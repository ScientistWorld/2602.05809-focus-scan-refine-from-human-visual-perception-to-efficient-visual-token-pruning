#!/usr/bin/env python3
"""Hidden (test-split) evaluator for visual-token pruning with a frozen VLM.

The test questions and answers live only in this directory
(``eval/test/test_items.jsonl``); the pruner inputs for them are produced at
test time in a scratch directory and deleted afterwards.

Every method is a *pruner program* registered as ``checkpoints/pruners/<name>.json``::

    {"command": ["python", "method/prune_fsr.py", "--alpha", "3"]}

For each registered method and each fixed token budget the evaluator

1. computes the visual inputs itself (CLIP ViT-L/14-336 penultimate-layer
   patch features, the LLaVA-1.5 projector outputs, [CLS] attention, CLIP
   joint-space patch/text embeddings) for the split's images and writes them
   to a scratch directory;
2. runs ``<command> --inputs DIR --budget K --output FILE`` on CPU only
   (no GPU visible, bounded time);
3. validates the returned selection: at most ``K`` slots per question, each
   slot a non-negative weighting of the image's own 576 visual tokens;
4. builds every LLM input itself from its private copy of the projected
   tokens (each slot becomes the weighted mean of its source tokens), runs the
   frozen LLaVA-1.5-7B language model greedily, parses the answers and scores
   them against ground truth.

A method whose pruner fails (crash, timeout, malformed or over-budget output)
receives the worst value (0.0) in every scored metric. Scored numbers are
relative accuracies: accuracy with pruned tokens divided by the accuracy of the
same frozen model with all 576 tokens, averaged over question families.

This module never imports method code; it only executes the registered
command as a subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT = "test"
EXPERIMENT = "token_pruning_core"
DEFAULT_ITEMS = Path(__file__).resolve().parent / "test_items.jsonl"
DEFAULT_OUTPUT = ROOT / "scoring" / "scores_test.json"
DEFAULT_PRUNERS = ROOT / "checkpoints" / "pruners"
DEFAULT_INPUTS = ROOT / "checkpoints" / ".pruner_inputs_test"
LOG_DIR = ROOT / "checkpoints" / "eval_logs" / SPLIT

LLAVA_DIR = ROOT / "data" / "downloads" / "llava-v1.5-7b"
CLIP_DIR = ROOT / "data" / "downloads" / "clip-vit-large-patch14-336"
IMAGE_DIR = ROOT / "data" / "downloads" / "coco" / "val2017"

BUDGET_MAIN = 64
BUDGET_LOW = 32
BUDGETS = (BUDGET_MAIN, BUDGET_LOW)
N_TOKENS = 576
GRID = 24
FAMILIES = ("exist", "count", "caption", "absent")
LOCAL_FAMILIES = ("exist", "count")
GLOBAL_FAMILIES = ("caption", "absent")
METRICS = (
    "performance_retained",
    "performance_retained_32",
    "local_evidence_retained",
    "global_context_retained",
)
WORST = {m: 0.0 for m in METRICS}
# Pruning is a light, plug-in step: CPU only, 60 s start-up + 0.1 s per question.
PRUNER_BASE_SEC = 60.0
PRUNER_PER_QUERY_SEC = 0.1
MAX_NEW_TOKENS = 6
BATCH_PRUNED = 48
BATCH_FULL = 12

SYSTEM = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

# Frozen artifacts: LFS sha256 for weights, git-blob sha1 for small files
# (liuhaotian/llava-v1.5-7b @ 4481d270, openai/clip-vit-large-patch14-336 @ ce19dc91).
PINNED_SHA256 = {
    "llava-v1.5-7b/pytorch_model-00001-of-00002.bin": "492fdfa63cf314097d99da81a3586d54a97adc1471a0ae0535f5caa269b86314",
    "llava-v1.5-7b/pytorch_model-00002-of-00002.bin": "c0eb94eb3fa578a4c63567f2eca6b0acf6a44a891c257aca0df88ffa891de72e",
    "llava-v1.5-7b/tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "clip-vit-large-patch14-336/pytorch_model.bin": "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801",
}
PINNED_GIT_SHA1 = {
    "llava-v1.5-7b/tokenizer_config.json": "740756b4bef305e27d0bb4d2e1a40dd8847797f7",
    "llava-v1.5-7b/special_tokens_map.json": "14761dcf1466dc232bd41de9c21d4c617b15755e",
    "clip-vit-large-patch14-336/config.json": "e9c5a5dc325b017eda0b6e802f79283a1f68bf27",
    "clip-vit-large-patch14-336/preprocessor_config.json": "a4604d0a1564fa0c45a6ff087bada5066a6335ea",
    "clip-vit-large-patch14-336/tokenizer.json": "564c0ebd5ce29c4ee4864004aee693deadd3128c",
    "clip-vit-large-patch14-336/tokenizer_config.json": "24feaf785502565f990d348aafa501c254710ed8",
    "clip-vit-large-patch14-336/special_tokens_map.json": "9bfb42aa97dcd61e89f279ccaee988bccb4fabae",
    "clip-vit-large-patch14-336/vocab.json": "182766ce89b439768edadda342519f33802f5364",
    "clip-vit-large-patch14-336/merges.txt": "bbfec752c9a675946c6dce106def6f35c882dcc2",
}

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
NUM_WORDS = {
    "zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}


class EvaluationError(RuntimeError):
    """The evaluator itself cannot produce valid scores."""


class MethodFailure(RuntimeError):
    """A registered method failed; it is scored with worst values."""


def log(msg: str) -> None:
    print(f"[eval:{SPLIT} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- inventory

def load_items(path: Path) -> list[dict]:
    items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not items:
        raise EvaluationError(f"no items in {path}")
    for it in items:
        if it["family"] not in FAMILIES:
            raise EvaluationError(f"unknown family {it['family']}")
    return items


def registered_methods(pruners_dir: Path) -> dict[str, Path]:
    if not pruners_dir.is_dir():
        return {}
    return {p.stem: p for p in sorted(pruners_dir.glob("*.json"))}


def write_scores(path: Path, results: dict, diagnostics: dict) -> None:
    payload = {"experiments": {EXPERIMENT: {"results": results}, "_diagnostics": diagnostics}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- integrity

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_artifacts(items: list[dict]) -> None:
    base = ROOT / "data" / "downloads"
    for rel, digest in PINNED_SHA256.items():
        p = base / rel
        if not p.is_file() or sha256_file(p) != digest:
            raise EvaluationError(f"frozen artifact missing or modified: {rel}")
    for rel, digest in PINNED_GIT_SHA1.items():
        p = base / rel
        if not p.is_file() or git_blob_sha1(p) != digest:
            raise EvaluationError(f"frozen artifact missing or modified: {rel}")
    seen: dict[str, str] = {}
    for it in items:
        seen[it["image"]] = it["image_sha256"]
    for name, digest in seen.items():
        p = IMAGE_DIR / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise EvaluationError(f"benchmark image missing or modified: {name}")


# ---------------------------------------------------------------- features

def expand2square(img, background):
    from PIL import Image

    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    out = Image.new(img.mode, (side, side), background)
    out.paste(img, (0, (side - h) // 2) if w > h else ((side - w) // 2, 0))
    return out


def compute_visual_inputs(image_names: list[str], projector, device, dtype) -> dict[str, np.ndarray]:
    import torch
    from PIL import Image
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    proc = CLIPImageProcessor.from_pretrained(CLIP_DIR)
    model = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_DIR, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    background = tuple(int(x * 255) for x in proc.image_mean)
    n = len(image_names)
    vis = np.zeros((n, N_TOKENS, 1024), dtype=np.float16)
    proj = np.zeros((n, N_TOKENS, 4096), dtype=np.float16)
    joint = np.zeros((n, N_TOKENS, 768), dtype=np.float16)
    cosine = np.zeros((n, N_TOKENS, N_TOKENS), dtype=np.float16)
    cls_attn = np.zeros((n, N_TOKENS), dtype=np.float32)
    bs = 8
    with torch.inference_mode():
        for start in range(0, n, bs):
            names = image_names[start:start + bs]
            imgs = [expand2square(Image.open(IMAGE_DIR / nm).convert("RGB"), background) for nm in names]
            px = proc.preprocess(imgs, return_tensors="pt")["pixel_values"].to(device, dtype)
            out = model.vision_model(pixel_values=px, output_hidden_states=True, output_attentions=True)
            feats = out.hidden_states[-2][:, 1:]                       # LLaVA select_layer=-2, patch tokens
            attn = out.attentions[-2][:, :, 0, 1:].float().mean(dim=1)  # [CLS]->patch, head mean
            emb = model.vision_model.post_layernorm(feats).float() @ model.visual_projection.weight.float().t()
            p = projector(feats)
            pf = p.float()
            pn = pf / pf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cos = pn @ pn.transpose(1, 2)
            sl = slice(start, start + len(names))
            vis[sl] = feats.float().cpu().numpy().astype(np.float16)
            proj[sl] = p.float().cpu().numpy().astype(np.float16)
            joint[sl] = emb.cpu().numpy().astype(np.float16)
            cosine[sl] = cos.cpu().numpy().astype(np.float16)
            cls_attn[sl] = attn.cpu().numpy()
    del model
    return {"vision_features": vis, "projected_features": proj, "joint_embeds": joint,
            "projected_cosine": cosine, "cls_attention": cls_attn}


def compute_text_embeds(questions: list[str], device, dtype) -> np.ndarray:
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    tok = CLIPTokenizerFast.from_pretrained(CLIP_DIR)
    model = CLIPTextModelWithProjection.from_pretrained(CLIP_DIR, torch_dtype=dtype).to(device).eval()
    seg_len = model.config.max_position_embeddings
    out = np.zeros((len(questions), 768), dtype=np.float32)
    with torch.inference_mode():
        for i, q in enumerate(questions):
            enc = tok(text=[q], return_tensors="pt")
            ids, mask = enc["input_ids"], enc["attention_mask"]
            n_seg = (ids.shape[1] - 1) // seg_len + 1
            pad = n_seg * seg_len - ids.shape[1]
            ids = torch.cat([ids, ids.new_zeros((1, pad))], dim=1).reshape(-1, seg_len)
            mask = torch.cat([mask, mask.new_zeros((1, pad))], dim=1).reshape(-1, seg_len)
            emb = model(input_ids=ids.to(device), attention_mask=mask.to(device)).text_embeds.float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            out[i] = emb.mean(dim=0).cpu().numpy()
    del model
    return out


def inputs_fingerprint(inputs_dir: Path) -> dict:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_mode)
            for p in sorted(inputs_dir.iterdir())} if inputs_dir.is_dir() else {}


def write_pruner_inputs(inputs_dir: Path, visual: dict, image_names: list[str],
                        queries: list[dict], text_embeds: np.ndarray) -> None:
    if inputs_dir.exists():
        for p in inputs_dir.iterdir():
            os.chmod(p, 0o644)
        shutil.rmtree(inputs_dir)
    inputs_dir.mkdir(parents=True)
    for key, arr in visual.items():
        np.save(inputs_dir / f"{key}.npy", arr)
    np.save(inputs_dir / "text_embeds.npy", text_embeds)
    with (inputs_dir / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")
    meta = {
        "format_version": 1,
        "num_images": len(image_names),
        "num_queries": len(queries),
        "num_visual_tokens": N_TOKENS,
        "grid": [GRID, GRID],
        "files": {
            "vision_features.npy": "[images, 576, 1024] float16 CLIP ViT-L/14-336 penultimate-layer patch features (row-major 24x24 grid)",
            "projected_features.npy": "[images, 576, 4096] float16 LLaVA-1.5 projector outputs: the tokens the language model consumes",
            "projected_cosine.npy": "[images, 576, 576] float16 cosine similarity between projected tokens",
            "joint_embeds.npy": "[images, 576, 768] float16 patch features mapped into CLIP's image-text joint space",
            "cls_attention.npy": "[images, 576] float32 [CLS]-to-patch attention of the penultimate vision layer, mean over heads",
            "text_embeds.npy": "[queries, 768] float32 mean of L2-normalised CLIP text embeddings of the question (77-token segments)",
            "queries.jsonl": "one query per line: qid, image (row index into the image arrays), question text",
        },
        "output_format": "JSONL, one line per qid: {\"qid\": str, \"slots\": [{\"tokens\": [int, ...], \"weights\": [float, ...]}, ...]}; "
                         "at most --budget slots; tokens are indices in [0, 576) of the query's image; weights optional (uniform), "
                         "finite, >= 0, positive sum. Each slot is fed to the language model as the weighted mean of its tokens' "
                         "projected features, in the order given.",
    }
    (inputs_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for p in inputs_dir.iterdir():
        os.chmod(p, 0o444)


# ---------------------------------------------------------------- pruners

def reserve_gpus(torch) -> list:
    """Hold all free accelerator memory while a pruner runs, so the frozen language
    model (or any other large network) cannot be run inside the pruning step."""
    blocks = []
    if not torch.cuda.is_available():
        return blocks
    torch.cuda.empty_cache()          # return cached blocks so the driver's free count is exact
    margin = 256 << 20
    for dev in range(torch.cuda.device_count()):
        try:
            free, _ = torch.cuda.mem_get_info(dev)
        except RuntimeError:
            continue
        chunk = 1 << 30
        while free > margin + (16 << 20):
            size = min(chunk, free - margin)
            try:
                blocks.append(torch.empty(size, dtype=torch.uint8, device=f"cuda:{dev}"))
                free -= size
            except RuntimeError:
                if chunk <= (16 << 20):
                    break
                chunk //= 4
    return blocks


def release_gpus(torch, blocks: list) -> None:
    blocks.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


HIDDEN_DIR = Path(__file__).resolve().parent
_HIDDEN_MODE = None


def hide_answers() -> None:
    """Make eval/test/ untraversable while a pruner runs (the answers are already in memory)."""
    global _HIDDEN_MODE
    try:
        if _HIDDEN_MODE is None:
            _HIDDEN_MODE = HIDDEN_DIR.stat().st_mode & 0o7777
        os.chmod(HIDDEN_DIR, 0o000)
    except OSError as exc:
        log(f"  could not hide {HIDDEN_DIR}: {exc}")


def unhide_answers() -> None:
    if _HIDDEN_MODE is not None:
        try:
            os.chmod(HIDDEN_DIR, _HIDDEN_MODE)
        except OSError as exc:
            log(f"  could not restore {HIDDEN_DIR}: {exc}")


def run_pruner(name: str, spec_path: Path, inputs_dir: Path, budget: int, out_path: Path,
               num_queries: int) -> None:
    try:
        spec = json.loads(spec_path.read_text())
        command = spec["command"]
        if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
            raise ValueError("command must be a non-empty list of strings")
    except Exception as exc:  # noqa: BLE001
        raise MethodFailure(f"invalid registration {spec_path.name}: {exc}") from exc
    if command[0] in ("python", "python3"):
        command = [sys.executable] + command[1:]
    if out_path.exists():
        out_path.unlink()
    env = dict(os.environ)
    # Pruning must be a light, plug-in step: the language model is not available to it.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["PYTHONHASHSEED"] = "0"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}_{budget}.log"
    cmd = command + ["--inputs", str(inputs_dir), "--budget", str(budget), "--output", str(out_path)]
    timeout = PRUNER_BASE_SEC + PRUNER_PER_QUERY_SEC * num_queries
    t0 = time.time()
    with log_path.open("w") as lf:
        try:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT,
                                  timeout=timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired as exc:
            raise MethodFailure(f"pruner exceeded its time budget of {timeout:.0f}s") from exc
        except OSError as exc:
            raise MethodFailure(f"pruner could not start: {exc}") from exc
    if proc.returncode != 0:
        tail = log_path.read_text()[-2000:]
        raise MethodFailure(f"pruner exited with {proc.returncode}: {tail}")
    log(f"  pruner {name} K={budget} finished in {time.time() - t0:.1f}s")


def _reject_constant(token: str):
    raise ValueError(f"non-finite JSON constant {token}")


def load_selection(path: Path, qids: list[str], budget: int) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    if not path.is_file():
        raise MethodFailure("pruner wrote no output file")
    wanted = set(qids)
    sel: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line, parse_constant=_reject_constant)
                qid = rec["qid"]
                slots = rec["slots"]
                if not isinstance(rec, dict) or set(rec) != {"qid", "slots"}:
                    raise ValueError("record must have exactly the keys 'qid' and 'slots'")
            except Exception as exc:  # noqa: BLE001
                raise MethodFailure(f"line {lineno}: malformed record ({exc})") from exc
            if not isinstance(qid, str) or qid not in wanted or qid in sel:
                raise MethodFailure(f"line {lineno}: unknown or duplicate qid {qid!r}")
            if not isinstance(slots, list) or not 1 <= len(slots) <= budget:
                raise MethodFailure(f"qid {qid}: needs 1..{budget} slots")
            parsed = []
            for slot in slots:
                if not isinstance(slot, dict) or not set(slot) <= {"tokens", "weights"}:
                    raise MethodFailure(f"qid {qid}: slot must be an object with only 'tokens' and 'weights'")
                toks = slot.get("tokens")
                if (not isinstance(toks, list) or not toks
                        or not all(isinstance(t, int) and not isinstance(t, bool) for t in toks)):
                    raise MethodFailure(f"qid {qid}: slot tokens must be a non-empty list of ints")
                if min(toks) < 0 or max(toks) >= N_TOKENS or len(set(toks)) != len(toks):
                    raise MethodFailure(f"qid {qid}: token indices out of range or repeated")
                weights = slot.get("weights")
                if weights is None:
                    w = np.ones(len(toks), dtype=np.float64)
                else:
                    if (not isinstance(weights, list) or len(weights) != len(toks)
                            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in weights)):
                        raise MethodFailure(f"qid {qid}: weights must match tokens")
                    w = np.asarray(weights, dtype=np.float64)
                    if not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
                        raise MethodFailure(f"qid {qid}: weights must be finite, >= 0, with positive sum")
                parsed.append((np.asarray(toks, dtype=np.int64), w / w.sum()))
            sel[qid] = parsed
    if len(sel) != len(wanted):
        raise MethodFailure(f"selection covers {len(sel)} of {len(wanted)} queries")
    return sel


# ---------------------------------------------------------------- language model

class FrozenLLaVA:
    def __init__(self, device, dtype):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

        self.torch = torch
        self.device = device
        self.dtype = dtype
        cfg = LlamaConfig(
            vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32,
            num_attention_heads=32, num_key_value_heads=32, max_position_embeddings=4096,
            rms_norm_eps=1e-5, hidden_act="silu", bos_token_id=1, eos_token_id=2, pad_token_id=0,
            tie_word_embeddings=False,
        )
        self.tok = LlamaTokenizer.from_pretrained(LLAVA_DIR, use_fast=False)
        self.model = LlamaForCausalLM.from_pretrained(
            LLAVA_DIR, config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        shard_path = LLAVA_DIR / "pytorch_model-00002-of-00002.bin"   # hash-verified above
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        proj = torch.nn.Sequential(torch.nn.Linear(1024, 4096), torch.nn.GELU(), torch.nn.Linear(4096, 4096))
        proj.load_state_dict({
            "0.weight": shard["model.mm_projector.0.weight"], "0.bias": shard["model.mm_projector.0.bias"],
            "2.weight": shard["model.mm_projector.2.weight"], "2.bias": shard["model.mm_projector.2.bias"],
        })
        del shard
        self.projector = proj.to(device, dtype).eval()
        self.embed = self.model.get_input_embeddings()
        self._cache: dict[str, tuple[list[int], list[int]]] = {}

    def prompt_ids(self, question: str, suffix: str) -> tuple[list[int], list[int]]:
        key = question + "\x00" + suffix
        if key not in self._cache:
            prompt = f"{SYSTEM} USER: <image>\n{question}\n{suffix} ASSISTANT:"
            before, after = prompt.split("<image>")
            ids_before = self.tok(before).input_ids               # keeps BOS
            ids_after = self.tok(after).input_ids[1:]             # LLaVA strips the chunk BOS
            self._cache[key] = (ids_before, ids_after)
        return self._cache[key]

    def answer(self, jobs: list[tuple[str, str, int, object]], batch_size: int) -> list[str]:
        """jobs: (question, suffix, n_visual, make_visual); make_visual() -> [n_visual, 4096] tensor.

        Questions are grouped by exact sequence length, so no padding is used and
        every method sees identical batching for identical inputs.
        """
        torch = self.torch
        groups: dict[int, list[int]] = {}
        for i, (q, s, n_vis, _) in enumerate(jobs):
            a, b = self.prompt_ids(q, s)
            groups.setdefault(len(a) + n_vis + len(b), []).append(i)
        out = [""] * len(jobs)
        with torch.inference_mode():
            for length in sorted(groups):
                idxs = groups[length]
                for start in range(0, len(idxs), batch_size):
                    chunk = idxs[start:start + batch_size]
                    embeds = []
                    for i in chunk:
                        q, s, n_vis, make_visual = jobs[i]
                        a, b = self.prompt_ids(q, s)
                        v = make_visual()
                        if v.shape != (n_vis, 4096):
                            raise EvaluationError("visual token tensor has the wrong shape")
                        ea = self.embed(torch.tensor(a, device=self.device))
                        eb = self.embed(torch.tensor(b, device=self.device))
                        embeds.append(torch.cat([ea, v.to(self.dtype), eb], dim=0))
                    x = torch.stack(embeds, dim=0)
                    mask = torch.ones(x.shape[:2], dtype=torch.long, device=self.device)
                    gen = self.model.generate(
                        inputs_embeds=x, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False, num_beams=1, temperature=None, top_p=None,
                        pad_token_id=0, eos_token_id=2,
                    )
                    texts = self.tok.batch_decode(gen, skip_special_tokens=True)
                    for i, t in zip(chunk, texts):
                        out[i] = t.strip()
        return out


def parse_answer(family: str, text: str) -> str | None:
    t = text.strip().lower()
    if family == "exist":
        words = re.findall(r"[a-z]+", t)
        if not words:
            return None
        if words[0] == "yes":
            return "yes"
        if words[0] == "no":
            return "no"
        return "no" if ("no" in words or "not" in words) else "yes"   # POPE rule
    if family == "count":
        m = re.search(r"\d+", t)
        if m:
            return str(int(m.group()))
        for w in re.findall(r"[a-z]+", t):
            if w in NUM_WORDS:
                return str(NUM_WORDS[w])
        return None
    m = re.match(r"^\W*([abcd])(?:\W|$)", t)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------- scoring

def family_accuracy(items: list[dict], answers: list[str]) -> tuple[dict[str, float], list[bool]]:
    correct = [parse_answer(it["family"], a) == it["answer"] for it, a in zip(items, answers)]
    acc = {}
    for fam in FAMILIES:
        flags = [c for it, c in zip(items, correct) if it["family"] == fam]
        acc[fam] = float(np.mean(flags)) if flags else float("nan")
    return acc, correct


def relative(acc: dict[str, float], full: dict[str, float], fams: tuple[str, ...]) -> float:
    vals = [acc[f] / full[f] for f in fams if full.get(f, 0) > 0 and not math.isnan(acc[f])]
    return float(np.mean(vals)) if vals else 0.0


def evaluate(args) -> int:
    import torch

    items = load_items(Path(args.items))
    if args.max_images:
        keep = sorted({it["image"] for it in items})[: args.max_images]
        items = [it for it in items if it["image"] in set(keep)]
    methods = registered_methods(Path(args.pruners_dir))
    out_path = Path(args.output)
    # Until scoring finishes, every registered method holds its worst values.
    write_scores(out_path, {m: dict(WORST) for m in methods}, {"status": "incomplete"})
    if not methods:
        log(f"no registered pruners in {args.pruners_dir}")
        return 1

    verify_artifacts(items)
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    log(f"{len(items)} questions, {len(methods)} registered methods, device={device}")

    llava = FrozenLLaVA(device, dtype)
    image_names = sorted({it["image"] for it in items})
    img_index = {nm: i for i, nm in enumerate(image_names)}
    visual = compute_visual_inputs(image_names, llava.projector, device, dtype)
    queries = [{"qid": it["qid"], "image": img_index[it["image"]], "question": it["question"]} for it in items]
    text_embeds = compute_text_embeds([it["question"] for it in items], device, dtype)
    inputs_dir = Path(args.inputs_dir)
    write_pruner_inputs(inputs_dir, visual, image_names, queries, text_embeds)
    fingerprint = inputs_fingerprint(inputs_dir)
    # The evaluator scores from its private in-memory copy, never from the shared files.
    projected = torch.from_numpy(visual["projected_features"])
    log(f"visual inputs ready for {len(image_names)} images")

    def full_tokens(image_row: int):
        return lambda: projected[image_row].to(device)

    def merged_tokens(image_row: int, weights: np.ndarray):
        # Each slot is the weighted mean of its source tokens' projected features.
        return lambda: torch.from_numpy(weights).to(device) @ projected[image_row].to(device).float()

    t0 = time.time()
    full_jobs = [(it["question"], it["suffix"], N_TOKENS, full_tokens(img_index[it["image"]])) for it in items]
    full_answers = llava.answer(full_jobs, BATCH_FULL)
    full_acc, full_correct = family_accuracy(items, full_answers)
    log(f"full-token accuracy {full_acc} ({time.time() - t0:.0f}s)")
    if not args.max_images and any(not full_acc[f] > 0 for f in FAMILIES):
        raise EvaluationError("full-token model scores zero on a family; environment is broken")
    save_dir = Path(args.save_answers) if args.save_answers else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        dump_answers(save_dir / "_full_tokens.jsonl", items, full_answers, full_correct)

    results: dict[str, dict] = {}
    diag: dict[str, dict] = {"_full_tokens": {"family_accuracy": full_acc}}
    qids = [it["qid"] for it in items]
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".selections_{SPLIT}_", dir=ROOT / "checkpoints"))
    for name, spec in methods.items():
        t0 = time.time()
        if not NAME_RE.match(name):
            results[name] = dict(WORST)
            diag[name] = {"status": "failed", "reason": "invalid method name"}
            continue
        try:
            per_budget = {}
            for budget in BUDGETS:
                sel_path = scratch / f"{name}_{budget}.jsonl"
                if inputs_fingerprint(inputs_dir) != fingerprint:
                    log("  pruner inputs were modified by a previous pruner; rewriting them")
                    write_pruner_inputs(inputs_dir, visual, image_names, queries, text_embeds)
                    fingerprint = inputs_fingerprint(inputs_dir)
                held = reserve_gpus(torch)
                hide_answers()
                try:
                    run_pruner(name, spec, inputs_dir, budget, sel_path, len(qids))
                finally:
                    unhide_answers()
                    release_gpus(torch, held)
                sel = load_selection(sel_path, qids, budget)
                jobs, n_slots = [], []
                for it in items:
                    slots = sel[it["qid"]]
                    w = np.zeros((len(slots), N_TOKENS), dtype=np.float32)
                    for r, (toks, wt) in enumerate(slots):
                        w[r, toks] = wt
                    jobs.append((it["question"], it["suffix"], len(slots), merged_tokens(img_index[it["image"]], w)))
                    n_slots.append(len(slots))
                del sel
                answers = llava.answer(jobs, BATCH_PRUNED)
                acc, correct = family_accuracy(items, answers)
                per_budget[budget] = {"family_accuracy": acc, "mean_slots": float(np.mean(n_slots))}
                if save_dir:
                    dump_answers(save_dir / f"{name}__{budget}.jsonl", items, answers, correct)
            acc64 = per_budget[BUDGET_MAIN]["family_accuracy"]
            acc32 = per_budget[BUDGET_LOW]["family_accuracy"]
            results[name] = {
                "performance_retained": round(relative(acc64, full_acc, FAMILIES), 6),
                "performance_retained_32": round(relative(acc32, full_acc, FAMILIES), 6),
                "local_evidence_retained": round(relative(acc64, full_acc, LOCAL_FAMILIES), 6),
                "global_context_retained": round(relative(acc64, full_acc, GLOBAL_FAMILIES), 6),
            }
            diag[name] = {"status": "ok", "budgets": {str(k): v for k, v in per_budget.items()}}
            log(f"{name}: {results[name]} ({time.time() - t0:.0f}s)")
        except MethodFailure as exc:
            results[name] = dict(WORST)
            diag[name] = {"status": "failed", "reason": str(exc)[:500]}
            log(f"{name}: FAILED -> worst values ({str(exc)[:300]})")
        write_scores(out_path, {**{m: dict(WORST) for m in methods}, **results}, {"status": "incomplete", **diag})
    shutil.rmtree(scratch, ignore_errors=True)
    diag["status"] = "complete"
    diag["num_questions"] = len(items)
    diag["num_images"] = len(image_names)
    write_scores(out_path, results, diag)
    log(f"wrote {out_path}")
    return 0


def dump_answers(path: Path, items: list[dict], answers: list[str], correct: list[bool]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for it, a, c in zip(items, answers, correct):
            fh.write(json.dumps({"qid": it["qid"], "family": it["family"], "answer": a, "correct": bool(c)}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--items", default=str(DEFAULT_ITEMS))
    parser.add_argument("--pruners-dir", default=str(DEFAULT_PRUNERS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--inputs-dir", default=str(DEFAULT_INPUTS),
                        help="scratch directory for the pruner inputs (deleted after scoring)")
    parser.add_argument("--save-answers", default=None, help="optional directory for per-question answers")
    parser.add_argument("--max-images", type=int, default=0, help="debug: evaluate only the first N images")
    parser.add_argument("--device", default=None, help="torch device for the frozen models (default: cuda if available)")
    args = parser.parse_args()
    try:
        code = evaluate(args)
    except EvaluationError as exc:
        log(f"EVALUATION ERROR: {exc} -- every registered method keeps its worst values")
        code = 1
    finally:
        unhide_answers()
        inputs_dir = Path(args.inputs_dir)
        if inputs_dir.exists():
            for p in inputs_dir.iterdir():
                os.chmod(p, 0o644)
            shutil.rmtree(inputs_dir, ignore_errors=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
