#!/usr/bin/env python3
"""Build the question sets for the visual-token pruning benchmark.

Maintainer tool. It reads the public COCO val2017 instance and caption
annotations, selects two disjoint image sets, and writes:

* ``data/train_items.jsonl``  -- the visible development split;
* ``eval/test/test_items.jsonl`` -- the hidden test split (lives only here).

It is not run at evaluation time. Because it can regenerate the hidden
answers, it lives in ``eval/test/`` only. Usage (paths are arguments so no
machine-specific location is committed):

    python eval/test/build_items.py \
        --instances /path/to/annotations/instances_val2017.json \
        --captions  /path/to/annotations/captions_val2017.json \
        --images    /path/to/val2017

Question families (all answered from COCO ground truth):

* ``exist``   -- POPE protocol object existence, yes/no. Per image two
                 present objects and two absent objects; absent objects are
                 sampled with the random / popular / adversarial strategies
                 of POPE in rotation.
* ``count``   -- "How many X are there?" for a category whose instances are
                 all clearly visible (no crowd annotation, no tiny instance).
* ``caption`` -- four-way caption matching; distractors are captions of
                 other images in the same split with the most similar object
                 sets (hard negatives).
* ``absent``  -- four-way "which object is not present"; three present
                 salient objects and one absent co-occurring object.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

SEED = "fsr-gym-coco-items-v1"
N_TRAIN = 300
N_TEST = 700
SALIENT_AREA = 0.01      # largest instance of a "salient" category covers >= 1% of the image
COUNT_MIN_AREA = 0.0025  # every counted instance covers >= 0.25% of the image
COUNT_RANGE = (1, 8)
COUNT_EXCLUDE = {"broccoli", "carrot", "book", "banana", "skis", "sports ball"}

PLURAL = {
    "person": "people", "bus": "buses", "sheep": "sheep", "scissors": "pairs of scissors",
    "mouse": "computer mice", "knife": "knives", "wine glass": "wine glasses",
    "sandwich": "sandwiches", "couch": "couches", "bench": "benches",
    "toothbrush": "toothbrushes", "hair drier": "hair driers", "tv": "tvs",
    "skis": "pairs of skis", "glass": "glasses", "dining table": "dining tables",
    "giraffe": "giraffes", "cell phone": "cell phones", "teddy bear": "teddy bears",
}

YESNO_SUFFIX = "Answer the question using a single word or phrase."
MCQ_SUFFIX = "Answer with the option's letter from the given choices directly."
LETTERS = "ABCD"


def plural(name: str) -> str:
    if name in PLURAL:
        return PLURAL[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


def article(name: str) -> str:
    return "an" if name[0] in "aeiou" else "a"


def stable_rng(*parts: object) -> random.Random:
    digest = hashlib.sha256(":".join([SEED, *map(str, parts)]).encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def clean_caption(text: str) -> str:
    text = " ".join(text.strip().split())
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text.rstrip(" .") + "."


def load(instances_path: Path, captions_path: Path):
    inst = json.loads(instances_path.read_text())
    caps = json.loads(captions_path.read_text())
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    images = {im["id"]: im for im in inst["images"]}
    anns = collections.defaultdict(list)
    for a in inst["annotations"]:
        anns[a["image_id"]].append(a)
    captions = collections.defaultdict(list)
    for c in sorted(caps["annotations"], key=lambda c: c["id"]):
        captions[c["image_id"]].append(clean_caption(c["caption"]))
    return cat_name, images, anns, captions


def image_summary(im: dict, anns: list[dict], cat_name: dict) -> dict:
    area = float(im["width"] * im["height"])
    present = set()
    crowd = set()
    largest = collections.defaultdict(float)
    smallest = {}
    counts = collections.Counter()
    for a in anns:
        name = cat_name[a["category_id"]]
        present.add(name)
        if a["iscrowd"]:
            crowd.add(name)
            continue
        frac = a["area"] / area
        largest[name] = max(largest[name], frac)
        smallest[name] = min(smallest.get(name, 1.0), frac)
        counts[name] += 1
    salient = sorted(n for n in present if largest.get(n, 0.0) >= SALIENT_AREA and n not in crowd)
    countable = sorted(
        n for n in counts
        if n not in crowd and n not in COUNT_EXCLUDE
        and smallest[n] >= COUNT_MIN_AREA
        and COUNT_RANGE[0] <= counts[n] <= COUNT_RANGE[1]
    )
    return {
        "present": present,
        "salient": salient,
        "countable": {n: counts[n] for n in countable},
        "largest": dict(largest),
    }


def build(instances_path: Path, captions_path: Path) -> tuple[list[dict], list[dict]]:
    cat_name, images, anns, captions = load(instances_path, captions_path)
    all_cats = sorted(cat_name.values())

    summaries = {iid: image_summary(images[iid], anns[iid], cat_name) for iid in anns}
    eligible = sorted(
        iid for iid, s in summaries.items()
        if len(s["salient"]) >= 3 and len(captions.get(iid, [])) >= 1
    )
    order = sorted(eligible, key=lambda iid: hashlib.sha256(f"{SEED}:split:{iid}".encode()).hexdigest())
    if len(order) < N_TRAIN + N_TEST:
        raise SystemExit(f"only {len(order)} eligible images")
    train_ids = order[:N_TRAIN]
    test_ids = order[N_TRAIN:N_TRAIN + N_TEST]

    # Popularity and co-occurrence statistics over the whole annotation set
    # (POPE's popular and adversarial negative sampling).
    freq = collections.Counter()
    cooc = collections.defaultdict(collections.Counter)
    for s in summaries.values():
        for a in s["present"]:
            freq[a] += 1
            for b in s["present"]:
                if a != b:
                    cooc[a][b] += 1
    def ranked_by_count(counter: collections.Counter) -> list[str]:
        # Deterministic: count descending, then name (independent of hash seed).
        return [c for c, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]

    popular = ranked_by_count(freq)

    def negatives(iid: int, k: int, start: int) -> list[str]:
        s = summaries[iid]
        rng = stable_rng("neg", iid)
        out: list[str] = []
        strategies = ["random", "popular", "adversarial"]
        for j in range(k):
            strategy = strategies[(start + j) % 3]
            absent = [c for c in all_cats if c not in s["present"] and c not in out]
            if strategy == "random":
                choice = rng.choice(absent)
            elif strategy == "popular":
                choice = next(c for c in popular if c in absent)
            else:
                score = collections.Counter()
                for p in sorted(s["present"]):
                    for c, v in cooc[p].items():
                        if c in absent:
                            score[c] += v
                ranked = ranked_by_count(score) or absent
                choice = ranked[0]
            out.append(choice)
        return out

    def adversarial_absent(iid: int, rng: random.Random) -> str:
        s = summaries[iid]
        score = collections.Counter()
        for p in sorted(s["present"]):
            for c, v in cooc[p].items():
                if c not in s["present"]:
                    score[c] += v
        ranked = ranked_by_count(score)[:5]
        if not ranked:
            ranked = [c for c in all_cats if c not in s["present"]]
        return rng.choice(ranked)

    def make_items(ids: list[int], split: str) -> list[dict]:
        items: list[dict] = []
        cat_sets = {iid: set(summaries[iid]["salient"]) for iid in ids}
        neg_counter = 0
        for iid in ids:
            s = summaries[iid]
            rng = stable_rng(split, iid)
            fname = images[iid]["file_name"]
            base = {"image": fname}

            # exist: 2 present salient objects, 2 absent objects
            pos = rng.sample(s["salient"], 2)
            neg = negatives(iid, 2, neg_counter)
            neg_counter += 2
            for name, ans in [(pos[0], "yes"), (neg[0], "no"), (pos[1], "yes"), (neg[1], "no")]:
                items.append({**base, "family": "exist",
                              "question": f"Is there {article(name)} {name} in the image?",
                              "suffix": YESNO_SUFFIX, "answer": ans})

            # count
            countable = s["countable"]
            if countable:
                multi = sorted(n for n, c in countable.items() if c >= 2)
                pool = multi or sorted(countable)
                name = rng.choice(pool)
                items.append({**base, "family": "count",
                              "question": f"How many {plural(name)} are there in the image?",
                              "suffix": YESNO_SUFFIX, "answer": str(countable[name])})

            # caption: true caption + 3 hard-negative captions from the same split
            true_cap = captions[iid][0]
            others = [j for j in ids if j != iid]
            others.sort(key=lambda j: (
                -len(cat_sets[iid] & cat_sets[j]) / max(1, len(cat_sets[iid] | cat_sets[j])),
                hashlib.sha256(f"{SEED}:{iid}:{j}".encode()).hexdigest(),
            ))
            distractor_imgs = rng.sample(others[:12], 3)
            options = [true_cap] + [captions[j][rng.randrange(len(captions[j]))] for j in distractor_imgs]
            perm = list(range(4))
            rng.shuffle(perm)
            opts = [options[p] for p in perm]
            answer = LETTERS[perm.index(0)]
            question = "Which of the following captions best describes the image?\n" + "\n".join(
                f"{LETTERS[k]}. {o}" for k, o in enumerate(opts))
            items.append({**base, "family": "caption", "question": question,
                          "suffix": MCQ_SUFFIX, "answer": answer})

            # absent: 3 present salient objects + 1 absent co-occurring object
            present3 = rng.sample(s["salient"], 3)
            missing = adversarial_absent(iid, rng)
            options = present3 + [missing]
            perm = list(range(4))
            rng.shuffle(perm)
            opts = [options[p] for p in perm]
            answer = LETTERS[perm.index(3)]
            question = "Which of the following objects is not present in the image?\n" + "\n".join(
                f"{LETTERS[k]}. {o}" for k, o in enumerate(opts))
            items.append({**base, "family": "absent", "question": question,
                          "suffix": MCQ_SUFFIX, "answer": answer})
        for k, item in enumerate(items):
            item["qid"] = f"{split}-{k:05d}"
        return items

    return make_items(train_ids, "train"), make_items(test_ids, "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--train-out", default="data/train_items.jsonl")
    parser.add_argument("--test-out", default="eval/test/test_items.jsonl")
    parser.add_argument("--images", required=True,
                        help="directory holding the COCO val2017 JPEGs (pins each image by SHA-256)")
    args = parser.parse_args()
    train, test = build(Path(args.instances), Path(args.captions))
    digests: dict[str, str] = {}
    for item in train + test:
        name = item["image"]
        if name not in digests:
            digests[name] = hashlib.sha256((Path(args.images) / name).read_bytes()).hexdigest()
        item["image_sha256"] = digests[name]
    for path, items in [(args.train_out, train), (args.test_out, test)]:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, sort_keys=True) + "\n")
        fams = collections.Counter(i["family"] for i in items)
        print(f"{path}: {len(items)} items over {len({i['image'] for i in items})} images {dict(fams)}")


if __name__ == "__main__":
    main()
