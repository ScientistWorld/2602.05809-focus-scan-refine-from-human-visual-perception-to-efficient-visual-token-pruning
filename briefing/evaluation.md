# Evaluation

## Metrics (all higher is better)

* `performance_retained`: at K = 64, the mean over the four question families of
  (accuracy with the pruned tokens) / (accuracy of the same frozen model with all 576 tokens).
  This is the "average relative performance" used in the visual-token pruning literature.
* `performance_retained_32`: the same at K = 32.
* `local_evidence_retained`: at K = 64, relative performance on the families answered from
  specific objects (existence, counting).
* `global_context_retained`: at K = 64, relative performance on the families that need the
  whole scene (caption matching, absent-object choice).

A method that fails (crash, timeout, malformed or over-budget output, missing questions)
gets 0.0 in every metric.

## Question families

COCO val2017 images, ground truth from COCO annotations:

* existence: "Is there a/an X in the image?" (yes/no; absent objects sampled at random,
  by popularity, and by co-occurrence with present objects, as in POPE);
* counting: "How many X are there in the image?" for clearly visible, non-crowd instances;
* caption: choose the matching caption among four (distractors from images with similar
  object sets);
* absent: choose which of four listed objects is not present (three present, one absent
  co-occurring object).

Prompts follow LLaVA-1.5's evaluation templates ("Answer the question using a single word or
phrase." / "Answer with the option's letter from the given choices directly."). Answers are
parsed with POPE's yes/no rule, the first number, or the first option letter.

## Protocol

1. Register a method as a pruner program: `checkpoints/pruners/<name>.json` containing
   `{"command": ["python", "path/to/your_pruner.py", "--any", "options"]}`. Every file there
   is scored; there is no other inventory.
2. The evaluator computes the inputs itself and runs
   `<command> --inputs DIR --budget K --output FILE` for K = 64 and 32, on CPU only, with a
   time budget of 60 s + 0.1 s per question. `DIR/meta.json` documents the inputs:
   `vision_features` (CLIP penultimate-layer patch features), `projected_features` (the
   projector outputs the language model consumes), `projected_cosine`, `joint_embeds` (patch
   features in CLIP's image-text space), `cls_attention` ([CLS]-to-patch attention of the
   penultimate vision layer), `text_embeds` (CLIP text embedding of the question), and
   `queries.jsonl`. `data/pruner_io.py` reads them.
3. Output: one JSON line per question, `{"qid": ..., "slots": [{"tokens": [i, ...],
   "weights": [w, ...]}, ...]}`, with at most K slots, token indices in [0, 576), and weights
   finite, >= 0, positive sum (optional, uniform if omitted). The evaluator turns each slot
   into the weighted mean of its tokens' projected features, from its own copy, and runs the
   frozen model.
4. `bash scripts/evaluate_train.sh` scores every registered method on the visible split
   (300 images, 2,100 questions) into `scoring/scores_train.json` and leaves that split's
   pruner inputs in `data/downloads/pruner_inputs_train/` for offline CPU development.
   `bash scripts/evaluate_test.sh` scores the hidden split (700 images, 4,898 questions) into
   `scoring/scores_test.json`. Both need a GPU for the frozen models (about 5 and 10 minutes
   on an H200 for five methods).

The frozen weights and benchmark images are verified against pinned hashes. Accuracy with
all 576 tokens on the hidden split: existence 0.916, counting 0.610, caption 0.883,
absent 0.637.

## Reference points

The source paper reports average relative performance on nine benchmarks for LLaVA-1.5-7B at
64 tokens (Table 1): 96.1% for its method, 95.7% for the strongest prior method. Those numbers
are kept in `scoring/reference.json`; this environment's question set differs, so values are
comparable in kind, not identical.
