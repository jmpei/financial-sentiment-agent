# Spec — Replace the random-head "baseline" with honest baselines

**Type:** Credibility fix. The current headline comparison is against a random
classifier, which a finance-savvy reviewer will reject. Add real baselines.

## Context (verified in code/README)

- `baseline.py` evaluates `distilbert-base-uncased` with a **randomly-initialised
  classification head** → weighted F1 ≈ 0.057. README itself notes "the F1 of 0.06
  is genuinely random, not a bug."
- README "Results" headlines **"+0.7738 vs baseline"** — i.e. the fine-tuned model
  vs a random head. Comparing against random is not a credible baseline.
- The fixed stratified split (seed 42, test n=485) is already shared across
  `baseline.py` / `train.py` / `lora.py` / `eval.py`.

## Problem

A domain interviewer asks, predictably:
1. "Your baseline is a random head — what about a **majority-class** predictor?"
2. "Why fine-tune DistilBERT when **FinBERT** (`ProsusAI/finbert`, purpose-built for
   financial sentiment) exists? How do you compare to it **zero-shot**?"

Right now there is no answer in the repo, and the `+0.77` number oversells.

## Goal

Add two credible baselines, evaluated on the **identical** test split, and report
them alongside the existing numbers so the improvement claim is honest.

1. **Majority-class** predictor (always predict the most frequent train label) — the
   true floor for a ~60% neutral dataset.
2. **FinBERT zero-shot** (`ProsusAI/finbert`) — the domain-standard reference; likely
   strong out of the box.

## Non-goals

- Changing the LoRA model or training.
- Temperature scaling (separate follow-up already noted in README).
- Trying 66/75/100-agreement subsets (separate experiment).

## Changes

### 1. `baselines.py` (new; or extend `baseline.py`)

- Reuse the **exact** split function + `RANDOM_SEED = 42` already used by the other
  scripts (do not re-split — apples-to-apples requires the same test indices).
- **Majority-class**: compute the most frequent label on the **train** split, predict
  it for all test rows, report weighted F1, macro F1, per-class.
- **FinBERT zero-shot**: load `ProsusAI/finbert` + its tokenizer; run inference on the
  test split. **Map labels carefully** — FinBERT's id order is
  `{0: positive, 1: negative, 2: neutral}`, which differs from this project's
  `{negative:0, neutral:1, positive:2}`. Remap by label *name*, not index. Report the
  same metrics.
- Write `baseline_results.json` with both baselines' metrics (same schema as
  `lora_results.json` / `eval_results.json`).

### 2. README "Results" table

Expand to four rows, all on the same test split (n=485):

| Model | Weighted F1 | Macro F1 |
|---|---|---|
| Majority-class (all-neutral) | _(fill)_ | _(fill)_ |
| FinBERT zero-shot | _(fill)_ | _(fill)_ |
| DistilBERT (random head) | 0.0571 | 0.1090 |
| **DistilBERT + LoRA (ours)** | **0.8309** | 0.8133 |

- Recompute the headline delta against the **strongest credible baseline**
  (majority or FinBERT), not the random head. State it plainly even if FinBERT is
  close to or above 0.83.

### 3. README "Why DistilBERT + LoRA over FinBERT?" (new short paragraph)

State the honest rationale and the measured delta. Candidate reasons (keep only the
true ones): smaller/faster for CPU serving, full control over the 3-class schema and
calibration analysis, and the exercise of doing the fine-tune end-to-end. **Do not
claim it beats FinBERT unless the numbers show it.**

## Acceptance criteria

- `uv run --with transformers --with torch --with scikit-learn python baselines.py`
  prints majority-class and FinBERT zero-shot weighted/macro F1 on the seed-42 test
  split and writes `baseline_results.json`.
- Both baselines are computed over the **same** test indices as `eval.py` (assert the
  test set size = 485, or reuse the shared split helper directly).
- README "Results" shows ≥4 rows; the headline improvement is stated **vs the
  strongest credible baseline**, and a one-paragraph "why not just FinBERT" answer
  exists with the real delta.

## Interview payoff

Turns "your baseline is random, why not FinBERT?" from a crack into a prepared
answer: *"Majority-class is X, FinBERT zero-shot is Y, my LoRA fine-tune is 0.83;
I chose DistilBERT+LoRA for [smaller/faster/3-class control], and here's how it
actually stacks up against the domain-standard model."*
