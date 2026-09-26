"""
Domain-shift eval: does the model trained on FinancialPhraseBank hold up on the
news the agent actually feeds it?

FinancialPhraseBank is sentences from company reports; the agent scores NewsAPI
headlines + descriptions. domain_shift_headlines.jsonl is a frozen snapshot of
live search_news results (scripts/fetch_headlines.py), labelled with the
FinancialPhraseBank guideline: from an investor's point of view, is this news
positive, negative or neutral for the company or market it is about. Rows
labelled off_topic (not about a company, market or economy) or duplicate (same
story as an earlier row) are excluded — the agent skips off-topic results.

The same three models as the in-domain table run on the same rows: our LoRA
model, FinBERT zero-shot, and the majority class. FinBERT's FinancialPhraseBank
advantage (it was trained on it) does not carry over to fresh news.

`label_source` records who labelled each row; the output reports the counts,
so a number from unreviewed labels is never mistaken for a hand-labelled one.

Output:
  domain_shift_results.json
"""

import json
from collections import Counter

import numpy as np
import torch
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DATA_PATH    = "domain_shift_headlines.jsonl"
BASE_MODEL   = "distilbert-base-uncased"
ADAPTER_DIR  = "./checkpoints/lora"
FINBERT_NAME = "ProsusAI/finbert"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
N_BOOTSTRAP  = 10_000
RANDOM_SEED  = 42

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = [ID2LABEL[i] for i in sorted(ID2LABEL)]   # ["negative", "neutral", "positive"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# 1. Labelled rows
rows = [json.loads(line) for line in open(DATA_PATH)]
kept = [r for r in rows if r["label"] in LABEL2ID]
texts  = [r["text"] for r in kept]
y_true = [LABEL2ID[r["label"]] for r in kept]
print(f"{len(rows)} fetched, {len(kept)} labelled positive/negative/neutral "
      f"({Counter(r['label'] for r in rows)})")


def _predict(tokenizer, model, to_project) -> list:
    preds = []
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            enc = tokenizer(
                texts[start:start + BATCH_SIZE],
                truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
            ).to(DEVICE)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            preds.extend(to_project[i] for i in logits.argmax(dim=-1).cpu().tolist())
    return preds


def _metrics(y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES,
        output_dict=True, zero_division=0,
    )
    return {
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "macro_f1":    round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "per_class": {
            lbl: {
                "precision": round(report[lbl]["precision"], 4),
                "recall":    round(report[lbl]["recall"], 4),
                "f1":        round(report[lbl]["f1-score"], 4),
                "support":   int(report[lbl]["support"]),
            }
            for lbl in LABEL_NAMES
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


# 2. LoRA model (temperature does not change the argmax, so it is not applied)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()
lora_pred = _predict(tokenizer, model, {i: i for i in ID2LABEL})
lora = _metrics(lora_pred)

# 3. FinBERT zero-shot — remap its label ids by name, as baselines.py does
fin_tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)
fin_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_NAME).to(DEVICE).eval()
fin_to_project = {int(i): LABEL2ID[name.lower()] for i, name in fin_model.config.id2label.items()}
fin_pred = _predict(fin_tokenizer, fin_model, fin_to_project)
finbert = _metrics(fin_pred)

# 4. Majority class of the FinancialPhraseBank train split (neutral, see baselines.py)
majority = _metrics([LABEL2ID["neutral"]] * len(kept))

# 5. Paired bootstrap: resample rows, recompute both models' weighted F1 on the same rows
rng = np.random.default_rng(RANDOM_SEED)
y, a, b = np.array(y_true), np.array(lora_pred), np.array(fin_pred)
lora_f1s, diffs = [], []
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(y), len(y))
    fa = f1_score(y[idx], a[idx], average="weighted", zero_division=0)
    fb = f1_score(y[idx], b[idx], average="weighted", zero_division=0)
    lora_f1s.append(fa)
    diffs.append(fa - fb)
bootstrap = {
    "resamples":             N_BOOTSTRAP,
    "lora_weighted_f1_ci95": [round(float(q), 4) for q in np.percentile(lora_f1s, [2.5, 97.5])],
    "lora_minus_finbert_ci95": [round(float(q), 4) for q in np.percentile(diffs, [2.5, 97.5])],
}

results = {
    "n":             len(kept),
    "label_counts":  dict(Counter(r["label"] for r in kept)),
    "label_sources": dict(Counter(r["label_source"] for r in kept)),
    "models": {
        "lora":              lora,
        "finbert_zero_shot": finbert,
        "majority_class":    majority,
    },
    "bootstrap": bootstrap,
}
for name, m in results["models"].items():
    print(f"{name:18s} weighted F1 {m['weighted_f1']:.4f}  macro F1 {m['macro_f1']:.4f}")
print(f"bootstrap 95% CI — LoRA weighted F1 {bootstrap['lora_weighted_f1_ci95']}, "
      f"LoRA − FinBERT {bootstrap['lora_minus_finbert_ci95']}")
print(f"label sources: {results['label_sources']}")

with open("domain_shift_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: domain_shift_results.json")
