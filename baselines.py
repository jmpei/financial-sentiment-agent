"""
Honest baselines on the *identical* seed-42 test split.

The random-head DistilBERT in baseline.py is not a credible reference for a
finance reviewer. This script adds the two baselines an interviewer will ask for:

  1. Majority-class  — always predict the most frequent *train* label. The true
     floor for a ~60% neutral dataset.
  2. FinBERT zero-shot (`ProsusAI/finbert`) — the domain-standard model, no
     fine-tuning on our part.

Both run on the same test indices as baseline.py / lora.py / eval.py (seed 42,
n=485), so weighted F1 is directly comparable to lora_results.json.

Output:
  baselines_results.json — majority-class + FinBERT metrics (per-class included)

Note: this writes baselines_results.json, NOT baseline_results.json. The latter
holds the random-head run and is consumed by train.py / lora.py with a flat
schema; it is left untouched.
"""

import json
import os
import time
import pandas as pd
import torch
import kagglehub
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 0. Resolve dataset path via kagglehub
def _load_phrasebank_50agree() -> str:
    """Download FinancialPhraseBank from Kaggle and return the 50%-agreement file path.

    Kaggle dataset layout (version 5):
      {cache}/FinancialPhraseBank/Sentences_50Agree.txt   ← what we want
      {cache}/all-data.csv                                 ← combined, do not use
    We walk the cache dir to stay robust if the upstream layout shifts.
    """
    cache_dir = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    for root, _, files in os.walk(cache_dir):
        for name in files:
            if name.lower() in ("sentences_50agree.txt", "sentences_50agree.csv"):
                return os.path.join(root, name)
    raise FileNotFoundError(f"50%-agreement file not found under {cache_dir}")


CSV_PATH = _load_phrasebank_50agree()
print(f"Dataset: {CSV_PATH}")

# Config
FINBERT_NAME = "ProsusAI/finbert"
MAX_LENGTH   = 128
BATCH_SIZE   = 32
RANDOM_SEED  = 42                                # identical to baseline.py / lora.py / eval.py

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = [ID2LABEL[i] for i in sorted(ID2LABEL)]   # ["negative", "neutral", "positive"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# 1. Reconstruct the exact same stratified split (seed 42)
df = pd.read_csv(
    CSV_PATH,
    sep="@",
    header=None,
    names=["sentence", "label"],
    encoding="latin-1",
)
df["label"] = df["label"].str.strip()
df["label_id"] = df["label"].map(LABEL2ID)
assert df["label_id"].isna().sum() == 0, "Unknown labels — check CSV format"

train_val_df, test_df = train_test_split(
    df, test_size=0.10, stratify=df["label_id"], random_state=RANDOM_SEED
)
train_df, _ = train_test_split(
    train_val_df,
    test_size=0.10 / 0.90,
    stratify=train_val_df["label_id"],
    random_state=RANDOM_SEED,
)

print(f"Split — train: {len(train_df)}, test: {len(test_df)}")
assert len(test_df) == 485, f"Expected test size 485, got {len(test_df)} — split drifted"

y_true = test_df["label_id"].tolist()


def _metrics(y_pred) -> dict:
    """Weighted/macro F1 + per-class report in the same schema as lora_results.json."""
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
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
    }


# 2. Majority-class baseline
# Most frequent label on the TRAIN split only — never peek at the test distribution.
majority_label = train_df["label"].value_counts().idxmax()
majority_id    = LABEL2ID[majority_label]
print(f"\nMajority class (train): {majority_label} (id={majority_id})")

majority_pred = [majority_id] * len(test_df)
majority_metrics = _metrics(majority_pred)
majority_metrics["predicted_label"] = majority_label
print(f"  Weighted F1 : {majority_metrics['weighted_f1']:.4f}")
print(f"  Macro F1    : {majority_metrics['macro_f1']:.4f}")

# 3. FinBERT zero-shot baseline
print(f"\nLoading {FINBERT_NAME} ...")
fin_tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)
fin_model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_NAME)
fin_model.to(DEVICE)
fin_model.eval()

# FinBERT's id order is {0: positive, 1: negative, 2: neutral} — DIFFERENT from
# this project's {negative:0, neutral:1, positive:2}. Remap by label *name*, read
# from the model's own config so we never rely on a hardcoded index order.
fin_id2name = {int(i): name.lower() for i, name in fin_model.config.id2label.items()}
print(f"  FinBERT id2label: {fin_id2name}")
fin_to_project = {fin_id: LABEL2ID[name] for fin_id, name in fin_id2name.items()}

sentences = test_df["sentence"].tolist()
fin_pred = []
t0 = time.perf_counter()
with torch.no_grad():
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start:start + BATCH_SIZE]
        enc = fin_tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(DEVICE)
        logits = fin_model(**enc).logits
        fin_ids = logits.argmax(dim=-1).cpu().tolist()
        fin_pred.extend(fin_to_project[i] for i in fin_ids)
elapsed_ms = (time.perf_counter() - t0) * 1000

finbert_metrics = _metrics(fin_pred)
finbert_metrics["model"] = FINBERT_NAME
print(f"  Inference: {elapsed_ms:.0f}ms total over {len(sentences)} samples")
print(f"  Weighted F1 : {finbert_metrics['weighted_f1']:.4f}")
print(f"  Macro F1    : {finbert_metrics['macro_f1']:.4f}")
print(f"\n{classification_report(y_true, fin_pred, labels=[0, 1, 2], target_names=LABEL_NAMES, zero_division=0)}")

# 4. Save
results = {
    "split_seed": RANDOM_SEED,
    "test_size":  len(test_df),
    "baselines": {
        "majority_class":    majority_metrics,
        "finbert_zero_shot": finbert_metrics,
    },
}
with open("baselines_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: baselines_results.json")
