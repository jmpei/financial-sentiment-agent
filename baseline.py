"""
Baseline evaluation: distilbert-base-uncased with NO fine-tuning.

Establishes the pre-training floor to compare against the LoRA fine-tuned model.
The classification head is randomly initialized — this is intentional.
Expected weighted F1 ~0.65 (compare against ~0.91 post fine-tuning).

Outputs:
  test_split.csv        — fixed test set reused for all future evaluations
  baseline_results.json — weighted F1, macro F1, per-class report
"""

import json
import os
import time
import numpy as np
import pandas as pd
import torch
import kagglehub
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
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
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
BATCH_SIZE  = 32
RANDOM_SEED = 42

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# 1. Load & split
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

# Stratified 80 / 10 / 10 split — fixed seed so every script uses the same test set
train_val_df, test_df = train_test_split(
    df, test_size=0.10, stratify=df["label_id"], random_state=RANDOM_SEED
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.10 / 0.90,   # 10% of original
    stratify=train_val_df["label_id"],
    random_state=RANDOM_SEED,
)

print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# Persist test split so fine-tuned model evaluates on the exact same rows
test_df.to_csv("test_split.csv", index=True)
print("Saved: test_split.csv")

# Verify stratification held
for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    dist = split_df["label"].value_counts(normalize=True).to_dict()
    print(f"  {split_name}: { {k: f'{v:.2f}' for k, v in dist.items()} }")

# 2. Dataset & DataLoader
class SentenceDataset(Dataset):
    def __init__(self, sentences, label_ids, tokenizer, max_length):
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(label_ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_dataset = SentenceDataset(
    sentences=test_df["sentence"].tolist(),
    label_ids=test_df["label_id"].tolist(),
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Load base model (no fine-tuning)
# num_labels=3 adds a randomly initialised classification head on top of the
# pretrained DistilBERT trunk — this is the baseline we're measuring.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,   # head is new; suppress HF warning
)
model.to(DEVICE)
model.eval()

# 4. Inference
all_preds  = []
all_labels = []

t0 = time.perf_counter()

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds  = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

elapsed_ms = (time.perf_counter() - t0) * 1000
per_sample_ms = elapsed_ms / len(test_df)

print(f"\nInference: {elapsed_ms:.0f}ms total, {per_sample_ms:.1f}ms/sample")

# 5. Metrics
label_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]

weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
macro_f1    = f1_score(all_labels, all_preds, average="macro")
report      = classification_report(
    all_labels, all_preds,
    target_names=label_names,
    output_dict=True,
)

print(f"\nBaseline results")
print(f"  Weighted F1 : {weighted_f1:.4f}   ← primary metric (goes on resume)")
print(f"  Macro F1    : {macro_f1:.4f}   ← recorded, not reported on resume")
print(f"\n{classification_report(all_labels, all_preds, target_names=label_names)}")

# 6. Save results
results = {
    "model":          MODEL_NAME,
    "fine_tuned":     False,
    "test_size":      len(test_df),
    "weighted_f1":    round(weighted_f1, 4),
    "macro_f1":       round(macro_f1, 4),
    "per_class":      {
        label: {
            "precision": round(report[label]["precision"], 4),
            "recall":    round(report[label]["recall"], 4),
            "f1":        round(report[label]["f1-score"], 4),
            "support":   int(report[label]["support"]),
        }
        for label in label_names
    },
    "split_seed":     RANDOM_SEED,
    "inference_ms_total":     round(elapsed_ms, 1),
    "inference_ms_per_sample": round(per_sample_ms, 2),
}

with open("baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved: baseline_results.json")
