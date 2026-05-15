"""
Full evaluation of the LoRA fine-tuned model.

Inference-only — no retraining. Reuses the exact same stratified test split
as baseline.py / train.py / lora.py (seed 42), so weighted F1 here should
match lora_results.json within floating-point noise.

Outputs (all under outputs/):
  confusion_matrix.png   — sklearn confusion matrix, counts annotated
  calibration_curve.png  — one-vs-rest calibration curve per class
  eval_results.json      — weighted F1, macro F1, per-class report, calibration notes
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import kagglehub
from peft import PeftModel
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
BASE_MODEL  = "distilbert-base-uncased"
ADAPTER_DIR = "./checkpoints/lora"
OUTPUT_DIR  = "./outputs"
MAX_LENGTH  = 128
BATCH_SIZE  = 32
RANDOM_SEED = 42                                # identical to earlier scripts

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Reconstruct the exact same test split
df = pd.read_csv(
    CSV_PATH,
    sep="@",
    header=None,
    names=["sentence", "label"],
    encoding="latin-1",
)
df["label"] = df["label"].str.strip()
df["label_id"] = df["label"].map(LABEL2ID)

train_val_df, test_df = train_test_split(
    df, test_size=0.10, stratify=df["label_id"], random_state=RANDOM_SEED
)
# val split is not used here, but the call is kept to consume the same seed state
_, _ = train_test_split(
    train_val_df,
    test_size=0.10 / 0.90,
    stratify=train_val_df["label_id"],
    random_state=RANDOM_SEED,
)

print(f"Test split size: {len(test_df)}")

# 2. Tokenizer & dataset
# Tokenizer comes from the LoRA save dir so it matches what training used.
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)


class SentenceDataset(Dataset):
    def __init__(self, sentences, label_ids):
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
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


test_dataset = SentenceDataset(test_df["sentence"].tolist(), test_df["label_id"].tolist())
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Load base model + LoRA adapter
# peft stores only the adapter; we attach it on top of a freshly loaded base.
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.to(DEVICE)
model.eval()

# 4. Inference
all_probs  = []   # softmax probabilities, shape (N, 3)
all_preds  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"]

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        preds  = probs.argmax(axis=-1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

all_probs  = np.concatenate(all_probs, axis=0)        # (N, 3)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# 5. Metrics
weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
macro_f1    = f1_score(all_labels, all_preds, average="macro")
report      = classification_report(
    all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True
)

print(f"\nLoRA eval (test split)")
print(f"  Weighted F1 : {weighted_f1:.4f}")
print(f"  Macro F1    : {macro_f1:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=LABEL_NAMES)}")

# 6. Confusion matrix
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(LABEL_NAMES)
ax.set_yticklabels(LABEL_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix — LoRA fine-tuned DistilBERT")

# Annotate each cell with the count; flip text color on dark cells
thresh = cm.max() / 2.0
for i in range(3):
    for j in range(3):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=12,
        )

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"Saved: {cm_path}")

# 7. Calibration curve (one-vs-rest per class)
# For each class, treat it as a binary problem:
#   y_true = (all_labels == class_id), y_prob = all_probs[:, class_id]
# Then plot predicted probability vs actual frequency in each bin.
fig, ax = plt.subplots(figsize=(6, 5))
colors = {"negative": "#C44E52", "neutral": "#4C72B0", "positive": "#55A868"}

calibration_summary = {}     # for the JSON output

for class_id, name in ID2LABEL.items():
    y_true = (all_labels == class_id).astype(int)
    y_prob = all_probs[:, class_id]

    # n_bins=10; strategy='uniform' = equal-width bins on [0, 1]
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label=name, color=colors[name])

    # Summary: mean predicted confidence for samples this class actually predicted,
    # and the actual accuracy on those samples.
    mask = (all_preds == class_id)
    if mask.sum() > 0:
        mean_conf = float(all_probs[mask, class_id].mean())
        accuracy  = float((all_labels[mask] == class_id).mean())
        calibration_summary[name] = {
            "n_predicted":            int(mask.sum()),
            "mean_predicted_conf":    round(mean_conf, 4),
            "actual_accuracy":        round(accuracy, 4),
            "overconfidence_gap":     round(mean_conf - accuracy, 4),
        }

# Perfect-calibration reference line
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
ax.set_xlabel("Predicted confidence")
ax.set_ylabel("Actual accuracy")
ax.set_title("Confidence Calibration — One-vs-Rest per Class")
ax.legend(loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
plt.tight_layout()
cal_path = os.path.join(OUTPUT_DIR, "calibration_curve.png")
plt.savefig(cal_path, dpi=150)
plt.close(fig)
print(f"Saved: {cal_path}")

# Pick the class with the largest overconfidence gap for the interview-ready note
most_overconfident = max(
    calibration_summary.items(),
    key=lambda kv: kv[1]["overconfidence_gap"],
)
class_name, stats = most_overconfident
calibration_notes = (
    f"{class_name}: mean predicted confidence {stats['mean_predicted_conf']:.2f} "
    f"vs actual accuracy {stats['actual_accuracy']:.2f} "
    f"(gap {stats['overconfidence_gap']:+.2f}). "
    f"Temperature scaling could correct this."
)
print(f"\nCalibration note: {calibration_notes}")

# 8. Save eval_results.json
results = {
    "model":         BASE_MODEL,
    "adapter_dir":   ADAPTER_DIR,
    "fine_tuned":    True,
    "lora":          True,
    "test_size":     len(test_df),
    "weighted_f1":   round(weighted_f1, 4),
    "macro_f1":      round(macro_f1, 4),
    "per_class": {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall":    round(report[lbl]["recall"],    4),
            "f1":        round(report[lbl]["f1-score"],  4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in LABEL_NAMES
    },
    "calibration":         calibration_summary,
    "calibration_notes":   calibration_notes,
    "confusion_matrix":    cm.tolist(),
    "split_seed":          RANDOM_SEED,
}

results_path = os.path.join(OUTPUT_DIR, "eval_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {results_path}")

# 9. Sanity check vs lora.py
try:
    with open("lora_results.json") as f:
        lora_run = json.load(f)
    delta = abs(weighted_f1 - lora_run["weighted_f1"])
    print(f"\nSanity check vs lora_results.json")
    print(f"  lora.py weighted F1 : {lora_run['weighted_f1']:.4f}")
    print(f"  eval.py weighted F1 : {weighted_f1:.4f}")
    print(f"  |Δ|                 : {delta:.4f}")
    if delta > 0.001:
        print("  ⚠️  Drift > 0.001 — check that the test split / model load matches.")
    else:
        print("  ✓ Match within tolerance.")
except FileNotFoundError:
    print("\nlora_results.json not found — run lora.py before eval.py.")
