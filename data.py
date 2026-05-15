"""
Load FinancialPhraseBank, inspect class distribution, compute class weights.

Output: class_weights dict ready to pass into WeightedTrainer.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; plt.savefig writes the PNG directly
import matplotlib.pyplot as plt
import kagglehub
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight


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

# 1. Load
# FinancialPhraseBank format: 'sentence@label' per line, latin-1 encoding, no header
df = pd.read_csv(
    CSV_PATH,
    sep="@",
    header=None,
    names=["sentence", "label"],
    encoding="latin-1",
)
df["label"] = df["label"].str.strip()

print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print(df.head(3))

# 2. Label encoding
# Trainer expects integer labels; keep explicit order so id2label is stable
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label  = {v: k for k, v in label2id.items()}

df["label_id"] = df["label"].map(label2id)

assert df["label_id"].isna().sum() == 0, "Unknown labels found — check CSV format"

# 3. Class distribution
counts = Counter(df["label"])
total  = len(df)

print("\nClass distribution:")
for lbl in ["neutral", "positive", "negative"]:
    n = counts[lbl]
    print(f"  {lbl:10s}: {n:4d}  ({n / total * 100:.1f}%)")

# 4. Plot
LABEL_ORDER = ["neutral", "positive", "negative"]
COLORS      = ["#4C72B0", "#55A868", "#C44E52"]

values = [counts[l] for l in LABEL_ORDER]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(LABEL_ORDER, values, color=COLORS, edgecolor="white", linewidth=0.8)

for bar, val in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        str(val),
        ha="center", va="bottom", fontsize=11,
    )

ax.set_title("FinancialPhraseBank — Class Distribution (50% agreement split)")
ax.set_ylabel("Count")
ax.set_ylim(0, max(values) * 1.15)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.close(fig)
print("\nSaved: class_distribution.png")

# 5. Class weights
# sklearn 'balanced': weight_i = n_samples / (n_classes * count_i)
# This upweights the minority negative class without duplicate samples.
class_ids = np.array(sorted(id2label.keys()))   # [0, 1, 2]

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=df["label_id"].values,
)

# Dict keyed by integer label id — matches what WeightedTrainer expects
class_weights = {int(i): float(w) for i, w in zip(class_ids, weights)}

print("\nClass weights (balanced):")
for idx, w in class_weights.items():
    print(f"  {id2label[idx]:10s} (id={idx}): {w:.4f}")

# 6. WeightedTrainer stub
# Drop this class into your training script unchanged.
# Pass class_weights via TrainingArguments or directly at instantiation.

import torch
from transformers import Trainer

class WeightedTrainer(Trainer):
    """Trainer subclass that applies per-class weights to cross-entropy loss."""

    def __init__(self, *args, class_weights: dict, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        # Build weight tensor ordered by label id
        weight_tensor = torch.tensor(
            [class_weights[i] for i in sorted(class_weights)],
            dtype=torch.float,
        ).to(device)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


# 7. Export
# These three objects are consumed by the training scripts.

LABEL2ID     = label2id       # {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL     = id2label       # {0: "negative", 1: "neutral", 2: "positive"}
CLASS_WEIGHTS = class_weights  # {0: w0, 1: w1, 2: w2}

if __name__ == "__main__":
    print("\nDeliverables")
    print(f"  LABEL2ID:      {LABEL2ID}")
    print(f"  CLASS_WEIGHTS: {CLASS_WEIGHTS}")
    print(f"  Dataset size:  {total}")
    print("\nAll checks passed.")
