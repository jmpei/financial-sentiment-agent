"""
Full DistilBERT fine-tuning (no LoRA) with WeightedTrainer.

Verifies the training loop, class weights, and metrics pipeline end-to-end
before introducing peft / LoRA. Kept as a comparison run.

Outputs:
  checkpoints/full_finetune/   — HF checkpoint (best by weighted F1)
  train_results.json           — weighted F1, macro F1, per-class report
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import kagglehub
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


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
MODEL_NAME   = "distilbert-base-uncased"
MAX_LENGTH   = 128
RANDOM_SEED  = 42          # must match baseline.py — same test split
OUTPUT_DIR   = "./checkpoints/full_finetune"

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
# Identical split to baseline.py — test set is the same rows.
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
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.10 / 0.90,
    stratify=train_val_df["label_id"],
    random_state=RANDOM_SEED,
)

print(f"Split — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

# 2. Class weights
# Computed on TRAIN split only — never let test distribution leak into weights.
class_ids = np.array(sorted(ID2LABEL.keys()))
raw_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=train_df["label_id"].values,
)
class_weights = {int(i): float(w) for i, w in zip(class_ids, raw_weights)}

print("\nClass weights (from train split):")
for idx, w in class_weights.items():
    print(f"  {ID2LABEL[idx]:10s} (id={idx}): {w:.4f}")

# 3. Tokenizer & Dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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


train_dataset = SentenceDataset(train_df["sentence"].tolist(), train_df["label_id"].tolist())
val_dataset   = SentenceDataset(val_df["sentence"].tolist(),   val_df["label_id"].tolist())
test_dataset  = SentenceDataset(test_df["sentence"].tolist(),  test_df["label_id"].tolist())

# 4. Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
model.to(DEVICE)

# 5. WeightedTrainer
class WeightedTrainer(Trainer):
    """CrossEntropyLoss with per-class weights to counter class imbalance."""

    def __init__(self, *args, class_weights: dict, **kwargs):
        super().__init__(*args, **kwargs)
        weight_tensor = torch.tensor(
            [class_weights[i] for i in sorted(class_weights)],
            dtype=torch.float,
        ).to(DEVICE)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# 6. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        # weighted_f1 is the primary metric — used for early stopping & resume
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        # macro_f1 recorded for completeness, not reported on resume
        "macro_f1":    f1_score(labels, preds, average="macro"),
    }

# 7. TrainingArguments
# ~3,876 train samples / batch_size 16 ≈ 243 steps/epoch
# 5 epochs = ~1,215 steps; warmup over first ~73 steps (6%)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # epochs & batch
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    # optimiser
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,          # ~1/3 of first epoch
    lr_scheduler_type="linear",

    # eval & checkpointing
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",
    greater_is_better=True,
    save_total_limit=2,         # keep only best + latest checkpoint

    # logging
    logging_dir="./logs/full_finetune",
    logging_steps=50,
    report_to="none",           # set to "wandb" if you want experiment tracking

    # reproducibility
    seed=RANDOM_SEED,
    data_seed=RANDOM_SEED,

    # Colab / CPU fallback
    fp16=torch.cuda.is_available(),   # mixed precision on GPU only
    dataloader_num_workers=0,          # 0 is safest on Colab
    push_to_hub=False,
)

# 8. Train
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    class_weights=class_weights,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[
        # Stop if val weighted_f1 doesn't improve for 2 epochs
        EarlyStoppingCallback(early_stopping_patience=2),
    ],
)

print("\nStarting training...")
train_result = trainer.train()
print("Training complete.")

# 9. Evaluate on test split
print("\nEvaluating on test split...")
test_output = trainer.predict(test_dataset)

preds  = np.argmax(test_output.predictions, axis=-1)
labels = test_output.label_ids

label_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]
weighted_f1 = f1_score(labels, preds, average="weighted")
macro_f1    = f1_score(labels, preds, average="macro")
report      = classification_report(labels, preds, target_names=label_names, output_dict=True)

print(f"\nFull fine-tune results (test split)")
print(f"  Weighted F1 : {weighted_f1:.4f}   ← primary metric")
print(f"  Macro F1    : {macro_f1:.4f}   ← recorded, not reported on resume")
print(f"\n{classification_report(labels, preds, target_names=label_names)}")

# 10. Save results
results = {
    "model":       MODEL_NAME,
    "fine_tuned":  True,
    "lora":        False,
    "test_size":   len(test_df),
    "weighted_f1": round(weighted_f1, 4),
    "macro_f1":    round(macro_f1, 4),
    "per_class":   {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall":    round(report[lbl]["recall"],    4),
            "f1":        round(report[lbl]["f1-score"],  4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in label_names
    },
    "train_runtime_s":   round(train_result.metrics.get("train_runtime", 0), 1),
    "train_loss":        round(train_result.metrics.get("train_loss", 0), 4),
    "split_seed":        RANDOM_SEED,
}

with open("train_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved: train_results.json")

# Save the best model + tokenizer together so eval.py can load them
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved: model + tokenizer → {OUTPUT_DIR}/")

# 11. Quick comparison with baseline
try:
    with open("baseline_results.json") as f:
        baseline = json.load(f)
    delta = weighted_f1 - baseline["weighted_f1"]
    print(f"\nvs. Baseline")
    print(f"  Baseline weighted F1 : {baseline['weighted_f1']:.4f}")
    print(f"  Full fine-tune       : {weighted_f1:.4f}")
    print(f"  Delta                : {delta:+.4f}")
except FileNotFoundError:
    print("\nbaseline_results.json not found — run baseline.py first.")
