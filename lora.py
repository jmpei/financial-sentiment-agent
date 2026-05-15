"""
LoRA fine-tuning with peft on top of the WeightedTrainer setup.

LoRA config: rank=16, target_modules=["q_lin", "v_lin"]
  - q_lin / v_lin are DistilBERT's query and value projections
  - targeting only these two avoids overfitting on 4,840 samples
  - full fine-tuning of all attention modules would overfit here

Outputs:
  checkpoints/lora/        — adapter weights (adapter_model.safetensors + adapter_config.json)
                             base model weights are NOT duplicated
  lora_results.json        — weighted F1, macro F1, trainable param count, per-class report
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import kagglehub
from peft import LoraConfig, TaskType, get_peft_model
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
MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
RANDOM_SEED = 42        # identical to baseline.py / train.py — same test split
OUTPUT_DIR  = "./checkpoints/lora"

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

# 2. Class weights (train split only)
class_ids   = np.array(sorted(ID2LABEL.keys()))
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

# 4. Base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

# 5. LoRA config
# rank=16: capacity sweet spot for a 4,840-sample dataset.
#   too low (r=4) → underfits; too high (r=32+) → overfits, negates LoRA benefit
#
# target_modules=["q_lin", "v_lin"]: query and value projections only.
#   adding k_lin or out_lin roughly doubles trainable params with no F1 gain
#   on this dataset size — confirmed by the project spec decision.
#
# lora_alpha=32: scaling factor = alpha/rank = 2.0, a common stable default.
#
# bias="none": don't train bias terms — keeps the frozen-trunk guarantee clean.
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(base_model, lora_cfg)
model.to(DEVICE)

# 6. Trainable parameter count
# DistilBERT: ~66.4M total params
# LoRA adds A (768×16) + B (16×768) per targeted projection × 2 modules × 6 layers
# + classification head (768×3 + 3)
# Expected: ~300K trainable / 66.4M total (~0.45%)
def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters : {trainable:>12,}")
    print(f"Total parameters     : {total:>12,}")
    print(f"Trainable ratio      : {100 * trainable / total:.4f}%")
    return trainable, total

trainable_params, total_params = print_trainable_parameters(model)

# 7. WeightedTrainer
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

# 8. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "macro_f1":    f1_score(labels, preds, average="macro"),
    }

# 9. TrainingArguments
# LoRA-specific notes vs full fine-tune:
#   lr: 3e-4 instead of 2e-5 — LoRA adapters are randomly init'd and need a
#       higher lr; the frozen trunk is unaffected by this larger step size.
#   epochs: 10 instead of 5 — LoRA trains far fewer params so convergence is
#       slower per epoch; early stopping prevents overfitting.
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    learning_rate=3e-4,         # higher lr appropriate for LoRA adapters
    weight_decay=0.01,
    warmup_ratio=0.06,
    lr_scheduler_type="linear",

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="weighted_f1",
    greater_is_better=True,
    save_total_limit=2,

    logging_dir="./logs/lora",
    logging_steps=50,
    report_to="none",

    seed=RANDOM_SEED,
    data_seed=RANDOM_SEED,

    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    push_to_hub=False,
)

# 10. Train
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    class_weights=class_weights,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # patience=3 instead of 2: LoRA loss curves are noisier epoch-to-epoch
)

print("\nStarting LoRA training...")
train_result = trainer.train()
print("Training complete.")

# 11. Evaluate on test split
print("\nEvaluating on test split...")
test_output = trainer.predict(test_dataset)

preds       = np.argmax(test_output.predictions, axis=-1)
labels      = test_output.label_ids
label_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]

weighted_f1 = f1_score(labels, preds, average="weighted")
macro_f1    = f1_score(labels, preds, average="macro")
report      = classification_report(labels, preds, target_names=label_names, output_dict=True)

print(f"\nLoRA results (test split)")
print(f"  Weighted F1 : {weighted_f1:.4f}   ← primary metric")
print(f"  Macro F1    : {macro_f1:.4f}   ← recorded, not reported on resume")
print(f"\n{classification_report(labels, preds, target_names=label_names)}")

# 12. Save results
results = {
    "model":             MODEL_NAME,
    "fine_tuned":        True,
    "lora":              True,
    "lora_rank":         16,
    "lora_alpha":        32,
    "lora_target_modules": ["q_lin", "v_lin"],
    "trainable_params":  trainable_params,
    "total_params":      total_params,
    "trainable_ratio_pct": round(100 * trainable_params / total_params, 4),
    "test_size":         len(test_df),
    "weighted_f1":       round(weighted_f1, 4),
    "macro_f1":          round(macro_f1, 4),
    "per_class": {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall":    round(report[lbl]["recall"],    4),
            "f1":        round(report[lbl]["f1-score"],  4),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in label_names
    },
    "train_runtime_s": round(train_result.metrics.get("train_runtime", 0), 1),
    "train_loss":      round(train_result.metrics.get("train_loss", 0), 4),
    "split_seed":      RANDOM_SEED,
}

with open("lora_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved: lora_results.json")

# peft saves adapter weights only — base model weights are not duplicated
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved: LoRA adapter → {OUTPUT_DIR}/")
print(f"  adapter_model.safetensors  (~{trainable_params * 4 / 1e6:.1f} MB, adapter only)")
print(f"  adapter_config.json")

# 13. Three-way comparison
print(f"\nResults comparison")
rows = [("LoRA fine-tune", weighted_f1, macro_f1)]

for fname, label in [
    ("baseline_results.json", "Baseline              "),
    ("train_results.json",    "Full fine-tune        "),
]:
    try:
        with open(fname) as f:
            d = json.load(f)
        rows.append((label, d["weighted_f1"], d["macro_f1"]))
    except FileNotFoundError:
        pass

rows.sort(key=lambda x: x[1])
print(f"  {'Model':<26}  Weighted F1  Macro F1")
print(f"  {'─'*26}  {'─'*11}  {'─'*8}")
for name, wf1, mf1 in rows:
    print(f"  {name:<26}  {wf1:.4f}       {mf1:.4f}")
