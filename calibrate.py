"""
Temperature scaling for the LoRA fine-tuned model.

eval.py found the model overconfident. Temperature scaling divides every logit
by one scalar T before the softmax: T > 1 softens confidences, and because the
argmax is unchanged, predictions (and so F1) are identical. T is fit by
minimising negative log-likelihood on the val split, then judged on test.

Same seed-42 split as lora.py / eval.py. Inference-only.

Output:
  calibration_results.json — T, ECE / NLL before and after on test, and the
                             per-class confidence-vs-precision summary eval.py reports
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import kagglehub
from peft import PeftModel
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# 0. Resolve dataset path via kagglehub
def _load_phrasebank_50agree() -> str:
    """Download FinancialPhraseBank from Kaggle and return the 50%-agreement file path."""
    cache_dir = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
    for root, _, files in os.walk(cache_dir):
        for name in files:
            if name.lower() in ("sentences_50agree.txt", "sentences_50agree.csv"):
                return os.path.join(root, name)
    raise FileNotFoundError(f"50%-agreement file not found under {cache_dir}")


CSV_PATH = _load_phrasebank_50agree()

# Config
BASE_MODEL  = "distilbert-base-uncased"
ADAPTER_DIR = "./checkpoints/lora"
MAX_LENGTH  = 128
BATCH_SIZE  = 32
RANDOM_SEED = 42                                # identical to earlier scripts
N_BINS      = 15                                # ECE bins, equal width on [0, 1]

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# 1. Reconstruct the exact same val and test splits
df = pd.read_csv(CSV_PATH, sep="@", header=None, names=["sentence", "label"], encoding="latin-1")
df["label"] = df["label"].str.strip()
df["label_id"] = df["label"].map(LABEL2ID)

train_val_df, test_df = train_test_split(
    df, test_size=0.10, stratify=df["label_id"], random_state=RANDOM_SEED
)
_, val_df = train_test_split(
    train_val_df,
    test_size=0.10 / 0.90,
    stratify=train_val_df["label_id"],
    random_state=RANDOM_SEED,
)
assert len(test_df) == 485, f"Expected test size 485, got {len(test_df)} — split drifted"
print(f"Split — val: {len(val_df)}, test: {len(test_df)}")

# 2. Model
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR).to(DEVICE).eval()


def _logits(sentences) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, len(sentences), BATCH_SIZE):
            enc = tokenizer(
                sentences[start:start + BATCH_SIZE],
                truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt",
            ).to(DEVICE)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


val_logits,  val_y  = _logits(val_df["sentence"].tolist()),  val_df["label_id"].to_numpy()
test_logits, test_y = _logits(test_df["sentence"].tolist()), test_df["label_id"].to_numpy()


# 3. Metrics
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits, y, T) -> float:
    p = _softmax(logits / T)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def _ece(probs, y) -> float:
    """Expected calibration error on the top-label confidence."""
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def _per_class(probs, y) -> dict:
    """For each predicted class: mean confidence vs precision (eval.py's summary)."""
    pred = probs.argmax(axis=1)
    out = {}
    for cid, name in ID2LABEL.items():
        mask = pred == cid
        conf, prec = float(probs[mask, cid].mean()), float((y[mask] == cid).mean())
        out[name] = {
            "n_predicted":         int(mask.sum()),
            "mean_predicted_conf": round(conf, 4),
            "precision":           round(prec, 4),
            "overconfidence_gap":  round(conf - prec, 4),
        }
    return out


def _report(T) -> dict:
    probs = _softmax(test_logits / T)
    return {
        "ece":         round(_ece(probs, test_y), 4),
        "nll":         round(_nll(test_logits, test_y, T), 4),
        "weighted_f1": round(f1_score(test_y, probs.argmax(axis=1), average="weighted"), 4),
        "per_class":   _per_class(probs, test_y),
    }


# 4. Fit T on val (1-D, bounded — NLL is smooth in T)
fit = minimize_scalar(lambda T: _nll(val_logits, val_y, T), bounds=(0.05, 20.0), method="bounded")
T = round(float(fit.x), 4)
print(f"Fitted temperature on val: T = {T}")

before, after = _report(1.0), _report(T)
assert before["weighted_f1"] == after["weighted_f1"], "temperature must not change predictions"

for tag, r in (("before (T=1)", before), (f"after  (T={T})", after)):
    print(f"\n{tag}: ECE {r['ece']:.4f}  NLL {r['nll']:.4f}  weighted F1 {r['weighted_f1']:.4f}")
    for name, s in r["per_class"].items():
        print(f"  predicted {name:8s} n={s['n_predicted']:3d}  conf {s['mean_predicted_conf']:.2f}"
              f"  precision {s['precision']:.2f}  gap {s['overconfidence_gap']:+.2f}")

# 5. Save
results = {
    "temperature": T,
    "fit_on":      f"val split (n={len(val_df)}), minimise NLL",
    "evaluated_on": f"test split (n={len(test_df)})",
    "ece_bins":    N_BINS,
    "before":      before,
    "after":       after,
    "split_seed":  RANDOM_SEED,
}
with open("calibration_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: calibration_results.json")
