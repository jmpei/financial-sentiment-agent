"""
FastAPI service for the LoRA fine-tuned DistilBERT.

The model is loaded once at startup via FastAPI's lifespan event and kept on
app.state. Per-request reload would push p95 latency into seconds.

POST /predict
  request  : {"text": "..."}
  response : {"label": "positive" | "negative" | "neutral",
              "confidence": float, "latency_ms": float}
"""

import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Config
BASE_MODEL  = "distilbert-base-uncased"
ADAPTER_DIR = "checkpoints/lora"
MAX_LENGTH  = 128

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# Lifespan: load model and tokenizer once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR).eval().to(DEVICE)

    app.state.tokenizer = tokenizer
    app.state.model = model
    app.state.device = DEVICE
    print(f"Model loaded on {DEVICE}")
    yield


app = FastAPI(
    title="Financial Sentiment API",
    version="1.0",
    lifespan=lifespan,
)


# Pydantic schemas
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    latency_ms: float


# Endpoints
@app.get("/healthz")
def healthz():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    tokenizer = request.app.state.tokenizer
    model     = request.app.state.model
    device    = request.app.state.device

    start = time.perf_counter()

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    ).to(device)

    # Pass only the fields DistilBERT.forward accepts — newer tokenizer versions
    # also produce `token_type_ids`, which DistilBERT rejects.
    with torch.no_grad():
        logits = model(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        ).logits
        probs  = torch.softmax(logits, dim=-1)[0]
        idx    = int(probs.argmax().item())

    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        label=ID2LABEL[idx],
        confidence=float(probs[idx].item()),
        latency_ms=round(latency_ms, 2),
    )
