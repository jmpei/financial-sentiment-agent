# Financial Sentiment Agent

## Overview

A **two-stage** financial NLP project. Project 1 trains and deploys a sentiment model; Project 2 consumes it inside an LLM agent.

1. **Project 1 — Model**: Fine-tune DistilBERT on FinancialPhraseBank using LoRA, deploy as a FastAPI service (`POST /predict`).
2. **Project 2 — Agent**: A LangChain agent that, given a financial question, calls NewsAPI for fresh articles, calls Project 1's `/predict` to score each article, then synthesizes a final answer via OpenAI.

The two projects share one service contract: Project 2 calls Project 1's HTTP endpoint. Do not re-implement sentiment logic inside the agent.

---

## Repository Structure

```
financial-sentiment-agent/
├── data.py                 # Data load, class distribution, class weights, WeightedTrainer
├── baseline.py             # Baseline evaluation: DistilBERT with no fine-tuning
├── train.py                # Full fine-tune with WeightedTrainer (no LoRA, comparison run)
├── lora.py                 # LoRA fine-tune (rank=16, q_lin / v_lin)
├── eval.py                 # Confusion matrix + calibration curve for fine-tuned model
├── api/
│   └── main.py             # FastAPI service with lifespan model loading
├── Dockerfile              # Container for the API
├── spaces/
│   └── app.py              # Gradio app for HuggingFace Spaces
├── src/                    # Agent
│   ├── agent.py            # AgentExecutor + tool registration
│   ├── tools.py            # search_news, analyze_sentiment
│   └── prompts.py          # SYSTEM_PROMPT
├── tests/
│   └── test_agent.py       # pytest with mocked tools
├── checkpoints/lora/       # adapter_model.safetensors + adapter_config.json (generated)
├── outputs/                # confusion_matrix.png, calibration_curve.png, *.json (generated)
├── .env.example            # OPENAI_API_KEY, NEWS_API_KEY, SENTIMENT_SERVICE_URL
├── .gitignore
├── requirements.txt
├── Process.md              # Internal build playbook (not user-facing)
├── CLAUDE.md               # This file
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | DistilBERT (`distilbert-base-uncased`) |
| Fine-tuning | LoRA via `peft` (rank=16, target=`q_lin`,`v_lin`) |
| Training framework | HuggingFace Transformers `Trainer` (subclassed as `WeightedTrainer`) |
| Dataset | FinancialPhraseBank, `sentences_50agree.csv`, 4,840 sentences |
| API | FastAPI + uvicorn |
| Container | Docker (`python:3.10-slim`) |
| Public demo | HuggingFace Spaces + Gradio |
| Agent framework | LangChain (`AgentExecutor`, ReAct style) |
| News source | NewsAPI |
| LLM | OpenAI API (`gpt-4o-mini`) |
| Testing | pytest + `unittest.mock` |
| Runtime | Python 3.10/3.11, venv |

---

## Project 1 — Model Spec

### Dataset
- File: `Sentences_50Agree.txt` (50% annotator agreement). Auto-downloaded by every script via `kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")` — the file lives under `~/.cache/kagglehub/.../FinancialPhraseBank/`. Do **not** use the 66%, 75%, or `all-data.csv` versions — numbers are not comparable.
- 3-class: `positive` / `negative` / `neutral`. Class distribution roughly: neutral ~60%, positive ~28%, negative ~12%.
- Stratified 80 / 10 / 10 split, `random_state=42` is the project-wide seed.

### Model & Training
- Base model: `distilbert-base-uncased` (40% fewer params than BERT, ~2× faster, <1% accuracy drop).
- Fine-tuning: LoRA, `rank=16`, `target_modules=["q_lin", "v_lin"]`, `lora_alpha=32`, `lora_dropout=0.1`, `bias="none"`.
  - Do **not** target all attention modules — overfits on 4,840 samples.
  - Do **not** do full fine-tuning — kept as a comparison run only (`train.py`).
- Class imbalance: `WeightedTrainer` (subclass of `Trainer`) applies `CrossEntropyLoss(weight=class_weights)`. Weights computed with `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced", ...)` on the **train** split only — never let test distribution leak in.

### Metrics
- **Primary**: weighted F1. Goes on the resume.
- **Secondary**: macro F1. Recorded but not reported on resume — pulled down by the ~12% negative class.
- Do **not** use accuracy — trivially high due to neutral majority.
- Target: weighted F1 > 0.88 (expected ~0.91 after LoRA). Baseline weighted F1 ~0.65 (random classification head).

### Evaluation Outputs (produced by `eval.py`)
1. weighted F1 and macro F1 saved to `outputs/lora_results.json`
2. confusion matrix saved to `outputs/confusion_matrix.png`
3. confidence calibration curve via `sklearn.calibration.calibration_curve` (one curve per class), saved to `outputs/calibration_curve.png`. Expected finding: model overconfident on negative class — interview hook: *"could apply temperature scaling to correct this."*

### API Contract

```
POST /predict
Request:  {"text": "Apple reported record earnings this quarter."}
Response: {"label": "positive", "confidence": 0.94, "latency_ms": 45.0}
```

- Labels: `positive` | `negative` | `neutral`
- Load the model **once** at startup using FastAPI `lifespan`, store in `app.state.model` / `app.state.tokenizer`. Loading inside the endpoint causes per-request reload and pushes p95 to seconds.
- Measure `latency_ms` inside the endpoint with `time.perf_counter()`.

### Latency Targets
- p50 < 30ms warm, p95 < 50ms warm (CPU).
- "Warm" = model already loaded, ≥5 prior requests sent.
- HuggingFace Spaces free-tier cold start (30–60s) is **not** included. Always say "warm inference latency" on resume and README.

---

## Project 2 — Agent Spec

### Sentiment Service Contract
The agent calls Project 1's FastAPI. **Do not re-implement sentiment logic.**

**Endpoint**: `POST {SENTIMENT_SERVICE_URL}/predict`
**Request**: `{ "text": "..." }`
**Response**: `{ "label": "positive" | "negative" | "neutral", "confidence": float, "latency_ms": float }`

### Tools (`src/tools.py`)
- **`search_news(query: str) → list[dict]`**
  Calls NewsAPI, returns up to 10 recent articles. Each article: `{ title, description, url, publishedAt }`. Reads `NEWS_API_KEY` from env. Raises a descriptive error on failure.
- **`analyze_sentiment(text: str) → dict`**
  POSTs to `SENTIMENT_SERVICE_URL/predict`. Returns the full response JSON. `timeout=10s`. Raises a descriptive error on failure.

### Agent (`src/agent.py`)
- LangChain `AgentExecutor`, ReAct-style reasoning, agent decides tool order autonomously.
- Model: `gpt-4o-mini`. Reads `OPENAI_API_KEY` from env.
- `verbose=True` in development, `verbose=False` in production.
- Entry point: `run(question: str) → str`.
- File ends with `if __name__ == "__main__":` REPL — input `quit` to exit.

### System Prompt (`src/prompts.py`)
Instructs the agent to:
1. Always call `search_news` first when receiving a financial question.
2. Call `analyze_sentiment` on each retrieved article (use title + description).
3. Aggregate the sentiment distribution (counts of positive/negative/neutral).
4. Synthesize a final conclusion grounded in the articles and the distribution.
5. If `search_news` returns empty, clearly tell the user no relevant news was found.

---

## Environment Variables

Copy `.env.example` to `.env`:

```
OPENAI_API_KEY=
NEWS_API_KEY=
SENTIMENT_SERVICE_URL=http://localhost:8000
```

Never commit `.env`. The `.gitignore` excludes it.

---

## Commands

```bash
# ── One-time setup ──
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Activate (optional, for shell convenience):
source .venv/bin/activate

# ── Project 1: training ──  (dataset auto-downloads via kagglehub on first run)
.venv/bin/python data.py        # class distribution, class weights
.venv/bin/python baseline.py    # baseline eval (no fine-tune)
.venv/bin/python train.py       # full fine-tune (comparison run, optional)
.venv/bin/python lora.py        # LoRA fine-tune (final model)
.venv/bin/python eval.py        # confusion matrix + calibration curve

# ── Project 1: serve ──
.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
# or:
docker build -t fin-sentiment . && docker run -p 8000:8000 fin-sentiment

# ── Project 2: agent ──
.venv/bin/python src/agent.py   # interactive REPL
.venv/bin/pytest tests/ -v      # mocked tests, no live HTTP
```

---

## Conventions

- **Reproducibility**: `RANDOM_SEED=42` is identical across `baseline.py`, `train.py`, `lora.py`, and `eval.py`. Same stratified split everywhere — the Day-2 test split is the reference for every later evaluation.
- **Label encoding**: fixed `LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}`. Do not reorder — checkpoints depend on it.
- **External API calls** are wrapped in `try/except` with descriptive error messages.
- **Pydantic models** for structured data between agent tools.
- **`latency_ms`** from the sentiment service is logged, not surfaced to the end user.
- **Class weights** are computed on the train split, never on the full dataset.
- **Model loading** in FastAPI/Gradio uses startup hooks (lifespan / on-load); never inside a request handler.
- **Latency reporting** always uses the "warm inference" qualifier when stated in README or resume.
