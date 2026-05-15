FROM python:3.10-slim

WORKDIR /app

# System deps kept minimal — slim image already has what HuggingFace needs.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# CPU-only PyTorch — the CUDA wheel is ~2 GB and useless in this image.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Runtime deps for the API only (no langchain / gradio / training extras).
RUN pip install \
    "transformers>=4.40" \
    "peft>=0.10" \
    "fastapi>=0.110" \
    "uvicorn[standard]>=0.29" \
    "pydantic>=2.5"

# Pre-cache the DistilBERT base weights inside the image so the first request
# doesn't trigger a 250 MB HuggingFace download. The LoRA adapter is mounted
# via COPY below.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
    AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')"

# Adapter weights (~3.4 MB) + API code.
COPY checkpoints/lora ./checkpoints/lora
COPY api ./api

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
