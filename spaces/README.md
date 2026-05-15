---
title: Financial Sentiment Analysis
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Financial Sentiment Analysis

DistilBERT fine-tuned with LoRA on the FinancialPhraseBank (50% annotator agreement) for 3-class sentiment classification (positive / negative / neutral).

| | |
|---|---|
| Base model | `distilbert-base-uncased` |
| Fine-tune | LoRA via `peft` (rank=16, alpha=32, target_modules=`q_lin`,`v_lin`) |
| Dataset | FinancialPhraseBank `Sentences_50Agree.txt`, 4,846 samples |
| Test split | 485 stratified samples (seed=42) |
| Weighted F1 | **0.8309** (baseline random head: 0.0571) |
| Trainable params | 887,811 / 67.8M = 1.31% |
| Adapter size | 3.4 MB |

Enter a piece of financial text below to get sentiment label, confidence score, and inference latency.

**Note**: latency on the Spaces free-tier CPU is roughly 50–80 ms warm. The first request after a cold start can take 30–60 seconds.
