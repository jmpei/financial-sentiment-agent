---
title: Financial Sentiment Agent
emoji: 🤖
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Financial Sentiment Agent

A LangChain agent that answers financial questions by:

1. Searching recent news via **NewsAPI**.
2. Scoring each article with a **DistilBERT + LoRA** sentiment classifier (loaded in-process).
3. Synthesising a final answer grounded in the article-level sentiment distribution.

| | |
|---|---|
| Agent framework | LangChain `AgentExecutor` (tool-calling) |
| LLM | `gpt-4o-mini` |
| Sentiment model | DistilBERT + LoRA (weighted F1 0.8309 on FinancialPhraseBank test) |
| News | NewsAPI `/v2/everything` |

## Rate limit

To keep the demo affordable, requests are rate limited per client IP:

- **10 / hour / IP**
- **30 / day / IP**
- **200 / day** global cap

Limits reset on container restart.

## Required secrets (Space settings → Secrets)

- `OPENAI_API_KEY`
- `NEWS_API_KEY`

## Companion Space

The sentiment classifier on its own: [jmpei/financial-sentiment-analysis](https://huggingface.co/spaces/jmpei/financial-sentiment-analysis).
