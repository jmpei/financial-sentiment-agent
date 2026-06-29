"""
Tools registered with the LangChain agent.

- search_news       : NewsAPI fetch (returns recent articles)
- analyze_sentiment : HTTP call to our FastAPI /predict endpoint
                       (which serves the LoRA-fine-tuned DistilBERT)
"""

import os
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool, ToolException

load_dotenv()

NEWS_API_KEY          = os.getenv("NEWS_API_KEY", "")
SENTIMENT_SERVICE_URL = os.getenv("SENTIMENT_SERVICE_URL", "http://localhost:8000").rstrip("/")

NEWS_API_URL          = "https://newsapi.org/v2/everything"
SENTIMENT_TIMEOUT_S   = 10


@tool
def search_news(query: str) -> List[Dict[str, Any]]:
    """Search recent financial news matching the query.

    Returns up to 10 articles, most recent first. Each article has:
      - title       (str)
      - description (str | None)
      - url         (str)
      - publishedAt (str, ISO 8601)

    Use this BEFORE answering any question about a stock, company, market,
    or financial event. Pass the company name or topic as the query.
    """
    if not NEWS_API_KEY:
        raise ToolException("NEWS_API_KEY is not set - fill .env first.")

    params = {
        "q":        query,
        "pageSize": 10,
        "sortBy":   "publishedAt",
        "language": "en",
    }
    # Key goes in the header, never the query string: requests echoes the URL
    # (with query params) inside RequestException messages, and those messages
    # are surfaced to the user — a key in the query would leak on any failure.
    try:
        r = requests.get(
            NEWS_API_URL, params=params, headers={"X-Api-Key": NEWS_API_KEY}, timeout=10
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise ToolException(f"NewsAPI failed: {e}") from e

    payload = r.json()
    if payload.get("status") != "ok":
        raise ToolException(f"NewsAPI returned error: {payload.get('message', payload)}")

    articles = payload.get("articles", []) or []
    return [
        {
            "title":       a.get("title") or "",
            "description": a.get("description"),
            "url":         a.get("url") or "",
            "publishedAt": a.get("publishedAt") or "",
        }
        for a in articles[:10]
    ]


@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Classify the sentiment of a piece of financial text.

    Calls our fine-tuned DistilBERT service. Returns:
      - label       ("positive" | "negative" | "neutral")
      - confidence  (float in [0, 1])
      - latency_ms  (float, server-side inference time)

    Call this on each article (title + description) returned by search_news.
    """
    if not text or not text.strip():
        raise ToolException("analyze_sentiment received empty text.")

    try:
        r = requests.post(
            f"{SENTIMENT_SERVICE_URL}/predict",
            json={"text": text},
            timeout=SENTIMENT_TIMEOUT_S,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise ToolException(f"Sentiment service failed: {e}") from e

    return r.json()


# Surfacing tool errors back to the LLM (as the tool's "observation") lets the
# agent decide how to react — apologise to the user, skip a failing article,
# etc. — instead of crashing the whole turn.
search_news.handle_tool_error       = True
analyze_sentiment.handle_tool_error = True
