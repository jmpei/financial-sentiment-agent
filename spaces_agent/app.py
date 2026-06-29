"""
Gradio app for HuggingFace Spaces — financial sentiment agent.

The agent has two tools:
  - search_news        : NewsAPI fetch
  - analyze_sentiment  : local DistilBERT+LoRA inference (loaded once at startup)

Per-client rate limit via sliding window, keyed on HF's per-user x-ip-token
(falling back to the connecting host). See ratelimit.py.
"""

import os
import time
from typing import Any, Dict, List

import gradio as gr
import requests
import torch
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import ToolException, tool
from langchain_openai import ChatOpenAI
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ratelimit import RateLimiter, client_key, HOUR_LIMIT, DAY_LIMIT, GLOBAL_DAY_LIMIT

# Local dev reads from .env; on HF Spaces these come from Space Secrets.
load_dotenv()

# ── sentiment model (load once at startup) ──────────────────────────────────
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

tokenizer = AutoTokenizer.from_pretrained("checkpoints/lora")
base = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID,
)
model = PeftModel.from_pretrained(base, "checkpoints/lora").eval()


# ── tools ──────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/everything"


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
        raise ToolException("NEWS_API_KEY is not configured on the server.")
    params = {
        "q":        query,
        "pageSize": 10,
        "sortBy":   "publishedAt",
        "language": "en",
    }
    # Key in the header, never the query string: requests echoes the URL inside
    # RequestException messages, and _format_trace renders those in the public
    # reasoning trace — a key in the query would leak on any NewsAPI failure.
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

    Uses the local fine-tuned DistilBERT+LoRA model. Returns:
      - label       ("positive" | "negative" | "neutral")
      - confidence  (float in [0, 1])
      - latency_ms  (float)

    Call this on each article (title + description) returned by search_news.
    """
    if not text or not text.strip():
        raise ToolException("analyze_sentiment received empty text.")
    t0 = time.perf_counter()
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(
            model(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]).logits,
            dim=-1,
        )[0]
    pred = int(probs.argmax())
    return {
        "label":      ID2LABEL[pred],
        "confidence": float(probs[pred]),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


search_news.handle_tool_error = True
analyze_sentiment.handle_tool_error = True


# ── agent ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a financial sentiment analyst. You have access to two tools:

1. `search_news(query)` — fetches up to 10 recent English financial news articles for a query (company, ticker, market, or topic). Each article contains title, description, url, publishedAt.

2. `analyze_sentiment(text)` — classifies a piece of financial text as positive / negative / neutral with a confidence score, using a fine-tuned model.

When the user asks any financial question, follow this procedure:

Step 1 — Always call `search_news` first with a concise query derived from the user's question. Never answer from prior knowledge alone.

Step 2 — If `search_news` returns an empty list, do NOT call `analyze_sentiment`. Respond exactly:
    "I could not find relevant recent news for your question."
    Then stop.

Step 3 — Otherwise, call `analyze_sentiment` once per article on the combined text of `title + ". " + description` (use just the title if description is empty). Track each article's predicted label.

Step 4 — Produce the final answer with this structure:
    - One short paragraph summarising the overall sentiment.
    - The sentiment distribution: how many articles were positive / negative / neutral (e.g. "6 positive, 3 neutral, 1 negative out of 10").
    - 2-3 representative article titles supporting your conclusion, each on its own line.

Be concise. Do not invent articles or sentiment scores - every claim must come from a tool result. Do not call the tools more than once per article. Do not surface the `latency_ms` field to the user.
"""


def _build_executor() -> AgentExecutor:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured on the server.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",      SYSTEM_PROMPT),
        ("human",       "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    tools = [search_news, analyze_sentiment]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=15,
        return_intermediate_steps=True,
    )


_executor: AgentExecutor | None = None


# ── rate limiting ──────────────────────────────────────────────────────────
# Keyed on HF's per-user x-ip-token (falling back to the connecting host); the
# leftmost X-Forwarded-For is spoofable and is not trusted. See ratelimit.py.
_limiter = RateLimiter()
_ip_token_checked = False


def _client_key(request: gr.Request | None) -> str:
    global _ip_token_checked
    if request is None:
        return "unknown"
    if not _ip_token_checked:
        _ip_token_checked = True
        has_token = any(k.lower() == "x-ip-token" for k in request.headers)
        print(f"[rate-limit] x-ip-token present: {has_token}")
    host = request.client.host if request.client else None
    return client_key(request.headers, host)


# ── handler ────────────────────────────────────────────────────────────────
def _format_trace(steps: List) -> str:
    if not steps:
        return "_(no tool calls)_"
    lines = []
    for i, (action, observation) in enumerate(steps, 1):
        lines.append(f"**Step {i} — Tool: `{action.tool}`**")
        lines.append(f"Input: `{action.tool_input}`")
        obs = str(observation)
        if len(obs) > 600:
            obs = obs[:600] + "..."
        lines.append(f"Observation: {obs}")
        lines.append("")
    return "\n".join(lines)


def ask(question: str, request: gr.Request | None = None):
    if not question or not question.strip():
        return "Please ask a question about a stock, company, or market.", ""

    key = _client_key(request)
    ok, msg = _limiter.check(key)
    if not ok:
        return f"⚠️ {msg}", ""

    global _executor
    if _executor is None:
        try:
            _executor = _build_executor()
        except Exception as e:
            return f"Failed to start agent: {e}", ""

    try:
        result = _executor.invoke({"input": question})
    except Exception as e:
        return f"Agent failed: {e}", ""

    return result.get("output", ""), _format_trace(result.get("intermediate_steps", []))


# ── UI ─────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Financial Sentiment Agent") as demo:
    gr.Markdown(
        "# Financial Sentiment Agent\n"
        "Ask a question about a stock, company, or market. The agent searches recent news "
        "(NewsAPI), scores each article with a fine-tuned DistilBERT+LoRA model, then synthesises "
        "an answer.\n\n"
        f"Rate limited to **{HOUR_LIMIT}/hour, {DAY_LIMIT}/day per IP** "
        f"(global cap {GLOBAL_DAY_LIMIT}/day)."
    )
    question = gr.Textbox(
        label="Your question",
        placeholder="What's the recent sentiment around Apple?",
        lines=2,
    )
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Markdown(label="Answer")
    with gr.Accordion("Agent reasoning trace", open=False):
        trace = gr.Markdown()

    gr.Examples(
        examples=[
            "What's the recent sentiment around Apple stock?",
            "How is Tesla being covered in the news this week?",
            "Should I be worried about NVIDIA?",
        ],
        inputs=question,
    )

    submit.click(fn=ask, inputs=question, outputs=[answer, trace])
    question.submit(fn=ask, inputs=question, outputs=[answer, trace])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
