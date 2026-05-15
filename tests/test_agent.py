"""
Integration tests for the financial sentiment agent.

The HTTP boundaries (NewsAPI, our FastAPI /predict) are mocked at the
`requests` level inside src.tools — no live HTTP traffic to either service.

The LLM is intentionally NOT mocked: we want to verify the agent actually
follows the system prompt's tool-orchestration policy. Tests skip cleanly
if OPENAI_API_KEY is missing, so they're safe to run in environments
without a key.

Run:
    .venv/bin/pytest tests/ -v
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

# Reset the lazy executor cache between tests so each test gets a clean build.
import src.agent as agent_module

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping agent integration tests",
)


# Fixtures
@pytest.fixture
def fake_articles():
    return [
        {
            "title":       "Apple beats Q4 earnings expectations",
            "description": "Apple reported record revenue, beating analyst forecasts.",
            "url":         "https://example.com/1",
            "publishedAt": "2026-05-15T08:00:00Z",
        },
        {
            "title":       "Apple stock surges on strong iPhone sales",
            "description": "Shares rose 5% after stronger-than-expected iPhone numbers.",
            "url":         "https://example.com/2",
            "publishedAt": "2026-05-15T09:00:00Z",
        },
        {
            "title":       "Analysts raise Apple price target",
            "description": "Several Wall Street firms upgraded Apple following the earnings beat.",
            "url":         "https://example.com/3",
            "publishedAt": "2026-05-15T10:00:00Z",
        },
    ]


@pytest.fixture(autouse=True)
def _reset_executor_cache():
    """The agent module memoises its AgentExecutor — clear it per test."""
    agent_module._executor = None
    yield
    agent_module._executor = None


@pytest.fixture(autouse=True)
def _stub_news_api_key(monkeypatch):
    """Make search_news think NEWS_API_KEY is set even if .env is unset."""
    monkeypatch.setattr("src.tools.NEWS_API_KEY", "test-key")


def _news_response(articles):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"status": "ok", "articles": articles}
    r.raise_for_status.return_value = None
    return r


def _sentiment_response(payload):
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


# Scenario 1: happy path
def test_happy_path(fake_articles):
    """3 articles, all positive sentiment — agent should produce a real answer."""
    with patch("src.tools.requests.get") as mock_get, \
         patch("src.tools.requests.post") as mock_post:

        mock_get.return_value  = _news_response(fake_articles)
        mock_post.return_value = _sentiment_response(
            {"label": "positive", "confidence": 0.9, "latency_ms": 40.0}
        )

        result = agent_module.run("What is the sentiment around Apple stock?")

    assert isinstance(result, str)
    assert len(result.strip()) > 0
    # search_news called at least once
    assert mock_get.call_count >= 1
    # analyze_sentiment called at least once (ideally 3 — one per article)
    assert mock_post.call_count >= 1
    # No spurious tool-call explosion (15 is the executor's max_iterations cap)
    assert mock_post.call_count <= len(fake_articles) + 2


# Scenario 2: empty news
def test_empty_news_short_circuits():
    """search_news returns [] — agent must NOT call analyze_sentiment."""
    with patch("src.tools.requests.get") as mock_get, \
         patch("src.tools.requests.post") as mock_post:

        mock_get.return_value = _news_response([])

        result = agent_module.run("What is the sentiment around XYZNonexistentTicker?")

    mock_post.assert_not_called()

    text = result.lower()
    assert any(
        phrase in text
        for phrase in ("could not find", "no relevant", "no recent", "no news")
    ), f"Expected a 'no news found' message, got: {result!r}"


# Scenario 3: sentiment service timeout
def test_sentiment_timeout_handled(fake_articles):
    """analyze_sentiment raises Timeout — agent must not propagate it."""
    with patch("src.tools.requests.get") as mock_get, \
         patch("src.tools.requests.post") as mock_post:

        mock_get.return_value  = _news_response(fake_articles[:2])
        mock_post.side_effect  = requests.exceptions.Timeout("timed out")

        # The tool wraps requests.Timeout into a RuntimeError. AgentExecutor's
        # handle_parsing_errors=True catches tool exceptions and feeds the
        # error back to the LLM — so run() should still return a string.
        try:
            result = agent_module.run("How is Apple doing?")
        except requests.exceptions.Timeout:
            pytest.fail("Timeout from the sentiment service should not propagate.")

    assert isinstance(result, str)
    assert len(result.strip()) > 0
