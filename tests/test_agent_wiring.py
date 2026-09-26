"""
Agent wiring tests with a scripted chat model — no OPENAI_API_KEY, no network.

tests/test_agent.py checks that the real LLM follows the prompt's policy, and
skips without a key. This file covers what does not depend on the LLM, so it
runs in CI: the compiled LangGraph agent routes tool calls to our tools, a
ToolException comes back as an observation instead of crashing the run, and
run() returns the final message text.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

import src.agent as agent_module
from src.prompts import SYSTEM_PROMPT
from src.tools import analyze_sentiment, search_news

ARTICLES = [
    {"title": "Apple beats earnings", "description": "Record revenue.",
     "url": "https://example.com/1", "publishedAt": "2026-05-15T08:00:00Z"},
    {"title": "Apple faces probe", "description": None,
     "url": "https://example.com/2", "publishedAt": "2026-05-15T09:00:00Z"},
]
FINAL = "Mixed. 1 positive, 1 negative out of 2."


class ScriptedModel(GenericFakeChatModel):
    """Replays fixed AIMessages; tool binding is a no-op."""

    def bind_tools(self, tools, **kwargs):
        return self


def _script():
    return iter([
        AIMessage(content="", tool_calls=[
            {"name": "search_news", "args": {"query": "Apple"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[
            {"name": "analyze_sentiment", "args": {"text": "Apple beats earnings. Record revenue."}, "id": "c2"},
            {"name": "analyze_sentiment", "args": {"text": "Apple faces probe"}, "id": "c3"}]),
        AIMessage(content=FINAL),
    ])


def _response(payload):
    r = MagicMock(spec=requests.Response)
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


@pytest.fixture(autouse=True)
def scripted_agent(monkeypatch):
    monkeypatch.setattr("src.tools.NEWS_API_KEY", "test-key")
    agent = create_agent(
        model=ScriptedModel(messages=_script()),
        tools=[search_news, analyze_sentiment],
        system_prompt=SYSTEM_PROMPT,
    )
    monkeypatch.setattr(agent_module, "_agent", agent)
    return agent


def test_tool_calls_reach_the_http_boundaries():
    with patch("src.tools.requests.get") as get, patch("src.tools.requests.post") as post:
        get.return_value = _response({"status": "ok", "articles": ARTICLES})
        post.side_effect = [
            _response({"label": "positive", "confidence": 0.9, "latency_ms": 5.0}),
            _response({"label": "negative", "confidence": 0.8, "latency_ms": 5.0}),
        ]
        answer = agent_module.run("How is Apple doing?")

    assert answer == FINAL
    assert get.call_args.kwargs["params"]["q"] == "Apple"
    assert sorted(c.kwargs["json"]["text"] for c in post.call_args_list) == [
        "Apple beats earnings. Record revenue.", "Apple faces probe"]


def test_tool_failure_becomes_observation_not_crash(scripted_agent):
    with patch("src.tools.requests.get") as get, patch("src.tools.requests.post") as post:
        get.return_value = _response({"status": "ok", "articles": ARTICLES})
        post.side_effect = requests.exceptions.Timeout("timed out")
        result = scripted_agent.invoke(
            {"messages": [{"role": "user", "content": "How is Apple doing?"}]},
            config={"recursion_limit": agent_module.RECURSION_LIMIT},
        )

    observations = [m.content for m in result["messages"]
                    if isinstance(m, ToolMessage) and m.name == "analyze_sentiment"]
    assert len(observations) == 2
    assert all("Sentiment service failed: timed out" in o for o in observations)
    assert result["messages"][-1].content == FINAL
