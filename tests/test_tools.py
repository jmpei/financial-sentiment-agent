"""
Unit tests for src.tools — no network, no OPENAI_API_KEY needed.

Regression guard for the NewsAPI key leak: the key must travel in the
`X-Api-Key` header, never in the request URL/query. requests echoes the URL
(with query string) inside HTTPError / ConnectionError messages, and those
messages are surfaced to the user (agent observation, Spaces reasoning trace),
so a key in the query string leaks on any request failure.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

import src.tools as tools

KEY = "SECRET_TEST_KEY_abcdef"


@pytest.fixture(autouse=True)
def _set_key(monkeypatch):
    monkeypatch.setattr(tools, "NEWS_API_KEY", KEY)


def _ok_response():
    r = MagicMock(spec=requests.Response)
    r.json.return_value = {"status": "ok", "articles": []}
    r.raise_for_status.return_value = None
    return r


def test_key_sent_in_header_not_query():
    with patch("src.tools.requests.get") as mock_get:
        mock_get.return_value = _ok_response()
        tools.search_news.func("apple")

    _, kwargs = mock_get.call_args
    # Key must be in the X-Api-Key header ...
    assert kwargs.get("headers", {}).get("X-Api-Key") == KEY
    # ... and must NOT appear anywhere in the query params (which end up in the URL).
    params = kwargs.get("params") or {}
    assert "apiKey" not in params
    assert KEY not in params.values()
