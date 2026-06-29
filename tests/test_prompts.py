"""Both copies of the agent system prompt must carry the untrusted-content clause."""

from pathlib import Path

from src.prompts import SYSTEM_PROMPT

CLAUSE = "untrusted external data"


def test_src_prompt_has_untrusted_clause():
    assert CLAUSE in SYSTEM_PROMPT


def test_spaces_agent_prompt_has_untrusted_clause():
    text = Path("spaces_agent/app.py").read_text()
    assert CLAUSE in text
