"""
The agent Space is a single-file deploy (spaces_agent/app.py), so it carries its
own copies of the agent's constants. These tests fail when a copy drifts from src/.

Constants are read with `ast` so the Space's model-loading app.py is never imported.
"""

import ast
from pathlib import Path

import src.agent as agent
from src.prompts import SYSTEM_PROMPT


def _constants(path: str) -> dict:
    tree = ast.parse(Path(path).read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }


SPACE = _constants("spaces_agent/app.py")


def test_src_prompt_has_untrusted_clause():
    assert "untrusted external data" in SYSTEM_PROMPT


def test_space_prompt_matches_src():
    assert SPACE["SYSTEM_PROMPT"] == SYSTEM_PROMPT


def test_space_model_and_recursion_limit_match_src():
    assert SPACE["OPENAI_MODEL"] == agent.OPENAI_MODEL
    assert SPACE["RECURSION_LIMIT"] == agent.RECURSION_LIMIT
