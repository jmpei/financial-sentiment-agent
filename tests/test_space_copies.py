"""
The Spaces are single-file deploys, so they carry their own copies of shared
constants. These tests fail when a copy drifts from its source: the agent
Space's prompt/model/recursion limit from src/, and every serving path's
TEMPERATURE from calibration_results.json.

Constants are read with `ast` so the model-loading app files are never imported.
"""

import ast
import json
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


def test_serving_temperature_matches_calibration():
    fitted = json.loads(Path("calibration_results.json").read_text())["temperature"]
    for path in ("api/main.py", "spaces/app.py", "spaces_agent/app.py"):
        assert _constants(path)["TEMPERATURE"] == fitted, path
