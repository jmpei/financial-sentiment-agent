import sys
from pathlib import Path

# spaces_agent/ is run as the cwd on HF Spaces, so app.py imports its sibling
# modules by top-level name (`from ratelimit import ...`). Mirror that here so
# tests can `import ratelimit` without importing the model-loading app.py.
sys.path.insert(0, str(Path(__file__).parent.parent / "spaces_agent"))
