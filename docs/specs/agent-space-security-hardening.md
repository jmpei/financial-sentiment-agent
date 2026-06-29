# Spec — Agent Space security hardening (3 findings)

**Type:** Security hardening. Follow-up to the NewsAPI key-leak fix
(`fe89320`). Three findings from the systematic-debugging pass on
`spaces_agent/app.py` and the agent prompt. Each fix and its option were
chosen by the project owner after web research (see "Decisions").

## Context (verified in code)

- `spaces_agent/app.py` is the **public** HuggingFace Spaces agent. It enforces
  a sliding-window rate limit: `HOUR_LIMIT=10`, `DAY_LIMIT=30`,
  `GLOBAL_DAY_LIMIT=200`.
- `_get_ip()` trusts the **leftmost** `X-Forwarded-For` entry as the client IP.
- `_ip_log` is a `defaultdict(deque)`; per-IP buckets are pruned only for the
  current request's IP and **empty buckets are never deleted** → the dict grows
  unbounded with one-shot IPs.
- The agent feeds fetched news (title + description) into the LLM during
  synthesis. The agent has **no side-effecting tools** (read-only NewsAPI +
  local classifier).
- The sliding-window logic (10/hr, 30/day, global 200/day) was re-verified as
  correct and must be preserved byte-for-byte in behaviour.

## Findings & Decisions

### Finding 1 — X-Forwarded-For spoofing → per-IP limit bypass
The leftmost XFF entry is client-controlled; an attacker can rotate it per
request to get a fresh bucket. Industry guidance: never trust the leftmost XFF;
on a proxy whose hop count is unknown (HF Spaces' topology is undocumented and
unstable) secure XFF parsing is impossible.

**Decision (owner): X-IP-Token priority + global backstop.**
Key the limit on HF's injected `x-ip-token` header (HF's own per-user signal,
added by trusted infra), falling back to the connecting host. **Do not trust
leftmost XFF.** The `GLOBAL_DAY_LIMIT=200` cap remains the real backstop.
**Caveat (accepted):** `x-ip-token` is documented mainly for ZeroGPU Spaces; on
this CPU Space its presence must be verified — emit a one-time log of whether
the header is present so it can be checked in the Space's runtime logs.

### Finding 2 — rate-limiter unbounded memory growth
`_ip_log` never removes idle/expired buckets.

**Decision (owner): `cachetools.TTLCache`.**
Store per-key buckets in a `TTLCache(maxsize, ttl=86400)` so idle keys are
evicted automatically; `maxsize` bounds the worst case. Sliding-window logic
unchanged; the bucket is re-inserted on each hit to refresh its TTL.

### Finding 3 — indirect prompt injection from news content
Article text flows into the LLM. A crafted article could try to redirect the
model. Impact is **low** (no side-effecting tools; worst case is a skewed text
answer), but the prompt should still defend.

**Decision (owner): harden the system prompt + isolate news text.**
Add a clause to the system prompt (both `src/prompts.py` and the inline copy in
`spaces_agent/app.py`) stating that fetched article content is untrusted
external data, never instructions, and must not redirect the model.

## Non-goals

- Configuring trusted-proxy XFF parsing (`FORWARDED_ALLOW_IPS` / hop counting)
  — rejected because HF's proxy topology is undocumented/unstable.
- An output-validation/rule-engine layer for injection (owner chose the
  prompt-hardening option, not the heavier output-validation option).
- A third-party rate-limit library (`slowapi`/`limits`) — owner chose
  `cachetools`.
- Any change to `api/main.py`, training scripts, or the model.

## Acceptance criteria

- A new, importable module `spaces_agent/ratelimit.py` holds the rate-limit
  logic with **no heavy imports** (stdlib + `cachetools` only), unit-tested in
  CI without loading the model.
- `client_key()` returns a token-based key when `x-ip-token` is present, a
  host-based key otherwise, and **ignores** any `X-Forwarded-For` header.
- `RateLimiter` preserves the 10/hr, 30/day, 200/day sliding-window behaviour
  and evicts idle keys (bounded memory).
- `spaces_agent/app.py` uses the new module; leftmost-XFF trust is gone; a
  one-time `x-ip-token present: <bool>` line is logged.
- Both system-prompt copies contain the untrusted-content clause; a test guards
  both.
- `cachetools` is added to `spaces_agent/requirements.txt` and the CI test job.
- `pytest tests/ -v` is green in CI (no `OPENAI_API_KEY`), including the new
  `test_ratelimit.py` and `test_prompts.py`.
- `python -m compileall -q .` passes (CI `compile` job).

## Interview payoff

Turns three latent weaknesses into a defensible story: *"I keyed the public
demo's rate limit on HF's per-user token instead of the spoofable
X-Forwarded-For, bounded the limiter's memory with a TTL cache, and hardened the
agent prompt to treat fetched news as untrusted data — and I extracted the
limiter so it's actually unit-tested in CI."*
