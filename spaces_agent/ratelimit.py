"""
Per-client rate limiting for the agent Space — extracted so it can be unit
tested without importing app.py (which loads the model at import time).

Client identity prefers HuggingFace's injected `x-ip-token` header (HF's own
per-user signal, added by trusted infra), falling back to the connecting host.
The leftmost X-Forwarded-For entry is deliberately NOT trusted: it is
client-controlled and trivially spoofable, so it cannot key a rate limit.

The global daily cap is the real backstop regardless of client identity.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Mapping, Optional

from cachetools import TTLCache

HOUR_LIMIT = 10
DAY_LIMIT = 30
GLOBAL_DAY_LIMIT = 200

_HOUR_SECONDS = 3_600
_DAY_SECONDS = 86_400


def client_key(headers: Mapping[str, str], client_host: Optional[str]) -> str:
    """Stable per-client key. Prefer HF's x-ip-token, else the connecting host.

    The x-ip-token lookup is case-insensitive. X-Forwarded-For is intentionally
    not consulted — its leftmost entry is client-spoofable.
    """
    token = None
    for k, v in headers.items():
        if k.lower() == "x-ip-token":
            token = v
            break
    if token:
        return f"tok:{token}"
    if client_host:
        return f"ip:{client_host}"
    return "unknown"


class RateLimiter:
    """Sliding-window limiter with auto-expiring per-key buckets.

    Per-key buckets live in a TTLCache (maxsize + 24h ttl) so idle clients are
    evicted automatically — no unbounded dict growth.
    """

    def __init__(
        self,
        hour_limit: int = HOUR_LIMIT,
        day_limit: int = DAY_LIMIT,
        global_day_limit: int = GLOBAL_DAY_LIMIT,
        maxsize: int = 10_000,
    ) -> None:
        self.hour_limit = hour_limit
        self.day_limit = day_limit
        self.global_day_limit = global_day_limit
        self._lock = threading.Lock()
        self._buckets: TTLCache = TTLCache(maxsize=maxsize, ttl=_DAY_SECONDS)
        self._global: deque = deque()

    def check(self, key: str, now: Optional[float] = None) -> tuple[bool, str]:
        """Return (allowed, message). Records the hit when allowed."""
        if now is None:
            now = time.time()
        hour_ago = now - _HOUR_SECONDS
        day_ago = now - _DAY_SECONDS
        with self._lock:
            while self._global and self._global[0] < day_ago:
                self._global.popleft()
            if len(self._global) >= self.global_day_limit:
                return False, f"Daily global limit reached ({self.global_day_limit}). Try again tomorrow."

            bucket = self._buckets.get(key) or deque()
            while bucket and bucket[0] < day_ago:
                bucket.popleft()

            if sum(1 for t in bucket if t > hour_ago) >= self.hour_limit:
                return False, f"Hourly limit reached ({self.hour_limit}/hr for your client). Wait an hour."
            if len(bucket) >= self.day_limit:
                return False, f"Daily limit reached ({self.day_limit}/day for your client). Try again tomorrow."

            bucket.append(now)
            self._buckets[key] = bucket   # re-insert: refresh TTL, keep active keys alive
            self._global.append(now)
            return True, ""
