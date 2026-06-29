"""Unit tests for the agent Space rate limiter — stdlib + cachetools only."""

import ratelimit
from ratelimit import RateLimiter, client_key


def test_client_key_prefers_x_ip_token():
    key = client_key({"X-IP-Token": "abc", "x-forwarded-for": "1.2.3.4"}, "10.0.0.1")
    assert key == "tok:abc"


def test_client_key_is_case_insensitive_for_token():
    assert client_key({"x-ip-token": "zzz"}, None) == "tok:zzz"


def test_client_key_falls_back_to_host_and_ignores_xff():
    # No token: must use the connecting host, NEVER the spoofable XFF.
    key = client_key({"x-forwarded-for": "9.9.9.9, 1.1.1.1"}, "10.0.0.1")
    assert key == "ip:10.0.0.1"
    assert "9.9.9.9" not in key


def test_client_key_unknown_when_nothing_available():
    assert client_key({}, None) == "unknown"


def test_hour_limit_blocks_after_10_within_an_hour():
    rl = RateLimiter()
    t = 1_000_000.0
    results = [rl.check("A", t + i)[0] for i in range(12)]
    assert all(results[:10])          # first 10 allowed
    assert not results[10] and not results[11]   # 11th, 12th blocked


def test_day_limit_blocks_after_30_in_a_day():
    rl = RateLimiter()
    t = 1_000_000.0
    allowed = 0
    for hr in range(4):               # 10 per hour across 4 hours = 30, then capped
        for i in range(10):
            allowed += rl.check("B", t + hr * 3600 + i)[0]
    assert allowed == 30


def test_global_cap_across_distinct_keys():
    rl = RateLimiter()
    t = 1_000_000.0
    allowed = sum(rl.check(f"k{i}", t + i)[0] for i in range(250))
    assert allowed == 200


def test_idle_keys_are_evicted_to_bound_memory():
    rl = RateLimiter(maxsize=5)
    t = 1_000_000.0
    for i in range(50):
        rl.check(f"k{i}", t + i)
    # TTLCache maxsize bounds the number of retained buckets.
    assert len(rl._buckets) <= 5
