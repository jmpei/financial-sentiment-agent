"""
Latency benchmark for the FastAPI /predict endpoint.

Sends 20 sequential requests, discards the first 5 as warm-up, reports
p50 / p95 / mean / max from the `latency_ms` field in each response.

Warm definition (matches the resume claim): model already loaded, at least
5 prior requests sent.

Usage:
  # Start the API first:
  .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 &
  .venv/bin/python scripts/benchmark.py
"""

import sys
import numpy as np
import requests

URL     = "http://localhost:8000/predict"
PAYLOAD = {"text": "Apple reported record earnings beating analyst expectations."}
N       = 20
WARMUP  = 5

print(f"Sending {N} sequential requests to {URL} ...")
latencies = []

for i in range(N):
    try:
        r = requests.post(URL, json=PAYLOAD, timeout=30)
        r.raise_for_status()
        ms = r.json()["latency_ms"]
    except requests.RequestException as e:
        print(f"  req {i + 1:2d}: ERROR {e}")
        sys.exit(1)

    latencies.append(ms)
    tag = "warmup" if i < WARMUP else "warm"
    print(f"  req {i + 1:2d} [{tag:>6s}]: {ms:6.2f} ms")

warm = latencies[WARMUP:]
p50  = float(np.percentile(warm, 50))
p95  = float(np.percentile(warm, 95))
mean = float(np.mean(warm))
mx   = float(np.max(warm))

print(f"\nLatency (warm-up of {WARMUP} excluded, n={len(warm)})")
print(f"  p50  : {p50:6.2f} ms")
print(f"  p95  : {p95:6.2f} ms")
print(f"  mean : {mean:6.2f} ms")
print(f"  max  : {mx:6.2f} ms")
print(f"\nTarget: p95 < 50ms warm")
print(f"Status: {'PASS' if p95 < 50 else 'FAIL — above 50ms'}")
