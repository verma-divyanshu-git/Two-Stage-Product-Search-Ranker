"""Latency benchmark: replay test queries through the API one at a time."""

import sys
import time

import numpy as np
from fastapi.testclient import TestClient

from ranker.api import app
from ranker.data import load_examples


def main(n):
    e = load_examples()
    queries = e[e.split == "test"]["query"].drop_duplicates().sample(n, random_state=0).tolist()
    stages, total = {}, []
    with TestClient(app) as c:
        for q in queries:
            t = time.perf_counter()
            r = c.get("/search", params={"q": q}).json()
            total.append((time.perf_counter() - t) * 1000)
            for name, v in r["ms"].items():
                stages.setdefault(name, []).append(v)
    print(f"{n} queries, ms per request")
    for name, v in [*stages.items(), ("total incl. HTTP", total)]:
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        print(f"  {name:17s} p50 {p50:6.1f}  p95 {p95:6.1f}  p99 {p99:6.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
