"""Drift check: compare logged traffic with a reference using PSI (population stability index)."""

import json
from pathlib import Path

import numpy as np

from ranker.api import LOG
from ranker.data import load_examples
from ranker.features import tokenize

REF = Path("artifacts/drift_ref.json")


def psi(ref, cur, bins=10):
    """Sum over bins of (cur% - ref%) * ln(cur% / ref%), bins cut at reference quantiles.
    Rule of thumb: < 0.1 stable, 0.1 to 0.25 watch, > 0.25 drifted."""
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    r = np.bincount(np.searchsorted(edges, ref), minlength=len(edges) + 1) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur), minlength=len(edges) + 1) / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def reference():
    """Query lengths of training queries, and top scores the pipeline gives 500 valid queries."""
    if REF.exists():
        return json.loads(REF.read_text())
    from ranker.pipeline import Searcher

    e = load_examples()
    train_q = e[e.split == "train"]["query"].unique()
    valid_q = e[e.split == "valid"]["query"].drop_duplicates().sample(500, random_state=0)
    s = Searcher()
    ref = {
        "q_len": [len(t) for t in tokenize(train_q)],
        "top_score": [s.search(q)[0][0]["score"] for q in valid_q],
    }
    REF.write_text(json.dumps(ref))
    return ref


def status(v):
    return "stable" if v < 0.1 else "watch" if v < 0.25 else "DRIFT"


def main():
    logs = [json.loads(line) for line in LOG.read_text().splitlines()]
    ref = reference()
    cur = {
        "q_len": [len(t) for t in tokenize([r["query"] for r in logs])],
        "top_score": [r["top_score"] for r in logs if r["top_score"] is not None],
    }
    print(f"{len(logs)} logged requests")
    for name in ref:
        v = psi(ref[name], cur[name])
        print(f"  {name:10s} psi {v:.3f} {status(v)}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(1, 1, 5000)) > 0.25
    main()
