"""Online search: hybrid retrieval, then a cascade of rankers, cheapest first."""

import sys
import time

import lightgbm as lgb
import numpy as np

from ranker import cross, dense
from ranker.data import load_examples
from ranker.features import Featurizer
from ranker.metrics import ndcg
from ranker.retrieve import Retriever, rrf
from ranker.train import MODELS


class Searcher:
    """query -> top k products.

    1. Retrieve `n` candidates from the catalog with BM25 + dense, fused by RRF.
    2. The light ranker scores all of them on cheap features.
    3. The cross-encoder scores only light's top `rerank`, since it costs one transformer pass per pair.
    4. The full ranker orders those using every feature."""

    def __init__(self):
        self.retriever = Retriever()
        self.fz = Featurizer()
        self.light = lgb.Booster(model_file=MODELS["light"])
        self.full = lgb.Booster(model_file=MODELS["full"])
        self.ce = cross.load_model(cross.FINETUNED)
        self.titles = self.fz.p.product_title.to_numpy()
        self.ids = self.fz.p.product_id.to_numpy()

    def search(self, query, k=10, n=200, rerank=50):
        ms = {}
        t = time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter()
            ms[name] = (now - t) * 1000
            t = now

        qv = dense.encode(self.retriever.model, [query])
        lap("encode")
        cands = np.array(rrf(self.retriever.lists([query], n, qv)[query], n))
        lap("retrieve")
        f = self.fz([query] * len(cands), cands, qv={query: qv[0]})
        lap("features")
        light = self.light.predict(f[self.light.feature_name()])
        top = np.argsort(-light, kind="stable")[:rerank]
        lap("light")
        f = f.iloc[top].assign(ce=cross.score(self.ce, [query] * len(top), self.titles[cands[top]], batch_size=rerank))
        lap("cross")
        full = self.full.predict(f[self.full.feature_name()])
        best = top[np.argsort(-full, kind="stable")][:k]
        lap("full")
        results = [{"product_id": self.ids[r], "title": self.titles[r]} for r in cands[best]]
        return results, ms


def evaluate(split):
    """NDCG@10 of the top 10 from the whole catalog, not just the labelled products.
    Unlabelled products count as gain 0, so this is a lower bound."""
    e = load_examples()
    e = e[e.split == split]
    gain = dict(zip(zip(e["query"], e.product_id), e.gain))
    ideal = e.groupby("query").gain.apply(list)
    s = Searcher()
    out = {"bm25": [], "hybrid": [], "pipeline": []}
    for q in ideal.index:
        b, d = s.retriever.lists([q], 200)[q]
        ranked = {
            "bm25": s.ids[b[:10]],
            "hybrid": s.ids[rrf((b, d), 10)],
            "pipeline": [r["product_id"] for r in s.search(q)[0]],
        }
        for name, ids in ranked.items():
            out[name].append(ndcg([gain.get((q, pid), 0.0) for pid in ids], ideal[q]))
    for name, v in out.items():
        print(f"{split} end-to-end {name} ndcg@10={np.mean(v):.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("valid", "test"):
        evaluate(sys.argv[1])
        sys.exit()
    s = Searcher()
    query = sys.argv[1] if len(sys.argv) > 1 else "80 cfm quiet bathroom exhaust fan"
    s.search(query)
    results, ms = s.search(query)
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r['product_id']} {r['title'][:90]}")
    print({k: round(v, 1) for k, v in ms.items()}, f"total {sum(ms.values()):.1f} ms")
    assert len(results) == 10 and len({r["product_id"] for r in results}) == 10
