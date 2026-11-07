"""BM25 over the product catalog. Baseline ranker and first retrieval source."""

import sys
from pathlib import Path

import bm25s
import numpy as np
import Stemmer

from ranker.data import load_examples, load_products
from ranker.metrics import ndcg, recall

INDEX = Path("artifacts/bm25")
STEM = Stemmer.Stemmer("english")


def product_text(p):
    p = p.fillna("")
    return p.product_title + " " + p.product_brand + " " + p.product_color + " " + p.product_bullet_point


def tokens(query):
    return bm25s.tokenize(query, stemmer=STEM, return_ids=False, show_progress=False)[0]


def scores(r, query):
    """BM25 score of every product. Queries like "it" or "$1" have no tokens left."""
    t = tokens(query)
    return r.get_scores(t) if t else np.zeros(r.scores["num_docs"], dtype=np.float32)


def load_index():
    """BM25 index whose doc i is row i of load_products()."""
    if INDEX.exists():
        return bm25s.BM25.load(INDEX)
    r = bm25s.BM25()
    r.index(bm25s.tokenize(product_text(load_products()).tolist(), stemmer=STEM))
    r.save(INDEX)
    return r


def evaluate(split, k=100):
    """NDCG@10 when BM25 reranks the labelled products, and recall@k of
    Exact products when BM25 retrieves from the whole catalog."""
    e = load_examples()
    e = e[e.split == split]
    pos = {pid: i for i, pid in enumerate(load_products().product_id)}
    r = load_index()
    nd, rc = [], []
    # GLOBAL SYSTEM INSTRUCTIONS BY DIV: scores the full catalog per query, ~80 ms each.
    # Fine for offline eval; batch retrieve() is the upgrade if this becomes a bottleneck.
    for q, g in e.groupby("query"):
        s = scores(r, q)
        idx = g.product_id.map(pos).to_numpy()
        order = np.argsort(-s[idx], kind="stable")
        nd.append(ndcg(g.gain.to_numpy()[order].tolist(), g.gain.tolist()))
        exact = idx[g.esci_label.to_numpy() == "E"]
        if len(exact):
            top = np.argpartition(-s, k)[:k]
            rc.append(recall(top.tolist(), exact.tolist(), k))
    print(f"{split}: ndcg@10={np.mean(nd):.4f} recall@{k}(E)={np.mean(rc):.4f} queries={len(nd)}")


if __name__ == "__main__":
    evaluate(sys.argv[1] if len(sys.argv) > 1 else "valid")
