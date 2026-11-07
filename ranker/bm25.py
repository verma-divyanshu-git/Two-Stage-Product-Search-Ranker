"""BM25 over the product catalog. Baseline ranker and first retrieval source."""

import sys
from pathlib import Path

import bm25s
import numpy as np
import pandas as pd
import Stemmer
from scipy.sparse import csc_matrix

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


def pair_scores(r, queries, rows):
    """BM25 score of product row rows[i] for queries[i], read straight from the
    index matrix. Much faster than scoring the whole catalog for each query."""
    s = r.scores
    m = csc_matrix((s["data"], s["indices"], s["indptr"]), shape=(s["num_docs"], len(s["indptr"]) - 1)).tocsr()
    rows = np.asarray(rows)
    out = np.zeros(len(rows), dtype=np.float32)
    for q, idx in pd.Series(queries).groupby(queries).indices.items():
        t = tokens(q)
        ids = r.get_tokens_ids(t) if t else []
        if ids:
            out[idx] = m[rows[idx]][:, ids].sum(axis=1).A1
    return out


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
    e = e.assign(row=e.product_id.map(pos))
    for q, g in e.groupby("query"):
        s = scores(r, q)
        idx = g.row.to_numpy()
        order = np.argsort(-s[idx], kind="stable")
        nd.append(ndcg(g.gain.to_numpy()[order].tolist(), g.gain.tolist()))
        exact = idx[g.esci_label.to_numpy() == "E"]
        if len(exact):
            top = np.argpartition(-s, k)[:k]
            rc.append(recall(top.tolist(), exact.tolist(), k))
    print(f"{split}: ndcg@10={np.mean(nd):.4f} recall@{k}(E)={np.mean(rc):.4f} queries={len(nd)}")


def check():
    r = load_index()
    queries = ["80 cfm fan", "fan fan", "it", "zzqqxx usb cable"]
    rows = np.arange(0, 400000, 1000)
    for q in queries:
        fast = pair_scores(r, np.array([q] * len(rows)), rows)
        assert np.allclose(fast, scores(r, q)[rows], atol=1e-4), q
    print("ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "valid"
    check() if arg == "check" else evaluate(arg)
