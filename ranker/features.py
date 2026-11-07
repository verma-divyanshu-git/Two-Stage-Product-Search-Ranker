"""Features for (query, product) pairs. The same code builds training rows and serving rows."""

import re

import bm25s
import numpy as np
import pandas as pd

from ranker.bm25 import STEM, load_index, pair_scores
from ranker.cross import FINETUNED, scores_for
from ranker.data import load_examples, load_products
from ranker.dense import encode, load_model, product_embeddings

NUM = re.compile(r"\d+(?:\.\d+)?")


def tokenize(texts):
    return [set(t) for t in bm25s.tokenize(list(texts), stemmer=STEM, return_ids=False, show_progress=False)]


def overlap(a, b):
    """Fraction of a that is in b. NaN when a is empty, so the model can tell 'no signal' from 'no match'."""
    return len(a & b) / len(a) if a else np.nan


class Featurizer:
    """Loads the catalog, BM25 index and bi-encoder once."""

    def __init__(self):
        self.p = load_products().fillna("")
        self.bm25 = load_index()
        self.model = load_model()
        self.emb = product_embeddings()

    def __call__(self, queries, rows, qv=None):
        """Every feature except the cross-encoder, for product row rows[i] and queries[i].
        qv: optional query -> embedding, to skip encoding queries again."""
        queries = np.asarray(queries)
        rows = np.asarray(rows)
        p = self.p.iloc[rows]
        uq = pd.unique(queries)
        q_tok = dict(zip(uq, tokenize(uq)))
        q_num = {q: set(NUM.findall(q)) for q in uq}
        if qv is None:
            qv = dict(zip(uq, encode(self.model, list(uq))))

        title = tokenize(p.product_title)
        brand = tokenize(p.product_brand)
        color = tokenize(p.product_color)
        title_num = [set(NUM.findall(t)) for t in p.product_title]
        qt = [q_tok[q] for q in queries]

        f = pd.DataFrame(index=range(len(rows)))
        f["bm25"] = pair_scores(self.bm25, queries, rows)
        f["cosine"] = np.einsum("ij,ij->i", np.stack([qv[q] for q in queries]), self.emb[rows])
        f["q_len"] = [len(t) for t in qt]
        f["title_len"] = [len(t) for t in title]
        f["title_overlap"] = [overlap(q, t) for q, t in zip(qt, title)]
        f["brand_in_query"] = [overlap(b, q) for q, b in zip(qt, brand)]
        f["color_in_query"] = [overlap(c, q) for q, c in zip(qt, color)]
        f["num_match"] = [overlap(q_num[q], n) for q, n in zip(queries, title_num)]
        f["has_desc"] = (p.product_description != "").to_numpy()
        f["has_bullets"] = (p.product_bullet_point != "").to_numpy()
        return f


def build(e):
    """Training features for the labelled pairs in e, with cached cross-encoder scores."""
    fz = Featurizer()
    rows = pd.Index(fz.p.product_id).get_indexer(e.product_id)
    f = fz(e["query"], rows).set_axis(e.index)
    f["ce"] = scores_for(e, FINETUNED)
    return f


if __name__ == "__main__":
    e = load_examples()
    f = build(e)
    print(f.describe().T.round(3))
    assert len(f) == len(e) and f.index.equals(e.index)
    for c in ["title_overlap", "brand_in_query", "color_in_query", "num_match"]:
        assert f[c].dropna().between(0, 1).all(), c
