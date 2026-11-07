"""Stage one: hybrid retrieval. BM25 and dense candidates merged by reciprocal rank fusion."""

import sys
from collections import defaultdict

from ranker import bm25, dense
from ranker.data import load_examples, load_products
from ranker.metrics import mean_recall


class Retriever:
    def __init__(self):
        self.bm25 = bm25.load_index()
        self.model = dense.load_model()
        self.hnsw = dense.hnsw_index(dense.product_embeddings())

    def lists(self, queries, depth):
        """query -> (BM25 top rows, dense top rows), each `depth` long."""
        hits = dense.search(lambda qv, k: dense.hnsw_knn(self.hnsw, qv, k), self.model, queries, depth)
        return {q: (bm25.top_rows(bm25.scores(self.bm25, q), depth), hits[q]) for q in queries}

    def __call__(self, queries, n, depth=200):
        """query -> top n product rows after fusion."""
        return {q: rrf(ls, n) for q, ls in self.lists(queries, depth).items()}


def rrf(lists, n, k=60):
    """Merge ranked lists of product rows. A product scores sum(1 / (k + rank)) over
    the lists it appears in. Uses ranks only, so BM25 and cosine scales never mix.
    k=60 is the constant from the original RRF paper (Cormack et al., 2009)."""
    score = defaultdict(float)
    for rows in lists:
        for rank, row in enumerate(rows, 1):
            score[int(row)] += 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)[:n]


def evaluate(split, depth=200):
    e = load_examples()
    e = e[e.split == split]
    pos = {pid: i for i, pid in enumerate(load_products().product_id)}
    lists = Retriever().lists(e["query"].unique(), depth)
    for n in (100, 200):
        bm = {q: b for q, (b, _) in lists.items()}
        dn = {q: d for q, (_, d) in lists.items()}
        hy = {q: rrf(ls, n) for q, ls in lists.items()}
        print(f"{split} recall@{n}(E): bm25={mean_recall(e, bm, pos, n):.4f} "
              f"dense={mean_recall(e, dn, pos, n):.4f} hybrid={mean_recall(e, hy, pos, n):.4f}")


def check():
    assert rrf([[1, 2, 3], [3, 4]], 10) == [3, 1, 2, 4]
    assert rrf([[1, 2, 3], [3, 4]], 2) == [3, 1]
    assert rrf([[], [5]], 10) == [5]
    print("ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "valid"
    check() if arg == "check" else evaluate(arg)
