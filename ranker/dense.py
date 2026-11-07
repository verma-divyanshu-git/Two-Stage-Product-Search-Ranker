"""Bi-encoder embeddings: queries and products become vectors, similar meaning means close vectors."""

import sys
import time
from functools import partial
from pathlib import Path

import hnswlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ranker.data import load_examples, load_products
from ranker.metrics import mean_recall

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB = Path("artifacts/product_emb.npy")
HNSW = Path("artifacts/hnsw.bin")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model():
    m = SentenceTransformer(MODEL_NAME, device=DEVICE)
    m.max_seq_length = 64
    return m


def product_text(p):
    return (p.product_title.fillna("") + " " + p.product_brand.fillna("")).tolist()


def encode(model, texts, batch_size=256):
    """Unit length float32 vectors, so a dot product is cosine similarity."""
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def product_embeddings():
    """Row i is product row i of load_products()."""
    if EMB.exists():
        return np.load(EMB)
    v = encode(load_model(), product_text(load_products()))
    EMB.parent.mkdir(exist_ok=True)
    np.save(EMB, v)
    return v


def exact_knn(v, qv, k, chunk=512):
    """Top k product rows per query vector, best first, by comparing with every product.
    NumPy instead of FAISS: faiss-cpu and torch each ship their own libomp, and
    running both in one process aborts or silently breaks threading on macOS."""
    out = []
    for i in range(0, len(qv), chunk):
        s = qv[i:i + chunk] @ v.T
        top = np.argpartition(-s, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
        out.append(np.take_along_axis(top, order, axis=1))
    return np.vstack(out)


def hnsw_index(v):
    """Approximate search over a neighbour graph. Built once, then saved."""
    index = hnswlib.Index(space="ip", dim=v.shape[1])
    if HNSW.exists():
        index.load_index(str(HNSW), max_elements=len(v))
    else:
        index.init_index(max_elements=len(v), M=32, ef_construction=200)
        index.add_items(v, np.arange(len(v)))
        index.save_index(str(HNSW))
    # Candidates explored per query. Must be at least k, higher trades speed for recall.
    index.set_ef(300)
    return index


def hnsw_knn(index, qv, k):
    return index.knn_query(qv, k=k)[0].astype(np.int64)


def search(knn, model, queries, k):
    """query -> top k product rows, best first. knn(query_vectors, k) does the lookup."""
    return dict(zip(queries, knn(encode(model, list(queries)), k)))


def evaluate(split, k=100):
    e = load_examples()
    e = e[e.split == split]
    pos = {pid: i for i, pid in enumerate(load_products().product_id)}
    v = product_embeddings()
    model = load_model()
    queries = e["query"].unique()
    for name, knn in [("exact", partial(exact_knn, v)), ("hnsw", partial(hnsw_knn, hnsw_index(v)))]:
        t = time.perf_counter()
        hits = search(knn, model, queries, k)
        ms = (time.perf_counter() - t) * 1000 / len(queries)
        print(f"{split}: dense {name} recall@{k}(E)={mean_recall(e, hits, pos, k):.4f} {ms:.2f} ms/query")


def check():
    v = product_embeddings()
    print(v.shape, v.dtype)
    assert v.shape == (len(load_products()), 384)
    assert np.allclose(np.linalg.norm(v[:1000], axis=1), 1, atol=1e-3)
    rows = exact_knn(v, v[:100], 5)
    assert (rows[:, 0] == np.arange(100)).mean() > 0.9, "a product should find itself"
    s = v[:100] @ v.T
    assert (np.diff(np.take_along_axis(s, rows, axis=1), axis=1) <= 1e-6).all(), "best first"
    print("ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "valid"
    check() if arg == "check" else evaluate(arg)
