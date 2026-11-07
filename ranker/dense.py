"""Bi-encoder embeddings: queries and products become vectors, similar meaning means close vectors."""

import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ranker.data import load_examples, load_products
from ranker.metrics import mean_recall

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB = Path("artifacts/product_emb.npy")
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


def flat_index(v):
    """Exact search: compares the query with every product."""
    index = faiss.IndexFlatIP(v.shape[1])
    index.add(v)
    return index


def search(index, model, queries, k):
    """query -> top k product rows, best first."""
    _, rows = index.search(encode(model, list(queries)), k)
    return dict(zip(queries, rows))


def evaluate(split, k=100):
    e = load_examples()
    e = e[e.split == split]
    pos = {pid: i for i, pid in enumerate(load_products().product_id)}
    hits = search(flat_index(product_embeddings()), load_model(), e["query"].unique(), k)
    print(f"{split}: dense recall@{k}(E)={mean_recall(e, hits, pos, k):.4f}")


def check():
    v = product_embeddings()
    print(v.shape, v.dtype)
    assert v.shape == (len(load_products()), 384)
    assert np.allclose(np.linalg.norm(v[:1000], axis=1), 1, atol=1e-3)
    _, rows = flat_index(v).search(v[:100], 1)
    assert (rows[:, 0] == np.arange(100)).mean() > 0.9, "a product should find itself"
    print("ok")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "valid"
    check() if arg == "check" else evaluate(arg)
