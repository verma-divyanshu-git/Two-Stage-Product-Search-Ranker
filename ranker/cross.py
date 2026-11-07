"""Cross-encoder: reads query and title together in one transformer pass and scores the pair.
More accurate than comparing two separate embeddings, but costs one pass per pair."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from ranker.data import load_examples, load_products
from ranker.metrics import mean_ndcg

ZERO_SHOT = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(name=ZERO_SHOT):
    return CrossEncoder(name, device=DEVICE, max_length=64)


def score(model, queries, titles, batch_size=256):
    return model.predict(list(zip(queries, titles)), batch_size=batch_size, show_progress_bar=False).astype(np.float32)


def cache_path(name):
    return Path("artifacts") / f"ce_{Path(name).name}.parquet"


def example_scores(name=ZERO_SHOT):
    """Cross-encoder score for every labelled pair, keyed by (query, product_id)."""
    path = cache_path(name)
    if path.exists():
        return pd.read_parquet(path)
    e = load_examples()[["query", "product_id"]].drop_duplicates()
    title = load_products().set_index("product_id").product_title.fillna("")
    e["ce"] = score(load_model(name), e["query"], title.loc[e.product_id])
    e.to_parquet(path)
    return e


def scores_for(e, name=ZERO_SHOT):
    """Cached cross-encoder score for each row of e."""
    return e[["query", "product_id"]].merge(example_scores(name), how="left").ce.to_numpy()


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ZERO_SHOT
    e = load_examples()
    s = scores_for(e, name)
    assert len(s) == len(e) and not np.isnan(s).any()
    for split in ("valid", "test"):
        print(f"{split} {Path(name).name} alone ndcg@10={mean_ndcg(e, s, split):.4f}")
