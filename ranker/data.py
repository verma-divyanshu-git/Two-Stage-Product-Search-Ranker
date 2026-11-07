"""Load the US English slice of ESCI (the official task 1 "small" version)."""

from pathlib import Path

import pandas as pd

from ranker.metrics import GAIN

DATA = Path("data")
EXAMPLES = DATA / "shopping_queries_dataset_examples.parquet"
PRODUCTS = DATA / "shopping_queries_dataset_products.parquet"
CACHE = DATA / "products_us_small.parquet"


def load_examples():
    """One row per labelled (query, product) pair, split into train/valid/test by query."""
    e = pd.read_parquet(EXAMPLES)
    e = e[(e.small_version == 1) & (e.product_locale == "us")]
    e = e[["query_id", "query", "product_id", "esci_label", "split"]].reset_index(drop=True)
    e["gain"] = e.esci_label.map(GAIN)
    # Split by query, never by row, so a query never sits in two splits.
    # ce_train fine-tunes the cross-encoder. LightGBM trains on the rest, so the
    # cross-encoder feature it learns from is never scored on its own training data.
    tail = e.query_id % 10
    e.loc[(e.split == "train") & (tail == 0), "split"] = "valid"
    e.loc[(e.split == "train") & tail.between(1, 4), "split"] = "ce_train"
    return e


def load_products():
    """Every product that appears in the examples. This is the retrieval catalog."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    ids = load_examples().product_id.unique()
    p = pd.read_parquet(PRODUCTS)
    p = p[(p.product_locale == "us") & p.product_id.isin(ids)]
    p = p.drop(columns="product_locale").reset_index(drop=True)
    p.to_parquet(CACHE)
    return p


if __name__ == "__main__":
    e = load_examples()
    p = load_products()
    print(e.groupby("split").query_id.nunique().to_dict())
    print(f"{len(e)} pairs, {len(p)} products")
    assert e.groupby("query_id").split.nunique().max() == 1
    assert set(e.product_id) <= set(p.product_id)
