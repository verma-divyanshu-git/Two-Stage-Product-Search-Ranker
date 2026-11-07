"""Train a LightGBM LambdaMART ranker on the labelled pairs."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from ranker.data import load_examples, load_products
from ranker.features import build
from ranker.metrics import GAIN, ndcg

MODEL = Path("artifacts/lgbm.txt")
LABEL = {"I": 0, "C": 1, "S": 2, "E": 3}
PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    # Same gains as our metric. LightGBM's default would be 2^label - 1.
    "label_gain": [GAIN[k] for k in sorted(LABEL, key=LABEL.get)],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


def load():
    """Examples sorted by query, since LightGBM needs each query's rows together."""
    e = load_examples().sort_values("query_id", kind="stable").reset_index(drop=True)
    return e, build(e, load_products())


def dataset(e, f, split):
    m = (e.split == split).to_numpy()
    group = e[m].groupby("query_id", sort=False).size().to_numpy()
    return lgb.Dataset(f[m], e.esci_label[m].map(LABEL), group=group)


def mean_ndcg(e, score, split):
    """Our NDCG@10. LightGBM's own counts all-Irrelevant queries as 1.0, ours as 0."""
    m = (e.split == split).to_numpy()
    s = e[m].assign(score=score[m])
    return np.mean([
        ndcg(g.sort_values("score", ascending=False, kind="stable").gain.tolist(), g.gain.tolist())
        for _, g in s.groupby("query_id")
    ])


def train():
    e, f = load()
    train_set = dataset(e, f, "train")
    valid_set = dataset(e, f, "valid")
    model = lgb.train(
        PARAMS, train_set, num_boost_round=3000, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    model.save_model(MODEL, num_iteration=model.best_iteration)
    print(f"valid bm25 only ndcg@10={mean_ndcg(e, f.bm25.to_numpy(), 'valid'):.4f}")
    print(f"valid lgbm      ndcg@10={mean_ndcg(e, model.predict(f), 'valid'):.4f}")


if __name__ == "__main__":
    train()
