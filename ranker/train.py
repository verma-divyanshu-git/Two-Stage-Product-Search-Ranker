"""Train a LightGBM LambdaMART ranker on the labelled pairs."""

import sys
from pathlib import Path

import lightgbm as lgb

from ranker.data import load_examples
from ranker.features import build
from ranker.metrics import GAIN, mean_ndcg

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
    return e, build(e)


def dataset(e, f, split):
    m = (e.split == split).to_numpy()
    group = e[m].groupby("query_id", sort=False).size().to_numpy()
    return lgb.Dataset(f[m], e.esci_label[m].map(LABEL), group=group)


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


def evaluate(split):
    e, f = load()
    model = lgb.Booster(model_file=MODEL)
    print(f"{split} bm25 only ndcg@10={mean_ndcg(e, f.bm25.to_numpy(), split):.4f}")
    # LightGBM matches features by position, so select them by the model's own names.
    print(f"{split} lgbm      ndcg@10={mean_ndcg(e, model.predict(f[model.feature_name()]), split):.4f}")
    gain = model.feature_importance("gain")
    for name, g in sorted(zip(model.feature_name(), gain / gain.sum()), key=lambda x: -x[1]):
        print(f"  {name:16s} {g:.1%}")


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    train() if split == "train" else evaluate(split)
