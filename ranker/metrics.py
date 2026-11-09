"""Ranking metrics. Gains follow the ESCI KDD Cup 2022 task 1 setup."""

import math

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}


def dcg(gains, k):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(ranked_gains, all_gains, k=10):
    """ranked_gains: gains in the order the model ranked them.
    all_gains: every labelled gain for the query, used for the ideal ranking."""
    ideal = dcg(sorted(all_gains, reverse=True), k)
    return dcg(ranked_gains, k) / ideal if ideal > 0 else 0.0


def recall(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def mean_ndcg(e, score, split, k=10):
    """Mean NDCG@k over the split's queries when each query's labelled products are
    sorted by score. LightGBM's built-in NDCG counts all-Irrelevant queries as 1.0, this counts them as 0."""
    m = (e.split == split).to_numpy()
    s = e[m].assign(score=score[m])
    out = [
        ndcg(g.sort_values("score", ascending=False, kind="stable").gain.tolist(), g.gain.tolist(), k)
        for _, g in s.groupby("query_id")
    ]
    return sum(out) / len(out)


def ndcg_both(e, score, split):
    """NDCG@10, and NDCG over each query's whole list, which is what the KDD Cup leaderboard scored."""
    return f"ndcg@10={mean_ndcg(e, score, split):.4f} ndcg={mean_ndcg(e, score, split, None):.4f}"


def mean_recall(e, retrieved, pos, k):
    """e: labelled pairs. retrieved: query -> ranked product rows. pos: product_id -> row.
    Queries with no Exact product are skipped."""
    e = e[e.esci_label == "E"]
    out = [recall(list(retrieved[q]), exact.tolist(), k) for q, exact in e.product_id.map(pos).groupby(e["query"])]
    return sum(out) / len(out)


if __name__ == "__main__":
    assert ndcg([1, 0.1, 0], [0, 0.1, 1]) == 1.0
    assert ndcg([0, 0.1, 1], [0, 0.1, 1]) < 1.0
    assert ndcg([0, 0], [0, 0]) == 0.0
    assert abs(ndcg([0, 1], [1, 0]) - 1 / math.log2(3)) < 1e-9
    assert ndcg([0] * 10 + [1], [1] + [0] * 10) == 0.0
    assert ndcg([0] * 10 + [1], [1] + [0] * 10, None) > 0.0
    assert recall(["a", "b", "c"], ["c", "d"], 2) == 0.0
    assert recall(["a", "b", "c"], ["c", "d"], 3) == 0.5
    print("ok")
