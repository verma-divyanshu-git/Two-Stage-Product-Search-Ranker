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


def mean_recall(e, retrieved, pos, k):
    """e: labelled pairs. retrieved: query -> ranked product rows. pos: product_id -> row.
    Queries with no Exact product are skipped."""
    out = []
    for q, g in e.groupby("query"):
        exact = g.product_id[g.esci_label == "E"].map(pos).tolist()
        if exact:
            out.append(recall(list(retrieved[q]), exact, k))
    return sum(out) / len(out)


if __name__ == "__main__":
    assert ndcg([1, 0.1, 0], [0, 0.1, 1]) == 1.0
    assert ndcg([0, 0.1, 1], [0, 0.1, 1]) < 1.0
    assert ndcg([0, 0], [0, 0]) == 0.0
    assert abs(ndcg([0, 1], [1, 0]) - 1 / math.log2(3)) < 1e-9
    assert recall(["a", "b", "c"], ["c", "d"], 2) == 0.0
    assert recall(["a", "b", "c"], ["c", "d"], 3) == 0.5
    print("ok")
