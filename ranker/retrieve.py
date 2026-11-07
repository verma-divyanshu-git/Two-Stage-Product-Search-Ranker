"""Stage one: hybrid retrieval. BM25 and dense candidates merged by reciprocal rank fusion."""

from collections import defaultdict


def rrf(lists, n, k=60):
    """Merge ranked lists of product rows. A product scores sum(1 / (k + rank)) over
    the lists it appears in. Uses ranks only, so BM25 and cosine scales never mix.
    k=60 is the constant from the original RRF paper (Cormack et al., 2009)."""
    score = defaultdict(float)
    for rows in lists:
        for rank, row in enumerate(rows, 1):
            score[int(row)] += 1 / (k + rank)
    return sorted(score, key=score.get, reverse=True)[:n]


if __name__ == "__main__":
    assert rrf([[1, 2, 3], [3, 4]], 10) == [3, 1, 2, 4]
    assert rrf([[1, 2, 3], [3, 4]], 2) == [3, 1]
    assert rrf([[], [5]], 10) == [5]
    print("ok")
