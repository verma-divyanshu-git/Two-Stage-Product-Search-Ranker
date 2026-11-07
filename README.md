# Two-Stage Product Search Ranker

Product search on Amazon's [Shopping Queries dataset (ESCI)](https://github.com/amazon-science/esci-data).
Stage one retrieves candidates from a 482k product catalog.
Stage two reranks them.

Data is the US English slice of the task 1 "small" version: 29.8k queries and 601k labelled query-product pairs.
Train queries are split 90/10 into train and valid by query, so no query lands in two splits.

## Setup

```sh
uv sync
uv run python -m ranker.download
```

## Results

NDCG@10 uses the KDD Cup gains: Exact 1.0, Substitute 0.1, Complement 0.01, Irrelevant 0.
Recall@100 counts labelled Exact products found in the top 100 of the full catalog.

| Model | Valid NDCG@10 | Test NDCG@10 | Test Recall@100 |
|---|---|---|---|
| Random order | 0.573 | - | - |
| BM25 | 0.701 | 0.703 | 0.578 |

Random order already scores 0.57 because most labels are Exact or Substitute.
