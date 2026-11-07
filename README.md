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
uv run python -m ranker.bm25 test    # BM25 baseline
uv run python -m ranker.train        # train LightGBM on train, early stop on valid
uv run python -m ranker.train test   # evaluate and print feature importance
```

## Results

NDCG@10 uses the KDD Cup gains: Exact 1.0, Substitute 0.1, Complement 0.01, Irrelevant 0.
Recall@100 counts labelled Exact products found in the top 100 of the full catalog.

| Model | Valid NDCG@10 | Test NDCG@10 | Test Recall@100 |
|---|---|---|---|
| Random order | 0.573 | - | - |
| BM25 | 0.701 | 0.703 | 0.578 |
| LightGBM LambdaMART, 9 lexical features | 0.725 | 0.723 | - |

Random order already scores 0.57 because most labels are Exact or Substitute.

The ranker uses BM25 score, query and title length, query-title word overlap, brand and color match, number match, and whether a description or bullets exist.
Title overlap carries the most gain (39%), then BM25 (27%) and number match (12%).

Tried and dropped, all within 0.001 of the baseline on valid:

- BM25 rank and score relative to the query's other candidates. These also change meaning when serving ranks retrieved candidates instead of labelled ones.
- A title-only BM25 score.
- Brand mismatch, for queries that name a brand the product doesn't have.

Lexical features stop helping around 0.725.
What's left is mostly telling Exact from Substitute when the words overlap, which needs a model of meaning.
