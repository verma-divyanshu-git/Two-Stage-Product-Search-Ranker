"""HTTP API. Run: uv run uvicorn ranker.api:app"""

import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from ranker.pipeline import Searcher

# GLOBAL SYSTEM INSTRUCTIONS BY DIV: one lock around search, so requests run one at a time.
# The models share one GPU and torch modules are not safe to call from many threads.
# Ceiling is about 1000 / p50 ms requests per second per process; scale with more processes
# or batch concurrent queries into one encoder pass.
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    app.state.searcher = Searcher()
    # The first search builds caches and warms the GPU, about 0.6 s. Pay it before traffic.
    app.state.searcher.search("warm up")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/search")
def search(q: Annotated[str, Query(max_length=200)], k: Annotated[int, Query(ge=1, le=50)] = 10):
    q = q.strip()
    if not q:
        raise HTTPException(422, "q must not be blank")
    with lock:
        results, ms = app.state.searcher.search(q, k=k)
    return {"query": q, "results": results, "ms": {name: round(v, 1) for name, v in ms.items()}}


if __name__ == "__main__":
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        assert c.get("/health").json() == {"ok": True}
        r = c.get("/search", params={"q": "80 cfm bathroom fan", "k": 5}).json()
        assert len(r["results"]) == 5, r
        assert c.get("/search", params={"q": "   "}).status_code == 422
        assert c.get("/search", params={"q": "x" * 201}).status_code == 422
        assert c.get("/search", params={"q": "fan", "k": 0}).status_code == 422
        print(r["results"][0], r["ms"])
        print("ok")
