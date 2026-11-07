"""Fine-tune the MS MARCO cross-encoder on ESCI ce_train pairs."""

from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from ranker.cross import ZERO_SHOT, DEVICE, load_model
from ranker.data import load_examples, load_products

OUT = Path("artifacts/ce-esci")
BATCH = 32
LR = 2e-5


def main():
    torch.manual_seed(0)
    e = load_examples()
    e = e[e.split == "ce_train"].sample(frac=1, random_state=0)
    title = load_products().set_index("product_id").product_title.fillna("")
    queries, titles, targets = e["query"].tolist(), title.loc[e.product_id].tolist(), e.gain.tolist()

    ce = load_model(ZERO_SHOT)
    model, tok = ce.model, ce.tokenizer
    model.train()
    steps = len(queries) // BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, steps // 10, steps)
    # Soft targets: the gain itself, so Substitutes sit between Exact and Irrelevant.
    loss_fn = torch.nn.BCEWithLogitsLoss()

    running = 0.0
    for step in range(steps):
        i = slice(step * BATCH, (step + 1) * BATCH)
        x = tok(queries[i], titles[i], truncation=True, max_length=64, padding=True, return_tensors="pt").to(DEVICE)
        y = torch.tensor(targets[i], dtype=torch.float32, device=DEVICE)
        loss = loss_fn(model(**x).logits.squeeze(-1), y)
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()
        running += loss.item()
        if (step + 1) % 500 == 0:
            print(f"step {step + 1}/{steps} loss {running / 500:.4f}", flush=True)
            running = 0.0

    model.eval()
    ce.save(str(OUT))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
