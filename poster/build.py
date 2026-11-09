"""Build the one-page poster from results.json: poster.html with inline SVG, then PDF and PNG via headless Chrome.

    python poster/build.py
"""

import json
import subprocess
from html import escape
from pathlib import Path

HERE = Path(__file__).parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H = 1650, 1350  # CSS px, 11:9 landscape
NAVY, BLUE, ORANGE, GREY, LIGHT = "#1f3a68", "#2f6fb5", "#e07b28", "#6b7280", "#eef3fa"


def box(x, y, w, h, text, fill=LIGHT, stroke=BLUE, size=13, bold=False):
    lines = text.split("\n")
    top = y + h / 2 - (len(lines) - 1) * size * 0.6
    tspans = "".join(
        f'<tspan x="{x + w / 2}" y="{top + i * size * 1.2 + size * 0.35}">{escape(t)}</tspan>' for i, t in enumerate(lines)
    )
    weight = "700" if bold else "400"
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>'
        f'<text font-size="{size}" font-weight="{weight}" text-anchor="middle" fill="#1b1b1b">{tspans}</text>'
    )


def arrow(x1, y1, x2, y2, label="", color=GREY):
    mid = f'<text x="{(x1 + x2) / 2 + 6}" y="{(y1 + y2) / 2 + 4}" font-size="11" fill="{GREY}">{escape(label)}</text>' if label else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.6" marker-end="url(#a)"/>{mid}'


def svg(w, h, body):
    marker = (
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{GREY}"/></marker></defs>'
    )
    return f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="inherit">{marker}{body}</svg>'


def split_diagram():
    b = box(0, 34, 140, 46, "official train\n20.9k queries", "#fff")
    b += box(0, 156, 140, 34, "official test, 9k q", "#fff", ORANGE)
    rows = [("train 50%", "LightGBM", 0), ("ce_train 40%", "cross-encoder", 52), ("valid 10%", "early stopping", 104)]
    for name, use, y in rows:
        b += arrow(140, 57, 196, y + 17)
        b += box(200, y, 120, 34, name) + f'<text x="330" y="{y + 21}" font-size="12.5" fill="{GREY}">{use}</text>'
    b += arrow(140, 173, 196, 173) + box(200, 156, 120, 34, "test", "#fdf0e6", ORANGE)
    b += f'<text x="330" y="177" font-size="12.5" fill="{GREY}">scored once, at the end</text>'
    return svg(490, 192, b)


def system_diagram():
    steps = [
        ("query", "", True),
        ("BM25 top 200  +  bi-encoder HNSW top 200", "stage 1: retrieve", False),
        ("reciprocal rank fusion: 200 candidates", "", False),
        ("light LightGBM, 10 cheap features", "stage 2: rank", False),
        ("fine-tuned cross-encoder on top 50", "", False),
        ("full LightGBM, adds cross-encoder score", "", False),
        ("top 10", "", True),
    ]
    b, y = "", 0
    for i, (text, note, edge) in enumerate(steps):
        b += box(120, y, 330, 34, text, "#fff" if edge else LIGHT, NAVY if edge else BLUE, 13, edge)
        if note:
            b += f'<text x="0" y="{y + 22}" font-size="12.5" font-weight="700" fill="{ORANGE}">{note}</text>'
        if i < len(steps) - 1:
            b += arrow(285, y + 34, 285, y + 48)
        y += 50
    return svg(490, y - 16, b)


def cross_diagram():
    b = box(0, 0, 150, 40, "[CLS] query [SEP]\nproduct title [SEP]", "#fff")
    b += box(190, 0, 150, 40, "MiniLM-L6, 22M\n6 layers, 64 tokens", "#fdf0e6", ORANGE, 13, True)
    b += box(380, 0, 110, 40, "relevance\nlogit")
    b += arrow(150, 20, 186, 20) + arrow(340, 20, 376, 20)
    b += box(190, 76, 150, 40, "target = ESCI gain\n1, 0.1, 0.01 or 0")
    b += box(380, 76, 110, 40, "BCE with\nlogits")
    b += arrow(435, 40, 435, 72) + arrow(340, 96, 376, 96)
    return svg(490, 118, b)


def ladder_chart(rows, lines):
    w, h, l, r, t, bot = 490, 300, 34, 40, 18, 44
    lo, hi = 0.55, 0.92
    n = len(rows)
    x = lambda i: l + 10 + i * (w - l - r - 10) / (n - 1)
    y = lambda v: t + (hi - v) / (hi - lo) * (h - t - bot)
    b = ""
    for v in (0.6, 0.7, 0.8, 0.9):
        b += f'<line x1="{l}" x2="{w}" y1="{y(v)}" y2="{y(v)}" stroke="#e5e7eb"/>'
        b += f'<text x="{l - 6}" y="{y(v) + 4}" font-size="11" text-anchor="end" fill="{GREY}">{v:.1f}</text>'
    for ref in lines:
        b += f'<line x1="{l}" x2="{w}" y1="{y(ref["value"])}" y2="{y(ref["value"])}" stroke="{GREY}" stroke-dasharray="4 4"/>'
        b += f'<text x="{w - 4}" y="{y(ref["value"]) + 14}" font-size="11" text-anchor="end" fill="{GREY}">{escape(ref["label"])}</text>'
    for key, color in (("ndcg", NAVY), ("ndcg10", ORANGE)):
        pts = [(x(i), y(row[key]), row[key]) for i, row in enumerate(rows) if row.get(key) is not None]
        b += f'<polyline points="{" ".join(f"{a},{c}" for a, c, _ in pts)}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        for a, c, v in pts:
            b += f'<circle cx="{a}" cy="{c}" r="3.5" fill="{color}"/>'
            ty = c - 9 if key == "ndcg" else c + 19
            b += f'<text x="{a}" y="{ty}" font-size="11.5" font-weight="700" text-anchor="middle" fill="{color}">{v:.3f}</text>'
    for i, row in enumerate(rows):
        for j, part in enumerate(row["step"].split("\n")):
            b += f'<text x="{x(i)}" y="{h - bot + 18 + j * 14}" font-size="11.5" text-anchor="middle">{escape(part)}</text>'
    return svg(w, h, b)


def bars(rows, keys, unit="", hi=None, fmt="{:.3f}"):
    w, label_w, bh, gap = 490, 150, 13, 12
    hi = hi or max(row[k] for row in rows for k, _ in keys)
    b, y = "", 0
    for row in rows:
        b += f'<text x="{label_w - 8}" y="{y + bh * len(keys) / 2 + 4}" font-size="12" text-anchor="end">{escape(row["name"])}</text>'
        for j, (k, color) in enumerate(keys):
            bw = row[k] / hi * (w - label_w - 60)
            b += f'<rect x="{label_w}" y="{y + j * bh}" width="{bw}" height="{bh - 2}" fill="{color}"/>'
            b += f'<text x="{label_w + bw + 4}" y="{y + j * bh + 10}" font-size="11">{fmt.format(row[k])}{unit}</text>'
        y += bh * len(keys) + gap
    return svg(w, y, b)


def legend(items):
    return "".join(f'<span class="key"><i style="background:{c}"></i>{escape(t)}</span>' for t, c in items)


def page(r):
    labels = "".join(
        f'<tr><td><b>{k}</b></td><td>{escape(name)}</td><td>{gain}</td><td>{share}</td></tr>'
        for k, name, gain, share in r["labels"]
    )
    takeaways = "".join(f"<li>{escape(t)}</li>" for t in r["takeaways"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: {W}px {H}px; margin: 0 }}
* {{ box-sizing: border-box }}
body {{ margin: 0; width: {W}px; height: {H}px; font-family: "Helvetica Neue", Arial, sans-serif; color: #1b1b1b; background: #fff }}
header {{ height: 118px; padding: 26px 48px 0; border-bottom: 4px solid {NAVY} }}
h1 {{ margin: 0; font-size: 40px; color: {NAVY}; letter-spacing: -0.5px }}
.sub {{ margin-top: 8px; font-size: 18px; color: {GREY} }}
main {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 40px; padding: 22px 48px 0 }}
h2 {{ margin: 18px 0 8px; font-size: 22px; color: {NAVY}; border-bottom: 2px solid {LIGHT}; padding-bottom: 4px }}
section:first-child h2 {{ margin-top: 0 }}
p, li {{ font-size: 14.5px; line-height: 1.45; margin: 6px 0 }}
ul {{ margin: 4px 0; padding-left: 20px }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px }}
td, th {{ padding: 4px 8px; border-bottom: 1px solid #e5e7eb; text-align: left }}
th {{ color: {GREY}; font-weight: 600 }}
.big {{ display: flex; gap: 14px; margin: 10px 0 }}
.big div {{ flex: 1; background: {LIGHT}; border-radius: 8px; padding: 10px 12px }}
.big b {{ display: block; font-size: 28px; color: {NAVY} }}
.big span {{ font-size: 12.5px; color: {GREY} }}
.key {{ font-size: 12px; margin-right: 14px; color: {GREY} }}
.key i {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px; vertical-align: -1px }}
.math {{ font-family: Georgia, serif; font-size: 15px; background: #fafafa; padding: 6px 10px; border-left: 3px solid {BLUE} }}
</style></head><body>
<header><h1>Two-Stage Product Search Ranker</h1>
<div class="sub">Hybrid retrieval, a LightGBM cascade and a fine-tuned cross-encoder on the Shopping Queries (ESCI) dataset</div></header>
<main>
<div>
<section><h2>Problem</h2>
<p>Given a shopping query, rank products from a 482k catalog so the ones that exactly match come first, within a 100 ms budget.</p>
<p>Every labelled query-product pair carries one of four labels:</p>
<table><tr><th></th><th>label</th><th>gain</th><th>share</th></tr>{labels}</table>
</section>
<section><h2>Data</h2>
<p>US English queries, split by query so none lands in two splits. The cross-encoders never train on pairs LightGBM learns from.</p>
{split_diagram()}
</section>
<section><h2>Metrics</h2>
<p class="math">NDCG@k = DCG@k / ideal DCG@k, &nbsp; DCG@k = &Sigma;<sub>i&le;k</sub> gain<sub>i</sub> / log<sub>2</sub>(i+1)</p>
<p><b>Full-list NDCG</b> scores each query's whole labelled list, the KDD Cup leaderboard metric. <b>NDCG@10</b> looks only at what a shopper sees and is much stricter.</p>
<p><b>Recall@k</b> is the share of Exact products that retrieval finds in its top k of the whole catalog.</p>
</section>
<section><h2>Latency, {r["bench_queries"]} replayed queries</h2>
<div>{legend([("p50", BLUE), ("p95", ORANGE)])}</div>
{bars(r["latency_bars"], [("p50", BLUE), ("p95", ORANGE)], " ms", fmt="{:g}")}
</section>
</div>
<div>
<section><h2>System</h2>{system_diagram()}</section>
<section><h2>Fine-tuning the cross-encoder</h2>
<p>An MS MARCO cross-encoder reads query and title together. One epoch on 8.3k ce_train queries, about 5 minutes on a laptop GPU, teaches it ESCI's notion of a match.</p>
{cross_diagram()}
<div>{legend([("NDCG@10", ORANGE), ("full-list NDCG", NAVY)])}</div>
{bars(r["cross"], [("ndcg10", ORANGE), ("ndcg", NAVY)], hi=1.0)}
</section>
<section><h2>End to end over the whole catalog</h2>
<p>Each test query searches all 482k products and the top 10 is scored. Products nobody labelled count as 0, so this is a lower bound.</p>
<div>{legend([("NDCG@10", ORANGE)])}</div>
{bars(r["end_to_end"], [("ndcg10", ORANGE)], hi=0.5)}
</section>

</div>
<div>
<section><h2>Results on the official test split</h2>
<div class="big">{"".join(f'<div><b>{h["value"]}</b><span>{escape(h["label"])}</span></div>' for h in r["headline"])}</div>
<div>{legend([("full-list NDCG", NAVY), ("NDCG@10", ORANGE)])}</div>
{ladder_chart(r["ladder"], r["reference_lines"])}
</section>
<section><h2>Retrieval recall of Exact products</h2>
<div>{legend([("recall@100", BLUE), ("recall@200", ORANGE)])}</div>
{bars(r["recall"], [("r100", BLUE), ("r200", ORANGE)], hi=1.0)}
</section>
<section><h2>Drift: watch the output too</h2>
<p>PSI against a reference, on query length (input) and the top result's score (output). Above 0.25 means drift.</p>
<div>{legend([("query length", BLUE), ("top score", ORANGE)])}</div>
{bars(r["drift"], [("q_len", BLUE), ("top_score", ORANGE)])}
</section>
<section><h2>Takeaways</h2><ul>{takeaways}</ul></section>
</div>
</main></body></html>"""


if __name__ == "__main__":
    html = HERE / "poster.html"
    html.write_text(page(json.loads((HERE / "results.json").read_text())))
    url = html.resolve().as_uri()
    common = [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--no-pdf-header-footer"]
    subprocess.run([*common, f"--print-to-pdf={HERE / 'poster.pdf'}", url], check=True, capture_output=True)
    subprocess.run(
        [*common, f"--window-size={W},{H}", "--force-device-scale-factor=2", f"--screenshot={HERE / 'poster.png'}", url],
        check=True, capture_output=True,
    )
    print("wrote", html, HERE / "poster.pdf", HERE / "poster.png")
