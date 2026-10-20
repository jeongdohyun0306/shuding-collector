import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import metrics
from bench.labels import load_labels, load_queries, qrels


def evaluate(name, score_fn, out_dir=ROOT / "results"):
    items, lab = load_labels()
    queries = load_queries()
    _, qr = qrels()
    S = score_fn([q["text"] for q in queries])
    rows = []
    for q, s in zip(queries, S):
        ranking = list(np.argsort(-np.asarray(s), kind="stable"))
        rel = qr[q["id"]]
        row = dict(id=q["id"], type=q["type"], split=q["split"], text=q["text"],
                   ndcg10=metrics.ndcg_at(ranking, rel, 10), ap=metrics.ap(ranking, rel),
                   p5=metrics.p_at(ranking, rel, 5), hit1=metrics.hit1(ranking, rel),
                   top10=[items[i]["id"] for i in metrics.condense(ranking, rel)[:10]])
        if q.get("neg"):
            row["viol10"] = metrics.violation_at(ranking, rel, lab, q["neg"], 10)
        rows.append(row)
    out_dir.mkdir(exist_ok=True)
    json.dump(rows, open(out_dir / f"{name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rows


def summary(rows):
    out = {}
    for split in ("dev", "test", "all"):
        for typ in ("single", "compound", "negation", "all"):
            rs = [r for r in rows if (split == "all" or r["split"] == split) and (typ == "all" or r["type"] == typ)]
            if not rs:
                continue
            d = {m: float(np.mean([r[m] for r in rs])) for m in ("ndcg10", "ap", "p5", "hit1")}
            vs = [r["viol10"] for r in rs if "viol10" in r]
            if vs:
                d["viol10"] = float(np.mean(vs))
            d["n"] = len(rs)
            out[f"{split}/{typ}"] = d
    return out


def print_summary(name, rows):
    s = summary(rows)
    print(f"== {name}")
    for k, d in s.items():
        extra = f" viol@10 {d['viol10']:.3f}" if "viol10" in d else ""
        print(f"  {k:16s} n={d['n']:2d} nDCG@10 {d['ndcg10']:.3f}  MAP {d['ap']:.3f}  P@5 {d['p5']:.3f}  Hit@1 {d['hit1']:.3f}{extra}")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    a = ap.parse_args()
    from sc.index import Corpus
    c = Corpus()
    if a.system == "v1":
        from sc.v1 import V1
        sysm = V1(c)
    elif a.system == "v1raw":
        from sc.v1 import V1
        sysm = V1(c, raw=True)
    else:
        raise SystemExit("unknown system")
    rows = evaluate(a.system, sysm.scores)
    print_summary(a.system, rows)
