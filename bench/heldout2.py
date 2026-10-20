import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.labels import codes_in
from bench.report_all import LIGHT, by_type, eval_queries, paired
from sc.index import Corpus
from sc.v1 import V1
from sc.v2 import V2, Cfg

VALID = {"L", "S", "T", "C", "sl", "yw", "mo", "gr", "ea", "pl", "lk", "gl", "ht", "cl", "tie", "in", "out", "gs",
         "sf", "bd", "bx", "sk", "wd", "cr", "lp", "tv", "sh", "pr", "hd", "dg", "mc"}


def collect(path):
    return json.load(open(path, encoding="utf-8"))["result"]


def main(journal):
    sets = collect(journal)
    kept, log = [], {"kept": 0, "fixed": 0, "dropped": 0, "dropped_items": []}
    for s in sets:
        verdict = {v["text"]: v for v in s["verdicts"]}
        for q in s["queries"]:
            v = verdict.get(q["text"])
            expr = q["expr"]
            if v is None:
                log["dropped"] += 1
                log["dropped_items"].append((q["text"], "no verdict"))
                continue
            if not v["matches"]:
                if v.get("fixed_expr"):
                    expr = v["fixed_expr"]
                    log["fixed"] += 1
                else:
                    log["dropped"] += 1
                    log["dropped_items"].append((q["text"], v["problem"][:120]))
                    continue
            if not codes_in(expr) <= VALID:
                log["dropped"] += 1
                log["dropped_items"].append((q["text"], f"bad codes {codes_in(expr) - VALID}"))
                continue
            negs = sorted({c for c in codes_in(expr) if f"not x.{c}" in expr or f"not x.__getattr__('{c}')" in expr})
            typ = "negation" if negs else q["type"] if q["type"] != "negation" else "compound"
            kept.append(dict(text=q["text"], expr=expr, type=typ, neg=negs, brief=s["key"]))
            log["kept"] += 1
    json.dump(kept, open(ROOT / "bench" / "heldout2_queries.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    c = Corpus()
    systems = {"v1": V1(c), "v2": V2(c, Cfg(**LIGHT), persist=False)}
    res, dropped_pos = eval_queries(systems, kept)
    out = {"log": log, "dropped_few_positives": dropped_pos, "rows": res}
    texts = [r["text"] for r in res["v1"]]
    brief_of = {q["text"]: q["brief"] for q in kept}
    summ = {"all": {"v1": by_type(res["v1"]), "v2": by_type(res["v2"]),
                    "paired": paired([r["ndcg10"] for r in res["v1"]], [r["ndcg10"] for r in res["v2"]])}}
    for b in sorted(set(brief_of.values())):
        m = [brief_of[t] == b for t in texts]
        r1 = [r for r, k in zip(res["v1"], m) if k]
        r2 = [r for r, k in zip(res["v2"], m) if k]
        summ[b] = {"v1": by_type(r1), "v2": by_type(r2), "paired": paired([r["ndcg10"] for r in r1], [r["ndcg10"] for r in r2])}
    out["summary"] = summ
    json.dump(out, open(ROOT / "results" / "heldout2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False, indent=1))
    print("dropped (fewer than 2 positives):", dropped_pos)
    for k, s in summ.items():
        p = s["paired"]
        line = f"{k:11s} n={s['v1']['all']['n']:2d}  v1 {s['v1']['all']['ndcg10']:.3f} -> v2 {s['v2']['all']['ndcg10']:.3f}  diff {p['diff']:+.3f} [{p['lo']:+.3f}, {p['hi']:+.3f}] p={p['p']:.4f}  better {p['better']} worse {p['worse']}"
        if "negation" in s["v1"]:
            line += f"  | neg {s['v1']['negation']['ndcg10']:.3f}->{s['v2']['negation']['ndcg10']:.3f} viol {s['v1']['negation'].get('viol10', 0):.3f}->{s['v2']['negation'].get('viol10', 0):.3f}"
        print(line)
    for typ in ("single", "compound", "negation"):
        if typ in summ["all"]["v1"]:
            print(f"  {typ:9s} n={summ['all']['v1'][typ]['n']:2d} v1 {summ['all']['v1'][typ]['ndcg10']:.3f} -> v2 {summ['all']['v2'][typ]['ndcg10']:.3f}")
    for r1, r2 in zip(res["v1"], res["v2"]):
        print(f"  {r1['ndcg10']:.2f} -> {r2['ndcg10']:.2f}  [{brief_of[r1['text']][:4]}|{r1['type'][:3]}] {r1['text']}")


if __name__ == "__main__":
    main(sys.argv[1])
