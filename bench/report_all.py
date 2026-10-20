import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import metrics
from bench.labels import judge, load_labels, load_queries
from bench.run_eval import evaluate, summary
from bench.agreement import parse_codes
from sc import index, kparse, models
from sc.index import Corpus
from sc.v1 import V1
from sc.v2 import V2, Cfg, z

OUT = ROOT / "results"
LIGHT = dict(w_eg2_cap=1.0, neg_lambda=1.0, neg_tau=1.0)
ABLATIONS = {
    "img_only": dict(parse_neg=False),
    "img_parse": dict(),
    "img_parse_penalty": dict(neg_lambda=1.0),
    "light": LIGHT,
    "light_no_parse": dict(w_eg2_cap=1.0, parse_neg=False),
    "qve_only": dict(w_eg2_img=0, w_qve_img=1.0, parse_neg=False),
    "light_qve": dict(LIGHT, w_qve_img=1.0),
    "light_rerank30": dict(LIGHT, rerank_k=30),
    "light_verify_neg": dict(LIGHT, verify_k=30, w_verify=2.0),
    "light_verify_pos": dict(LIGHT, verify_k=30, w_verify=2.0, verify_pos=True),
    "light_english": dict(LIGHT, w_en=1.0),
}


def paired(a, b, seed=0):
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(d, len(d)).mean() for _ in range(20000)]
    perm = np.mean([abs((d * rng.choice([-1, 1], len(d))).mean()) >= abs(d.mean()) - 1e-12 for _ in range(20000)])
    return dict(diff=float(d.mean()), lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                p=float(perm), better=int((d > 0.005).sum()), worse=int((d < -0.005).sum()), same=int((abs(d) <= 0.005).sum()))


def eval_queries(systems, qs, lab_override=None):
    items, lab = load_labels()
    lab = lab_override or lab
    kept = []
    for q in qs:
        rel = {i: j for i in range(len(items)) if (j := judge(q["expr"], *lab[i])) is not None}
        if sum(rel.values()) >= 2:
            kept.append((q, rel))
    out = {}
    for name, s in systems.items():
        S = s.scores([q["text"] for q, _ in kept])
        rows = []
        for (q, rel), sc in zip(kept, S):
            rk = list(np.argsort(-sc, kind="stable"))
            row = dict(text=q["text"], type=q["type"], ndcg10=metrics.ndcg_at(rk, rel), ap=metrics.ap(rk, rel),
                       hit1=metrics.hit1(rk, rel), npos=int(sum(rel.values())))
            if q.get("neg"):
                row["viol10"] = metrics.violation_at(rk, rel, lab, q["neg"])
            rows.append(row)
        out[name] = rows
    return out, len(qs) - len(kept)


def by_type(rows):
    res = {}
    for typ in ("single", "compound", "negation", "all"):
        rs = [r for r in rows if typ == "all" or r["type"] == typ]
        if rs:
            d = {m: float(np.mean([r[m] for r in rs])) for m in ("ndcg10", "ap", "hit1")}
            v = [r["viol10"] for r in rs if "viol10" in r]
            if v:
                d["viol10"] = float(np.mean(v))
            d["n"] = len(rs)
            res[typ] = d
    return res


def main():
    c = Corpus()
    summ = {}
    t0 = time.time()
    v1 = V1(c)
    rows_v1 = evaluate("v1", v1.scores)
    rows_v1raw = evaluate("v1raw", V1(c, raw=True).scores)
    summ["bench_v1"] = summary(rows_v1)
    summ["bench_v1raw"] = summary(rows_v1raw)
    abl = {}
    for name, kw in ABLATIONS.items():
        eng = V2(c, Cfg(**kw))
        rows = evaluate(f"ablation_{name}", eng.scores)
        abl[name] = summary(rows)
        if name == "light":
            rows_v2 = rows
            evaluate("v2", eng.scores)
        models.free()
        print(f"[ablation] {name}: dev {abl[name]['dev/all']['ndcg10']:.3f} test {abl[name]['test/all']['ndcg10']:.3f}", flush=True)
    summ["ablations"] = abl
    grid = {}
    for lam in (0.5, 1.0, 2.0):
        for tau in (0.5, 1.0, 1.5, 2.0):
            rows = evaluate("tmp", V2(c, Cfg(neg_lambda=lam, neg_tau=tau)).scores)
            s = summary(rows)
            grid[f"{lam}/{tau}"] = {"dev": s["dev/all"]["ndcg10"], "dev_neg": s["dev/negation"]["ndcg10"]}
    summ["penalty_grid_dev"] = grid
    a = [r["ndcg10"] for r in rows_v1]
    b = [r["ndcg10"] for r in rows_v2]
    summ["bench_paired"] = {"all": paired(a, b)}
    for typ in ("single", "compound", "negation"):
        m = [r["type"] == typ for r in rows_v1]
        summ["bench_paired"][typ] = paired([x for x, k in zip(a, m) if k], [x for x, k in zip(b, m) if k])
    v2 = V2(c, Cfg(**LIGHT))
    h1 = json.load(open(ROOT / "bench" / "audit" / "heldout_queries.json", encoding="utf-8"))
    res, dropped = eval_queries({"v1": v1, "v2": v2}, h1)
    summ["heldout1"] = {"v1": by_type(res["v1"]), "v2": by_type(res["v2"]), "dropped": dropped,
                        "paired": paired([r["ndcg10"] for r in res["v1"]], [r["ndcg10"] for r in res["v2"]])}
    json.dump(res, open(OUT / "heldout1_after_parser.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    items, lab = load_labels()
    blind = {r["idx"]: parse_codes(r["codes"]) for r in json.load(open(ROOT / "bench" / "audit" / "blind_labels.json", encoding="utf-8"))}
    lab2 = {}
    for i, (s, u) in lab.items():
        if i in blind:
            s2, u2 = blind[i]
            dis = (s ^ s2) - u - u2
            lab2[i] = (s - dis, u | dis | u2)
        else:
            lab2[i] = (s, u)
    res2, _ = eval_queries({"v1": v1, "v2": v2}, load_queries(), lab2)
    summ["disputed_excluded"] = {"v1": by_type(res2["v1"]), "v2": by_type(res2["v2"])}
    mrl = {}
    qs = load_queries()
    P = [kparse.parse(q["text"]) for q in qs]
    Qp = index.eg2_texts([p["pos"] for p in P])
    negs = sorted({n for p in P for n in p["neg"]})
    Qn = dict(zip(negs, index.eg2_texts(negs))) if negs else {}
    for dim in (768, 512, 256, 128):
        def nrm(x):
            x = x[..., :dim]
            return x / np.linalg.norm(x, axis=-1, keepdims=True)
        I, C = nrm(v2.eg2_img), nrm(v2.caps)

        def sc(_texts):
            S = z(nrm(Qp) @ I.T) + z(nrm(Qp) @ C.T)
            for qi, p in enumerate(P):
                for n in p["neg"]:
                    v = nrm(Qn[n][None])
                    S[qi] -= np.maximum(0, (z(v @ I.T) + z(v @ C.T))[0] - 1.0)
            return S
        mrl[dim] = summary(evaluate("tmp", sc))["all/all"]["ndcg10"]
    summ["mrl"] = mrl
    lat = {}
    v2f = V2(c, Cfg(**LIGHT), persist=False, memo=False)
    v2f.scores(["예열"])
    v1.scores(["예열"])
    for _ in range(2):
        for q in qs:
            for name, eng in (("v1", v1), ("v2", v2f)):
                t = time.perf_counter(); eng.scores([q["text"]]); ms = (time.perf_counter() - t) * 1000
                lat.setdefault(name, {}).setdefault(q["type"], []).append(ms)
    summ["latency_ms"] = {name: {typ: {"median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)} for typ, v in d.items()} | {"all": {"median": float(np.median(sum(d.values(), []))), "max": float(np.max(sum(d.values(), [])))}} for name, d in lat.items()}
    summ["elapsed_s"] = time.time() - t0
    json.dump(summ, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    (OUT / "tmp.json").unlink(missing_ok=True)
    print(json.dumps({k: summ[k] for k in ("bench_paired", "heldout1", "disputed_excluded", "mrl", "latency_ms")}, ensure_ascii=False, indent=1))
    for k in ("bench_v1", "bench_v1raw"):
        print(k, {kk: round(v["ndcg10"], 3) for kk, v in summ[k].items()})
    for k, s in abl.items():
        print(f"{k:20s} dev {s['dev/all']['ndcg10']:.3f}  test {s['test/all']['ndcg10']:.3f}  all {s['all/all']['ndcg10']:.3f}  neg {s['all/negation']['ndcg10']:.3f}  cmp {s['all/compound']['ndcg10']:.3f}  sgl {s['all/single']['ndcg10']:.3f}")


if __name__ == "__main__":
    main()
