import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench import metrics
from bench.labels import codes_in, judge, load_labels, load_queries
from bench.report_all import LIGHT, paired
from sc import kparse
from sc.index import Corpus
from sc.v1 import V1
from sc.v2 import V2, Cfg

CATS = {
    "장소": {"in", "out", "gs", "sf", "bd", "bx", "sk", "wd", "cr"},
    "주변 물건": {"lp", "tv", "sh"},
    "자세·행동": {"L", "S", "T", "C", "sl", "yw", "mo", "gr", "ea", "pl", "lk"},
    "착용": {"gl", "ht", "cl", "tie"},
    "함께 있는 대상": {"pr", "hd", "dg", "mc"},
}
FORMS = [
    ("안/못 + 동사 (띄어 씀)", re.compile(r"(^|\s)(안|못)\s")),
    ("안/못 + 동사 (붙여 씀)", re.compile(r"(^|\s)(안|못)(?!경|쪽|에|기|겨|고|락|의|이|을|은|으|아|는)[가-힣]")),
    ("~지 않/못하/말", re.compile(r"지(도|는|를)?\s?(않|못하|말)")),
    ("없이·없는·없고", re.compile(r"없(이|는|고|어|음)")),
    ("말고·빼고·아닌·아니라·아니고", re.compile(r"말고|빼고|아닌|아니라|아니고")),
]


def negated(expr):
    return {c for c in codes_in(expr) if re.search(rf"not\s+x\.(__getattr__\('{c}'\)|{c}\b)", expr)}


def score_all(systems, texts):
    return {k: s.scores(texts) for k, s in systems.items()}


def rows_for(S, qs, lab, n_items, subset=None):
    idx = list(range(n_items)) if subset is None else subset
    out = []
    for q, s in zip(qs, S):
        rel = {i: j for i in idx if (j := judge(q["expr"], *lab[i])) is not None}
        if sum(rel.values()) < 1 or (subset is None and sum(rel.values()) < 2):
            out.append(None)
            continue
        sc = np.full(n_items, -1e9, dtype=np.float32)
        sc[idx] = s[idx]
        rk = list(np.argsort(-sc, kind="stable"))
        k = 10 if subset is None else 3
        row = dict(text=q["text"], expr=q["expr"], type=q["type"], ndcg=metrics.ndcg_at(rk, rel, k),
                   ap=metrics.ap(rk, rel), hit1=metrics.hit1(rk, rel), p5=metrics.p_at(rk, rel, 5), npos=int(sum(rel.values())))
        if q.get("neg"):
            row["viol"] = metrics.violation_at(rk, rel, lab, q["neg"], k)
        out.append(row)
    return out


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def group(name, r1, r2, sel):
    pairs = [(a, b) for a, b in zip(r1, r2) if a is not None and b is not None and sel(a)]
    if not pairs:
        return None
    a = [p[0]["ndcg"] for p in pairs]
    b = [p[1]["ndcg"] for p in pairs]
    d = dict(name=name, n=len(pairs), v1=mean(a), v2=mean(b))
    if len(pairs) >= 4:
        d.update({k: v for k, v in paired(a, b).items() if k in ("lo", "hi", "p", "better", "worse")})
    else:
        d.update(better=int(sum(y - x > 0.005 for x, y in zip(a, b))), worse=int(sum(x - y > 0.005 for x, y in zip(a, b))))
    v1v = [p[0]["viol"] for p in pairs if "viol" in p[0]]
    if v1v:
        d["viol_v1"] = mean(v1v)
        d["viol_v2"] = mean([p[1]["viol"] for p in pairs if "viol" in p[1]])
    return d


def main():
    items, lab = load_labels()
    n = len(items)
    c = Corpus()
    bench = load_queries()
    held = json.load(open(ROOT / "bench" / "heldout2_queries.json", encoding="utf-8"))
    sets = {"벤치마크 47": bench, "처음 보는 44": held}
    report = {}
    for sname, qs in sets.items():
        systems = {"v1": V1(c), "v2": V2(c, Cfg(**LIGHT), persist=False)}
        kept = [q for q in qs if sum(j for i in range(n) if (j := judge(q["expr"], *lab[i])) is not None) >= 2]
        qs = kept
        S = score_all(systems, [q["text"] for q in qs])
        r1 = rows_for(S["v1"], qs, lab, n)
        r2 = rows_for(S["v2"], qs, lab, n)
        sec = {}
        sec["overall"] = {m: {"v1": mean([r[m] for r in r1 if r]), "v2": mean([r[m] for r in r2 if r])} for m in ("ndcg", "ap", "hit1", "p5")}
        sec["by_category_positive"] = [g for cat, codes in CATS.items()
                                       if (g := group(cat, r1, r2, lambda r, codes=codes: bool((codes_in(r["expr"]) - negated(r["expr"])) & codes)))]
        sec["by_category_negated"] = [g for cat, codes in CATS.items()
                                      if (g := group(cat + " (빼는 조건)", r1, r2, lambda r, codes=codes: bool(negated(r["expr"]) & codes)))]
        sec["by_form"] = [g for fname, rx in FORMS if (g := group(fname, r1, r2, lambda r, rx=rx: bool(negated(r["expr"])) and bool(rx.search(r["text"]))))]
        sec["by_type"] = [g for t in ("single", "compound", "negation") if (g := group(t, r1, r2, lambda r, t=t: r["type"] == t))]
        sec["name"] = [g for g in (group("이름(슈딩) 들어감", r1, r2, lambda r: "슈딩" in r["text"]),
                                   group("이름 없음", r1, r2, lambda r: "슈딩" not in r["text"])) if g]
        lost = []
        for a, b in zip(r1, r2):
            if a and b and a["ndcg"] - b["ndcg"] > 0.05:
                lost.append(dict(text=a["text"], expr=a["expr"], v1=a["ndcg"], v2=b["ndcg"], parse=kparse.parse(a["text"])))
        sec["v2_lost"] = lost
        won = sorted([(b["ndcg"] - a["ndcg"], a["text"], a["ndcg"], b["ndcg"]) for a, b in zip(r1, r2) if a and b], reverse=True)[:8]
        sec["v2_biggest_wins"] = [dict(text=t, v1=x, v2=y, gain=g) for g, t, x, y in won]
        real = [i for i, it in enumerate(items) if it["id"].startswith("shuding_")]
        q1 = rows_for(S["v1"], qs, lab, n, subset=real)
        q2 = rows_for(S["v2"], qs, lab, n, subset=real)
        judged_real = {k: sum(1 for i in real if judge(q["expr"], *lab[i]) is not None) for k, q in enumerate(qs)}
        pairs = [(a, b) for k, (a, b) in enumerate(zip(q1, q2)) if a and b and a["npos"] < judged_real[k]]
        sec["real_shuding_13"] = {"n_queries": len(pairs), "v1_ndcg3": mean([a["ndcg"] for a, _ in pairs]),
                                  "v2_ndcg3": mean([b["ndcg"] for _, b in pairs]),
                                  "v1_hit1": mean([a["hit1"] for a, _ in pairs]), "v2_hit1": mean([b["hit1"] for _, b in pairs]),
                                  "queries": [dict(text=a["text"], v1=a["ndcg"], v2=b["ndcg"], npos=a["npos"]) for a, b in pairs]}
        report[sname] = sec
    json.dump(report, open(ROOT / "results" / "breakdown.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for sname, sec in report.items():
        print(f"\n######## {sname}")
        print("overall", {m: f"{v['v1']:.3f}->{v['v2']:.3f}" for m, v in sec["overall"].items()})
        for key in ("by_type", "by_category_positive", "by_category_negated", "by_form", "name"):
            print(f"-- {key}")
            for g in sec[key]:
                ci = f" [{g['lo']:+.3f},{g['hi']:+.3f}] p={g['p']:.3f}" if "lo" in g else ""
                vv = f"  위반 {g['viol_v1']:.1%}->{g['viol_v2']:.1%}" if "viol_v1" in g else ""
                print(f"   {g['name']:24s} n={g['n']:2d}  {g['v1']:.3f} -> {g['v2']:.3f} ({g['v2'] - g['v1']:+.3f}){ci}  좋아짐 {g['better']} 나빠짐 {g['worse']}{vv}")
        rs = sec["real_shuding_13"]
        print(f"-- 실제 슈딩이 13장만: 검색어 {rs['n_queries']}개  nDCG@3 {rs['v1_ndcg3']:.3f} -> {rs['v2_ndcg3']:.3f}  1위 적중 {rs['v1_hit1']:.2f} -> {rs['v2_hit1']:.2f}")
        print("-- v2가 진 검색어")
        for l in sec["v2_lost"]:
            print(f"   {l['v1']:.2f} -> {l['v2']:.2f}  {l['text']}  | {l['parse']}")
        print("-- 가장 크게 이긴 검색어")
        for w in sec["v2_biggest_wins"]:
            print(f"   {w['v1']:.2f} -> {w['v2']:.2f}  {w['text']}")


if __name__ == "__main__":
    main()
