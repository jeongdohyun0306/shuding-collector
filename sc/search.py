import argparse
import json
from pathlib import Path

import numpy as np

from . import kparse
from .index import ROOT, Corpus
from .v1 import V1
from .v2 import FINAL, V2, Cfg

NEEDED = ("eg2_img.npy", "captions.json", "eg2_cap.npy")


def load(index_dir=None, names=None):
    if isinstance(names, str):
        names = [names]
    if index_dir:
        idx = Path(index_dir).resolve()
        corpus_json = idx / "corpus.json"
    else:
        idx, corpus_json = ROOT / "index", ROOT / "data" / "corpus.json"
    if not corpus_json.exists():
        raise SystemExit(f"{corpus_json} not found. Build it first: python -m sc.build --photos <folder> --out {idx}")
    c = Corpus(corpus_json, idx)
    for name in NEEDED:
        if not c.path(name).exists():
            raise SystemExit(f"{c.path(name)} is missing. Build it first: python -m sc.build --photos <folder> --out {idx}")
    cfg = Cfg(**FINAL, names=list(names) if names else list(kparse.NAMES))
    return c, V2(c, cfg, persist=False, memo=False)


def rank(c, scores, k):
    order = np.argsort(-scores, kind="stable")[:k]
    return [{"rank": n + 1, "id": c.items[i]["id"], "file": c.files[i], "score": round(float(scores[i]), 3)}
            for n, i in enumerate(order)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--index")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--name", action="append", dest="names")
    ap.add_argument("--v1", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.k < 1:
        ap.error("--k must be at least 1")
    c, v2 = load(a.index, a.names)
    out = []
    S2 = v2.scores(a.queries)
    S1 = V1(c, names=v2.cfg.names).scores(a.queries) if a.v1 else None
    for qi, q in enumerate(a.queries):
        row = {"query": q, "parse": v2.parse(q), "v2": rank(c, S2[qi], a.k)}
        if S1 is not None:
            row["v1"] = rank(c, S1[qi], a.k)
        out.append(row)
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    for row in out:
        print(row["query"])
        print(f"  pos: {row['parse']['pos']}")
        print(f"  neg: {row['parse']['neg']}")
        for key in ("v2", "v1"):
            if key in row:
                print(f"  [{key}]")
                for r in row[key]:
                    print(f"    {r['rank']:2d}  {r['score']:+.3f}  {r['id']}  {r['file']}")


if __name__ == "__main__":
    main()
