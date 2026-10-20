import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.labels import load_labels

CODES = ["L", "S", "T", "C", "sl", "yw", "mo", "ea", "pl", "lk", "gl", "ht", "cl", "tie",
         "in", "out", "gs", "sf", "bd", "bx", "sk", "wd", "cr", "lp", "tv", "sh", "pr", "hd", "dg", "mc", "nc"]


def parse_codes(s):
    toks = s.split()
    return {t for t in toks if not t.startswith("?")}, {t[1:] for t in toks if t.startswith("?")}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(path):
    _, lab = load_labels()
    other = {r["idx"]: parse_codes(r["codes"]) for r in json.load(open(path, encoding="utf-8"))}
    rows = []
    tot_a = tot_d = 0
    for c in CODES:
        a, b, diff = [], [], []
        for i, (s2, u2) in other.items():
            s1, u1 = lab[i]
            if c in u1 or c in u2:
                continue
            a.append(int(c in s1))
            b.append(int(c in s2))
            if a[-1] != b[-1]:
                diff.append(i)
        n_pos = sum(a) + sum(b)
        agree = sum(x == y for x, y in zip(a, b))
        tot_a += agree
        tot_d += len(a)
        rows.append((c, len(a), agree / len(a) if a else 1, kappa(a, b) if n_pos else float("nan"), sum(a), sum(b), diff))
    print(f"photos compared: {len(other)}   overall code agreement {tot_a / tot_d:.3f}")
    for c, n, ag, k, p1, p2, diff in rows:
        print(f"{c:4s} n={n:3d} agree {ag:.3f} kappa {k:5.2f}  mine {p1:2d} other {p2:2d}  disagree {diff[:12]}")


if __name__ == "__main__":
    main(sys.argv[1])
