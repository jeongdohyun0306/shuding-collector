import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sc.kparse import parse


def same(pos, neg, case):
    return pos == case["expected_pos"] and sorted(neg) == sorted(case["expected_neg"])


def main():
    cases = json.load(open(ROOT / "bench" / "audit" / "parser_stress.json", encoding="utf-8"))["cases"]
    judged = sum(c["correct"] for c in cases)
    old = sum(same(c["got_pos"], c["got_neg"], c) for c in cases)
    now = 0
    for c in cases:
        p = parse(re.sub(r"^\[[^\]]*\]\s*", "", c["query"]))
        now += same(p["pos"], p["neg"], c)
    print(f"{len(cases)} cases")
    print(f"old whitespace rules: auditor judged correct {judged}, exact match {old}")
    print(f"current sc/kparse.py: exact match {now}")


if __name__ == "__main__":
    main()
