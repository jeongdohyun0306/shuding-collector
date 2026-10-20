import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_labels():
    items = json.load(open(ROOT / 'data' / 'corpus.json', encoding='utf-8'))
    lab = {}
    for line in open(ROOT / 'data' / 'labels_raw.txt', encoding='utf-8'):
        m = re.match(r'^(\d+):\s*(.*)$', line.strip())
        if not m:
            continue
        idx, codes = int(m.group(1)), m.group(2).split()
        sure = {c for c in codes if not c.startswith('?')}
        unsure = {c[1:] for c in codes if c.startswith('?')}
        lab[idx] = (sure, unsure)
    assert len(lab) == len(items), (len(lab), len(items))
    return items, lab


class Img:
    def __init__(self, sure, unsure):
        self.sure, self.unsure = sure, unsure
        self.touched = set()

    def __getattr__(self, k):
        if k in ('sure', 'unsure', 'touched'):
            raise AttributeError(k)
        self.touched.add(k)
        return k in self.sure


def judge(expr, sure, unsure):
    if 'nc' in sure:
        return 0
    from itertools import product
    codes = sorted((unsure - sure) & codes_in(expr))
    vals = set()
    for bits in product([False, True], repeat=len(codes)):
        assumed = set(sure) | {c for c, b in zip(codes, bits) if b}
        vals.add(bool(eval(expr, {}, {'x': Img(assumed, set())})))
        if len(vals) > 1:
            return None
    return int(vals.pop())


def codes_in(expr):
    return set(re.findall(r"x\.(?!__)(\w+)", expr)) | set(re.findall(r"__getattr__\('(\w+)'\)", expr))


def load_queries():
    return json.load(open(ROOT / 'bench' / 'queries.json', encoding='utf-8'))


def qrels():
    items, lab = load_labels()
    out = {}
    for q in load_queries():
        rel = {}
        for i in range(len(items)):
            j = judge(q['expr'], *lab[i])
            if j is not None:
                rel[i] = j
        out[q['id']] = rel
    return items, out


if __name__ == '__main__':
    items, lab = load_labels()
    from collections import Counter
    c = Counter()
    for s, u in lab.values():
        c.update(s)
    print(sorted(c.items(), key=lambda t: -t[1]))
    if (ROOT / 'bench' / 'queries.json').exists():
        items, qr = qrels()
        for q in load_queries():
            r = qr[q['id']]
            print(q['id'], q['split'], q['type'], q['text'], 'pos', sum(r.values()), 'judged', len(r))
