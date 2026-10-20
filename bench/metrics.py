import math


def condense(ranking, rel):
    return [i for i in ranking if i in rel]


def ndcg_at(ranking, rel, k=10):
    r = condense(ranking, rel)[:k]
    dcg = sum(rel[i] / math.log2(p + 2) for p, i in enumerate(r))
    npos = sum(rel.values())
    idcg = sum(1 / math.log2(p + 2) for p in range(min(k, npos)))
    return dcg / idcg if idcg else 0.0


def ap(ranking, rel):
    r = condense(ranking, rel)
    npos = sum(rel.values())
    hit, s = 0, 0.0
    for p, i in enumerate(r):
        if rel[i]:
            hit += 1
            s += hit / (p + 1)
    return s / npos if npos else 0.0


def p_at(ranking, rel, k=5):
    r = condense(ranking, rel)[:k]
    return sum(rel[i] for i in r) / k


def hit1(ranking, rel):
    r = condense(ranking, rel)
    return float(bool(r) and rel[r[0]] == 1)


def violation_at(ranking, rel, labels, neg, k=10):
    r = condense(ranking, rel)[:k]
    bad = sum(1 for i in r if labels[i][0] & set(neg))
    return bad / k
