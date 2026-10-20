import numpy as np

from . import models, vlm


def english(engine, phrases):
    key = lambda p: f"tr|{p}"
    miss = [p for p in phrases if key(p) not in engine.qcache]
    if miss:
        for p, e in zip(miss, vlm.translate(miss)):
            engine.qcache[key(p)] = e
        engine.save_cache()
    return [engine.qcache[key(p)] for p in phrases]


def statements(engine, parsed, use_pos=False):
    out = [("neg", e) for e in english(engine, parsed["neg"])]
    if use_pos:
        out += [("pos", e) for e in english(engine, [parsed["pos"]])]
    return out


def p_yes(engine, idxs, stmt):
    key = lambda i: f"yn|{stmt}|{engine.c.items[i]['id']}"
    miss = [i for i in idxs if key(i) not in engine.qcache]
    if miss:
        ims = [models.load_image(engine.c.files[i]) for i in miss]
        for i, p in zip(miss, vlm.p_yes(ims, [stmt] * len(ims))):
            engine.qcache[key(i)] = p
        engine.save_cache()
    return np.array([engine.qcache[key(i)] for i in idxs], dtype=np.float32)


def penalty(engine, idxs, stmts):
    pen = np.zeros(len(idxs), dtype=np.float32)
    for kind, st in stmts:
        p = p_yes(engine, idxs, st)
        pen += p if kind == "neg" else (1 - p)
    return pen
