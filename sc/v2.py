import json
from dataclasses import dataclass, field

import numpy as np

from . import index, kparse, models


def z(s):
    s = np.asarray(s, dtype=np.float32)
    return (s - s.mean(-1, keepdims=True)) / (s.std(-1, keepdims=True) + 1e-6)


@dataclass
class Cfg:
    w_eg2_img: float = 1.0
    w_eg2_cap: float = 0.0
    w_qve_img: float = 0.0
    parse_neg: bool = True
    neg_lambda: float = 0.0
    neg_tau: float = 1.0
    rerank_k: int = 0
    w_rerank: float = 1.0
    verify_k: int = 0
    w_verify: float = 4.0
    verify_pos: bool = False
    w_en: float = 0.0
    names: list = field(default_factory=lambda: list(kparse.NAMES))


class V2:
    def __init__(self, corpus, cfg: Cfg, persist=True, memo=True):
        self.c, self.cfg, self.persist, self.memo = corpus, cfg, persist, memo
        self.eg2_img = corpus.cached("eg2_img.npy", lambda: index.eg2_images(corpus.files))
        self._caps = None
        self._qve = None
        self.cache_path = corpus.path("query_cache.json")
        self.qcache = {}
        if persist:
            try:
                self.qcache = json.load(open(self.cache_path, encoding="utf-8"))
            except FileNotFoundError:
                pass

    def save_cache(self):
        if self.persist:
            json.dump(self.qcache, open(self.cache_path, "w", encoding="utf-8"), ensure_ascii=False)

    @property
    def caps(self):
        if self._caps is None:
            caps = self.c.cached("captions.json", None)
            self._caps = self.c.cached("eg2_cap.npy", lambda: index.eg2_texts(caps, query=False))
        return self._caps

    @property
    def qve_img(self):
        if self._qve is None:
            self._qve = self.c.cached("qve_img.npy", lambda: index.qve_images(self.c.files))
        return self._qve

    def parse(self, q):
        if self.cfg.parse_neg:
            return kparse.parse(q, self.cfg.names)
        return {"pos": kparse.normalize(q, self.cfg.names), "neg": []}

    def _cached_vec(self, kind, texts, fn):
        if not self.memo:
            return np.asarray(fn(list(texts)), dtype=np.float32)
        miss = [t for t in texts if f"{kind}|{t}" not in self.qcache]
        if miss:
            for t, v in zip(miss, fn(miss)):
                self.qcache[f"{kind}|{t}"] = v.tolist()
            self.save_cache()
        return np.array([self.qcache[f"{kind}|{t}"] for t in texts], dtype=np.float32)

    def first_stage(self, texts, en=True):
        cfg = self.cfg
        S = 0
        if cfg.w_en and en:
            from . import verify
            S = cfg.w_en * self.first_stage(verify.english(self, texts), en=False)
        if cfg.w_eg2_img or cfg.w_eg2_cap:
            q = self._cached_vec("eg2", texts, index.eg2_texts)
            if cfg.w_eg2_img:
                S = S + cfg.w_eg2_img * z(q @ self.eg2_img.T)
            if cfg.w_eg2_cap:
                S = S + cfg.w_eg2_cap * z(q @ self.caps.T)
        if cfg.w_qve_img:
            q = self._cached_vec("qve", texts, index.qve_texts)
            S = S + cfg.w_qve_img * z(q @ self.qve_img.T)
        return np.asarray(S, dtype=np.float32)

    def scores(self, queries):
        cfg = self.cfg
        parsed = [self.parse(q) for q in queries]
        S = self.first_stage([p["pos"] for p in parsed])
        negs = sorted({n for p in parsed for n in p["neg"]})
        if negs and cfg.neg_lambda:
            N = dict(zip(negs, self.first_stage(negs)))
            for qi, p in enumerate(parsed):
                for n in p["neg"]:
                    S[qi] -= cfg.neg_lambda * np.maximum(0, N[n] - cfg.neg_tau)
        if cfg.rerank_k:
            S = self.rerank(queries, parsed, S)
        if cfg.verify_k:
            S = self.verify(queries, parsed, S)
        return S

    def rerank(self, queries, parsed, S):
        from . import rerank
        out = S.copy()
        for qi, p in enumerate(parsed):
            top = np.argsort(-S[qi])[: self.cfg.rerank_k]
            r = rerank.scores(self, p["pos"], top)
            out[qi, top] = S[qi, top].max() + 1 + self.cfg.w_rerank * z(r) + z(S[qi, top])
        return out

    def verify(self, queries, parsed, S):
        from . import verify
        out = S.copy()
        for qi, p in enumerate(parsed):
            stmts = verify.statements(self, p, self.cfg.verify_pos)
            if not stmts:
                continue
            top = np.argsort(-S[qi])[: self.cfg.verify_k]
            pen = verify.penalty(self, top, stmts)
            base = S[qi, top]
            out[qi, top] = base - self.cfg.w_verify * pen
        return out
