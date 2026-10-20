import numpy as np

from . import index, kparse

NAMES = list(kparse.NAMES)


def preprocess(q, names=NAMES):
    return kparse.normalize(q, names)


class V1:

    def __init__(self, corpus, raw=False, names=NAMES):
        self.c = corpus
        self.raw = raw
        self.names = names
        self.img = corpus.cached("eg2_img.npy", lambda: index.eg2_images(corpus.files))

    def scores(self, queries):
        qs = queries if self.raw else [preprocess(q, self.names) for q in queries]
        q = index.eg2_texts(qs)
        return q @ self.img.T

    def search(self, query, k=None):
        s = self.scores([query])[0]
        order = np.argsort(-s)
        return order[:k] if k else order, s
