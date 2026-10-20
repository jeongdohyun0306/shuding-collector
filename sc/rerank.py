import numpy as np
import torch

from . import models

PROMPT = "Retrieve photos of a cat that match the user's description."


@torch.inference_mode()
def scores(engine, query, idxs):
    key = lambda i: f"rr|{query}|{engine.c.items[i]['id']}"
    miss = [i for i in idxs if key(i) not in engine.qcache]
    if miss:
        m = models.qvr()
        pairs = [(query, {"image": models.load_image(engine.c.files[i])}) for i in miss]
        vals = m.predict(pairs, batch_size=4, prompt=PROMPT)
        for i, v in zip(miss, vals):
            engine.qcache[key(i)] = float(v)
        engine.save_cache()
    return np.array([engine.qcache[key(i)] for i in idxs], dtype=np.float32)
