import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import models

ROOT = Path(__file__).resolve().parent.parent


CACHE_FILES = ("eg2_img.npy", "captions.json", "eg2_cap.npy", "qve_img.npy", "query_cache.json")


class Corpus:
    def __init__(self, corpus_json=ROOT / "data" / "corpus.json", index_dir=ROOT / "index"):
        self.items = json.load(open(corpus_json, encoding="utf-8"))
        self.files = [str(ROOT / it["file"]) for it in self.items]
        self.dir = Path(index_dir)
        self.dir.mkdir(exist_ok=True)
        self.check_fingerprint()

    def fingerprint(self):
        h = hashlib.sha256()
        for it, f in zip(self.items, self.files):
            p = Path(f)
            digest = hashlib.sha1(p.read_bytes()).hexdigest() if p.exists() else "missing"
            h.update(f"{it['id']}|{digest}\n".encode("utf-8"))
        return h.hexdigest()

    def check_fingerprint(self):
        fp_path = self.path("fingerprint.txt")
        fp = self.fingerprint()
        old = fp_path.read_text(encoding="utf-8").strip() if fp_path.exists() else None
        if old == fp:
            return
        legacy_ok = old is None and self.path("eg2_img.npy").exists() and len(np.load(self.path("eg2_img.npy"))) == len(self.items)
        if not legacy_ok:
            stale = [n for n in CACHE_FILES if self.path(n).exists()]
            for n in stale:
                self.path(n).unlink()
            thumbs = self.path("thumbs")
            if thumbs.exists():
                shutil.rmtree(thumbs)
            if stale:
                print(f"[index] photo list changed, cleared {', '.join(stale)}")
        fp_path.write_text(fp, encoding="utf-8")

    def path(self, name):
        return self.dir / name

    def cached(self, name, fn):
        p = self.path(name)
        if p.exists():
            out = np.load(p) if p.suffix == ".npy" else json.load(open(p, encoding="utf-8"))
            if isinstance(out, (list, np.ndarray)) and len(out) != len(self.items):
                raise RuntimeError(f"{p} has {len(out)} rows but the corpus has {len(self.items)} photos; delete it and rebuild")
            return out
        t = time.time()
        out = fn()
        if p.suffix == ".npy":
            np.save(p, out)
        else:
            json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"[index] {name}: {time.time() - t:.1f}s")
        return out


def l2n(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


@torch.inference_mode()
def eg2_images(files, bs=8):
    m = models.eg2()
    out = []
    for s in range(0, len(files), bs):
        ims = [models.load_image(f) for f in files[s:s + bs]]
        out.append(m.encode(ims, batch_size=bs, convert_to_numpy=True))
    return l2n(np.concatenate(out))


@torch.inference_mode()
def eg2_texts(texts, query=True):
    m = models.eg2()
    kw = {"prompt_name": "SearchQuery"} if query else {"prompt_name": "Document"}
    return l2n(m.encode(list(texts), batch_size=32, convert_to_numpy=True, **kw))


QVE_DOC = "Represent this photo of a cat for retrieval: describe the cat's action, pose, place, objects and anything it wears."
QVE_QRY = "Retrieve photos of a cat that match the user's description."


@torch.inference_mode()
def qve_images(files, bs=4):
    m = models.qve()
    out = []
    for s in range(0, len(files), bs):
        ims = [{"image": models.load_image(f)} for f in files[s:s + bs]]
        out.append(m.encode(ims, batch_size=bs, convert_to_numpy=True, prompt=QVE_DOC))
    return l2n(np.concatenate(out))


@torch.inference_mode()
def qve_texts(texts, prompt=QVE_QRY):
    m = models.qve()
    return l2n(m.encode([{"text": t} for t in texts], batch_size=16, convert_to_numpy=True, prompt=prompt))
