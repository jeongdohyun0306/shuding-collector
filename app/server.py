import argparse
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from sc import index as sindex
from sc import models
from sc.v1 import V1
from sc.v2 import V2, Cfg

app = FastAPI()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
STATE = {}
MAX_QUERY = 200


def photo_size(path):
    with Image.open(path) as im:
        w, h = im.size
        try:
            orient = im.getexif().get(0x0112, 1)
        except Exception:
            orient = 1
    return (h, w) if orient in (5, 6, 7, 8) else (w, h)


def thumb(i, size):
    c = STATE["corpus"]
    src = Path(c.files[i])
    p = c.path("thumbs") / f"{size}_{c.items[i]['id']}_{int(src.stat().st_mtime)}.jpg"
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        im = models.load_image(src, max_side=size)
        tmp = p.with_suffix(f".{os.getpid()}.tmp")
        im.save(tmp, format="JPEG", quality=85)
        os.replace(tmp, p)
    return p


@app.get("/")
def home():
    return HTMLResponse((ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/api/photos")
def photos():
    c = STATE["corpus"]
    out = [{"i": i, "id": it["id"], "w": w, "h": h} for i, (it, (w, h)) in enumerate(zip(c.items, STATE["sizes"]))]
    return {"photos": out, "name": STATE["name"]}


@app.get("/thumb/{i}")
def get_thumb(i: int, s: int = 480):
    if not 0 <= i < len(STATE["corpus"].items):
        raise HTTPException(404)
    return FileResponse(thumb(i, 480 if s <= 480 else 1400), media_type="image/jpeg")


def run(engine, q, k):
    t = time.perf_counter()
    s = engine.scores([q])[0]
    order = np.argsort(-s, kind="stable")[:k]
    ms = round((time.perf_counter() - t) * 1000)
    return [{"i": int(i), "id": STATE["corpus"].items[i]["id"]} for i in order], ms


@app.get("/api/search")
def search(q: str, mode: str = "v2", k: int = 12):
    q = q.strip()[:MAX_QUERY]
    if not q:
        return JSONResponse({"q": q, "error": "검색어를 입력하세요."}, status_code=400)
    if mode not in ("v1", "v2", "both"):
        return JSONResponse({"q": q, "error": "mode는 v1, v2, both 중 하나여야 합니다."}, status_code=400)
    k = max(1, min(int(k), 50))
    out = {"q": q}
    if mode in ("v2", "both"):
        res, ms = run(STATE["v2"], q, k)
        p = STATE["v2"].parse(q)
        out["v2"] = {"results": res, "ms": ms, "pos": p["pos"], "neg": p["neg"]}
    if mode in ("v1", "both"):
        res, ms = run(STATE["v1"], q, k)
        out["v1"] = {"results": res, "ms": ms}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None)
    ap.add_argument("--config", default=str(ROOT / "app" / "config.json"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.index:
        idx = Path(a.index).resolve()
        corpus_json = idx / "corpus.json"
        if not corpus_json.exists():
            raise SystemExit(f"{corpus_json} not found. Build it first: python -m sc.build --photos <folder> --out {idx}")
    else:
        idx = ROOT / "index"
        corpus_json = ROOT / "data" / "corpus.json"
    c = sindex.Corpus(corpus_json, idx)
    for name in ("eg2_img.npy", "captions.json", "eg2_cap.npy"):
        if not c.path(name).exists():
            raise SystemExit(f"{c.path(name)} is missing. Build the index first: python -m sc.build --photos <folder> --out {idx}")
    cfg_d = json.load(open(a.config, encoding="utf-8"))
    cfg = Cfg(**{f.name: cfg_d[f.name] for f in fields(Cfg) if f.name in cfg_d})
    STATE["corpus"] = c
    STATE["name"] = cfg.names[0] if cfg.names else "고양이"
    STATE["sizes"] = [photo_size(f) for f in c.files]
    STATE["v1"] = V1(c, names=cfg.names)
    STATE["v2"] = V2(c, cfg, persist=False, memo=False)
    STATE["v2"].scores(["안경을 안 쓴 고양이"])
    STATE["v1"].scores(["고양이"])
    print(f"ready: http://127.0.0.1:{a.port}  ({len(c.items)} photos)", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
