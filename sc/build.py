import argparse
import json
from pathlib import Path

from PIL import Image

from . import index, models, vlm

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}


def readable(p):
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception as e:
        print(f"[skip] {p}: {type(e).__name__}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    photos = Path(a.photos).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in photos.rglob("*")
                   if p.suffix.lower() in EXTS and out not in p.parents and readable(p))
    if not files:
        raise SystemExit(f"no readable photos under {photos}")
    items = [{"id": f"p{k:04d}_{p.stem}", "file": str(p)} for k, p in enumerate(files)]
    json.dump(items, open(out / "corpus.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    c = index.Corpus(out / "corpus.json", out)
    print(f"{len(items)} photos")
    c.cached("eg2_img.npy", lambda: index.eg2_images(c.files))
    models.free("eg2")
    caps = c.cached("captions.json", lambda: vlm.caption_files(c.files))
    models.free("qvl")
    c.cached("eg2_cap.npy", lambda: index.eg2_texts(caps, query=False))
    print("done", out)


if __name__ == "__main__":
    main()
