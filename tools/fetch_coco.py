import concurrent.futures as cf
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fetch(item):
    dst = ROOT / item["file"]
    if dst.exists():
        return "have"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(item["coco_url"], tmp)
    os.replace(tmp, dst)
    return "got"


def main():
    items = json.load(open(ROOT / "data" / "coco_meta" / "chosen.json", encoding="utf-8"))
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(fetch, items))
    print(f"{len(items)} photos: {res.count('got')} downloaded, {res.count('have')} already present")


if __name__ == "__main__":
    main()
