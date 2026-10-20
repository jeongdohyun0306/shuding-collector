import gc
import torch
from PIL import Image, ImageOps

DTYPE = torch.bfloat16
EG2 = "google/embeddinggemma-2"
QVE = "Qwen/Qwen3-VL-Embedding-2B"
QVR = "Qwen/Qwen3-VL-Reranker-2B"
QVL = "Qwen/Qwen3-VL-2B-Instruct"

_cache = {}

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def load_image(path, max_side=896):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((max_side, max_side))
    return im


def free(name=None):
    names = [name] if name else list(_cache)
    for n in names:
        _cache.pop(n, None)
    gc.collect()
    torch.cuda.empty_cache()


def eg2():
    if "eg2" not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache["eg2"] = SentenceTransformer(
            EG2, device="cuda",
            model_kwargs={"torch_dtype": DTYPE},
            config_kwargs={"audio_config": None},
        )
    return _cache["eg2"]


def qve():
    if "qve" not in _cache:
        from sentence_transformers import SentenceTransformer
        _cache["qve"] = SentenceTransformer(
            QVE, device="cuda",
            model_kwargs={"torch_dtype": DTYPE},
            processor_kwargs={"min_pixels": 64 * 28 * 28, "max_pixels": 512 * 28 * 28},
        )
    return _cache["qve"]


def qvr():
    if "qvr" not in _cache:
        from sentence_transformers import CrossEncoder
        _cache["qvr"] = CrossEncoder(
            QVR, device="cuda",
            model_kwargs={"torch_dtype": DTYPE},
            processor_kwargs={"min_pixels": 64 * 28 * 28, "max_pixels": 512 * 28 * 28},
        )
    return _cache["qvr"]


def qvl():
    if "qvl" not in _cache:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained(QVL, min_pixels=64 * 28 * 28, max_pixels=768 * 28 * 28)
        model = AutoModelForImageTextToText.from_pretrained(QVL, dtype=DTYPE, device_map="cuda")
        model.eval()
        _cache["qvl"] = (model, proc)
    return _cache["qvl"]
