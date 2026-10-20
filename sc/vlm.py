import torch

from . import models

CAPTION_PROMPT = (
    "Describe this photo for a photo search index in 3 short sentences. "
    "Say what the cat is doing (pose and action; whether its eyes are closed, whether its mouth is open), "
    "where it is (place, furniture, indoor or outdoor), anything the cat is wearing, "
    "and which people, other animals or notable objects are next to it. "
    "Only describe what is visible."
)


def _chat(proc, image, text):
    content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]
    msgs = [{"role": "user", "content": content}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate(images, prompts, max_new_tokens=160):
    model, proc = models.qvl()
    texts = [_chat(proc, im, p) for im, p in zip(images, prompts)]
    ims = [im for im in images if im is not None] or None
    proc.tokenizer.padding_side = "left"
    batch = proc(text=texts, images=ims, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, batch["input_ids"].shape[1]:]
    return [s.strip() for s in proc.batch_decode(out, skip_special_tokens=True)]


def caption_files(files, bs=6):
    caps = []
    for s in range(0, len(files), bs):
        ims = [models.load_image(f, 768) for f in files[s:s + bs]]
        caps += generate(ims, [CAPTION_PROMPT] * len(ims))
    return caps


_yes_no = {}


def _yn_ids(tok):
    if "ids" not in _yes_no:
        y = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
        n = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]
        _yes_no["ids"] = (y, n)
    return _yes_no["ids"]


@torch.inference_mode()
def p_yes(images, statements, bs=8):
    model, proc = models.qvl()
    y, n = _yn_ids(proc.tokenizer)
    proc.tokenizer.padding_side = "left"
    out = []
    for s in range(0, len(images), bs):
        ims, sts = images[s:s + bs], statements[s:s + bs]
        texts = [_chat(proc, im, f"Does this photo show: \"{st}\"? Answer Yes or No.") for im, st in zip(ims, sts)]
        batch = proc(text=texts, images=ims, return_tensors="pt", padding=True).to(model.device)
        logits = model(**batch).logits[:, -1, :].float()
        ly = torch.logsumexp(logits[:, y], dim=-1)
        ln = torch.logsumexp(logits[:, n], dim=-1)
        out += torch.sigmoid(ly - ln).tolist()
    return out


TRANSLATE_PROMPT = "Translate this Korean phrase into natural English. Reply with the English phrase only.\nKorean: {q}"


def translate(phrases):
    return [o.strip().strip('"').rstrip(".") for o in generate([None] * len(phrases), [TRANSLATE_PROMPT.replace("{q}", p) for p in phrases], max_new_tokens=40)]
