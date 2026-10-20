import pandas as pd, random, json, os, urllib.request, concurrent.futures as cf
random.seed(1210)
cat = pd.read_pickle('data/coco_meta/cats.pkl')
cat['cap'] = cat.captions.apply(lambda c: ' '.join(c).lower())
groups = [
    ('yawn', r'yawn', 10), ('mouth', r'mouth open|open mouth|teeth|tongue|meow', 12),
    ('glasses', r'glasses|sunglasses|goggles', 14), ('hat', r'\bhat\b|helmet|crown', 10),
    ('costume', r'costume|sweater|shirt|dressed|outfit|jacket', 10), ('tie', r'\btie\b|bow tie|necktie', 6),
    ('groom', r'licking|groom|cleaning itself', 8), ('stretch', r'stretch', 6),
    ('grass', r'grass|lawn|yard|garden', 14), ('sofa', r'couch|sofa', 12), ('bed', r'\bbed\b|blanket', 12),
    ('laptop', r'laptop|keyboard|computer', 12), ('sink', r'sink|bathroom|toilet', 10),
    ('window', r'window|windowsill|sill', 12), ('car', r'\bcar\b|hood|vehicle', 10),
    ('box', r'\bbox\b|suitcase|luggage|\bbag\b|basket', 12), ('dog', r'\bdog\b|puppy', 10),
    ('person', r'person|\bman\b|woman|girl|boy|child|\blap\b', 12), ('eat', r'eating|drinking|food|bowl', 10),
    ('tv', r'television|\btv\b', 8), ('shoe', r'shoe|boot', 8), ('play', r'playing|toy|ball', 8),
    ('outside', r'outside|outdoor|street|sidewalk|fence|porch|roof', 12), ('bench', r'bench', 6),
    ('sleep', r'sleep|asleep|nap', 14),
]
chosen = []
seen = set()
for name, pat, n in groups:
    pool = cat[cat.cap.str.contains(pat) & ~cat.image_id.isin(seen)]
    pick = pool.sample(min(n, len(pool)), random_state=random.randint(0, 10**6))
    for _, r in pick.iterrows():
        seen.add(r.image_id); chosen.append(r)
rest = cat[~cat.image_id.isin(seen)].sample(30, random_state=7)
for _, r in rest.iterrows():
    chosen.append(r)
os.makedirs('data/coco_photos', exist_ok=True)
meta = []
def dl(r):
    fn = f"data/coco_photos/coco_{r.image_id}.jpg"
    if not os.path.exists(fn):
        urllib.request.urlretrieve(r.coco_url, fn)
    return fn
with cf.ThreadPoolExecutor(8) as ex:
    fns = list(ex.map(dl, chosen))
for r, fn in zip(chosen, fns):
    meta.append({'id': f'coco_{r.image_id}', 'file': fn, 'captions': list(r.captions)})
json.dump(meta, open('data/coco_meta/chosen.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(len(meta))
