import glob
import json
import os
import sys
from PIL import Image, ImageDraw, ImageFont

real = sorted(glob.glob('data/real_photos/*.jpg'))
coco = json.load(open('data/coco_meta/chosen.json', encoding='utf-8'))
items = [{'id': os.path.splitext(os.path.basename(f))[0], 'file': f.replace(os.sep, '/')} for f in real]
items += [{'id': c['id'], 'file': c['file'].replace(os.sep, '/')} for c in coco]
json.dump(items, open('data/corpus.json', 'w', encoding='utf-8'), indent=1)

out = sys.argv[1] if len(sys.argv) > 1 else 'sheets'
os.makedirs(out, exist_ok=True)
C, R, W, H = 4, 3, 440, 400
font = ImageFont.truetype('arial.ttf', 26)
for s in range(0, len(items), C * R):
    sheet = Image.new('RGB', (C * W, R * H), 'white')
    d = ImageDraw.Draw(sheet)
    for k, it in enumerate(items[s:s + C * R]):
        im = Image.open(it['file']).convert('RGB')
        im.thumbnail((W - 8, H - 8))
        x, y = (k % C) * W, (k // C) * H
        sheet.paste(im, (x + 4, y + 4))
        d.rectangle([x + 4, y + 4, x + 64, y + 36], fill='yellow')
        d.text((x + 8, y + 6), str(s + k), fill='black', font=font)
    sheet.save(f'{out}/sheet_{s // (C * R):02d}.jpg', quality=88)
print(len(items))
