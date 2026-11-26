"""LayerBeauty loader: one item with its layer and part weights (used by the tests; a starting point for your own code).

The dataset is stored as Parquet on the Hub, with configs creative_scene and daily_scene; iter_hub() streams its rows.
load() reads one item kept as files, as in tests/data: <root>/<config>/<split>/<id>/{madeup.webp, bare.webp, layer.webp,
parts.npz, meta.json}, where meta.json holds the metadata columns of the row.
layer.webp is RGBA (RGB = F, A = alpha) and covers meta['layer_info']['box'] = [x0, y0, x1, y1].
Both return dict(I uint8 made-up, B uint8 bare, a (H, W), F (H, W, 3), w (5, H, W) part weights in the order lips, eyes,
brows, face, forehead (float [0, 1]), le (H, W) lips + eyes weight, face (H, W) bool or None, lm (478, 2) bare landmarks or
None, meta)."""
import io, json, os
import numpy as np, cv2
from PIL import Image

REPO = 'ming0531/LayerBeauty'
IMAGES = ('madeup', 'bare', 'layer')


def items(root, config, split):
    d = os.path.join(root, config, split)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def load(root, config, split, iid):
    d = os.path.join(root, config, split, iid)
    with open(f'{d}/meta.json') as f: meta = json.load(f)
    with np.load(f'{d}/parts.npz') as npz: z = {k: npz[k] for k in npz.files}
    return assemble(config, Image.open(f'{d}/madeup.webp'), Image.open(f'{d}/bare.webp'), Image.open(f'{d}/layer.webp'), z, meta)


def iter_hub(config, split, repo=REPO, revision=None):
    """Stream one split from the Hub (needs `pip install datasets`)."""
    from datasets import Image as ImageFeature, load_dataset
    ds = load_dataset(repo, config, split=split, streaming=True, revision=revision)
    for c in IMAGES: ds = ds.cast_column(c, ImageFeature(decode=False))
    for row in ds:
        with np.load(io.BytesIO(row['parts_npz'])) as npz: z = {k: npz[k] for k in npz.files}
        I, B, L = (Image.open(io.BytesIO(row[c]['bytes'])) for c in IMAGES)
        yield row['id'], assemble(config, I, B, L, z, {k: v for k, v in row.items() if k not in IMAGES + ('parts_npz',)})


def assemble(config, I, B, L, z, meta):
    I, B = np.asarray(I.convert('RGB')), np.asarray(B.convert('RGB'))
    L = np.asarray(L.convert('RGBA')).astype(np.float32) / 255
    H, W = B.shape[:2]
    a = np.zeros((H, W), np.float32); F = np.zeros((H, W, 3), np.float32)
    x0, y0, x1, y1 = (int(v) for v in meta['layer_info']['box'])
    a[y0:y1, x0:x1] = L[..., 3]; F[y0:y1, x0:x1] = L[..., :3]
    if config == 'creative_scene':
        w = z['w'].astype(np.float32) / 255
        return dict(I=I, B=B, a=a, F=F, w=w, le=np.clip(w[0] + w[1], 0, 1), face=z['face'].astype(bool), lm=None, meta=meta)
    # daily_scene: exact part labels inside the layer box (names: decor lips liner eyeshadow brows face)
    names = [str(x) for x in z['part_names']]
    pl = np.full((H, W), 255, np.uint8); pl[y0:y1, x0:x1] = z['part_labels']
    w = np.zeros((5, H, W), np.float32)
    w[0] = pl == names.index('lips'); w[1] = np.isin(pl, (names.index('liner'), names.index('eyeshadow'))); w[2] = pl == names.index('brows')
    w[:3] = np.stack([cv2.GaussianBlur(x, (0, 0), 1.0) for x in w[:3]])
    w[3] = np.clip(1 - w[:3].sum(0), 0, 1)
    le = np.zeros((H, W), np.float32); le[y0:y1, x0:x1] = np.isin(z['regions'], (1, 2))      # regions: 1 lips, 2 eyes
    return dict(I=I, B=B, a=a, F=F, w=w, le=le, face=None, lm=z['lm_bare'].astype(np.float32), meta=meta)
