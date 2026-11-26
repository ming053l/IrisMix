"""Shared helpers of the inference path: session caches (LRU, read-only arrays, content keys), uint8 conversion, the
1.9x face crop, Lab, and numpy <-> torch conversion of images in [-1, 1].
numpy / OpenCV only at import time; torch is imported by the tensor helpers when they are called."""
import hashlib, os, threading
from collections import OrderedDict
import numpy as np, cv2

R = 512                       # decomposer / renderer face-crop resolution

# ------------------------------------------------------------------------------------------------ caches
# Every per-image analysis (parsing, landmarks, rule contexts, warp meshes) is a pure function of the
# image bytes, so it is computed once per session and returned READ-ONLY: a write into a cached array raises instead of
# silently corrupting later calls. Keys: shape + dtype + blake2b of the bytes.
LOCK = threading.RLock()      # the caches may be shared by worker threads


class LRU(OrderedDict):
    def __init__(self, n): super().__init__(); self.n = n

    def get(self, k, d=None):
        with LOCK:
            if k in self: self.move_to_end(k); return self[k]
            return d

    def put(self, k, v):
        with LOCK:
            self[k] = v; self.move_to_end(k)
            while len(self) > self.n: self.popitem(last=False)
            return v

    def __contains__(self, k):
        with LOCK: return super().__contains__(k)


def read_only(x):
    """mark numpy arrays (also inside tuples / lists / dicts) read-only; returns x."""
    if isinstance(x, np.ndarray): x.flags.writeable = False
    elif isinstance(x, (tuple, list)): [read_only(v) for v in x]
    elif isinstance(x, dict): [read_only(v) for v in x.values()]
    return x


def image_key(img):
    a = np.ascontiguousarray(img)
    return (a.shape, a.dtype.str, hashlib.blake2b(a.data, digest_size=16).digest())


NCACHE = int(os.environ.get('IRISMIX_CACHE_ITEMS', '48'))     # cached images per analysis


# ------------------------------------------------------------------------------------------------ images
def u8(x):
    """float [0, 1] -> uint8, round half up."""
    return np.floor(np.clip(x, 0, 1) * 255 + .5).astype(np.uint8)


def face_box(p, shape, k=1.9):
    """square face crop (X0, Y0, side): k x the landmark extent, centred, clamped to the image (decomposer, renderer and the
    composite data use the same crop). p: landmarks (N, >= 2) in px."""
    H, W = shape[:2]
    x0, y0 = p[:, :2].min(0); x1, y1 = p[:, :2].max(0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = int(min(k * max(x1 - x0, y1 - y0), H, W))
    X0 = int(np.clip(cx - side / 2, 0, W - side)); Y0 = int(np.clip(cy - side / 2, 0, H - side))
    return X0, Y0, side


def crop_rs(x, box, n=R):
    """crop box (X0, Y0, side) of a full-frame map and resize it to n x n (float32; area when shrinking, else cubic)."""
    X0, Y0, side = box
    c = x[Y0:Y0 + side, X0:X0 + side]
    return cv2.resize(c.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA if side > n else cv2.INTER_CUBIC)


# ------------------------------------------------------------------------------------------------ colour
def lab(x):
    """sRGB float [0, 1] (H, W, 3) -> CIELAB (OpenCV float convention: L 0-100, a / b ~ -127-127)."""
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_RGB2LAB)


def rgb(L):
    """CIELAB (H, W, 3) -> sRGB float [0, 1] (clipped)."""
    return np.clip(cv2.cvtColor(L.astype(np.float32), cv2.COLOR_LAB2RGB), 0, 1)


# ------------------------------------------------------------------------------------------------ numpy <-> torch
def to_device(arrs, dev):
    """{name: (H, W[, 3]) float [0, 1] numpy} -> {name: (1, 3, H, W) tensor in [-1, 1] on dev} (2-D maps are
    repeated to 3 channels): the raw array is copied once (pinned, asynchronous on CUDA) and mapped to [-1, 1] on the
    device (x * 2 - 1 is exact in IEEE float32 on either device)."""
    import torch
    pin = str(dev).startswith('cuda')
    out = {}
    for k, x in arrs.items():
        t = torch.from_numpy(np.ascontiguousarray(x))
        if pin: t = t.pin_memory()
        t = t.to(dev, non_blocking=pin)
        if t.ndim == 2: t = t[..., None].expand(-1, -1, 3)
        out[k] = (t.permute(2, 0, 1) * 2 - 1)[None]
    return out


def from_t(h, i=0):
    """(B, 3, H, W) tensor in [-1, 1] -> (H, W, 3) float32 numpy in [0, 1] of item i; the affine map and the clip run on the
    tensor's device and only the result is copied (same values as ((h[i].permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0, 1)
    for a float32 h)."""
    return ((h[i].permute(1, 2, 0) + 1) / 2).clamp(0, 1).cpu().numpy()
