"""Transfer rules: face parsing, MediaPipe landmarks, piecewise-affine landmark warp of a reference layer onto a target
bare face, reference-side cleaning, lips / brows rules, hair occlusion on the target, exact alpha-over composite.

A reference layer (a, P = aF, w) is warped from the reference bare face onto the target with a piecewise-affine warp over a
Delaunay triangulation of 478 MediaPipe landmarks (+ two forehead rings), then I = a'F' + (1 - a') B exactly.

Every per-image analysis is cached per session and returned read-only (parsing, landmarks, the target / reference rule
contexts, warp meshes). Module-level knobs:
  SEG_DEVICE    face-parser device, None = auto (env IRISMIX_SEG_DEVICE forces it); CUDA probabilities match the CPU
                ones to ~2e-6
  SEG_THREADS   torch CPU threads when the parser runs on the CPU (None = min(16, #cores))
  SEG_BATCH     images per parser forward in parse_many
  LMK_FALLBACK  retry landmark detection with confidence 0.1 when the default (0.5) finds no face (in-the-wild photos)"""
import os, threading
import numpy as np, cv2
from PIL import Image
from scipy.spatial import Delaunay

from irismix import config
from irismix.utils import LRU, LOCK, read_only, image_key, NCACHE

SEG = None
SEG_DEVICE = None
SEG_THREADS = None
SEG_BATCH = 8
LMK_FALLBACK = False
PART_NAMES = ['lips', 'eyes', 'brows', 'face', 'forehead']   # order of the 5 part weights
ALLOWED = (1, 2, 6, 7, 11, 12)                 # jonathandinu/face-parsing: skin, nose, brows, upper / lower lip
EYES = (4, 5)
HAIR = [13, 14]                                # hair, hat: occluders above the makeup
BROWS = (6, 7)
GLASSES = 3
LIPS = (11, 12)
LIP_PTS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
NOSE_BASE = [64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48]   # nose tip / alae / base
ORN_A = 0.85                                   # opaque ornament threshold inside the brow region
LOW_A, LOW_RAMP = 0.08, 0.12                    # alpha below ~0.08 is removal residue, ramps in to 0.20
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
DESIGN_A0, DESIGN_RAMP = 0.35, 0.15
MIN_DESIGN = 0.0012                             # visible makeup must cover >= 0.12 % of the image
WASH_AB0, WASH_RAMP = 5.0, 5.0                  # chroma shift (Lab a*b*, smoothed) that marks a coloured wash
# MediaPipe face-oval landmark indices (clockwise from the forehead top) and its upper half (above the eye line on a frontal
# face: forehead and temples)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
             150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
UPPER_OVAL = [10, 338, 297, 332, 284, 251, 389, 162, 21, 54, 103, 67, 109]

_MODEL_LOCK = threading.RLock()   # SegFormer and MediaPipe calls are serialised
_DETECTORS = {}                   # low confidence? -> MediaPipe FaceLandmarker
_PARSE = LRU(NCACHE)          # image key -> C x H x W probabilities
_LMK = LRU(4 * NCACHE)        # image key -> (478 x 3) normalised MediaPipe landmarks or None
DERIVED = LRU(8 * NCACHE)        # derived per-image quantities (argmax, target_mask, rule contexts), keyed by image key


# ------------------------------------------------------------------------------------------------ face parsing
def seg_device():
    """device of the face parser: env IRISMIX_SEG_DEVICE (cpu | cuda) > SEG_DEVICE > auto (cuda when available)."""
    import torch
    return os.environ.get('IRISMIX_SEG_DEVICE') or SEG_DEVICE or ('cuda' if torch.cuda.is_available() else 'cpu')


def _face_parser():
    global SEG
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    if SEG is None:
        dev = seg_device()
        if dev == 'cpu': torch.set_num_threads(SEG_THREADS or int(os.environ.get('IRISMIX_THREADS') or 0) or min(16, os.cpu_count() or 1))
        kw = dict(revision=config.PARSER_REVISION)
        SEG = (SegformerImageProcessor.from_pretrained(config.PARSER_ID, **kw),
               SegformerForSemanticSegmentation.from_pretrained(config.PARSER_ID, **kw).eval().to(dev))
    return SEG


def parse_many(imgs):
    """face parsing (SegFormer) of several images -> list of read-only C x H x W
    probability maps, cached per image. Uncached images go through SegFormer in batches of SEG_BATCH (the processor
    resizes every image to 512 x 512, so mixed sizes batch). fp32 (the probabilities are used continuously: support /
    hair sums); TF32 off on CUDA so the probabilities match the CPU path to float rounding."""
    import torch
    keys = [image_key(x) for x in imgs]
    with _MODEL_LOCK:
        todo = []
        for k, x in zip(keys, imgs):
            if k not in _PARSE and k not in [t[0] for t in todo]: todo.append((k, x))
        if todo:
            proc, net = _face_parser(); dev = next(net.parameters()).device
            for b0 in range(0, len(todo), SEG_BATCH):
                chunk = todo[b0:b0 + SEG_BATCH]
                with torch.no_grad(), torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=False, allow_tf32=False):
                    x = proc(images=[Image.fromarray(np.ascontiguousarray(im)) for _, im in chunk], return_tensors='pt')['pixel_values'].to(dev)
                    lg = net(pixel_values=x).logits
                    for j, (k, im) in enumerate(chunk):
                        pr = torch.nn.functional.interpolate(lg[j:j + 1], size=im.shape[:2], mode='bilinear').softmax(1)[0]
                        _PARSE.put(k, read_only(pr.cpu().numpy()))
        return [_PARSE.get(k) for k in keys]


def parse(img):
    """face parsing of one image (cached, read-only C x H x W probabilities); see parse_many."""
    return parse_many([img])[0]


def memoise_on_map(tag, prob, fn):
    """memoise fn(prob) for a cached (read-only) probability map, keyed by its identity (the cache keeps prob alive)."""
    if prob.flags.writeable: return fn()
    k = (tag, id(prob)); hit = DERIVED.get(k)
    if hit is not None and hit[0] is prob: return hit[1]
    return DERIVED.put(k, (prob, read_only(fn())))[1]


def label_map(prob):
    """prob.argmax(0) for a C x H x W map, identical result (first maximum on ties) but ~10x faster: numpy's argmax over
    the leading axis makes a transposed copy of the whole map (~80 MB at 1024^2), this scans it channel by channel."""
    return memoise_on_map('argmax0', prob, lambda: _label_map(prob))


def _label_map(prob):
    best = prob[0].copy(); idx = np.zeros(prob.shape[1:], np.intp)
    for c in range(1, prob.shape[0]):
        m = prob[c] > best
        np.copyto(best, prob[c], where=m); idx[m] = c
    return idx


def target_mask(prob):
    """(support, hair) on the target, both soft.
    support: where makeup can exist on this face -- skin, nose, brows, lips, plus the eye rim (eye label minus an eroded
             core, so liner and lash lines survive but the eyeball stays clean); not teeth, mouth interior or background.
             Hair is NOT removed here: the layer is amodal (the full design, also under the hair).
    hair:    hair / hat probability = the occluder. The photo is I = B + (1 - hair) * a (F - B): makeup lies under the hair."""
    return memoise_on_map('target_mask', prob, lambda: _target_mask(prob))


def _target_mask(prob):
    lab = label_map(prob)
    ok = prob[list(ALLOWED) + HAIR].sum(0)                         # hair pixels may carry (hidden) makeup on the skin below
    eye = np.isin(lab, EYES).astype(np.uint8)
    k = max(3, int(round(lab.shape[0] / 256)))
    rim = (eye - cv2.erode(eye, np.ones((2 * k + 1, 2 * k + 1), np.uint8))).astype(np.float32)
    support = np.clip(cv2.GaussianBlur(np.clip(ok + rim, 0, 1).astype(np.float32), (0, 0), 1.0), 0, 1)
    # brows and lashes are hair-like and get a high 'hair' probability; they are skin-attached features, not occluders
    feat = cv2.dilate(np.clip(prob[list(BROWS) + list(EYES)].sum(0), 0, 1).astype(np.float32), np.ones((7, 7), np.float32))
    return support, np.clip(prob[HAIR].sum(0), 0, 1) * (1 - feat)


def nostrils(img, pt):
    """soft mask of the nostril openings: inside the nose-base hull, pixels much darker than the nose skin."""
    nb = np.zeros(img.shape[:2], np.uint8); cv2.fillConvexPoly(nb, cv2.convexHull(np.round(pt[NOSE_BASE]).astype(np.int32)), 1)
    if nb.sum() < 50: return np.zeros(img.shape[:2], np.float32)
    L = img.astype(np.float32).mean(-1) / 255
    nskin = float(np.percentile(L[nb > 0], 70))
    nos = ((L < 0.62 * nskin) & (nb > 0)).astype(np.uint8)
    nos = cv2.morphologyEx(nos, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(3, int(round(img.shape[0] / 256)))
    return cv2.GaussianBlur(cv2.dilate(nos, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(np.float32), (0, 0), k / 2)


def brow_region(pt, shape):
    """soft brow region from the 10 MediaPipe points of each brow (parsing splits brows between 'brow' and 'hair')."""
    m = np.zeros(shape[:2], np.uint8)
    for idx in (BROW_L, BROW_R):
        cv2.fillConvexPoly(m, cv2.convexHull(np.round(pt[idx]).astype(np.int32)), 1)
    k = max(3, int(round(shape[0] / 170)))
    return cv2.GaussianBlur(cv2.dilate(m, np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(np.float32), (0, 0), k / 2)


# ------------------------------------------------------------------------------------------------ landmarks
def _detector(low=False):
    """MediaPipe FaceLandmarker (default confidence 0.5; low: detection / presence confidence 0.1), created once."""
    import mediapipe as mp
    if low not in _DETECTORS:
        kw = dict(min_face_detection_confidence=0.1, min_face_presence_confidence=0.1) if low else {}
        _DETECTORS[low] = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=config.landmarker_path()), num_faces=1, **kw))
    return _DETECTORS[low], mp


def clear_caches():
    """drop the per-image analysis caches (parsing maps, landmarks, derived contexts, warp meshes) and the face parser."""
    global SEG
    with _MODEL_LOCK:
        for c in (_PARSE, _LMK, DERIVED, piecewise_warp.cache, piecewise_warp.mesh):
            with LOCK: c.clear()
        SEG = None


def close_models():
    """close the MediaPipe landmarkers (call before interpreter exit to avoid a teardown warning of mediapipe)."""
    with _MODEL_LOCK:
        for d in _DETECTORS.values(): d.close()
        _DETECTORS.clear()


def detect_landmarks(img, low=False):
    """MediaPipe FaceLandmarker on img, cached per image: (478, 3) float64 normalised (x, y, z) or None (read-only)."""
    with _MODEL_LOCK:
        k = image_key(img) + (('low',) if low else ())
        if k in _LMK: return _LMK.get(k)
        det, mp = _detector(low)
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
        v = None if not r.face_landmarks else read_only(np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]], np.float64))
        return _LMK.put(k, v)


def _landmarks_normalised(img, fallback=None):
    """detect_landmarks with the low-confidence retry when the default finds no face (default LMK_FALLBACK; faces found by
    the default are unchanged)."""
    q = detect_landmarks(img)
    if q is None and (LMK_FALLBACK if fallback is None else fallback): q = detect_landmarks(img, low=True)
    return q


def landmarks(img, fallback=None):
    """478 x 2 landmarks in px (float32, a fresh array) or None; see _landmarks_normalised for fallback."""
    q = _landmarks_normalised(img, fallback)
    if q is None: return None
    h, w = img.shape[:2]
    return np.stack([q[:, 0] * w, q[:, 1] * h], 1).astype(np.float32)      # same float64 products as p.x * w


def landmarks_3d(img, fallback=None):
    """478 x 3 landmarks (x, y px; z in px scale, float32, a fresh array) or None; same detection as landmarks."""
    q = _landmarks_normalised(img, fallback)
    if q is None: return None
    h, w = img.shape[:2]
    return np.stack([q[:, 0] * w, q[:, 1] * h, q[:, 2] * w], 1).astype(np.float32)


def require_landmarks_3d(img):
    """landmarks_3d for a photo that must show a face: raises ValueError (a message for the user) instead of returning None."""
    p = landmarks_3d(img)
    if p is None: raise ValueError('no face found in the photo')
    return p


def extend_landmarks(p, shape):
    """478 landmarks + two outer rings derived from the upper face oval, so the mesh also covers the forehead up to
    the hairline: each oval point above the eyes is pushed away from the face centre (x1.35, x1.75). The same rule on
    source and target keeps the rings in correspondence."""
    c = p[FACE_OVAL].mean(0)
    up = p[UPPER_OVAL]                          # FIXED index set (same on every face): source and target must get the
    rings = []                                  # same number of ring points, in correspondence
    for s_up in (1.35, 1.75):                   # extend only above the eyes (forehead to hairline); below the eyes the
        rings.append(c + s_up * (up - c))       # 478 points reach the jaw, and rings there pull face-edge residue onto the neck
    q = np.concatenate([p] + rings, 0)
    q[:, 0] = np.clip(q[:, 0], 0, shape[1] - 1); q[:, 1] = np.clip(q[:, 1], 0, shape[0] - 1)
    return q.astype(np.float32)


# ------------------------------------------------------------------------------------------------ warp
def piecewise_warp(src, ps, pd, shape):
    """piecewise-affine warp over the Delaunay triangulation of the destination landmarks, as ONE remap: a triangle-index
    map of the destination and the per-triangle affine dst -> src give every pixel its source coordinate (same result as
    warping each triangle and pasting it). Pixels outside the mesh get 0.
    The maps are evaluated only inside the bounding box of the destination mesh (outside it every pixel is off-mesh:
    map = -1e4 -> 0), the destination triangulation is cached per (pd, frame), the maps per (ps, pd, frame), and the last
    map is kept per thread (stretch_mask)."""
    H, W = shape[:2]
    src = np.asarray(src, np.float32)
    key = (np.ascontiguousarray(ps, np.float32).tobytes(), np.ascontiguousarray(pd, np.float32).tobytes(), H, W)
    hit = piecewise_warp.cache.get(key)
    if hit is None:
        mk = (np.ascontiguousarray(pd).tobytes(), np.asarray(pd).dtype.str, H, W)
        mesh = piecewise_warp.mesh.get(mk)
        if mesh is None:                        # destination mesh: Delaunay + per-pixel triangle index inside the mesh
            dt = Delaunay(pd)                   # bounding box, shared by every reference warped onto the same target
            tris = dt.simplices
            x0 = int(np.clip(np.floor(pd[:, 0].min()), 0, W)); x1 = int(np.clip(np.ceil(pd[:, 0].max()) + 1, 0, W))
            y0 = int(np.clip(np.floor(pd[:, 1].min()), 0, H)); y1 = int(np.clip(np.ceil(pd[:, 1].max()) + 1, 0, H))
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            # exact point-in-triangle (no rasterisation rounding: thin triangles can neither overlap nor leave gaps that
            # would sample far-away source pixels as thin streaks)
            idx = dt.find_simplex(np.stack([xx.ravel(), yy.ravel()], 1)).reshape(yy.shape).astype(np.int32)
            u_, v_ = pd[tris[:, 1]] - pd[tris[:, 0]], pd[tris[:, 2]] - pd[tris[:, 0]]
            area = np.abs(u_[:, 0] * v_[:, 1] - u_[:, 1] * v_[:, 0])         # 2-D cross product
            idx[(idx >= 0) & (area[np.clip(idx, 0, None)] <= 1e-3)] = -1
            ii = np.where(idx < 0, len(tris), idx)
            inside = np.zeros((H, W), bool); inside[y0:y1, x0:x1] = idx >= 0
            mesh = piecewise_warp.mesh.put(mk, read_only((tris, (y0, y1, x0, x1), ii, xx, yy, inside)))
        tris, (y0, y1, x0, x1), ii, xx, yy, inside = mesh
        A = np.stack([cv2.getAffineTransform(pd[tri].astype(np.float32), ps[tri].astype(np.float32)) for tri in tris]
                     + [np.array([[0, 0, -1e4], [0, 0, -1e4]], np.float32)])
        # per-coefficient gathers instead of the (H, W, 2, 3) float64 gather A[ii]: same float64 products and sums in
        # the same order -> bit-identical maps
        g = lambda r, c: A[:, r, c][ii]
        mx = np.full((H, W), -1e4, np.float32); my = np.full((H, W), -1e4, np.float32)
        mx[y0:y1, x0:x1] = (g(0, 0) * xx + g(0, 1) * yy + g(0, 2)).astype(np.float32)
        my[y0:y1, x0:x1] = (g(1, 0) * xx + g(1, 1) * yy + g(1, 2)).astype(np.float32)
        hit = piecewise_warp.cache.put(key, read_only((mx, my, inside, (y0, y1, x0, x1))))
    mx, my, inside, (y0, y1, x0, x1) = hit
    out = np.zeros((H, W) + src.shape[2:], np.float32)
    if y1 > y0 and x1 > x0:
        o = cv2.remap(src, mx[y0:y1, x0:x1], my[y0:y1, x0:x1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[y0:y1, x0:x1] = o.reshape(o.shape[:2] + src.shape[2:])
    piecewise_warp.tls.last_map = (mx, my, inside)
    return out if out.ndim == src.ndim else out[..., None]


piecewise_warp.cache = LRU(16)                           # (ps, pd, H, W) -> maps (main + lips + brow mesh of several references)
piecewise_warp.mesh = LRU(8)                             # (pd, H, W) -> destination triangulation and triangle-index map
piecewise_warp.tls = threading.local()


def stretch_mask(lo=0.2, hi=5.0):
    """soft mask of target pixels whose local area ratio (source px per target px, from the last warp map) is in
    [lo, hi] and not folded: strongly compressed / stretched / flipped triangles at a turned cheek or the jaw would
    squeeze reference residue into fragments."""
    mx, my, inside = piecewise_warp.tls.last_map
    J = np.gradient(mx, axis=1) * np.gradient(my, axis=0) - np.gradient(mx, axis=0) * np.gradient(my, axis=1)
    J = cv2.medianBlur(J.astype(np.float32), 5)
    med = np.median(J[inside]) if inside.any() else 1.0
    r = J / max(med, 1e-6)
    ok = ((r > lo) & (r < hi) & inside).astype(np.float32)
    return cv2.GaussianBlur(cv2.erode(ok, np.ones((5, 5), np.float32)), (0, 0), 2.0)


# ------------------------------------------------------------------------------------------------ transfer
def _ref_bare_terms(Br):
    return dict(Lb=Br.astype(np.float32).mean(-1) / 255, labB=cv2.cvtColor((Br / 255).astype(np.float32), cv2.COLOR_RGB2LAB))


def reference_context(Br, Im, pr):
    """reference-side quantities of transfer_layer that do not depend on the layer (cached per (Br, Im)): parsing support /
    hair / mouth / nostril factors, the chroma-wash evidence of madeup vs bare, the bare's lightness and Lab."""
    k = ('ref', image_key(Br), image_key(Im)); hit = DERIVED.get(k)
    if hit is not None: return hit
    pref = parse(Br); ref_support, ref_hair = target_mask(pref)
    er = max(3, int(round(Br.shape[0] / 128)))                      # drop the reference's face-edge residue (hair /
    ref_support = cv2.erode(ref_support, np.ones((2 * er + 1, 2 * er + 1), np.float32))   # shadow along the contour)
    mref = np.zeros(Br.shape[:2], np.uint8); cv2.fillPoly(mref, [np.round(pr[INNER_LIPS]).astype(np.int32)], 1)
    sig = max(2.0, Br.shape[0] / 256)
    # coloured-makeup evidence (see design_only in transfer_layer): chroma shift between the reference's madeup and bare,
    # smoothed
    Im_ = cv2.GaussianBlur(np.asarray(Im).astype(np.float32) / 255, (0, 0), 1.5 * sig)
    Bm_ = cv2.GaussianBlur(Br.astype(np.float32) / 255, (0, 0), 1.5 * sig)
    dab_ = np.sqrt(((cv2.cvtColor(Im_, cv2.COLOR_RGB2LAB)[..., 1:] - cv2.cvtColor(Bm_, cv2.COLOR_RGB2LAB)[..., 1:]) ** 2).sum(-1))
    wash_ = np.clip((dab_ - WASH_AB0) / WASH_RAMP, 0, 1)
    wash_ = cv2.GaussianBlur(cv2.dilate(wash_, np.ones((5, 5), np.uint8)), (0, 0), sig)
    c = dict(ref_support=ref_support, one_m_hair=1 - ref_hair, one_m_mouth=1 - cv2.GaussianBlur(mref.astype(np.float32), (0, 0), 1.0),
             one_m_nos=1 - nostrils(Br, pr), wash_=wash_, **_ref_bare_terms(Br))
    return DERIVED.put(k, read_only(c))


def target_context(Bt, pt):
    """target-side quantities of transfer_layer (cached per target): final support and hair occluder (brow rule, mouth
    interior, nostrils), brow region, glasses fraction, lip mask, lip / brow shading terms, the float target."""
    k = ('tgt', image_key(Bt)); hit = DERIVED.get(k)
    if hit is not None: return hit
    sh = Bt.shape
    prob = parse(Bt); support, hair = target_mask(prob)
    breg = brow_region(pt, sh)
    # brows are often labelled 'hair'. Inside the brow region, hair occludes only if it belongs to a hair component that
    # also extends outside the brow region (bangs / fringe falling over the brow); isolated 'hair' blobs are the brows.
    hb = np.isin(label_map(prob), HAIR).astype(np.uint8)
    nlab, lab = cv2.connectedComponents(hb)
    outside = (breg < 0.05)
    keep = np.zeros(nlab, bool); keep[np.unique(lab[(hb > 0) & outside])] = True; keep[0] = False
    real = keep[lab].astype(np.float32)
    hair = hair * (1 - breg) + hair * breg * real
    support = np.maximum(support, breg)
    glasses = float((label_map(prob) == GLASSES).mean())
    tl = np.clip(prob[list(LIPS)].sum(0), 0, 1)
    tl = cv2.GaussianBlur(cv2.dilate(tl, np.ones((3, 3), np.float32)), (0, 0), 1.0)
    # lip shading of the target (creases, highlights): modulates the recoloured lip pigment mildly, as for the brows
    Lt = Bt.astype(np.float32).mean(-1) / 255; inl = tl > 0.5
    mLt = float(Lt[inl].mean()) if inl.any() else 0.5
    shade = 1 + 0.75 * (np.clip(Lt / max(mLt, 1e-3), 0.3, 2.0) - 1)
    # mouth interior (inner-lip landmarks: teeth and tongue) and nostril openings never carry makeup
    mouth = np.zeros(sh[:2], np.uint8); cv2.fillPoly(mouth, [np.round(pt[INNER_LIPS]).astype(np.int32)], 1)
    support = support * (1 - cv2.GaussianBlur(mouth.astype(np.float32), (0, 0), 1.0))
    support = support * (1 - nostrils(Bt, pt))
    # brow tint terms: coverage follows each pixel's hair darkness against the surrounding skin (strands strongly, skin
    # between strands weakly); colour keeps the target's strand shading
    Bf = Bt.astype(np.float32) / 255; L = Bf.mean(-1)
    reg = breg
    ring = (cv2.dilate((reg > 0.3).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (reg < 0.1) & (prob[1] > 0.5)
    skinL = float(np.median(L[ring])) if ring.sum() > 20 else float(np.percentile(L[reg > 0.3], 90)) if (reg > 0.3).any() else 0.5
    strand = np.clip((skinL - L) / max(0.45 * skinL, 1e-3), 0, 1)        # brow-hair darkness (NOT the occluder)
    mL = float(L[reg > 0.3].mean()) if (reg > 0.3).any() else 0.5
    c = dict(support=support, hair=hair, breg=breg, glasses=glasses, tl=tl, shade=shade, strand_w=(0.25 + 0.75 * strand),
             brow_fac=(1 + 0.3 * (np.clip(L / max(mL, 1e-3), 0.3, 2.0) - 1))[..., None], B=Bf)
    return DERIVED.put(k, read_only(c))


def transfer_layer(layer, Br, Im, Bt, parts=None, check_glasses=True, design_only=True, min_design=True, lip_mode='warp'):
    """transfer a reference layer onto a target bare face.
    layer: (a HxW, F HxWx3, w 5xHxW part weights in the order lips, eyes, brows, face, forehead), all float [0, 1] on the
    reference frame, or a callable returning that tuple; Br: reference bare (uint8 HxWx3, same frame); Im: reference
    made-up photo (uint8, same frame) or a callable returning it; Bt: target bare (uint8). parts: None = whole layer, else
    a subset of {'lips', 'eyes', 'brows', 'face', 'forehead'} (cut by the part weights). lip_mode: 'warp' (warp the
    reference lip texture into the target lips) or 'recolour' (recolour the target's own lips with the reference pigment).
    design_only: transfer only the opaque design and coloured washes of the layer. min_design=False disables the
    'too little opaque design' rejection.
    -> dict(a, F, av, hair, I, le, pt) full frame on Bt (amodal a / F, visible av, composite I) or dict(error=...)."""
    pr, pt = landmarks(Br), landmarks(Bt)
    if pr is None or pt is None: return dict(error='no landmarks')
    pt_face = pt.copy()                                                   # 478 face points only (for crops)
    pr, pt = extend_landmarks(pr, Br.shape), extend_landmarks(pt, Bt.shape)
    a, F, w = layer() if callable(layer) else layer
    a, F, w = (np.asarray(x, np.float32).copy() for x in (a, F, w))
    if parts is not None:                                                 # keep the requested parts only
        a = a * np.clip(sum(w[PART_NAMES.index(k)] for k in parts), 0, 1)
    # reference-side cleaning: layer content on the reference's hair, background, eyeballs or mouth interior is never
    # makeup (closed-form residue of the removal), and a low-coverage skin residue everywhere would imprint the
    # reference's features (ghost eyes / nostrils) on the target -> drop both.
    rc = reference_context(Br, Im() if callable(Im) else Im, pr)
    a = a * rc['ref_support'] * rc['one_m_hair'] * rc['one_m_mouth']
    a = a * rc['one_m_nos']                                              # the reference's nostril shadows are not makeup
    # skin-texture residue: klein removal also smooths the skin (freckles, pores, stubble), so the closed form turns that
    # texture into a grainy, semi-transparent layer (its colour is amplified by F = B + d / a, so colour cannot be used).
    # In the semi-transparent part (a < ~0.4) keep only the LOW-frequency alpha: smooth washes (blush, contour, shadow)
    # stay, high-frequency grain goes; opaque design (paint, liner, gems, glitter: a > ~0.4) is untouched.
    sig = max(2.0, Br.shape[0] / 256)
    alow = cv2.GaussianBlur(a, (0, 0), 2 * sig)
    wash_ = rc['wash_']
    opq = np.clip((a - 0.3) / 0.15, 0, 1)
    a = a * opq + np.minimum(a, alow) * (1 - opq)
    # sheen residue: the reference's own specular highlights (cheekbone / nose glow) differ between madeup and klein's
    # bare, so the closed form stores them as a semi-transparent DARK patch with grain -> pasted onto another face it
    # looks like dirt. Semi-transparent pixels whose layer colour is darker than the reference bare are dropped.
    Lb = rc['Lb']; Lfc = F.mean(-1)
    darker = np.clip((Lb - Lfc - 0.04) / 0.08, 0, 1)
    a = a * (1 - darker * (1 - opq) * (1 - wash_))                       # eyeshadow is darker than skin: keep washes
    # lighter sheen: the reference's specular glow is also missing from klein's bare, so it becomes a light, low-chroma,
    # semi-transparent layer; drop it (coloured highlighter with clear chroma is kept)
    labF = cv2.cvtColor(F.astype(np.float32), cv2.COLOR_RGB2LAB); labB = rc['labB']
    dch = np.sqrt(((labF[..., 1:] - labB[..., 1:]) ** 2).sum(-1))
    lighter = np.clip((Lfc - Lb - 0.03) / 0.06, 0, 1) * np.clip(1 - (dch - 8) / 8, 0, 1)
    a = a * (1 - lighter * (1 - opq) * (1 - wash_))
    if design_only:
        # only the opaque DESIGN of a Creative layer is transferred (paint, liner, lipstick, gems, glitter): its
        # semi-transparent washes (contour, blush, highlight) are indistinguishable from klein-removal residue
        # (skin texture, sheen, freckles) and are left out; daily washes come from LayerBeauty-Syn's exact layers
        dmask = np.clip((cv2.GaussianBlur(a, (0, 0), sig) - DESIGN_A0) / DESIGN_RAMP, 0, 1)
        # coloured washes (eyeshadow, blush, light paint): the reference's madeup vs bare differ in CHROMA (Lab a*b*,
        # smoothed); removal residue (grain, sheen, freckles) changes mostly lightness and fine texture -> keep the
        # low-frequency alpha where the chroma shift is clear
        wash = wash_
        keepm = np.maximum(dmask, wash)
        # close holes inside design / wash regions (pale centre of a paint block)
        hk = (keepm > 0.5).astype(np.uint8); ff = hk.copy(); hh, ww = ff.shape
        msk = np.zeros((hh + 2, ww + 2), np.uint8); cv2.floodFill(ff, msk, (0, 0), 1)
        holes = (1 - ff).astype(np.float32)
        keepm = np.maximum(keepm, holes)
        keepm = cv2.GaussianBlur(cv2.dilate(keepm, np.ones((3, 3), np.uint8)), (0, 0), 0.75 * sig)
        alow = cv2.GaussianBlur(a, (0, 0), sig)
        a = np.maximum(a * dmask, np.minimum(a, alow) * keepm) * (1 - w[0]) + a * w[0]   # lips: own rule
    # keep only COHERENT makeup (hysteresis on the smoothed layer): a region survives only if it is connected to a strong
    # seed (smoothed alpha >= 0.30) through pixels with smoothed alpha >= 0.10 and covers >= 0.05 % of the image.
    # Diffuse removal residue (grain, sheen, texture changes over the whole face) has no strong seed and is dropped;
    # washes (blush, shadow, contour) are smooth and strong enough at their centre to survive.
    sm = cv2.GaussianBlur(a, (0, 0), 1.5 * sig)
    low = (sm >= 0.10).astype(np.uint8)
    nl, lab_, st_, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    seeds = np.unique(lab_[(sm >= 0.30) & (low > 0)])
    keep = np.zeros(nl, bool); keep[seeds] = True; keep[0] = False
    keep &= st_[:, cv2.CC_STAT_AREA] >= 0.0005 * a.size
    coh = keep[lab_].astype(np.float32)
    coh = cv2.GaussianBlur(cv2.dilate(coh, np.ones((5, 5), np.uint8)), (0, 0), sig)
    a = a * np.maximum(coh, wash_)
    a = a * np.clip((a - LOW_A) / LOW_RAMP, 0, 1)
    sh = Bt.shape
    wl = w[0]                                                            # lips part handled separately below
    st = piecewise_warp(np.dstack([a * (1 - wl), (a * (1 - wl))[..., None] * F, w[0] + w[1]]), pr, pt, sh)
    a2, P2, le = st[..., 0], st[..., 1:4], st[..., 4]
    okj = stretch_mask(); a2 = a2 * okj; P2 = P2 * okj[..., None]          # no fragments from folded / squeezed mesh
    # lips: warp with the lip landmarks only (dense, local), then confine to the TARGET's lip shape (lips are
    # identity like brows); holes inside the target lips are filled with the reference lip pigment
    li = LIP_PTS
    stl = piecewise_warp(np.dstack([a * wl, (a * wl)[..., None] * F]), pr[li], pt[li], sh)
    al, Pl = stl[..., 0], stl[..., 1:4]
    tc = target_context(Bt, pt)
    support, hair, breg = tc['support'], tc['hair'], tc['breg']          # final target support / occluder
    if check_glasses and tc['glasses'] > 0.002: return dict(error='target wears glasses')
    tl = tc['tl']
    mlp = a * wl
    if mlp.sum() > 50:
        pig_l = (F * mlp[..., None]).sum((0, 1)) / mlp.sum()
        cov_l = float(np.percentile(a[wl > 0.5], 60)) if (wl > 0.5).any() else 0.0
        if lip_mode == 'recolour':
            # lip SHAPE is identity: recolour the target's own lips with the reference lip pigment instead of warping the
            # reference lip texture (no seam between warped texture and hole fill), modulated by the target's lip shading
            shade = tc['shade']
            al = np.full(sh[:2], cov_l, np.float32); Pl = al[..., None] * np.clip(pig_l[None, None] * shade[..., None], 0, 1)
        else:
            fill = tl * cov_l * (al < 0.5 * cov_l)
            Pl = Pl + fill[..., None] * pig_l; al = al + fill
    al = al * tl; Pl = Pl * tl[..., None]
    P2 = Pl + (1 - al[..., None]) * P2; a2 = al + (1 - al) * a2
    # brows: the reference's brow SHAPE is identity, not makeup -> drop the warped brow part and recolour the target's own
    # brows with the reference brow pigment (alpha-weighted mean colour and coverage). Opaque ornaments in the brow region
    # (gems, crystals, paint strokes: alpha >= ORN_A) are design, not brow tint: they keep their warped shape.
    orn = np.clip((a - ORN_A) / 0.1, 0, 1)
    wb_r = w[2] * (1 - orn); mb = a * wb_r
    wb_t = piecewise_warp(wb_r[..., None], pr, pt, sh)[..., 0]
    a2 = a2 * (1 - wb_t); P2 = P2 * (1 - wb_t)[..., None]
    if mb.sum() > 50:
        pig = (F * mb[..., None]).sum((0, 1)) / mb.sum(); cov = float(np.percentile(a[wb_r > 0.5], 75)) if (wb_r > 0.5).any() else 0.0
        # texture-preserving brow tint on the TARGET's brows; the tint lies UNDER the warped design (face paint over the
        # brow stays on top)
        reg = breg
        tb = cov * reg * tc['strand_w']
        Fb = np.clip(pig * tc['brow_fac'], 0, 1)                         # mild strand shading
        P2 = P2 + (1 - a2)[..., None] * tb[..., None] * Fb; a2 = a2 + (1 - a2) * tb
    F2 = P2 / np.maximum(a2, 1e-4)[..., None]
    a2 = a2 * support; le = le * support                                 # amodal layer (full design, under hair too)
    av = a2 * (1 - hair)                                                 # visible layer (what the photo shows)
    B = tc['B']
    I = av[..., None] * F2 + (1 - av[..., None]) * B                     # exact alpha-over
    if min_design and design_only and parts is None and float(av.mean()) < MIN_DESIGN:
        return dict(error='too little opaque design')
    return dict(a=a2, F=F2, av=av, hair=np.array(hair), I=I, le=le, pt=pt_face)
