"""Renderer inference helpers: reference-layer quantities of a look (warp, design units, layer crops, caption), the
512-crop preparation (K / A cut to the design, Omega) and the masked-residual paste-back at full resolution. Conditions are
built exactly like the training data (conditions.py) on the source.
Output: I = (1 - m) S + m (C + up(H512 - C512)), m = Omega upsampled into the crop box (0 outside -> source exactly)."""
import numpy as np, cv2

from irismix import config, rules, conditions as CM, edit_region as ER, face_parts as FP
from irismix.utils import face_box, crop_resize, to_device, tensor_to_image

ORDER = {'face': 0, 'forehead': 1, 'brows': 2, 'eyes': 3, 'lips': 4}         # paint order of a look (finer parts on top)
NAMES = rules.PART_NAMES
CAP_PART = {'eyes': ('Eyes',), 'lips': ('Lips',), 'face': ('Face', 'Cheeks', 'Accents'), 'forehead': ('Forehead',), 'brows': ('Brows',)}
OMEGA_R = config.OMEGA_R           # Omega dilation in px (0 = edit_region default, 6 px)


# ------------------------------------------------------------------------------------------------ reference layers
def warp_reference_layer(a, F, Br, S):
    """full reference layer (a, F) on the reference bare Br -> (alpha, F) warped onto the source S (landmark mesh)."""
    pr, pt = rules.landmarks(Br), rules.landmarks(S)
    st = rules.piecewise_warp(np.dstack([a, a[..., None] * F]), rules.extend_landmarks(pr, Br.shape), rules.extend_landmarks(pt, S.shape), S.shape)
    return np.clip(st[..., 0], 0, 1), np.clip(st[..., 1:4], 0, 1)


def design_unit_mask(a, w, parts):
    """design units mostly inside the requested parts: a connected design (smoothed alpha >= 0.1) whose alpha lies mostly
    (>= 50 %) in the requested parts is taken whole (same rule as the training data)."""
    m = np.clip(sum(w[NAMES.index(k)] for k in parts), 0, 1)
    sm_ = cv2.GaussianBlur(a, (0, 0), max(2.0, a.shape[0] / 256))
    nl_, lab_u = cv2.connectedComponents((sm_ >= 0.1).astype(np.uint8), connectivity=8)
    whole = np.zeros(a.shape, np.float32)
    for c in range(1, nl_):
        cm = lab_u == c; s = float(a[cm].sum())
        if s > 0 and float((a * m)[cm].sum()) / s >= 0.5: whole[cm] = 1
    k = max(3, a.shape[0] // 100)
    return np.clip(cv2.GaussianBlur(cv2.dilate(whole, np.ones((2 * k + 1, 2 * k + 1), np.uint8)), (0, 0), k / 2), 0, 1)


def reference_crops(Ir, Br, a, F, w=None):
    """reference-frame layer crops: landmarks, eyeball, mouth interior, design
    mask, part regions (lip polygon dilated 12 % of the mouth width; eye zone + brows where the design covers them)."""
    p = rules.landmarks(Br)
    if p is None: p = rules.landmarks(Ir)
    prob = rules.parse(Br)
    eyeball = np.isin(rules.label_map(prob), rules.EYES).astype(np.float32)
    mouth = np.zeros(Br.shape[:2], np.uint8); cv2.fillPoly(mouth, [np.round(p[rules.INNER_LIPS]).astype(np.int32)], 1)
    sc = face_box(p, Br.shape)[2] / 512
    D = ER.design_mask(a, scale=sc)
    mw = np.linalg.norm(p[61] - p[291]); kk = max(1, int(round(0.12 * mw)))
    lipreg = np.zeros(Br.shape[:2], np.uint8); cv2.fillPoly(lipreg, [np.round(p[FP.LIP_OUT]).astype(np.int32)], 1)
    lipreg = cv2.dilate(lipreg, np.ones((2 * kk + 1, 2 * kk + 1), np.uint8))
    if w is None:                                                     # a decomposed layer: landmark part weights on the crop
        bx = face_box(p, Br.shape); pc = ((p - [bx[0], bx[1]]) * (512 / bx[2])).astype(np.float32)
        w5 = FP.landmark_part_weights(pc); w = np.zeros((5,) + Br.shape[:2], np.float32)
        X0, Y0, side = bx
        for k in range(5): w[k, Y0:Y0 + side, X0:X0 + side] = cv2.resize(w5[k], (side, side))
    eyereg = (w[1] > 0.3) | ((w[2] > 0.3) & D)
    lc = lambda i0, i1, kind, reg: ER.layer_crop(a, F, p, i0, i1, kind, eyeball=eyeball, mouth_in=mouth.astype(np.float32), D=D, region=reg)
    return dict(lip=lc(*CM.MOUTH_C, 'lips', lipreg), eyeL=lc(*CM.EYE_L_C, 'eye', eyereg), eyeR=lc(*CM.EYE_R_C, 'eye', eyereg))


def layer_caption(Ref, Bru, a, F, parts=None):
    """structured caption of a decomposed layer: measured part colours (alpha-weighted median of F per landmark part on the
    512 crop) in the training caption format, restricted to the caption keys of `parts` (None: all). -> dict."""
    pr = rules.landmarks(Bru); comps = {}
    if pr is not None:
        bx = face_box(pr, Ref.shape); pc = ((pr - [bx[0], bx[1]]) * (512 / bx[2])).astype(np.float32)
        wv = FP.landmark_part_weights(pc); a512 = crop_resize(a, bx); F512 = crop_resize(F, bx)
        for k, nm in ((0, 'lips'), (1, 'eyeshadow'), (3, 'face'), (4, 'decor')):
            sel = (wv[k] > 0.5) & (a512 > 0.15)
            if sel.sum() > 30: comps[nm] = dict(cov=float(a512[wv[k] > 0.5].mean()), area=float(sel.mean()), rgb=np.median(F512[sel], 0).tolist())
    st = CM.describe_components(comps)
    cap = {x['part'] if x['part'] in ('Lips', 'Cheeks', 'Accents') else 'Eyes':
           dict(colour=x['colour'], finish='', motif='', placement='', edges='soft', opacity=x['opacity'][0] if x['opacity'] else 'medium') for x in st}
    if parts is not None:
        keep = set().union(*[CAP_PART[p] for p in parts])
        cap = {k: v for k, v in cap.items() if k in keep}
    return cap


# ------------------------------------------------------------------------------------------------ 512 crop + paste-back
def prepare(it):
    """source conditions on the 512 face crop: K / A cut to the design and K over grey, layer crops, Omega = edit_region with
    the protected zones (eyeball, mouth interior, hair, glasses, parts NOT in the look). -> (batch arrays, bookkeeping)."""
    S = it['S']
    cond = CM.source_conditions(S)
    if cond is None: raise ValueError('no face found in the source photo')
    box = cond['box']; cr = lambda x: crop_resize(x, box)
    A5, av5 = np.clip(cr(it['A']), 0, 1), np.clip(cr(it['av']), 0, 1)
    D5 = ER.design_mask(A5).astype(np.float32); A5 = A5 * D5
    K5 = np.clip(cr(it['K']) * D5[..., None] + (1 - A5[..., None]) * 0.5, 0, 1)
    CM.set_removed(cond, A5, av5)
    prob = rules.parse(S); lab = rules.label_map(prob)
    p = cond['p']; mouth = np.zeros((512, 512), np.uint8); cv2.fillPoly(mouth, [np.round(p[rules.INNER_LIPS]).astype(np.int32)], 1)
    unreq = None
    if it.get('parts'):
        w = FP.landmark_part_weights(p)
        unreq = np.clip(sum(w[NAMES.index(k)] for k in NAMES if k not in it['parts']), 0, 1)
    D = ER.design_mask(A5, av5)
    Pm = ER.protect_mask(eyeball=cr(np.isin(lab, rules.EYES).astype(np.float32)), mouth_in=mouth.astype(np.float32), hair=cr(cond['hair']),
                         glasses=cr((lab == rules.GLASSES).astype(np.float32)), unrequested=unreq, D=D)
    om = ER.omega(D, protect=Pm, r=OMEGA_R) if OMEGA_R else ER.omega(D, protect=Pm)
    C5 = cr(it['C'])
    arr = dict(C=C5, K=K5, A=A5, S1=cond['S1'], S2=cond['S2'], S3=cond['S3'], G1=cond['G1'], G2=cond['G2'], **it['crops'])
    return arr, dict(box=box, C5=C5, om=om, D=D)


def compose_full(S, C, H5, info, av):
    """masked residual at full resolution: the render's change inside Omega (512 crop, upsampled into the crop box) is
    added to the paste C; outside Omega the source S is kept exactly (outside the crop box: the paste where av > 0.02).
    -> (I float [0, 1], m)."""
    X0, Y0, side = info['box']; R = 512
    up = lambda x: cv2.resize(x, (side, side), interpolation=cv2.INTER_CUBIC if side > R else cv2.INTER_AREA)
    Sf = S.astype(np.float32) / 255
    r = np.zeros_like(Sf); r[Y0:Y0 + side, X0:X0 + side] = up(H5 - info['C5'])
    m = np.zeros(S.shape[:2], np.float32); m[Y0:Y0 + side, X0:X0 + side] = np.clip(up(info['om']), 0, 1); m[m < 1 / 512] = 0
    out_crop = np.ones(S.shape[:2], bool); out_crop[Y0:Y0 + side, X0:X0 + side] = False
    m = np.where(out_crop & (av > 0.02), 1.0, m).astype(np.float32)
    return np.clip((1 - m[..., None]) * Sf + m[..., None] * np.clip(C + r, 0, 1), 0, 1), m


def render_one(rm, arr, text, sigma=None, steps=None, seed=None):
    """one sample of the renderer on prepared arrays -> H5 (512, 512, 3) float [0, 1] (bf16 autocast)."""
    import torch
    b = to_device(arr, rm.dev)
    with torch.no_grad(), torch.autocast(torch.device(rm.dev).type, dtype=torch.bfloat16):
        h = rm.sample(b, [text], sigma=config.SIGMA if sigma is None else sigma, steps=config.STEPS if steps is None else steps,
                      seed=config.SEED if seed is None else seed).float()
    return tensor_to_image(h)
