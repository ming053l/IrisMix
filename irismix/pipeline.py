"""A look = a source photo + references, each supplying some parts ('lips', 'eyes', 'brows', 'face', 'forehead') or the
whole layer (parts None). End to end:
  1. the decomposer splits every reference photo into (bare, layer)                         decompose()
  2. the transfer rules warp each reference layer onto the source and paste the requested parts in the paint order
     face < forehead < brows < eyes < lips (exact alpha-over) -> warp-only composite C   paste()
  3. renderer conditions: the full (unfiltered) reference layers restricted to whole design units of the requested parts,
     warped onto the source (A, K), reference layer crops (lips / eyes), per-part colour caption   look_item()
  4. render_steps.prepare -> render_steps.render_one (one step, sigma 0.65) -> render_steps.compose_full (masked residual)
  5. layer_recover.layer_from_render: closed-form layer (alpha~, F') of the render for edits
  6. part_masks + apply_edits: closed-form edits of that layer (strength, part toggles, recolour, hue)
Steps 2, 3, 5 and 6 run on the CPU (face parsing on the GPU when available)."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2

from irismix import decomposer as DEC, render_steps as RD, captions as CP, rules, edits as ED, layer_recover as LR, face_parts as FP
from irismix.utils import face_box, rgb_to_lab


def _parallel_map(fn, items):
    """fn over the items in worker threads (the per-image caches are lock-protected; OpenCV and NumPy release the GIL),
    results in input order. Every item is computed exactly as in a plain loop, so the outputs do not change."""
    items = list(items)
    if len(items) < 2: return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(len(items), 4)) as ex: return list(ex.map(fn, items))


def decompose(s1, ref):
    """Decomposition of one reference photo (uint8) -> dict(a, F, w, Br, Im) in the reference frame, or dict(error=...)."""
    L = DEC.predict_layer(s1, ref)
    if 'error' in L: return L
    return dict(a=L['a'], F=L['F'], w=L['w'], Br=L['Br'], Im=ref)


def design_transform(S, dx=0.0, dy=0.0, scale=1.0, mirror=False):
    """2x3 affine on the source frame that moves a transferred design: mirror about the face's vertical axis, scale about the
    face centre, then shift by (dx, dy) face widths. -> None for the identity."""
    if not mirror and scale == 1.0 and dx == 0 and dy == 0: return None
    p = rules.require_landmarks_3d(S)[:, :2]; c = p.mean(0); fw = float(np.ptp(p[:, 0]))
    A = np.diag([-scale if mirror else scale, scale])
    t = c + np.array([dx, dy]) * fw - A @ c
    return np.hstack([A, t[:, None]]).astype(np.float32)


def move_layer(a, K, M, shape, support=None):
    """warp a layer given as (alpha, premultiplied colour K) by the affine M; optional paintable-skin support."""
    H, W = shape[:2]
    a2 = cv2.warpAffine(a.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    K2 = cv2.warpAffine(K.astype(np.float32), M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    if support is not None: s = support.astype(np.float32); a2, K2 = a2 * s, K2 * s[..., None]
    return np.clip(a2, 0, 1), np.clip(K2, 0, 1)


def paste(S, refs):
    """warp-only composite. refs: list of dict(layer=decompose(...), parts=None | [part, ...], M=None | 2x3 affine of
    design_transform). -> (C float [0, 1], av combined visible coverage, per-reference transfer results) or raises ValueError."""
    def transfer(r):
        L = r['layer']
        return rules.transfer_layer((L['a'], L['F'], L['w']), L['Br'], L['Im'], S, parts=r['parts'], check_glasses=False,
                                    design_only=True, min_design=False, lip_mode='recolour')
    tps = []; sup = None
    for r, tp in zip(refs, _parallel_map(transfer, refs)):           # the references are transferred in parallel
        if 'error' in tp: raise ValueError(f"transfer failed ({tp['error']}) for parts {r['parts'] or 'all'}")
        if r.get('M') is not None:                            # repositioned design, kept on paintable skin
            if sup is None: sup = LR.support_of(S)[0]
            a2, K2 = move_layer(tp['av'], tp['av'][..., None] * tp['F'], r['M'], S.shape, sup)
            tp['av'], tp['F'] = a2, np.clip(K2 / np.maximum(a2, 1e-6)[..., None], 0, 1)
        tp['order'] = -1 if r['parts'] is None else min(RD.ORDER[p] for p in r['parts'])
        tps.append(tp)
    C = S.astype(np.float32) / 255; av = np.zeros(S.shape[:2], np.float32)
    for tp in sorted(tps, key=lambda t: t['order']):
        a = tp['av'][..., None]; C = a * tp['F'] + (1 - a) * C
        av = 1 - (1 - av) * (1 - tp['av'])
    return C, av, tps


def look_item(S, refs, C, av):
    """Renderer item of a look: source S, paste C / av, warped full reference layers A / K, crops, caption, parts."""
    A = np.zeros(S.shape[:2], np.float32); K = np.zeros(S.shape, np.float32)
    g = np.full((256, 256, 3), 0.5, np.float32); crops = dict(lip=g, eyeL=g, eyeR=g); cap = {}
    single = len(refs) == 1 and refs[0]['parts'] is None
    order = lambda r: -1 if r['parts'] is None else min(RD.ORDER[p] for p in r['parts'])

    def per_ref(r):                                           # everything that depends on one reference only
        L, ps = r['layer'], r['parts']
        a = L['a']
        if ps is not None:                                    # whole design units of the requested parts
            a = a * np.maximum(RD.design_unit_mask(a, L['w'], ps), np.clip(sum(L['w'][RD.NAMES.index(p)] for p in ps), 0, 1))
        aw, Pw = RD.warp_reference_layer(a, L['F'], L['Br'], S)
        if r.get('M') is not None: aw, Pw = move_layer(aw, Pw, r['M'], S.shape)
        rc = RD.reference_crops(L['Im'], L['Br'], L['a'], L['F']) if ps is None or set(ps) & {'lips', 'eyes'} else None
        return aw, Pw, rc, RD.layer_caption(L['Im'], L['Br'], L['a'], L['F'], ps)
    ordered = sorted(refs, key=order)
    for r, (aw, Pw, rc, c) in zip(ordered, _parallel_map(per_ref, ordered)):   # computed in parallel, folded in paint order
        ps = r['parts']
        K = Pw + (1 - aw[..., None]) * K; A = aw + (1 - aw) * A
        if rc is not None:
            if ps is None or 'lips' in ps: crops['lip'] = rc['lip']
            if ps is None or 'eyes' in ps: crops['eyeL'], crops['eyeR'] = rc['eyeL'], rc['eyeR']
        cap.update(c)
    parts = None if single else sorted({p for r in refs for p in (r['parts'] or RD.NAMES)})
    return dict(S=S, C=C, av=av, A=A, K=K, crops=crops, text=CP.caption_text(cap), parts=parts)


def part_masks(S):
    """5 landmark part weights of the source in its full frame, in render_steps.NAMES order (lips, eyes, brows, face, forehead)."""
    p3 = rules.require_landmarks_3d(S); X0, Y0, side = face_box(p3, S.shape)
    w5 = FP.landmark_part_weights(((p3[:, :2] - [X0, Y0]) * (512 / side)).astype(np.float32))
    w = np.zeros((5,) + S.shape[:2], np.float32)
    for k in range(5): w[k, Y0:Y0 + side, X0:X0 + side] = cv2.resize(w5[k], (side, side))
    return w


def px_per_mm(S):
    """pixels per millimetre of a face photo from its inter-pupillary distance (iris centres; 63 mm average)."""
    p = rules.require_landmarks_3d(S)
    return float(np.linalg.norm(p[468, :2] - p[473, :2])) / 63.0


def apply_edits(at, F, S, w, strength=1.0, off=(), recolour=None, hue=None, part_strength=None, soft=None, finish=None,
                mm=None):
    """closed-form edits of a layer (alpha~, F') over the target S (uint8), w = part_masks(S). Applied in this order:
      soft          {part: radius in mm}: Gaussian softening (mm = pixels per millimetre, px_per_mm(S) when None)
      part_strength 5 per-part strengths in render_steps.NAMES order (selection s = sum_k s_k w_k); off: parts switched off
      recolour      {part: CIELAB triple}; hue {part: degrees}
      finish        {part: k}: lip finish, k = -1 matte ... +1 gloss
    then composited with the global strength s: out = min(1, s alpha) F + (1 - min(1, s alpha)) S.
    -> (edited image float [0, 1], edited alpha, edited F)."""
    idx = RD.NAMES.index
    if soft:
        mm = mm or px_per_mm(S)
        for p, r in soft.items():
            if r > 0: at, F = ED.soften(at, F, w[idx(p)], r * mm)
    ps = list(part_strength) if part_strength is not None else [1.0] * 5
    for p in off: ps[idx(p)] = 0.0
    if any(x != 1.0 for x in ps): at = ED.select(at, w, ps)
    for p, t in (recolour or {}).items():
        F = ED.recolour_to(F, (w[idx(p)] > 0.5) & (at > LR.EPS), np.asarray(t, np.float32))
    for p, d in (hue or {}).items():
        F = ED.rotate_hue(F, (w[idx(p)] > 0.5) & (at > LR.EPS), float(d))
    for p, k in (finish or {}).items():
        m = (w[idx(p)] > 0.5) & (at > LR.EPS)
        if k and m.any():
            F = ED.lip_finish(F, m, float(k), float(rgb_to_lab(F)[..., 0][m].mean()))
    return ED.compose(at, F, S.astype(np.float32) / 255, strength), at, F
