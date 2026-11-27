"""Renderer source conditions (shared by the data builder, the trainer and inference).

Conditions are computed on the SOURCE (the bare face the layer is pasted on) inside the 512 face crop
(1.9 x landmark extent, as the decomposer and the composites), so training and inference use the same code:
  S1 (256)  signed eyelid / lip-outline / brow-line distance maps (signed_S1)
  S2 (256)  R part id (lips 1, eyes .8, brows .6, forehead .4, face .2, 0 off-face), G per-eye openness painted on
            each eye region (openness x 2), B alpha removed by the transfer pipeline (warped full layer - pasted, >= 0)
  S3 (256)  R crease-ring distance, G eye-corner blobs, B source eyeball mask (parsed eye labels)
  G1 (256)  mesh normals (0.5 + 0.5 n) from the MediaPipe 3D landmarks, 0 off-mesh
  G2 (256)  R hair / hat occluder, G glasses (separate, NOT merged into the paintable area), B paintable support
  K, A (512) the warped full reference layer over black (aF) and its alpha, not part-cut, not filtered
  crops (256) native-resolution, landmark-normalised reference crops of the lips and of each eye
plus the caption helpers of a decomposed reference layer (caption_colour_name / describe_components).
Per-source results are cached (read-only arrays; source_conditions returns a shallow copy of the cached dict)."""
import numpy as np, cv2

from irismix import rules, face_parts as FP
from irismix.utils import R, face_box, crop_resize, image_key, read_only

RS = 256          # resolution of the low-frequency maps and the crops
EYE_L_C, EYE_R_C = (33, 133), (362, 263)            # eye corners (outer, inner) / (inner, outer) in MediaPipe order
LID = dict(L=(159, 145, 33, 133), R=(386, 374, 362, 263))
CREASE_L = [113, 225, 224, 223, 222, 221, 189]       # 3rd ring above the upper lid (crease proxy)
CREASE_R = [342, 445, 444, 443, 442, 441, 413]
MOUTH_C = (61, 291)


# ------------------------------------------------------------------------------------------------ landmarks / parsing
def openness(p):
    """per-eye openness = lid gap / eye width (MediaPipe), (left, right)."""
    out = []
    for k in ('L', 'R'):
        u, l, c0, c1 = LID[k]
        out.append(float(np.linalg.norm(p[u, :2] - p[l, :2]) / max(np.linalg.norm(p[c0, :2] - p[c1, :2]), 1e-3)))
    return out


def support_and_hair(prob, pt):
    """occluder with the brow rule of rules.transfer_layer (isolated 'hair' blobs in the brow region are brows).
    Memoised per (cached probability map, landmarks): source_conditions and layer_recover.support_of share it."""
    pk = np.ascontiguousarray(pt).tobytes()
    return rules.memoise_on_map(('support_and_hair', pk), prob, lambda: _support_and_hair(prob, pt))


def _support_and_hair(prob, pt):
    support, hair = rules.target_mask(prob)
    breg = rules.brow_region(pt, prob.shape[1:])
    hb = np.isin(rules.label_map(prob), rules.HAIR).astype(np.uint8)
    nlab, lab = cv2.connectedComponents(hb)
    keep = np.zeros(nlab, bool); keep[np.unique(lab[(hb > 0) & (breg < 0.05)])] = True; keep[0] = False
    hair = hair * (1 - breg) + hair * breg * keep[lab].astype(np.float32)
    return np.clip(np.maximum(support, breg), 0, 1).astype(np.float32), np.clip(hair, 0, 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ source conditions
def _signed(lines_closed, p, S, ramp=32.0):
    """signed distance to a closed polygon (negative inside)."""
    m = np.zeros((S, S), np.uint8); cv2.fillPoly(m, [np.round(p[lines_closed]).astype(np.int32)], 1)
    din = cv2.distanceTransform(m, cv2.DIST_L2, 5); dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    return np.clip(dout - din, -ramp, ramp) / ramp


def signed_S1(p, S=512, out=RS):
    """p: landmarks in the 512 crop. R: eye openings (min over both eyes; < 0 inside the opening), G: lips (outer
    contour; mouth interior forced to -1), B: brow line distance (unsigned). 0.5 + 0.5 * value, at RS."""
    eyes = np.minimum(_signed(FP.EYE_L, p, S), _signed(FP.EYE_R, p, S))
    lips = _signed(FP.LIP_OUT, p, S)
    mi = np.zeros((S, S), np.uint8); cv2.fillPoly(mi, [np.round(p[FP.LIP_IN]).astype(np.int32)], 1); lips[mi > 0] = -1
    bl = np.full((S, S), 255, np.uint8)
    for g in FP.BROWLINE_L + FP.BROWLINE_R: cv2.polylines(bl, [np.round(p[g]).astype(np.int32)], False, 0, 1)
    brow = np.minimum(cv2.distanceTransform(bl, cv2.DIST_L2, 5), 63.75) / 63.75 * 2 - 1
    x = 0.5 + 0.5 * np.stack([eyes, lips, brow], -1)
    return cv2.resize(x.astype(np.float32), (out, out), interpolation=cv2.INTER_AREA)


def _normals(p3, n):
    from scipy.spatial import Delaunay
    xy = p3[:, :2]; dt = Delaunay(xy); tris = dt.simplices
    v = p3[tris]; nrm = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    nrm[nrm[:, 2] > 0] *= -1                                         # towards the camera (MediaPipe z grows away)
    vn = np.zeros_like(p3)
    for k in range(3): np.add.at(vn, tris[:, k], nrm)
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-6)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    q = np.stack([xx.ravel() + .5, yy.ravel() + .5], 1)
    s = dt.find_simplex(q); ok = s >= 0
    T = dt.transform[s[ok]]; b = np.einsum('nij,nj->ni', T[:, :2], q[ok] - T[:, 2]); b = np.c_[b, 1 - b.sum(1)]
    out = np.zeros((n * n, 3), np.float32)
    out[ok] = (b[..., None] * vn[tris[s[ok]]]).sum(1)
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
    img = (0.5 + 0.5 * out.reshape(n, n, 3)) * ok.reshape(n, n, 1)
    img[..., 1] = np.where(ok.reshape(n, n), 1 - img[..., 1], 0)    # y up
    return img.astype(np.float32)


def source_conditions(S, p3=None, prob=None):
    """S uint8 full frame (the bare / source). Returns dict(box, maps at RS: S1, S2 (B channel 0), S3, G1, G2,
    full-frame support/hair (for the paste), openness, p3). Cached per source (default p3 / prob): a shallow copy of the
    cached dict is returned (set_removed replaces S2), the arrays are read-only."""
    if p3 is None and prob is None:
        k = ('source_conditions', image_key(S)); hit = rules.DERIVED.get(k)
        if hit is None:
            hit = _source_conditions(S)
            hit = rules.DERIVED.put(k, None if hit is None else read_only(hit))
        return None if hit is None else dict(hit)
    return _source_conditions(S, p3, prob)


def _source_conditions(S, p3=None, prob=None):
    if p3 is None: p3 = rules.landmarks_3d(S)
    if p3 is None: return None
    if prob is None: prob = rules.parse(S)
    box = face_box(p3, S.shape); X0, Y0, side = box
    sc = R / side
    p = ((p3[:, :2] - np.array([X0, Y0], np.float32)) * sc).astype(np.float32)   # landmarks in the 512 crop
    support, hair = support_and_hair(prob, p3[:, :2])
    lab = rules.label_map(prob)
    w = FP.landmark_part_weights(p)
    face = FP.polygon_mask(cv2.convexHull(np.concatenate([p[FP.FACE_OVAL], p[FP.FACE_OVAL].mean(0) + 1.75 * (p[FP.UPPER_OVAL] - p[FP.FACE_OVAL].mean(0))]).astype(np.float32))[:, 0], 0, 2)
    code = np.array([1.0, 0.8, 0.6, 0.2, 0.4], np.float32)           # lips, eyes, brows, face, forehead
    pid = code[w.argmax(0)] * face
    op = openness(p3)
    opm = np.zeros((R, R), np.float32)
    for k, (eye, o) in enumerate(((FP.EYE_L, op[0]), (FP.EYE_R, op[1]))):
        e = p[eye]; reg = FP.polygon_mask(cv2.convexHull(FP.scale_about_centre(e, np.array([1.6, 2.4], np.float32)).astype(np.float32))[:, 0], 0, 3)
        opm = np.maximum(opm, reg * np.clip(o * 2, 0, 1))
    m = np.full((R, R), 255, np.uint8)
    for g in (CREASE_L, CREASE_R): cv2.polylines(m, [np.round(p[g]).astype(np.int32)], False, 0, 1)
    crease = np.minimum(cv2.distanceTransform(m, cv2.DIST_L2, 5), 63.75) / 63.75
    corners = np.zeros((R, R), np.float32)
    for i in (33, 133, 362, 263):
        cv2.circle(corners, tuple(int(round(v)) for v in p[i]), 3, 1.0, -1)
    corners = np.clip(cv2.GaussianBlur(corners, (0, 0), 3) * 3, 0, 1)
    eyeball = crop_resize(np.isin(lab, rules.EYES).astype(np.float32), box)
    glasses = crop_resize((lab == rules.GLASSES).astype(np.float32), box)
    nrm = _normals(np.c_[p, p3[:, 2] * sc].astype(np.float64), R)
    dn = lambda x: cv2.resize(x, (RS, RS), interpolation=cv2.INTER_AREA)
    return dict(box=box, p=p, p3=p3, open=op, support=support, hair=hair,
                S1=signed_S1(p),
                S2=dn(np.stack([pid, opm, np.zeros_like(pid)], -1)),
                S3=dn(np.stack([crease, corners, eyeball], -1)),
                G1=dn(nrm), G2=dn(np.stack([crop_resize(hair, box), glasses, crop_resize(support, box)], -1)))


def set_removed(cond, A512, av512):
    """S2.B = alpha removed by the transfer pipeline; A512 must already be cut to the design mask."""
    cond['S2'] = cond['S2'].copy(); cond['S2'][..., 2] = cv2.resize(np.clip(A512 - av512, 0, 1), (RS, RS), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------------------------------------ text
def caption_colour_name(rgb):
    lab = cv2.cvtColor(np.array([[rgb]], np.float32), cv2.COLOR_RGB2LAB)[0, 0]
    L, a, b = lab; C = np.hypot(a, b); h = (np.degrees(np.arctan2(b, a)) + 360) % 360
    if C < 8: return 'black' if L < 25 else 'white' if L > 85 else 'grey' if C < 4 else ('brown' if L < 55 else 'nude')
    if L < 45 and 20 < h < 90: return 'brown'
    for hi, nm in ((15, 'pink'), (40, 'red'), (60, 'coral'), (80, 'orange'), (100, 'gold'), (150, 'green'), (210, 'teal'),
                   (275, 'blue'), (330, 'purple'), (361, 'pink')):
        if h < hi: return ('deep ' if L < 40 else 'pale ' if L > 78 else '') + nm
    return 'nude'


def describe_components(comps):
    """layer components {name: {cov, area, rgb}} -> structured parts (colour names + opacity)."""
    nm = {'lips': 'Lips', 'liner': 'Eyeliner', 'eyeshadow': 'Eyeshadow', 'brows': 'Brows', 'decor': 'Accents', 'face': 'Cheeks'}
    st = []
    for k, v in comps.items():
        if not v or v.get('rgb') is None or v.get('area', 0) < 3e-4: continue
        c = caption_colour_name(v['rgb']); cov = v.get('cov', 0)
        op = 'opaque' if cov > 0.6 else 'medium' if cov > 0.3 else 'sheer'
        st.append(dict(part=nm.get(k, k.title()), desc=f'{c} {k if k != "face" else "blush"}', colour=[c], finish=[], edges=[], opacity=[op]))
    return st
