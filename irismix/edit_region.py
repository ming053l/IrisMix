"""Design mask D, edit region Omega and identity-safe reference crops (numpy, CPU) of the renderer.

design_mask(A, av)       D: geodesic hysteresis on the warped full alpha. Seeds A_s >= 0.2 (A_s = A blurred sigma 1);
                         grown at most REACH = 8 px (at 512) through A_s >= 0.06; components < 6 px dropped; union the
                         pasted alpha av > 0.02 (bounded growth: low closed-form skin residue cannot bridge designs
                         across the whole face).
omega(D, protect, ...)   Omega = blur(dilate(D, r = 6 px), sigma = 2) * (1 - protect), protect = eyeball (parsing, eroded
                         1 px so lash-line liner stays editable) + mouth interior (inner-lip polygon) + hair / hat occluder
                         (> 0.5) + glasses + (inference, part toggles) the support of every part that is NOT requested,
                         minus the design core that actually lies there (a design pixel is never protected).
layer_crop(...)          reference part crop = the reference LAYER over neutral grey, alpha cut to D, eyeball and mouth
                         interior removed, in the landmark-normalised native frame: alpha F + (1 - alpha) 0.5. No bare
                         reference skin, iris, natural brow hair, lip-line shape of the skin or moles can enter."""
import numpy as np, cv2

REACH, TAU_HI, TAU_LO, MIN_CC = 8, 0.20, 0.06, 6
R_DIL, FEATHER = 6, 2.0


def design_mask(A, av=None, reach=REACH, hi=TAU_HI, lo=TAU_LO, scale=1.0):
    """A, av: (H, W) float [0, 1] at 512 (scale = H / 512 for other sizes). -> bool D."""
    As = cv2.GaussianBlur(A.astype(np.float32), (0, 0), 1.0 * scale)
    seed = (As >= hi).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    small = np.where(st[:, cv2.CC_STAT_AREA] < MIN_CC * scale * scale)[0]
    seed[np.isin(lab, small[small > 0])] = 0
    low = (As >= lo).astype(np.uint8); k = np.ones((3, 3), np.uint8)
    for _ in range(int(round(reach * scale))):                  # geodesic growth, bounded reach
        seed = cv2.dilate(seed, k) & low
    D = seed.astype(bool)
    if av is not None: D |= av > 0.02
    return D


def protect_mask(eyeball=None, mouth_in=None, hair=None, glasses=None, unrequested=None, D=None):
    z = None
    for m, er in ((eyeball, 1), (mouth_in, 0), (hair, 0), (glasses, 0)):
        if m is None: continue
        m = (m > 0.5).astype(np.uint8)
        if er: m = cv2.erode(m, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
        z = m if z is None else np.maximum(z, m)
    if unrequested is not None:
        u = (unrequested > 0.5).astype(np.uint8)
        if D is not None: u = u * (1 - cv2.dilate(D.astype(np.uint8), np.ones((3, 3), np.uint8)))
        z = u if z is None else np.maximum(z, u)
    return np.zeros_like(D, np.float32) if z is None else z.astype(np.float32)


def omega(D, protect=None, r=R_DIL, feather=FEATHER, scale=1.0):
    r = max(1, int(round(r * scale)))
    m = cv2.dilate(D.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(np.float32)
    if protect is not None: m = m * (1 - protect)
    return np.clip(cv2.GaussianBlur(m, (0, 0), feather * scale), 0, 1)


# ----------------------------------------------------------------------------------------------- reference crops
def layer_crop(a, F, p, i0, i1, kind, n=256, eyeball=None, mouth_in=None, D=None, grey=0.5, region=None):
    """reference layer (a, F) in the REFERENCE frame -> landmark-normalised crop of the layer over grey.
    a (H, W) [0, 1], F (H, W, 3) [0, 1], p landmarks (478, 2); i0, i1: mouth corners / eye corners (conditions.MOUTH_C, EYE_*_C).
    Similarity normalisation (two landmarks on fixed positions); no photo pixel enters."""
    a = a.astype(np.float32).copy()
    if D is not None: a *= D.astype(np.float32)
    if region is not None: a *= region.astype(np.float32)        # part-restricted: lip polygon / eye + painted-brow zone
    for m in (eyeball, mouth_in):
        if m is not None: a *= 1 - (m > 0.5)
    x, y = p[i0, :2].astype(np.float64), p[i1, :2].astype(np.float64)
    if x[0] > y[0]: x, y = y, x
    if kind == 'lips': da, db = np.array([0.2 * n, 0.5 * n]), np.array([0.8 * n, 0.5 * n])
    else: da, db = np.array([0.22 * n, 0.58 * n]), np.array([0.78 * n, 0.58 * n])
    M, _ = cv2.estimateAffinePartial2D(np.stack([x, y]).astype(np.float32), np.stack([da, db]).astype(np.float32))
    if M is None: return np.full((n, n, 3), grey, np.float32)
    flag = cv2.INTER_AREA if np.linalg.norm(y - x) > 0.6 * n else cv2.INTER_CUBIC
    pm = cv2.warpAffine(np.dstack([a, a[..., None] * F]).astype(np.float32), M, (n, n), flags=flag, borderMode=cv2.BORDER_CONSTANT)
    al = np.clip(pm[..., 0], 0, 1)[..., None]
    return np.clip(pm[..., 1:4] + (1 - al) * grey, 0, 1)      # premultiplied over grey


