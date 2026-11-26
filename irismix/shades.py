"""Shade try-on and recommendation (the paper's shade-recommendation application; numpy / OpenCV after one analysis).

A shade is a layer (alpha, F) over the target T, composited exactly like every IrisMix layer, I = alpha F + (1 - alpha) T:
  foundation  alpha = s * paintable skin (coverage 0.6 by default); F = shade colour + the face's own low-frequency shading
              (CIELAB offsets of a blurred copy of the face), so the base is not a flat sticker; finish matte / satin / dewy
  blush       alpha = s * cheek-apple blob x paintable skin; F = shade with the same low-frequency shading
  lipstick    alpha = s * lip mask (landmark lip contour, mouth interior removed); F = shade x the lip's own shading;
              finish matte / satin / sheer / gloss
Scene light: the white point and exposure are estimated from the sclera (eye contour minus iris, whitest low-chroma pixels),
gray-world over the background as a fallback. Measured skin and lip colours are corrected to D65 (von Kries) before
classification, and every shade is converted back to the photo's light before compositing.
Recommendation (a heuristic for illustration, not a validated colour-matching method): depth from the individual typology
angle, undertone from the hue angle of the corrected skin; foundations ranked by CIEDE2000 with undertone as tie-breaker;
lipsticks restricted to the undertone's family and picked as everyday / brighter / deeper relative to the natural lip."""
import json, os
import numpy as np, cv2

from irismix.utils import lab, rgb

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'shades.json')
TIE = 1.0
HUE_COOL, HUE_WARM = 56.0, 64.0          # undertone thresholds on the corrected skin hue angle (deg)
FINISH = {'matte': (0.55, 0.25, -0.35), 'satin': (0.8, 0.3, 0.0), 'dewy': (1.0, 0.35, 0.45)}        # lam_L, lam_ab, sheen
LIP_FINISH = {'matte': (1.0, 0.45, 0.0), 'satin': (0.95, 0.75, 0.08), 'sheer': (0.6, 0.85, 0.12), 'gloss': (0.85, 0.9, 0.35)}
FAMILY = {'warm': ('coral', 'orange', 'brown', 'red', 'nude', 'peach', 'terracotta', 'brick'),
          'cool': ('rose', 'pink', 'plum', 'berry', 'mauve', 'red', 'wine')}
L_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
R_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
SCLERA_REF = np.array([0.343, 0.309, 0.287])       # linear RGB of the visible sclera under neutral light
EXP_DAMP = 0.5


def lab1(c): return lab(np.asarray(c, np.float32).reshape(1, 1, 3))[0, 0]
def rgb1(L): return rgb(np.asarray(L, np.float32).reshape(1, 1, 3))[0, 0]
def hex2rgb(h): return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255
def rgb2hex(c): return '#%02X%02X%02X' % tuple(np.clip(np.round(np.asarray(c) * 255), 0, 255).astype(int))
def srgb_to_linear(x): x = np.clip(x, 0, 1); return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
def linear_to_srgb(x): x = np.clip(x, 0, 1); return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
def ita(L): return float(np.degrees(np.arctan2(L[0] - 50, L[2])))
def hue(L): return float(np.degrees(np.arctan2(L[2], L[1])))
def chroma(L): return float(np.hypot(L[1], L[2]))


def ciede2000(l1, l2):
    L1, a1, b1 = l1; L2, a2, b2 = l2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2); Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p, h2p = np.degrees(np.arctan2(b1, a1p)) % 360, np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    dh = h2p - h1p
    if C1p * C2p == 0: dh = 0
    elif dh > 180: dh -= 360
    elif dh < -180: dh += 360
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lm, Cmp = (L1 + L2) / 2, (C1p + C2p) / 2
    hm = h1p + h2p if C1p * C2p == 0 else ((h1p + h2p + 360) / 2 if abs(h1p - h2p) > 180 else (h1p + h2p) / 2)
    T = 1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm)) + 0.32 * np.cos(np.radians(3 * hm + 6)) \
        - 0.20 * np.cos(np.radians(4 * hm - 63))
    dth = 30 * np.exp(-((hm - 275) / 25) ** 2); RC = 2 * np.sqrt(Cmp ** 7 / (Cmp ** 7 + 25 ** 7))
    SL = 1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2); SC = 1 + 0.045 * Cmp; SH = 1 + 0.015 * Cmp * T
    RT = -np.sin(np.radians(2 * dth)) * RC
    return float(np.sqrt((dL / SL) ** 2 + (dC / SC) ** 2 + (dH / SH) ** 2 + RT * (dC / SC) * (dH / SH)))


def load_shades(path=DATA):
    with open(path, encoding='utf-8') as f: lib = json.load(f)['shades']
    out = []
    for s in lib:
        s = dict(s)
        if 'lab' in s: s['lab'] = [float(x) for x in s['lab']]; s['hex'] = rgb2hex(rgb1(s['lab']))
        else: s['lab'] = [float(x) for x in lab1(hex2rgb(s['hex']))]
        s.setdefault('finish', 'satin'); s.setdefault('strength', 0.6)
        out.append(s)
    return out


# ------------------------------------------------------------------------------------------------ scene light
def _sclera(pts, shape):
    m = np.zeros(shape, np.uint8)
    for poly, ir in ((L_EYE, 468), (R_EYE, 473)):
        e = np.zeros(shape, np.uint8); cv2.fillPoly(e, [pts[poly].round().astype(np.int32)], 1)
        c = pts[ir]; r = np.linalg.norm(pts[ir + 1:ir + 5] - c, axis=1).mean() * 1.15
        cv2.circle(e, tuple(int(v) for v in np.round(c)), int(round(r)), 0, -1)
        m |= cv2.erode(e, np.ones((3, 3), np.uint8))
    return m.astype(bool)


def estimate_light(img, pts, face_m):
    """img float RGB -> dict(gain (3,), exposure, src, reliable)."""
    X = srgb_to_linear(img); m = _sclera(pts, img.shape[:2]); src = 'sclera'; v = X[m]
    if len(v) >= 40:
        Y = v.mean(1); sat = cv2.cvtColor(img[m][None].astype(np.float32), cv2.COLOR_RGB2HSV)[0][:, 1]
        keep = (Y >= np.percentile(Y, 40)) & (sat <= np.percentile(sat, 20))
        w = np.median(v[keep], 0) if keep.sum() >= 15 else np.median(v, 0)
    else:
        w = X[face_m < 0.05].mean(0); w = w / w.mean() * SCLERA_REF.mean(); src = 'gray-world'
    gain = np.clip((w.mean() / w) * (SCLERA_REF / SCLERA_REF.mean()), 0.6, 1.6)
    exposure = float(np.clip(SCLERA_REF.mean() / max(w.mean(), 1e-3), 0.6, 1.8))
    return dict(gain=gain, exposure=exposure, src=src, reliable=bool(0.7 < exposure < 1.6 and np.abs(np.log(gain)).max() < 0.3))


def to_d65(c, Lt): return linear_to_srgb(srgb_to_linear(np.asarray(c, np.float64)) * Lt['gain'] * Lt['exposure'] ** EXP_DAMP)
def to_scene(c, Lt): return linear_to_srgb(srgb_to_linear(np.asarray(c, np.float64)) / (Lt['gain'] * Lt['exposure'] ** EXP_DAMP))


# ------------------------------------------------------------------------------------------------ face analysis
class Face:
    """masks and colour statistics of one target photo (full frame), computed once (face parsing + landmarks)."""

    def __init__(self, T8):
        from irismix import rules, layer_recover as LR
        self.T = T8.astype(np.float32) / 255; H, W = T8.shape[:2]
        pt = rules.landmarks(T8, fallback=True)
        if pt is None: raise ValueError('no face found')
        prob = rules.parse(T8); lb = rules.argmax0(prob); sup, p3 = LR.support_of(T8)
        p = p3[:, :2].astype(np.float32); self.pts = p; fw = float(np.ptp(p[:, 0])); self.fw = fw
        skin = np.isin(lb, (1, 2)).astype(np.float32) * sup                       # paintable skin minus features
        feat = np.isin(lb, rules.EYES + rules.BROWS + rules.LIPS).astype(np.uint8)
        for idx in (rules.LIP_PTS, rules.BROW_L, rules.BROW_R):
            cv2.fillConvexPoly(feat, cv2.convexHull(np.round(p[idx]).astype(np.int32)), 1)
        k = max(3, int(0.03 * fw) | 1); el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.erode(skin * (1 - cv2.dilate(feat, el)), el)
        self.base_m = np.clip(cv2.GaussianBlur(m, (0, 0), 0.02 * fw), 0, 1).astype(np.float32)
        tc = rules.target_context(T8, pt)
        mouth = np.zeros((H, W), np.uint8); cv2.fillPoly(mouth, [np.round(pt[rules.INNER_LIPS]).astype(np.int32)], 1)
        self.lip_m = np.clip(tc['tl'] * (1 - cv2.GaussianBlur(mouth.astype(np.float32), (0, 0), 1.0)), 0, 1).astype(np.float32)
        self.lip_shade = tc['shade'].astype(np.float32)
        yy, xx = np.mgrid[:H, :W].astype(np.float32); bl = np.zeros((H, W), np.float32)
        for i, j in ((50, 234), (280, 454)):                                        # apple of the cheek, nudged to the ear
            c = 0.8 * pt[i][:2] + 0.2 * pt[j][:2]; r = 0.13 * fw
            bl = np.maximum(bl, np.exp(-((xx - c[0]) ** 2 + ((yy - c[1]) / 0.8) ** 2) / (2 * r * r)))
        self.blush_m = bl * self.base_m
        self.Tlab = lab(self.T)
        low = lab(cv2.GaussianBlur(self.T, (0, 0), 0.012 * fw)); mid = lab(cv2.GaussianBlur(self.T, (0, 0), 0.004 * fw))
        sk = self.base_m > 0.6
        if sk.sum() < 50: sk = self.base_m > 0.2
        self.off_low = low - np.median(low[sk], 0); self.sheen = np.clip(mid[..., 0] - low[..., 0], 0, None)
        self.light = estimate_light(self.T, p, self.base_m)
        v = self.T[sk]; Lv = self.Tlab[sk]; lo, hi = np.percentile(Lv[:, 0], [10, 90]); v = v[(Lv[:, 0] >= lo) & (Lv[:, 0] <= hi)]
        self.skin = lab1(to_d65(np.median(v, 0), self.light))                       # corrected to D65
        lm = self.lip_m > 0.6
        self.lip = lab1(to_d65(np.median(self.T[lm], 0), self.light)) if lm.sum() > 20 else np.array([50, 25, 15], np.float32)

    def _scene_lab(self, shade): return lab1(to_scene(rgb1(shade['lab']), self.light))

    def foundation(self, shade, s=None, finish=None):
        s = shade['strength'] if s is None else s
        lamL, lamab, sh = FINISH[finish or shade.get('finish', 'satin')]; c = self._scene_lab(shade)
        Fl = np.empty_like(self.off_low)
        Fl[..., 0] = c[0] + lamL * self.off_low[..., 0] + sh * self.sheen; Fl[..., 1:] = c[1:] + lamab * self.off_low[..., 1:]
        return np.clip(s * self.base_m, 0, 1), rgb(Fl)

    def blush(self, shade, s=None):
        s = shade['strength'] if s is None else s; c = self._scene_lab(shade)
        Fl = np.empty_like(self.off_low); Fl[..., 0] = c[0] + 0.7 * self.off_low[..., 0]; Fl[..., 1:] = c[1:]
        return np.clip(s * self.blush_m, 0, 1), rgb(Fl)

    def lipstick(self, shade, s=None, finish=None):
        s = shade['strength'] if s is None else s
        cov, g, gloss = LIP_FINISH[finish or shade.get('finish', 'satin')]
        F = np.clip(to_scene(hex2rgb(shade['hex']), self.light)[None, None] * (1 + g * (self.lip_shade - 1))[..., None], 0, 1)
        if gloss: F = np.clip(F + gloss * (np.clip((self.lip_shade - 1.05) / 0.3, 0, 1) * self.lip_m)[..., None] * (1 - F), 0, 1)
        return np.clip(s * cov * self.lip_m, 0, 1).astype(np.float32), F.astype(np.float32)

    # -------------------------------------------------------------------------------------------- recommendation
    def undertone(self):
        h = hue(self.skin)
        return 'cool' if h < HUE_COOL else 'warm' if h > HUE_WARM else 'neutral'

    def report(self):
        s = self.skin
        return dict(skin_hex=rgb2hex(rgb1(s)), ita=round(ita(s), 1), undertone=self.undertone(), lip_hex=rgb2hex(rgb1(self.lip)),
                    light=self.light['src'], light_reliable=self.light['reliable'])

    def recommend(self, shades, n=4):
        """-> dict(foundations=[...n, best first], lipsticks={everyday, brighter, deeper}, blush=shade)."""
        ut = self.undertone()
        F = [s for s in shades if s['category'] == 'foundation']
        rows = sorted(((ciede2000(self.skin, s['lab']), s) for s in F), key=lambda r: r[0])
        best = rows[0]
        for r in rows[1:]:                                                          # undertone as tie-breaker
            if r[0] - best[0] > TIE: break
            if r[1].get('undertone') == ut and best[1].get('undertone') != ut: best = r; break
        found = [best[1]] + [s for _, s in rows if s is not best[1]][:n - 1]
        P = [s for s in shades if s['category'] == 'lipstick']
        fam = [s for s in P if ut == 'neutral' or any(k in (s.get('name_en', '') + ' ' + s.get('family', '')).lower() for k in FAMILY[ut])] or P
        L0 = self.lip; de = {s['id']: ciede2000(L0, s['lab']) for s in fam}; used = set(); lips = {}

        def take(role, cands, key):
            cands = [s for s in cands if s['id'] not in used]
            if cands: s = min(cands, key=key); used.add(s['id']); lips[role] = s
        take('everyday', [s for s in fam if abs(s['lab'][0] - L0[0]) < 12], lambda s: de[s['id']])
        take('brighter', [s for s in fam if chroma(s['lab']) > chroma(L0) + 6], lambda s: de[s['id']])
        take('deeper', [s for s in fam if s['lab'][0] < L0[0] - 8], lambda s: de[s['id']])
        for role in ('everyday', 'brighter', 'deeper'):
            if role not in lips: take(role, fam, lambda s: de[s['id']])
        B = [s for s in shades if s['category'] == 'blush']
        pref = {'warm': ('peach', 'coral', 'bronze'), 'cool': ('rose', 'berry'), 'neutral': ('rose', 'peach')}[ut]
        blush = next((s for p in pref for s in B if p in s.get('name_en', '').lower()), B[0] if B else None)
        return dict(foundations=found, lipsticks=lips, blush=blush)
