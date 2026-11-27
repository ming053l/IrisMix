"""Layer edits by plain alpha-over (numpy, CPU): a layer (alpha (H, W) in [0, 1], F (H, W, 3) in [0, 1]) over a target T.
These are the formulas of the paper's layer edits:
  compose(at, F, T, s)          out = s alpha F + (1 - s alpha) T                 (strength s; alpha clipped to [0, 1])
  part_off(at, w)               alpha (1 - w)                                     (switch a part off: w = its part weight)
  recolour_to(F, m, Lab)        F inside m -> the target CIELAB colour, keeping the part's lightness variation (shading)
  rotate_hue(F, m, deg)         F inside m with its a*b* chroma rotated by deg
  select(at, w, s)              alpha min(1, sum_k s_k w_k alpha)                  (spatial selection / per-part strength)
  soften(at, F, m, r)           Gaussian blur of radius r px inside m by normalized convolution of (alpha, alpha F)
  lip_finish(F, m, k, mu)       highlights above the part's mean lightness mu scaled by 1 + k (k < 0 matte, k > 0 gloss)
  stack(layers)                 ordered stacking, bottom to top: alpha = a2 + (1 - a2) a1, K = K2 + (1 - a2) K1
  blend(l1, l2, t)              linear blending alpha_t = (1 - t) a1 + t a2, K_t = (1 - t) K1 + t K2"""
import numpy as np
import cv2
from irismix.utils import rgb_to_lab, lab_to_rgb


def compose(at, F, T, strength=1.0):
    a = np.clip(at * strength, 0, 1)[..., None]
    return a * F + (1 - a) * T


def part_off(at, w):
    return at * (1 - w)


def recolour_to(F, m, target):
    """F (H, W, 3) [0, 1]; m bool (H, W) region; target CIELAB triple. The region's mean lightness is moved to the target
    L (shading kept), a* b* are set to the target."""
    L = rgb_to_lab(F); mu = L[m].mean(0)
    L2 = L.copy(); L2[..., 0] = L[..., 0] - mu[0] + target[0]; L2[..., 1] = target[1]; L2[..., 2] = target[2]
    return np.where(m[..., None], lab_to_rgb(L2), F)


def rotate_hue(F, m, deg):
    """rotate the a* b* chroma of F inside m by deg degrees (lightness and chroma magnitude kept)."""
    L = rgb_to_lab(F); t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    a, b = L[..., 1].copy(), L[..., 2].copy(); L[..., 1] = c * a - s * b; L[..., 2] = s * a + c * b
    return np.where(m[..., None], lab_to_rgb(L), F)


def select(at, w, s):
    """per-part strength: w (5, H, W) part weights (summing to 1), s 5 strengths (0 switches a part off)."""
    sel = np.tensordot(np.asarray(s, np.float32), w, axes=1)
    return np.clip(at * sel, 0, 1)


def soften(at, F, m, r):
    """blur the layer by a Gaussian of radius r px inside the soft region m (H, W) in [0, 1]; colour is extended by
    normalized convolution (blur alpha F and alpha, then divide), so edges soften without darkening."""
    if r <= 0: return at, F
    K = at[..., None] * F
    ab = cv2.GaussianBlur(at, (0, 0), r); Kb = cv2.GaussianBlur(K, (0, 0), r)
    Fb = np.clip(Kb / np.maximum(ab, 1e-4)[..., None], 0, 1)
    m = m.astype(np.float32)
    return (1 - m) * at + m * ab, np.where((m > 0)[..., None] & (ab > 1e-4)[..., None], (1 - m[..., None]) * F + m[..., None] * Fb, F)


def lip_finish(F, m, k, mu):
    """matte (k = -1) removes the highlights painted into the layer above its mean lightness mu, gloss (k > 0) amplifies them."""
    L = rgb_to_lab(F); h = np.maximum(L[..., 0] - mu, 0)
    L[..., 0] = np.clip(L[..., 0] + k * h, 0, 100)
    return np.where(m[..., None], lab_to_rgb(L), F)


def stack(layers):
    """[(alpha, F), ...] bottom to top -> combined (alpha, F)."""
    A = np.zeros(layers[0][0].shape, np.float32); K = np.zeros(layers[0][1].shape, np.float32)
    for a, F in layers:
        K = a[..., None] * F + (1 - a[..., None]) * K; A = a + (1 - a) * A
    return A, np.clip(K / np.maximum(A, 1e-6)[..., None], 0, 1)


def blend(l1, l2, t):
    """linear blending of two aligned layers on the same source (cross-fades appearance, not geometry)."""
    (a1, F1), (a2, F2) = l1, l2
    A = (1 - t) * a1 + t * a2; K = (1 - t) * a1[..., None] * F1 + t * a2[..., None] * F2
    return A, np.clip(K / np.maximum(A, 1e-6)[..., None], 0, 1)
