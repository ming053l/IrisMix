"""Closed-form layer recovery after rendering (the paper's rendering section).

  alpha~ = max(A_D * support, av, alpha_I)
         A_D      warped full reference coverage cut to its design mask (edit_region.design_mask, as the K / A conditions),
                  restricted to the source's paintable skin (support_and_hair support)
         av       pasted coverage (already limited by the transfer rules)
         alpha_I  the smallest coverage that explains the render over the source with a pigment in [0, 1] (the
                  triangulation-matting bound, per channel: (I - T) / (1 - T) where I > T, (T - I) / T where I < T, max over
                  RGB). Needed because the renderer restores design where the rules removed it and darkens / brightens
                  beyond max(A_D, av) (without it F' would clip on a large share of the pixels). alpha_I = 0 wherever
                  I_hat = T, i.e. outside Omega.
  F'     = clip((I_hat - (1 - alpha~) T) / alpha~, 0, 1)  where alpha~ > EPS, else F' = 0 and alpha~ = 0 (so the saved
           layer composites to T there instead of darkening it; the render differs from T there by at most EPS * 255)
Edits are then plain alpha-over: out = s alpha~ F'' + (1 - s alpha~) T (strength s, recolored F'', parts switched off by
multiplying alpha~ with 1 - part weight). Full frame, float [0, 1].
EPS = 0.02: re-compositing is exact (0 / 255) where alpha~ > EPS and within about 5 / 255 everywhere; a larger EPS drops
more pixels, a smaller one lets F' clip where 8-bit noise is divided by a tiny alpha~."""
import numpy as np
from irismix import rules, conditions as CM, edit_region as ER
from irismix.utils import face_box

EPS = 0.02


def support_of(S, p3=None):
    """paintable skin of the source S (uint8 full frame): the hair / occluder rule of the transfer pipeline."""
    if p3 is None: p3 = rules.require_landmarks_3d(S)
    prob = rules.parse(S)
    sup, hair = CM.support_and_hair(prob, p3[:, :2])
    return np.clip(sup * (1 - hair), 0, 1).astype(np.float32), p3


def alpha_tilde(A, av, support, I_hat=None, T=None, scale=1.0):
    D = ER.design_mask(A, scale=scale).astype(np.float32)
    at = np.maximum(A * D * support, av)
    if I_hat is not None: at = np.maximum(at, alpha_needed(I_hat, T))
    return np.clip(at, 0, 1).astype(np.float32)


def alpha_needed(I_hat, T):
    d = I_hat - T
    need = np.where(d > 0, d / np.maximum(1 - T, 1e-3), np.where(d < 0, -d / np.maximum(T, 1e-3), 0.0))
    return np.clip(need.max(-1), 0, 1).astype(np.float32)


def recover(I_hat, T, at, eps=EPS):
    """I_hat, T (H, W, 3) [0, 1]; at (H, W) alpha~. -> (alpha~ zeroed where <= eps, F' (H, W, 3) [0, 1])."""
    ok = at > eps
    F = np.where(ok[..., None], (I_hat - (1 - at[..., None]) * T) / np.maximum(at, eps)[..., None], 0.0)
    return (at * ok).astype(np.float32), np.clip(F, 0, 1).astype(np.float32)


def layer_from_render(I_hat, S, A, av, eps=EPS):
    """full pipeline for one render: I_hat (H, W, 3) [0, 1] or uint8, S source uint8, A warped full reference coverage,
    av pasted coverage (full frame). -> (alpha~, F')."""
    f = lambda x: x.astype(np.float32) / 255 if x.dtype == np.uint8 else x.astype(np.float32)
    I_hat, T = f(I_hat), f(S)
    sup, p3 = support_of(S)
    sc = face_box(p3, S.shape)[2] / 512
    at = alpha_tilde(f(A) if A.dtype == np.uint8 else A.astype(np.float32), av.astype(np.float32), sup, I_hat, T, scale=sc)
    return recover(I_hat, T, at, eps)


def save(path, at, F):
    """npz with float16 alpha~ (H, W) and F' (H, W, 3) in [0, 1] (F' = 0 and alpha~ = 0 where alpha~ <= eps). float16, not
    uint8: 8-bit alpha~ and F' re-composite with up to 4 / 255 error, float16 with <= 1 / 255."""
    np.savez_compressed(path, alpha=at.astype(np.float16), F=F.astype(np.float16), eps=EPS)
