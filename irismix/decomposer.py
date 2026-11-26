"""Decomposer: decomposition of a reference photo into (bare face, makeup layer) with one network evaluation.

FLUX.2-klein-4B + the decomposer LoRA (flux.OneStep, task 'E'): the reference is cropped to a face-centred square (1.9 x the
MediaPipe landmark extent), resized to 512 and mapped in ONE step to the canvas [bare | layer on white | layer on black].
Closed form: alpha = 1 - mean_c(W - K), F = K / alpha. The layer, the predicted bare and 5 landmark part weights
(face_parts.rule_parts on the predicted bare) are pasted back into the reference's own frame (alpha = 0 outside the crop;
the bare there is the reference photo itself).
Landmarks of in-the-wild photos: when the default MediaPipe face landmarker (confidence 0.5) finds no face, the crop and the
predicted bare retry with confidence 0.1."""
import os
import numpy as np, cv2

from irismix import config
from irismix import rules, face_parts as FP
from irismix.utils import R, face_box, from_t


def image_face_box(img):
    """square face crop (x0, y0, side) of an image from its MediaPipe landmarks (low-confidence retry), or None."""
    p = rules.landmarks(img, fallback=True)
    return None if p is None else face_box(p, img.shape)


def closed_form_alpha(W, K):
    """layer alpha from the layer over white and over black: alpha = 1 - mean_c(W - K), in [0, 1] (F = K / alpha)."""
    return np.clip(1 - (W - K).mean(-1), 0, 1)


class Decomposer:
    def __init__(self, ckpt=None, dev='cuda', base_model=None, lora_init=None, cache_dir=None, skip_text_encoder=False):
        import torch
        from irismix import flux
        self.torch, self.dev = torch, dev
        cache = os.path.join(cache_dir or config.CACHE_DIR, 'decomposer_prompt_embeds.pt')
        pipe = flux.load_pipeline(dev, base_model=base_model, lora_init=lora_init,
                                  skip_text_encoder=skip_text_encoder and os.path.exists(cache))   # only with cached embeddings
        # prompt embeddings: cached on disk, so the 4B text encoder never has to sit on the GPU
        if os.path.exists(cache): emb = torch.load(cache, weights_only=True)
        else:
            pipe.text_encoder.to(dev)
            with torch.no_grad():
                emb = {k: tuple(x.cpu() for x in pipe.encode_prompt(prompt=v, device=dev)) for k, v in (('E', flux.P_E), ('H', flux.P_H))}
            pipe.text_encoder.to('cpu')
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            part = f'{cache}.{os.getpid()}.part'; torch.save(emb, part); os.replace(part, cache)   # never a truncated cache
        self.one_step = flux.OneStep(pipe, dev, emb={k: tuple(x.to(dev) for x in v) for k, v in emb.items()})
        sd = flux.load_state(ckpt or config.DECOMPOSER_CKPT)
        flux.load_lora_ckpt(pipe, sd, self.one_step.skips); pipe._irismix_lora = self
        self.lora = flux.lora_state(pipe, sd, dev)              # kept: the base pipeline is shared with the renderer
        self.one_step.skips.eval()

    def canvas(self, x512, seed=0):
        """x512 uint8 512x512x3 -> (B, W, K) float [0,1] 512x512x3 each."""
        from irismix import flux
        torch = self.torch
        flux.use_lora(self.one_step.p, self, self.lora)
        t = flux.to_t(x512.astype(np.float32) / 255)[None].to(self.dev)
        torch.manual_seed(seed)
        with torch.no_grad(), torch.autocast(torch.device(self.dev).type, dtype=torch.bfloat16):
            c = self.one_step.gen([t], 'E', R).float()
        c = from_t(c)
        return c[:, :R], c[:, R:2 * R], c[:, 2 * R:]


def predict_layer(s1, ref):
    """Decomposition of a full reference photo (uint8). Returns dict(a, F, w, Br, box, canvas, a512) in the reference frame."""
    box = image_face_box(ref)
    if box is None: return dict(error='no face in reference')
    X0, Y0, side = box
    crop = cv2.resize(ref[Y0:Y0 + side, X0:X0 + side], (R, R), interpolation=cv2.INTER_AREA if side > R else cv2.INTER_CUBIC)
    B, W, K = s1.canvas(crop)
    a = closed_form_alpha(W, K)
    pb = rules.landmarks((B * 255).round().astype(np.uint8), fallback=True)
    if pb is None: pb = rules.landmarks(crop, fallback=True)
    w = FP.rule_parts(pb) if pb is not None else np.stack([np.zeros((R, R))] * 3 + [np.ones((R, R)), np.zeros((R, R))]).astype(np.float32)
    up = lambda x: cv2.resize(x.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR if side > R else cv2.INTER_AREA)
    H, Wd = ref.shape[:2]
    af = np.zeros((H, Wd), np.float32); Pf = np.zeros((H, Wd, 3), np.float32)
    wf = np.zeros((5, H, Wd), np.float32); wf[3] = 1                    # outside the crop: 'face' (the remainder part)
    Br = ref.astype(np.float32) / 255
    af[Y0:Y0 + side, X0:X0 + side] = up(a); Pf[Y0:Y0 + side, X0:X0 + side] = up(K)
    Br[Y0:Y0 + side, X0:X0 + side] = up(B)
    for k in range(5): wf[k, Y0:Y0 + side, X0:X0 + side] = up(w[k])
    wf /= np.maximum(wf.sum(0, keepdims=True), 1e-6)
    af = np.clip(af, 0, 1); Ff = np.clip(Pf / np.maximum(af, 1e-3)[..., None], 0, 1)
    return dict(a=af, F=Ff, w=wf, Br=(Br * 255).round().astype(np.uint8), box=box, canvas=(B, W, K), a512=a)
