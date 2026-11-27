"""Occlusion-aware transfer: glasses or bangs are lifted off before transfer and put back on top afterwards.

  1. remove   the base FLUX.2-klein editor (no IrisMix LoRA; 4 steps, guidance 1.0) removes the occluder from the photo W;
              the edit is kept only inside the occluder region (parsed glasses / hair on the forehead, united with what the
              editor changed next to it, dilated and feathered), so every other pixel is the photo -> occluder-free face B
  2. transfer IrisMix renders the look on B -> R
  3. restore  the occluder's own pixels come back from the photo, I = t R + (1 - t) W, with the transmittance t built from the
              minimum-coverage occluder alpha (W against B), stacked three times, and face parsing of the photo:
              bangs    t = (1 - alpha_o)^3 (1 - P_hair): parsed hair stays opaque even where alpha_o is low; gaps between
                       strands, where the photo shows the forehead, show the makeup
              glasses  only the frames go back (outer band of the parsed glasses plus the temples); the eyes behind the
                       lenses come from R, because the lenses refract the eyes, which then differ between W and B.
Limitation: when face parsing misses part of a frame, that part is not restored (small gaps in the frame)."""
import numpy as np, cv2

from irismix import rules

PROMPTS = {
    'glasses': 'Remove her eyeglasses completely: no frames, no lenses, no nose pads. Show her bare eyes, eyelids and the '
               'bridge of her nose with natural skin where the frames were, with no frame marks, no lens reflections and no '
               'frame shadows. Keep everything else exactly the same: her face, skin, eyes, eyebrows, makeup, expression, '
               'head position, hair outside this area, clothes, background, lighting and framing.',
    'bangs': 'Push her fringe back off her face and tuck it smoothly into the rest of her hair, so her entire forehead is bare '
             'skin from the eyebrows up to the natural hairline, with no hair strands on the forehead. No clips, pins, bands '
             'or any hair accessory are visible. Keep exactly the same hair colour, hair length and the hair falling at the '
             'sides and back. Keep everything else exactly the same: her face, skin, eyes, eyebrows, makeup, expression, '
             'head position, hair outside this area, clothes, background, lighting and framing.'}
EYES = ([263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246])


def load_editor(dev='cuda', base_model=None, share=None):
    """the base FLUX.2-klein pipeline. share: a loaded pipeline of the same base model (e.g. the renderer's) whose
    text encoder, tokenizer and VAE are reused, so only the plain transformer is loaded again (about 9 GB less)."""
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
    from irismix import config
    from irismix.flux import _placement
    base = base_model or config.BASE_MODEL
    kw = dict(vae=share.vae) if share is not None else {}
    if share is not None and share.text_encoder is not None: kw.update(text_encoder=share.text_encoder, tokenizer=share.tokenizer)
    kw['transformer'] = Flux2Transformer2DModel.from_pretrained(base, subfolder='transformer', torch_dtype=torch.bfloat16,
                                                                **_placement(dev))   # read straight into GPU memory
    pipe = Flux2KleinPipeline.from_pretrained(base, torch_dtype=torch.bfloat16, **kw)
    pipe.transformer.to(dev); pipe.vae.to(dev); pipe.text_encoder.to(dev)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _forehead_band(lb, k=0.40):
    face = np.isin(lb, [1, 2, 4, 5, 6, 7, 10, 11, 12]); ys, xs = np.nonzero(face)
    band = np.zeros_like(face)
    if not len(ys): return band
    by = np.nonzero(np.isin(lb, [6, 7]))[0]; top, chin = ys.min(), ys.max()
    bt = int(np.percentile(by, 10)) if len(by) else int(top + 0.3 * (chin - top))
    fh = int(k * (chin - bt)); band[max(0, bt - fh):bt, max(0, xs.min() - 10):xs.max() + 10] = True
    return band


def remove(pipe, W8, cls, seed=0):
    """occluder-free face B (uint8) of the photo W8: the editor's output inside the occluder region, the photo elsewhere."""
    import torch
    from PIL import Image
    h, w = W8.shape[:2]
    pipe.text_encoder.to(pipe.transformer.device)            # a shared text encoder may have been parked on the CPU
    raw = pipe(image=Image.fromarray(W8), prompt=PROMPTS[cls], num_inference_steps=4, guidance_scale=1.0,
               width=(w // 16) * 16, height=(h // 16) * 16, generator=torch.Generator(pipe.device).manual_seed(seed)).images[0]
    E = np.asarray(raw.resize((w, h), Image.LANCZOS).convert('RGB'))
    la, le = rules.argmax0(rules.parse(W8)), rules.argmax0(rules.parse(E))
    if cls == 'glasses': R = (la == rules.GLASSES) | (le == rules.GLASSES)
    else: R = np.isin(la, rules.HAIR) & _forehead_band(la) | _forehead_band(la) & (np.abs(E.astype(int) - W8).max(-1) > 40)
    near = cv2.dilate(R.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    R = R | (near & (np.abs(E.astype(np.float32) - W8).mean(-1) > 40))
    R = cv2.dilate(R.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
    soft = cv2.GaussianBlur(R, (0, 0), 3.0)[..., None]
    return np.clip(soft * E + (1 - soft) * W8 + 0.5, 0, 255).astype(np.uint8)


def alpha_occ(W, B):
    """minimum-coverage occluder alpha (dead zone 0.06, slight blur) inside the region the removal changed."""
    d = W - B; e = 1e-4
    a = np.maximum(np.where(d > 0, d / np.maximum(1 - B, e), 0), np.where(d < 0, -d / np.maximum(B, e), 0)).max(-1)
    a = np.clip((a - 0.06) / 0.94, 0, 1); a = cv2.GaussianBlur(a, (0, 0), 0.6 * W.shape[0] / 1024)
    reg = (np.abs(d).max(-1) > 0.08).astype(np.uint8)
    reg = cv2.dilate(cv2.morphologyEx(reg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), np.ones((31, 31), np.uint8))
    return a * reg


def _eye_mask(img, grow=1.6):
    p = rules.landmarks(img); m = np.zeros(img.shape[:2], np.uint8)
    for idx in EYES:
        q = p[list(idx)][:, :2]; c = q.mean(0); cv2.fillConvexPoly(m, np.round(c + grow * (q - c)).astype(np.int32), 1)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 4)


def restore(W8, B8, R, cls):
    """W8 photo with the occluder, B8 its occluder-free face (uint8), R the render on B8 (float [0, 1]) -> float result."""
    W, B = W8.astype(np.float32) / 255, B8.astype(np.float32) / 255
    a = alpha_occ(W, B); pr = rules.parse(W8)
    if cls == 'bangs':
        t = (1 - a) ** 3 * (1 - np.clip(pr[rules.HAIR].sum(0), 0, 1) * (a > 0))
    else:
        gb = (rules.argmax0(pr) == rules.GLASSES).astype(np.uint8); fw = W8.shape[1]
        r = max(3, int(round(fw * 0.022))); ring = gb - cv2.erode(gb, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
        side = cv2.dilate(gb, np.ones((5, max(9, int(round(fw * 0.07)))), np.uint8)) * (1 - gb)
        g = cv2.GaussianBlur(np.clip(ring + side, 0, 1).astype(np.float32), (0, 0), 1.5)
        diff = cv2.GaussianBlur(np.abs(W - B).max(-1), (0, 0), 0.8)
        frame = np.maximum(np.clip((a - 0.12) / 0.20, 0, 1), np.clip((diff - 0.10) / 0.12, 0, 1))
        glare = cv2.GaussianBlur(((W - B).mean(-1) > 0.05).astype(np.float32) * gb, (0, 0), 1.5)  # glare would bite the makeup
        frame = frame * (1 - glare)
        t = (1 - np.maximum(a, frame) * g * frame * (1 - _eye_mask(W8))) ** 3
        sideg = cv2.GaussianBlur(side.astype(np.float32), (0, 0), 3)
        t = t * (1 - np.clip(pr[rules.HAIR].sum(0), 0, 1) * sideg)                               # hair in front of temples
    return np.clip(t[..., None] * R + (1 - t[..., None]) * W, 0, 1)
