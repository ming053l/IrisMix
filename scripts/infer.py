"""IrisMix inference: source photo + per-part reference photos -> made-up photo and its makeup layer.

  # whole look from one reference
  python scripts/infer.py --src source.jpg --ref reference.jpg --out out/
  # compose a look from several references (each optional; one reference may supply several parts)
  python scripts/infer.py --src source.jpg --eyes ref_a.jpg --lips ref_b.jpg --face ref_c.jpg --forehead ref_c.jpg --out out/
  # closed-form edits of the recovered layer
  python scripts/infer.py ... --strength 0.6 --recolour lips=#b0303a --hue eyes=40 --off forehead

Outputs in --out: result.png (rendered result), paste.png (warp-only composite), layer.npz (alpha~, F' float16) and
layer.png (RGBA), edited.png when an edit is given. The decomposer and the renderer share one base pipeline (the weights are
read once, straight into GPU memory), so a GPU with about 16 GB free is enough. Sampler: config.SIGMA / STEPS / OMEGA_R (0.65, one Euler step, 2 px)."""
import argparse, gc, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from irismix import config                                    # noqa: E402
config.limit_cpu_threads()                                    # before numpy / torch start their thread pools
import numpy as np                                            # noqa: E402
from PIL import Image                                         # noqa: E402
from irismix import rules                                     # noqa: E402

PARTS = ('eyes', 'lips', 'face', 'forehead')


def read_rgb(p): return np.asarray(Image.open(p).convert('RGB'))


def hex_to_lab(h):
    import cv2
    h = h.lstrip('#'); rgb = np.array([[[int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]]], np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]


def part_edits(items, value):
    """['PART=VALUE', ...] -> {part: value(VALUE)}; raises ValueError with the offending argument."""
    out = {}
    for x in items:
        p, sep, v = x.partition('=')
        if not sep or p not in rules.PART_NAMES: raise ValueError(f'{x!r}: expected PART=VALUE with PART in {rules.PART_NAMES}')
        out[p] = value(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', required=True, help='source photo (the face to make up)')
    ap.add_argument('--ref', help='one reference for the whole look (cannot be combined with per-part references)')
    for p in PARTS: ap.add_argument(f'--{p}', help=f'reference photo for the {p}')
    ap.add_argument('--out', required=True)
    ap.add_argument('--decomposer', default=config.DECOMPOSER_CKPT); ap.add_argument('--renderer', default=config.RENDERER_CKPT)
    ap.add_argument('--lora_init', default=config.LORA_INIT); ap.add_argument('--base_model', default=config.BASE_MODEL)
    ap.add_argument('--cache_dir', default=config.CACHE_DIR)
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--sigma', type=float, default=config.SIGMA); ap.add_argument('--steps', type=int, default=config.STEPS)
    ap.add_argument('--warp_only', action='store_true', help='stop after the transfer rules (no rendering)')
    ap.add_argument('--strength', type=float, default=1.0, help='edit: layer strength s (out = s a F + (1 - s a) S)')
    ap.add_argument('--recolour', action='append', default=[], help='edit: PART=#RRGGBB, recolour a part of the layer')
    ap.add_argument('--hue', action='append', default=[], help='edit: PART=DEG, rotate the hue of a part of the layer')
    ap.add_argument('--off', action='append', default=[], choices=['lips', 'eyes', 'brows', 'face', 'forehead'], help='edit: switch a part off')
    a = ap.parse_args()
    per_part = {p: getattr(a, p) for p in PARTS if getattr(a, p)}
    if bool(a.ref) == bool(per_part): ap.error('give either --ref or at least one of --eyes / --lips / --face / --forehead')
    try:                                                      # checked before any model is loaded
        rec, hue = part_edits(a.recolour, hex_to_lab), part_edits(a.hue, float)
    except ValueError as e:
        ap.error(str(e))
    os.makedirs(a.out, exist_ok=True)
    import torch
    from irismix import decomposer as DEC, flux, pipeline as PL, render_steps as RD, layer_recover as LR
    from irismix.utils import to_uint8
    rules.LMK_FALLBACK = True                                    # in-the-wild photos: low-confidence landmark retry
    S = read_rgb(a.src)
    # group parts by reference file: one decomposition per reference, a reference may supply several parts
    groups = {a.ref: None} if a.ref else {}
    for p, f in per_part.items(): groups.setdefault(f, []).append(p)

    # ---- decomposer: decompose every reference (meanwhile the kernel reads the renderer's text encoder from disk)
    if not a.warp_only: flux.prefetch(a.base_model, 'text_encoder')
    s1 = DEC.Decomposer(a.decomposer, a.device, base_model=a.base_model, lora_init=a.lora_init, cache_dir=a.cache_dir)
    refs = []
    for f, ps in groups.items():
        L = PL.decompose(s1, read_rgb(f))
        if 'error' in L: sys.exit(f'{f}: {L["error"]}')
        refs.append(dict(layer=L, parts=ps))
    del s1; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- transfer rules + warp (CPU): warp-only composite
    try:
        C, av, _ = PL.paste(S, refs)
    except ValueError as e:
        sys.exit(f'{a.src}: {e}')
    Image.fromarray(to_uint8(C)).save(f"{a.out}/paste.png", compress_level=1)
    if a.warp_only: print('warp-only composite ->', f'{a.out}/paste.png'); rules.close_models(); return

    # ---- renderer: one-step render inside Omega, masked residual paste-back
    it = PL.look_item(S, refs, C, av)
    arr, info = RD.prepare(it)
    from irismix.renderer import Renderer
    rm = Renderer(a.device, ckpt=a.renderer, base_model=a.base_model, lora_init=a.lora_init); rm.skips.eval()
    with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16): rm.text([it['text']])   # cached embedding
    rm.p.text_encoder = None; gc.collect()                      # caption encoded (or cached): free the 4B encoder if loaded
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    H5 = RD.render_one(rm, arr, it['text'], sigma=a.sigma, steps=a.steps, seed=a.seed)
    I, _ = RD.compose_full(S, C, H5, info, av)
    Image.fromarray(to_uint8(I)).save(f'{a.out}/result.png', compress_level=1)   # lossless; level 1 writes ~4x faster

    # ---- closed-form layer of the render
    at, F = LR.layer_from_render(I, S, it['A'], av)
    LR.save(f'{a.out}/layer.npz', at, F)
    Image.fromarray(np.dstack([to_uint8(F), to_uint8(at)])).save(f'{a.out}/layer.png', compress_level=1)
    print('caption:', it['text']); print('saved', f'{a.out}/result.png', f'{a.out}/layer.npz')

    # ---- optional edits (plain alpha-over of the edited layer on the source)
    if a.strength != 1.0 or a.recolour or a.hue or a.off:
        E, _, _ = PL.apply_edits(at, F, S, PL.part_masks(S), a.strength, a.off, rec, hue)
        Image.fromarray(to_uint8(E)).save(f'{a.out}/edited.png', compress_level=1); print('saved', f'{a.out}/edited.png')
    rules.close_models()


if __name__ == '__main__':
    main()
