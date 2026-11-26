"""ComfyUI nodes for IrisMix.

Install: link this folder into ComfyUI/custom_nodes (it imports the irismix package from the repository root, one level
up), and install the repository requirements into ComfyUI's Python environment. See comfyui/README.md.

Nodes (category IrisMix):
  IrisMix Decompose          photo -> bare face, makeup layer (preview over a checkerboard), IRISMIX_LAYER
  IrisMix Compose by Part    target photo + optional references per part (eyes, lips, face, forehead) -> composite, IRISMIX_LOOK
  IrisMix Render             IRISMIX_LOOK -> rendered photo, IRISMIX_RESULT (the render as a layer over the target)
  IrisMix Edit Layer         IRISMIX_RESULT -> edited photo (strength, parts, lip color, hue per part; no new render)

To put makeup on a face that already wears some, connect the bare_face output of a Decompose node to the target input of
Compose. The decomposer and the renderer share one FLUX.2 [klein] base pipeline; the text encoder stays on the CPU."""
import gc, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # realpath: custom_nodes holds a link to this folder
if ROOT not in sys.path: sys.path.insert(0, ROOT)

PARTS = ('eyes', 'lips', 'face', 'forehead')
_MODELS = {}


def _u8(t):
    """ComfyUI IMAGE (first image of the batch) -> uint8 H x W x 3"""
    return (t[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def _img(a):
    a = a.astype(np.float32) / 255 if a.dtype == np.uint8 else a.astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(np.clip(a, 0, 1)))[None]


def _checker(a, F):
    h, w = a.shape[:2]; yy, xx = np.mgrid[:h, :w]; s = max(4, min(h, w) // 40)
    ck = np.where(((yy // s + xx // s) % 2)[..., None] == 0, .97, .87)
    return F * a[..., None] + ck * (1 - a[..., None])


def _dev(): return 'cuda' if torch.cuda.is_available() else 'cpu'


def _decomposer():
    if 'renderer' in _MODELS: del _MODELS['renderer']; gc.collect()
    if 'decomposer' not in _MODELS:
        from irismix import config, decomposer as DEC, rules
        rules.LMK_FALLBACK = True                                       # in-the-wild photos: low-confidence landmark retry
        _MODELS['decomposer'] = DEC.Decomposer(config.DECOMPOSER_CKPT, _dev(), base_model=config.BASE_MODEL,
                                               lora_init=config.LORA_INIT, cache_dir=config.CACHE_DIR)
    return _MODELS['decomposer']


def _renderer():
    if 'decomposer' in _MODELS: del _MODELS['decomposer']; gc.collect()
    if 'renderer' not in _MODELS:
        from irismix import config
        from irismix.renderer import Renderer
        rm = Renderer(_dev(), ckpt=config.RENDERER_CKPT, base_model=config.BASE_MODEL, lora_init=config.LORA_INIT); rm.skips.eval()
        _MODELS['renderer'] = rm
    return _MODELS['renderer']


def _hex_lab(h):
    import cv2
    h = h.strip().lstrip('#')
    if len(h) != 6: raise ValueError(f'IrisMix: color {h!r} is not #RRGGBB')
    rgb = np.array([[[int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]]], np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]


class IrisMixDecompose:
    """Split a photo into its bare face and its makeup layer (coverage and color) with the decomposer (one step)."""
    CATEGORY = 'IrisMix'
    RETURN_TYPES = ('IMAGE', 'IMAGE', 'IRISMIX_LAYER')
    RETURN_NAMES = ('bare_face', 'makeup_layer', 'layer')
    FUNCTION = 'run'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'photo': ('IMAGE',)}}

    def run(self, photo):
        from irismix import pipeline as PL
        G = _u8(photo)
        with torch.no_grad(): L = PL.decompose(_decomposer(), G)
        if 'error' in L: raise ValueError(f"IrisMix: {L['error']} (a tighter crop around the face usually helps)")
        return (_img(L['Br']), _img(_checker(L['a'], L['F'])), L)


class IrisMixCompose:
    """Place the parts of one or more decomposed references on the target face with the transfer rules (warp only)."""
    CATEGORY = 'IrisMix'
    RETURN_TYPES = ('IMAGE', 'IRISMIX_LOOK')
    RETURN_NAMES = ('composite', 'look')
    FUNCTION = 'run'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'target': ('IMAGE',)},
                'optional': {f'{p}_from': ('IRISMIX_LAYER',) for p in PARTS}}

    def run(self, target, **refs):
        from irismix import pipeline as PL
        S = _u8(target)
        groups = {}                                                     # one reference may supply several parts
        for p in PARTS:
            L = refs.get(f'{p}_from')
            if L is not None: groups.setdefault(id(L), [L, []])[1].append(p)
        if not groups: raise ValueError('IrisMix: connect at least one reference layer (eyes_from, lips_from, face_from or forehead_from)')
        rl = [dict(layer=L, parts=ps) for L, ps in groups.values()]
        C, av, _ = PL.paste(S, rl)
        return (_img(C), dict(S=S, refs=rl, C=C, av=av))


class IrisMixRender:
    """Render a composed look in one step (sigma 0.65 by default) and return it as a layer over the target."""
    CATEGORY = 'IrisMix'
    RETURN_TYPES = ('IMAGE', 'IRISMIX_RESULT')
    RETURN_NAMES = ('rendered', 'result')
    FUNCTION = 'run'

    @classmethod
    def INPUT_TYPES(cls):
        from irismix import config
        return {'required': {'look': ('IRISMIX_LOOK',), 'seed': ('INT', {'default': config.SEED, 'min': 0, 'max': 2 ** 31 - 1})}}

    def run(self, look, seed):
        from irismix import config, pipeline as PL, render as RD, layer_recover as LR
        S, C, av = look['S'], look['C'], look['av']
        it = PL.look_item(S, look['refs'], C, av)
        arr, info = RD.prepare(it)
        rm = _renderer()
        with torch.no_grad(), torch.autocast(torch.device(_dev()).type, dtype=torch.bfloat16): rm.text([it['text']])
        rm.p.text_encoder.to('cpu')
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        H5 = RD.render_one(rm, arr, it['text'], sigma=config.SIGMA, steps=config.STEPS, seed=seed)
        I, _ = RD.compose_full(S, C, H5, info, av)
        at, F = LR.layer_from_render(I, S, it['A'], av)
        return (_img(I), dict(S=S, at=at, F=F))


class IrisMixEditLayer:
    """Closed-form edits of the rendered layer; nothing is rendered again."""
    CATEGORY = 'IrisMix'
    RETURN_TYPES = ('IMAGE',)
    RETURN_NAMES = ('edited',)
    FUNCTION = 'run'

    @classmethod
    def INPUT_TYPES(cls):
        req = {'result': ('IRISMIX_RESULT',), 'strength': ('FLOAT', {'default': 1.0, 'min': 0.0, 'max': 1.5, 'step': 0.05})}
        for p in PARTS: req[p] = ('BOOLEAN', {'default': True})
        req['lip_color'] = ('STRING', {'default': '', 'tooltip': '#RRGGBB, empty keeps the rendered color'})
        for p in ('eyes', 'face', 'forehead'): req[f'{p}_hue'] = ('INT', {'default': 0, 'min': -180, 'max': 180, 'step': 5})
        return {'required': req}

    def run(self, result, strength, eyes, lips, face, forehead, lip_color, eyes_hue, face_hue, forehead_hue):
        from irismix import pipeline as PL
        S, at, F = result['S'], result['at'], result['F']
        off = [p for p, on in (('eyes', eyes), ('lips', lips), ('face', face), ('forehead', forehead)) if not on]
        rec = {'lips': _hex_lab(lip_color)} if lip_color.strip() else None
        hue = {p: d for p, d in (('eyes', eyes_hue), ('face', face_hue), ('forehead', forehead_hue)) if d}
        E, _, _ = PL.apply_edits(at, F, S, PL.part_masks(S), strength, off, rec, hue or None)
        return (_img(E),)


NODE_CLASS_MAPPINGS = {'IrisMixDecompose': IrisMixDecompose, 'IrisMixCompose': IrisMixCompose, 'IrisMixRender': IrisMixRender,
                       'IrisMixEditLayer': IrisMixEditLayer}
NODE_DISPLAY_NAME_MAPPINGS = {'IrisMixDecompose': 'IrisMix Decompose', 'IrisMixCompose': 'IrisMix Compose by Part',
                              'IrisMixRender': 'IrisMix Render', 'IrisMixEditLayer': 'IrisMix Edit Layer'}
