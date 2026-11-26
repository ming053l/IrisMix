"""CPU smoke test (no GPU, no diffusion weights): imports, layer algebra, the LayerBeauty loader, and the CPU part of the
inference path on a bundled LayerBeauty pair -- transfer rules + warp (pipeline.paste), renderer conditions
(pipeline.look_item, render.prepare), the masked-residual paste-back with an identity render (render.compose_full) and the
closed-form layer recovery (layer_recover.layer_from_render). The exact LayerBeauty layer of the reference stands in for the
decomposition.
Needs the face parser (Hugging Face hub, downloaded once) and the MediaPipe landmarker (downloaded into the cache, or
$IRISMIX_LANDMARKER). About 1-2 minutes on a CPU.
usage: python -m pytest tests/test_smoke.py -q      or      python tests/test_smoke.py"""
import os, sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''                 # CPU only
os.environ.setdefault('IRISMIX_SEG_DEVICE', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
import numpy as np                                       # noqa: E402

DATA = os.path.join(HERE, 'data', 'layerbeauty')
REF = ('creative_scene', 'test', 'cs_006634')           # green eye blocks + green lips
SRC = ('daily_scene', 'test', 'ds_001545')                      # the bare face of this item is the source


def test_imports():
    import importlib
    for m in ('config', 'utils', 'face_parts', 'rules', 'edit_region', 'conditions', 'captions', 'render',
              'layer_recover', 'edits', 'pipeline', 'decomposer', 'flux', 'renderer', 'viewer.engine', 'viewer.server', 'viewer.cli'):
        importlib.import_module(f'irismix.{m}')


def test_layer_algebra():
    from irismix import edits as ED, layer_recover as LR
    rng = np.random.default_rng(0)
    T = rng.uniform(0.05, 0.95, (64, 64, 3)).astype(np.float32)
    a = rng.uniform(0, 1, (64, 64)).astype(np.float32); F = rng.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    I = ED.compose(a, F, T)
    assert np.all(LR.alpha_needed(I, T) <= a + 1e-5)              # the closed-form bound never exceeds the true alpha
    at, F2 = LR.recover(I, T, a)
    ok = at > LR.EPS
    assert np.abs(ED.compose(at, F2, T) - I)[ok].max() < 1e-4     # exact re-composite where alpha~ > eps
    assert np.allclose(ED.compose(a, F, T, 0.0), T)               # strength 0 = source
    assert np.allclose(ED.part_off(a, np.ones_like(a)), 0)


def test_loader_recompose():
    import layerbeauty as LB
    for subset, split, iid in (REF, SRC):
        X = LB.load(DATA, subset, split, iid)
        I = np.floor(np.clip(X['a'][..., None] * X['F'] + (1 - X['a'][..., None]) * X['B'] / 255., 0, 1) * 255 + .5)
        assert np.abs(I - X['I']).max() <= 2, iid                 # creative: bare stored quantised (<= 2 / 255)
        assert X['w'].shape == (5,) + X['B'].shape[:2]


def _look(parts):
    import layerbeauty as LB
    from irismix import pipeline as PL, rules
    rules.LMK_FALLBACK = True
    X = LB.load(DATA, *REF)
    S = LB.load(DATA, *SRC)['B']
    refs = [dict(layer=dict(a=X['a'], F=X['F'], w=X['w'], Br=X['B'], Im=X['I']), parts=parts)]
    C, av, _ = PL.paste(S, refs)
    return S, refs, C, av


def test_cpu_inference_path():
    from irismix import pipeline as PL, render as RD, layer_recover as LR
    from irismix.utils import u8
    S, refs, C, av = _look(['eyes', 'lips'])
    Sf = S.astype(np.float32) / 255
    assert av.max() > 0.5 and (av > 0.05).mean() > 0.002          # the eye blocks and lips were pasted
    assert np.abs(C - Sf)[av == 0].max() < 1e-4                    # alpha-over leaves the rest of the source untouched
    it = PL.look_item(S, refs, C, av)
    assert it['parts'] == ['eyes', 'lips'] and it['text'].strip()
    arr, info = RD.prepare(it)
    assert arr['C'].shape == (512, 512, 3) and arr['S1'].shape == (256, 256, 3) and arr['lip'].shape == (256, 256, 3)
    om = info['om']
    assert 0 < (om > 0.5).mean() < 0.5                             # Omega: the design neighbourhood, not the face
    I, m = RD.compose_full(S, C, arr['C'], info, av)               # identity render (H = C)
    assert np.array_equal(u8(I)[m == 0], S[m == 0])                 # outside Omega: the source exactly
    assert np.abs(I - ((1 - m[..., None]) * Sf + m[..., None] * C)).max() < 1e-5   # zero residual: blend of S and C
    at, F = LR.layer_from_render(I, S, it['A'], av)
    R = at[..., None] * F + (1 - at[..., None]) * Sf
    assert np.abs(R - I)[at > LR.EPS].max() < 2 / 255             # the recovered layer re-composites the render
    assert np.abs(R - I).max() < 6 / 255


def test_single_reference_item():
    from irismix import pipeline as PL, render as RD
    S, refs, C, av = _look(None)
    it = PL.look_item(S, refs, C, av)
    assert it['parts'] is None
    arr, info = RD.prepare(it)
    assert np.abs(arr['lip'] - 0.5).max() > 0.05                   # lip crop of the reference layer (green lips)


def teardown_module(module=None):
    from irismix import rules
    rules.close_models()


if __name__ == '__main__':
    import time
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            t0 = time.time(); fn(); print(f'{name}: ok ({time.time() - t0:.1f} s)', flush=True)
    teardown_module()
    print('all smoke tests passed')
