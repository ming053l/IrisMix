"""GPU end-to-end test of the viewer engine (skipped without CUDA or the IrisMix weights): full renders with the bundled
examples, the cached-reference path, design repositioning, occlusion-aware transfer (bangs), re-make-up with kept parts, look
recommendation and, when IRISMIX_TEST_VLM points to a VLM folder, the stylist. IRISMIX_TEST_GLASSES may point to a photo with
glasses to cover the glasses path too.
usage: python -m pytest tests/test_viewer_gpu.py -q -s      (fits a 24 GB GPU, a few minutes)"""
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np                                       # noqa: E402
import pytest                                            # noqa: E402
from PIL import Image                                    # noqa: E402

torch = pytest.importorskip('torch')
from irismix import config                               # noqa: E402

EX = os.path.join(config.REPO, 'assets', 'examples')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(config.RENDERER_CKPT),
                                reason='needs CUDA and the IrisMix weights')


def load(name): return np.asarray(Image.open(os.path.join(EX, name)).convert('RGB'))


@pytest.fixture(scope='module')
def eng():
    from irismix.viewer.engine import Engine
    return Engine('cuda', keep_models=False)


def run(e, render=True):
    t = time.time(); o = e.run(render=render)
    print(f"  {o['timings']} total {time.time() - t:.1f} s")
    assert o['result'].shape == o['target'].shape and np.isfinite(o['alpha']).all()
    return o


def test_render_and_cache(eng):
    eng.set_source(load('source-studio.webp')); eng.set_ref('whole', load('reference-satin.webp'))
    o = run(eng)
    assert o['rendered'] and o['alpha'].max() > 0.3
    o = run(eng)
    assert o['timings']['decompose'] < 1.0                   # the reference is decomposed once


def test_reposition_and_bangs(eng):
    eng.set_ref('whole', load('reference-lily.webp'))
    eng.set_options(xform=dict(dx=0.0, dy=-0.05, scale=1.0, mirror=True)); a = run(eng)['alpha']
    eng.set_options(xform=dict(dx=0.0, dy=0.0, scale=1.0, mirror=False)); b = run(eng)['alpha']
    assert np.abs(a - b).mean() > 1e-3                    # the mirrored design lands elsewhere
    eng.set_ref('whole', load('reference-merlot.webp')); eng.set_options(target='bangs')
    o = run(eng)
    assert 'restore' in o['timings']
    eng.set_options(target='as_is')


def test_remake_keep_parts(eng):
    eng.set_source(load('reference-merlot.webp')); eng.refs.clear(); eng.set_ref('lips', load('reference-satin.webp'))
    eng.set_options(target='remake', keep=['eyes'])
    o = run(eng)
    assert o['bare'] is not None
    eng.set_options(target='as_is', keep=[])


@pytest.mark.skipif(not os.environ.get('IRISMIX_TEST_GLASSES'), reason='set IRISMIX_TEST_GLASSES to a photo with glasses')
def test_glasses(eng):
    eng.set_source(np.asarray(Image.open(os.environ['IRISMIX_TEST_GLASSES']).convert('RGB')))
    eng.refs.clear(); eng.set_ref('whole', load('reference-merlot.webp')); eng.set_options(target='glasses')
    assert 'restore' in run(eng)['timings']
    eng.set_options(target='as_is')


def test_recommend_and_stylist(eng, tmp_path):
    from irismix.viewer.server import App
    eng.set_source(load('source-studio.webp')); eng.refs.clear()
    vlm = os.environ.get('IRISMIX_TEST_VLM'); stylist = None
    if vlm:
        from irismix.viewer.vlm import Stylist
        stylist = Stylist(vlm, os.environ.get('IRISMIX_TEST_VLM_PYTHON'))
    app = App(eng, str(tmp_path), None, stylist)
    app.examples = [e for e in app.examples if e['kind'] == 'reference'][:6]

    def wait():
        while eng.status['busy']: time.sleep(0.5)
        assert eng.status['error'] is None, eng.status['error']
    app.recommend(dict(which='examples')); wait()
    ranked = [r for r in app.rec if 'score' in r]
    assert len(ranked) >= 4 and ranked[0]['score'] >= ranked[-1]['score'] and ranked[0]['reasons']
    print('  ranking', [(r['name'], round(r['score'])) for r in ranked])
    if stylist:
        app.ask_stylist(dict(n=4, lang='en')); wait()
        assert app.vlm and all(v['reason'] for v in app.vlm)
        print('  stylist', app.vlm); stylist.close()
