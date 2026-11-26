"""CPU test of the viewer (no GPU, no diffusion weights): the HTTP API end to end with a warp-only look. The exact LayerBeauty
layer of the bundled reference is put into the engine's decomposition cache, so no decomposition is run; then upload, transfer,
the result images, and the exact export (checked against pipeline.apply_edits).
usage: python -m pytest tests/test_viewer.py -q"""
import io, json, os, sys, threading, time, urllib.request

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('IRISMIX_SEG_DEVICE', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'scripts'))
import numpy as np                                       # noqa: E402
from PIL import Image                                    # noqa: E402

DATA = os.path.join(HERE, 'data', 'layerbeauty')
REF = ('creative_scene', 'test', 'cs_006634')
SRC = ('daily_scene', 'test', 'ds_001545')


def _png(x):
    b = io.BytesIO(); Image.fromarray(x).save(b, 'PNG'); return b.getvalue()


def _req(base, path, data=None, raw=False):
    r = urllib.request.urlopen(urllib.request.Request(base + path, data=data, method='POST' if data is not None else 'GET'))
    body = r.read()
    return (body, r.headers) if raw else json.loads(body)


def test_layer_ops():
    from irismix import edits as ED
    rng = np.random.default_rng(1)
    a1, a2 = rng.uniform(0, 1, (2, 32, 32)).astype(np.float32); F1, F2 = rng.uniform(0, 1, (2, 32, 32, 3)).astype(np.float32)
    T = rng.uniform(0, 1, (32, 32, 3)).astype(np.float32)
    A, F = ED.stack([(a1, F1), (a2, F2)])                          # stacking = compositing one layer after the other
    assert np.abs(ED.compose(A, F, T) - ED.compose(a2, F2, ED.compose(a1, F1, T))).max() < 1e-4
    A, F = ED.blend((a1, F1), (a2, F2), 0.3)                       # blending cross-fades the two composites
    assert np.abs(ED.compose(A, F, T) - (0.7 * ED.compose(a1, F1, T) + 0.3 * ED.compose(a2, F2, T))).max() < 1e-4
    w = np.zeros((5, 32, 32), np.float32); w[0] = 1
    assert np.allclose(ED.select(a1, w, [0, 1, 1, 1, 1]), 0) and np.allclose(ED.select(a1, w, [1] * 5), a1)
    at, Fs = ED.soften(a1, F1, np.ones((32, 32), np.float32), 2.0)
    assert at.std() < a1.std() and Fs.shape == F1.shape


def test_viewer_api(tmp_path):
    import layerbeauty as LB
    from irismix import pipeline as PL
    from irismix.viewer.engine import Engine
    from irismix.viewer.server import serve
    X = LB.load(DATA, *REF); S = LB.load(DATA, *SRC)['B']
    eng = Engine(device='cpu')
    app, httpd = serve(eng, '127.0.0.1', 0, str(tmp_path), DATA)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{httpd.server_address[1]}'
    try:
        assert b'IrisMix' in _req(base, '/', raw=True)[0]
        assert len(_req(base, '/api/examples')) >= 4
        _req(base, '/api/upload?slot=source', _png(S))
        st = _req(base, '/api/upload?slot=whole', _png(X['I']))
        assert st['inputs']['source'] and st['inputs']['whole']
        h = eng.refs['whole'][0]                                   # decomposition cache: the exact layer of the reference
        eng.layers[h] = dict(a=X['a'], F=X['F'], w=X['w'], Br=X['B'], Im=X['I'])
        _req(base, '/api/run', json.dumps(dict(render=False)).encode())
        for _ in range(600):
            st = _req(base, '/api/state')
            if not st['busy']: break
            time.sleep(0.2)
        assert st['error'] is None and st['result']['width'] == S.shape[1], st
        for k in ('photo', 'target', 'paste', 'result', 'F', 'alpha', 'masks', 'masks2'):
            im = np.asarray(Image.open(io.BytesIO(_req(base, f'/api/img/{k}.png', raw=True)[0])))
            assert im.shape[:2] == S.shape[:2], k
        o = eng.out
        Sf = S.astype(np.float32) / 255                            # the warp-only layer re-composites the paste
        R = o['alpha'][..., None] * o['F'] + (1 - o['alpha'][..., None]) * Sf
        assert np.abs(R - o['paste'])[o['alpha'] > 0.02].max() < 2 / 255 and np.abs(R - o['paste']).max() < 6 / 255
        ed = dict(strength=0.6, part_strength=[0, 1, 1, 1, 1], recolour={'eyes': [55.0, 20.0, -30.0]}, hue={}, soft={}, finish={})
        body, hd = _req(base, '/api/export', json.dumps(dict(edits=ed)).encode(), raw=True)
        got = np.asarray(Image.open(io.BytesIO(body)))
        ref, _, _ = PL.apply_edits(o['alpha'], o['F'], S, o['w'], 0.6, ['lips'], {'eyes': [55.0, 20.0, -30.0]}, {})
        assert np.abs(got.astype(int) - np.round(ref * 255)).max() <= 1
        d = hd['X-Saved-To']
        assert all(os.path.exists(os.path.join(d, f)) for f in ('edited.png', 'result.png', 'layer.npz', 'edits.json'))
        # keep two looks, stack and blend them; softness; shade analysis and try-on
        _req(base, '/api/keep', json.dumps(dict(edits=dict(strength=1.0))).encode())
        st = _req(base, '/api/keep', json.dumps(dict(edits=dict(strength=0.5, hue={'lips': 90}))).encode())
        assert [l['id'] for l in st['looks']] == [1, 2]
        st = _req(base, '/api/stack', json.dumps(dict(order=[1, 2], visible={}, strength={}, blend=None)).encode())
        assert st['flags']['stack']
        st = _req(base, '/api/stack', json.dumps(dict(order=[1, 2], blend=dict(a=1, b=2, t=0.5))).encode())
        assert st['flags']['stack']
        v = st['versions']['layer']
        st = _req(base, '/api/soft', json.dumps(dict(soft={'eyes': 1.5})).encode())
        assert st['versions']['layer'] == v + 1
        info = _req(base, '/api/shades')
        assert info['recommended']['foundations'] and len(info['recommended']['lipsticks']) == 3
        st = _req(base, '/api/shades', json.dumps(dict(foundation=info['recommended']['foundations'][0], blush=None,
                                                       lipstick=info['recommended']['lipsticks']['everyday'])).encode())
        assert st['flags']['under'] and st['flags']['lip']
        for k in ('stackF', 'stackA', 'underF', 'underA', 'lipF', 'lipA', 'look_1'):
            _req(base, f'/api/img/{k}.png', raw=True)
        # shade comparison cards and look recommendation (the user's reference; its layer is cached)
        for cat in ('foundation', 'lipstick', 'blush'):
            d = _req(base, '/api/compare', json.dumps(dict(cat=cat, edits=dict(strength=1.0))).encode())
            assert len(d['cards']) >= 3 and sum(c['best'] for c in d['cards']) == 1 and d['overall']
            assert d['cards'][0]['face'].startswith('data:image/jpeg;base64,') and d['cards'][0]['notes']
        _req(base, '/api/recommend', json.dumps(dict(which='mine')).encode())
        for _ in range(1200):
            st = _req(base, '/api/state')
            if not st['busy']: break
            time.sleep(0.25)
        assert st['error'] is None and st['rec'] and 0 <= st['rec'][0]['score'] <= 100, st.get('error')
        _req(base, f"/api/img/rec_{st['rec'][0]['i']}.png", raw=True)
        # design repositioning: mirrored and shifted face paint changes the paste
        C0 = eng.out['paste'].copy()
        _req(base, '/api/options', json.dumps(dict(xform=dict(dx=0.05, mirror=True))).encode())
        _req(base, '/api/run', json.dumps(dict(render=False)).encode())
        for _ in range(600):
            st = _req(base, '/api/state')
            if not st['busy']: break
            time.sleep(0.2)
        assert st['error'] is None and np.abs(eng.out['paste'] - C0).max() > 0.05
    finally:
        httpd.shutdown(); httpd.server_close()
        from irismix import rules
        rules.close_models()
