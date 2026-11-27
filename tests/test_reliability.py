"""CPU tests of the failure paths (no GPU, no diffusion weights): a photo without a face, bad checkpoints, bad command-line
edits and bad viewer requests must end in a clear message, and the files a fresh clone needs must be in the package.
usage: python -m pytest tests/test_reliability.py -q"""
import http.client, json, os, subprocess, sys, threading, types, urllib.error, urllib.request

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('IRISMIX_SEG_DEVICE', 'cpu')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np                                       # noqa: E402
import pytest                                            # noqa: E402

BLANK = np.full((256, 256, 3), 128, np.uint8)


def test_shade_library_is_bundled():
    from irismix import shades as SH
    assert len(SH.load_shades()) > 0


def test_no_face_raises_a_message():
    from irismix import pipeline as PL, render_steps as RD, layer_recover as LR
    for call in (lambda: PL.design_transform(BLANK, dx=0.1), lambda: PL.part_masks(BLANK), lambda: PL.px_per_mm(BLANK),
                 lambda: LR.support_of(BLANK), lambda: RD.prepare(dict(S=BLANK))):
        with pytest.raises(ValueError, match='no face'): call()


def test_checkpoint_mismatch_raises():
    from irismix import flux
    pipe = types.SimpleNamespace(transformer=types.SimpleNamespace(named_parameters=lambda: []))
    with pytest.raises(ValueError, match='LoRA tensors'): flux.load_lora_ckpt(pipe, {})


@pytest.mark.parametrize('edit', (['--hue', 'nose=40'], ['--hue', 'eyes'], ['--recolour', 'lips=#zzzzzz']))
def test_infer_rejects_bad_edits_before_loading_models(edit, tmp_path):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'infer.py'), '--src', 'a.png', '--ref', 'b.png',
                        '--out', str(tmp_path), *edit], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2, r.stderr
    assert 'Traceback' not in r.stderr


@pytest.fixture
def viewer(tmp_path):
    from irismix.viewer.engine import Engine
    from irismix.viewer.server import serve
    app, httpd = serve(Engine(device='cpu'), '127.0.0.1', 0, str(tmp_path))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield app, httpd.server_address[1]
    httpd.shutdown(); httpd.server_close()


def _status(port, path, data=None):
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data)
    try:
        with urllib.request.urlopen(req) as r: return r.status
    except urllib.error.HTTPError as e:
        return e.code


@pytest.mark.parametrize('path, data, code', (
    ('/api/upload?slot=source', b'not an image', 400),
    ('/api/upload?slot=nose', b'', 400),
    ('/api/clear?slot=nose', b'{}', 400),
    ('/api/use_rec', b'{"i": 0}', 400),
    ('/api/soft', b'{"soft": {"lips": 1}}', 400),
    ('/api/keep', b'{}', 400),
    ('/api/compare', b'{"cat": "nose"}', 400),
    ('/api/shades', b'{}', 400),
    ('/api/run', b'not json', 400),
    ('/api/nope', b'{}', 404),
    ('/static/missing.js', None, 404),
    ('/examples/99999', None, 404),
    ('/examples/-1', None, 404),
    ('/examples/abc', None, 400)))
def test_viewer_bad_requests(viewer, path, data, code):
    assert _status(viewer[1], path, data) == code


def test_viewer_refuses_an_oversized_body(viewer):
    from irismix.viewer.server import MAX_UPLOAD
    c = http.client.HTTPConnection('127.0.0.1', viewer[1], timeout=30)
    c.putrequest('POST', '/api/upload?slot=source'); c.putheader('Content-Length', str(MAX_UPLOAD + 1)); c.endheaders()
    r = c.getresponse()
    assert r.status == 400 and 'larger' in json.loads(r.read())['error']
    c.close()


def test_only_one_background_job_at_a_time(viewer):
    app = viewer[0]; gate = threading.Event(); claims = []

    def claim():
        try:
            app._background(gate.wait); claims.append('ok')
        except ValueError:
            claims.append('busy')
    ts = [threading.Thread(target=claim) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    gate.set()
    assert sorted(claims) == ['busy'] * 7 + ['ok']


def teardown_module(module=None):
    from irismix import rules
    rules.close_models()
