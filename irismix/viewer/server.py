"""HTTP server of the IrisMix viewer (standard library only): a single-page web app plus a small JSON / PNG API.

  GET  /                          the web app (static/index.html; static/i18n.js holds its English, Chinese, Japanese and
                                  Korean strings and the guided tour)
  GET  /api/state                 inputs, options, progress, result metadata, kept looks
  GET  /api/examples              example photos: sources and references (name, url)
  GET  /api/img/<name>.png        photo | target | paste | result | F | alpha | masks (lips, eyes, brows) | masks2 (face,
                                  forehead) | stackF | stackA | underF | underA | lipF | lipA | thumb_<slot> | look_<id>
                                  (RGB / gray PNGs only: browsers premultiply an alpha channel)
  GET  /api/shades                skin / lip analysis of the target, recommended shades and the shade library
  POST /api/upload?slot=<slot>    body = image file; slot = source | whole | eyes | lips | face | forehead
  POST /api/clear?slot=<slot>     remove a reference
  POST /api/options               {"target": as_is | remake | glasses | bangs, "keep": [parts], "xform": {dx, dy, scale, mirror}}
  POST /api/run                   {"render": true | false}; runs in the background, poll /api/state
  POST /api/soft                  {"soft": {part: mm}}: softened working layer (new F / alpha)
  POST /api/keep                  {"edits": {...}, "name": str}: keep the edited working layer as a look
  POST /api/stack                 {"order", "visible", "strength", "blend"}: combined layer of the kept looks
  POST /api/shades                {"foundation", "blush", "lipstick", "strength", "finish"}: shade layers
  POST /api/recommend             {"which": "examples" | "mine" | "all"}: score candidate looks on the target (background)
  POST /api/stylist               ask the optional VLM stylist about the top recommended looks (background; --vlm)
  POST /api/compare               {"cat": foundation | lipstick | blush, "edits", "shades"}: shade comparison cards
  POST /api/free                  unload the networks and drop the caches (memory shown in /api/state "mem")
  POST /api/export                {"edits", "stack", "shades", "show_work"}: exact (numpy) composite, saved with the render,
                                  its layer and the parameters into the output directory; returns the PNG
The browser applies the closed-form edits on the GPU (WebGL) for interactive feedback; export recomputes them exactly."""
import base64, io, json, mimetypes, os, tempfile, threading, time, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from irismix import config, layer_recover as LR, edits as ED
from irismix.utils import u8
from irismix.viewer.engine import SLOTS

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
EXAMPLES = os.path.join(config.REPO, 'assets', 'examples')
MAX_SIDE = 2048
MAX_UPLOAD = 64 * 2 ** 20      # bytes
IMG_EXT = ('.png', '.jpg', '.jpeg', '.webp')


def png(x):
    b = io.BytesIO(); Image.fromarray(x).save(b, 'PNG', compress_level=1); return b.getvalue()


def read_file(path):
    with open(path, 'rb') as f: return f.read()


def decode(data):
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    except (OSError, SyntaxError, Image.DecompressionBombError) as e:   # truncated, corrupt or not an image at all
        raise ValueError(f'not a readable image: {e}') from e
    if max(im.size) > MAX_SIDE: im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.asarray(im)


def find_examples(extra=None):
    """bundled examples (assets/examples/examples.json) plus every image of an optional extra folder."""
    ex = []
    man = os.path.join(EXAMPLES, 'examples.json')
    if os.path.exists(man):
        with open(man) as f: m = json.load(f)
        for kind in ('sources', 'references'):
            for e in m[kind]:
                ex.append(dict(name=e['name'], kind=kind[:-1], path=os.path.join(EXAMPLES, e['file'])))
    if extra and os.path.isdir(extra):
        for d, _, fs in sorted(os.walk(extra)):
            for f in sorted(fs):
                if f.lower().endswith(IMG_EXT) and not f.startswith('layer'):
                    kind = 'source' if f.startswith('bare') else 'reference' if f.startswith('madeup') else 'any'
                    ex.append(dict(name=os.path.basename(d) if f.startswith(('bare', 'madeup')) else os.path.splitext(f)[0],
                                   kind=kind, path=os.path.join(d, f)))
    return ex


def layer_png(at, F):
    return png(u8(F)), png(u8(at))


class App:
    def __init__(self, engine, out_dir, examples_dir=None, stylist=None):
        self.E, self.out_dir, self.stylist = engine, out_dir, stylist
        self.rec, self.vlm = [], None
        self.rec_imgs = {}                                    # index of a ranked look -> its reference image
        self.job_lock = threading.Lock()                      # claiming the single background job
        self.examples = find_examples(examples_dir)
        self.cache = {}                                       # name -> png bytes
        self.v = dict(photo=0, base=0, layer=0, stack=0, shade=0, looks=0, rec=0)
        self.flags = dict(stack=False, under=False, lip=False)

    # ---------------------------------------------------------------- state
    def state(self):
        E, o = self.E, self.E.out
        st = dict(E.status, inputs=dict(source=E.src is not None, **{s: s in E.refs for s in SLOTS}), options=E.opts,
                  versions=self.v, flags=self.flags, cached_refs=len(E.layers), keep_models=E.keep,
                  sampler=dict(sigma=E.sigma, steps=E.steps, seed=E.seed),
                  looks=[dict(id=l['id'], name=l['name']) for l in E.looks], stylist=self.stylist is not None, vlm=self.vlm,
                  rec=[{k: v for k, v in r.items() if k != 'crop'} for r in self.rec], mem=E.memory())
        if E.src is not None: st['photo'] = dict(width=int(E.src.shape[1]), height=int(E.src.shape[0]))
        if o and 'result' in self.cache:
            st['result'] = dict(width=int(o['target'].shape[1]), height=int(o['target'].shape[0]), caption=o.get('caption', ''),
                                timings={k: round(v, 3) for k, v in o['timings'].items()}, total=round(o['total'], 2),
                                mean_L=o['mean_L'], rendered=o['rendered'], eps=LR.EPS, target=E.opts['target'],
                                has_bare=o.get('bare') is not None, has_free=o.get('free') is not None)
        return st

    def _base_images(self):
        o = self.E.out; w = o['w']
        d = dict(photo=png(u8(o['photo'])), target=png(u8(o['target'])), paste=png(u8(o['paste'])), result=png(u8(o['result'])),
                 masks=png(u8(np.stack([w[0], w[1], w[2]], -1))), masks2=png(u8(np.stack([w[3], w[4], 0 * w[4]], -1))))
        d['F'], d['alpha'] = layer_png(o['alpha'], o['F'])
        if o.get('bare') is not None: d['bare'] = png(o['bare'])
        if o.get('free') is not None: d['free'] = png(o['free'])
        return d

    # ---------------------------------------------------------------- actions
    def upload(self, slot, data):
        img = decode(data)
        if slot == 'source':
            with self.E.lock: self.E.set_source(img)
            self.cache = {k: v for k, v in self.cache.items() if k.startswith('thumb_')}
            self.cache['photo'] = png(img); self.v['photo'] += 1
            self.flags = dict(stack=False, under=False, lip=False); self.v['looks'] += 1
        else:
            with self.E.lock: self.E.set_ref(slot, img)
        t = Image.fromarray(img); t.thumbnail((320, 320))
        self.cache[f'thumb_{slot}'] = png(np.asarray(t))
        if getattr(self.E, 'auto_preload', False): self.E.preload()          # weights load while the user picks the rest
        if slot == 'whole':
            for s in SLOTS[1:]: self.cache.pop(f'thumb_{s}', None)
        elif slot != 'source': self.cache.pop('thumb_whole', None)

    def run(self, render):
        def fn():
            self.E.run(render=render)
            self.cache.update(self._base_images()); self.v['base'] += 1; self.v['layer'] += 1
        self._background(fn)

    def soft(self, p):
        with self.E.lock:
            at, F = self.E.soft_layer({k: float(v) for k, v in (p.get('soft') or {}).items()})
            self.cache['F'], self.cache['alpha'] = layer_png(at, F); self.v['layer'] += 1

    def keep(self, p):
        with self.E.lock:
            lid = self.E.keep_look(p.get('edits') or {}, p.get('name'))
            l = self.E.looks[-1]; T = self.E.out['target']
            im = ED.compose(l['alpha'], l['F'], T); t = Image.fromarray(u8(im)); t.thumbnail((240, 240))
            self.cache[f'look_{lid}'] = png(np.asarray(t)); self.v['looks'] += 1

    def stack(self, p):
        with self.E.lock:
            r = self.E.stack_layer(p)
            self.flags['stack'] = r is not None
            if r is not None: self.cache['stackF'], self.cache['stackA'] = layer_png(*r)
            self.v['stack'] += 1

    def _shade_target(self):
        o = self.E.out
        return o['target_u8'] if o else self.E.src

    def shades_info(self):
        from irismix import shades as SH
        if self.E.src is None: raise ValueError('choose a photo first')
        with self.E.lock:
            f = self.E.shade_face(self._shade_target())
            lib = SH.load_shades(); rec = f.recommend(lib)
        pick = lambda s: s and dict(id=s['id'], name=s['name_en'], hex=s['hex'], category=s['category'], finish=s.get('finish'),
                                    strength=s.get('strength'), depth=s.get('depth'), undertone=s.get('undertone'))
        return dict(report=f.report(), library=[pick(s) for s in lib],
                    recommended=dict(foundations=[s['id'] for s in rec['foundations']],
                                     lipsticks={k: s['id'] for k, s in rec['lipsticks'].items()},
                                     blush=rec['blush']['id'] if rec['blush'] else None))

    def shades(self, p):
        with self.E.lock:
            under, lip = self.E.shade_layers(p, self._shade_target())
            self.flags['under'], self.flags['lip'] = under is not None, lip is not None
            if under is not None: self.cache['underF'], self.cache['underA'] = layer_png(*under)
            if lip is not None: self.cache['lipF'], self.cache['lipA'] = layer_png(*lip)
            self.v['shade'] += 1

    def _background(self, fn):
        """run fn under the engine lock in a worker thread; one job at a time (two requests cannot both claim it)."""
        E = self.E
        with self.job_lock:
            if E.status['busy']: raise ValueError('busy')
            E.status.update(busy=True, error=None)

        def job():
            try:
                with E.lock: fn()
            except Exception as e:                            # surface the message in the UI
                traceback.print_exc(); E.status['error'] = str(e); E.say(f'error: {e}')
            finally:
                E.status['busy'] = False
        threading.Thread(target=job, daemon=True).start()

    def recommend(self, p):
        """candidates: the bundled example looks and / or the references the user uploaded."""
        which = p.get('which', 'examples'); cands = []
        if which in ('examples', 'all'):
            cands += [(e['name'], decode(read_file(e['path']))) for e in self.examples if e['kind'] == 'reference']
        if which in ('mine', 'all'):
            cands += [(f'your {s}', img) for s, (h, img) in self.E.refs.items()]
        if not cands: raise ValueError('no candidate looks')
        imgs = dict(enumerate(c[1] for c in cands))

        def fn():
            rows = self.E.recommend_looks(cands, self.E.say)
            for r in rows:
                if 'crop' in r: self.cache[f"rec_{r['i']}"] = png(r['crop'])
            self.rec, self.vlm = rows, None; self.rec_imgs = imgs; self.v['rec'] += 1
            self.E.say(f'ranked {len([r for r in rows if "score" in r])} looks')
        self._background(fn)

    def use_rec(self, p):
        """put a recommended look into the whole-look slot."""
        img = self.rec_imgs.get(int(p.get('i', -1)))
        if img is None: raise ValueError('rank some looks first')
        self.E.set_ref('whole', img); t = Image.fromarray(img); t.thumbnail((320, 320))
        self.cache['thumb_whole'] = png(np.asarray(t))
        for s in SLOTS[1:]: self.cache.pop(f'thumb_{s}', None)

    def ask_stylist(self, p):
        if self.stylist is None: raise ValueError('start the viewer with --vlm to use the stylist')
        top = [r for r in self.rec if 'crop' in r][:int(p.get('n', 6))]
        if len(top) < 2: raise ValueError('rank some looks first')

        def fn():
            self.E.say('asking the stylist')
            n = len(top); cols = 3; S = 320; grid = Image.new('RGB', (cols * S, ((n + cols - 1) // cols) * S), 'white')
            d = ImageDraw.Draw(grid)
            for k, r in enumerate(top):
                x, y = (k % cols) * S, (k // cols) * S; grid.paste(Image.fromarray(r['crop']), (x, y))
                d.rectangle([x + 6, y + 6, x + 40, y + 40], fill='white'); d.text((x + 17, y + 16), str(k + 1), fill='black')
            with tempfile.TemporaryDirectory(prefix='irismix_stylist_') as tmp:
                f = os.path.join(tmp, 'looks.png'); grid.save(f)
                rank = self.stylist.ask(f, n, p.get('lang', 'en'))
            self.vlm = [dict(i=top[int(x['look']) - 1]['i'], name=top[int(x['look']) - 1]['name'], reason=x.get('reason', ''))
                        for x in rank if 1 <= int(x.get('look', 0)) <= n]
            self.v['rec'] += 1; self.E.say('stylist answered')
        self._background(fn)

    def compare(self, p):
        with self.E.lock:
            cards, overall = self.E.compare_shades(p.get('cat', 'foundation'), p.get('edits') or {}, p.get('shades') or {})
        enc = lambda x: 'data:image/jpeg;base64,' + base64.b64encode(self._jpg(x)).decode()
        return dict(overall=overall, cards=[dict({k: v for k, v in c.items() if k not in ('face', 'zoom')}, face=enc(c['face']),
                                                 zoom=enc(c['zoom'])) for c in cards])

    @staticmethod
    def _jpg(x):
        b = io.BytesIO(); Image.fromarray(x).save(b, 'JPEG', quality=90); return b.getvalue()

    def free_memory(self):
        """unload the networks and drop caches; the photo, the result, kept looks and shown images stay."""
        E = self.E
        if E.status['busy']: raise ValueError('busy')
        with E.lock:
            for r in self.rec: r.pop('crop', None)
            self.rec_imgs = {}; E.free_memory()
            E.say('memory released')

    def export(self, p):
        E = self.E
        if not E.out: raise ValueError('nothing to export yet')
        with E.lock:
            o = E.out; T = o['target']
            under, lip = E.shade_layers(p.get('shades') or {}, o['target_u8']) if p.get('shades') else (None, None)
            st = E.stack_layer(p['stack']) if p.get('stack') else None
            layers = [l for l in (under, st) if l is not None]
            if p.get('show_work', True): layers.append(E.edited_layer(p.get('edits') or {}))
            if lip is not None: layers.append(lip)
            img = T.copy()
            for a, F in layers: img = ED.compose(a, F, img)
            img = u8(img)
            d = os.path.join(self.out_dir, time.strftime('%Y%m%d-%H%M%S')); os.makedirs(d, exist_ok=True)
            Image.fromarray(img).save(f'{d}/edited.png')
            Image.fromarray(u8(o['result'])).save(f'{d}/result.png'); Image.fromarray(u8(o['paste'])).save(f'{d}/paste.png')
            LR.save(f'{d}/layer.npz', o['alpha'], o['F'])
            Image.fromarray(np.dstack([u8(o['F']), u8(o['alpha'])])).save(f'{d}/layer.png')
            with open(f'{d}/edits.json', 'w') as f: json.dump(dict(p, caption=o.get('caption', ''), options=E.opts), f, indent=1)
        print('[viewer] exported', d, flush=True)
        return png(img), d


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def send(self, code, body, ctype='application/json'):
            if isinstance(body, (dict, list)): body = json.dumps(body).encode()
            self.send_response(code); self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
            self.end_headers(); self.wfile.write(body)

        def body(self):
            n = int(self.headers.get('Content-Length', 0))
            if n > MAX_UPLOAD: raise ValueError(f'request larger than {MAX_UPLOAD // 2 ** 20} MB')
            return self.rfile.read(n) if n else b''

        def do_GET(self):
            p = urlparse(self.path).path
            try:
                if p in ('/', '/index.html'):
                    return self.send(200, read_file(os.path.join(STATIC, 'index.html')), 'text/html; charset=utf-8')
                if p.startswith('/static/') and '/' not in p[len('/static/'):] and p.endswith('.js'):
                    f = os.path.join(STATIC, p[len('/static/'):])
                    if not os.path.isfile(f): return self.send(404, dict(error='not found'))
                    return self.send(200, read_file(f), 'text/javascript; charset=utf-8')
                if p == '/api/state': return self.send(200, app.state())
                if p == '/api/examples':
                    return self.send(200, [dict(name=e['name'], kind=e['kind'], url=f'/examples/{i}') for i, e in enumerate(app.examples)])
                if p.startswith('/examples/'):
                    i = int(p.rsplit('/', 1)[1])
                    if not 0 <= i < len(app.examples): return self.send(404, dict(error='no such example'))
                    e = app.examples[i]
                    return self.send(200, read_file(e['path']), mimetypes.guess_type(e['path'])[0] or 'image/png')
                if p == '/api/shades': return self.send(200, app.shades_info())
                if p.startswith('/api/img/'):
                    k = p[len('/api/img/'):].removesuffix('.png')
                    if k in app.cache: return self.send(200, app.cache[k], 'image/png')
                    return self.send(404, dict(error=f'no image {k}'))
                self.send(404, dict(error='not found'))
            except ValueError as e:
                self.send(400, dict(error=str(e)))
            except Exception as e:
                traceback.print_exc(); self.send(500, dict(error=str(e)))

        def do_POST(self):
            u = urlparse(self.path); p, q = u.path, parse_qs(u.query)
            try:
                if p == '/api/upload':
                    slot = q.get('slot', [''])[0]
                    if slot not in ('source',) + SLOTS: return self.send(400, dict(error=f'bad slot {slot}'))
                    if app.E.status['busy']: return self.send(409, dict(error='busy'))
                    app.upload(slot, self.body()); return self.send(200, app.state())
                if p == '/api/export':
                    img, d = app.export(json.loads(self.body() or b'{}'))
                    self.send_response(200); self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', str(len(img))); self.send_header('X-Saved-To', d)
                    self.end_headers(); self.wfile.write(img); return
                a = json.loads(self.body() or b'{}')
                if p == '/api/clear':
                    slot = q.get('slot', [''])[0]
                    if app.E.status['busy']: return self.send(409, dict(error='busy'))
                    with app.E.lock: app.E.set_ref(slot, None)
                    app.cache.pop(f'thumb_{slot}', None)
                elif p == '/api/options': app.E.set_options(**a)
                elif p == '/api/run': app.run(bool(a.get('render', True)))
                elif p == '/api/soft': app.soft(a)
                elif p == '/api/keep': app.keep(a)
                elif p == '/api/stack': app.stack(a)
                elif p == '/api/shades': app.shades(a)
                elif p == '/api/recommend': app.recommend(a)
                elif p == '/api/use_rec': app.use_rec(a)
                elif p == '/api/stylist': app.ask_stylist(a)
                elif p == '/api/compare': return self.send(200, app.compare(a))
                elif p == '/api/free': app.free_memory()
                else: return self.send(404, dict(error='not found'))
                self.send(200, app.state())
            except ValueError as e:
                self.send(400, dict(error=str(e)))
            except Exception as e:
                traceback.print_exc(); self.send(500, dict(error=str(e)))
    return H


def serve(engine, host='127.0.0.1', port=7007, out_dir='outputs/viewer', examples_dir=None, stylist=None):
    app = App(engine, out_dir, examples_dir, stylist)
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    return app, httpd
