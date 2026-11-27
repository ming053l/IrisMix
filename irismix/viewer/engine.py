"""The look engine behind the viewer: one photo, references per part, the paper's applications and the working layer.

Target preparation (one of):
  as is         the look is transferred onto the photo
  remake        re-make-up: the decomposer strips the photo's own makeup, the look is applied to the recovered bare face, and
                parts of the photo's own makeup can be kept (its own layer is added as a reference for those parts)
  glasses/bangs occlusion-aware transfer (irismix.occluder): the occluder is lifted off, the look transferred, the occluder
                put back on top; the layer is then expressed over the original photo
Look: references per part (or one for the whole look); face paint can be moved, scaled and mirrored before rendering
(pipeline.design_transform). Transfer = decomposer + rules + warp; Render adds the renderer and the closed-form layer recovery.
Editing (closed form, also live in the browser): per-part strength, recolour, hue, softness, lip finish; kept looks stack
in order or blend linearly; shade try-on adds foundation / blush under the look and a lipstick on top.

Decompositions are cached by image content (a reference is decomposed once and reused across photos and looks).
Models load on first use (or in the background as soon as a photo is uploaded, preload()); keep_models=False (default,
about 24 GB) holds one network on the GPU at a time. start_idle_release(minutes) gives the GPU memory back after that many
idle minutes; the next render loads the models again."""
import collections, ctypes, gc, hashlib, sys, threading, time
import numpy as np, cv2

from irismix import config, rules, pipeline as PL, render as RD, layer_recover as LR, edits as ED
from irismix.utils import face_box, lab, u8

SLOTS = ('whole', 'eyes', 'lips', 'face', 'forehead')       # 'whole': one reference for the whole look
PAINT = ('face', 'forehead')                                 # parts that the design transform moves
TARGETS = ('as_is', 'remake', 'glasses', 'bangs')
MAX_LAYERS = 16                                              # cached decompositions (about 40 MB each at 1024^2)
MAX_FREE = 4                                                 # cached occluder-free faces


def image_sha1(img): return hashlib.sha1(img.tobytes() + bytes(str(img.shape), 'ascii')).hexdigest()


class Engine:
    def __init__(self, device='cuda', keep_models=False, decomposer=None, renderer=None, lora_init=None, base_model=None,
                 cache_dir=None, sigma=config.SIGMA, steps=config.STEPS, seed=config.SEED):
        self.dev, self.keep = device, keep_models
        self.paths = dict(decomposer=decomposer or config.DECOMPOSER_CKPT, renderer=renderer or config.RENDERER_CKPT,
                          lora_init=lora_init or config.LORA_INIT, base_model=base_model or config.BASE_MODEL,
                          cache_dir=cache_dir or config.CACHE_DIR)
        self.sigma, self.steps, self.seed = sigma, steps, seed
        self.models = {}                    # 's1' | 'rm' | 'editor'
        self.layers = collections.OrderedDict()   # sha1 of an image -> decomposed layer (least recently used dropped first)
        self.free = collections.OrderedDict()     # (sha1 of photo, cls) -> occluder-free face
        self.lock = threading.Lock()
        self.src = None; self.refs = {}     # slot -> (sha1, uint8 image)
        self.opts = dict(target='as_is', keep=[], xform=dict(dx=0.0, dy=0.0, scale=1.0, mirror=False))
        self.out = {}                       # the working look
        self.looks = []                     # kept looks: dict(id, name, alpha, F)
        self.face = None                    # shades.Face of the target (lazy)
        self.status = dict(busy=False, stage='idle', log=[], error=None)
        self.model_lock = threading.RLock()   # loading / dropping networks (a background preload must not block uploads)
        self.last_use = time.time()
        rules.LMK_FALLBACK = True              # in-the-wild photos: low-confidence landmark retry

    # ------------------------------------------------------------------ inputs
    def set_source(self, img):
        self.src = img; self.out = {}; self.looks = []; self.face = None; self.last_use = time.time()

    def set_ref(self, slot, img):
        if slot not in SLOTS: raise ValueError(f'unknown slot {slot}')
        if img is None: self.refs.pop(slot, None); return
        if slot == 'whole': self.refs = {}
        else: self.refs.pop('whole', None)
        self.refs[slot] = (image_sha1(img), img); self.last_use = time.time()

    def set_options(self, **o):
        if 'target' in o and o['target'] not in TARGETS: raise ValueError(f'unknown target {o["target"]}')
        if 'keep' in o: o['keep'] = [p for p in o['keep'] if p in RD.NAMES]
        if 'xform' in o: o['xform'] = dict(self.opts['xform'], **{k: v for k, v in o['xform'].items() if k in ('dx', 'dy', 'scale', 'mirror')})
        self.opts.update(o)

    # ------------------------------------------------------------------ models (one on the GPU unless keep_models)
    def _free_gpu(self):
        """return freed memory: Python objects, the CUDA caching allocator and the C heap (glibc keeps freed arenas)."""
        gc.collect()
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized(): torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except (OSError, AttributeError):
            pass

    @staticmethod
    def _cap(cache, n):
        while len(cache) > n: cache.popitem(last=False)

    def free_memory(self):
        """unload every network and drop the caches (decompositions, occluder-free faces, skin analysis); kept looks and
        the current result stay."""
        self.models.clear(); self.layers.clear(); self.free.clear(); self.face = None
        if 'irismix.flux' in sys.modules: sys.modules['irismix.flux'].release_pipelines()
        self.__dict__.pop('_scorer', None)
        rules.clear_caches(); rules.close_models()
        self._free_gpu()

    def memory(self):
        """resident memory of the process and GPU memory held by PyTorch, in GB."""
        m = dict(ram=None, gpu=None, gpu_reserved=None, cached_layers=len(self.layers))
        try:
            with open('/proc/self/status') as f:
                m['ram'] = round(int(next(l for l in f if l.startswith('VmRSS')).split()[1]) / 2 ** 20, 2)
        except (OSError, StopIteration):
            pass
        torch = sys.modules.get('torch')
        try:                                                  # torch may still be importing in the worker thread
            if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
                m['gpu'] = round(torch.cuda.memory_allocated() / 2 ** 30, 2); m['gpu_reserved'] = round(torch.cuda.memory_reserved() / 2 ** 30, 2)
        except AttributeError:
            pass
        return m

    def _gpu_loaded(self):
        return bool(self.models) or bool(getattr(sys.modules.get('irismix.flux'), '_PIPES', None))

    def preload(self, min_free_gb=14.0):
        """load the decomposer (and with it the shared base pipeline) in a background thread when nothing is loaded yet and
        the GPU has room, so the first Render after an upload does not wait for the weights. Uploads are not blocked."""
        if self.status['busy'] or self._gpu_loaded(): return
        def job():
            if not self.model_lock.acquire(blocking=False): return
            try:
                import torch
                if not torch.cuda.is_available() or self._gpu_loaded(): return
                if torch.cuda.mem_get_info(torch.device(self.dev))[0] < min_free_gb * 2 ** 30: return      # a shared GPU without room: load on demand
                self._model('s1'); self.say('decomposer ready')
            except Exception as e:                                                   # a failed preload is retried by Render
                self.say(f'preload skipped: {e}')
            finally:
                self.model_lock.release()
        threading.Thread(target=job, daemon=True).start()

    def release_models(self, reason=''):
        """drop every network and the shared base pipeline (GPU memory and the parked text encoder); decompositions and
        looks stay cached."""
        with self.model_lock:
            self.models.clear(); self.__dict__.pop('_scorer', None)
            if 'irismix.flux' in sys.modules: sys.modules['irismix.flux'].release_pipelines()
            self._free_gpu()
        if reason: self.say(reason)

    def start_idle_release(self, minutes):
        """give the GPU memory back after `minutes` without use (0: never)."""
        if not minutes or minutes <= 0: return
        def loop():
            while True:
                time.sleep(30)
                if self.status['busy'] or not self._gpu_loaded() or time.time() - self.last_use < minutes * 60: continue
                if not self.lock.acquire(blocking=False): continue
                try:
                    if not self.status['busy']:
                        self.release_models(f'GPU memory released after {minutes:g} idle minutes (the next render loads the models again)')
                finally:
                    self.lock.release()
        threading.Thread(target=loop, daemon=True).start()

    def _model(self, name):
        with self.model_lock: return self._model_locked(name)

    def _model_locked(self, name):
        self.last_use = time.time()
        if name not in self.models:
            if not self.keep and self.models: self.models.clear(); self._free_gpu()
            if name == 'editor' and not self.keep:          # the editor brings its own transformer: drop the shared base
                from irismix import flux; flux.release_pipelines(); self._free_gpu()
            p = self.paths
            self.say({'s1': 'loading the decomposer', 'rm': 'loading the renderer', 'editor': 'loading the occluder editor'}[name])
            if name == 's1':
                from irismix import decomposer as DEC
                m = DEC.Decomposer(p['decomposer'], self.dev, base_model=p['base_model'], lora_init=p['lora_init'], cache_dir=p['cache_dir'])
            elif name == 'rm':
                from irismix.renderer import Renderer
                m = Renderer(self.dev, ckpt=p['renderer'], base_model=p['base_model'], lora_init=p['lora_init']); m.skips.eval()
            else:
                from irismix import occluder as OC
                rm = self.models.get('rm')                       # share its text encoder and VAE when it is loaded
                m = OC.load_editor(self.dev, p['base_model'], share=rm.p if rm is not None else None)
            self.models[name] = m
        return self.models[name]

    def _decompose(self, img, label):
        h = image_sha1(img)
        if h in self.layers: self.layers.move_to_end(h)
        else:
            s1 = self._model('s1'); self.say(f'Decomposer: decomposing {label}')
            L = PL.decompose(s1, img)
            if 'error' in L: raise ValueError(f'{label}: {L["error"]}')
            self.layers[h] = L; self._cap(self.layers, MAX_LAYERS)
        return self.layers[h]

    # ------------------------------------------------------------------ run
    def say(self, msg):
        self.status['stage'] = msg; self.status['log'].append(f'{time.strftime("%H:%M:%S")}  {msg}')
        del self.status['log'][:-30]

    def run(self, render=True):
        """prepare the target -> decompose (cached) -> transfer rules + warp -> [renderer] -> closed-form layer."""
        import torch
        if self.src is None: raise ValueError('choose a photo first')
        if not self.refs and not (self.opts['target'] == 'remake' and self.opts['keep']):
            raise ValueError('choose at least one reference')
        W, t, T0 = self.src, {}, time.time()
        tgt = self.opts['target']; free = bare = None
        # ---- target
        if tgt in ('glasses', 'bangs'):
            k = (image_sha1(W), tgt)
            if k not in self.free:
                from irismix import occluder as OC
                ed = self._model('editor'); self.say(f'lifting the {tgt} off'); t0 = time.time()
                self.free[k] = OC.remove(ed, W, tgt, self.seed); t['occluder'] = time.time() - t0
                ed = None; self._cap(self.free, MAX_FREE)
            S = free = self.free[k]
        elif tgt == 'remake':
            t0 = time.time(); Ls = self._decompose(W, 'the photo'); t['remake'] = time.time() - t0
            S = bare = Ls['Br']
        else:
            S = W
        # ---- references (one decomposition per distinct image); the design transform moves face paint only
        groups = {}
        for slot, (h, img) in self.refs.items():
            g = groups.setdefault(h, dict(img=img, parts=[]))
            g['parts'] = None if slot == 'whole' else g['parts'] + [slot]
        M = PL.design_transform(S, **self.opts['xform'])
        refs = []; t0 = time.time()
        for i, g in enumerate(groups.values()):
            L = self._decompose(g['img'], f'reference {i + 1}/{len(groups)}')
            if M is not None and (g['parts'] is None or set(g['parts']) & set(PAINT)):
                ps = list(RD.NAMES) if g['parts'] is None else g['parts']
                paint, rest = [p for p in ps if p in PAINT], [p for p in ps if p not in PAINT]
                if paint: refs.append(dict(layer=L, parts=paint, M=M))
                if rest: refs.append(dict(layer=L, parts=rest))
            else:
                refs.append(dict(layer=L, parts=g['parts']))
        if tgt == 'remake' and self.opts['keep']:             # mixed: keep some of the photo's own parts
            refs.insert(0, dict(layer=Ls, parts=list(self.opts['keep'])))
        t['decompose'] = time.time() - t0
        self.say('transfer rules and warp'); t0 = time.time()
        C, av, _ = PL.paste(S, refs); t['warp'] = time.time() - t0
        out = dict(photo=W.astype(np.float32) / 255, target=S.astype(np.float32) / 255, paste=C, bare=bare, free=free)
        if render:
            it = PL.look_item(S, refs, C, av)
            arr, info = RD.prepare(it)
            rm = self._model('rm'); self.say('Renderer: rendering')
            with torch.autocast(torch.device(self.dev).type, dtype=torch.bfloat16): rm.text([it['text']])
            rm.park_text_encoder(); self._free_gpu()
            t0 = time.time()
            H5 = RD.render_one(rm, arr, it['text'], sigma=self.sigma, steps=self.steps, seed=self.seed)
            I, _ = RD.compose_full(S, C, H5, info, av); t['render'] = time.time() - t0
            I = I.astype(np.float32) / 255 if I.dtype == np.uint8 else I.astype(np.float32)
            A = it['A']; out['caption'] = it['text']; rm = H5 = None      # no local reference may keep the renderer alive
        else:
            I, A = C, av; out['caption'] = ''
        if free is not None:                                   # put the occluder back; the layer is over the photo
            from irismix import occluder as OC
            self.say(f'putting the {tgt} back'); t0 = time.time()
            I = OC.restore(W, free, I, tgt).astype(np.float32); t['restore'] = time.time() - t0
            S = W; out['target'] = out['photo']
        self.say('closed-form layer recovery'); t0 = time.time()
        if render or free is not None: at, F = LR.layer_from_render(I, S, A, av)
        else: at, F = LR.recover(C, out['target'], av)
        t['recover'] = time.time() - t0
        w = PL.part_masks(S)
        out.update(result=I, alpha=at, F=F, w=w, mm=PL.px_per_mm(S), mean_L=self._mean_L(at, F, w), timings=t,
                   total=time.time() - T0, rendered=bool(render), target_u8=S)
        if not self.keep: self.models.clear(); self._free_gpu()
        if self.face is not None and self.face.key != image_sha1(S): self.face = None
        self.out = out; self.last_use = time.time(); self.say(f'done in {out["total"]:.1f} s')
        return out

    @staticmethod
    def _mean_L(at, F, w):
        L = lab(F)[..., 0]
        return {p: float(L[m].mean()) if (m := (w[k] > 0.5) & (at > LR.EPS)).any() else 50.0 for k, p in enumerate(RD.NAMES)}

    # ------------------------------------------------------------------ edits (exact, numpy)
    def _working_look(self):
        if not self.out: raise ValueError('transfer or render a look first')
        return self.out

    def edited_layer(self, e):
        """the working layer with the edit parameters e (dict of the browser's controls) -> (alpha, F)."""
        o = self._working_look()
        _, at, F = PL.apply_edits(o['alpha'], o['F'], o['target_u8'], o['w'], 1.0, (), e.get('recolour') or None,
                                  e.get('hue') or None, e.get('part_strength'), e.get('soft') or None, e.get('finish') or None,
                                  o['mm'])
        return np.clip(at * float(e.get('strength', 1.0)), 0, 1), F

    def soft_layer(self, soft):
        """the working layer after softness only (the browser applies the other edits live on top of it)."""
        o = self._working_look()
        at, F = o['alpha'], o['F']
        for p, r in (soft or {}).items():
            if r > 0: at, F = ED.soften(at, F, o['w'][RD.NAMES.index(p)], r * o['mm'])
        return at, F

    # ------------------------------------------------------------------ looks: keep, stack, blend
    def keep_look(self, e, name=None):
        at, F = self.edited_layer(e)
        lid = max([l['id'] for l in self.looks] + [0]) + 1
        self.looks.append(dict(id=lid, name=name or f'Look {lid}', alpha=at.astype(np.float32), F=F.astype(np.float32)))
        return lid

    def stack_layer(self, spec):
        """spec: dict(order=[ids bottom to top], visible={id: bool}, strength={id: s}, blend=None | dict(a, b, t))
        -> combined (alpha, F) or None."""
        L = {l['id']: l for l in self.looks}
        b = spec.get('blend')
        if b and b.get('a') in L and b.get('b') in L:
            return ED.blend((L[b['a']]['alpha'], L[b['a']]['F']), (L[b['b']]['alpha'], L[b['b']]['F']), float(b.get('t', 0.5)))
        vis = spec.get('visible', {}); st = spec.get('strength', {})
        ls = [(np.clip(L[i]['alpha'] * float(st.get(str(i), st.get(i, 1.0))), 0, 1), L[i]['F'])
              for i in spec.get('order', []) if i in L and vis.get(str(i), vis.get(i, True))]
        return ED.stack(ls) if ls else None

    # ------------------------------------------------------------------ shades
    def shade_face(self, T8):
        from irismix import shades as SH
        if T8 is None: raise ValueError('choose a photo first')
        if self.face is None or self.face.key != image_sha1(T8):
            self.say('analysing skin and lips'); self.face = SH.Face(T8); self.face.key = image_sha1(T8)
        return self.face

    def shade_layers(self, spec, T8):
        """spec: dict(foundation=id|None, blush=id|None, lipstick=id|None, strengths, finishes) -> (under, lip) layers."""
        from irismix import shades as SH
        f = self.shade_face(T8); lib = {s['id']: s for s in SH.load_shades()}
        st = spec.get('strength', {}); fin = spec.get('finish', {})
        under = []
        if spec.get('foundation') in lib: under.append(f.foundation(lib[spec['foundation']], st.get('foundation'), fin.get('foundation')))
        if spec.get('blush') in lib: under.append(f.blush(lib[spec['blush']], st.get('blush')))
        lip = f.lipstick(lib[spec['lipstick']], st.get('lipstick'), fin.get('lipstick')) if spec.get('lipstick') in lib else None
        return (ED.stack(under) if under else None), lip

    # ------------------------------------------------------------------ recommendation
    def recommend_looks(self, cands, progress=None):
        """score each candidate reference on the current target: warp-only transfer (decomposition cached), aesthetic and CLIP
        preference of the made-up face crop, personal-colour harmony. -> list of dicts, best first."""
        from irismix import aesthetics as AE
        T8 = self.out['target_u8'] if self.out else self.src
        if T8 is None: raise ValueError('choose a photo first')
        face = self.shade_face(T8); crops, rows = [], []
        for i, (name, img) in enumerate(cands):
            if progress: progress(f'trying look {i + 1}/{len(cands)}: {name}')
            try:
                L = self._decompose(img, name)
                C, _, _ = PL.paste(T8, [dict(layer=L, parts=None)])
            except ValueError as e:
                rows.append(dict(i=i, name=name, error=str(e))); continue
            crops.append(face_crop(u8(C), T8))
            h, why = AE.harmony(face, AE.look_colours(L))
            rows.append(dict(i=i, name=name, harmony=h, reasons=why, crop=crops[-1]))
        if not self.keep: self.models.clear(); self._free_gpu()
        ok = [r for r in rows if 'error' not in r]
        if not ok: raise ValueError('no candidate could be transferred')
        if progress: progress('scoring aesthetics')
        import torch
        sc = getattr(self, '_scorer', None) or AE.Scorer('cuda' if torch.cuda.is_available() and self.dev != 'cpu' else 'cpu')
        if self.keep: self._scorer = sc
        aes, pref = sc([r['crop'] for r in ok])
        z = lambda x: (x - x.min()) / max(float(np.ptp(x)), 1e-6)
        za, zp = z(aes), z(pref)
        for r, a, p, na, npf in zip(ok, aes, pref, za, zp):
            r.update(aesthetic=float(a), preference=float(p), score=float(100 * (0.4 * na + 0.25 * npf + 0.35 * r['harmony'])))
        if not self.keep: del sc; self._free_gpu()
        ok.sort(key=lambda r: -r['score'])
        return ok + [r for r in rows if 'error' in r]


    def compare_shades(self, cat, edits, spec):
        """comparison cards for one shade category on the current look: the recommended shades, each with a face crop, a
        zoom (cheek or lips), a subtitle, who it suits and short notes as codes the browser words in its language, the best one
        marked. -> (cards, overall)"""
        from irismix import shades as SH, aesthetics as AE
        if cat not in ('foundation', 'lipstick', 'blush'): raise ValueError(f'unknown shade category {cat}')
        T8 = self.out['target_u8'] if self.out else self.src
        f = self.shade_face(T8); lib = SH.load_shades(); byid = {s['id']: s for s in lib}; rec = f.recommend(lib)
        ut = f.undertone(); skin = f.skin
        if cat == 'foundation': cands = rec['foundations'][:4]
        elif cat == 'lipstick':
            cands = list(rec['lipsticks'].values())
            bold = max((s for s in lib if s['category'] == 'lipstick' and s not in cands), key=lambda s: SH.chroma(s['lab']))
            cands = (cands + [bold])[:4]
        else: cands = [s for s in lib if s['category'] == 'blush'][:4]
        if cat == 'lipstick':                                  # the pick must agree with its own explanation
            best = max(enumerate(cands), key=lambda c: (AE.harmony(f, {'lips': np.asarray(c[1]['lab'])})[0], -c[0]))[1]['id']
        else: best = cands[0]['id'] if cat == 'foundation' else (rec['blush'] or cands[0])['id']
        base = self.out['target'].copy() if self.out else T8.astype(np.float32) / 255
        other = dict(spec or {}); other[cat] = None
        under, lip = self.shade_layers(other, T8)
        work = self.edited_layer(edits) if self.out else None
        pts = f.pts; fw = f.fw
        if cat == 'lipstick': zc, zw = pts[[61, 291, 0, 17]].mean(0), 1.7 * float(np.ptp(pts[[61, 291], 0]))
        else:                                                  # the cheek less covered by hair
            zw = 0.30 * fw
            def skin_cover(c):
                x0, y0 = int(max(c[0] - zw / 2, 0)), int(max(c[1] - zw / 2, 0))
                return float(f.base_m[y0:y0 + int(zw), x0:x0 + int(zw)].mean())
            zc = max((0.75 * pts[50] + 0.25 * pts[1], 0.75 * pts[280] + 0.25 * pts[1]), key=skin_cover)
        role = {v['id']: k for k, v in rec['lipsticks'].items()}
        cards = []
        for s in cands:
            layer = (f.foundation(s, spec.get('strength', {}).get('foundation'), spec.get('finish', {}).get('foundation')) if cat == 'foundation'
                     else f.blush(s, spec.get('strength', {}).get('blush')) if cat == 'blush'
                     else f.lipstick(s, spec.get('strength', {}).get('lipstick'), spec.get('finish', {}).get('lipstick')))
            img = base
            seq = [under] if cat == 'lipstick' else []
            if cat != 'lipstick': seq = [layer] if under is None else [ED.stack([under, layer])]
            seq += [work] if work is not None else []
            seq += [layer] if cat == 'lipstick' else ([lip] if lip is not None else [])
            for l in seq:
                if l is not None: img = ED.compose(l[0], l[1], img)
            u = u8(img)
            dE = SH.ciede2000(skin, s['lab']); dL = s['lab'][0] - skin[0]; notes = []      # codes, worded by the browser
            if cat == 'foundation':
                sub = dict(k='depth_ut', depth=s.get('depth'), ut=s.get('undertone'))
                notes.append(dict(k='f_closest', dE=round(dE, 1)) if dE < 3 else dict(k='f_brighter') if dL > 2
                             else dict(k='f_deeper') if dL < -2 else dict(k='f_close', dE=round(dE, 1)))
                notes.append(dict(k='f_same_ut', ut=ut) if s.get('undertone') == ut else dict(k='f_shift_ut', ut=s.get('undertone')))
                notes.append(dict(k='finish', f=spec.get('finish', {}).get('foundation') or s.get('finish', 'satin')))
                suits = dict(k='f_best') if s['id'] == best else dict(k='f_brighter_look') if dL > 2 else dict(k='f_sunkissed') if dL < -2 \
                    else dict(k='f_cast', ut=s.get('undertone'))
            else:
                w = AE.name_colour(np.asarray(s['lab']))
                if cat == 'lipstick':
                    r = role.get(s['id'], 'bold'); sub = dict(k='role', role=r)
                    _, why = AE.harmony(f, {'lips': np.asarray(s['lab'])})
                    notes += [dict(k='harmony', **why[0]), dict(k='l_role', role=r), dict(k='contrast', strong=bool(skin[0] - s['lab'][0] > 35))]
                    suits = dict(k='best') if s['id'] == best else dict(k='l_suits', role=r)
                else:
                    sub = dict(k='colour', colour=w)
                    notes += [dict(k='b_flush', colour=w), dict(k='b_match' if s['id'] == best else 'b_alt')]
                    suits = dict(k='best') if s['id'] == best else dict(k='b_other')
            cards.append(dict(id=s['id'], name=s['name_en'], hex=s['hex'], sub=sub, suits=suits, notes=notes, best=s['id'] == best,
                              face=face_crop(u, T8, 1.25, 300), zoom=box_crop(u, zc, zw, 300)))
        overall = dict(cat=cat, ut=ut, ita=round(SH.ita(skin)), best=byid[best]['name_en'])
        return cards, overall


# ---------------------------------------------------------------------- recommendation helpers (look scoring, shade cards)
def face_crop(img, T8, k=1.5, n=320):
    """square face crop of img in the frame of the target T8 (its landmarks), resized to n."""
    p = rules.landmarks(T8, fallback=True)
    if p is None: raise ValueError('no face found in the photo')
    X0, Y0, s = face_box(p, T8.shape, k)
    return cv2.resize(np.ascontiguousarray(img[Y0:Y0 + s, X0:X0 + s]), (n, n), interpolation=cv2.INTER_AREA)


def box_crop(img, c, w, n=320):
    H, W = img.shape[:2]; w = int(max(16, min(w, H, W))); x0 = int(np.clip(c[0] - w / 2, 0, W - w)); y0 = int(np.clip(c[1] - w / 2, 0, H - w))
    return cv2.resize(np.ascontiguousarray(img[y0:y0 + w, x0:x0 + w]), (n, n), interpolation=cv2.INTER_AREA)
