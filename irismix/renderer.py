"""Renderer: frozen FLUX.2-klein-4B + one rank-32 render LoRA + zero-init VAE skips from the pasted
composite C (the one-step generator code of flux.py: VAE encode with encoder features, decode with skips).

Token layout (OminiControl-style in-context conditions):
  output   x        512 -> 32 x 32 tokens, ids (T=0,  h, w)
  aligned  C, K, A  512 -> 32 x 32 each,  ids (T=10/20/30, h, w)        same (h, w) grid as the output
  maps     S1 S2 S3 G1 G2  256 -> 16 x 16, ids (T=40..80, 2h+.5, 2w+.5)  same grid at half density
  crops    lip, eyeL, eyeR 256 -> 16 x 16, ids (T=90/100/110, 64+h, 64+w)  own position offset (not aligned)
Input-anchored flow: x_sigma = (1 - sigma) z(C) + sigma n, straight path to z(I) with constant velocity
v = (x_sigma - z(I)) / sigma; any t in (0, sigma]: x_t = z(I) + t v, so a few Euler steps from sigma to 0 are exact
when v is learned. Output used as a masked residual: out = C + Omega (dec(x_0) - C) (render.compose_full).

GPU work per sample: one VAE encode per condition (11; only C keeps encoder features for the skips), STEPS transformer
evaluations, one decode. Position ids and the output grid are built once per shape on the device; prompt embeddings are
cached per text (Renderer.text): in memory for the process and on disk (<cache>/caption_embeds, IRISMIX_CAPTION_CACHE_MB,
default 512, 0 = off), so the 4B text encoder is loaded and moved to the GPU only for a caption it has not seen."""
import collections, hashlib, os
import numpy as np, torch

from irismix.flux import OneStep, Skips, load_lora_ckpt, load_pipeline, load_state, lora_state, use_lora, unpack_grid

ALIGNED = ('C', 'K', 'A')
MAPS = ('S1', 'S2', 'S3', 'G1', 'G2')
CROPS = ('lip', 'eyeL', 'eyeR')
TXT_LEN = 256
_EMB = collections.OrderedDict()   # (base model, caption) -> (embeddings, text ids) on the CPU, shared by Renderer instances
_EMB_MAX = 256


def _caption_file(base, text):
    from irismix import config
    mb = float(os.environ.get('IRISMIX_CAPTION_CACHE_MB', '512'))
    if mb <= 0: return None, 0
    h = hashlib.sha1(f'{base}\n{TXT_LEN}\n{text}'.encode()).hexdigest()
    return os.path.join(config.CACHE_DIR, 'caption_embeds', f'{h}.pt'), mb


def _prune(d, mb):
    """keep the on-disk caption cache under mb megabytes (oldest files first)."""
    fs = sorted((os.path.join(d, f) for f in os.listdir(d) if f.endswith('.pt')), key=os.path.getmtime)
    tot = sum(os.path.getsize(f) for f in fs)
    while fs and tot > mb * 2 ** 20:
        f = fs.pop(0); tot -= os.path.getsize(f); os.remove(f)


class Renderer:
    """ckpt: renderer checkpoint."""
    def __init__(self, dev='cuda', ckpt=None, text_encoder=True, base_model=None, lora_init=None):
        """text_encoder=False: never encode new captions (only cached embeddings can be used)."""
        from irismix import config
        # rank-32 LoRA modules; the text encoder is added to the shared pipeline only when a caption is not cached
        pipe = load_pipeline(dev, False, base_model=base_model, lora_init=lora_init, skip_text_encoder=True)
        self._load = dict(base_model=base_model, lora_init=lora_init); self.base = base_model or config.BASE_MODEL
        self.allow_text_encoder = text_encoder
        self.p, self.dev = pipe, dev
        self.one_step = OneStep(pipe, dev, emb={}, skips=False)
        self.mu, self.sd = self.one_step.mu, self.one_step.sd
        self.skips = Skips().to(dev)               # decoder skips, weights from the renderer checkpoint
        if ckpt: self.load(ckpt)
        self.emb_cache = {}
        self.dev_cache = {}                        # per-shape device constants

    # -------------------------------------------------------------------------------------------- weights
    def load(self, path, skips=True):
        sd = load_state(path)
        load_lora_ckpt(self.p, sd, self.skips if skips else None); self.p._irismix_lora = self
        self.lora = lora_state(self.p, sd, self.dev)              # kept: the base pipeline is shared with the decomposer

    # -------------------------------------------------------------------------------------------- text
    @torch.no_grad()
    def text(self, prompts):
        """prompt embeddings and text ids on the device; exactly what encode_prompt returns (cached copies are bitwise equal)."""
        out = []
        for s in prompts:
            if s not in self.emb_cache:
                if len(self.emb_cache) > 64: self.emb_cache.clear()
                self.emb_cache[s] = tuple(x.to(self.dev) for x in self._embedding(s))
            out.append(self.emb_cache[s])
        return torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out])

    def _embedding(self, s):
        key = (self.base, s)
        if key in _EMB: _EMB.move_to_end(key); return _EMB[key]
        f, mb = _caption_file(self.base, s)
        e = None
        if f and os.path.exists(f):
            try: e = tuple(torch.load(f, map_location='cpu', weights_only=True))
            except Exception: e = None                        # a damaged file is recomputed
        if e is None:
            if not self.allow_text_encoder: raise RuntimeError('caption not cached and the text encoder is disabled')
            self.p = load_pipeline(self.dev, True, **self._load)  # adds / moves the text encoder (shared pipeline)
            e = tuple(x.cpu() for x in self.p.encode_prompt(prompt=s, device=self.dev, max_sequence_length=TXT_LEN))
            if f:
                os.makedirs(os.path.dirname(f), exist_ok=True)
                part = f'{f}.{os.getpid()}.part'; torch.save(e, part); os.replace(part, f)
                _prune(os.path.dirname(f), mb)
        _EMB[key] = e
        while len(_EMB) > _EMB_MAX: _EMB.popitem(last=False)
        return e

    def park_text_encoder(self):
        """after the captions are encoded: move the text encoder (if loaded) off the GPU."""
        te = getattr(self.p, 'text_encoder', None)
        if te is not None: te.to('cpu')

    # -------------------------------------------------------------------------------------------- tokens
    def enc(self, x, feats=True): return self.one_step.enc(x, feats)

    def _ids(self, z, t0, scale=1.0, off=0.0):
        """position ids (B, h*w, 4) float on the device: T = t0, (h, w) * scale + off (cached per shape)."""
        key = ('ids', tuple(z.shape), t0, scale, off); c = self.dev_cache
        if key not in c:
            g = self.p._prepare_latent_ids(z).to(self.dev).float(); g[..., 0] = t0
            if scale != 1.0: g[..., 1:3] = scale * g[..., 1:3] + off
            elif off: g[..., 1:3] = g[..., 1:3] + off
            c[key] = g
        return c[key]

    def _encode_conditions(self, b):
        """-> ({name: latent}, encoder features of C). One encode per condition (only C keeps its skip features). Not batched
        on purpose: in bf16 a batched encode picks other convolution kernels and changes the render by a few levels."""
        zC, fC = self.enc(b['C'])
        zs = {'C': zC}
        for name in (ALIGNED + MAPS + CROPS)[1:]: zs[name] = self.enc(b[name], feats=False)[0]
        return zs, fC

    def cond_tokens(self, b):
        """b: dict of tensors in [-1, 1]: C, K, A (B,3,512,512), maps (B,3,256,256), crops (B,3,256,256).
        Returns (tokens, ids, latent of C, encoder features of C)."""
        p = self.p; toks, ids = [], []
        zs, fC = self._encode_conditions(b)
        for k, name in enumerate(ALIGNED):
            toks.append(p._pack_latents(zs[name])); ids.append(self._ids(zs[name], 10 * (k + 1)))
        for k, name in enumerate(MAPS):
            toks.append(p._pack_latents(zs[name])); ids.append(self._ids(zs[name], 40 + 10 * k, 2.0, 0.5))
        for k, name in enumerate(CROPS):
            toks.append(p._pack_latents(zs[name])); ids.append(self._ids(zs[name], 90 + 10 * k, 1.0, 64))
        return torch.cat(toks, 1), torch.cat(ids, 1), zs['C'], fC

    def velocity(self, x, t, ctok, cids, pe, ti):
        p = self.p; Bn = x.shape[0]
        grid = self.grid(Bn)
        v = p.transformer(hidden_states=torch.cat([x, ctok], 1).to(p.transformer.dtype),
                          timestep=t.to(p.transformer.dtype), guidance=None,
                          encoder_hidden_states=pe, txt_ids=ti,
                          img_ids=torch.cat([grid, cids], 1), return_dict=False)[0][:, :x.shape[1]]
        return v

    def grid(self, Bn, h=32):
        key = ('grid', Bn, h); c = self.dev_cache
        if key not in c:
            g = torch.cartesian_prod(torch.arange(1), torch.arange(h), torch.arange(h), torch.arange(1)).float().to(self.dev)
            c[key] = g[None].expand(Bn, -1, -1)
        return c[key]

    def decode(self, x, fC):
        return self.one_step.dec(unpack_grid(x, 32, 32), fC, self.skips)

    # -------------------------------------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, b, prompts, sigma=0.7, steps=4, seed=0):
        """-> decoded output (B,3,512,512) in [-1, 1] (before the Omega residual). The reported configuration is
        config.SIGMA / config.STEPS (0.65, 1 step); the defaults here are the grid the renderer was mostly trained on."""
        if getattr(self, 'lora', None) is not None: use_lora(self.p, self, self.lora)
        pe, ti = self.text(prompts)
        ctok, cids, zC, fC = self.cond_tokens(b)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        n = torch.randn(zC.shape, device=self.dev, dtype=zC.dtype, generator=g)
        x = self.p._pack_latents((1 - sigma) * zC + sigma * n)
        ts = np.linspace(sigma, 0, steps + 1)
        for i in range(steps):
            t = torch.full((x.shape[0],), float(ts[i]), device=self.dev)
            v = self.velocity(x, t, ctok, cids, pe, ti)
            x = x + float(ts[i + 1] - ts[i]) * v.to(x.dtype)
        return self.decode(x, fC)
