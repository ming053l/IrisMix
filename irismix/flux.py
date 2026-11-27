"""One-step FLUX.2-klein-4B generator shared by the decomposer (decompose 'E' / harmonise 'H') and the renderer:
VAE encode with encoder skip features, rectified-flow single step, decode with zero-initialised encoder -> decoder skips.

  x_1 = noise, x_0 = x_1 - v(x_1, t=1 | refs, prompt)   (sigma < 1: start from the noised first input, ART / pix2pix-Turbo)
E's output is the canvas [B | W | K] (one 1x3 latent, as the initial LoRA was trained); the skips (pix2pix-Turbo) feed the
input's features into the bare panel of E and the output of H. The training scripts import these classes unchanged."""
import os
import numpy as np, torch

P_E = ("Decompose this made-up face into three panels with exactly the same framing as the photo. Left panel: the same "
       "person with all makeup and face paint removed, natural bare skin, natural lips and brows, identical identity, "
       "expression, hair, lighting and background. Middle panel: only the makeup, over a pure white background. Right "
       "panel: the same makeup, over a pure black background. Skin, eyes, teeth, hair, clothes and background are not "
       "part of the makeup.")

P_H = ("Make this made-up face look like a natural, seamless real photograph: blend the makeup edges into the skin and "
       "match the makeup to the face's lighting. Keep the makeup design, colours and position exactly, and keep the "
       "person's identity, expression, hair and background exactly the same.")


def to_t(x):  # HxWx3 [0,1] -> 3xHxW [-1,1]
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1) * 2 - 1


def unpack_grid(x, h, w):
    """(B, h*w, C) tokens on the regular raster grid -> (B, C, h, w) contiguous: pipe._unpack_latents_with_ids with the
    identity index map of a full grid (same values, no device -> host sync for the grid size)."""
    return x.reshape(x.shape[0], h, w, x.shape[2]).permute(0, 3, 1, 2).contiguous()


class Skips(torch.nn.Module):
    """pix2pix-Turbo style encoder->decoder skips with zero-initialised 1x1 convs: features of the INPUT image at 1, 1/2
    and 1/4 resolution are added after decoder up-blocks 0-2, so unchanged content is copied from the input instead of
    regenerated. Zero init -> identical to the plain decoder."""
    def __init__(self):
        super().__init__()
        self.z = torch.nn.ModuleList([torch.nn.Conv2d(256, 512, 1), torch.nn.Conv2d(128, 512, 1), torch.nn.Conv2d(128, 256, 1)])
        for c in self.z: torch.nn.init.zeros_(c.weight); torch.nn.init.zeros_(c.bias)


class OneStep:
    """one-step klein generator. E: canvas [B | W | K]; H: one output aligned with the inputs.
    emb: precomputed prompt embeddings {'E': (pe, ids), 'H': (pe, ids)} (None: encoded here, the text encoder must be on
    dev); skips=False: no per-task skip modules (the renderer brings its own)."""
    def __init__(self, pipe, dev, emb=None, skips=True):
        self.p, self.dev = pipe, dev
        if emb is None:
            with torch.no_grad():
                emb = {k: pipe.encode_prompt(prompt=v, device=dev) for k, v in (('E', P_E), ('H', P_H))}
        self.emb = emb
        bn = pipe.vae.bn
        self.mu = bn.running_mean.view(1, -1, 1, 1).to(dev); self.sd = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(dev)
        if skips: self.skips = torch.nn.ModuleDict({'E': Skips(), 'H': Skips()}).to(dev)

    def enc(self, x, feats=True):
        """B,3,H,W [-1,1] -> normalised patchified latent (B,128,h,w) + encoder features (f0, f1, f2) for the skips (None
        with feats=False). The encoder trunk (conv_in, down blocks 0-1) runs ONCE and is shared by the skip features and
        the latent (same layers on the same input as vae.encode -> the same latent); the mean of the posterior is used."""
        v = self.p.vae; x = x.to(v.dtype)
        with torch.no_grad():
            e = v.encoder
            if v.use_tiling or v.use_slicing or hasattr(v, '_hf_hook') or len(e.down_blocks) < 2:   # generic path
                f = None
                if feats: f0 = e.conv_in(x); f1 = e.down_blocks[0](f0); f2 = e.down_blocks[1](f1); f = (f0, f1, f2)
                z = v.encode(x).latent_dist.mode()
            else:
                f0 = e.conv_in(x); f1 = e.down_blocks[0](f0); f2 = e.down_blocks[1](f1); h = f2
                for db in e.down_blocks[2:]: h = db(h)
                h = e.conv_act(e.conv_norm_out(e.mid_block(h)))
                h = e.conv_out(h)
                if v.quant_conv is not None: h = v.quant_conv(h)
                z = torch.chunk(h, 2, dim=1)[0]                              # DiagonalGaussianDistribution(h).mode()
                f = (f0, f1, f2) if feats else None
        z = (self.p._patchify_latents(z) - self.mu) / self.sd
        return z, f

    def dec(self, lat, feats=None, sk=None):
        v = self.p.vae; z = self.p._unpatchify_latents(lat * self.sd + self.mu).to(v.dtype)
        if v.post_quant_conv is not None: z = v.post_quant_conv(z)
        d = v.decoder; ck = lambda f, x: torch.utils.checkpoint.checkpoint(f, x, None, use_reentrant=False) if torch.is_grad_enabled() else f(x, None)
        h = d.conv_in(z); h = ck(d.mid_block, h)                             # decoder blocks checkpointed when training
        for k, ub in enumerate(d.up_blocks):
            h = ck(ub, h)
            if feats is not None and k < 3:                                  # skip into the left (input-sized) part only
                f = sk.z[k](feats[2 - k].float()).to(h.dtype)
                h = torch.cat([h[..., :f.shape[-1]] + f, h[..., f.shape[-1]:]], -1)
        h = d.conv_act(d.conv_norm_out(h)); return d.conv_out(h).float().clamp(-1, 1)

    def gen(self, refs, task, R, sigma=1.0):
        """refs: list of B,3,R,R. Returns E: (B,3,R,3R) canvas [B|W|K]; H: (B,3,R,R).
        sigma < 1 (H only): the output starts from the noised first input, x_s = (1 - s) z_in + s n, and one step gives
        x0 = x_s - s v (ART / pix2pix-Turbo: start from the input, not from pure noise, so identity is kept)."""
        p, B = self.p, refs[0].shape[0]
        enc = [self.enc(r, feats=(i == 0)) for i, r in enumerate(refs)]     # skip features: first input only
        zs = [e[0] for e in enc]
        ref_ids = p._prepare_image_ids([z[:1] for z in zs]).to(self.dev).repeat(B, 1, 1)
        rz = torch.cat([p._pack_latents(z) for z in zs], 1)
        wide = 3 if task == 'E' else 1                                       # E: canvas, one 1x3 latent
        # `noise` only fixes the latent shape / grid, but its draw advances the RNG: kept so seeded outputs are unchanged
        noise = torch.randn((B, 128, R // 16, wide * R // 16), device=self.dev, dtype=rz.dtype)
        out_ids = p._prepare_latent_ids(noise).to(self.dev)                  # (B, hw, 4): T=0, H, W, L=0
        x = p._pack_latents(torch.randn_like(noise))
        if sigma < 1: x = (1 - sigma) * p._pack_latents(zs[0]).to(x.dtype) + sigma * x
        pe, ti = self.emb[task]
        v = p.transformer(hidden_states=torch.cat([x, rz], 1).to(p.transformer.dtype),
                          timestep=torch.full((B,), sigma, device=self.dev, dtype=p.transformer.dtype),
                          guidance=None, encoder_hidden_states=pe.repeat(B, 1, 1), txt_ids=ti.repeat(B, 1, 1),
                          img_ids=torch.cat([out_ids, ref_ids], 1), return_dict=False)[0][:, :x.shape[1]]
        x0 = x - sigma * v.to(x.dtype)                                       # rectified flow, one step t = sigma -> 0
        lat = unpack_grid(x0, noise.shape[2], noise.shape[3])                # grid: full raster
        return self.dec(lat, enc[0][1], self.skips[task])                    # skips: E bare panel / H output


def load_state(path):
    """checkpoint -> state dict (.safetensors, or a torch .pt written by an older trainer)."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location='cpu', weights_only=True)


def load_lora_ckpt(pipe, path_or_sd, skips=None, prefix='skips.'):
    """copy the LoRA tensors of a checkpoint (transformer LoRA + skip convs) into the pipeline's LoRA modules (cast to
    their dtype) and the skip convs under `prefix` into `skips` (optional). Every LoRA module must be matched."""
    from irismix.config import N_LORA
    sd = load_state(path_or_sd) if isinstance(path_or_sd, str) else path_or_sd; n = 0
    for k, q in pipe.transformer.named_parameters():
        if k in sd: q.data.copy_(sd[k].to(q.dtype)); n += 1
    n_ck = sum('lora' in k for k in sd)
    if not n == n_ck == N_LORA: raise ValueError(f'LoRA tensors: matched {n}, in checkpoint {n_ck}, expected {N_LORA}')
    if skips is not None:
        sk = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not sk: raise ValueError(f'no skip weights with prefix {prefix!r}')
        skips.load_state_dict(sk)
    return n


_PIPES = {}                    # (base model, LoRA init, device) -> shared pipeline


def _placement(dev):
    """from_pretrained arguments that load the weights straight onto a CUDA device (no staging copy in host RAM)."""
    return dict(device_map=str(dev)) if torch.device(dev).type == 'cuda' else {}


def _load_text_encoder(base, dev):
    """the Qwen3 text encoder and its tokenizer of the base model, the encoder loaded straight onto dev."""
    from transformers import AutoTokenizer, Qwen3ForCausalLM
    te = Qwen3ForCausalLM.from_pretrained(base, subfolder='text_encoder', torch_dtype=torch.bfloat16, **_placement(dev))
    return te.eval(), AutoTokenizer.from_pretrained(base, subfolder='tokenizer')


def load_pipeline(dev, text_encoder=False, base_model=None, lora_init=None, skip_text_encoder=False, shared=True):
    """FLUX.2-klein-4B (bf16) with the rank-32 LoRA modules of step A; transformer and VAE on dev (text encoder too when
    asked; otherwise it stays on the CPU). skip_text_encoder: do not load it at all (prompt embeddings come from a cache).
    shared: the decomposer and the renderer use the SAME base pipeline (they differ only in the LoRA tensors and the skip
    convs, which each model keeps and copies in with use_lora before it runs), so switching between them does not reload
    the 4B transformer and the VAE from disk. A shared pipeline loaded without its text encoder gets only the encoder added
    when a later caller needs it. On a CUDA device the weights are read straight into GPU memory. release_pipelines() frees
    it."""
    from irismix import config
    key = (base_model or config.BASE_MODEL, os.path.abspath(lora_init or config.LORA_INIT), str(dev))
    pipe = _PIPES.get(key) if shared else None
    if pipe is None:
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
        on = _placement(dev)
        comp = dict(transformer=Flux2Transformer2DModel.from_pretrained(key[0], subfolder='transformer', torch_dtype=torch.bfloat16, **on),
                    vae=AutoencoderKLFlux2.from_pretrained(key[0], subfolder='vae', torch_dtype=torch.bfloat16, **on))
        if skip_text_encoder: comp.update(text_encoder=None, tokenizer=None)
        else: comp['text_encoder'], comp['tokenizer'] = _load_text_encoder(key[0], dev if text_encoder else 'cpu')
        pipe = Flux2KleinPipeline.from_pretrained(key[0], torch_dtype=torch.bfloat16, **comp)
        lora = key[1]
        pipe.load_lora_weights(os.path.dirname(lora), weight_name=os.path.basename(lora))
        pipe.transformer.to(dev); pipe.vae.to(dev); pipe._irismix_lora = None
        if shared: _PIPES[key] = pipe
    elif pipe.text_encoder is None and not skip_text_encoder:
        pipe.text_encoder, pipe.tokenizer = _load_text_encoder(key[0], dev if text_encoder else 'cpu')
    if text_encoder: pipe.text_encoder.to(dev)
    return pipe



def prefetch(base_model=None, subfolder='text_encoder'):
    """ask the kernel to start reading a component's weight files into the page cache in the background (readahead), so a
    later load does not wait on the disk; nothing is pinned or copied, and a missing local copy is simply skipped."""
    import glob
    from irismix import config
    base = base_model or config.BASE_MODEL
    d = os.path.join(base, subfolder) if os.path.isdir(base) else None
    if d is None:
        try:
            from huggingface_hub import snapshot_download
            d = os.path.join(snapshot_download(base, allow_patterns=[f'{subfolder}/*'], local_files_only=True), subfolder)
        except Exception:
            return
    for f in glob.glob(os.path.join(d, '*.safetensors')):
        try:
            fd = os.open(f, os.O_RDONLY)
            try: os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
            finally: os.close(fd)
        except (OSError, AttributeError):                  # no posix_fadvise (Windows / macOS): nothing to do
            return


def release_pipelines():
    """drop the shared base pipelines (the caller frees the CUDA cache)."""
    _PIPES.clear()


def lora_state(pipe, path_or_sd, dev):
    """the LoRA tensors of a checkpoint as {parameter name: tensor} on dev in the parameters' dtype (checked like
    load_lora_ckpt), for use_lora."""
    from irismix.config import N_LORA
    sd = load_state(path_or_sd) if isinstance(path_or_sd, str) else path_or_sd
    out = {k: sd[k].to(dev, q.dtype) for k, q in pipe.transformer.named_parameters() if k in sd}
    n_ck = sum('lora' in k for k in sd)
    if not len(out) == n_ck == N_LORA: raise ValueError(f'LoRA tensors: matched {len(out)}, in checkpoint {n_ck}, expected {N_LORA}')
    return out


@torch.no_grad()
def use_lora(pipe, owner, state):
    """make `state` (from lora_state) the active LoRA of the shared pipeline unless `owner` already is (exact copies)."""
    if getattr(pipe, '_irismix_lora', None) is owner: return
    params = dict(pipe.transformer.named_parameters())
    for k, v in state.items(): params[k].data.copy_(v)
    pipe._irismix_lora = owner
