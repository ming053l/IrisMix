"""Paths, model ids and the reported inference configuration.

Every path can be overridden by an environment variable (and the scripts expose the important ones as CLI arguments):
  IRISMIX_CACHE         cache directory (default ~/.cache/irismix): prompt embeddings, the MediaPipe landmarker
  IRISMIX_WEIGHTS       directory with the released weights (default <repo>/weights)
  IRISMIX_BASE_MODEL    FLUX.2-klein-4B, a Hugging Face id or a local diffusers folder
  IRISMIX_LANDMARKER    MediaPipe face_landmarker.task (downloaded into the cache when missing)
  IRISMIX_SEG_DEVICE    cpu | cuda for the face parser (default: cuda when available)
renderer sampler (paper): sigma 0.65, one Euler step, Omega dilated by 2 px, brows inside Omega, seed 0, bf16.
The appendix configuration is reachable with IRISMIX_SIGMA=0.7 IRISMIX_STEPS=4 IRISMIX_OMEGA_R=0 (0 = 6 px default)."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.expanduser(os.environ.get('IRISMIX_CACHE', os.path.join('~', '.cache', 'irismix')))
WEIGHTS_DIR = os.path.expanduser(os.environ.get('IRISMIX_WEIGHTS', os.path.join(REPO, 'weights')))

BASE_MODEL = os.environ.get('IRISMIX_BASE_MODEL', 'black-forest-labs/FLUX.2-klein-4B')
LORA_INIT = os.path.join(WEIGHTS_DIR, 'irismix_lora_init.safetensors')    # rank-32 LoRA modules (init of both models)
DECOMPOSER_CKPT = os.path.join(WEIGHTS_DIR, 'irismix_decomposer.safetensors')       # decomposer LoRA + skip convs (E / H)
RENDERER_CKPT = os.path.join(WEIGHTS_DIR, 'irismix_renderer.safetensors')       # renderer LoRA (EMA) + skip convs
N_LORA = 120                                                                # LoRA tensors per checkpoint

# face parser (SegFormer, CelebAMask-HQ labels; non-commercial licence, see NOTICE), pinned revision
PARSER_ID = 'jonathandinu/face-parsing'
PARSER_REVISION = '758b82e15a0178c9db39c1ff666a8b56e3a550c8'
# MediaPipe face landmarker (Apache-2.0)
LANDMARKER_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/'
                  'face_landmarker.task')


def landmarker_path():
    """path of face_landmarker.task: $IRISMIX_LANDMARKER, else <cache>/face_landmarker.task (downloaded once)."""
    p = os.environ.get('IRISMIX_LANDMARKER') or os.path.join(CACHE_DIR, 'face_landmarker.task')
    if not os.path.exists(p):
        import shutil, urllib.request
        os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
        part = f'{p}.{os.getpid()}.part'
        try:
            with urllib.request.urlopen(LANDMARKER_URL, timeout=60) as r, open(part, 'wb') as f: shutil.copyfileobj(r, f)
            os.replace(part, p)
        finally:
            if os.path.exists(part): os.remove(part)
    return p


SIGMA = float(os.environ.get('IRISMIX_SIGMA', '0.65'))
STEPS = int(os.environ.get('IRISMIX_STEPS', '1'))
OMEGA_R = int(os.environ.get('IRISMIX_OMEGA_R', '2'))        # px; 0 = edit_region default (6 px)
SEED = 0
