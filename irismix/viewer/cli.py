"""IrisMix viewer: an interactive web app for composing, rendering and editing makeup looks.

  python scripts/viewer.py                       # http://localhost:7007   (or: irismix-viewer after pip install -e .)
  python scripts/viewer.py --port 7010 --keep_models

On a remote GPU machine, forward the port and open the address in a local browser:
  ssh -L 7007:localhost:7007 <user>@<host>

Upload a source photo and one reference for the whole look (or one per part), then Transfer (warp only, about a second)
or Render (the decomposer and the renderer). Strength, part toggles, recolouring and hue edits of the recovered layer run live in the
browser; Export recomputes them exactly and saves edited.png, result.png, layer.npz and edits.json in --out."""
import argparse, getpass, os, socket

from irismix import config



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=7007)
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to accept connections from other machines')
    ap.add_argument('--out', default=os.path.join(config.REPO, 'outputs', 'viewer') if config.IN_CHECKOUT else os.path.join(os.getcwd(), 'irismix_outputs'),
                    help='export directory')
    ap.add_argument('--examples', default=None, help='extra folder of example photos (the bundled assets/examples are always shown)')
    ap.add_argument('--keep_models', action='store_true', help='keep the decomposer and the renderer on the GPU (about 40 GB)')
    ap.add_argument('--idle_release', type=float, default=10, help='minutes without use after which the GPU memory is given back (0: never)')
    ap.add_argument('--no_preload', action='store_true', help='do not start loading the decomposer in the background when a photo is uploaded')
    ap.add_argument('--threads', type=int, default=0, help='CPU threads (default: $IRISMIX_THREADS, else min(8, #cores))')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--decomposer', default=config.DECOMPOSER_CKPT); ap.add_argument('--renderer', default=config.RENDERER_CKPT)
    ap.add_argument('--lora_init', default=config.LORA_INIT); ap.add_argument('--base_model', default=config.BASE_MODEL)
    ap.add_argument('--sigma', type=float, default=config.SIGMA); ap.add_argument('--steps', type=int, default=config.STEPS)
    ap.add_argument('--seed', type=int, default=config.SEED)
    ap.add_argument('--vlm', help='optional VLM stylist for look recommendation: a local image-text model folder, e.g. Qwen3-VL-4B-Instruct-FP8')
    ap.add_argument('--vlm_python', help='python executable of an environment that can run the VLM (default: this one)')
    a = ap.parse_args()

    config.limit_cpu_threads(a.threads or None)              # before numpy / torch start their thread pools
    os.environ.setdefault('IRISMIX_CACHE_ITEMS', '8')        # interactive use: keep few per-image analyses (about 0.1 GB each)
    from irismix.viewer.engine import Engine
    from irismix.viewer.server import serve
    eng = Engine(a.device, a.keep_models, a.decomposer, a.renderer, a.lora_init, a.base_model, sigma=a.sigma, steps=a.steps,
                 seed=a.seed)
    if not a.keep_models: eng.start_idle_release(a.idle_release)
    eng.auto_preload = not a.no_preload
    stylist = None
    if a.vlm:
        from irismix.viewer.vlm import Stylist
        stylist = Stylist(a.vlm, a.vlm_python)
    try:
        _, httpd = serve(eng, a.host, a.port, a.out, a.examples, stylist)
    except OSError as e:                                      # usually an older viewer still running on the port
        if e.errno != 98: raise
        raise SystemExit(f'\nPort {a.port} is already in use, probably by another IrisMix Studio. Stop it first '
                         f'(Ctrl+C in its terminal, or the stop button in VS Code), or start this one with --port {a.port + 1}.\n')
    url = f'http://localhost:{a.port}'
    line = '─' * 66
    print(f'\n{line}\n  IrisMix viewer\n{line}')
    print(f'  local      {url}')
    if a.host == '0.0.0.0': print(f'  network    http://{socket.gethostname()}:{a.port}')
    print(f'  remote     ssh -L {a.port}:localhost:{a.port} {getpass.getuser()}@{socket.gethostname()}')
    print(f'             then open {url}')
    print(f'  exports    {a.out}')
    print(f'  models     loaded when a photo is uploaded ({"kept on the GPU" if a.keep_models else "one stage at a time"})'
          + ('' if a.keep_models or not a.idle_release else f', GPU memory released after {a.idle_release:g} idle minutes'))
    if a.vlm: print(f'  stylist    {a.vlm}')
    print(f'{line}\n  Ctrl+C to stop\n', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        httpd.server_close()
        from irismix import rules
        rules.close_models()


if __name__ == '__main__':
    main()
