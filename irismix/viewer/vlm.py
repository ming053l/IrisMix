"""Optional VLM stylist for the viewer's look recommendation (e.g. a local Qwen VLM checkpoint; off unless --vlm is given).

The model runs in a separate worker process (python -m irismix.viewer.vlm --worker), possibly from another environment with
a newer transformers (--vlm_python), so the viewer itself needs no extra dependencies. The worker reads one JSON request per
line ({"image": path, "prompt": text}) and answers one JSON line ({"text": ...} or {"error": ...})."""
import json, os, subprocess, sys

PROMPT = ('The image shows the same person with {n} candidate makeup looks, labelled 1 to {n}. As a professional makeup '
          'artist, choose the three looks that suit her best (skin tone, features, harmony, how flattering and wearable they '
          'are). Write each reason as one short sentence in {language}. Answer only with JSON: '
          '{{"ranking": [{{"look": <number>, "reason": "<reason>"}}, ...]}}.')
LANGUAGES = {'en': 'English', 'zh': 'Traditional Chinese (Taiwan)', 'ja': 'Japanese', 'ko': 'Korean'}


class Stylist:
    def __init__(self, model, python=None):
        self.model, self.python = model, python or sys.executable
        self.p = None

    def ask(self, image_path, n, lang='en'):
        if self.p is None or self.p.poll() is not None:
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                                               os.environ.get('PYTHONPATH', '')]))
            self.p = subprocess.Popen([self.python, '-m', 'irismix.viewer.vlm', '--worker', self.model], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, env=env)
        try:
            self.p.stdin.write(json.dumps(dict(image=image_path, prompt=PROMPT.format(n=n, language=LANGUAGES.get(lang, 'English')))) + '\n'); self.p.stdin.flush()
        except OSError as e:
            raise ValueError(f'the VLM worker exited: {e}') from e
        r = json.loads(self.p.stdout.readline() or '{"error": "the VLM worker exited"}')
        if 'error' in r: raise ValueError(r['error'])
        t = r['text']; t = t[t.find('{'):t.rfind('}') + 1]
        return json.loads(t)['ranking']

    def close(self):
        if self.p is not None and self.p.poll() is None: self.p.kill()


def worker(model):
    reply, sys.stdout = sys.stdout, sys.stderr                # library prints must not corrupt the JSON replies
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model)
    kw = {}
    with open(os.path.join(model, 'config.json')) as f: q = json.load(f).get('quantization_config', {})
    if q.get('quant_method') == 'fp8':                         # FP8 checkpoints: dequantize to bf16 when the FP8 kernels are missing
        try:
            import kernels  # noqa: F401
        except ImportError:
            from transformers import FineGrainedFP8Config
            kw['quantization_config'] = FineGrainedFP8Config(dequantize=True)
    net = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.bfloat16, device_map='auto', **kw).eval()
    for line in sys.stdin:
        try:
            q = json.loads(line)
            msgs = [dict(role='user', content=[dict(type='image', image=Image.open(q['image']).convert('RGB')), dict(type='text', text=q['prompt'])])]
            x = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt',
                                         enable_thinking=False).to(net.device)
            with torch.no_grad():
                y = net.generate(**x, max_new_tokens=400, do_sample=False)
            out = dict(text=proc.decode(y[0, x['input_ids'].shape[1]:], skip_special_tokens=True))
        except Exception as e:
            out = dict(error=f'{type(e).__name__}: {e}')
        reply.write(json.dumps(out) + '\n'); reply.flush()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker': worker(sys.argv[2])
