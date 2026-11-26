"""Look recommendation scores: an aesthetic score of the made-up face and a colour-harmony score of the look for that face.

  aesthetic  the LAION improved aesthetic predictor (Apache-2.0; an MLP on CLIP ViT-L/14 image embeddings, 1-10 scale) on
             the face crop, plus a CLIP zero-shot preference for a flattering over a clashing look (both relative across the
             candidates on one face, so the photo itself cancels out)
  harmony    a personal-colour heuristic: the look's lip and eye colours (median CIELAB of the reference layer) against the
             face's undertone and depth (corrected skin colour, irismix.shades): warm undertones favour coral / brick / gold
             hues, cool undertones rose / berry / plum; deeper skin carries richer, more saturated colour
Weights: CLIP from Hugging Face (openai/clip-vit-large-patch14); the predictor MLP is downloaded once into the cache."""
import os
import numpy as np

from irismix import config

MLP_URL = ('https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/'
           'sac%2Blogos%2Bava1-l14-linearMSE.pth')
CLIP_ID = 'openai/clip-vit-large-patch14'
PROMPTS = ('a portrait of a woman wearing flattering, harmonious makeup that suits her skin tone',
           'a portrait of a woman wearing unflattering makeup that clashes with her skin tone')


class Scorer:
    def __init__(self, device='cpu'):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.dev = torch, device
        self.clip = CLIPModel.from_pretrained(CLIP_ID).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID, use_fast=False)
        path = os.path.join(config.CACHE_DIR, 'aesthetic_l14.pth')
        if not os.path.exists(path):
            import urllib.request
            os.makedirs(config.CACHE_DIR, exist_ok=True); urllib.request.urlretrieve(MLP_URL, path + '.part'); os.replace(path + '.part', path)
        nn = torch.nn
        self.mlp = nn.Sequential(nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2), nn.Linear(128, 64),
                                 nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
        sd = torch.load(path, map_location='cpu', weights_only=True)
        self.mlp.load_state_dict({k.replace('layers.', ''): v for k, v in sd.items()}); self.mlp.to(device).eval()
        with torch.no_grad():
            t = self.proc(text=list(PROMPTS), return_tensors='pt', padding=True).to(device)
            e = self.clip.text_projection(self.clip.text_model(**t).pooler_output); self.text = e / e.norm(dim=-1, keepdim=True)

    def __call__(self, imgs):
        """list of uint8 face crops -> (aesthetic 1-10, preference logit) arrays."""
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=list(imgs), return_tensors='pt').to(self.dev)
            e = self.clip.visual_projection(self.clip.vision_model(pixel_values=x['pixel_values']).pooler_output)
            e = e / e.norm(dim=-1, keepdim=True)
            aes = self.mlp(e.float()).squeeze(-1).cpu().numpy()
            pref = (100 * e @ self.text.T).softmax(-1)[:, 0].cpu().numpy()
        return aes, pref


def look_colours(L):
    """median CIELAB of the lips and eyes of a decomposed layer (reference frame), or None where the part is (almost) bare."""
    from irismix.utils import lab
    from irismix import render as RD
    out = {}
    Lab = lab(L['F'])
    for p in ('lips', 'eyes'):
        m = (L['a'] > 0.3) & (L['w'][RD.NAMES.index(p)] > 0.5)
        out[p] = np.median(Lab[m], 0) if m.sum() > 40 else None
    return out


def _hue(c): return float(np.degrees(np.arctan2(c[2], c[1])) % 360)
def _chroma(c): return float(np.hypot(c[1], c[2]))


def name_colour(c):
    """a plain colour word for a CIELAB colour (for the explanation)."""
    h, C, L = _hue(c), _chroma(c), c[0]
    if C < 12: return 'nude' if L > 55 else 'brown' if L > 30 else 'black'
    if h < 15 or h >= 345: w = 'berry' if L < 40 else 'rose'
    elif h < 40: w = 'red' if C > 45 else 'rose'
    elif h < 60: w = 'coral' if L > 50 else 'brick'
    elif h < 85: w = 'gold' if L > 60 else 'bronze'
    elif h < 170: w = 'green'
    elif h < 260: w = 'blue'
    else: w = 'plum' if L < 45 else 'violet'
    return w


def harmony(face, cols):
    """personal-colour harmony in [0, 1] and structured reasons (the viewer words them in the chosen language):
    dict(part='lips', colour, rel='match' | 'warmer' | 'cooler' | 'neutral', ut, rich) and dict(part='eyes', colour,
    rel='flatter' | 'contrast', ut). face: shades.Face (corrected skin, undertone, ITA)."""
    from irismix import shades as SH
    ut, ita = face.undertone(), SH.ita(face.skin); deep = ita < 28
    score, reasons, n = 0.0, [], 0
    lip = cols.get('lips')
    if lip is not None:
        h, C = _hue(lip), _chroma(lip); n += 1
        warm_h = 25 <= h <= 70; cool_h = h < 25 or h > 300
        if ut == 'warm': s = 1.0 if warm_h else 0.45; rel = 'match' if warm_h else 'cooler'
        elif ut == 'cool': s = 1.0 if cool_h else 0.45; rel = 'match' if cool_h else 'warmer'
        else: s = 0.85; rel = 'neutral'
        if deep: s *= 0.7 + 0.3 * min(C / 45, 1)
        else: s *= 1.0 if lip[0] > 32 or C > 25 else 0.8
        score += s; reasons.append(dict(part='lips', colour=name_colour(lip), rel=rel, ut=ut, rich=bool(deep and C > 35)))
    eye = cols.get('eyes')
    if eye is not None:
        h = _hue(eye); n += 1
        if ut == 'warm': s = 1.0 if 20 <= h <= 110 else 0.7
        elif ut == 'cool': s = 1.0 if h < 20 or h > 180 else 0.7
        else: s = 0.9
        score += s; reasons.append(dict(part='eyes', colour=name_colour(eye), rel='flatter' if s >= 0.9 else 'contrast', ut=ut))
    return (score / n if n else 0.6), reasons
