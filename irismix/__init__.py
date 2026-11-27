"""IrisMix: compositional makeup transfer with layered diffusion transformers.

  config         model ids, weight paths, cache directory and the renderer sampler (sigma 0.65, one Euler step, Omega 2 px)
  utils          image I/O with session caches, face crop, Lab, numpy <-> torch
  rules          face parsing, MediaPipe landmarks, piecewise-affine landmark warp and the transfer rules (transfer_layer)
  face_parts     landmark part weights on the 512 face crop
  flux           the one-step FLUX.2-klein generator shared by both models
  decomposer     Decomposer: decomposition of a reference photo into (bare face, makeup layer)
  conditions     Renderer source conditions (with the signed distance maps) and reference-layer caption helpers
  edit_region    design mask D, edit region Omega, reference layer crops
  captions       structured per-part captions
  render_steps   the steps around the renderer: conditions of a look, 512-crop preparation, masked-residual paste-back
  renderer       the renderer (Renderer.sample)
  pipeline       a look = source + per-part references, end to end (used by scripts/infer.py)
  layer_recover  closed-form layer (alpha~, F') of a render
  edits          layer edits (strength, part toggles, recolour, hue rotation) by plain alpha-over
  viewer         interactive browser app (scripts/viewer.py): compose, render and edit looks over an SSH-forwarded port

numpy / OpenCV / PIL at import time; torch, transformers, diffusers and mediapipe are imported when first needed."""

__version__ = '1.0.0'
