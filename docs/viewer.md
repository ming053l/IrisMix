# IrisMix Studio

IrisMix Studio is the interactive viewer of IrisMix. A standard-library HTTP server runs on the GPU machine and serves one
WebGL page. The interface is available in English (default), Traditional Chinese, Japanese and Korean, and the **Guide** button
starts a ten-step tour; the tour also opens once on the first visit.

## Starting

```bash
python scripts/viewer.py --port 7007                       # add --keep_models to keep the networks on the GPU
python scripts/viewer.py --vlm <folder of Qwen3-VL-4B-Instruct-FP8> --vlm_python <python of an env with transformers >= 5>
ssh -L 7007:localhost:7007 <user>@<server>                  # then open http://localhost:7007
```

In VS Code with Remote-SSH, the **IrisMix Studio** launch configuration (`.vscode/launch.json`) starts the server and opens the
forwarded port; *Ports → Preview in Editor* or *Simple Browser: Show* shows it inside the editor.

## Files

| File | Role |
|---|---|
| `irismix/viewer/engine.py` | look engine: target preparation (as is, re-make-up, glasses, bangs), cached decompositions, transfer, rendering, layer recovery, exact edits, kept looks, shades, look recommendation, shade comparison cards |
| `irismix/viewer/server.py` | HTTP API (JSON and PNG) and static files |
| `irismix/viewer/vlm.py` | optional VLM stylist in a worker process |
| `irismix/viewer/static/index.html` | the page: layout, WebGL compositing of the layers and of the closed-form edits, guided tour |
| `irismix/viewer/static/i18n.js` | all interface strings in four languages, including the wording of recommendation reasons and card notes |
| `irismix/shades.py`, `irismix/aesthetics.py`, `irismix/occluder.py` | shade analysis and try-on, recommendation scores, occlusion-aware transfer |

## Look recommendation

Each candidate look is transferred onto the face with the transfer rules only (cached decomposition plus the warp; no rendering, for
speed), and the face crop is scored:

| Term | Computation | Weight |
|---|---|---|
| Aesthetic | LAION improved aesthetic predictor (MLP on CLIP ViT-L/14 image embeddings, 1–10) | 0.40 |
| Preference | CLIP probability of "flattering, harmonious makeup that suits her skin tone" against "unflattering makeup that clashes with her skin tone" | 0.25 |
| Harmony | personal-colour rules below, 0–1 | 0.35 |

Aesthetic and preference are min–max normalised over the candidates on one face, so the photo itself cancels out:
score = 100 (0.4 aesthetic + 0.25 preference + 0.35 harmony).

Harmony uses the skin colour corrected for the scene light (white point from the sclera), its undertone (hue angle below 56°
cool, above 64° warm, neutral between) and its depth (individual typology angle), and the median CIELAB colour of the lips and
eyes of the look's decomposed layer. Lips in the undertone's hue family (cool: below 25° or above 300°; warm: 25–70°) score 1.0,
others 0.45, and 0.85 on neutral skin; deeper skin favours saturated colours. Eyes score 1.0 in the family and 0.7 outside it.
The reasons shown under each card are the outcomes of these rules (for example "rose lips match your cool undertone"), worded
in the selected language. The thresholds are hand-set heuristics, not a validated colour-matching method.

The optional stylist sends the top looks as one labelled grid to a local VLM (Qwen3-VL-4B-Instruct-FP8 is recommended; FP8
weights are dequantized to bf16 when the FP8 kernels are not installed, about 9 GB) and asks for a ranking with one short
reason per look in the selected language.

## Tests

```bash
python -m pytest tests -q                                   # CPU: smoke test, layer algebra, viewer API (about 1 minute)
python -m pytest tests/test_viewer_gpu.py -q -s             # GPU: renders, bangs, re-make-up, recommendation
IRISMIX_TEST_VLM=<vlm folder> IRISMIX_TEST_VLM_PYTHON=<python> python -m pytest tests/test_viewer_gpu.py -q -s
```
