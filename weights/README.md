# IrisMix weights

LoRA adapters (rank 32) and skip-connection weights for
[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). The base model is not included; it is
downloaded from Hugging Face on first use.

| file | role | tensors | size (bytes) | sha256 |
|---|---|---|---|---|
| `irismix_lora_init.safetensors` | initial LoRA: creates the rank-32 LoRA modules (diffusers / PEFT format) | 120 LoRA | 66,866,216 | `6f05958ff2e5735de51acce6f2125c18246bf97c3e11c3e6eb87653124ea8786` |
| `irismix_decomposer.safetensors` | Decomposer, the reported model: transformer LoRA + skip convs `skips.E.*` (decompose) and `skips.H.*` (harmonise; not used at inference) | 120 LoRA + 12 skip | 68,709,512 | `efa679e478c6087a033a3a168d7e40dfcc621d53e63e2e25dec6b02aea19f0c8` |
| `irismix_renderer.safetensors` | Renderer, the reported model: bias-corrected EMA of the render LoRA + skip convs `skips.z.*` | 120 LoRA + 6 skip | 67,786,352 | `1d4b1651bdd1888426ae71ee3f76920f94f5f18b95457e81d09a4420d967de5b` |

LoRA tensor names follow the loaded PEFT modules (`transformer_blocks.<i>.attn.to_q.lora_A.default_0.weight`, ...);
`irismix.flux.load_lora_ckpt` copies them into the modules created from the initial LoRA and checks that all 120 match.
Dtypes: LoRA bf16, skip convs fp32. The files were converted from the training checkpoints without changing any value
(tensor-by-tensor equality checked).

Download:

    huggingface-cli download ming0531/IrisMix --local-dir weights

or point the scripts elsewhere with `--decomposer / --renderer / --lora_init` or `IRISMIX_WEIGHTS=/path/to/weights`.

Verify:

    sha256sum weights/*.safetensors

Licence of the weights: **CC BY-NC 4.0** (https://creativecommons.org/licenses/by-nc/4.0/), non-commercial use only. The
face parser required at inference is itself non-commercial (see NOTICE).
