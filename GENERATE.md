# Generation parameters

Reference for every parameter accepted by the three image-producing
subcommands — `generate`, `stylize`, `upscale` — with what each does, how
it affects output, and the ranges that actually work in practice.

For batch generation via `scenario`, the same parameters are accepted as
top-level fields in the HJSON file (with `kebab-case` instead of CLI
`kebab-case` — the names match). The scenario-only fields (`scene`,
`weather`, `tasks`, prompt-assembly headers/footers, `upscale` block) are
documented in [README.md](README.md#scenario-configuration).

---

## `plakat generate`

Single text-to-image inference.

### Required

#### `<PROMPT>` (positional)

What you want generated. Free text — no token limit on the input, but
CLIP/T5 tokenizers truncate to their max context (77 tokens for SD, 256
for Flux schnell, 512 for Flux dev), so the *end* of a very long prompt
is dropped.

**Style**: Stable Diffusion responds well to **comma-separated tags**:
`subject, environment, lighting, medium, style, mood`. Flux handles
**natural language** better — `"a knight standing in a moonlit forest"`
works as well as a tag list.

### Model + dimensions

#### `--model <ALIAS|REPO>` (default `sd15`)

Which base model to use. Accepts a short alias or any HuggingFace repo id.

| Alias | Resolves to | Notes |
|---|---|---|
| `sd15` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 512² native. ~2 GB download. |
| `sd21` | `stabilityai/stable-diffusion-2-1` | 768² native. |
| `sdxl` | `stabilityai/stable-diffusion-xl-base-1.0` | 1024² native. ~7 GB. |
| `sdxl-turbo` | `stabilityai/sdxl-turbo` | 512–1024. Requires `--steps 4 --guidance 0`. |
| `flux-schnell` | `black-forest-labs/FLUX.1-schnell` | 1024² typical. 4 steps. ~31 GB. |
| `flux-dev` | `black-forest-labs/FLUX.1-dev` | Gated — needs `HF_TOKEN`. 20–50 steps. |

Custom HF repos: pass the full `org/name`. For SD-family the repo must
have the diffusers layout (`unet/`, `vae/`, `text_encoder[_2]/`,
`tokenizer[_2]/`). For Flux variants we expect the BFL-native single-file
layout (`flux1-{schnell,dev}.safetensors` + `ae.safetensors`).

#### `--size <WxH>`

Output dimensions. Must be multiples of 8 (16 for Flux). Examples:
`512x512`, `768x512`, `1024x1024`.

**Effect on quality**: each model was trained at a specific resolution.
Generating outside the trained band introduces specific artifacts:

- Too small → blurry, low detail, sometimes simplified subjects.
- Too large → duplication (two heads, two faces, repeated structure), or
  "ribboning" at boundaries.

Stay near each model's native resolution unless you have a reason not to.

#### `--aspect <N:M>` + `--base <SIDE>` (alternative to `--size`)

Specify an aspect ratio (`16:9`, `2:3`, `1:1`) and a base resolution
(default `768`). The shorter side becomes `--base`, the longer side scales
to match the ratio (rounded down to a multiple of 8).

Use `--aspect 16:9 --base 1024` for cinematic SDXL, etc. Exclusive with
`--size`.

#### `--count <N>` / `-n <N>` (default `1`)

How many images to generate from this prompt. Each gets a unique seed
(`seed`, `seed+1`, …). Within one invocation, the model is loaded once
and shared across all `N` images — cheap to bump.

### Sampling

#### `--steps <N>` (default `28`)

Denoising steps. Each step refines the latent further toward the prompt.

| Model | Sweet spot |
|---|---|
| SD 1.5 / 2.1 | 25–35 |
| SDXL | 25–40 |
| SDXL-Turbo | **4** (hard requirement of the trained schedule) |
| Flux schnell | **4** (rectified flow, very few steps suffice) |
| Flux dev | 20–50 |

Diminishing returns past the sweet spot. 50 steps rarely beat 30. With
LCM-LoRA on SD 1.5, 4–8 steps is enough.

#### `--guidance <FLOAT>` (default `7.5`)

Classifier-free guidance scale. Controls prompt adherence vs.
naturalness. Higher = follows prompt more strictly but with more "burn"
(oversaturated, plastic-looking, ringing artifacts).

| Model | Sweet spot |
|---|---|
| SD 1.5 / 2.1 | 7.0–9.0 |
| SDXL | 5.0–7.5 (lower than 1.5) |
| SDXL-Turbo | **0.0** (model was trained without CFG; non-zero breaks output) |
| Flux schnell | **1.0** (no CFG) |
| Flux dev | 3.0–4.5 |

If output looks oversaturated or melted, drop guidance by 1.0. If
composition is vague and ignoring details in the prompt, raise it.

#### `--negative <TEXT>` (default empty)

Negative prompt — what you *don't* want. The model's CFG pushes the
output away from this. A baseline that fixes 50% of bad outputs:

```
blurry, low quality, deformed, ugly, watermark, text, signature, jpeg artifacts, oversaturated, bad anatomy, extra limbs
```

**SDXL-Turbo / Flux ignore the negative prompt** because they don't use
CFG.

#### `--seed <U64>` (default random)

Random seed. Same seed + same prompt + same model = same image. Use this
to:
- Iterate prompts deterministically (find one composition you like, vary
  prompt at the same seed).
- Reproduce a result you already generated.
- Sweep over seeds (`--seed 0 -n 4` tries 0, 1, 2, 3).

On Metal, the seed is masked to `u32` because candle's Metal RNG accepts
only `u32`. CPU device doesn't support seeding at all — generations are
non-deterministic there (warned via debug log).

#### `--scheduler <KIND>` (default `default`)

Numerical solver for the reverse diffusion ODE.

| Choice | When |
|---|---|
| `default` | Use the variant's built-in (DDIM for SD 1.5/2.1/SDXL, Euler-A for SDXL-Turbo). |
| `ddim` | Deterministic baseline. Reproduces exactly given a seed. |
| `euler-a` | Euler-Ancestral. **Often higher quality at the same step count** for SD 1.5 / SDXL. Mildly stochastic (varies even with the same seed across runs). |
| `unipc` | DPM-Solver++ family. **CUDA / CPU only** — candle's UniPC uses F64 ops Metal doesn't implement. |

If unsure: try `euler-a`. It's almost always at least as good as DDIM
for the same step count.

#### `--refine <N>` + `--refine-strength <FLOAT>` (defaults: off, 0.3)

After the main denoise loop completes, run `N` extra steps of img2img on
the produced latents using the **same base model**. Sharpens details and
removes some artifacts.

`--refine-strength` controls how much fresh noise gets re-added before
the polish loop:

- `0.0` → no effect.
- `0.2–0.4` → subtle sharpening, recommended.
- `0.5+` → significant re-rendering, can change subject features.

This is the **same-model polish pass** — distinct from the real refiner
below. Useful for SD 1.5 / 2.1 (no separate refiner exists for those) or
when you want extra detail without the 6 GB refiner download.

#### `--refiner` + `--refiner-frac <FLOAT>` (defaults: off, 0.8)

Use the **official SDXL refiner** (`stabilityai/stable-diffusion-xl-refiner-1.0`)
for the last fraction of the schedule. The base SDXL UNet handles the
first `frac×N` steps; the refiner UNet handles the remaining
`(1−frac)×N` on the same latents.

- **`--refiner-frac 0.8`** (default) — last 20% of steps run on refiner.
  Diffusers reference uses this.
- Lower values (0.6) give the refiner more responsibility — different
  trade-off, sometimes better for portraits / fine textures.
- Higher values (0.9) keep more of the base's composition.

SDXL / SDXL-Turbo only — errors on SD 1.5/2.1 or Flux. Adds **~6 GB
download** for the refiner UNet on first run.

**Known limitation in plakat's refiner port** — candle 0.8's UNet has no
`add_embedding` projection, so the refiner's pooled-CLIP-G + time_ids
micro-conditioning is unused. The refiner still runs and produces
recognizably better output than base alone, but the gap to the diffusers
reference is real (~70–90% of reference quality). The same gap already
applies to plakat's base SDXL.

`--refiner` and `--refine` are independent — you can stack both (refiner
for the last 20%, then a polish pass on top).

### LoRA

#### `--lora <SPEC>` (repeatable)

Apply a LoRA. SD-family only — Flux ignores LoRAs with a warning.

The `<SPEC>` accepts three forms:

| Form | Meaning |
|---|---|
| `./local/file.safetensors` | Local path. |
| `org/repo` | HF repo. plakat picks the `.safetensors` (canonical names first, then largest). |
| `org/repo#sub/path.safetensors` | HF repo, explicit file. |

Append `:0.7` for per-LoRA scale: `--lora foo.safetensors:0.7`. Multiple
`--lora` flags stack.

**Quality dependencies**:
- The LoRA must target the same base model as `--model`. An SDXL LoRA on
  SD 1.5 will mismatch every cross-attention layer; plakat detects this,
  prints the right "use `--model sdxl`" message, and either errors (if
  zero targets match) or warns (if some match).
- Most LoRA trigger tokens need to appear in the prompt to activate. Read
  the LoRA's README on HF.

**Memory**: peak ~3.4 GB on SD 1.5 during merge, ~10 GB on SDXL. Merge
happens once per `generate` call.

**Targets merged**: plakat runs the merge against each base-model
component the LoRA touches:

- **UNet** (always)
- **CLIP-L text encoder** (when the file has `lora_te_*` or `lora_te1_*` keys, or PEFT `text_encoder.*` keys)
- **CLIP-G text encoder** (SDXL only, when the file has `lora_te2_*` or PEFT `text_encoder_2.*` keys)

Each pass reports a separate `merged X/Y targets` line — so a civitai
LoRA with both UNet and text-encoder targets shows two lines,
e.g.:

```
LoRA … → UNet: 192/192 targets merged
LoRA … → text encoder: 72/72 targets merged
```

**Formats recognized**:

| Format | Detection key(s) | Math |
|---|---|---|
| Standard LoRA / LoCon / DyLoRA | `lora_down` + `lora_up` (kohya), `lora_A` + `lora_B` (PEFT) | `W ← W + (α/rank) · (B · A) · scale` |
| DoRA | LoRA keys + `dora_scale` (per-row magnitude vector) | `W ← scale · (W + ΔW) / rowwise_L2(W + ΔW)` |
| LyCORIS LoHa | `hada_w1_a/b` + `hada_w2_a/b` | `W ← W + (W1_b·W1_a) ⊙ (W2_b·W2_a) · α/rank` |
| LyCORIS LoHa (Tucker) | LoHa keys + `hada_t1` + `hada_t2` | `W ← W + tucker(t1,a1,b1) ⊙ tucker(t2,a2,b2) · α/rank` |
| LyCORIS LoKr | `lokr_w1` (or `_a`+`_b`) + `lokr_w2` (or `_a`+`_b`) | `W ← W + kron(W1, W2) · α/dim` |

DyLoRA stores as standard LoRA at inference time (its "dynamic rank" is a
training-time feature), so it's covered by the standard arm with no
extra detection.

Conv weight shapes handled automatically: 2D Linear, 1×1 conv, and 3×3
conv (LCM-LoRA exercises all three). LoKr w2 conv weights are flattened
along trailing dims for the Kronecker reshape. `base_model.model.` /
`diffusion_model.` prefixes are stripped before matching.

Tucker LoHa uses two matmul contractions to evaluate the einsum
`"r1 r2 kh kw, r2 in, out r1 -> out in kh kw"` (the two ranks `r1`/`r2`
are both `lora_dim` in practice; the spatial dims pass through). Only
emitted by LyCORIS when training on conv layers with non-1 kernels — 2D
Linear targets don't ship a Tucker form.

#### `--lora-scale <FLOAT>` (default `1.0`)

Multiplier applied to every LoRA's per-file scale. So
`--lora foo:0.8 --lora-scale 0.5` runs `foo` at effective 0.4.

### Prompt enhancement

#### `--enhance <PROVIDER>`

Pass the prompt through an LLM (DeepSeek or Gemini) that rewrites it
with concrete visual detail (composition, lighting, medium, style) before
generation.

```
plakat generate "knight" --enhance deepseek
```

| Provider | Env var |
|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |

The enhancer is the cheap-but-effective way to compensate for a model
whose prompt comprehension is weaker than you'd like (SD 1.5 / 2.1 in
particular).

### Output

#### `--out <DIR>` (default `./out`)

Directory for generated images. Created if absent. Files are named
`plakat-<seed>.png` (or `plakat-flux-<seed>.png` for Flux).

---

## `plakat stylize`

IP-Adapter style transfer: take an input image and a reference image,
produce an output that keeps the input's content with the reference's
style. SD 1.5 base only.

#### `--in <IN>` (required)

Input image (the thing whose content you want to keep). Any common
format (PNG, JPEG, WebP). Aspect ratio is preserved; dims are rounded to
multiples of 8 and capped at 768.

#### `--ref <REF>` (required)

Style reference image. plakat resizes it to 224×224 and CLIP-normalizes
internally — high resolution isn't required.

#### `--out <OUT>` (required)

Output PNG path. Created (with parent dirs) if absent.

#### `--strength <FLOAT>` (default `0.7`)

How much to redraw IN. Higher = more like REF, less like IN.

- `0.0` → no change to IN (output ≈ input).
- `0.3` → subtle restyling, IN mostly preserved.
- `0.6` → balanced — recognizable IN with REF's style.
- `0.9` → heavy restyling, IN may lose recognizable detail.

#### `--steps <N>` (default `30`)

Denoising steps for the img2img pass. 20–40 is the useful range.

#### `--seed <U64>` (default random)

Same semantics as `generate`. Reproducible given the same IN + REF +
strength + seed.

#### `--model <ALIAS|REPO>` (default `sd15`)

Base diffusion model for the denoise loop. **Currently SD 1.5 only.** The
IN image's original generator can be anything (SDXL, Flux, a photo) —
stylize operates on image bytes, not on latents from another model.

### What stylize is and isn't

Stylize uses IP-Adapter's image projection (the `image_proj.*` weights
from `h94/IP-Adapter`) to map the CLIP-H image embedding of REF into the
text-token space, then concats those tokens onto the (empty) text prompt
for the UNet.

The **reference IP-Adapter** also adds decoupled cross-attention (separate
`to_k_ip`/`to_v_ip` projections per UNet layer) — this isn't wired in
plakat because candle's UNet has no attention hook. Quality is roughly
50–70% of the reference implementation.

For consistent results: REF should have a clear visual style (painting,
sketch, distinctive photo). REFs that are themselves stylistically neutral
won't push IN very far.

---

## `plakat upscale`

Classical image upscaling — no ML model, no weight download.

#### `--in <IN>` (required)

Source image. Any common format.

#### `--out <OUT>` (required)

Output path. Extension determines format (`.png`, `.jpg`, `.webp`).

#### `--scale <FLOAT>` (default `2.0`)

Scale factor. Non-integer values OK: `--scale 1.5`, `--scale 2.5`. The
new dimensions are `round(orig × scale)`.

#### `--method <FILTER>` (default `lanczos`)

Two families: **classical** filters (pure CPU resize) and **ML** models
(RRDBNet / Real-ESRGAN).

**Classical** — fast, predictable, no weights:

| Filter | Quality | Best for |
|---|---|---|
| `nearest` | Lowest — pixelated | Pixel art, retro look |
| `bilinear` | Blurry edges | Thumbnails |
| `bicubic` | Decent, slight softening | General-purpose |
| `lanczos` | Sharpest detail preservation, slight ringing on hard edges | Photographic / detailed images |

**ML — Real-ESRGAN** (RRDBNet ported to candle):

| Method | Scale | Variant | Weights | Use case |
|---|---|---|---|---|
| `real-esrgan-x2` | ×2 | xinntao x2plus | ~17 MB | Subtle resolution boost, photographic |
| `real-esrgan-x4` | ×4 | xinntao x4plus | ~17 MB | Standard 4× upscale, recovers fine detail |
| `real-esrgan-anime-x4` | ×4 | xinntao x4plus_anime_6B | ~9 MB | Line art / anime / illustration |

ML methods:
- **Ignore `--scale`** — the model's architecture fixes the factor.
- Download weights from `hlky/RealESRGAN_*` on first use (small, ~17 MB max).
- Run on the device chosen by `--device` (Metal/CUDA/CPU). Memory scales with the **output** size — a 4× upscale of 1024² = 4096² requires room for that 64 MB output tensor in F32 plus intermediates.

For aggressive upscales beyond 4× (e.g. 8×), pipe through twice
(`plakat upscale ... --method real-esrgan-x4 ... && plakat upscale ...
--method real-esrgan-x2 ...`). Diminishing returns past that — ESRGAN
models weren't trained for cascaded use.

---

## Global flags (every subcommand)

#### `-v` / `-vv` (default off)

Increase log verbosity. `-v` enables debug logs from plakat; `-vv` adds
trace.

#### `--device <SPEC>` (default `auto`)

| Spec | Meaning |
|---|---|
| `auto` | Try CUDA, then Metal, then fall back to CPU. |
| `cuda` / `cuda:N` | NVIDIA GPU N. Requires building with `--features cuda`. |
| `metal` | Apple Silicon GPU. Requires `--features metal`. |
| `cpu` | CPU. Works everywhere. Slow on full-size generations. |

#### `--cache-dir <PATH>` (env: `PLAKAT_CACHE_DIR`)

Where HF model weights are cached. Resolution order:
1. `--cache-dir` flag
2. `PLAKAT_CACHE_DIR` env var
3. `HUGGINGFACE_HUB_CACHE` env var
4. `HF_HOME` env var (with `/hub` appended)
5. Default `~/.cache/huggingface/hub`

---

## Common workflows

### Iterate on a prompt at fixed composition

```bash
plakat generate "a tranquil koi pond" \
    --model sdxl --size 1024x1024 \
    --steps 30 --scheduler euler-a \
    --seed 42 -n 1
# tweak prompt, keep --seed 42 — same composition, different details
```

### Find a good composition then refine

```bash
# Step 1 — sample 4 seeds
plakat generate "a tranquil koi pond" --model sdxl --size 1024x1024 \
    --seed 0 -n 4

# Step 2 — pick the best, refine it
plakat generate "a tranquil koi pond" --model sdxl --size 1024x1024 \
    --steps 35 --scheduler euler-a --refine 8 --refine-strength 0.25 \
    --seed 2
```

### Style transfer at high fidelity to IN

```bash
plakat stylize \
    --in photo.jpg \
    --ref reference_painting.jpg \
    --out styled.png \
    --strength 0.4 \
    --steps 30
```

### Generate → stylize → upscale (one pipeline via scenario)

See [README.md](README.md#scenario-configuration) — the `scenario`
subcommand runs this chain over many task definitions and shares the
loaded model across all of them.
