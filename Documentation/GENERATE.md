# Generation parameters

Reference for every parameter accepted by the three image-producing
subcommands — `generate`, `stylize`, `upscale` — with what each does, how
it affects output, and the ranges that actually work in practice.

For batch generation via `scenario`, the same parameters are accepted as
top-level fields in the HJSON file (with `kebab-case` instead of CLI
`kebab-case` — the names match). The scenario-only fields (`scene`,
`weather`, `tasks`, prompt-assembly headers/footers, `upscale` block) are
documented inline in the annotated example
[`examples/scenario.hjson`](../examples/scenario.hjson). A runnable
batch demo with artefact placement also lives under
[`examples/tutorials/ZONES/scenario.hjson`](../examples/tutorials/ZONES/scenario.hjson).

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
| `flux-fill-dev` | `black-forest-labs/FLUX.1-Fill-dev` | BFL's dedicated Flux inpaint checkpoint. Driven via `plakat img2img --mask`. |
| `flux-canny-dev` | `black-forest-labs/FLUX.1-Canny-dev` | BFL "concept" Flux with canny conditioning baked into `img_in` (128 channels = 64 noise + 64 canny latent). Pass the canny map via `--concept-image PATH`. Recommended guidance ~30. Gated. |
| `flux-depth-dev` | `black-forest-labs/FLUX.1-Depth-dev` | BFL "concept" Flux with depth-map conditioning. Same shape as Canny-dev. Pass the depth map via `--concept-image PATH`. Gated. |
| `flux-dev-gguf` | `city96/FLUX.1-dev-gguf` | 4-bit quantized FLUX.1-dev. ~7 GB transformer (vs ~24 GB BF16). Pair with `--quant-level` to pick precision. |
| `flux-schnell-gguf` | `city96/FLUX.1-schnell-gguf` | 4-bit quantized FLUX.1-schnell. |
| `flux-fill-dev-gguf` | `city96/FLUX.1-Fill-dev-gguf` | 4-bit quantized Flux Fill. |
| `flux-dev-nf4` | `lllyasviel/flux1-dev-bnb-nf4-v2` | NF4 (bitsandbytes 4-bit) quantization. ~6 GB transformer. Composes with `--loras`. |
| `sd35-medium` | `stabilityai/stable-diffusion-3.5-medium` | Stable Diffusion 3.5 Medium. 2.5B-param MMDiT. Gated. |
| `sd35-large` | `stabilityai/stable-diffusion-3.5-large` | SD3.5 Large flagship. 8B-param MMDiT. Gated. |
| `sd35-large-turbo` | `stabilityai/stable-diffusion-3.5-large-turbo` | 4-step distillation of SD3.5 Large. `--steps 4 --guidance 0`. Gated. |
| `sd3-medium` | `stabilityai/stable-diffusion-3-medium` | original SD3 Medium (June 2024). Superseded by 3.5. Gated. |

Custom HF repos: pass the full `org/name`. For SD-family the repo must
have the diffusers layout (`unet/`, `vae/`, `text_encoder[_2]/`,
`tokenizer[_2]/`). For Flux variants we expect the BFL-native single-file
layout (`flux1-{schnell,dev}.safetensors` + `ae.safetensors`). GGUF
repos ship a single `flux1-{variant}-{LEVEL}.gguf` file; the matching
`ae.safetensors` + text encoders come from the original BFL repo
("donor") on first run.

#### `--quant-level <LEVEL>` (default `Q4_K_S`)

GGUF quant level for the Flux transformer when running a `flux-*-gguf`
model. city96 publishes Q2_K through Q8_0 plus F16. Common picks:

| Level | Approx footprint | Notes |
|---|---|---|
| `Q3_K_S` | ~5.5 GB | Tightest VRAM; noticeable quality drop. |
| `Q4_K_S` | ~7 GB | Default; balanced. |
| `Q5_K_M` | ~8.5 GB | Sweet spot for quality/memory tradeoff. |
| `Q8_0` | ~13 GB | Near-BF16 quality at half memory. |
| `F16` | ~24 GB | Equivalent to BF16. |

Ignored on BF16 Flux and SD-family models.

#### `--quantize-t5` (default off)

Load T5-XXL via city96's GGUF mirror instead of the BF16 sharded
safetensors. Drops T5 from ~10 GB to ~3 GB (Q4_K_M default).
`--t5-quant-level Q5_K_M` picks a higher level. Requires `--model
flux-*-gguf` (bails loud otherwise — pairing quantized T5 with a BF16
transformer wastes T5 quality without unlocking the memory budget
that needs it).

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
| `unipc` | UniPC corrector with Karras sigmas — DPM-Solver++ family. Predictor-corrector tends to be smooth at low step counts. **CUDA / CPU only** (Metal-guarded). |
| `dpmpp-2m` | DPM-Solver++ 2M Karras — same UniPC class with corrector disabled. Crisper edges than `unipc` at the same step count; widely considered an A1111/ComfyUI "safe default". **CUDA / CPU only**. |
| `unipc-exp` | UniPC with exponential sigma schedule (vs Karras). Different noise-step distribution; sometimes better at very low step counts. **CUDA / CPU only**. |
| `lcm` | LCM consistency-function scheduler. **Pair with an LCM-LoRA** (e.g. `latent-consistency/lcm-lora-sdv1-5`) at 4–8 steps with `--guidance 1.0–2.0`. Pure F32 math — works on Metal/CUDA/CPU. `--steps` must be ≤ 50. |
| `euler` | Deterministic Euler. Same algorithm as `euler-a` minus the per-step noise injection. Reproducible across runs given a seed. Pure F32 — works on Metal/CUDA/CPU. |
| `heun` | Heun second-order predictor-corrector. **2× the UNet evaluations** per `--steps` value (interleaved predictor + corrector). Quality typically beats Euler at the same number of model calls. Pure F32 — works on Metal/CUDA/CPU. |
| `ddpm` | The original DDPM. Slow (best at high step counts, since each step is a single Markov transition). Mainly useful as a reference. Pure F32 — works on Metal/CUDA/CPU. |

If unsure on Metal: `euler-a` (stochastic) or `euler` (deterministic).
On CUDA/CPU: `dpmpp-2m` is a strong all-purpose default. With LCM-LoRA
at low step counts: `lcm`. For maximum quality at moderate step count
where 2× UNet evaluations is fine: `heun`.

> Naming note: `--scheduler euler` means **deterministic** Euler, matching
> A1111/ComfyUI convention. The aliases `euler-a`, `eulera`, and
> `euler-ancestral` all hit Euler-Ancestral.

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

Apply a LoRA. Supported on SD-family (SD 1.5 / 2.1 / SDXL / SDXL-Turbo),
Flux (BF16 / GGUF / NF4), and **SD3 / SD3.5**.

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

### Tiled hi-res generation

For outputs above the model's trained working resolution (4K SDXL,
2K–4K Flux) without OOM, use MultiDiffusion-style tiled denoise:

| Flag | Default | Description |
|---|---|---|
| `--tiled` | off | Enable tiled denoise. The transformer/UNet only ever sees `--tile-size` worth of tokens per call; per-step noise predictions are blended via a 2D Hann window. |
| `--tile-size <PX>` | `1024` | Tile side length in pixels. Default matches SDXL's native and Flux's working scale. Must be a multiple of 8 (SD) or 16 (Flux + SD3). |
| `--tile-stride <PX>` | `768` | Stride between tile origins. Smaller = more overlap = smoother seams + more compute. |

Composes with: GGUF + LoRA + img2img on Flux; ControlNet on
SDXL and Flux (each tile gets its CN conditioning cropped to its
region); **Flux.1-Fill-dev** (per-tile masked-latent + mask
packing); **SD3 / SD3.5 img2img + inpaint** (the rectified-flow
init lerp + RePaint mask blend compose with the per-tile
velocity blend). Does **not** compose with the SDXL refiner,
Flux concept variants (Canny-dev / Depth-dev), SD3 ControlNet,
or `--hires-fix`.

SD3 / SD3.5 join the tiled lineup. MMDiT's `pos_embed_max_size`
caps the patched tile dim at 192 (SD3 / SD3.5-Large) or 384
(SD3.5-Medium); the default 1024-px tile patches to 64×64, well
within either cap. Each per-step prediction is the post-CFG
velocity per tile, Hann-blended into a full-canvas update applied
via the Euler step.

Inpaint + tiled note: tile seams can become visible near sharp
mask boundaries because the Hann blend doesn't know about the
mask. Pass `--mask-feather PX` to smooth the transition.

```bash
# 4K SDXL
plakat generate "ultra-detailed architectural diagram" \
 --model sdxl --size 3072x2048 --tiled

# 2K Flux with depth-guided structure across every tile
plakat generate ".." --model flux-dev --size 2048x2048 \
 --tiled --control-spec 'depth:from=ref.jpg'

# 2K SD3.5-Medium
plakat generate ".." --model sd35-medium --size 2048x2048 \
 --tiled --tile-size 1024 --tile-stride 768
```

### Artefact compositing

`plakat generate` (and `plakat portrait`) accept three related flags
for placing named PNG cutouts into the generated image:

| Flag | Purpose |
|---|---|
| `--artefact NAME[@ZONE[:SCALE]]` (repeatable) | Alpha-composite a named cutout from the library. |
| `--artefact-library <DIR>` | Override the bundled library path. |
| `--artefact-blend` / `--artefact-blend-strength F` | v2 — masked low-strength img2img pass to soften edges (~2–5 s on GPU). |
| `--smart-zones` | v3 — derive zones from each image's depth + luminance instead of the rigid 4×3 grid. Requires Depth-Anything-V2 (~99 MB, downloaded on first use). |

Full reference: [`ARTEFACTS.md`](ARTEFACTS.md). Runnable end-to-end
walkthrough: [`examples/tutorials/ZONES/`](../examples/tutorials/ZONES/).

---

### `--redux-image <SPEC>`

Repeatable image-conditioning input for Flux Redux. Each spec is a
path (weight = 1.0) or `path:weight=F.F` (custom weight; 0.0 turns
the image off, ≤2.0 typical range). Up to 4 references; the
attention cost grows quadratically with the seq length so 1–2 is
typical.

```bash
# Single ref
plakat generate "in this style" --model flux-dev \
 --redux-image style.png

# Multi-ref with weights
plakat generate ".." --model flux-dev \
 --redux-image style.png:weight=0.8 \
 --redux-image subject.png:weight=0.4
```

Composes with: BF16 / GGUF / NF4 Flux, LoRA, ControlNet, img2img,
tiled denoise. **Does not** compose with `flux-fill-dev` (Fill's
384ch `img_in` is incompatible).

### `--concept-image <PATH>` / `--concept-from <PATH>`

Conditioning map for Flux.1-Canny-dev / Flux.1-Depth-dev. Pass a
canny edge map (with `--model flux-canny-dev`) or a depth map (with
`--model flux-depth-dev`) at the target output resolution. The image
is VAE-encoded and packed alongside the noise tokens — the "concept"
checkpoint's `img_in` Linear is 128 channels wide (64 noise + 64
conditioning).

Two ways to supply the map:

* **`--concept-image PATH`** — a pre-rendered canny / depth map you
 already have on disk.
* **`--concept-from PATH`** — a source photo to auto-annotate. Plakat
 runs Canny edge detection (for `flux-canny-dev`) or
 Depth-Anything-V2 (for `flux-depth-dev`) on the source, writes the
 result to a temporary PNG, and feeds it to the model the same way
 `--concept-image` would.

```bash
# Auto-annotate a reference photo with the matching annotator
plakat generate "a Victorian mansion, gothic, twilight" \
 --model flux-canny-dev --concept-from photo.jpg --guidance 30

plakat generate "a polished marble statue of an angel" \
 --model flux-depth-dev --concept-from photo.jpg --guidance 30

# Or supply a pre-rendered map
plakat generate ".." --model flux-canny-dev \
 --concept-image edges.png --guidance 30
```

The two flags are mutually exclusive. `--concept-from` is only valid
with the two concept variants — passing it with `flux-dev` or any
other model raises an explicit error.

Caveats:
* Doesn't compose with `--tiled`, `--init-image` / `--mask`,
 `--redux-image`, or `--control-spec` — pairing raises an explicit
 error.
* `--guidance 30` is BFL's recommendation per their model cards.
 Lower values (3-7) underrespect the conditioning; higher
 oversharpen.

### `--fast <PRESET>`

Bundles a published distillation LoRA + recommended step + guidance
in one flag. Presets:

* `hyper-8` — ByteDance Hyper-FLUX 8-step (CFG-free)
* `hyper-16` — ByteDance Hyper-FLUX 16-step (CFG-free)
* `turbo-alpha` — alimama-creative FLUX.1-Turbo-Alpha 8-step

```bash
plakat generate ".." --model flux-dev --fast hyper-8
```

The preset LoRA gets prepended to `--loras`; `--steps` / `--guidance`
are overridden **only** when you didn't pass them explicitly.
Requires a non-Fill Flux model.

### Wildcards

Inline alternation + file-backed random picks in the prompt.
Auto1111 / NovelAI / ComfyUI grammar.

| Flag | Default | Description |
|---|---|---|
| `--wildcard-dir <DIR>` | (none) | Directory holding `<name>.txt` wildcard files for `__name__` expansion. |

* **Inline alternation**: `{red|blue|green}` picks one of the three
 at random. Nestable: `{a {b|c}|d}` → `a b`, `a c`, or `d`. Works
 without `--wildcard-dir`.
* **File wildcards**: `__name__` reads
 `<wildcard-dir>/<name>.txt` and picks a uniformly-random
 non-empty, non-comment (`#`) line. Names accept letters, digits,
 `-`, and `_` (so `__warm-colors__` and `__warm_colors__` both
 work).

```bash
plakat generate "a {red|blue|green} {fox|cat|owl}" \
 --model sd15 --count 4 --seed 42

mkdir -p wildcards
echo -e "ruby\ncrimson\namber" > wildcards/warm-colors.txt
plakat generate "a __warm-colors__ fox" \
 --wildcard-dir ./wildcards --model sd15
```

The wildcard RNG is seeded from `--seed` when set (reproducible
expansion) and from OS entropy otherwise. Expansion runs **before**
`--enhance` so the enhancer sees a concrete prompt.

### CLIP-skip (SD 1.5 / SD 2.1)

| Flag | Default | Description |
|---|---|---|
| `--clip-skip <N>` | `1` | Use the N-th-from-last CLIP-L hidden state. `1` = last (diffusers default; byte-identical to). `2` = penultimate (community default for SD 1.5 anime checkpoints). |

SDXL ignores `--clip-skip > 1` with a warning (the dual-encoder
path already uses penultimate by training default). Flux / SD3
ignore the flag entirely.

### ADetailer face refinement (SD-family)

After the t2i pass, runs SCRFD on each output to detect faces,
crops + img2img-refines each face, and feather-composites the
refined crop back. Reuses the t2i SdCore — no second model load.

| Flag | Default | Description |
|---|---|---|
| `--adetailer` | off | Enable the post-t2i face refinement pass. |
| `--adetailer-strength <F>` | `0.4` | img2img strength on each face crop. Lower preserves identity + colour; `0.6+` re-imagines. |
| `--adetailer-padding <F>` | `0.25` | Bbox expansion per side. More = better blending, less res per face. |
| `--adetailer-feather <F>` | `0.25` | Outer fraction of the bbox that fades to 0 opacity at the edge. |
| `--adetailer-confidence <F>` | `0.5` | SCRFD score threshold. Faces below are skipped. |
| `--adetailer-size <PX>` | `512` | Working resolution for the face img2img (square, snapped to /8). |
| `--adetailer-prompt <STR>` | (generic) | Override the face-pass prompt. Default: "detailed face, sharp focus, high quality". |

**Required setup**: SCRFD weights via `PLAKAT_SCRFD_WEIGHTS` (local
path) or `PLAKAT_SCRFD_HF` (HF spec). Same env vars the FaceID
portrait flow uses.

**Restrictions**: SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo only. Flux /
SD3 bail loud. Composes with `--lora`, `--seed`, and `--hires-fix`
(ADetailer runs after hires fix so face refinement operates on the
upscaled image).

```bash
plakat generate "a woman walking through a forest, full body shot" \
 --model sd15 --size 768x1024 --adetailer
```

### Hires fix (SD-family)

Mitigates the "multi-head problem" SD/SDXL produce when sampled
above their trained resolution. Generate at the trained res, then
upscale + img2img-refine.

| Flag | Default | Description |
|---|---|---|
| `--hires-fix` | off | Enable the post-t2i upscale + refine pass. |
| `--hires-scale <F>` | `2.0` | Upscale factor for classical upscalers. ML upscalers (Real-ESRGAN) use native scale and ignore this. |
| `--hires-strength <F>` | `0.5` | img2img strength on the upscaled image. Lower preserves composition; `0.7+` allows more reinterpretation. |
| `--hires-upscaler <MODE>` | `lanczos` | `lanczos / bicubic / bilinear / nearest / real-esrgan-x2 / real-esrgan-x4 / real-esrgan-anime-x4`. |
| `--hires-steps <N>` | (main `--steps`) | Step count for the refine pass. |

**Restrictions**: SD 1.5 / SD 2.1 / SDXL / SDXL-Turbo only. Flux /
SD3 already have `--tiled` for high-res output and bail loud. Does
**not** compose with `--artefact*` (upscale changes dims; the
artefact compositor reads the original t2i dims).

```bash
# 1.5k SD 1.5 via 2x Lanczos + img2img
plakat generate "an astronaut on a beach" \
 --model sd15 --size 768x768 --hires-fix

# 4K poster: hires-fix + ADetailer
plakat generate "a vintage travel poster of Tokyo at night" \
 --model sd15 --size 768x768 --steps 30 \
 --hires-fix --hires-upscaler real-esrgan-x2 --hires-strength 0.45 \
 --adetailer
```

### Textual Inversion (partial)

| Flag | Description |
|---|---|
| `--embedding <SPEC>` (repeatable) | Textual Inversion `.safetensors`. Format: `PATH_OR_REPO[:trigger][:scale]`. |

**Status**: parser + `plakat embedding info` inspector ship and
work today. Runtime injection into the SD CLIP-L tokenizer +
token_embedding matrix is gated by candle 0.8's private
`clip::Config.vocab_size` — `--embedding` plumbs end-to-end and
bails loud at SD load with a deferral message + the "convert TI
to LoRA via kohya-ss" workaround. SD 1.5 / SD 2.1 only when the
runtime path opens; SDXL dual-encoder TIs bail in the parser.

Use `plakat embedding info PATH` to inspect a TI file today.

---

## `plakat outpaint`

Extend an input image past its borders. A thin wrapper around the
inpaint pipeline: expand the canvas, replicate-fill the new region
from the input's edge pixels, build a mask covering the new region
(white = inpaint, black = preserve), then hand off to `plakat
img2img --mask`.

```bash
# 256 px on every side, default sdxl-inpaint
plakat outpaint photo.png --prompt ".." --expand 256

# Panoramic: extend only horizontally
plakat outpaint photo.png --prompt "wide mountain valley" \
 --left 512 --right 512

# Flux Fill outpaint
plakat outpaint photo.png --prompt ".." --expand 256 \
 --model flux-fill-dev
```

| Flag | Default | Description |
|---|---|---|
| `<INPUT>` | required | Path to the input image. |
| `--prompt` | required | Describes the whole output (including the new region). |
| `--left / --right / --top / --bottom <PX>` | `0` | Per-side padding. At least one must be > 0. |
| `--expand <PX>` | (off) | Shorthand for all four sides equally. Conflicts with per-side flags. |
| `--model` | `sdxl-inpaint` | Inpaint-capable model. `sd15-inpaint` and `flux-fill-dev` also work, as do non-inpaint models in RePaint mode. |
| `--mask-feather <PX>` | `16` | Softens the seam between original + outpainted regions. Higher than `img2img`'s default 8. |

Padding is snapped up to the model's VAE / patch granularity (8 for
SD, 16 for Flux). Edge replication gives the inpaint UNet a smooth
low-frequency hint at the seam — beats flat gray, which often biases
the denoise toward "wall" content.

---

## `plakat portrait`

Portrait generation, optionally guided by a reference photo. The default
strategy is **IP-Adapter-Plus-Face** on SD 1.5 — the photo flows through
CLIP-H's penultimate hidden state, a 4-layer Perceiver resampler emits
16 image tokens, those concat onto the text token sequence, and the
standard SD denoise loop runs from pure noise. Other strategies
(`plus-face-sdxl`, `faceid`, `faceid-sdxl`) follow the same pattern with
different model variants and identity encoders. Without `--photo`, the
command degrades to a portrait-tuned text-only generate (3:4 aspect,
face/anatomy negatives baked in) with no extra download.

```bash
plakat portrait "cinematic close-up, soft Rembrandt lighting" \
 --photo face.jpg --face-strength 0.8
```

#### `<PROMPT>` (positional, required)

Describes the portrait: pose, lighting, framing, style. The photo seeds
identity; the prompt shapes everything else.

#### `--photo <PATH>` (optional)

Reference photo. Provide a tight head-and-shoulders crop, or use
`--face-bbox` to mark the face region. Optional SCRFD auto-detection
(`PLAKAT_SCRFD_*` env vars) fills landmarks automatically — see
PERSONA.md. With no `--photo`, the identity branch is skipped entirely
and no Plus-Face / CLIP-H weights are downloaded.

#### `--identity <KIND>` (default `plus-face`)

Identity strategy. Must match the `--model` you pass (the validator
refuses cross-attn-dim mismatches at load):

- `plus-face` — IP-Adapter-Plus-Face, SD 1.5. Use with `--model sd15`
 (or any HF SD 1.5 repo).
- `plus-face-sdxl` (aliases: `plusface-sdxl`, `plus-face-xl`,
 `sdxl-plus-face`) — IP-Adapter-Plus-Face SDXL via the `vit-h` variant
 (reuses the SD 1.5 CLIP-H image encoder). Use with `--model sdxl`.
- `faceid` — IP-Adapter-FaceID, SD 1.5. Uses the InsightFace ArcFace
 IR-ResNet50 embedding for stronger identity preservation than
 CLIP-H-based strategies. Requires ArcFace weights (set via
 `PLAKAT_ARCFACE_WEIGHTS` or `PLAKAT_ARCFACE_HF`). See PERSONA.md
 "FaceID setup".
- `faceid-sdxl` — IP-Adapter-FaceID on SDXL. Same ArcFace backbone;
 SDXL UNet target. Use with `--model sdxl`.
- `instantid` — roadmap; not yet implemented.

#### `--face-strength <FLOAT>` (default `0.8`)

Scales the image-token contribution before concatenation. Standard
IP-Adapter `scale` parameter equivalent.

- `0.0` — image tokens vanish; equivalent to running without `--photo`.
- `0.5` — light influence, prompt dominates.
- `0.8` — default; strong likeness while keeping prompt steering.
- `1.0+` — over-amplifies the face; useful for tough photos but the prompt
 can start losing control.

Ignored without `--photo`.

#### `--model <ALIAS|REPO>` (default `sd15`)

Either `sd15` (Stable Diffusion 1.5, default) or `sdxl` (SDXL base 1.0).
Any HF SD-1.5/SDXL repo id also works. Flux is not supported for
portraits. Pair `--model sdxl` with `--identity plus-face-sdxl`.

#### `--size <WxH>` / `--aspect <N:M>` + `--base <SIDE>`

Same semantics as `generate`. Defaults to `--aspect 3:4 --base 768`
(`768×1024`).

#### `--count <N>` / `-n <N>` (default `1`)

Number of portraits per invocation. Each gets `seed + i` if `--seed` is set.

#### `--steps <N>` (default `30`)

Slightly higher than `generate`'s `28` because faces benefit from a few
extra refinement steps. With `--scheduler lcm` + an LCM-LoRA you can drop
to 4–8.

#### `--guidance <FLOAT>` (default `7.0`)

Tuned a touch below `generate`'s `7.5` — IP-Adapter conditioning already
pulls strongly toward the reference, so very high CFG tends to over-saturate.

#### `--negative <TEXT>` (default: face-and-anatomy fixers)

The baseline negative covers `deformed face, asymmetric eyes, extra fingers,
cross-eyed, low quality, blurry, watermark, jpeg artifacts, bad anatomy,
cropped head, disfigured, extra limbs, low resolution`. Pass an explicit
`--negative ""` to disable, or a custom string to fully replace it.

#### `--scheduler <KIND>` (default `euler-a`)

Defaults to Euler-Ancestral — its stochasticity helps skin-tone gradients
look less plasticky than deterministic samplers on SD 1.5. All schedulers
from `generate` are available.

#### `--lora <SPEC>` / `--lora-scale <FLOAT>`

Same syntax as `generate`. A realistic-portrait LoRA stacks cleanly on top
of the Plus-Face conditioning — the LoRA controls aesthetic, the photo
controls identity.

#### `--refine <N>` / `--refine-strength <FLOAT>` (defaults: off, `0.3`)

Same-model polish pass on the final latents. Identity conditioning persists
through the polish loop, so refining usually sharpens without losing the
likeness.

#### `--seed <U64>` / `--enhance <PROVIDER>` / `--out <DIR>`

Identical to `generate`. Output files are named `plakat-portrait-<seed>.png`
to distinguish them from `generate`'s `plakat-<seed>.png`.

### What portrait is and isn't

For `plus-face` / `plus-face-sdxl` strategies: candle 0.8's UNet exposes
no cross-attention hooks, so the *decoupled* IP-Adapter path — separate
`to_k_ip` / `to_v_ip` projections in every block — is not wired up.
Identity tokens travel through the same cross-attention as text. Result:
identity is recognisable but not pixel-perfect, ~50–70% of the diffusers
reference. The `faceid` / `faceid-sdxl` strategies bypass this ceiling
using InsightFace ArcFace embeddings + an automatically-applied UNet
LoRA, landing closer to ~80–90% of reference — at the cost of needing
ArcFace weights set up (see PERSONA.md).

Plus-Face strategies have no face detector — pass a head-and-shoulders
crop, or use `--face-bbox`. FaceID strategies can optionally use SCRFD
for automatic face detection (`PLAKAT_SCRFD_*` env vars).

First-run downloads (`plus-face` / SD 1.5):

- Plus-Face safetensors (~50 MB).
- CLIP-H image encoder (~2.5 GB) — shared with `stylize`, cached once.
- SD 1.5 base (~4 GB) — shared with `generate` / `stylize`.

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

How much to redraw IN. Higher = more like REF, less like IN. This is the
single most important knob — pick it based on **what IN is**, not what
you want REF to do.

| Strength | Face input | Non-face input |
|---|---|---|
| `0.2` | reference barely visible, face exact | barely visible, photo unchanged |
| `0.3 – 0.4` | subtle restyling, face preserved | mild palette shift |
| `0.5` | clear style shift, mild face drift | balanced restyling |
| `0.6` | strong style shift, face starts to drift | recognisable IN, distinctive REF |
| `0.7` (default) | heavy restyle, identity wobble — **too much for faces** | balanced→heavy restyle |
| `0.8 +` | IN is essentially a composition hint; identity gone | heavy redraw |

The default `0.7` is tuned for scenes/landscapes. For face inputs use
`0.35` (or `--for portrait`, below).

#### `--for <PRESET>` (default off)

Strength preset shortcut — picks a documented `--strength` for a use
case. Explicit `--strength` always wins if both are passed.

| Preset | Strength | Use for |
|---|---|---|
| `portrait` (aliases: `face`, `person`) | `0.35` | Face inputs — preserves identity while picking up REF's palette/brushwork. |
| `scene` (aliases: `landscape`, `balanced`) | `0.55` | Landscapes, architecture, objects. Clear style shift, structure preserved. |
| `grading` (aliases: `grade`, `tonal`, `color`) | `0.25` | Tonal/colour grading — adds REF's character without redrawing anything. Safest preset for photos you want to keep recognisable. |

Examples:

```bash
# Face-preserving style transfer.
plakat stylize --in face.jpg --ref painting.jpg --out styled.png --for portrait

# Landscape restyling.
plakat stylize --in landscape.jpg --ref ukiyoe.jpg --out styled.png --for scene

# Just borrow the reference's colour palette.
plakat stylize --in photo.jpg --ref gradient.jpg --out graded.png --for grading

# Preset + explicit override → explicit wins (warns about the conflict).
plakat stylize .. --for portrait --strength 0.45 # uses 0.45
```

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

The **reference IP-Adapter** also adds decoupled cross-attention
(separate `to_k_ip`/`to_v_ip` projections per UNet layer) — plakat
doesn't implement that path because candle's UNet has no attention
hook. Quality is roughly 50–70% of the reference implementation.

For consistent results: REF should have a clear visual style (painting,
sketch, distinctive photo). REFs that are themselves stylistically neutral
won't push IN very far.

---

## `plakat upscale`

Resize an image. Two families: classical filters (pure CPU resize, no
weights) and Real-ESRGAN (RRDBNet ported to candle, ML).

#### `--in <IN>` (required)

Source image. Any common format (PNG, JPEG, WebP).

#### `--out <OUT>` (required)

Output path. Extension determines format (`.png`, `.jpg`, `.webp`).

#### `--scale <FLOAT>` (default `2.0`)

Scale factor for **classical** methods. Non-integer values OK:
`--scale 1.5`, `--scale 2.5`. New dimensions are `round(orig × scale)`.
**Ignored for ML methods** — those have a fixed factor baked into the model.

#### `--method <METHOD>` (default `lanczos`)

**Classical** (fast, predictable, no weights):

| Filter | Quality | Best for |
|---|---|---|
| `nearest` | Lowest — pixelated | Pixel art, retro look |
| `bilinear` | Blurry edges | Thumbnails |
| `bicubic` | Decent, slight softening | General-purpose |
| `lanczos` | Sharpest, slight ringing on hard edges | Photographic / detailed images |

**ML** — Real-ESRGAN (downloads ~64 MB on first use):

| Method | Scale | Variant | Use case |
|---|---|---|---|
| `real-esrgan-x2` | ×2 | xinntao x2plus | Subtle resolution boost, photographic |
| `real-esrgan-x4` | ×4 | xinntao x4plus | Standard 4× upscale, recovers fine detail |
| `real-esrgan-anime-x4` | ×4 | xinntao x4plus_anime_6B | Line art / anime / illustration |

ML methods run on the device chosen by `--device`. Memory scales with the
**output** size — a 4× upscale of 1024² → 4096² requires room for that
64 MB output tensor in F32 plus intermediates.

For 8× or more, chain two calls (`real-esrgan-x4` → `real-esrgan-x2`).
ESRGAN models weren't trained for cascaded use past 4–8×, so quality
degrades.

---

## `plakat transparent`

Make every pixel whose colour matches the **upper-left corner** transparent.
Useful for stripping solid-colour backgrounds from generated images, logos,
or screenshots.

#### `--in <IN>` (required)

Source image.

#### `--out <OUT>` (required)

Output. Must be a format that supports alpha (`.png` or `.webp`); `.jpg` is
rejected with a clear error.

#### `--tolerance <N>` (default `0`)

Per-channel tolerance for the colour match (0–255). `0` is an exact match;
~`10` absorbs JPEG noise on solid backgrounds; ~`30+` extends to anti-aliased
edges and similar shades.

```bash
plakat transparent --in logo.png --out logo-alpha.png --tolerance 12
# → ✓ key #ffffff • 512×512 • 238921/262144 pixels transparent (91.1%)
```

---

## `plakat models`

Browse HuggingFace and manage the local cache.

| Subcommand | Purpose |
|---|---|
| `models search <QUERY>` | Free-text HF search. |
| `models recommend [--query Q] [--sort downloads\|likes\|trending\|recent] [--limit N]` | T2I-filtered recommendations. |
| `models size <REPO>` | Total Hub size + the subset plakat would actually download. |
| `models pull <REPO>` | Pre-download SD/Flux weight files for a repo. |
| `models ls` | List cached models with disk usage. |
| `models rm <REPO>.. [--yes]` | Delete cached models (with size + confirmation by default). |

---

## `plakat civitai`

Browse + download Civitai assets — community LoRAs, checkpoints,
embeddings, ControlNet variants.

| Subcommand | Purpose |
|---|---|
| `civitai search <QUERY> [--type lora\|checkpoint\|ti\|controlnet\|vae] [--limit N] [--page P] [--include-nsfw]` | Free-text search. Filters NSFW by default; pages 1-indexed. |
| `civitai info <REF>` | Show model/version details — trigger words, base model, files. |
| `civitai download <REF> [--file NAME]` | Fetch the asset into the local cache. Prints the absolute path. |

`<REF>` accepts: bare integer model ID (`123456`), `civitai:`
shorthand (`civitai:123456`), or any of the `https://civitai.com/`
URL shapes (`/models/<id>`, `/models/<id>/<slug>`,
`?modelVersionId=..`, and the `/api/download/models/<vid>`
direct-download form).

Downloads stream into
`<plakat-cache>/civitai/model-<id>/version-<id>/`. Cache-hit
short-circuits on matching size. Authenticated downloads use
`CIVITAI_API_KEY` from the env. Drop the printed path into
`--lora` or `--model`.

```bash
plakat civitai search "watercolor" --type lora --limit 10
plakat civitai info 12345
plakat civitai download "https://civitai.com/models/12345?modelVersionId=789"
```

---

## `plakat embedding`

Inspect Textual Inversion (`.safetensors`) files + Flux IP-Adapter
weights. Runtime injection lands when candle exposes the seam —
parsers + inspectors ship today.

| Subcommand | Purpose |
|---|---|
| `embedding info <PATH_OR_REPO> [--trigger NAME]` | Inspect a TI file: trigger word, vector count × dim, matching SD variant. |
| `embedding flux-ip-adapter-info <PATH_OR_REPO>` | Inspect an XLabs Flux IP-Adapter: SigLIP feature dim, Flux hidden dim, per-block attention count. |

```bash
plakat embedding info ./my-style.safetensors
plakat embedding info sd-concepts-library/cat-toy
plakat embedding flux-ip-adapter-info XLabs-AI/flux-ip-adapter
```

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

For Apple hardware (chip tiers, expected speeds, memory headroom),
see [`APPLE_REQUIREMENTS.md`](APPLE_REQUIREMENTS.md).

#### `--cache-dir <PATH>` (env: `PLAKAT_CACHE_DIR`)

Where HF model weights are cached. Resolution order:
1. `--cache-dir` flag
2. `PLAKAT_CACHE_DIR` env var
3. `HUGGINGFACE_HUB_CACHE` env var
4. `HF_HOME` env var (with `/hub` appended)
5. Default `~/.cache/huggingface/hub`

---

## Common workflows

### Iterate on a prompt at a fixed composition

Set `--seed` to anything, vary the prompt — same composition, different
details.

```bash
plakat generate "a tranquil koi pond, soft light" \
 --model sdxl --size 1024x1024 --steps 30 --scheduler euler-a \
 --seed 42

# Now tweak just the prompt:
plakat generate "a tranquil koi pond, soft light, autumn leaves" \
 --model sdxl --size 1024x1024 --steps 30 --scheduler euler-a \
 --seed 42
```

### Sample many seeds, then refine the best

```bash
# Step 1 — sample 4 candidates
plakat generate "a tranquil koi pond, soft light" \
 --model sdxl --size 1024x1024 --seed 0 -n 4

# Step 2 — pick seed=2 (say), refine with extra polish
plakat generate "a tranquil koi pond, soft light" \
 --model sdxl --size 1024x1024 \
 --steps 35 --scheduler euler-a \
 --refine 8 --refine-strength 0.25 \
 --seed 2
```

### Fast generation with LCM-LoRA

LCM-LoRA + the matching LCM scheduler render usable output in 4 steps on
any device.

```bash
plakat generate "a serene mountain landscape at sunset" \
 --model sd15 --size 512x512 \
 --steps 4 --guidance 1.5 \
 --scheduler lcm \
 --lora latent-consistency/lcm-lora-sdv1-5
```

### Style transfer that mostly preserves IN

Lower strength keeps more of the input photo's content while picking up
REF's style.

```bash
plakat stylize \
 --in photo.jpg \
 --ref reference_painting.jpg \
 --out styled.png \
 --strength 0.4 --steps 30
```

### Img2img / inpaint / outpaint

For prompt-driven transforms of an existing image (with or without a
region mask), use the dedicated `plakat img2img` subcommand. SD
1.5 / 2.1 / SDXL, Flux (`flux-dev` for img2img, `flux-fill-dev`
for inpaint, both available as GGUF variants), and SD3 / SD3.5
all work. To extend a canvas past its borders, use `plakat outpaint`. Full
reference: [`IMG2IMG.md`](IMG2IMG.md). Runnable walkthrough:
[`examples/tutorials/IMG2IMG/`](../examples/tutorials/IMG2IMG/).

### ControlNet (layout conditioning)

For structural guidance from a depth map or canny edge map,
every SD-family subcommand accepts:

```bash
--control <depth|canny>
--control-image PATH # pre-rendered conditioning
--control-from PATH # OR auto-annotate any image
--control-strength F # default 1.0
```

Works on SD 1.5, SD 2.1, SDXL, **Flux** (BF16 / GGUF / NF4, via
Shakker-Labs Union Pro v2), and **SD3 / SD3.5** (via the
InstantX adapter family). The architecture is
auto-detected from `--model` and the resolver picks the matching
adapter repo per variant. Full reference:
[`CONTROLNET.md`](CONTROLNET.md). Runnable walkthrough:
[`examples/tutorials/CONTROL/`](../examples/tutorials/CONTROL/).

### Generate → 4× ML upscale

```bash
plakat generate "an isometric tiny diorama of a forest cabin" \
 --model sd15 --size 512x512 --seed 7 --out ./out

plakat upscale --in ./out/plakat-7.png --out ./out/plakat-7-4x.png \
 --method real-esrgan-x4 --device metal
# 512×512 → 2048×2048
```

### Generate → stylize → upscale (one pipeline via scenario)

For batch runs that chain all three steps with shared pipelines, see
the annotated example at
[`examples/scenario.hjson`](../examples/scenario.hjson). The `scenario`
subcommand assembles per-task prompts from a catalog of scenes and
weather, runs the per-image pipeline, and reuses loaded weights across
every task.

#### Scenario fieldsAll are
optional; existing scenarios keep working unchanged.

**Scenario-level (top of the HJSON):**

| Field | Type | Description |
|---|---|---|
| `quantize-t5:` | `bool` | Load T5-XXL as a quantized GGUF. Requires a GGUF Flux model. |
| `quant-level:` | string | Flux GGUF quant level (e.g. `Q5_K_M`). Defaults to `Q4_K_S`. |
| `t5-quant-level:` | string | T5 GGUF quant level. Defaults to `Q4_K_M`. |
| `tiled:` | `{ size, stride }` | MultiDiffusion-style tiled denoise for every task. Defaults: `1024` / `768`. |

**Per-task (inside each `tasks: [..]` entry):**

| Field | Type | Description |
|---|---|---|
| `controls:` | `Vec<ControlSpec>` | Multi-ControlNet — residuals sum per step. Mutually exclusive with the singular `control:`. |
| `init-image:` | path | Source image for img2img / inpaint / outpaint. |
| `mask:` | path | Inpaint mask. Requires `init-image:`. |
| `strength:` | float | img2img strength `[0, 1]`. Ignored for Fill. |
| `mask-feather:` | int | SD inpaint mask feather radius (px). |
| `mask-invert:` | `bool` | Flip mask polarity (black = inpaint). |
| `outpaint:` | `{ expand | left/right/top/bottom }` | Canvas expansion. Synthesises mask from padding amounts. Requires `init-image:`. |

```hjson
{
 model: flux-dev-gguf,
 quant-level: Q5_K_M,
 quantize-t5: true,
 tiled: { size: 1024, stride: 768 },
 tasks: [
 // Multi-CN, depth + canny stacked
 { name: gen1, prompt: "..", controls: [
 { kind: depth, image: ./d.png, strength: 0.8 },
 { kind: canny, auto-from: ./ref.jpg, strength: 0.5, end: 0.5 },
 ]},
 // Outpaint via Flux Fill
 { name: panorama, prompt: "wide landscape", model: flux-fill-dev,
 init-image: ./photo.png, outpaint: { left: 512, right: 512 } },
 // SD inpaint
 { name: fix-sky, model: sdxl-inpaint, prompt: "blue sky",
 init-image: ./in.png, mask: ./sky-mask.png,
 mask-feather: 16, strength: 1.0 },
 ]
}
```

Notes on scenario pipeline caching:

* **SD 1.5 / 2.1 / SDXL** — t2i tasks share a single `Arc<SdCore>`;
  img2img/inpaint tasks within the same scenario still reload the
  SD pipeline per task (the img2img dispatcher doesn't yet share
  the t2i SdCore).
* **Flux** (BF16 / GGUF / NF4) — one pipeline shared across every
  task. Per-task `loras:` swap at runtime via the LoraLinear stack.
* **SD3 / SD3.5** — one MMDiT pipeline shared across every task.
  Per-task `loras:` swap at runtime via the MMDiT LoraLinear stack.

For **SD-family per-task LoRA** (different LoRA stacks per task),
plakat runs a preflight at scenario start and either prints a hint
to fold uniform stacks to scenario-level `loras:` or bails loud
with three concrete workarounds (switch to Flux/SD3.5, split
scenarios, or fold). The SD UNet runtime LoraLinear vendor is
deferred — candle 0.8's UNet internals are private.
