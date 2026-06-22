# plakat 1.10.0 — roadmap

The map track is complete and polished (1.4–1.9). 1.10.0's headline is the
**model-training expansion** — closing the LoRA / TI gaps across the families —
per the standing [`PLAN_TRAINING_EXPANSION.md`](PLAN_TRAINING_EXPANSION.md). Map
optional features and carried debt remain available as off-track work.

The through-line holds: training output is non-deterministic, so each new trainer
lands with a `corpus/*_train.sh` driver + a committed **showcase** (not a byte-check),
and is verified **on-box** where possible. Build with a GPU backend
(`--features metal` on Apple Silicon).

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — model-training expansion (headline)

In the recommended sequence from the plan:

- [x] **SD 2.1 — style LoRA + DreamBooth — DONE.** `train_style_lora_sd` now branches
      on `is_sd21`: a dedicated `sd21_unet_config` (cross-attn 1024, linear projection,
      `[5,10,20,20]` heads) and a **v-prediction** loss target (`v_target` = √ᾱ·ε −
      √(1−ᾱ)·x0; ε for SD 1.5). The 1024-dim CLIP conditioning comes from SdCore for
      free; DreamBooth (`--class-dir`) rides the same loop. CLI: `style train --base
      sd21`; `corpus/style_train.sh sd21`. **Verified on-box (Metal):** trained the
      watercolour set → 128 attention adapters, v-pred loss trended down, a kohya LoRA
      that loads **128/128 merged** into sd21 inference and renders. 2 unit tests
      (config + v-pred math). *(Full showcase = run `style_train.sh sd21` ~120 steps.)*
- [ ] **PixArt-Σ — style/subject LoRA**. Retrofit the DiT attention/MLP projections to
      `LoraLinear` + a PixArt `install_train_adapters`; training forward (VAE encode,
      BF16 T5, IDDPM-ε through the frozen DiT). The reusable transformer-adapter
      groundwork.
- [ ] **SD 3.5 — Textual Inversion**. A placeholder token learned in CLIP-L + CLIP-G
      **and** T5; rectified-flow loss through the frozen MMDiT; save a triple embedding
      file. Completes SD3.5's training trio.
- [ ] **Stable Cascade — Stage-C LoRA**. Train in the Würstchen semantic space;
      Stage-C attention adapters (the merge path already loads the result).
- [ ] **Flux — BACK-BURNER**. Implementable but unverifiable on Metal (Flux inference
      is broken on Metal); park until a CUDA/CI verify path exists.

See the plan for per-family steps, effort, risk, and the shared adapter spine.

## B — map optional features (off-track, opt-in)

Carried from the 1.9 roadmap — pick as wanted:

- [ ] **River + dry canyons** — carve gorges along high-flow channels + realize
      `terrain.rift_valleys` as dry canyons (`--map-canyons`). The remaining
      terrain-realism gap.
- [ ] **Lakes + marshland depth** — lake reflection tint, marsh hatching for Wetland
      regions, river deltas at navigable mouths.
- [ ] **Plateaus / mesas** — realize `terrain.plateaus` (a schema stub) as flat-topped
      scarped terrain.
- [ ] **Political layer** — borders + polity fills/labels from the unused
      `RegionSpec.political`.
- [ ] **Seasonal palettes** (`--map-season`), **game-grid overlay** (`--map-grid`),
      **multi-tile world maps**.

## C — corpus / verification

- [ ] **Fill `corpus/images/train/`** — run `resume_train.sh` (a few GPU-minutes; the
      one ungenerated, non-memory-blocked proof).
- [ ] **Map gallery section** — add the town + eroded-island + painted renders to
      `GALLERY.md`.

## D — carried product debt

- Flux regional prompting (Metal-blocked → code + CI), IC-Light relighting, the
  memory-bound SD3.5 DreamBooth / `regional.sh sdxl/sd35` renders.

## Notes

- `--features metal` (Apple Silicon) / `--features cuda` (NVIDIA) for GPU; the default
  build is CPU-only.
- New deps gated behind features where practical (e.g. the map's `shaped-labels`).
