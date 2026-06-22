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

- [ ] **SD 2.1 — style LoRA + DreamBooth** *(start here)*. Reuse the SD 1.5 UNet
      trainer with sd21's 1024-dim CLIP + the **v-prediction** loss target. Route
      `sd21` into `train_style_lora_sd`; extend `style train --base` / `dreambooth`.
      On-box verifiable → `corpus/style_train.sh sd21` + a committed showcase.
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
