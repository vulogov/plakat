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
- [x] **PixArt-Σ — style/subject LoRA — DONE (code), NOT on-box-verified.** Retrofitted
      the DiT attention projections (`attn1`/`attn2` `to_q/k/v/out`) + FeedForward to
      `LoraLinear` via a shared `LoraRegistry` threaded through `PixArtBlock::new` →
      `PixArtSigmaXL::new` (public ctor unchanged), plus `install_train_adapters`
      (attention-only filter). `pixart::train_style_lora` runs the **DDPM ε-prediction**
      loop (linear betas 1e-4→2e-2, first-4-channel ε, Σ res/aspect conditioning, AdamW +
      grad-clip, numbered checkpoints, `--resume`, DreamBooth `--class-dir`) and saves
      diffusers-PEFT keys the existing `pixart_lora` merge path loads. CLI: `style train
      --base pixart`; `corpus/style_train.sh pixart`. Builds clean (`--features metal`),
      43 pixart DiT tests green. **NOT verified on this 24 GB box: memory-bound.** T5-XXL
      (4.7 B) drives Phase-A peak past 32 GB → swap-thrash on 24 GB unified (host stayed up;
      see below). Showcase wants ≥ 36 GB unified or CUDA. Same memory class as SD3.5
      DreamBooth (carried debt).
  - **OOM-guard gap found + closed (this cycle):** the watchdog
    ([`memwatch::MemoryGuard`]) was wired only into `generate` / `scenario`, never the
    training paths — so the run above was unguarded. Now installed in `style train`
    (covers every base). Note the guard's contract is *host-crash* prevention on
    sustained **CRITICAL** kernel pressure; it deliberately does NOT fire on low free-RAM
    or slow swap-thrash (which never crossed the kernel's Critical cliff here — the box
    kept swapping, just uselessly slowly). So it would not have aborted this run; the real
    fix is sizing the box to the base.
- [x] **SD 3.5 — Textual Inversion — DONE (code), NOT on-box-verified (memory-bound).**
      Both halves landed; completes SD3.5's training trio (LoRA + DreamBooth + TI).
      - **Training** — a placeholder token learned in CLIP-L + CLIP-G **and** T5 via a
        differentiable splice into each encoder's init-word slot, rectified-flow loss
        through the frozen MMDiT, saved as a triple file (`clip_l`+`clip_g`+`t5`).
        Required vendoring candle's T5 (`vendored_t5.rs`) to expose `embed_tokens` +
        `forward_from_input_embeds`; a guard test proves the copy is byte-faithful to
        candle's T5 on random weights (so SD3.5 inference / LoRA / DreamBooth are
        unaffected). `embedding train --base sd35`.
      - **Inference loading (runtime splice)** — SD3 `LoadRequest`/`Request` gained an
        `embeddings` field; on load each triple TI is read and its trigger registered as
        an added token in all three tokenizers. `encode_prompt` early-branches to
        `encode_prompt_ti` **only when a TI is loaded** (the verified path is otherwise
        untouched): it clamps the trigger's OOB id for the embedding lookup, splices the
        learned vector·scale into that row via `slice_assign`, and runs each encoder
        from-embeds — including the clamped-ids argmax fix so CLIP-G pooling stays
        correct. No weight files rewritten. `--embedding PATH:trigger:scale`.
      - Memory wall: CLIP-L + CLIP-G + T5-XXL + MMDiT resident (training adds autograd) →
        >24 GB on the canonical checkpoint. `corpus/embedding_train.sh sd35` is a recipe
        for ≥36 GB / CUDA, not a committed proof.
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
