# plakat 2.4.0 — roadmap: the performance pass

2.3.0 made plakat a first-class **library** (`plakat::api`). 2.4.0's anchor is a **performance
pass** — the one dimension the 2.x line hasn't touched, and directly user-visible.

The organizing idea plays to plakat's unique asset: **the verify harness makes this a
provably-safe perf pass.** Every optimization is gated behind `plakat verify` — Tier 1 (golden
tensors, corr 1.0) + Tier 2 (end-to-end SSIM 1.0) — so **speed never costs correctness**. No
other local generator can optimize with that guarantee.

Goal: implement all the perf-pass items below. (`plakat serve` — the persistent daemon — is
**deferred**: the user is authoring an architecture RFC for it; do NOT implement serve here.)

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Phase 0 — Measure (build the ruler first)

- [x] **`plakat bench`** — new subcommand (`src/cli/bench.rs`). Real generation, decomposed into
      **load / encode+first-step / per-step (mean) / VAE-decode tail / total** + **peak RSS**
      (background sampler). Fixed prompt+seed; `--json` for CI; `--repeat K` reports the best.
      SD family wired (sd15/sd21/sdxl/pony/turbo); PixArt/SD3.5/Cascade/Flux in a later phase.
- [~] **Freeze baselines** — first data point: **sd15 CPU 256² 4-step → total 29.9 s, of which
      _load is 21.8 s (73%)_**, VAE tail 2.1 s, per-step 1.47 s, peak 5.6 GB. → confirms the
      hypothesis: **cold load dominates**; weight-load + a persistent model (the deferred daemon)
      are the top prize, not per-step micro-tuning. Still to freeze: Metal + the other families.
- [x] **No optimization lands without a before/after number** from this harness.

## Phase 1 — Profile the hot paths

- [ ] Instrument load / text-encode / per-step denoise / VAE decode; attribute wall-clock.
      Expected culprits to confirm (not assume): weight load (cold), attention (per-step),
      VAE decode (tail), needless F16↔F32 casts.

## Phase 2 — Optimize (highest-leverage first; each verified-safe)

- [ ] **Step-caching** — TeaCache / DeepCache / first-block caching for the DiT/MMDiT/Flux
      models. Algorithmic 1.5–2× with near-zero quality drift. (Direction #2.) Gate: Tier-2 SSIM
      stays within the perceptual bound; expose as an opt-in knob (quality↔speed).
- [ ] **Attention** — Metal SDPA / fused paths, F16 accumulation, eliminate redundant casts.
- [ ] **VAE decode** — tiled + F16 decode (memory *and* the latency tail).
- [ ] **Weight load** — parallel mmap, lazy/on-demand submodules, warm-cache reuse across a batch.
- [ ] **Metal-native schedulers** — reimplement UniPC / DPM++ 2M Karras sigma math in **F32** so
      they stop being rejected on Metal (`scheduler.rs check_device_support`). Removes a
      correctness-forced fallback *and* speeds sampling. (Direction #4.)

## Phase 3 — Lock it in

- [ ] **Perf CI gate** — `plakat bench` thresholds fail a PR on regression (the speed analogue of
      the Tier-0 correctness gate).
- [ ] **Every optimization verified** — `plakat verify` Tier 1 (corr 1.0) + Tier 2 (SSIM 1.0)
      green after each change; document the measured speedup per model.

## Deferred / parked (see memory `reference_feature_directions`)

- [⏸] **`plakat serve`** (Direction #1) — user's own architecture RFC pending. Not in scope.
- Feature backlog (PAG/FreeU/CFG-rescale, SDXL Lightning/Hyper, ControlNet-Tile + diffusion
  tiled-upscale, Sana, one-shot edit commands, fp8/broader quant) — future cycles.

## Parked follow-ups (not the anchor; pick up opportunistically)

- [ ] **Regional prompting in Bund** (`plakat.region.*`) — needs `t2i::GenRequest` to gain a
      `regions` field + SD regional-denoise wiring, then a `region.{add,clear,list}` namespace.
- [ ] **Per-builder API knobs** — ControlNet/embeddings/refiner/tiled/regions/flux-quant on the
      `plakat::api` builders (additive to the locked surface).
- [ ] **`cargo public-api` in CI** — full public-surface diff scoped to `plakat::api`.
- [ ] **`sd21 unet.out` verify symmetry** — trivial (same candle UNet as sd15); author the golden.

## House-keeping

- [x] **Open 2.4.0** — branch off `main` (2.3.0 release), version bump `2.3.0 → 2.4.0`.
