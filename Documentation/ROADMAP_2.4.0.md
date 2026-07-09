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
- [x] **Metal baselines frozen** (`--size` pinned per model, 20 steps, F32 Metal). `load` is the
      first cold load this session (page-cache empty) except sd15 (warm):

  | model | size | load (ms) | encode+first | per-step | VAE tail | gen | **total** | peak RSS |
  |---|---|---|---|---|---|---|---|---|
  | sd15 | 512² | 797 *(warm)* | 23 | 711 | 3154 | 16686 | **17.5 s** | 3.5 GB |
  | sd21 | 512² | 61385 | 69 | 660 | 3118 | 15719 | **77.1 s** | 3.8 GB |
  | sdxl | 1024² | 67194 | 1421 | 4278 | **17800** | 100499 | **167.7 s** | 10.4 GB |
  | pixart | 512² | **174832** | 1759 | 1560 | 4037 | 35442 | **210.3 s** | 13.5 GB |
  | sd35-medium | 512² | 137027 | **14408** | 1537 | 3413 | 47024 | **184.1 s** | 9.4 GB |

  **Findings that redirect the pass:**
  1. **Cold weight-load dominates** — 61–175 s, dwarfing generation. sd15 warm-load 0.8 s vs its
     own 21.5 s cold ⇒ it's largely I/O + page cache. → biggest prize (load + persistent model).
  2. **VAE decode is a fat tail at high-res** — SDXL 1024² spends **17.8 s** just decoding.
  3. **T5-XXL text-encode is a one-time monster** — SD3.5's encode+first is **14.4 s** (the T5-XXL
     forward), separate from per-step.
  4. Per-step only dominates on SDXL@1024 (4.3 s/step); step-caching's payoff is real but #4, not #1.
- [ ] **Still to freeze**: Cascade + Flux (once the bench covers them), and CPU parity.
- [x] **No optimization lands without a before/after number** from this harness.

## Phase 1 — Profile the hot paths

- [ ] Instrument load / text-encode / per-step denoise / VAE decode; attribute wall-clock.
      Expected culprits to confirm (not assume): weight load (cold), attention (per-step),
      VAE decode (tail), needless F16↔F32 casts.

## Phase 2 — Optimize (reordered by the Phase-0 data; each verified-safe)

1. [ ] **Weight load** *(the ~60–175 s elephant)* — parallel mmap, lazy/on-demand submodules,
       warm-cache reuse across a batch, avoid redundant dtype conversions at load. The persistent
       model (deferred `serve` daemon, user's RFC) captures the warm-load win; do the in-process
       wins here.
2. [ ] **VAE decode** *(SDXL 1024² = 17.8 s)* — tiled + F16 decode (memory *and* the latency tail).
3. [ ] **T5-XXL text-encode** *(SD3.5 = 14.4 s)* — wire the existing int8 T5 (`--quantize-t5`)
       into the bench + cache the encode across a batch; measure the quality/speed trade.
4. [ ] **Step-caching** — TeaCache / DeepCache / first-block caching for DiT/MMDiT/Flux. Algorithmic
       1.5–2×; opt-in knob (quality↔speed). (Direction #2.) Biggest per-step win, esp. SDXL@1024.
5. [ ] **Attention** — Metal SDPA / fused paths, F16 accumulation, eliminate redundant casts.
6. [ ] **Metal-native schedulers** — reimplement UniPC / DPM++ 2M Karras sigma math in **F32** so
       they stop being rejected on Metal (`scheduler.rs check_device_support`). (Direction #4.)

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
