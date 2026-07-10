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
  | stable-cascade | 512² | **198429** | 6467 | 5133 | 1172 | 105174 | **303.6 s** | 7.3 GB |

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

- [x] **Load path profiled** — env-gated `PLAKAT_PROFILE_LOAD=1` prints per-phase deltas in
      `sd_core::SdCore::load` (download-resolve / tokenizers+vae / **unet** / text-enc+rest).
      **Findings (sd15, Metal/CPU):**
      - **The UNet phase is 70–76% of load** (sd15 14.7 s of 19.3 s; sdxl 46.5 s of 66 s).
        download-resolve is negligible (<0.6 s) — it's not the network/cache-lookup.
      - **It's pure disk I/O, not compute.** Back-to-back runs: the UNet phase collapses
        **14.7 s → 0.42 s (35×)** once the page cache is warm. Construct is ~0.4 s; the rest is
        reading the weights.
      - **The read is inefficient** — sd15's ~3.4 GB UNet in 14.7 s ≈ **230 MB/s**, far below SSD
        sequential (1–3 GB/s). The lazy-mmap **random page-fault** pattern is the culprit; there's
        no sequential prefetch.
      - Load time swings 0.8 s ↔ 21 s with cache state; loading big models evicts the cache
        (memory pressure), so cold re-reads recur in real use.
- [x] **Conclusion → Phase-2 weight-load approach:** (1) **sequential prefetch** — `madvise`
      `WILLNEED`/`SEQUENTIAL` (or an explicit read-ahead pass) to turn the 230 MB/s random-fault
      read into a 1–3 GB/s streaming read; (2) **parallel sub-file loads** (unet ∥ vae ∥ text
      encoders — currently sequential); (3) the persistent daemon (deferred, user's RFC) keeps
      weights resident and sidesteps reload entirely. Per-step/VAE profiling deferred (already
      bounded by the baselines).

## Phase 2 — Optimize (reordered by the Phase-0 data; each verified-safe)

1. [x] **Weight load** — implemented `sd_core::prefetch_files`: a parallel sequential pre-read of
       the big weight files before the build, so the mmap faults hit warm pages. Correctness-safe
       by construction (only warms the page cache; compute is byte-identical). Env-disable
       `PLAKAT_NO_PREFETCH=1`. **Measured (sd15, cache evicted): 22.3 s → 20.1 s (~10%).**

       **BUT the number reframed the whole load problem — the disk is the wall, not the access
       pattern.** Raw `dd` of the UNet = **127 MB/s** single-stream; random mmap-fault ~116 MB/s;
       parallel prefetch ~152 MB/s. This box's weight reads top out ~130–250 MB/s (a ~1.7 GB F16
       UNet takes ~14 s to read). So **in-process load optimization is capped at ~10–30%** — the
       parallelism win is real but small, because the storage, not the code, is the bottleneck.
       The high-leverage load wins are therefore **out of the read path**: (i) the persistent
       daemon (deferred, user's RFC) — don't re-read at all; (ii) **read fewer bytes** — quantized
       weights (a Q4 model is ¼ the file → ~4× faster load), tying into the quant direction.
       *(Note: 127 MB/s is slow for a Mac NVMe — the HF cache volume may be a bottleneck worth
       checking, but it's the user's environment.)*
   → **Strategic pivot:** load is disk-bound with little in-process headroom; the remaining
     leverage is in the **compute** phases (below), which are not I/O-bound.
2. [⏸] **VAE decode** *(SDXL 1024² = 15.6 s)* — **tiled decode rejected (measured negative).**
   Confirmed the cost is the mid-block spatial self-attention going quadratic (512²→1024² VAE-tail
   2.76 s→17.8 s = 6.45× for 4× pixels; solving linear+quadratic ⇒ ~9 s is attention at 1024²).
   But `tile_decode_2d` (64 tile / 56 stride) made it **worse: 15.6 s → 21.5 s** — the VAE is slow
   *per unit area* (a 64² tile decode is 2.76 s on candle/Metal), so the overlapping-tile **conv
   redundancy** (9 tiles) outweighs the attention saving. A real win needs kernel-level VAE work
   (the slow Metal groupnorm/conv/attention), not tiling — deferred.
3. [⏸] **T5-XXL text-encode** *(SD3.5 = 14.4 s)* — **int8 T5 blocked on this hardware (two
       confirmed reasons):** (1) candle's `quantized_t5` encoder `forward(input_ids)` has **no
       padding attention-mask** param, so it can't do SD3's pad-masking → dropping it in would
       **regress the v2.1 caption fix** (SD3 uses `vendored_t5` precisely for `forward_with_mask`);
       (2) it runs on **`QMatMul`, the candle-0.10.2 Metal quantized-matmul kernel that produces
       corrupted output** — the same general bug that defers GGUF Flux (`gguf_metal_blocked`).
       So the int8 *compute* win isn't reachable here. Not shipping a broken/regressing path.
       Achievable alternatives (future): (a) **int8-on-disk → BF16-in-compute** to cut the T5
       portion of the 137 s *load* (fewer bytes, Metal-safe, masked forward unchanged) — but
       that's the load metric, not the 14.4 s encode; (b) **cache the T5 encode across a batch**
       (Metal-safe; helps count>1 / scenario, not single-image); (c) revisit with Flux once the
       candle Metal quant kernel is fixed.
4. [x] **Step-caching** — TeaCache-lite loop cache (`PLAKAT_STEP_CACHE=<thresh>`, unset = exact ⇒
       verify unaffected): accumulate the per-step input change, reuse the cached model output
       under threshold. Ported to **PixArt, SD3.5, and Cascade Stage-C**. Measured (512², 20
       steps, Metal):
       - **PixArt** @0.10 → **1.37×**, SSIM **0.9987** (imperceptible); @0.30 → 2.6× / SSIM 0.856.
       - **SD3.5** @0.10 → **1.92×** (gen 81 s → 42 s); @0.20 → 2.41×. Bigger win (heavier MMDiT
         per-step). SSIM confirmation blocked this session by Metal memory pressure (the exact
         path OOM-guards on a fresh T5-XXL load) — expected ~PixArt-level (identical mechanism);
         `sudo purge` + regen to confirm.
       - **Cascade** @0.10 → **~0%**, @0.20 → ~5%. **Defeated by the stochastic Wuerstchen
         scheduler** (fresh noise each step inflates the per-step change ⇒ the accumulator crosses
         threshold every step ⇒ no cache hits). Kept (opt-in, harmless) but ineffective as-is.

       **General finding: step-caching pays off on DETERMINISTIC transformer samplers (PixArt
       1.4×, SD3.5 1.9×) but NOT on stochastic ones (Cascade).** Sweet spot ~0.10–0.15. Flux
       deferred (untestable on this hardware). To make Cascade work would need a deterministic
       indicator (e.g. the timestep embedding) instead of the noise-polluted latent delta.
5. [~] **Attention — fused SDPA (the surprise win).** Probe (`examples/sdpa_probe.rs`) decided
       GO: candle's **Metal SDPA kernel is correct (~1e-6 vs eager) and 15–17× faster** at
       realistic dims — unlike VAE/int8, the kernel *works*, and it takes a mask + supports the
       models' head_dim (64/72). Wired into **PixArt self-attention** (GPU-only — candle SDPA has
       no CPU impl, so CPU + verify stay on eager; masked cross-attn stays eager to keep the v2.1
       fix bit-identical). Measured: **PixArt self-attn per-step 1560→1026 ms = 1.52×** (isolated
       SSIM 0.99873, mad 2.3/255); **SD3.5 MMDiT joint-attn 1910→1555 ms = 1.23×** (probe covers
       the exact dims h=24 s=1024 d=64 at 3.6e-6; sd35 end-to-end isolation OOM-blocked this
       session). Escape hatch `PLAKAT_NO_SDPA=1`. Compounds with step-caching + the Metal samplers.
       **Parked (post-2.4):** SD UNet self-attn — unlike the DiTs, the SD generation UNet builds
       its attention from **candle's registry** `stable_diffusion` (not plakat-editable), so it
       needs *vendoring* candle's SD attention (+ blocks) or re-architecting generation onto
       plakat's own `sd_train` UNet; also head-dim-partial (SDXL 64 full; SD1.5/2.1 40/80/160 →
       80-only). Deferred as a focused follow-up. Flux stays deferred (untestable here); the masked
       cross-attn paths (SDPA additive-mask shape) also parked.
6. [x] **Metal schedulers unblocked** — DPM++ 2M Karras / UniPC / UniPC-exp were rejected on
       Metal (their solver-coefficient math builds F64 tensors on the device; candle 0.10.2 has
       no F64 Metal backend). Instead of a full F32 rewrite or vendoring candle's 1005-line
       `uni_pc.rs`, wrapped it in a **`CpuHopScheduler`** that routes the tiny per-step scheduler
       tensors through the CPU (F64 works there) and moves the result back. **Verified: det-init
       Metal-DPM++ vs CPU-DPM++ = SSIM 1.0000** (mad 0.10/255). `check_device_support` is now a
       no-op. Payoff: the community-standard samplers on Apple Silicon → good quality at fewer
       steps (DPM++ ~20 vs Euler ~28–30; UniPC ~10–15), compounding with step-caching.

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
