# plakat 2.5.0 — roadmap (planning)

First cycle on the **road to 3.0** (see [`ROADMAP_TO_3.0.md`](ROADMAP_TO_3.0.md)). Theme: **finish
2.4's parked performance threads + cheap image-quality wins.** Scope decided with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Candidate anchor items

- [?] **candle 0.11 spike** — 0.11.0 is released (candle is pre-1.0). Bump on a throwaway branch,
      fix the API breaks, run `plakat verify` across all 7 families (any kernel regression =
      corr < 1.0), and test the three 2.4 blockers: **GGUF Flux on Metal**, a **quantized-matmul**
      sanity check, and **DPM++ without the CpuHopScheduler**. If 0.11 fixes the Metal quant kernel,
      migration unblocks GGUF Flux + int8 T5 — the biggest latent win. Decide from the data.
- [ ] **SD UNet SDPA** — bring the 2.4 attention win to the SD1.5/2.1/SDXL workhorses. Needs
      vendoring candle's `stable_diffusion` attention (+ the UNet blocks that call it) since it's
      registry code; SDXL gets full head-dim-64 coverage, SD1.5/2.1 partial. Verify-gated.
- [ ] **Free-quality guidance** — **PAG** (perturbed-attention guidance; now tractable post-SDPA),
      **FreeU**, **CFG-rescale** / dynamic thresholding. No new weights, real quality; opt-in flags,
      verify-safe. Highest quality ROI.

## Carried-over (opportunistic)

- [ ] Masked cross-attn SDPA (finish the rollout; validate the additive-mask shape).
- [ ] `sd21 unet.out` verify symmetry (trivial).
- [ ] Perf-CI gate — `plakat bench` thresholds fail a PR on regression.

## House-keeping

- [x] **Open 2.5.0** — branch off `main` (2.4.0 release), version bump `2.4.0 → 2.5.0`.
