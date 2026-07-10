# plakat 2.5.0 — roadmap (planning)

First cycle on the **road to 3.0** (see [`ROADMAP_TO_3.0.md`](ROADMAP_TO_3.0.md)). Theme: **finish
2.4's parked performance threads + cheap image-quality wins.** Scope decided with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Candidate anchor items

- [⏸] **candle 0.11 spike — DEFERRED until candle stabilizes** (owner call, 2026-07-10). candle
      is pre-1.0; 0.11 carries breaking changes and the payoff (does it fix the Metal quant kernel
      → unblock GGUF Flux + int8 T5?) is unconfirmed. Not worth churning the dependency yet. Stay
      on `0.10` (pinned). Revisit when a candle release reaches real stability (a matured 0.1x or
      1.0). The spike playbook is preserved for then: bump → `plakat verify` all 7 families
      (regression = corr < 1.0) → test the 3 blockers (GGUF-Flux-Metal / quant-matmul / DPM++
      without the CpuHopScheduler).
- [~] **SD UNet SDPA** — **Step 1 done:** SDPA in plakat's own SD attention (`sd_train/attention.rs`)
      → **stylize/instantstyle** get it now (GPU-only, head-dim-guarded, unmasked, probe-verified
      kernel). **Step 2 (the workhorses):** route main t2i through plakat's `sd_train::unet` (already
      exposes `forward`/`forward_sdxl` matching `SdCore`) instead of candle's `SdxlUNet2DConditionModel`
      — feasible but touches the `SdUNet` enum + LoRA + ControlNet + motion + weight-load keys; must be
      verify-gated (switch → `plakat verify` → corr 1.0 confirms UNet equivalence). Focused follow-up.
      *(Vendoring the SD UNet also decouples plakat from candle's churny surface — aligns with the
      candle-stability wariness.)*
- [~] **Free-quality guidance** — **PAG landed for PixArt** (opt-in `PLAKAT_PAG_SCALE=<s>`, default
      off ⇒ verify-safe; extra conditional forward with self-attn→identity, guided = cfg +
      pag·(cond−cond_ptb)). pag-off bit-identical by construction; pag-on coherent. Perturbs all
      self-attn blocks (strong; scale ~1–2). **Remaining:** SD3 MMDiT PAG, layer-subset selection,
      a `--pag-scale` CLI flag, then **FreeU** + **CFG-rescale** / dynamic thresholding.

## Carried-over (opportunistic)

- [ ] Masked cross-attn SDPA (finish the rollout; validate the additive-mask shape).
- [ ] `sd21 unet.out` verify symmetry (trivial).
- [ ] Perf-CI gate — `plakat bench` thresholds fail a PR on regression.

## House-keeping

- [x] **Open 2.5.0** — branch off `main` (2.4.0 release), version bump `2.4.0 → 2.5.0`.
