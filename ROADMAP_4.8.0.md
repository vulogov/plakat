# plakat 4.8.0 — roadmap: round out the Sana family

4.5–4.7 built Sana (DC-AE + Gemma-2 + Linear-DiT), then deepened it (DPM++, img2img, variants,
Sana-1.5, inpaint) and fixed the Metal DC-AE encode bug. 4.8.0 adds the two remaining reach items:
**outpaint** (canvas extension) and a **Sana ControlNet** path. Both build on the verified,
now-Metal-correct components; frozen paths stay byte-identical.

Ground rules: additive; each phase lands with a reference-corr or coherence check; `Cargo.lock` in sync;
no Anthropic/Claude attribution anywhere.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — Sana outpaint (`plakat outpaint --model sana`) — DONE

The `plakat outpaint` command is already model-agnostic: it builds a replicate-fill canvas + a
white-border mask and hands off to `img2img --mask` (the inpaint flow), which Sana now supports (4.7).
So the only real gap was the VAE snap constraint.

- [x] `outpaint.rs`: `snap` is now model-aware — Sana → **multiple of 32** (DC-AE 32×). SD=8 / Flux=16
      kept. The input-divisibility bail enforces mult-of-32 Sana inputs; Sana img2img defaults (20 steps,
      guidance 4.5, strength 1.0 for masked) flow through the existing inpaint dispatch.
- [x] Verify (Metal): `outpaint plakat-sana-7.png --model sana --right 64` → 512²→576×512, preserved
      region mean|Δ| **7.6** (≤ AE floor), new strip coherently continued (buildings/stalls/sky blend).

## Phase 2 — Sana ControlNet (`plakat generate --model sana --control-* …`)

Real + verifiable: diffusers 0.38 ships `SanaControlNetModel` + `SanaControlNetPipeline`; checkpoints
exist (`ishan24/Sana_600M_1024px_ControlNetPlus_diffusers`,
`Efficient-Large-Model/Sana_600M_1024px_ControlNet_diffusers`). Architecture = the standard ControlNet
copy-of-early-blocks pattern, reusing components we already have:
- Control image is **DC-AE-encoded** to a 32-ch latent (the encode we just Metal-fixed), patch-embedded,
  passed through an `input_block` (zero-conv-like) and **added to the main patch embedding**.
- A copy of the **first 7** Sana DiT blocks produces per-block residuals; each goes through a
  zero-init `controlnet_block` linear, is scaled by `conditioning_scale`, and is **added into the main
  DiT's hidden state after the matching block** (`hidden += residual[i-1]` for blocks 1..=7).

- [ ] New `sana_controlnet.rs`: `SanaControlNet` = `input_block` + N copied DiT blocks (reuse the
      `sana_dit` block type) + N zero `controlnet_blocks`; `Config` from the ControlNet `config.json`
      (num_layers=7, in=32, inner=heads·head_dim). Forward → `Vec<Tensor>` of scaled residuals.
- [ ] `sana_dit::SanaTransformer::forward` gains an optional `controlnet_residuals: Option<&[Tensor]>`
      arg, adding `residual[i-1]` after block `i` for `1..=len`. No-op / byte-identical when `None`.
- [ ] Pipeline wiring (`sana.rs`): load the ControlNet when a control image is given; DC-AE-encode the
      control image once; thread residuals through the denoise loop (both CFG passes). Reuse plakat's
      existing control-image preprocessing (canny/depth/etc. already produce the conditioning image).
- [ ] CLI/dispatch: route `--control` / `--control-image` / `--control-from` for Sana (they currently
      bail or are SD-only). Alias(es) for the ControlNet repo(s) in `hf/mod.rs`; capability note.
- [ ] Verify: `SanaControlNet` single-forward residuals match a diffusers dump
      (`tools/reference/sana_controlnet_dump.py`) at corr > 0.999; end-to-end a canny/HED control run
      on Metal produces an image that follows the control.

## Phase 3 — docs + release

- [ ] GENERATE / CONTROLNET tutorials: Sana outpaint + ControlNet notes; README what's-new; capability
      hints. Update [[reference_sana]]. Cut the 4.8.0 release.

## Notes / risks

- The public Sana ControlNets are **600M** — they pair with the `sana-600m` base (already supported),
  not the 1.6B. Document that `--model sana-600m` is the ControlNet base.
- Control conditioning is at the coarse 32× DC-AE latent grid (like inpaint); fine control edges soften.
- Sana-1.5 ControlNet / LoRA / multi-CN stacking: out of scope. Still deferred: Sana LoRA (no real LoRAs).
