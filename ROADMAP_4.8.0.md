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

## Phase 2 — Sana ControlNet (`plakat generate --model sana-600m --control-* …`) — DONE

Real + verifiable: diffusers 0.38 ships `SanaControlNetModel` + `SanaControlNetPipeline`; checkpoints
exist (`ishan24/Sana_600M_1024px_ControlNetPlus_diffusers`,
`Efficient-Large-Model/Sana_600M_1024px_ControlNet_diffusers`). Architecture = the standard ControlNet
copy-of-early-blocks pattern, reusing components we already have:
- Control image is **DC-AE-encoded** to a 32-ch latent (the encode we just Metal-fixed), patch-embedded,
  passed through an `input_block` (zero-conv-like) and **added to the main patch embedding**.
- A copy of the **first 7** Sana DiT blocks produces per-block residuals; each goes through a
  zero-init `controlnet_block` linear, is scaled by `conditioning_scale`, and is **added into the main
  DiT's hidden state after the matching block** (`hidden += residual[i-1]` for blocks 1..=7).

- [x] `SanaControlNet` in `sana_dit.rs` (same module → reuses the private `Block` + helpers): an
      `input_block` Linear on the patch-embedded control latent + N copied blocks + N zero
      `controlnet_blocks` Linears + its own time/caption front-matter. `Config` from the ControlNet
      `config.json` (7 layers, inner 1152). `forward → Vec<Tensor>` of scaled residuals.
- [x] `SanaTransformer::forward_control(..., Option<&[Tensor]>)` — adds `residual[i-1]` after block
      `i` for `1..=len` (diffusers window). `forward` delegates with `None` → byte-identical.
- [x] Pipeline wiring (`sana.rs`): `load_controlnet` (async, from the CN repo `controlnet/` subfolder,
      bails if hidden-dim ≠ base DiT → "use sana-600m"); DC-AE-encode the control once via `encode_init`;
      per-step the CN runs on the doubled latent+control and its residuals steer the DiT; freed with the DiT.
- [x] CLI/dispatch: the Sana arm resolves one `--control`/`--control-image`/`--control-from` spec
      (auto-annotate via the shared annotator), single-CN only; repo pinned to `SANA_CONTROLNET_REPO`.
      (No new model alias → not a `doctor --capability` item.)
- [x] Verify: `SanaControlNet` residuals match a diffusers dump (`tools/reference/sana_controlnet_dump.py`)
      at **corr 1.000000** (worst 0.999996); end-to-end canny-guided run on Metal follows the control —
      ablation NCC-to-source **0.166 with vs 0.079 without** (same prompt+seed → 2× structural alignment).

## Phase 3 — docs + release

- [x] GENERATE tutorial: Sana outpaint + ControlNet subsections (600M/coarse-grid notes); README banner
      + "what's new in 4.8.0". `reference_sana` memory updated (4.8 section).
- [x] Cut the 4.8.0 release — v4.8.0 @ 9c67ace: tag pushed → Release CI green (6 assets + SHA256SUMS),
      `cargo publish --locked` (on crates.io), main fast-forwarded, notes set via `gh release edit`.

## Notes / risks

- The public Sana ControlNets are **600M** — they pair with the `sana-600m` base (already supported),
  not the 1.6B. Document that `--model sana-600m` is the ControlNet base.
- Control conditioning is at the coarse 32× DC-AE latent grid (like inpaint); fine control edges soften.
- Sana-1.5 ControlNet / LoRA / multi-CN stacking: out of scope. Still deferred: Sana LoRA (no real LoRAs).
