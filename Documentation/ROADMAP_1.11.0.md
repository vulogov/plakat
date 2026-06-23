# plakat 1.11.0 — roadmap

1.10.0 shipped the **model-training expansion**: SD 2.1 LoRA + DreamBooth (verified),
PixArt-Σ LoRA, SD 3.5 Textual Inversion, and Stable Cascade Stage-C LoRA. The honest
caveat from that cycle: the three **transformer** trainers (PixArt / SD3.5 / Cascade)
keep their giant encoders resident with autograd and are **memory-bound** — code-complete
but not on-box-verifiable on 24 GB. 1.11.0's natural headline is to **close that gap** and
pick up carried debt.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — make the memory-bound trainers verifiable (INVESTIGATED → blocked on 24 GB)

**Outcome (2026-06-22):** measured all three transformer trainers on-box. **Every one OOMs
at the first backward pass** — load + encode succeed; the autograd graph for the first
training step is the wall. Root cause: **candle has no gradient checkpointing**, so the
activation graph can't be shrunk, and this 24 GB box shares ~7 GB with other apps (~16 GB
free). Even PixArt's small 0.6 B DiT (trained F32) spikes past that. True verification needs
**≥ 32 GB unified / CUDA**, or a *fully* freed 24 GB box. The honest 1.10.0 state stands:
the trainers are code-complete + memory-bound.

- [x] **OOM guard validated under real training load** — it fired + aborted cleanly every
      time, no host crash. (The 1.10.0 fix that installed it on the training path works.)
- [x] **Encoder-drop (Cascade)** — precompute x0 + text, then drop CLIP-G + effnet before
      Stage C loads/trains (~1.7 GB). Correct footprint reduction; helps on a ≥ 32 GB box.
- [~] **Gradient checkpointing** — the real fix, but candle's autograd has no native support
      (would need manual detach + recompute / a custom op). Parked; revisit if candle gains it
      or a CUDA/≥36 GB verify path appears. Frozen-conditioning caching + CPU-offload were the
      cheaper ideas but don't dent the per-step autograd spike that is the actual wall.

## B — carried map optional features (DONE — canyons + plateaus + political)

- [x] **Dry canyons (rift valleys)** — `apply_canyons` carves narrow oriented gorges
      (floor above sea level → dry, only-lowers, erosion-wandered) realizing
      `terrain.rift_valleys`. `NamedRegion` gained optional orientation/length/size.
- [x] **Plateaus / mesas** — `apply_plateaus` raises flat-topped tablelands with steep
      scarp rims (preserves higher peaks) from `terrain.plateaus`.
- [x] **Political layer** — `draw_political` realizes `RegionSpec.political`: dashed
      territorial rings (name-hashed colour), inter-region borders styled by kind
      (disputed/river/mountain), polity labels.
- All three are pure fns of (spec, seed/style), gated so empty data → byte-identical;
      `corpus/map/realms.spec.json` showcase + heightmap/render proofs + byte-checks.
- [x] **Seasonal palettes** (`--map-season` spring/summer/autumn/winter) + **game-grid overlay**
      (`--map-grid N`, A1/B2 cells). On `Style`, gated → defaults byte-identical. **B COMPLETE.**

## C — carried product debt (Flux items skipped — broken on this hardware)

- [~] **IC-Light relighting** (carried since 1.1) — IN PROGRESS. SD1.5-based relighting:
      the UNet input conv is widened to take the subject latent as extra channels; reuse
      SdCore (SD1.5 UNet + VAE) + the U2Net matte for foreground extraction.
- [ ] Fill `corpus/images/train/`; map gallery section in `GALLERY.md`.
- ~~Flux regional prompting / Flux training~~ — **skipped**: Flux is broken on this
      hardware (Metal), so unverifiable. Parked until a CUDA/CI path exists.

## Notes

- `--features metal` (Apple Silicon) / `--features cuda` (NVIDIA) for GPU; default build is CPU-only.
- Training output is non-deterministic → each trainer lands with a `corpus/*_train.sh` driver +
  a committed **showcase**, verified on-box where memory permits.
