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

## B — carried map optional features (off-track, opt-in)

- [ ] **River + dry canyons** — carve gorges along high-flow channels + realize
      `terrain.rift_valleys` (`--map-canyons`). The remaining terrain-realism gap.
- [ ] **Plateaus / mesas** — realize `terrain.plateaus` as flat-topped scarped terrain.
- [ ] **Political layer** — borders + polity fills/labels from `RegionSpec.political`.
- [ ] **Seasonal palettes** (`--map-season`), **game-grid overlay** (`--map-grid`).

## C — carried product debt

- [ ] **IC-Light relighting** (carried since 1.1).
- [ ] **Flux regional prompting** (Metal-blocked → code + CI).
- [ ] **Flux training** — implementable (rectified-flow, mirrors SD3) but unverifiable on
      Metal; unblock once a CUDA/CI verify path exists.
- [ ] Fill `corpus/images/train/`; map gallery section in `GALLERY.md`.

## Notes

- `--features metal` (Apple Silicon) / `--features cuda` (NVIDIA) for GPU; default build is CPU-only.
- Training output is non-deterministic → each trainer lands with a `corpus/*_train.sh` driver +
  a committed **showcase**, verified on-box where memory permits.
