# plakat 1.11.0 — roadmap

1.10.0 shipped the **model-training expansion**: SD 2.1 LoRA + DreamBooth (verified),
PixArt-Σ LoRA, SD 3.5 Textual Inversion, and Stable Cascade Stage-C LoRA. The honest
caveat from that cycle: the three **transformer** trainers (PixArt / SD3.5 / Cascade)
keep their giant encoders resident with autograd and are **memory-bound** — code-complete
but not on-box-verifiable on 24 GB. 1.11.0's natural headline is to **close that gap** and
pick up carried debt.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## A — make the memory-bound trainers verifiable (headline candidate)

The blocker is peak unified memory during the autograd forward, not correctness. Options,
cheapest first:

- [ ] **Gradient checkpointing** on the transformer denoisers (recompute block activations
      in the backward pass instead of storing them) — the standard fix; should bring PixArt /
      SD3.5 / Cascade Stage-C training under 24 GB so they can be **verified on-box**.
- [ ] **8-bit / paged optimizer state** + keep frozen encoders in BF16, adapters F32 (already
      done) — trims optimizer + activation memory further.
- [ ] **CPU-offload the frozen text encoders** during the training loop (T5-XXL is the hog;
      it only encodes the trigger once — cache its output and drop it before the loop).
      Caching the (frozen) conditioning is the biggest single win and is trainer-local.
- [ ] **Verify each trainer end-to-end** once it fits: a `corpus/*_train.sh` run + a committed
      showcase, same convention as SD 2.1.

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
