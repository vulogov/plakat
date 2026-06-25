# plakat 1.12.0 — roadmap

1.11.0 shipped IC-Light relighting, the map terrain/cartography features, and the
single-instance guard + `models pull` fixes. That cycle surfaced — repeatedly — that
the real ceiling on this class of hardware is **unified memory**: the transformer
trainers OOM at the first backward, even a default-size SD 1.5 render can spike Metal's
single-buffer allocation, and concurrent runs thrash. 1.12.0's natural theme is
**memory & stability** — make the heavy paths fit and fail gracefully — plus carried
map / debt items.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## 0 — multiperson scene placement (landed)

- [x] **`plakat multiperson` — M1 (place specific personas into a generated scene).**
      Each persona gets a relative location in words — position (left/center/right) ·
      distance (closer/farther) · facing (front/side/back); omit it and a scene-aware LLM
      auto-places them. Renders via the portrait pipeline (scene base → per-persona inpaint,
      farther→closer). Reuses `region_mask` + IP-Adapter + portrait inpaint; no new attention
      plumbing. `corpus/multiperson.sh` + showcase; verified on-box (the chess scene).
      Design/critique in `RFC_MULTIPERSON_REVIEW.md`. M2 (regional eps-blend, seam-free) and
      M3 (masked decoupled IP, single-pass + fidelity) remain optional upgrades.

## A — memory & stability (headline)

- [ ] **Cap render size on Metal** — the default `generate` size spikes Metal's
      single-buffer allocation on SD 1.5 (and large SDXL) and can OOM a 24 GB box.
      Auto-cap / warn, or transparently tile large single-pass renders. (1.11.0 worked
      around it by pinning the resume-train render to 512².)
- [ ] **Gradient checkpointing** — the blocker for verifying the PixArt / SD3.5 / Cascade
      trainers on 24 GB. candle has no native support; prototype manual detach + recompute
      (or a custom op) on one denoiser and measure. If infeasible, document the dead end.
- [ ] **OOM-guard tuning** — the guard correctly aborts genuine pressure, but it can fire
      on a transient first-backward / decode spike that the OS would otherwise absorb via
      swap. Consider a longer sustained window or a "training vs inference" sensitivity.
- [ ] **Single-instance guard polish** — optional `models`/`gallery` exemptions are in;
      consider a `--wait` mode (block until the running instance frees the host) and
      surfacing the guard in `doctor`.

## B — map optional features (off-track, opt-in)

- [ ] **Multi-tile world maps** — stitch adjacent tiles into a seamless world map.
- [ ] River **deltas** at navigable mouths; **marsh hatching** for Wetland regions.
- [ ] Carry: seasonal palette on the **painted** (`--map-render-sd`) path; the political
      layer in GeoJSON/SVG export.

## C — carried product debt

- [ ] Memory-bound **SD3.5 DreamBooth** / `regional.sh sdxl|sd35` renders (need the
      memory headroom from A).
- ~~Flux regional prompting / Flux training~~ — **skipped** while Flux is broken on Metal
      (unverifiable). Park until a CUDA/CI path exists.

## Notes

- `--features metal` (Apple Silicon) / `--features cuda` (NVIDIA) for GPU; default build
  is CPU-only.
- Training output is non-deterministic → each trainer lands with a `corpus/*_train.sh`
  driver + a committed showcase, verified on-box where memory permits.
