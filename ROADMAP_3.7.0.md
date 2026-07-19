# plakat 3.7.0 — roadmap (planning)

Opening after 3.6.0 (generation in the manager + the true panorama). `plakat photos` is a deep
studio; 3.7 fills the remaining **non-AI** gaps a serious photo editor/manager still has — creative
effects, lens/geometry corrections, and two manager gaps — reusing the existing machinery (replayable
slider `EditOp`s, the homography solver, the perceptual dHash, the Laplacian).

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — creative & tonal (replayable slider EditOps) — DONE

- [x] **Tilt-shift / miniature** (chord `x m`) — in-focus band + graduated blur + saturation pop.
- [x] **Creative blurs** — **motion** (`x C`, h/v/diagonal), **zoom** (`x w`), **spin** (`x q`); a
      shared `path_blur` samples & averages along a line / radial / arc.
- [x] **Channel-mixer B&W** (`x B` + presets) — weighted mono (red/green/blue/orange/luminance).
- [x] **Film-negative conversion** (`x N`) — invert + per-channel auto-stretch (orange-mask removal).

## Track B — lens & geometry

- [x] **Chromatic-aberration removal** (`x A`) — rescale R/B channels radially about the centre.
- [x] **Lens distortion correction** (`g d`) — barrel / pincushion radial warp (bipolar).
- [ ] **4-point perspective rectify** — needs an interactive corner-pick mode; folded into the
      pick-mode stretch below (shares UX with clone/heal), so it lands with that or later.

## Track C — manager gaps (non-AI) — DONE

- [x] **Duplicate / near-duplicate finder** — the `dedup_scan` (perceptual dHash, keep-best +
      tag `dup`) was NL-only; now on the **edit palette + chord `m f`** too.
- [x] **Quality auto-cull (non-AI)** (`src/photos/quality.rs`, chord `m q`, NL `cull blurry`) —
      Laplacian-variance **sharpness** (adaptive floor = 35 % of the shoot's median) + **exposure**
      bounds → reject soft / too-dark / too-bright frames (metadata, undoable). Offline complement to
      the AI aesthetic cull.

## Stretch (needs a new interactive pick-mode)

- [ ] **Spot heal / clone stamp** (+ red-eye, dodge/burn brush) — the biggest "real editor" gap; a
      crosshair pick-mode would unlock all three. Deferred unless we invest in the UX this cut.

## Deferred (need explicit go-ahead)

- [ ] **Track C distribution** — publish 3.6.0/3.7.0 to crates.io + GitHub release + merge to `main`.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
