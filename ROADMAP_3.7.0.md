# plakat 3.7.0 — roadmap (planning)

Opening after 3.6.0 (generation in the manager + the true panorama). `plakat photos` is a deep
studio; 3.7 fills the remaining **non-AI** gaps a serious photo editor/manager still has — creative
effects, lens/geometry corrections, and two manager gaps — reusing the existing machinery (replayable
slider `EditOp`s, the homography solver, the perceptual dHash, the Laplacian).

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — creative & tonal (replayable slider EditOps)

- [ ] **Tilt-shift / miniature** — graduated blur above/below an in-focus band + a saturation/contrast
      pop. Single slider (blur amount).
- [ ] **Creative blurs** — **motion** (directional, angle presets), **zoom** (radial from centre),
      **spin** (rotational). Sample-along-a-path averaging; slider = amount.
- [ ] **Channel-mixer B&W** — weighted mono (red/green/blue/orange filter presets) — a real mono
      conversion, not the flat desaturate.
- [ ] **Film-negative conversion** — invert + per-channel auto-stretch to remove the orange C-41 mask
      (scanned negative → positive).

## Track B — lens & geometry

- [ ] **Chromatic-aberration removal** — per-channel radial scale about the centre + fringe desaturate.
- [ ] **Lens distortion correction** — barrel / pincushion warp (manual amount).
- [ ] **4-point perspective rectify** — pick the corners of a plane → rectify (reuses
      `homography::solve_homography` from the panorama stitcher).

## Track C — manager gaps (non-AI)

- [ ] **Duplicate / near-duplicate finder** — a library-wide perceptual-dHash pass; group + flag the
      dups for review (today the lookalike only compares against one image).
- [ ] **Quality auto-cull (non-AI)** — Laplacian-variance **blur** score + under/over-exposure flags →
      auto-reject soft / badly-exposed shots (offline complement to the AI aesthetic cull).

## Stretch (needs a new interactive pick-mode)

- [ ] **Spot heal / clone stamp** (+ red-eye, dodge/burn brush) — the biggest "real editor" gap; a
      crosshair pick-mode would unlock all three. Deferred unless we invest in the UX this cut.

## Deferred (need explicit go-ahead)

- [ ] **Track C distribution** — publish 3.6.0/3.7.0 to crates.io + GitHub release + merge to `main`.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
