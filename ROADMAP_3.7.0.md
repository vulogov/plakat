# plakat 3.7.0 — roadmap (SHIPPED)

**Shipped — retouch + the last non-AI gaps.** An interactive crosshair retouch pick-mode (spot heal /
clone / red-eye / dodge-burn / 4-point perspective), creative effects (tilt-shift, motion/zoom/spin
blur, B&W channel mixer, film-negative), lens corrections (chromatic aberration, distortion), and two
manager passes (duplicate finder on the palette/chord, non-AI quality cull). All reuse existing
machinery (replayable slider `EditOp`s, the homography solver, the perceptual dHash, the Laplacian).

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

## Stretch — interactive pick-mode — DONE

- [x] **Crosshair pick-mode** (`src/photos/mod.rs` `PickState`/`handle_pick_key`/`pick_preview`; chord
      category `r`) — one interactive mode (arrows move, `+/-` brush size, Enter sets each point, Esc
      cancels) that unlocks all of:
  - [x] **Spot heal** (`r h`) — fill a disc from its boundary (dust / blemish removal).
  - [x] **Clone stamp** (`r c`) — pick source then destination; feathered copy.
  - [x] **Red-eye removal** (`r e`) — neutralise the red pupil glare in a disc.
  - [x] **Dodge / burn brush** (`r d` / `r b`) — soft lighten / darken.
  - [x] **4-point perspective rectify** (`r p`) — pick the corners → warp to fill the frame (reuses
        `homography::solve_homography`). Closes the deferred Track B item.
  - All five are **replayable EditOps** carrying per-mille coordinates.

## Deferred (need explicit go-ahead)

- [ ] **Track C distribution** — publish 3.6.0/3.7.0 to crates.io + GitHub release + merge to `main`.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
