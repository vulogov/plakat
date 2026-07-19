# plakat 3.6.0 — roadmap (SHIPPED)

**Shipped — generation in the manager + the true panorama.** The post-3.5.0 work that closed out the
3.5 cycle's Tracks A/B, cut as its own release.

## Shipped

- **Generation integrated into `plakat photos`** — `generate` (txt2img), `img2img`, `portrait`,
  `multiperson`, via the stable `crate::api` builders. Reachable from three surfaces: the new
  **`Ctrl-B n` "AI create" chord category** + edit palette (`n g`/`n i`/`n p`/`n m`), the **ML menu**
  (`M` → `g`/`p`/`m`), and the **`:` cmd pane** (NL verbs `generate`/`portrait`/`scene`/`img2img`).
  Prompt-driven; portrait uses the cursor image as the identity face, multiperson the selection as the
  people. TUI-suspended, OOM-guarded, lands a new `ai_*.png`. (`run_create_job`, `CreateOp`.)
- **Homography panorama** (`hm`) — `src/photos/homography.rs`: FAST corners → normalised-patch NCC
  matching → RANSAC homography (normalised 4-point DLT + Gaussian elimination, refit on inliers) →
  imageproc warp + feathered blend. Corrects rotation/perspective; edge-to-edge fallback per pair.
- **Aligned panorama** (`ha`/`va`) — cross-correlation overlap registration + cross-faded seam.
- **Mosaic / scrapbook collage** (`W`) — justified-rows layout with varied cell sizes.
- **Crystallize** (Voronoi / low-poly) — new 0–100 % filter (chord `s z`).
- **EditOp `Copy` refactor** — `EditOp`/`EditCmd` are now `Clone`-not-`Copy`, so ops carry
  `String`/`PathBuf` payloads. **Watermark** and **LUT** are now **replayable edits** (undo /
  copy-paste / presets), re-rendered on rebuild; missing font → bitmap fallback, missing `.cube` →
  no-op.

## Not done (deferred — need explicit go-ahead)

- **Track C distribution** — `cargo publish` (crates.io) + GitHub release assets + merge to `main`.
  Deferred across the whole 3.x line so far.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary — generation verbs produce a new album
  image (create-only); no external read, no exec. (Watermark/LUT name an external file → palette-only.)
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
