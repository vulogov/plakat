# plakat 3.5.0 — roadmap (SHIPPED)

**Shipped — AI comes to the manager.** Resident ML worker (img2img/relight/upscale, inline progress
+ cancel + residency, OOM-guarded), memory indicator; AI menu: aesthetic auto-cull, analyze &
generate, face-scan; CLIP visual-search verified. Hybrid **face polish** (AI-detected mask + non-AI
skin smooth, 0–100 %). More non-AI: better-sky, Kelvin/auto white balance, gradient-map, cross-hatch,
Apple/iOS look filters. Track B (below) + Track C distribution carry to 3.6.0.

---


Opening the cycle after 3.4.0 (the photos "full studio" — a deep non-AI creative/finishing pass +
composites). The non-AI editing surface is now very complete. With the **models cache reconnected**,
3.5 is the natural point to (re)open the **AI track** inside `plakat photos`, alongside a few
remaining non-AI polish items and the distribution loop. Candidate tracks — narrowed with the owner
before build.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — AI in the manager (cache now connected)

- [x] **Phase A — resident ML worker (photos-local).** `src/photos/mlworker.rs`: img2img / relight /
      upscale run on a background thread that holds the pipelines **resident** across ops; progress +
      cancel drain **inline** (no TUI suspend). OOM-safe (MemoryGuard + pressure preflight + panic
      catch + abort hook) and a top-bar memory indicator with "avoid AI when low" guidance. `plakat ui`
      untouched. See `Documentation/Photos/AI_LOADING_REVIEW.md`.


- [x] **CLIP visual search — live-verify.** DONE — cache reconnected;
      `cargo test --lib --features photos clip_loads_and_embeds_into_joint_space -- --ignored` passes
      (1 passed, 18.4s): real `openai/clip-vit-large-patch14` weights load and embed text+image into
      the joint space. The in-tree CLIP path is confirmed end-to-end. (Next: exercise `V` text→image +
      `Ctrl-B L` image→image on a real album interactively.)
- [x] **Face-scan** — DONE (`src/photos/faces.rs`, AI menu `f`). SCRFD (auto-download) detects faces
      across the library; when `PLAKAT_ARCFACE_WEIGHTS` is set, ArcFace embeds + greedy-clusters into
      `person-N`. Tags each image (`has-face` / `faces-N` / `person-K`) so the tag filter + smart
      albums surface people. Degrades to detection-only counts without ArcFace weights.
- [x] **Analyze-and-generate** — DONE (AI menu `n`). Describes the reference with the configured LLM
      vision provider → a compact prompt → img2img on the resident worker into a "reimagined" variant
      (also stored as the image's caption).
- [x] **Aesthetic auto-cull** — DONE (`src/photos/cull.rs`, AI menu `r`). Ranks the album with the
      LAION predictor (CLIP ViT-L + MLP), flags the top-N (you type N), rejects the rest (metadata
      only, undoable), writes scores to the `.json` sidecars. All three arm the OOM guard + refuse on
      critical memory pressure.

## Track A½ — hybrid (AI-assist for a non-AI filter)

- [x] **Face polish** (edit palette / chord `x c`, 0–100 % slider). SCRFD detects the faces (the mask
      you'd normally paint by hand), converts each to a compact ellipse stored *in* the `FacePolish`
      edit op, then a non-AI edge-preserving skin smooth runs limited to those regions. The AI cost is
      paid once at creation; the slider preview + replay are pure geometry (op stays `Copy`, no model
      reload). Detection is TUI-suspended + OOM-guarded; identity at strength 0 / no faces.

## Track A¾ — major generation features in the manager

- [x] **generate / img2img / portrait / multiperson** integrated into `plakat photos` — DONE, via the
      stable `crate::api` builders (`Generate`/`Img2img`/`Portrait`/`Multiperson`). Reachable from all
      three surfaces: the **command palette** + **`Ctrl-B n` chords** (new "AI create" category:
      `n g`/`n i`/`n p`/`n m`), the **ML menu** (`M` → `g`/`p`/`m`), and the **`:` cmd pane** (NL verbs
      `generate …`/`portrait …`/`scene …`/`img2img …`). Generate is prompt-only; portrait uses the
      cursor image as the identity face; multiperson uses the selected images as the people. Run
      TUI-suspended with the OOM guard + critical-pressure refuse; land a new `ai_*.png` in the album.

## Track B — remaining non-AI polish

- [x] **EditOp `Copy` refactor** — DONE. `EditOp`/`EditCmd` are now `Clone`-not-`Copy`, so ops can
      carry `String`/`Option<String>` payloads. **Watermark** (`EditOp::Watermark { text, font }`) and
      **LUT** (`EditOp::Lut { path }`) are now **replayable edits** in the stack (undo / copy-paste /
      presets), re-rendered on rebuild — replacing the old bake-a-new-file path. A missing font falls
      back to the bitmap font; a missing `.cube` replays as a no-op, so the stack never breaks. Still
      palette-only (they name an external file → out of the `:` planner).
- [x] **Mosaic / scrapbook collage** (varied cell sizes) — DONE (`stitch::mosaic`, grid `W`, NL
      "mosaic"). Justified-rows (Flickr-style) layout: images keep aspect, each row scaled to fill the
      width so cell sizes vary.
- [x] **Aligned panorama** (`ha` / `va`) — overlap registration by cross-correlation (translation) +
      cross-faded seam, with an edge-to-edge fallback.
- [x] **Homography panorama** (`hm`, NL "panorama homography") — DONE. `src/photos/homography.rs`: FAST
      corners → normalised-patch NCC matching (Lowe ratio, permissive → RANSAC filters) → RANSAC
      homography (normalised 4-point DLT + Gaussian elimination, refit on inliers) → imageproc `warp`
      + feathered (blurred-mask) blend. Corrects **rotation + perspective**. Per-pair confidence guard
      (≥12 inliers, sane canvas bounds) → falls back to edge-to-edge concat. Own 3×3 linalg + LCG
      (deterministic). Verified: recovers a known synthetic warp within tolerance.
- [x] Quick creative leftovers: **gradient map** (warm/cyanotype/fire/teal-orange), **cross-hatch**,
      **Kelvin white balance**, **crystallize (Voronoi / low-poly)** (`stitch`-free pure filter, chord
      `s z`, NL "crystallize/voronoi") — all adjustable 0–100 % (Kelvin bipolar −100..100), palette +
      chords + NL verbs.
- [x] **"Better sky"** (`EnhanceSky`) — no-AI, no-manual-mask sky enhancer: soft sky mask (vertical
      prior × blue-dominance/overcast-brightness) + polarizer (deepen & saturate blue); adjustable
      0–100 %, chord `x y`, NL "enhance sky". **Auto white balance** (`AutoWhiteBalance`, gray-world),
      chord `k a`. **Apple Photos / iOS classic filter looks**: fade, chrome, process, transfer,
      instant, tonal + cinematic bleach-bypass & teal-&-orange (palette-only look presets).

## Track C — distribution & housekeeping

- [ ] **Publish 3.3.0 + 3.4.0** — `cargo publish` (crates.io) + GitHub release assets (deferred at
      both cuts).
- [ ] **Merge 3.3.0 / 3.4.0 → `main`.**
- [ ] Carry-throughs from the 2.4 performance pass.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays inside the closed, album-scoped vocabulary — `export`/`convert` the only
  create-only outward writes; no external read, no exec. (Watermark/LUT read a named file, so they
  stay palette-only.)
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
