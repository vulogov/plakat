# RFC QUALITY-8 — naturalize round 5: model-aware presets · EXIF-aware · TUI (6.17.0)

**Status:** SHIPPED (6.17.0) · **Depth cycle** on [`RFC_QUALITY_7.md`](RFC_QUALITY_7.md). Make de-slop aware of
*where the image came from*, and interactive.

**All phases shipped:** P1 model-aware presets + `--preset auto` · P2 EXIF-aware (gentle touch on real
photos) · P3 Naturalize tab in `plakat ui` (9th `ActiveScreen`, live scorecard) · P4 open external images in
the tab (`o` path-input) · P5 naturalize + etch + verify in `plakat photos`.

## What ships

### P1 — model-aware presets + `--preset auto`
Each model family has its own tells — SDXL over-saturates, SD 1.5 is soft, Flux is clean-but-plastic,
Cascade/PixArt/SD3.5/Sana differ again. Add per-model preset specs and **`--preset auto`**: read the input's
generation metadata (the plakat sidecar / A1111 `parameters` chunk), identify the model, and apply the
preset tuned to that model's fingerprint. No metadata → fall back to the analysis-driven recommendation.

### P2 — EXIF / metadata-aware de-slop
- **Detect** the source model from metadata to drive `--preset auto` (P1's input).
- **Preserve** the image's metadata through the pass (today the PNG `parameters`/`etch` text chunks are
  carried; extend to EXIF where present so a de-slopped photo keeps its camera tags).
- **Lighter touch on real photos**: if EXIF says it's a genuine camera capture (Make/Model/exposure present),
  default to a gentler recipe — de-slop shouldn't fight a real photograph.

### P3 — naturalize tab in `plakat ui` (interactive tuning)
**Owner decision (2026-08-24): a new tab in the main `plakat ui`, NOT a standalone TUI** — reuse the
existing ratatui event loop / image display / navigation instead of duplicating them. Add a 9th
`ActiveScreen::Naturalize` with a `screens/naturalize.rs` `NaturalizeState` (open an image, dial the
weight-free knobs — polish / micro / grain / desaturate / paper — with a **live scorecard** and image
preview updating on each apply, then save). Weight-free only (instant, no model). The naturalize *core*
stays always-compiled + gate-tested; the tab is a thin presentation layer behind the existing `ui` feature,
so the `--no-default-features` gate is unaffected.

### P4 — open external images in the Naturalize tab
Owner: work with external photos, not just generations. Add an **"open path"** input to the Naturalize tab
(`o` → type a file path → Enter loads it); the tab already falls back to the workspace out-dir. Small — a
capture-input mode on the screen.

### P5 — naturalize + etch + verify in `plakat photos`
Owner: integrate de-slop **and provenance** into the photo manager (the home for **external** photos — it
imports HEIC/AVIF/raw into albums with EXIF), so photos becomes a full **import → de-slop → etch → verify**
pipeline. Three edit-palette actions, all reusing existing cores:
- **naturalize** — `EditOp::Naturalize(strength)`, a first-class edit op (the naturalize *Photo* recipe
  scaled 0–100). Deterministic → replay-safe, so it inherits undo/redo/versioning through the edit
  pipeline. Palette "naturalize (weight-free de-slop)…" (chord `an`) opens the live strength slider.
- **etch provenance** — palette "etch plakat provenance (into file)…" (chord `me`) → confirm →
  `etch::fresh_etch` a fresh L0 manifest + L1 pixel mark into the target file(s) in place (batch).
- **verify provenance** — palette "verify provenance (is it plakat?)" (chord `mv`) → the offline L0+L1
  `doctor --if-plakat` engine on the cursor image → verdict in the status line.

### P6 — parity + docs + cut 6.17.0
Tutorial + QUALITY + doctor + README; corpus. Cut 6.17.0 (bump Cargo+lock, gate `--test-threads=1`,
turbofish on new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
Model detection is metadata-driven (best-effort — stripped metadata → fall back to analysis). The per-model
presets are tuned heuristics, not per-image guarantees. The TUI tunes the weight-free pass only.

## Sequencing
**P1** model-aware presets → **P2** EXIF-aware → **P3** TUI → **P4** cut. Independent phases.
