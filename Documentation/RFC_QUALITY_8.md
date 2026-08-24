# RFC QUALITY-8 — naturalize round 5: model-aware presets · EXIF-aware · TUI (6.17.0)

**Status:** draft (6.17.0) · **Depth cycle** on [`RFC_QUALITY_7.md`](RFC_QUALITY_7.md). Make de-slop aware of
*where the image came from*, and interactive.

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

### P3 — naturalize TUI (interactive tuning)
`naturalize <img> --tui`: an interactive terminal panel (reuse the `plakat ui` / ratatui stack) to dial the
weight-free knobs (polish / micro / grain / desaturate / paper …) with the **live scorecard** updating and a
written preview on each apply, so you tune by eye instead of guessing flags. Weight-free only (instant).

### P4 — parity + docs + cut 6.17.0
Tutorial + QUALITY + doctor + README; corpus. Cut 6.17.0 (bump Cargo+lock, gate, turbofish on new
`.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
Model detection is metadata-driven (best-effort — stripped metadata → fall back to analysis). The per-model
presets are tuned heuristics, not per-image guarantees. The TUI tunes the weight-free pass only.

## Sequencing
**P1** model-aware presets → **P2** EXIF-aware → **P3** TUI → **P4** cut. Independent phases.
