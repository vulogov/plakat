# RFC QUALITY-9 — naturalize round 6: batch · presets · per-image LUT (6.18.0)

**Status:** draft (6.18.0) · **Depth cycle** on [`RFC_QUALITY_8.md`](RFC_QUALITY_8.md). Round 6 makes the
weight-free de-slop **faster at scale**, **reusable**, and **more honestly captured**. Weight-free only
(instant, no model) — the reliable headline stays the headline.

## What ships

### P1 — per-image adaptive LUT (close the 6.16 honest limit)
Today `--export-lut grade.cube` bakes only the **fixed** film grade (desaturate + warm); the per-image
white-balance and auto-levels are *not* captured (documented limitation in 6.16). Round 6 adds
**`--export-lut` from an input image**: fit the per-image colour transform on that image (gray-world WB
gains + ratio-preserving auto-level black/white points + vibrance + the grade), then bake *those fitted
scalars* into the `.cube` by replaying them on the identity lattice. Honest carve-out: the **spatial**
stages (unsharp, micro-texture) are not colour LUTs and are dropped — the exported LUT is the full
**colour** grade for that image, not the sharpening. So `naturalize photo.png --export-lut look.cube`
gives a Resolve/Premiere/OBS grade that matches what de-slop did to *that* photo's colour.

### P2 — save / load a tuned preset
Round 5 shipped 6 built-in named presets (read-only). Round 6 makes them **user-authored**:
- **`naturalize --save-preset <name>`** serialises the current knob spec (a new `naturalize::to_spec`)
  into a user preset store (`$XDG_CONFIG_HOME/plakat/naturalize.presets`, INI-simple `name = spec`).
- **`--preset <name>`** and `--list-presets` resolve **user presets** too (user store shadows built-ins).
- In the **`plakat ui` Naturalize tab**: **`w`** saves the current knobs as a named preset (prompt), and
  the tab can **load** a named preset (cycle with `[`/`]` or a small picker), so a dialed-in look is reusable
  across images without re-tuning.

### P3 — batch de-slop in `plakat photos`
Round 5 made naturalize a first-class `EditOp` on the **cursor** image. Round 6 adds **"naturalize all
selected"** to the edit palette (chord `aN`): apply the op across every selected target via the existing
`apply_edit_ops_to_targets` path (so it's recorded, undoable, versioned per image). Turns a multi-select
into a one-keystroke folder de-slop inside the manager.

### P4 — parity + docs + cut 6.18.0
Tutorial + QUALITY + doctor + README; corpus. Cut 6.18.0 (bump Cargo+lock, gate `--test-threads=1`,
turbofish on new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
The per-image LUT captures **colour** only (WB + levels + vibrance + grade); spatial sharpening/micro
can't be a colour LUT and are excluded — documented. User presets are spec strings (the same weight-free
knobs), not per-image guarantees. Batch de-slop uses one strength across the selection (per-image tuning
is still the tab / single-image path).

## Sequencing
**P1** per-image LUT → **P2** presets → **P3** photos batch → **P4** cut. Independent phases.

## Deferred (not this cycle)
Interactive region **drawing** in the ui tab (mouse/rubber-band region select) — heavier ratatui input
work; the CLI `--region`/`--auto-regions` (6.15) already covers region de-slop. Revisit if asked.
