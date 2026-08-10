# plakat comic — multi-page / series (6.8.1)

A comic strip spans many pages that share a cast, a style, and an engine while the dialogue and panels
change. 6.8.0 ships one `ComicSpec` = one page. This cycle makes the **shared world propagate** across
pages without re-authoring it, and reused settings recur. Part of the 6.8 comic line (RFC COMIC-1).

The split: **structure** (weight-free — a page is inherited world + its own panels) lands first and ships
value alone; **visual reference-lock** (identity/style that actually holds at book scale) layers on top.

## Phases

### M1 — multi-page structure (weight-free)
- `ComicSpec` gains `pages: [ { name?, layout?, reading?, panels[] } ]`. The top-level `cast`/`style`/
  `model`/`seed`/`steps`/`page` are the **shared world**; each page carries only its own layout + panels.
  A single-page spec (top-level `panels`, no `pages`) still works unchanged (back-compat).
- `scenes: { alley: "…" }` — a **named scene library**; a panel `scene: "@alley"` resolves to it, so a
  setting recurs across pages by reference (define once, reuse).
- `extends: "series.hjson"` — a page/issue spec **inherits** a base spec (shared world in one file, pages
  in another). Merge: base = defaults; child overrides scalars, merges `cast`/`scenes` by name, replaces
  `pages`/`panels` when non-empty.
- Layout engine refactored to resolve **per logical page** (each page indexes its own panels).
- `render_spec` loops pages: `out.png` → `out_00.png, out_01.png, …` (single page keeps `out.png`); a
  **global panel-seed counter** across pages so panels stay distinct + reproducible.
- CLI `comic show`/`lint`/`layout`/`letter`/`render` all page-aware; `--page N` to target one.
- Docs + corpus (a 2-page series) + the walkthrough.

### M2 — visual reference-lock (needs a model)
- **Cast reference sheet**: persona renders each character once → a canonical portrait; every panel
  conditions on it (IP-Adapter / face-swap — plakat has `ip_adapter` + `inswapper`) so the *same face*
  survives book-wide, beyond description-level drift.
- **Shared style lock**: a style LoRA / InstantStyle reference applied to every panel of every page.
- Scene-art reuse: a panel may reuse a previously-rendered panel's image (exact repeat of a setting).

### M3 — integration + cut 6.8.1
Parity refresh (scenario/compile/Bund/api/doctor page-aware), corpus + docs, **CUT 6.8.1** (Cargo+lock,
gate `--no-default-features --lib`, pin turbofish on new `.parse()`, FF main, tag → 6-asset CI,
`cargo publish`, `gh release edit`, verify the Windows leg, NO Claude/Anthropic coauthor).

## Sequencing
**M1** (structure, ships value weight-free) → **M2** (reference-lock, the visual hold) → **M3** (cut).
Honest limit today: identity is description-level and drifts over dozens of pages — M2 is what fixes it.
