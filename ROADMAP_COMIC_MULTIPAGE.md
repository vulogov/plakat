# plakat comic — multi-page / series (6.8.1)

A comic strip spans many pages that share a cast, a style, and an engine while the dialogue and panels
change. 6.8.0 ships one `ComicSpec` = one page. This cycle makes the **shared world propagate** across
pages without re-authoring it, and reused settings recur. Part of the 6.8 comic line (RFC COMIC-1).

The split: **structure** (weight-free — a page is inherited world + its own panels) lands first and ships
value alone; **visual reference-lock** (identity/style that actually holds at book scale) layers on top.

## Phases

### M1 — multi-page structure (weight-free) — **DONE (commit `d76e713`)**
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

### M2 — visual reference-lock (needs a model) — **DONE (commit `0f5403e`)**
- **Cast reference sheet** — `comic cast` renders one canonical portrait per character (`ref_<name>.png` +
  a lettered `cast_sheet.png`) at the model native square. `render --lock` builds them, loads
  `FaceSwapper::load_resolved` (SCRFD+ArcFace+inswapper), embeds each reference to an identity latent, and
  face-swaps it onto every **single-character** panel (unambiguous). Best-effort — no weights → stays
  description-level. **Live-proven Metal**: 2-page issue, 3 panels locked, same face on pages 0 + 1.
- **Style lock** — spec `style_lora` (+ `style_lora_scale`) on every panel + the reference portraits.
- `cast[].reference` overrides the rendered portrait; `cast[].lock:false` opts out.
- **Deferred**: multi-character-panel disambiguation (v1 skips ≥2 locked chars); scene-art reuse of a
  prior panel image; a `restore-faces` cleanup pass on small swapped faces.

### M3 — integration + cut 6.8.1 — **IN PROGRESS**
Parity already page-aware from M1/M2 (scenario `lock:`, `api::Comic::lock`, doctor, compile unchanged).
Corpus: `comic_series.hjson` (shared world) + `comic_issue.hjson` (`extends`, 2 pages) + `comic_run.sh`
multi-page/cast/lock steps + `COMIC_CORPUS.md`. Docs: COMIC.md multi-page + reference-lock; README 6.8.1
blockquote. **CUT 6.8.1**: Cargo+lock → 6.8.1; gate `--no-default-features --lib`; no new `.parse()`; FF
`git push 6.8.1:main`; tag → 6-asset CI; `cargo publish --locked --allow-dirty --no-default-features`;
`gh release edit` + bg waiter; verify the Windows leg. NO Claude/Anthropic coauthor.

## Sequencing
**M1** (structure, ships value weight-free) → **M2** (reference-lock, the visual hold) → **M3** (cut).
Honest limit today: identity is description-level and drifts over dozens of pages — M2 is what fixes it.
