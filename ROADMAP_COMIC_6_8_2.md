# plakat comic 6.8.2 — finish the reference-lock deferrals

Three items deferred from 6.8.1's reference-lock (RFC COMIC-1), each closing an honest limit.

## Phases

### D1 — multi-character-panel identity disambiguation
6.8.1 only face-locks **single-character** panels (unambiguous). Extend `lock_panel` to N characters:
map each detected face to a character by **reading-order position** — sort faces by centroid x
(ascending ltr / descending rtl), zip with `panel.chars` order (the author controls who's where by list
order), and swap each character's reference onto its face. Skip characters without a reference; align by
order when face/char counts differ. Weight-free logic (needs the swap weights only to run live).

### D2 — scene-art reuse (exact repeat, not just recurring descriptions)
Today `@scene` names recur a *description* (re-generated each time → different art). Add exact art reuse:
`Panel.id` labels a panel; `Panel.reuse: "@id"` renders that panel as a **copy of the labelled panel's
image** (book-wide, so an establishing shot repeats identically). Render plumbing carries an id→art map
across pages; a reuse panel skips generation (and re-locking). Weight-free plumbing.

### D3 — restore-faces pass on small swapped faces
A face-swap onto a small face (distant or group shot) can look rough. After locking, when the swapped face
is small relative to the panel, run the existing `restore-faces` (adetailer) refine on that panel to clean
it up. Opt-in (`--restore-faces` / `restore: true`); best-effort (needs the restore pipeline).

### D4 — docs + corpus + cut 6.8.2
COMIC.md + doctor update; a corpus panel exercising reuse + a 2-character panel; **CUT 6.8.2** (Cargo+lock,
gate `--no-default-features --lib`, pin turbofish on new `.parse()`, FF main, tag → 6-asset CI,
`cargo publish`, `gh release edit`, verify the Windows leg, NO Claude/Anthropic coauthor).

## Status — ALL DONE (cutting 6.8.2)
- **D1** ✅ multi-character lock by reading-order position (`reading_order` helper + `lock_panel` chains
  swaps; faces matched to `chars` order). Live-proven (2-shot → both faces swapped). Honest caveat
  documented: model arrangement may disagree with `chars` order → order `chars` to match / pin in `scene`.
- **D2** ✅ `Panel.id` + `Panel.reuse: "@id"`; `render_page_panels` carries a book-wide id→art map, reuse
  skips generation. Lint flags unknown ids. Live-proven (reuse panel shared the exact opening art).
- **D3** ✅ `restore_small_faces` (adetailer `refine_files`, strength 0.35) over panels whose swapped face
  < 22% of panel height; `--restore-faces` / `restore:`. render_spec split into render+lock → restore →
  composite so the refine lands before assembly. Live-proven (1 panel refined).
- **D4** — corpus (issue `id`/`reuse` + driver `--restore-faces`), COMIC.md/doctor, cut 6.8.2.

## Sequencing
**D1** + **D2** (weight-free logic, testable) → **D3** (restore pipeline) → **D4** (cut).
