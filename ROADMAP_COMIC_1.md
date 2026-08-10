# ROADMAP — plakat 6.8.0 · `plakat comic` (RFC COMIC-1)

A new flagship: multi-panel comic pages from a script. Sibling of `bookart` (page layout) + `persona`
(character identity across panels). Branch `6.8.0` (off `main` @ `0c54de6`, v6.7.0). RFC:
`Documentation/RFC_COMIC_1.md`.

**Grounding:** reuse points confirmed — `bookart::Page`/page canvas + `imageops::overlay`,
`bookart::glyph` `ab_glyph` lettering, `persona` identity, `api::Generate` per-panel art.

---

## G0 — the one novel weight-free algorithm: balloon placement + lettering
The panel-layout engine and page composite are deterministic geometry (low risk, in-track tests). The
genuinely novel piece is **speech-balloon placement + lettering**, so de-risk it first:

- **G0.1 — balloon layout probe (`examples/comic_balloon_probe.rs`).** Given a panel size, a subject/face
  exclusion mask, N dialogue strings, and speaker points: (a) **word-wrap + fit** each string into a
  rounded balloon (binary-search font size / wrap so the text fills without overflow, `ab_glyph` metrics),
  (b) **place** balloons in open space — not over the exclusion mask, not overlapping siblings, biased to
  the reading corner, (c) a **tail** from the balloon toward its speaker. Measure on synthetic panels:
  all balloons placed, zero overlap, text fully inside, tails point the right way. **Exit:** a clean,
  measurable placement on a busy synthetic panel (≥4 balloons + a face mask). Pure/weight-free.

### G0.1 RESULT — PASS (commit `d81c928`, `examples/comic_balloon_probe.rs`)
Busy panel (face mask + 4 speakers): 4/4 placed, 0 overlaps, all off-mask, text fits, tails correct.
Key insight: **shrink-retry** — try progressively narrower balloons until one fits an open gap (narrower
wraps taller but squeezes into side gutters), what a letterer does when the reading corner is full.
Monospace metric approximation; P2 swaps in `ab_glyph`. Tail-routing-around-mask is a P2 refinement. →
**P2 uses this algorithm.**

---

## Phases

### P1 — spec + layout engine + page composite (weight-free; front-loaded) — **DONE (commit `5181b3f`)**
`src/comic/{spec,lint,layout,page,mod}.rs`: `ComicSpec` (permissive serde) → resolve → panel
`Rect`s (rows of relative-width cells + gutter + border) → composite **supplied** panel images into a
bordered page + `panels.json` (rect + reading index). CLI `comic new|lint|show|layout` (`--panels <dir>`),
`Command::Comic` wired. 6 unit tests + full-pipeline smoke (us-letter 300dpi = 2550×3300, `[[1,1],[1],[1,1,1]]`
resolves correctly); gate 1802 green. **Ships a working page pipeline with no GPU.** (No separate
`compile.rs` — resolve lives in `layout.rs`.)

### P2 — balloons + lettering (the G0 algorithm) — **DONE (commit `09e3d52`)**
`src/comic/balloon.rs` (from G0): word-wrap/fit (largest legible box) + open-space placement (off an
optional mask, non-overlapping, `at`-anchor / top-reading-corner biased) + tails; the four `kind`s
(speech = rounded+straight tail, thought = rounded+bubble trail, shout = spiky burst, caption =
tinted/tailless); reading-order-aware. Lettering rides the always-compiled 5×7 bitmap face
(`crate::map::labels`, all-caps; `--features shaped-labels` + font overrides for non-Latin).
`page::letter()` draws each panel's dialogue border-inset; CLI `comic letter` = layout + lettering.
2 unit tests + full-page visual smoke (7/7 lines placed, every kind legible + inside frame); gate 1804
green. **Face-aware masks + real speaker tails deferred to P3** (need the persona cast + face centroids).

### P3 — scene art + character consistency — **DONE (commit `0e40520`)**
`src/comic/render.rs`: per-panel `api::Generate` at the panel aspect; `chars` injected via the `persona`
identity layer (a `persona:` cast member → the deterministic persona compile, bare attributes; `describe:`
= seed-locked text) + a shared `style` suffix so the page reads as one hand. Negatives always exclude drawn
text (we letter separately). **Face-aware lettering closes the P2 deferral**: `detect_faces` (SCRFD,
best-effort) → cover-fit-mapped into panel-interior coords → balloon masks + tails toward the nearest face,
falling back to P2 defaults when no detector/faces. `balloon::place` now takes multiple masks. CLI `comic
render` (`--panels-out`/`--device`/`--no-letter`). map 5×7 font gained `? ! " ( )` for dialogue (previously
blank → place-name corpus byte-stable). 5 unit tests + **live Metal render** (3-panel strip: 3/3 panels,
3/3 balloons, persona held across panels 1–2, 2 faces → face-aware tails); gate 1808 green. RFC Q1 (persona
holds across varied actions) confirmed live. Per-panel pose hint: not needed for P3 (deferred as optional).

### P4 — integration + corpus + docs + cut
Parity (scenario `type: comic` / compile / Bund `plakat.comic.*` / `api::Comic` / doctor — bookart A1–A5
template); a demo page (`corpus/comic_*` + driver); `Documentation/COMIC.md` + tutorial + README; **CUT
6.8.0** (bump Cargo+lock, gate `--no-default-features --lib`, **pin turbofish on new `.parse()`**, FF
`git push 6.8.0:main`, tag → CI 6-asset, `cargo publish --locked --allow-dirty --no-default-features`, `gh
release edit` + bg waiter, **verify the Windows leg**, NO Claude/Anthropic coauthor).

## Sequencing
G0.1 (balloon algorithm) → **P1** (weight-free page pipeline, ships value alone) → **P2** (balloons) →
**P3** (scene art + persona cast) → **P4** (cut). Front-load the weight-free half — a full lettered page
can be composed from supplied panel images before any generation exists.

## Non-goals (RFC)
Clean grids only (no free-form / inset panels v1); you write the panels (no auto-story); lettering is
Latin/Cyrillic via a supplied font.
