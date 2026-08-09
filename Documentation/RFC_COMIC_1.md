# RFC COMIC-1: `plakat comic`
## Multi-panel comic pages from a script

**Status:** Proposed (6.8) · **Surface:** `plakat comic` (new subcommand) · **Feature:** in DEFAULT
features (the weight-free half — layout, balloons, page composite — needs no GPU).

---

## Overview

`plakat comic` composes a **multi-panel comic page** from a small HJSON `ComicSpec`: a page layout, a
named **cast**, and an ordered list of **panels** (each a scene + who's in it + dialogue). It resolves the
panel grid, renders each panel's scene with its characters drawn **consistently across panels**, letters
the dialogue into **speech balloons** with tails to the speaker, and composites everything onto a
print-size page in reading order.

It is the decorative/narrative sibling of two shipped flagships: **`bookart`** (page layout, DPI, print
sizing, transparent compositing) and **`persona`** (a *specific, reusable* synthetic person rendered
recognisably across scenes — exactly what a comic needs from panel to panel). COMIC-1 is the layer that
turns "a person in a scene" into "the same people telling a story across a page."

The design premise, stated up front: **panel geometry, balloons, lettering, and page composition are
deterministic and weight-free; only the per-panel *scene art* needs a model.** So the whole layout /
balloon / page-composite half runs with no GPU and is fully testable, and a supplied set of panel images
can be turned into a finished lettered page offline.

---

## The `ComicSpec`

```hjson
{
  schema: "comic/1"
  page:   { size: "us-letter", dpi: 300, gutter: 24, border: 6, bg: "white" }
  reading: "ltr"                       // ltr (western) | rtl (manga)
  layout: "rows: [ [1,1], [1], [1,1,1] ]"   // 3 rows: 2 panels, 1 wide, 3 panels (relative widths)
  cast: [
    { name: "mika",  persona: "mika.hjson" }      // a PersonaSpec → consistent identity
    { name: "robot", describe: "a dented brass service robot, single glowing eye" }
  ]
  panels: [
    { scene: "a rain-slick neon alley at night, wide establishing shot", chars: [], caption: "Tuesday. 3 a.m." }
    { scene: "mika crouched behind a dumpster, tense", chars: ["mika"],
      balloons: [ { by: "mika", say: "Did you hear that?", at: "top-left" } ] }
    { scene: "the robot rounds the corner, backlit", chars: ["robot"],
      balloons: [ { by: "robot", say: "SCANNING…", kind: "shout" } ] }
  ]
  model: "sdxl"  seed: 7  steps: 30
}
```

Permissive serde like `BookArtSpec` / `PersonaSpec`: every field optional, enums carried as strings
(lint catches typos, not a hard failure), unknown keys ignored.

---

## The pieces

### 1. Panel layout engine (weight-free)
`layout` resolves to a set of **panel rectangles** on the page canvas: rows of relative-width cells with
`gutter` between them and a `border` stroke, at the page's exact pixel size (`page.size`/`dpi`, reusing
`bookart::Page`). Supports the common grid; `rows: [...]` gives per-row cell counts/weights. Irregular
(overlapping / inset) panels are a later add — COMIC-1 ships clean grids. Output: an ordered
`Vec<PanelRect>` + a `panels.json` sidecar (rect + reading index per panel).

### 2. Per-panel scene art + character consistency
Each panel's `scene` prompt is generated (t2i) at the panel's aspect. Characters named in `chars` are
composited **consistently**: a cast member with a `persona` renders that *specific* person (reusing the
`persona` layer's identity machinery — the whole reason a comic needs persona, not a fresh face each
panel); a `describe`-only member uses a stable seed-locked description. The panel art is fit (cover-crop)
to its `PanelRect`.

### 3. Speech balloons + lettering (weight-free — the novel algorithm)
For each panel's `balloons`: **letter** the text with `ab_glyph` (the `bookart::glyph` path), **word-wrap**
to fit a rounded balloon, **place** the balloon in open space (away from faces/subject, non-overlapping
with sibling balloons), and draw a **tail** toward the speaker. `kind`: `speech` (default) / `thought`
(cloud + bubbles) / `shout` (jagged) / `caption` (a cornered box, no tail). Reading-order-aware placement
(top-to-bottom, `reading` L-to-R or R-to-L).

### 4. Page composite + reading order
Panels + borders + gutters + balloons composited onto the page canvas in reading order (Z for ltr, mirror
for rtl). Output: the print-size page PNG (+ optional per-panel PNGs) + `panels.json`.

---

## CLI surface

```
plakat comic new  <out.hjson>                 scaffold a spec
plakat comic lint <spec>                       validate (weight-free)
plakat comic show <spec>                       the resolved plan: page, panel rects, cast, reading order
plakat comic layout <spec> --out page.png      render the EMPTY grid + balloons from supplied panel imgs (no GPU)
plakat comic render <spec> --out page.png      the full page (generates panel art)
plakat comic letter <panel.png> --say "…" --at … --out   letter one panel (weight-free)
```

`--panels <dir>` supplies pre-rendered panel images (skip generation) → a fully weight-free page.

---

## Reuse map

- **`bookart`**: `Page` (size/DPI/margins), the page canvas + `imageops::overlay` composite, `glyph.rs`
  `ab_glyph` lettering, the `render_spec`/scenario/compile/Bund/api integration template.
- **`persona`**: the cast's identity — render a *specific* person recognisably across panels.
- **`t2i` / `api::Generate`**: per-panel scene art.
- **`texture::trim`** atlas/UV-region idea informs the `panels.json` sidecar shape.

---

## Implementation phases (each independently useful)

- **P0 — G0**: the balloon-placement + lettering algorithm (novel, weight-free) — word-wrap to fit,
  non-overlap placement avoiding a subject mask, tail geometry to a speaker point. De-risk first.
- **P1 — spec + layout + page composite (weight-free)**: `ComicSpec`, panel-layout engine, `comic
  new/lint/show/layout`, `panels.json`, page composite from **supplied** panel images. Ships a working
  lettered-page pipeline with no GPU.
- **P2 — balloons + lettering**: the G0 algorithm wired in (`comic letter` + balloons in `layout`), all
  four `kind`s, reading-order placement.
- **P3 — scene art + character consistency**: per-panel generation + persona-consistent cast → `comic
  render`.
- **P4 — integration + corpus + docs + cut**: scenario `type: comic` / compile / Bund / `api::Comic` /
  doctor; a demo page; `Documentation/COMIC.md`; the 6.8.0 release.

## Non-goals
Not a full sequential-art authoring tool (no free-form panel shapes / inset overlaps in v1 — clean grids
only); not automatic script→story (you write the panels); not hand-drawn line-art style enforcement
(that's the model/LoRA's job via the scene prompt); lettering is Latin/Cyrillic via a supplied font (no
automatic multi-script shaping beyond `ab_glyph`).

## Open questions
1. **Character consistency across *poses/actions*** — persona is proven for identity across *scenes*;
   comics demand varied *actions* per panel. How far does persona hold, and does COMIC-1 need a
   per-panel pose hint? (Measure in P3.)
2. **Balloon placement quality** — the open-space search + face-avoidance is heuristic; the G0 fixes the
   algorithm, but real panels will stress it (busy scenes, many speakers). Calibrate in P2.
3. **Font** — bundle a comic-lettering font, or require `--font`? (Licensing.) Default to the `bookart`
   font path; ship with a note.
