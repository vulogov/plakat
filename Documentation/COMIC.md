# `plakat comic` — a script → a lettered, multi-panel comic page

`comic` turns a small HJSON script (a `ComicSpec`) into a finished **multi-panel comic page**: a panel
grid, per-panel **scene art**, a **recurring cast** whose identity holds from panel to panel, and
**speech balloons + captions** placed and lettered over the art. It is the plakat 6.8 flagship (RFC
[`RFC_COMIC_1.md`](RFC_COMIC_1.md)), and it is **fully additive** — no existing command or output
changes.

It is a structural sibling of [`bookart`](BOOKART.md) (page layout + real-font lettering) and
[`persona`](PERSONA.md) (a *specific* synthetic person, reproducible from a spec). A comic needs both:
a page is a composition of framed panels, and a character must be the *same* person in panel 5 as in
panel 1. So the page is **authored**, not prompted — a text prompt is a poor instrument for "six panels,
this cast, these lines, in this order."

**The weight-free half needs no GPU.** Layout, page composite, balloon placement, and lettering are pure
CPU. Only the per-panel *scene art* (`comic render`) needs a model — and you can supply your own panel
images instead (`comic letter`), skipping the model entirely.

## The pipeline

```
ComicSpec ─▶ layout ─▶ [scene art per panel] ─▶ composite ─▶ letter ─▶ page.png + panels.json
  (author)   (grid)     (model, or your PNGs)    (cover-fit)  (balloons)
```

| Stage | Command | Weights? |
|---|---|---|
| resolve the grid + inspect | `comic show` | no |
| validate the script | `comic lint` | no |
| composite supplied panels | `comic layout --panels <dir>` | no |
| composite + letter supplied panels | `comic letter --panels <dir>` | no |
| the full flagship (generate → composite → letter) | `comic render` | **yes** |

## The `ComicSpec`

A permissive HJSON document — every field is optional, unknown keys are ignored (forward-compatible),
and enums are carried as strings (a typo is a lint warning, not a hard failure). `comic new` scaffolds a
working one.

```hjson
{
  schema: "comic/1"
  page:    { size: "us-letter", dpi: 300, gutter: 24, border: 6, bg: "white" }
  reading: "ltr"                       // ltr (western) | rtl (manga)
  layout:  { rows: [[1,1],[1],[1,1,1]] } // 2 half-width | 1 full | 3 third-width panels
  style:   "noir comic book art, heavy ink shadows, cel shading"  // one hand across the page

  model: "sdxl"  seed: 7  steps: 30

  cast: [
    { name: "mika", persona: "mika.hjson" }                 // a SPECIFIC recurring person
    { name: "bot",  describe: "a tall brass robot, glowing blue eye" }  // or seed-locked text
  ]

  panels: [
    { scene: "a rain-slick neon alley, wide shot", caption: "Tuesday. 3 a.m." }
    { scene: "mika crouches by a door", chars: ["mika"],
      balloons: [ { by: "mika", say: "Someone's been here.", at: "top-left" } ] }
    { scene: "a brass robot fills the frame", chars: ["bot"],
      balloons: [ { by: "bot", say: "SCANNING. TARGET ACQUIRED.", kind: "shout", at: "top" } ] }
  ]
}
```

- **`page.size`** — `us-letter` · `a4` · `a5` · `tabloid` · `square` · `custom` (with `w_in`/`h_in`).
  Pixel size = size × `dpi`.
- **`layout.rows`** — rows of *relative-width* cells. `[[1,1],[1]]` is two equal panels over one wide
  panel. Omit `layout` to auto-grid the panels into a near-square. `row_heights` sets per-row heights.
- **`reading`** — `rtl` reverses panel order within each row (manga).
- **`cast`** — each member is either a **`persona:`** path (a `PersonaSpec` → that *specific* person,
  compiled deterministically so the face/wardrobe recur) or a **`describe:`** string (seed-locked text).
- **`panels[].chars`** — cast names present in the panel; each injects its identity into the prompt.
- **`panels[].balloons[]`** — `by` (speaker, for the tail) · `say` (the line) · `kind`
  (`speech` | `thought` | `shout` | `caption`) · `at` (a placement hint: `top-left`, `top`, … or
  `auto`).
- **`panels[].caption`** — a narration box (tailless), drawn tinted.

## Balloons + lettering — the one novel piece

Placing speech balloons is the algorithm a letterer does by hand, and it is the part of `comic` that is
genuinely new (the rest is composition). For each line, in reading order, `comic`:

1. **fits** the text — word-wraps and picks the largest legible size whose text fills a bounded box,
   retrying progressively narrower boxes so a line can squeeze into a side gutter;
2. **places** the box in open space — off the subject (a detected face becomes an exclusion mask),
   never overlapping a sibling balloon, biased toward the `at` hint or the top reading corner;
3. **draws** it — the four kinds are visually distinct: **speech** (rounded, straight tail), **thought**
   (rounded, a trail of shrinking bubbles), **shout** (a spiky burst), **caption** (a tinted, tailless
   box) — with a **tail** pointing at the speaker.

Lettering uses the built-in all-caps bitmap face (the classic hand-lettered look), so `comic letter` is
**byte-stable and asset-free**. Build with `--features shaped-labels` and it will use a real
TrueType/OpenType font instead (for non-Latin scripts), exactly as map labels do.

When `comic render` generates the art, a configured face detector (SCRFD) makes the balloons
**face-aware**: tails point at the actual face, and balloons steer clear of it. Without a detector, it
falls back to the open-space defaults — still correct, just not face-locked.

## Multi-page comics (6.8.1)

A strip spans many pages that share a cast, a style, and an engine while the panels and dialogue change.
The shared **world** lives at the top of the spec; each **page** carries only its own layout + panels.

```hjson
{
  cast:   [ { name: "mira", persona: "mira.hjson" } ]   // the world — propagates to every page
  style:  "noir comic book art, heavy ink"
  model:  "sdxl"  seed: 7
  scenes: { alley: "a rain-slick neon alley", booth: "a payphone booth" }  // named, reusable

  pages: [
    { name: "opening", layout: { rows: [[1,1]] },
      panels: [ { scene: "@alley", chars: ["mira"], caption: "Midnight." },
                { scene: "@booth", balloons: [ { by: "mira", say: "Who's there?", kind: "shout" } ] } ] }
    { name: "turn", layout: { rows: [[1],[1,1]] },
      panels: [ { scene: "@alley", caption: "Nobody." }, { scene: "a shadow moves" }, { scene: "@alley" } ] }
  ]
}
```

`comic render page.png` writes **`page_00.png`, `page_01.png`, …** (a single-page spec still writes
`page.png`). Each page inherits the cast/style/model/page-format; only `panels` and `layout` differ.

- **Named scenes** — `scenes: { alley: "…" }` defines a setting once; a panel `scene: "@alley"` references
  it, so the same establishing shot recurs across pages without re-typing it.
- **`extends`** — split the world and the pages into separate files. A per-issue spec inherits a series
  file:
  ```hjson
  { extends: "series.hjson", pages: [ … ] }   // series.hjson holds cast/style/model/scenes
  ```
  Merge rules: the base provides defaults; the child overrides scalars, merges `cast`/`scenes` by name,
  and replaces `panels`/`pages` when it provides them.
- **Deterministic seeds** — panels seed off a *global* counter across all pages, so every panel is
  distinct and reproducible book-wide.

### Reference-lock — holding identity across pages

Cast identity is injected as text into every panel, which propagates *by construction* but is
**description-level**: over dozens of pages a model reinterprets the description and the face drifts.
`--lock` fixes that visually:

```bash
plakat comic cast issue.hjson --out refs/          # render each character once → a reference sheet
plakat comic render issue.hjson --out page.png --lock   # face-swap that reference onto every panel
```

- **Cast reference sheet** — `comic cast` renders one canonical portrait per character (`ref_<name>.png`
  + a combined `cast_sheet.png`) from its persona/`describe`. `render --lock` builds the same references,
  then **face-swaps** each character's reference onto its panel (SCRFD locate → ArcFace identity →
  inswapper), so the same face recurs book-wide. It acts on **single-character panels** (unambiguous which
  face is whom); multi-character panels stay description-level for now.
- **Explicit reference** — `cast[].reference: "mira.png"` locks to your own image instead of a rendered
  portrait. `cast[].lock: false` opts a character out.
- **Style-lock** — `style_lora: "…"` (+ `style_lora_scale`, default 0.8) applies one LoRA to *every* panel
  of *every* page, holding the look beyond the text style suffix.

Reference-lock is **best-effort**: it needs the face-swap weights (SCRFD + ArcFace + inswapper); without
them `--lock` prints a note and identity stays description-level. Small faces are swapped but may benefit
from a follow-up `restore-faces` pass.

## Character consistency — why a comic needs `persona`

A prompt can't keep a character the same across panels; that's the whole reason `persona` exists. In a
`ComicSpec`, a `persona:` cast member is compiled through the deterministic persona layer into a stable
attribute description that is injected into *every* panel the character appears in, so the same person
recurs. A `describe:` member does the lighter version — the same text, seed-locked. A shared `style`
suffix keeps the whole page in one hand.

## Commands

```
plakat comic new    <out.hjson>                       # scaffold a 6-panel template
plakat comic lint   <spec.hjson>                      # schema / vocab / cast cross-refs
plakat comic show   <spec.hjson>                      # resolved page + panel rects + cast
plakat comic layout <spec.hjson> --out page.png [--panels <dir>]   # composite supplied art
plakat comic letter <spec.hjson> --out page.png [--panels <dir>]   # composite + balloons
plakat comic render <spec.hjson> --out page.png [--panels-out <dir>] [--device auto] [--no-letter] [--lock]
plakat comic cast   <spec.hjson> --out refs/          # M2: build the cast reference sheet
```

`--panels <dir>` supplies your own panel images (sorted by name → panel order); missing panels draw a
placeholder. `comic render --panels-out <dir>` keeps the generated per-panel PNGs. Every page writes a
**`page.panels.json`** sidecar — each panel's page rectangle + reading index — so a re-lettering or
DCC pass can map back to panels.

## Integration

`comic` is wired into the rest of plakat like every flagship:

- **Library**: [`plakat::api::Comic`](API.md) — the stable builder (`Comic::new(spec).out(path).run()`).
- **Scenario**: a `type: "comic"` task in an HJSON scenario batch.
- **Bund**: `plakat.comic.*` scripting words (`spec` / `layout` / `letter` / `render`).
- **Compile**: `plakat compile` recognises a comic request in prose.
- **Doctor**: `plakat doctor` reports the comic capability.

## Limits (honest)

- Character consistency is **description-level** (persona attributes + seed), not a locked reference
  face; strong across panels, but not identity-swap exact. A per-panel reference-image lock is a future
  step.
- Lettering is **all-caps** by default (the bitmap face); use `--features shaped-labels` + a font for
  mixed case / non-Latin.
- Balloon placement avoids **faces**; it does not yet parse arbitrary busy regions of the art.
