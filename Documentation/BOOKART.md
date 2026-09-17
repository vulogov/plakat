# `plakat bookart` — controllable black-and-white book-ornament composition

`bookart` composes a *reusable, print-ready, transparent* black-and-white book ornament — a chapter
headpiece, a tailpiece, a decorated initial, a border, a corner-piece, a vignette, a frontispiece —
from a small HJSON document (a `BookArtSpec`) and renders it at an exact page size, in a chosen
illustration tradition and drawing technique. It is the plakat 6.0 flagship (RFC
[`RFC_BOOKART_1.md`](RFC_BOOKART_1.md)), and it is **fully additive** — no existing command or output
changes.

`bookart` is the sibling of [`persona`](PERSONA.md): the same *spec → resolver → conditioned render →
composite → measure* spine, applied to decorative ornament instead of human identity. Where `persona`
anchors *identity*, `bookart` anchors a **motif** and a **drawing hand**. And like persona, it exists
because text prompts are a poor instrument for the job: "a symmetric black-and-white woodcut border"
comes out tinted grey, lopsided, opaque, arbitrarily sized, and scrawled with fake lettering — five
categorically different failures (colour, symmetry, transparency, print size, stray glyphs) that need
five different remedies, not a better prompt.

## The output contract

Every ornament is **generated → transparented → sized**. The primary, always-emitted artifact is a
**transparent, correctly-page-sized PNG** at the target DPI (with the DPI written into the file). SVG
is a **secondary, by-request** extra (`--svg`) — born-vector for the procedural tier only, off the
critical path. There is no "generate grey, then remove the background": B/W has a better transparency
model (ink darkness *is* opacity), which is the counter-intuitive core kept in its own document,
[`BOOKART_TRANSPARENCY.md`](BOOKART_TRANSPARENCY.md).

## The layer model

| Layer | What | Command(s) | Weights? |
|---|---|---|---|
| 0 — spec + lexicon | the HJSON schema + origin×technique×motif presets | `new` · `lint` · `show` | no |
| 1 — resolver | `(spec, lexicon) → RenderPlan`, pure & byte-stable | (inside `show`) | no |
| 2 — geometry | ornament layout · symmetry engine · page/text-block/DPI | (inside `render`) | no |
| 3 — render router | procedural \| diffusion \| composite | `render` · `illustrate` · `kit` · `manuscript` | tier-dependent |
| 4 — finisher | technique binarise → transparency → symmetry → (opt) vectorise | (inside all renders) | no |
| 6 — scorecard | measure a render against its spec | `verify` | detect only |
| — edit/lineage | class-aware diff · in-place post-edit · tradition blend | `diff` · `edit` · `blend` | no |
| — proof | contact sheet from a set | `proof` | no |

The determinism contract (RFC §5.2): everything except the diffusion step is a **pure, byte-stable
function** — testable in CI without weights or a GPU. Diffusion is the one stochastic step; it is
seed-locked and reproducible on a given device.

## Commands

```
plakat bookart new        <out.hjson> [--origin O --technique T --type K --page a5]   scaffold a spec
plakat bookart lint       <spec>                                                      validate (schema · vocab · ranges · page)
plakat bookart show       <spec>                                                      what it resolves to (tier · symmetry · canvas · prompt)
plakat bookart origins    [--details]                                                 list origins × techniques × ornaments + LoRA hosting
plakat bookart verify     <spec> --image IMG [--out O] [--finished] [--symmetrize] [--page]   the scorecard
plakat bookart render     <spec> --out O [--model sd15 --seed 0 --steps 28] [--svg] [--attempts N] [--font F] [--cache-raw] [--import ALBUM]
plakat bookart illustrate "<prompt>" --out O [--origin O --technique T --page a5 --type frontispiece …] [--font F] [--cache-raw] [--import ALBUM]
plakat bookart kit        <spec> --out DIR [--model --steps --svg --no-coherence]     a coherent matched set (flagship)
plakat bookart manuscript <book.md|list|book.epub> --kit <spec> --out DIR [--latex --svg]   a per-chapter set for a whole book
plakat bookart proof      <dir> --out sheet.png                                       a contact sheet
plakat bookart diff       <old> <new>                                                 classify an edit (post · re-raster · re-gen)
plakat bookart edit       <png> --out O [--tint T] [--symmetry S] [--ink-weight W] [--transparency M] [--fade F]   cheap post-edit, no GPU
plakat bookart blend      <a> <b> --out O                                             lineage: origin(A) × technique(B)
plakat bookart vectorize  <raster> --out svg [--tint T --dpi N]                        raster→SVG trace   (feature: bookart-trace)
plakat bookart font       --out dingbats.otf [--family NAME]                           export ornaments as an OpenType dingbat font
plakat bookart typst      --border|--corner ORN --out page.typ [--page a5 --margin 12 --corner-size 18 --rule 0.6 --title T --body F --image IMG --spec S --verify]   a bordered Typst book page (reusable template)
plakat bookart title-page <spec.hjson> --out title.typ [--page a5 --style letterpress|engraved|modern|playbill --margin 22 --fit --historical --print|--bleed 3 --crop-marks --verify]   a title page → Typst
plakat bookart cover      <spec.hjson> --out cover.typ [--page a5 --style S --pages N --paper 0.06 --flap 0 --historical --print|--bleed 3 --crop-marks --verify]   a cover / dust jacket (back·spine·front) → Typst
plakat bookart book       <manuscript.md> --out book.typ [--page a5 --title-page T.typ --running-head "…" --headpiece H --tailpiece T --divider D --colophon "…" --verify]   a whole typeset book → Typst
plakat bookart endpaper   --motif ORN.png --out endpaper.png [--page a5 --layout grid|half-drop|diamond --tile 22 --gap 10 --bg '#f4efe6']   a seamless patterned endpaper (PNG)
```

> **Opt-in features.** A few of the above need a Cargo feature the prebuilt release binaries don't
> include: **`bookart-trace`** (`vectorize` + `--svg` tracing on the diffusion/composite tiers) and
> **`epub`** (`manuscript book.epub`). Build them with `cargo install plakat --features bookart-trace,epub`.
> Glyph-driven initials use **`shaped-labels`**, which *is* on by default (via `photos`). `bookart font`
> and all six origin LoRAs work in the release binaries as-is.

### `new` — scaffold a spec

Writes a valid partial `BookArtSpec` HJSON you then edit, and lints it. Flags: `--origin` (default
`generic`), `--technique` (`line`), `--type` (`headpiece`), `--page` (`a5`). Refuses to overwrite an
existing file.

### `lint` — validate without weights

Checks schema version, vocabulary (with nearest-match suggestions — `woodcutt` → `woodcut`), numeric
ranges, page validity, and the `ornament`-xor-`kit` contradiction. Exits non-zero on any error, so it
can gate CI. No network, no weights.

### `show` — the resolved plan

Prints what a spec resolves to: origin/technique, motif, ornament type, the chosen **render tier**,
the **symmetry** group, the print **canvas** (px @ DPI, plus mm and bleed), the **finisher** chain
(transparency mode, binariser, ink colour/weight, tint), the output formats, and the compiled
diffusion prompt + negative (or `(procedural tier — no prompt)`).

### `render` — one ornament, end to end

Resolves the spec, lays it out against the text block, dispatches the tier, finishes to transparency,
applies symmetry, places it on the exact page canvas, and writes a DPI-tagged PNG **plus a `.json`
recipe sidecar** (origin / technique / tier / a stable spec-hash) that is also embedded as an Auto1111
`parameters` PNG `tEXt` chunk — so an ornament is searchable and re-runnable. `--attempts N` turns on
rejection sampling for the diffusion tier: it tries up to N seeds and keeps the first that clears the
scorecard (else the fewest-issues one). `--model` selects the diffusion base (`sd15`, which the origin
LoRAs target); `--seed`/`--steps` tune the diffusion step.

- **`--svg`** emits a born-vector SVG on the **procedural** tier. On the diffusion/composite (pixel)
  tiers it emits a **traced** SVG when built with the `bookart-trace` feature, else a one-line note (the
  PNG is the deliverable). See `vectorize` below.
- **`--font <ttf/otf>`** supplies a font for a glyph-driven `initial` — the ornament is built around the
  real letterform in `ornament.glyph` (see the vocabulary table). Needs `shaped-labels` (default-on).
- **`--cache-raw`** also writes `<out>.raw.png` (the pre-finish gray) + `<out>.plan.json`, so
  `bookart edit --ink-weight/--transparency/--fade` can re-finish without re-rendering.
- **`--import <album>`** lands the ornament + its recipe sidecar in a `plakat photos` album
  (auto-tagged from the recipe). Needs the `photos` feature (default-on).

### `origins` — the vocabulary + LoRA hosting

```sh
plakat bookart origins --details
```

Lists every **origin** (with its LoRA-hosting status — `[hosted LoRA]`, `[scaffold only]`, or the
LoRA-free `generic` path — and `(custom)` for lexicon additions), every **technique** (→ its binariser
+ prompt cue), and every **ornament** type (→ default tier + symmetry), plus the status of the optional
`assets/bookart/lexicon.hjson` override. `--details` also prints each origin's prompt scaffold + default
technique + motifs. Six origins ship trained sd15 LoRAs — **russian / english / japanese** (Bilibin /
Beardsley / Hokusai) and **american / european / chinese** (Pyle / Doré / woodblock outline) — hosted
at `vulogov98/plakat-bookart` and auto-resolved; see [`BOOKART_STYLES.md`](BOOKART_STYLES.md).

### `illustrate` — a standalone B/W plate from a prompt

The diffusion tier exposed directly, for when you don't want to author a spec:

```sh
plakat bookart illustrate "a wolf in a snowy pine forest" --origin japanese --out wolf.png
```

Synthesises a diffusion-tier spec (`--type frontispiece` page-fill by default, or `vignette` for a
centred spot), styled to `--origin`×`--technique`, finished + page-placed like any render. Same
`--model`/`--seed`/`--steps`/`--attempts`, and the same `--cache-raw` / `--import` as `render`.

### `verify` — the print/ink scorecard

Finishes a raw render per the spec (binarise → transparency) and scores it: **chroma** purity (is it
truly B/W?), **alpha-halo** (partial-alpha ring — a clean key has none), **symmetry RMS** (fold about
the declared axis/order), **ink coverage**, and **resolution** (does it match `size×dpi`?). `--out`
writes the finished transparent PNG; `--finished` scores an already-finished PNG as-is; `--symmetrize`
applies the plan's symmetry (§6.3 — the one thing the finisher provably can't fix); `--page` places
the result on the exact page canvas so `--out` is page-sized with the DPI recorded.

### `kit` — a coherent matched set (flagship)

Renders every ornament in the spec's `kit.ornaments` block sharing **one origin+technique (one hand)**,
**one motif DNA**, and **one seed lineage** (deterministic per-ornament seeds). Emits `NN_<type>.png`
(+ SVG when `--svg`), a **contact sheet**, a **`manifest.json`**, and a **CLIP style-coherence** score
(min/mean pairwise cosine across the set; `--no-coherence` skips loading CLIP). Coherence is
informational, not an auto-gate — a kit legitimately spans geometric and pictorial types. See §10.

### `manuscript` — a per-chapter set for a whole book (flagship)

```sh
plakat bookart manuscript book.md --kit style.hjson --out ornaments/ --latex
```

Parses a book's chapter structure — Markdown `#`/`##` headings, one title per line, **or an `.epub`**
(the spine/TOC is read via the NCX → nav → `<title>` fallbacks; needs the `epub` feature) — and emits a
**frontispiece** (the pictorial plate, diffusion) plus, per chapter, a **procedural headpiece band**
(rules + central medallion + interweaving guilloché braid + fleuron ends) and a **procedural tailpiece**
(a cul-de-lampe tapering to a point). The per-chapter seed *diversifies* the bands — a denser braid, a
different scroll count — so they read as kin, not clones, while staying in one hand. The
`--kit` spec supplies the style (origin/technique/motif/page) and its `kit.seed` seeds the lineage.
Writes per-file PNGs, a chapter→assets `manifest.json`, a contact sheet, and (with `--latex`) an
`includes.tex` of `\newcommand`s. See §11.

### `proof` — a contact sheet

Tiles every ornament PNG in a directory (cropped to its ink, on white) into one sheet — the kit/
manuscript modes emit one automatically; this runs it over any directory.

### `diff` · `edit` · `blend` — edit & lineage

- **`diff <old> <new>`** classifies each changed field by the *cheapest* action it forces: `post` (a
  tint or symmetry change — recolour/re-tile a finished PNG, no GPU), `re-raster` (a page/size change —
  re-place the same raster), or `re-gen` (origin/motif/prompt/technique — a full re-render). It reports
  the overall cheapest sufficient action.
- **`edit <png> --out …`** applies `post`-class repairs with **no re-render**. Two paths:
  - *On a finished PNG* — recolour the ink (`--tint black|sepia|#rrggbb`) and/or re-apply symmetry
    (`--symmetry bilateral|radial:N`), operating on the pixels directly.
  - *Re-finishing from a cache* — `--ink-weight W` / `--transparency luminance|threshold|fade` /
    `--fade F` re-run the finisher (binarise → transparency → symmetry → page) on the gray cached by
    `render --cache-raw`, so ink weight and transparency become cheap edits instead of a full re-gen. It
    bails with a clear note if the `<png>.raw.png` / `.plan.json` cache is absent.
- **`blend <a> <b> --out`** is lineage: it writes a new spec crossing the **origin of A** with the
  **technique of B**, unioning both motifs, and lints it (e.g. Russian firebird motif drawn with a
  Japanese line hand).

### `vectorize` — raster→SVG trace *(feature: `bookart-trace`)*

```sh
plakat bookart vectorize scan.png --out scan.svg --tint black --dpi 300
```

Traces any raster ornament (the diffusion/composite tiers, or a scan) into a compact SVG: the
transparent art is flattened onto white, traced to filled B/W paths, retinted to the ink colour, and
stamped with the physical (mm) print size from `--dpi`. The **procedural** tier is already born-vector
(`render --svg`), so this is for the pixel tiers. Behind the `bookart-trace` feature (it pulls an extra
tracing stack); without it the command explains how to enable it.

### `font` — an OpenType dingbat font

```sh
plakat bookart font --out dingbats.otf --family PlakatDingbats
```

Exports a small set of procedural ornaments (`a`–`h` → fleurons / dinkus / rosettes / divider / corner)
as a real **OpenType dingbat font** for inline use in InDesign / LaTeX — type a letter, get an ornament.
Self-contained (a from-scratch TrueType writer, no font-toolkit dependency); the file loads + renders in
any font-aware application.

### `typst` — bordered book pages as Typst templates

```sh
plakat bookart typst --corner corner.png --page a5 --margin 16 --corner-size 15 \
  --title "Chapter One" --body chapter.txt --out page.typ --verify
```

Wraps an ornament into a self-contained, PDF-compilable **Typst** artifact where the *text* is the
subject and the frame never dominates the page. Two styles:

- **`--border <ornament>`** — a full border, sized to the margin box (`page − per-side margins`). The
  text box is **fitted to the border's *measured* clear window** (the largest ink-free rectangle inside
  the ornament, corner motifs included), so text can never overlap the art — and it **refuses to emit**
  if nothing usable fits ("do not generate what doesn't fit").
- **`--corner <ornament>`** — a restrained **thin-rule frame + the corner ornament mirrored to match at
  all four corners** (a `bookart corner` render is cropped to one square tile, flipped in place). Text
  keeps the whole interior.

The output is a **reusable template**: it defines `book-page`, whose `set page(background: …)` repeats the
frame on *every* page, so you apply it to a whole book with `#show: book-page` and Typst paginates your
text across as many pages as it needs — no per-page setup. Compiling the file directly renders a preview;
`#import "page.typ": book-page` takes only the helpers. Also emitted: **`text-box(body, width:, inset:,
fill:, alignment:…)`** (put a run of text in a sizeable box) and **`place-on-page(dx, dy, …)`** (absolute
overlay). Per-side `--margin[-top/-bottom/-left/-right]`, `--safety`, `--corner-size`, `--rule`; `--spec`
reads the page size from a spec; **`--verify`** compiles the artifact to a PDF via the `typst` CLI to prove
it renders. Referenced assets are copied beside the `.typ` so it compiles anywhere.

### `title-page` — an old-style title page from HJSON

```sh
plakat bookart title-page title.hjson --out title.typ --verify
```

Generates a **book / chapter title page** in the historical **letterpress** hierarchy — tracked small-caps
series lines, a big bold display title, subtitle, part, an author line, and an imprint at the foot, with an
optional ornamental **border** or emblem — as a compilable **Typst** artifact usable in a Typst book.
Weight-free (the typography *is* the artefact — no diffusion). Generic over a `style:` (default
`letterpress`; more styles can be added). The output is reusable: it defines `#let title-page = { … }` and
previews it, so it compiles standalone (`--verify`) and `#import`s into a book (`#import "title.typ":
title-page` then `#title-page`).

The spec is a small HJSON — a `style`, an optional `border` image, and a vertical stack of `lines`, each a
`role` + its `text` (or `src` for an image). Roles: `series` · `title` · `subtitle` · `part` · `subchapter`
(a subordinate section mark — small tracked small-caps, for section pages under a chapter) · `author` ·
`note` · `epigraph` · `imprint` (placed at the foot) · `rule` · `ornament`/`image` (an emblem) · `space`.
`\n` in a line's text splits it into stacked centred lines; `size` overrides a role's point size (for an
`image`/`ornament` line it is the width — a page-% for `image`, mm for `ornament` — and `width:` is an
accepted alias). With a
`border`, the type box is auto-fitted to the ornament's **measured clear window** (shared with `typst`), so
text never overlaps the frame. An `image`/`ornament` line's `src` is **auto-cropped to its ink** — a
`bookart render` ornament arrives on a full page canvas, so it's trimmed to the device before it's centred
inline. So a `border` (a rendered `border` ornament) and a chapter `ornament` (a rendered `fleuron`
rosette) drop straight in — see [`corpus/bookart_titlepage.sh`](../corpus/bookart_titlepage.sh).

**Styles.** The `style:` (or `--style`, which overrides it) picks the hand — same roles, different
typography, all weight-free (no display fonts): **`letterpress`** (default — dense antique book type, bold
upper display, small-caps series), **`engraved`** (copperplate/atlas — regular-weight wide-tracked caps,
italic subtitle & author, airy), **`modern`** (minimal — regular weight, as-authored case, tiny
wide-tracked labels), **`playbill`** (Victorian poster — everything heavy bold upper, big size jumps, thick
rules). Author display lines in title-case and each style cases them as it likes. See the sampler:
[`corpus/bookart_titlepage_styles.sh`](../corpus/bookart_titlepage_styles.sh).

**Fit & typography.** A dense hierarchy can be taller than its page — which Typst silently spills onto a
second sheet. `--fit` measures the layout with `typst` and shrinks the type (and any plates) just enough to
fit **one** page, reporting the scale it used; without it, `--verify` still **warns** when a page overflows.
`--historical` (or `historical: true` in the spec) turns on old-style figures and historical ligatures for
an antique feel — weight-free, and harmless where the font lacks them.

```hjson
{ style: "letterpress", page: "a5", border: "frame.png", lines: [
  { role: "series", text: "УЧЕБНЫЯ РУКОВОДСТВА\nдля ВОЕННО-УЧЕБНЫХЪ ЗАВЕДЕНІЙ" }
  { role: "rule" }
  { role: "title",  text: "ИСТОРИЧЕСКОЙ ГРАММАТИКИ" }
  { role: "part",   text: "ЧАСТЬ I. ЭТИМОЛОГІЯ" }
  { role: "author", text: "Ѳ. Буслаевымъ." }
  { role: "imprint", text: "МОСКВА.\nВъ университетской типографіи.\n1858." } ] }
```

**Press-ready output.** `title-page` and `cover` take `--bleed <mm>` (grow the sheet past the trim so ink
runs off the cut edge) and `--crop-marks` (hairline trim marks in the bleed margin — plus **spine fold
ticks** on a cover); `--print` is the shorthand for **3 mm bleed + crop marks**. The page grows by the
bleed on every side, the content and any full-bleed background shift into the trim, and the marks sit in the
new margin — a sheet you can hand to a press.

### `cover` — a book cover / dust jacket from HJSON

```
plakat bookart cover cover.hjson --out cover.typ --verify
```

Lays the three panels of a wrap — **back · spine · front** — flat on one wide sheet, with the **spine width
computed from the page count** (`pages × paper` mm/page `+ board`, or an explicit `spine_mm`), optional
`flap`s, and dashed fold guides so the `.typ` is a usable printer's layout. Each panel is authored with the
same roles as a title page (`title`/`subtitle`/`author`/`imprint`/`ornament`/`image`/…) and shares the
`style` hand, so the cover matches the book. The spine's type is set small and rotated to read top-to-bottom;
`border` places a full-bleed image behind the front type. Page/style/pages/paper/flap are overridable on the
CLI (`--page`/`--style`/`--pages`/`--paper`/`--flap`).

```hjson
{ style: "letterpress", page: "a5", pages: 320,
  front: [ { role: "title", text: "The Open Sea" }, { role: "author", text: "By J. Hawkins" },
           { role: "ornament", src: "emblem.png", size: 34 } ],
  spine: [ { role: "title", text: "The Open Sea", size: 13 }, { role: "author", text: "Hawkins", size: 10 } ],
  back:  [ { role: "note", text: "An account of divers discoveries…" },
           { role: "imprint", text: "London · The Admiralty Press" } ] }
```

### `book` — assemble a whole typeset book (the capstone)

```
plakat bookart book manuscript.md --out book.typ --title-page 01-title.typ --headpiece rosette.png --verify
```

The capstone that makes every other piece add up. From a **Markdown** manuscript it assembles one
compilable Typst book. The manuscript is real Markdown: `# Title` opens a **chapter** (text before the
first is front matter), `## Head` is an in-chapter **section head**, `> …` lines are a **blockquote**, a
`***` line is a **scene break** (a typographic asterism, or a `--divider` ornament), blank lines separate
paragraphs, and `*italic*` / `**bold**` (also `_`/`__`) are honoured inline. From it comes: an `#include`d
**title page** (a `bookart title-page` artifact),
**chapter openers** (an optional headpiece ornament · `CHAPTER N` · the title), body prose with a **raised
decorated initial + small-caps opening** on each chapter's first paragraph, **running heads + page folios**,
an optional **tailpiece** per chapter, and a **colophon**. Weight-free typography; the only images are the
title page and the head/tailpiece ornaments you pass. Because the title page is `#include`d, it must sit in
the book's directory with its own assets (the tool references it by basename when it already does).

The natural pipeline: render a `kit` → build a `title-page` (and `cover`) → `bookart book` lays the
manuscript into them. See the full demo [`corpus/bookart_titlepage.sh`](../corpus/bookart_titlepage.sh),
which ends by assembling `thebook.pdf`.

### `endpaper` — a seamless patterned diaper from a motif

```
plakat bookart endpaper --motif fleuron.png --out endpaper.png --layout diamond --bg "#f4efe6"
```

Tiles one motif (a transparent B/W ornament — auto-cropped to its ink) into a **seamless repeating pattern**
across a print-sized page → a PNG endpaper. Three lattices — a straight **`grid`**, a **`half-drop`**, and a
diagonal **`diamond`** diaper — with `--tile`/`--gap` (mm) setting the repeat and motifs **clipped at the
trim** so the pattern reads as continuing past the page. Transparent by default (overprintable); `--bg
#rrggbb` lays it on a tint (e.g. a warm laid paper). Weight-free — pure compositing, no model.

## The ornament vocabulary

The `ornament.type` key (the RFC's named vocabulary, §4). Each type carries a default tier and default
symmetry the resolver applies unless you override them:

| Type | What it is | Default tier | Default symmetry |
|---|---|---|---|
| **headpiece** | a chapter-opening band (bandeau · *застАвка*) | composite | bilateral |
| **tailpiece** | a chapter-closing tapering piece (cul-de-lampe · *концовка*) | composite | bilateral |
| **initial** | a decorated drop-cap built around a legible letter² | composite | none |
| **border** | a frame assembled from a tileable edge unit + corner unit | procedural | bilateral |
| **corner** | an L-shaped piece placed at 4 corners by reflection | procedural | none¹ |
| **divider** | a thin centred rule between sections | procedural | bilateral |
| **fleuron** | a printer's flower — a small centred mark | procedural | bilateral |
| **dinkus** | a section break (asterism ⁂) | procedural | none |
| **vignette** | a pictorial spot illustration | diffusion | none |
| **frontispiece** | a full-page pictorial plate | diffusion | none |
| **colophon** | a printer's device / closing mark | diffusion | radial:8 |
| **endpaper** | a seamless repeating pattern | procedural | radial:8 |
| **marginalia** | a small pictorial margin mark | diffusion | none |

¹ `corner` is *placed* four times by the layout engine (inward-flipped at each corner), so it needs no
per-piece symmetry.

² `initial` with `ornament.glyph: "<letter>"` + `render --font <ttf/otf>` rasterises the **real
letterform** (any script, incl. Cyrillic) via `ab_glyph` and frames it — a legible historiated initial,
no diffusion faking letters. Without a font/glyph it renders as a decorative composite cell.

## The `BookArtSpec` schema

Permissive serde, exactly like `PersonaSpec`: **every field is optional** (a bare `{}` resolves — it
defaults to a `divider`), enums are carried as strings (unknown values load and are caught by `lint`
with a nearest-match suggestion, not a hard failure), and unknown *keys* are ignored (forward-
compatible). A full single-ornament spec:

```hjson
{
  schema: "bookart/1"
  origin: "russian"            # tradition preset — LoRA + prompt scaffold + default motifs
  technique: "woodcut"         # drawing method → LoRA + finisher binariser
  motif: ["firebird", "oak-leaf"]

  ink: {
    color: "black"             # black (default) | sepia | #rrggbb — recolours ink, alpha unchanged
    weight: 0.7                # [0,1] stroke/coverage — biases the binariser
    transparency: "luminance"  # luminance (default) | threshold | matte | fade
  }

  page: {
    size: "a5"                 # a4|a5|a6|b5|letter|legal|trade|mass-market | custom
    dpi: 300
    orientation: "portrait"    # portrait (default) | landscape
    bleed_mm: 3
    margins: { top: 18, bottom: 20, inner: 18, outer: 15 }   # mm; derive the text block
    gutter_mm: 0
    custom: { w_mm: 0, h_mm: 0 }   # only when size == "custom"
  }

  transparent: true
  output: { formats: ["png"], tint: "black" }   # png always; add "svg" to also emit vector (opt-in)

  ornament: {
    type: "headpiece"          # the vocabulary above
    symmetry: "bilateral"      # bilateral | radial:N | frieze:GROUP | none
    tier: "auto"               # auto (router) | procedural | diffusion | composite
    prompt: "a firebird among oak branches"   # pictorial inlay (diffusion / composite)
    frame: "filigree"          # procedural scaffold family for a composite frame
    taper: 0.0                 # tailpiece taper: 0 = a band, 1 = a point
    glyph: "В"                 # a single decorated initial's letter (any script)
    glyphs: "cyrillic-upper"   # a glyph-set name for an initial *series*
    lines: 3                   # initial cell height in text lines
    places: 4                  # corner replication count
    fade: 0.0                  # vignette/spot edge fade, [0,1]
    motif: ["firebird"]        # per-ornament motif override (else the top-level motif)
  }
}
```

A spec carries **either** `ornament` (a single piece) **or** `kit` (a matched set), never both — `lint`
enforces the xor. The `kit` block:

```hjson
kit: {
  seed: 42                              # one lineage — each ornament derives its own seed
  ornaments: [
    { type: "headpiece", tier: "composite" }
    { type: "tailpiece", taper: 0.6 }
    { type: "divider" }
    { type: "corner", places: 4 }
    { type: "frontispiece", prompt: "the firebird in a winter forest" }
  ]
}
```

## The three render tiers

The router picks a tier from the ornament type (geometric → procedural, pictorial → diffusion, framed-
pictorial → composite); an explicit `ornament.tier` in the spec overrides the router.

- **`procedural`** — vector-native, **zero-weight** geometric ornament from self-contained parametric
  generators (rosette, guilloché, bead-and-reel border, L-corner scrollwork). Deterministic, crisp,
  instantly symmetric, and the **only** tier that emits born-vector SVG. Default for border, corner,
  divider, fleuron, dinkus, endpaper.
- **`diffusion`** — pictorial ornament via **sd15** + the origin LoRA (or the generic line-art path,
  no LoRA), finished through the technique binariser (XDoG etc.) to clean line art, then transparented.
  Default for vignette, frontispiece, marginalia, colophon.
- **`composite`** — the elegant hybrid: a **procedural frame** (born-vector, symmetric, crisp) with a
  **diffusion picture** inlaid into its window and finished to transparency. The direct analog of
  persona's geometry-map + detail-composite. Default for headpiece, tailpiece, initial. (Symmetry is
  skipped — the frame is already symmetric and the picture is a scene, not a mirror-double.)

## The print/ink scorecard

`bookart verify` measures a render against the spec that produced it, so quality is falsifiable and
repairable. The reported probes (all pure — no weights):

| Probe | Checks |
|---|---|
| **chroma** | max saturation < ε — truly black-and-white, no residual tint |
| **alpha-halo** | fraction of partial-alpha pixels — a clean key leaves no halo ring |
| **symmetry RMS** | fold about the declared axis/order — the finisher's one blind spot (fix with `--symmetrize`) |
| **ink coverage** | black coverage vs the spec/kit |
| **resolution** | px == `size × dpi`, with the pHYs DPI correct |

The scorecard drives `render --attempts N` rejection sampling. (The RFC's stray-glyph and aesthetic
probes are wired at the render layer as a fast-follow; verify today reports the five above.)

## Integration surfaces

`bookart` is not only a subcommand — the same render core drives every automation surface (6.1):

- **scenario** — a `type: bookart` task (inline `spec:` or `spec_file:` + `model`/`seed`/`steps`/`svg`)
  renders ornaments inside a batch scenario.
- **compile** — a `type: bookart` block (`bookart-origin` / `-technique` / `-type` / `-page` / `-svg`
  directives; the prose is the ornament prompt) compiles a prose prompts file to a bookart scenario.
- **Bund** — `plakat.bookart.render` / `.illustrate` / `.origin` / `.technique` push a **transparent,
  page-sized image handle** into the existing `plakat.save` / `.metadata.write` / `.upscale` pipeline.
- **library API** — `plakat::api::BookArt` (`load` / `from_spec` · `model`/`seed`/`steps`/`svg`/`attempts`
  · `run` → an in-memory `Rendered`), mirroring `Generate` / `Portrait`.
- **photos** — `render|illustrate --import <album>` lands an ornament in a `plakat photos` album,
  curated with its recipe.
- **Typst** — `bookart typst` wraps an ornament into a `book-page` Typst *template* (the frame repeats on
  every page) plus `text-box` / `place-on-page` helpers, so a bordered book paginates itself; `bookart
  title-page` generates an old-style letterpress title page from HJSON. Both emit compilable Typst usable in
  a Typst book, fit text to the ornament's clear window, and `--verify` to PDF.

## Honest scope

- **B/W only.** This is an ink idiom. `ink.color`/`output.tint` recolour the alpha channel (§7.4); they
  do not make colour art. No CMYK, no ICC — output is K-only / 1-bit / greyscale.
- **Procedural SVG is born-vector; pixel-tier SVG is a trace.** `render --svg` emits born-vector SVG for
  the procedural tier (always). Diffusion/composite `--svg` and `bookart vectorize` *trace* the raster —
  behind the `bookart-trace` feature. The PNG is always the deliverable.
- **Origins are tradition-level, from public-domain corpora.** Six origins ship trained sd15 LoRAs
  (russian/english/japanese + american/european/chinese); other origins run the generic line-art path.
  Not any living illustrator's exact hand. See [`BOOKART_STYLES.md`](BOOKART_STYLES.md).
- **`matte` transparency is a convenience.** U2Net is trained on natural photos, so its silhouette
  quality on B/W ornament is unverified — the luminance model (§7.2) is the primary path.
- **Glyph initials render one letter.** `ornament.glyph` + `--font` builds a historiated initial around
  a real letterform (any script, incl. Cyrillic); an initial *series* / auto-drop-cap across a whole
  book is still a fast-follow.

## Companion documents

- [`RFC_BOOKART_1.md`](RFC_BOOKART_1.md) — the full design.
- [`Tutorials/BOOKART_TUTORIAL.md`](Tutorials/BOOKART_TUTORIAL.md) — a hands-on walkthrough.
- [`BOOKART_TRANSPARENCY.md`](BOOKART_TRANSPARENCY.md) — the luminance-alpha model, tint, binarisers,
  born-vector SVG, exact-print sizing, and the symmetry engine (the counter-intuitive core).
- [`BOOKART_STYLES.md`](BOOKART_STYLES.md) — the origin × technique system and the origin LoRAs.
