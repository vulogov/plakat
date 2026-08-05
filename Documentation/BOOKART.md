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
