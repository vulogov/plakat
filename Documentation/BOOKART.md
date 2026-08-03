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
plakat bookart verify     <spec> --image IMG [--out O] [--finished] [--symmetrize] [--page]   the scorecard
plakat bookart render     <spec> --out O [--model sd15 --seed 0 --steps 28] [--svg] [--attempts N]
plakat bookart illustrate "<prompt>" --out O [--origin O --technique T --page a5 --type frontispiece …]
plakat bookart kit        <spec> --out DIR [--model --steps --svg --no-coherence]     a coherent matched set (flagship)
plakat bookart manuscript <book.md|list> --kit <spec> --out DIR [--latex --svg]       a per-chapter set for a whole book
plakat bookart proof      <dir> --out sheet.png                                       a contact sheet
plakat bookart diff       <old> <new>                                                 classify an edit (post · re-raster · re-gen)
plakat bookart edit       <png> --out O [--tint T] [--symmetry S]                      cheap post-edit, no GPU
plakat bookart blend      <a> <b> --out O                                             lineage: origin(A) × technique(B)
```

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
applies symmetry, places it on the exact page canvas, and writes a DPI-tagged PNG. `--svg` also emits
a born-vector SVG (**procedural tier only**; other tiers print a note — the raster trace is a
fast-follow). `--attempts N` turns on rejection sampling for the diffusion tier: it tries up to N
seeds and keeps the first that clears the scorecard (else the fewest-issues one). `--model` selects
the diffusion base (`sd15`, which the origin LoRAs target); `--seed`/`--steps` tune the diffusion
step.

### `illustrate` — a standalone B/W plate from a prompt

The diffusion tier exposed directly, for when you don't want to author a spec:

```sh
plakat bookart illustrate "a wolf in a snowy pine forest" --origin japanese --out wolf.png
```

Synthesises a diffusion-tier spec (`--type frontispiece` page-fill by default, or `vignette` for a
centred spot), styled to `--origin`×`--technique`, finished + page-placed like any render. Same
`--model`/`--seed`/`--steps`/`--attempts`.

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

Parses a book's chapter structure (Markdown `#`/`##` headings, else one title per line) and emits a
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
- **`edit <png> --out --tint --symmetry`** applies the `post`-class repairs directly to a *finished*
  PNG — recolour the ink (`--tint black|sepia|#rrggbb`) and/or re-apply symmetry
  (`--symmetry bilateral|radial:N`) — with **no re-render**. Anything else needs `render`.
- **`blend <a> <b> --out`** is lineage: it writes a new spec crossing the **origin of A** with the
  **technique of B**, unioning both motifs, and lints it (e.g. Russian firebird motif drawn with a
  Japanese line hand).

## The ornament vocabulary

The `ornament.type` key (the RFC's named vocabulary, §4). Each type carries a default tier and default
symmetry the resolver applies unless you override them:

| Type | What it is | Default tier | Default symmetry |
|---|---|---|---|
| **headpiece** | a chapter-opening band (bandeau · *застАвка*) | composite | bilateral |
| **tailpiece** | a chapter-closing tapering piece (cul-de-lampe · *концовка*) | composite | bilateral |
| **initial** | a decorated drop-cap built around a legible letter (§ glyph path) | composite | none |
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
pictorial → composite); `ornament.tier` or `--tier`-style overrides win.

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

## Honest scope

- **B/W only.** This is an ink idiom. `ink.color`/`output.tint` recolour the alpha channel (§7.4); they
  do not make colour art. No CMYK, no ICC — output is K-only / 1-bit / greyscale.
- **SVG is procedural-only and by-request.** The born-vector path emits SVG for the procedural tier;
  the raster→SVG *trace* for diffusion/composite is a documented fast-follow. The PNG is always the
  deliverable.
- **Origins are tradition-level, from public-domain corpora.** `russian`/`english`/`japanese` are
  trained sd15 LoRAs; the other origins run the generic line-art path. Not any living illustrator's
  exact hand. See [`BOOKART_STYLES.md`](BOOKART_STYLES.md).
- **`matte` transparency is a convenience.** U2Net is trained on natural photos, so its silhouette
  quality on B/W ornament is unverified — the luminance model (§7.2) is the primary path.
- **Font export, glyph-driven initial *series*, EPUB parsing, and a repair command** are noted in the
  RFC as fast-follows; the shipped surface is the twelve commands above.

## Companion documents

- [`RFC_BOOKART_1.md`](RFC_BOOKART_1.md) — the full design.
- [`Tutorials/BOOKART_TUTORIAL.md`](Tutorials/BOOKART_TUTORIAL.md) — a hands-on walkthrough.
- [`BOOKART_TRANSPARENCY.md`](BOOKART_TRANSPARENCY.md) — the luminance-alpha model, tint, binarisers,
  born-vector SVG, exact-print sizing, and the symmetry engine (the counter-intuitive core).
- [`BOOKART_STYLES.md`](BOOKART_STYLES.md) — the origin × technique system and the origin LoRAs.
