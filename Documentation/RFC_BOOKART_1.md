# RFC BOOKART-1 — Controllable black-and-white book-ornament composition

**Status:** draft
**Track:** B (bookart)
**Scope:** a new `plakat bookart` subcommand, a `BookArtSpec` HJSON schema, a
deterministic spec resolver, an ornament-layout geometry engine, a
procedural+diffusion **hybrid render router**, a B/W **finisher** (transparency +
binarization + exact-print sizing, with **optional** vectorization to SVG), a
print/ink scorecard, and — as the flagship — coherent **kit** and
**manuscript-aware** set generation.

**Output contract:** the primary, always-emitted artifact is a **transparent,
correctly-page-sized PNG**. Every ornament is *generated → transparented → sized*.
Vector (SVG/EPS/PDF) is a **secondary, by-request** format (`--format svg`), not
the default and not on the critical path.
**Compatibility:** fully additive. No existing flag, scenario field, Bund word,
sidecar key, or default image output changes. Sibling in shape to `persona`
(RFC PERSONA-1): the same *spec → resolver → conditioned render → composite →
measure* spine, applied to decorative ornament instead of human identity.

---

## 1. Summary

`plakat bookart` turns a declarative HJSON description of a **black-and-white
book ornament** — a chapter headpiece, a tailpiece, a decorated initial, a
border, a corner-piece, a divider, a vignette, a frontispiece — into a
*reproducible, print-ready, transparent* asset, rendered in a chosen illustration
tradition and drawing technique, at an exact page size.

The spec resolves through a deterministic pipeline into an ornament layout, a
render decision (procedural, diffusion, or a composite of both), a B/W-native
transparency pass, and placement on an exact page-size canvas at a target DPI —
**always emitting a transparent, correctly-sized PNG**. On request, that same
finished ornament is also traced to a **vector (SVG)**. Any rendered ornament can
then be *measured* against the spec that produced it — chroma purity, alpha
cleanliness, symmetry, ink weight, resolution — and the measurement drives retry
and repair.

Single-ornament generation is the v1. The flagship is a **kit** — a coherent
matched *set* of ornaments that share one style and one decorative motif — and a
**manuscript-aware** mode that emits a per-chapter matched set for a whole book.

The through-line: **a book ornament is a spec, a transparent print-sized image,
and a measurement — not a prompt fragment.** Where `persona` anchors *identity*,
`bookart` anchors a *motif* and a *drawing hand*.

---

## 2. Motivation

### 2.1 The problem

Generating usable book ornament by prompting a diffusion model fails in five
distinct, compounding ways.

**2.1.1 Not actually black-and-white.** "Black and white illustration" yields a
*desaturated photo*, not ink on paper — grey gradients, soft focus, no clean line
structure. Book ornament is a line/mark idiom (pen, woodcut, engraving,
silhouette), not a tonal-photo idiom.

**2.1.2 Not actually transparent.** Ornament must sit on a page of any colour or
texture. "Remove the background" of a grey illustration leaves halos and eats
thin lines. B/W has a *better* transparency model available (§7) that prompting
cannot reach.

**2.1.3 Not symmetric.** Most ornament is bilaterally or radially symmetric by
construction. Diffusion cannot hold symmetry; a "symmetric border" comes out
lopsided. Symmetry is a *geometric* guarantee, not a prompt hope (§6.3).

**2.1.4 Not print-sized, not transparent-at-size.** Raster output at an arbitrary
resolution with an opaque background is unusable for print. Ornament must be
emitted at **exact page geometry** (A5 text-block width, 3 mm bleed, 300 DPI) and
transparent (§2.1.2). Prompting has no notion of page geometry or DPI. (Vector —
crisp at any DPI, editable — is valuable *on request*, but the primary,
always-emitted artifact is a correctly-sized transparent **raster PNG**, §7.5.)

**2.1.5 Fake lettering.** Diffusion scrawls gibberish letterforms into decorative
space — fatal for a book ornament, where any glyph must be intentional and
legible. (We hit exactly this in persona rendering; the anti-text remedy carries
over and is made a *probe* here, §9.)

These are five categorically different failures needing five different remedies —
a colour idiom, a transparency model, a symmetry engine, a vector+print pipeline,
and a text guard — which is why prompt-only approaches plateau.

### 2.2 Why the existing surfaces do not solve this

plakat already has most of the *mechanisms*, none of them wired to ornament:

| Existing surface | Provides | Gap |
|---|---|---|
| `generate` + lineart/scribble ControlNet | line-structured B/W generation | no ornament layout, no transparency, no vector, no print size |
| `style` catalog + `style train` (SD3.5 LoRA) | style presets + LoRA training | no illustration-tradition packs; not B/W-tuned |
| `remove` / `replace-bg` (U2Net matte, alpha) | transparency | photo-matte, not luminance-native ink alpha |
| `fractals` engine (pure Rust, 17 families, compose) | deterministic procedural art, infinite-res | not shaped as ornament primitives (rosette/knot/guilloché/border) |
| `persona` (HJSON spec → resolver → composite → verify) | the whole architectural spine | anchors identity, not motif |
| upscale / DPI / image ops | resolution handling | no page-size / text-block / bleed model |

There is no path from *"a decorative scheme I am inventing"* to *"a coherent,
print-ready ornament set."* This RFC supplies it and reuses every mechanism above.

### 2.3 Baseline measurement is a prerequisite

Before implementation, quantify the naive control: for each origin×technique, one
fixed ornament prompt, N = 16 seeds, "generate grey + `replace-bg`" at A5/300.
Report: **chroma leakage** (fraction of pixels with saturation > ε), **alpha halo
width** (partial-alpha ring thickness), **symmetry RMS** (fold-and-compare),
**stray-glyph rate** (text-hallucination probe), and **line cleanliness** (does it
vectorise to < K paths without noise). These are the control every phase reports
against, committed as a corpus entry.

---

## 3. Goals and non-goals

### 3.1 Goals

- **G1.** A declarative, human-editable, version-tolerant `BookArtSpec` HJSON
  schema covering ornament type, origin, technique, motif, ink, transparency,
  symmetry, and page geometry.
- **G2.** A deterministic resolver: `(spec, lexicon)` → layout geometry, render
  plan, finisher chain, and canvas placement, as a pure, byte-stable function.
- **G3.** **Out-of-box transparency native to B/W** — ink darkness *is* opacity
  (luminance→alpha), plus threshold→alpha and matte modes; no photo-style halos.
- **G4.** A **hybrid render router**: procedural (from the fractal engine + new
  ornament primitives) for geometric/knotwork/guilloché ornament; diffusion
  (LoRA + lineart ControlNet) for pictorial ornament; **composite** ornaments
  that inlay a diffusion picture inside a procedural frame.
- **G5.** **A transparent, exact-page-sized PNG as the primary, always-emitted
  artifact** — every ornament is generated, made transparent (§7.2), and placed on
  the exact page canvas at the target DPI, with no further flags. **Vector
  (SVG/EPS/PDF) is a secondary, by-request output** (§7.5, `--format svg`) —
  born-vector for procedural ornament, traced for diffusion — off the critical
  path.
- **G6.** **Exact print sizing** — named page sizes (A-series, US, trade, custom)
  at a target DPI, with a text-block / margin / gutter / bleed model and DPI
  embedded in output.
- **G7.** An **origin × technique** style system: 2–3 trained origin LoRAs plus a
  **generic line-art path** that works for any combination without a LoRA.
- **G8.** **Symmetry-locked generation**: generate a fundamental domain, then
  mirror/rotate/translate for a guaranteed-symmetric result at fractional compute.
- **G9.** **Coherent kit** (a matched set sharing one motif "DNA" and one hand)
  and a **manuscript-aware** set (per-chapter matched ornaments for a whole book)
  — the flagship.
- **G10.** A **print/ink scorecard** (chroma purity, alpha cleanliness, symmetry
  error, ink weight, resolution/DPI, safe-area, stray glyphs) driving retry +
  attribute-targeted repair.
- **G11.** Full reach into scenario, compile, scripting, and the library API.

### 3.2 Non-goals

- **N1.** Colour illustration. This is a B/W idiom. "Ink tint" (§7.4) recolours
  the alpha channel; it does not make colour art.
- **N2.** New base-model ports. Ships no new generative family.
- **N3.** A typesetting / page-layout engine. `bookart` emits *assets* (and an
  optional proof/preview, §12); it is not an InDesign/LaTeX replacement.
- **N4.** Free text rendering inside ornament. Legible letterforms come *only*
  from the glyph-driven initial path (§6.5); everywhere else, text is suppressed
  and probed against.
- **N5.** Forging a specific living illustrator's exact hand. Origins are
  *tradition-level* (Bilibin-lubok, Beardsley-era pen, ukiyo-e sumi), trained on
  public-domain corpora (§8.4, §13).
- **N6.** Full CMYK colour management. Print output is K-only / 1-bit / greyscale;
  no ICC pipeline.
- **N7.** Arbitrary photo → ornament. The `illustrate` path (§6.6) synthesises
  from a *prompt*, not by converting a supplied photograph.

---

## 4. Terminology and naming

- The **subcommand** is `plakat bookart`, one word, reads as a noun the verbs
  operate on (`bookart render`, `bookart kit`).
- An **ornament** is a single decorative element; a **kit** is a coherent matched
  set; a **plate** is a full-page pictorial illustration.

| Term | Meaning |
|---|---|
| **Spec** | the `BookArtSpec` HJSON document (one ornament or one kit) |
| **Lexicon** | data mapping origin/technique/motif/ornament vocabulary to prompts, geometry, primitives, and finisher chains |
| **Origin** | an illustration *tradition* preset (russian, english, japanese, …) |
| **Technique** | a *drawing method* (line, woodcut, engraving, stipple, silhouette, ink-wash), orthogonal to origin |
| **Motif** | the shared decorative DNA threaded through every ornament in a kit |
| **Render tier** | `procedural` \| `diffusion` \| `composite` — the source router (§5.3) |
| **Finisher** | the post chain: technique binarization → transparency → symmetry tiling → vectorize |
| **Page model** | size, DPI, orientation, margins, gutter, bleed, text-block |
| **Fundamental domain** | the sub-region generated before symmetry replication (§6.3) |
| **Scorecard** | the per-attribute print/ink measurement of a render vs its spec |
| **Kit coherence** | cross-set consistency of hand + motif (the kit analog of identity coherence) |

Named ornament vocabulary (the resolver's type keys, with traditional synonyms):
**headpiece** (bandeau · *застАвка*), **tailpiece** (cul-de-lampe · *концовка*),
**initial** (drop-cap; historiated/inhabited), **corner**, **border** (frame),
**divider** (rule), **fleuron** (printer's flower) & **dinkus** (asterism ⁂),
**vignette** (spot), **frontispiece** (plate), **colophon** (device),
**endpaper** (seamless pattern), **marginalia**.

---

## 5. Architecture

### 5.1 Layers

```
                       bookart.hjson  (Layer 0)
                             │
                   ┌─────────┴─────────┐
             lexicon (Layer 0b): origin × technique × motif presets
                             │
                    resolver (Layer 1)  — pure, byte-stable
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                     │
  ornament geometry    render router          finisher plan
   engine (Layer 2)    (Layer 3, hybrid)      (Layer 4 chain)
  layout · symmetry     proc │ diff │ comp    binarize·alpha·
  · text-block          ─────┼──────┼─────    symmetry·vectorize
        │             fractal│ LoRA+│ frame+
        │             engine │ line │ inlay
        └──────────┬─────────┴──────┴──────────┬──────────┘
                   │                            │
          print/canvas compositor (Layer 5)  scorecard (Layer 6)
          page-size · DPI · bleed · export    chroma·alpha·symmetry·
          PNG + SVG + PDF                     ink·DPI·safe·glyph
                   │
        ┌──────────┴───────────┐
      kit (§10)         manuscript (§11)   ← the flagship
   motif DNA +         chapter structure →
   coherence           per-chapter matched set
```

The layer split mirrors persona: a pure, testable, GPU-free front half (Layers
0–2, and most of 4 and 6), and a weights-bearing render half (Layer 3). Layers 1,
2, 4-finisher, and 6-scorecard are deterministic and CI-testable without a GPU.

### 5.2 Determinism contract

`resolve(spec, lexicon) → RenderPlan` is a pure function. Layout geometry,
symmetry replication, procedural ornament, transparency math, and vectorisation of
procedural output are byte-stable across machines and golden-tested. Diffusion
(Layer 3, diffusion tier) is the one stochastic step; it is seed-locked and, on a
given device, reproducible. A `lock.hjson` (§ artifact layout) pins the lexicon,
motif, origin/technique versions, seeds, and DPI so an ornament re-renders
identically months later.

### 5.3 The hybrid render router (Decision D1)

Each ornament type carries a **default tier**, overridable per-ornament:

| Tier | Source | Default for |
|---|---|---|
| `procedural` | fractal engine + ornament primitives (§ Layer 3a) | border, corner, divider, fleuron, dinkus, endpaper, geometric headpiece/tailpiece, rosette/knot/guilloché motifs |
| `diffusion` | LoRA + lineart ControlNet + XDoG finish (§ Layer 3b) | vignette, frontispiece, pictorial headpiece/tailpiece, historiated initial imagery |
| `composite` | procedural scaffold + diffusion inlay, alpha-composited | headpiece/tailpiece/initial with a pictorial centre inside an ornamental frame; frontispiece with a decorative border |

`composite` is the elegant hybrid: a *procedural frame* (born-vector, symmetric,
crisp) with a *diffusion picture* inlaid into its window and finished to
transparency — the direct analog of persona's geometry-map + detail-composite.
The router chooses the tier from the ornament type and whether the motif is
geometric or pictorial; `--tier` and per-ornament `tier:` override it.

---

## 6. Layer 0 — `BookArtSpec`, Layer 2 — geometry

### 6.1 Single-ornament spec

```hjson
{
  schema: bookart/1
  origin: russian            # a lexicon preset (LoRA + prompt scaffold + motifs)
  technique: woodcut         # drawing method → LoRA + finisher binarizer
  motif: ["firebird", "oak-leaf"]
  ink: { color: black, weight: 0.7, transparency: luminance }  # luminance|threshold|matte
  page: { size: a5, dpi: 300, orientation: portrait, bleed_mm: 3 }
  transparent: true
  output: { formats: [png], tint: black }   # png always; add "svg" to also emit vector (opt-in)

  ornament: {
    type: headpiece          # named vocabulary (§4)
    symmetry: bilateral       # bilateral | radial:N | frieze:GROUP | none
    tier: composite           # override the router
    prompt: "a firebird among oak branches"   # pictorial inlay (diffusion)
    frame: filigree            # procedural scaffold family
    taper: 0.0                 # tailpiece taper; 0 for a band
  }
}
```

Permissive serde like `PersonaSpec`: every field optional, string enums,
untagged colour, unknown keys tolerated + linted. A bare
`{ ornament: { type: divider } }` resolves to a full plan via lexicon defaults.

### 6.2 Ornament-layout geometry (Layer 2, pure)

Each ornament type has a canonical layout resolved against the **text block**
(not the raw page):

- **headpiece** — a band, aspect from the lexicon (≈4:1–8:1), width = text-block
  width, anchored above the first text line.
- **tailpiece** — a `taper`-parameterised tapering region (point-down), centred
  under the last text line.
- **corner** — an L-shaped fundamental domain, placed at `places: 4` corners by
  reflection.
- **border** — assembled from a *tileable edge unit* + a *corner unit* around the
  text-block or page perimeter (procedural assembly, the ornament analog of
  persona's detail compositing).
- **initial** — a square cell sized to N text lines (`--lines 3`), containing a
  glyph (§6.5).
- **divider / fleuron / dinkus** — small centred marks with fixed aspect.
- **vignette / frontispiece** — free / page-fill aspect with edge `fade`.

Layout is a pure function of `(type, page, lexicon)`; golden-tested.

### 6.3 The symmetry engine (Decision-adjacent to D1; G8)

Symmetry groups: `bilateral` (mirror about a vertical axis), `radial:N`
(N-fold rotation), `frieze:GROUP` (translational border strips), `none`.

Procedure: compute the **fundamental domain** (a half, a 1/N wedge, or a repeat
unit), render *only* that domain, then replicate by reflection/rotation/tiling
with seam-blending at the join. Two payoffs: (1) a **guaranteed** symmetric result
that diffusion cannot produce, (2) fractional generation area. For `procedural`
tier the symmetry is inherent to the generator; for `diffusion`/`composite` the
domain is generated at higher resolution and replicated. Seam handling and the
mirror-line blend are the one subtlety; §14 R4.

### 6.4 The page / print model (G6)

```
PageSpec { size, dpi=300, orientation, bleed_mm=3, margins{top,bottom,inner,outer}, gutter_mm }
```

Named sizes in mm → px at DPI (A4 210×297 → 2480×3508 @300; A5, A6, B5, US
Letter/Legal, Trade 6×9″, Mass-market, `custom: {w_mm,h_mm}`). The **text block**
is derived from margins+gutter; ornaments anchor to it. Output canvas is exactly
`size×dpi` px; DPI is written to the PNG `pHYs` chunk and as physical mm on the
SVG. Optional crop/bleed marks. **1-bit** and **halftone** output modes for
letterpress/riso.

### 6.5 Glyph-driven initials (N4 exception)

For `initial`, the target letter (`glyph: "В"`, any script incl. Cyrillic) is
rasterised from a font and passed as a **mask/ControlNet** so the ornament is
built *around* a legible letterform. The letter is the only intentional text in
the system; everywhere else text is suppressed and probed (§9).

### 6.6 `illustrate` — a B/W plate from a prompt (your ask #4)

`bookart illustrate "<prompt>"` synthesises a single B/W illustration suitable as
a `vignette`/`frontispiece`/spot, styled to a given origin×technique (or, inside a
kit, auto-fit to the kit's hand). It is the `diffusion`-tier path exposed directly,
with the full finisher (transparency + vectorise + page placement).

---

## 7. Layer 4 — the finisher (transparency, the headline of your ask)

Runs after the source render, before scoring. Ordered chain:

### 7.1 Technique binarisation

Per `technique`: **line** → XDoG / adaptive threshold to clean contour; **woodcut**
→ high-contrast threshold + bold-mass cleanup; **engraving** → white-on-black
lines (invert + fine-line preserve); **stipple** → error-diffusion dither;
**silhouette** → solid fill via matte; **ink-wash** → retain grey (halftone if
1-bit output). Deterministic; parameters from the lexicon + `ink.weight`.

### 7.2 Transparency — B/W-native (G3)

The differentiator versus "generate grey then remove background":

- **`luminance`** (default for line/hatch): `alpha = curve(1 − L)`, ink =
  `output.tint`. Ink darkness *is* opacity; grey hatching becomes semi-transparent
  grey marks that sit correctly on any page. No matting model, no halo.
- **`threshold`** (crisp 1-bit line): hard cut with a soft ramp near the threshold
  to preserve anti-aliased edges.
- **`matte`** (solid silhouette pieces): U2Net (reused from the edit verbs).
- **`fade` / `vignette`**: feather the alpha at ornament edges for spot art.

Optional **premultiplied** output; optional **two-layer** export (ink layer +
separate tone layer).

### 7.3 Symmetry tiling

Replicate the fundamental domain (§6.3) with mirror-line blend.

### 7.4 Ink tint

Keep alpha from §7.2, recolour ink (`tint: black|sepia|#hex`) — one transparent
asset drops onto any page *and* re-tints without regeneration.

### 7.5 Vectorisation → SVG (secondary, by request)

The mandatory finisher ends at §7.4 + Layer-5 sizing: a **transparent,
page-sized PNG** is always emitted. Vectorisation is **off by default** and runs
only when `--format svg` (or `pdf`/`eps`) is requested:

- **Procedural tier — parametric generators** (guilloché, knotwork, border
  edge/corner units): **born vector** — the generator emits SVG paths directly, no
  tracing, mathematically exact (near-free when SVG is asked for). *Note (G0.4):* the
  fractal-engine-backed procedural sources (flame-rosette, L-system scrollwork) are
  **raster**, so those take the trace path below like diffusion.
- **Diffusion / composite / fractal-raster**: the finished raster → **trace** to SVG
  paths with a **permissively-licensed** tracer — **`vtracer`/`visioncortex`
  (MIT/Apache-2.0)**, resolved in G0.1; *not* GPL potrace.
- Transparency carried as path fill-opacity; export **SVG + EPS + PDF**. This is
  the escape hatch for users who need infinite-DPI / editable assets; it does not
  replace the PNG.

---

## 8. Layer 0b — the origin × technique lexicon (G7, Decision D4)

A catalog mirroring `plakat style` (`catalog.json` + optional LoRA
`safetensors` + prompt scaffold + default motifs + default technique + finisher
overrides). Two orthogonal axes:

### 8.1 Origins (traditions)

| Origin | Character | v1 |
|---|---|---|
| **russian** | Bilibin folk borders, lubok, strong застАвка/концовка | **LoRA** |
| **english** | Beardsley black-mass + fine line, Rackham pen+silhouette, Morris foliate | **LoRA** |
| **japanese** | ukiyo-e sumi line, sumi-e brush, kirie paper-cut | **LoRA** |
| american | Pyle/Brandywine, Rockwell Kent woodcut, Art Deco | generic path / fast-follow LoRA |
| european | Dürer/Doré engraving, German woodcut, Mucha Art Nouveau | generic path / fast-follow LoRA |
| chinese | 白描 baimiao outline, 木刻 woodblock, nianhua | generic path / fast-follow LoRA |

**Ship 2–3 strong LoRAs (russian, english, japanese) + a generic line-art path**
so every origin×technique is reachable day one without a trained LoRA (prompt
scaffold + lineart ControlNet + XDoG finish). Remaining origins land as
fast-follow LoRAs.

### 8.2 Techniques (orthogonal)

`line · woodcut · engraving (white-on-black) · stipple · cross-hatch ·
silhouette · ink-wash · scratchboard`. Technique drives the LoRA selection *and*
the §7.1 binariser, so "russian × line" and "russian × woodcut" are both
reachable.

### 8.3 Motif vocabulary

A motif is a named decorative element (`firebird`, `oak-leaf`, `wave`,
`celtic-knot`, `crane`) threaded into prompts (diffusion) and generator
parameters (procedural). In a kit it is the shared **DNA** (§10).

### 8.4 Training data

Origin LoRAs train on **public-domain** illustration corpora (pre-1929 books,
museum/library scans — Bilibin, Beardsley, Rackham, Hokusai *manga*, Doré) via
the existing `style train` (SD3.5 LoRA, mixed precision). Legal/quality sourcing
is a gating item (§13).

---

## 9. Layer 6 — the print/ink scorecard (G10)

Probes (most pure, no weights):

| Probe | Checks | Weights? |
|---|---|---|
| **chroma-purity** | max pixel saturation < ε (truly B/W) | no |
| **alpha-clean** | background fully transparent; no partial-alpha halo ring | no |
| **symmetry-error** | fold about the declared axis/order; RMS < τ | no |
| **ink-weight** | black coverage + estimated stroke width vs spec/kit | no |
| **resolution/DPI** | px == size×dpi; pHYs correct | no |
| **safe-area/bleed** | content inside margins / within bleed | no |
| **stray-glyph** | text-hallucination detector (OWL-ViT/OCR-lite) flags fake letterforms | reuse |
| **vectorability** | traces to < K paths without noise speckle | no |
| **aesthetic** (opt) | LAION as a distant secondary key on candidates | reuse |

`bookart verify <spec> --image` reports the scorecard; `bookart render` uses it
for rejection sampling (`--min-score`) and retry. Auto-fixes: recolour stray
chroma to grey, re-matte alpha, re-tile symmetry, re-vectorise, drop a
stray-glyph candidate and retry (the persona anti-text remedy, now measured).

---

## 10. Kit — the coherent matched set (G9, flagship part 1)

```hjson
{
  schema: bookart/1
  origin: russian
  technique: woodcut
  motif: ["firebird", "oak-leaf"]        # the shared DNA
  page: { size: a5, dpi: 300 }
  ink: { transparency: luminance }
  kit: {
    seed: 42                              # one lineage; each ornament derives its seed
    ornaments: [
      { type: headpiece, tier: composite }
      { type: tailpiece, taper: 0.6 }
      { type: initial, glyphs: "cyrillic-upper" }   # А–Я decorated set
      { type: corner, places: 4 }
      { type: divider }
      { type: frontispiece, prompt: "the firebird in a winter forest" }
    ]
  }
}
```

`bookart kit` generates the whole set sharing **one origin+technique (one hand)**,
**one motif DNA** (threaded into every prompt + generator param), and **one seed
lineage** (deterministic per-ornament seeds). A **kit-coherence** score (pairwise
CLIP-embedding similarity + ink-weight/technique consistency across the set — the
ornament analog of persona's identity coherence) gates the result; below
threshold, re-derive. Output: a directory of transparent **PNG + SVG** per
ornament, a **contact-sheet PDF** proof, and optionally a **dingbat font** (§12).

---

## 11. Manuscript-aware set (G9, flagship part 2)

```
bookart manuscript book.md --kit kit.hjson --out ornaments/
```

Parse a book's **chapter structure** (Markdown headings, EPUB spine, or a plain
chapter-title list) → emit a *matched* set: **one headpiece per chapter** as a
seed-varied *variation* of the motif (coherent, not identical), a **tailpiece**
per chapter, a **decorated initial** for each chapter's first letter, and a
**frontispiece**. Emits a `manifest.json` mapping chapter → asset paths, and
optional **LaTeX/InDesign includes** (correct-DPI PNGs + `\lettrine` drop-cap
assets, or `\headpiece` macros). This is the scale flagship — the ornament analog
of persona's cast set and the `photos` corpus.

---

## 12. Output & publishing

- **Raster (primary, always)** — transparent PNG at exact page px + DPI metadata;
  1-bit / halftone variants for letterpress/riso.
- **Vector (secondary, on request)** — SVG/EPS/PDF per ornament (§7.5).
- **Dingbat font** — small fleurons/dinkus → an `.otf` so ornaments are usable
  inline in InDesign/LaTeX (`bookart font <dir>`; fast-follow candidate, §14).
- **Proof** — `bookart proof` renders a contact sheet, or drops the kit onto a
  mock text page (a **page proof**) so ornaments are seen in situ (this is preview
  only — not a typesetting engine, N3).

---

## 13. Gating research (do first, architecture-shaping)

Like persona's gating, resolve before building:

1. **Vectorizer license (gates only the opt-in SVG path).** potrace is **GPL** —
   incompatible with this repo's Unlicense/permissive posture. Survey permissive
   Rust tracers (`vtracer`/`visioncortex` — MIT) for quality on B/W ornament; this
   decides the §7.5 dependency. Since vector is by-request, this never blocks the
   core PNG path — worst case, SVG ships a phase later. *Provisional pick:*
   `visioncortex`/`vtracer`.
2. **Baseline (§2.3).** Quantify the naive control so every phase is falsifiable.
3. **Origin-LoRA corpora.** Source + license public-domain illustration sets;
   confirm `style train` produces a usable B/W-line LoRA (vs the generic path).
4. **Procedural coverage.** Which ornament families the fractal engine already
   covers vs new primitives needed (rosette/guilloché/knotwork/tileable-border).
5. **Symmetry seams.** Mirror-line blend quality on diffusion tiers.
6. **Transparency curve.** The luminance→alpha ramp (gamma) that best preserves
   thin lines without greying the page.

---

## 14. Risks

- **R1 fake lettering** (fatal) → anti-text negatives + stray-glyph probe + prefer
  procedural for text-adjacent ornament; glyph path is the only intentional text.
- **R2 tracer license** (GPL potrace) → permissive tracer only (§13.1); affects
  the opt-in SVG path only, never the primary PNG.
- **R3 origin LoRAs need PD data / may underperform** → the **generic line-art
  path** is the guaranteed fallback; the feature works without any LoRA.
- **R4 symmetry seams** on diffusion tiers → higher-res domain + mirror blend;
  prefer procedural or composite where symmetry is strict.
- **R5 alpha halos** from diffusion soft edges → luminance-alpha + threshold
  cleanup + alpha-clean probe.
- **R6 print-color scope creep** → K-only/1-bit/greyscale; no CMYK/ICC (N6).
- **R7 DPI correctness across tools** → pHYs + physical-mm SVG + a resolution probe.
- **R8 lexicon/motif churn** → versioned lexicon + `lock.hjson`.

---

## 15. CLI surface

```
plakat bookart new       <out.hjson> [--origin O --technique T --motif M ... --page a5]
plakat bookart lint      <spec>
plakat bookart show      <spec> [--json]         # resolved layout, tier, finisher chain, canvas
plakat bookart origins                           # list origin × technique presets (like `style`)
plakat bookart render    <spec> [--out DIR] [--page a5 --dpi 300]
                                 [--format png|svg|pdf|eps ...]   # default png; others opt-in
                                 [--tier auto|procedural|diffusion|composite]
                                 [--attempts N] [--min-score S]
plakat bookart illustrate "<prompt>" [--origin O --technique T --page a5 ...]   # a single B/W plate
plakat bookart vectorize <image> [--out svg]     # standalone raster → SVG escape hatch
plakat bookart kit       <kit.hjson> [--out DIR] # coherent matched set (flagship)
plakat bookart manuscript <book.md|epub|list> --kit K [--out DIR] [--latex|--indesign]
plakat bookart verify    <spec> --image IMG [--json]
plakat bookart repair    <spec> --image IMG --attr ink.weight|symmetry|...
plakat bookart proof     <DIR|kit> [--out proof.pdf]
plakat bookart font      <DIR> [--out ornaments.otf]     # fleurons → dingbat font
```

Grouped `--help` sections: Spec, Style, Page, Render, Finish/Output, Scoring, plus
the shared global group.

---

## 16. Artifact layout

Plain-text, no hidden database; everything except `bookart.hjson` is derived and
safely deletable.

```
<bookart>/<name>/
  bookart.hjson          # source of truth; hand-editable
  resolved.hjson         # cached resolution; derived
  ornaments/
    headpiece.png  headpiece.svg
    tailpiece.png  tailpiece.svg
    initial-А.png  initial-Б.png  ...
    frontispiece.png  frontispiece.svg  frontispiece.pdf
  proof.pdf
  ornaments.otf          # optional dingbat font
  manifest.json          # manuscript mode: chapter → assets
  scorecards/<ts>.json
  lock.hjson             # origin/technique/lexicon/motif versions + seeds + DPI
```

`lock.hjson` pins the lexicon version, motif set, origin/technique/LoRA ids, seed
lineage, and page/DPI — so a kit re-renders identically later. `export`/`import`
bundle the directory (spec-only share optional).

---

## 17. Integration surfaces

| Surface | Integration |
|---|---|
| **scenario** | `type: bookart` task; per-task page/format/tier overrides |
| **compile** | a `bookart:` directive resolves to a bookart task |
| **scripting** | `plakat.bookart.render` · `.kit` · `.illustrate` · `.vectorize` · `.verify`, pushing image handles into the existing pipeline (save/upscale/metadata) |
| **library API** | a `BookArt` builder mirroring the existing builder shape, returning in-memory transparent images (+ optional SVG) + scorecard |
| **doctor** | a `bookart` capability section (origins, techniques, tracer, LoRA presence) |

---

## 18. Documentation deliverables

- `Tutorials/BOOKART_TUTORIAL.md` — new → render → kit → manuscript.
- `BOOKART.md` — reference: schema, ornament vocabulary, tiers, page model, every flag.
- `BOOKART_STYLES.md` — origins × techniques, the generic path, adding a LoRA.
- `BOOKART_TRANSPARENCY.md` — the luminance-alpha model, tint, vectorisation, print sizing (the counter-intuitive core, kept separate like persona's details how-to).
- Corpus: 2–3 authored specs + a kit + a manuscript demo under `./corpus`.

---

## 19. Phasing

Front-load the pure, measurable half; build the finisher/scorecard *before*
generation so quality is falsifiable from the first render (the persona lesson).

- **B0** — `BookArtSpec` schema + resolver + lexicon skeleton + `lint`/`show`. Pure, CI without GPU.
- **B1** — the **finisher** (transparency modes + technique binarisers) + **scorecard**. Mostly pure/CI; the always-on transparent-PNG path (generate→transparent→size) is proven here first. *(Vectoriser is deferred to B6 — it is opt-in and off the critical path.)*
- **B2** — ornament-layout geometry + **symmetry engine** + **page/text-block model + DPI sizing**. Pure, golden-tested. Completes the primary output contract: transparent + exactly page-sized.
- **B3** — **procedural** ornament generators (leverage the fractal engine; born-vector). Deterministic, CI-testable.
- **B4** — **diffusion** path: generic line-art (prompt + lineart CN + XDoG), then the 2–3 origin **LoRAs**. First weights phase.
- **B5** — the **composite** tier (procedural frame + diffusion inlay) + the render router.
- **B6** — single-ornament `render` / `illustrate` end-to-end, plus the **opt-in vectoriser** (`--format svg`) / `vectorize` escape hatch. **v1 milestone.**
- **B7** — **kit** (motif DNA + kit coherence). Flagship part 1.
- **B8** — **manuscript** set + `proof` + `font` export. Flagship part 2.
- **B9** — `repair`/edit loop + lineage/blend (variant kit, origin blend).
- **B10** — scenario/compile/Bund/API integration + docs + corpus. **cut.**

B0–B3 and the B1 finisher/scorecard are deterministic → CI without GPUs, exactly
as persona P0–P3 were.

---

## 20. Open questions

1. **Font export (§12) in v1 or fast-follow?** Leaning fast-follow (B8/9) — high
   value but an isolated dependency.
2. **Manuscript input formats** — Markdown + plain chapter-list for v1; EPUB
   parsing fast-follow?
3. **`illustrate` styling** — always auto-fit to a kit when invoked inside one, or
   keep standalone-stylable? (Proposed: both — kit-fit inside a kit, explicit
   `--origin/--technique` standalone.)
4. **Tone in v1** — pure line/threshold only, or ship `ink-wash`/halftone grey too?
   (Proposed: ship line + woodcut + silhouette; grey/halftone in B4+.)
5. **Radial default** — per-ornament sensible `radial:N` defaults (rosette, mon,
   colophon) in the lexicon, or always explicit?

---

## 21. Alternatives considered

- **Pure procedural (no diffusion).** Crisp, deterministic, free — but cannot do
  pictorial ornament (a firebird, a vignette). Rejected; kept as one tier.
- **Pure diffusion (no procedural, no vector).** Cannot hold symmetry, cannot
  print-crisp, scrawls text. Rejected; kept as one tier. (Decision D1 = hybrid.)
- **Vector-first (SVG as the primary artifact).** Rejected by owner decision: the
  primary, always-emitted output is a **transparent, page-sized PNG**; vector is a
  secondary, by-request format (`--format svg`). Raster is the core; vector is the
  add-on for infinite-DPI / editable needs. (Supersedes the earlier "SVG headline"
  framing.)
- **Extend `persona` directly.** Same *pattern*, wrong *domain* — no landmarks, no
  identity, different geometry and scorecard. Reuse the architecture, not the code.
- **Kit-first (no single ornament).** Too big a first cut; single-ornament is the
  testable v1, kit/manuscript the flagship (Decision D3).

---

## 22. Future work

- More origins as LoRAs (american, european, chinese) beyond the generic path.
- Colour-plate mode (chromolithograph / two-colour) — explicitly out of v1 (N1).
- Learned ornament vectorisation (raster→SVG trained on PD engravings).
- Seamless **endpaper** pattern designer + tiling preview.
- Guilloché / security-print rosette generator (procedural, from the fractal core).
- Full EPUB round-trip (inject ornaments back into the book).
- A composition TUI (like persona's) for interactive ornament authoring.

---

## 23. Summary of what must be true for this to work

1. B/W transparency is **luminance-native**, not photo-matte — ink darkness is
   opacity (§7.2). This is the ask's core and the clean differentiator.
2. Symmetry is a **geometric guarantee** (fundamental domain + replicate), not a
   diffusion hope (§6.3).
3. The primary output is a **transparent, exact-print-sized PNG** — every ornament
   is generated, transparented, and placed at true page geometry and DPI (§6.4,
   §7.2); **vector (SVG) is a by-request add-on** (§7.5), never the default.
4. The **hybrid router** puts procedural where geometry/symmetry/vector matter and
   diffusion where pictorial content matters, and **composites** the two (§5.3).
5. A **generic line-art path** makes every origin×technique reachable without a
   trained LoRA; 2–3 LoRAs raise the ceiling (§8).
6. A **kit** shares one hand + one motif DNA, and **manuscript** mode scales it to
   a whole book — the flagship (§10, §11).
7. Every ornament is **measured** (chroma, alpha, symmetry, ink, DPI, glyphs) so
   quality is falsifiable and repairable (§9).
