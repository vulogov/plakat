# plakat paint — tuning controls per technique

`plakat paint` renders a painting in **stroke space**: the model makes a low‑res **armature**, and a
weight‑free **stroke engine** paints the full‑res picture from it in a chosen **medium**. Every knob below is
authored in the **HJSON spec** (the control surface) and can be overridden on the **CLI**. The **medium sets the
default behaviour** for each technique; the spec/CLI only override what you want to change.

Precedence: **CLI flag → HJSON field → medium default**.

```
plakat paint painting.paint.hjson --out out.png     # spec-driven (armature from `subject:`, or `reference:`)
plakat paint from photo.png --style fidelity ...     # paint an existing image (CPU, no GPU)
plakat paint new painting.paint.hjson                # scaffold a spec with all controls documented
```

The **target quality** for each technique is a real painting in that medium — a palette‑knife impasto oil, a
luminous watercolour, a matte gouache, a graded ink wash, a graphite pencil landscape.

---

## Global controls (any technique)

| HJSON field        | CLI flag            | Range / values                         | What it does |
|--------------------|---------------------|----------------------------------------|--------------|
| `subject`          | —                   | prose                                  | What to render; the model builds the armature from it. |
| `reference`        | —                   | image path                             | Paint an existing image instead of rendering an armature. |
| `negative`         | —                   | prose                                  | What to avoid in the armature (`deformed hands, back turned`). |
| `model`            | —                   | `sdxl` `sd35` `sana` `sd15` `pixart` `pony` `flux` | Armature model — the **ceiling** for the painting. |
| `steps`            | —                   | 1–100                                  | Armature diffusion steps. More = clearer figure / net / faces. |
| `medium`           | —                   | see techniques below                   | The technique — sets all behaviour defaults. |
| `palette`          | —                   | `zorn` `split-primary` `verdaccio` `earth` `limited-landscape` `sumi` | Pigment set (subtractive Kubelka‑Munk mixing). |
| `style`            | `--style`           | `fidelity` \| `legible` \| `impressionist` | How closely to track the armature. |
| `define`           | `--define`          | 0..1                                   | Edge hardness — how many boundaries meet crisply. |
| `haze`             | `--haze`            | 0..1                                   | Aerial perspective (background recession). Use **0** with `fidelity`. |
| `budget.strokes`   | `--strokes`         | int                                    | Total marks. Auto‑derived from size × medium if unset. |
| `stroke_length`    | `--stroke-length`   | 0.2..4  (1 = default)                  | Mark **length** — longer = clean sweeping strokes; shorter = choppier. |
| `stroke_width`     | `--stroke-width`    | 0.3..3  (1 = default)                  | Mark **width** — wider = fewer/broader; narrower = finer/more. |
| `bleed`            | `--bleed`           | 0..1                                   | Wet‑into‑wet fusion / bloom (wet media). |
| `opacity`          | `--opacity`         | 0.1..1                                 | Body — 1 = opaque; low = transparent (ground glows through). |
| `pickup`           | `--pickup`          | 0..1                                   | Dirty‑brush drag — fuses neighbouring colour. |
| `impasto`          | `--impasto`         | 0..1                                   | Textured thick‑paint relief, relit from stroke height (oil/knife). |
| `chroma`           | `--chroma`          | 0.3..2  (1 = neutral)                  | Saturation range — >1 vivid (oil); <1 muted (gouache/watercolour). |
| `dry_shift`        | `--dry-shift`       | −0.4..0.4                              | Value change on drying — + lighter (watercolour); − matte mid (gouache). |
| `granulate`        | `--granulate`       | 0..1                                   | Pigment settling into the paper tooth — mottled watercolour/graphite grain. |
| `sheen`            | `--sheen`           | 0..1                                   | Gloss specular on paint ridges (oil glossy; watercolour/gouache matte). |
| `lift`             | —                   | 0..1                                   | Wipe removability — oil lifts freely; watercolour staining barely. |
| `surface.size`     | `--size`            | `WxH`                                  | Output size. |
| `seed`             | —                   | int                                    | Reproducible. |

**Application vs. material.** `bleed`/`opacity`/`pickup`/`impasto`/`stroke_*` control *how the paint is
applied*; `chroma`/`dry_shift`/`granulate`/`sheen`/`lift` model *how the paint itself behaves* (its saturation
range, drying value‑shift, granulation, gloss, and removability) — so a watercolour and an oil differ at the
material level, not only in softness. Every material value defaults per medium and is recorded in the score, so
`plakat paint replay` reproduces the technique exactly. The **palette** also defaults to the one that suits the
medium (sumi for ink/pencil, split‑primary for gouache, limited‑landscape for watercolour) when the spec names
none.

### `style` — the fidelity register
- **`fidelity`** — paint **closely** from a sharp armature: tight edge‑following, clean low‑waver strokes, most
  passes resolve detail. For **recognizable, detailed** results. Pair with `haze: 0`.
- **`legible`** — resolves features but keeps a painterly looseness (soft masses + a detail tier).
- **`impressionist`** — deliberately loose: soft masses of colour, no detail tier, no edge hardening.

### `model` — the armature is the ceiling
The stroke engine can only paint what the armature contains. `sd35` (best anatomy/hands/faces; low‑mem for
24 GB Metal) · `sdxl` (balanced default) · `sana` (light/fast, painterly — good drafts) · `sd15` (fastest) ·
`pixart` (artistic) · `pony` (stylized) · `flux` (coherent — **GGUF Flux broken on Metal**).
**Workflow:** draft cheap with `sana`/`sd15`, finalize with `sdxl`/`sd35`.

### Per‑element brushes (composition)
For a `composition.elements` scene, each element takes a `brush` from the vocabulary —
`flat` (masses/skies) · `filbert` (general) · `round` (faces/detail) · `fan` (foliage/waves) ·
`rigger` (ropes/masts/fine lines) · `knife` (broad opaque slabs) · `wash` (sky/watercolour washes).

---

## Techniques and their tuning

Each medium ships characteristic defaults; override any of them in the spec.

### Oil — `medium: oil-direct` (alla‑prima) / `oil-indirect` (layered)  🎯 *palette-knife impasto landscape*
Opaque, buttery, **distinct** marks that sit on top of one another; **thick paint catches light (impasto)**;
a dirty brush harmonises colour; broken colour laid in adjacent dabs.
- Defaults: `opacity 1.0` · `impasto 0.60` (oil‑direct) / `0.40` (indirect) · `bleed 0.08` · `pickup 0.30`/`0.40`.
- Tune: **`impasto` up (→0.8)** and **`stroke_width` up (→1.3)** for a chunky palette‑knife feel; `pickup` up for
  more wet‑blend; `stroke_length` up for confident strokes. Try `brush: knife` per element. Keep `bleed` low.
- Good with: `style: fidelity`, `palette: zorn` / `limited-landscape` / `split-primary`.

```hjson
medium: oil-direct
style: fidelity
impasto: 0.8
stroke_width: 1.3
pickup: 0.35
```

### Watercolour — `medium: watercolour`  🎯 *luminous wet-in-wet washes*
Transparent, luminous, **the paper glows through**; pigment **bleeds and blooms** wet‑into‑wet; whites are the
reserved paper.
- Defaults: `opacity 0.45` · `bleed 0.55` · `pickup 0.15` · `impasto 0` · whites reserved automatically.
- Tune: `bleed` up (→0.7) for looser washes/bloom; `opacity` down (→0.35) for more paper glow; `stroke_length`
  up for flowing washes. Keep `pickup` low, `impasto` 0.
- Good with: `style: fidelity`/`legible`, `palette: limited-landscape`/`sumi`, `haze: 0`.

```hjson
medium: watercolour
style: fidelity
bleed: 0.6
opacity: 0.5
pickup: 0.15
```

### Gouache — `medium: gouache`  🎯 *matte opaque, flat washes + crisp marks*
Opaque **matte** body colour; flat even coverage; crisp opaque marks laid light‑over‑dark (back‑to‑front); no
gloss.
- Defaults: `opacity 1.0` · `impasto 0.15` (matte, slight body) · `bleed 0.05` · `pickup 0.80`.
- Tune: `pickup` down (→0.4) for flatter poster‑like blocks; `stroke_width` up for broad flat fills; keep
  `bleed` low and `impasto` low (matte, not oily).
- Good with: `style: legible`, opaque palettes (`split-primary`, `earth`).

```hjson
medium: gouache
opacity: 1.0
impasto: 0.15
pickup: 0.45
stroke_width: 1.2
```

### Ink wash — `medium: ink-wash`  🎯 *graded monochrome washes (sumi-e)*
Flowing transparent washes; strong bleed; whites reserved; graded from light to dark.
- Defaults: `opacity 0.40` · `bleed 0.70` (very fluid) · `pickup 0.90` · `impasto 0`.
- Tune: `bleed`/`pickup` are the character — lower for tighter washes. `palette: sumi`, `style: legible`.

```hjson
medium: ink-wash
palette: sumi
bleed: 0.7
opacity: 0.4
```

### Pen & ink — `medium: pen-ink`  🎯 *crisp line + hatching (line-and-wash)*
Crisp lines and **hatching / cross‑hatching**; value built by mark **density**, not paint body. No bleed/pickup.
For *line‑and‑wash*, render the ink first, then a separate watercolour pass (compose two layers).
- Defaults: `bleed 0.0` · `opacity 1.0` · `pickup 0.0` · density marks (darker = more hatch).
- Tune: `stroke_length` for longer hatches; `define` up (→0.8) for crisp contours. `palette: sumi`.
  `bleed`/`opacity` have little effect (marks are discrete).

```hjson
medium: pen-ink
palette: sumi
style: legible
define: 0.8
```

### Pencil / graphite — `medium: pencil`  🎯 *soft graphite landscape, hatched shading*
Monochrome **graphite grey** (not black); value by **hatching / shading density**; soft graded tones from light
**smudging**; fine lines; paper white for highlights. No colour, no impasto.
- Defaults: `body 0.7` (greys, semi‑transparent) · `bleed 0.15` (smudge → soft gradients) · `pickup 0.10` ·
  density marks · reserved paper whites.
- Tune: `bleed` up (→0.3) for smoother, smudgier shading; `bleed` down for crisper pencil lines;
  `stroke_length` for longer strokes; `define` up for a sharper drawn line. Use `palette: sumi`.

```hjson
medium: pencil
palette: sumi
style: fidelity
bleed: 0.2
define: 0.7
```

### Tempera — `medium: tempera`  🎯 *fine cross-hatched build-up (egg tempera)*
Fine, semi‑opaque **cross‑hatching**; many small disciplined marks, little blending.
- Defaults: `opacity 0.85` · `bleed 0.03` · `pickup 0.05` · `impasto 0.12` · density marks · many stages tolerated.
- Tune: `stroke_width` down (→0.8) and moderate `stroke_length` for fine hatching; keep `bleed`/`pickup` low.

```hjson
medium: tempera
style: fidelity
stroke_width: 0.8
```

---

## Quick reference — default behaviour by medium

| medium        | body | bleed | pickup | impasto | chroma | dry_shift | granulate | sheen | lift | marks      | whites   | palette |
|---------------|:----:|:-----:|:------:|:-------:|:------:|:---------:|:---------:|:-----:|:----:|------------|----------|---------|
| oil-direct    | 1.00 | 0.08  | 0.30   | 0.60    | 1.08   | 0.00      | 0.00      | 0.15  | 0.80 | continuous | pigment  | zorn |
| oil-indirect  | 0.90 | 0.10  | 0.40   | 0.40    | 1.05   | 0.00      | 0.00      | 0.12  | 0.70 | continuous | pigment  | zorn |
| gouache       | 1.00 | 0.05  | 0.80   | 0.15    | 0.85   | −0.05     | 0.00      | 0.00  | 0.50 | continuous | pigment  | split-primary |
| watercolour   | 0.45 | 0.55  | 0.15   | 0.00    | 0.92   | +0.08     | 0.35      | 0.00  | 0.20 | continuous | reserved | limited-landscape |
| ink-wash      | 0.40 | 0.70  | 0.90   | 0.00    | 0.90   | +0.05     | 0.22      | 0.00  | 0.10 | continuous | reserved | sumi |
| pen-ink       | 1.00 | 0.00  | 0.00   | 0.00    | 0.80   | 0.00      | 0.00      | 0.00  | 0.00 | density    | reserved | sumi |
| pencil        | 0.70 | 0.15  | 0.10   | 0.00    | 0.70   | 0.00      | 0.30      | 0.05  | 0.60 | density    | reserved | sumi |
| tempera       | 0.85 | 0.03  | 0.05   | 0.12    | 0.95   | 0.00      | 0.10      | 0.05  | 0.30 | density    | pigment  | verdaccio |

Every value is a **default** — set the same‑named field in the spec (or the CLI flag) to override it. The score
records the applied `bleed` / `opacity` / `impasto` so a re‑render (`plakat paint replay`) reproduces the
technique exactly.
