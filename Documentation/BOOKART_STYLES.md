# `bookart` styles — origin × technique

A `bookart` style is **two orthogonal axes**: an **origin** (an illustration *tradition*) and a
**technique** (a *drawing method*). They compose freely — `russian × line` and `russian × woodcut` are
both reachable, as are `japanese × engraving` and `generic × silhouette`. This is the ornament analog
of the `plakat style` catalog, and it mirrors persona's separation of *what a face is* from *how it is
rendered*.

```
             origin  (tradition)  ─────────────────────────────►
technique    russian   english   japanese   american  european  chinese  generic
 (method)  ┌────────────────────────────────────────────────────────────────────
   line     │   ●         ●          ●          ●         ●         ●        ○
   woodcut   │   ●         ●          ●          ●         ●         ●        ○
   engraving │   ●         ●          ●          ●         ●         ●        ○
   stipple   │   ●         ●          ●          ●         ●         ●        ○
     …       │
```

`●` = reachable via a trained origin LoRA; `○` = the LoRA-free `generic` line-art path (a prompt
scaffold, no LoRA). All **six** named origins now ship a trained LoRA. **Every cell renders** — no
combination is a dead end.

## Origins (traditions)

The `origin` field selects a tradition preset: a prompt scaffold (a tradition cue), a default technique,
a default motif set, and — for the six named origins — a trained LoRA.

| Origin | Character | Default technique | Default motifs | Style source |
|---|---|---|---|---|
| **russian** | Bilibin folk borders, lubok, strong *застАвка*/*концовка* | woodcut | firebird · oak-leaf · vine | **LoRA** (Bilibin) |
| **english** | Beardsley line + Morris foliate ornament | line | rose · vine · peacock | **LoRA** (Beardsley) |
| **japanese** | ukiyo-e / sumi brush line | line | wave · crane · pine | **LoRA** (Hokusai) |
| **american** | golden-age pen/woodcut illustration | woodcut | oak-leaf · star | **LoRA** (Howard Pyle) |
| **european** | Dürer / Doré wood-engraving | engraving | acanthus · laurel | **LoRA** (Gustave Doré) |
| **chinese** | baimiao outline + woodblock | line | lotus · crane · cloud | **LoRA** (Shan Hai Jing · Gujin Tushu Jicheng) |
| **generic** | neutral clean-ink ornamental line | line | leaf · scroll | generic path (no LoRA) |

The scaffold sets the tradition cue even without a LoRA — `european` prompts "in the tradition of
European engraving, Dürer and Doré line" whether or not a weight file exists.

**Custom traditions.** Drop an `assets/bookart/lexicon.hjson` (or point `PLAKAT_BOOKART_LEXICON` at one)
to add or re-scaffold origins with no rebuild — each entry is `{ name, scaffold, default_technique,
motif: [...], hosted_lora }`. A custom origin renders through its scaffold immediately; set `hosted_lora`
only if you host a matching `<name>-sd15.safetensors`. `bookart origins` shows them tagged `(custom)`.

## Techniques (drawing methods)

The `technique` field is orthogonal to origin. It drives **both** the LoRA selection *and* the finisher
binariser (§7.1), so the same origin reads as a genuinely different hand across techniques:

`line · woodcut · engraving · stipple · cross-hatch · silhouette · ink-wash · scratchboard`

| Technique | Prompt cue | Binariser |
|---|---|---|
| `line` | clean black ink line art, no shading | xdog |
| `woodcut` | bold woodcut, high contrast | threshold-bold |
| `engraving` | fine engraving, cross-hatched linework | engrave-invert (white-on-black) |
| `stipple` | stippled, dotted shading | dither (Floyd–Steinberg) |
| `cross-hatch` | cross-hatched pen lines | xdog |
| `silhouette` | solid black silhouette | matte-solid |
| `ink-wash` | sumi ink wash | halftone |
| `scratchboard` | white lines on black | threshold-invert |

Binariser behaviour is detailed in [`BOOKART_TRANSPARENCY.md`](BOOKART_TRANSPARENCY.md); `ink.weight`
biases each one.

## The origin LoRAs

**Six** origins ship as **trained sd15 LoRAs**, hosted publicly at **`vulogov98/plakat-bookart`**:

```
vulogov98/plakat-bookart
  russian-sd15.safetensors     english-sd15.safetensors     japanese-sd15.safetensors
  american-sd15.safetensors    european-sd15.safetensors    chinese-sd15.safetensors
```

Each was trained (via `plakat style train --base sd15`) on ~40 grayscale-prepped **public-domain**
images from Wikimedia — Bilibin / Beardsley / Hokusai (russian/english/japanese), and Howard Pyle /
Gustave Doré / Shan Hai Jing + Gujin Tushu Jicheng woodblock plates (american/european/chinese) — at
rank 16, 256², 180 steps. The fetch → grayscale → train → host pipeline is scripted under
`tools/bookart/` (`fetch_training_corpus.py`, `prep_grayscale.py`, `train_origins.sh`).

**You never fetch these manually.** When a spec's `origin` is one of the six, the diffusion tier
resolves the LoRA by the `repo#file` source form
`vulogov98/plakat-bookart#<origin>-sd15.safetensors`; hf-hub downloads and caches it, the loader merges
it (128/128 UNet targets), and the render weaves the `bookart_<origin> style` trigger into the prompt:

```
  ↳ origin LoRA: bookart_russian (sd15)
```

The LoRAs are measurably cleaner than the generic path — on the baseline metrics the russian LoRA
renders at chroma 0.053 / symmetry-RMS 0.134 versus the generic path's 0.16–0.38 / 0.29–0.48. They
**raise the ceiling**; they are not on the critical path. `corpus/bookart_origins.sh` renders a
signature ornament in every origin's hand for side-by-side comparison.

## `generic` — the guaranteed fallback

`generic` is the **LoRA-free** origin. It renders from a neutral prompt scaffold + the technique cue
through the finisher — **no LoRA required** — so every `origin × technique` combination works day one,
with or without a weight file. This is the design's safety net (RFC R3): if an origin LoRA is
unavailable, `generic` still renders the whole matrix. (A custom origin with no hosted LoRA renders the
same way — through its scaffold.)

```
  ↳ generic scaffold path (no hosted LoRA)
```

## Choosing a style

- **Geometric ornament** (border, corner, divider, fleuron, dinkus, endpaper) is procedural and
  **weight-free** — origin/technique only tint its prompt-adjacent metadata; the geometry is the same
  crisp vector art regardless. Pick any origin.
- **Pictorial ornament** (vignette, frontispiece, marginalia, colophon) is where origin×technique
  actually shows — use any of the six trained origins for its tradition's hand, or the generic path for
  a neutral line look. `bookart origins` lists them.
- **Composite ornament** (headpiece, tailpiece, initial) combines a procedural frame with a diffusion
  inlay, so both axes matter: the frame is geometry, the inlay carries the origin's hand.

To cross traditions, `bookart blend <a> <b> --out c.hjson` writes a new spec with the **origin of A**
and the **technique of B**, unioning both motifs — e.g. a Russian firebird drawn with a Japanese line
hand.

## See also

- [`BOOKART.md`](BOOKART.md) — the command + schema reference and the ornament vocabulary.
- [`BOOKART_TRANSPARENCY.md`](BOOKART_TRANSPARENCY.md) — the technique binarisers, the luminance-alpha
  model, and exact print sizing.
- [`Tutorials/BOOKART_TUTORIAL.md`](Tutorials/BOOKART_TUTORIAL.md) — a hands-on walkthrough.
- [`STYLES_TUTORIAL.md`](Tutorials/STYLES_TUTORIAL.md) · [`HOW_TO_CREATE_MY_OWN_STYLE.md`](Tutorials/HOW_TO_CREATE_MY_OWN_STYLE.md)
  — the general `plakat style` catalog these origins mirror.
