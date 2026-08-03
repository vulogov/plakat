# `plakat bookart` — corpus walkthrough

A reproducible demonstration of the BOOKART-1 feature (plakat 6.0.0). Everything here renders from the
small HJSON specs in this directory; run [`bookart_run.sh`](bookart_run.sh) to regenerate the images
under `corpus/images/bookart/`.

```bash
cargo build --release --features metal   # once
corpus/bookart_run.sh                     # renders the whole corpus
```

## The specs

| file | what it shows |
|---|---|
| [`bookart_ornament.hjson`](bookart_ornament.hjson) | a single **procedural** ornament (a Russian border) — vector-native, **zero weights**, `png` + born-vector `svg` |
| [`bookart_kit.hjson`](bookart_kit.hjson) | a coherent **kit** — 5 ornaments (procedural border/divider/fleuron/corner + a diffusion firebird vignette) sharing one origin, technique, motif DNA, and seed lineage |
| [`bookart_book.md`](bookart_book.md) | a 3-chapter book, input to `bookart manuscript` |

## What the driver produces

1. **`bookart render bookart_ornament.hjson`** → `border.png` (transparent, A5 @ 300 DPI) + `border.svg`
   (a print-sized born-vector file, crisp at any DPI). No GPU, no model download.
2. **`bookart illustrate "a firebird among oak branches" --origin russian`** → `plate.png` — a standalone
   B/W plate; the `russian-sd15` LoRA auto-resolves from `vulogov98/plakat-bookart`.
3. **`bookart render bookart_kit.hjson`** → `composite.png` — the kit's first ornament as a composite
   (procedural frame + diffusion inlay).
4. **`bookart kit bookart_kit.hjson`** → `kit/` — every ornament, a `contact_sheet.png`, a
   `manifest.json`, and a CLIP style-**coherence** score.
5. **`bookart manuscript bookart_book.md --kit bookart_kit.hjson --latex`** → `manuscript/` — a
   frontispiece + a seed-varied headpiece & a tailpiece **per chapter**, a `manifest.json`, a contact
   sheet, and an `includes.tex` you can `\input` into a LaTeX book.

## The idea

A book ornament is a **spec**, a **transparent print-sized image**, and a **measurement** — not a prompt
fragment. `bookart` resolves the spec deterministically, renders it through the right tier (procedural
for geometry, diffusion for pictures, composite for both), makes it transparent with a B/W-native alpha
model (ink darkness *is* opacity), and places it on an exact page canvas. See
[`Documentation/BOOKART.md`](../Documentation/BOOKART.md) and
[`Documentation/BOOKART_TRANSPARENCY.md`](../Documentation/BOOKART_TRANSPARENCY.md).
