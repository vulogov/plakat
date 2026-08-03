# `plakat bookart` tutorial

A hands-on pass through the whole pipeline: scaffold a spec, see what it resolves to, render a
zero-weight procedural border and a diffusion vignette, ask for SVG, then build a coherent **kit**, a
**manuscript** set for a whole book, and finish with edit/lineage. `bookart` composes *reusable,
print-ready, transparent* black-and-white book ornaments — the reference is
[`../BOOKART.md`](../BOOKART.md); the counter-intuitive transparency core is
[`../BOOKART_TRANSPARENCY.md`](../BOOKART_TRANSPARENCY.md).

Build the release binary first — debug diffusion is ~50× slower:

```sh
cargo build --release
alias plakat=./target/release/plakat
export PLAKAT_OOM_GUARD_GB=0        # the macOS free-page guard mis-fires under render loops
```

Only the diffusion and composite tiers need a model (sd15, auto-downloaded once) and a GPU. **The
procedural tier needs nothing** — no weights, no download — so start there.

## 1. Author a spec (no weights)

Scaffold, then edit:

```sh
plakat bookart new border.hjson --type border --origin generic --technique line --page a5
```

That writes a valid partial `BookArtSpec` and lints it. Open it — every field is optional and
commented; change `motif`, `ink.weight`, `page.size`, or the ornament's `symmetry` to taste. Validate
whenever you like:

```sh
plakat bookart lint border.hjson       # schema · vocabulary (nearest-match) · ranges · page
```

`lint` exits non-zero on errors, so it can gate CI, and suggests fixes — misspell `technique: linee`
and it points you at `line`.

## 2. See what it resolves to (no weights)

```sh
plakat bookart show border.hjson
```

This is the resolver's output — no rendering. You'll see the chosen **render tier** (a `border` defaults
to `procedural`), the **symmetry** (`bilateral`), the print **canvas** (`1748 × 2480 px @ 300 DPI`,
with mm and bleed), the **finisher** chain (transparency mode, binariser, ink, tint), and — for a
procedural ornament — `(procedural tier — no prompt)`. For a diffusion type it prints the compiled
prompt + negative instead.

## 3. Render a procedural border (no weights)

```sh
plakat bookart render border.hjson --out border.png
```

Instant, deterministic, no model. Out comes a publication-quality A5 frame — nested rules, an even
bead-and-reel run, four guilloché corner rosettes — crisp, exactly symmetric, and **transparent**
(drop it onto any page colour). The line you'll see:

```
wrote border.png  (1748 × 2480 px @ 300 DPI · procedural · border · bilateral · 1 piece(s))
```

Ask for the born-vector SVG too — procedural ornament is vector-native, so this is near-free and
infinite-DPI:

```sh
plakat bookart render border.hjson --out border.png --svg
#   ↳ born-vector SVG → border.svg
```

Try a few more geometric types with `new` + `render`: `fleuron` (a centred rosette), `divider` (a thin
rule), `corner` (placed four times). All zero-weight.

## 4. Render a diffusion vignette (weights)

Pictorial ornament comes from the diffusion tier. Scaffold a vignette and give it a prompt:

```sh
plakat bookart new bird.hjson --type vignette --origin russian --technique woodcut
#   edit bird.hjson → ornament: { type: "vignette", prompt: "a firebird among oak branches" }
plakat bookart render bird.hjson --out bird.png --steps 28 --seed 7
```

Because the origin is `russian`, the render resolves and attaches the hosted Bilibin LoRA
automatically (`vulogov98/plakat-bookart#russian-sd15.safetensors`) and weaves its trigger; `generic`
would use the LoRA-free line-art path instead (see [`../BOOKART_STYLES.md`](../BOOKART_STYLES.md)). The
raw sd15 render is binarised to clean line art and transparented — no grey slab, no halo.

Don't want to author a spec at all? `illustrate` is the diffusion tier as a one-liner:

```sh
plakat bookart illustrate "a wolf in a snowy pine forest" --origin japanese --out wolf.png
```

If a seed misbehaves, let the scorecard pick: `--attempts 4` renders up to four seeds and keeps the
first that clears chroma/alpha/ink (else the fewest-issues one).

## 5. Verify a render (no weights)

Measure any render against its spec:

```sh
plakat bookart verify bird.hjson --image bird.png --out bird-finished.png
```

It reports **chroma** (should be 0.000 — truly B/W), **alpha-halo** (0.000 — clean key), **symmetry
RMS**, **ink coverage**, and **resolution**. Symmetry is the finisher's one blind spot — if a piece
should be symmetric but isn't, fix it geometrically and place it on the page in one shot:

```sh
plakat bookart verify bird.hjson --image bird.png --symmetrize --page --out bird-page.png
```

## 6. Build a coherent kit (flagship, weights)

A kit is a matched *set* sharing one hand, one motif DNA, and one seed lineage. Author a spec with a
`kit` block instead of an `ornament`:

```hjson
{
  schema: "bookart/1"
  origin: "russian"
  technique: "woodcut"
  motif: ["firebird", "oak-leaf"]
  page: { size: "a5", dpi: 300 }
  ink: { transparency: "luminance" }
  kit: {
    seed: 42
    ornaments: [
      { type: "border" }
      { type: "divider" }
      { type: "fleuron" }
      { type: "corner", places: 4 }
      { type: "vignette", prompt: "the firebird in a winter forest" }
    ]
  }
}
```

```sh
plakat bookart kit firebird-kit.hjson --out kit/ --steps 28
```

Each ornament renders with the shared origin/technique/motif and its own derived seed. You get
`00_border.png … 04_vignette.png`, a **contact sheet**, a **`manifest.json`**, and a **CLIP
style-coherence** score (min/mean pairwise similarity — informational, since a kit legitimately spans
crisp procedural pieces and a pictorial vignette). Add `--svg` for born-vector SVG on the procedural
pieces; `--no-coherence` skips loading CLIP if you don't need the score.

## 7. Ornament a whole book (flagship, weights)

Point `manuscript` at a Markdown book (chapters = `#`/`##` headings) or a plain one-title-per-line
list, and supply a kit spec for the shared style:

```sh
plakat bookart manuscript book.md --kit firebird-kit.hjson --out ornaments/ --latex
```

It emits a **frontispiece** plus, per chapter, a **seed-varied headpiece** banner (a coherent
*variation* of the motif, not a clone) and a **tailpiece**, all in one hand — with a chapter→assets
`manifest.json`, a contact sheet, and (`--latex`) an `includes.tex` of `\newcommand`s you can `\input`
straight into your document. Preview any ornament directory as a sheet at any time:

```sh
plakat bookart proof ornaments/ --out proof.png
```

## 8. Edit and blend (mostly no weights)

Before you spend compute, ask what a change *costs*:

```sh
plakat bookart diff firebird-kit.hjson firebird-kit-v2.hjson
```

It classifies each changed field: `post` (tint/symmetry — a finished-PNG tweak), `re-raster` (page/size
— re-place the raster), or `re-gen` (origin/motif/prompt — a full re-render), and names the cheapest
sufficient action. The `post` class needs no GPU at all:

```sh
plakat bookart edit border.png --out border-sepia.png --tint sepia          # recolour ink, no render
plakat bookart edit bird.png   --out bird-sym.png    --symmetry bilateral   # re-apply symmetry
```

Finally, lineage — cross two traditions into a new spec (origin of A × technique of B, motifs unioned),
auto-linted:

```sh
plakat bookart blend firebird-kit.hjson wolf-japanese.hjson --out crossed.hjson
# → a russian × japanese-line spec with both motif sets
```

## Where to go next

- [`../BOOKART.md`](../BOOKART.md) — the command + schema reference, the ornament vocabulary, the tiers.
- [`../BOOKART_TRANSPARENCY.md`](../BOOKART_TRANSPARENCY.md) — why ink darkness *is* opacity, the
  binarisers, born-vector SVG, exact print sizing, the symmetry engine.
- [`../BOOKART_STYLES.md`](../BOOKART_STYLES.md) — the origin × technique system and the origin LoRAs.
