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
infinite-DPI (and compact — the paths are simplified sub-pixel):

```sh
plakat bookart render border.hjson --out border.png --svg
#   ↳ born-vector SVG → border.svg
```

Try the rest of the geometric repertoire with `new` + `render` — all zero-weight, all crisp:
`fleuron` (a guilloché rosette), `divider` (a rule with a running guilloché wave + fleuron ends),
`corner` (a bold L placed four times), `headpiece` (a band — rules + central medallion +
interweaving braid), `tailpiece` (a cul-de-lampe tapering to a point). Change `--seed` to *diversify* a
piece — the same type renders as a sibling, not a copy, which is what keeps a kit or a book coherent
without being repetitive.

## 4. Render a diffusion vignette (weights)

Pictorial ornament comes from the diffusion tier. Scaffold a vignette and give it a prompt:

```sh
plakat bookart new bird.hjson --type vignette --origin russian --technique line
#   edit bird.hjson → ornament: { type: "vignette", prompt: "a firebird among oak branches" }
plakat bookart render bird.hjson --out bird.png --steps 28 --seed 7
```

Because the origin is `russian`, the render resolves and attaches the hosted Bilibin LoRA
automatically (`vulogov98/plakat-bookart#russian-sd15.safetensors`) and weaves its trigger; `generic`
would use the LoRA-free line-art path instead (see [`../BOOKART_STYLES.md`](../BOOKART_STYLES.md)). The
raw sd15 render is binarised and transparented — no grey slab, no halo.

**Pick the idiom on purpose.** `technique: line` (the default, and Bilibin's actual pen idiom) extracts
clean delicate outlines — the airy, refined book-illustration look. `technique: woodcut` keeps bold
black masses (authentic lubok, but heavy — better for a full-bleed plate than a small spot). If a
diffusion piece comes out too dark, switch it to `line`.

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
  technique: "line"
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

It emits a **frontispiece** (the pictorial plate, diffusion) plus, per chapter, a **procedural
headpiece band** (a заставка — rules, a central medallion, an interweaving guilloché braid, fleuron
ends) and a **procedural tailpiece** (a cul-de-lampe tapering to a point). The per-chapter seed diversifies
the bands so they read as *kin, not clones* — a denser braid here, a different scroll count there —
while staying in one hand. You get a chapter→assets `manifest.json`, a contact sheet, and (`--latex`) an
`includes.tex` of `\newcommand`s you can `\input` straight into your document. (Headpieces/tailpieces
are procedural — clean, airy, weight-free line ornament; the motif lives in the frontispiece plate.)
Preview any ornament directory as a sheet at any time:

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

**Ink weight and transparency, without re-rendering.** Render once with `--cache-raw`, then re-finish
from the cache — the (expensive) diffusion sampling is skipped:

```sh
plakat bookart illustrate "a firebird among oak branches" --origin russian --out bird.png --cache-raw
plakat bookart edit bird.png --out bird-bold.png  --ink-weight 0.85          # heavier ink, no render
plakat bookart edit bird.png --out bird-soft.png  --transparency threshold   # different alpha curve
```

Finally, lineage — cross two traditions into a new spec (origin of A × technique of B, motifs unioned),
auto-linted:

```sh
plakat bookart blend firebird-kit.hjson wolf-japanese.hjson --out crossed.hjson
# → a russian × japanese-line spec with both motif sets
```

## 9. The rest of the toolbox (6.1)

```sh
plakat bookart origins --details              # the six origins × techniques × ornaments + LoRA hosting
```

Six origins ship trained LoRAs — russian / english / japanese (Bilibin / Beardsley / Hokusai) and
american / european / chinese (Pyle / Doré / woodblock). `bookart illustrate "…" --origin european`
resolves `european-sd15.safetensors` from HuggingFace automatically. Add your own tradition with an
`assets/bookart/lexicon.hjson`.

```sh
# a spec with  ornament: { type: "initial", glyph: "A" }  →  render frames the real letterform:
plakat bookart render initial.hjson --out A.png --font /Library/Fonts/Georgia.ttf   # historiated initial
plakat bookart font --out dingbats.otf                                # an OpenType dingbat font (type a–h)
plakat bookart vectorize bird.png --out bird.svg                      # raster→SVG trace (feature: bookart-trace)
plakat bookart manuscript book.epub --kit style.hjson --out orn/      # a whole EPUB's per-chapter set (feature: epub)
plakat bookart render border.hjson --out b.png --import ~/albums/ornaments   # land it in a plakat photos album
```

These are the 6.1 additions: glyph-driven **initials** (a real letterform, any script, framed), an
OpenType **dingbat font**, raster→SVG **tracing**, **EPUB** manuscripts, and `--import` into a `plakat
photos` album with the render's recipe. `bookart-trace` and `epub` are opt-in Cargo features (see
[`../BOOKART.md`](../BOOKART.md)); the rest work in the prebuilt binaries.

## Where to go next

- [`../BOOKART.md`](../BOOKART.md) — the command + schema reference, the ornament vocabulary, the tiers.
- [`../BOOKART_TRANSPARENCY.md`](../BOOKART_TRANSPARENCY.md) — why ink darkness *is* opacity, the
  binarisers, born-vector SVG, exact print sizing, the symmetry engine.
- [`../BOOKART_STYLES.md`](../BOOKART_STYLES.md) — the origin × technique system and the origin LoRAs.
