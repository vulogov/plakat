# `plakat paint` palettes

A **palette** is the small set of pigments (2–8) every colour in the painting is mixed from. Painting from ONE
limited palette is what buys **colour harmony for free** — the whole image shares a common genetic set, so it
reads as a painting, not a photo recolour. Pick the palette by the **light and mood** of the scene.

Set it with `--palette <name>` (CLI) or `palette: <name>` (HJSON). Names are case‑insensitive.

```bash
plakat paint from photo.png --medium watercolour --palette early-evening -o out.png
```

Colours mix **subtractively** (Kubelka–Munk), like real pigment — so a small warm+cool set already reaches a
wide, harmonious gamut. Every built‑in palette spans value (has a light and a dark) so it can carry a full image.

---

## Derive a palette from the image — `--palette image`

`--palette image` (alias `auto`) **derives** a bespoke palette from the reference's own dominant colours
(k‑means + white/black + a saturation safety‑net). This is the best choice for **photos and portraits** — the
painting keeps the subject's real colours instead of forcing them through a fixed set.

```bash
plakat paint from photo.png --medium watercolour --palette image -o out.png
```

The derived pigments are **written into the stroke score**, so a derived‑palette painting now **replays**
(`plakat paint replay out.strokes`) like any other — the score is self‑contained and needs no external palette
definition. (Every score carries its pigment definitions as `P <name> <r> <g> <b>` lines.)

---

## Classic painter's palettes

| Name | Pigments | Use it for |
|------|----------|-----------|
| `zorn` | yellow‑ochre · cadmium‑red · ivory‑black · titanium‑white | Portraits & figures — a whole warm‑skinned figure from four. |
| `split-primary` | cadmium/lemon yellow · cadmium‑red · quinacridone‑rose · ultramarine · phthalo‑blue · white | Full gamut — a warm+cool of each primary reaches most colours. |
| `verdaccio` | yellow‑ochre · ivory‑black · terre‑verte · white | Tempera / green‑grey underpainting. |
| `earth` | raw‑umber · burnt‑sienna · yellow‑ochre · ivory‑black · white | Muted earth scenes, rustic subjects. |
| `limited-landscape` | ultramarine · burnt‑sienna · yellow‑ochre · cadmium‑yellow · white | Plein‑air landscape — sky, earth, and their greys. |
| `sumi` | ivory‑black · white | Ink wash, grisaille, monochrome. |

---

## Scene & mood presets

Pick by **time of day**, **weather**, **biome**, or **overall saturation**. Each is a curated harmony spanning a
light and a dark.

### Time of day
| Name | Character | Pigments |
|------|-----------|----------|
| `early-morning` | Cool, soft, pale first light with a warm touch | sky‑blue · lavender · rose‑madder · cerulean · yellow‑ochre · payne's‑grey · white |
| `bright-day` | Clear high‑key daylight | cerulean · ultramarine · sap‑green · cadmium‑yellow · cadmium‑red · burnt‑sienna · white |
| `early-evening` | Golden hour — warm gold/orange over cool shadow | gold‑ochre · cadmium‑orange · cadmium‑red · quinacridone‑rose · ultramarine · raw‑umber · white |
| `late-evening` | Dim warm dusk sinking into purple | dioxazine‑purple · alizarin · burnt‑sienna · indigo · gold‑ochre · ivory‑black · white |
| `night` | Deep cool dark, sparse light | indigo · ultramarine · payne's‑grey · phthalo‑blue · dioxazine‑purple · ivory‑black · white |

### Weather
| Name | Character | Pigments |
|------|-----------|----------|
| `rain` | Desaturated cool greys, muted blue/green | payne's‑grey · slate · cerulean · terre‑verte · raw‑umber · ivory‑black · white |
| `storm` | Dark, dramatic, bruised | indigo · payne's‑grey · deep‑green · burnt‑sienna · dioxazine‑purple · ivory‑black · white |
| `snow` | Cool whites with blue shadow | white · cerulean · payne's‑grey · lavender · sky‑blue · slate · ivory‑black |

### Biome
| Name | Character | Pigments |
|------|-----------|----------|
| `desert` | Warm sand & ochre under a clean sky | sand · gold‑ochre · burnt‑sienna · cadmium‑orange · cerulean · raw‑umber · white |
| `forest` | Earthy greens & browns | sap‑green · olive‑green · terre‑verte · burnt‑sienna · raw‑umber · yellow‑ochre · white |
| `rainforest` | Saturated humid greens, deep shadow | viridian · sap‑green · deep‑green · teal · gold‑ochre · raw‑umber · white |

### Saturation register
| Name | Character | Pigments |
|------|-----------|----------|
| `vivid-colors` | Saturated primaries & secondaries, full strength | cadmium‑yellow/orange/red · quinacridone‑rose · ultramarine · phthalo‑blue · viridian · white |
| `rich-colors` | Deep jewel tones | alizarin · dioxazine‑purple · phthalo‑blue · viridian · burnt‑sienna · gold‑ochre · ivory‑black · white |
| `muted-colors` | Greyed, desaturated harmony | yellow‑ochre · terre‑verte · raw‑umber · payne's‑grey · rose‑madder · slate · white |

---

## Inspect a palette

```bash
plakat paint palette                 # list every built-in palette
plakat paint palette early-evening   # show a palette's pigments (name + colour)
```

## Notes

- If you name a `--medium` but no palette, paint **derives** one from the image (a fixed medium palette rarely
  fits an arbitrary photo). Name a palette to override, or `--palette image` to be explicit.
- Palettes are medium‑agnostic — `sumi` in oil is a grisaille; `vivid-colors` in watercolour is a bright wash.
- For the loose‑watercolour look and portrait recipe, see [`PAINT_WATERCOLOR.md`](PAINT_WATERCOLOR.md); for the
  full control list, [`PAINT_CONTROLS.md`](PAINT_CONTROLS.md).
