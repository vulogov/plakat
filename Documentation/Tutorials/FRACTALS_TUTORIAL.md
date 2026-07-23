# Fractals — render them, then paint scenes from their structure

`plakat fractals` does two things:

1. **Track A — a pure-CPU fractal renderer.** Deterministic, offline, no model, no
   GPU. Mandelbrot & friends, IFS, L-systems, flames, strange attractors, and 3D
   raymarched fractals. *Same spec → byte-identical pixels.*
2. **Track B — an optional AI paint pass.** Feed the fractal's *structure* into
   Stable Diffusion (ControlNet) and generate a real image — a seascape, a
   landscape, a creature — **shaped by the fractal**.

This tutorial covers both, with lots of copy-paste examples. It assumes you built
plakat with the `fractals` feature (it's in the default build). For GPU painting,
add a backend: `cargo build --release --features metal,fractals` (macOS) or
`--features cuda,fractals` (NVIDIA).

> Everything in Track A is instant and needs nothing downloaded. Track B needs a
> model (SDXL by default, ~7 GB, auto-downloaded once) and is far faster on a GPU.

---

## Quick start

```bash
# 1. A classic Mandelbrot (Track A only)
plakat fractals --fractal-out out/mandel.png

# 2. A deep-zoomed, ice-blue Julia set
plakat fractals --fractal-kind julia --fractal-julia-c=-0.8,0.156 \
  --fractal-palette ice --fractal-zoom 1.4 --fractal-out out/julia.png

# 3. Generate a golden-hour seascape SHAPED BY a fractal (Track B, needs a GPU)
plakat fractals --fractal-center=-0.745,0.113 --fractal-zoom 60 --fractal-palette ice \
  --fractal-paint --fractal-prompt "wide seascape at golden hour, sun on the horizon, ocean waves"
```

Example 3 writes **two** files: `out/fractal.png` (the raw fractal) and
`out/fractal.painted.png` (the AI scene). More on that in Part 7.

---

# Part 1 — Track A: rendering fractals

## 1.1 The families

Pick a family with `--fractal-kind`. Here's one example per family.

### Escape-time (the complex-plane classics)

```bash
# Mandelbrot (the default)
plakat fractals --fractal-kind mandelbrot --fractal-out out/mandelbrot.png

# Julia — set the constant c
plakat fractals --fractal-kind julia --fractal-julia-c=-0.7,0.27 --fractal-out out/julia.png

# Burning Ship — jagged, fiery
plakat fractals --fractal-kind burning-ship --fractal-center=-1.75,-0.03 --fractal-zoom 40 \
  --fractal-iter 800 --fractal-palette electric --fractal-out out/ship.png

# Tricorn (Mandelbar) — conjugate symmetry
plakat fractals --fractal-kind tricorn --fractal-center=0,0 --fractal-zoom 0.7 --fractal-out out/tricorn.png

# Multibrot — Mandelbrot with a higher power
plakat fractals --fractal-kind multibrot --fractal-power 5 --fractal-center=0,0 --fractal-out out/multibrot.png

# Newton — root basins (great with angle coloring)
plakat fractals --fractal-kind newton --fractal-power 3 --fractal-center=0,0 --fractal-zoom 0.5 \
  --fractal-coloring angle --fractal-palette neon --fractal-out out/newton.png

# Nova, Phoenix, Magnet, Sine, Exp — more exotic escape maps
plakat fractals --fractal-kind phoenix --fractal-center=0,0 --fractal-zoom 0.6 --fractal-out out/phoenix.png
plakat fractals --fractal-kind sine   --fractal-center=0,0 --fractal-zoom 0.3 --fractal-out out/sine.png
```

The full escape-time list: `mandelbrot`, `julia`, `burning-ship`, `tricorn`,
`multibrot`, `newton`, `nova`, `phoenix`, `magnet`, `sine`, `exp`.

### Buddhabrot (density of escaping orbits)

```bash
plakat fractals --fractal-kind buddhabrot --fractal-size 1024x1024 \
  --fractal-buddha-samples 20000000 --fractal-seed 1 --fractal-palette midnight \
  --fractal-out out/buddhabrot.png
```

Stochastic but **deterministic per seed**. More samples = smoother (and slower).

### IFS — chaos-game attractors

```bash
plakat fractals --fractal-kind ifs --fractal-ifs-preset barnsley-fern \
  --fractal-size 800x1000 --fractal-palette earth --fractal-out out/fern.png
```

Presets: `barnsley-fern`, `sierpinski`, `dragon`, `levy`, `tree`, `spiral`.

### L-systems — turtle line drawings

```bash
plakat fractals --fractal-kind lsystem --fractal-lsystem-preset koch-snowflake \
  --fractal-supersample 2 --fractal-palette ice --fractal-out out/snowflake.png

plakat fractals --fractal-kind lsystem --fractal-lsystem-preset plant \
  --fractal-supersample 2 --fractal-palette earth --fractal-out out/plant.png
```

Presets: `koch`, `koch-snowflake`, `sierpinski`, `dragon`, `hilbert`, `gosper`,
`plant`, `bush`. Tune with `--fractal-lsystem-angle` and `--fractal-lsystem-depth`.

### Fractal flames (glowing, painterly)

```bash
plakat fractals --fractal-kind flame --fractal-flame-preset flame \
  --fractal-palette neon --fractal-out out/flame.png

# Kaleidoscopic symmetry
plakat fractals --fractal-kind flame --fractal-flame-preset swirl --fractal-flame-symmetry 6 \
  --fractal-palette electric --fractal-out out/flame_mandala.png
```

Presets: `sierpinski`, `spherical`, `swirl`, `spiral`, `flame`.

### Strange attractors (chaotic trajectories)

```bash
plakat fractals --fractal-kind attractor --fractal-attractor-preset clifford \
  --fractal-palette ice --fractal-out out/clifford.png

plakat fractals --fractal-kind attractor --fractal-attractor-preset lorenz \
  --fractal-palette fire --fractal-out out/lorenz.png
```

Presets: `clifford`, `dejong`, `bedhead`, `duffing`, `ikeda`, `lorenz`, `rossler`.

### 3D raymarched fractals

```bash
plakat fractals --fractal-kind raymarch --fractal-raymarch-shape mandelbulb \
  --fractal-raymarch-yaw 40 --fractal-raymarch-pitch 22 --fractal-palette fire \
  --fractal-size 800x800 --fractal-out out/mandelbulb.png
```

Shapes: `mandelbulb`, `mandelbox`, `menger`, `sierpinski3d`, `quat-julia`. Orbit the
camera with `--fractal-raymarch-yaw` / `-pitch` / `-dist`. *(Raymarching is
compute-heavy — use a release build.)*

## 1.2 Framing: center, zoom, size, iterations

```bash
plakat fractals \
  --fractal-center=-0.745,0.113 \   # complex-plane center (RE,IM) — note the '=' for negatives
  --fractal-zoom 250 \              # higher = deeper zoom
  --fractal-iter 1500 \            # more iterations = more boundary detail when deep
  --fractal-size 1600x1000 \       # output pixels (WxH)
  --fractal-out out/seahorse.png
```

> **Negative coordinates:** use the `=` form — `--fractal-center=-0.745,0.113`. The
> space form `--fractal-center -0.745,0.113` also works now, but `=` is unambiguous.

Deeper zooms need more iterations, or the fine structure disappears into solid color.

## 1.3 Coloring modes

`--fractal-coloring` (escape-time families):

| Mode | Look | Example |
|------|------|---------|
| `smooth` | continuous gradient (default) | `--fractal-coloring smooth` |
| `histogram` | even color spread across the frame | `--fractal-coloring histogram` |
| `distance` | thin, evenly-lit filaments | `--fractal-coloring distance` |
| `orbit-trap` | color by closest approach to a shape | `--fractal-coloring orbit-trap --fractal-trap-shape circle` |
| `angle` | final-iterate angle (Newton basins) | `--fractal-coloring angle` |
| `stripe` | flame-like angular bands | `--fractal-coloring stripe --fractal-stripe-freq 8` |
| `image` | sample a photo at the orbit (Part 5) | `--fractal-trap-image photo.jpg` |

```bash
plakat fractals --fractal-coloring stripe --fractal-stripe-freq 8 \
  --fractal-palette pastel --fractal-out out/stripe.png

plakat fractals --fractal-coloring distance --fractal-palette midnight \
  --fractal-out out/filaments.png
```

## 1.4 Palettes

`--fractal-palette`: `fire`, `ice`, `electric`, `neon`, `pastel`, `monochrome`,
`midnight`, `earth`. All are **Lab-space** gradients (perceptually smooth — no muddy
grey mid-tones).

Custom stops and interior color:

```bash
plakat fractals --fractal-stops "#000010,#0050a0,#40d0ff,#ffffff" \
  --fractal-out out/custom.png
# interior (non-escaping) color is part of the spec's palette; edit via a spec file.
```

## 1.5 Anti-aliasing (supersampling)

```bash
plakat fractals --fractal-supersample 3 --fractal-out out/smooth_edges.png
```

Renders at N× per axis then downsamples (1–8). Great for line families (L-systems)
and crisp boundaries; 3× is a good default for final art.

## 1.6 Per-family knob cheat-sheet

| Family | Key flags |
|--------|-----------|
| escape-time | `--fractal-center`, `--fractal-zoom`, `--fractal-iter`, `--fractal-julia-c`, `--fractal-power`, `--fractal-coloring` |
| buddhabrot | `--fractal-buddha-samples`, `--fractal-seed` |
| ifs | `--fractal-ifs-preset`, `--fractal-ifs-iterations` |
| lsystem | `--fractal-lsystem-preset`, `--fractal-lsystem-angle`, `--fractal-lsystem-depth` |
| flame | `--fractal-flame-preset`, `--fractal-flame-symmetry`, `--fractal-flame-iterations` |
| attractor | `--fractal-attractor-preset`, `--fractal-attractor-iterations` |
| raymarch | `--fractal-raymarch-shape`, `--fractal-raymarch-power`, `--fractal-raymarch-yaw/-pitch/-dist` |

---

# Part 2 — The spec: reproducibility & sharing

Every render is described by a **FractalSpec**. You can print it, save it, and
recover it from any PNG plakat wrote.

```bash
# Print the fully-resolved spec (renders nothing)
plakat fractals --fractal-kind julia --fractal-zoom 3 --fractal-dump-spec

# Save it, edit it, re-run it (HJSON — comments + unquoted keys allowed)
plakat fractals --fractal-kind flame --fractal-dump-spec > my_flame.hjson
# ...edit my_flame.hjson...
plakat fractals --fractal-spec my_flame.hjson --fractal-out out/edited.png

# Every PNG embeds its spec — reconstruct the exact image from the file
plakat fractals --fractal-clone out/edited.png --fractal-out out/copy.png
```

CLI flags always override the base (clone → spec-file → prose → default), so you can
load a spec and tweak one field: `--fractal-spec my.hjson --fractal-zoom 500`.

---

# Part 3 — Prose → spec (`--fractal-from`)

> **Important:** `--fractal-from` shapes the **fractal itself** from *fractal keywords*.
> It is **not** a scene generator. If you want a picture of "a winding forest path,"
> that description belongs in `--fractal-prompt` with `--fractal-paint` (Part 7).

`--fractal-from` reads fractal keywords from your text and builds a starting spec
(offline, deterministic — no LLM). CLI flags still override.

```bash
# See what it decided (renders nothing):
plakat fractals --fractal-from "an icy julia with stripes" --fractal-dump-spec
#   → kind=julia, palette=ice, coloring=stripe

plakat fractals --fractal-from "a cosmic nebula" --fractal-out out/nebula.png
#   → kind=buddhabrot, palette=midnight

plakat fractals --fractal-from "a fiery burning ship, deep zoom, intricate" \
  --fractal-out out/from_prose.png
#   → kind=burning-ship, palette=fire, zoom×40, high iterations
```

It understands **families** (mandelbrot, julia, burning ship, newton, flame, fern,
plant, dragon, koch, lorenz, clifford, mandelbulb, …), **moods → palettes** (fiery,
icy, neon, pastel, cosmic, earthy…), **coloring** (stripes, filaments), **symmetry**
(kaleidoscope), and **depth** (deep zoom, intricate).

**Smarter mapping with an LLM (optional).** Add `--fractal-provider` to have a language
model map an arbitrary description to a spec — it falls back to the offline keyword mapper
on any failure, so it's always safe:

```bash
plakat fractals --fractal-from "a stormy alien coastline at dusk" \
  --fractal-provider auto --fractal-out out/llm.png
#   auto | deepseek | gemini | local | local:<alias>
```

**Any text that names no family** still gives you something distinctive — plakat hashes
the words into a unique Julia set, so every phrase yields different art (never the same
default twice), and any mood word still steers the palette:

```bash
plakat fractals --fractal-from "winding path in the forest" --fractal-out out/forest.png
#   → a hash-derived Julia set in the 'earth' palette (from "forest"); a *different*
#     Julia for every phrase. It also prints a hint reminding you that scenes go in
#     --fractal-prompt.
```

### Prose → fractal → painted scene

The natural combo: use `--fractal-from` to pick a fractal *vibe*, and `--fractal-prompt`
to paint the *scene* on top of it:

```bash
plakat fractals --fractal-from "icy julia, intricate" \
  --fractal-paint --fractal-prompt "a frozen alien landscape under aurora, cinematic"
```

---

# Part 3a — Batch rendering with `plakat scenario`

A scenario can batch-render fractals with `type: fractal` tasks — single fractals, compose grids,
animations, or AI-painted — alongside `generate` / `map` tasks. **Quote keys and values inside the
inline `fractal: { … }` block** (HJSON is strict about hyphenated keys):

```hjson
{
  out: "out/fractals"
  seed: 100
  tasks: [
    { name: "mandel", type: "fractal", fractal: { kind: "mandelbrot", size: "800x800", palette: "fire" } }
    { name: "fern",   type: "fractal", fractal: { kind: "ifs", "ifs-preset": "barnsley-fern", palette: "earth" } }
    { name: "sweep",  type: "fractal", fractal: { compose: "julia-sweep", grid: "4x4", size: "1200x1200" } }
    { name: "scene",  type: "fractal", fractal: { kind: "julia", center: "0,0", zoom: 40,
        paint: true, prompt: "a frozen alien landscape, aurora" } }
  ]
}
```

```bash
plakat scenario my_fractals.hjson
#   → out/fractals/mandel/fractal.png, .../fern/fractal.png, .../sweep/fractal.png,
#     .../scene/fractal.painted.png
```

Each task's `fractal:` block accepts the same options as the CLI flags (without the `--fractal-`
prefix): `kind`, `center`, `zoom`, `iter`, `size`, `palette`, `coloring`, the `*-preset` /
`raymarch-shape` selectors, `compose`/`grid`, `animate`/`frames`/`fps`, and paint (`paint`, `prompt`,
`paint-mode`, `sd-model`, `sd-strength`, `sd-control-strength`). `seed` defaults to the scenario seed.

# Part 3b — Bund scripting (`plakat.fractal.*`)

From a Bund script (`plakat run`), five words render or paint a fractal straight into an image
**handle** so it flows into the rest of the pipeline (`plakat.save`, `plakat.upscale`,
`plakat.relight`, `plakat.metadata.write`) — the same handle plumbing generated images use:

```text
plakat.fractal.size    ( w h -- )                 output-size override for the words below
plakat.fractal.render  ( src -- h )               Track-A CPU render → handle (no GPU)
plakat.fractal.compose ( src mode rows cols -- h ) grid contact sheet → handle
plakat.fractal.paint   ( src -- h )               AI paint (txt2img + ControlNet) → handle
plakat.fractal.animate ( src mode frames fps out -- out )  zoom / sweep to a video / GIF file
```

`src` is the same **spec source** `--control-fractal` accepts: a spec file, a `kind` /
`kind:preset` shorthand (`"flame"`, `"ifs:barnsley-fern"`, `"raymarch:menger"`), or prose.

```bund
"512" "512" plakat.fractal.size

// Pure-CPU render → ML upscale → save.
"ifs:barnsley-fern" plakat.fractal.render   // handle 1
1 "real-esrgan-x4" plakat.upscale           // handle 2
"fern-4k.png" plakat.save

// A 2×2 palette contact sheet.
"mandelbrot" "palette-grid" "2" "2" plakat.fractal.compose
"sheet.png" plakat.save

// A zoom GIF (no ffmpeg needed for .gif).
"mandelbrot" "zoom" "48" "24" "zoom.gif" plakat.fractal.animate
```

See `SCRIPTING_TUTORIAL.md` §14 for the full scripting reference.

# Part 4 — Composition grids (`--fractal-compose`)

Make a "contact sheet" of related fractals in one image.

```bash
# 16 Julia sets sweeping c around a circle
plakat fractals --fractal-compose julia-sweep --fractal-grid 4x4 \
  --fractal-size 1200x1200 --fractal-palette electric --fractal-out out/julia_sweep.png

# Progressive zoom into one point (self-similarity)
plakat fractals --fractal-compose zoom-grid --fractal-grid 3x3 \
  --fractal-center=-0.745,0.113 --fractal-palette fire --fractal-out out/zoom_grid.png

# The same fractal through every palette
plakat fractals --fractal-compose palette-grid --fractal-grid 2x4 --fractal-out out/palettes.png

# Seed/parameter variations (great for flame / attractor)
plakat fractals --fractal-kind flame --fractal-compose variation-sweep --fractal-grid 3x3 \
  --fractal-out out/flame_variations.png
```

Modes: `julia-sweep`, `zoom-grid`, `palette-grid`, `variation-sweep`.

## Part 4a — Keep the best cells (`--fractal-keep-best`)

A sweep makes a contact sheet, but which cells are actually good? Add
`--fractal-keep-best K` and every cell is scored by the LAION aesthetic predictor
(the same model behind `plakat rank`). The top-K cells are **highlighted in gold** in
the grid *and* written out individually as `<out>_best-1.png`, `<out>_best-2.png`, …
(each with its own embedded spec, so you can clone or re-render it):

```bash
plakat fractals --fractal-compose julia-sweep --fractal-grid 4x4 \
  --fractal-size 1600x1600 --fractal-keep-best 3 --fractal-out out/sweep.png
#   → out/sweep.png              (grid, top-3 cells framed in gold)
#     out/sweep_best-1.png …-3   (the three highest-scoring Julia sets, standalone)
#   --fractal-keep-best: scored 16 cells, kept top 3:
#     #1   6.021  out/sweep_best-1.png
#     #2   5.874  out/sweep_best-2.png
#     #3   5.610  out/sweep_best-3.png
```

Loads a small scoring model on first use (cached thereafter). Runs on whatever
`--device` resolves to; CPU is fine — scoring a cell is cheap next to rendering it.

---

# Part 5 — Color a fractal with a photo (`--fractal-trap-image`)

The **image orbit-trap**: instead of a gradient, sample a photograph at each orbit's
closest approach. A Julia set textured by your photo.

```bash
plakat fractals --fractal-kind julia --fractal-julia-c=-0.8,0.156 \
  --fractal-trap-image my_photo.jpg --fractal-size 800x800 --fractal-out out/photo_julia.png
```

Passing `--fractal-trap-image` sets `--fractal-coloring image` automatically. Tune the
sampling window with `--fractal-trap-point RE,IM`. Works best on Julia / Mandelbrot.

## Part 5a — From inside `plakat photos`

The same image orbit-trap is one keystroke away in the photo manager. In the image
view, `Ctrl-B n f` (**fractalize**) renders a Julia set textured by the current photo
and drops a new `*_fractal.png` beside it — pure-CPU, no model. Because the output
embeds its `FractalSpec`, the info panel (`i` / `I`) then shows a **fractal** section
for it (kind, framing, palette, and the AI-paint recipe when present). Every
plakat-made fractal PNG gets that panel, so a folder of renders is self-describing.

---

# Part 6 — Explore interactively (`--fractal-explore`)

A live TUI: pan, zoom, and retune with instant preview (needs a graphics terminal —
Kitty, iTerm2, WezTerm, or a Sixel terminal).

```bash
plakat fractals --fractal-kind mandelbrot --fractal-explore
```

| Key | Action |
|-----|--------|
| arrows / `hjkl` | pan |
| `+` / `-` | zoom in / out |
| `[` / `]` | fewer / more iterations |
| `p` | cycle palette |
| `c` | cycle coloring |
| `n` / `N` | next / previous family |
| `r` | reset view |
| `s` | **save** current view (full-res, to `--fractal-out`) |
| `q` / `Esc` | quit |

---

# Part 7 — Generating images FROM fractal structure (Track B)

This is the headline feature: turn a fractal into a real picture whose composition is
driven by the fractal's geometry.

## 7.1 The idea

The fractal's structure is turned into a **ControlNet map** (edges via Canny for 2D
families; a real **depth** map for the 3D raymarched ones), and Stable Diffusion
generates an image guided by that map. There are two ways to do it:

| Mode | The fractal is… | Result | Use when |
|------|-----------------|--------|----------|
| **`txt2img`** (default) | ControlNet **only** (no init image) | a real scene — sky, horizon, lighting from the prompt — *shaped by* the fractal's contours | you want a **believable scene** (landscape, seascape, creature) |
| **`img2img`** | the init image **+** ControlNet | a scene *made of* the fractal — keeps its colors & layout, more abstract | you want the **fractal's own colors/shape** to dominate |

Switch with `--fractal-paint-mode {txt2img|img2img}`. **txt2img is the default** because
it produces real scenes; img2img gives a more abstract, fractal-forward look.

## 7.2 Turn it on

```bash
plakat fractals \
  --fractal-center=-0.745,0.113 --fractal-zoom 60 --fractal-palette ice --fractal-size 768x768 \
  --fractal-paint \
  --fractal-prompt "wide seascape at golden hour, sun low on the horizon, orange and pink sky, ocean waves"
```

Output: `out/fractal.png` (the fractal) + `out/fractal.painted.png` (the scene). Add
`--fractal-paint-out path.png` to choose the painted file's name.

You'll see which device it uses:

```
painting via sdxl (txt2img, control: canny 0.4)…
painting on Metal GPU…
```

## 7.3 The one rule that matters most: **frame detail**

A whole Mandelbrot is ~90% featureless black interior — there's nothing for the model
to build on, and you'll get a dark, empty result. **Zoom onto the boundary filigree**
(as above) so the fractal has rich structure. Good regions to try:

```bash
--fractal-center=-0.745,0.113 --fractal-zoom 60      # seahorse valley
--fractal-center=-0.16,1.035  --fractal-zoom 40      # antenna
--fractal-center=0.285,0.535  --fractal-zoom 80      # spirals
```

Or use a family that's structure-rich everywhere: `julia`, `flame`, `ifs`, and the 3D
`raymarch` shapes.

## 7.4 The master dial: `--fractal-sd-control-strength`

This governs **how much the fractal drives the composition**:

| Control | Effect |
|---------|--------|
| `0.3–0.4` (default) | a free, believable scene; the fractal is a gentle influence |
| `0.5–0.7` | the fractal's structure clearly drives the layout — zoom/shape changes become obvious |
| `0.8–1.0` | contours snap hard to the fractal (can look abstract / top-down again) |

At the loose default, two different fractals can paint to similar scenes (the prompt
dominates). Raise control and the *specific* fractal structure takes over:

```bash
# Fractal structure strongly drives the scene:
plakat fractals --fractal-center=-0.745,0.113 --fractal-zoom 60 --fractal-palette ice \
  --fractal-paint --fractal-sd-control-strength 0.7 \
  --fractal-prompt "seascape at golden hour, ocean waves, dramatic sky"
```

## 7.5 More knobs

| Flag | Meaning | Default |
|------|---------|---------|
| `--fractal-prompt` | what to paint | per-family auto prompt |
| `--fractal-negative` | what to avoid | a sensible default |
| `--fractal-sd-model` | model alias | `sdxl` |
| `--fractal-sd-control-strength` | fractal influence (§7.4) | 0.4 (txt2img) / 0.55 (img2img) |
| `--fractal-sd-strength` | img2img only: how far from the fractal | 0.78 |
| `--fractal-sd-guidance` | prompt adherence (CFG) | 6.5 |
| `--fractal-sd-steps` | diffusion steps | 28 |
| `--fractal-sd-control` | override control type (canny/lineart/softedge/depth) | per-family |
| `--fractal-sd-lora` | add a LoRA (HF `org/name[:scale]`, `civitai:ID`, or a path) | — |

A strong negative prompt helps a lot — push away failure modes:

```bash
--fractal-negative "aerial view, top-down, map, blurry, dark, low quality"
```

## 7.6 img2img — the "made of the fractal" look

When you *want* the fractal's own colors and shapes:

```bash
plakat fractals --fractal-center=-0.745,0.113 --fractal-zoom 60 --fractal-palette fire \
  --fractal-paint --fractal-paint-mode img2img --fractal-sd-strength 0.6 \
  --fractal-prompt "molten lava, glowing embers, volcanic"
```

- Lower `--fractal-sd-strength` (0.4–0.6) keeps it recognizably the fractal.
- Higher (0.8+) reimagines more freely (but a dark fractal stays dark — that's the
  reason txt2img is the default).

## 7.6b The inverse: a fractal as ControlNet for `plakat generate`

Track B paints *from* the fractals command. The **inverse** lives on `plakat generate`:
`--control-fractal` renders a fractal and uses its structure as ControlNet conditioning for
any prompt — the fractal shapes a normal generation.

```bash
plakat generate "a stained glass window, intricate" --model sd15 --size 768x768 \
  --control-fractal "julia" --control-strength 0.9
#   → a stained-glass window whose leading follows the Julia set's boundary
```

`--control-fractal` accepts a fractal spec file, a `kind[:preset]` shorthand
(`flame`, `ifs:barnsley-fern`, `raymarch:menger`), or prose. The control type is auto per
family (canny / lineart / depth) unless you set `--control`; combine with the usual
`--control-strength`.

## 7.7 3D fractals + depth control

For raymarched fractals, txt2img uses a **depth** ControlNet — the fractal's actual 3D
form guides the scene, which is the strongest structural signal of all:

```bash
plakat fractals --fractal-kind raymarch --fractal-raymarch-shape mandelbulb \
  --fractal-size 768x768 --fractal-paint \
  --fractal-prompt "an ancient alien temple carved from black stone, cinematic, volumetric light"
```

## 7.8 Worked example: dialing in a seascape

1. **Find structure.** Render Track A first and look at `out/fractal.png`:
   ```bash
   plakat fractals --fractal-center=-0.745,0.113 --fractal-zoom 60 --fractal-palette ice --fractal-out out/base.png
   ```
2. **Paint it (default txt2img).**
   ```bash
   plakat fractals --fractal-center=-0.745,0.113 --fractal-zoom 60 --fractal-palette ice \
     --fractal-paint --fractal-prompt "wide seascape at golden hour, sun on the horizon, waves, orange sky" \
     --fractal-negative "aerial, top-down, map, dark, blurry"
   ```
   → a golden-hour seascape whose waves follow the fractal boundary.
3. **Want the fractal to show more?** Raise control to 0.6–0.7.
4. **Want a different structure?** Change the region/zoom, or switch to
   `--fractal-kind julia`/`flame` — the composition follows.

---

# Reference — every flag

Render `plakat fractals --help` for the authoritative list. The essentials:

**Base / output**
`--fractal-from TEXT`, `--fractal-spec FILE`, `--fractal-clone PNG`,
`--fractal-out PATH`, `--fractal-dump-spec`, `--fractal-explore`

**Fractal**
`--fractal-kind`, `--fractal-center=RE,IM`, `--fractal-zoom`, `--fractal-iter`,
`--fractal-julia-c=RE,IM`, `--fractal-power`, `--fractal-size WxH`, `--fractal-seed`

**Coloring / palette**
`--fractal-coloring`, `--fractal-palette`, `--fractal-stops`, `--fractal-supersample`,
`--fractal-trap-shape`, `--fractal-trap-point`, `--fractal-stripe-freq`,
`--fractal-de-scale`, `--fractal-trap-image`

**Per family**
`--fractal-ifs-preset/-iterations`, `--fractal-lsystem-preset/-angle/-depth`,
`--fractal-flame-preset/-symmetry/-iterations`,
`--fractal-attractor-preset/-iterations`,
`--fractal-raymarch-shape/-power/-yaw/-pitch/-dist`, `--fractal-buddha-samples`

**Composition**
`--fractal-compose`, `--fractal-grid RxC`

**AI paint (Track B)**
`--fractal-paint`, `--fractal-paint-out`, `--fractal-paint-mode`, `--fractal-prompt`,
`--fractal-negative`, `--fractal-sd-model`, `--fractal-sd-strength`,
`--fractal-sd-control-strength`, `--fractal-sd-guidance`, `--fractal-sd-steps`,
`--fractal-sd-control`, `--fractal-sd-lora/-lora-scale`

---

# Appendix — presets & enums (reference tables)

### Fractal families — `--fractal-kind`

```
┌──────────────┬─────────┬──────────────────────────────────────────┐
│ kind         │ type    │ what it is                               │
├──────────────┼─────────┼──────────────────────────────────────────┤
│ mandelbrot   │ escape  │ the classic set; z ← z² + c              │
│ julia        │ escape  │ fixed c (--fractal-julia-c), z₀ = pixel  │
│ burning-ship │ escape  │ z ← (|Re z|+i|Im z|)² + c; fiery hulls   │
│ tricorn      │ escape  │ mandelbar; conjugate z ← z̄² + c         │
│ multibrot    │ escape  │ z ← zⁿ + c (--fractal-power)             │
│ newton       │ escape  │ root basins (pair with coloring=angle)   │
│ nova         │ escape  │ relaxed Newton with an added c           │
│ phoenix      │ escape  │ uses the previous iterate                │
│ magnet       │ escape  │ magnet type-I rational map               │
│ sine         │ escape  │ z ← c·sin(z) (transcendental)            │
│ exp          │ escape  │ z ← c·exp(z) (transcendental)            │
│ buddhabrot   │ density │ density of escaping orbits (seeded)      │
│ flame        │ density │ fractal flame (variations + log-density) │
│ attractor    │ density │ strange-attractor trajectory             │
│ ifs          │ line    │ chaos-game attractor (fern, dragon…)     │
│ lsystem      │ line    │ Lindenmayer turtle drawing               │
│ raymarch     │ 3D      │ distance-estimated 3D fractal            │
└──────────────┴─────────┴──────────────────────────────────────────┘
```

### Coloring — `--fractal-coloring` (escape families)

```
┌────────────┬────────────────────────────────────────────────────┐
│ mode       │ look                                               │
├────────────┼────────────────────────────────────────────────────┤
│ smooth     │ continuous gradient (default)                      │
│ histogram  │ even color spread across the frame                 │
│ distance   │ thin, evenly-lit filaments                         │
│ orbit-trap │ color by closest approach to a shape               │
│ angle      │ final-iterate angle (Newton basins)                │
│ stripe     │ flame-like angular bands                           │
│ image      │ sample a photo at the orbit (--fractal-trap-image) │
└────────────┴────────────────────────────────────────────────────┘
```

### Palettes — `--fractal-palette`

```
┌────────────┬───────────────────────────────────┐
│ preset     │ mood                              │
├────────────┼───────────────────────────────────┤
│ fire       │ black → red → orange → white      │
│ ice        │ deep blue → cyan → white          │
│ electric   │ violet → magenta → cyan           │
│ neon       │ hot pink / orange / yellow / blue │
│ pastel     │ soft muted tones                  │
│ monochrome │ black → white grayscale           │
│ midnight   │ deep-space blues                  │
│ earth      │ browns & natural tones            │
└────────────┴───────────────────────────────────┘
```

### IFS presets — `--fractal-ifs-preset`

```
┌───────────────┬───────────────────────┐
│ preset        │ attractor             │
├───────────────┼───────────────────────┤
│ barnsley-fern │ the classic fern      │
│ sierpinski    │ Sierpiński triangle   │
│ dragon        │ Heighway dragon curve │
│ levy          │ Lévy C curve          │
│ tree          │ branching tree        │
│ spiral        │ logarithmic spiral    │
└───────────────┴───────────────────────┘
```

### L-system presets — `--fractal-lsystem-preset`

```
┌────────────────┬─────────────────────────────┐
│ preset         │ curve                       │
├────────────────┼─────────────────────────────┤
│ koch           │ Koch curve                  │
│ koch-snowflake │ Koch snowflake              │
│ sierpinski     │ Sierpiński arrowhead        │
│ dragon         │ dragon curve                │
│ hilbert        │ Hilbert space-filling curve │
│ gosper         │ Gosper (flowsnake) curve    │
│ plant          │ branching plant             │
│ bush           │ bushy plant                 │
└────────────────┴─────────────────────────────┘
```

### Flame presets — `--fractal-flame-preset`

```
┌────────────┬─────────────────────────────────────────┐
│ preset     │ note                                    │
├────────────┼─────────────────────────────────────────┤
│ flame      │ the default mixed set (spherical+swirl) │
│ sierpinski │ 3 linear maps                           │
│ spherical  │ spherical variation                     │
│ swirl      │ swirl variation                         │
│ spiral     │ spiral variation                        │
└────────────┴─────────────────────────────────────────┘
```

Flame variations (for custom `functions` in a spec file): linear, sinusoidal,
spherical, swirl, horseshoe, polar, handkerchief, heart, disc, spiral, hyperbolic,
diamond, ex, fisheye, exponential, power, cosine, bubble.

### Strange attractors — `--fractal-attractor-preset`

```
┌──────────┬─────────────────────────┐
│ preset   │ kind                    │
├──────────┼─────────────────────────┤
│ clifford │ 2D map                  │
│ dejong   │ 2D map (Peter de Jong)  │
│ bedhead  │ 2D map                  │
│ duffing  │ 2D map                  │
│ ikeda    │ 2D map                  │
│ lorenz   │ 3D ODE (butterfly), RK4 │
│ rossler  │ 3D ODE, RK4             │
└──────────┴─────────────────────────┘
```

### 3D shapes — `--fractal-raymarch-shape`

```
┌──────────────┬───────────────────────────────────────────────────────┐
│ shape        │ note                                                  │
├──────────────┼───────────────────────────────────────────────────────┤
│ mandelbulb   │ spherical-power Mandelbrot (--fractal-raymarch-power) │
│ mandelbox    │ box-fold + sphere-fold                                │
│ menger       │ Menger sponge                                         │
│ sierpinski3d │ Sierpiński tetrahedron                                │
│ quat-julia   │ quaternion Julia                                      │
└──────────────┴───────────────────────────────────────────────────────┘
```

### Composition — `--fractal-compose`

```
┌─────────────────┬───────────────────────────────┐
│ mode            │ grid of…                      │
├─────────────────┼───────────────────────────────┤
│ julia-sweep     │ Julia sets, c around a circle │
│ zoom-grid       │ progressive deep zoom         │
│ palette-grid    │ one fractal, every palette    │
│ variation-sweep │ seed / parameter variations   │
└─────────────────┴───────────────────────────────┘
```

### Orbit-trap shapes — `--fractal-trap-shape`

```
┌────────┬───────────────────────────────────┐
│ shape  │ distance to…                      │
├────────┼───────────────────────────────────┤
│ point  │ a point (--fractal-trap-point)    │
│ cross  │ the nearer axis through the point │
│ circle │ a circle around the point         │
└────────┴───────────────────────────────────┘
```

### AI paint — modes & control (Track B)

```
┌──────────────────────┬────────────────────────────────────────┐
│ setting              │ values                                 │
├──────────────────────┼────────────────────────────────────────┤
│ --fractal-paint-mode │ txt2img (default) · img2img            │
│ control (auto)       │ escape→canny · ifs/lsystem→lineart     │
│ control (auto)       │ flame/attractor/buddhabrot→softedge    │
│ control (auto)       │ raymarch→depth                         │
│ --fractal-sd-control │ override: canny·lineart·softedge·depth │
└──────────────────────┴────────────────────────────────────────┘
```

# Tips & gotchas

- **Negative coordinates:** prefer the `=` form: `--fractal-center=-0.745,0.113`.
- **Deep zoom looks flat?** Raise `--fractal-iter` (detail budget scales with depth).
- **Painting is dark/empty?** You're painting the black interior — zoom onto the
  boundary (§7.3).
- **Zoom didn't change the painting?** Control is too loose — raise
  `--fractal-sd-control-strength` (§7.4).
- **Paint on CPU is slow.** Build with `--features metal` / `--features cuda`; the
  first run downloads the model (~7 GB for SDXL).
- **Reproducibility:** Track A is byte-identical per spec; Track B is deterministic per
  seed on the same device. Every PNG carries its spec (`--fractal-clone`).
- **Track A never changes** when you paint — it's always saved separately.

Happy fractal-ing. For the pure-image compositor see
[`COMPOSE_TUTORIAL.md`](COMPOSE_TUTORIAL.md); for general SD generation see
[`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md).
