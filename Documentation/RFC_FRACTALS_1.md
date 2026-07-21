# RFC FRACTALS-1: `plakat fractals`
## Non-AI and AI-Assisted Fractal Generation

**Status:** Implementation in progress (v4.1 line) · **Binary:** `plakat fractals` ·
**Feature:** `--features fractals` · **Architecture:** two-track (pure CPU render + optional AI paint),
mirroring `plakat map`.

---

## Overview

`plakat fractals` generates fractals two ways:

- **Track A** — a pure-CPU, deterministic escape-time / IFS / L-system / flame / attractor / raymarch
  engine (rayon-parallel; `same spec + seed → same pixels`; no GPU, no model, offline). Always saved.
- **Track B** — an optional AI enhancement pass (`--fractal-paint`): the Track-A render feeds a
  ControlNet-conditioned img2img through the existing generation stack. Non-deterministic; opt-in.

The single authoritative description is a **`FractalSpec`** (LLM-emittable, seed-stable, human-writable
HJSON), embedded as a `fractalspec` tEXt chunk in every output PNG (`--fractal-clone` extracts it).

## Taxonomy (7 families)

1. **Escape-time** (complex plane): mandelbrot · julia · burning-ship · tricorn · multibrot · newton ·
   nova · phoenix · magnet · sine · exp. Default Track-B control: **Canny**.
2. **IFS** (iterated function systems): chaos game / deterministic; presets (Sierpiński, Barnsley fern,
   Koch, Lévy C, Dragon, Hilbert). Control: **Lineart**.
3. **L-system** (Lindenmayer + turtle): axiom + rules + angle; presets (Koch, Sierpiński, Dragon,
   Hilbert, Gosper, plants, bush). Control: **Lineart**.
4. **Fractal flame** (IFS + non-linear variations + log-density): 20 V1 variations; symmetry;
   density-estimation AA; optional `.flam3` import. Control: **SoftEdge**.
5. **Strange attractors** (density accumulation): lorenz · rössler · clifford · de-jong · duffing ·
   ikeda · bedhead; parameter-search mode. Control: **SoftEdge**.
6. **3D distance-estimated (raymarched)**: mandelbulb · mandelbox · menger · kleinian · quat-julia
   (nalgebra Vector3, SDF + sphere-tracing + Phong/AO). Control: **Depth** (the distance field IS the
   depth map).
7. **Hybrid / composed**: see Composition.

## Coloring engine

Escape-time: smooth (Bernstein) · histogram-equalization · distance-estimation · orbit-trap
(point/shape/**image**/fractal) · angle · stripe-average · buddhabrot. Density families: log-density +
structural color. L-system: solid / depth-gradient / length-gradient.

Palettes are **Lab-space** interpolated (`palette 0.7`, avoids the "through-grey" hue shift) between N
stops. Presets: fire · ice · electric · neon · pastel · monochrome · midnight · earth · custom. The
**orbit-trap-image** mode samples a user photograph at the orbit position — the bridge to
`plakat photos` (a Julia set's boundary coloured by a photo).

## Composition

Spatial zone grid (`imaging::grid`) · alpha blend · alpha-key overlay (`photos::layers`) · orbit-trap
cross-family · Julia c-sweep · zoom grid · HDR fusion (`photos::multishot`) · panorama
(`photos::homography`). "Random but regular" canvases: multi-seed sweep · Julia sweep · attractor
parameter search — each scored by `AestheticScorer`, `--keep-best N`.

## Track B (AI pass)

`fractal_prompt(spec)` (per-kind default prompts) + auto ControlNet (`default_control_for_kind`) →
`controlnet_annotator::annotate()` → `cli::img2img::run()` (all model families; SDXL default, tiled
for large canvas). Auto-applies the Made-Of-Fractals SDXL LoRA for escape-time. Mirrors
`map::render_sd`.

## Ecosystem integration

`plakat photos` (`Ctrl-v F` = fractal-from-photo orbit-trap; `fractalspec` tEXt panel; CLIP search over
a fractal library) · `plakat scenario` (a `fractal` task type) · Bund `plakat.fractal.*` words
(handles interop with relight/stylize/upscale/save) · `plakat generate` (fractal as ControlNet
conditioning) · `prompt::complete()` (prose → FractalSpec).

## Module layout

`src/fractals/{mod,spec,palette}.rs`, `render/{escape_time,ifs,lsystem,flame,attractor,raymarch,
buddhabrot}.rs`, `coloring/{escape_colors,density_colors}.rs`, `compose/*`, `ai_pass.rs`, `prompt.rs`,
`explorer.rs`, `cache.rs`; `cli/fractals.rs`; `scripting/words/fractals.rs`.

## Cargo

One new dep (`palette 0.7`); `rayon` / `num-complex` / `nalgebra` promoted from transitive to direct;
`rand` already direct. Feature `fractals = ["dep:palette","dep:rayon","dep:num-complex","dep:nalgebra"]`
— standalone-buildable (photos-integration is `#[cfg(feature="photos")]`-gated).

## Implementation phases (each independently shippable)

1. **Escape-time core + CLI** — FractalSpec + serde + tEXt; viewport→complex; Mandelbrot/Julia/
   Burning-Ship (rayon + num-complex); smooth coloring + Lab palettes; core CLI; `--fractal-clone`.
   **DONE.**
2. **Extended coloring + escape-time variants** — histogram/distance/orbit-trap/angle/stripe; the full
   escape-time family (+tricorn/multibrot/newton/nova/phoenix/magnet/sine/exp); supersampling AA;
   buddhabrot (deterministic seeded density). **DONE.**
3. **L-system + IFS** — chaos-game IFS (6 presets: fern/sierpinski/dragon/levy/tree/spiral,
   deterministic seeded, two-pass bounds+density) + L-system turtle (8 presets, custom axiom/rules,
   gradient-along-path, thickened strokes). Aspect-preserving fit; both AA via supersample. A progress
   callback (`indicatif` bar) now spans **all** generators. **DONE.**
4. **AI enhancement pass (Track B)** — opt-in (`ai.enabled` / `--fractal-paint`): the Track-A render
   is both the img2img init and the ControlNet source (auto per-family: canny/lineart/softedge), a
   per-family default prompt, reusing `pipelines::img2img` + `pipelines::controlnet`. GPU auto-detected
   (respects `--device cpu`); Track A always saved first, untouched. **DONE.**
5. **Fractal flame + strange attractors** — flame (18 non-linear variations, log-density color tone
   mapping, rotational symmetry, 5 presets) + strange attractors (7: clifford/dejong/bedhead/duffing/
   ikeda maps + lorenz/rossler RK4 ODEs, density-accumulated). Both deterministic, both AA via
   supersample. (`.flam3` XML import deferred.) **DONE.**
6. **TUI explorer** (`--fractal-explore`) — interactive pan/zoom/retune with live inline preview
   (ratatui-image: Kitty/iTerm2/Sixel); cycle kind/palette/coloring, adjust iterations, `s` to save
   full-res (spec embedded). Gated on the `ui` feature (render engine stays feature-free); graceful
   error on a non-graphics terminal. **DONE.**
7. **3D raymarched fractals** — distance-estimated sphere tracing (nalgebra): mandelbulb, mandelbox,
   menger, sierpinski3d, quat-julia; orbit camera (yaw/pitch/dist/fov), Phong + ambient occlusion +
   depth fog, orbit-trap palette coloring. Rayon per-row, deterministic. Track-B control = Depth (the
   distance field *is* the depth map). **DONE.**
8. **Composition + spec generation + the photos bridge** — grid compositions (`--fractal-compose`:
   julia-sweep / zoom-grid / palette-grid / variation-sweep, each cell a full Track-A render);
   image orbit-trap coloring (`--fractal-trap-image`, a Julia textured by a photo — the `plakat photos`
   bridge); offline prose→FractalSpec (`--fractal-from`, deterministic keyword mapper). **DONE.**
   *Deferred follow-ups:* aesthetic `--keep-best` scoring on sweeps, `plakat scenario` fractal task,
   Bund `plakat.fractal.*` words, the photos-TUI fractalspec panel, `plakat generate` fractal-as-
   ControlNet, LLM-backed `prompt::complete` prose, and perturbation-theory deep zoom.

## Open questions (see full RFC)

Deep zoom = perturbation theory (pure Rust) over `rug` (C dep); 20 flame variations for Phase 5;
`.flam3` import as `fractals-flame-compat`; 3D in the default feature; `Ctrl-v F` in photos;
LLM system-prompt calibration; `fractals` standalone via `#[cfg(feature="photos")]`.
