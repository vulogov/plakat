# plakat 4.3.0 — roadmap: `plakat fractals` → ecosystem

**The mature fractals flagship (4.1 render + 4.2 depth) becomes pervasive — wired into the rest of
plakat.** The deferred RFC-FRACTALS-1 Phase-8 ecosystem items, each independently shippable.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — `plakat scenario` fractal task (DONE)

- [x] A `fractal` task type in the HJSON scenario system: batch-render fractals (and optionally paint
      them) alongside `generate` / `map` tasks, with the same seed / count / out conventions.

## Phase 2 — `plakat generate --control-fractal` (DONE)

- [x] Use a fractal as ControlNet conditioning for a normal generation: render a fractal (spec or a
      quick `kind:...` string) and feed its canny/lineart into `generate` — fractal *structure* guiding
      any prompt, from the generate side (the inverse of `--fractal-paint`).

## Phase 3 — Bund `plakat.fractal.*` scripting words (DONE)

- [x] Host words to render fractals from a Bund script — `plakat.fractal.size` (output override),
      `.render` (Track-A CPU), `.compose` (grid), `.paint` (Track-B AI, GPU), `.animate` (video/GIF).
      Render/compose/paint push an image **handle** so they interop with the existing
      save/relight/upscale/metadata words; animate writes a file and pushes its path. `src` reuses the
      `--control-fractal` resolver (spec file / `kind[:preset]` / prose). Numeric args accept quoted or
      bare integers via bund's native `conv`. Gated on the `fractals` feature. Live-verified end-to-end
      (`plakat run`): render, compose, animate → real distinct outputs; +section in SCRIPTING_TUTORIAL §14.

## Phase 4 — `plakat photos` integration (DONE)

- [x] `fractalspec` panel — the image-view info panel (`i` / `I`) shows a **fractal** section for any
      image carrying an embedded `FractalSpec` (read once at decode time into `view_fractal`): kind,
      framing / per-family knobs, palette, seed, and the AI-paint recipe when present.
- [x] Fractal-from-photo — `Ctrl-B n f` (**fractalize**) renders a Julia set textured by the selected
      photo(s) via the image orbit-trap (`coloring = image`, `trap_image = <photo>`), landing a new
      `*_fractal.png` (with embedded spec) in the album. Pure-CPU, no model. Both bits are
      `#[cfg(feature = "fractals")]`-gated so `photos` still builds without `fractals`. Verified: the
      trap-image render produces a photo-palette-textured Julia carrying a `fractalspec` chunk that the
      panel reads back.

## Phase 5 — aesthetic `--keep-best` on compose sweeps

- [ ] Score the cells of a `--fractal-compose` sweep with the `AestheticScorer` and keep / highlight
      the top-N (reuses `plakat rank`'s LAION predictor).

## Ground rules (kept)

- Default CLI image output byte-identical; new logic lands with unit tests.
- `Cargo.lock` in sync with `Cargo.toml`; `fractals` standalone-buildable; photos-integration bits
  `#[cfg(feature = "photos")]`-gated.
