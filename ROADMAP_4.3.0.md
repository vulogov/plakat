# plakat 4.3.0 — roadmap: `plakat fractals` → ecosystem

**The mature fractals flagship (4.1 render + 4.2 depth) becomes pervasive — wired into the rest of
plakat.** The deferred RFC-FRACTALS-1 Phase-8 ecosystem items, each independently shippable.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — `plakat scenario` fractal task (DONE)

- [x] A `fractal` task type in the HJSON scenario system: batch-render fractals (and optionally paint
      them) alongside `generate` / `map` tasks, with the same seed / count / out conventions.

## Phase 2 — `plakat generate --control fractal`

- [ ] Use a fractal as ControlNet conditioning for a normal generation: render a fractal (spec or a
      quick `kind:...` string) and feed its canny/lineart into `generate` — fractal *structure* guiding
      any prompt, from the generate side (the inverse of `--fractal-paint`).

## Phase 3 — Bund `plakat.fractal.*` scripting words

- [ ] Host words to render fractals from a Bund script (`plakat.fractal.render`, `.compose`, `.animate`,
      handles interop with the existing save/relight/upscale words).

## Phase 4 — `plakat photos` integration

- [ ] `fractalspec` panel — show a fractal PNG's embedded spec in the photos viewer.
- [ ] Fractal-from-photo — the image orbit-trap (`--fractal-trap-image`) driven from a selected photo
      (a Julia set textured by the picture).

## Phase 5 — aesthetic `--keep-best` on compose sweeps

- [ ] Score the cells of a `--fractal-compose` sweep with the `AestheticScorer` and keep / highlight
      the top-N (reuses `plakat rank`'s LAION predictor).

## Ground rules (kept)

- Default CLI image output byte-identical; new logic lands with unit tests.
- `Cargo.lock` in sync with `Cargo.toml`; `fractals` standalone-buildable; photos-integration bits
  `#[cfg(feature = "photos")]`-gated.
