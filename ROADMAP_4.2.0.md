# plakat 4.2.0 — roadmap: `plakat fractals` → depth

**The fractals flagship gets deeper.** Four fronts, each independently shippable, building on the
4.1.0 renderer + spec + paint stack.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase A — perturbation-theory deep zoom (the marquee)

`f64` runs out of mantissa around zoom ≈ 1e13 (the center can't be resolved finer → pixelation). Fix it
with **perturbation theory**: compute one high-precision **reference orbit**, then render every pixel as
a cheap `f64` **delta** relative to it. Only the reference (and the center coordinate) need arbitrary
precision — the per-pixel math stays `f64`, so deep zoom is fast.

- [ ] Pure-Rust arbitrary-precision float for the reference orbit (no `rug`/GMP C dep).
- [ ] High-precision center in the spec (decimal strings) + auto-enable perturbation past the `f64` limit.
- [ ] Reference orbit + `δ_{n+1} = 2·Zₙ·δₙ + δₙ² + δc` per-pixel iteration; smooth coloring.
- [ ] **Glitch handling** (Pauldelbrot's criterion) + secondary references so deep zooms stay correct.
- [ ] CLI: `--fractal-center` accepts high-precision decimals; verify vs f64 at moderate zoom.

## Phase B — fractal animation → video

- [ ] Zoom-in animation (geometric zoom into a point) and Julia `c`-sweep / parameter-sweep animations,
      rendered frame-by-frame (Track A) and encoded to mp4/gif via the existing `animate` ffmpeg path.
- [ ] `--fractal-animate {zoom|julia-sweep|param-sweep}`, `--fractal-frames N`, `--fractal-fps`.

## Phase C — more families & flame variations

- [ ] Complete the flame variation set (→ the full ~20+ V1/V2 variations: julia, bent, waves, popcorn,
      rings, fan, blob, pdj, …) usable from spec-file `functions`.
- [ ] A few more escape families / attractors where they earn their keep.

## Phase D — LLM-backed prose → spec

- [ ] Wire `prompt::complete()` (the same enhancer `map`/`generate` use) so `--fractal-from` can map an
      arbitrary description to a FractalSpec, with the deterministic keyword mapper as the offline
      fallback (robust-by-design, like `map`'s parser).

## Ground rules (kept)

- Default CLI image output byte-identical; new logic lands with unit tests.
- `Cargo.lock` in sync with `Cargo.toml`; `fractals` standalone-buildable.
- Perturbation matches the `f64` renderer where both are valid (a regression anchor).
