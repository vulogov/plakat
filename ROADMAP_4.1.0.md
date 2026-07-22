# plakat 4.1.0 — roadmap (SHIPPED)

**`plakat fractals` — a pure-Rust fractal studio + an AI paint pass that turns a fractal's structure
into a real image (RFC FRACTALS-1), plus a CLI-wide `--help` readability pass.**

Status: `[x]` done.

## `plakat fractals` (RFC FRACTALS-1) — all 8 phases DONE

- [x] **Phase 1 — escape-time core + CLI.** `FractalSpec` (HJSON serde + embedded PNG `fractalspec`
      tEXt chunk → `--fractal-clone`), viewport→complex mapping, Mandelbrot / Julia / Burning Ship
      (rayon-parallel), smooth coloring, Lab-space palettes.
- [x] **Phase 2 — full escape family + coloring + AA + buddhabrot.** +tricorn/multibrot/newton/nova/
      phoenix/magnet/sine/exp; histogram / distance-estimate / orbit-trap / angle / stripe coloring;
      supersampling; deterministic seeded buddhabrot.
- [x] **Phase 3 — IFS + L-systems + progress bars.** Chaos-game IFS (6 presets) + L-system turtle
      (8 presets); an `indicatif` progress callback across every generator.
- [x] **Phase 4 — Track B AI paint pass.** ControlNet-conditioned img2img reusing the generation
      stack; per-family default control + prompt; GPU auto-detected (respects `--device`).
- [x] **Phase 5 — fractal flame + strange attractors.** Draves flames (18 variations, log-density
      color, symmetry) + 7 attractors (Clifford/DeJong/Bedhead/Duffing/Ikeda maps + Lorenz/Rössler
      RK4 ODEs).
- [x] **Phase 6 — interactive TUI explorer** (`--fractal-explore`): pan / zoom / retune live with an
      inline preview (ratatui-image); gated on the `ui` feature, graceful on a non-graphics terminal.
- [x] **Phase 7 — 3D distance-estimated raymarched fractals.** Mandelbulb / Mandelbox / Menger /
      Sierpiński3D / quaternion-Julia, orbit camera, Phong + AO + depth fog (nalgebra).
- [x] **Phase 8 — composition + spec generation + the photos bridge.** Grid compositions
      (`--fractal-compose`), image orbit-trap (`--fractal-trap-image`), offline prose→FractalSpec
      (`--fractal-from`, distinctive per phrase).

## Paint polish (post-phase)

- [x] **txt2img mode** (`--fractal-paint-mode`) — the fractal as ControlNet-only structure; **default**,
      so the paint pass produces real scenes (sky / horizon / lighting) shaped by the fractal.
- [x] Paint-friendly defaults, negative-coordinate CLI parsing, per-family control (canny / lineart /
      softedge / depth).

## CLI `--help` normalization

- [x] **Grouped help across the whole CLI** via `help_heading`: a universal **Global options** group
      on every command, fully-grouped `generate` (92 flags) + `fractals` (54), a shared heading
      vocabulary applied CLI-wide, and per-command named groups (map / multiperson / compile / animate
      / …). Cosmetic only — no parsing change.

## Docs

- [x] `Documentation/Tutorials/FRACTALS_TUTORIAL.md` — a detailed, example-heavy guide (families,
      coloring, palettes, spec/clone, prose→spec, composition, image-trap, explorer, the full Track-B
      paint section) + an appendix of ASCII reference tables for every preset/enum.

## Ground rules (kept)

- Default CLI image output byte-identical; new logic lands with unit tests.
- `Cargo.lock` committed in sync with `Cargo.toml` (the 3.11.0 `--locked` lesson).
- `fractals` is standalone-buildable; the explorer's TUI bits are `#[cfg(feature = "ui")]`-gated.
