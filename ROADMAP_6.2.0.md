# ROADMAP — plakat 6.2.0 (consolidation / polish)

After four consecutive flagships (photos 3.x → fractals 4.x → persona 5.0 → bookart 6.0/6.1), 6.2.0 is a
deliberate **breather**: no new flagship. Harden the surface that's there — docs, corpus, robustness,
perf, and a bug backlog. Nothing here changes the public contract; every item is a refinement.

**Theme:** *make what exists solid, documented, and reproducible.* Additive and low-risk by design.

---

## Track A — documentation & tutorials refresh

The 6.1 cycle shipped a lot of surface with only roadmap-level notes. Bring the docs current.

- **A1 — bookart docs for the 6.1 surface. DONE.** Brought `BOOKART.md`, `BOOKART_STYLES.md`,
  `BOOKART_TRANSPARENCY.md`, and `Tutorials/BOOKART_TUTORIAL.md` current: command table + sections for
  `origins` (+ lexicon override), `vectorize` + pixel-tier `--svg` (feature `bookart-trace`), `font`,
  `edit --ink-weight/--transparency/--fade` (+ `render --cache-raw`), glyph-driven `initial` (+ `--font`;
  fixed the example to use `render`+spec, since `illustrate` doesn't wire `ornament.glyph`), EPUB
  manuscripts (feature `epub`), `--import`, the six trained origins (Bilibin/Beardsley/Hokusai +
  Pyle/Doré/woodblock), and an opt-in-features callout. Added an "Integration surfaces" section
  (scenario/compile/Bund/API/photos). Every documented flag verified against `--help`.
- **A2 — integration surfaces.** Document scenario `type: bookart`, compile `type: bookart`, the Bund
  `plakat.bookart.*` words, and `plakat::api::BookArt` in the relevant guides (scenario / compile / Bund
  / API docs), matching how `persona` / `fractals` are documented.
- **A3 — docs audit sweep.** Grep the whole `Documentation/` + `README` for version strings, dead
  command names, and superseded guidance (e.g. "SVG procedural-only", "three origin LoRAs"). Fix drift.

## Track B — corpus regen + robustness soak

- **B1 — regenerate the proof corpus.** Re-run every corpus driver against the 6.1 binary; confirm
  byte-plausible outputs. Wire the new `corpus/bookart_origins.sh` into the corpus index / `BOOKART_CORPUS.md`.
- **B2 — robustness soak.** Exercise the edge paths surfaced during 6.1: faint `line`-technique renders,
  empty/degenerate specs, missing font / missing `--cache-raw`, malformed EPUB / lexicon override,
  origins with no hosted LoRA. Each should fail clearly or degrade gracefully — add tests where missing.
- **B3 — quality nits from 6.1.** The `line` technique on some origins (japanese/chinese) renders faint,
  and `woodcut` renders very heavy — tune the technique→binariser defaults / ink-weight so a default
  render reads well without hand-tuning. Keep the change measured (the scorecard is the check).

## Track C — performance profiling pass

- **C1 — profile the hot paths.** Bench a representative set (t2i sd15/sdxl/sd3.5, a bookart render, a
  fractal render) with the existing bench harness; capture a baseline and look for regressions since the
  last perf pass (2.4.0).
- **C2 — opportunistic wins.** Only land changes that are safe + measured (no speculative rewrites).
  Candidates: redundant image round-trips (bookart writes temp PNGs for diffusion — could stay in-mem),
  allocation churn in the finisher/procedural rasteriser.

## Track D — bug backlog

- **D1 — triage + fix.** Collect known issues (test-suite `#[ignore]`s worth revisiting, the map
  `--verbose` clap-collision class of footguns, any Metal fp-drift asserts) and clear what's cheap.
- **D2 — CI hygiene.** Confirm the `--no-default-features --lib` gate stays the source of truth; make
  sure the opt-in features (`bookart-trace`, `epub`) at least *compile* in CI (a feature-matrix check).

---

## Sequencing

Track A (docs) first — it's the highest-value, lowest-risk, and forces a full re-read of the surface
that seeds B/C/D. Then B (corpus/robustness), C (perf), D (bugs) opportunistically. Cut 6.2.0 when the
docs are current, the corpus regenerates clean, and the bug backlog is drained to a comfortable point —
this is a **patch-flavoured minor**, so the bar is "solid + documented", not "new capability".

## Release-flow reminders (from auto-memory)

Bump `Cargo.toml` **+ `Cargo.lock`** in sync (`--locked` CI); gate = `cargo test --no-default-features
--lib`; new capability surfaces in `doctor`; **no Claude/Anthropic co-authoring**; FF `main` via
`git push 6.2.0:main` + tag → 6-asset CI release + `cargo publish --locked --allow-dirty`; `gh release
edit` for notes (**GH_TOKEN env = vulogov owner, valid — do NOT `env -u GH_TOKEN`**); `assets/filipok.md`
+ `corpus/images` untracked → `--allow-dirty`.
