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
- **A2 — integration surfaces. DONE.** Documented the bookart integration in each home guide:
  **API.md** gains a `BookArt` builder section (+ Contents entry), mirroring `Map`; **SCRIPTING.md** gains
  a `plakat.bookart.*` words subsection (render/illustrate/origin/technique); **COMPILE.md** gains a new
  "Task-type blocks (`type: …`)" section documenting `type: bookart` (+ `type: map`, previously
  undocumented) with a worked example. Verified the compile example against the CLI and the API builder
  against the real `plakat::api::BookArt`. (Scenario `type: bookart` is covered by BOOKART.md's
  Integration-surfaces section from A1.)
- **A3 — docs audit sweep. DONE.** Swept `Documentation/` + `README` for drift and fixed: the README's
  stale "*Deferred to 6.1*" note (those items shipped → "6.1 added"); the `Documentation/README.md` index
  descriptions for `BOOKART.md` (6.1 commands + integration + six origins) and the very-stale
  `SCRIPTING.md` entry ("seven host words" → namespace-grouped); a broken self-link in `API.md`
  (`../Documentation/BOOKART.md` → `BOOKART.md`); and the previously-undocumented `plakat.map.*` /
  `plakat.fractal.*` Bund word namespaces (added to SCRIPTING.md). Verified: no age-gate drift, every
  documented `bookart` command exists, no other broken self-links, anchors resolve.

## Track B — corpus regen + robustness soak

- **B1 — regenerate the proof corpus. DONE.** Re-ran the full `bookart_run.sh` against the 6.1 binary
  (+ B3 tuning): border/plate/composite/kit/manuscript all regenerate clean (kit coherence 0.631/0.748,
  border.svg RDP-compacted to 48 KB). Updated the driver (6.0.0→6.1; added `origins` + `font` steps + a
  pointer to `bookart_origins.sh`) and `BOOKART_CORPUS.md` (version, a specs-&-drivers table incl.
  `bookart_composite.hjson` / `bookart_origins.sh` / `bookart_script.bund`, and the 6.1 step).
- **B2 — robustness soak. DONE.** Added edge-path tests: malformed / name-less **lexicon override** →
  parse-error (loader falls back, no panic); **degenerate spec** (unknown origin/technique/ornament) →
  resolves to a usable plan + no LoRA-404, no panic; **solid-slab** ornament → scorecard flags it
  (blank was already covered); **malformed / missing EPUB** → `Err`, not panic (feature `epub`). The
  empty-spec → procedural-divider path was already covered. Faint-`line` was the B3 fix.
- **B3 — quality nits from 6.1. DONE.** Root cause of faint `line`: `xdog` (a) **ignored `ink_weight`**
  entirely and (b) had no contrast normalisation, so a low-contrast LoRA render gave sparse specks.
  Fixed `binarize.rs`: `xdog(g, ink_weight)` now **autocontrasts** (1st/99th-pct stretch — no-op on
  full-range input) and biases `tau` + a darkening gamma by `ink_weight` (a real, monotone dial); and
  `threshold-bold` (woodcut) dilation now scales with `ink_weight`, pivoting to **0 px at the 0.6
  default** so woodcut is no longer a slab. Measured: chinese `line` ink coverage 0.004→**0.030**
  (near-blank → legible cloud/dragon woodblock); russian `woodcut` a bold-but-breathing border at
  **0.317** (was a near-slab). Both eyeballed. New `ink_weight_dials_line_boldness` test.

## Track C — performance profiling pass

- **C1 — profile the hot paths. DONE.** Ran `plakat bench sd15 512² 20 --repeat 3` and compared to the
  frozen 2.4.0 baseline (`out/baseline/baseline-sd15.json`): **no regression** — per-step **714 ms** vs
  711, VAE tail 3225 vs 3154, encode+first 16 vs 23, and peak RSS *lower* (2.65 vs 3.53 GB, the own-UNet
  default is leaner). The dominant cost is unchanged base diffusion (GPU); the bookart finisher /
  procedural rasteriser are negligible beside it, so no speculative micro-opt is warranted.
- **C2 — opportunistic win. DONE.** Removed the bookart diffusion **temp-PNG round-trip**: `diffuse()`
  now converts the `api::Image` to an `RgbImage` in memory (was `img.save(tmp)` → `image::open(tmp)`
  per render — a full PNG encode+decode + two disk ops, pure waste since PNG is lossless). The rare
  `matte` transparency path writes its own short-lived temp (U2Net's `matte` takes a path). Output is
  byte-identical; verified live (russian firebird renders correctly, no temp files). Suite 1739 green.

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
