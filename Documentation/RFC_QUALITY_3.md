# RFC QUALITY-3 — naturalize as *de-slop*: make a genuinely better picture

**Status:** draft (6.12.0) · **Depth cycle** on [`RFC_QUALITY_1.md`](RFC_QUALITY_1.md) /
[`RFC_QUALITY_2.md`](RFC_QUALITY_2.md).

## The reframe

QUALITY-1 framed `naturalize` as "reduce the AI-generated *fingerprint*" — stamping analog imperfections
(grain, aberration, vignette) so a render reads as human-sourced. Owner feedback corrects the goal:

> Not making computer art become genuine human art. But AI/computer output must be **not sloppy**.
> Geometry, colours and all other features of naturalize must make a **better photo or art picture**.

So the purpose is **de-slop**, not disguise: fix the things that make AI output look cheap. Chromatic
aberration and heavy grain are *degradations* — they get demoted; the headline becomes genuine
**quality improvement**.

## What ships

### 1. `polish` — the weight-free quality core (runs FIRST)
A real correction pass, deterministic, no GPU:
- **gray-world white balance** — neutralise the AI colour cast (clamped ±15% so a true sunset survives),
- **robust auto-levels** — stretch a muddy/washed histogram to true black/white (0.5 / 99.5 luminance
  percentiles), **ratio-preserving** (scale luminance, apply the same per-pixel gain to R/G/B) so contrast
  lifts without shifting colour,
- **vibrance** — tame blown-out oversaturation *and* lift dull colour toward a natural mid,
- **unsharp** — crisp the soft AI mush.

`--polish <0..1>`; preset defaults (subtle 0.55 … photo 0.70). It runs even with every analog knob at 0.

### 2. `micro-texture` — the fix for plastic skin
Real skin has pores and micro-wrinkles; AI skin is a perfect gradient. `micro_texture` adds fine
two-octave detail **only where the image is unnaturally smooth** (local-variance gated, so hair/fabric are
left alone) and **only in mid-tones** (where skin lives). Heavy on the `People` focus (`micro: 0.85`).
`--micro <N>` to tune.

### 3. Structural / realism improvement (model-backed)
Geometry, coherence and overall realism **cannot** be fixed weight-free — they need the model. The
corrective focuses run BEFORE the weight-free pass, so the order is **fix structure → improve colour/detail
→ light finish**:
- `--geometry` / `--anatomy` — img2img re-resolve with a **realism-led** prompt ("photorealistic, natural
  coherent detail, believable depth, well-formed structure") + an **anti-faceting** negative (faceted,
  fragmented, glassy shards, kaleidoscope) at a gentle strength (0.26) so structure is corrected without
  repainting the scene,
- the hi-res fix (`generate --hires`, QUALITY-2) injects coherent detail.

### 4. Presets rebalanced + analog demoted
Chromatic aberration cut to ~0 (it's fringing = sloppy); grain lightened; `polish`/`micro` dialled up. All
presets stay realism-oriented (no vintage).

### 5. `naturalize --etch` freshly etches; etch out of the quality story
`naturalize --etch` on a **non-plakat** image now mints a fresh valid etch (`fresh_etch`, the same claim
`generate --etch` makes) instead of silently writing an un-etched file. But **etch is provenance, not
quality** — its L1 DCT-QIM mark adds a faint fixed-lattice texture (visible on smooth gradients, a
robustness tradeoff; the step is shared with the decoder so it can't be lowered without breaking every
etched image). It's therefore **off** in the quality demos.

## Drivers
- `corpus/naturalize_run.sh` — photoreal portrait + steppe (no etch); `--polish`/`--micro` isolation.
- `corpus/naturalize_art_run.sh` — **new**, `assets/citystreet.png` (AI-slop watercolour): de-slop an
  AI *art* picture; RENDER=1 adds the model geometry fix.

## Honest limits
- Weight-free `polish`/`micro` improve **colour, contrast, detail, surface realism** — not structure.
- Structural/geometry/realism gains are **model-backed** (img2img/upscale); they can shift style and
  won't invent correct physics.
- The AI-tell score stays a coarse ranking heuristic.

## Sequencing
**P1** polish · **P2** micro-texture · **P3** corrective realism tuning + presets · **P4** fresh-etch +
drivers · **P5** parity + docs + cut 6.12.0.
