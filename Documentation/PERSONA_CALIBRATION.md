# PERSONA-1 calibration (RFC §13)

Calibration is what turns the persona schema from a suggestion box into an instrument. It is a slow,
offline, **per-family** process whose outputs are committed tables in
`assets/persona/calibration/<family>.hjson`. This note documents the table format, how the numbers are
used, and how to (re)generate a table.

## What a table holds

- **`identity`** — the measurement conditions (population, prompt, sampler, steps, size, aligner,
  topology, lexicon version, `provisional`). A mismatch vs the current environment surfaces a
  **staleness** warning rather than a silent wrong answer (§13.4). `provisional: true` marks a
  bootstrap seed, not a real sweep.
- **`priors`** — per landmark metric, the `median` (= the meaning of `0.5`, §13.1) and `p5`/`p95`
  (the usable range). The scorecard maps a realised metric to a `[0,1]` scalar against this.
- **`curves`** — per geometric attribute, the empirical `requested → realised` transfer function
  (normalised `[0,1]` samples) plus its fitted `slope`, `variance`, and derived `grade`. The compiler
  **pre-distorts** through the curve inverse so a requested value *lands* at that value (§13.2); the
  grade (`strong`/`moderate`/`weak`/`experimental`) is **measured, never asserted** (§13.3).
- **`harmonise`** — per detail kind, the img2img strength that blends a composite into skin without
  erasing it (§13.2).
- **`spontaneous_detail_rate` / `prompted_detail_hit_rate`** — the §13.1 detail baselines.

## How the numbers are used

| Consumer | Uses |
|---|---|
| `persona verify --model <family>` | scores `eyes.spacing` / `mouth.width` / `face.width` against the prior; weights by grade |
| `persona geometry --calibrate <family>` | pre-distorts the deformation through the curve inverses (§13.2) |
| `persona show --model <family>` | shows the per-family controllability grade badge on each attribute |

The three geometric scalars the aligner can measure today are `eyes.spacing` (interpupillary/face-width),
`mouth.width` (mouth/face-width), and `face.width` (face aspect, inverted — a wider face has a smaller
height/width). More attributes become scorable as the aligner metric set grows.

## Regenerating a table

**Bootstrap (no renders).** Reproduces the committed provisional seed — priors from the geometry
engine's mean-template metrics, grades from the lexicon `control` defaults:

```
plakat persona calibrate sdxl --bootstrap --out assets/persona/calibration/sdxl.hjson
```

**Measured sweep (the offline compute job).** Render a sweep, then measure it. Each render is named
`<attr>__<requested>__<seed>.png` (e.g. `eyes.spacing__0.75__3.png`); sweep each attribute across its
range holding everything else fixed, several seeds per step, plus a `requested = 0.5` population for the
prior. Then:

```
plakat persona calibrate sdxl --from ./sweep-sdxl --out assets/persona/calibration/sdxl.hjson \
  --prompt "…" --sampler euler --steps 30 --size 1024
```

`calibrate` detects + aligns each render (SCRFD → PIPNet-98), groups by attribute and step, takes the
median realised metric per step, normalises against the measured prior, fits a monotone curve, and
derives the grade. The **render half is the scheduled compute job**; the measurement half runs on any
sweep directory today (the same split as the §2.3 baselines).

## Recalibration triggers (§13.4)

A table goes stale on: a new family; a change to a family's default sampler / step count / native size;
a lexicon change that alters a deformation basis or gain; a landmark topology change; a change to the
landmark aligner; and (for the harmonisation constants) a change to the compositor. Each table records
the inputs it was measured under; `staleness()` reports a mismatch.
