# `plakat naturalize` / `--quality` — make generations read human-sourced

Diffusion output has a fingerprint: too clean, over-saturated, grain-free, with a "repeating painterly"
texture, and sometimes a ghost-signature smudge from signed-painting training data. `plakat naturalize`
(the 6.10 feature, RFC [`RFC_QUALITY_1.md`](RFC_QUALITY_1.md)) reduces that fingerprint so an image reads
as **contemporary, human-made — not aged/vintage, and not obviously AI**. It is **fully additive**.

**The analog half is weight-free.** Grain, aberration, vignette, bloom, and the film grade run on CPU; a
supplied image → a more human look with no GPU. Only the *corrective* focuses (fixing structure/anatomy/
lookalikes) and `--quality` (generation-time levers) use a model.

> **What it is and isn't.** Naturalize reduces the machine *fingerprint*; it does **not** fix physical
> reasoning (muddled reflections, impossible joinery) — those are model-capability failures. It's a look,
> not a forensic disguise, and it stays **realism-focused**: the grade only *desaturates* (the loudest
> tell); the warm lift and vignette stay small, because a strong warm + heavy vignette read as an applied
> *filter* — its own artifact.

## The analog post-pass

```
plakat naturalize IN.png --out OUT.png [--preset subtle|photo|painting]
  [--grain N] [--aberration N] [--vignette N] [--bloom N] [--desaturate N] [--warm N] [--defocus N]
```

- **Presets** — `subtle` (default, barely-there), `photo` (a real-camera look), `painting` (canvas-like).
- **Film grain** (luminance-weighted), **chromatic aberration** (radial R-out/B-in ∝ r²), **vignette**,
  **bloom/halation**, a desaturating **film grade**, optional **defocus** — each individually tunable.

## Content focus qualifiers

Different subjects have different tells, so pre-tune the pass to one (all **combine**; `N` is a blend
weight, 0 = off, 1 = midpoint, >1 = stronger):

| Analog (weight-free) | targets |
|---|---|
| `--people N` | plastic/waxy skin → desaturate + fine grain, minimal aberration/vignette (wrong on faces) |
| `--sky N` | banding / too-smooth → fine de-banding grain |
| `--vegetation N` | the cloud-like repeating foliage mush → stronger broadband grain + a little defocus |
| `--cityscape N` | razor-clean geometry → edge chromatic aberration |
| `--landscape N` | atmosphere → gentle vignette + desaturation |
| `--sea` / `--river N` | water surface → specular bloom + fine grain |
| `--mechanics N` | metal / transports → specular bloom + edge aberration |
| `--household N` | indoor → soft grain, gentle vignette |

Example — a forest: `plakat naturalize forest.png --out out.png --vegetation 1 --sky 1`.

### Corrective focuses (model-backed — grain can't fix structure)

| Corrective (needs a model) | how |
|---|---|
| `--geometry N` | whole-image **img2img** re-resolves incoherent structure / joinery |
| `--anatomy N` | img2img re-resolves proportions / hands |
| `--no-twins N` | **detect** faces (SCRFD) + **inpaint** each duplicate with a distinct seed so lookalikes diverge |

These run **before** the analog pass. `--model` / `--refine-steps` / `--device` tune them.

## Ghost-signature removal

```
plakat naturalize IN.png --out OUT.png --designature br      # br | bl | tr | tl
```

Dissolves a **foreign** training-data signature smudge from a corner (weight-free content-aware fade). It
is scoped to foreign artifacts and **never touches plakat's own `--etch` provenance**.

## `--quality` at generation time

```
plakat generate "…" --quality high
```

Bundles the levers that fight the AI look, per model family, filling only knobs left at their default (so
explicit flags win): **CFG-rescale** (kills oversaturation) + **FreeU** (`low`), + **PAG** +
**dynamic-threshold** (`medium`), + **ADetailer** (`high`).

## As a pass in generate & scenario

- `plakat generate "…" --naturalize "photo vegetation=1 sky=0.5"` — naturalizes each output **in place**,
  preserving the PNG metadata (including the `--etch` L0 chunk).
- A scenario `naturalize: "<spec>"` field naturalizes every image the run produces.

## AI-tell score

A weight-free score in `0..1` (higher = reads more AI-generated) from the two loudest tells —
**oversaturation** and **over-smoothness**. `naturalize` reports it, and it's available via
`plakat::api::Naturalize::ai_tell_score`. It's a **batch-ranking heuristic** (pick the least-AI-looking of
N candidates), not a per-image guarantee.

## Etch preservation

`naturalize` carries plakat's own provenance forward: the L0 JSON sidecar is copied and the PNG text
chunks are spliced onto the output, so `doctor --if-plakat` still resolves it. (A full re-etch — re-embedding
the L1 pixel mark into the changed pixels with a `parent` chain — is a documented follow-up; `--no-reetch`
writes a clean, un-etched output.)

## Integration

`plakat::api::Naturalize` · Bund `plakat.naturalize` · `generate --naturalize`/`--quality` · scenario
`naturalize:` · `plakat doctor`.

## Honest limits

- **Not a physics fix** — reflections/geometry/anatomy are model-capability limits; the corrective img2img
  helps but won't invent correct physics.
- **The AI-tell score is a coarse heuristic** for ranking, not a verdict.
- **Full L1 re-etch after naturalize is a follow-up** — today the L0 provenance is carried, but the changed
  pixels no longer match the original L1 mark (the verdict degrades honestly rather than being lost).
