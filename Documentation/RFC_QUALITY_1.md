# RFC QUALITY-1 — `plakat naturalize`: make generations read human-sourced

**Status:** draft (6.10.0 kickoff) · **Focus:** reduce the "AI-generated" fingerprint of plakat output ·
**Leads with:** a weight-free **naturalize post-pass** · **Stands on:** the guidance bundle (CFG-rescale /
FreeU / PAG / dynamic-threshold), `upscale --diffusion` (tile-ControlNet), the LAION aesthetic scorer, and
the OWL-ViT + inpaint edit path.

## Motivation

A plakat SD3 watercolor drew an independent assessment listing five tells that betray AI origin. They are
a precise spec for this cycle:

| Tell (from the assessment) | Root cause | Lever |
|---|---|---|
| **Repeating "AI-painterly" texture**, too-clean, oversaturated | high-CFG over-guidance + zero analog grain | CFG-rescale / FreeU / PAG (**have**, not defaulted) + **analog naturalize pass** (🔨 the novel piece) |
| **Incoherent geometry / joinery** | model coherence at native res | **hi-res fix** (gen → tile-CN upscale-diffuse to inject real detail — pieces exist in `upscale --diffusion`) |
| **Dissolving background** | detail starvation | same hi-res fix (+ optional depth-ControlNet) |
| **Ghost signature** (corner smudge) | signed-painting training data | negative-prompt (**have**) + **corner signature detect → inpaint** (🔨) |
| **Muddled reflections** | physics the model can't reason | region-inpaint refine — *honest limit: hardest; won't fully resolve in a diffusion pipeline* |

The single biggest visual win is **killing the over-clean, over-saturated digital look and adding physical
imperfection** — a too-perfect image is the #1 tell, and a grain / aberration / vignette / grade pass plus
CFG-rescale defaults move it toward "scanned from real media" fast, on **any** image, with no GPU.

## Three pillars

### 1. `plakat naturalize` — the analog post-pass (the lead, weight-free)

A deterministic image-processing pass that stamps **physical-media imperfections** onto a finished image
(generated or not), breaking the digital-clean fingerprint. All CPU:

- **Film grain** — luminance-dependent monochromatic + a touch of chromatic noise (real sensors/film are
  noisier in the mids/shadows).
- **Chromatic aberration** — a small RGB channel shift that grows with radius from centre (lens dispersion).
- **Vignette** — gentle radial darkening at the corners.
- **Bloom / halation** — bright regions bleed a soft glow (blur a highlight mask, screen it back).
- **Color grade** — a subtle tone curve / split-tone (film / warm / cool / teal-orange) + a small
  **desaturation** (the oversaturation tell is the loudest one).
- **Defocus** *(optional)* — a faint edge/lens softness so the frame isn't uniformly razor-sharp.

```
plakat naturalize IN.png --out OUT.png [--preset subtle|photo|painting]
  [--grain 0.4] [--aberration 0.3] [--vignette 0.3] [--bloom 0.2] [--grade film] [--desaturate 0.1] [--defocus 0.0]
```

Presets bundle sensible values — `subtle` (default), `photo` (a real-camera look), `painting` (canvas-like
for painterly renders). **All aim at contemporary realism, NOT a retro/"vintage" look**: the grade only
*desaturates* (kills the AI oversaturation tell); the warm lift and vignette stay small, because a strong
warm grade + heavy vignette read as an applied *filter* — its own artifact, not naturalness. Also reachable
as `generate --naturalize [preset]` and `api::Naturalize` / Bund `plakat.naturalize`.

### 2. `--quality` — better generation defaults (bundle the levers we already have)

A per-family curated preset so good-looking output is the default, not a tuning exercise:

```
plakat generate "…" --quality high
```

`--quality high` sets, per model family, a tuned combination of **CFG-rescale** (kills oversaturation),
**FreeU** + **PAG** (coherence/detail), **dynamic-threshold**, **ADetailer** (faces), and a **hi-res fix**
(generate at native → `upscale --diffusion` tile-CN refine to inject real detail and fix cloud-foliage /
dissolving backgrounds). `low` / `medium` / `high` trade speed for polish; the naturalize pass can chain
on the end.

### 3. AI-tell scorer — select the least-AI-looking candidate

Extend the LAION aesthetic scorer with an **AI-tell penalty**: oversaturation (mean HSV saturation),
texture uniformity (local-variance / FFT flatness — the "cloud-like repeating" texture), and a
ghost-signature check. `generate --keep-best K` then ranks on *aesthetic − ai-tell*, and a `plakat rank
--ai-tells` mode surfaces the score so a batch can be pruned to the most human-looking frames.

## Integration

- **Library**: `plakat::api::Naturalize` (+ `Generate::naturalize`).
- **Bund**: `plakat.naturalize` ( `in out -- handle` ).
- **Generate flag**: `generate --naturalize [preset]` / `--quality <low|medium|high>`.
- **Doctor**: `plakat doctor` reports the quality/naturalize capability.
- *(No `scenario type:` / `compile` — naturalize is a post-pass/flag on the generate path, not a new task
  type; `generate --naturalize` covers scenarios via the existing generate task.)*

## Reuse

The guidance bundle (`--cfg-rescale` / `--freeu` / PAG / `--dynamic-threshold`), `upscale --diffusion`
(SUPIR-lite tile-ControlNet) for the hi-res fix, `pipelines::aesthetic::AestheticScorer` for the scorer,
OWL-ViT (`remove --what "signature"`) + the inpaint path for signature removal, `img2img` for refine.

## Etch preservation (mandatory)

Removing **foreign** AI traces (a training-data ghost signature) must never strip plakat's **own**
provenance etch (RFC ETCH-1). The naturalize pass mutates pixels — grain, aberration, desaturate — which
would degrade the L1 pixel mark, and a naive re-save drops the L0 `tEXt` chunk. So the whole quality
pipeline is **etch-aware**:

- **Order in `generate`:** naturalize / removal run **before** the etch, so `--etch` writes L0 + L1 into
  the *final* naturalized pixels and they survive intact. (Naturalize → etch, never etch → naturalize.)
- **Standalone `naturalize` / signature-removal on an already-etched image:** detect the incoming plakat
  etch first; apply the pass; then **re-etch** — carry the original `EtchId` forward as the `parent`
  (ETCH-1's parent-chain), **re-embed L1** into the new pixels, **re-fingerprint L3**, and re-write the L0
  `tEXt` chunk + sidecar. The output stays a valid, verifiable plakat artifact (`doctor --if-plakat` still
  resolves it, now as a derivative of the original).
- **Signature/trace removal is scoped to foreign artifacts** (corner signature smudge, model watermarks) —
  it explicitly excludes plakat's own etch carriers.
- **Minimum bar (P1):** even the weight-free pass carries the L0 metadata forward on save; full re-etch
  (L1/L3) lands with the etch integration in P2. A `--no-reetch` escape hatch exists for users who want a
  clean non-etched output.

## Non-goals / honest limits

- **Naturalize does not fix physical reasoning.** Muddled reflections, impossible joinery, and wrong
  shadows are model-capability failures; the post-pass makes an image read as *physical media*, not as
  *physically correct*. Hi-res fix helps geometry/detail but won't invent correct reflections.
- **It's a look, not a lie.** The goal is to reduce the machine fingerprint (over-clean, over-saturated,
  grain-free, signature-ghosted), not to pass forensic analysis. (Orthogonal to `--etch` provenance, which
  stays available and invisible.)
- **Overdone naturalize degrades.** Heavy grain/vignette/desaturation destroys detail; the presets stay
  conservative and the G0 probe measures that content is preserved.

## Open questions (for the owner)

- **Q1 — scope:** all three pillars phased over P1–P4, or the naturalize post-pass alone first?
  *Recommendation:* full phased, naturalize-first (P1 ships a weight-free pass usable on any image).
- **Q2 — `--quality high` default-on?** Should a tuned `--quality` become the *default* for `generate`
  (opt-out), or stay opt-in? *Recommendation:* opt-in for one cycle, then revisit once measured.
- **Q3 — naturalize default preset:** `subtle` (barely-there) vs `photo` (clearly analog) as the default
  when `--naturalize` is passed bare. *Recommendation:* `subtle`.
