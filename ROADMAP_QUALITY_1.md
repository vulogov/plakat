# plakat naturalize / quality — roadmap (6.10.0, RFC QUALITY-1)

Make generations read human-sourced, not AI-generated. Leads with the weight-free **naturalize
post-pass** (the biggest visual win, works on any image); the model-backed `--quality` preset + hi-res
fix and the AI-tell scorer layer on top. Stands on the guidance bundle, `upscale --diffusion`, the
aesthetic scorer, and OWL-ViT + inpaint.

## G0 — de-risk the one novel weight-free algorithm: analog naturalize

The novel piece is the **analog-imperfection math** — turning a too-clean, over-saturated digital image
into one that reads as physical media, *without destroying it*. Prove it before building the cycle.

- **G0.1 — naturalize probe (`examples/naturalize_probe.rs`).** Apply film grain + chromatic aberration +
  vignette + bloom + a desaturating film grade to a synthetic "too-clean, over-saturated" image (flat
  gradients + a hard-edged shape + a uniform high-freq tile standing in for "AI-painterly" texture).
  **Measure an "AI-tell" delta**: mean HSV **saturation drops**, high-frequency **noise/variance rises**
  (grain breaks the uniform texture), a **radial channel offset** appears (aberration), **corners darken**
  (vignette) — while **structure is preserved** (content correlation to the input stays high, i.e. we
  degraded the fingerprint, not the picture). PASS → P1 uses the algorithm; settles RFC Q3 (default preset).

## P1 — `plakat naturalize` (weight-free post-pass; front-loaded)

`src/naturalize/{grain,aberration,vignette,bloom,grade,mod}.rs` + `src/cli/naturalize.rs`: the analog
stages from G0, driven by params + presets (`subtle`/`photo`/`painting`/`vintage`). CLI `naturalize IN
--out OUT [--preset …] [--grain/--aberration/--vignette/--bloom/--grade/--desaturate/--defocus]` and
`generate --naturalize [preset]` (post-pass on the generate path). **Ships value alone** — any image →
a more analog/human look, no GPU.

## P2 — the model-backed quality half: `--quality` preset + hi-res fix + signature removal

`--quality <low|medium|high>` on `generate` — per-family tuned CFG-rescale + FreeU + PAG +
dynamic-threshold + ADetailer + a **hi-res fix** (native gen → `upscale --diffusion` tile-CN refine to
inject real detail; fixes cloud-foliage / dissolving backgrounds / incoherent geometry). Naturalize gains
an optional `--refine` (img2img/tile detail pass) and **ghost-signature removal** (OWL-ViT "signature"
detect → inpaint, with a weight-free corner-clean fallback). Verify the preset lifts a fixed prompt across
two families.

## P3 — AI-tell scorer + keep-best selection

Extend `pipelines::aesthetic` with an **AI-tell penalty** (oversaturation + texture-uniformity +
ghost-signature); `generate --keep-best K` ranks on *aesthetic − ai-tell*; `plakat rank --ai-tells`
surfaces the score. Selects the least-AI-looking candidate from a batch.

## P4 — integration + corpus + docs + cut 6.10.0

Parity (`api::Naturalize` + `Generate::naturalize` · Bund `plakat.naturalize` · doctor); a before/after
corpus (a deliberately over-clean render → naturalized) + driver; `Documentation/QUALITY.md` + README;
**CUT 6.10.0** (bump Cargo+lock, gate `--no-default-features --lib`, **pin turbofish on new `.parse()`**,
FF `git push 6.10.0:main`, tag → 6-asset CI, `cargo publish --locked --allow-dirty --no-default-features`,
`gh release edit` + bg waiter, **verify the Windows leg**, NO Claude/Anthropic coauthor).

## Sequencing
**G0** (analog naturalize math) → **P1** (weight-free post-pass, ships value) → **P2** (`--quality` +
hi-res fix + signature removal) → **P3** (AI-tell scorer) → **P4** (cut). Front-load the weight-free half —
any image → a more human look with no GPU.

## Honest limit (restate on every phase)
Naturalize reduces the machine *fingerprint* (over-clean / over-saturated / grain-free / signature-ghost);
it does **not** fix physical reasoning (muddled reflections, impossible joinery). Hi-res fix helps detail/
geometry but won't invent correct physics. It's a look, not a forensic disguise.
