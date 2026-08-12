# plakat naturalize / quality — roadmap (6.10.0, RFC QUALITY-1)

Make generations read human-sourced, not AI-generated. Leads with the weight-free **naturalize
post-pass** (the biggest visual win, works on any image); the model-backed `--quality` preset + hi-res
fix and the AI-tell scorer layer on top. Stands on the guidance bundle, `upscale --diffusion`, the
aesthetic scorer, and OWL-ViT + inpaint.

## G0 — de-risk the one novel weight-free algorithm: analog naturalize

The novel piece is the **analog-imperfection math** — turning a too-clean, over-saturated digital image
into one that reads as physical media, *without destroying it*. Prove it before building the cycle.

- **G0.1 — naturalize probe (`examples/naturalize_probe.rs`) — PASS.** Film grain + chromatic aberration
  (radial R-out/B-in ∝ r²) + vignette + bloom + a desaturating warm film grade on a synthetic over-clean
  image (flat gradient + a uniform high-freq "AI-painterly" checker + a hard corner edge). **5/5 measures
  green**: saturation **0.74 → 0.66**, flat-region hi-freq variance **0.02 → 6.17** (grain breaks the
  uniform texture), R/B edge separation **0.0 → 4.3px** (aberration), corner luminance **120 → 84**
  (vignette), luminance-correlation **0.948** (structure preserved — degraded the fingerprint, not the
  picture). Two measurement-placement bugs found + fixed (grain measured in a flat region; the aberration
  edge moved to a large-radius corner on a uniform surround). → **P1 uses this algorithm**
  (`src/naturalize/`). Default preset (RFC Q3) = `subtle`. The probe's demonstration strengths (grain 0.4 /
  aberr 0.6 / vig 0.35) are stronger than the shipped presets — owner feedback: naturalize aims at
  contemporary **realism, NOT a retro/"vintage" look**, so the shipped grade only desaturates (kills the
  oversaturation tell) and keeps warm/vignette small; the `Vintage` preset was removed.

## P1 — `plakat naturalize` (weight-free post-pass; front-loaded) — **DONE (commit pending)**
`src/naturalize/mod.rs` (the analog stages from G0 + `Params`/`Preset` — realism-focused: `subtle`
(default) / `photo` / `painting`, **no vintage**) + `src/cli/naturalize.rs`. CLI `naturalize IN --out OUT
[--preset …] [--grain/--aberration/--vignette/--bloom/--desaturate/--warm/--defocus] [--no-reetch]`. 2 unit
tests (fingerprint-degrade-but-preserve-structure + deterministic/preset-parse) + live (subtle/photo on a
real render — grain on the smooth sky, aberration on trunk edges, gentle vignette; structure held). **Etch
bar DONE**: L0 JSON sidecar carried forward + PNG `tEXt`/`zTXt`/`iTXt` chunks verbatim-spliced (so
`plakat metadata`/`clone` + the etch tEXt carrier survive); `--no-reetch` writes a clean output.
**`generate --naturalize` MOVED to P2** (it pairs with `--quality` and needs the naturalize-before-etch
order inside the generate flow). **Ships value alone** — any image → a more human look, no GPU.

## P2 — the model-backed quality half — DONE (partial: --quality preset + ghost-signature removal; hi-res fix + full re-etch deferred)

`generate --naturalize [preset]` — apply the P1 pass after generation, **before `--etch`** (so the etch
writes into the final pixels). `--quality <low|medium|high>` on `generate` — per-family tuned CFG-rescale + FreeU + PAG +
dynamic-threshold + ADetailer + a **hi-res fix** (native gen → `upscale --diffusion` tile-CN refine to
inject real detail; fixes cloud-foliage / dissolving backgrounds / incoherent geometry). Naturalize gains
an optional `--refine` (img2img/tile detail pass) and **ghost-signature removal** (OWL-ViT "signature"
detect → inpaint, scoped to FOREIGN artifacts only, with a weight-free corner-clean fallback). **Etch
integration (mandatory):** standalone naturalize / removal on an already-etched image detects the plakat
etch, then **re-etches** — original `EtchId` carried as `parent`, L1 re-embedded into the new pixels, L3
re-fingerprinted, L0 re-written; `--no-reetch` opts out; verify `doctor --if-plakat` still resolves the
naturalized image (as a derivative). Verify the `--quality` preset lifts a fixed prompt across two families.

## P3 — AI-tell scorer — DONE (weight-free ai_tell_score + reported per run + api::Naturalize::ai_tell_score; rank --ai-tells deferred)

Extend `pipelines::aesthetic` with an **AI-tell penalty** (oversaturation + texture-uniformity +
ghost-signature); `generate --keep-best K` ranks on *aesthetic − ai-tell*; `plakat rank --ai-tells`
surfaces the score. Selects the least-AI-looking candidate from a batch.

## P4 — integration + corpus + docs + cut 6.10.0 — DONE (api::Naturalize + Bund + doctor + QUALITY.md + corpus + README; cutting)

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
