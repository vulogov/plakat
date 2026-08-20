# plakat quality-2 — roadmap (6.11.0, RFC QUALITY-2)

Close the three 6.10 naturalize deferrals. A **depth** cycle (no new studio, no novel weight-free
algorithm → **no G0**); each phase reuses an existing piece and ships value alone.

## P1 — hi-res fix — DONE (commit pending)
`generate --hires <factor>`: after generation, **before `--etch`**, run the existing `upscale --diffusion`
(tile-ControlNet / SUPIR-lite) at ×factor on each output to inject real, coherent detail — fixes the
low-res tells (cloud-foliage, dissolving background, incoherent geometry) that the analog pass can't.
Fold into `--quality high` (→ `--hires 1.5`). Order: gen → hires → naturalize → etch. Reuse `cli::upscale`
diffusion path. Verify it lifts detail on a fixed prompt without changing composition.

## P2 — full L1 re-etch after naturalize — DONE (commit pending)
`naturalize` on an etched image (unless `--no-reetch`): read the input `EtchId` (L0 manifest) →
`etch::set_parent(id)` → save the naturalized buffer via `imaging::io::save_rgb_u8_with_metadata`
(re-embeds L1 into the new pixels + L0 with the parent chain + enqueues L3). `doctor --if-plakat` then
resolves the output as a **derived** image with a valid L1 (not a stale mark). Un-etched input → no etch
invented. Closes the QUALITY-1 P1/P2 honest limit. Verify `doctor --if-plakat` on a naturalized etched
image reports `derived`.

## P3 — AI-tell ranking — DONE (commit pending)
`rank --ai-tells` sorts by `naturalize::ai_tell_score` (ascending — least-AI first, weight-free, no scorer
load) + writes `ai_tell` into the sidecar; `generate --keep-best K --ai-tells` ranks on
*aesthetic − λ·ai-tell* (λ=2.0) to prune a batch to the most human-looking frames, recording both `score`
and `ai_tell`. New `GenerationMetadata.ai_tell` field + `imaging::io::patch_sidecar_ai_tell`. Reuses
`pipelines::aesthetic` + `naturalize::ai_tell_score`. 1 unit test (sidecar round-trip + score coexists +
no-sidecar no-op).

## P4 — parity + docs + cut 6.11.0
Extend `Documentation/QUALITY.md` (hires / re-etch / ai-tells) + doctor note + README; a corpus step for
the hi-res + re-etch demo. **CUT 6.11.0** (bump Cargo+lock, gate `--no-default-features --lib`, **pin
turbofish on new `.parse()`**, FF `git push 6.11.0:main`, tag → 6-asset CI, `cargo publish --locked
--allow-dirty --no-default-features`, `gh release edit` + bg waiter, **verify the Windows leg**, NO
Claude/Anthropic coauthor).

## Sequencing
**P1** (hi-res fix) → **P2** (re-etch) → **P3** (AI-tell ranking) → **P4** (cut). Independent phases.
