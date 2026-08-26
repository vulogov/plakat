# RFC FACESWAP-2 — finish face-swap: colour-match + standalone verb (6.20.0)

**Status:** SHIPPED (6.20.0). Additive to the (already-complete) face-swap engine — a colour-match quality
fix + a standalone verb. **Reality check** (the
`ROADMAP_FACESWAP.md` `[ ]` boxes were stale): Phases 1–3 (SCRFD reference · inswapper_128 generator ·
ArcFace recognition) are done and numerically verified; **Phase 4 align + paste-back is also done**
(`FaceSwapper::swap_into` = `norm_crop` 5-pt similarity align → `inswapper.forward` → `paste_back`
inverse-warp + feather); and **multiperson already integrates it** — `multiperson --swap` flows through the
shared persona pipeline (`persona.rs` → `swap_into`). What actually remains:

## What ships

### P1 — colour-match the paste-back (the Phase-4 polish that was skipped)
`paste_back` feather-blends the swapped 128² crop back but does **no colour match**, so a swapped face can
carry the *source's* skin tone / white-balance / exposure — a visible "pasted head" tell under different
scene lighting. Add a **clamped skin-tone match**: shift the swapped crop's per-channel mean to the
**target** face region it replaces (the same principle as 6.19.0 Q2's `adetailer::tone_match`), before the
feather blend. Clamped so identity/detail is preserved and only tone aligns. Benefits **every** caller
(standalone, persona, multiperson) since they all go through `swap_into`.

### P2 — `plakat faceswap` standalone command (the missing Phase-5 verb)
Face-swap is only reachable today through persona/multiperson (which also *generate* a scene). Expose the
proven engine as a first-class verb for **existing** images:
`plakat faceswap <scene> --source <face.png> [--face N | --all] [--out PATH] [--restore]` — load
`FaceSwapper`, `detect` the scene faces (largest-first), `source_latent(source)`, and `swap_into` the
selected face(s). `--all` swaps every detected face with the one source; `--face N` picks the N-th
(largest = 0). Optional `--restore` runs the ADetailer detail pass. Reuses the whole pipeline; no new model.

### P3 — docs + license + cut 6.20.0
Tutorial section + README + **non-commercial license note** for inswapper (insightface weights are
non-commercial). Refresh `ROADMAP_FACESWAP.md` (mark 4/5 done). Cut 6.20.0 (bump Cargo+lock, gate
`--test-threads=1`, turbofish on new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**,
NO Claude coauthor).

## Honest limits
inswapper_128 is a 128² identity transfer — it transfers *identity*, not high-frequency skin detail;
`--restore` helps but can slightly drift identity (documented). Colour-match aligns the mean tone, not
per-pixel relighting. SCRFD must find the target face (small/occluded/profile faces may miss). Weights are
**non-commercial** (insightface) — gated behind explicit opt-in, hosted safetensors.

## Sequencing
**P1** colour-match (pipeline-wide win) → **P2** standalone verb → **P3** docs + cut.
