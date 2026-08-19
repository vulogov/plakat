# RFC QUALITY-2 — finish the naturalize deferrals (hi-res fix · re-etch · AI-tell ranking)

**Status:** draft (6.11.0 kickoff) · **Depth cycle** on [`RFC_QUALITY_1.md`](RFC_QUALITY_1.md) — not a new
studio, so **no G0** (every piece already exists; this is integration + closing three honest deferrals).

## Summary

6.10 shipped `plakat naturalize` and left three deferrals documented. 6.11 closes them:

1. **Hi-res fix** — the structural half of "less AI": generate at native, then a **tile-ControlNet
   upscale-diffuse** injects real detail, fixing the low-res tells (cloud-foliage, dissolving backgrounds,
   incoherent geometry) that grain can't. Reuses `upscale --diffusion` (SUPIR-lite).
2. **Full L1 re-etch after naturalize** — today naturalize carries the L0 provenance but the changed pixels
   no longer match the original L1 mark. Re-embed L1 into the new pixels with a `parent` chain, so
   `doctor --if-plakat` resolves a naturalized etched image as a **verifiable derivative**.
3. **AI-tell ranking** — `rank --ai-tells` and `generate --keep-best` select the **least-AI-looking**
   candidate, using the weight-free AI-tell score.

All additive.

## 1. Hi-res fix

```
plakat generate "…" --hires 2            # native gen → 2× tile-CN upscale-diffuse
plakat generate "…" --quality high       # `high` implies --hires 1.5
```

- `--hires <factor>` (e.g. `1.5`, `2`) runs, after generation and **before `--etch`**, a diffusion
  upscale on each output: pre-upscale ×factor, then a tiled img2img refine with **ControlNet-Tile** guiding
  each tile to hallucinate *coherent* detail (the existing `upscale --diffusion` path). This is the piece
  that actually fixes geometry/detail, where the naturalize analog pass only changes the *surface look*.
- Folds into `--quality high` (which the QUALITY-1 roadmap always intended to include it).
- Order: gen → hires → naturalize → etch (etch writes into the final pixels).
- Reuse: `cli::upscale` diffusion path (`upscale --diffusion --scale`), tile-CN model = SD 1.5 / SDXL.

## 2. Full L1 re-etch after naturalize

When `naturalize` (or `--designature`) rewrites the pixels of an image that **was** plakat-etched, and
`--no-reetch` is not set:

1. read the input's `EtchId` from its L0 manifest (sidecar / `etch` tEXt chunk);
2. `etch::set_parent(input_id)` — chain the original as the derivation `parent` (the same path
   img2img/outpaint/relight already use);
3. save the naturalized buffer through `imaging::io::save_rgb_u8_with_metadata`, which re-runs
   **L1 embed** (into the new pixels), **L0 manifest** (with the parent), and enqueues the **L3**
   re-fingerprint.

Result: `doctor --if-plakat OUT.png` resolves it as `derived` from the original id, with a *valid* L1 in
the current pixels — not a stale mark. `--no-reetch` still writes a clean, un-etched output. (If the input
was never etched, nothing changes — no etch is invented.)

## 3. AI-tell ranking

```
plakat rank imgs/ --ai-tells             # least-AI-looking first
plakat generate "…" --count 8 --keep-best 2 --ai-tells   # keep the 2 least-AI (aesthetic − ai-tell)
```

- `rank --ai-tells` sorts by `naturalize::ai_tell_score` (ascending — least AI first) instead of the LAION
  aesthetic score, and writes it into each sidecar (`ai_tell`).
- `generate --keep-best K --ai-tells` ranks candidates on **aesthetic − λ·ai-tell** so a batch is pruned to
  the most human-looking frames.
- Weight-free (the score is CPU-only).

## Integration & reuse

`cli::upscale` (diffusion tile-CN) · `etch::{set_parent, active}` + `imaging::io::save_rgb_u8_with_metadata`
· `pipelines::aesthetic` + `naturalize::ai_tell_score`. Surfaces: `generate --hires`/`--quality high`,
`rank --ai-tells`, `generate --keep-best --ai-tells`, `naturalize` (re-etch by default). Docs: extend
`Documentation/QUALITY.md`.

## Non-goals / honest limits

- Hi-res fix **improves** detail/geometry; it does not *invent correct physics* (a wrong reflection
  upscales to a sharper wrong reflection).
- Re-etch re-embeds L1 into the naturalized pixels — it does **not** claim the naturalized image is the
  original (it's a `parent`-chained derivative, which is the honest claim).
- The AI-tell score stays a coarse ranking heuristic, not a verdict.

## Sequencing (roadmap)
**P1** hi-res fix → **P2** full L1 re-etch → **P3** AI-tell ranking → **P4** parity + docs + cut 6.11.0.
Each phase is independent and ships value alone.
