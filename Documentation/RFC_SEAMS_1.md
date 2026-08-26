# RFC SEAMS-1 — seam quality: texture tiling + upscale tiles (6.19.0)

**Status:** SHIPPED (6.19.0). An **improvement** cycle — depth on existing features, no new flagship.

**All phases shipped:** P1 frequency-aware texture feather + `seam_score` + `mode: "mirror"` · P2 upscale
smoothstep overlap feather + cross-tile colour match · P3 pigment-aware (chroma-gated) normal-from-photo.

**Additional improvements (same cut):** P5 texture `mode: "auto"` (measure raw seam → pick feather band,
mirror fallback on a hard seam; scorecard already reports per-channel tileability) · P6 quieter etch L1
mark (variance-gate: skip the QIM in flat blocks while keeping a per-bit decode majority — invisible on
smooth content, still recoverable) · P7 upscale pre-sharpen (light unsharp on the Lanczos base so
ControlNet-Tile locks onto crisper structure) · P8 fractals deep-zoom central re-reference (pick a
well-behaved *and* central glitched pixel → fewer residual glitch blobs at extreme depth).

Both
`plakat texture` (seamless tiling) and `plakat upscale --diffusion` (tiled refine) rely on **feather
blends** because the native circular-conv path was G0-killed on this hardware. Feathering works but can
leave a visible soft band (texture) or tile-to-tile drift (upscale). This cycle makes those seams better,
weight-free and measured.

## What ships

### P1 — texture seam quality (`seamless.rs`)
Today `feather_seam` is a **linear** cross-fade over a fixed `band`, and `make_tileable` offset-and-heals
with the same linear feather. A linear blend over a wide band blurs texture; over a narrow band it leaves a
tonal step. Improve:
- **Smoothstep feather** (C¹-continuous `3t²−2t³`) instead of linear → no hard band edges.
- **Frequency-aware blend** — match the **low frequencies** across the wrap boundary (kills the tonal step)
  while **preserving high-frequency** detail (no blur): split each side into low/high via a box/Gaussian,
  cross-fade the low band wide + the high band narrow.
- **Mirror-blend mode** (`mode: "mirror"`) — reflect across the boundary for textures that read better
  mirrored (fabric, organic) than wrapped.
- **Measure-first auto-band** — a pure `seam_score` (mean gradient energy *across* the wrap boundary vs the
  interior baseline); pick the band/method that **minimises** it, and report the residual. Extends the
  existing measure-first (G0.2) discipline instead of guessing a band.

### P2 — upscale tile seams (`tiled.rs` / `diffusion_upscale.rs`)
Tiled img2img already feathers the `overlap`, but each tile denoises independently → **tile drift** (mean
brightness / colour cast differs across the seam, so the feather blends two different exposures). Improve:
- **Smoothstep overlap feather** (matches P1).
- **Cross-tile colour/tone match** — before compositing, normalise each tile's mean+contrast (per channel)
  to its already-placed neighbour **within the overlap region**, so the blend joins matched exposures.
  Weight-free, applied in the compositor — no change to the diffusion.

### P3 — normal-from-photo accuracy (`derive.rs`)
`normal_from_height` builds the normal from luminance-as-height, so **albedo micro-contrast becomes fake
geometry** (a dark speck reads as a pit). Improve the height estimate for the image-to-material path:
- **Frequency-separate** the photo — bilateral / edge-preserving pre-smooth so pigment detail doesn't drive
  the height, and gate high-frequency height by local structure.
- Tune the Sobel strength/normalisation against a small synthetic gradient set (the existing tests).

### P4 — parity + docs + cut 6.19.0
Tutorial/TEXTURE + upscale docs; corpus. Cut 6.19.0 (bump Cargo+lock, gate `--test-threads=1`, turbofish on
new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
Seam repair is **post-hoc** — it fixes the boundary, not a genuinely tiling generator (still G0-blocked on
Metal). Frequency-aware blending trades a hairline tonal match for a slightly wider affected band; the
`seam_score` picks the balance but can't invent detail that isn't there. Tile colour-match assumes neighbour
exposure is the reference — a genuinely mis-denoised tile is reduced, not perfectly hidden. Normal-from-photo
is still a heuristic (no true geometry capture).

## Sequencing
**P1** texture → **P2** upscale → **P3** normal → **P4** cut. Independent phases; P1's smoothstep + seam_score
helpers are shared into P2.
