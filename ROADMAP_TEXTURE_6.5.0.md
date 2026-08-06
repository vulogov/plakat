# ROADMAP — plakat 6.5.0 · "Trim sheets & decals"

A deepening of `plakat texture` (RFC TEXTURE-1) into **material layout** — beyond a single tiling
texture. Two capabilities, both mostly **weight-free** (compositing existing material sets):

1. **Trim sheets** — compose several sub-materials into one **atlas** (stacked bands, each a strip that
   tiles along its run axis), with a UV-region sidecar so an engine can map faces to bands. The standard
   way games texture pipes / trims / panels / edges from one material.
2. **Decals** — **alpha-masked overlay** materials (a rust streak, crack, sign, logo) that layer onto a
   base material: an albedo+alpha + normal + roughness set, and a compositor that places a decal onto a
   base PBR set (alpha-blended albedo, reoriented-normal blend, per-channel merge).

Branch `6.5.0` (off `main` @ `b039514`, v6.4.0). Reference: `Documentation/RFC_TEXTURE_1.md`.

> **Cycle history:** 6.5.0 opened as "native seamless generation," but G0 (on the real stack) proved
> per-step latent-roll fails and native circular conv needs a ~1500-line vendor of candle's UNet blocks
> for a mild smear feather already handles — so the cycle **pivoted** here. Those findings are preserved
> in *Appendix A* + commits `4bb1bcb`/`f5a25a8`.

---

## G0 — de-risk the one non-trivial algorithm (decal normal compositing)

Trim/decal is mostly image compositing (low risk), with **one** algorithm worth a probe first:

- **G0.1 — reoriented normal blend.** Layering a decal's tangent-space normal over a base normal is NOT
  a simple lerp (that flattens detail). Prototype **Reoriented Normal Mapping (RNM)** / the UDN blend on
  a synthetic base+detail fixture; verify the blended normal stays unit-length and the detail rides the
  base slope correctly (vs a naive lerp). Pure math, weight-free. Exit: RNM validated → Track B uses it.

Other pieces (band atlas composite, alpha matte, UV-region sidecar) are deterministic image ops — covered
by unit tests in-track, no separate gate.

### G0.1 RESULT — RNM validated, PASS (commit `b47ef4f`, `examples/texture_rnm_probe.rs`)
RNM vs a naive lerp on 3 falsifiable fixtures: flat detail over a tilted base → RNM returns the base
tilt EXACT (err 0.000) where lerp flattens it (err 0.251); detail bump over a flat base → RNM keeps
100% amplitude where lerp keeps 51%; bumps over a tilted base → base-mean preserved (err 0.024, u8
quant), detail amplitude carried 87%, unit-length 100%. **→ Track B's decal compositor uses RNM.**

---

## Track A — Trim sheets
- **A1 — `TrimSpec` + compose.** An HJSON spec: an ordered list of **bands**, each `{ material: <dir|
  spec>, height: <fraction|px>, tile: x|y|none }`. Compose the bands into one atlas per channel
  (albedo/normal/roughness/metallic/height/AO + ORM), stacked along V, each band resized to its slot and
  tiled along its run axis. Weight-free when bands are pre-rendered dirs; a band given as a spec is
  rendered first (GPU).
- **A2 — UV-region sidecar.** `trim.json` — each band's normalized UV rect + label, so an engine / DCC
  can map faces to bands. Plus per-band scorecard (does each band tile along its axis?).
- **A3 — `texture trim <spec|dirs…> --out` + preview.** A preview that shows the atlas flat + a lit
  strip. `--bands a=0.5,b=0.25,c=0.25` shorthand for quick trims from material dirs.

## Track B — Decals
- **B1 — decal material.** A decal = base PBR channels + an **alpha** (opacity) mask. Build one from: a
  procedural `--shape` (stripe/splatter/crack/ring), an `--image` (matte its background), or a `--mask`
  PNG. `texture decal new` scaffolds; `texture decal make <src>` produces `albedo`(+alpha)/`normal`/
  `roughness`/`height`/`opacity`.
- **B2 — RNM normal blend (per G0.1).** The compositor blends the decal normal over the base via
  Reoriented Normal Mapping so detail rides the base surface, not flattens it.
- **B3 — `texture decal apply <base> <decal> --at x,y --scale s --rotate d --out`.** Composite a decal
  onto a base material: alpha-blend albedo/roughness/metallic/height, RNM-blend normal, re-derive AO if
  height changed, re-score. Weight-free. `--tile` for a repeating decal.

## Track C — integration, corpus, docs, cut
- **C1 — integration parity.** Spec `type: trim` / `type: decal` in scenario + compile; Bund words
  (`plakat.texture.trim` / `.decal`); `api::Texture` companions (`texture_trim`, `decal_apply`); doctor
  refresh. (bookart/6.4 template.)
- **C2 — corpus.** A trim-sheet demo (e.g. metal-panel + rivet-strip + pipe bands) and a decal demo
  (rust-streak decal applied to the stone material). Wire into `texture_run.sh`; update TEXTURE_CORPUS.md.
- **C3 — docs.** `Documentation/TEXTURE.md` — a "Trim sheets & decals" section (spec schema, the atlas +
  UV-region model, the decal/RNM model, all new flags verified vs `--help`); tutorial passage.
- **C4 — CUT 6.5.0** (bump Cargo.toml+lock, gate `cargo test --no-default-features --lib`, FF `git push
  6.5.0:main`, tag → CI 6-asset, `cargo publish --locked --allow-dirty --no-default-features`, `gh
  release edit` GH_TOKEN=vulogov + bg waiter, NO Claude/Anthropic coauthor).

---

## Sequencing
G0.1 (RNM) → **A** (trim, weight-free, front-loaded — value from supplied material dirs) → **B** (decals,
uses the G0.1 RNM) → **C** (integration + corpus + docs + cut). Front-load the weight-free compositors;
generation only enters when a band/decal is authored from a prompt.

## Non-goals
- Not a full UV-unwrapper or atlas-packer with arbitrary bin-packing (bands are ordered strips; decals
  are placed, not auto-packed).
- Not a runtime decal-projection system (this produces *baked* material sets, not a shader).
- Normal compositing is RNM (an approximation), not a displacement-accurate resolve.

---

## Appendix A — dropped: native seamless generation (G0 findings, preserved)
6.5.0's original headline. Dropped after measure-first G0 on the real stack:
- **Per-step latent-roll FAILS** (commit `4bb1bcb`): 3-way gravel (SDXL/DDIM) — baseline-feather 0.05 /
  raw-no-feather 2.65 / **roll-no-feather 3.48 (worse)**. A zero-pad conv's edge artifact is at the
  tensor boundary regardless of content; rolling never makes opposite edges adjacent — only circular
  padding does. The synthetic G0.B "90%" modeled a fixed linear operator, not a generative denoise.
- **Native circular conv scope** (commit `f5a25a8`): inference uses candle's upstream `unet_2d_blocks`
  (both sd15/sdxl), whose seam-bearing convs are built inside candle's containers → circular padding
  needs vendoring the whole ~1500-line block module + rewiring + fork maintenance, for a mild smear
  feather already handles (0.05). Deferred as a documented hard item; feather + adaptive band (6.4) stand.
