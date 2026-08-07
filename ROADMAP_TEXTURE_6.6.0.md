# ROADMAP — plakat 6.6.0 · "Engine-interop breadth"

A deepening of `plakat texture` (RFC TEXTURE-1) into **interop** — make a plakat material drop cleanly
into more engines/DCCs, in each one's native convention. Today's export is ORM (R=AO/G=rough/B=metal) +
a *minimal* glTF + `plakat`/`unity`/`unreal` filename conventions. This cycle adds: a **complete,
validated glTF** (proper metallic-roughness + occlusion, and the `KHR_materials_anisotropy` extension
driven by the 6.4 anisotropy map), **MaterialX** (`.mtlx`) output, **Unity HDRP mask-map** packing
(which packs *differently* from ORM), and an `--engine` preset that picks the right naming + packing +
material document in one shot. Mostly **weight-free** (export-layer). Fully additive.

Branch `6.6.0` (off `main` @ `aaabb25`, v6.5.1). Reference: `Documentation/RFC_TEXTURE_1.md`,
`src/texture/export.rs`.

---

## Why — the packing conventions genuinely differ (and getting them wrong is the whole risk)

| target | packed map | R | G | B | A | notes |
|---|---|---|---|---|---|---|
| glTF 2.0 | metallicRoughness | — | roughness | metallic | — | occlusion is a *separate* texture (or R of the same) |
| glTF 2.0 | occlusion | AO | — | — | — | can reuse the metallicRoughness R (glTF allows ORM sharing) |
| Unreal / Godot | **ORM** | AO | roughness | metallic | — | plakat's existing pack — correct as-is |
| **Unity HDRP** | **mask map** | metallic | AO | detail | **smoothness** | smoothness = 1 − roughness; NOT the same as ORM |
| Unity (Standard) | metallic+smoothness | metallic | — | — | smoothness | separate occlusion |

The failure mode is silent (a material that *looks* wrong in-engine), so **G0 pins these exact
conventions in a table + tests** before building the emitters.

---

## G0 — pin the channel conventions (verification, not measurement)

- **G0.1 — convention table + packing tests.** Encode each target's packing in one place
  (`export::EnginePack`) and unit-test that a known (AO, rough, metal, aniso) input lands in the correct
  channel for ORM, HDRP mask map, and glTF metallic-roughness (incl. smoothness = 255 − roughness for
  HDRP). Pure, weight-free. **Exit:** the table matches each engine's published docs and the tests lock
  it. (No probe run needed — the risk is correctness, not a measurement.)

---

## Track A — a complete, validated glTF (+ anisotropy)
- **A1 — full PBR glTF.** Extend `gltf_document` to a valid glTF 2.0 `material`: `baseColorTexture`,
  `metallicRoughnessTexture` (ORM GB), `normalTexture` (with scale), `occlusionTexture` (ORM R + strength),
  and reference the actual written filenames. Validate the emitted JSON structurally (a test parses it +
  checks required fields); note it targets the glTF-Validator conventions.
- **A2 — `KHR_materials_anisotropy`.** When the material has an anisotropy map (6.4), emit the extension:
  `anisotropyStrength`, `anisotropyRotation`, and `anisotropyTexture` (the RG flow → the extension's
  tangent-rotation encoding). Declare it in `extensionsUsed`.

## Track B — MaterialX (`.mtlx`)
- **B1 — `standard_surface` document.** Emit a MaterialX 1.38 doc: a `standard_surface` node with
  `base_color` / `metalness` / `specular_roughness` / `normal` / (occlusion via a multiply) wired to
  `<image>` nodes referencing the channel files. The interchange format for USD / Arnold / Karma /
  Substance. Weight-free XML; a test parses it + checks the node graph.

## Track C — Unity HDRP mask map + the `--engine` preset
- **C1 — HDRP mask-map packing.** `mask_map_pack(m)` = R:metallic, G:AO, B:detail(=neutral 128 or a
  supplied detail mask), A:smoothness(=255−roughness). A 4-channel RGBA PNG. Distinct from ORM.
- **C2 — `texture export --engine <target>`.** One preset selects naming + packing + material doc:
  `gltf` (A) · `unreal` (ORM + `T_*` + a `.uasset`-friendly note) · `unity-hdrp` (mask map + HDRP names) ·
  `godot` (ORM + a `.tres`-friendly note) · `materialx` (B) · `plakat` (raw). Backwards-compatible: the
  existing `--naming` still works; `--engine` is the new one-shot.

## Track D — parity, corpus, docs, cut
- **D1 — integration parity.** `--engine` reachable on `texture render`/`derive`/`export`; `api::Texture`
  gets `.engine(...)`; `api::texture_export(dir, engine, out)`; Bund/doctor refreshed.
- **D2 — corpus.** Export the steel (anisotropy → glTF `KHR_materials_anisotropy`) + stone materials to
  each engine target; wire into `texture_run.sh`; update TEXTURE_CORPUS.md.
- **D3 — docs.** `Documentation/TEXTURE.md` — an "Engine export" section with the convention table + the
  `--engine` targets + what each writes (flags verified vs `--help`); tutorial passage.
- **D4 — CUT 6.6.0** (bump Cargo.toml+lock, gate `cargo test --no-default-features --lib`, **pin
  turbofish on any new `.parse()`** — the CI Windows toolchain is stricter (6.5.0 lesson), FF `git push
  6.6.0:main`, tag → CI 6-asset, `cargo publish --locked --allow-dirty --no-default-features`, `gh release
  edit` GH_TOKEN=vulogov + bg waiter, NO Claude/Anthropic coauthor).

---

## Sequencing
G0.1 (pin conventions) → **A** (glTF, the most-used target) → **B** (MaterialX) → **C** (HDRP + `--engine`
preset) → **D** (parity + corpus + docs + cut). Front-load G0.1 so every emitter builds on one verified
convention table.

## Non-goals
- Not writing binary engine formats (`.uasset` / `.tres` / `.unitypackage` / `.usdz`) — plakat emits the
  *textures* + a standard material doc (glTF / MaterialX) + correctly-named/packed channels; engine import
  is the user's step.
- Not KTX2 / basis texture compression (PNG only this cycle).
- Not a full USD stage — MaterialX doc only.
