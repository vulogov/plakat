# `plakat texture` — corpus walkthrough

A reproducible demonstration of the TEXTURE-1 feature (plakat 6.3). A prompt becomes a **seamless,
tileable PBR material set**; run [`texture_run.sh`](texture_run.sh) to regenerate the images under
`corpus/images/texture/`.

```bash
cargo build --release --features metal   # once
corpus/texture_run.sh                     # renders the whole corpus
```

## The spec

| file | what it shows |
|---|---|
| [`texture_stone.hjson`](texture_stone.hjson) | a `texture/1` spec — weathered granite cobblestones, seamless (circular), `height: auto` (depth) + `roughness: from-albedo`, ORM + glTF + a lit preview. A matte **dielectric**: `metallic: 0.0` → `metallic.png` is solid **black** (stone is a non-metal — the physically correct value) |
| [`texture_steel.hjson`](texture_steel.hjson) | the **conductor** counterpart — brushed stainless steel, `metallic: 1.0` → `metallic.png` is solid **white**. **6.4.0:** `anisotropy: 0.85` adds a brushed **grain** → an anisotropy flow map + a preview highlight that stretches along the grain |
| [`texture_leaves.hjson`](texture_leaves.hjson) | a richly-coloured organic dielectric — a carpet of fallen autumn leaves, `roughness: from-albedo` so drier pale leaves read rougher than damp dark ones |
| [`texture_river.hjson`](texture_river.hjson) | a **wet** dielectric — smooth river cobblestones under streaming water; a low scalar `roughness: 0.18` gives the glossy sheen (the shine is low roughness, **not** metal — `metallic: 0.0`) |
| [`texture_rust.hjson`](texture_rust.hjson) | **6.4.0 headline — a COMPOSITE material.** Rusted iron (bare steel + orange rust in one tile) with `metallic: "auto"` + `roughness: "auto"` → a **structured** metal mask (bare-metal patches white, rust black) where single-class materials give a flat map. The one spec whose `metallic.png` carries real spatial detail |
| [`texture_trim.hjson`](texture_trim.hjson) | **6.5.0 — a trim sheet.** Three sub-materials (steel / rust / stone) composed into one banded **atlas**, each band tiling along U, with a `trim.json` UV-region sidecar. `texture trim`. |

## What the driver produces

1. **`texture lint` / `texture show`** — validate the spec and print the resolved plan (seamless mode,
   size, channel sources, the compiled albedo prompt). **No weights.**
2. **`texture render texture_stone.hjson --out stone/`** → a full material directory: `albedo.png`,
   `normal.png`, `roughness.png`, `metallic.png`, `height.png`, `ao.png`, `orm.png` (R=AO/G=rough/B=metal),
   `preview.png` (a lit sphere), `material.gltf`, and `material.json` (recipe + scorecard). Every channel
   tiles. The SDXL albedo downloads from HuggingFace on first use.
3. **`texture derive stone/albedo.png --out stone_derived/`** — re-derive the whole set from just the
   albedo (**no GPU**): normal/AO from a circular Sobel/cavity, roughness from a from-albedo heuristic.
4. **`texture verify stone/`** — the tileability / PBR-validity scorecard (tileability-x/-y,
   normal-validity, albedo-flatness, channel-consistency). **No weights.**
5. **`texture export stone/ --naming unreal --gltf`** — re-pack for Unreal (`T_BaseColor`…`T_ORM`) + a
   glTF material. **No weights.**

### New in 6.4.0 — "Deepen texture"

6. **Spatially-varying channels** — `texture derive rust/albedo.png --metallic auto --roughness auto`
   region-votes a **structured** metal mask for the composite rust (bare-metal white, rust black); the
   contrast `--metallic 0` is the flat mask a known dielectric wants. Metal↔dielectric is separated by
   **saturation** (bare metal is near-grey), so `auto` is opt-in — a plain grey dielectric should pass
   `--metallic 0`. **No GPU.**
7. **Anisotropy** — `texture derive steel/albedo.png --anisotropy 0.85 --anisotropy-angle 0` writes an
   `anisotropy.png` flow map and a preview whose highlight **stretches along the grain** (brushed metal).
   Omit the angle to auto-detect it from the height. **No GPU.**
8. **Blend** — `texture blend stone/ leaves/ --mask mix` blends two materials through a **tileable** mask
   (the default `mix`; `radial` also tiles; `x`/`y` are intentional transition sheets). **No GPU.**
9. **Variations** — `texture render texture_stone.hjson --variations 3 --keep-best` renders three seed
   variants side-by-side (`var-0/`…) and copies the best-scoring one to the output root.

### New in 6.5.0 — material layout (all weight-free)

10. **Trim sheet** — `texture trim texture_trim.hjson --out trim_panel/` composes steel/rust/stone into
    one banded **atlas** (each strip tiles along U) + a `trim.json` UV-region sidecar mapping each band's
    label to its UV rect. **No GPU.**
11. **Decals** — `texture decal make --shape crack` (and `--shape splatter`) build alpha-masked overlay
    materials; `texture decal apply <base> <decal> --at --scale` stamps them onto the stone — alpha-
    blending the channels and blending the normal via **Reoriented Normal Mapping** so the crack/rust
    relief rides the surface instead of flattening it. The driver layers both onto one weathered stone.
    **No GPU.**

`derive`, `blend`, `verify`, `preview`, `export`, `lint`, `show` all run **weight-free** — you can
build, score, and re-pack a full engine-ready material (including the new auto channels, anisotropy, and
blends) from a supplied albedo before any generation is involved.

### New in 6.6.0 — engine interop ([`texture_interop_run.sh`](texture_interop_run.sh))

A separate driver demonstrates the export/interop breadth. It renders steel (with a brushed anisotropy
grain) + stone, then **re-packs each to every engine target** with one `--engine` flag — showing how the
naming + packing + material document differ:

- `--engine gltf` → `material.gltf` (a complete glTF 2.0 material; steel's carries **`KHR_materials_
  anisotropy`** from its flow map) + ORM.
- `--engine unreal` → `T_BaseColor`…`T_ORM` (ORM = R:AO/G:rough/B:metal).
- `--engine unity-hdrp` → **`mask_map.png`** (RGBA: R:metal/G:AO/B:detail/A:smoothness) — the *same data
  packed differently* from ORM.
- `--engine godot` → ORM. `--engine materialx` → `material.mtlx` (1.38 `standard_surface`, for USD /
  Arnold / Substance).

The re-pack half is **weight-free** (no GPU). Run [`texture_interop_run.sh`](texture_interop_run.sh).

## The idea

A material is a **spec**, a **set of coherent tileable channels**, and a **measurement** — not a prompt
fragment. `plakat texture` resolves the spec, generates a flat, tileable albedo, derives the rest of the
PBR set with circular (so tileable) image ops, measures the result, and exports it engine-ready with a
lit preview. See [`Documentation/TEXTURE.md`](../Documentation/TEXTURE.md) and
[`Documentation/RFC_TEXTURE_1.md`](../Documentation/RFC_TEXTURE_1.md).
