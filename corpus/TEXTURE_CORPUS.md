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
| [`texture_steel.hjson`](texture_steel.hjson) | the **conductor** counterpart — brushed stainless steel, `metallic: 1.0` → `metallic.png` is solid **white**. Shows the metallic channel working (stone black ⇄ steel white) |
| [`texture_leaves.hjson`](texture_leaves.hjson) | a richly-coloured organic dielectric — a carpet of fallen autumn leaves, `roughness: from-albedo` so drier pale leaves read rougher than damp dark ones |
| [`texture_river.hjson`](texture_river.hjson) | a **wet** dielectric — smooth river cobblestones under streaming water; a low scalar `roughness: 0.18` gives the glossy sheen (the shine is low roughness, **not** metal — `metallic: 0.0`) |

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

## The idea

A material is a **spec**, a **set of coherent tileable channels**, and a **measurement** — not a prompt
fragment. `plakat texture` resolves the spec, generates a flat, tileable albedo, derives the rest of the
PBR set with circular (so tileable) image ops, measures the result, and exports it engine-ready with a
lit preview. See [`Documentation/TEXTURE.md`](../Documentation/TEXTURE.md) and
[`Documentation/RFC_TEXTURE_1.md`](../Documentation/RFC_TEXTURE_1.md).
