# Make a fantasy map (`plakat map`)

`plakat map` turns a prose world description into a fantasy map — a coastline,
mountains, rivers, biomes, towns, roads, and labelled landmarks. The geometry is a
**pure function of (spec, seed)**: the same spec + seed always produces the same
map, byte-for-byte, with no GPU and no network. You can then style it as crisp
linework, paint it with Stable Diffusion, or export it as scalable vectors.

## 1. From prose to a map

The quickest start — describe a world, get a map:

```bash
plakat map "a volcanic island kingdom ringed by salt marshes, a great river \
running from the central peaks to a southern harbour town" --map-render isle.png
```

Under the hood this runs two stages:

1. **Parse** — an LLM (the same `--enhance` provider stack) turns your prose into a
   `MapSpec` — a structured description of terrain, water, regions, landmarks, and
   infrastructure. Positions are **anchors** (relationships like *"at the mouth of
   the river"*), never pixel coordinates.
2. **Render** — the geometry engine builds the map from the spec and a seed, and
   draws it.

Save the parsed spec to inspect or reuse it (this skips the LLM next time):

```bash
plakat map "a volcanic island kingdom…" --map-dump-spec isle.json
plakat map --map-spec isle.json --map-render isle.png      # no LLM, deterministic
```

`--seed N` (default 42) picks the world; the same spec with a different seed is a
different island with the same *features*.

### Specs can be HJSON

`--map-spec` accepts **HJSON** as well as JSON — comments, no commas, the same
relaxed syntax as scenarios and `compile`. Existing `.json` specs still parse
unchanged (HJSON is a strict superset).

> **Gotcha — one field per line.** HJSON quoteless strings run to the *end of the
> line*, so inline objects like `{ cols: 2 rows: 2 }` are **invalid**. Expand
> every object so each field is on its own line:
>
> ```hjson
> tile_grid:
> {
>   cols: 2
>   rows: 2
> }
> ```

The canonical, all-features example is
[`corpus/map/realms.hjson`](../../corpus/map/realms.hjson) — see §10 below.

## 2. Styling the linework map

`--map-style` selects the palette:

```bash
plakat map --map-spec isle.json --map-render isle.png --map-style parchment   # default
plakat map --map-spec isle.json --map-render isle.png --map-style inked        # high-contrast
plakat map --map-spec isle.json --map-render isle.png --map-style blueprint    # cyan on dark
```

The render gives you hill-shaded terrain, an ink coastline, biome fills, rivers,
roads, per-kind landmark symbols (cities, fortresses, ports, lighthouses, …),
collision-routed labels, and cartographic furniture — a title cartouche, a compass
rose, a scale bar, and a legend.

### Seasonal palettes

`--map-season` tints the land palette by season — handy for showing the same
world in different moods:

```bash
plakat map --map-spec isle.json --map-render isle.png --map-season autumn
plakat map --map-spec isle.json --map-render isle.png --map-season winter
```

`spring | summer | autumn | winter`. **`summer` is the neutral default** — it's
byte-identical to passing no season at all, so existing renders don't change.

### Tabletop coordinate grid

`--map-grid N` overlays an `N×N` reference grid with `A1` / `B2` cell labels
(column letters across the top, row numbers down the left) — for hex/RPG-style
"the party is in C4":

```bash
plakat map --map-spec isle.json --map-render isle.png --map-grid 8
```

`0` (the default) draws no grid; `N` is capped at 26 (A–Z). The lines are faint
so they don't fight the map underneath. Composes with `--map-season` and
`--map-style`.

## 3. Tuning the geography

Two knobs control how the natural features look. Both work on the **CLI**, in a
**scenario**, and from **scripting** — and all three produce identical output.

**Erosion** — how rugged the coasts and mountains are:

```bash
plakat map --map-spec isle.json --map-render isle.png --map-erosion 0      # smooth, idealized
plakat map --map-spec isle.json --map-render isle.png --map-erosion 1      # natural (default)
plakat map --map-spec isle.json --map-render isle.png --map-erosion 2.5    # deep fjords + jagged peaks
```

At `0` the coast is near-circular and ranges are smooth ovals; at `1` you get bays,
peninsulas, and wandering ridgelines; above `1` it becomes dramatically rugged.

You can also set it in the spec (`"terrain": { "erosion": 1.5 }`) so it round-trips.
Erosion governs the irregularity of coastlines, mountain ridges, canyons, mesas,
and lake shores alike.

## 4. More terrain: canyons, mesas, and borders

Beyond mountains and water, the spec realizes a few more landforms (all under
`terrain`, all eroded by the `--map-erosion` knob).

**Dry canyons** — `terrain.rift_valleys` carves narrow, oriented gorges whose
floor stays *above* sea level (so they read as dry rifts, not flooded channels).
Each is a named region with an `orientation`, a `length_fraction`, and a `size` of
`shallow | moderate | deep | chasm` (deeper = cut closer to the sea):

```hjson
rift_valleys:
[
  {
    id: the_cleft
    name: The Cleft
    anchor: { kind: cardinal position: center }
    orientation: north-south
    length_fraction: 0.6
    size: deep
  }
]
```

**Plateaus / mesas** — `terrain.plateaus` raises flat-topped tablelands ringed by
a steep scarp. Same `NamedRegion` shape; `size` is `small | moderate | large`:

```hjson
plateaus:
[
  { id: highreach name: The Highreach
    anchor: { kind: cardinal position: east }
    size: large }
]
```

**Coastal shaping** — three arrays cut a jagged shoreline (all `NamedRegion`s
anchored to the coast they shape; omit them all for a smooth coast):

- `terrain.peninsulas` — land spits jutting into the sea; `size` `narrow | moderate | broad`.
- `terrain.inlets` — bays / coves cutting *into* the land; `size` `shallow | moderate | deep`.
- `terrain.fjords` — deep, narrow, steep-walled sea arms; `size` `moderate | deep`.

```hjson
peninsulas: [ { id: the_horn name: The Horn anchor: { kind: cardinal position: southeast } size: broad } ]
fjords:     [ { id: cold_arm name: The Cold Arm anchor: { kind: cardinal position: northwest } size: deep } ]
```

The prose parser maps coastal language for you: *"a fjord-cut northern coast"* →
`fjords`, *"a wide bay"* → `inlets`, *"a long cape"* → `peninsulas`. A committed
showcase lives at `corpus/map/coastal.spec.json`. Like every terrain feature, the
coast responds to `--map-erosion` (higher = more ragged).

**Political layer** — any `region` can carry a `political` block, drawing a
territorial ring, kind-styled borders to neighbouring regions, and a polity label.
`borders[].kind` is `river | mountain | disputed`:

```hjson
regions:
[
  {
    id: westmark name: Westmark biome: temperate_grassland
    anchor: { kind: cardinal position: west }
    coverage: 0.3
    political:
    {
      polity_name: The Westmark League
      polity_kind: confederation
      borders: [ { with_region: eastreach kind: river } ]
    }
  }
]
```

`with_region` references another region by `id`; the border is drawn between the
two regions' anchors and styled by `kind`. Each polity gets a deterministic muted
colour from its name, so it reads on any palette.

## 5. Town maps (urban scale)

A city or town spec (`scale_tier` 10–12, with an `urban` block) renders a **street
map** instead of a regional map: a wall with gates, arterials, ring/grid streets,
building lots, a waterfront with piers, and labels at urban anchors.

```bash
plakat map --map-spec town.json --map-render town.png
```

The street plan is configurable with `--map-urban-layout` (or `urban.layout` in the
spec), or inferred from context when omitted:

```bash
plakat map --map-spec town.json --map-render town.png --map-urban-layout radial    # medieval rings + radials
plakat map --map-spec town.json --map-render town.png --map-urban-layout grid       # planned / Roman
plakat map --map-spec town.json --map-render town.png --map-urban-layout organic    # winding old town
```

Inference defaults: **mountainous** terrain → `organic`, a **walled** town →
`radial`, **plains** / unwalled → `grid`. A straight grid is available — but only
when you ask for it; medieval walled towns default to the organic radio-concentric
plan.

A minimal urban spec:

```json
{
  "version": 2, "name": "Saltmere Town", "scale_tier": 11,
  "tile_grid": { "cols": 2, "rows": 2 },
  "terrain": { "dominant_elevation": "lowland" },
  "urban": {
    "layout": "radial",
    "wall": { "shape": "round", "radius": 0.74 },
    "gates": [ { "id": "north_gate", "name": "Kingsgate", "bearing": "north" },
               { "id": "harbor_gate", "name": "Harbourgate", "bearing": "south" } ],
    "districts": [ { "id": "market", "name": "Market Square",
                     "anchor": { "kind": "cardinal", "position": "center" } } ],
    "waterfront": "south",
    "piers": [ { "id": "long_pier", "name": "The Long Pier", "position": 0.45 } ]
  },
  "landmarks": [
    { "id": "hall", "name": "Town Hall", "kind": "city", "anchor": { "kind": "city_center" } },
    { "id": "keep", "name": "The Keep", "kind": "fortress", "anchor": { "kind": "at_gate", "gate": "north_gate" } },
    { "id": "pharos", "name": "The Pharos", "kind": "lighthouse", "anchor": { "kind": "pier_tip", "pier": "long_pier" } }
  ]
}
```

Landmarks place themselves against the town: `city_center`, `at_gate`,
`in_district`, `pier_tip`, `along_street`, `on_wall`.

## 6. Painting the map with Stable Diffusion

`--map-render-sd` runs the styled base through SD img2img + a Canny ControlNet so it
looks **hand-painted**, then re-composites the crisp linework + labels on top. This
is the only GPU step — build with a backend (`cargo build --release --features
metal` on Apple Silicon, `--features cuda` on NVIDIA):

```bash
plakat map --map-spec isle.json --map-render-sd isle.png \
    --map-sd-model sdxl --map-sd-lora Muapi/fantasy-map
```

- `--map-sd-model` — any plakat model (sdxl default, or sd15/sd21/turbo/an HF repo).
- `--map-sd-lora` — optional; SDXL-family defaults to the `Muapi/fantasy-map` style
  LoRA, `none` disables it.
- `--map-sd-strength` / `--map-sd-steps` / `--map-sd-guidance` — sampling knobs.
- `--map-sd-tile N` — for large maps, paint in overlapping memory-safe tiles.
- `--map-sd-raw` — keep the bare painting (skip the linework + label overlay).

The painted output is not deterministic (it's SD); the *conditioning* base is. Use
`--map-dump-conditioning` to see the exact image the paint starts from.

## 7. Vector export

Export the same geometry as scalable vectors for print or further editing:

```bash
plakat map --map-spec isle.json --map-export-svg isle.svg          # standalone SVG
plakat map --map-spec isle.json --map-export-geojson isle.geojson  # GIS FeatureCollection
```

GeoJSON gives you the coastline (closed rings), rivers + roads (LineStrings),
landmarks (Points), and — for any region with a `political` block — both a polity
**Point** (the label anchor) and a `territory` **Polygon** (the boundary ring as
real GIS geometry). All `name`/`kind`/`id` properties, normalized to `[0,1]`,
north-up. The SVG draws the territory ring as a faint dashed polygon.

## 8. Maps in batches: scenario, compile, scripting

A map is a first-class step everywhere, all rendering identically to `--map-render`.

**Scenario** — a `type: map` task interleaves maps with renders/animations:

```hjson
{ out: "./out", seed: 42
  tasks: [ { name: "isle", type: "map", map-spec: "isle.json",
             map-style: "parchment", map-erosion: 1.5 } ] }
```

Fields: `map-spec`, `map-style`, `map-paint` (SD), `map-scale`/`map-tiles`,
`map-sd-model`/`map-sd-lora`, `map-layout`, `map-erosion`, `map-provider`. Each task
writes `<out>/<name>/map.png`. Every spec-level feature (coastal terrain,
canyons, marsh, deltas, political layer) flows through automatically — it lives in
the spec the task loads, so nothing extra is needed to use it from automation.

Set `map-render-tiles: true` to slice the world into a grid of **seamless tiles**
(over `map-tiles`/`map-scale`) instead of a single `map.png` — the task then writes
`<out>/<name>/world.png` + `tile_r{R}_c{C}.png`, byte-identical to the CLI
`--map-render-tiles`:

```hjson
{ out: "./out", seed: 7
  tasks: [ { name: "world", type: "map", map-spec: "coastal.spec.json",
             map-tiles: "2x2", map-render-tiles: true } ] }
```

`world.png` is the full stitched map; the tiles are its quadrant slices (they
reassemble pixel-exact). Add `map-tile-furniture: true` (CLI: `--map-tile-furniture`)
to draw a frame + grid coordinate (`R1C2`) + north arrow on each tile, so a single
tile is a usable standalone map — the stitched `world.png` stays clean.

**Compile** — a `type: map` block in a `prompts.txt` compiles to a scenario map task,
so prose worldbuilding and maps live in one document:

```
type: map
map-spec: isle.json
map-style: parchment

name: my-isle
```

**Scripting** — a Bund script renders a map into an image handle:

```
"42" "seed" plakat.config.set
"grid" plakat.map.layout       \ set the town plan
2.0   plakat.map.erosion       \ set erosion
"isle.json" "parchment" plakat.map.render   \ ( spec-path style -- handle ); .paint for SD
"isle.png" plakat.save

\ tile a world into seamless tiles (world.png + tile_r{R}_c{C}.png) in a dir:
"coastal.spec.json" "parchment" "./tiles" plakat.map.tiles   \ ( spec-path style out-dir -- count )
```

## 9. Inspecting the geometry layers

Each layer of the engine can be dumped — useful for debugging a spec:

```bash
plakat map --map-spec isle.json \
  --map-dump-heightmap hm.png --map-dump-rivers rv.png --map-dump-coast co.png \
  --map-dump-biome bi.png --map-dump-landmarks lm.png --map-dump-roads rd.png \
  --map-dump-features fe.png --map-dump-streets st.png       # streets = urban specs
```

## 10. Non-Latin labels

The built-in label font is a Latin bitmap face (so the default render is asset-free
and byte-stable). For Cyrillic, CJK, and other scripts, build with the
`shaped-labels` feature and supply a font:

```bash
cargo build --release --features metal,shaped-labels
plakat map --map-spec ru_town.json --map-render town.png --map-font /path/to/font.ttf
```

The font you provide must cover the script you're using; nothing is bundled. (Cyrillic
and CJK render directly; complex scripts that need contextual shaping/RTL — Arabic —
render glyphs but unshaped, for now.)

## 11. A worked all-features example

[`corpus/map/realms.hjson`](../../corpus/map/realms.hjson) — "The Sundered Realms"
— is one continental spec that exercises **every** geographical feature the engine
realizes: two mountain ranges, a large plateau, a dry canyon, two lakes, a wetland
region, rivers + coast, several biomes, the political layer (polity rings + a
disputed border), landmarks, and a road with a bridge. It's authored in idiomatic
HJSON (comments, no commas, one field per line).

[`corpus/realms.sh`](../../corpus/realms.sh) compiles it into its full artifact
set — every geometry layer, the styled map in all three styles, the seasonal +
tabletop-grid variants, and the GeoJSON/SVG export:

```bash
./corpus/realms.sh
# styled + autumn/winter + 8×8-grid renders, all layers, vector export →
#   corpus/images/realms/  and  corpus/map/export/
```

Because the geometry is a pure function of (spec, seed), the whole thing runs with
**no GPU, no network, no API key**. Read `realms.hjson` as a copy-paste template
for your own world, then run individual lines:

```bash
plakat map --map-spec corpus/map/realms.hjson --map-render realms.png
plakat map --map-spec corpus/map/realms.hjson --map-season autumn --map-grid 8 \
    --map-render realms-autumn.png
```

## What's next

- The full geometry/anchor schema lives in the parser's system prompt — dump a spec
  (`--map-dump-spec`) to see the shape, then hand-edit it.
- For reproducible map proofs, see `corpus/map.sh` / `corpus/map_urban.sh` — every
  layer is byte-checked against a committed image.
