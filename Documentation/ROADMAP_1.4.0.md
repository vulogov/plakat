# plakat 1.4.0 — roadmap

The **compile** track is done (1.2.0 core + point-fixes + 1.3.0 Tera). 1.4.0 opens
**Track M — `plakat map`** (procedural fantasy maps), per
[`RFC_MAP_COMPILE_PLAN.md`](RFC_MAP_COMPILE_PLAN.md). 1.4.0 = **MAP-1 + MAP-2**:
spec + LLM parser + the geometry engine — **no SD render yet** (that's 1.5.0
linework / 1.6.0 tiled SD). All SemVer-additive.

The through-line (the 1.1.0 memory-wall lesson): geometry is a **pure function of
(spec, seed)** → byte-stable on-box corpus proofs. The non-deterministic LLM parse
is decoupled via a committed `--map-spec`; the memory-bound tiled SD render is the
1.6.0 capstone.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## MAP-1 — MapSpec v2 + LLM geographic parser

- [ ] **`MapSpec v2` structs + `Anchor` enum** (`src/map/spec.rs`) — typed spatial
      relations (`MouthOf`, `Bearing{constraint}`, `PassBetween`, …), serde round-trip.
- [ ] **LLM parser** (`src/map/parser.rs`) — prose → MapSpec via `prompt::complete`
      (reuses the `--enhance` stack, same as compile). Built-in system prompt; the
      3-stage robustness fallback (strip fences → retry → minimal spec, never abort).
- [ ] **`--map-spec` / `--map-dump-spec` / `--map-cache`** — load a committed spec
      (skip the LLM), dump the parsed spec, SHA-256 cache.
- **Gate:** `plakat map "…" --map-dump-spec` produces valid JSON for ≥5 prompts incl.
  a Tier-4 and an urban; a committed `corpus/map/island.spec.json` loads via `--map-spec`.

## MAP-2 — layered geography engine (Layers 0–7)

- [ ] **L0 canvas + L1 tectonic** (`spade` Voronoi + `noise` fBm heightmap). Gate:
      `--map-dump-heightmap` → byte-stable PNG.
- [ ] **L2 hydraulics** (D8 flow + the **breach algorithm** — spike-first with a
      Tier-4 "river must reach the sea" test, the highest-risk component).
- [ ] **L3 coastline** (marching squares + `rstar` index) + **L4 biome** + **L5
      landmark resolver** (topological sort over the anchor graph; all anchor types
      resolve or error on a cycle).
- [ ] **L6 infrastructure** (`petgraph` Dijkstra roads) + **L7 conditioning assembly**
      (heightmap/biome/feature-overlay full-canvas + per-tile crops).
- **Gate:** committed `island.spec.json` → byte-stable `--map-dump-{heightmap,biome,
  features}` PNGs; anchor-resolution unit tests (`mouth_of`, `bearing+constraint`,
  `pass_between`). **Determinism invariant**: no unseeded map output (render twice,
  identical bytes).

## New deps (pure Rust, gate behind a `map` feature if heavy)

`spade`, `ndarray`, `geo`/`geo-types`, `rstar`, `petgraph`, `noise` (geometry);
`ab_glyph`, `unicode-bidi`/`-normalization` come with the 1.5.0 label compositor.

## Later in Track M (see the plan)

1.5.0 **linework render** (labels + furniture on the feature overlay — a complete
map, NO SD; memory-wall-free) + GeoJSON/SVG export. 1.6.0 tiled SD render (1×1
on-box; multi-tile memory-bound). 1.7–1.8 Bund hooks + urban fabric.

## Opportunistic / debt (off the critical path)

- 1.1.0 carryovers: Flux regional (Flux broken on Metal → code-only), IC-Light (L).
- COMPILE-1 `map:` block (E-C4) — unblocks once `plakat map` exists.
- Memory-bound render debt: SD3.5 DreamBooth render, `regional.sh sdxl/sd35`.
