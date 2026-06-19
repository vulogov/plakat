# Plan: `plakat map` + `plakat compile` — two-track delivery across 1.x

**Status:** Accepted plan (response to the v0.50–v0.51 Master RFC)
**Re-scoped onto the 1.x track** (the RFC predates 1.0; its v0.50/v0.51 numbers are
remapped to 1.2.0+).
**Date:** 2026-06-19

This document splits the Master RFC into two independent delivery **tracks**,
sequences them across several 1.x releases each gated by a **committed corpus
proof**, reviews what the RFC proposes (strengths + risks), and proposes
**extensions**. The single most important re-framing — learned the hard way this
cycle — is the **memory wall** (see §3.1): every track is planned so that progress
is provable on a 24 GB box *without* depending on a heavy SD render.

---

## 1. The two tracks

| Track | Feature | RFC parts | Risk | GPU | New deps |
|---|---|---|---|---|---|
| **C — Compile** | `plakat compile` (+ Tera pre-pass) | II, III (COMPILE-1/2) | Low | None | `tera` (gated), `serde_json` |
| **M — Map** | `plakat map` | I (MAP-1…5) | High | Heavy (tiled SD + ControlNet) | `spade`, `ndarray`, `geo`, `rstar`, `petgraph`, `noise`, `ab_glyph`, `unicode-bidi/-normalization` |

**Why this split.** They share *nothing* in code (the RFC confirms `src/compile/`
and `src/map/` are disjoint) — only the LLM provider stack (`src/llm/`, what the RFC
calls the "`--enhance` stack") and the Bund VM. So they can be developed in parallel
and **shipped interleaved**, one feature per release.

**Track C ships first.** It is self-contained, needs no GPU, carries one optional
pure-Rust dependency, and delivers immediate workflow value (write prose → get a
scenario). It is also the lower-risk way to exercise the shared LLM-reuse pattern
both features depend on. Track M is a long arc whose hardest parts (geometry
correctness, the breach algorithm, DCEL block subdivision, the memory-bound render)
benefit from Track C having already de-risked the LLM plumbing.

---

## 2. Release sequencing (each line ships when its corpus proof is committed)

> Numbers are proposals; the **gates** are the contract. We are on `1.2.0` now.

### Track C — Compile

| Release | Scope | Corpus gate (committed, deterministic) |
|---|---|---|
| **1.2.0** | COMPILE-1: parser → resolver (model-family) → assembler → emitter (JSON→HJSON post-pass, **OQ-COMPILE-4**) → cache → positive+negative LLM → `--diff`, `scenario -` stdin | `corpus/compile/basic.txt` → `--no-enhance` produces a **byte-stable** HJSON (committed); a second proof runs the live LLM and asserts *structure* (N scenes, every scene has positive+negative, family profile applied) not exact text |
| **1.3.0** | COMPILE-2: Tera pre-pass behind `--features templates`; filters/functions; file+line error citations; `--dump-rendered-only` | `corpus/compile/series.tera` + `--vars series.json` → `--dump-rendered-only` produces a **byte-stable** `prompts.txt` (committed); piped into `compile --no-enhance` → stable HJSON |

### Track M — Map (geometry is provable on-box; SD render is the memory-bound capstone)

| Release | Scope | Corpus gate (committed, deterministic) |
|---|---|---|
| **1.4.0** | MAP-1 (MapSpec v2 + Anchor + LLM parser + cache) **and** MAP-2 (Layers 0–7 geometry engine). **No SD.** | Commit `corpus/map/island.spec.json` (a hand-written spec — decouples from the LLM). `plakat map --map-spec island.spec.json --map-dump-{heightmap,biome,features}` → three **byte-stable** PNGs (committed). Anchor-resolution unit tests: `mouth_of`, `bearing+constraint:coastline`, `pass_between` land where claimed. |
| **1.5.0** | **MAP-3 linework render** (the §4 E-M1 extension): label compositor + cartographic furniture rendered onto the feature overlay — **a complete, useful map with NO SD inference**. + GeoJSON/SVG export (E-M4). | `--map-spec island.spec.json --map-style linework` → a committed labelled vector-style map PNG (deterministic, CPU). BiDi proof: an Arabic spec renders RTL. This is the release that makes `map` useful **without** the memory wall. |
| **1.6.0** | MAP-3 **SD render**: tiled ControlNet (depth+canny) inference + Hann stitch + feature overlay on top. | **1×1 tier** (single tile) SD map renders on the dev box → committed proof. Larger tiers (≥3×3) are **memory-bound** on 24 GB and recorded CANNOT-VERIFY-here (like sd35) — verify on a bigger box / CI with more RAM. |
| **1.7.0** | MAP-4 (Bund hooks at every layer boundary) + MAP-5 **phase A** (urban: morphology → street network → DCEL blocks). | `--map-script` example mutates a river terminus → reflected in the dumped features (deterministic). `--map-dump-urban` shows a street grid + non-degenerate blocks for a committed settlement spec. |
| **1.8.0** | MAP-5 **phase B** (waterfront, transit routing, notable buildings, fabric fill, street-label-along-path, urban Bund vocab). | `--map-dump-urban` proof: piers extend into water, tram follows the street graph, walls close with gates at road entries — all on a committed settlement spec, all CPU-dumpable. |

**Carryovers from the current 1.2.0 roadmap** (Flux regional, IC-Light) drop to
*opportunistic* — slot them into a release only if a cycle has slack; they are not on
the critical path of either track.

---

## 3. RFC review — strengths and risks

### Strengths (keep these)

- **Anchor-based positioning.** Expressing every object as a typed spatial relation
  (`MouthOf`, `Bearing{constraint: Coastline}`, `PassBetween`) instead of pixels is
  the right abstraction: it is what an LLM can actually emit reliably, it is
  scale-independent, and it makes the geometry engine a *constraint resolver* (a
  topological sort over the anchor graph) rather than a layout hack. This is the
  best idea in the RFC.
- **LLM-as-optional, spec-as-truth.** `--map-spec` / `--compile` caching means the
  reproducible artifact is the *spec/prompts file*, not the LLM call. This is exactly
  what makes corpus verification possible (see §5).
- **Staged delivery with dump-based gates.** `--map-dump-*` lets every geometry layer
  be confirmed before the renderer exists. The RFC already has the right shape here.
- **Provider/Bund/feature-gate reuse.** Confirmed real: `src/llm/`, `src/scripting/`,
  the Cargo feature pattern all exist.

### Risks the RFC under-weights

#### 3.1 The memory wall (the dominant risk — RFC does not mention it)

The map render does **tiled SD with ControlNet at internal resolutions up to
16384²**. This cycle we hit hard OOM ceilings on a 24 GB Mac for *single* renders:
SD3.5 (OOMs at the LoRA merge even at 512²/CPU), SDXL 1024², even SD 2.1 768². A
multi-tile ControlNet map render is **strictly heavier** than any of those.

**Consequence for the plan:** the SD render must not be on the critical path of
*proving the feature works*. The plan above pushes all geometry + a complete
**linework** map (§4 E-M1) ahead of the SD render, so `map` is demonstrably useful by
1.5.0 with zero SD inference. The SD render (1.6.0) is verified only at the **1×1
tier** on-box; multi-tile is honestly marked memory-bound, exactly as we did for the
SD3.5 DreamBooth render. The RFC's acceptance criterion "tile seams not visible on a
6×6 render" is **not** a dev-box gate — it is a bigger-box / CI gate.

#### 3.2 Tiling model is *not* the hi-res tiled path

The RFC says MAP-3 "Reuses `src/generate/tiled.rs` directly." But `tiled.rs` is the
**base-anchored MultiDiffusion hi-res refine** (which I rebuilt this cycle precisely
because anchorless latent blending produced *global incoherence*). Map tiling is a
**different model**: each tile is an *independent* SD generate conditioned on its own
ControlNet crop; global coherence comes from the **shared conditioning images**
(heightmap/canny), not from latent overlap. The only reusable piece is the *image-space
Hann blend* of the finished tiles. The plan should reuse the blend helper and **not**
route map tiles through the latent MultiDiffusion path. Flag this explicitly in MAP-3.

#### 3.3 ControlNet availability + Flux fallback (OQ-MAP-1/2)

Map leans entirely on depth+canny ControlNet. Confirm the existing `controlnet.rs`
path runs at tile sizes on Metal for SD1.5/SDXL before committing MAP-3. Flux has no
CN in plakat (and is broken on Metal anyway) → for `--model flux*`, **error with a
suggestion**, do not silently img2img-fallback (a low-strength heightmap img2img
produces a fundamentally different, worse result; silent degradation reads as a bug).

#### 3.4 MAP-5 (urban) is its own multi-release effort, not a phase

Urban fabric is eight sub-systems (morphology, DCEL blocks, waterfront, four transit
kinds, notable buildings, fabric fill, path-following labels, a whole Bund vocab). It
is comparable in size to MAP-1–4 combined. The plan gives it **two** releases (1.7,
1.8) and still treats DCEL robustness (OQ-URBAN-4) and street-text-along-path
(OQ-URBAN-2) as the gating risks. Do not bundle urban into the map MVP.

#### 3.5 Highest-risk algorithms — prototype standalone first

- **Breach algorithm (Lindsay 2016, OQ-MAP-7)** for continental drainage. The RFC
  already flags it; the plan makes it a **spike inside 1.4.0** with its own unit test
  (a Tier-4 spec whose river must reach the sea, not pool inland) *before* Layer 2 is
  declared done.
- **DCEL planar block subdivision** on organic street graphs (near-parallel edges,
  T-junctions, coincident nodes). Gate 1.7.0 on a snap-tolerance pass + a
  "no degenerate faces" property test.

#### 3.6 Determinism (corpus depends on it)

Every byte-stable corpus proof requires the geometry to be a pure function of
`(spec, seed)`. The RFC's noise/Voronoi/Dijkstra are seedable, but the plan must make
"**no `Math.random`, no unseeded HashMap iteration order in geometry output**" an
invariant with a test (render twice, assert identical PNG bytes). This is OQ-URBAN-1
promoted to a track-wide rule.

---

## 4. Proposed extensions

### Compile

- **E-C1 `--decompile` (scenario HJSON → prompts.txt).** The inverse pass. Lets users
  round-trip an existing scenario into the friendly editable format. Cheap; high
  ergonomic value; makes the format a true two-way bridge.
- **E-C2 `--lint` (no LLM).** Validate a `prompts.txt`: unknown commands, bad merge
  (e.g. `skip:` in the global block — OQ-COMPILE-1), empty blocks, model typos. Pure,
  instant, and a perfect deterministic corpus gate.
- **E-C3 Cost/Token estimate in `--dry-run`.** Extend the existing dry-run with a
  per-provider token + (where known) cost estimate, so a 200-scene book run is
  predictable before spending.
- **E-C4 `map:` block type → Track bridge.** A block whose body is a world description
  and whose command is `map:` compiles to a `plakat map` invocation in the scenario
  (closes OQ-MAP-6 and literally joins the two tracks). Lands after both features exist.

### Map

- **E-M1 Linework / vector render (`--map-style linework`, no SD).** *The* key
  extension. The geometry engine already produces a labelled feature overlay; rendering
  it as a clean cartographic linework map (parchment fill + ink linework + labels +
  furniture) is a **complete, shippable map with zero SD inference**. It (a) sidesteps
  the memory wall, (b) is fully deterministic/corpus-friendly, and (c) is a genuinely
  desirable output (crisp vector-style maps, not just AI-painted ones). This is why
  1.5.0 ships before the SD render.
- **E-M2 GeoJSON + SVG export (`--map-export geojson|svg`).** The `geo` crate gives
  GeoJSON for free; SVG is a small writer over the same polylines. Maps become
  **editable vector data** (open in Inkscape/QGIS), not just raster. Pure-Rust,
  on-box-verifiable, and a strong differentiator.
- **E-M3 Real-terrain import (`--map-heightmap dem.png`).** Skip the procedural Layers
  1–2; feed an external DEM/heightmap and run Layers 3–7 (coastline, biome, render) on
  real terrain. Reuses 80% of the engine for a whole new use case.
- **E-M4 Map → `compose` / artefact.** A rendered map is just a PNG — it already feeds
  the new `compose` `load:`/`generate:` layers (a map on a table, a torn map fragment).
  Document the recipe; no new code.
- **E-M5 Cartography-LoRA auto-pick.** For the SD render, reuse the existing
  `--smart-discovery` (Civitai LLM judge, shipped v0.46) to choose a map/isometric
  LoRA per `--map-style`. Ties the new feature to existing infrastructure.

---

## 5. Corpus verification strategy (the through-line)

The hard lesson of the 1.1.0 cycle: **separate the deterministic, the non-deterministic,
and the memory-bound.** Applied here:

| Layer | Deterministic? | On-box? | Corpus role |
|---|---|---|---|
| `prompts.txt` / `MapSpec` (committed) | yes (it's an input) | yes | the **reproducible source of truth** |
| Parser / assembler / geometry / linework | **yes** (pure fn of spec+seed) | yes (CPU) | **byte-stable committed proofs** — the spine of every gate |
| LLM enhancement / parse | no (provider variance) | yes | assert **structure**, not exact text; `--no-enhance` / `--map-spec` give a deterministic path |
| Tiled SD render (≥3×3) | seed-stable but heavy | **no** (OOMs 24 GB) | **capstone**, verified at 1×1 on-box; larger = documented memory-bound debt |

Net: by **1.3.0** Compile is fully shipped and corpus-proven; by **1.5.0** Map produces
a complete, corpus-proven *linework* map with no GPU; the SD render is the only piece
that inherits the memory caveat, and it inherits it honestly.

---

## 6. Open questions — dispositions

- **OQ-MAP-1 (CN auto-download):** yes, auto-pull like Real-ESRGAN; gate behind a
  one-line "downloading ControlNet…" spinner. **OQ-MAP-2 (Flux fallback):** error +
  suggest an SD/SDXL model; no silent img2img. **OQ-MAP-3 (font):** Noto Serif OFL-1.1
  is redistributable — bundle it; OFL is compatible. **OQ-MAP-4 (schema v):** keep
  `version` and a best-effort v1→v2 upgrader that warns, never hard-fails.
  **OQ-MAP-5 (non-square furniture):** anchor furniture to output corners, not tile
  grid. **OQ-MAP-6 (scenario `map:`):** ship as **E-C4** after both tracks land.
  **OQ-MAP-7 (breach):** spike-first inside 1.4.0 (§3.5).
- **OQ-COMPILE-1:** `skip:` in global → parse error (caught by **E-C2 `--lint`**).
  **OQ-COMPILE-4 (HJSON emit):** confirmed needed — `deser-hjson` is deserialize-only;
  write the ~100-line JSON→HJSON comment/multiline post-pass (1.2.0). **OQ-COMPILE-5
  (rate limits):** reuse the existing enhance backoff.
- **OQ-TEMPLATE-1:** rename `include_prompts` → `include_raw` (clearer that it does not
  re-render). **OQ-TEMPLATE-4 (`--vars-env` secrets):** document the leak; never log
  rendered output at info level. **OQ-TEMPLATE-6 (style order):** **scene-after-global**
  is correct (later = stronger signal); keep as specified.

---

## 7. What ships when (summary)

```
1.2.0  Compile core            (Track C)  ── byte-stable HJSON proof
1.3.0  Compile + Tera          (Track C)  ── byte-stable rendered-prompts proof
1.4.0  Map spec + geometry     (Track M)  ── byte-stable heightmap/biome/feature dumps
1.5.0  Map LINEWORK render     (Track M)  ── complete labelled map, NO SD  ★ memory-wall-free
1.6.0  Map SD render (tiled)   (Track M)  ── 1×1 on-box; multi-tile = memory-bound
1.7.0  Map Bund + urban A      (Track M)  ── street grid + DCEL blocks
1.8.0  Map urban B             (Track M)  ── waterfront/transit/notable/fabric/labels
```

Compile is done and proven by 1.3.0. Map is *useful and proven* (linework) by 1.5.0,
with the SD render and urban fabric as honestly-caveated follow-ons.
