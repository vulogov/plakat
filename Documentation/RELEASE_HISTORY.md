# plakat — release history

"What's new" sections for v0.13 through 1.12. The current
release's notes live in the [main README](../README.md). Older
cycles are archived here so the README stays focused on what's
new this turn.

For commit-level history see `git log`; for migration notes the
per-cycle commits carry the rationale + before/after.

## What's new in 2.2.0 — every model, end-to-end regression-gated

2.0.0 built `plakat verify`; 2.1.0 used it to fix a real caption bug. 2.2.0 takes the harness
to **full coverage** — every rendering model now has a whole-pipeline regression gate, and
that gate is cheap enough to run in CI.

**End-to-end gates for all six rendering models.** The Tier-2 perceptual gate — render a
fixture through the *entire* real pipeline and compare to a frozen reference image — now
covers SD 1.5, SD 2.1, SDXL, PixArt-Σ, Stable Cascade, and SD 3.5, across **four different
rendering architectures**. If any change ever perturbs a model's output, the gate catches it.

**Finishing the coverage found two more determinism bugs** — exactly what the harness is for:
SD 3.5's text-to-image path and Stable Cascade's (stochastic) sampler weren't fully
reproducible; both are now pinned. A third prompt fixture (tokenization edge cases — numbers,
punctuation) rounds out the per-module checks at three prompt lengths.

**Cheaper CI.** The weight-backed verification job now caches model weights across runs, so
after the first run it's just build + inference — practical enough to gate a release on. The
zero-download structural tier still runs on every push.

Day-to-day use is unchanged — this is a confidence release: the same one loop, now with a
safety net under every model. See [`VERIFY.md`](VERIFY.md).

## What's new in 2.1.0 — the harness pays off: a real caption bug, fixed

2.0.0 shipped `plakat verify`, a self-contained model-correctness harness. 2.1.0 is that
harness earning its keep: it found and fixed a **real, image-affecting bug** in PixArt-Σ and
SD 3.5 — and grew broader coverage across the board.

**The bug: unmasked T5 captions.** PixArt and SD 3/3.5 encode your prompt through a T5 text
encoder, then pad it to a fixed length. plakat wasn't masking those padding tokens — so the
actual prompt words were attending to *hundreds* of pad tokens inside T5, drifting the
caption. Measured against the reference, the caption for your real words was only **~70%
correlated** with the correct output; every PixArt / SD 3.5 image was subtly off-prompt.

2.1.0 masks the pad tokens in T5 self-attention **and** stops image tokens from attending to
pad-position captions in the DiT cross-attention — matching the reference **exactly** (corr
0.70 → 1.0). Your PixArt and SD 3.5 prompts now land as intended. Nothing about how you use
plakat changes; the output is just correct.

**Broader verification.** The harness grew a second prompt fixture (conditioning is now
checked at two very different prompt lengths), a new PixArt tap (the final-adaLN timestep),
and **end-to-end** perceptual regression gates for **SDXL and PixArt** on top of SD 1.5 — a
whole-pipeline safety net across two rendering families. It also learned its own limits: a
full-DiT tap and an SD 1.5 mid-block tap were evaluated and honestly dropped (documented) as
not worth their cost.

**Still pure Rust, still self-contained.** No new dependencies; `plakat verify` fetches its
golden reference tensors from Hugging Face exactly like model weights. See
[`VERIFY.md`](VERIFY.md).

## What's new in 2.0.0 — `plakat verify`: proof the models are right

Every prior cycle has, at some point, rescued a *silently-wrong* model by hand-comparing
plakat's math against a reference implementation. 2.0.0 turns that method into a
**committed, repeatable subcommand** — and it earned its keep by catching a real SDXL bug
on its very first run.

```bash
plakat verify                        # run the whole harness
plakat verify --tier 0               # structural checks — zero downloads, ~0.2s
plakat verify --tier 1 --model sdxl  # per-module correctness vs a frozen reference
```

**A self-contained correctness harness.** `plakat verify` checks that each model's *real*
internals match a frozen reference — the text encoders, the pooled conditioning, the VAE,
and the **denoiser / transformer core** of all seven families (SD 1.5, SD 2.1, SDXL,
PixArt-Σ, Stable Cascade, SD 3.5, AnimateDiff). Three tiers: **structural** (no downloads —
a fast, always-on gate), **per-module** (correlation vs golden tensors), and **end-to-end**
(render a fixture, compare to a golden image). Every core tap matches the reference at
correlation 1.0.

**Pure Rust — nothing new to install.** The shipped binary never touches Python or torch:
it fetches the golden reference tensors from Hugging Face exactly like it fetches model
weights. The reference authoring (which *does* use diffusers) runs offline, once, and is
excluded from the crate. plakat stays self-contained.

**It found a real bug.** On its first run it flagged SDXL's `clip.encoded` at correlation
0.991: SDXL's CLIP-L text encoder was padding with the wrong token (`"!"` instead of
end-of-text). Fixed — every SDXL (and SD 3.5) render now matches the reference exactly.

**Wired into CI.** Every push runs the structural tier; a one-click Actions job runs the
full weight-backed verification. See [`VERIFY.md`](VERIFY.md) for the operator guide and
[`RFC_VERIFY.md`](RFC_VERIFY.md) for the design.

## What's new in 1.22.0 — power-user polish + a source-wide stability pass

1.22.0 finishes the power-user loop **and** hardens the whole codebase: a six-part
source audit found ~45 issues, and this release fixes **40 of them** (plus one
mitigated) — security, host-crash, silently-wrong-output, and input-driven-panic
bugs across the UI, pipelines, scenario runner, and training.

```bash
plakat ui            # the interactive terminal UI
```

**Power-user loop**
- **Per-model generation size** — **`/size 1024x768`** (or `native`) sets a size each model
  remembers, budget-guarded against free RAM. **`/steps`** / **`/cfg`** override denoise
  steps / guidance; all three fold into named **presets**.
- **`/vary` into presets** — a preset now carries the model + LoRA stack + negative + size +
  steps + guidance.
- **Undo / redo** — **`Ctrl-Z`** / **`Ctrl-Shift-Z`** step the live image back / forward
  through the session history.
- **Faster scenario runs** — a matching all-SD scenario **reuses the loaded Chat model** (no
  reload) when it's provably the same base.

**Stability pass — the headline fixes**
- **Security** — the Civitai downloader could be tricked into overwriting arbitrary files
  (path traversal) by a hostile file name; now confined to the cache.
- **Won't crash your host, won't false-abort** — the OOM watchdog now catches a fast
  single-buffer allocation *and* only aborts when plakat itself is the memory culprit
  (no more self-terminating on another app's pressure).
- **No more silent garbage** — fixed **SDXL AnimateDiff** (every frame past the first paired
  with the wrong prompt → incoherent video), regional-SDXL conditioning, LoKr/Flux/SDXL
  pooling, and more.
- **No more input-driven panics** — deeply-nested prompts, `steps=0`, oversized maps, and a
  dead render thread used to crash or wedge the UI; all handled cleanly now.
- **Safer training** — atomic checkpoint writes (a mid-write OOM can't destroy your only
  LoRA) + an LR warm-up on `--resume`.

Everything is still one loop, and every output is a normal plakat PNG (recipe embedded)
that a compiled scenario runs headlessly.

> Inline images use the terminal's graphics protocol (Kitty/Ghostty/WezTerm/iTerm2/Sixel);
> the UI runs without one (placeholders). It's behind the default-on `ui` feature. Flux in
> the UI is postponed until it can be verified on capable hardware.

See [`BUGFIX_PLAN.md`](BUGFIX_PLAN.md) for the full audit + every fix, and
[`Tutorials/UI_TUTORIAL.md`](Tutorials/UI_TUTORIAL.md).

## What's new in 1.21.0 — the build-an-image loop + power-user workflow

1.20.0 made `plakat ui` memory-aware; 1.21.0 sharpened the **loop**: the Chat title shows
the live mode (`evolve · seed …` / `anchored 0.60` / `inpaint` / `identity` / `fresh`) and
**Ctrl-T** toggles evolve ↔ anchored; **/vary N** fans out variations at fresh seeds into
the filmstrip; **/scenario** grabs an image's exact recipe (prompt+negative+seed) into a
Scenarios task; **named presets** (`/preset save`) snapshot the model + LoRA stack +
negative; and an **F1 cheatsheet** + ambient free-RAM/loaded-model status readout.

## What's new in 1.20.0 — memory-aware `plakat ui`

On 24 GB unified memory a big model can pin RAM to 100%. 1.20.0 made `plakat ui`
**memory-aware**: it warns before a load that won't fit, gives idle memory back on its
own, and can hand the whole GPU pool back in one keystroke.

- **Memory-budget warning before a load** — Models `[L]` estimates the model's resident
  footprint (exact from the on-disk cache, else a per-family guess) vs free RAM; if it
  would over-commit, a confirm modal appears instead of firing a multi-GB download+load
  that may hard-OOM. Reloading the resident model never prompts.
- **Idle auto-unload + resume** — after 10 min of no keypresses the model unloads to return
  its memory; the next keypress reloads it in the background (LoRAs intact).
- **Hard reset (free all GPU memory)** — palette → *Restart plakat* → confirm → plakat
  re-execs a fresh process, the only way to fully return candle's Metal buffer pool.
- **Cache doctor** — palette on Models → sweeps stale download locks + reports cached weight
  GB / partial / not-cached + a gated hint.

Also cleaned every build warning (release + test) by *using* the dead fields.

## What's new in 1.19.0 — semantic search + a calmer, clearer UI

1.18.0 made `plakat ui` sturdy; 1.19.0 added **semantic search**, hardened it against
out-of-memory crashes, and fixed two confusing spots in the build-an-image loop.

- **Semantic search in History** (`?`) — rank every image by *relevance* to a query
  (TF-IDF cosine over filename + tags + recipe), most-related first. "snowy peak"
  surfaces "a mountain in winter, fresh snow" with no shared substring. Local + instant,
  no model or network. (`/` stays the plain substring filter.)
- **Canvas shows the image** — the mask grid renders the **current** picture (each cell
  tinted with that region's colour), so you can see *where* you're painting; painted cells
  get a bright-green overlay. The Canvas masks + inpaints the **latest** render, so you can
  **gradually build an image** — mask → inpaint → mask again on top — model-agnostic.
- **Add-an-object nudge** — the first time you type "add a …" in Chat, a tip points you at
  Canvas inpaint (prompt-evolve re-describes the whole scene and can't reliably *insert*
  an object).
- **OOM hardening** — the memory watchdog now covers **all** `plakat ui` generations, so an
  out-of-memory abort exits plakat cleanly instead of crashing the host — and it restores
  your terminal on the way out.

## What's new in 1.18.0 — `plakat ui`, sturdier

1.17.0 made the `plakat ui` terminal UI fast and pleasant; 1.18.0 made it **sturdy**:

- **In-process runner — no double-load OOM** — scenario runs + People quick-gen run on
  the model thread, which frees the loaded Chat pipeline first, so only one model is
  resident on unified memory.
- **Identity-preserving Chat continuation** — a continued portrait keeps its face: each
  refine re-runs the person's IP-Adapter pass (same reference photos + seed).
- **People auto-encode + invalidation** — the `E` encoding-quality score computes on
  first ENCODING-tab view and re-scores on a ref/strategy change (fingerprint sidecar).
- **Download manager, complete** — `U` Civitai version-update detection + ≤2 concurrent
  downloads (FIFO queue) + range-resume, alongside 1.17.0's SHA-256 verify.

## What's new in 1.17.0 — `plakat ui`, polished

1.16.0 broadened the `plakat ui` terminal UI; 1.17.0 made it *fast and pleasant*:

- **Command palette** (`Ctrl-K`) — a fuzzy action launcher on every screen.
- **`@mention`** people and LoRAs inline in the Chat prompt.
- **Chat sessions + filmstrip** — scrub frames (`Ctrl-←/→`), roll back (`Ctrl-B`), vary
  (`Ctrl-Y`).
- **History thumbnail grid** (`v`, lazy + LRU) + background decode + `/` filter + recipe
  compare.
- **People depth** — six DETAIL sub-tabs, **`E` encoding-quality score** (SCRFD + ArcFace
  ref consistency), mixed-family multiperson.
- **Tera mode** (`Ctrl-T`) with a live variable panel; **Canvas** face-aware `B` + `g`
  finer-grid masks.
- **LoRA Hub** — 24h assessment caching, two-stage HF pre-filter, SHA-256-verified
  Civitai downloads.
- **Fixes** — scenario runs no longer corrupt the TUI or hide outputs from History; the
  Prompt Workspace crash fixed; a new scenario is runnable as-is + nameable (`Ctrl-R`).

## What's new in 1.16.0 — `plakat ui` grows up

1.15.0 shipped the eight-screen `plakat ui` terminal UI; 1.16.0 paid down its depth
deferrals — more models, smarter refinement, and the workflow glue that turns
*exploring* into *producing*.

- **More models in the UI** — SD3 / 3.5, PixArt-Σ, and Stable Cascade load + generate in
  Chat alongside SD-family; Chat refine / Canvas inpaint get live preview + mid-denoise
  cancel (StepHook-wired img2img); `/auto` LLM edit-vs-new routing.
- **LoRA Hub completeness** — per-LoRA weights (`+`/`-`, `★w`), `Ctrl-R` LLM stack
  suggestions, HuggingFace base-model markers, 1h search caching, `●` download indicator.
- **Chat → Scenario** — `Ctrl-G` distills the Chat thread into a scenario task.
- **People** — `I` imports a scenario persona into the editable library; `Del` is the
  right to be forgotten (type-name confirmation).
- **History** — `/` filter (filename / tags / recipe), `T` tag, `X` export, `d` recipe
  compare.
- **Canvas** — `M` outpaint. **Prompt Workspace** — `Ctrl-Tab` cycle, `Ctrl-N` new.
  **Chat sessions** — `/save` · `/load` · `/sessions`.

Flux in the UI was postponed (hardware).

## What's new in 1.15.0 — `plakat ui`, the interactive terminal UI

plakat has always been a powerful but flag-heavy CLI. 1.15.0 added **`plakat ui`** — a
full-screen, keyboard-driven terminal application over the *same* engine. Load a model
once and **talk** to it: conversational generation + refinement with inline images,
browse everything you've made, drop in a specific person, paint an inpaint mask, search
and apply LoRAs, compile prose into scenarios — eight integrated screens (RFC TUI-1).

- **Chat** — type a prompt, the image renders inline; the next prompt **evolves** it
  (accumulated prompt at a stable seed). Commands: `/new` `/negative` `/enhance`
  `/strength` `/seed`, `Ctrl-P/N` prompt recall, a 2-line wrapping editor.
- **Models** — load/unload with live RAM + swap gauges and rerouted progress.
- **Scenarios** — browse, edit (built-in editor), run batch jobs with a per-task board.
- **History** — every output under `out/`, by date, with its embedded recipe; `C`
  continues any image in Chat.
- **People** — an identity library (incl. personas read from scenarios); `G` makes a
  portrait, or mark two+ for a multiperson scene.
- **LoRA Hub** — LOCAL (family + compatibility, `A` applies) + CivitAI/HuggingFace
  search/download + LLM assess / recommend.
- **Prompt Workspace** — prose → live structural compile; `Ctrl-R` LLM compile;
  `Ctrl-O` → Scenarios editor.
- **Canvas** — paint a coarse inpaint mask (presets); `Enter` hands Chat a full-res mask.

The UI loaded SD-family models at 1.15.0; SD3/PixArt/Cascade in the UI plus the depth
features landed in 1.16.0.

## What's new in 1.14.0 — every feature, every surface

1.12 shipped **multiperson + face-swap** and 1.13 shipped the **map coastlines & worlds** —
both as CLI-only. 1.14.0 **productizes** them: people-in-scene and the new map features
become first-class in the automation surfaces (**scenario / compile / scripting**), each
dispatching the *same* pipeline as the CLI, so a given spec renders identically everywhere.

- **Multiperson everywhere** — a `type: multiperson` scenario task (scene + `people` that
  reference the top-level `personas` by name + identity mode swap/composite/pose/harmonize)
  and a `plakat.multiperson` scripting word, both building the same request as the CLI.
- **Maps in the automation surfaces** — multi-tile worlds from scenario/scripting
  (`plakat.map.tiles`), per-tile furniture (`--map-tile-furniture`), political territory
  **polygons** in GeoJSON/SVG, and prose-parser coastal words (`peninsulas`/`inlets`/`fjords`).
- **Confidence** — a comprehensive `corpus/map.sh` demonstrates every map feature byte-stably.
- **Fixes** — river deltas no longer draw across open ocean (the fan forms land-side).

(1.14.1 followed: `candle-onnx` made opt-in behind an `onnx` feature so `cargo install`
and release builds don't require `protoc`.)

## What's new in 1.13.0 — a steadier host and a richer world map

1.13.0 pays down the **memory & stability** debt that had been deferred twice, and fills
out **`plakat map`** with the coastline + world-scale features it was missing.

- **Memory & stability** — a render-size guard before Metal's single-buffer OOM; OOM-guard
  tuning (sustained window 3→5 inference, 12 training; `PLAKAT_OOM_GUARD_SUSTAINED`); gradient
  checkpointing investigated + documented as a dead end on candle 0.10.2.
- **`plakat map` coastlines & worlds** (deterministic, byte-stable, no GPU) — peninsulas,
  inlets, fjords (`terrain.{peninsulas,inlets,fjords}`); multi-tile world maps
  (`--map-render-tiles DIR` + `--map-tiles CxR`); marsh hatching; river deltas; political
  layer in GeoJSON/SVG export; seasonal palette on the painted path.
- **`plakat multiperson`** — `--scale LABEL:0.7` sizes a child/teen persona's `--pose`
  skeleton shorter.

## What's new in 1.12.0 — put specific people into a scene (`plakat multiperson`)

1.12.0 places specific people into a generated scene, with a complete,
numerically-verified face-swap stack ported to candle from InsightFace's ONNX
models, plus a `convert-onnx` command to build the weights yourself.

- **`plakat multiperson`** — give each person a photo + a relative location
  (`--at "alice:left closer front"`). `--swap` face-swaps each figure (with `--pose`
  pinning an OpenPose skeleton per region so the right face lands on the right figure);
  `--composite` mattes the real photos in (exact identity, any model).
- **Verified face-swap stack** — SCRFD-500MF (~1–3 px), ArcFace `w600k_r50` (cosine 1.0),
  `inswapper_128` (1e-5), all checked vs onnxruntime; `plakat convert-onnx` builds them.
- **Three never-verified components fixed** — SCRFD architecture, ArcFace stem PReLU, and a
  180°-rotation bug in the face-alignment transform (also lifts `--identity faceid`).
- Honest scope: identity is strongest on few, prominent, roughly-frontal faces from photos.

## What's new in 1.11.0 — relighting, richer maps, and a steadier host

1.11.0 adds **IC-Light relighting**, fills out the **`plakat map`** generator with the
terrain + cartography features it was missing, and hardens the CLI against concurrent
runs and broken `pull`s.

- **IC-Light relighting** — `plakat relight <subject> --prompt "<lighting>"` re-illuminates
  a foreground subject from a text description while preserving identity (SD 1.5: widened
  4→8-channel input conv + `lllyasviel/ic-light` offset; low guidance 1.5–3).
- **`plakat map` geography** — dry canyons, plateaus/mesas, a political layer, seasonal
  palettes (`--map-season`), a tabletop grid (`--map-grid N`), comment-friendly HJSON specs,
  natural lake/coast irregularity (`terrain.erosion`).
- **CLI robustness** — single-instance guard (a second heavy run refuses while one is busy;
  `--enable-multiple-instances` overrides); `models pull` rewritten to enumerate a repo's
  actual files (PixArt / SD3 / Cascade / single-file / `civitai:N` all pull).

## What's new in 1.10.0 — train your own style on every model family

1.10.0 is the **model-training expansion**: `plakat` learns a style (or subject) on
**four** model families, closing the LoRA / Textual-Inversion gaps. Each trainer freezes
the base and learns only the adapter / embedding, saving a file the matching `--lora` /
`--embedding` loader reads.

- **SD 2.1 — style LoRA + DreamBooth** *(verified on-box)* — 1024-dim CLIP UNet config +
  **v-prediction** loss. `style train --base sd21`.
- **PixArt-Σ — style/subject LoRA** — DiT attention → trainable `LoraLinear`; DDPM-ε
  through the frozen DiT (BF16 T5). `style train --base pixart`.
- **SD 3.5 — Textual Inversion** — a token learned in **all three** encoders (CLIP-L +
  CLIP-G + T5) via differentiable splice; rectified-flow loss through the frozen MMDiT;
  triple embedding. Required a faithful vendored T5 (proven byte-identical by a guard
  test). `embedding train --base sd35`.
- **Stable Cascade — Stage-C LoRA** — trains in the Würstchen semantic space; shipped the
  missing **effnet encoder** (image → 16×24×24 latent, verified on real weights).
  `style train --base cascade`.

The reusable `LoraLinear` + `install_train_adapters` spine now spans the MMDiT, PixArt
DiT, and Cascade Stage-C. The three transformer trainers are **memory-bound** (≥ 36 GB
unified / CUDA); SD 2.1 runs on 24 GB.

## What's new in 1.9.0 — map polish (lakes, lots, painted scripts, shaped labels)

1.9.0 polishes `plakat map` after the track went feature-complete in 1.8: real
lakes, river labels that follow the right channel, towns drawn at the building
scale, SD-painted maps from scripts, and non-Latin labels. The geometry stays a
**pure function of (spec, seed)** — byte-stable, no GPU for the linework path. New
this cycle: a full **[map tutorial](Tutorials/MAP_TUTORIAL.md)**.

- **Lakes are real water** — a spec lake was labelled but never drawn; now it carves
  a sub-sea-level basin so the coast/biome/hydrology pipeline renders it as a blue
  tarn with a shoreline, and rivers drain into it.
- **Named-river ↔ channel matching** — each named river labels the traced channel
  whose mouth is nearest its resolved mouth (not just the longest); GeoJSON exports
  its real id + name.
- **Town lot subdivision** — each block splits into building lots with thin lanes +
  tone variation, so towns read at the building scale.
- **`plakat.map.paint`** — a Bund word `( spec-path style -- handle )` paints a map
  via SD into an image handle, completing the `plakat.map.*` scripting surface.
- **Non-Latin labels** — a `shaped-labels` feature + `--map-font <PATH.ttf>`
  rasterizes Cyrillic / CJK labels via `ab_glyph`; the Latin bitmap font stays the
  byte-stable default.

## What's new in 1.8.0 — town maps + believable, tunable geography (`plakat map`)

1.8.0 closed the planned map track with **MAP-5, the urban fabric** — a city/town
map from a `petgraph` street graph — and made the geography realistic + tunable.

- **Town maps** — a city/town spec (`urban` block) renders a street graph: wall +
  gates, arterials, ring/grid streets, block parcels, a waterfront with piers, and
  labels at urban anchors (`at_gate`, `in_district`, `pier_tip`, `along_street`, …).
- **Configurable street plans** — `radial` (medieval), `grid` (planned), or `organic`
  (winding), via `urban.layout` / `--map-urban-layout`, or inferred from context.
- **Eroded geography** — coastlines noise-warped into bays + peninsulas, mountain
  ridgelines that wander — no more smooth-potato islands or oval ranges.
- **One erosion knob** — `terrain.erosion` (0 idealized … 1 natural … >1 rugged),
  reachable from `--map-erosion`, the scenario `map-erosion` field, and the
  `plakat.map.erosion` scripting word, each byte-identical.

## What's new in 1.7.0 — maps everywhere (`map` in scenarios, compile & scripts)

1.7.0 wired `plakat map` into the rest of plakat, so a map is a first-class step in
any batch — not just a one-off command. The three host systems — scenarios, compile,
and Bund scripting — can each emit a map, all converging on the same deterministic
render (each byte-identical to the direct `--map-render`).

- **Scenario `map` task** — `type: map` (fields `map-spec`/`map-style`/`map-paint`/
  `map-scale`/`map-sd-model`/`map-sd-lora`/…, merged scenario⊕task) → linework (no GPU)
  or SD paint, to `<out>/<name>/map.png`.
- **`map:` compile block** — a `type: map` block in a `prompts.txt` compiles to a
  scenario map task (deterministic; no LLM for the directive).
- **`plakat.map.render`** — a Bund word `( spec-path style -- handle )` renders a map
  into an in-memory image handle.

## What's new in 1.6.0 — a *painted* map (`plakat map --map-render-sd`)

1.6.0 was the map track's render capstone: run the 1.5.0 geometry through SD
img2img + a Canny ControlNet so the map looks hand-painted, then re-composite the
crisp linework + labels on top. The only GPU step on the track — the styled-base
conditioning stays a pure fn of (spec, seed); only the paint is non-deterministic.

- **`--map-render-sd`** — the styled base is the img2img init + Canny source;
  coastline/rivers/roads/labels re-composite over the paint (`--map-sd-raw` for the
  bare painting). Any model (`--map-sd-model`) with optional LoRA (`--map-sd-lora`;
  SDXL-family defaults to `Muapi/fantasy-map`, `none` disables).
- **Tiled multi-tile** (`--map-sd-tile`) — large canvases paint in overlapping
  Hann-feathered tiles, each a memory-safe full pass; pipeline + LoRA load once.
- **Broad SDXL LoRA compatibility** — a compvis→diffusers UNet key remap so
  kohya-format `lora_unet_input_blocks_*` LoRAs merge into the UNet across every
  LoRA path (generate / portrait / img2img / scenario / map), not just maps.

## What's new in 1.5.0 — a finished map you can read (`plakat map --map-render`)

1.5.0 turned the 1.4.0 geometry engine into the **first complete, user-facing map**:
a styled, labelled image with cartographic furniture, plus a scalable vector export.
Still no SD — a pure function of (spec, seed), byte-stable on-box.

- **MAP-3 linework render** — `--map-render` (+ `--map-style parchment|inked|blueprint`):
  paper-tinted biomes, ink coastline, NW hill-shading, per-kind landmark symbols,
  collision-routed labels (a hand-authored 5×7 bitmap font — no font asset, so the PNG
  is byte-identical across machines), and furniture (title cartouche, compass rose,
  1/2/5-rounded scale bar, present-kinds legend, double frame).
- **MAP-3b vector export** — `--map-export-geojson` (coastline via Moore-neighbour
  contour rings + river/road LineStrings + landmark Points, normalized [0,1] north-up)
  and `--map-export-svg` (a standalone scalable map).

## What's new in 1.4.0 — procedural fantasy maps, layer by layer (`plakat map`)

1.4.0 opened **Track M — `plakat map`**: turn a prose world description into a
fantasy map. This cut shipped the front half — the **spec** + an **eight-layer
geometry engine** — with no SD render (the linework render arrived in 1.5.0). Pure
function of (spec, seed), every layer a byte-stable on-box corpus image.

- **MAP-1** — `MapSpec v2` geographic schema + a tagged `Anchor` type (`mouth_of`,
  `natural_harbor`, `bearing`, `range_slope`, `pass_between`, …); landmarks placed
  relative to features, not pixels. LLM parser (prose → spec, 3-stage robustness);
  `--map-spec` loads a committed spec and skips the LLM.
- **MAP-2** — the geometry engine, L0–L7: fBm terrain + range ridges, priority-flood/
  D8 hydrology (rivers end at the coast), spec-driven coastline, biomes, a fixpoint
  landmark resolver, Dijkstra roads + bridges, and a composited feature overlay. The
  only new dependency for the whole engine is `noise`.

## What's new in 1.3.0 — generate a scene series from data (`compile` + Tera)

1.3.0 completed the **compile** track with an optional **Tera template pre-pass**.
A `.tera`/`.j2` input renders to a `prompts.txt` first — so a whole scene *series*
comes from one data file (loops, conditionals, shared macros), then compiles to a
`scenario` and renders. SemVer-additive, behind the `templates` feature.

- **`.tera` / `.j2` inputs** render to a `prompts.txt` before the parser, with
  context from `--var KEY=VALUE` / `--vars <json|toml>` / `--vars-env PREFIX` /
  built-in `plakat.*`. Loop over a data file, branch with `{% if %}`, share macros.
- **Filters & functions** — `scene_name`, `prompt_join`, `zero_pad`, `model_family`,
  `include_raw`, `scene_separator`, …; `--dump-rendered[-only]` to inspect the
  rendered `prompts.txt` before spending any LLM calls.
- **Compile polish** — `--compile-parallel N` (concurrent scenes, output order
  preserved) and a `--dry-run` token estimate. Proven end to end: `series.tera`
  (+ `series.json`) → a branched two-character scenario → rendered (mage + ranger).

## What's new in 1.2.0 — write prose, get a render plan (`plakat compile`)

1.2.0 opened the two-track `compile` + `map` arc; Track C is `plakat compile`.

- **`plakat compile prompts.txt`** → a `scenario` HJSON, one task per blank-line
  block: free-text descriptions + `key: value` commands, global→scene inheritance,
  model-family-aware prompt rewriting (SD15 / SDXL / Flux) + auto-negatives via the
  `--enhance` stack.
- **Deterministic core** — `--no-enhance --no-negative` assembles verbatim (no LLM,
  byte-stable). `--lint` / `--dry-run` validate without a call.
- **Workflow glue** — `scenario -` stdin pipe, `--decompile` (scenario → prompts.txt),
  `--diff`, two-namespace `--compile-cache`, `--compile-parallel`, `translate:` /
  `persona:`.

## What's new in 1.1.0 — train your own words, compose live, select by depth

1.1.0 finished the "train your own everything" and "compose & edit" threads from 1.0.

- **Textual-Inversion training** — `plakat embedding train` learns a new token
  embedding from a few images, the model frozen. SD 1.5 / 2.1 learn one CLIP-L
  vector; **SDXL** learns a CLIP-L + CLIP-G pair (a dual-encoder TI). Loads via
  `--embedding PATH:trigger[:scale]`.
- **Compose `generate:` / `matte:` layers** — a `compose` layer's pixels can come
  from `load:`, `matte:` (U2Net cutout on the fly), or `generate:` (t2i inline) —
  build a scene with nothing on disk.
- **`segment --depth-band LO,HI`** — a click-free mask source via Depth-Anything-V2
  (1.0 = nearest), combinable with `--point` to intersect.
- **SD3.5 DreamBooth** — `style train --base sd35 --class-dir … --prior-weight`
  ports class prior-preservation to the MMDiT rectified-flow trainer.

## What's new in 1.0.0 — compose & edit scenes, train your own everything

The **1.0 release**: SemVer-stable contracts, plus two capability themes on top of
the v0.47 freeze.

**Compose & edit scenes**

- **Select** — `plakat segment --point X,Y` masks an object via **MobileSAM**
  (`--grow`/`--feather` for clean edges); the mask feeds `img2img --mask`, so
  *select → remove / replace / swap background* composes from owned pieces.
- **Compose** — `plakat compose <scene.hjson>` stacks image layers (background +
  cut-outs) by z-order, position, scale, and opacity. No GPU.
- **Regional prompting** — `--region "x0,y0,x1,y1:prompt"` puts different prompts
  in different canvas regions of one image (SD 1.5 / SDXL / SD3.5; also a scenario
  `regions` key), feather-blended into one scene.

**Train your own everything**

- **Subject (DreamBooth) LoRAs** — `style train --class-dir … --class-prompt …
  --prior-weight` learns a specific *subject* (not a style) with class
  prior-preservation, so its token binds your subject without overrunning the class.
- **Resumable training** — `style train --resume …-step<N>.safetensors --steps M`
  continues an interrupted (or finished) run; all bases.

**Rock-solid on Metal**

- **OOM guard** — a background watchdog (macOS kernel memory-pressure aware) that
  aborts plakat *cleanly* before a unified-memory exhaustion can crash the host.
- Fixes: 9-channel inpaint models now honour `--strength`; tiled hi-res is
  base-anchored (globally coherent) and per-tile memory-bounded.

**Stable & honest** — SemVer from 1.0; CLI flags / scenario HJSON / Bund word-set
frozen (`STABILITY.md`); Flux scoped CPU/CUDA-only (`FEATURE_TO_MODEL.md`).

## What's new in v0.47.0 — InstantStyle, smart cut-outs, and 1.0-ready

v0.47 landed the last marquee features and **froze the surface for 1.0**.

- **InstantStyle** — `stylize --instantstyle`: true painterly STYLE transfer via a
  decoupled IP cross-attention into the SDXL style block (`up_blocks.0.attentions.1`)
  — the reference's brushwork without cloning its content.
- **Smart transparent** — `transparent --matte`: content-aware U2Net background
  removal off *any* background (verified bit-for-bit vs a PyTorch reference).
- **Integral artefacts** — `--artefact … --artefact-blend`: composites cut-outs as
  *part of* the scene (canvas-relative scale, contact-shadow grounding, colour
  harmony, a canny-ControlNet re-paint).
- **Friendly crashes** (`human-panic`) + the **1.0 contract freeze**: final CLI
  renames (`--lora`, `--preset`, `--asset-type`, `--flux-quant-level`); CLI flags /
  scenario HJSON / Bund word-set frozen in `STABILITY.md`; Flux scoped CPU/CUDA-only.

## What's new in v0.46.0 — Train your own style, on any base

v0.45 shipped style training on SD 3.5 only. v0.46 **completed the trilogy:
SD 1.5 and SDXL style-LoRA training** — plakat vendored candle's SD UNet and
wired a trainable LoRA attention path into it, so you learn a style on whichever
base you render with (a LoRA is bound to its base).

Around the trainer, a verification-and-polish cycle hardened the surface against
a committed proof corpus:

- **`plakat doctor --capability`** — a RAM-budgeted table of which models run on
  *your* hardware, before downloading 30 GB.
- **`--smart-discovery`** — for `--look` / `--genre`, a local LLM judges the
  Civitai candidate pool and picks the best *style* LoRA, rejecting character
  LoRAs. All 8 bundled looks render clean on SDXL.
- **Civitai by id** — `--lora civitai:<id>:scale` pulls straight from Civitai (a
  `timeout(0)` bug that instantly failed every download is fixed).
- **Outpaint, clean** — the masked region is conditioned on mid-gray (no dark
  bands) with a binary mask (no feather seams); extensions blend in.
- **SDXL stylize** — `plakat stylize --model sdxl` (sharper, native 1024²).
  Honest scope: a ref-*variation* tool (content/palette, not painterly texture) —
  the true-style path (InstantStyle) landed in v0.47.

## What's new in v0.45.0 — Train your own style

**`plakat style train` learns a style from a folder of images into a LoRA**
you drop onto any generation — creation, not just detection. Train on nine
watercolour references and fresh subjects render in that exact style. Phase 1
shipped SD 3.5 (mixed precision — frozen BF16 base + F32 LoRA on the attention
projections; encode-then-drop the text/VAE stack; train at 256² with periodic
checkpoints). Output is a standard diffusers-PEFT `.safetensors`; training and
generation stay separate. (SD 1.5 + SDXL bases followed in v0.46.)

Also fixed: **SD 3 `--lora`** on diffusers checkpoints (sd35-medium) was
silently 0/N-merged — now remaps diffusers→SAI before merging (191/191), fixing
*any* SD3 LoRA; **SD 2.1** repointed off the newly-gated stabilityai repo to an
ungated 768 v-prediction mirror.

## What's new in v0.44.0 — SD 3.5 rescued + corpus breadth

v0.44 continues the verification cycle. The headline: **SD 3.5-medium —
listed as supported since the SD3 line landed but never once rendering an
image — now generates end to end** on a 24 GB Mac (BF16-native on Metal,
the strongest open model plakat can GPU-accelerate there). It was the
sixth "shape-tested, never verified" model the proof corpus caught.

```
SD 3.5-medium  → loads (diffusers→SAI MMDiT remapper) + generates;
                 MMDiT verified against diffusers at corr 1.0
corpus breadth → ML upscale, img2img restyle, portrait + reference
                 lookalike, and a scene × weather demonstration (19 → 38)
```

### How SD 3.5 was broken — and fixed

It didn't even **load**: plakat's MMDiT loader expects the SAI single-file
layout (fused `joint_blocks` QKV), but Stability ships the diffusers
transformer (split `transformer_blocks` Q/K/V) — a 404 on the first
tensor. A diffusers→SAI remapper fixed the load. Then the **forward** and
**conditioning** hid five more bugs, none catchable by a single-forward
check:

| Bug | Effect |
|---|---|
| pooled-`y` concatenated `[CLIP-G, CLIP-L]` vs diffusers' `[CLIP-L, CLIP-G]` | scrambled the vector that drives adaLN across the whole MMDiT → it **never denoised** (a grid) |
| flow-match timestep passed raw `[0,1]`, not `×1000` | wrong time embedding |
| `AdaLayerNormContinuous` read `(shift, scale)` vs diffusers' `(scale, shift)`, ×2 | a 2700-magnitude outlier the final norm propagated |
| missing QK-norm on the context-qkv-only block; F16 timestep embed; `sd35-medium` variant mis-detect | load / precision |

The MMDiT now matches diffusers' `SD3Transformer2DModel` at **corr 1.0** —
found the same way as the prior campaigns, plus a full `encode_prompt`
diff (which is what caught the pooled-`y` swap).

### Corpus breadth (19 → 38 entries)

`sd35.hjson` (incl. a legible **"FRESH BREAD"** sign), `upscale.sh`
(Real-ESRGAN ×2 — opens **Transforms & post**), `img2img.sh` (prompt-
steered restyle), `portrait.sh` + `portrait.hjson` (text personas + a
reference-photo lookalike via IP-Adapter-Plus-Face), `weather-scene.hjson`
(one area across the `scene` × `weather` axes).

## What's new in v0.43.0 — Proof corpus + the bugs it caught

v0.43 is a **verification** cycle: a reproducible, self-documenting body
of images — and the tooling to regenerate and index it — that proves
plakat's pipelines actually work end to end. Rendering it surfaced (and
fixed) a stack of correctness bugs that shape-tests had hidden for dozens
of versions.

```
plakat gallery    → build a browsable Markdown index from generated images
                    (AnimateDiff clips fold in as single animated-GIF entries)
proof corpus      → scenario-driven stills (Cascade / SDXL / SD 1.5 / PixArt-Σ)
                    + AnimateDiff clips, each embedding its full recipe
```

### What the corpus caught

| Area | Was | Now |
|---|---|---|
| **SD 1.5 / 2.1** | pure **noise** on every backend since v0.16 | `clip_skip=1` now applies the CLIP final layer norm — the regression had been feeding the UNet pre-LN (un-normalized) embeddings |
| **SDXL** | **black** image on Metal | the stock VAE overflows F16 → madebyollin `sdxl-vae-fp16-fix` drop-in for non-CPU |
| **AnimateDiff** | pure **noise** since v0.26 (never verified) | every motion module matches diffusers at **corr 1.0** after **7 fixes**; coherent video on an aesthetic base (`--model Lykon/dreamshaper-8`) |
| **PixArt-Σ** | errored on load → **black** → **noise** | generates; the DiT matches diffusers (pos-embed H/W + scaling, final-adaLN, BF16 T5, IDDPM linear betas) |
| **Flux GGUF / Metal** | crash / garbage | fails fast with guidance (a candle 0.10.2 kernel bug, a layer below plakat) |

The method, reused from the Cascade campaign: dump diffusers' intermediate
activations on a fixed input and diff plakat's against them stage by stage
until every forward matches.

## What's new in v0.42 — Stable Cascade completeness + polish

v0.41 made Stable Cascade *generate*. v0.42 makes it **complete**: real
LoRA support, image-conditioning, and ControlNet on every surface —
plus a graceful guard for a Flux-on-Metal candle bug. See the new
[Stable Cascade tutorial](Documentation/Tutorials/CASCADE_TUTORIAL.md).

```
LoRA / DoRA       → community kohya & PEFT LoRAs actually merge (was a silent no-op)
image variation   → condition on a reference image's semantics (unCLIP-style)
faithful img2img  → hold the init's content, not just its structure
scripting CN      → plakat.cascade honours plakat.controlnet.* — the last CN surface
```

### Phases

| # | Phase | What |
|---|---|---|
| 0 | `--decoder-guidance` | Stage B (decoder) CFG scale, decoupled from the prior's `--guidance` (default 1.1). Threaded through CLI, img2img, scenarios, scripting. |
| 1 | LoRA / DoRA, for real | Community Cascade LoRAs silently no-op'd. Fixed two load-bearing bugs, verified against a real DoRA: **kohya/sd-scripts prefix** recognition (`lora_prior_unet_…`, not just dotted PEFT), and the **DoRA magnitude axis** — kohya stores it per input-column (dim 0), PEFT per output-row (dim 1); renorming the wrong axis scrambles every weight regardless of strength. `apply_dora` now auto-detects the axis (CoV-based for square weights, length for non-square), so kohya **and** PEFT DoRAs both fuse. Full noise → coherent styled output. |
| 2 | *(dropped)* | An "exact CannyFilter resize-to-224" normalization was investigated and **empirically falsified** — 224 makes the effnet emit a 7×7 feature map that the residual injection upsamples into a grid; the v0.41 full-resolution path was already correct. Reverted. |
| 3 | Stage C image encoder | Wired the **CLIP ViT-L/14** image encoder (`openai/clip-vit-large-patch14`, the one the prior's `clip_img_mapper` expects) into Stage C's previously-zeroed image slot. Two entry points: **`--image-variation PATH`** (unCLIP-style; prompt optional) and **`img2img --faithful`** (adds Stage C semantic conditioning on top of the Stage B VAE seed). Encoder lazy-loads only when requested. |
| 4 | Scripting Cascade CN | `plakat.cascade` honours a canny `ControlSpec` from the shared `plakat.controlnet.*` words — closing the **last ControlNet surface** (CLI + img2img + scenarios + scripting). Surfaced and fixed a pre-existing bug: `plakat.load` couldn't load Cascade **or PixArt** at all in scripting (mis-routed to the SD-only loader). |

Bonus: GGUF Flux on Apple Metal now **fails fast with guidance**
instead of crashing/emitting garbage — candle 0.10.2's Metal
matrix×matrix quantized matmul kernel is buggy (a layer below plakat);
the transformer body is now F32-correct so the path works the day
candle fixes the kernel. Override with `PLAKAT_ALLOW_GGUF_METAL=1`.

### Try it

```bash
# LoRA / DoRA — community Cascade LoRAs now actually apply
plakat generate "a girl in a flower field, anime style" \
    --model stable-cascade --lora ~/loras/cascade_anime.safetensors:1.0

# Image variation — vary on a reference image's semantics
plakat generate "" --model stable-cascade --image-variation ref.png

# Faithful img2img — hold the init's content
plakat img2img cottage.png --prompt "a cottage in winter snow" \
    --model stable-cascade --strength 0.6 --faithful

# Decoupled decoder guidance
plakat generate "a baroque cathedral interior" \
    --model stable-cascade --guidance 4.0 --decoder-guidance 1.3
```

Cascade ControlNet now works in scripting too — push a spec with
`plakat.controlnet.annotate`, then `plakat.cascade` (see
[`tools/verify_phase4_cascade_cn.bund`](tools/verify_phase4_cascade_cn.bund)).

## What's new in v0.41 — Stable Cascade actually generates

v0.40 shipped Stable Cascade "architecture-verified, quality-pending"
— it ran end to end but produced noise. v0.41 makes it **generate
real, photorealistic images** across text-to-image, ControlNet, and
img2img. The path there was a reference-comparison campaign: per-stage
Python harnesses dump diffusers' (and torchvision's) intermediate
activations on fixed inputs, and Rust tests diff ours against them
until every forward matches to <0.001. That harness caught **24
distinct bugs** that inspection alone had missed.

```
text-to-image     → coherent landscapes, scenes, complex multi-subject prompts
+ ControlNet      → a canny house outline → a photorealistic cottage on the lines
+ img2img         → a cottage → the same cottage in winter snow
+ img2img + CN    → init texture and edge structure composed together
```

### Phases

| # | Phase | What |
|---|---|---|
| 0 | Wuerstchen scheduler | Ratio-timestep `CascadeScheduler` (cosine α-cumprod, shift 0.008) replacing the SDXL integer-timestep DDPM the model was never trained on. |
| 1 | sca/crp conditioning | `sca_emb` / `crp_emb` use the sinusoidal embedding of a zero scalar (upstream's `sca=None` default), not the `t_emb` placeholder. |
| 2 | Visual quality | The bulk of the cycle: **16 numerical bugs** fixed across all three stages, each pinned to a diffusers reference. Headliners: F16→**BF16** dtype (Cascade trains in bf16; F16 overflowed to NaN → all-black); the **sinusoidal time embedding** (missing the ×10000 scale, wrong divisor, wrong sin/cos order); the missing **clip_norm** (KV stream off 80×); **switch_level=false** Stage C topology; Stage B's **pixels_mapper(zeros)** always-applied term and **up_repeat_mappers** [3,3,2,2]; Stage A's **ReplicationPad2d** (not reflection); the **decoder CFG**; and CLIP-G **`hidden_states[-1]`** not `[-2]` (the complex-prompt melt). |
| 3 | ControlNet rebuild | The canny CN was broken on every axis and never ran. Rebuilt the backbone as **EfficientNetV2-S** (it was mislabeled MobileNetV3 — now matches torchvision to 0.00004), fixed the LeakyReLU projections, rewrote injection to the upstream `controlnet_blocks=[0,4,8,12,51,55,59,63]` (bilinear-resized), and wired it into generation. Scenario `control:` support. |
| 4 | CN UX + img2img | `--control-from` auto-annotates via Canny; `--cascade-control-weights` is now optional (the CN auto-resolves from the repo). Implemented Cascade img2img (was a bail stub) — Stage-A-encode the init, seed Stage B at a strength-truncated schedule — and made ControlNet compose with it. |

### Verification

The reference harnesses (`tools/cascade_ref_dump*.py`) + the
`*_matches_diffusers_reference` / `cn_forward_matches_reference` tests
are permanent regression guards — they pin every Cascade stage to the
upstream reference to <0.001 (the CN one to torchvision EfficientNetV2-S).

### Follow-ups (most closed by v0.42)

- A proper `--decoder-guidance` flag (was a fixed 1.1) — **closed v0.42.**
- Scripting (Bund) Cascade ControlNet — **closed v0.42.**
- The exact upstream CannyFilter normalization — investigated in v0.42
  and found the "resize to 224" premise wrong for plakat's residual
  injection; the working full-resolution `[0,1]→[-1,1]` path was kept.
- Multi-CN is N/A for Cascade — single cnet, canny-only checkpoint.

## What's new in v0.40 — Stable Cascade end-to-end on real weights

v0.40 closed the v0.39 "load-correct, generate-pending" caveat:
plakat ran Stable Cascade end-to-end on the real upstream
`stabilityai/stable-cascade` + `stabilityai/stable-cascade-prior`
weights — CLIP-G text → Stage C denoise → effnet conditioning →
Stage B denoise → PixelShuffle bridge → Stage A decode → image.
Five feature phases + close-out; three real-weight smoke-iteration
rounds resolved every architectural gap (Conv2d-no-bias for
BN-followed convs, ResBlock skip-concat at up-path boundaries, CN
stage-4 variable expansion, Stage A `up_blocks.0.0` nesting,
effnet/pixels spatial alignment). 1230 lib + 47 integration tests.

The end-to-end smoke produced structurally valid but visually
**noisy** output — architecture verified, numerical quality
(scheduler, sca/crp conditioning, more steps) explicitly deferred
to v0.41. That deferral became the bulk of v0.41.

## What's new in v0.39 — Stable Cascade architectural rewrite

v0.39 closed the "tensor-naming alignment" caveat that v0.37 and
v0.38 both shipped: plakat's Cascade modules were rewritten to
match the actual upstream Würstchen v3 / Stable Cascade
architecture, verified position-by-position against the inspected
`stabilityai/stable-cascade` + `stabilityai/stable-cascade-prior`
safetensors headers.

The cycle started with a planned ~300-LOC rename; inspection in
phase 0 revealed the gap was structural. One headline phase, eight
sub-phases (0a–0h), ~3600 LOC rewrite, +12 net lib tests.

### Architectural changes (vs v0.37/v0.38)

- **ResBlock**: SD-style `norm/conv/norm/conv` → ConvNeXt-v2
  (`depthwise + LayerNorm2d + channelwise(Linear → GELU → GRN →
  Linear)` + skip).
- **TimestepBlock**: 1 mapper → 1/2/3 mappers (`mapper`,
  `mapper_sca`, `mapper_crp`).
- **AttnBlock**: separate self+cross modules → fused single
  attention with `kv_mapper` (KV = `cat(flatten(image),
  kv_mapper(text))`).
- **Stage C UNet**: 3 levels → 2 levels at uniform c_hidden=2048
  with strict `[Res, Time, Attn]` triples; `blocks_per_level =
  [8, 24]`.
- **Stage B UNet**: variable widths `[320, 640, 1280, 1280]`;
  `effnet_mapper` + `pixels_mapper` Sequentials; attention only
  at deepest 2 levels; Strided 2×2 stride-2 Conv/ConvTranspose.
- **Stage A VAE**: Paella v3 VQ-GAN — ConvNeXt blocks + 8192-code
  codebook + PixelUnshuffle/PixelShuffle at input/output.
- **ControlNet**: 4 strided convs → MobileNetV3-Large backbone
  (8 stages, `[2,4,4,6,9,15]` blocks) + 8 projection heads →
  2048-ch residuals.

### Honest scope notes (closed by v0.40)

v0.39 shipped **load-correct, generate-pending**:
`Pipeline::generate()` bailed with a v0.40 pointer rather than
shipping subtly-incorrect output. v0.40 closed this caveat with
real-weight end-to-end smoke iteration.

### By the numbers

- **1214 lib + 47 integration = 1261 active tests** (+12 net lib).
- 8 sub-phases of phase 0 + close-out.
- 2689 LOC of old SD-style modules deleted; ~3600 LOC of
  upstream-aligned modules added.

## What's new in v0.38 — Stable Cascade completeness

v0.38 closed the two load-bearing correctness deferrals from v0.37
(FiLM time injection + effnet conditioning Stage C → Stage B) and
layered productivity on top. Five feature phases + close-out.
Test count grew 1173 → 1199 lib tests (+26 across the cycle).

### Architecture-level closures (phases 0–1)

- **FiLM timestep injection.** New `TimestepBlock` module
  interleaved between ResBlock and AttentionBlock at every
  encoder + decoder slot. Wired the time embedding that v0.37
  phase 2 left as a silent pass-through.
- **Effnet conditioning.** Stage B's `in_conv` grew from 4 to 20
  input channels (4 noise + 16 Stage C effnet, spatially
  upsampled). `Config::effnet_input_channels` toggled the path;
  `forward_with_effnet` API enforced it.

### Productivity surface (phases 2–5)

- **`plakat.cascade ( prompt -- handle )`** Bund word mirroring
  `plakat.pixart`. `ScriptCtx.loaded_cascade` cache amortised
  ~14 GB cold load.
- **`--stage-c-steps` / `--stage-b-steps`** CLI flags + config
  keys. Unset → split `--steps` 2/3 + 1/3.
- **LoRA on both prior UNets.** Diffusers PEFT format with
  load-time tempfile merge. Stage-specific resolvers.
- **img2img.** Encode → Stage A → Stage C → Stage B with
  truncated denoise schedule. Strength captured in PNG metadata.
- **ControlNet on Stage C.** Compact image-to-residual encoder
  producing (B, 16, 24, 24) residual injected before in_conv,
  gated by `[start, end)` timestep window. Single-CN only.

### Honest scope notes (closed by v0.39)

v0.38 shipped architecture-complete and surface-complete, but
**tensor-naming alignment with upstream `stabilityai/stable-
cascade` checkpoints remained gating** for production-quality
output. v0.39 closed this caveat with a full architectural
rewrite.

### By the numbers

- **1199 lib + 47 integration = 1246 active tests** (+26 lib
  across cycle).
- Both v0.37 correctness deferrals closed.
- Fully additive surface.

## What's new in v0.37 — Stable Cascade (diversify-5)

plakat's **fifth model family** lands: Stable Cascade. A 3-stage
architecture distinct from every existing family — not a single
UNet (SD), not DiT (PixArt), not MMDiT (SD3), not Flux DiT. Three
coupled models chain at inference: `text → Stage C → Stage B →
Stage A → image`.

Six phases shipped. Test count grew 1141 → 1173 lib tests
(+32 across the cycle).

### `plakat generate "..." --model stable-cascade`

```bash
plakat generate "a misty forest at dawn, painterly" \
    --model stable-cascade --size 1024x1024 --steps 30 \
    --guidance 4.0 --seed 42
```

Aliases: `stable-cascade`, `cascade` → `stabilityai/stable-cascade`.
The single `--steps` budget splits **2/3 to Stage C** (the heavy
semantic stage) + **1/3 to Stage B**; dedicated step flags landed
in v0.38.

### Stage A VAE — Paella v3 (phase 1)

Small ~3.6M-param VAE for image ↔ latent mapping at 32× per-axis
compression. Continuous latents (Würstchen v3 / Stable Cascade
dropped the codebook the earlier designs used):

```
image (B, 3, 1024, 1024)
  → Encoder (5 down blocks: 64 → 128 → 256 → 384 → 512 → 4 ch)
  → latent (B, 4, 32, 32)
  → Decoder (5 up blocks, mirror)
  → image (B, 3, 1024, 1024)
```

Each `down/up_block` is a `ResBlock` + strided Conv2d (encoder)
or nearest 2× upsample + Conv2d refinement (decoder). ResBlock
is `GroupNorm → SiLU → Conv2d → GroupNorm → SiLU → Conv2d + skip`.

### Stage B latent prior UNet (phase 2)

~1.5B-param UNet that takes Stage C's output + text and produces
Stage A's latent. The same `StableCascadeUnet` skeleton serves
**both Stage B and Stage C** with different `Config` instances.

Block structure:
- `in_conv` (channels → first level)
- `TimeEmbedding` — sinusoidal + 2-layer MLP
- N **encoder levels**: ResBlocks (+ optional `AttentionBlock`
  per RB) + Downsample
- N **decoder levels** (mirror): skip-concat from matching encoder
  level + ResBlocks + Upsample
- `out_norm + silu + out_conv`

`AttentionBlock` is `norm → self-attn → norm → cross-attn-to-text
→ norm → 2-layer FF MLP`. Self-attention + cross-attention to the
CLIP-G text sequence at deeper levels.

**Full + Lite variant routing.** `Config::stage_b_for_alias`
picks Lite from the substring `"lite"`; otherwise Full.

### Stage C high-res prior UNet (phase 3)

The headline ~3.6B-param model. Text → 24×24×16 super-compressed
prior latent. **16 input/output channels** (vs Stage B's 4) at a
tiny spatial grid. Attention at every level — the short sequence
keeps it affordable.

Reuses the `StableCascadeUnet` skeleton from phase 2; phase 3 was
mostly configuration + wiring.

### 3-stage orchestration (phase 4)

```text
prompt
  ↓ CLIP-G encode (penult + pooled)
  ↓ Stage C CFG denoise (DPM++ default)        → 24×24×16 latent
  ↓ Stage B CFG denoise                        → 32×32×4 latent
  ↓ Stage A decode                             → 1024×1024 image
```

Seed plumbing through `pipelines::seeds::prepare_seed`. PNG
sidecar metadata. Stable Cascade earns a ✓ row in `plakat doctor
--reproducibility-check`.

### CLI integration + scenarios (phase 5)

- Doctor row: **Stable Cascade (3-stage)** classified Guaranteed.
- v0.25 look preset routing automatic via `BaseFamily::
  StableCascade`.
- Scenario integration: new `cascade_pipeline` cache slot
  mirroring `pixart_pipeline` from v0.36 phase 0.

### Honest scope notes

v0.37 shipped **shape-correct end-to-end orchestration**. Two
architectural pieces deferred to v0.38:

- **FiLM timestep injection** into ResBlocks. ✓ closed v0.38 phase 0.
- **Effnet conditioning** — Stage C output feeding Stage B's
  denoise. ✓ closed v0.38 phase 1.

### By the numbers

- **1173 lib + 47 integration tests = 1220 active tests** (+32
  lib across the cycle).
- 6 phase commits + RFC + close-out.
- Fifth model family alongside SD-family, SD3, Flux, and PixArt.

## What's new in v0.36 — PixArt completeness

Closes every PixArt deferral from v0.35 while DiT / T5 / LoRA
context was fresh. Mirrors v0.34's audit follow-through after
v0.33 — finish what got started before context fades. Five
phases shipped; PixArt is now usable in scenarios, scripting,
across three resolution variants (512 / 1024 / 2K), with LCM-LoRA
composition.

Test count grew 1123 → 1141 lib tests (+18 across the cycle).

### Scenario PixArt dispatch (phase 0)

PixArt models now run in scenarios alongside SDXL t2i / Flux /
SD3 / AnimateDiff, with VAE-cache sharing across kind switches.
`pixart_pipeline` cache slot mirrors `flux_pipeline` /
`sd3_pipeline`: load once at scenario start; mixed-kind
scenarios that share an alias with SDXL t2i reuse the VAE Arc
via the v0.34 phase 3 cache.

### Scripting `plakat.pixart` Bund word (phase 1)

```bund
"pixart" plakat.load
"a misty forest at dawn" plakat.pixart   // → image handle
```

Same shape as `plakat.generate`: pulls prompt off the stack,
pushes the 1-based image handle. Pipeline cached on
`ScriptCtx.loaded_pixart` — multi-call scripts amortise the
~12 GB cold load.

### PixArt-Σ-XL-2-512-MS variant (phase 2)

New aliases `pixart-512` + `pixart-sigma-512`. Same DiT-XL/2
architecture as 1024-MS — `sample_size: 32` is informational
(plakat computes the positional embedding from the actual grid
at forward time). Faster CPU smoke + smaller VRAM at 512².

### KV-compression + PixArt-Σ-XL-2-2K-MS variant (phase 3)

PixArt-Σ's headline addition over PixArt-α: a per-block depthwise
Conv2d downsamples the image-token K/V sequence in self-attention,
making 2048² output computationally tractable. Σ paper §3.2: "We
apply KV compression on all 28 transformer blocks."

```bash
plakat generate "..." --model pixart-2k --size 2048x2048
```

- `Config::sigma_xl_2k` with `kv_compression: Some(scale=2)`.
- `Attention::new_with_compression` registers a depthwise Conv2d
  (`groups=hidden_size`, `kernel=stride=2`). Tensor key
  `<attn-prefix>.kv_proj_conv2d.{weight,bias}` matches the
  diffusers PixArt-Σ convention.
- Only `attn1` (self-attn) uses compression — `attn2` (cross-
  attn to T5) stays uncompressed.

### LCM with PixArt — composition path (phase 4)

LCM 2-step / 4-step PixArt generation works today via the
existing v0.35 phase 4 LoRA merge + v0.28 phase 1 LCM scheduler:

```bash
plakat generate "..." \
    --model pixart \
    --lora civitai:NNNNNN:1.0 \   # PixArt LCM-LoRA
    --scheduler lcm --steps 4 --guidance 1.5
```

Native PixArt-α-LCM checkpoint integration is deferred to v0.38+
(requires an α/Σ architectural fork in `pixart_dit`). A new
`is_pixart_sigma_repo` guard at `Pipeline::load` detects α-style
repo paths and surfaces the exact LCM-LoRA recipe.

### Documentation

- [`RFC_v0.36_PIXART_COMPLETENESS.md`](RFC_v0.36_PIXART_COMPLETENESS.md)
  — design doc, locked decisions (phase 3 KV-compression locked,
  not stretch), 5-phase plan.

### By the numbers

- **1141 lib + 47 integration tests = 1188 active tests** (+18
  lib across the cycle).
- 5 phase commits + RFC + close-out.
- All 6 PixArt items from the v0.35 deferral list closed (4
  shipped, 2 deferred with documented mitigations).

### v0.35 → v0.36 migration

v0.36 is fully additive. Every existing flag, host word, config
key, scenario field, and PNG sidecar from v0.35 still works
unchanged. New surface:

- ✅ `--model pixart-512` / `pixart-sigma-512` / `pixart-2k` /
  `pixart-sigma-2k`.
- ✅ Scenarios accept `model: pixart` (and aliases) for batch
  PixArt runs.
- ✅ `plakat.pixart` Bund word for single-image scripting.
- ✅ `pixart_dit::KvCompressionConfig` +
  `Attention::new_with_compression` +
  `Attention::forward_self_attn(x, grid_dims)` — public APIs.
- ✅ `is_pixart_sigma_repo` early bail at `Pipeline::load`.

## What's new in v0.35 — PixArt Sigma (diversify-4)

plakat's **fourth model family** lands: PixArt-Σ-XL-2-1024-MS, a
Diffusion Transformer (DiT) with a T5-XXL text encoder. Breaks
the two-cycle polish chain (v0.33 + v0.34) and reuses the T5
infrastructure SD3 and Flux already ship for partial
implementation savings.

Five phases shipped. Test count grew 1099 → 1123 lib tests (+24
across the cycle).

### `plakat generate "..." --model pixart`

```bash
# Canonical 1024² PixArt-Σ inference.
plakat generate "a misty forest at dawn, painterly" \
    --model pixart --size 1024x1024 --steps 20 \
    --guidance 4.5 --seed 42 --scheduler dpmpp-karras
```

Aliases: `pixart`, `pixart-sigma`, `pixart-1024` — all resolve to
`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`. CFG denoise loop with
DPM++ as the recommended scheduler.

### DiT-XL/2 architecture in candle (v0.35 phase 1)

`src/pipelines/pixart_dit.rs` ships the full transformer with
tensor names matching the diffusers `PixArtTransformer2DModel`
safetensors layout verbatim:

- **DiT-XL/2 backbone** — 28 layers, hidden 1152, 16 heads, ~600M
  params.
- **adaLN-single + scale_shift_table** — PixArt-α's parameter-
  saving trick. Single global MLP turns timestep + Σ-conditioning
  (resolution + aspect_ratio) into one `(6 × hidden)` vector;
  each block adds its own `(6, hidden)` `scale_shift_table`.
- **adaLN-zero modulation** — shift/scale/gate for MSA and MLP
  (6-way split per block).
- **Cross-attention to T5** — image tokens form Q; T5 hidden
  states (after `caption_projection`) form K/V. No KV-compression
  on cross-attn — `kv_compression: None` per the 1024-MS config;
  2K-MS variant deferred.
- **Σ-specific conditioning** — `resolution_embedder` +
  `aspect_ratio_embedder` sit alongside the timestep embedder
  inside `adaln_single.emb`.

### T5-XXL + DiT + VAE assembly (v0.35 phase 2)

`pixart::Pipeline` carries `t5_enc + t5_tok` (same
`candle_transformers::models::t5::T5EncoderModel` SD3 uses),
`dit` (the v0.35 phase 1 module), and `vae` (`Arc<AutoEncoderKL>`
shared via the v0.34 phase 3 cache).

Seed plumbing routes through `pipelines::seeds::prepare_seed`
(v0.34 phase 1 chokepoint) — PixArt earns a ✓ row in
`plakat doctor --reproducibility-check`.

### PixArt LoRA + sidecar metadata (v0.35 phase 4)

```bash
plakat generate "a cat in a meadow" --model pixart \
    --lora civitai:12345:0.7 --lora-scale 1.0
```

Diffusers-format PEFT LoRA parser (`pipelines/pixart_lora.rs`).
Accepts every per-block target: `attn1/2.{to_q,to_k,to_v,
to_out.0}`, `ff.net.{0.proj,2}`. Civitai PixArt LoRAs match via
`BaseFamily::PixArt::civitai_matches`.

PixArt is the **first non-t2i pipeline since v0.34 phase 0 to
emit `GenerationMetadata`** — closes one corner of the v0.34
"no metadata for non-t2i pipelines" deferral. PNG sidecars carry
prompt / negative / model / seed / steps / guidance / scheduler /
size / `lora_stack` / `lora_scale`:

```json
{
  "lora_stack": [
    {"display": "civitai:12345", "scale": 0.7, "source": "civitai"}
  ]
}
```

### Documentation

- [`RFC_v0.35_PIXART_SIGMA.md`](RFC_v0.35_PIXART_SIGMA.md)
  — design doc, locked decisions (1024-MS first, LoRA locked
  phase 4 not stretch), 5-phase plan.

### By the numbers

- **1123 lib + 47 integration tests = 1170 active tests** (+24
  lib across the cycle).
- 5 phase commits + RFC + close-out.
- Fourth model family alongside SD-family, SD3, and Flux.

### v0.34 → v0.35 migration

v0.35 is fully additive. Every existing flag, host word, config
key, scenario field, and PNG sidecar from v0.34 still works
unchanged. New surface:

- ✅ `--model pixart` / `pixart-sigma` / `pixart-1024`.
- ✅ `pixart::Pipeline` + `pixart_dit::PixArtSigmaXL` +
  `pixart_lora::merge_pixart_loras_into_weights` — public APIs
  for scripting / scenario integration in v0.36+.
- ✅ `Variant::PixArt` + `BaseFamily::PixArt` +
  `BaseModel::PixArt` — exhaustive matches across the codebase
  stay sound.
- ✅ Doctor row, look preset routing, sidecar `lora_stack` — all
  populated for PixArt.

## What's new in v0.34 — audit follow-through

v0.34 closes the gaps v0.33 left behind, while the audit table
and metadata-builder context were still fresh. Three of the four
feature phases turned v0.33's "half-shipped" outputs into
"actually useful"; the fourth cleared every remaining v0.32
carry. No new model families, no new pipelines — fewer headline-
worthy items than v0.33, but every win acts on something the
previous cycle deferred or surfaced.

Four phases shipped, all additive. Test count grew 1073 → 1099
lib tests (+26 across the cycle).

### Pipeline-side structured stack population

v0.33 added `lora_stack`, `embedding_stack`, and `control_stack`
to `GenerationMetadata`, but the CLI passed `None` everywhere —
the new fields stayed empty in practice. v0.34 phase 0 wires the
t2i pipeline to populate the LoRA + ControlNet stacks from the
specs at the metadata-build site:

```json
{
  "lora_stack": [
    {"display": "civitai:12345", "scale": 0.7, "source": "civitai"},
    {"display": "user/style-lora", "scale": 0.5, "source": "hub"}
  ],
  "control_stack": [
    {"kind": "canny", "image": "./edges.png", "strength": 0.85, "start": 0.0, "end": 1.0}
  ]
}
```

PNG sidecars from `plakat generate` now carry the resolved
metadata Civitai importers, gallery cataloguers, and scenario
regression diff tools already wanted. Source kind
(`local` / `hub` / `civitai`) per entry; HF pinned revision
captured when present.

Scope is t2i (SD 1.5 + SDXL) — the only pipeline that builds
`GenerationMetadata` in-pipeline today. SD3, Flux, AnimateDiff,
stylize, and portrait don't emit `GenerationMetadata` at all and
would need separate metadata-emitting paths added; deferred.
Embedding-stack population also deferred — `EmbeddingEntry`
needs `embed_dim` / `num_tokens` / `dual_encoder` which require
loading the safetensors, making it more than "data plumbing."

### Determinism fixes from the v0.33 audit

The phase 3 audit shipped with 8 ⚠ Metal-u32 rows and 2 ?
NEEDS-VERIFICATION rows. v0.34 phase 1 fixes both:

- **VAE encode `set_seed()` placement.** In `stylize.rs` and
  `img2img.rs`, the VAE's `init_dist.sample()` is RNG-touching —
  but the existing code ran `set_seed(seed)` AFTER the sample.
  Init latents used leftover RNG state and ignored `--seed`.
  Fix: hoist `set_seed` to run before the VAE encode.
- **Metal u32 seed truncation.** New
  `pipelines::seeds::prepare_seed(seed, device)` applies
  SplitMix64 + reduces to u32 when device is Metal AND
  seed > u32::MAX. Identity passthrough below 2^32 preserves
  byte-identical output for existing users. Plumbed through 13
  `set_seed` call sites across t2i, sd3, flux, animatediff
  (both variants + FreeNoise), portrait, stylize, img2img, and
  the animate CLI.

```
$ plakat doctor --reproducibility-check
   ✓    t2i (SD-family)            v0.34 phase 1: seeds::prepare_seed mixes full u64 entropy...
   ✓    AnimateDiff (SD 1.5)       v0.34 phase 1: seeds::prepare_seed at per-window + FreeNoise
   ✓    Stylize (SD 1.5)           v0.34 phase 1: set_seed moved BEFORE VAE encode
   ✓    img2img / inpaint          v0.34 phase 1: per-iter set_seed inserted BEFORE vae_encode_image_file
```

Audit went from 3 ✓ rows + 8 ⚠ + 2 ? to **11 ✓ + 0 ⚠ + 0 ?**.
Remaining 2 ✗ rows are intentional (`rand::random()` fallback
when `--seed` omitted, and remote DeepSeek / Gemini enhancers).
A regression-lock test asserts neither tier ever reappears.

### Per-task failure capture in `--json-summary`

`TaskRunRecord.error: Option<String>` populates on
`status: "failed"` with the full anyhow error chain:

```json
{
  "tasks": [
    {"name": "alpha", "status": "ok", "seed": 42},
    {"name": "beta",  "status": "failed", "seed": 43,
     "error": "loading LoRA civitai:404404: HTTP 404 from civitai.com"},
    {"name": "gamma", "status": "ok", "seed": 44}
  ]
}
```

The dispatch loop now wraps every task in a catch-and-record
guard; failures push a record + continue rather than aborting
the scenario. Summary file is written first, then the scenario
exits non-zero if any task failed. CI consumers see every
failure in one shot.

### v0.32 carry closures

Three deferrals from two cycles back, all closed:

- **Animate-side VAE cache.** AnimateDiff{,Sdxl}Pipeline's VAE
  field rewrapped as `Arc<AutoEncoderKL>` (mirrors `SdCore` from
  v0.32 phase 2). Mixed-kind scenarios stop paying the ~330 MB
  SDXL VAE rebuild cost on every `t2i ↔ animate` kind switch.
- **Scripting `plakat.load` VAE cache.** Same Arc cache surfaces
  in `ScriptCtx`; scripts running
  `plakat.load sdxl; plakat.animate sdxl` share one VAE handle.
- **Auto1111 two-files SDXL TI convention.**
  `plakat generate --embedding mystyle_clip_l.safetensors`
  auto-discovers the `mystyle_clip_g.safetensors` companion and
  stitches both halves into a dual-encoder TI. Bare `_clip_g`
  input rejected with a hint at the `_clip_l` primary.

### Documentation

- [`RFC_v0.34_AUDIT_FOLLOWTHROUGH.md`](RFC_v0.34_AUDIT_FOLLOWTHROUGH.md)
  — design doc, scope contraction after pre-phase-0 survey,
  4-phase plan.

### By the numbers

- **1099 lib + 47 integration tests = 1146 active tests** (+26
  lib across the cycle).
- 4 phase commits + RFC + close-out.
- v0.33 phase 0 metadata-half-shipped gap **closed** (t2i side).
- v0.33 phase 3 audit gaps (Metal-u32 + VAE encode placement)
  **closed**; audit table is now all-green for pipelines plakat
  controls.
- All three v0.32 carries (animate VAE cache, scripting cache,
  Auto1111 two-files TI) **closed**.

### v0.33 → v0.34 migration

v0.34 is mostly additive. Every existing flag, host word, config
key, scenario field, and PNG sidecar from v0.33 still works
unchanged. Two intentional behavioural shifts on previously-
broken paths:

- ✅ `--seed N` with `N < 2^32` on any backend: **byte-identical**
  output.
- ⚠ `--seed N` with `N >= 2^32` on Metal: previously collided to
  `N mod 2^32`; now distinct via SplitMix64. (Fix, not regression.)
- ⚠ `stylize` / `img2img` with `--seed N --strength X`:
  numerically changed because `set_seed` now runs before VAE
  encode. (Fix, not regression — output was non-deterministic
  before.)
- ⚠ Scenarios with one failing task: previously aborted at first
  failure with bare error; now records each failure + writes
  summary + exits non-zero.
- ✅ Animate / scripting Load APIs gained a `vae_cache: Option<...>`
  parameter — callers pass `None` for the v0.33 behaviour.

## What's new in v0.33 — production polish bundle

v0.33 closes the long-standing **production polish** deferral
from v0.32+: structured metadata, actionable error hints, machine-
readable scenario output, and a reproducibility audit. No new
pipelines, no new model families — every win is on the boundary
between plakat and the operator.

Four phases shipped, all additive. No flag rename, no behaviour
change for existing runs. Test count climbed from 1030 → 1073
lib tests (+43 across the cycle).

### Structured metadata fields

PNG `tEXt` chunks and JSON sidecars carry the full visible
configuration — stylistic presets, LoRA stack, TI stack, ControlNet
stack, enhancer state, FreeNoise flag — alongside the existing
Auto1111-compatible "Parameters:" string.

```bash
plakat generate "a misty forest" --model sd15 --look anime \
    --genre fantasy --negative-preset crisp \
    --lora detail:0.6 --lora style:0.4 \
    --embedding cinematic-style --controlnet canny ./edges.png
```

Every flag shows up under its own key in the JSON sidecar AND
as a `Look: anime, Genre: fantasy, Negative preset: crisp, ...`
suffix in the A1111 string. Downstream tooling (Civitai
importers, gallery cataloguers, scenario regression diff) no
longer has to re-parse free-form prompt text.

New `GenerationMetadata` fields are `#[serde(default)] +
skip_serializing_if`, so every v0.32 sidecar still parses
unchanged (regression-locked by `v032_sidecar_still_parses`).

### Actionable error hints

Three new decorators on the user-facing error path:

```
$ plakat generate "x" --model sd1.5
Error: unknown --model alias 'sd1.5'. Did you mean 'sd15'?
       Run `plakat --help` or `plakat hf list` for the full list.

$ plakat generate "x" --model flux --width 2048 --height 2048
Error: out of memory loading Flux at 2048×2048.
       Try: --quant nf4, lower --width/--height, or close
       other GPU consumers. See FLUX.md for VRAM guidance.

$ plakat scenario broken.hjson
Error: HJSON parse error on line 14 in task 'beta':
       expected `,` or `}` after value.
       Inspect the task block starting near `name: beta`.
```

Levenshtein-based typo suggestion for `--model` and `--look`;
pipeline-tagged OOM decorator that names the right mitigation
(quant for Flux, `--vae-tiled` for SD3.5, frame count for
AnimateDiff); scenario parse errors point at the offending task
by name, not just byte offset. 21 unit tests cover the matching
logic.

### `plakat scenario --json-summary PATH`

Scenarios now emit a machine-readable run summary alongside the
existing log output:

```json
{
  "scenario_file": "/tmp/forest.hjson",
  "model": "sd15",
  "out_dir": "/tmp/out",
  "total_tasks": 12,
  "ran": 10,
  "skipped": 2,
  "failed": 0,
  "wall_time_secs": 184.21,
  "plakat_version": "0.33.0",
  "tasks": [
    {"name": "alpha", "kind": "generate",  "status": "ok",      "seed": 42},
    {"name": "beta",  "kind": "animatediff","status": "ok",     "seed": 43},
    {"name": "gamma", "kind": "generate",  "status": "skipped", "note": "--only filter excluded"}
  ]
}
```

CI now has a single file to consume — pass/skip/fail counts,
wall time, per-task seed and status. Records every code path:
`--only` skip, `--limit` skip, `--resume` cache hit, dry-run
early-continue, animate dispatch, normal generate end. Survives
mixed `--dry-run` + real runs in the same scenario.

### `plakat doctor --reproducibility-check`

```
$ plakat doctor --reproducibility-check
◆ Top warnings
  ! Reproducibility REQUIRES `--seed N`...
  ! Metal backend truncates seeds to u32...
  ! VAE encode placement in img2img / stylize paths...

◆ Per-pipeline determinism table
status  pipeline                code path              note
   ⚠    t2i (SD-family)         Pipeline::run randn    Seed masked to u32...
   ⚠    AnimateDiff (SD 1.5)    denoise_window         set_seed() before randn
   ✓    Prompt wildcards        StdRng                 Seeded from --seed
   ?    img2img/inpaint         VAE encode             Needs verification
   ✗    Any pipeline (no --seed) rand::random()        Non-deterministic
```

Hand-curated audit of every RNG-touching path across plakat's
pipelines, classified into 4 tiers: **GUARANTEED**, **GUARANTEED
(Metal u32)**, **NEEDS VERIFICATION**, **NON-DETERMINISTIC**.
Color-coded human output; composes with `--json` for CI.
Descriptive, not prescriptive — fixes for the `?`-tier rows defer
to v0.34.

### Documentation

- [`RFC_v0.33_PRODUCTION_POLISH.md`](RFC_v0.33_PRODUCTION_POLISH.md)
  — design doc, additive-schema constraint, 4-phase plan.

### By the numbers

- **1073 lib + 47 integration tests = 1120 active tests** (+43
  lib across the cycle).
- 4 phase commits + RFC + close-out.
- v0.32+ production polish deferral **closed**.
- Reproducibility audit surfaces 13 RNG paths + 5 top-level
  warnings — input for v0.34 determinism fixes.

### v0.32 → v0.33 migration

v0.33 is **fully additive**. Every existing flag, host word,
config key, scenario field, PNG sidecar, and A1111 parameter
string keeps its v0.32 shape. New surface:

- ✅ 9 new `GenerationMetadata` fields (`look`, `genre`,
  `negative_preset`, `lora_stack`, `embeddings`,
  `embedding_stack`, `control_stack`, `enhancement`,
  `free_noise`). All `Option`/`Vec` with serde `default` +
  `skip_serializing_if` — v0.32 sidecars parse unchanged.
- ✅ `plakat scenario --json-summary PATH` (optional flag).
- ✅ `plakat doctor --reproducibility-check` + `--json`.
- ✅ New `error_hints` module — opt-in decorators on the
  existing error path. Pure additions.

## What's new in v0.32 — animate-lite + diversify-3

After two consecutive diversify cycles (v0.30 + v0.31), v0.32
pays down the animate quality backlog with the single most-
visible item — **FreeNoise long-form** — while continuing the
diversify momentum with two architectural / perf wins: a
**vendored CLIP rollout** to every SD-family pipeline (unlocks
`--embedding` everywhere in future cycles), and **SDXL VAE
caching** across scenario kind switches.

Three phases shipped: one animate quality win + two carry
closures. Cleanest cycle in five turns.

### FreeNoise long-form for AnimateDiff (closes v0.27 deferral)

```bash
# Existing long-form syntax — random noise per window (v0.27).
plakat animate --animatediff --model sd15 \
    --from "a misty forest at dawn" \
    --frames 64 --window-size 16 --window-overlap 4 \
    --format mp4

# Add --free-noise to share noise across overlapping windows —
# eliminates the cross-fade seam artefact (v0.32).
plakat animate --animatediff --model sd15 \
    --from "a misty forest at dawn" \
    --frames 64 --window-size 16 --window-overlap 4 \
    --free-noise --format mp4
```

Cao et al., "FreeNoise: Tuning-Free Longer Video Diffusion." The
flag pre-generates a `(total_frames, 4, H/8, W/8)` noise tensor
at the user's seed, then slices it per sliding-window — adjacent
windows automatically share noise in the overlap region because
they slice the SAME underlying tensor. The v0.27 phase 5
linear-ramp latent blend still runs, but the two sides being
blended now come from the same noise sequence, so the seam
disappears.

SD 1.5 + SDXL both ship. Opt-in flag preserves byte-identical
output on `--seed N --frames 64` when the flag is OFF (existing
runs unchanged). Composes with multi-CN, AnimateLCM, motion LoRAs.
Single-window runs (frames ≤ window-size) are no-ops.

### Vendored CLIP rollout (closes v0.30 architectural deferral)

v0.30 phase 0 vendored CLIP for `SdCore` only (to enable TI
runtime injection). v0.32 phase 1 finishes the rollout —
every SD-family pipeline now holds plakat's vendored CLIP-L
text encoder instead of candle's:

- `AnimateDiffPipeline::text_encoder` (SD 1.5)
- `AnimateDiffSdxlPipeline::text_encoder_l`
- `sd3::Pipeline::clip_l + clip_l_cfg`
- `flux::Pipeline::clip_text + clip_cfg`
- `stylize::Pipeline::text_encoder`

Numerically identical to candle's path (per the v0.30 phase 0
forward-pass tests). User-visible impact: zero in v0.32 — but
this is the architectural foundation. Future cycles can wire
`--embedding` (TI runtime injection) through the same
`Config::with_vocab(n)` pattern v0.30 phase 0 established for
SdCore, now that every pipeline holds the vendored type. A new
compile-time type-lock test in `vendored_clip::tests` prevents
silent regressions.

### SDXL VAE caching across mixed-kind scenarios

Scenarios that mix `type: generate` and `type: animatediff`
tasks used to rebuild the ~330 MB SDXL VAE on every kind switch
(after v0.31 phase 3's pipeline eviction kicked in). v0.32 phase
2 wraps `SdCore.vae` in `Arc<AutoEncoderKL>` and adds a
scenario-level cache keyed by base alias — subsequent t2i loads
against the same SDXL base reuse the cached Arc instead of
rebuilding from disk.

Auto-deref keeps every `.vae.encode(...)` / `.vae.decode(...)`
call site at the pipeline boundary unchanged.

```
INFO plakat: SdCore: reusing cached VAE (skipping ...vae.safetensors build)
```

shows up in logs when the cache hits. AnimateDiff load functions
don't yet take a vae param — animate-side sharing waits for
v0.33+. The cache helper itself (`vae_cache_lookup`) is a pure
generic function unit-tested with 6 decision cases.

### Documentation

- [`ANIMATEDIFF.md`](ANIMATEDIFF.md) — bumped to v0.32 with the
  FreeNoise long-form quick-start. Capability matrix adds the
  "shared-noise long-form" row.
- [`RFC_v0.32_ANIMATE_LITE_DIVERSIFY_3.md`](RFC_v0.32_ANIMATE_LITE_DIVERSIFY_3.md)
  — design doc, locked decisions, 4-phase plan.

### By the numbers

- **1030 lib + 47 integration tests = 1077 active tests** (+10
  lib across the cycle).
- 3 phase commits + RFC + close-out.
- v0.27 FreeNoise animate quality deferral **closed**.
- v0.30 vendored CLIP architectural deferral **closed**
  (rollout to all SD-family pipelines).
- v0.32 phase 2 partially closes the mixed-kind VAE rebuild
  cost — t2i side complete; animate-side sharing defers to v0.33.

### v0.31 → v0.32 migration

v0.32 is **fully additive**. Every existing flag, host word,
config key, and scenario field keeps its v0.31 shape. New surface:

- ✅ `--free-noise` flag on `plakat animate` (opt-in; off by
  default preserves byte-identical numerics).
- ✅ `SdLoadRequest.vae_cache` + `t2i::LoadRequest.vae_cache`
  fields (`Option<Arc<AutoEncoderKL>>`). External callers pass
  `None` for the v0.31 behaviour.
- ✅ Every SD-family pipeline's CLIP-L field type is now the
  vendored CLIP — same forward-pass numerics, source-level
  type-lock prevents future regressions.

## What's new in v0.31 — diversify-2 (INT8 bail, four wins)

The cycle started as **diversify-2 + INT8 SDXL headline** — close
the v0.30 SDXL dual-encoder TI stretch goal, ship INT8 SDXL UNet
quantization for 12 GB GPUs, add weighted wildcards, close the
v0.29 mixed-kind pipeline cache carry. Phase 1's INT8 validation
spike rejected the codec direction (candle 0.10.2 has no
quantized Conv2d, and SDXL UNet is conv-heavy). The RFC's bail
plan triggered: swap phase 1 for **Pony preset + `plakat civitai
sync`** — two visible polish wins from the same deferral list.

Four phases shipped: two carry closures (SDXL dual TI from v0.30,
mixed-kind cache from v0.29) + two new feature surfaces (Pony +
civitai sync) + one prompt-power addition (weighted wildcards).

### SDXL dual-encoder TI parser

```bash
plakat generate "a portrait in <my-sdxl-style> art" --model sdxl \
    --embedding ./my-sdxl-style.safetensors
```

v0.30's vendored CLIP + tokenizer-mutation infrastructure was
CLIP-L-only. v0.31 phase 0 drops the parser bail for the
`clip_l` + `clip_g` dual format and mirrors the v0.30 extension
pattern through SDXL CLIP-G. A stack can mix single-encoder and
dual-encoder TIs.

### Pony Diffusion preset

`--model pony` (also `pony-v6`, `pony-diffusion-v6`) resolves to
`AstraliteHeart/pony-diffusion-v6-xl`. `--look pony` prepends
the Pony quality tags (`score_9, score_8_up, ...`) and applies a
Pony-tuned negative.

### `plakat civitai sync USERNAME --out DIR`

Bulk-download a Civitai creator's library. Walks API cursor
pagination, picks each model's primary version + primary file,
copies into `--out DIR`. Idempotent on rerun; honours
`CIVITAI_API_KEY`.

### Weighted wildcards

`{WEIGHT::CHOICE|...}` syntax adds explicit weights to inline
alternation (relative; normalized; omitted defaults to 1.0).
Composes with the v0.16 nested syntax.

### Mixed-kind scenarios pipeline cache

Closes the v0.29 carry. Scenarios that mix `type: generate` and
`type: animatediff` tasks now drop the opposite-kind cached
pipeline at kind boundaries — peak memory drops by ~5-10 GB.

### Release workflow finally green

After four iterations across v0.29/v0.30/v0.31, the arm64
cross-build's apt-source dance landed clean: wholesale wipe of
`/etc/apt/sources.list`, `/etc/apt/sources.list.d/*`, and
`/etc/apt/apt-mirrors.txt`, then writes only two explicit
sources.

### By the numbers

- 1020 lib + 47 integration tests = **1067 active tests** (+23
  lib across the cycle).
- 4 phase commits + RFC + close-out + CI workflow fix.

## What's new in v0.30 — diversify + one animate theme

After three consecutive AnimateDiff cycles (v0.27 scope, v0.28
single-script polish, v0.29 batch polish), v0.30 picked fresh
ground. It closed the **longest-running open carry** (Textual
Inversion runtime injection, deferred since v0.16 — eight cycles
ago), extended v0.28's LCM scheduler wiring to single-image t2i,
shipped the **most-requested animate carry** (per-frame video
ControlNet), and enriched `plakat doctor` to cover the new
surface. Five phases, ~2150 LOC delta.

### Embedding (Textual Inversion) runtime injection

```bash
plakat generate "a portrait in <my-style> art" \
    --embedding ./my-style.safetensors
```

Civitai TIs now apply at inference time on SD 1.5 / SD 2.1 / SDXL
CLIP-L. The parser, merger, and `plakat embedding info`
inspector had shipped since v0.16; what was missing was a text
encoder that would accept a vocab larger than candle's stock
`clip::Config.vocab_size` (private). v0.30 vendors a minimal
~430-LOC CLIP text encoder (`src/pipelines/vendored_clip.rs`)
with a public `vocab_size` + `Config::with_vocab()` builder. The
existing merger appends TI vectors to `token_embedding.weight`
via a tempfile (mirroring LoRA's merge pattern); the tokenizer
gets the new trigger tokens registered via
`Tokenizer::add_tokens`. Multi-vector TIs render as `N`
consecutive tokens (`trigger`, `trigger_1`, ..., `trigger_{N-1}`).
SDXL dual-encoder TIs (files with both `clip_l` and `clip_g`)
were still rejected in v0.30; closed in v0.31 phase 0.

### LCM-LoRA in t2i

```bash
# Auto-detects from the LoRA source — no flag needed for canonical names
plakat generate "a misty forest, dawn light" \
    --lora latent-consistency/lcm-lora-sdv1-5
# → scheduler=lcm, steps=4, guidance=1.5 (~10× speedup)
```

Extends v0.28's AnimateLCM scheduler wiring to single-image
generation. Substring heuristic on `--lora` sources catches the
canonical `latent-consistency/lcm-lora-*` repos automatically;
explicit `--lcm` flag covers non-canonical names. SD 1.5 + SDXL.

### Per-frame video ControlNet (video-to-video)

```bash
plakat animate --animatediff --model sd15 \
    --from "a glowing neon dragon, cyberpunk alley, rain" \
    --control-spec 'openpose:video=./reference.mp4:strength=0.9' \
    --frames 16 --format mp4
```

The headline animate carry on every deferral list since v0.27.
`video=PATH` triggers ffmpeg input decode, even sub-sampling to
the animate frame budget, per-frame annotation, and per-frame CN
residuals injected through SD 1.5 + SDXL AnimateDiff sampling.
Composes with sliding-window long-form. HJSON scenarios pick up
the new `video:` field on `controls[]` entries.

### `plakat doctor` enrichment

Two new sections (human + `--json`): ffmpeg version probe (warn
when missing) and HF / Civitai API token presence (boolean only —
never the value).

### By the numbers

- 997 lib + 47 integration tests = **1044 active tests** (+35
  lib + +6 integration across the cycle).
- New module: `pipelines::vendored_clip` (~430 LOC).
- v0.16 phase 9 carry **closed**.
- v0.27 video CN deferral **closed**.

## What's new in v0.29 — batch productivity completion

v0.28 made the **single-script** AnimateDiff surface pleasant.
v0.29 makes it pleasant **at production scale** — closing the
loudest v0.28 deferrals: animate in HJSON scenarios (the biggest
plakat batch-driver gap), SDXL in `plakat.animate`, and the
final `animate_format` Bund config key. Six phases, ~1100 LOC
delta, zero new architectural risk.

### Animate in HJSON scenarios

```hjson
{
    model: sd15
    type: animatediff       # scenario default — every task is animate
    frames: 16
    lcm: true               # 4-step AnimateLCM (~5× speedup)
    format: gif
    out: ./out/animations
    scene:   [ { name: dawn,  prompt: "at dawn" } ]
    weather: [ { name: mist,  prompt: "misty" } ]
    tasks: [
        { name: cottage, scene: dawn, weather: mist,
          prompt: "a watercolor cottage" }
        { name: knight,  scene: dawn, weather: mist,
          prompt: "a knight in a forest, oil painting",
          frames: 32, format: mp4 }     # per-row overrides
    ]
}
```

```bash
plakat scenario animate.hjson --dry-run    # preview the plan
plakat scenario animate.hjson              # render every task
plakat scenario animate.hjson --resume     # skip rendered tasks
plakat scenario animate.hjson --only knight
```

The same scenario filters (`--resume` / `--only` / `--limit` /
`--dry-run`) work unchanged. Per-task overrides for `frames`,
`window-size`, `window-overlap`, `lcm`, `motion-lora`,
`motion-lora-scale`, `format`, `gif-delay-ms` compose with
scenario-level defaults. ControlNet through the existing
`control:` / `controls:` fields (multi-CN sum). All-animate
scenarios don't need the `enhancer:` field set (prompt
enhancement is t2i-only). Mixed-kind scenarios still need it for
the generate tasks.

Closes the **largest plakat batch-driver gap** identified in the
v0.28 cycle audit: `cli/scenario.rs::TaskDef` had zero animate-
related fields before this release.

### SDXL `plakat.animate`

```bund
"sdxl" plakat.load
16   "animate_frames" plakat.config.set
1024 "width"          plakat.config.set
1024 "height"         plakat.config.set
"a knight in a forest, oil painting" "./out" plakat.animate
```

Removes the v0.28 SD-1.5-only restriction in scripting. New
`ScriptCtx::loaded_animatediff_sdxl` cache slot mirrors the
v0.26 stylize slot pattern so multi-call scripts amortise the
~7 GB SDXL backbone load. SD 1.5 keeps its own cache slot
(`loaded_animatediff`) with a key encoding the LCM mode —
toggling `animate_lcm` between calls swaps the pipeline.
AnimateLCM-SDXL still bails loud (upstream repo not publicly
available).

### `animate_format` Bund config key

```bund
"sd15" plakat.load
"mp4" "animate_format" plakat.config.set
"a watercolor cottage" "./out" plakat.animate
```

The final v0.28 Bund surface gap. `animate_format` accepts the
same five strings as the CLI's `--format`: `frames | gif | mp4 |
webm | all`. MP4 / WebM need ffmpeg on `$PATH`; the availability
check fires before inference so install pointers come fast.

### CI workflow fix

The arm64 cross-build matrix step in `.github/workflows/release.yml`
gained a rewrite over `/etc/apt/apt-mirrors.txt` to use only the
single Azure mirror. The runner image's mirrorlist
(`URIs: mirror+file:///etc/apt/apt-mirrors.txt`) had grown
multiple amd64-only entries (security.ubuntu.com,
archive.ubuntu.com) that the v0.26 phase fix to `.list` / `.sources`
files didn't reach. v0.29 patches that hole.

### By the numbers

- 962 lib tests + 41 integration tests = **1003 active tests**
  (+9 lib + +4 integration across the cycle).
- 6 phase commits + RFC + CI workflow fix.
- 50 host words (unchanged); 1 new config key (`animate_format`).

### v0.28 → v0.29 migration

v0.29 was **fully additive**. Every existing flag, host word,
config key, and scenario field kept its v0.28 shape.

## What's new in v0.28 — AnimateDiff productivity polish

v0.27 made AnimateDiff feature-complete. v0.28 made it pleasant
to use in practice — closing the loudest v0.27 deferrals with
four targeted productivity wins.

### 4-step animate via AnimateLCM

`--lcm` switches the motion adapter to `wangfuyun/AnimateLCM`
(17 modules: 16 + a V1/V2-style mid-block), the scheduler to LCM,
and applies the diffusers-recommended defaults `steps=4
guidance=1.5`. SD 1.5 only (SDXL AnimateLCM not publicly
available).

### Multi-ControlNet stacking through animate

`--control-spec` repeatable, same grammar as `plakat generate`'s.
Mutually exclusive at parse time with the legacy single-CN flags.
Both SD 1.5 + SDXL supported.

### `plakat.animate` Bund host word

The last major CLI verb missing from Bund scripting. Stack effect:
`( prompt out_dir -- )`. Four new config keys: `animate_frames`
(16), `animate_window_size` (16), `animate_window_overlap` (4),
`animate_lcm` (false). SD 1.5 only in v0.28 (SDXL animate
scripting landed in v0.29).

### `plakat motion-adapter` inspection

`plakat motion-adapter list` + `plakat motion-adapter info REPO`,
parallel to `plakat civitai info`. Dumps motion adapter config +
per-block tensor breakdown.

### By the numbers

- 953 lib + 37 integration tests green (+5 lib + +6 integration
  across the cycle).
- 6 phase commits + RFC.
- 49 → 50 host words.
- AnimateLCM loader added; shares `load_from_repo` +
  `load_with_motion_loras` with V3 + SDXL beta.

## What's new in v0.27 — AnimateDiff feature complete

Eight phases close every AnimateDiff carry from the v0.26 cycle.
End-to-end inference now works on both SD 1.5 + SDXL, each with
optional ControlNet conditioning and a sliding-window long-form
mode that lifts V3's 32-frame cap.

### AnimateDiff inference (SD 1.5 + SDXL)

V3 SD 1.5 (`guoyww/animatediff-motion-adapter-v1-5-3`, 16 motion
modules) and SDXL beta (`guoyww/animatediff-motion-adapter-sdxl-beta`,
12 motion modules) share the same `MotionAdapterConfig` schema; the
only difference is `block_out_channels` matching each base UNet's
block layout. Both paths use the block-boundary motion splice
(sequential motion modules at each down/up block output).

### AnimateDiff + ControlNet

Single conditioning image, same hint applied to every frame. Five
kinds supported (`depth` / `canny` / `openpose` / `lineart` /
`softedge`). The CN runs at the full per-step batch and feeds
residuals into the motion UNet's down + mid hooks.

### Long-form sliding window

V3's 32-frame `motion_max_seq_length` was a hard cap on a single
window. Long-form mode chains overlapping windows with linear-ramp
latent-space blend, lifting the practical cap to ~256 frames.

### Motion-module tensor naming fix

A latent bug from the v0.26 phase 2 motion-module code: the
referenced tensor key paths (`motion_modules.{j}.temporal_transformer.norm`,
`attention_blocks.{0,1}`, `norms.{0,1,2}`, `pos_encoder.pe`) don't
exist in the real upstream safetensors. v0.27 phase 2 fixed all the
paths and collapsed `Vec<TemporalTransformerBlock>` → a single
inner block per motion module. Verified upstream against V3 SD 1.5
(540 keys) and SDXL beta (405 keys); both share the same convention.

### By the numbers

- 948 lib tests + 32 integration tests green (+14 lib across the cycle).
- 8 phase commits + RFC.
- 49 host words (unchanged from v0.26).
- SD 1.5 + SDXL motion adapter loaders share `load_from_repo` +
  `load_with_motion_loras` helpers.
- Shared free helpers `validate_long_form_window` +
  `stitch_long_form<F>` agree the SD 1.5 and SDXL `generate_long`
  paths on the blend math by construction.

## What's new in v0.26 — AnimateDiff infrastructure + every v0.25 carry closed

Thirteen phases close **every v0.25 carry** in one cycle: SD3 /
SD3.5 animate, Bund look/genre apply on Flux + SD3, scenario
auto-LoRA discovery, Real-ESRGAN in `plakat.upscale`,
`plakat.metadata.write`, `plakat.stylize` cache slot, plus the
full **AnimateDiff infrastructure** (motion adapter, temporal
modules, vendored UNet with motion splice, motion LoRAs, output
formats). Host word count 48 → 49.

End-to-end AnimateDiff inference was deferred to v0.26.1, then
folded into v0.27 phase 0 (where it closed alongside SDXL +
ControlNet + long-form sliding window).

### SD3 / SD3.5 animate

Closes the v0.20-era `plakat animate` SD3 bail:

```bash
# Three-encoder lerp (CLIP-L + CLIP-G + T5) + flow-match per frame.
plakat animate --model sd35-medium \
    --from "a quiet temple at dawn" \
    --to   "a quiet temple at sunset" \
    --frames 16 --gif-delay-ms 125
```

### `plakat.look.*` / `plakat.genre.*` on Flux + SD3

The v0.25 look/genre apply was SD-family only. v0.26 wires it on
the Flux + SD3 generate paths in `script_entry.rs`. LoRA
discovery filters by `BaseFamily::Flux` / `BaseFamily::Sd3` so
the cache key `(name, base_family)` finds the right LoRAs.

### Scenario auto-LoRA discovery (smart-cached)

100 tasks with `look: watercolor` fire **one** network call —
the discovered LoRA is cached per `(look_name, base_family)`
across the scenario.

### Real-ESRGAN in `plakat.upscale`

```bund
1 "real-esrgan-x4" plakat.upscale         // ML x4 upscale
1 "real-esrgan-anime-x4" plakat.upscale   // anime-tuned variant
```

`plakat.upscale` dispatches on arg type: integer (2 / 4) =
Lanczos, string = Real-ESRGAN ML method.

### `plakat.save` metadata + `plakat.metadata.write`

`plakat.save` writes the A1111 `parameters` PNG tEXt chunk +
JSON sidecar automatically when the handle has metadata
attached. New host word `plakat.metadata.write` re-attaches
metadata to an existing file.

### `plakat.stylize` cache slot

Closes the v0.24 'caching is a v0.25+ optimisation' deferral.
Multi-call scripts amortise the ~5 GB SD1.5 + IP-Adapter
weights load.

### AnimateDiff infrastructure

- AnimateDiff V3 motion adapter loader
  (`guoyww/animatediff-motion-adapter-v1-5-3`)
- 16 per-block temporal-attention modules built from real V3 weights
- Vendored SD 1.5 UNet (`Sd15MotionUNet`) with motion-module splice
  at block-output boundaries
- Motion LoRA composition (`--motion-lora SPEC`) — merges into the
  motion-adapter weights via the existing LoRA-merge pipeline
- `--format {frames, gif, mp4, webm, all}` with ffmpeg integration
- `AnimateDiffPipeline` assembly type

### By the numbers

- 934 lib tests + 20 integration tests green (+32 lib across the cycle).
- 13 phase commits + RFC.
- 48 → 49 host words.
- 7 v0.25 carries closed.
- New `MergeTarget::MOTION_ADAPTER` LoRA-merge variant.
- New `loaded_stylize` cache slot on `ScriptCtx`.

## What's new in v0.25 — art-medium presets + auto-LoRA discovery

Twelve phases ship two new preset axes — `--look` (art medium)
and `--genre` (subject domain) — with **automatic LoRA discovery**
from Civitai / HuggingFace / your local cache. Pick "watercolor"
or "ink-wash" with one flag and plakat composes the prompt, picks
a sampler, and finds a compatible LoRA matched to your loaded
base model. Host word count 42 → 48.

### `--look` — eight bundled art mediums

```bash
plakat generate --model sd15 --look watercolor "a cottage in the woods"
plakat generate --model sdxl --look oil-painting "a still life"
```

Eight mediums ship: `ink-wash`, `watercolor`, `oil-painting`,
`charcoal`, `pencil`, `chalk-pastel`, `linocut`, `gouache`. Each
bundles a prompt prefix/suffix + recommended sampler/steps/guidance
+ a `lora_query` that drives auto-discovery.

### `--genre` — independent subject-domain axis

```bash
plakat generate --model sdxl --look watercolor --genre anime "a knight"
```

`anime` ships built-in; user-extensible via `$CONFIG_DIR/genres/*.json`.

### Auto-LoRA discovery chain (Civitai → HF → local)

`--lora` empty → plakat searches for a compatible LoRA, filtering
by base-model compatibility. Trigger words from the discovered LoRA
auto-prepend to the prompt. `--offline` short-circuits to cache +
local-scan only.

### Surfaces

`--look` / `--genre` / `--offline` work on every prompt-driven
subcommand (`generate`, `portrait`, `img2img`, `inpaint`,
`outpaint`), in scenarios at both global + per-task level, and via
six new Bund host words (`plakat.look.{apply,clear,list}` +
`plakat.genre.{apply,clear,list}`).

### By the numbers

- 902 lib tests + 20 integration tests green (+85 lib across the cycle).
- 12 phase commits + RFC.
- 42 → 48 host words.
- 8 bundled looks + 1 bundled genre + user-extension directories wired.
- 3-source discovery chain (Civitai + HF Hub + local-cache scan)
  with on-disk caching keyed by `(name, base_model)`.

### Documentation

- [`LOOKS.md`](LOOKS.md) — flag reference + user-extension format.
- [`GENRES.md`](GENRES.md) — subject-domain axis.
- [`LOOKS_TUTORIAL.md`](Tutorials/LOOKS_TUTORIAL.md) — walkthrough.
- [`GENRES_TUTORIAL.md`](Tutorials/GENRES_TUTORIAL.md) — companion.
- [`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md) — design doc.

## What's new in v0.24 — persona depth + scripting completion

Eleven phases finish the Bund scripting surface. The
v0.21/22/23 arc closes: after v0.24 there's no "use the CLI
for X" gap for users staying in scripts. Word count 33 → 42.

### Persona depth (phases 1–3)

```bund
// Multi-photo portrait (BREAKING — see migration below).
"./alice.jpg" 0.7 plakat.portrait.photo.add
"./bob.jpg"   0.3 plakat.portrait.photo.add
"a couple at the beach" plakat.portrait

// Face alignment overrides (CLI parity with --face-bbox /
// --face-landmarks).
"0.2,0.1,0.8,0.7" "face_bbox" plakat.config.set
"0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68"
    "face_landmarks" plakat.config.set

// Identity-encoder variant override (the four FaceID kinds).
"face-id-sdxl" "identity_kind" plakat.config.set
```

CLI flags `--photo` / `--face-bbox` / `--face-landmarks` / all
four `--identity` variants already existed; v0.24 wires the
scripting surface to them.

### Scripting completion (phases 4–9)

Six v0.20+ carries close:

- **`plakat.outpaint`** — `( prompt input expand-spec -- handle )`.
- **`plakat.embedding.*`** — Textual Inversion stack (add / clear / list).
- **`plakat.stylize`** — IP-Adapter style transfer.
- **`plakat.metadata.read`** — JSON sidecar reader (read-only).
- **Flux + SD3 ControlNet `from=`** — lazy first-generate
  annotation; cached per-pipeline; dim changes invalidate.
- **Flux inpaint** via `plakat.inpaint` — wires the
  flux-fill-dev variant + channel-concat path.

### v0.23 → v0.24 migration

One backwards-incompatible change in phase 1:

```bund
// v0.23:
"alice.jpg" "a portrait" plakat.portrait

// v0.24:
"alice.jpg" 1.0 plakat.portrait.photo.add
"a portrait" plakat.portrait
```

`plakat.portrait` no longer takes a photo arg; photos come from
the `plakat.portrait.photo.add` collection stack.

### By the numbers

- 817 lib tests green (+45 across the cycle).
- 11 phase commits + RFC.
- 33 → 42 host words.
- Three new config keys (face_bbox / face_landmarks /
  identity_kind).

### Documentation

- [`SCRIPTING.md`](SCRIPTING.md) — full reference, updated for v0.24.
- [`SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md) §11.
- [`RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md`](RFC_v0.24_SCRIPT_SURFACE_COMPLETION.md) — design doc.

## What's new in v0.23 — Bund deferrals closed

Nine phases close every "deferred to v0.23" stub v0.22 explicitly
took on, plus add two new things: the `plakat.style.*` catalog
namespace and the `plakat.inpaint` host word. Word count
28 → 33; smaller cycle than v0.22 (~7 phases of real work).

### Cache architecture: the SdT2i slot

`plakat.load "sdxl"` now warms a `t2i::Pipeline` slot in addition
to (or sharing `Arc<SdCore>` with) the v0.22 `portrait::Pipeline`
slot. `plakat.generate`'s SD-family path routes through SdT2i,
which carries the SDXL refiner UNet hook + the CLIP-skip encode
path that the v0.22 portrait cache didn't expose.

### `plakat.style.*` namespace (phase 4)

```bund
"poster-bold" plakat.style.apply       // by id
"./ref.jpg"   plakat.style.detect      // CLIP-H detect from photo
plakat.style.list                      // ( -- ...ids count )
plakat.style.clear
0.7 "style_strength" plakat.config.set
"a town square" plakat.generate
```

Resolution runs lazily at `plakat.generate` request-build time:
catalog LoRAs override the user LoRA stack for the load (CLI
parity with `--style ID`); trigger prepends to prompt;
`negative_extras` appends to negative.

### `plakat.inpaint` host word (phase 5)

```bund
"stained glass window in the wall"
   "./photo.png" "./mask.png"
   plakat.inpaint
   "result.png" plakat.save
```

Stack: `( prompt input mask -- handle )`. SD-family + SD3 wired;
Flux bail closed in v0.24.

### Flux + SD3 ControlNet (phases 6–7)

CN stack wires into `LoadRequest.controlnets` at pipeline-load
time. Stack mutations call `mark_controlnets_changed` which
drops the Flux/SD3 slot. v0.23 cap: `image=` specs only;
auto-annotate (`from=`) deferred to v0.24 phase 8.

### By the numbers

- 772 lib tests green (+14 across the cycle).
- 28 → 33 host words. `style_catalog` config key added; six v0.22
  deferred items closed.

### Documentation

- [`SCRIPTING.md`](SCRIPTING.md) — full reference, updated for v0.23.
- [`SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md) §10.
- [`RFC_v0.23_BUND_DEFERRALS.md`](RFC_v0.23_BUND_DEFERRALS.md) — design doc.

## What's new in v0.22 — Bund words expansion

The v0.21 scripting MVP graduates to feature parity with the
CLI. Twelve phases ship 21 new host words across 7 namespaces,
a pipeline cache so multi-call scripts don't reload the model,
all three model families on the scripting surface, and 50+
new Category-B config keys.

### The 28 `plakat.*` words

```
// v0.21 core (cache-aware in v0.22):
plakat.load / .generate / .img2img / .portrait / .upscale / .save / .config.set / .echo

// LoRA stack (phase 4):
plakat.lora.add / .clear / .list

// ControlNet stack (phase 5):
plakat.controlnet.add / .annotate / .spec / .clear / .list

// Post-process toggles:
plakat.refiner.{enable,disable}      // phase 6: SDXL refiner switch
plakat.adetailer.{enable,disable}    // phase 7: SCRFD face refinement
plakat.hires.{enable,disable}        // phase 8: upscale + img2img refine
plakat.artefact.add / .clear / .list / .blend.{enable,disable}   // phase 9

// Prompt enhancer (phase 10):
plakat.enhance ( prompt -- enhanced )
```

### Pipeline cache + all three families

`ScriptCtx::loaded` holds one `LoadedPipeline` enum
(`SdFamily(portrait::Pipeline)` / `Flux(flux::Pipeline)` /
`Sd3(sd3::Pipeline)`). Same-alias reuse skips the model load
entirely. Family-specific config keys (`quantize_t5`,
`kontext_bucket` for Flux; `tiled`, `tile_size`,
`tile_stride` for SD3) flow through the same
`plakat.config.set` surface.

### By the numbers

- 758 lib tests green (+124 new across the cycle).
- 12 phase commits + 2 RFC commits + 1 release-notes commit.
- 28 host words (was 8 in v0.21). ~60 `GenerationConfig` keys.
- Four composition tests exercise multi-namespace state
  interaction end-to-end.

### Deferred to v0.23 (all closed in v0.23)

- Flux + SD3 ControlNet load-time wiring.
- SDXL refiner UNet load.
- `plakat.inpaint` (mask path argument).
- `clip_skip` wiring.
- `plakat.style.*` namespace.

## What's new in v0.21 — Bund scripting

One big swing this cycle: `plakat run SCRIPT.bund` ships a
stack-based DSL for driving plakat's pipelines from a script.
Composition wins that were awkward at the CLI (`generate →
upscale`, `generate → img2img → save`, multi-variation runs at a
pinned seed) become one-liners; an interactive REPL on the same
surface lands for exploration.

### The seven `plakat.*` words

```
plakat.load        ( model-alias -- )
plakat.generate    ( prompt -- handle )
plakat.img2img     ( prompt input -- handle )      // input: path OR handle
plakat.portrait    ( prompt photo -- handle )      // photo: path OR handle
plakat.upscale     ( handle scale -- handle )      // Lanczos x2/x4
plakat.save        ( handle path -- )
plakat.config.set  ( value key -- )                // steps/guidance/seed/...
```

```bund
"sdxl" plakat.load
40   "steps"     plakat.config.set
3.5  "guidance"  plakat.config.set
"a fox in a meadow" plakat.generate    // handle 1
  2 plakat.upscale                     // handle 2 (2048x2048)
  "fox-2k.png" plakat.save
"a fox in a meadow, painterly oil"
  1  plakat.img2img                    // refine handle 1 → handle 3
  "fox-refined.png" plakat.save
```

Handles address rendered images in an in-memory registry —
chains compose without disk round-trips. Sources aren't consumed
by downstream words, so the same generation can fan out into
upscale + img2img + portrait variants from one root.

### Interactive REPL

```text
$ plakat run --repl
plakat REPL (v0.21). Type .help for commands, .q to exit.
plakat> "sd15" plakat.load
plakat> 50 "steps" plakat.config.set
plakat> "a fox" plakat.generate
=> 1
plakat> "fox.png" plakat.save
plakat> .s
  [0] 1
plakat> .q
```

Persistent state across lines, history at `<plakat-config-dir>/repl_history`,
Forth-style meta-commands (`.q` / `.s` / `.help`), the `=>` echo
shows the top of the workbench after each successful eval.

### v0.21 limitations

- **SD-family only** — `sd15`, `sd21`, `sdxl`, `sdxl-turbo`.
  Flux + SD3 / SD3.5 bail at `plakat.load` with a clear "phase
  2b" pointer; both land in v0.22.
- **Single-photo portrait** — no FaceID / multi-photo / manual
  landmarks. v0.22.
- **Lanczos x2/x4 upscale only** — Real-ESRGAN ML upscaling in
  v0.22.
- **No LoRA / ControlNet / refiner words** — use the CLI directly
  if you need them; the scripting surface stays minimal in v0.21.
- **No pipeline cache** — every `plakat.generate` reloads the
  model. Acceptable for the MVP; cache work is v0.22.

### Documentation

- [`SCRIPTING_TUTORIAL.md`](Tutorials/SCRIPTING_TUTORIAL.md)
  — narrative walkthrough: syntax in 60 seconds, every word with
  examples, composition patterns, the REPL, limitations.
- [`SCRIPTING.md`](SCRIPTING.md) — reference manual.
- [`RFC_v0.21_BUND_SCRIPTING.md`](RFC_v0.21_BUND_SCRIPTING.md)
  — design RFC + the seven locked architectural decisions + phase
  plan. Read if you're contributing a new `plakat.*` word.

### By the numbers

- 634 lib tests green (+65 new across the cycle).
- 8 phase commits + 1 RFC commit + 1 release-notes commit.
- 8 host words (7 MVP + `plakat.echo` smoke). 9 `GenerationConfig`
  knobs.

## What's new in v0.20 — recipe replay, project bootstrap, Flux animate, Kontext + tiled

Nine features in three groups. v0.20 picks up where v0.19 left
off on workflow polish (recipe-driven replay, project
bootstrap, user-defined negative-preset catalogs, Civitai
trigger-word display), then lands two unlock-grade composition
wins: Flux Kontext + tiled denoise for hi-res reference edits,
and Flux animate (T5 + CLIP-L lerp, flow-match per frame).

### Top picks (3 features)

- **`plakat generate --recipe FILE.json`**. Replay any prior
  generation from its JSON sidecar. Recipe fields fill in only
  where the CLI didn't set the flag explicitly — `--model`,
  `--seed`, `--negative`, etc. pass through unchanged when you
  override them. Useful for "re-render at higher steps" /
  "swap one LoRA, keep everything else" iterations. The
  prompt is never overridden — the recipe is structural, the
  prompt is creative.

  ```bash
  # Re-render a v0.17 generation at higher quality
  plakat generate "$(cat in.prompt)" --recipe in.json --steps 50
  ```

- **Flux + SD3 WebP output**. v0.19 shipped WebP on SD-family
  only with a warn+fallback for the modern backbones. v0.20
  threads `--format png|webp` through the Flux and SD3
  pipelines too. WebP is ~30% smaller at perceptually-
  equivalent quality; the JSON sidecar still works on every
  backbone (so `plakat metadata` / `plakat clone` round-trip
  unchanged).

- **Civitai LoRA trigger-word display**. When a `--lora
  civitai:NNNNNN` resolves (cache hit or fresh download),
  plakat now prints the LoRA's trained trigger words inline:

  ```text
    ✦ Civitai LoRA 2595428 (v2614696) trigger words: watercolor_(medium), some_trigger
      → consider adding these to your prompt for the LoRA to activate
  ```

  Silent LoRAs (no apparent effect because triggers were
  missing from the prompt) is one of the most common
  Civitai-LoRA friction points; this surfaces the fix at the
  exact moment users need it.

### Round-out (4 features)

- **`plakat models aliases [--family F] [--repo] [--gated]`**.
  Enumerates every `--model` short-name plakat recognises,
  grouped by family. `--family flux` filters; `--repo` prints
  bare HF repo ids (pipes into `xargs plakat models pull`);
  `--gated` lists HF_TOKEN-only repos. Refactor: the
  hand-written alias `match` became a static `ALIAS_TABLE`
  so adding an entry updates both resolution and the listing.

- **`plakat init [DIR]`**. Bootstraps a runnable starter
  project — `scenario.hjson` (sd15, `enhancer: local`, two
  tasks), `wildcards/` (subject / style / lighting with three
  options each), and a focused `.gitignore`. Targets the
  ungated SD 1.5 + on-device LLM enhancer so first-run users
  with no HF token + no API key can generate end-to-end.
  Companion fix: `scenario`'s enhancer validator gained the
  `local` / `local:<alias>` / `auto` providers (previously
  cloud-only — the gap is why a fresh init scenario couldn't
  dry-run).

- **User-defined negative-preset catalogs**. Drop a `.txt` file
  into `<plakat-config-dir>/negative-presets/` and the
  filename becomes a `--negative-preset` name. User files
  override built-ins; safety-checked names; empty files fall
  through to the built-in. Error output marks entries as
  `<name> (user)` or `<name> (user override)`.

- **`--enhance-keep-original`**. New flag on `plakat generate`
  and `plakat portrait`: joins the enhancer's rewrite with
  the user's original prompt via the SD-family `BREAK`
  keyword (each chunk gets its own 77-token CLIP slot, so
  original terms aren't diluted by the enhancer's added
  detail). SD-family only by design; Flux / SD3 warn once
  (their T5 ignores BREAK and has the budget to carry both
  phrasings).

### Big swings — Kontext + tiled, Flux animate (2 features)

- **Flux Kontext + tiled denoise**. Lifts the v0.18 bail at
  the Kontext + `--tiled` junction. Each tile slices the
  matching region of the reference latent, packs it,
  seq-concats onto the tile's noise tokens, pads CN
  residuals for the reference half, runs forward, strips the
  reference tail. Per-tile RoPE budget check fires up front
  (Kontext + tiled doubles the per-tile sequence; the bail
  interpolates the largest safe `--tile-size` into the error
  message, typically ≤608 px for Kontext-dev).

  ```bash
  plakat generate "fold the dress into a flowing cape" \
      --model flux-kontext-dev --concept-image portrait.jpg \
      --size 2048x2048 --tiled --tile-size 512
  ```

- **Flux animate**. `plakat animate --model flux-dev` (and
  `--model flux-schnell`) now work. Pre-encodes both endpoint
  prompts through CLIP-L + T5-XXL **once**, then per frame:
  lerp the `(clip_pooled, t5_emb)` pair → run Flux's
  flow-match denoise → save. T5 encode dominates the cost,
  so amortising it across frames is the whole point of
  animate. New `pub fn animate_frame` on `flux::Pipeline`;
  Kontext / Fill / Canny / Depth refused (no place for a
  reference per call). Flux is guidance-distilled, so
  `--negative` is a no-op (warns) and `--guidance` is the
  scalar that goes straight to the model — drop to 3.5
  (Dev) / 0.0 (Schnell).

  ```bash
  plakat animate \
      --from "an oil painting of a fox in a meadow" \
      --to   "an oil painting of a cat in a meadow" \
      --frames 24 --seed 42 --guidance 3.5 \
      --model flux-dev --size 1024x1024 --out ./morph --gif
  ```

### Deferred to v0.21

- **SD3 / SD3.5 animate** — the three-encoder (CLIP-L +
  CLIP-G + T5) lerp + MMDiT rectified-flow integrator wiring
  is its own refactor. `plakat animate --model sd35-*` bails
  with a clear "deferred" message; Flux animate in v0.20 is
  the proving ground for the per-frame-encoding approach
  SD3 will follow.
- **AnimateDiff** — motion-adapter weights + temporal-attention
  injection into the SD UNet. Genuinely new architecture
  (not covered by candle 0.10.2); slated for v0.21+ as its
  own multi-cycle effort rather than rushed into v0.20.

569 lib tests green; +60 new tests across the cycle.

## What's new in v0.19 — local enhancer polish, partial-rerun, WebP, Kontext compositions

Nine features in three groups. Pairs with v0.18's larger surface
(Flux Kontext, A1111 attention on Flux+SD3, BREAK, local prompt
enhancer) — v0.19 sands down rough edges and unblocks the two
Kontext compositions deferred from v0.18.

### Top picks (3 features)

- **Enhancer CLI flag surface + disk cache**. The v0.18 local
  enhancer's internals get CLI flags: `--enhance-temp F` (default
  greedy / `0.0`), `--enhance-max-tokens N` (default 96),
  `--enhance-system PATH` (custom system prompt), and
  `--enhance-cache` (opt-in SHA-256 disk cache at
  `~/.cache/plakat/enhance/`). Cache hits skip the LLM forward
  entirely — scenarios re-enhancing the same prompts across runs
  go from ~3-5s per prompt to instant.
- **`plakat animate --resume`**. Long animates that crash on frame
  23 of 24 no longer require re-rendering all 24. The flag scans
  `<out>/frame-NNNN.png`, skips frames already on disk, re-runs
  only what's missing. Mirrors the scenario `--resume` pattern
  added in v0.17.
- **scenario `--only TASK[,TASK,…]` + `--limit N`**. Partial-rerun
  affordances for long batches. `--only` runs just the named
  tasks (typo'd names bail up front with the supported list);
  `--limit` runs the first N. Both compose with `--resume` and
  `--dry-run`. `seed_offset` advances on skipped tasks so a
  partial run produces seeds identical to the full batch — no
  drift when iterating.

### Round-out (4 features)

- **`plakat doctor --json`**. Structured CI / scripting output
  alongside the v0.18 health-check sections. Covers build /
  runtime device match, libcuda driver shim probe, HF cache disk
  usage. `jq` consumers can assert `.device.aligned == true` or
  `.cache.severity == "ok"`.
- **`--negative-preset photo | painting | anime | cinematic`**.
  Four bundled negative-prompt presets. Combine with `--negative`
  for preset-plus-user-extras. Saves the daily-driver
  `"blurry, low quality, watermark, ..."` copy-paste.
- **`plakat clone PNG`**. Reverse of `plakat metadata` (v0.18):
  reads a generated PNG's recipe + emits the `plakat generate`
  shell command that would re-create it. JSON sidecar preferred
  for lossless reproduction; falls back to parsing the Auto1111
  `parameters` chunk for Civitai uploads / A1111 outputs.
  `--one-line` for pipes.
- **WebP output format**. `--format png | webp` on
  `plakat generate`. WebP ships ~30% smaller files at perceptually-
  equivalent quality. Trade-off: WebP can't carry the Auto1111
  tEXt chunk (no drag-and-drop into A1111 / Civitai / ComfyUI);
  the JSON sidecar still works, so `plakat metadata` / `plakat
  clone` round-trip on WebP outputs. SD-family pipeline only in
  this release; Flux / SD3 warn and fall back.

### FLUX.1 Kontext composition unlocks (2 features)

- **Kontext + ControlNet**. Lifts the v0.18 phase 2 bail.
  ControlNet residuals (computed per-block from the CN forward on
  noise tokens) get zero-padded along the seq dim for Kontext's
  reference half before being added to the per-block flux
  intermediate state. The reference tokens get no CN contribution
  — they're already conditioning via cross-attention. Unlocks
  "edit this image, preserve the depth/canny structure" workflows.

  ```bash
  plakat generate "make it golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --control-spec 'depth:from=input.png:strength=0.7'
  ```

- **Kontext + Redux**. Lifts the v0.18 phase 2 bail with a RoPE
  budget gate. Total effective attention seq (txt + img + ref + N
  Redux tokens) is computed at dispatch; soft warn at 3500
  positions, hard bail at 4096 with actionable cleanup hints.
  Unlocks "edit this image in the style of these references" —
  Kontext provides the layout, Redux provides the aesthetic.

  ```bash
  plakat generate "the same scene at golden hour" \
      --model flux-kontext-dev \
      --concept-image input.png \
      --redux-image style_ref.png:weight=0.5
  ```

### Two new tutorials

- [`SCENARIOS_TUTORIAL.md`](Documentation/Tutorials/SCENARIOS_TUTORIAL.md)
  — batch generation via HJSON. Cross-product expansion, per-task
  overrides, partial-rerun filters, real-world series-production
  examples.
- [`OUTPAINT_TUTORIAL.md`](Documentation/Tutorials/OUTPAINT_TUTORIAL.md)
  — `plakat outpaint INPUT.png`. Per-side flag grammar,
  VAE-snapped dimensions, model choice, iterative-stage workflow.

509 lib tests green; +40 new tests across the cycle.

## What's new in v0.18 — Flux Kontext, SDXL animate, BREAK, local enhancer, polish

The largest single-version cycle yet. Three workstreams plus a
follow-on wave of QoL features and three new tutorials.

### Top picks + round-out (7 phases)

- **A1111 attention syntax on Flux + SD3**. The v0.17 per-token
  weight broadcast (CLIP) now applies to T5-XXL hidden states on
  Flux and to all three penultimate streams on SD3 / SD3.5
  (CLIP-L + CLIP-G + T5). Every Civitai Flux LoRA card already
  embeds `(token:1.4)`-style emphasis in its example prompts;
  these now work as written. Sentencepiece alignment caveat
  documented.
- **SDXL `plakat animate`**. The prompt-morph animator (v0.17)
  extended from SD 1.5 / SD 2.1 to SDXL. Dual CLIP-L + CLIP-G
  hidden lerp, pooled `add_text_embeds` lerp, `add_time_ids`
  micro-conditioning threaded through.
- **Animate frame metadata**. Each `frame-NNNN.png` carries the
  Auto1111 `parameters` PNG tEXt chunk + a JSON sidecar with the
  lerp `t` parameter + `Animate from` / `Animate to` extras.
- **LCM-LoRA SD 1.5 `--fast` preset**. Same recipe as v0.17's
  `lcm-sdxl` against the smaller backbone — `--fast lcm-sd15` for
  4-step inference on SD 1.5 hardware.
- **`--grid` on `img2img` / `portrait` / `outpaint`**. The v0.17
  grid bundling now works on every `--count`-bearing subcommand.
  Per-backbone filename prefix preserved.
- **`--negative` attention verification**. Tests confirming the
  per-token weight broadcast works on the uncond branch across
  SD 1.5 / 2.1, SDXL, SD3.
- **`plakat doctor` enhancements**. Build / runtime device match,
  `libcuda.so.1` driver shim probe (Linux + `--features cuda`),
  HF cache disk usage report. Catches the CI-style "binary built
  with CUDA, no driver on host" silent fallback.

### FLUX.1 Kontext (BFL image editing)

Four phases bringing BFL's image-editing Flux variant online:

- **`--model flux-kontext-dev`** on `plakat generate` and
  `plakat img2img`. Reference image is VAE-encoded and
  sequence-concatenated onto the noise tokens (with
  `img_ids[..., 0] = 1` as the RoPE marker) — distinct mechanism
  from Canny/Depth which widen `img_in` to 128 channels.
- **`--concept-image PATH`** reused as the reference flag (same
  grammar as Canny/Depth, semantically the "image to edit"). On
  `plakat img2img`, the input positional becomes the reference
  natively.
- **GGUF support** via `unsloth/FLUX.1-Kontext-dev-GGUF`
  (`--model flux-kontext-dev-gguf`). Composes with LoRA (Kontext
  shares Dev's transformer layer names) and `--quantize-t5`.
- **`--kontext-bucket`** opt-in flag — snaps `--size` to the
  closest of 17 BFL-recommended Kontext resolutions before VAE
  encoding (off by default, surprise-free for non-Kontext flows).

### Follow-on wave (6 features)

- **`plakat metadata FILE.png`**. New subcommand reads the v0.17
  `parameters` PNG tEXt chunk + JSON sidecar back into the
  terminal. `--json-only` pipes cleanly to `jq`.
- **`--aspect`** on `plakat img2img`. Resolution priority:
  `--size > --aspect + --base > input image dims`. Composes with
  `--kontext-bucket`.
- **`plakat scenario --dry-run` polish**. The summary line now
  reads `(dry-run) would have generated …` instead of `✓ done`,
  and per-task previews show the output directory path so you can
  see file layout before launching a long batch.
- **A1111 inline `<lora:NAME[:weight]>` syntax**. Civitai LoRA
  cards embed these directly; plakat extracts them at the CLI
  boundary, parses via the v0.17 `LoraSpec` grammar (paths /
  HF repos / `civitai:NNN` shorthand), prepends to the LoRA
  stack, removes from the prompt before encoding.
- **`BREAK` keyword in prompts**. A1111 convention for chunking
  past CLIP's 77-token cap. Each chunk gets its own 77-token
  CLIP context; hidden states sequence-concatenate before
  cross-attention. SD 1.5 / 2.1 / SDXL; Flux + SD3 strip + warn
  (their T5 already has a 256/512-token budget).
- **Local prompt enhancer**. `--enhance local` runs a small
  instruction-tuned LLM in-process via candle's quantized
  backends. Qwen2.5-1.5B-Instruct (Q4_K_M, ~1 GB) as default,
  SmolLM2-360M (~230 MB) as CPU-budget fallback. Greedy decoding
  for reproducibility; `--enhance auto` picks DeepSeek → Gemini →
  local based on what env vars are set. No API key required for
  the local arm.

### Three new tutorials

- [`ADVANCED_PROMPTING_TUTORIAL.md`](Documentation/Tutorials/ADVANCED_PROMPTING_TUTORIAL.md)
  — attention syntax, BREAK, inline `<lora:>` as a coherent set.
- [`PROMPT_ENHANCER_TUTORIAL.md`](Documentation/Tutorials/PROMPT_ENHANCER_TUTORIAL.md)
  — `--enhance deepseek / gemini / local / auto`.
- [`METADATA_TUTORIAL.md`](Documentation/Tutorials/METADATA_TUTORIAL.md)
  — recovering recipes from PNG metadata.

465 lib tests green; +84 new tests across the cycle.

## What's new in v0.17 — the prompt + reproducibility release

Ten phases focused on **prompt expressiveness**, **reproducibility**,
and **animation**. The cycle also upgrades the underlying candle ML
framework two minor versions and adds the long-asked `--lora civitai:`
shorthand:

- **A1111 prompt syntax**. `(red:1.4)` emphasis / `[blue]`
  de-emphasis / `((nested))` compounding / `\(escape\)` — the
  grammar every Civitai LoRA card uses in its example prompts.
  Applied to CLIP penultimate hidden states via per-token broadcast.
  SD 1.5 / SD 2.1 / SDXL.
- **PNG metadata + JSON sidecar**. Outputs ship with the
  Auto1111-compatible `parameters` PNG tEXt chunk + a sibling
  `<filename>.json` carrying the full recipe. A1111 / Civitai /
  ComfyUI / sd-prompt-reader all surface the prompt + seed + LoRAs
  + scheduler inline. `--no-metadata` opts out.
- **`--grid` output**. `--count N > 1` + `--grid` writes a single
  `plakat-grid-<seed>.png` combining all N outputs in a near-square
  layout. `--grid-cols` / `--grid-padding` for fine control.
- **`plakat animate`**. New subcommand for prompt-morph animation:
  lerp CLIP embeddings between two prompts at a fixed seed,
  producing a smooth N-frame sequence. `--gif` bundles into an
  animated GIF. SD 1.5 / SD 2.1.
- **Live preview during denoise**. `--preview-every N` writes a
  cheap latent-projection PNG every N steps so long runs aren't a
  black box. Microseconds per write — no meaningful runtime cost.
- **scenario `--resume` / `--force`**. Crashed scenario picks up
  where it left off by probing for already-existing output PNGs.
  No more restart-from-task-0.
- **`--lora civitai:NNNNNN`**. Skip the explicit
  `plakat civitai download` step — the LoRA spec parser now
  downloads + caches Civitai assets on first use via the
  shorthand. `civitai-version:NNNNNN` pins a specific version.
- **LCM-LoRA SDXL `--fast lcm-sdxl`**. Latent-Consistency
  distillation for SDXL bundled with the right scheduler and
  4-step / CFG-1.5 defaults. ~5× speedup over stock SDXL.
- **candle 0.8 → 0.10.2 upgrade**. Single 8-line trait-impl fix
  for `SimpleBackend::get_unchecked`. GGUF / NF4 / MMDiT /
  vendored Flux all intact, 304 tests still green at upgrade
  time.
- **SDXL refiner cleanup**. The "Known limitation" about missing
  `add_embedding` on the refiner was outdated since v0.11 phase
  8e. Stale docs replaced; regression tests pin the 5-time-id
  config so future refactors can't silently break the refiner's
  `text_time` micro-conditioning.

381 lib tests green; +77 new tests across the cycle.

## What's new in v0.16 — the productivity release

A dozen quality-of-life landings that connect community workflows
(Civitai browsing, ADetailer face fix, Hires fix, wildcards) to the
existing plakat backbone, plus deeper SD3 integration:

- **SD3 ControlNet (InstantX)**. `--control-spec` works on SD3 /
  SD3.5 via the InstantX adapter family. Multi-CN composition,
  step-gating, auto-annotation from a reference photo — same
  ergonomics SDXL + Flux ControlNet ship.
- **Tiled Flux Fill**. `--tiled` composes with Flux.1-Fill-dev for
  4K+ inpaint. Per-tile masked-latent + mask packing.
- **Tiled SD3 img2img + inpaint**. The rectified-flow init lerp +
  RePaint mask blend compose with the per-tile Hann blend.
- **Wildcards**. `{red|blue|green}` inline alternation +
  `__name__` file wildcards (Auto1111 / NovelAI grammar). Seeded
  from `--seed` for reproducibility.
- **CLIP-skip**. `--clip-skip N` for SD 1.5 / SD 2.1 — N=2 is the
  community default for anime checkpoints.
- **ADetailer-style face refinement**. `--adetailer` runs SCRFD
  on each output, crops + img2img-refines each face, feather-
  composites back. Reuses the t2i SdCore — no extra model load.
- **Hires fix**. `--hires-fix` escapes the trained-resolution
  ceiling: upscale (Lanczos / Real-ESRGAN) + img2img-refine.
  Composes with `--adetailer` for a 4K → fixed faces pipeline.
- **Civitai browser + downloader**. `plakat civitai search`,
  `info`, `download` — drop the resulting path into `--lora` /
  `--model`. Atomic streaming downloads with cache-hit
  short-circuit.
- **Auto-annotation for Flux concept variants**. `--concept-from
  PATH` auto-annotates a photo through Canny / Depth before feeding
  Flux.1-Canny-dev / Flux.1-Depth-dev.
- **SD3 pipeline caching + per-task LoRA**. Scenarios with
  `--model sd35-*` now share one SD3 pipeline across tasks; per-
  task `loras:` swap at runtime via the LoraLinear stack.
- **Textual Inversion** *(partial)*. Parser + `plakat embedding
  info` inspector. Runtime injection blocked by candle 0.8's
  private `clip::Config.vocab_size` — wiring lands when the
  candle API surface opens or alongside a vendored CLIP path.
- **SD UNet per-task LoRA preflight** *(partial)*. Detects the
  blocker upfront and emits actionable YAML-fold hints; bails
  loud with three concrete workarounds. Full UNet vendoring
  deferred — same candle private-internals blocker.
- **XLabs Flux IP-Adapter parser** *(partial)*. Inspector that
  reports per-block attention count + SigLIP/Flux dims. Per-block
  injection blocked by Flux's private `double_block_forward`;
  use `--redux-image` for working image conditioning today.

## What's new in v0.15 — runtime LoRA + SD3 maturation

- **Per-task LoRA in scenarios**. `tasks: [{ loras: [...] }]` applies
  and clears LoRAs between tasks at runtime — no model reload.
  Composes with the scenario-level LoRA set. Flux (BF16 / GGUF / NF4).
- **NF4 + ControlNet**. NF4 Flux composes with `--control-spec` via
  the residual-aware forward — same residual interleave the BF16 and
  GGUF backbones use, so a single CN checkpoint works on all three.
- **SD3 / SD3.5 img2img + inpaint**. RePaint-style inpaint with
  per-step mask blend, rectified-flow truncated schedule. Works
  across the lineup (Medium / Large / Turbo).
- **SD3 / SD3.5 LoRA**. Diffusers PEFT format, MMDiT-targeted keys.
- **Flux Canny-dev / Depth-dev variants**. BFL "concept" Flux
  checkpoints with conditioning baked into the 128-channel `img_in`.
  Pass `--concept-image PATH` with `--model flux-canny-dev`.
- **Tiled SD3**. MultiDiffusion-style tiled denoise for MMDiT —
  1024-px tiles work on every SD3 variant within the variant's
  `pos_embed_max_size` cap.
- **Scenario ↔ generate sync**. Per-task `fast`, `concept-image`,
  `enhance`, `tiled` overrides.
- **Two new tutorials**:
  [`FLUX_TUTORIAL.md`](Documentation/Tutorials/FLUX_TUTORIAL.md)
  walks through the Flux feature set end-to-end;
  [`SD3_TUTORIAL.md`](Documentation/Tutorials/SD3_TUTORIAL.md) does
  the same for the SD3 / SD3.5 family.

## What's new in v0.14 — the SD3.5 + NF4 + Redux release

- **Stable Diffusion 3 / 3.5 (MMDiT)**. New family — `sd35-medium`,
  `sd35-large`, `sd35-large-turbo`, `sd3-medium`. Triple text encoder
  (CLIP-L + CLIP-G + T5-XXL), 16-channel VAE, rectified-flow sampler
  with SD3 time-shift. CFG via `[neg, pos]` double-batch.
- **NF4 quantized Flux**. `--model flux-dev-nf4` loads lllyasviel's
  bitsandbytes NF4 pack — ~6 GB transformer at inference (4× weight
  savings vs BF16), pure-CPU dequant codec means it runs on any
  candle device. Phase 8b adds **NF4 + LoRA composition** via the
  same selective-dequant trick GGUF uses.
- **Flux Redux**. `--redux-image PATH` adds image conditioning via
  SigLIP-so400m + BFL's Redux adapter (729 tokens → seq-concat onto
  T5). Repeatable for multi-image stacks (`--redux-image
  style.png:weight=0.8 --redux-image subject.png:weight=0.5`). Cap
  of 4 with attention-cost guardrails. Composes with GGUF, NF4,
  LoRA, ControlNet, img2img, tiled.
- **Tiled SD 1.5 / 2.1**. `--tiled` now supported on the smaller
  SD backbones too (was SDXL-only in v0.12).
- **Flux Fill + ControlNet**. `plakat img2img --model flux-fill-dev
  --mask ... --control-spec depth:from=...` composes with the
  auto-annotator and multi-CN.
- **Hyper-FLUX / FLUX-Turbo presets**. `--fast hyper-8 | hyper-16 |
  turbo-alpha` bundles the matching distillation LoRA + recommended
  step count + guidance in one flag.
- **Shared SdCore**. Scenarios with mixed t2i + img2img tasks now
  load the SD backbone **once** (was: per-task). The t2i Pipeline's
  `Arc<SdCore>` is reused by img2img via the existing `from_core`
  path.

## What's new in v0.13 — the Flux modernization release

- **Quantized Flux (GGUF)**. Run FLUX.1-dev on 16 GB GPUs.
  `--model flux-dev-gguf` loads the 4-bit transformer (~7 GB vs ~24 GB BF16).
  `--quantize-t5` drops T5-XXL to ~3 GB. `--quant-level Q5_K_M` picks a
  different precision (Q2_K..F16 supported); same for `--t5-quant-level`.
- **Flux LoRA on quantized**. Diffusers PEFT and AI-Toolkit / kohya
  formats both compose with the GGUF backbone — affected Linears are
  dequantized once at load, rest of the model stays 4-bit.
- **Flux Inpainting**. `--model flux-fill-dev` + `--mask` runs BFL's
  dedicated 384-channel inpaint checkpoint via `plakat img2img`.
- **Flux Img2Img**. Rectified-flow init: `plakat img2img init.png
  --model flux-dev --strength 0.7 --prompt "..."`.
- **Tiled Flux denoise**. MultiDiffusion-style 2K–4K outputs on any
  Flux variant: `--tiled --tile-size 1024 --tile-stride 768`. Composes
  with ControlNet (per-tile residuals) and the tiled VAE decode.
- **Flux ControlNet polish**. Auto-annotators wire through to Flux
  (`--control-spec depth:from=photo.jpg` is now a one-liner). Step gating
  via `start=…:end=…`. Multi-Flux-CN with summed residuals.
- **Outpainting**. New `plakat outpaint` subcommand expands a canvas
  and hands off to the inpaint pipeline (SDXL-Inpaint, SD15-Inpaint, or
  Flux.1-Fill-dev).
- **Scenarios**. Every v0.13 feature above is now expressible in
  scenario HJSON: `quant-level:`, `t5-quant-level:`, `tiled:`,
  per-task `init-image:` / `mask:` / `strength:` / `outpaint:`, plus
  multi-CN via `controls: [...]`.

