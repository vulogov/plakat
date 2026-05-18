# ZONES — artefact compositing into named zones

End-to-end runnable tutorial for plakat's artefact compositing
feature. Every script in `scripts/` is independent and self-
contained. Run them in order for a walkthrough, or jump straight to
whichever one matches your use case.

This example demonstrates all three tiers of the feature:

| Tier | Flag | What it does |
|---|---|---|
| **v1** | (default) | Alpha-composite PNG cutouts into rigid 4×3 zone grid. |
| **v2** | `--artefact-blend` | Masked low-strength img2img pass after the composite to soften edges. |
| **v3** | `--smart-zones` | Derive zones from each generated image's depth + luminance instead of the rigid grid. |

The tiers stack: you can enable v2 + v3 together (recommended for
production).

## What's in here

```
ZONES/
├── README.md                  ← this file
├── library/                   ← self-contained artefact library
│   ├── library.json
│   ├── sky/      (sun.png, moon.png, cloud.png)
│   ├── trees/    (oak.png, pine.png)
│   └── houses/   (cottage.png)
├── scenario.hjson             ← batch demo (4 tasks)
└── scripts/
    ├── 01_basic.sh            ← one artefact, library default zone
    ├── 02_zones.sh            ← multiple artefacts, explicit zones
    ├── 03_scale.sh            ← zone + scale overrides
    ├── 04_blend.sh            ← v2: --artefact-blend
    ├── 05_smart_zones.sh      ← v3: --smart-zones
    ├── 06_full_stack.sh       ← v1 + v2 + v3 together
    └── 07_scenario.sh         ← batch run of scenario.hjson
```

Outputs land under `out/zones-tutorial/<script-name>/`.

## Prerequisites

- A `plakat` binary, found in any of:
  1. `plakat` on `$PATH` (installed via `cargo install plakat`).
  2. `target/release/plakat` (built via `cargo build --release` from
     the repo root).
  3. `target/debug/plakat` (built via `cargo build`).

  The scripts source `scripts/_plakat.sh`, which probes these in
  order and exits with a clear message if none are found. GPU
  strongly recommended (Apple Silicon Metal or NVIDIA CUDA); CPU
  works but each image takes minutes.
- For script `05_smart_zones.sh` and `06_full_stack.sh`: a one-time
  ~99 MB download of the Depth-Anything-V2-small checkpoint. Cached
  by HuggingFace's hub library after the first run.
- For script `04_blend.sh` and `06_full_stack.sh`: SD 1.5 weights
  (also cached on first use).
- For script `07_scenario.sh`: an LLM API key in the environment.
  Scenarios run task prompts through an enhancer; set either
  `DEEPSEEK_API_KEY` or `GEMINI_API_KEY` before running, and match
  the corresponding `enhancer:` field in `scenario.hjson`.
- Scripts `01_basic.sh` through `06_full_stack.sh` need no API keys.

## Quick start

```bash
cd examples/tutorials/ZONES
./scripts/01_basic.sh
open out/zones-tutorial/01_basic/plakat-1001.png
```

If that produces an image with a stylized oak silhouette in the
middle band, everything is wired up correctly.

## Walkthrough

### 1. The artefact library

The `library/` subdirectory is a complete plakat artefact library
— `library.json` plus the PNG cutouts it references. Every script
passes `--artefact-library library` so plakat uses this local set
instead of the bundled placeholders.

The library ships with six CC0 silhouettes:

| Artefact | Natural zone | Anchor | Use |
|---|---|---|---|
| `sun` | sky | center | Daylight scenes |
| `moon` | sky | center | Night scenes |
| `cloud` | sky | center | Any sky |
| `oak` | middle_plan | bottom_center | Mid-distance tree |
| `pine` | middle_plan | bottom_center | Alternative tree |
| `cottage` | close_plan | bottom_center | Foreground building |

The silhouettes are deliberately simple — they prove the pipeline
works without making aesthetic commitments. For production, replace
them with your own PNGs and edit `library.json` accordingly.

### 2. Run `01_basic.sh` — one artefact, library default

```bash
./scripts/01_basic.sh
```

What it shows:

- `--artefact oak` with no `@ZONE` suffix → plakat uses `oak`'s
  library default (`middle_plan`).
- `--artefact-library library` → use the local library, not the
  bundled assets.
- Seed pinned to `1001` → deterministic output for comparison.

Expected output: meadow scene with an oak silhouette near the
middle-bottom of the canvas. The cutout will look obviously pasted
— that's v1 baseline. Later scripts soften this.

### 3. Run `02_zones.sh` — multiple artefacts, explicit zones

```bash
./scripts/02_zones.sh
```

Five artefacts arranged across the 4×3 grid using
`NAME@DEPTH/HORIZONTAL` syntax. Demonstrates:

- Multiple `--artefact` flags compose.
- Z-order = flag order (later flags render on top).
- The full 4×3 zone vocabulary: `sky` / `far_plan` /
  `middle_plan` / `close_plan` × `left` / `center` / `right`.

The layout reads as a complete scene rather than a single subject
on a backdrop:

```
   sky/left           sky/right
   ┌───────────────┐
   │ cloud   sun   │
   │               │
   │ pine          │  ← far_plan/left
   │       oak     │  ← middle_plan/right
   │   cottage     │  ← close_plan/center
   └───────────────┘
```

### 4. Run `03_scale.sh` — depth-of-field via scale

```bash
./scripts/03_scale.sh
```

Three oaks at 0.5×, 0.8×, and 1.2× scale across `far_plan`,
`middle_plan`, `close_plan`. Each `:SCALE` suffix multiplies the
library's default size, giving a cheap depth cue. None of the three
specifies an explicit offset, so auto-stagger doesn't trigger
(they're already in different zones).

### 5. Run `04_blend.sh` — v2 edge integration

```bash
./scripts/04_blend.sh
```

Same scene as `02_zones.sh` (same seed), plus
`--artefact-blend --artefact-blend-strength 0.30`. After the alpha
composite, plakat runs a short masked img2img pass over a feathered
union of the artefact zones. The pasted silhouettes are no longer
sharply outlined — they integrate with the surrounding lighting.

Compare side-by-side:

```bash
open out/zones-tutorial/02_zones/plakat-1002.png
open out/zones-tutorial/04_blend/plakat-1002.png
```

Cost: ~2–5 s extra per image on GPU.

**Strength dial.** 0.3 is the recommended default. Lower (0.15–0.20)
for edge feathering only; higher (0.40+) lets the model add texture
and may drift the artefact silhouette.

### 6. Run `05_smart_zones.sh` — v3 depth-aware zones

```bash
./scripts/05_smart_zones.sh
```

First run downloads Depth-Anything-V2-small (~99 MB). The script
uses a 16:9 panoramic prompt with a low horizon — exactly the kind
of scene where the rigid grid misplaces sky-zone artefacts. Smart
zones reads the actual painted scene, finds the depth quantiles,
and tracks the painted horizon.

To see the difference, run the script, then run it again with
`--smart-zones` removed (or `02_zones.sh` for the rigid-grid path).
The `sun@sky/right` placement should be visibly higher in the
rigid-grid version (forced to top 25 %) vs. lower / wider in the
smart version (matching the actual sky region).

Fallback: if the depth model can't be downloaded (offline, mirror
unreachable), plakat warns and silently uses the rigid grid. The
flag never blocks a generation.

### 7. Run `06_full_stack.sh` — all three tiers together

```bash
./scripts/06_full_stack.sh
```

Smart zones place each artefact relative to the painted scene; the
blend pass softens the edges. This is the recommended setup for
production use.

### 8. Run `07_scenario.sh` — batch with the HJSON form

```bash
./scripts/07_scenario.sh
```

Reads `scenario.hjson` and produces several outputs, one per task,
each in its own subdirectory under `out/zones-tutorial/scenario/`.
The scenario demonstrates:

- `bare_defaults` — minimal: artefacts referenced by name only.
- `layered_zones` — multi-artefact composition with explicit zones.
- `full_object_form` — the per-artefact `offset`, `anchor`, `flip`,
  `alpha` overrides only available via HJSON.
- `smart_zones_off` — per-task override disabling smart zones for a
  scene where the depth signal isn't useful.

Top-level scenario fields enable smart zones and blend across all
tasks; individual tasks override as needed.

## Modifications to try

**Swap your own artefact in.** Drop a PNG into `library/`, add an
entry to `library.json`, and reference it by name in any script.
The schema is documented in
[`Documentation/ARTEFACTS.md`](../../../Documentation/ARTEFACTS.md#the-artefact-library).

**Use a different model.** Add `--model sdxl` (or any HF repo id)
to any script. The artefact compositing is model-agnostic; only the
generation step differs.

**Override zone extents.** Edit `scenario.hjson` and add a `zones:`
field at the top level to change the rigid grid's default band
positions. Useful if you've disabled smart zones but still want to
adapt the grid to a wide aspect ratio:

```hjson
zones:
{
    sky:         [0.0, 0.40]
    far_plan:    [0.40, 0.55]
    middle_plan: [0.55, 0.80]
    close_plan:  [0.80, 1.0]
}
```

**Inspect the library.** `plakat artefact list --library library`
prints every entry. `plakat artefact show oak --library library`
dumps one in detail.

## Cleaning up

```bash
rm -rf out/zones-tutorial
```

The depth and SD model caches live under HuggingFace's hub cache
(`~/.cache/huggingface/`) and are intentionally kept across
invocations.

## Further reading

- [`Documentation/ARTEFACTS.md`](../../../Documentation/ARTEFACTS.md)
  — full reference: schema, every flag, every override.
- [`Documentation/Tutorials/ARTEFACTS_TUTORIAL.md`](../../../Documentation/Tutorials/ARTEFACTS_TUTORIAL.md)
  — narrative tutorial covering the same material at a slower pace.
- [`Documentation/GENERATE.md`](../../../Documentation/GENERATE.md)
  — for the broader `plakat generate` / scenario surface that
  artefacts plug into.

## License

Everything in this directory is CC0 / Unlicense — same as plakat
itself. The artefact PNGs are procedurally generated by plakat and
carry no third-party copyright.
