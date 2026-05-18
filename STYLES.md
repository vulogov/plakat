# Art-style detection and transfer

`plakat` can look at a reference photo, identify its art style, and load
the matching LoRAs to render your prompt in that style — all through the
standard generation pipeline. No IP-Adapter required.

This is different from `plakat stylize --ref` (which transfers the
specific visual texture of one image via IP-Adapter image conditioning).
Style detection resolves to a *named, reusable* style → reproducible
across runs, works on every base model with catalog coverage, and
composes cleanly with the LoRA stack.

## Implementation status

| Surface | Status |
|---|---|
| Detection module (`src/style/`) | **shipped** |
| `plakat style detect <PHOTO>` | **shipped** |
| `plakat style list` | **shipped** |
| `plakat style show <ID>` | **shipped** |
| Catalog → LoRA resolution (`StyleCatalog::resolve`) | **shipped** |
| `--style-ref` / `--style` / `--style-strength` / `--style-catalog` on `plakat generate` | **shipped** |
| `--style-ref` / `--style` / `--style-strength` / `--style-catalog` on `plakat portrait` | **shipped** |
| `style-ref` / `style` / `style-strength` / `style-catalog` in scenarios (global) | **shipped** |
| Bundled catalog (`watercolor` with real LoRA, `photorealistic` trigger-only) | **shipped** |
| End-to-end style transfer through standard SD 1.5 pipeline | **shipped** |
| Revision-SHA threading from catalog → HF download | catalog records the SHA, downloader uses `main` (see Limits) |
| Per-task style-ref in scenarios | not yet implemented (global only) |
| `plakat style probe` | **shipped** |
| `style-ref` field in scenarios | designed, not implemented |
| Catalog-build tool (`tools/build-style-catalog/`) | spike uses `examples/spike_catalog.rs` |
| Expanded catalog (10 seed styles + LoRA pins) | not yet curated |

Sections below covering not-yet-implemented surfaces describe the
designed behavior. They are kept here so the documented end-state is
visible while the rest is being built; each carries an inline
**Status:** note.

## Quick start

```bash
plakat style detect ./inspiration.jpg
```

Output:

```
Detected: watercolor (0.5037) [picked]

Top 2:
  1. watercolor           0.5037  ✓ picked
  2. photorealistic       0.0977
```

What works today: detection prints what style your reference photo looks
most like. What's coming: passing `--style-ref` to `plakat generate`
will resolve the detected style to a LoRA stack and run generation in
that style.

## How it works

1. **Encode** — the reference photo is resized to 224², CLIP-normalized,
   and passed through the CLIP-H/14 vision encoder shipped with
   IP-Adapter (`h94/IP-Adapter`). The output is a 1024-dimensional
   pooled embedding.
2. **Normalize** — L2-normalize so that cosine similarity reduces to
   plain dot product.
3. **Match** — compute cosine similarity against every exemplar
   embedding in the bundled catalog. Per-style scores are aggregated
   across the style's exemplars (default `top3-mean`).
4. **Decide** — the top-scoring style is picked if it clears
   `min_confidence` (default 0.22). If the top-1 doesn't beat top-2 by
   `margin_over_runner_up` (default 0.02), the result is flagged
   ambiguous.
5. **Resolve** — *(not yet implemented)* the matched style is looked up
   in the catalog, which maps it to one or more LoRAs for your active
   base model (SD 1.5, SDXL, or Flux) plus a trigger phrase.
6. **Generate** — *(not yet implemented)* the LoRAs are downloaded
   (cached after first use) and merged into the UNet. The trigger
   phrase is prepended to your prompt. Generation proceeds through the
   standard pipeline.

## The catalog

The catalog is a directory containing two files:

```
assets/style_catalog/
├── catalog.json           # style routing metadata
└── exemplars.safetensors  # CLIP-H pooled embeddings of exemplar images
```

`catalog.json` declares the schema version, the encoder used to produce
exemplars, detection policy thresholds, and per-style routing
(exemplar keys, per-base-model LoRA refs, trigger phrases).
`exemplars.safetensors` holds the actual embeddings keyed
`<style_id>/<idx>`, L2-normalized, f16 for compactness.

### catalog.json schema

```jsonc
{
  "schema_version": 1,

  "encoder": {
    "id": "clip-h-laion2b",           // must match runtime CLIP encoder
    "embed_dim": 1024,
    "exemplars_file": "exemplars.safetensors",
    "preprocess": "clip-standard-224"
  },

  "detection": {
    "aggregation": "top3-mean",       // or "max" | "mean"
    "min_confidence": 0.22,           // cosine threshold for auto-pick
    "margin_over_runner_up": 0.02     // top1 - top2 needed to be unambiguous
  },

  "styles": [
    {
      "id": "watercolor",             // slug; used by --style <id>
      "display_name": "Watercolor",
      "description": "Wet-on-wet pigment washes, ink lineart, paper texture.",

      // Tensor keys in exemplars.safetensors. Typically 3-5 per style.
      "exemplar_keys": ["watercolor/00", "watercolor/01", "watercolor/02", "watercolor/03"],

      // Per-base-model routing. Missing slot means: style is detectable
      // but not transferable onto that base. Status: schema is read by
      // the loader today; the resolve API that consumes it is not yet
      // implemented.
      "models": {
        "sd15": {
          "loras": [
            {
              "spec": "Arczisan/ink-watercolor:0.8",
              "revision": "main",
              "license": "creativeml-openrail-m",
              "license_url": "https://huggingface.co/Arczisan/ink-watercolor"
            }
          ],
          "trigger": "watercolor painting, soft washes, ink lineart",
          "negative_extras": "3d render, photo, glossy"
        },
        "sdxl": {
          "loras": ["ostris/watercolor-style-lora-sdxl:0.75"],
          "trigger": "watercolor, painterly",
          "negative_extras": "photo, glossy"
        }
      }
    }
  ]
}
```

#### Field reference

**Top-level**
- `schema_version` — integer. Bumps on breaking JSON-schema changes.
  Loader refuses unknown majors.
- `encoder.id` — fingerprint of the encoder that produced the
  exemplars. The runtime asserts this matches the encoder it's about to
  use. Cosines across mismatched embedding spaces are meaningless.
- `encoder.embed_dim` — must equal the encoder's pooled-output
  dimension. CLIP-H/14 with the IP-Adapter visual projection: 1024.
- `encoder.exemplars_file` — relative path; resolved against the JSON's
  directory.
- `detection.aggregation` — how to collapse per-exemplar similarities
  into one per-style score: `"max"` | `"mean"` | `"top3-mean"`. Default
  `"top3-mean"` is robust to one weak exemplar without averaging across
  all.
- `detection.min_confidence` — below this cosine, detection returns
  "no pick" (caller falls back to no style or to `--style <id>`).
- `detection.margin_over_runner_up` — top-1 must beat top-2 by at least
  this much; otherwise the result is flagged ambiguous and both are
  surfaced.

**Per style**
- `id` — slug; the value `--style <id>` will accept (when that flag
  ships). Also used as the prefix for exemplar tensor keys.
- `display_name` — for `plakat style detect` / `plakat style list`
  output and human-facing docs.
- `description` — one-line summary.
- `exemplar_keys` — list of keys in `exemplars.safetensors`. 1-N per
  style; more exemplars usually buys robustness for visually
  heterogeneous styles.
- `models` — map keyed by base-model family: `sd15` | `sdxl` | `flux`.
  Missing slot means: this style can't be transferred onto that base;
  detection still finds it but the transfer step will report "no LoRA
  available for `<base>`".

**Per `models.<base>` slot**
- `loras` — array of LoRA entries. Each entry is either a shorthand
  string like `"org/repo:0.8"` (matching plakat's existing
  `LoraSpec::from_str` grammar) **or** an object:
  ```json
  {
    "spec": "org/repo:0.8",
    "revision": "sha-or-tag-or-main",
    "license": "creativeml-openrail-m",
    "license_url": "https://..."
  }
  ```
  Empty/missing is allowed (trigger-only styles).
- `trigger` — phrase prepended to your prompt at generation time so the
  LoRA's training tokens are present. Per-base because SD 1.5 and SDXL
  LoRAs often have different trigger words.
- `negative_extras` — appended to the user's negative prompt.

### Where the catalog lives

The bundled catalog is at `assets/style_catalog/` relative to the
plakat source tree (also the working directory during `cargo run`).
Override with `--catalog <DIR>`:

```bash
plakat style detect ./inspiration.jpg --catalog ./my_catalog/
```

For installed binaries, the deployment story (system data dir vs.
user-config dir vs. auto-download) is not yet decided. Use the explicit
`--catalog` flag for now.

## CLI

### `plakat style detect <PHOTO>` — shipped

Detect art style from a photo. Prints top-K matches, doesn't generate.

```bash
plakat style detect <PHOTO> [--top-k N] [--format text|json] [--catalog DIR]
```

| Flag | Default | Description |
|---|---|---|
| `<PHOTO>` | required | Reference photo to detect from. |
| `--top-k` | `5` | Number of ranked matches to display. |
| `--format` | `text` | Output format: `text` (human) or `json` (scripting). |
| `--catalog` | bundled | Override the catalog directory. |

#### Text output

```
$ plakat style detect ./inspiration.jpg
Detected: watercolor (0.5037) [picked]

Top 5:
  1. watercolor           0.5037  ✓ picked
  2. photorealistic       0.0977
```

States the output can take:
- `[picked]` — top score cleared `min_confidence` AND beat runner-up by
  `margin_over_runner_up`. Confident detection.
- `[ambiguous]` — top score cleared `min_confidence` but runner-up was
  within margin. Runner-up is also shown explicitly; consider re-running
  with each candidate via `--style <id>` (once that flag ships).
- `(none above min_confidence)` — no style matched well enough. The
  closest is still shown for context; either expand the catalog, pick a
  style by name, or lower the threshold in the catalog JSON.

#### JSON output

```json
{
  "ambiguous": false,
  "picked": "watercolor",
  "top": [
    { "style_id": "watercolor",    "display_name": "Watercolor",    "score": 0.5037 },
    { "style_id": "photorealistic", "display_name": "Photorealistic", "score": 0.0977 }
  ]
}
```

`picked` is `null` when nothing cleared the confidence threshold.
`ambiguous` is `true` when the top two scores are within margin.
Suitable for `jq`-style scripting or CI integration.

### `plakat style list` — shipped

Lists every style in the catalog with a one-line description.

```bash
plakat style list [--base sd15|sdxl|flux] [--format text|json] [--catalog DIR]
```

| Flag | Default | Description |
|---|---|---|
| `--base` | (all) | Filter to styles with LoRA mappings for the specified base model. Styles with empty `models` are hidden when this flag is set. |
| `--format` | `text` | `text` for a human-readable table, `json` for scripting. |
| `--catalog` | bundled | Override the catalog directory. |

#### Text output

```
ID              Display name     Ex  Bases       Description
──────────────  ──────────────  ───  ──────────  ────────────────────
photorealistic  Photorealistic    4  (none)      Photographic realism; lens characteristics; lighting physicality.
watercolor      Watercolor        4  (none)      Wet-on-wet pigment washes, ink lineart, visible paper texture.

2 styles.
```

The `Ex` column shows exemplar count; `Bases` lists every base-model
slot with at least one LoRA configured. `(none)` means the style is
detection-only.

### `plakat style show <ID>` — shipped

Full info for one style: description, exemplar count, per-base LoRA
specs (with revision pins), trigger phrases, negative-prompt
additions.

```bash
plakat style show <ID> [--format text|json] [--catalog DIR]
```

#### Text output (detection-only style)

```
ID:              watercolor
Display name:    Watercolor
Description:     Wet-on-wet pigment washes, ink lineart, visible paper texture.
Exemplars:      4 in catalog

Note: this style is detection-only — no LoRAs / triggers configured.
```

#### Text output (style with LoRAs configured)

```
ID:              watercolor
Display name:    Watercolor
Description:     Wet-on-wet pigment washes, ink lineart, visible paper texture.
Exemplars:      4 in catalog

Models:
  sd15:
    loras:
      - Arczisan/ink-watercolor:0.8 (revision: main)
    trigger:   "watercolor painting, soft washes, ink lineart"
    negative+: "3d render, photo, glossy"
  sdxl:
    loras:
      - ostris/watercolor-style-lora-sdxl:0.75
    trigger:   "watercolor, painterly"
```

### `plakat style probe` — shipped

Confirms every LoRA in the catalog still resolves on HuggingFace.
HEAD-requests `https://huggingface.co/<repo>/resolve/<revision>/<file>`
for each LoRA and reports the HTTP status. Network-dependent.

```bash
plakat style probe [--id <ID>] [--format text|json] [--timeout SECS] [--catalog DIR]
```

| Flag | Default | Description |
|---|---|---|
| `--id` | (all styles) | Probe only the LoRAs for one style id. |
| `--format` | `text` | `text` or `json`. JSON is suitable for CI consumption. |
| `--timeout` | `10` | Per-request network timeout, seconds. |
| `--catalog` | bundled | Override the catalog directory. |

The command exits with status `0` when every LoRA resolves, `1` when
any fails. Suitable as a periodic CI job to catch upstream repo
deletions or file renames before users hit them.

#### Text output

```
$ plakat style probe
Probing 1 style(s), 1 LoRA(s) total…

  ✓ Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8 (sd15 @ cd8b7d93)

✓ all 1 LoRA(s) resolved
```

The `(sd15 @ cd8b7d93)` suffix shows the base-model slot and the
8-character revision SHA prefix the catalog pinned for that LoRA. When
no explicit revision is set, only the base slot is shown.

#### Mixed-result example

```
$ plakat style probe --catalog ./test_catalog
Probing 1 style(s), 3 LoRA(s) total…

  ✓ Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8 (sd15)
  ✗ ghost-user/missing-repo#fake.safetensors:0.5 (sd15) HTTP 401
  ✗ Arczisan/ink-watercolor#missing-file.safetensors:0.5 (sd15) HTTP 404

✗ 2 / 3 LoRA(s) failed to resolve
```

(HuggingFace returns `401 Unauthorized` for repos that don't exist —
the same response gated/private repos give. `404` means the repo
exists but the file path inside it doesn't.)

#### JSON output

```json
{
  "probed": 3,
  "failures": 2,
  "results": [
    {
      "style_id": "watercolor",
      "base": "sd15",
      "spec": "Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8",
      "revision": "cd8b7d93ec0b6c0aa31a640f1287837583d702d0",
      "status": "ok",
      "detail": {
        "url": "https://huggingface.co/Arczisan/ink-watercolor/resolve/cd8b7d93.../inkwatercolor.safetensors"
      }
    },
    {
      "style_id": "watercolor",
      "base": "sd15",
      "spec": "ghost-user/missing-repo#fake.safetensors:0.5",
      "revision": null,
      "status": "not_found",
      "detail": {
        "url": "...",
        "http_status": 401
      }
    }
  ]
}
```

`status` values: `ok` | `local_ok` | `not_found` | `local_missing` |
`network_error` | `bad_spec`. The first two pass; the others fail.

### What `probe` does not check

- **Auto-discovery specs.** A LoRA spec without an explicit `#file`
  (e.g. `org/repo:0.8`) only verifies that the repo exists via the
  models API. The actual `.safetensors` filename is discovered at
  download time. If the repo exists but no `.safetensors` files do,
  probe passes but a real generation would fail. The fix is to use
  explicit `#file` specs in catalog entries.
- **License compatibility.** Probe confirms availability, not whether
  you're allowed to use the LoRA for your purpose. See the catalog's
  recorded license + url via `plakat style show <ID>`.
- **Weight integrity.** A HEAD request returns 200 even for a
  zero-byte safetensors. Actual content validation happens at
  generation time when plakat tries to load the file.

## Integration with `plakat generate`

### Flags

| Flag | Default | Description |
|---|---|---|
| `--style-ref <PATH>` | — | Reference photo to detect style from. Detected style's LoRAs are loaded and trigger is prepended to the prompt. |
| `--style <ID>` | — | Bypass detection; pick a style by id. Can be combined with `--style-ref` (overrides the detection result) or used alone. |
| `--style-strength <F>` | `1.0` | Multiplier applied to each catalog LoRA's `:scale`. `<1.0` for subtler style, `>1.0` for stronger. Above ~1.8 most LoRAs degrade the prompt. |
| `--style-catalog <DIR>` | bundled | Override the catalog directory. |

### Quick start

```bash
plakat generate \
  "a fox in a forest clearing" \
  --style-ref ./inspiration.jpg
```

Plakat encodes the reference photo, finds the closest catalog style,
resolves it to the LoRAs configured for the active base model
(detected from `--model`), prepends the trigger phrase to your prompt,
and runs generation through the standard pipeline.

#### Sample log

```
  → style: watercolor
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors → UNet: 192/192 targets merged (scale 0.80)
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors → text encoder: 72/72 targets merged (scale 0.80)
→ ./out/plakat-690272596.png
```

The catalog-resolved LoRA is downloaded once (cached after first use),
merged into both the UNet and the text encoder at the catalog's
configured scale, and the style trigger phrase is prepended to your
prompt. Generation proceeds through plakat's standard SD 1.5 pipeline
exactly as it would with a manually-specified `--lora`.

### Pick a style by id

```bash
plakat generate "a fox" --style watercolor
```

Skips detection entirely. Useful when you already know which style you
want, or when detection on a previous photo gave the wrong result.

When combined with `--style-ref`, the named style wins; the photo is
still encoded so the detection result can be shown for context, but
the named style is what gets resolved.

### Behavior when `--style-ref` is combined with `--lora`

When you pass your own `--lora` specs alongside `--style-ref`, plakat
emits a warning and uses catalog LoRAs only:

```
⚠ --style-ref overrides 2 user-specified LoRA(s); using catalog LoRAs only
```

To stack user LoRAs *on top of* a named style, use `--style <id>` —
that path is designed to compose with user LoRAs but is not yet
implemented; today both `--style` and `--style-ref` replace user LoRAs.

### Behavior when the detected style has no LoRA for the active base

Three cases:

- **Style has no `models` section at all.** Treated as
  "detection-only" — generation proceeds with no LoRAs and no trigger
  injection, with a one-line warning. This is what happens with the
  spike catalog today (no LoRAs configured yet).
- **Style has `models` for other bases but not the active one.**
  Hard error with an actionable message listing which bases the style
  does support:
  ```
  Error: style 'watercolor' has no LoRA mapping for sdxl;
  supported bases for this style: [sd15]
  ```
  Switch model or pick a different style.
- **Style has `models` for the active base.** Normal flow — LoRAs
  download and merge into the UNet, trigger is prepended.

## Integration with `plakat portrait`

**Status:** shipped. Same four flags, same precedence with `--lora`,
same warning behaviors as `plakat generate`.

`plakat portrait` keeps two reference photos separate by design:

- `--photo` → **identity** reference (who the portrait depicts).
  Used by FaceID / Plus-Face / similar identity adapters. See PERSONA.md.
- `--style-ref` → **style** reference (how the portrait is rendered).
  Used by the catalog → LoRA path described in this document.

Both can be set on the same invocation; they don't conflict.

### Quick start

```bash
# Style-only portrait (no identity photo)
plakat portrait \
  "a woman with red hair smiling" \
  --style-ref ./inspiration.jpg

# Identity + style (FaceID for the face, watercolor LoRA for the style)
plakat portrait \
  "a portrait wearing a knit sweater" \
  --photo ./me.jpg --identity faceid \
  --style-ref ./inspiration.jpg
```

The style flow runs before the identity adapter loads, so the trigger
phrase is prepended to your prompt and the catalog LoRAs are merged
into the UNet before any FaceID / Plus-Face conditioning kicks in.

#### Sample log

```
  → style: watercolor
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors → UNet: 192/192 targets merged (scale 0.80)
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors → text encoder: 72/72 targets merged (scale 0.80)
→ ./out/plakat-portrait-3943102168.png
```

### Interaction with portrait's default negative prompt

`plakat portrait` ships with a face-and-anatomy negative prompt
baseline (deformed face, asymmetric eyes, etc.) applied unless you
pass `--negative ""` to disable it. When a style is applied, the
catalog's `negative_extras` is appended to whatever negative was
resolved — so the watercolor style's "3d render, photo, glossy"
joins the portrait baseline rather than replacing it. The two are
complementary: portrait negatives fix anatomy, style negatives push
away from competing rendering styles.

## Integration with scenarios

**Status:** shipped at the global level (whole scenario inherits a
single style). Per-task `style-ref` overrides are not yet supported.

The scenario HJSON accepts four new top-level fields:

```hjson
{
    # Apply style detection globally to every task in this scenario.
    # The reference photo is encoded once at scenario load time, the
    # detected style is resolved against the active base model, and
    # the LoRAs + trigger are wired into the prompt-assembly path for
    # all tasks.
    style-ref: ./inspiration.jpg

    # OR pick a style by name (overrides style-ref if both are set).
    style: watercolor

    # Optional multiplier on catalog LoRA scales. Defaults to 1.0.
    style-strength: 1.0

    # Optional override of the bundled catalog directory.
    style-catalog: ./my_catalog/

    # ... existing scenario fields ...
}
```

### Sample log

```
$ DEEPSEEK_API_KEY=... plakat scenario ./my_scenario.hjson
  → style: watercolor
  ⚠ --style-ref overrides 1 user-specified LoRA(s); using catalog LoRAs only
scenario  3 task(s) × 2 image(s) = 6 image(s) to generate
  model:     sd15
  loras:     1 (scale 1)
...
```

The style flow runs once at scenario load — every task inherits the
detected style's LoRAs, trigger phrase, and negative additions.

### Interaction with the existing `loras: [...]` field

Same precedence as the CLI: if both `style-ref`/`style` AND `loras` are
set on the scenario, plakat warns once at load time and uses catalog
LoRAs only. The `lora-scale` global multiplier still applies to the
catalog-resolved LoRAs, composed multiplicatively with `style-strength`
(both default to 1.0).

### Interaction with `lora-header` / `lora-footer`

The catalog trigger is prepended to the scenario's `lora-header`. The
existing prompt assembly:

```
final_prompt = lora-header + ENHANCED(prompt-header + scene + weather + task + prompt-footer) + lora-footer
```

becomes (when style is applied):

```
final_prompt = (STYLE_TRIGGER + lora-header) + ENHANCED(...) + lora-footer
```

A simple exact-substring dedup prevents duplicating the trigger if it's
already present in `lora-header`. Paraphrases (`"watercolor painting"`
vs `"painted in watercolor"`) are not dedup'd — if you've semantically
duplicated the trigger in your `lora-header`, the catalog still appends
its own; usually harmless but can over-weight the style.

### Interaction with the per-task `negative` override

Tasks can override the global negative via their own `negative:` field.
That per-task negative is taken as-authored and does **not** receive the
style's `negative_extras`. The style's negative additions are merged
into the *global* negative only.

If you want a task-specific negative AND style negatives, you have two
options:

- Don't set a per-task `negative:` — the global (which already includes
  the style's `negative_extras`) applies.
- Set a per-task `negative:` that literally includes the style's
  `negative_extras` text. Acceptable for one-off task tuning.

A future iteration may auto-merge the style negatives into per-task
overrides; today it's an explicit choice.

### Per-task style-ref — designed, not implemented

A later iteration may allow per-task `style-ref:` overriding the
global. Costs one CLIP-H encode per task that uses it; the encoder
loads once and is reused. Not in scope today — every task inherits
the scenario's single global style.

## Building a custom catalog

**Status:** today, only the spike builder exists at
`examples/spike_catalog.rs`. The full curation tool is designed but
not yet implemented.

### Spike builder

```bash
cargo run --release --example spike_catalog -- \
    --fixtures tests/fixtures/style_catalog \
    --out      assets/style_catalog \
    --device   cpu
```

Layout it expects:

```
<fixtures>/
├── <style_id_1>/    # any image files (.jpg/.jpeg/.png); each becomes an exemplar
├── <style_id_2>/
└── holdout/         # skipped by the builder; used by the smoke test
```

The spike builder hardcodes routing metadata for the two spike styles
(`watercolor`, `photorealistic`). Extend it by editing
`spike_style_metadata()` in `examples/spike_catalog.rs`, or wait for
the full catalog-build tool.

### Designed full builder

The post-spike tool will read a curator-authored YAML and an
exemplar-images directory, and emit `catalog.json` +
`exemplars.safetensors` plus sidecar `LICENSES.md` and `provenance.json`
files. The YAML lives at `tools/style_sources/catalog.yaml` and looks
like:

```yaml
schema_version: 1
encoder:
  id: clip-h-laion2b
  embed_dim: 1024
detection:
  aggregation: top3-mean
  min_confidence: 0.22
  margin_over_runner_up: 0.02
styles:
  - id: watercolor
    display_name: Watercolor
    description: Wet-on-wet pigment washes, ink lineart, paper texture.
    exemplars:
      - watercolor/01.jpg
      - watercolor/02.jpg
      - watercolor/03.jpg
      - watercolor/04.jpg
    models:
      sd15:
        loras:
          - spec: Arczisan/ink-watercolor:0.8
            revision: main
            license: creativeml-openrail-m
            license_url: https://huggingface.co/Arczisan/ink-watercolor
        trigger: "watercolor painting, soft washes, ink lineart"
        negative_extras: "3d render, photo, glossy"
      sdxl:
        loras:
          - ostris/watercolor-style-lora-sdxl:0.75
        trigger: "watercolor, painterly"
```

The builder validates that every referenced exemplar image loads, every
`spec` parses through `LoraSpec::from_str`, and (with `--probe-hf`) that
every `repo+revision` resolves on HuggingFace. Sidecars track licenses
and per-exemplar SHA-256 hashes for reproducible rebuilds.

## Tuning

### When detection picks the wrong style

- **Confident-but-wrong** — the catalog's exemplars for that style
  aren't representative of your reference photo. Pick by name once
  `--style <id>` ships; today, the workaround is to expand the catalog
  or use a different reference photo.
- **Ambiguous** — `plakat style detect` will surface both candidates.
  Re-run generation with each (once integrated).
- **No pick (below threshold)** — your reference doesn't resemble any
  catalog style closely enough. Either pick by name, expand the catalog,
  or lower `min_confidence` in `catalog.json`. Lowering the threshold
  is appropriate when you have few but well-separated styles; raise it
  when the catalog is dense and noisy.

### Aggregation policy

- `top3-mean` (default) — robust to one anomalous exemplar without
  averaging across all. Best general-purpose choice.
- `max` — single best exemplar wins. Use when you expect tight
  intra-style clustering and want sensitivity to even one strong match.
  Vulnerable to a single noisy exemplar inflating a style's score.
- `mean` — average across all exemplars. Use when you've curated a
  large set of consistent exemplars and want to penalize stylistic
  outliers within the set.

### `min_confidence` threshold

The spike's `0.22` default works for the seed styles. Re-tune when:

- You expand the catalog and see false-positive picks → raise it.
- You see correct picks getting rejected → lower it.

CLIP-H pooled cosines on style-discrimination tasks typically land in
the `[0.15, 0.55]` range; the spike's observed values (`0.10` for
out-of-style, `0.50+` for in-style) span most of that range, giving
plenty of headroom.

### `margin_over_runner_up`

Default `0.02` is suitable when scores are typically separated by
`0.1+`. Raise it (e.g. to `0.05`) when adjacent styles in your catalog
have genuinely overlapping CLIP signatures (e.g. art nouveau vs art
deco) — you want the user to see the ambiguity rather than commit to a
coin-flip pick.

## First-run downloads

| Asset | Size | Source |
|---|---|---|
| CLIP-H image encoder | ~2.5 GB | `h94/IP-Adapter` — shared with FaceID / Plus-Face / scenario stylize |

The CLIP-H download is a one-time cost shared across plakat features.
If you've already used `plakat portrait --identity faceid`, `plakat
stylize`, or any IP-Adapter-based path, CLIP-H is already cached and
`plakat style detect` reuses it.

LoRA downloads (per detected style) are deferred until the
`generate`/`portrait` integration ships.

## Limits

- **CLIP-H is coarse on fine-grained styles.** Watercolor vs anime is
  easy. Art nouveau vs art deco is harder. Use `plakat style detect`
  to confirm the pick before committing to a long generation.
- **Style coverage = catalog coverage.** A style not in the catalog
  cannot be detected. The shipping catalog covers 2 styles today
  (`watercolor` with a real SD 1.5 LoRA, `photorealistic` trigger-only);
  will grow to ~10 in the MVP curation pass.
- **Revision pinning is recorded but not yet enforced at download.**
  The catalog stores a revision SHA per LoRA and `plakat style show`
  displays it, but the underlying HF downloader currently uses `main`.
  In practice this means catalog LoRA references aren't perfectly
  reproducible against upstream force-pushes. A future fix threads
  `ResolvedLoraRef.revision` through the download call. Until then,
  prefer LoRAs from authors known not to rewrite history.
- **LoRA availability isn't guaranteed.** Catalog LoRAs point at HF
  repos that can occasionally disappear or be renamed. `plakat style
  probe` (when shipped) will catch this early.
- **Plakat doesn't redistribute LoRA weights.** The catalog stores
  references; users download on demand. Each LoRA's license is the
  user's responsibility — `plakat style show <ID>` displays the
  catalog's recorded license + url so you can review before generating.
  Some LoRAs are bundled with bespoke license terms (non-commercial,
  attribution-required, etc.) that the catalog cannot enforce for you.
- **Cross-base support varies.** Some styles will only have a viable
  LoRA for SD 1.5 (or only for SDXL). `plakat style list --base sdxl`
  filters to SDXL-supported styles.
- **`--style-ref` + `--lora` don't combine.** When both are set,
  catalog wins and user LoRAs are dropped with a warning. To stack
  user LoRAs on top of a style, use `--style <id>` (without
  `--style-ref`).
- **Trigger-phrase dedup is exact-substring.** If you've paraphrased
  the trigger into your prompt or `lora-header`, the catalog appends
  its own — usually fine, occasionally over-weights the style. Match
  prompts to triggers literally to avoid this.

## Troubleshooting

**`Error: style catalog at <path> uses schema_version=N, plakat supports 1`**
The catalog was authored against a newer schema than this plakat
supports. Either downgrade the catalog or upgrade plakat.

**`Error: style catalog was built with encoder 'X' but runtime is using 'Y'`**
The catalog's exemplar embeddings were produced by a different CLIP
encoder than the one plakat is about to use. Rebuild the catalog with
the current encoder.

**`Error: style 'X' references missing exemplar key 'X/00'`**
`catalog.json` references an exemplar key that's absent from
`exemplars.safetensors`. The two files are out of sync — rebuild.

**`Error: embedding dim N doesn't match catalog embed_dim M`**
A loaded exemplar has the wrong shape, or `encoder.embed_dim` in the
JSON doesn't match what was actually written. Rebuild the catalog.

**`Detected: (none above min_confidence)`**
Reference photo doesn't match any catalog style strongly. Either:
- Use `--style <id>` (once shipped) to force a specific style.
- Expand the catalog with more representative exemplars.
- Lower `min_confidence` in `catalog.json`.

**`Detected: <style> [ambiguous]`**
Top-1 is within `margin_over_runner_up` of top-2. Both candidates are
shown. Run generation against each (once `--style <id>` ships) and
pick the better result.

## Testing

The spike's smoke test lives at `tests/style_detect_smoke.rs`. It runs
three checks against the bundled catalog:

1. A held-out watercolor (`Sargent — Under the Willows`) picks the
   `watercolor` style with score above `min_confidence`.
2. A held-out photograph (`Apollo 17 — Blue Marble`) picks
   `photorealistic`.
3. Re-encoding a catalog exemplar scores higher than re-encoding a
   different photo of the same style (the "exemplar beats holdout"
   identity check — catches L2-norm and catalog-key bugs).

These tests download CLIP-H weights (~2.5 GB) on first run, so they're
marked `#[ignore]` and excluded from default `cargo test`. Run them
explicitly:

```bash
cargo test --test style_detect_smoke --release -- --ignored --nocapture
```

Expected output:

```
watercolor holdout scores:
        watercolor  0.5037
    photorealistic  0.0977
photo holdout scores:
    photorealistic  0.5530
        watercolor  0.2140
exemplar score: 0.5382 (watercolor)
holdout score:  0.5037 (watercolor)
test result: ok. 3 passed; 0 failed
```

The cosine numbers above are the spike's validation: CLIP-H pooled
embeddings carry plenty of style signal to discriminate the two seed
styles with a 0.3+ margin between right and wrong.

## See also

- `PERSONA.md` — identity preservation (FaceID, IP-Adapter-Plus-Face).
  Distinct from style detection: identity is about *who*, style is
  about *how*.
- `GENERATE.md` — the standard text-to-image pipeline that style
  detection will plug into.
- `README.md` — top-level plakat documentation.
