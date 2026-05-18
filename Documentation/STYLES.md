# Art-style detection and transfer

`plakat` can look at a reference photo, identify its art style, and load
the matching LoRAs to render your prompt in that style — all through the
standard generation pipeline. No IP-Adapter required.

This is different from `plakat stylize --ref` (which transfers the
specific visual texture of one image via IP-Adapter image conditioning).
Style detection resolves to a *named, reusable* style → reproducible
across runs, works on every base model with catalog coverage, and
composes cleanly with the LoRA stack.

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
5. **Resolve** — the matched style is looked up in the catalog,
   which maps it to one or more LoRAs for your active base model
   (SD 1.5, SDXL, or Flux) plus a trigger phrase.
6. **Generate** — the LoRAs are downloaded (cached after first use)
   and merged into the UNet. The trigger phrase is prepended to your
   prompt. Generation proceeds through the standard pipeline.

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
      // but not transferable onto that base.
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

### Bundled catalog

The bundled catalog ships 5 styles:

| Id | Display name | Status | LoRA | Trigger |
|---|---|---|---|---|
| `watercolor` | Watercolor | full | `Arczisan/ink-watercolor` @ `cd8b7d93` | `"colorful inkpainting"` |
| `photorealistic` | Photorealistic | trigger-only | — | `"photograph, photorealistic, 35mm film, natural lighting"` |
| `oil_painting` | Oil Painting | full | `Jehugging/oilpaint_lora` @ `957cbf5d` | `"oil painting, impressionism, brush touch style, warm light"` |
| `ukiyo_e` | Ukiyo-e | full | `py-img-gen/lora-ukiyo-e-face-blip2-captions` @ `64553e15` | `"ukiyo-e, edo period woodblock print, traditional japanese"` |
| `art_nouveau` | Art Nouveau | trigger-only | — | `"art nouveau, alphonse mucha style, decorative borders, flowing lines, ornamental"` |

Each style ships with 4 public-domain exemplar images. See
`tests/fixtures/style_catalog/ATTRIBUTION.md` for sources — all
pre-1928 paintings/prints or NASA-PD photographs.

Per-style notes:

- **`photorealistic`** is intentionally trigger-only. SD 1.5 produces
  photographic output natively; no LoRA needed. The trigger nudges
  vocabulary toward photography, the `negative_extras` pushes away
  from painterly drift.
- **`oil_painting`** uses `Jehugging/oilpaint_lora` — community SD 1.5
  oil-paint LoRA with no declared license. Public on HF with the
  `text-to-image` tag, plakat doesn't redistribute. Trigger is broad
  ("oil painting, impressionism, brush touch style, warm light") so
  the LoRA's impressionist bias doesn't lock out cleaner classical
  output.
- **`ukiyo_e`** uses `py-img-gen/lora-ukiyo-e-face-blip2-captions` —
  CreativeML OpenRAIL-M, trained on BLIP2-captioned ukiyo-e face
  images, so output skews toward portraits. Still recognizably
  ukiyo-e for landscapes and other subjects, just with portrait bias.
  Only UNet targets merge (128/128); the LoRA wasn't trained with
  text-encoder LoRAs.
- **`art_nouveau`** is trigger-only because the only available SD 1.5
  art_nouveau LoRA (`SidXXD/Art_Nouveau_modern`) uses DreamBooth-style
  layer naming that doesn't merge against plakat's kohya-format
  merger (0/191 targets). SDXL alternatives exist but don't apply to
  the SD 1.5 slot. The trigger still pushes SD 1.5 toward
  Mucha-adjacent decorative output — meaningfully less specific than
  a trained LoRA would be, but recognizable.

### Where the catalog lives

The bundled catalog is at `assets/style_catalog/` relative to the
plakat source tree (also the working directory during `cargo run`).
Override with `--catalog <DIR>`:

```bash
plakat style detect ./inspiration.jpg --catalog ./my_catalog/
```

For installed binaries, point `--catalog` (or `--style-catalog`) at
the directory containing your built catalog files.

## CLI

### `plakat style detect <PHOTO>`

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

### `plakat style list`

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

### `plakat style show <ID>`

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

### `plakat style init`

Scan a directory of images and emit a starter catalog HJSON for the
catalog-build tool to consume. Useful for bootstrapping a personal
catalog from a corpus you've already organized by style.

```bash
plakat style init --from-dir <DIR> [--out <PATH>] [--force]
```

| Flag | Default | Description |
|---|---|---|
| `--from-dir` | required | Corpus directory. Each subdirectory becomes a style; the subdir name becomes the slugified style id; `.jpg`/`.jpeg`/`.png` files inside become exemplars. |
| `--out` | `<from-dir>/catalog.hjson` | Where to write the emitted HJSON. Exemplar paths in the file are resolved relative to this path's parent. |
| `--force` | (off) | Overwrite the output file if it already exists. |

What gets skipped: subdirectories named `holdout` (reserved for
smoke-test queries), subdirectories starting with `.` (hidden), and
subdirectories with no images (warning printed).

#### Example workflow

Given a corpus laid out as:

```
~/my_styles/
├── Moody Landscapes/    # 3 images
├── bright-portraits/    # 2 images
└── sketches/            # 1 image
```

Run init:

```
$ plakat style init --from-dir ~/my_styles
==> scanning /Users/.../my_styles
    moody_landscapes       3 images
    bright_portraits       2 images  ⚠ <3 exemplars
    sketches               1 images  ⚠ <3 exemplars

✓ wrote /Users/.../my_styles/catalog.hjson with 3 style(s)

Next steps:
  1. Edit /Users/.../my_styles/catalog.hjson to fill in descriptions,
     triggers, and LoRA pins.
  2. Build the catalog:
       cargo run --release --bin build_catalog -- \
         --sources /Users/.../my_styles/catalog.hjson \
         --out     /Users/.../my_styles/built
  3. Use it:
       plakat style detect <PHOTO> --catalog /Users/.../my_styles/built
```

The emitted HJSON is a valid `build_catalog` input out of the box —
every style has `models: {}` (detection-only) and an empty
`description`. The user edits the file to add transfer information
(LoRA references, trigger phrases, negative_extras) before building.

#### Slug rules

Directory names are normalized to style ids by lowercasing, replacing
runs of non-alphanumeric characters with `_`, and trimming
leading/trailing `_`. Examples:

| Directory name | Style id | Display name |
|---|---|---|
| `Moody Landscapes` | `moody_landscapes` | `Moody Landscapes` |
| `bright-portraits` | `bright_portraits` | `Bright Portraits` |
| `Mucha 1896!` | `mucha_1896` | `Mucha 1896` |

The display name is title-cased best-effort; the user is expected to
edit the HJSON anyway.

#### When `init` is the right tool

- **Bootstrapping a detection-only catalog** from a folder of images
  you've already grouped by style. Skip the manual HJSON authoring;
  edit the emitted file to add descriptions.
- **Bootstrapping a transfer catalog** when you know which LoRAs you
  want — init gets the exemplar layout in place, then you fill in the
  `models: {}` blocks per style.

#### When init is the *wrong* tool

- **You only have one style.** A single-style catalog isn't useful;
  detection always picks the one entry. init errors with a hint if
  zero usable subdirectories are found.
- **Your corpus is a flat folder, not subdirectories.** init expects
  one subdir per style. Reorganize first.
- **You want fully-automatic LoRA suggestions.** init doesn't search
  HuggingFace for matching LoRAs; that's the editorial work the
  curator has to do.

### `plakat style probe`

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
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors @ cd8b7d93 → UNet: 192/192 targets merged (scale 0.80)
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors @ cd8b7d93 → text encoder: 72/72 targets merged (scale 0.80)
→ ./out/plakat-690272596.png
```

The `@ cd8b7d93` suffix is the catalog's pinned revision SHA (8-char
prefix) — it appears when the catalog declared a `revision` for the
LoRA. Without a pin, plakat downloads from the repo's current `main`
and no suffix is shown. The cache is keyed by `(repo, revision, file)`
so different revisions don't collide.

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

Both `--style` and `--style-ref` replace user LoRAs when set.

### Behavior when the detected style has no LoRA for the active base

Three cases:

- **Style has no `models` section at all.** Treated as
  "detection-only" — generation proceeds with no LoRAs and no trigger
  injection, with a one-line warning.
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

Same four flags, same precedence with `--lora`, same warning
behaviors as `plakat generate`. `plakat portrait` keeps two reference
photos separate by design:

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

Scenarios apply a style either globally (whole scenario inherits a
single style) or per-task (each task can override the global with its
own `style-ref:`). The scenario HJSON accepts four new top-level
fields:

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

### Per-task `style-ref`

A task can override the scenario's global style with its own
`style-ref:`. The CLIP-H encoder is loaded once and shared via an
internal `StyleSession`, so a scenario with N per-task style refs
pays for the ~2.5 GB encoder load exactly once.

```hjson
{
    # No global style — every task picks its own.
    enhancer: deepseek
    scene: [ ... ]
    weather: [ ... ]
    tasks:
    [
        {
            name: forest_natural
            scene: forest
            weather: dawn
            prompt: "a fox in tall grass"
            style-ref: ./inspiration/fox_photo.jpg     # detects photorealistic
        }
        {
            name: forest_painted
            scene: forest
            weather: dawn
            prompt: "the same scene, painted"
            style-ref: ./inspiration/turner_watercolor.jpg  # detects watercolor
        }
    ]
}
```

Per-task style detection runs inside the task loop, just after the
task header is printed. The catalog's trigger phrase + `negative_extras`
for the picked style apply **for that task only** — the next task
without a `style-ref:` falls back to the scenario's global style (or no
style if none was set globally).

#### Important: trigger + negative only, not LoRAs

Plakat scenarios pre-load **one** generation pipeline at scenario start
with the global LoRAs baked into the UNet. Swapping LoRAs per task
would require reloading that pipeline (multi-GB UNet re-merge), which
is too expensive for a typical multi-task batch.

So per-task style overrides change the **trigger phrase** and the
**negative_extras** but NOT the LoRAs. When a per-task style resolves
to a different LoRA set than the global one, plakat warns:

```
▶ [2/3] t2_per_task_watercolor (scene=forest, weather=dawn)
  pre-enhance: a forest, dawn light, a deer
  → style: watercolor
  ⚠ per-task style 'watercolor' wants 1 LoRA(s); scenarios share one
    pipeline so only trigger + negative apply (global LoRAs stay loaded)
  ...
  final: colorful inkpainting, ...
```

This is useful when:
- Per-task style is **trigger-only** (e.g., `photorealistic`) — the
  warning doesn't fire, the trigger fully applies.
- Per-task style **happens to use the same LoRAs** as global (e.g.,
  `same_lora_set` matches) — no warning, full behavior.

It's less useful when:
- Per-task style needs **different LoRAs** than global. The trigger
  phrase will pull the prompt in the right direction, but without the
  LoRA's visual contribution the output won't fully take on the style.
  Restructure into separate scenarios per LoRA set for cases like this.

#### How per-task interacts with global

Per-task style **fully replaces** the global style for that task:
- The catalog trigger from the per-task style replaces the global one.
- The catalog `negative_extras` from the per-task style replaces the
  global one.
- (LoRAs unchanged from global — see above.)

The per-task trigger is prepended to the scenario's **bare**
`lora-header` (not to the global-trigger-modified one). Same for
`negative_extras` — combined with `negative:` from the scenario root,
not with the global style's contribution. This symmetry mirrors how
global style applies to user-authored values rather than to anything
already modified.

#### Constraints on per-task style fields

- **Per-task `style:` (pick by id) is not available.** The existing
  per-task `style:` field already means "IP-Adapter REF stylize-pass
  photo." To pick a catalog style by id per-task, use the scenario's
  global `style:` field and group same-style tasks into one scenario.
- **Per-task `style-strength:` is not available.** Same reasoning;
  the existing per-task `style-strength:` controls the IP-Adapter
  pass. Catalog strength is global-only.
- **Per-task LoRA swaps are not supported.** See "trigger + negative
  only" above — scenarios share one pre-loaded pipeline.

## Building a custom catalog

The catalog-build tool is at `src/bin/build_catalog.rs` — built
alongside the main `plakat` binary as a secondary executable. It reads
a curator-authored HJSON config, encodes exemplar images through
CLIP-H, and emits a complete catalog with sidecar files.

### Usage

```bash
cargo run --release --bin build_catalog -- \
    --sources tools/style_sources/catalog.hjson \
    --out     assets/style_catalog
```

Or, after a `cargo build --release`, run the cached binary directly:

```bash
./target/release/build_catalog \
    --sources tools/style_sources/catalog.hjson \
    --out     assets/style_catalog
```

| Flag | Default | Description |
|---|---|---|
| `--sources` | required | Path to the curator's HJSON config. Exemplar paths are resolved relative to this file's directory. |
| `--out` | required | Output directory; created if missing. Existing files are overwritten. |
| `--device` | `auto` | Encoder device (`auto` / `cuda[:N]` / `metal` / `cpu`). |
| `--probe-hf` | (off) | HEAD-check every catalog LoRA on HuggingFace before encoding. Fails the build on any non-200. Recommended in CI. |

### Curator HJSON schema

The shipped tools/style_sources/catalog.hjson is the live source for
the bundled catalog. It looks like:

```hjson
{
    schema_version: 1
    encoder: { id: clip-h-laion2b, embed_dim: 1024 }
    detection:
    {
        aggregation: top3-mean
        min_confidence: 0.22
        margin_over_runner_up: 0.02
    }
    styles:
    [
        {
            id: watercolor
            display_name: Watercolor
            description: "Wet-on-wet pigment washes, ink lineart, paper texture."
            exemplars:
            [
                ../../tests/fixtures/style_catalog/watercolor/01_durer_hare.jpg
                ../../tests/fixtures/style_catalog/watercolor/02_sargent_alligators.jpg
                # ... 3-5 exemplars per style is typical
            ]
            models:
            {
                sd15:
                {
                    loras:
                    [
                        {
                            spec: "Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8"
                            revision: cd8b7d93ec0b6c0aa31a640f1287837583d702d0
                            license_url: "https://huggingface.co/Arczisan/ink-watercolor"
                        }
                    ]
                    trigger: "colorful inkpainting"
                    negative_extras: "3d render, photo, glossy"
                }
            }
        }
    ]
}
```

LoRA entries accept either a shorthand string (`"org/repo:0.8"`) or
a full object with `spec`, `revision`, `license`, `license_url`. The
builder normalizes both to the full-object form in the emitted
`catalog.json`.

### What the builder does

1. **Parse + validate** — HJSON parse, schema_version check, duplicate
   style ids, missing exemplar paths, unknown base-model slot names
   (`sd15` / `sdxl` / `flux` only), unparseable LoRA specs.
2. **Soft warnings** — styles with <3 exemplars (sparse, detection may
   be unreliable); per-base entries that have neither LoRAs nor a
   trigger (declaring them does nothing).
3. **Optional HF probe** — when `--probe-hf` is set, every LoRA's
   URL is HEAD-requested before encoding starts (so curators don't sit
   through a 60-second encode pass just to find their LoRA reference
   is broken).
4. **Encode exemplars** — every exemplar image is loaded, CLIP-preprocessed,
   passed through CLIP-H, L2-normalized, downcast to f16 and stored at
   key `<style_id>/<idx>` in `exemplars.safetensors`.
5. **Emit outputs**:
   - `catalog.json` — runtime-format catalog
   - `exemplars.safetensors` — embeddings
   - `LICENSES.md` — markdown table of every LoRA's license + url
   - `provenance.json` — exemplar_key → source path mapping, for
     reproducible rebuilds

### Sample output

```
$ cargo run --release --bin build_catalog -- \
      --sources tools/style_sources/catalog.hjson \
      --out assets/style_catalog --probe-hf

==> probing 1 LoRA reference(s) on HuggingFace
  ✓ Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8 (sd15 @ cd8b7d93)
==> loading CLIP-H image encoder
==> watercolor (4 exemplars)
    watercolor/00 ← ../../tests/fixtures/style_catalog/watercolor/01_durer_hare.jpg
    ...
==> wrote assets/style_catalog/exemplars.safetensors (8 tensors)
==> wrote assets/style_catalog/catalog.json
==> wrote assets/style_catalog/LICENSES.md
==> wrote assets/style_catalog/provenance.json

✓ built catalog: 2 style(s), 8 exemplar embedding(s), 1 LoRA reference(s)
```

### Sidecars

**`LICENSES.md`** — markdown table per style, per base-model:

```markdown
## Watercolor (`watercolor`)

### `sd15` base

| Spec | Revision | License | URL |
|---|---|---|---|
| `Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8` | `cd8b7d93` | (not declared) | https://... |
```

**`provenance.json`** — exemplar key → source image path, so rebuilds
from the same sources should produce the same `exemplars.safetensors`:

```json
{
  "schema_version": 1,
  "encoder_id": "clip-h-laion2b",
  "exemplars": {
    "watercolor/00": { "source": "../../tests/fixtures/style_catalog/watercolor/01_durer_hare.jpg" },
    ...
  }
}
```


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

The default `0.22` works for the bundled styles. Re-tune when:

- You expand the catalog and see false-positive picks → raise it.
- You see correct picks getting rejected → lower it.

CLIP-H pooled cosines on style-discrimination tasks typically land in
the `[0.15, 0.55]` range; the bundled catalog's observed values
(`0.10` for out-of-style, `0.50+` for in-style) span most of that
range, giving plenty of headroom.

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
  cannot be detected. The shipping catalog covers 5 styles:
  `watercolor`, `photorealistic`, `oil_painting`, `ukiyo_e`,
  `art_nouveau`. Three carry working SD 1.5 LoRAs; two are
  trigger-only (`photorealistic` because SD 1.5 does it natively,
  `art_nouveau` because no usable SD 1.5 LoRA was findable as of the
  curator pass — see "Bundled catalog" below).
- **Revision pinning is honored end-to-end.** When the catalog records
  a `revision` SHA / tag for a LoRA, `plakat generate --style-ref` and
  `plakat portrait --style-ref` request that exact revision from
  HuggingFace. The download log displays the short SHA when present
  (`Arczisan/ink-watercolor/inkwatercolor.safetensors @ cd8b7d93`).
  `plakat style probe` also checks the pinned revision URL, so probe
  and download cannot diverge.
- **LoRA availability isn't guaranteed.** Catalog LoRAs point at HF
  repos that can occasionally disappear or be renamed. `plakat style
  probe` catches this early.
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
- Use `--style <id>` to force a specific style.
- Expand the catalog with more representative exemplars.
- Lower `min_confidence` in `catalog.json`.

**`Detected: <style> [ambiguous]`**
Top-1 is within `margin_over_runner_up` of top-2. Both candidates are
shown. Run generation against each (once `--style <id>` ships) and
pick the better result.

## Testing

The smoke test lives at `tests/style_detect_smoke.rs`. It runs three
checks against the bundled catalog:

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

The cosine numbers above demonstrate that CLIP-H pooled embeddings
carry plenty of style signal to discriminate the bundled styles with
a 0.3+ margin between right and wrong.

## See also

- `PERSONA.md` — identity preservation (FaceID, IP-Adapter-Plus-Face).
  Distinct from style detection: identity is about *who*, style is
  about *how*.
- `GENERATE.md` — the standard text-to-image pipeline that style
  detection will plug into.
- `README.md` — top-level plakat documentation.
