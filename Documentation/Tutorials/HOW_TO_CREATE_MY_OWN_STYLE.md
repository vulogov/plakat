# Build your own style catalog from a folder of images

This walks you from "I have a folder of art I like" to "I can detect
that style in any photo and apply it to my generations." End-to-end:
corpus → init → build → use.

## What you'll learn

- The three-file structure that makes up a plakat style catalog
- How to turn a folder of images into a working catalog in ~5 minutes
- How to test detection and refine your exemplars
- How to (optionally) add a LoRA so the style actually transfers — not
  just detected
- The tradeoffs between **detection-only** catalogs (quick) and full
  **transfer** catalogs (more setup, but you can render in the style)

## Before you start

- Work through `GENERATE_TUTORIAL.md` so you've generated at least one
  image with plakat.
- Skim `STYLES_TUTORIAL.md` so you've used the bundled styles and know
  what a catalog "feels like" from the user side.
- Have a corpus of style examples — say, 3-30 images per style you
  want to teach plakat. JPEGs or PNGs, 512px+ on the short side ideal.
- About 1-2 GB free disk space for plakat's CLIP-H download (one-time;
  shared with portrait + bundled style features).

---

## 1. The big picture

A plakat style catalog is two files in a directory:

```
my_catalog/
├── catalog.json           # routing metadata
└── exemplars.safetensors  # CLIP-H embeddings of your example images
```

You don't write these by hand. Plakat builds them from a curator's
config — an HJSON file (a JSON variant with comments and relaxed
syntax) that describes:

- One or more **styles** (e.g., `watercolor`, `my_landscape_style`).
- For each style: a list of **exemplar images** (the actual files
  representing what that style looks like).
- Optionally per style: **LoRA refs**, **trigger phrases**, **negative
  extras** (the bits that *transfer* style during generation).

Building the catalog runs each exemplar through CLIP-H, gets a 1024-d
fingerprint per image, packs them into the safetensors file, and
emits a `catalog.json` with the routing metadata.

The three-step workflow:

1. **Organize** your images into subdirectories named after the
   styles.
2. **Init** — plakat scans the layout and emits a starter HJSON.
3. **Build** — plakat encodes the exemplars and produces the catalog.

After that, you can use the catalog anywhere plakat takes a
`--catalog` / `--style-catalog` flag.

---

## 2. Step 1 — Organize your images

Decide what styles you want. Make one subdirectory per style. Put 3+
images representing that style inside each.

Example: you collect art and want plakat to recognize three distinct
sub-genres of your collection.

```
~/my_styles/
├── moody_landscapes/
│   ├── 01.jpg
│   ├── 02.jpg
│   ├── 03.jpg
│   └── 04.jpg
├── bright_portraits/
│   ├── 01.jpg
│   ├── 02.jpg
│   └── 03.jpg
└── geometric_abstracts/
    ├── 01.jpg
    ├── 02.jpg
    ├── 03.jpg
    ├── 04.jpg
    └── 05.jpg
```

### What makes a good exemplar set?

- **3 minimum, ideally 4-8 per style.** Fewer than 3 makes detection
  noisy; more than ~10 has diminishing returns.
- **Diverse subjects within the style.** If you're teaching plakat
  "moody landscapes," include forests, seascapes, mountains — all in
  the *style* you want but with different *content*. The model should
  learn the style, not the content.
- **Avoid the obvious trap.** Don't put 4 photos of the same painting
  from different angles. Plakat will learn "this specific painting"
  not "this style."
- **Reasonable resolution.** 512px+ on the short side. The CLIP-H
  preprocessor downsizes to 224x224 anyway, but starting from too
  small loses detail.
- **JPEG, JPEG2000, or PNG.** Avoid HEIC, WebP, AVIF — plakat doesn't
  decode those.

### Names you should avoid for subdirectories

- `holdout/` — reserved; plakat skips it (it's where smoke-test images
  live in dev workflows).
- Hidden directories (`.git`, `.DS_Store`, anything starting with `.`)
  are skipped.
- Empty subdirectories are skipped with a warning.

---

## 3. Step 2 — Bootstrap the HJSON with `plakat style init`

Plakat scans your corpus and emits a starter HJSON:

```bash
plakat style init --from-dir ~/my_styles
```

You'll see:

```
==> scanning /Users/.../my_styles
    moody_landscapes       4 images
    bright_portraits       3 images
    geometric_abstracts    5 images

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

Notes on what just happened:

- Directory names got slugified (`bright-portraits` becomes
  `bright_portraits`).
- Display names got title-cased (`bright_portraits` → `Bright Portraits`).
- Each style has `models: {}` — meaning **detection-only** for now.
  No LoRAs, no trigger phrases. You can add those later or never.
- Exemplar paths in the HJSON are relative to the HJSON's directory,
  so the catalog stays portable.

The emitted HJSON is *immediately valid* — you can build it as-is for
detection-only use, or edit it first.

---

## 4. Step 3 — Edit the HJSON (optional)

Open the generated `catalog.hjson`:

```hjson
{
    # Auto-generated by `plakat style init`. Edit to fill in
    # descriptions, trigger phrases, and LoRA references.

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
            id: moody_landscapes
            display_name: "Moody Landscapes"
            description: ""

            exemplars:
            [
                moody_landscapes/01.jpg
                moody_landscapes/02.jpg
                moody_landscapes/03.jpg
                moody_landscapes/04.jpg
            ]

            # Add per-base-model LoRA references + trigger phrases below.
            # Leave `models: {}` for a detection-only style (no transfer).
            models: {}
        }
        ...
    ]
}
```

### What you might want to edit

- **`description`**: 1-2 sentences. Shown in `plakat style list` /
  `plakat style show`. Useful for your future self.
- **Add `models: { sd15: { ... } }`** if you want this style to
  actually transfer during generation (next section).
- **Add or remove exemplars** as you refine. The HJSON is the source
  of truth; you can edit it after the initial scan.

For **detection-only catalogs** (just want to *recognize* the style
without rendering in it), you can skip all editing. Move on to
step 4 directly.

---

## 5. Step 4 — Build the catalog

Encode the exemplars into the runtime format:

```bash
cargo run --release --bin build_catalog -- \
    --sources ~/my_styles/catalog.hjson \
    --out     ~/my_styles/built
```

Or if you've already built the release binary:

```bash
./target/release/build_catalog \
    --sources ~/my_styles/catalog.hjson \
    --out     ~/my_styles/built
```

Output:

```
==> loading CLIP-H image encoder
==> moody_landscapes (4 exemplars)
    moody_landscapes/00 ← moody_landscapes/01.jpg
    moody_landscapes/01 ← moody_landscapes/02.jpg
    moody_landscapes/02 ← moody_landscapes/03.jpg
    moody_landscapes/03 ← moody_landscapes/04.jpg
==> bright_portraits (3 exemplars)
    bright_portraits/00 ← bright_portraits/01.jpg
    bright_portraits/01 ← bright_portraits/02.jpg
    bright_portraits/02 ← bright_portraits/03.jpg
==> geometric_abstracts (5 exemplars)
    geometric_abstracts/00 ← geometric_abstracts/01.jpg
    geometric_abstracts/01 ← geometric_abstracts/02.jpg
    geometric_abstracts/02 ← geometric_abstracts/03.jpg
    geometric_abstracts/03 ← geometric_abstracts/04.jpg
    geometric_abstracts/04 ← geometric_abstracts/05.jpg
==> wrote ~/my_styles/built/exemplars.safetensors (12 tensors)
==> wrote ~/my_styles/built/catalog.json
==> wrote ~/my_styles/built/LICENSES.md
==> wrote ~/my_styles/built/provenance.json

✓ built catalog: 3 style(s), 12 exemplar embedding(s), 0 LoRA reference(s)
```

What was produced:

- `catalog.json` — runtime routing (which style id maps to which
  exemplar keys, trigger phrase, LoRAs).
- `exemplars.safetensors` — your image fingerprints (CLIP-H 1024-d
  embeddings, f16, L2-normalized). Small — typically <100 KB total.
- `LICENSES.md` — automatic sidecar listing every LoRA's license
  (empty for detection-only catalogs).
- `provenance.json` — maps each embedding back to the source image
  path. Useful for verifying rebuilds and auditing what went into
  the catalog.

First run downloads ~2.5 GB CLIP-H weights from HuggingFace. Cached
afterward; subsequent rebuilds are fast (~1 second per exemplar on
CPU).

---

## 6. Step 5 — Test detection

Before using your catalog in real generation, sanity-check it. Pick
an image that *should* match one of your styles (but isn't already in
the catalog — a "holdout") and ask plakat to detect:

```bash
plakat style detect ~/new_image.jpg --catalog ~/my_styles/built
```

Expected output (assuming your styles are well-separated in CLIP
space):

```
Detected: moody_landscapes (0.5841) [picked]

Top 3:
  1. moody_landscapes      0.5841  ✓ picked
  2. geometric_abstracts   0.2104
  3. bright_portraits      0.1822
```

The picked style should be obviously the right one, with the runner-
up well behind. Healthy detection has the top score above 0.3 and a
margin of 0.1+ over the runner-up.

### Signs your catalog needs work

- **Top score below 0.22 (no auto-pick).** Either your exemplars are
  too few, too diverse, or your holdout isn't actually in any of your
  styles. Either add more representative exemplars or accept that
  this image is out-of-distribution.
- **Top two scores within 0.02 of each other ("ambiguous").** Two
  styles are blurring together in CLIP space. Either the exemplars
  are too similar (the styles aren't really distinct visually) or
  you need to add more discriminating exemplars to one of them.
- **The wrong style consistently wins.** Your exemplars for the
  "correct" style might be unrepresentative. Replace some with better
  examples and rebuild.

### Iterate

Rebuild is fast after the first run (CLIP-H stays cached). The loop
is:

1. Notice a detection problem.
2. Add/replace exemplars in the right subdirectory.
3. Re-run `plakat style init` *or* (more commonly) just edit
   `catalog.hjson` to add/remove the exemplar paths.
4. Re-run `build_catalog`.
5. Re-test with `plakat style detect`.

---

## 7. Step 6 — Use the catalog

With a working catalog at `~/my_styles/built`, use `--catalog` (for
`plakat style ...` commands) or `--style-catalog` (for generation):

```bash
# List your styles
plakat style list --catalog ~/my_styles/built

# Detect on a new photo
plakat style detect ./photo.jpg --catalog ~/my_styles/built

# Generate (uses your catalog instead of the bundled one)
plakat generate "a small house on a hill" \
    --style-ref ./inspiration.jpg \
    --style-catalog ~/my_styles/built

# Generate (pick by name from your catalog)
plakat generate "a small house on a hill" \
    --style moody_landscapes \
    --style-catalog ~/my_styles/built

# In a scenario, set the style-catalog field globally:
# style-catalog: ~/my_styles/built
```

For **detection-only catalogs**, the trigger phrase is empty and no
LoRAs apply, so the style is detected but generation looks just like
plain `plakat generate`. Detection is what you got. To make the style
*transfer*, you need a LoRA (next section).

---

## 8. Going further — adding a LoRA to actually transfer the style

Detection-only catalogs are useful for *recognizing* a style. To
*render* in a style, plakat needs a LoRA — a small model add-on that
changes how the base model paints.

### Finding a LoRA

You have to source the LoRA yourself. Options:

- **HuggingFace search** — look for SD 1.5 LoRAs matching your
  style. Filter to `text-to-image` pipeline + `lora` tag.
- **Train your own** — outside plakat's scope. Tools like kohya_ss
  train LoRAs from a corpus of images.
- **Trigger-only** — skip the LoRA entirely. The trigger phrase
  alone can push the model in the right direction for styles the base
  model already does well.

For this tutorial, assume you found `someone/cool-watercolor-lora` on
HuggingFace. Pin its current SHA:

```bash
curl -s https://huggingface.co/api/models/someone/cool-watercolor-lora \
    | python3 -c "import json,sys; print(json.load(sys.stdin).get('sha'))"
```

Now add it to your catalog. Open `~/my_styles/catalog.hjson` and edit
the `moody_landscapes` style:

```hjson
{
    id: moody_landscapes
    display_name: "Moody Landscapes"
    description: "Atmospheric outdoor scenes; muted palette, soft lighting."

    exemplars:
    [
        moody_landscapes/01.jpg
        ...
    ]

    # Was: models: {}
    # Now: per-base-model routing.
    models:
    {
        sd15:
        {
            loras:
            [
                {
                    spec: "someone/cool-watercolor-lora:0.8"
                    revision: "abc123def456..."
                    license: "creativeml-openrail-m"
                    license_url: "https://huggingface.co/someone/cool-watercolor-lora"
                }
            ]
            trigger: "moody atmospheric watercolor, soft palette"
            negative_extras: "3d render, glossy, vibrant, photorealistic"
        }
    }
}
```

Fields:

- `spec`: HuggingFace `repo:scale` syntax — or `repo#explicit_file:scale`
  if the repo has multiple safetensors. `0.8` is the LoRA's
  application strength (catalog default; runtime `--style-strength`
  multiplies it).
- `revision`: HuggingFace commit SHA, tag, or branch. Pinning a SHA
  means your generations are reproducible even if upstream changes.
- `license` / `license_url`: declarative metadata; plakat displays this
  in `plakat style show` so future users see what they're agreeing to.
- `trigger`: words the LoRA was trained to recognize. Get these from
  the LoRA's README on HuggingFace. Plakat prepends this to every
  prompt that uses this style.
- `negative_extras`: pushed into the negative prompt to discourage
  competing styles.

Rebuild:

```bash
./target/release/build_catalog \
    --sources ~/my_styles/catalog.hjson \
    --out     ~/my_styles/built \
    --probe-hf      # confirms LoRA URL still resolves before encoding
```

The `--probe-hf` flag is recommended when adding new LoRAs — it
HEAD-checks the URL upfront so a typo'd repo name fails fast instead
of after a long encode pass.

### Verify the LoRA loads

```bash
plakat style show moody_landscapes --catalog ~/my_styles/built
```

Output should include:

```
Models:
  sd15:
    loras:
      - someone/cool-watercolor-lora:0.8 (revision: abc123de...)
    trigger:   "moody atmospheric watercolor, soft palette"
    negative+: "3d render, glossy, vibrant, photorealistic"
```

And in a real generation:

```bash
plakat generate "a small house on a hill" \
    --style moody_landscapes \
    --style-catalog ~/my_styles/built
```

You should see:

```
  → style: moody_landscapes
 INFO LoRA someone/cool-watercolor-lora/... → UNet: 192/192 targets merged (scale 0.80)
 INFO LoRA someone/cool-watercolor-lora/... → text encoder: 72/72 targets merged (scale 0.80)
→ ./out/plakat-<seed>.png
```

`192/192` and `72/72` mean every weight in the LoRA found a target in
the base model — full merge. `0/192` means the LoRA's layer naming
isn't compatible with plakat's merger (different LoRA training
script). In that case, the LoRA isn't actually applying — fall back
to trigger-only by removing the `loras:` entry but keeping the
`trigger:`.

---

## 9. Sharing or distributing your catalog

A catalog is portable. To share:

1. Bundle the **built** directory: `catalog.json`,
   `exemplars.safetensors`, plus the sidecars `LICENSES.md` and
   `provenance.json`.
2. **Don't** include the original exemplar images — the embeddings
   stand alone for detection. Less to ship, less licensing risk.
3. Recipient runs: `plakat style detect <photo> --catalog
   <bundled-dir>` or `plakat generate ... --style-catalog
   <bundled-dir>`.

If you also share the **source HJSON + exemplar images**, your
recipient can rebuild the catalog and tweak/extend it. Useful for
collaboration.

---

## 10. Common issues

**`init` finds zero styles.**
The corpus directory has no subdirectories (just images at the top
level). Reorganize: each style needs its own subdir.

**`build_catalog` fails on "exemplar not found".**
The HJSON's exemplar paths are relative to the HJSON's directory. If
you moved the HJSON without moving the images, paths break. Either
move both together or update the HJSON paths.

**Build is slow on first run.**
First run downloads CLIP-H (~2.5 GB). Subsequent rebuilds reuse the
cache. CLIP-H is also reused with `plakat portrait --identity faceid`
and the bundled `plakat style detect` flow, so if you've used those
features the weights are already cached.

**Detection is unreliable / scores are low across the board.**
Three usual causes:

- **Too few exemplars** (<3 per style). Add more.
- **Style is too broad.** "All my paintings" isn't a style. Split
  into more specific categories.
- **CLIP-H just isn't discriminating well for your domain.** This is
  a real limit; CLIP-H was trained on web image-text pairs and
  doesn't always cluster the way humans cluster style. Try
  alternative aggregation (`detection: { aggregation: max }`) — it
  sometimes helps when single-image likeness matters more than
  averaging.

**LoRA loads but merges 0 targets.**
The LoRA was trained with a non-kohya-format training script. Plakat
only supports kohya-format LoRAs (the most common). Workaround: drop
the LoRA from the catalog and use the style trigger-only.

**My catalog works alone but breaks when I also pass `--lora` flags.**
`--style-ref` and `--style-catalog` swap out the LoRA list with the
catalog's. User `--lora` flags are dropped with a warning. Use
`--style <id>` (without `--style-ref`) if you want catalog LoRAs +
user LoRAs to stack. This is by-design.

---

## Where to next

- **Use your catalog day-to-day** → `STYLES_TUTORIAL.md`
- **Full reference for catalog schema + every flag** →
  `Documentation/STYLES.md`
- **How styles compose with portraits** → `PORTRAIT_TUTORIAL.md`
- **The shipped catalog's `tools/style_sources/catalog.hjson`** —
  a realistic example with mixed detection-only + LoRA-bearing
  entries
