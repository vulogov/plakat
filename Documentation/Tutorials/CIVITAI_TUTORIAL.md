# Browsing Civitai from the command line

[Civitai](https://civitai.com) is the largest community hub for
Stable Diffusion checkpoints, LoRAs, embeddings, and ControlNet
adapters. Everything you'd otherwise click through the web UI to
find, download, and reference by local path can be done from inside
plakat with one subcommand: `plakat civitai`.

This tutorial walks through the full loop:

1. Searching by free-text query + type filter.
2. Reading the search output (what every column means).
3. Drilling into a single model with `info`.
4. Downloading the right file into plakat's cache.
5. Using the downloaded path with `--lora`, `--model`, or
   `plakat embedding info`.

By the end you should be able to find a community LoRA, pull it
locally, and slot it into a `plakat generate` invocation without
ever leaving the terminal.

## Prerequisites

- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You
  should be comfortable with `--lora`, `--model`, and seed
  reproducibility.
- A working internet connection. The first Civitai search hits the
  public API; downloads stream from Civitai's CDN.
- **Optional**: a [Civitai API key](https://civitai.com/user/account)
  if you want to download gated / NSFW / "early access" assets.
  Public models work without one — every example in this tutorial
  works anonymously.

## 1. Your first search

The simplest invocation is one positional argument — your query:

```bash
plakat civitai search "watercolor"
```

This returns the top 10 most-relevant matches across **every**
asset type (checkpoints, LoRAs, embeddings, controlnet, vae,
poses, ...) — not always what you want, since a mixed result list
is hard to scan.

Filter to a specific type:

```bash
plakat civitai search "watercolor" --type lora
```

Other types accepted by `--type`:

| Value | What it filters to |
|---|---|
| `lora` | LoRA / LoCon adapters |
| `checkpoint` (or `ckpt`, `model`) | Full base models (SD, SDXL, Flux community fine-tunes) |
| `ti` (or `embedding`, `textualinversion`) | Textual Inversion files |
| `controlnet` (or `cn`) | ControlNet adapters |
| `vae` | VAE replacements |
| `locon` (or `lycoris`) | LyCORIS LoCon variants |
| `hypernetwork` | Legacy hypernetwork files |
| `poses` | Pose-only assets (depth maps, OpenPose skeletons) |

## 2. Reading the search output

A typical search response looks like:

```
• 9 match(es)
2595428  Some LoRA Name  [LORA]  by @some_creator
  base=Illustrious  triggers=(watercolor_(medium), some_trigger) downloads=273
  ★ MyLoRA_v1.safetensors (20.6 MB)
2614696  Another LoRA  [LORA]  by @other_creator
  base=Anima  triggers=(@something, watercolor painting style) downloads=247
  ★ NeonGenesisEvangelionArtBookAnima.safetensors (126.2 MB)
  · config.yaml (4.0 KB)
...
```

Each entry shows:

- **Numeric ID** (`2595428`) — the model ID. You'll use this when
  drilling in with `info` or downloading.
- **Name** and **type** (`[LORA]`).
- **Creator** (`@username`).
- **`base=`** — the SD variant the LoRA was trained against
  (`SD 1.5`, `SDXL 1.0`, `Flux.1 D`, `Illustrious`, `Pony`, ...).
  **Pair the LoRA with the matching `--model`** or it won't apply
  cleanly.
- **`triggers=`** — the trigger words the creator used. Most LoRAs
  need at least one of these in your prompt to activate. Read them
  carefully; some LoRAs ship dozens.
- **`downloads=`** — popularity proxy. Higher generally means
  better-vetted, but new high-quality LoRAs start at 0.
- **File list** — each file shipped with the version. The **★**
  marks the primary file (the one plakat downloads by default);
  **·** marks secondary files (configs, alternative weights).
  Sizes are shown.

## 3. Drilling into one model

Once you've picked something promising, `info` shows the full
details — every version the creator has released, every file in
each version:

```bash
plakat civitai info 2595428
```

Output:

```
2595428  Some LoRA Name  [LORA]
  by @some_creator
  downloads=273  thumbs_up=12  rating=4.80
  tags: anime, watercolor, character
  • v1.0 (id=2614696, base=Illustrious)
    triggers: watercolor_(medium), some_trigger
    ★ MyLoRA_v1.safetensors (20.6 MB)
  • v0.9 (id=2598123, base=Illustrious)
    triggers: watercolor_(medium)
    ★ MyLoRA_v0.9.safetensors (18.4 MB)
```

`info` also accepts the same `<REF>` grammar as `download`:

```bash
# Bare integer (model ID)
plakat civitai info 2595428

# The civitai: shorthand
plakat civitai info civitai:2595428

# A full URL pasted from the browser
plakat civitai info "https://civitai.com/models/2595428"

# A URL with a pinned version
plakat civitai info "https://civitai.com/models/2595428?modelVersionId=2614696"

# A direct download URL (version-level)
plakat civitai info "https://civitai.com/api/download/models/2614696"
```

## 4. Downloading

Once you know which version + file you want, `download` fetches
it into plakat's cache:

```bash
# Latest version, primary file — the common case
plakat civitai download 2595428

# Pin to a specific older version
plakat civitai download "https://civitai.com/models/2595428?modelVersionId=2598123"

# Pick a non-primary file by name
plakat civitai download 2614696 --file "config.yaml"
```

plakat prints the absolute path the file landed at:

```
✓ /Users/you/.cache/plakat/civitai/model-2595428/version-2614696/MyLoRA_v1.safetensors (20.6 MB)
→ drop this path into --lora or --model PATH
```

The downloader is **atomic**: the file streams into a `.partial`
sibling first, then renames on success. If you Ctrl-C halfway, no
corrupt file is left in the cache slot — the next run re-downloads
cleanly.

**Cache hit**: if the same `(model, version, file)` is already in
the cache at the expected size, `download` short-circuits without
hitting the network:

```
✓ /Users/you/.cache/plakat/civitai/model-2595428/version-2614696/MyLoRA_v1.safetensors (cached)
```

## 5. Using the downloaded asset

The path plakat printed is the same kind of path `--lora` and
`--model` already accept:

```bash
# LoRA
plakat generate "a fox in tall grass, watercolor_(medium), some_trigger" \
    --model sd15 \
    --lora ~/.cache/plakat/civitai/model-2595428/version-2614696/MyLoRA_v1.safetensors

# Stack with another LoRA
plakat generate "..." \
    --model sd15 \
    --lora ~/.cache/plakat/civitai/model-2595428/version-2614696/MyLoRA_v1.safetensors:0.7 \
    --lora some/other-style-lora:0.4

# Custom checkpoint (downloaded via `civitai download` with --type checkpoint)
plakat generate "..." \
    --model ~/.cache/plakat/civitai/model-12345/version-67890/the_checkpoint.safetensors
```

For Textual Inversion files (`--type ti`):

```bash
plakat civitai download <id-of-a-TI>
# Inspect it first:
plakat embedding info ~/.cache/plakat/civitai/model-XXX/version-YYY/the_ti.safetensors
```

The TI inspector reports the trigger word, vector count, and the
matching SD variant. (Runtime injection of TIs into the SD pipeline
isn't wired yet — see the GENERATE_TUTORIAL §17 for the workaround
options.)

## 6. NSFW filtering

By default `civitai search` filters out NSFW-tagged models. Pass
`--include-nsfw` to see them too:

```bash
plakat civitai search "portrait" --type lora --include-nsfw
```

The filter is client-side (the API still returns the entries; we
drop them post-fetch). Civitai's NSFW classification has caveats —
treat the filter as advisory.

## 7. Pagination

`--limit N` controls the page size (1-100, default 10). `--page P`
walks to the requested page:

```bash
# Top 50 LoRAs for "watercolor"
plakat civitai search "watercolor" --type lora --limit 50

# Second page of 10 (matches 11-20)
plakat civitai search "watercolor" --type lora --page 2
```

Implementation note: Civitai's API uses **cursor-based** pagination
when a query string is set (so deep `--page` walks issue one
HTTP round-trip per intermediate page). For typical browsing
(`--page 1` / `--page 2` / `--page 3`) this is invisible. For
deep paging (`--page 20`), prefer refining the query — the
20× round-trip cost is real, and Civitai's search ranking is
strong enough that anything past page 3 is rarely worth the wait.

## 8. Gated assets + your API key

Some Civitai models are gated — usually for the creator's
"early access" or specific NSFW assets. Downloads of those return
401 unless you set `CIVITAI_API_KEY`:

```bash
export CIVITAI_API_KEY="your-key-here"
plakat civitai download 12345
```

Get a key from your [Civitai account page](https://civitai.com/user/account)
under "API Keys". Public assets continue to work without one.

When a 401 happens, plakat reports it with a clear pointer to the
setup:

```
Error: Civitai download returned 401 — this asset is gated. Set
CIVITAI_API_KEY from https://civitai.com/user/account → API Keys.
```

## 9. Cache layout + cleanup

Everything plakat downloads lives under:

```
<plakat-cache>/civitai/
├── model-<id>/
│   └── version-<id>/
│       ├── the-file.safetensors
│       └── metadata.json     ← serialized version record
```

`metadata.json` carries the full version record (trigger words,
base model, files, hashes) — useful for revisiting what a
particular cached file actually is months later.

The cache root follows the same resolution as the HF cache:
`--cache-dir` flag → `PLAKAT_CACHE_DIR` env → `HUGGINGFACE_HUB_CACHE`
env → `HF_HOME/hub` → `~/.cache/huggingface/hub`. plakat creates
the `civitai/` sibling automatically.

To clean up, just delete the directories — there's no `plakat
civitai rm` subcommand yet. The most space-efficient cleanup is
to remove the per-version dirs of assets you no longer use; the
per-model dir is empty without its versions.

## 10. Common gotchas

- **Base model mismatch.** A LoRA trained against SDXL won't
  meaningfully apply to SD 1.5 — plakat detects this and warns,
  but the warning is easy to miss. Pair the `base=` field from
  the search output with the right `--model` from
  [`GENERATE.md`](../GENERATE.md).
- **Trigger words must appear in the prompt.** Most LoRAs are
  trained with a specific tag (e.g. `watercolor_(medium)`,
  `bxz`, `@kimura_(katatema)`). Without it, the LoRA contributes
  almost nothing. Copy the triggers verbatim — Civitai trains
  with literal token strings including the parens, underscores,
  and `\` escapes.
- **"Illustrious" / "Pony" / "Anima" base models** are SD-family
  community fine-tunes — drive them with `--model sdxl` (most
  Illustrious / Pony bases) or pull the matching checkpoint from
  Civitai too. They're not built-in plakat aliases.
- **`--type checkpoint` downloads are big.** 5-7 GB is typical
  for SDXL fine-tunes; some Flux community checkpoints are
  20-30 GB. Watch your `--cache-dir` capacity.
- **Search results change over time.** Civitai's ranking algorithm
  is opaque + churns popularity. A search that returned 9 matches
  yesterday may return 12 today.

## Where to next

- **Full subcommand reference**:
  [`GENERATE.md`](../GENERATE.md) — the `plakat civitai` section
  lists every flag.
- **Using downloaded LoRAs**:
  [`GENERATE_TUTORIAL.md §7`](GENERATE_TUTORIAL.md) — the LoRA
  fundamentals.
- **Inspecting downloaded TI files**:
  [`GENERATE_TUTORIAL.md §17`](GENERATE_TUTORIAL.md) — the Textual
  Inversion section.
- **Style detection from a Civitai-downloaded checkpoint**:
  [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md) — pair a downloaded
  base model with the bundled style catalog.
