# Looks — art-medium presets (v0.25)

`plakat` can render your prompt in a chosen **art medium** —
watercolor, oil painting, charcoal, pencil, chalk pastel, linocut,
gouache, or ink wash — with one flag. The preset bundles a prompt
prefix/suffix, a recommended sampler + step count + CFG scale, a
negative-prompt extension, and (when you haven't passed `--lora`
yourself) automatically discovers a compatible LoRA from Civitai /
HuggingFace / your local cache.

Looks are *prescriptive*: you pick a medium by name. Contrast with
`--style` ([STYLES.md](STYLES.md)) which is *detective* — it
analyzes a reference photo and matches against a curated CLIP-H
catalog.

## Quick start

```bash
# Watercolor cottage on SD 1.5 — Civitai discovery on first run.
plakat generate --model sd15 --look watercolor "a cottage in the woods"

# Anime watercolor (looks + genres compose).
plakat generate --model sdxl --look watercolor --genre anime "a knight"

# Air-gapped: only the discovery cache + local LoRA scan are consulted.
plakat generate --model sd15 --look watercolor --offline "a cottage"
```

`--look` is available on every prompt-driven subcommand:
`generate`, `portrait`, `img2img`, `inpaint` (via `img2img --mask`),
and `outpaint`. `upscale` is image-filter only and doesn't take
a look.

## The 8 bundled looks

| Name | Medium | Steps | CFG | Scheduler |
|---|---|--:|--:|---|
| `ink-wash` | East Asian sumi-e brushwork | 32 | 6.5 | `dpmpp-2m` |
| `watercolor` | Transparent washes, paper texture | 32 | 6.0 | `dpmpp-2m` |
| `oil-painting` | Impasto brushwork, canvas | 40 | 7.0 | `dpmpp-2m` |
| `charcoal` | Monochrome smudged shading | 30 | 6.5 | `euler-a` |
| `pencil` | Fine graphite, hatching | 28 | 6.5 | `euler-a` |
| `chalk-pastel` | Soft pastels, vibrant matte | 32 | 6.0 | `dpmpp-2m` |
| `linocut` | Bold carved lines, high contrast | 28 | 7.5 | `euler-a` |
| `gouache` | Opaque matte color, illustrative | 32 | 6.5 | `dpmpp-2m` |

Run `plakat run -e 'plakat.look.list'` from a Bund script to enumerate
them programmatically.

## How it works

1. **Apply the preset.** The look's `prompt_prefix` prepends to your
   prompt; `prompt_suffix` appends; `negative_extras` joins onto your
   `--negative`. Sampler / steps / guidance fill in **only when you
   didn't pass them explicitly** — your CLI flags always win.

2. **Auto-LoRA discovery.** When `--lora` is empty AND the look has
   a `lora_query`, plakat searches for a compatible LoRA in this
   order:

   1. **Disk cache** at `$PLAKAT_CACHE_DIR/look-discovery/` keyed by
      `(look_name, base_model)`. Hit → skip the network.
   2. **Civitai** (`https://civitai.com/api/v1/models`) — search by
      the look's tags + keywords, filter by base-model compatibility,
      pick the first non-NSFW result. The discovered LoRA is
      downloaded via the existing `civitai::download` cache.
   3. **HuggingFace Hub** — `https://huggingface.co/api/models?search=Q+lora`,
      filter by repo-id / tag pattern against the base model.
   4. **Local-cache scan** — walk `$PLAKAT_CACHE_DIR/civitai/` for
      already-downloaded LoRAs whose metadata.json matches the
      look's keywords.

3. **Trigger words injected.** If the discovered LoRA exposes
   `trained_words` (Civitai standard), they're prepended to your
   prompt via the dedup-aware `style::prepend_trigger` helper —
   the same machinery `--style` uses.

4. **Generate.** Pipeline runs with the modified prompt, negative,
   sampler, and LoRA stack.

## Override semantics

Looks are **suggestions**, not overrides. The rule:

| Field | Semantics |
|---|---|
| `steps` / `guidance` / `scheduler` | Override-only-if-user-didn't-pass. Your explicit `--steps 50` wins over the look's `32`. |
| `prompt_prefix` / `prompt_suffix` | **Always applied** — they compose your prompt rather than replace it. |
| `negative_extras` | Always appended to your `--negative` (comma-joined). |
| `lora_query` | Discovery fires **only when `--lora` is empty**. User-supplied LoRAs always win. |

Same rule the v0.14 `--fast` distillation presets follow.

## User-extension catalog

Drop a JSON file under `$CONFIG_DIR/looks/` to add your own:

```text
Linux:   ~/.config/plakat/looks/cyberpunk.json
macOS:   ~/Library/Application Support/ai.plakat.plakat/looks/cyberpunk.json
Windows: %APPDATA%\plakat\plakat\config\looks\cyberpunk.json
```

**One file per look.** The filename stem (`cyberpunk`) is the catalog
key — that's what the user passes to `--look`. The `name` field
inside the JSON must match; mismatches log a warning and the stem
wins.

### File shape

```jsonc
{
  "name":            "cyberpunk",       // must match filename stem
  "display_name":    "Cyberpunk",
  "description":     "Neon-lit cityscapes ...",

  // Compositional fields (always applied):
  "prompt_prefix":   "cyberpunk illustration, neon lighting, dystopian cityscape",
  "prompt_suffix":   ", holographic UI overlays, rain-slick streets",
  "negative_extras": "pastoral, daylight, soft watercolor",

  // Override-only fields (apply only when user didn't pass them):
  "scheduler_hint":  "dpmpp-2m",
  "steps":           30,
  "guidance":        7.0,

  // Discovery query (only when --lora is empty):
  "lora_query": {
    "tags":     ["cyberpunk"],          // exact-match Civitai tags
    "keywords": ["cyberpunk", "neon"]   // fuzzy-search free text
  },

  // null = compatible with all base models; otherwise an allow-list:
  // ["sd15", "sd21", "sdxl", "flux", "sd3"]
  "base_compat": null
}
```

All fields except `name` / `display_name` / `description` are
optional (`null` disables that piece). A minimal user look that just
adds a prompt prefix:

```json
{
  "name": "noir",
  "display_name": "Noir",
  "description": "High-contrast 1940s film",
  "prompt_prefix": "noir film still, high contrast black and white"
}
```

### User entries shadow bundled

If you create `~/.config/plakat/looks/watercolor.json`, your version
fully replaces the bundled `watercolor`. Useful for tightening the
prefix / pinning a specific LoRA / dropping a `lora_query` so
discovery doesn't fire.

### Validation

Files are loaded best-effort: bad JSON, unsafe filenames, or
non-matching `name` fields are logged via `tracing::warn` and
skipped — the bundled catalog still loads. Failures don't block
generation.

## Compositions

### Look + genre

```bash
plakat generate --model sdxl --look watercolor --genre anime \
    "a knight in a forest"
```

Both axes' prompt prefixes/suffixes/negatives compose. Sampler
fields follow the override-only rule with `--look` applied first;
the genre fills only what the look left unset.

### Look + style

```bash
# Discovery + your own LoRA: user LoRA wins, discovery skips.
plakat generate --model sdxl --look watercolor \
    --lora my-favorite-watercolor.safetensors:0.7 \
    "a cottage"
```

The look's prompt prefix / sampler still apply; auto-discovery is
gated off because `--lora` is non-empty.

### Look + fast preset

```bash
# Distillation (--fast) authority over step count wins; the look's
# prompt prefix + negative still apply.
plakat generate --model flux-dev --fast hyper-8 --look oil-painting \
    "a still life"
```

`--fast` runs first and sets `steps=8`; the look sees a non-default
step count and doesn't override it.

## Offline mode

`--offline` short-circuits Civitai + HF Hub. Only the on-disk
discovery cache and the local-cache scan run:

```bash
plakat generate --model sd15 --look watercolor --offline "a cottage"
```

Useful for CI, reproducibility, and air-gapped environments. First
run still needs to be online (cache miss → network); subsequent
runs work offline.

## In scenarios

```hjson
{
    model: sdxl
    out: ./out
    look: watercolor      # scenario-level
    genre: anime
    offline: false

    tasks: [
        {
            name: cottage
            prompt: "a stone cottage"
            # inherits look=watercolor + genre=anime
        }
        {
            name: knight
            prompt: "a knight"
            look: oil-painting   # task overrides scenario
        }
    ]
}
```

Per-task `look:` / `genre:` / `offline:` override the
scenario-level setting. Scenario-mode auto-LoRA discovery is
**deferred to v0.26** — `lora_query` is ignored in scenarios for
now. Supply `loras:` explicitly at the scenario or task level if
you want a specific LoRA.

## In Bund scripts

```bund
"sd15" plakat.load
"watercolor" plakat.look.apply
"anime"      plakat.genre.apply
"true"  "offline_discovery" plakat.config.set
"a cottage at dawn" plakat.generate
"out.png" plakat.save
```

Three host-word namespaces:

| Word | Stack | Description |
|---|---|---|
| `plakat.look.apply` | `( name -- )` | Set the active look |
| `plakat.look.clear` | `( -- )` | Forget the look |
| `plakat.look.list` | `( -- l_1 ... l_n n )` | Push every catalog name + count |
| `plakat.genre.apply` | `( name -- )` | Set the active genre |
| `plakat.genre.clear` | `( -- )` | Forget the genre |
| `plakat.genre.list` | `( -- g_1 ... g_n n )` | Push every genre + count |

The `offline_discovery` config key (`plakat.config.set
"offline_discovery" "true"`) mirrors the CLI `--offline` flag.

Bund script generate-time apply currently fires on the SD-family
path only; Flux / SD3 paths set state correctly but the apply
runs at the CLI level instead — use the CLI on those families for
v0.25.

## See also

- [`STYLES.md`](STYLES.md) — `--style` (detection-flavored, CLIP-H matching)
- [`GENRES.md`](GENRES.md) — `--genre` (subject-domain axis)
- [`SCRIPTING.md`](SCRIPTING.md) — full Bund host-word reference
- [`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md) — design rationale + locked decisions
