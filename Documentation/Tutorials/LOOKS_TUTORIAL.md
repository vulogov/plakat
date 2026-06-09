# Using looks — art-medium presets

This tutorial covers plakat's `--look` flag (v0.25): a one-flag way
to render your prompt in a chosen art medium — watercolor, oil
painting, charcoal, pencil, chalk pastel, linocut, gouache, or ink
wash. Behind the scenes, applying a look composes your prompt,
picks a sampler suited to the medium, and (when you haven't passed
`--lora` yourself) automatically discovers a compatible LoRA from
Civitai / HuggingFace / your local cache.

No prior text-to-image experience assumed.

## What you'll learn

- The difference between a "look" (`--look`, prescriptive) and a
  "style" (`--style`, detective)
- How to render in each of the 8 bundled mediums with one flag
- How auto-LoRA discovery works + when it fires
- The override-only rule: explicit `--steps` / `--guidance` /
  `--scheduler` always win
- How to use looks in scenarios + Bund scripts
- Adding your own custom looks via `$CONFIG_DIR/looks/`
- Composing looks with the `--genre` axis (medium + subject domain)

## Before you start

- Work through `GENERATE_TUTORIAL.md` first.
- First-time use of a look hits Civitai over the network to
  discover a compatible LoRA. Cached afterward — subsequent runs
  with the same look + base model are network-free.
- For air-gapped / reproducibility use, `--offline` short-circuits
  network discovery (uses cache + local LoRA scan only).
- **`--smart-discovery`** (v0.46) improves discovery for generic medium
  terms: a small local LLM judges the Civitai candidate pool and picks the
  best *style* LoRA, **rejecting character LoRAs** (so "watercolour" no
  longer resolves to "an anime girl tagged watercolour"). If nothing fits,
  it falls back to the prompt-only preset. Best paired with `--model sdxl`.

---

## 1. Looks vs. styles

Both `--look` and `--style` end up applying a LoRA + a prompt
prefix, but they're picked differently:

| Feature | `--look NAME` | `--style ID` / `--style-ref PHOTO` |
|---|---|---|
| How you pick | By name (you know what medium you want) | By name OR by CLIP-H matching against a reference photo |
| Catalog source | `assets/looks/catalog.json` (8 mediums) | `assets/style_catalog/` (CLIP-H exemplars + curated per-base LoRAs) |
| LoRA source | Civitai → HF → local-cache **discovery** | Curated catalog entries (manually authored) |
| When to use | "Give me watercolor." | "Give me whatever this photo's style is." |

They compose cleanly — you can `--style watercolor --look watercolor`
and get both the curated catalog LoRA AND the discovered LoRA in
the stack.

---

## 2. The bundled looks

Eight art mediums ship with plakat:

| Name | Best for |
|---|---|
| `ink-wash` | East Asian sumi-e brushwork on rice paper |
| `watercolor` | Soft transparent washes with paper texture |
| `oil-painting` | Thick impasto on canvas, classical painterly |
| `charcoal` | Monochrome smudged shading, dramatic light |
| `pencil` | Fine graphite lines, hatching, sketchbook feel |
| `chalk-pastel` | Soft pastels, vibrant matte pigments |
| `linocut` | Bold carved lines, high contrast, limited palette |
| `gouache` | Opaque matte color, illustrative |

To list them from the CLI:

```bash
plakat generate --help | grep -A2 "\-\-look"
```

Or from a Bund script:

```bash
plakat run -e 'plakat.look.list'
```

---

## 3. Your first look

Render a watercolor cottage on SD 1.5:

```bash
plakat generate --model sd15 --look watercolor "a cottage in the woods"
```

What happens:

1. **Prompt composes.** Your "a cottage in the woods" becomes
   *"watercolor painting, soft transparent washes, a cottage in
   the woods, on cold-pressed paper, traditional watercolor
   techniques, visible paper texture"*. The look's `prompt_prefix`
   prepends; `prompt_suffix` appends.

2. **Sampler suggestions apply.** Steps becomes 32 (from 28),
   guidance 6.0 (from 7.5), scheduler `dpmpp-2m` — all because you
   didn't pass `--steps`, `--guidance`, or `--scheduler`.

3. **Negative composes.** The look's `"photographic, smooth digital
   gradients, oil painting"` appends to your `--negative` (empty
   in this case).

4. **Auto-discovery fires.** Your `--lora` stack is empty, so
   plakat searches Civitai for a watercolor LoRA matching SD 1.5,
   downloads it (cached after first run), and prepends its trigger
   words to the prompt.

5. **Generate.** SD 1.5 runs with the modified prompt + the
   discovered LoRA.

On first run you'll see lines like:

```
  look 'watercolor': prompt/negative composed, steps=32, guidance=6.0,
                     lora-discovery=pending (phase 4)
  discovered LoRA 'Watercolor Style v3' (scale=0.8) for 'watercolor'
                   — https://civitai.com/models/12345
  trigger words prepended: watercolor wash, soft pigment
```

Subsequent runs hit the discovery cache and skip the network.

---

## 4. Override rules — what wins

The look is a **suggestion**, not a replacement. Two kinds of fields:

### Override-only (you win when you pass it explicitly)

`--steps`, `--guidance`, `--scheduler`:

```bash
# Look suggests steps=32; you say 50; you get 50.
plakat generate --model sd15 --look watercolor --steps 50 "a cottage"
```

The override-only rule uses a clap-default comparison — if the
flag matches its built-in default, the look fills in; if it
differs, you win.

### Compositional (always apply)

`prompt_prefix`, `prompt_suffix`, `negative_extras`:

These are additive — they compose your input rather than replace
it. There's no "skip the prefix" flag in v0.25. If you don't want
a look's prefix, don't pass `--look`.

### Discovery (only when `--lora` is empty)

```bash
# Discovery skipped — you passed a LoRA, you win.
plakat generate --model sd15 --look watercolor \
    --lora my-favorite-watercolor.safetensors:0.7 \
    "a cottage"
```

The look's prompt prefix + sampler hints still apply; only the
network discovery is gated off.

---

## 5. Looks across every prompt-driven subcommand

`--look` works on all five prompt-driven subcommands. The semantics
are identical:

```bash
# Generate (text → image)
plakat generate --model sdxl --look oil-painting "a still life"

# Portrait (with identity)
plakat portrait --model sdxl --look ink-wash --photo ./alice.jpg \
    "alice at a tea ceremony"

# Img2img (transform existing)
plakat img2img --model sd15 --look pencil --prompt "as a pencil sketch" \
    ./photo.png

# Inpaint (img2img with mask)
plakat img2img --model sd15 --look watercolor \
    --mask ./mask.png --prompt "watercolor garden" ./photo.png

# Outpaint (extend canvas)
plakat outpaint --model sd15 --look ink-wash --expand 256 \
    --prompt "extend the scene" ./photo.png
```

`upscale` is image-filter only (no prompt, no LoRA) — it doesn't
take a look.

---

## 6. Composing with `--genre`

`--look` chooses the **medium**; `--genre` chooses the **subject
domain**. They're independent axes that compose:

```bash
plakat generate --model sdxl --look watercolor --genre anime \
    "a knight in a forest"
```

The look applies first (medium is more specific); the genre adds
its prompt prefix + negative on top. Sampler fields follow the
override-only rule — the look fills, the genre sees `Some(_)` and
skips its own sampler suggestion.

v0.25 ships `anime` as the only bundled genre. See
[`GENRES_TUTORIAL.md`](GENRES_TUTORIAL.md) for the subject-domain axis
in depth.

---

## 7. Offline mode — repeatable & air-gapped

`--offline` short-circuits Civitai + HF Hub. Only the on-disk
discovery cache and a local-cache scan run:

```bash
plakat generate --model sd15 --look watercolor --offline "a cottage"
```

What happens:
- **Cache hit** → instant LoRA resolution, no network
- **Cache miss + local scan finds a match** → use that, log the
  source
- **Both miss** → log "no compatible LoRA found (offline)" and
  generate without auto-discovery (prompt prefix + sampler hints
  still apply)

Use cases: CI runs, reproducibility (pin a specific LoRA via the
cache), air-gapped environments, deterministic batch jobs.

---

## 8. Looks in scenarios

Scenarios (HJSON batch config) accept `look:` / `genre:` /
`offline:` at both the global and per-task levels. Per-task wins.

```hjson
{
    model: sdxl
    out: ./out
    look: watercolor       # scenario-wide medium
    genre: anime           # scenario-wide subject
    offline: false

    enhancer: deepseek
    scene: [ { name: morning, prompt: "in the morning" } ]
    weather: [ { name: sunny, prompt: "sunny day" } ]

    tasks: [
        {
            name: cottage
            scene: morning
            weather: sunny
            prompt: "a stone cottage"
            # inherits look=watercolor + genre=anime
        }
        {
            name: knight
            scene: morning
            weather: sunny
            prompt: "a knight"
            look: oil-painting      # per-task overrides the scenario
        }
    ]
}
```

`plakat scenario --dry-run my-scenario.hjson` shows the effective
preset per task before running:

```
▶ [1/2] cottage (scene=morning, weather=sunny)
  (dry-run) presets: look=watercolor, genre=anime

▶ [2/2] knight (scene=morning, weather=sunny)
  (dry-run) presets: look=oil-painting, genre=anime
```

**Scenario limitation (v0.25):** auto-LoRA discovery is **not**
wired in scenario mode. The prompt prefix + sampler hints apply,
but the `lora_query` is ignored — `loras:` at scenario or task
level still works as before. Discovery integration in scenarios
is deferred to v0.26.

---

## 9. Looks in Bund scripts

The same surface is available from `plakat run SCRIPT.bund`:

```bund
"sd15" plakat.load

// Pick a look.
"watercolor" plakat.look.apply

// Optional: pick a genre too (independent axis).
"anime" plakat.genre.apply

// Optional: skip network discovery.
"true" "offline_discovery" plakat.config.set

// Generate + save.
"a cottage at dawn" plakat.generate
"cottage.png" plakat.save
```

Six new host words ship in v0.25:

| Word | Stack effect |
|---|---|
| `plakat.look.apply` | `( name -- )` set the active look |
| `plakat.look.clear` | `( -- )` forget the look |
| `plakat.look.list` | `( -- l_1 ... l_n n )` push every catalog name + count |
| `plakat.genre.apply` | `( name -- )` set the active genre |
| `plakat.genre.clear` | `( -- )` forget the genre |
| `plakat.genre.list` | `( -- g_1 ... g_n n )` push every genre + count |

Plus one config key:

```bund
"true" "offline_discovery" plakat.config.set    // mirror of --offline
```

**Bund limitation (v0.25):** the generate-time apply currently
fires on the SD-family `plakat.generate` path only. Flux and SD3
set the state correctly but don't apply the preset in-script.
For those families, use the CLI `--look` flag.

---

## 10. Adding your own look

Drop a JSON file under `$CONFIG_DIR/looks/`:

```text
Linux:   ~/.config/plakat/looks/cyberpunk.json
macOS:   ~/Library/Application Support/ai.plakat.plakat/looks/cyberpunk.json
Windows: %APPDATA%\plakat\plakat\config\looks\cyberpunk.json
```

One PresetSpec object per file. The filename stem is the catalog
key — that's what `--look NAME` matches against. `name` inside the
JSON must match (mismatches log a warning + use the stem).

Minimal example — just a prompt prefix:

```json
{
  "name": "noir",
  "display_name": "Noir",
  "description": "High-contrast 1940s film aesthetic",
  "prompt_prefix": "noir film still, high contrast black and white"
}
```

Then:

```bash
plakat generate --model sdxl --look noir "a detective in an alley"
```

Full example with discovery + sampler hints:

```jsonc
{
  "name": "cyberpunk",
  "display_name": "Cyberpunk",
  "description": "Neon-lit cityscapes, holographic UI, rain on asphalt",

  "prompt_prefix": "cyberpunk illustration, neon lighting, dystopian cityscape",
  "prompt_suffix": ", holographic UI overlays, rain-slick streets",
  "negative_extras": "pastoral, daylight, soft watercolor",

  "scheduler_hint": "dpmpp-2m",
  "steps": 30,
  "guidance": 7.0,

  "lora_query": {
    "tags": ["cyberpunk"],
    "keywords": ["cyberpunk", "neon", "dystopian"]
  },
  "base_compat": null
}
```

Your user-extension entry **shadows the bundled** by name — if
you create `~/.config/plakat/looks/watercolor.json`, your version
fully replaces the bundled `watercolor`. Useful for tightening a
prefix, pinning a specific LoRA, or dropping `lora_query` so
discovery doesn't fire.

To verify your look loaded:

```bash
plakat run -e 'plakat.look.list'
```

The list should include your new entry. Bad JSON / unsafe
filenames are logged via `tracing::warn` and skipped — the
bundled catalog still loads.

---

## 11. Composition matrix

How looks interact with other plakat features:

| Combined with | Behavior |
|---|---|
| `--style` (CLIP-H detection) | Both apply. Style's curated LoRAs land first (filling the LoRA stack), so look's discovery skips. Look's prompt prefix + sampler still apply. |
| `--fast` (distillation preset) | Fast applies first and sets steps (e.g., `hyper-8` → 8 steps). Look's sampler hints see a non-default value and don't override. Look's prompt + negative still compose. |
| `--lora` (explicit LoRA) | Discovery skipped (user wins on the LoRA stack). Look's prompt + sampler still apply. |
| `--negative-preset` (built-in negative) | Both apply: negative-preset's value joins with look's `negative_extras`. |
| `--genre` (subject domain) | Both apply. Look first → sampler / discovery; genre second → additive prompt + negative. |

---

## Common questions

**"Where do the look's prompt prefix + suffix come from?"** Each
look in `assets/looks/catalog.json` has a `prompt_prefix` /
`prompt_suffix` / `negative_extras` field — hand-curated for that
medium. Read the JSON to see them.

**"Can I disable just the auto-discovery and keep the prompt
prefix?"** Two options: pass `--offline` (no network, but local
cache + scan still find a LoRA if cached), or pass `--lora
<anything>` (discovery gates on `args.loras.is_empty()`).

**"The discovery picked a LoRA I don't like."** Cache it out:
```bash
rm $PLAKAT_CACHE_DIR/look-discovery/watercolor__sdxl.json
```
Next run re-discovers. Alternatively, pin your own LoRA with
`--lora` to skip discovery entirely.

**"Civitai is blocked / down."** Use `--offline`. The first run
without network won't have a cache, so discovery returns "no
compatible LoRA found (offline)" — that's fine, the prompt
prefix + sampler still apply. Pre-cache by running once with
network access, or pre-populate the discovery cache directory by
hand.

**"What's the difference between `--look watercolor` and
`--style watercolor`?"** `--style` looks up `watercolor` in the
v0.23 CLIP-H catalog (5 entries, curated per-base LoRAs).
`--look` looks up `watercolor` in the v0.25 medium catalog (8
entries, auto-discovery). They're separate catalogs — same name,
different mechanisms. Both compose; pick one or both.

---

## See also

- [`Documentation/LOOKS.md`](../LOOKS.md) — flag reference + field
  shape details
- [`Documentation/GENRES.md`](../GENRES.md) — the `--genre` axis
- [`GENRES_TUTORIAL.md`](GENRES_TUTORIAL.md) — companion tutorial
  for subject-domain presets
- [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md) — `--style` (the
  detection-flavored sibling)
- [`SCRIPTING_TUTORIAL.md`](SCRIPTING_TUTORIAL.md) — Bund language
  fundamentals (§12 covers what's new in v0.25)
- [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md) — scenario
  fundamentals
