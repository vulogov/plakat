# Using genres — subject-domain presets

This tutorial covers plakat's `--genre` flag (v0.25). Where
[`--look`](LOOKS_TUTORIAL.md) chooses the **medium** (watercolor,
oil, pencil, ...), `--genre` chooses the **subject domain** (anime
— and any genres you add yourself). The two axes are independent
and compose: `--look watercolor --genre anime` renders a knight as
a watercolor anime illustration.

## What you'll learn

- Why genres are split out of `--look`
- How to use the bundled `anime` genre across SD 1.5 / SDXL / Pony
  / Illustrious / Flux
- Composing genres with looks (medium + subject)
- Adding your own genres (photoreal, fantasy, cyberpunk, ...)
- Using genres in scenarios + Bund scripts

## Before you start

- Work through [`LOOKS_TUTORIAL.md`](LOOKS_TUTORIAL.md) first. The
  genre axis mirrors the look axis exactly — every concept you
  learn here (override semantics, discovery, user-extension dirs,
  catalogs) is the same.

---

## 1. Why a second axis?

Looks describe **how the image is rendered**: paint vs. ink vs.
pencil. Genres describe **what kind of subject you're rendering**:
anime characters vs. photoreal portraits vs. cyberpunk cityscapes.
They're orthogonal — every combination makes sense:

- watercolor + anime = a watercolor anime illustration
- oil-painting + cyberpunk = a painterly cyberpunk scene
- pencil + photoreal = a graphite-rendered photographic study

If `--look` and `--genre` were one axis, you'd have to pick "anime
OR watercolor" — folding both into a "style" word would lose
the composition. The split lets you mix them freely.

Mechanically, both axes share the same `PresetSpec` shape (same
JSON fields, same discovery flow, same override rules). The only
practical difference is the catalog they live in.

---

## 2. The bundled genre — anime

v0.25 ships **`anime`** as the only built-in genre. The bundled
catalog is intentionally minimal; the genre axis is validated with
one curated entry, while the user-extension directory is wired so
you can add more without waiting on plakat releases.

```bash
plakat generate --model sdxl --genre anime "a knight in a forest"
```

What `--genre anime` does:

- Prepends `"anime illustration, cel-shaded, clean line art"` to
  your prompt
- Appends `", anime aesthetic, expressive style, stylized
  proportions"`
- Adds `"photographic, realistic skin texture, photorealistic,
  3d render, hyperrealism"` to your negative
- Suggests steps=24, guidance=8.0, scheduler `euler-a` (all
  override-only — your explicit flags win)
- Auto-discovers a compatible anime LoRA from Civitai → HF Hub
  → local cache (when `--lora` is empty)

The discovery is what makes this work across the anime LoRA
universe (SDXL: Pony / Illustrious / NoobAI / Animagine; SD 1.5:
Anything-v5 derivatives; Flux: various anime fine-tunes). plakat
filters discovered LoRAs by `BaseFamily` of your loaded model.

---

## 3. Composing with `--look`

The most common pairing — pick a medium + a subject:

```bash
# Watercolor anime knight
plakat generate --model sdxl --look watercolor --genre anime "a knight"

# Ink-wash anime forest
plakat generate --model sdxl --look ink-wash --genre anime "a forest temple"

# Oil-painting anime portrait
plakat generate --model sdxl --look oil-painting --genre anime --photo ./alice.jpg \
    "alice in a kimono"  # plakat portrait
```

Order of application:
1. **Look applies first** — its prompt prefix + suffix +
   `negative_extras` lands; its sampler / steps / guidance fill
   `Option<>` slots; its `lora_query` drives discovery if `--lora`
   is empty.
2. **Genre applies second** — its prompt prefix + suffix +
   `negative_extras` stacks additively on top; sampler fields see
   `Some(_)` from the look and skip (override-only rule).

So in `--look watercolor --genre anime "a knight"`:

```text
prompt   = "anime illustration, cel-shaded, clean line art,
            watercolor painting, soft transparent washes,
            a knight,
            on cold-pressed paper, traditional watercolor techniques,
            anime aesthetic, expressive style, stylized proportions"

negative = "<your --negative>, photographic, smooth digital gradients,
            oil painting, photographic, realistic skin texture, ..."

steps    = 32 (watercolor)        ← look's value, since user passed nothing
guidance = 6.0 (watercolor)       ← look's value
scheduler= dpmpp-2m (watercolor)  ← look's value
```

If you only want one axis, just pass that one — they're optional
and independent.

---

## 4. Across every prompt-driven subcommand

Same surface as `--look`:

```bash
plakat generate --model sdxl --genre anime "a knight"
plakat portrait --model sdxl --genre anime --photo ./alice.jpg "alice as a samurai"
plakat img2img  --model sdxl --genre anime --prompt "as anime" ./photo.png
plakat outpaint --model sdxl --genre anime --expand 256 \
                --prompt "extend the scene" ./photo.png
```

---

## 5. Adding your own genres

Drop a JSON file under `$CONFIG_DIR/genres/`:

```text
Linux:   ~/.config/plakat/genres/cyberpunk.json
macOS:   ~/Library/Application Support/ai.plakat.plakat/genres/cyberpunk.json
Windows: %APPDATA%\plakat\plakat\config\genres\cyberpunk.json
```

Same `PresetSpec` shape as looks (see
[`LOOKS_TUTORIAL.md`](LOOKS_TUTORIAL.md) §10 for the field
reference). The filename stem is the catalog key.

### Example: photoreal

```jsonc
{
  "name": "photoreal",
  "display_name": "Photorealistic",
  "description": "Photographic realism, lens characteristics, lighting physicality",

  "prompt_prefix": "photograph, photorealistic, 35mm film",
  "prompt_suffix": ", natural lighting, sharp focus, professional photography",
  "negative_extras": "painting, illustration, cartoon, 3d render, drawing, sketch",

  "scheduler_hint": "dpmpp-2m",
  "steps": 30,
  "guidance": 6.5,

  "lora_query": {
    "tags": ["photoreal", "photography"],
    "keywords": ["photoreal", "photograph", "realistic", "photography"]
  },
  "base_compat": null
}
```

### Example: fantasy

```jsonc
{
  "name": "fantasy",
  "display_name": "Fantasy",
  "description": "High-fantasy concept art, sword & sorcery",

  "prompt_prefix": "fantasy concept art, dramatic lighting, detailed",
  "prompt_suffix": ", epic composition, painterly",
  "negative_extras": "modern, urban, photographic, contemporary",

  "scheduler_hint": "dpmpp-2m",
  "steps": 35,
  "guidance": 7.5,

  "lora_query": {
    "tags": ["fantasy", "concept-art"],
    "keywords": ["fantasy", "concept art", "sword and sorcery"]
  },
  "base_compat": null
}
```

### Example: cyberpunk

```jsonc
{
  "name": "cyberpunk",
  "display_name": "Cyberpunk",
  "description": "Neon-lit cityscapes, dystopian future",

  "prompt_prefix": "cyberpunk, neon lighting, dystopian cityscape",
  "prompt_suffix": ", holographic UI, rain-slick streets",
  "negative_extras": "pastoral, daylight, medieval, rural",

  "scheduler_hint": "dpmpp-2m",
  "steps": 32,
  "guidance": 7.0,

  "lora_query": {
    "tags": ["cyberpunk"],
    "keywords": ["cyberpunk", "neon", "dystopian", "future"]
  },
  "base_compat": null
}
```

Then use them:

```bash
plakat generate --model sdxl --genre photoreal "a violinist on stage"
plakat generate --model sdxl --genre fantasy --look oil-painting "a wizard's tower"
plakat generate --model sdxl --genre cyberpunk --look watercolor "a street market"
```

Verify your genre loaded:

```bash
plakat run -e 'plakat.genre.list'
```

The list should include your new entry alongside the bundled
`anime`. Bad JSON / unsafe filenames are logged via `tracing::warn`
and skipped — the bundled catalog still loads.

---

## 6. In scenarios

`genre:` at both the scenario and task level, with task overriding
scenario:

```hjson
{
    model: sdxl
    out: ./out
    look: watercolor
    genre: anime              # default for every task

    enhancer: deepseek
    scene: [ { name: morning, prompt: "in the morning" } ]
    weather: [ { name: sunny, prompt: "sunny day" } ]

    tasks: [
        {
            name: knight
            scene: morning
            weather: sunny
            prompt: "a knight"
            # inherits genre=anime
        }
        {
            name: temple
            scene: morning
            weather: sunny
            prompt: "a temple"
            genre: fantasy    # overrides for this task only
        }
    ]
}
```

Dry-run preview:

```text
▶ [1/2] knight
  (dry-run) presets: look=watercolor, genre=anime

▶ [2/2] temple
  (dry-run) presets: look=watercolor, genre=fantasy
```

Same scenario limitation as looks: auto-LoRA discovery is **not**
wired in scenario mode for v0.25 (the prompt prefix + sampler hints
still apply; `lora_query` is ignored). Supply `loras:` explicitly
at scenario or task level. Deferred to v0.26.

---

## 7. In Bund scripts

```bund
"sdxl" plakat.load
"watercolor" plakat.look.apply
"anime"      plakat.genre.apply
"a knight at dawn" plakat.generate
"knight.png" plakat.save
```

Three host words mirroring the look namespace:

| Word | Stack effect |
|---|---|
| `plakat.genre.apply` | `( name -- )` set the active genre |
| `plakat.genre.clear` | `( -- )` forget the genre |
| `plakat.genre.list` | `( -- g_1 ... g_n n )` push every genre + count |

Same Bund limitation as looks: the generate-time apply currently
fires on the SD-family `plakat.generate` path only. Flux + SD3
set the state correctly but apply at the CLI level. For those
families, use `plakat generate --genre anime` directly.

---

## 8. Should I make this a look or a genre?

When you're authoring your own preset, the question comes up:
medium or subject domain?

| It's a look if... | It's a genre if... |
|---|---|
| It describes **how** the image is rendered (texture, surface, mark-making) | It describes **what** the image is of (subject domain, cultural register) |
| You'd describe it with a noun like "watercolor", "etching", "stained glass" | You'd describe it with a noun like "anime", "fantasy", "noir", "vintage advertising" |
| It pairs naturally with another subject domain | It pairs naturally with another medium |
| Pencil + watercolor doesn't make sense (pick one medium) | Anime + fantasy doesn't make sense (pick one genre) |

Borderline cases:
- **"Manga"** is a genre (subject domain — Japanese comics)
- **"Ukiyo-e"** is a look (medium — woodblock printing)
- **"Pixel art"** is a look (rendering technique)
- **"Cottagecore"** is a genre (cultural register)

When in doubt, ask "does this compose with watercolor?" If yes,
it's a genre. If you'd never pair them, it's probably a competing
look.

---

## 9. Limitations and roadmap

- **Bundled-`anime`-only** for v0.25. We deliberately ship one
  curated genre and rely on the user-extension directory for the
  rest. The reason: anime has a deep LoRA ecosystem and well-tuned
  sampler conventions; other genres are subject to author taste.
- **Bund Flux/SD3** support is deferred — set state via
  `plakat.genre.apply`, but use the CLI to actually apply the
  preset on those families until v0.26.
- **Scenario auto-discovery** is deferred — the scenario LoRA
  pipeline needs careful integration.

---

## See also

- [`LOOKS_TUTORIAL.md`](LOOKS_TUTORIAL.md) — the companion tutorial
  for art-medium presets
- [`Documentation/GENRES.md`](../GENRES.md) — flag reference
- [`Documentation/LOOKS.md`](../LOOKS.md) — flag reference for the
  shared field shape
- [`SCRIPTING_TUTORIAL.md`](SCRIPTING_TUTORIAL.md) — Bund language
  (§12 covers what's new in v0.25)
- [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md) — scenario
  basics
