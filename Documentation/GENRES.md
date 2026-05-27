# Genres — subject-domain presets (v0.25)

`--genre` is an independent axis from `--look`. Where `--look`
chooses the **medium** (watercolor, oil, charcoal, ...), `--genre`
chooses the **subject domain** (anime — and whatever you add via
user extensions).

The two axes compose: `--look watercolor --genre anime` runs
through both presets, stacking prompt prefixes / suffixes /
negatives and using sampler fields from whichever was applied
first (look wins by default — it's the more specific axis).

## Quick start

```bash
plakat generate --model sdxl --genre anime "a knight in a forest"
plakat generate --model sdxl --look ink-wash --genre anime "a knight"
```

`--genre` is available on every prompt-driven subcommand —
`generate`, `portrait`, `img2img`, `inpaint`, `outpaint` — and in
scenarios + Bund scripts. Mirror of [`--look`](LOOKS.md).

## Bundled genres

v0.25 ships **`anime` only** as a built-in genre. The bundled
catalog at `assets/genres/catalog.json` is intentionally minimal —
the genre axis pattern is validated with one curated entry, while
the user-extension directory is wired so you can add more without
waiting on plakat releases.

```text
anime    steps=24  cfg=8.0  scheduler=euler-a
         tags=[anime, manga, cel-shaded]
```

### Why anime is built-in

Three reasons:
1. **Big LoRA universe.** Civitai + HF Hub host thousands of anime
   LoRAs across SD 1.5 / SDXL / Pony / Illustrious / Flux. Discovery
   has plenty to match against.
2. **Established sampler conventions.** Anime models often ship as
   distillations or favor specific schedulers (`euler-a` is
   community-standard).
3. **Distinct from `--look`.** Anime is a *finetune domain* (it
   pulls SDXL-derivative bases like Pony / Illustrious), not a
   medium — fits the genre axis cleanly.

Other genres (photoreal, fantasy, cyberpunk, ...) belong in your
user catalog rather than the bundled one.

## User-extension catalog

Drop a JSON file under `$CONFIG_DIR/genres/`:

```text
Linux:   ~/.config/plakat/genres/cyberpunk.json
macOS:   ~/Library/Application Support/ai.plakat.plakat/genres/cyberpunk.json
Windows: %APPDATA%\plakat\plakat\config\genres\cyberpunk.json
```

The file shape is **identical to looks** — see
[`LOOKS.md`](LOOKS.md) §"User-extension catalog" for the full
field reference. The only difference is the directory name
(`genres/` vs `looks/`) and the catalog the entry merges into.

Example fantasy genre:

```jsonc
{
  "name": "fantasy",
  "display_name": "Fantasy",
  "description": "High-fantasy concept art, sword & sorcery",
  "prompt_prefix": "fantasy concept art, dramatic lighting, detailed",
  "prompt_suffix": ", epic composition",
  "negative_extras": "modern, urban, photographic",
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

Then:

```bash
plakat generate --model sdxl --genre fantasy "a knight in a temple"
```

## Override semantics — composing with `--look`

When both axes are set, the **look applies first**. Under the
override-only rule, the look fills any `None` sampler fields; the
genre then sees `Some(_)` and skips its sampler suggestions.
Compositional fields from both axes stack:

```text
--look watercolor --genre anime "a knight"
  →  prompt = "anime illustration, cel-shaded, watercolor painting,
                soft transparent washes, a knight, on cold-pressed
                paper, anime aesthetic"
  →  negative = "<your --negative>, photographic, smooth digital
                  gradients, oil painting, photographic, realistic
                  skin texture"
  →  steps = 32  (from watercolor, since user didn't pass --steps)
  →  guidance = 6.0  (from watercolor)
  →  scheduler = dpmpp-2m  (from watercolor)
```

The genre's `lora_query` is only consulted if the look has no
`lora_query` of its own — the look's discovery wins.

## Compositions

### Genre + style detection

```bash
# Detect style from a reference photo + apply genre framing.
plakat generate --model sdxl --style-ref ./ref.jpg --genre anime \
    "a knight"
```

`--style` (CLIP-H detection) and `--genre` are orthogonal — style
chooses a catalog entry's curated LoRA stack; genre adds prompt
prefix + sampler hints. They compose cleanly.

### Genre + fast preset

```bash
# Distillation steps win; genre's prompt prefix still applies.
plakat generate --model flux-dev --fast hyper-8 --genre anime \
    "a knight"
```

## Bund scripting

```bund
"sdxl" plakat.load
"anime" plakat.genre.apply
"a knight at dawn" plakat.generate
"out.png" plakat.save
```

Three host words mirroring the look namespace:

| Word | Stack | Description |
|---|---|---|
| `plakat.genre.apply` | `( name -- )` | Set the active genre |
| `plakat.genre.clear` | `( -- )` | Forget the genre |
| `plakat.genre.list` | `( -- g_1 ... g_n n )` | Push every genre + count |

## In scenarios

```hjson
{
    model: sdxl
    look: watercolor
    genre: anime          # scenario-level

    tasks: [
        { name: knight
          prompt: "a knight" }
        { name: temple
          prompt: "a temple"
          genre: fantasy }   # task overrides scenario
    ]
}
```

Per-task `genre:` overrides the scenario-level setting. Same
resolution as look + offline.

## Why split anime out of `--look`?

Looks are mediums; genres are subject domains. Anime is a finetune
domain, not a medium — it pulls a different model universe (Pony,
Illustrious, NoobAI on SDXL; Animagine on SDXL; etc.) and follows
different sampler conventions. Folding it into `--look` would muddy
the abstraction: "watercolor" and "anime" aren't comparable choices.

Keeping the axes separate lets you say "watercolor anime" — and
have both contributions apply additively.

## See also

- [`LOOKS.md`](LOOKS.md) — `--look` (art-medium axis)
- [`STYLES.md`](STYLES.md) — `--style` (detection-flavored)
- [`RFC_v0.25_LOOKS_AND_GENRES.md`](RFC_v0.25_LOOKS_AND_GENRES.md) — design rationale
