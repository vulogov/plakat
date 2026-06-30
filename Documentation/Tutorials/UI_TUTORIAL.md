# `plakat ui` — the terminal UI

`plakat ui` is a full-screen terminal application that wraps plakat's
generation engine in an interactive, conversational workflow. Instead
of composing a 30-flag command per image, you load a model once and
then *talk* to it: type a prompt, watch the image appear inline, type
a follow-up to refine it, browse everything you've made, drop in a
specific person, paint an inpaint region, apply a LoRA — all without
leaving the keyboard.

It is built on the same pipelines as the CLI, so anything you make in
the UI is a normal plakat output (PNG + embedded recipe) and anything
you compile here becomes a runnable scenario.

```
plakat ui
```

> **Terminal support.** Inline images use the terminal's graphics
> protocol (Kitty, Ghostty, WezTerm, iTerm2, or any Sixel terminal).
> In a terminal without graphics support the UI still runs — you just
> get placeholders where images would be. `plakat ui` is behind the
> default-on `ui` feature; a lean build (`--no-default-features`) omits
> it.

---

## 1. The workspace

On first run `plakat ui` offers to create a **workspace** — a small
project directory with a `plakat-workspace.hjson` config and folders
for your outputs, scenarios, people, LoRAs, and prompt buffers:

```
my-project/
  plakat-workspace.hjson   # default model, steps, guidance, dirs
  out/                     # everything you generate (+ recipe sidecars)
  scenarios/               # *.hjson batch jobs
  people/                  # <name>/person.hjson identities
  loras/                   # local .safetensors LoRAs
  prompts/                 # .txt / .tera / .hjson prompt buffers
```

The model/LoRA caches are **global** (shared with the CLI), so a model
you've already pulled doesn't download again.

---

## 2. Getting around

Eight screens, switched with **`Ctrl-1`…`Ctrl-8`** or **`Tab`** /
**`Shift-Tab`** (use Tab if your terminal eats `Ctrl-<digit>`):

| # | Screen | What it's for |
|---|--------|---------------|
| 1 | **Chat** | Conversational generation + refinement |
| 2 | **Models** | Load/unload a model; live RAM + swap gauges |
| 3 | **Scenarios** | Browse / edit / run batch HJSON jobs |
| 4 | **History** | Browse everything under `out/`; continue in Chat |
| 5 | **LoRA Hub** | Local + CivitAI + HuggingFace LoRAs |
| 6 | **People** | Identity library; one-key portraits / group scenes |
| 7 | **Prompts** | Prompt Workspace — compile prose → scenario |
| 8 | **Canvas** | Paint an inpaint mask, hand it to Chat |

- **`Ctrl-K`** opens the **command palette** — a fuzzy launcher for every
  action available on the current screen (plus jump-to-any-screen and
  quit). Type to filter, `↑/↓` to move, `Enter` to run, `Esc` to dismiss.
  It works from anywhere, even mid-typing.
- **`Ctrl-Q`** quits from anywhere.
- **`Ctrl-C`** cancels a running generation (or quits if none).
- A shared **Output pane** at the bottom shows live download / denoise
  progress for whatever is running, on every screen.

---

## 3. Models (start here)

Before you can generate, load a model. Press **`Ctrl-2`**, move with
`j`/`k`, and press **`L`** to load (or **`U`** to unload). The download
+ load progress streams into the Output pane; the RAM/Swap gauges at
the top warn you before a load over-commits memory.

The workspace's `default_model` is pre-selected and, if its weights
are already cached, auto-loaded on startup. SD-family models
(sd15 / sd21 / sdxl) are supported in the UI today.

---

## 4. Chat — generate by talking

Press **`Ctrl-1`**. Type a prompt and press **Enter**:

```
> a watercolor fox in a misty forest
```

The image renders inline on the right; the live denoise bar shows in
the Output pane. Generation happens at the model's native resolution
(sd15 = 512², sdxl = 1024²) so it's always Metal-safe.

### Refining — just keep typing

Once you have an image, your **next prompt evolves it**:

```
> a watercolor fox in a misty forest      ▸ (fresh)
> add a red autumn leaf in its mouth      ↻ (refine)
> make the background snowy                ↻ (refine)
```

Each follow-up **accumulates** onto the description and re-renders at
the **same stable seed**, so the composition stays recognizable while
your edits reliably appear (`↻` marks a refine, `▸` a fresh render).
Every step is saved as `plakat-<seed>-1.png`, `-2.png`, … so you never
lose an earlier version.

### Commands

| Command | Effect |
|---------|--------|
| `/new <prompt>` | Start a fresh image (resets the refinement thread) |
| `/negative <text>` | Set a session negative prompt (bare `/negative` clears) |
| `/enhance <prompt>` | AI-expand the prompt (DeepSeek/Gemini/local) then generate |
| `/strength <0.1–1>` | Switch refinement to **image-anchored** (img2img) mode |
| `/strength off` | Back to the default prompt-evolve mode |
| `/seed <n>` \| `/seed random` | Pin a seed for reproducible / comparable runs |
| `/save [name]` | Save the whole Chat session (thread + prompt + seed + base) |
| `/load <name>` | Reload a saved session and keep refining where you left off |
| `/sessions` | List your saved sessions |

`/enhance` uses your configured enhancer (`auto` by default) — set a
`DEEPSEEK_API_KEY` (env or `~/.config/plakat/config.toml`) and it
routes to DeepSeek, exactly like the CLI.

### `@mention` people and LoRAs

Type **`@`** in the prompt to pop up a completion list of your **people**
(`◆`) and local **LoRAs** (`★`); keep typing to filter, `↑/↓` to move,
**`Tab`**/**`Enter`** to accept, `Esc` to dismiss.

- Accepting a **person** leaves a readable `@name` token in the prompt —
  at generation time it expands to that person's prompt fragment (so
  `a portrait of @alice` renders Alice's look).
- Accepting a **LoRA** applies it to Chat right away (the model reloads
  with it merged) and removes the token from your prompt.

### Session filmstrip — scrub, roll back, vary

Once you've made a few images, a **filmstrip** appears under the image
pane — one numbered cell per generated frame this session.

- **`Ctrl-←`** / **`Ctrl-→`** scrub through the frames; the selected one
  shows in the image pane (past the newest returns to the live latest).
- **`Ctrl-B`** **rolls back** to the selected frame — it becomes the live
  base (its prompt + seed recovered), so your next prompt **branches**
  from there instead of the latest.
- **`Ctrl-Y`** makes a **variation** of the selected frame: its prompt
  re-rendered at a new seed.

### Other keys

- **`Ctrl-P`** / **`Ctrl-N`** — recall previous prompts into the editor.
- The prompt box is a 2-line soft-wrapping editor with full cursor
  editing (←/→, Home/End, Up/Down across wrapped rows).

---

## 5. History — find and reuse your work

Press **`Ctrl-4`**. Every PNG under `out/` is listed by date, newest
first. Move with `j`/`k`; the selected image previews on the right
along with its **embedded recipe** (prompt, seed, steps, model).

Press **`C`** to **continue in Chat**: the image loads as an
image-anchored base seeded with its recovered prompt, and you're
dropped into Chat to keep editing it. Because Chat writes the recipe
into every PNG it makes, this round-trips your own chat images too.

### Find, tag, compare, and export

- **`/`** **filters** the list live — type any text and it matches
  against the filename, an image's **tags**, and its **recipe** (prompt,
  seed, steps, model, …). `Enter` keeps the filter, `Esc` clears it.
- **`T`** **tags** the selected image — type a label and press `Enter`.
  Tags are stored in a `<image>.tags` sidecar and shown as `#label` in
  the list; filter by one to gather a collection.
- **`X`** **exports** the current (filtered) set into `out/export/` —
  filter to a tag, then `X`, to build a collection folder.
- **`d`** marks the selected image as a **compare baseline** (`◆`); move
  to another image and press `d` again to see a **recipe diff** (only
  the fields that changed — seed, steps, prompt, …). `d` on the baseline
  again clears it.

---

## 6. People — put a specific person in the picture

Press **`Ctrl-6`**. People are *identities* — reference photos + a
strategy — stored under `people/<name>/person.hjson`. The library also
surfaces any personas defined in your scenario files (read-only,
tagged `◇`).

- Move with `j`/`k`; the primary reference photo previews on the right.
- The detail pane has **six sub-tabs** (cycle with **`←`/`→`** or
  `h`/`l`): **REFS** (weighted photos + angle-coverage guidance),
  **ENCODING** (strategy/mode/quality + on-disk encodings), **PORTFOLIO**
  (generated images + consistency score), **TEST** (the four fixed
  identity test renders), **KNOWN-GOOD** (recorded parameter combos), and
  **SETTINGS** (consent + a privacy audit).
- **`G`** generates a **portrait** of the selected person — it opens in
  Chat ready to refine.
- **`Space`** marks people (`●`); with **two or more marked, `G`**
  generates a **multiperson scene** placing each person in their own
  region.
- **`I`** **imports** a scenario-defined persona (`◇`) into your editable
  `people/` library — it copies the reference photos into
  `people/<name>/refs/` and writes a `person.hjson`, so you can encode,
  re-use, and edit it like any other identity (conflict-aware; an
  existing dir is never overwritten).
- **`Del`** **removes** a `people/` identity — the *right to be
  forgotten*. You must **type the identity's name** to confirm; on a
  match the whole `people/<name>/` directory (refs + encodings) is
  deleted. (Scenario personas are read-only here — edit the scenario.)

---

## 7. LoRA Hub — find, apply, and assess LoRAs

Press **`Ctrl-5`**. Three tabs, switched with **`←`/`→`**:

- **LOCAL** — every `.safetensors` in your workspace + caches. Each
  shows its inferred **family** (read from the safetensors header) and
  a **compatibility marker** (`✓`/`✗`) against the currently-loaded
  model.
  - **`A`** applies a LoRA to Chat — the model reloads with it merged
    in (`★` marks applied; an incompatible LoRA is refused with a note).
    Your next Chat generation uses it.
  - **`R`** asks the LLM for a one-sentence **assessment** of the LoRA.
- **CIVITAI** / **HUGGINGFACE** — type a query, **Enter** to search,
  `j`/`k` to browse, **`D`** to download. Civitai LoRAs land in the
  shared cache; HF LoRAs are copied into your `loras/` dir — either way
  they appear in LOCAL, ready to `A`-apply.
  - **`R`** asks the LLM to **recommend** which result best fits your
    current Chat prompt.

---

## 8. Canvas — paint an inpaint mask

Press **`Ctrl-8`**. The current Chat base image shows on the left; a
coarse cell grid on the right is your **mask** (white = the region to
regenerate).

- Move the cursor with arrows / `hjkl`; **`Space`** toggles a cell;
  **`Shift`+move** paints while moving.
- Preset regions: **`S`** sky, **`B`** background, **`F`** foreground,
  **`L`**/**`R`** halves, **`P`** person column, **`C`** clear.
- **`B` is face-aware**: it masks the background *but punches out any
  detected faces*, so the inpaint regenerates the scene while preserving
  the people. (Face detection runs once per base in the background; if no
  detector is configured it just fills plainly.)
- **`g`** cycles the **grid density** (16×12 → 24×18 → 32×24) for finer
  control — switching density clears the current mask.
- **`Enter`** rasterizes the grid to a full-resolution mask and hands
  it to Chat. Your next prompt **inpaints only the painted region**.

### Outpaint — extend the canvas (`M`)

Press **`M`** for **outpaint mode**: instead of masking *inside* the
image, you grow it. Pick an edge with the arrows (`←/→/↑/↓`), set how
far with `+`/`-` (each band = 128px, up to 4), and press `Enter`. The
Canvas builds a grey-padded, enlarged base with a mask over the new
strip and hands it to Chat — your next prompt **paints the new region**
(e.g. extend a landscape rightward). `M` or `Esc` cancels.

### Finer masks — `g` density, or an external editor

The Canvas is *regional* masking. For more control, press **`g`** to step
up the grid density (up to 32×24). When you need **pixel-precise** edges —
a hairline, a hand, text — paint the mask in any image editor instead:

1. Export or copy the base image you're refining (every Chat image is a
   normal PNG under `out/chat/`).
2. In an external editor (Photoshop, GIMP, Krita, Preview…), paint the
   region to regenerate **pure white** on a **black** background, at the
   image's exact resolution, and save it as a PNG.
3. Run the edit from the CLI with that mask:

   ```bash
   plakat img2img out/chat/plakat-<seed>-1.png \
     --prompt "your edit" \
     --mask my-mask.png --strength 0.85
   ```

White = regenerate, black = keep. This is the same inpaint path the
Canvas drives, just with a hand-authored mask. `plakat img2img --help`
lists the related flags (`--mask-feather`, `--mask-invert`).

---

## 9. Scenarios — batch jobs

Press **`Ctrl-3`**. Browse the `.hjson` files in `scenarios/`:

- **`Enter`** runs the selected scenario; a live **per-task board**
  shows each task go pending → running → ✓/✗ while its progress streams
  to the Output pane.
- **`e`** edits the file in a built-in editor (`Ctrl-S` saves), **`n`**
  starts a new one from a template. The template is **runnable as-is** —
  `n` → `Ctrl-S` → `Enter` generates without any API key.
- **`Ctrl-R`** (in the editor) **names** the scenario — type a file name
  and press `Enter`; it renames `untitled.hjson` (or an existing file) to
  `<name>.hjson`.

### Grab a task from your Chat session (`Ctrl-G`)

Inside the editor, **`Ctrl-G`** turns the conversation you've been having
in Chat into a reusable scenario task. plakat takes the whole refinement
thread — `a fox` → `make it autumn` → `add falling leaves` — and asks the
LLM to **distill it into one coherent prompt**, then inserts a
`{ name: from-chat, prompt: "…" }` block at the cursor:

```
> tasks: [
>   ▏            ← cursor here, press Ctrl-G
> ]
```

becomes

```
> tasks: [
>   {
>     name: from-chat
>     prompt: "a fox in an autumn forest, falling leaves, soft light"
>   }
> ]
```

It runs in the background (the editor stays live); the prompt value is
quoted and escaped so commas and quotes survive. `Ctrl-S` to save. This
is the fast path from *exploring* an image in Chat to *batching* it
(seeds, weather, scene variants) as a scenario.

See [`SCENARIOS_TUTORIAL.md`](SCENARIOS_TUTORIAL.md) for the scenario
format itself.

---

## 10. Prompt Workspace — prose → scenario

Press **`Ctrl-7`**. Write prompts on the left; the **structural
compile** (deterministic, no LLM) updates live on the right, showing
the scenario HJSON your prose produces.

- **`Ctrl-R`** runs the full **LLM compile** (family-aware enhancement
  + auto-negative).
- **`Ctrl-T`** toggles **Tera mode**: the buffer is rendered through the
  Tera template engine *before* compiling. A live **variable panel**
  appears on the right listing every `{{ variable }}` the template reads;
  press **`Ctrl-V`** to jump into it, `↑/↓` to select, and type to set a
  value — the compiled output re-renders as you go. (Needs a build with
  `--features templates`; otherwise the pane shows a recompile hint.)
- **`Ctrl-S`** saves the buffer; **`Ctrl-O`** writes the compiled HJSON
  into `scenarios/` and opens it in the Scenarios editor.
- **`Esc`** jumps to the buffer list (`.txt`/`.tera`/`.hjson`);
  **Enter** loads one.
- **`Ctrl-N`** starts a fresh buffer (name it by saving); **`Ctrl-Tab`**
  cycles through your saved buffers without leaving the editor.

See [`COMPILE_TUTORIAL.md`](COMPILE_TUTORIAL.md) for the prose format.

---

## 11. A full loop

Putting it together, a session might be:

1. **Models** (`Ctrl-2`) → `L` to load `sdxl`.
2. **LoRA Hub** (`Ctrl-5`) → CIVITAI → search "watercolor" → `D`, then
   LOCAL → `A` to apply it.
3. **Chat** (`Ctrl-1`) → "a fox in a forest" → "make it autumn" →
   "add falling leaves".
4. **Canvas** (`Ctrl-8`) → `S` (sky) → `Enter`, back in Chat: "dramatic
   sunset sky".
5. **People** (`Ctrl-6`) → mark two people → `G` for a group scene.
6. **History** (`Ctrl-4`) → find the best frame → `C` to keep refining.

Everything you make is a normal plakat PNG with its recipe embedded —
inspect it with `plakat metadata FILE.png`, or run a compiled scenario
headlessly with `plakat scenario FILE.hjson`.

---

## Notes & limits

- The UI loads **SD-family** (sd15 / sd21 / sdxl), **SD3 / 3.5**, **PixArt-Σ**, and
  **Stable Cascade** models. Only **Flux** remains CLI-only for now. PixArt / Cascade
  have no persistent pipeline, so they reload on each generation (slower) and support
  fresh / prompt-evolve generation but not image-anchored refine or Canvas inpaint.
- People quick-gen and Scenario runs load their **own** model
  alongside any Chat model — on a 24 GB box, unload the Chat model
  first if memory is tight.
- Continuing a portrait in Chat keeps the *look* but not strict face
  identity (Chat refinement is plain img2img, not identity-aware).
