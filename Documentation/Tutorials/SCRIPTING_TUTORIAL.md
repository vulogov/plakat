# Scripting plakat with Bund (v0.21)

`plakat run SCRIPT.bund` runs a small **Bund** script — a stack-based,
Forth-flavoured DSL — against the same pipelines `plakat generate` /
`img2img` / `portrait` / `upscale` use. The point is **composition**:
generate-then-upscale, generate-then-refine-as-img2img, batch a few
variations with shared config, drive the whole thing from one file.

If you've never written Forth, the syntax looks weird for about ten
minutes. After that it reads cleanly. This tutorial gets you from
"never heard of Bund" to writing real scripts.

## Prerequisites

- plakat v0.21 or later (`plakat --version` to check).
- Finished [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md). You should
  know what `--model`, `--steps`, `--guidance`, `--seed` mean.
- A working `plakat generate` against `sd15` or `sdxl` — scripting
  reuses those pipelines unchanged.

No HF token required if you stick with `sd15`. SDXL works too;
Flux + SD3 are deferred to v0.22 (see "Limitations" at the end).

## 1. Your first script

Save this as `hello.bund`:

```bund
// hello.bund — a one-liner that exercises the integration.
"hello from plakat scripting" plakat.echo
```

Run it:

```bash
plakat run hello.bund
```

That's it. `plakat.echo` is a smoke-test word that pushes a tagged
copy of its input back onto the stack — it doesn't produce an
image, just proves the embedding works. Real work starts in §3.

## 2. The syntax in 60 seconds

Bund is **stack-based** (a.k.a. concatenative or RPN). Each token
either:

- **Pushes a value onto the stack** — `42`, `3.5`, `"a fox"`.
- **Pops some values, computes, pushes a result** — `+`, `plakat.generate`.

Top-of-stack is the most recently pushed value. Words pop from the top.

```bund
1 2 +              // push 1, push 2, then `+` pops both and pushes 3
"a fox" "fox.png"  // two strings on the stack — fox.png on top
```

A few one-liners worth memorising:

| Bund | What it does |
|---|---|
| `"text"` | Push a string |
| `42` | Push an integer |
| `3.5` | Push a float |
| `drop` | Pop top, discard |
| `swap` | Swap top two |
| `dup` | Duplicate top |
| `// comment` | Line comment (rest of line ignored) |
| `:name { ... } register` | Define a named lambda |

You don't need most of those yet. `drop` is handy for tidying the
stack when a word pushes a result you don't care about; `swap`
helps when arg order is awkward.

## 3. The seven `plakat.*` words

```text
plakat.load        ( model-alias -- )
plakat.generate    ( prompt -- handle )
plakat.img2img     ( prompt input -- handle )
plakat.portrait    ( prompt photo -- handle )
plakat.upscale     ( handle scale -- handle )
plakat.save        ( handle path -- )
plakat.config.set  ( value key -- )
```

Stack-effect comments use the Forth convention: `( in1 in2 -- out )`
means "pops in1 and in2; pushes out." Top of stack is the rightmost
input (popped first).

### `plakat.load`

Records which model the script should use. Idempotent — calling twice
with the same alias is a no-op. Phase 2 doesn't preload the pipeline,
so the model loads on the first `plakat.generate` call.

```bund
"sd15" plakat.load        // SD 1.5
"sdxl" plakat.load        // SDXL
```

Supported aliases: `sd15`, `sdxl`, `sdxl-turbo`. **Flux, SD3, SD3.5
all bail** with a clear "Phase 2b" pointer — they need additional
plumbing and land in a follow-up cycle.

### `plakat.generate ( prompt -- handle )`

Renders one image. Pushes an **integer handle** that other words use
to reference the result. Handles start at 1 (handle 0 is reserved).

```bund
"sd15" plakat.load
"a fox in a meadow, painterly" plakat.generate
// → stack now has [1]
```

The image lives in an in-memory registry until the script ends.
Multiple `plakat.generate` calls produce handles 2, 3, 4, … each
addressing the corresponding image.

### `plakat.save ( handle path -- )`

Writes a handle's image to disk. Relative paths resolve against the
output dir (`--out`, default `./out`). The handle is **not consumed**
— you can save the same image to multiple paths.

```bund
"sd15" plakat.load
"a fox" plakat.generate        // handle 1
"fox.png" plakat.save
"fox-copy.png" plakat.save     // same image, different filename
```

### `plakat.img2img ( prompt input -- handle )`

Re-imagines an existing image. `input` is either a filesystem path
string OR an **integer handle** to a previously-generated image:

```bund
"sd15" plakat.load
"a fox in burnished gold, ornate" "./photo.jpg" plakat.img2img
  "out.png" plakat.save

// Or chain from a handle (no disk round-trip):
"a fox" plakat.generate                        // handle 1
"a fox, painterly oil"  1  plakat.img2img      // handle 2
  "refined.png" plakat.save
```

Strength controls how much of the input survives — see §4.

### `plakat.portrait ( prompt photo -- handle )`

Identity-preserving portrait via IP-Adapter-Plus-Face. Same two
input shapes as `img2img`:

```bund
"sdxl" plakat.load
0.85 "face_strength" plakat.config.set
"a renaissance oil portrait, ornate frame" "./me.jpg" plakat.portrait
  "portrait.png" plakat.save
```

Identity strategy auto-picked from the loaded model: SD 1.5 →
PlusFace, SDXL → PlusFaceSdxl. SD 2.1 bails (no shipped Plus-Face
checkpoint).

Phase 5 MVP: single reference photo only. FaceID + multi-photo
blends + manual face_bbox/landmarks all land in v0.22.

### `plakat.upscale ( handle scale -- handle )`

Lanczos-3 resize at x2 or x4. The source handle stays — `upscale`
pushes a new handle.

```bund
"sdxl" plakat.load
"a fox" plakat.generate         // handle 1, 1024x1024
  2 plakat.upscale              // handle 2, 2048x2048
  "fox-2k.png" plakat.save
1 4 plakat.upscale              // handle 3, 4096x4096 from source
  "fox-4k.png" plakat.save
```

ML upscaling (Real-ESRGAN) is deferred to v0.22.

### `plakat.config.set ( value key -- )`

Tunes one knob. Settings persist across calls within one script.
Available keys:

| Key | Type | Notes |
|---|---|---|
| `steps` | int | Denoise steps (> 0) |
| `guidance` | float | CFG (finite) |
| `seed` | int | Seed (≥ 0); pins reproducibility across calls |
| `width` | int | Output width (multiple of 8, ≤ 4096) |
| `height` | int | Output height (same) |
| `negative` | string | Negative prompt |
| `scheduler` | string | One of `default \| ddim \| euler-a \| euler \| heun \| unipc \| dpmpp-2m \| unipc-exp \| lcm \| ddpm` |
| `strength` | float | img2img denoise strength `[0, 1]` (default `0.75`) |
| `face_strength` | float | portrait IP-Adapter scale `[0, 1]` (default `0.8`) |

```bund
"sdxl" plakat.load
40    "steps"        plakat.config.set
3.5   "guidance"     plakat.config.set
1024  "width"        plakat.config.set
1024  "height"       plakat.config.set
42    "seed"         plakat.config.set
"blurry, low quality"  "negative"  plakat.config.set
"euler-a"              "scheduler" plakat.config.set

"a fox in a meadow" plakat.generate "fox.png" plakat.save
```

If you don't call `plakat.config.set width/height`, the script
defaults to **per-family** sizes: 512² for SD 1.5 / 2.1, 1024² for
SDXL. Override either dim to pin a custom size.

## 4. Composition patterns

The seven words are designed to chain. Here are the common shapes:

### Generate → upscale → save

```bund
"sdxl" plakat.load
"a renaissance oil painting of a fox" plakat.generate
  2 plakat.upscale
  "fox-2k.png" plakat.save
```

### Generate → refine via img2img → save

```bund
"sd15" plakat.load
0.55 "strength" plakat.config.set      // lighter denoise for refinement
"a fox" plakat.generate                // handle 1, rough draft
"a fox, painterly oil, intricate details, golden hour"
  1  plakat.img2img                    // handle 2, refined
  "fox-refined.png" plakat.save
```

### Multiple variations at the same seed family

```bund
"sd15" plakat.load
42 "seed" plakat.config.set

"a fox in a meadow"     plakat.generate "fox.png"     plakat.save
"a deer in a meadow"    plakat.generate "deer.png"    plakat.save
"a rabbit in a meadow"  plakat.generate "rabbit.png"  plakat.save
```

Pinning the seed means all three share the same initial noise, so
they look like siblings rather than three random renders.

### Portrait from a generated source

```bund
"sd15" plakat.load
"a 19th-century photograph of a man" plakat.generate    // handle 1

"an oil portrait in the style of Sargent"
  1                                                      // reuse
  plakat.portrait                                        // handle 2
  "sargent.png" plakat.save
```

The source handle (1) stays addressable — you can portrait the
same source through multiple prompts without re-generating.

## 5. The REPL

Interactive line editor for the same surface. Persistent state across
lines, so you can build up a session:

```text
$ plakat run --repl
plakat REPL (v0.21). Type .help for commands, .q to exit.
plakat> "sd15" plakat.load
plakat> 50 "steps" plakat.config.set
plakat> "a fox" plakat.generate
=> 1
plakat> "fox.png" plakat.save
plakat> .s
  [0] 1
plakat> .q
```

Three meta commands (start with `.` — Forth convention):

- `.q` / `.quit` — exit
- `.s` / `.stack` — print the workbench bottom-up (`[0]` is the bottom)
- `.help` — list every `plakat.*` word + meta commands

After each successful eval that left a value on top, the REPL echoes
`=> <value>` so you see what just landed. Eval errors print to stderr
and the REPL stays open — you can fix the line and try again.

History persists at `<plakat-config-dir>/repl_history` (use the up
arrow to recall).

## 6. Patterns to lean on

- **Always `plakat.load` first.** Skipping it makes the next image-
  producing word bail with a clear pointer; it's not silently expensive.
- **Set seed for reproducibility.** Pinning `seed` makes every
  `plakat.generate` deterministic — same script, same output forever.
  Without it, plakat picks a random seed per call.
- **Use handles for chains.** `generate → upscale → save` reads
  cleanly via handles; you don't need a temp PNG in between.
- **Comment your scripts.** `//` to end-of-line. Scripts get long
  fast; future-you will want the comments.

## 7. Patterns to avoid

- **Don't mix `plakat.config.set` for the same key twice in a row.**
  The second call silently overwrites the first. Set each knob once;
  override later only with intent.
- **Don't `plakat.save` a handle you intend to keep.** Save doesn't
  free the handle, but if your script later only refers to a handle
  by integer and you've lost track, things get confusing fast. Use
  `.s` in the REPL to keep your bearings.
- **Don't use absolute paths in `plakat.save` unless you have to.**
  Relative paths resolve against `--out`; that makes scripts portable
  across machines without edits.

## 8. Limitations (v0.21)

- **SD-family only** — `sd15`, `sd21`, `sdxl`, `sdxl-turbo`. Flux,
  SD3, SD3.5 all bail at `plakat.load` with a "phase 2b" pointer.
  Adding them is mechanical but adds Flux + SD3 surface that's bigger
  than the v0.21 budget. Lands in a follow-up cycle.
- **SD 2.1 portrait** — bails because there's no shipped IP-Adapter-
  Plus-Face checkpoint for SD 2.1. Use `sd15` or `sdxl` for
  `plakat.portrait`.
- **No pipeline cache.** Every `plakat.generate` / `img2img` / `portrait`
  reloads the model. Scripts that produce many images pay the load
  cost N times. Cache work is deferred to a follow-up; for now, prefer
  fewer-but-larger generations over many small ones.
- **Single reference photo for portraits.** FaceID variants (ArcFace +
  face landmarks) + multi-photo identity blends are v0.22.
- **Lanczos upscale only.** Real-ESRGAN ML upscaling lands in v0.22.
- **`scale` must be 2 or 4** on `plakat.upscale`. No arbitrary scales
  for v0.21 — keeps the surface minimal.
- **No `--lora` / `--control` / `--refiner`.** LoRA stacking,
  ControlNet, and the SDXL refiner aren't yet exposed to scripts.
  Use `plakat generate` directly for those.
- **No scenarios from scripts.** HJSON scenarios already exist
  (`plakat scenario`); duplicating that surface in scripts would be
  redundant. If you need batch generation, use scenarios.

## 9. What's new in v0.22

The v0.21 tutorial above covers the seven foundational words.
v0.22 added 21 more host words across 7 namespaces, a pipeline
cache, all three model families (SD / Flux / SD3), and 50+
Category-B config keys. The full reference is
[`SCRIPTING.md`](../SCRIPTING.md); this section walks through
the patterns you're most likely to need.

### Pipeline cache (phase 1)

`plakat.load "sd15"` now caches the loaded SD pipeline. Calling
`plakat.generate` multiple times on the same alias pays the model
load cost once. Switching aliases drops the cached pipeline.
LoRA / ControlNet mutations also drop the cache so the next
generate rebuilds with the current state.

### All three families (phases 2-3)

```bund
"flux-schnell" plakat.load        // Flux works (with quant D-keys)
"sd35-medium"  plakat.load        // SD3 / SD3.5 work too
```

Family-specific config knobs (Flux quantisation, SD3 tiled
denoise) live in the same `plakat.config.set` surface. See
SCRIPTING.md §"Flux D-keys" and §"Tiled" for the full key list.

### LoRA stack (phase 4)

```bund
"./style-v1.safetensors" 0.7 plakat.lora.add
"civitai:123456"         1.0 plakat.lora.add
0.9 "lora_scale" plakat.config.set     // global multiplier
"sd15" plakat.load
"a knight" plakat.generate
```

`plakat.lora.list` pushes each entry as `"<spec>:<scale>"` plus
the count. `plakat.lora.clear` empties the stack.

### ControlNet stack (phase 5)

```bund
"depth" "./depth.png"        plakat.controlnet.add        // pre-rendered map
"canny" "./photo.jpg"        plakat.controlnet.annotate   // auto-annotate
"depth:0.6:0.0:0.7@image=./d.png" plakat.controlnet.spec  // full grammar
"sd15" plakat.load
"a knight" plakat.generate
```

SD-family ControlNet flows through `Request.controls` per call —
mutating the stack doesn't invalidate the cache. Flux + SD3
ControlNet need load-time setup; the generate paths bail loud
with a v0.23 pointer.

### Post-process toggles (phases 6-9)

```bund
plakat.refiner.enable               // SDXL refiner (v0.23 wiring)
plakat.adetailer.enable             // SCRFD face refinement
plakat.hires.enable                 // upscale + img2img refine
plakat.artefact.blend.enable        // post-composite blend pass
"oak" plakat.artefact.add           // composite an artefact
```

Compose order at generate time: **artefacts → hires →
adetailer**. Hires + artefacts are mutually exclusive (mirrors
the CLI `--hires-fix` vs `--artefact` gate). All four post-
process toggles work cleanly together so long as you don't
combine hires with artefacts.

### Prompt enhancer (phase 10)

```bund
"local" "enhance_provider" plakat.config.set
"true"  "enhance_keep_original" plakat.config.set
"sd15" plakat.load
"a knight" plakat.enhance plakat.generate
```

Pure prompt transformer. The local LLM weight cache is global,
so back-to-back enhance calls pay the GGUF load cost once.
`enhance_keep_original` BREAK-joins on SD-family (no-ops on
Flux/SD3).

### Misc Category-B keys (phase 11)

```bund
"16:9" "aspect"           plakat.config.set
512    "base"             plakat.config.set
"./wildcards" "wildcard_dir" plakat.config.set
"photo" "negative_preset" plakat.config.set
"low quality" "negative"  plakat.config.set
"a knight at __environment__" plakat.generate
```

`aspect` + `base` derive working size when `width`/`height` are
unset. `wildcard_dir` enables `__name__` file wildcards (inline
`{a|b|c}` always works). `negative_preset` combines with
`negative` at request-build time.

## 10. What's new in v0.23

v0.23 closes every "deferred to v0.23" stub v0.22 explicitly took
on, plus adds two new things (`plakat.style.*` namespace + the
`plakat.inpaint` host word). Word count: 28 → 33. Smaller cycle
than v0.22 (~7 phases vs. 12). Full reference is
[`SCRIPTING.md`](../SCRIPTING.md).

### Cache architecture: the SdT2i slot

`plakat.load "sdxl"` now warms a `t2i::Pipeline` slot (not just
`portrait::Pipeline`). `plakat.generate` runs through the t2i
slot, which carries the SDXL refiner UNet hook + the CLIP-skip-
aware encode path. `plakat.img2img` + `plakat.portrait` keep
using `portrait::Pipeline`. Both slots share `Arc<SdCore>`, so
mixed generate+portrait scripts pay one weight load.

### SDXL refiner finally loads

```bund
"sdxl" plakat.load
plakat.refiner.enable             // ~6 GB extra download first time
0.85 "refiner_frac" plakat.config.set
"a knight" plakat.generate        // base UNet 80% → refiner UNet 20%
```

The v0.22 toggle was a state flag with a generate-time bail; in
v0.23 it actually drives `use_refiner` at load time. Non-SDXL
aliases silently downgrade with a warn.

### `clip_skip` wires through

```bund
2 "clip_skip" plakat.config.set   // SD 1.5 anime checkpoints
"sd15" plakat.load
"a fox in tall grass" plakat.generate
```

SDXL warns (penultimate is the design); Flux + SD3 ignore (T5).

### `plakat.style.*` namespace

```bund
"poster-bold" plakat.style.apply       // by id
"./ref.jpg"   plakat.style.detect      // CLIP-H detect from photo
plakat.style.list                      // ( -- ...ids count )
plakat.style.clear
0.7 "style_strength" plakat.config.set
"a town square" plakat.generate        // catalog LoRAs override user LoRAs
```

CLI parity: style LoRAs replace the user LoRA stack for the
load; trigger phrase prepends to the prompt; `negative_extras`
appends to the negative. Subsequent generates with the same
style cache-hit the style-laden pipeline.

### `plakat.inpaint`

```bund
16   "mask_feather" plakat.config.set       // declared v0.22, now firing
"true" "mask_invert" plakat.config.set
"stained glass window in the wall"
   "./photo.png" "./mask.png"
   plakat.inpaint
   "result.png" plakat.save
```

Stack: `( prompt input mask -- handle )`. `input` accepts a
string path or an image handle; `mask` is a string path. SD-family
+ SD3 work end-to-end. Flux inpaint requires the `flux-fill-dev`
variant + channel-concat wiring (not in scope); bail message
points at the CLI workaround.

### Flux + SD3 ControlNet from scripts

```bund
"./depth-map.png" "depth" plakat.controlnet.add
"flux-dev" plakat.load
"a cyberpunk street" plakat.generate
```

CN stack wires into `LoadRequest.controlnets` at load time. Stack
mutations invalidate the Flux/SD3 slot. **Scope cap**: `image=`
specs only (pre-rendered conditioning). `from=` (auto-annotate
via `plakat.controlnet.annotate`) bails on Flux/SD3 — the loader
doesn't know the per-generate dims yet. Pre-render depth/canny/
pose maps and use `.add`.

### Composition order, refined

The SD-family run order:

1. **Style resolve** (if `style_id` / `style_ref` set) → catalog
   LoRAs override user LoRAs; trigger prepends; negative extras
   append.
2. **t2i.generate** with the resolved LoRA stack + refiner gate
   + `clip_skip`.
3. **Artefacts** (compose + optional blend).
4. **Hires fix**. Mutually exclusive with artefacts.
5. **ADetailer** (face refinement at the final resolution).

Order matters: style → generate → artefacts → hires → adetailer.

## 11. What's new in v0.24

v0.24 adds 9 new host words (33 → 42), exposing the last
CLI-only persona and post-process features to scripts. Three
config keys for face-alignment overrides, one new namespace
(`plakat.portrait.photo.*`) for multi-photo identity, plus
namespaces for outpaint, Textual Inversion, style transfer, and
metadata read. The two v0.23 limitations (Flux/SD3 `from=`
auto-annotate, Flux inpaint via `flux-fill-dev`) also close.
Full reference is [`SCRIPTING.md`](../SCRIPTING.md).

### Multi-photo portrait (phase 1) — BREAKING CHANGE

```bund
// v0.23:
"alice.jpg" "a portrait" plakat.portrait

// v0.24:
"alice.jpg" 1.0 plakat.portrait.photo.add
"a portrait" plakat.portrait
```

`plakat.portrait` no longer takes a photo arg. Photos come from
`plakat.portrait.photo.add ( path-or-handle weight -- )`. Weight
`-1.0` means auto-fill (CLI's default); explicit values pass
through. Weights normalise sum-to-1 at request-build time.

Multi-photo identity blends:

```bund
"alice.jpg" 0.7 plakat.portrait.photo.add
"bob.jpg"   0.3 plakat.portrait.photo.add
"a couple at the beach" plakat.portrait
```

### Face alignment overrides (phases 2-3)

```bund
"0.2,0.1,0.8,0.7" "face_bbox" plakat.config.set
"0.40,0.40,0.60,0.40,0.50,0.55,0.42,0.68,0.58,0.68"
   "face_landmarks" plakat.config.set
"face-id" "identity_kind" plakat.config.set
```

CLI parity with `--face-bbox`, `--face-landmarks`, and the four
identity-encoder variants (`plus-face`, `plus-face-sdxl`,
`face-id`, `face-id-sdxl`). Empty string clears any of them.

### `plakat.outpaint` (phase 4)

```bund
"sd15" plakat.load
"wide mountain valley, panorama"
   "./photo.jpg" "left=512,right=512"
   plakat.outpaint
```

Stack: `( prompt input expand-spec -- handle )`. Spec is
`"expand=N"` (all four sides) or `"left=L,right=R,top=T,bottom=B"`.
Builds a replicate-edge canvas + a mask, dispatches to
`plakat.inpaint`.

### `plakat.embedding.*` Textual Inversion (phase 5)

```bund
"./my-ti.safetensors:mytrigger:0.7" plakat.embedding.add
"civitai:99999:concept" plakat.embedding.add
"sdxl" plakat.load
"mytrigger a knight" plakat.generate
```

Spec grammar matches `--embedding`. Effective only on
`plakat.generate`'s SdT2i path; `plakat.img2img` +
`plakat.portrait` silently ignore (CLI parity).

### `plakat.stylize` (phase 6)

```bund
"sd15" plakat.load
0.35 "strength" plakat.config.set
"alice.jpg" "renaissance.jpg" plakat.stylize
   "alice-renaissance.png" plakat.save
```

Stack: `( subject style -- handle )`. No prompt — image-driven.
SD 1.5 only. One-shot load per call (~5 GB).

### `plakat.metadata.read` (phase 7)

```bund
"fox.png" plakat.metadata.read    // ( -- k_1 v_1 ... k_n v_n n )
```

Reads the JSON sidecar. Pushes every populated field as a
`(key, value)` string pair plus a count.

### Flux + SD3 ControlNet `from=` auto-annotate (phase 8)

```bund
"sd35-medium" plakat.load
"depth" "./photo.jpg" plakat.controlnet.annotate
"a knight" plakat.generate    // depth map auto-derived from photo.jpg
```

Annotation runs lazily at first generate using the call's
width/height; the annotated PNG caches on a per-pipeline
tempdir. Same dims → cache hit; dim mismatch re-annotates.

### Flux inpaint via flux-fill-dev (phase 9)

```bund
"flux-fill-dev" plakat.load
"stained glass window in the wall"
   "./photo.png" "./mask.png"
   plakat.inpaint
```

The v0.23 `plakat.inpaint` Flux bail is gone — when the loaded
alias resolves to `FluxFillDev`, the mask threads through
`flux::GenRequest.mask`.

## 12. What's new in v0.25

v0.25 adds the **art-medium** (`--look`) and **subject-domain**
(`--genre`) axes. Both ship as Bund host word namespaces alongside
the CLI flags + scenario fields. Host word count: 42 → 48.

The big new thing is **auto-LoRA discovery**: when you apply a
look and your LoRA stack is empty, plakat searches Civitai → HF
Hub → local cache for a compatible LoRA matched to the loaded
base model and the look's tags/keywords. Trigger words from the
discovered LoRA prepend to the prompt automatically.

### The new namespaces

```bund
// Looks (the medium axis) — 3 words.
"watercolor" plakat.look.apply         // pick a medium
plakat.look.list                       // 8 bundled names + count
plakat.look.clear                      // forget it

// Genres (subject-domain axis) — 3 words, identical shape.
"anime" plakat.genre.apply
plakat.genre.list
plakat.genre.clear

// Discovery on/off — config key.
"true" "offline_discovery" plakat.config.set
```

Bundled looks (8): `ink-wash`, `watercolor`, `oil-painting`,
`charcoal`, `pencil`, `chalk-pastel`, `linocut`, `gouache`.
Bundled genres (1): `anime`.

### Override semantics

Three field buckets, same rules as the CLI:

| Bucket | Rule |
|---|---|
| Compositional (`prompt_prefix`, `prompt_suffix`, `negative_extras`) | **Always applied** — they compose your prompt + negative rather than replace them. |
| Override-only (`steps`, `guidance`, `scheduler_hint`) | Fill `ctx.config` slots only when you left them at defaults. Explicit `plakat.config.set "steps" "50"` wins. |
| Discovery-gating (`lora_query`, `base_compat`) | Discovery fires only when `ctx.loras` is empty. User-supplied LoRAs always win. |

### Composing look + genre

```bund
"sdxl" plakat.load
"watercolor" plakat.look.apply
"anime"      plakat.genre.apply
"a knight in a forest" plakat.generate
"knight.png" plakat.save
```

Both axes' prompt prefixes/suffixes/negatives stack. Sampler
fields follow the override-only rule with the **look applied
first** — the genre fills only what the look left unset.

### User-extension catalogs

Drop a JSON file under `$CONFIG_DIR/looks/` or `$CONFIG_DIR/genres/`:

```text
~/Library/Application Support/ai.plakat.plakat/looks/cyberpunk.json
~/.config/plakat/looks/cyberpunk.json
```

One PresetSpec object per file; filename stem is the catalog key.
See [`LOOKS.md`](../LOOKS.md) for the field reference. User
entries shadow bundled by name.

`plakat.look.list` and `plakat.genre.list` enumerate the merged
catalog (bundled + user).

### Offline discovery

```bund
"true" "offline_discovery" plakat.config.set
"watercolor" plakat.look.apply
"a cottage" plakat.generate
```

Skips Civitai + HF Hub; uses on-disk discovery cache + local
LoRA scan. First-time use still needs network access to populate
the cache; subsequent calls are network-free.

### Scope

Bund-side apply currently fires on the SD-family `plakat.generate`
path only. Flux + SD3 paths set the look/genre state correctly
but don't apply the preset at generate time — for those families,
use the CLI flags (`--look watercolor`) which apply on every
pipeline family. v0.26 will extend the Bund apply to Flux/SD3.

Scenario-mode auto-LoRA discovery is also deferred to v0.26 (the
scenario LoRA pipeline has two stages and needs careful
integration). The prompt prefix / sampler hints still apply in
scenarios — set `loras:` explicitly if you want a specific LoRA.

## 13. The full word reference

```text
plakat.echo        ( s -- s' )       Phase 1 smoke; pushes "[out=...] <s>"
plakat.load        ( alias -- )      Set the model alias for subsequent words
plakat.generate    ( prompt -- h )   Text→image; pushes handle
plakat.img2img     ( prompt in -- h ) Re-imagine; `in` is string path or int handle
plakat.portrait    ( prompt ph -- h ) IP-Adapter portrait; `ph` is path or handle
plakat.upscale     ( h scale -- h' ) Lanczos x2/x4
plakat.save        ( h path -- )     Write to disk (relative → --out)
plakat.config.set  ( val key -- )    Tune one knob (see §3)
```

Stack notation: lowercase letters are values, `h` is a handle, `--`
separates inputs from outputs. Top-of-stack is on the right of each
side. Words on the right of the `--` get pushed in the order shown.

## Where to next

- **`GENERATE_TUTORIAL.md`** — the foundation. Everything `plakat.*`
  words do is also reachable from the CLI; the tutorial is the
  one-stop reference for the underlying surface.
- **`PORTRAIT_TUTORIAL.md`** — identity preservation in depth. The
  IP-Adapter / FaceID surface plakat exposes is much richer than
  what `plakat.portrait` covers in v0.21.
- **`Documentation/SCRIPTING.md`** — the full reference for every
  v0.22 host word + config key. Source of truth when this tutorial
  feels too narrative.
- **`Documentation/RFC_v0.22_BUND_WORDS_EXPANSION.md`** — the v0.22
  design doc covering the 7 namespaces + cache + Category-B keys.
- **`Documentation/RFC_v0.21_BUND_SCRIPTING.md`** — the v0.21
  foundations doc.
- **External Bund material** — `vulogov/Bund` on GitHub has the full
  language reference (lambdas, currying, list ops, control flow).
  Most of it isn't exposed to plakat scripts (RFC decision #2: build
  our own VM, no Bund stdlib), but the syntax + concepts are the
  same.
