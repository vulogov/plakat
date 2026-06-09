# plakat — CLI conventions (1.0 contract)

The naming rules the CLI follows, frozen for 1.0. New subcommands and flags must
conform; existing ones that deviate are flagged below.

## Input / output

- **The primary input is positional.**
  - Prompt-only commands take the **prompt** positionally:
    `plakat generate "a cat"`, `plakat portrait "a knight"`.
  - Image-transform commands take the **source image** positionally, with the
    prompt as `--prompt`:
    `plakat img2img photo.png --prompt "…"`, `plakat outpaint photo.png`.
- **Secondary / reference inputs are named:** `--ref` (a reference image —
  stylize's style, portrait's identity), `--mask` (the inpaint mask),
  `--control-image` / `--control-from` (ControlNet), `--redux-image` (Flux),
  `--image-variation` (Cascade).
- **Output is always `--out`** (a file for single-output commands; a directory
  for batch/scenario commands). Never `--output`.

### Known deviation (flagged for the freeze)

`stylize`, `transparent`, and `upscale` take their primary image via **`--in`**
rather than positionally. For `stylize` this is defensible (symmetric with
`--ref`); for `transparent` / `upscale` it is historical. **Decision pending:**
unify to positional, or freeze the `--in` exception. Until decided, `--in` is
the named primary-image flag for these three.

## Repeatable lists

Repeatable inputs use a **singular flag, repeated** — not a plural comma-list:

```
--lora a.safetensors --lora b.safetensors    # not --loras a,b
--artefact balloon@sky --artefact pine@middle_plan
--redux-image style.png --redux-image subject.png
```

`--lora` is uniform across `generate` / `portrait` / `img2img` / `outpaint`.

## Common flags (identical everywhere they apply)

`--model` · `--steps` · `--seed` · `--negative` · `--guidance` · `--size` ·
`--device` · `--cache-dir` · `-v/--verbose`. Named once, never re-spelled.

## Flag families (kebab-case, shared prefix)

`control-*` · `hires-*` · `enhance-*` · `adetailer-*` · `artefact-*` · `grid-*` ·
`tile-*` · `window-*` · `motion-lora*` · `ref-*` (stylize). A new knob for an
existing feature joins its family; it does not invent a sibling spelling.

## Environment variables

**Public — supported, documented contract:**

| Var | Effect |
|---|---|
| `PLAKAT_CACHE_DIR` | HuggingFace download cache dir (= `--cache-dir`). |
| `PLAKAT_DEVICE` | Default device (= `--device`). |
| `PLAKAT_TRAIN_SINGLE_FILE` | Style training overwrites one checkpoint file instead of writing numbered ones. |
| `PLAKAT_ALLOW_GGUF_METAL` | Opt into GGUF on Metal (broken upstream; escape hatch). |

**Internal / advanced — may change without notice (not part of the 1.0 contract):**
`PLAKAT_ARCFACE_HF` / `PLAKAT_ARCFACE_WEIGHTS`, `PLAKAT_SCRFD_HF` /
`PLAKAT_SCRFD_WEIGHTS`, `PLAKAT_FACEID_LORA` — portrait identity / face-detector /
FaceID-LoRA model-source overrides.
