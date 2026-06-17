# plakat — Stability & frozen contracts (1.0)

From **1.0.0**, plakat follows SemVer. The contracts below are **frozen**:
breaking changes to them are a major-version event. Everything outside them
(internal Rust APIs, rendering numerics, model weight sources) may change in a
minor release.

Adding an **optional** flag / scenario key / Bund word is *non-breaking*.
Removing or renaming one is *breaking*.

---

## Frozen contract #1 — CLI surface

The subcommands and their flags are plakat's primary interface. The full rules
(flag families, the positional-vs-`--in` input convention, the public/internal
env-var split) live in [`CLI_CONVENTIONS.md`](CLI_CONVENTIONS.md).

The 0.47 flag set is the 1.0 surface. The pre-1.0 renames are final:
`--lora` (was `--loras`), `--preset` (was `--for`), `--asset-type` (was
`--type`), `--flux-quant-level` (was `--quant-level`). Common flags
(`steps`/`seed`/`model`/`out`/`negative`/`guidance`/`size`) are uniform across
subcommands; the `control-*` / `hires-*` / `enhance-*` / `adetailer-*` /
`artefact-*` / `grid-*` / `tile-*` / `window-*` / `motion-lora*` families are
stable.

**Input convention (frozen, deliberate exception):** `generate` takes a
positional *prompt*; `img2img` / `outpaint` take a positional *image* (the
artifact you are creatively continuing); the mechanical post-process tools
`stylize` / `transparent` / `upscale` take their source image via `--in` (an
operand, not a creative seed). This split is intentional and frozen.

## Frozen contract #2 — Scenario HJSON schema

The keys accepted by `plakat scenario <file.hjson>` are frozen (defined by the
`Scenario` / `Task` structs in `src/cli/scenario.rs`). Stable key set:

- **Run:** `model`, `device`, `size`, `steps`, `guidance`, `seed`, `count`,
  `scheduler`, `out`, `negative`, `enhancer`/`enhance`.
- **Axes & tasks:** `scene[]`, `weather[]`, `tasks[]`, `prompt-header`,
  `prompt-footer`, `name`, `prompt`, `regions`.
- **Adapters:** `lora-header`, `lora-footer`, `lora-scale`, `controls`,
  `control-*`, `style-catalog`, `style-ref`, `style-strength`.
- **Transforms:** `init-image`, `mask-feather`, `mask-invert`, `refine-strength`,
  `refiner-frac`, `redux-images`, `concept-image`, `kontext-bucket`.
- **Artefacts / motion / faces:** `artefact-library`, `artefact-blend`,
  `artefact-blend-strength`, `smart-zones`, `frames`, `format`, `gif-delay-ms`,
  `motion-lora`, `motion-lora-scale`, `window-size`, `window-overlap`,
  `face-bbox`, `face-landmarks`, `face-strength`.
- **Flux quant:** `flux-quant-level`, `quantize-t5`, `t5-quant-level`.

## Frozen contract #3 — Bund scripting word-set

The ~50 host words in the `plakat.*` namespace are frozen (registered in
`src/scripting/words/`; documented in [`SCRIPTING.md`](SCRIPTING.md)). Stable
namespaces: `plakat.load` / `generate` / `save` / `echo`; `plakat.lora.*`,
`plakat.controlnet.*`, `plakat.embedding.*`, `plakat.look.*`, `plakat.genre.*`,
`plakat.style.*`, `plakat.artefact.*`, `plakat.portrait*`; `plakat.animate`,
`plakat.inpaint`, `plakat.outpaint`, `plakat.upscale`, `plakat.stylize`,
`plakat.enhance`; `plakat.cascade` / `pixart` / `pixart`; `plakat.hires.*`,
`plakat.tiled.*`, `plakat.refiner.*`, `plakat.adetailer.*`, `plakat.config.set`,
`plakat.metadata.*`.

## NOT a frozen contract — the Rust library API

**plakat is a CLI.** The `plakat::*` Rust library API is an implementation
detail and may change in any release — it is *not* SemVer-bound. Consume plakat
through one of the three stable interfaces above (CLI, scenario HJSON, Bund
scripting), not by depending on the crate as a library. (This deliberately
avoids binding plakat's large internal surface — 7 model families, dozens of
pipelines — under SemVer.)

## Explicitly NOT frozen

- Internal Rust modules, functions, and types.
- **Model weight sources** — HF repos may be repointed without notice (e.g.
  SD 2.1 → an ungated mirror, EasyNegative → `embed/`); the *capability* is the
  contract, not the source repo.
- **Rendering numerics** — Metal renders are not bit-reproducible; identical
  inputs may differ slightly across runs/backends.
- The proof corpus, the gallery, and the example drivers.
