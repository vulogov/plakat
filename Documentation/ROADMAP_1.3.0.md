# plakat 1.3.0 — roadmap

1.2.0 shipped **`plakat compile`** (Track C, COMPILE-1): prose `prompts.txt` →
scenario HJSON, deterministic core + LLM enhancement + `--lint`/`--diff`/
`--decompile`/`--compile-cache`. 1.3.0 is **COMPILE-2**: the optional Tera template
pre-pass — the second half of the compile track (see
[`RFC_MAP_COMPILE_PLAN.md`](RFC_MAP_COMPILE_PLAN.md)). All SemVer-additive.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## 1.3.0 — `compile` Tera pre-pass (Track C, COMPILE-2)

A Tera render pass that fires *before* the `prompts.txt` parser, gated behind the
`templates` Cargo feature. `.tera`/`.j2`/`.jinja` inputs (or `--template`) render
to a `prompts.txt` string, then flow through the existing compile pipeline
unchanged.

- [x] **Feature gate** — `templates = ["dep:tera"]` (serde_json/toml already deps);
      `template_stub.rs` errors clearly when absent. Default binary unchanged.
- [x] **Render pass** (`src/compile/template.rs`) — Tera context: built-in `plakat.*`
      < `--vars <json|toml>` < `--vars-env PREFIX` < `--var KEY=VALUE`. TOML datetimes
      serialize to strings (**OQ-TEMPLATE-3**).
- [x] **Custom filters/functions** — `scene_name`, `prompt_join`, `prompt_clean`,
      `zero_pad`, `sentence_case`; `include_raw` (verbatim — **OQ-TEMPLATE-1** taken as
      the explicit name), `scene_separator`, `model_family`. `{% include %}`/`{% import %}`
      resolve sibling template files.
- [x] **Diagnostics** — template parse/render errors surface the template name + the
      Tera error/source chain. `--dump-rendered` / `--dump-rendered-only`.
- [x] **Corpus gate MET:** `corpus/compile/series.tera` + `--vars series.json` →
      **byte-stable** `series.rendered.txt` (committed) → `compile --no-enhance` →
      validated via `scenario --dry-run` (`corpus/compile.sh`, conditional on the
      feature). The whitespace-trim (`{%- -%}`) pitfall is in `COMPILE_TEMPLATES.md`.
      4 template unit tests (run with `--features templates`).

## Then — Track M (`plakat map`) begins at 1.4.0

Spec + geometry (1.4.0) → **linework render, no SD** (1.5.0) → tiled SD render
(1.6.0) → Bund + urban (1.7–1.8). Geometry/linework are corpus-provable on-box; the
SD render is the memory-bound capstone. See the plan.

## Opportunistic / debt (off the critical path)

- COMPILE-1 leftovers: per-provider rate-limit `--compile-parallel`, token/cost
  estimate in `--dry-run`, `map:` block type (E-C4, after Track M lands).
- 1.1.0 carryovers: Flux regional (Flux broken on Metal), IC-Light (L stretch).
- Memory-bound render debt: SD3.5 DreamBooth render, `regional.sh sdxl/sd35`.

## Explicitly out of scope (still)

- Flux-on-Metal (candle GGUF kernel broken upstream); `plakat serve` HTTP daemon.
