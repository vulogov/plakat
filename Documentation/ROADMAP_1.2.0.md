# plakat 1.2.0 — roadmap

1.1.0 shipped "train your own words" (TI incl. SDXL dual-encoder), live compose,
depth-band selection, and SD3.5 DreamBooth.

**1.2.0 opens the two-track `map` + `compile` arc** (see
[`RFC_MAP_COMPILE_PLAN.md`](RFC_MAP_COMPILE_PLAN.md)). 1.2.0 = **Track C, COMPILE-1**:
`plakat compile` — prose `prompts.txt` → scenario HJSON, reusing the `src/llm/`
provider stack. Self-contained, no GPU, one optional pure-Rust dep.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## 1.2.0 — `plakat compile` core (Track C, COMPILE-1)

- [x] **Parser** (`src/compile/parser.rs`) — blocks, `key: value` commands, per-command
      merge strategies. `--dry-run` shows the block summary.
- [x] **Resolver** — global→scene inheritance (concatenate / accumulate / last-wins) +
      model-family classification (SD15/SDXL/Flux).
- [x] **Assembler** — header + text + footer; family-aware positive + negative system
      prompts; `--no-enhance` = verbatim.
- [x] **Emitter** — hand-written JSON→HJSON formatter (unquoted keys + JSON-escaped
      values + `//` comments; **deterministic, no timestamp**). Output passes
      `scenario --dry-run`. (**OQ-COMPILE-4** resolved.)
- [x] **LLM integration** — positive + auto-negative via `prompt::complete` (custom
      system prompt threaded to deepseek/gemini/local/auto); graceful fallback to
      verbatim/seed-terms on failure.
- [x] **Pipe** — `scenario -` (stdin). **`--lint`** (E-C2, deterministic) shipped.
- [ ] *Deferred to a 1.2.x point-fix:* two-namespace `--compile-cache`, `--diff`,
      `--decompile` (E-C1), `translate:`/`persona:` execution (parsed, not yet wired).
- [x] **Corpus gate MET:** `corpus/compile/basic.txt` → `--no-enhance --no-negative`
      **byte-stable** `basic.hjson` (committed), validated via `scenario --dry-run` +
      the stdin pipe (`corpus/compile.sh`). 21 compile unit tests.

## Later in the arc (see the plan)

- **1.3.0** — COMPILE-2: Tera pre-pass (`--features templates`). **1.4.0+** — Track M
  (`plakat map`): spec+geometry → linework render (no SD, 1.5.0) → tiled SD render
  (1.6.0) → Bund + urban (1.7–1.8). Geometry/linework are corpus-provable on-box; the
  SD render is the memory-bound capstone.

## Opportunistic carryovers from 1.1.0 (off the critical path)

- [ ] **Flux regional prompting** — *(M)* `--region` for Flux (today it bails).
      Flux's flow-matching transformer needs its own per-region velocity blend.
      Note: Flux is broken on Metal (candle GGUF kernel bug), so this lands
      **code-only / unverifiable** on the dev box — verify on CPU/CUDA.
- [ ] **IC-Light relighting** — *(L, stretch)* relight composited artefacts so they
      sit in the scene's light, not just on it. SD 1.5-based; porting the model is
      the work. Pairs naturally with the new `compose` `matte:`/`generate:` layers.

## Verification debt (memory-bound on 24 GB — needs more RAM or a bigger box)

- [ ] **SD3.5 DreamBooth render** — training + LoRA-merge verified; the render OOMs
      at the merge step even at 512² / `--device cpu`. Code-complete; proof awaits RAM.
- [ ] `regional.sh sdxl` / `sd35` (sd15 verified; sd35 likely OOMs 24 GB).
- [ ] `resume_train.sh` final render (resume verified; the render OOM'd).

## Ideas (unscheduled)

- Multi-vector Textual Inversion (`--vectors N`) — more capacity than the single
  vector for subjects/styles that one vector under-captures.
- `compose` `segment:` layer source (point/depth mask → cut-out inline), closing
  the segment → compose loop without a temp file.

## Explicitly out of scope (still)

- Flux-on-Metal (candle GGUF kernel broken upstream); `plakat serve` HTTP daemon
  (its own cycle); additional model families.
