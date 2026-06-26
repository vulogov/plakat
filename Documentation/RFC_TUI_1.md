# RFC TUI-1: `plakat ui` — Terminal User Interface

**Target versions:** 1.15.0 → 1.18.0 (4 development cycles)
**Status:** Accepted (with §0 codebase reconciliation)
**Author:** Vladimir Ulogov
**Invocation:** `plakat ui` (subcommand of the plakat binary, feature flag: `ui`, default-on)

---

## 0. Codebase reconciliation (added on commit, 2026-06-26)

This section records where the original RFC's stated foundations differ from the
plakat codebase as it actually stands, and the decisions taken. The screen specs
(§6–§13) and UX (§5, §14–§18) below are the design of record and unchanged; only
the load-bearing engineering assumptions are corrected here. **Where this section
conflicts with the body, this section wins.**

**R0-1 — There is no `plakat-core` crate.** plakat is a *single binary-only crate*
(no `[lib]`, no `[workspace]`). Every reference to "`plakat-core`'s `ScenarioRunner`"
means an **in-crate module** (`src/ui/services/` consuming a refactored
`src/cli/scenario.rs`), not a crate split. The TUI lives in `src/ui/` in the same
binary, exactly like every other subcommand.

**R0-2 — The reusable scenario runner must be extracted.** Today scenario execution
is one monolithic `pub async fn run(args: ScenarioArgs)` (`src/cli/scenario.rs`).
The TUI RUNNER (§8.3) needs a runner that (a) accepts an **already-loaded model**
and (b) emits progress/preview over a channel instead of writing to the terminal.
**This extraction is a Release-2 prerequisite**, tracked as a foundation task.

**R0-3 — The samplers have no step hook (the biggest single cost).** Every denoise
loop drives an `indicatif` bar directly (`bar.inc(1)`), across t2i/flux/sd3/pixart/
cascade/animatediff — there is **no callback, no cancellation token, and no
intermediate-latent decode hook**. Progressive preview (§15) and `Ctrl-C`
cancellation (§14) require threading a `StepHook` through every sampler.
**This is a Release-1 prerequisite** and is built *before* any screen code; the
existing CLI keeps identical behaviour by passing an indicatif-backed hook.

**R0-4 — New dependencies.** `ratatui`, `ratatui-image`, `crossterm`,
`tui-textarea`, and `uuid` are **all new** (the RFC's "ratatui already present"
is incorrect — plakat uses `indicatif`/`console`). `serde_json` is already a
**hard** dependency, so `templates` stays `["dep:tera"]` (unchanged from 1.3.0)
and `serde_json` is never feature-gated. The corrected feature block:

```toml
[features]
default       = ["ui", "templates"]            # ui is DEFAULT-ON (owner decision)
ui            = ["dep:ratatui", "dep:ratatui-image", "dep:crossterm", "dep:tui-textarea", "dep:uuid"]
templates     = ["dep:tera"]                    # unchanged (1.3.0); serde_json is a hard dep
shaped-labels = ["dep:ab_glyph"]                # unchanged
onnx          = ["dep:candle-onnx"]             # unchanged (1.14.1)
```

A CLI-only build (no TUI deps, no protoc) remains `cargo build --no-default-features`.

**R0-5 — `default = ["ui"]` is the owner's explicit decision.** Noted that this
reverses the lean-install direction of 1.14.1 (every `cargo install plakat` now
pulls the TUI dep tree); accepted so `plakat ui` works out of the box.

---

## Contents

1. [Overview and philosophy](#1-overview-and-philosophy)
2. [Prerequisites and terminal detection](#2-prerequisites-and-terminal-detection)
3. [Workspace concept](#3-workspace-concept)
4. [Application architecture](#4-application-architecture)
5. [Global chrome](#5-global-chrome)
6. [Screen 1: Chat](#6-screen-1-chat)
7. [Screen 2: Models](#7-screen-2-models)
8. [Screen 3: Scenarios](#8-screen-3-scenarios)
9. [Screen 4: History](#9-screen-4-history)
10. [Screen 5: LoRA Hub](#10-screen-5-lora-hub)
11. [Screen 6: People](#11-screen-6-people)
12. [Screen 7: Prompt Workspace](#12-screen-7-prompt-workspace)
13. [Screen 8: Canvas](#13-screen-8-canvas)
14. [Generation queue and cancellation](#14-generation-queue-and-cancellation)
15. [Progressive image display](#15-progressive-image-display)
16. [Model lifecycle service](#16-model-lifecycle-service)
17. [Keyboard conventions](#17-keyboard-conventions)
18. [Error handling](#18-error-handling)
19. [Crate dependencies](#19-crate-dependencies)
20. [Module structure](#20-module-structure)
21. [Release plan](#21-release-plan)
22. [Acceptance criteria](#22-acceptance-criteria)
23. [Open questions](#23-open-questions)

---

## 1. Overview and philosophy

`plakat ui` is a full-featured terminal user interface for plakat, invoked as a
subcommand of the existing `plakat` binary. It is the primary interactive
interface for users who prefer the terminal over the CLI's one-shot model. It is
a persistent application that owns the model lifecycle, maintains session state,
and provides eight screens covering every major plakat workflow.

The CLI (`plakat generate`, `scenario`, `multiperson`, `map`, `compile`) remains
the scriptable, pipeline-friendly interface. `plakat ui` is the interactive face
of the same capabilities, sharing all generation pipelines, model loading, the
LLM provider stack, and recipe serialization through the same codebase.

`plakat ui` is a match arm in the existing clap subcommand router, parallel to
every other subcommand. The TUI code lives in `src/ui/` gated behind the
default-on `ui` feature (see §0-R0-4 for the corrected feature block).

The eight screens: **Chat** (conversational generation), **Models** (lifecycle +
memory), **Scenarios** (author/select/run), **History** (gallery + recipes),
**LoRA Hub** (local + CivitAI + HF + LLM recs), **People** (identity stewardship),
**Prompt Workspace** (prose→scenario authoring), **Canvas** (regional masks).

**What it is not:** a CLI replacement (scripts use `scenario`/`compile`), a
separate binary, a web app, or a pixel-accurate editor.

---

## 2. Prerequisites and terminal detection

`plakat ui` requires a terminal with a pixel graphics protocol. At startup,
before ratatui takes the terminal, it queries capabilities via
`ratatui_image::picker::Picker::from_query_stdio()`. Protocol priority: Kitty >
iTerm2 > Sixel; Halfblocks is rejected as insufficient. If no protocol is
detected, `plakat ui` exits *before* raw mode with a message listing supported
terminals (Kitty/iTerm2/WezTerm/Ghostty/foot/Sixel) and pointing CLI users at
`plakat generate` / `plakat scenario`.

It also queries `\x1b[14t` for the display area in pixels (needed for correct
image scaling); a zero response → assume 10×18 px/cell + warn.

Invocation:

```sh
plakat ui                                  # current dir — create workspace if none
plakat ui --workspace ~/projects/family    # specific workspace (create if absent)
plakat ui --screen chat|models|scenarios   # open to a screen
plakat ui scenarios/family_portrait.hjson  # pre-load a scenario
```

---

## 3. Workspace concept

A workspace is a directory with a known structure used as the working context;
all screens resolve paths relative to its root.

**Detection order:** `--workspace <dir>` → `plakat-workspace.hjson` in cwd →
walk up parents → else run the **creation wizard** in cwd. There is no degraded
mode: if no workspace is found, one is always created so every screen has full
functionality from first launch.

**Creation wizard** (runs before raw mode, ≤10s, three Enter-able questions):
name, default model, default LLM provider, include global LoRA dir. Creates
`plakat-workspace.hjson` + `people/ scenarios/ loras/ refs/ prompts/ chat/ out/
.plakat_cache/` + a `.gitignore` (excludes `.plakat_cache/`, `out/`, and
`people/*/encoding{,_tests}/`; deliberately leaves `people/*/refs/` to the user
since reference photos may be personal).

**Migration:** in a directory with existing plakat files (scenarios, `out/`,
LoRAs) but no workspace marker, the wizard offers to adopt the structure —
creating only what's missing, moving/modifying nothing.

**`--workspace <dir>` auto-creates**: absent dir → create + wizard; existing dir
without a marker → wizard in place.

### Workspace structure

```
my_project/
  plakat-workspace.hjson     ← root marker + project config
  people/<name>/             ← person.hjson, refs/, encoding/, encoding_tests/, portfolio/
  scenarios/                 ← *.hjson + *.run.hjson run-history sidecars
  loras/                     ← *.safetensors + *.plakat.hjson sidecars
  refs/                      ← shared non-person reference images
  prompts/                   ← prompts.txt + *.tera
  chat/                      ← serialized chat sessions
  out/                       ← generated images (per scenario/session)
  .plakat_cache/             ← ephemeral (gitignored): lora_search, scene_layouts, llm_assessments
  .gitignore                 ← auto-generated
```

### `plakat-workspace.hjson`

```hjson
{
  name: "My Portrait Project"
  created: "2026-06-07"
  default_model: sdxl
  default_identity: plus-face-sdxl
  default_steps: 35
  default_guidance: 7.5
  default_size: "1280x768"
  layout_provider: deepseek
  enhancer: deepseek
  out_dir: out
  people_dir: people
  scenarios_dir: scenarios
  loras_dir: loras
  prompts_dir: prompts
  chat_dir: chat
  global_lora_dirs: ["~/.plakat/loras"]
  preview_every_n_steps: 5
  default_consent: { permitted_uses: ["personal"], restrictions: [] }
}
```

These override the global `~/.config/plakat/config.hjson`; both are editable any
time.

---

## 4. Application architecture

### Entry point

```rust
// src/ui/mod.rs
pub struct UiArgs { pub workspace: Option<PathBuf>, pub screen: Option<String>, pub file: Option<PathBuf> }

pub async fn run(args: UiArgs) -> Result<()> {
    let picker    = check_terminal_support()?;            // exits cleanly if unsupported
    let workspace = resolve_or_create_workspace(args.workspace)?;
    App::new(workspace, picker, args)?.run().await
}
```

### App + event loop

`App` holds the active screen, the `Workspace`, the background services
(`ModelService`, `GenQueue`, `DownloadManager`, `LlmPool` behind `Arc`), per-screen
state structs (persist across switches), and global state (notifications, command
palette, `Picker`). The loop polls input (100ms), drains the gen/download/llm
channels, drains notifications, and redraws each tick.

### Background services (tokio tasks + channels)

- **ModelService** — owns loaded components, serialises load/unload, hands
  `Arc<LoadedModel>` to generations.
- **GenQueue** — ordered `GenerationRequest` queue (depth 5), a single
  `spawn_blocking` worker (Metal device exclusivity), emits `GenMessage`
  (Progress/Preview/Done/Error). **Built on the §0-R0-3 `StepHook`.**
- **DownloadManager** — concurrent resumable HTTP, SHA-256 verify.
- **LlmPool** — concurrent provider calls (scene analyser, enhancer, LoRA recs).

---

## 5. Global chrome

- **Tab bar** (top, 1 line): the eight screens, active highlighted, `●` =
  background activity. `Ctrl-1`…`Ctrl-8` switch instantly; running generation
  continues regardless of the visible screen.
- **Status bar** (bottom, 1 line): model · format · size · queue depth · Metal
  memory · clock; during generation shows `step k/N · s/step · ETA`.
- **Command palette** (`Ctrl-Space`): fuzzy overlay over all registered actions.
- **Help overlay** (`?` / `Ctrl-H`): global + screen-specific keybindings.

---

## 6. Screen 1: Chat  *(Release 1)*

Primary interactive generation. Left: session history (utterances + step records).
Right: progressive preview pane + a scrollable step-thumbnail strip. Bottom: input
line with inline param chips and a `Ctrl-P` parameter sidebar.

Each utterance goes to the LLM, which classifies intent into a structured
`ChatOperation` (Generate, ImgToImg{prompt_delta, action, strength}, Inpaint,
FaceRefine, ParameterChange, Rollback, Variations, StyleTransfer, Upscale,
SaveSession). ImgToImg strength guidance: 0.2–0.4 texture/colour, 0.4–0.6
aesthetic, 0.6–0.8 recomposition.

`ChatState` holds the session id, `Vec<ChatStep>` (utterance + operation + recipe
+ full_path + thumbnail + duration + seed), current step, gen state, input,
params, scroll positions, sidebar flag.

- **Progressive display**: preview every N steps (default 5) at half-res; three
  visual states (spinner → preview+bar → final+annotation).
- **Rollback** (instant, no regen): utterance ("go back to step 1…") → `Rollback`,
  or thumbnail-strip arrows + Enter; next gen branches from there.
- **Variations** (`Ctrl-V`): N seeds → sub-steps `3a..3d`.
- **Parameter sidebar** (`Ctrl-P`): edit + Enter re-generates.
- **`@mention`**: `@alice` → `people/alice/person.hjson`; ≥2 → multiperson, 1 →
  portrait.
- **Session**: `/save`, `/load <session>` to/from `chat/`.

---

## 7. Screen 2: Models  *(Release 1)*

Segmented memory bar (per-component: T5/CLIP/Transformer/VAE + free) over a
local/remote model list and a detail pane (architecture, sizes per format, text
encoders, VAE, compatibility, projected memory-if-loaded). Actions: `L` load,
`U` unload, `F` format (bf16/fp8/int8), `D` download.

Component-level tracking matters on Apple-Silicon unified memory: T5-XXL shared
between Flux and SD 3.5 reuses the already-mapped component (same `Arc`). Memory
is projected before any load; >95% of budget → warn + offer to unload. Loading
progress hooks the safetensors header (tensor count) and increments per mmap'd
tensor.

---

## 8. Screen 3: Scenarios  *(Release 2)*

Three sub-tabs (`Tab` cycles): **SELECT**, **EDITOR**, **RUNNER**.

- **SELECT**: scenario browser with per-file last-run status from a `.run.hjson`
  sidecar (last 10 runs: started/completed/duration/model/tasks_passed/failed +
  per-task results). New-scenario wizard (blank / portrait_series /
  multiperson_shoot / map_generation). `R` retries only `status: failed` tasks.
- **EDITOR**: `tui-textarea` with 300ms-debounced HJSON parse + schema check
  (red `●` gutter markers + detail panel), `Ctrl-/` context completion,
  person-reference validation against `people/<name>/`, `Ctrl-P` resolved
  preview, `Ctrl-R` save+validate+run (stays in editor on failure).
- **RUNNER**: two nested progress bars (scenario tasks / current step) + preview
  pane (same `GenMessage::Preview` channel), pause/skip/edit-during-run. `Ctrl-C`
  → confirmation; "use last preview" saves a partial with `cancelled: true`.

**Uses the extracted in-process scenario runner (§0-R0-2)** — the already-loaded
model is passed in, no subprocess, no reload between tasks. The same runner backs
the `plakat scenario` CLI.

---

## 9. Screen 4: History  *(Release 2)*

Date-grouped lazy thumbnail grid (LRU cache, default 100) over `out/`, with
semantic search across any recipe field, a side-by-side comparison view
(`Ctrl-C`) that highlights recipe diffs, `C` to load an image as Chat step 0,
and tag (`T`) + export (`X`) for collection building. Lazy loading must never
block the event loop.

---

## 10. Screen 5: LoRA Hub  *(Release 3 — 3a LOCAL · 3b CivitAI+HF · 3c LLM recs)*

Tabs: **LOCAL** (scan workspace + global lora dirs, read `.plakat.hjson` sidecar,
safetensors-header compatibility vs the *currently loaded* model, live-updating),
**CIVITAI** (structured API search with `baseModel` filter, optional key, 1h
cache, `civitai_base_to_plakat_family()` mapping + post-download header scan),
**HUGGINGFACE** (two-stage: fast pre-filter discards non-`.safetensors`/non-
diffusion/zero-download<24h, then an LLM scores the remaining 30–40 for relevance
+ base-model guess + one-line assessment; HF results cached 1h, LLM 24h).

Sidecar `.plakat.hjson`: name, civitai ids, base_model, rank, trigger_words,
compatible_models, weight_range/default, notes, preview image, cached
`llm_assessments[]` (provider + context_hash + text), and a `download` block
(url, sha256, size, verified). LLM features: `R` assess selected, `R` from search
= recommend-for-context, `Ctrl-R` = combination suggestions (populates Chat's
active LoRA list). Downloads: ≤2 concurrent, range-resume, SHA-256 verify,
version-update detection, `●` in the tab bar.

---

## 11. Screen 6: People  *(Release 3a)*

A Person is an **identity**, not an asset — it persists across all generation
contexts. Directory: `people/<name>/{person.hjson, refs/, encoding/,
encoding_tests/, portfolio/}`.

`person.hjson`: name, display_name, weighted `refs[]` (path/weight/angle/lighting/
notes), identity strategy, encoding_mode (weighted_average|concatenated),
face_strength, encoding_quality, behavioural prompt/negative, `known_good[]`
parameter combos, a `consent` block (granted_by/date/permitted_uses/restrictions/
notes), and a plakat-maintained `stats` block (appearances, scenarios, sessions,
last_used, consistency score).

- **LIBRARY**: person list + coverage summary + encoding quality + actionable
  guidance ("no profile-right photo → left-facing scenes may be less consistent").
- **DETAIL** (six sub-tabs): REFS (weights + angle/lighting/expression coverage
  analysis via face-bbox detection), ENCODING (per-strategy quality + re-encode +
  side-by-side strategy comparison), PORTFOLIO (lazy grid + pairwise-similarity
  consistency score), TEST (4 fixed test gens), KNOWN GOOD (editable param table,
  apply-to-chat), SETTINGS (consent + privacy audit log).
- **Quick generate** (`G`, multi-select): 1 → portrait, ≥2 → multiperson; result
  opens in Chat.
- **Import** (`I`) from scenario files (conflict-aware; rewrites refs to
  `people/<name>/`).
- **Encoding**: auto on first strategy+model use, explicit via `E`, invalidated by
  ref-photo or strategy/model-family change (NOT by prompt/face-strength). Emits
  progress + a quality score (face similarity vs refs).
- **Right to be forgotten** (`Del`): type-name confirmation, removes the dir,
  updates scenario refs, offers to delete `out/` images.
- **Consent enforcement** at generation time: `no_nsfw` appends the negative,
  `no_political` injects analyser instructions — silent, visible in SETTINGS.

---

## 12. Screen 7: Prompt Workspace  *(Release 4)*

`tui-textarea` editor + a live structural-compile pane (parse + inheritance + `//`
split, no LLM, 500ms debounce). `Ctrl-R` runs the full LLM compile (model-family-
aware enhancement + auto-negative, using the compile cache). `Ctrl-T` Tera mode
adds a variable panel (live render). Buffer list (`Ctrl-Tab`) over `.txt`/`.tera`/
`.hjson`. `Ctrl-Enter` saves compiled HJSON and opens it in Scenarios EDITOR.

---

## 13. Screen 8: Canvas  *(Release 4)*

Full-terminal image (ratatui-image) with cell-grid mask painting (arrows + Space,
Shift+arrows to paint-while-moving) and preset regions covering ~80% of cases:
`S` sky (top 30%), `B` background (top 60% minus faces), `L`/`R` halves, `F`
foreground (bottom 40%), `P-left/center/right` person columns. Each cell ≈ 10×18
px — coarse *regional* masking only (documented); fine masks need an external
editor + `--mask-path`. `Enter` rasterizes to a full-res PNG and hands Chat a
pre-populated inpaint mask. `M` toggles outpaint mode (arrow at boundary extends
by 128px units, grey-border preview).

---

## 14. Generation queue and cancellation

Queue depth 5 (full → reject with a clear message). `Ctrl-C` during generation
opens a confirmation (not an immediate quit): **U** use last preview as output
(saved at preview res with `cancelled: true`), **D** discard, **K** keep running.
The sampler checks the cancellation token between steps (via the §0-R0-3
`StepHook`) — cancellation takes effect at the next step boundary, not mid-step.

---

## 15. Progressive image display

Every N steps (default 5) the current latent is decoded at half output resolution
(~100ms on Metal) and sent as `GenMessage::Preview`. **This requires the §0-R0-3
`StepHook`** — the decode + emit happen inside the sampler via the hook; no
sampler writes to the terminal directly anymore.

```rust
pub enum GenMessage {
    Progress { step: u32, total: u32, elapsed: Duration, steps_per_sec: f32 },
    Preview  { step: u32, image: DynamicImage },
    Done     { full_path: PathBuf, thumbnail: DynamicImage, recipe: Recipe },
    Error    { message: String, partial_path: Option<PathBuf> },
}
```

The 100ms tick stores the newest `Preview` as `Arc<DynamicImage>`; ratatui-image
renders it (Kitty/iTerm2/Sixel encoding handled transparently). Preview consumers:
Chat (primary), Scenarios RUNNER (current task), People ENCODING (≤3 side-by-side),
LoRA Hub DETAIL.

---

## 16. Model lifecycle service

Component-keyed tracking `(family, component, quantization)`; shared components
reuse one `Arc`. Memory polled from `task_info` (macOS) every 2s. Loading:
header-scan → per-tensor progress (`LoadProgress{phase, tensors_done/total,
bytes_done/total}`). `project_load(model)` returns the projected layout bar before
confirming; >95% budget → warn + offer unload. Quantization change = unload→load
shown as one progress sequence.

---

## 17. Keyboard conventions

Global: `Ctrl-1..8` screens · `Ctrl-Space` palette · `Ctrl-Q` quit (confirm if
generating) · `?`/`Ctrl-H` help. Nav: `Tab`/`Shift-Tab` panels · arrows or
`hjkl` · `Enter` select · `Esc` cancel/back · `/` search · `Space` multi-select.
Generation: `Ctrl-Enter` submit · `Ctrl-C` cancel · `Ctrl-Z` rollback (Chat) ·
`Ctrl-V` variations (Chat) · `Ctrl-P` param sidebar (Chat). Items: `Enter` open ·
`D` delete · `E` edit · `C` continue-in-chat · `A` add-to-context · `G` generate
(People) · `R` assess/recommend. Canvas: arrows move · `Space` paint · `Shift+`
paint-move · `S/B/L/R/F/P` presets · `M` mode · `C` clear · `Enter` apply · `Esc`
cancel.

**`tui-textarea` focus** (EDITOR, Prompt Workspace): global keys suspended except
`Ctrl-Q`; a `EDITOR — Esc for nav mode` indicator shows; `Esc` returns to nav
(see OQ-1).

---

## 18. Error handling

Three non-blocking tiers: **Notification** (bottom banner, auto-dismiss 5s, e.g.
"cache hit"), **Warning** (persistent banner under the tab bar, `Esc` to dismiss,
e.g. "face refine fell back to centroid"), **Error popup** (modal, requires
acknowledgment, e.g. "projected memory exceeds budget — unload sdxl? [Y/N]").

---

## 19. Crate dependencies

See §0-R0-4 for the corrected, compile-true feature block. New `ui`-feature deps:
`ratatui` (0.29), `ratatui-image` (5), `crossterm` (0.28), `tui-textarea` (0.7),
`uuid` (1, v4). No new C FFI, no new `unsafe`. (Exact versions pinned at
implementation time against the installed ratatui generation.) CLI-only:
`cargo build --no-default-features`.

---

## 20. Module structure

```
src/
  main.rs                  — ui::run dispatched here
  cli/scenario.rs          — refactored: thin CLI shell over the extracted runner (R0-2)
  pipelines/step_hook.rs   — StepHook trait + IndicatifHook + ChannelHook (R0-3, Release 1)
  ui/
    mod.rs                 — UiArgs, run(), terminal detection, workspace resolve/create
    app.rs                 — App, event loop, channel draining, render
    screens/{chat,models,scenarios/{select,editor,runner},history,
             lora_hub/{local,civitai,huggingface,recommendations},
             people/{library,detail/*,import,wizard},prompt_workspace,canvas}.rs
    widgets/{preview,progress_bar,thumbnail_strip,thumbnail_grid,command_palette,
             help_overlay,notification,memory_bar,tab_bar,status_bar}.rs
    services/{model_service,gen_queue,download_manager,llm_pool,scenario_runner}.rs
    workspace.rs           — Workspace, detection, wizard, migration, .gitignore
    config.rs              — global config
```

---

## 21. Release plan

- **Release 1 (1.15.0) — Foundation: Chat + Models.** *First lands the §0-R0-3
  `StepHook`* (CLI behaviour-preserving), then: `plakat ui` subcommand + terminal
  detection + workspace wizard/migration/.gitignore + global chrome (screens 3–8
  show "coming later") + services (ModelService, GenQueue, LlmPool) + **Chat**
  (full conversational gen, operation classifier, progressive preview, thumbnail
  strip, rollback, variations, param sidebar, `@mention`, save/load) + **Models**
  (list, load/unload, format, component memory bar, load progress) + palette
  (Chat/Models actions).
- **Release 2 (1.16.0) — Scenarios + History.** *First lands the §0-R0-2 runner
  extraction.* Scenarios SELECT/EDITOR/RUNNER, `.run.hjson` sidecar, History
  (lazy grid, search, compare, continue-in-chat, export).
- **Release 3 (1.17.0) — LoRA Hub + People.** 3a People (library + 6 detail tabs +
  import + quick-gen + encoding + consent) + LoRA Hub LOCAL; 3b CivitAI + HF +
  download manager; 3c LLM recommendations. *People (3a) before remote LoRA — it
  reuses existing pipelines, needs no network/keys (OQ-10).*
- **Release 4 (1.18.0) — Prompt Workspace + Canvas.** Prompt Workspace (live +
  LLM compile, Tera, buffers, send-to-scenarios), Canvas (masks, presets,
  outpaint, mask→Chat), full palette registration.

---

## 22. Acceptance criteria

Per release, as authored in the original RFC (Release 1–4 checklists), plus two
foundation gates added here:

- [ ] **F-1 (Release 1):** the `StepHook` is threaded through the t2i sampler(s);
      `plakat generate` output + progress are byte/behaviour-identical to pre-hook;
      a `ChannelHook` delivers `Progress`/`Preview`/cancel to a test harness.
- [ ] **F-2 (Release 2):** `plakat scenario` CLI and the TUI RUNNER call the *same*
      extracted runner; a scenario run from a pre-loaded model performs **no** model
      reload between same-model tasks.

(Original Release 1–4 acceptance lists carried verbatim from the source RFC.)

---

## 23. Open questions

OQ-1 textarea keybinding suspension (proposed: global suspended except `Ctrl-Q`,
`Esc` to nav). OQ-2 `mask_schedule` (hidden behind Advanced). OQ-3 SCRFD fallback
to centroid + warning. OQ-4 portfolio consistency as background task, cached.
OQ-5 CivitAI NSFW filtered by default. OQ-6 HF gated-model 403 → show license URL.
OQ-7 `plakat ui --init` (defer to Release 1 stretch). OQ-8 Windows portfolio
symlink fallback to `.ref.hjson` pointer. OQ-9 `--workspace` with content → migration
wizard, single-Enter confirmable. OQ-10 People (3a) before remote LoRA (3b) —
**resolved: yes** (see §21).
