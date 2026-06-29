# plakat 1.16.0 — roadmap: `plakat ui` depth

1.15.0 shipped the **`plakat ui`** terminal UI end-to-end (RFC TUI-1: all eight
screens — Chat, Models, Scenarios, History, People, LoRA Hub, Prompt Workspace,
Canvas — over the same engine as the CLI). 1.16.0 pays down the deferrals that were
explicitly carried out of the TUI-1 cycle: the depth features each screen stubbed,
plus the cross-cutting engine work the UI now wants.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

Reference: [`Documentation/RFC_TUI_1.md`](RFC_TUI_1.md) (the design of record).

## A — generation engine (cross-cutting, highest leverage)

- [x] **StepHook-wired img2img** — threaded an optional `StepHook` through
      `blend_latents_one` → `img2img::run_with_pipeline_hooked` → the model thread's
      `ChannelHook`, so a Chat refine / inpaint now has live preview + mid-denoise
      cancel. (RFC §0-R0-3)
- [x] **LLM operation classifier** — `/auto on|off` (opt-in): an LLM classifies each
      follow-up as EDIT (refine) vs NEW (fresh) before dispatch, instead of the
      always-refine heuristic. (RFC §6)
- [~] **Non-SD families in the UI** — done family-by-family.
  - [x] **SD3 / 3.5** — `ModelService` now holds a `Loaded` enum (`Sd` | `Sd3`); SD3
        loads via `sd3::Pipeline::load` and generates via its hooked `generate_hooked`
        (one call covers txt2img / img2img / inpaint), so the full Chat flow — refine,
        Canvas inpaint, LoRA apply, live preview/cancel — works on SD3.5 too.
  - [x] **PixArt-Σ** / **Stable Cascade** — `run_hooked(RunRequest)` (both already
        StepHook-wired). No persistent pipeline, so `Loaded::{PixArt,Cascade}` hold
        only the applied LoRA set and generate load-per-call (the first gen shows the
        download/load). txt2img only (no img2img): prompt-evolve refine works,
        anchored/inpaint falls back to a fresh render.
  - [ ] **Flux** — `flux::run` is load-per-call with a ~25-field `Request` (+ the
        GGUF-Metal block); needs its own dispatch. Last family.
- [ ] **In-process scenario / portrait runner** — Scenario runs and People quick-gen
      load their **own** model alongside any Chat model (double load, memory
      pressure). **Large refactor**: the scenario runner is monolithic and selects its
      own model; sharing the loaded pipeline means extracting a runner that accepts an
      already-loaded model + reconciling the scenario's model field. (RFC §0-R0-2)

## B — People depth

- [ ] **Detail sub-tabs** (RFC §11) — REFS (angle/lighting coverage analysis via
      face-bbox), ENCODING (per-strategy quality + re-encode + comparison), PORTFOLIO
      (lazy grid + pairwise-similarity consistency), TEST (4 fixed test gens),
      KNOWN-GOOD (editable param table, apply-to-Chat), SETTINGS (consent + privacy
      audit). The `person.hjson` schema already reserves these fields.
- [ ] **Import** (`I`) — pull personas out of a scenario file into `people/<name>/`
      (conflict-aware; rewrite ref paths).
- [ ] **Re-encode** (`E`) — explicit identity encoding with a quality score; auto on
      first strategy+model use; invalidated by ref/strategy/model change.
- [ ] **Right to be forgotten** (`Del`) — type-name confirmation; remove the dir,
      update scenario refs, offer to delete `out/` images.
- [ ] **Mixed-family multiperson** — route a marked set by the personas' strategies
      instead of forcing plus-face/sd15 for the whole scene.
- [ ] **Identity-preserving Chat continuation** — continuing a portrait keeps the
      *look* but not strict face identity (Chat refine is plain img2img). Add an
      IP-Adapter-aware refine path.

## C — LoRA Hub completeness

- [x] **`Ctrl-R` combination suggestions** — on LOCAL, the LLM suggests a LoRA *stack*
      (1–3 + rough weights) for the current Chat prompt from the compatible LoRAs;
      shown in the detail pane.
- [x] **Per-LoRA weight control** — `active_loras` carries a per-LoRA weight; `+`/`-`
      on an applied LoRA nudges it (0.1–1.5) and reloads; the list shows `★<weight>`.
- [x] **HF base-model marker** — `RemoteHit.family` is inferred (Civitai from
      `baseModel`, HF guessed from the repo id), so HF hits show `✓`/`✗`.
- [~] **Download manager** — download *progress* already streams to the Output pane
      (rerouted bars) and the tab bar now shows `● downloading` while a download is in
      flight (single-flight, guarded). Still deferred (deep robustness, low leverage):
      ≤2 concurrent, range-resume, explicit SHA-256 verify, version-update detection.
- [x] **Search caching** — Civitai/HF results cached **1h** to a sidecar JSON under the
      shared cache root (`<cache>/plakat-ui/lora-search/`, honors `--cache-dir`), keyed
      by `(source, normalized query)`, fresh via file mtime; an identical recent query
      is served from disk with a `(cached)` status. `RemoteHit`/`DownloadRef` are now
      `Serialize`/`Deserialize`. (LLM-assessment 24h caching + the two-stage HF
      pre-filter remain as smaller follow-ups.)

## C′ — Chat → Scenario (new this cycle)

- [x] **Grab Chat session as a task** (`Ctrl-G` in the Scenarios EDITOR) — distill the
      whole Chat refinement thread into one coherent prompt via the LLM and insert a
      `{ name, prompt }` task at the cursor (quoted + escaped; background job; opens a
      fresh template if no editor is active). Closes the explore-in-Chat →
      batch-as-scenario loop.

## D — History richness

- [ ] **Semantic search** across any recipe field.
- [ ] **Side-by-side compare** (`Ctrl-C`) that highlights recipe diffs.
- [ ] **Tag** (`T`) + **export** (`X`) for collection building.
- [ ] **True thumbnail grid** (lazy, LRU cache) instead of list + single preview.
- [ ] **Background image decode** — move the selected-image decode off the event-loop
      tick to a worker so large (upscaled) PNGs never hitch navigation.

## E — Prompt Workspace + Canvas finish

- [ ] **Tera mode** (`Ctrl-T`) — toggle the compile Tera pre-pass (`templates`
      feature) with a live variable panel.
- [ ] **Buffer cycling** (`Ctrl-Tab`) polish + new-buffer naming.
- [ ] **Canvas outpaint** (`M`) — arrow at a boundary extends by 128px units, grey
      preview; hands Chat an outpaint job.
- [ ] **Canvas face-aware `B`** — exclude detected faces from the background preset.
- [ ] **Finer masks** — document the external-editor + `--mask-path` path more
      prominently; optionally a finer grid toggle.

## F — polish

- [ ] **Command palette** (RFC §5) — fuzzy action launcher per screen.
- [ ] **Save/load Chat session** + thumbnail strip + explicit rollback/variations.
- [ ] **`@mention`** people / LoRAs / styles inline in the Chat prompt.
