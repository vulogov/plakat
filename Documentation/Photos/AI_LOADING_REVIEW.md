# Bringing `plakat ui`'s model/LoRA loading into `plakat photos`

A review (3.5.0, Track A prep) of how the `plakat ui` TUI loads models and LoRAs, what
`plakat photos` does today, and the concrete path to unify them.

## How `plakat ui` does it

`plakat ui` never touches a pipeline on the event-loop thread. It owns a dedicated
background thread — `plakat-model-svc` — that holds the resident model and talks to the
`App` over `std::sync::mpsc`:

- **Resident model, one at a time.** `ModelService` (`src/ui/tui/services/model_service.rs`)
  spawns the thread in `model_loop`; the loaded pipeline lives as a local
  `Option<(alias, Loaded)>` on that thread's stack (SD/SD3 kept resident so refines are
  fast; PixArt/Cascade record only their LoRA set and reload per-gen). Loading a new model
  drops the old one first — unified-memory discipline.
- **Two channels + a per-gen channel.** App→thread `ModelCommand { Load{alias,loras}, Unload,
  Generate(GenJob), RunScenario, … }`; thread→App status `ModelMessage { LoadStarted, Loaded{
  alias, used_gb }, Unloaded, Error }`, drained non-blocking each tick. Each generation gets
  its own fresh `Receiver<GenMessage>` + `CancelFlag` (`GenMessage { Progress, Preview, Done,
  Error }`) so progress renders **inline** — the TUI is never suspended.
- **LoRA = merge-at-load.** No runtime toggle; the App keeps `active_loras: Vec<(PathBuf, f32)>`,
  converts them to `Vec<LoraSpec>` (`LoraSource::Local(path)`, `scale`) and passes them to
  `model_svc.load(alias, specs)`. Changing the set reloads the model. A **LoRA hub**
  (`screens/lorahub.rs`) scans local dirs + searches/downloads Civitai/HF, and gates by
  `BaseFamily` compatibility.
- **Gating.** `capability::resident_estimate(alias)` (footprint), `hf::download::is_cached(alias)`
  (cache presence → auto-load only if already local), `capability::native_res(alias)`
  (Metal-safe size), `clean_stale_locks` before each load.

## How `plakat photos` does it today

`src/photos/mledit.rs` — a thin, correct, but **basic** path:

- `MlJob::run()` calls the stable `crate::api` builders (`Upscale`, `Img2img`, `Relight`) with
  `.device("auto").run().await`. Each call **loads the model fresh, runs, drops it** — no
  residency, so back-to-back ML edits pay the full load each time.
- Runs with the **TUI suspended** (`run_ml_job` drops the alternate screen so the pipeline's
  own stderr progress bars show on the real terminal), then resumes and picks up the new file.
- Hardcoded `sdxl` for prompt ops; **no LoRA support, no model choice, no footprint/cache
  gating, no cancel.** One job at a time.

## The gap & the recommendation

The two are on opposite ends: `ui` has a resident-model service with LoRAs, inline progress,
gating, and cancel; `photos` has a fire-and-forget suspend-the-UI call with none of that. The
machinery to close the gap already exists and is **self-contained** — `ModelService` depends
only on a `Device` + a tokio `Handle`, not on any `ui` App state.

**Recommended: adopt `ModelService` in `plakat photos`, incrementally.**

1. **Phase A — reuse the service, keep the current ops.** Give `App` (photos) a
   `model_svc: ModelService` spawned once; route `MlOp::Img2img`/`Relight`/`Upscale` through
   it instead of `MlJob::run().await` under a suspend. Immediate wins: model stays resident
   across successive edits, progress renders **inline** (no more screen-drop fl&icker), and
   generations become **cancellable**. Requires: an img2img/relight `ModelCommand` path (today
   `generate` is t2i-oriented; the refine reuse `img2img::run_with_pipeline_hooked` is already
   on the thread — expose it as a command variant, or add `ModelCommand::Edit(EditJob)`).
2. **Phase B — model choice + footprint/cache gating.** Surface `native_res` /
   `resident_estimate` / `is_cached` the same way `ui` does, so photos can pick a Metal-safe
   size and refuse loads that won't fit, and auto-skip a cold download unless the user opts in.
3. **Phase C — LoRAs.** Reuse `active_loras: Vec<(PathBuf, f32)>` → `LoraSpec` and, if wanted,
   the `lorahub` scan (local dirs) for a photos-side "apply style LoRA to this image" flow.
   Compatibility gating by `BaseFamily` comes for free.

**Shared-code note:** `ModelService` currently lives under `src/ui/tui/services/`. Both the
`ui` and `photos` features pull the same TUI stack, so the cleanest move is to lift
`model_service.rs` (+ `gen_channel` is already in `src/pipelines/`) to a location both features
compile — e.g. `src/services/model_service.rs` behind `any(feature="ui", feature="photos")` —
rather than have `photos` depend on the `ui` module tree.

**What stays out of scope / unchanged:** the `:` NL planner stays a closed, album-scoped
vocabulary (no model-name free-text → no arbitrary external reads); ML ops remain
non-destructive (new variant file); default CLI output stays byte-identical.

## Concrete first steps (Track A)

- [x] **CLIP visual search live-verify** — DONE. `cargo test --lib --features photos
  clip_loads_and_embeds_into_joint_space -- --ignored` passes (18.4s): real
  `openai/clip-vit-large-patch14` weights load from the reconnected cache and embed text+image into
  the joint space. The in-tree CLIP path (`src/pipelines/clip_embed.rs` → `src/photos/visual_search.rs`)
  is confirmed end-to-end.
- [ ] Phase A above (resident `ModelService` for the existing ML ops) — the highest-leverage,
  lowest-risk reuse; unlocks inline progress + cancel + residency with no new model concepts.
