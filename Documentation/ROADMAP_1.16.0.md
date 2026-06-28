# plakat 1.16.0 — roadmap: `plakat ui` depth

1.15.0 shipped the **`plakat ui`** terminal UI end-to-end (RFC TUI-1: all eight
screens — Chat, Models, Scenarios, History, People, LoRA Hub, Prompt Workspace,
Canvas — over the same engine as the CLI). 1.16.0 pays down the deferrals that were
explicitly carried out of the TUI-1 cycle: the depth features each screen stubbed,
plus the cross-cutting engine work the UI now wants.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

Reference: [`Documentation/RFC_TUI_1.md`](RFC_TUI_1.md) (the design of record).

## A — generation engine (cross-cutting, highest leverage)

- [ ] **StepHook-wired img2img** — the refinement / inpaint path runs `img2img`
      without a `StepHook`, so there's **no live preview and no mid-denoise cancel**
      during a refine (only the final image + Output-pane progress). Thread the hook
      through `img2img::run_with_pipeline` like the txt2img sampler. (RFC §0-R0-3)
- [ ] **In-process scenario / portrait runner** — Scenario runs and People quick-gen
      load their **own** model alongside any Chat model (double load, memory
      pressure). Extract a runner that accepts the already-loaded pipeline. (RFC §0-R0-2)
- [ ] **Non-SD families in the UI** — the UI loads only SD-family (sd15/sd21/sdxl);
      Flux, SD3, PixArt, and Cascade are CLI-only. Wire their loaders into
      `ModelService` (per-family `t2i_load_check`).
- [ ] **LLM operation classifier** — Chat refinement is a heuristic (image-exists →
      refine, `/new` → fresh). Add the RFC §6 intent classifier (edit vs new scene
      vs variation) so the mode is inferred from the prompt.

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

- [ ] **`Ctrl-R` combination suggestions** — LLM-suggested LoRA *stacks* for a prompt
      (populates Chat's active-LoRA list). (RFC §10 3c, the last LLM feature.)
- [ ] **Per-LoRA weight control** — a weight slider per applied LoRA (sidecar
      `weight_range`/`default`), instead of the fixed 0.8.
- [ ] **Download manager** — inline progress bar (currently status-line only), ≤2
      concurrent, range-resume, SHA-256 verify, version-update detection, `●` in the
      tab bar.
- [ ] **Search caching** — Civitai/HF results cached 1h, LLM assessments 24h
      (sidecar `llm_assessments[]`); the two-stage HF pre-filter (discard
      non-`.safetensors`/non-diffusion/zero-download<24h before the LLM scores 30–40).
- [ ] **HF base-model marker** — infer/guess a base family for HF hits (the API
      doesn't expose one) so the compat `✓`/`✗` works before download.

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
