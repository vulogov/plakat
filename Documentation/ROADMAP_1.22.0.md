# plakat 1.22.0 — roadmap (open cycle)

1.20.0 (robustness) + 1.21.0 (build-an-image loop + workflow + polish) completed the
post-RFC-TUI-1 improvement plan. RFC TUI-1 itself is done. This is an open cycle — pick
the next direction with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## Candidate directions

- [x] **Per-model generation-size override** — `/size <WxH>` \| `<N>` \| `native` sets a
      per-model generation size (each model remembers its own; absent = native square, always
      Metal-safe). Guarded by `capability::generation_estimate_gb(alias, w, h)` (the
      resolution-scaled working set) vs free RAM — an over-committing size still sets but warns
      that the memory guard will abort cleanly. Shown in the Chat mode readout (`· 1024×768`).
      *(Persisting the size into presets is the "presets: size + sampler" item below.)*
- [x] **Presets: size + steps + guidance** — named presets now also carry the per-model
      generation **size**, **steps** (`/steps`), and **guidance** (`/cfg`); save captures the
      current overrides, apply restores them. `serde(default)` keeps old preset files loadable.
      *(Sampler isn't included: the Chat path always uses each model's default scheduler —
      there's no per-session sampler selection to snapshot. A sampler picker would be its own
      feature.)*
- [ ] **Linear undo (`Ctrl-Z`)** — if users want a linear undo on top of the filmstrip
      scrub + rollback (deferred in 1.21.0 as redundant; reassess on demand).
- [x] **Shared pipeline for scenario runs** — the ModelService now hands a *vanilla* resident
      SD pipeline to the scenario runner, which reuses it as the run's SD base when the models
      match, there are no scenario-level LoRAs, and no refiner (`can_reuse_sd_pipeline`) — so a
      matching all-SD scenario skips the reload. Any mismatch (different model, LoRA'd Chat
      pipeline, non-SD family, mixed/animate run) drops the handoff up front and loads fresh, so
      reuse can never change output. RFC §0-R0-2.

## Carry

- [⏸] **Flux in the UI** — hardware-blocked (Flux too large to verify here; GGUF-Flux-Metal
      known broken). The `ModelService::Loaded` enum + family dispatch have the seam.

> No committed scope yet — this file is a landing pad for the next cut's decisions.
