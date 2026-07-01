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
- [ ] **Presets: size + sampler** — extend named presets beyond model + LoRA + negative to
      the full recipe (size, steps, guidance, sampler).
- [ ] **Linear undo (`Ctrl-Z`)** — if users want a linear undo on top of the filmstrip
      scrub + rollback (deferred in 1.21.0 as redundant; reassess on demand).
- [ ] **Shared pipeline for scenario runs** — reuse the TUI's loaded model for a scenario
      run instead of the runner loading its own (RFC §0-R0-2), saving a reload.

## Carry

- [⏸] **Flux in the UI** — hardware-blocked (Flux too large to verify here; GGUF-Flux-Metal
      known broken). The `ModelService::Loaded` enum + family dispatch have the seam.

> No committed scope yet — this file is a landing pad for the next cut's decisions.
