# plakat 1.20.0 — roadmap: `plakat ui` robustness

The RFC TUI-1 surface is complete. This cycle (and the next) act on the post-completion
improvement brainstorm; the user split it across **two releases**:

- **1.20.0 — robustness** (this file): memory budget warning, idle auto-unload/reload,
  TUI hard reset, cache doctor.
- **1.21.0 — the build-an-image loop + workflow + polish** (see `ROADMAP_1.21.0.md`).

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## Robustness (1.20.0)

- [ ] **Memory budget warning at load** — before loading, estimate the model's resident
      footprint (cached size + overhead) vs available RAM; if it would over-commit,
      require a confirm before the download+load. Reuses `capability` + `hw`.
- [ ] **Idle auto-unload + auto-reload** — after N minutes idle with a model loaded, unload
      it (free memory) and remember `(alias, loras)`; reload it automatically when the user
      resumes an activity that needs it.
- [ ] **TUI hard reset** — restart plakat in place (re-exec) to fully return candle's Metal
      buffer pool (no in-process force-clear API). Surfaced in the command palette + a
      confirm (Ctrl-R is taken by screen editors, so it can't be a bare global key).
- [ ] **Cache doctor in the UI** — on the Models screen, show each model's cache status
      (cached GB / not cached / gated) + a repair action (sweep stale locks, report
      partial state). Reuses the 1.19.0 lock-sweep + `capability` sizing.

## Carry

- [⏸] **Flux in the UI** — hardware-blocked (Flux too large to verify here; GGUF-Flux-Metal
      known broken). The `ModelService::Loaded` enum + family dispatch have the seam.
