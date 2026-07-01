# plakat 1.20.0 — roadmap: `plakat ui` robustness

The RFC TUI-1 surface is complete. This cycle (and the next) act on the post-completion
improvement brainstorm; the user split it across **two releases**:

- **1.20.0 — robustness** (this file): memory budget warning, idle auto-unload/reload,
  TUI hard reset, cache doctor.
- **1.21.0 — the build-an-image loop + workflow + polish** (see `ROADMAP_1.21.0.md`).

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## Robustness (1.20.0)

- [x] **Memory budget warning at load** — `capability::resident_estimate(alias)` (exact
      from the cached snapshot, else a coarse per-family guess) vs `hw::available_ram_gb()`;
      Models `[L]` on an over-committing model raises a centered confirm modal ([Y] load
      anyway / [N]·Esc cancel) instead of firing the download+load. Reloading the resident
      model never prompts.
- [x] **Idle auto-unload + auto-reload** — after `IDLE_UNLOAD` (10 min) of no keypresses
      with a model loaded, `idle_tick` unloads it and records the alias in `suspended`;
      the next keypress (`resume_if_suspended`) kicks a background reload (current LoRA set
      persists in `active_loras`). Never fires mid-generation.
- [x] **TUI hard reset** — palette → "Restart plakat (free all GPU memory)" → centered
      confirm → `should_reset` breaks the event loop and `run()` restores the terminal
      then `reexec()`s (Unix `CommandExt::exec`; Windows spawn+exit). A fresh process is
      the only way to fully return candle's Metal buffer pool. Palette-hosted because
      Ctrl-R is taken by screen editors (Prompts compile / LoRA assess / Scenarios rename).
- [x] **Cache doctor in the UI** — palette (Models) → "Cache doctor: sweep locks + report":
      `run_cache_doctor` sweeps stale download locks (1.19.0 `clean_stale_locks`) and reports
      cached weight GB (`capability::cached_size_gb`) / partial / not-cached + a gated hint,
      to the Output pane.

## Carry

- [⏸] **Flux in the UI** — hardware-blocked (Flux too large to verify here; GGUF-Flux-Metal
      known broken). The `ModelService::Loaded` enum + family dispatch have the seam.
