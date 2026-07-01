# plakat 1.21.0 — roadmap: the build-an-image loop + workflow + polish

Second half of the post-RFC-TUI-1 improvement plan (1.20.0 was the robustness half:
memory-budget warning, idle unload/reload, hard reset, cache doctor). This cycle turns to
the **build-an-image loop** itself, **workflow / power-user** ergonomics, and **polish**.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked.

## The build-an-image loop

- [x] **Tighter refine loop** — the Chat pane title now shows what the next Enter does —
      `Chat · evolve · seed 12345` / `anchored 0.60 · seed …` / `inpaint` / `identity` /
      `fresh`, `· auto` when routing is on (`chat_mode_hint`, synced each tick). **Ctrl-T**
      one-key toggles evolve ↔ anchored (also in the palette + input hint).
- [x] **Variation batch** — `/vary [n]` (default 4, clamp 2–8) fans out N variations of the
      current image at fresh seeds. The model thread is serial, so `pump_variations` runs
      them one at a time; each lands in the filmstrip to scrub (Ctrl-←/→) and keep (Ctrl-B).
      (A *parallel* grid isn't possible on one unified-memory device — one model instance.)
- [x] **Undo/redo across the filmstrip** — covered by the existing filmstrip: `Ctrl-←/→`
      scrub the full frame history and `Ctrl-B` rolls back (branches) to the selected
      frame. A separate `Ctrl-Z` stack would duplicate this (and `Ctrl-Y` is already
      vary), so it's documented rather than added. Reassess if users ask for a linear undo.

## Workflow / power-user

- [x] **Recipe → scenario in one step** — `/scenario` (or a palette entry) reads the current
      Chat image's embedded A1111 recipe and inserts an **exactly-reproducing** task
      (prompt + negative + seed, no LLM) via `insert_recipe_task`, then jumps to Scenarios.
      Complements the existing Chat→LLM-summary→scenario path (`Ctrl-G`).
- [x] **Named presets** — `/preset save <name>` snapshots the current model + LoRA stack +
      negative to workspace `presets.json`; `/preset list` lists; `/preset <name>` or a
      palette *Preset: …* entry re-applies (sets the stack + negative, loads the model
      through the memory-budget guard). New `ui::tui::presets` module.
- [x] **Keybinding cheatsheet overlay** (**F1** — never conflicts with text input, unlike a
      bare `?`) — a centered modal listing the global keys + the active screen's keys
      (`screen_help`/`render_help`); any key closes. Also in the palette + the status-bar
      hint.

## Polish

- [x] **Consistent modal styling** — one `centered_modal(f, title, border, body, size)`
      helper now backs every overlay (load warning, hard reset, F1 cheatsheet), so they
      share centering/Clear/bordered-block styling instead of three hand-rolled copies.
- [x] **Status-line memory readout** — the global status bar now shows, right-aligned and
      headroom-tinted (red < 3 GB / yellow < 6 GB / green), `<loaded-model> · free/total GB
      free` (`mem_readout`), so the memory picture is ambient, not just at load time.
- [x] **Tutorial + RFC refresh** — `UI_TUTORIAL.md` updated: memory-aware loading (budget
      warning / idle unload / cache doctor / hard reset), the ambient status-bar readout +
      F1 cheatsheet, the Chat mode readout + `Ctrl-T`, and `/vary` / `/scenario` / `/preset`.

## Carry

- [⏸] **Flux in the UI** — hardware-blocked (see 1.20.0). Seam exists in `ModelService`.

> Pulled from the 16-item improvement brainstorm; scope per cycle by what lands cleanly.
