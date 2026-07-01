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
- [ ] **Variation grid** — from any frame, fan out N seeds at once (a small parallel batch)
      and pick the keeper into the filmstrip.
- [ ] **Undo/redo across the filmstrip** — `Ctrl-Z`/`Ctrl-Y` over the frame history (roll
      back is there; make it a proper stack).

## Workflow / power-user

- [ ] **Recipe → scenario in one step** — promote the current Chat frame's embedded recipe
      straight into a scenario task (today it's Chat→summary→scenario; add a direct path).
- [x] **Named presets** — `/preset save <name>` snapshots the current model + LoRA stack +
      negative to workspace `presets.json`; `/preset list` lists; `/preset <name>` or a
      palette *Preset: …* entry re-applies (sets the stack + negative, loads the model
      through the memory-budget guard). New `ui::tui::presets` module.
- [x] **Keybinding cheatsheet overlay** (**F1** — never conflicts with text input, unlike a
      bare `?`) — a centered modal listing the global keys + the active screen's keys
      (`screen_help`/`render_help`); any key closes. Also in the palette + the status-bar
      hint.

## Polish

- [ ] **Consistent modal styling** — the 1.20.0 confirm modals (load warning / hard reset)
      established a pattern; fold the older ad-hoc prompts into it.
- [x] **Status-line memory readout** — the global status bar now shows, right-aligned and
      headroom-tinted (red < 3 GB / yellow < 6 GB / green), `<loaded-model> · free/total GB
      free` (`mem_readout`), so the memory picture is ambient, not just at load time.
- [ ] **Tutorial + RFC refresh** — document the 1.20.0 memory model + the loop ergonomics.

## Carry

- [⏸] **Flux in the UI** — hardware-blocked (see 1.20.0). Seam exists in `ModelService`.

> Pulled from the 16-item improvement brainstorm; scope per cycle by what lands cleanly.
