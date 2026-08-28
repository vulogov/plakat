# RFC UI-GALLERY-1 — History quick-actions in `plakat ui` (6.24.0)

**Status:** draft (6.24.0). An **improvement** cycle on the `plakat ui` History tab. **Reality check:** the
thumbnail **grid** (`v` toggles), **filter** (`/`), and **semantic search** (Tab, TF-IDF) already exist —
so the gap the "gallery" ask really points at is **quick-actions**: today the only thing you can do to a
selected frame is `Continue` it in Chat. Add direct actions so History is a real workbench, not just a
browser.

## What ships

### P1 — Naturalize the selected frame (`n`)
Weight-free de-slop of the selected image **in place** (writes `<stem>_naturalized.png` next to it, then
rescans so it appears) — instant, no model, reusing the `naturalize` core (the same pass as the Naturalize
tab / CLI, `photo` preset). Non-destructive.

### P2 — Vary the selected frame (`y`)
Queue **variations** of the selected frame via the App's existing `vary_frame` (re-render at fresh seeds
from the frame's embedded recipe) — the History equivalent of Chat's vary, without switching tabs.

### P3 — Delete the selected frame (to trash)
`Delete` moves the selected frame to `<out_dir>/.trash/` (recoverable, not a hard unlink), with a
**single confirm** (press twice — the status line asks). Then rescan.

### P4 — docs + cut 6.24.0
Update the History F1 help + UI tutorial; README note. Cut 6.24.0 (bump Cargo+lock, gate
`--test-threads=1`, turbofish on new `.parse()`, FF main, tag → CI, publish, notes, **verify Windows**,
NO Claude coauthor).

## Honest limits
Naturalize is the weight-free pass (the reliable headline — model repair isn't wired here). Vary needs the
frame to carry an embedded recipe (no recipe → a friendly no-op, as in Chat). Delete is a move-to-trash,
not a purge — the `.trash/` dir is the user's to empty.

## Sequencing
**P1** naturalize (self-contained) → **P2** vary (reuse `vary_frame`) → **P3** delete-to-trash → **P4** cut.
