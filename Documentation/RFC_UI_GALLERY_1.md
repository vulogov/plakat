# RFC UI-GALLERY-1 — History quick-actions in `plakat ui` (6.24.0)

**Status:** SHIPPED (6.24.0). An **improvement** cycle on the `plakat ui` History tab. **All shipped:**
P1 naturalize (`n`) · P2 vary (`y`) · P3 delete-to-trash (`Delete`, confirm). (Grid/filter/semantic search
already existed.)

**Follow-ups (same cut) — owner steer "no new tabs; put relight/faceswap in existing tabs":** V1 History
`e` etch (sync provenance) · V1 Chat **`/relight <preset>`** + **`/faceswap <src> [N]`** editing the latest
frame (reuse the `active_gen` channel; evict the t2i model first for Metal single-device exclusivity — the
heavy-op path needs live model validation) · V3 palette + F1-help discoverability. Also FIXED 2 stale ui
nav tests (asserted 8 screens; broken since the 6.17 Naturalize tab made 9 — the gate excludes `ui` so it
went unnoticed). **Reality check:** the
thumbnail **grid** (`v` toggles), **filter** (`/`), and **semantic search** (Tab, TF-IDF) already exist —
so the gap the "gallery" ask really points at is **quick-actions**: today the only thing you can do to a
selected frame is `Continue` it in Chat. Add direct actions so History is a real workbench, not just a
browser.

**Round 2 (same 6.24.0 cut) — "improve `plakat ui`, no new tabs" (W1–W4):**
- **W1 — more Chat edit commands** on the latest frame, reusing the `active_gen` channel:
  **`/etch`** (in-place provenance, weight-free), **`/upscale [2|3|4]`** (weight-free Lanczos → new
  frame), **`/remove-bg`** (U2Net matte → transparent cutout, evicts the t2i model). `/restore-faces`
  DEFERRED (needs a full SD base + SCRFD ADetailer config — a bigger wire; own follow-up).
- **W2 — History multi-select + batch:** `Space` toggles a frame into the selection (✓ marker); then
  `n`/`e`/`Delete` act on the **whole set** (naturalize / etch / trash a batch). Weight-free and sync →
  fully unit-tested (`multi_select_batches_the_weight_free_ops`).
- **W3 — settings surfaced in Chat:** `/settings` reports device / default+loaded model / output dir /
  LoRAs / auto-route / etch-on-gen; **`/settings etch on|off`** toggles provenance-on-generate (finished
  Chat frames gain an EtchId in the `Done` handler — alpha images like cutouts are skipped so
  transparency survives). No new tab. Tested (`settings_reports_and_toggles_etch`).
- **W4 — preview maximize:** **Ctrl-F** gives the whole middle row to the image (hides the transcript
  column) for a bigger look; Ctrl-F again restores the 55/45 split. Tested
  (`ctrl_f_toggles_preview_maximize`).

## What ships (round 1)

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
