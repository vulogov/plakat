# RFC RELIGHT-1 — named lighting presets + directional control for `plakat relight` (6.23.0)

**Status:** draft (6.23.0). An **improvement** cycle on the existing IC-Light relighting. Today
`plakat relight <subject> --prompt "<free-text lighting>"` is a blunt instrument — the user must hand-write
a good lighting prompt, and IC-Light only gets *words* (the subject is composited on flat neutral grey). Make
relight a **menu with real directional control**.

## What ships

### P1 — named lighting presets (`--light <name>`)
A curated table of lighting looks, each mapping to a well-crafted **prompt + negative**:
`key-left` · `key-right` · `top` · `rim` (backlight) · `softbox` (studio) · `golden-hour` · `sunset` ·
`moonlight` · `candlelight` · `neon` · `overcast`. `--light key-left` picks the look; `--prompt` becomes
**optional** (a user prompt is appended to the preset, or used alone as today). `--list-lights` prints the
table. Reachable from api/scenario/Bund too.

### P2 — directional backdrop (real spatial control)
IC-Light responds to the *conditioning image*, not just the prompt. Instead of compositing the subject on
**flat grey**, composite it on a **directional gradient** — brighter on the light side, darker opposite —
derived from the preset's direction (left / right / top / rim / flat). This gives IC-Light a genuine
spatial cue, so "key from the left" actually lights from the left rather than hoping the words land.
`--light-angle <deg>` (optional) overrides the direction for a custom gradient.

### P3 — parity + docs + cut 6.23.0
`api::Relight::light(preset)`, scenario/Bund parity where relight already has surfaces; update
`RELIGHT_TUTORIAL.md` + README. Cut 6.23.0 (bump Cargo+lock, gate `--test-threads=1`, turbofish on new
`.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
IC-Light is a *relighting* model, not a physically-based renderer — presets steer it, they don't guarantee
a photometric result. The directional backdrop is a soft cue (the model can still override it for a strong
prompt). Presets are tuned heuristics. Subject matting quality still bounds the result (U2Net).

## Sequencing
**P1** presets (prompt table) → **P2** directional backdrop (the real control) → **P3** parity + cut.
The backdrop is the substantive quality win; presets are the ergonomic headline.
