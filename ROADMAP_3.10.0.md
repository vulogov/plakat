# plakat 3.10.0 — roadmap (in progress)

**Theme — collaboration, deepened.** 3.9 made a shared library safe and showed *who* last touched a
record. 3.10 builds on that: a rolling **edit history** per image, an interactive **conflict review**
pane, and a **presence** heartbeat so you can see which instances are live.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — per-image edit history

- [x] **Rolling edit history** — beyond the single `last_editor`, a bounded per-record log
      (`history: Vec<EditNote{by,at}>`, cap `HISTORY_CAP`=12). `merge_album_stamped` unions our history
      with the disk record's (so a concurrent editor's entries survive), appends our new entry, sorts
      by time and caps. Shown in the info panel (latest + up to 3 recent `↳ who · when`). 1 test.

## Track B — review conflicts

- [ ] **Conflict review pane** — accumulate conflicts surfaced by the merge; a modal list (file +
      "also edited by …" + when) reachable from the status warning. Enter jumps to the image; a
      "take theirs" action adopts the disk record for that file; Esc closes.

## Track C — presence

- [ ] **Presence heartbeat** — each instance writes a small `<root>/.plakat_presence/<id>.json`
      heartbeat (editor id + pid + time), refreshed periodically and removed on exit. Live peers
      (fresh heartbeats) are counted in the status bar and listable (`:who`). Stale entries (crashed
      instances) age out.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- Shared-volume safe: all writes go through the lock-free three-way merge (+ `flock` fast-path).
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
