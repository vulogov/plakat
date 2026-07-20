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

## Track B — review conflicts — DONE

- [x] **Conflict review pane** (`:conflicts`) — `save_album` records each merge conflict (path +
      other editor) into `App::conflicts`; a modal list shows "file ← also edited by … (time)".
      **Enter** jumps to the image (opening its album), **`t`** takes theirs (adopts the on-disk
      record via `conflict_take_theirs`), **`c`** clears, **Esc** closes. The save warning now points
      to `:conflicts`.

## Track C — presence — DONE

- [x] **Presence heartbeat** — `src/photos/presence.rs`: each instance writes
      `<root>/.plakat_presence/<pid>-<who>.json` (`Presence{who,pid,at}`), refreshed every ~30 s
      (`tick_presence`) and removed on exit (`depart`). `live()` returns peers whose heartbeat is
      within `TTL_SECS` (90 s) — stale ones (crashed instances) age out and are cleaned up. Other
      instances are counted in the status bar (`👥 N`) and listed by `:who`. 2 tests.

## Track D — polish (backlog)

- [x] **Slideshow: shuffle + rating-weighted dwell** — `r` toggles random order while the slideshow
      runs (a fresh permutation each loop, `rand`), and higher-rated images linger longer
      (`slide_dwell_factor`: 0★ = 1×, 5★ = 2.5×). A `🔀` marker in the `▶` badge.
- [x] **EXIF write-back: keyword tags** — the record's `tags` are written as **XPKeywords** (0x9C9E,
      UTF-16LE, `;`-separated) alongside the other fields in `d w`, so tags travel to other tools.
      (WebP/TIFF EXIF write-back stays deferred — those need container/IFD rewrites.)
- [x] **Presence album context** — each heartbeat carries the open album (relative to root); `:who`
      now reads e.g. `alice@box (in Iceland)`, so you can see not just who's here but where.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- Shared-volume safe: all writes go through the lock-free three-way merge (+ `flock` fast-path).
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
