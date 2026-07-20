# plakat 3.12.0 — roadmap (in progress)

**Theme — scale: the derived index.** `album.hjson` stays the authoritative, human-editable source of
truth. 3.12 adds a **derived, rebuildable index** so the manager stays fast on large libraries: a
persisted snapshot of every image's curation + EXIF, incrementally synced against disk, with
smart-albums and search running over the in-memory index instead of re-walking every album each time.

Backend: **pure-Rust in-memory + a serde snapshot** (no new C dependency, no release-CI risk). A
SQLite backend could slot in later behind the same query surface if out-of-core storage is ever needed.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — the index core

- [ ] **`src/photos/index.rs`** — `LibraryIndex`: an entry per image (`path`, `album`, the cached
      `ImageRecord`), plus per-album `(mtime, len)` stamps for `album.hjson` and the directory.
      `sync(dirs)` reconciles incrementally — untouched albums are skipped, changed/added ones
      re-read, removed ones dropped. Snapshot save/load (serde JSON under XDG cache, keyed by root).
      Pure + unit-tested.

## Track B — integration & payoff

- [ ] **Load on startup + incremental sync** — the app loads the snapshot (fast cold start) then syncs
      only changed albums; `collect_library` reads from the index instead of walking + parsing every
      `album.hjson` on each smart-album / search build. Snapshot saved on exit.
- [x] **`:reindex`** — force a full rebuild from the authoritative `album.hjson` files (the index is
      always safe to delete + rebuild).
- [x] **Tree counts from the index + faster cold-start walk** — `library::walk` now does **one
      `read_dir` per directory** (was three: count + subdir-probe + child-list) — the cold-start
      bottleneck. `LibraryIndex::counts()` gives the per-album image count; `refresh_tree_counts`
      updates the tree badges from the index after each full `collect_library` sync, so counts reflect
      images added/removed (import, generate, move) without a fresh directory scan.
- [x] **Matched-only query** — `LibraryIndex::filter(pred)` clones **only the matching rows** (a smart
      album matching 200 of 50k images clones 200, not the whole library); `App::query_library(q)`
      syncs then filters in place. `open_smart` + `materialize_smart` route through it (search /
      lookalike still take all rows, as they rank everything).
- [x] **`:stats`** — aggregate library facets computed from the index in one pass (`LibraryIndex::stats`
      → `IndexStats`): total images / albums, a rating histogram (with bars), flagged / rejected /
      tagged / geotagged counts, mean aesthetic score, top cameras, and the capture-year span. Shown in
      a dismissable overlay — instant at any library size.

## Track C — fast visual search (persist CLIP)

- [x] **Proactive library embed** (`:embed`) — CLIP image embeddings are already persisted per album
      (`.plakat_clip`) + reloaded across sessions, but only computed *lazily on first search*. `:embed`
      pre-computes + persists them for the **whole library** up front (`visual_search::embed_all`,
      lazy model load — fully-embedded → offline), so the first visual search / lookalike is instant.
      Reuses the seed-from-disk + save-to-disk path; TUI-suspended with progress. 1 offline-path test.
- [x] **CLIP vectors folded into the index store** — a compact **binary** vector sidecar
      (`<snapshot>.vec`, `LibraryIndex::save_vectors` / `load_vectors`) beside the JSON snapshot (768
      f32/image would bloat JSON). The whole library's embeddings load in **one read at startup** (seeds
      `clip_cache`), and are re-persisted on exit + after `:embed`/search. 1 round-trip test.

## Track D — keep the index hot

- [x] **Incremental update on save** — `LibraryIndex::update_album` refreshes just the edited album's
      entries in place from the merged `album.hjson` (no directory scan) and records the album.hjson
      stamp, so an in-app edit is reflected in the index **immediately** and the next smart-album /
      search build doesn't re-read that album. Wired into `save_album` (open albums; smart views edit
      per-album). Leaves the directory stamp so a later file add/remove still re-syncs. 1 test.

## Track E — index-backed browse

- [x] **All Photos grid** (`:all`) — the whole library in one grid, sourced straight from the index
      (no re-parse of every `album.hjson`). On a **warm launch** (a persisted snapshot exists) it opens
      automatically as the cold-start view, so a large library is browsable immediately; first-ever run
      (empty snapshot) stays on the tree. Curation routes to each source album; `/` filters live.

## Ground rules

- The index is **derived + non-authoritative**: delete it and it rebuilds from `album.hjson`.
- Shared-volume-safe: stamps pick up another instance's writes; nothing bypasses the three-way merge.
- Verify-safe: default CLI image output byte-identical; the index core lands with unit tests.
