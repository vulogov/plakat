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

## Track C — (stretch) fast visual search

- [ ] Persist CLIP embeddings alongside the index (a compact binary vector store keyed by path), so
      visual / semantic search doesn't re-embed the library each time. (May slip to a later cycle.)

## Ground rules

- The index is **derived + non-authoritative**: delete it and it rebuilds from `album.hjson`.
- Shared-volume-safe: stamps pick up another instance's writes; nothing bypasses the three-way merge.
- Verify-safe: default CLI image output byte-identical; the index core lands with unit tests.
