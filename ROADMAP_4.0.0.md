# plakat 4.0.0 — roadmap (in progress)

**The major cut — `plakat photos` is done.** The 3.x line built the flagship TUI photo/image manager
end to end (organize · edit · AI · collaborate · output · maps · formats · scale). 4.0 is the
**confidence** release: the visual-search-at-scale follow-ups, then a hardening / consistency / docs
pass — no new pillars, just making the flagship solid enough to put a "1.0-grade" 4.0 stamp on.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track V — visual-search polish (the 3.13 follow-ups) — DONE

- [x] **Lookalike through the ANN** — CLIP image→image lookalike now routes through the HNSW ANN above
      `ANN_THRESHOLD` (embed library + query, then `ann_search`), keeping the resident-model linear
      scan for smaller libraries and the fully-offline path when everything's cached.
- [x] **Persist the HNSW graph** — `instant-distance` serializes `HnswMap` (its `with-serde` feature);
      `AnnIndex::save`/`load` snapshot it (compact **bincode**) beside the index snapshot (`.hnsw`),
      keyed by vector count. `ann_search` loads the persisted graph if it matches, else builds once +
      saves — so a huge library skips the O(N log N) rebuild after the first time. 1 round-trip test.
- [x] **Search UX** — the status line now shows the **best similarity score** + whether the **ANN** or
      linear **scan** ran (both text search + lookalike).

## Track C — consolidation / stability (4.0 readiness) — DONE (first pass)

- [x] **Clippy + bug sweep** on the `photos` module — clippy surfaces **no correctness / robustness
      findings** across `photos` (only style/pedantic lints: complex types, clamp patterns, sort_by_key,
      redundant casts) — a good signal the module is solid. Fixed the genuine tidy-ups: completed the
      `AnnIndex` API (`is_empty`) and removed two dead same-type casts (`hjson`, `stitch`).
- [x] **Docs currency** — KEYMAP carries the 3.10–3.13 additions (the "Library commands" `:` section
      for `:all`/`:stats`/`:reindex`/`:embed`/`:conflicts`/`:who`, the "Shared volumes & multiple
      instances" section, HEIC + EXIF-write-back notes, brush masks, slideshow), maintained each cycle.

## Track R — 4.0 readiness (hardening + consistency + docs)

- [ ] **Robustness soak** — lock in graceful handling of adversarial / edge inputs with tests: corrupt
      / truncated images, an empty or missing library, malformed `album.hjson` / index snapshot /
      `.plakat_clip` / `.hnsw`, weird filenames, zero-byte files. Fix any panic found.
- [ ] **Command-surface consistency audit** — every `nl::Action` has a dispatch arm and a parseable
      verb; no orphaned commands / dead ends. Programmatic check + spot review.
- [ ] **Tutorial + feature tour** — a complete `Documentation/Photos/TUTORIAL.md` walking the whole
      manager (organize → edit → AI → collaborate → output → search-at-scale), current through 4.0.
- [ ] **The 4.0 stamp** — README "what's new" tells the flagship's story; RELEASE_HISTORY archives
      3.13; full suite + clippy-clean gate; declare 4.0.

## Ground rules

- Non-destructive; `album.hjson` authoritative; the index + ANN are derived + rebuildable.
- Verify-safe: default CLI image output byte-identical; new logic lands with unit tests.
- `Cargo.lock` committed with the version bump (the 3.11.0 lesson).
