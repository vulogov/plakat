# plakat 3.14.0 — roadmap (in progress)

**Theme — polish & consolidate (toward 4.0).** Two threads: finish the visual-search-at-scale story
(3.13 follow-ups), and a stability / polish / docs pass to harden the 3.x flagship.

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

## Ground rules

- Non-destructive; `album.hjson` authoritative; the index + ANN are derived + rebuildable.
- Verify-safe: default CLI image output byte-identical; new logic lands with unit tests.
- `Cargo.lock` committed with the version bump (the 3.11.0 lesson).
