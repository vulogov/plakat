# RFC PHOTOS-PEOPLE-1 — plakat photos: people, hybrid search, quality-aware keeper (6.26.0)

**Status:** SHIPPED (6.26.0). An **improvement** cycle on `plakat photos`. The survey found the
manager already very complete — near-dup with keeper-selection (`dedup.rs`), ArcFace people
**clustering** (`faces.rs` → `person-N` tags), Laplacian **sharpness** + exposure cull
(`quality.rs`), HNSW **visual search** (`ann.rs`), and a rich faceted **filter** grammar
(`matches_filter`: rating/tag/date/iso/camera/gps…). So 6.26.0 fills the four remaining holes:

## P1 — People management
Face clustering produced opaque `person-N` tags with no human-in-the-loop. Added a `people`
command (typed in the NL command box):
- **`people list`** — every person cluster + image count.
- **`people rename <from> <name>`** — retag a cluster across the whole library (`person-3` → `alice`),
  persisted per record; `tag:alice` then browses that person.
- **`people merge <a> <b>`** — fold cluster `a` into `b` (a rename onto an existing name).

Core is a pure, tested `faces::rename_tag(tags, from, to)` (case-insensitive, de-duping); the
manager applies it library-wide via `edit_record_at` + `rebuild_view`.

## P2 — Hybrid / faceted search
Perceptual **lookalike** search scanned the whole library, ignoring the active filter. Now it
**honours the current facets**: with a filter active (`tag:beach date>=2024` / `tag:alice` /
`rating>=4`), lookalike ranks only the matching subset — "find similar to THIS, but only among
these". The query image is always kept in the set. `matches_filter` (the existing grammar) is
reused, so every facet composes with visual similarity.

## P3 — Quality-aware dedup keeper
The near-dup keeper picked by `(rating, aesthetic)` and ignored sharpness. Now it ranks by
**rating → sharpness → aesthetic**, so the auto-kept frame is the **crispest** of the group.
`dedup_scan` loads one 256px thumbnail per image and computes both the dHash and
`quality::sharpness` in the same pass. Pure `dedup::pick_keeper(&[KeeperKey])`, tested.

## P4 — Person-aware quality (`soft-face`)
New `quality::region_sharpness(img, boxes)` — the Laplacian variance measured **only inside the
face boxes**, so a sharp background with a soft face scores low. The **face scan reuses the boxes
it already detects** (no extra model pass) to compute per-image face-region sharpness, and flags
frames below **40% of the library median** with a **`soft-face`** tag — adaptive to the library's
scale. Filter `tag:soft-face` to find blurry-face shots (and `-tag:soft-face` to keep the good
ones). `region_sharpness` is tested; the scan integration is best-effort (relative floor, ≥4
scored faces needed to arm it) and flagged for live validation.

## Honest limits
- People rename stores the name as a lowercased tag (matches the case-insensitive filter grammar);
  there's no separate person database or per-person contact-sheet pane yet — `tag:<name>` is the
  browse. Merge/rename are the human-in-the-loop primitives; a visual People pane is a follow-up.
- `soft-face` uses a relative floor (library median × 0.4); it flags the *relatively* blurriest
  faces, not an absolute focus judgment. Needs ≥4 face-images to activate.

## Sequencing
**P3** keeper (pure, self-contained) → **P4** `region_sharpness` + scan `soft-face` → **P1** people
rename/merge/list → **P2** faceted lookalike → cut 6.26.0 (bump Cargo+lock, gate
`--test-threads=1`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).
