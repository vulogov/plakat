# plakat 3.8.0 — roadmap (planning)

Opening after 3.7.0 (retouch + the last non-AI editing gaps). The **editing** surface is deep; 3.8
turns to the **manager** side — browsing, organizing, metadata — plus one substantial editing feature
(local masked adjustments). Kicked off by a real bug the tree surfaced: a mixed album (loose images +
sub-albums) read its recursive count, hiding what "open" actually shows. Fixing that pointed straight
at the browsing gaps below.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — browse & organize

- [x] **Tree badge = an album's own image count** (not recursive) — the bug that opened this cycle
      (`fix 83928a2`). Mixed albums now read `[direct]`, sub-albums surface as `[1]` children.
- [x] **Recursive / flatten browse** (`*` in tree/grid, `:flatten`) — `open_recursive` shows **all**
      images beneath a folder / mixed album (across sub-albums) in one grid; curation routes to each
      source album via the smart-view source map.
- [x] **Move / copy selected → album** (`m o` / `m p`) — `move_targets` carries file + `.json` sidecar
      + curation record to the destination album; move also removes the source record.
- [x] **Trash / soft-delete + restore** (`m t` / `m b`, `:restore` / `:empty trash`) — soft-delete to a
      hidden `<root>/.trash` with a `.manifest`, restore-to-origin, browse, and permanent empty.

## Track B — metadata — DONE

- [x] **EXIF / IPTC editor** — edit **title / author / copyright** (new record fields) + **capture
      date** and **geotag** (into the record's cached EXIF), via the new `d` metadata chord group
      (`d t`/`d a`/`d c`/`d d`/`d g`/`d e`). Non-destructive (album.hjson); shown in the info panel
      (record-preferred over the file's EXIF). *Writing back into the file's binary EXIF is a future
      enhancement — the scrub GPS-redact machinery shows it's feasible.*
- [x] **EXIF-based smart-album fields** — `matches_filter` now supports `iso`/`focal` (numeric
      `>`/`>=`/`<`/`<=`/`=`), `camera:`/`lens:`, `date` (`>YYYY`/`<YYYY`/`:text`), `has-gps` /
      `-has-gps`, and `author:`/`copyright:`/`title:` — all read from the cached record (no file I/O).

## Track C — editing — DONE

- [x] **Local masked adjustments** — `EditOp::LocalAdjust { adjust, amount, shape, dir }`: the base
      adjustment (exposure / brightness / contrast / saturation / warmth / vibrance / definition /
      blur) is applied globally then blended back through a **linear-gradient** (from an edge) or
      **radial** (centre / edges) mask. Reuses every adjustment via `local_base_op` → the base op's own
      `apply`, then `adjust::local_mask` + `blend_masked`. Slider = amount; curated palette set (`a g`
      graduated exposure, `a i` radial exposure, + saturation/warmth/blur variants). *Parametric masks
      only (linear/radial presets); interactive placement + freeform brush masks are a future
      enhancement (they'd reuse the pick-mode + the layer image-matte).*

## Stretch

- [x] **Persistent thumbnail cache** — already existed (`loader::get_or_render_thumb`, XDG
      `<cache>/plakat/photos/thumbs`, keyed by `sha256(path + size + mtime + byte-len)` so any on-disk
      change auto-invalidates — the byte-size disambiguates same-second in-place edits). Added a
      **global clear** (`clear_thumb_cache`) to reclaim disk (per-album `regen` already existed).
- [x] **Map / geo view** — DONE. `src/photos/geomap.rs`: plot geotagged photos on ratatui's built-in
      vector world map (`canvas::Map`, **fully offline** — no tiles/network), grid `m` / `:map`;
      pan/zoom, grid-bin clustering with counts, centre crosshair, Enter → geo-filtered smart view.
      **Reverse-geocode** (`src/photos/geodata.rs`, manage palette / `:geocode`): downloads a Natural
      Earth populated-places gazetteer **once**, caches it (XDG), then works offline — tags each
      geotagged image with `place:<nearest-city>` (filter e.g. `place:tokyo`). Runtime stays offline;
      the only network step is the one-time gazetteer fetch, exactly like model weights.

## Deferred

- [ ] Distribution is handled per-cut (crates.io + GH release via the tag-triggered CI + FF `main`).

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
