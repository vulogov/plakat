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

## Track B — metadata

- [ ] **EXIF / IPTC editor** — edit **capture date** (timezone/clock fixes), **copyright / author /
      title**, and **geotag** (lat/lon), written to the sidecar + (where lossless) the file.
- [ ] **EXIF-based smart-album fields** — expose shot metadata (ISO, focal length, camera, date) to the
      filter grammar so smart albums can select on them.

## Track C — editing

- [ ] **Local masked adjustments** — apply *any* tonal/colour adjustment through a **linear-gradient /
      radial / brush** mask (the Lightroom local-adjust model), generalising the fixed graduated-ND /
      radial dodge-burn ops. Interactive mask placement reuses the crop / pick-mode patterns.

## Stretch

- [ ] **Persistent thumbnail cache** — a per-library on-disk thumb store so big libraries open fast.
- [ ] **Map / geo view** — cluster photos by GPS; reverse-geocode to place tags. The "wow" manager
      feature, fully non-AI.

## Deferred

- [ ] Distribution is handled per-cut (crates.io + GH release via the tag-triggered CI + FF `main`).

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays a closed, album-scoped vocabulary; no external read, no exec.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
