# Genres catalog (v0.25)

Each entry in `catalog.json` describes a **subject-domain** preset.
Genres are an independent axis from looks: `--look watercolor`
selects the medium, `--genre anime` selects the subject domain.
They compose.

Loaded at startup. User-added entries from
`$CONFIG_DIR/genres/*.json` shadow bundled entries by `name`.

## Field shape (`GenreSpec`)

Same shape as `LookSpec` — see `assets/looks/README.md`. Bundled
schema is identical so a future refactor can unify them; today
they live in separate JSON files to keep CLI semantics
(`--look` vs `--genre`) explicit.

## Bundled

v0.25 ships **`anime`** only as a built-in. Other genres
(photoreal, fantasy, cyberpunk, …) live in user catalogs.

## Adding your own genre

Create `$CONFIG_DIR/genres/my-genre.json` (single object). The
loader picks it up at startup. Name conflicts shadow bundled
entries.

See `Documentation/RFC_v0.25_LOOKS_AND_GENRES.md` for the full
design.
