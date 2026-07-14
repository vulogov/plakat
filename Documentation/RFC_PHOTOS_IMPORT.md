# RFC PHOTOS-IMPORT — `--import` for the generation commands

**Status:** Plan (implementation-ready) · **Depends on:** RFC PHOTOS-1 (`src/photos/hjson.rs`)

## Goal

Close the **generate → collection** loop. Any image-producing command gains `--import <ALBUM>`: each
output is copied into the target photo album, and its generation parameters (the existing
`GenerationMetadata` — prompt, model, seed, steps, guidance, loras, …) are written into the album's
per-image HJSON record. Generate once, and the result is already curated in the manager.

## Which commands

Every command that writes an image: `generate`, `upscale`, `portrait`, `multiperson`, `img2img`,
`outpaint`, `stylize`, `relight`, `compose` (and later `animate` frames / `map`). Each gains a
`--import <ALBUM_DIR>` flag (and optional `--import-move` to move instead of copy).

## Mechanism (non-invasive post-step — mirrors `generate --score`/`--keep-best`)

The generation flow is unchanged; import is a **post-generation step** over the outputs:

1. **Snapshot** the output dir before generation (the pattern already in `generate::run`), so only
   this run's new files are imported. (Commands that write a single known `--out` path skip the
   snapshot and pass that path directly.)
2. After generation, call `photos::import::import_outputs(album, &new_files, move_files)`.

## `import_outputs(album, files, move_files) -> Result<usize>`

For each output file:
1. **Album** — create `<album>` if missing (`mkdir` + empty `album.hjson`).
2. **Copy/move** the PNG into `<album>/`, dedup the name if it collides (`name-2.png`). Copy the
   `.json` sidecar alongside too (so the standalone recipe travels).
3. **Read params** — `GenerationMetadata` from the sidecar, else the PNG `parameters` tEXt chunk via
   `imaging::io::read_parameters_chunk`. Carry the aesthetic `score` if the sidecar has one.
4. **EXIF** — `photos::exif::read_exif` (usually empty for generated PNGs, but free + uniform).
5. **Record** — read `album.hjson`, set `images[name] = ImageRecord { generation: Some(meta), score,
   exif, ..existing }` (merge if the name already had a record), atomic-write `album.hjson`.

Returns the number imported. Best-effort per file (one failure doesn't abort the batch; logged).

## Data-model change

Add one additive field to `hjson::ImageRecord`:

```rust
/// v3.x: generation parameters when this image was produced by plakat (`--import`).
#[serde(default, skip_serializing_if = "Option::is_none")]
pub generation: Option<crate::imaging::metadata::GenerationMetadata>,
```

`GenerationMetadata` is already `serde` + additive, so it embeds cleanly and the manager's EXIF panel
can show a `plakat` block (prompt/seed/steps/CFG) directly from the record — no PNG re-parse.

## Live synergy with `plakat photos`

If the manager is open on the target album, the `notify` watcher already picks up the new file and
rescans → the imported image appears in the grid **live**, with its gen params + score in the record.
Generate in one terminal, watch it land curated in another.

## Edge cases

- `--count N`: import every output.
- Album vs folder: importing into a folder turns it into an album (it now holds images) — consistent
  with the contents-based classification.
- Concurrent `album.hjson` writes (photos open + import): both use atomic `.tmp → rename`; import
  re-reads immediately before writing, so it merges rather than clobbers (last-writer-wins per record).
- `--out` + `--import`: both allowed — output lands in `--out` as usual **and** a copy is imported
  (or moved, with `--import-move`, leaving only the album copy).

## Implementation order

1. `ImageRecord.generation` field + `src/photos/import.rs` (`import_outputs`) — behind `--features
   photos` (imports gate on the photos module; the flag is a no-op/error without it).
2. Wire `--import` into `generate` (reuse its existing before/after snapshot wrapper).
3. Wire into `upscale` / `portrait` / `multiperson` / `img2img` / `outpaint` / `stylize` / `relight`
   (each has a known output path → call `import_outputs` directly; no snapshot needed).
4. Tests: `generate --import <album>` → `album.hjson` carries the record with `generation` populated;
   the file is in the album; a second import dedups the name.
