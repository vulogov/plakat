# plakat 3.5.0 — roadmap (planning)

Opening the cycle after 3.4.0 (the photos "full studio" — a deep non-AI creative/finishing pass +
composites). The non-AI editing surface is now very complete. With the **models cache reconnected**,
3.5 is the natural point to (re)open the **AI track** inside `plakat photos`, alongside a few
remaining non-AI polish items and the distribution loop. Candidate tracks — narrowed with the owner
before build.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — AI in the manager (cache now connected)

- [ ] **CLIP visual search — live-verify.** The cache disk is back; run the ignored test and confirm
      the in-tree CLIP path end-to-end: `cargo test --features photos -- --ignored clip_loads_and_embeds`.
      Then exercise `V` (text→image) + `Ctrl-B L` (image→image) on a real album.
- [ ] **Face-scan** — detect/group faces across the library (SCRFD + ArcFace are already in-tree);
      surface as a People view / smart grouping.
- [ ] **Analyze-and-generate** — turn a reference image's analysis into a generation recipe (the
      last Phase-7 item).
- [ ] **Aesthetic auto-cull** — already have `rank`/`--keep-best`; wire a one-key "rank + keep top N"
      curation pass into the manager.

## Track B — remaining non-AI polish

- [ ] **EditOp `Copy` refactor** → makes watermark/LUT replayable edits (currently file-ops) and
      unblocks text/path-carrying ops in the edit log.
- [ ] **True panorama stitch** (feature-matched alignment) as an optional upgrade to the current
      concatenation stitch; **mosaic/scrapbook collage** (varied cell sizes).
- [ ] Quick creative leftovers: **gradient map / tritone**, **cross-hatch**, **Kelvin white balance**,
      **crystallize (Voronoi)**.

## Track C — distribution & housekeeping

- [ ] **Publish 3.3.0 + 3.4.0** — `cargo publish` (crates.io) + GitHub release assets (deferred at
      both cuts).
- [ ] **Merge 3.3.0 / 3.4.0 → `main`.**
- [ ] Carry-throughs from the 2.4 performance pass.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` authoritative + additive.
- The `:` planner stays inside the closed, album-scoped vocabulary — `export`/`convert` the only
  create-only outward writes; no external read, no exec. (Watermark/LUT read a named file, so they
  stay palette-only.)
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
