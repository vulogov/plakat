# plakat 2.8.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). **Theme: finish the curate-&-finish arc + start the
collection-manager plumbing in earnest.** 2.7 shipped aesthetic scoring, PAG-for-workhorses, and
synced the guidance bundle across CLI/scenarios/TUI; 2.8 closes the remaining sync gaps, picks up a
Tier-2 restoration/reach item, and begins the 3.x manager's data layer. Scope with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.7 — close the loops

- [x] **Aesthetic score → gen PNG metadata** (6673030). `generate --score` (+ `--keep-best`) scores
      each output and writes it into the `.json` sidecar via `io::patch_sidecar_score`; `rank --write`
      (4a07cf5) does the same for existing folders. Optional `score` field on GenerationMetadata
      (additive). Never loads the scorer unless a flag is set. The manager's on-disk sort key.
- [ ] **Scenario remainder:** `keep-best` (needs per-task integration across the scattered generate-
      dispatch arms — a whole-scenario keep-best would delete whole tasks' outputs; deferred, not a
      quick win) + **diffusion-upscale as a task kind** + per-task guidance overrides.
- [ ] **`plakat ui` remainder:** the diffusion upscaler + `rank`/`--keep-best` as UI actions; surface
      the older subsystems the TUI never got (map, compose/segment, relight, style/embedding train).

## Tier 2 — pick one (still open from 2.7)

- [x] **Face/detail restoration** (1c89c09) — `plakat restore-faces`. Delivered via the diffusion
      approach (the existing ADetailer engine: SCRFD-detect → img2img-refine each face → feather-
      composite) as a STANDALONE command for existing images, instead of a GFPGAN/CodeFormer GAN port
      — reuses `refine_files`, pairs with `upscale --diffusion`. Validated: refined a portrait's face
      cleanly, seamless composite. (A dedicated GAN restorer stays available as future work if a
      non-diffusion path is wanted.)
- [x] **sd35 T5-XXL memory relief** (0213e31) — DONE. Low-mem mode: T5-XXL and the MMDiT are lazy +
      droppable and never co-resident (T5 encodes → freed → MMDiT loads → denoises), dropping peak from
      sum to max (~17→12 GB). `PLAKAT_SD3_LOWMEM` (auto on Metal when RAM < 22 GB). **Validated**:
      sd35-medium generated a clean detailed fox @512 on Metal — where every attempt OOM'd all the 2.6
      cycle. Unblocks SD3.5 on 24 GB **and** SD3-PAG calibration. Limitation: runtime LoRAs skip in
      low-mem (warns).

## Road to 3.0 — first real manager plumbing

- [ ] **Collection index (design + first slab).** Define what a "collection" is and how it's stored
      (SQLite or a JSON store): per-image prompt/params/seed/model + aesthetic score + tags, indexed for
      the History semantic search + the 3.x TUI collection manager. The metadata sidecars + scores are
      the raw material; give them a queryable home. See ROADMAP_TO_3.0.md.

## House-keeping

- [x] **Open 2.8.0** — branch off `main` (2.7.0 release), version bump `2.7.0 → 2.8.0`.
