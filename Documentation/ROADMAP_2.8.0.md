# plakat 2.8.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). **Theme: finish the curate-&-finish arc + start the
collection-manager plumbing in earnest.** 2.7 shipped aesthetic scoring, PAG-for-workhorses, and
synced the guidance bundle across CLI/scenarios/TUI; 2.8 closes the remaining sync gaps, picks up a
Tier-2 restoration/reach item, and begins the 3.x manager's data layer. Scope with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.7 — close the loops

- [ ] **Aesthetic score → gen PNG metadata** (small; the persistent manager sort key). Write the score
      into the generation `.json` sidecar at gen time — but only when scoring already happens
      (`--keep-best` / a new `--score` flag), so it never loads the scorer on every generate.
- [ ] **Scenario remainder:** `keep-best` (scenario-global post-process pruning each task's outputs
      by aesthetic score) + **diffusion-upscale as a task kind** (`upscale --diffusion` isn't a scenario
      `type:` yet) + per-task guidance-bundle overrides (currently scenario-global only).
- [ ] **`plakat ui` remainder:** the diffusion upscaler + `rank`/`--keep-best` as UI actions; surface
      the older subsystems the TUI never got (map, compose/segment, relight, style/embedding train).

## Tier 2 — pick one (still open from 2.7)

- [?] **Face/detail restoration (GFPGAN or CodeFormer)** — SCRFD detect + ArcFace align (both present)
      → restore → paste back. Pairs with `upscale --diffusion` + `multiperson`. High effort (GAN/VQ
      port + weights). **or**
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
