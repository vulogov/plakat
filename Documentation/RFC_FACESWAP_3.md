# RFC FACESWAP-3 — face-swap depth: dry-run · batch · per-face sources · video (6.21.0)

**Status:** SHIPPED (6.21.0). Depth on the 6.20.0 `plakat faceswap` verb.

**All phases shipped:** P1 `--dry-run`/`--preview` (live-proven) · P2 directory batch · P3 repeatable
`--source` · P4 video (every frame). `plan_swaps` mapping unit-tested.

**Controls / parity / provenance (same cut):** S2 swap controls — `--feather PX`, `--no-color-match`,
`--match rank|identity`, `--report` (print source→face + cosine); FaceSwapper gained `feather`/`color_match`
fields · S3 Bund **`plakat.faceswap`** word (`scene source out -- handle`, via `script_entry::faceswap_one`)
— scenario `type: faceswap` + `api::FaceSwap` builder DEFERRED (kept the cut focused) · S4 `--etch`
provenance on the output (`fresh_etch` when `--etch` active); `--restore` documented as already face-scoped
(ADetailer only touches detected face boxes).

**Additional improvements (same cut):** R1 **recognition-matched** sources — each `--source` matched to its
closest face by ArcFace cosine (`match_sources_to_faces`, greedy; identity-follows-face, robust to size
reorder across frames), replacing the size-rank heuristic for multi-source · R3 `--source-face N` (pick the
identity face in a multi-face source photo) · R4 small-face sharpen (unsharp the 128² target crop when
inter-ocular < 40px so inswapper locks onto structure — marginal, documented). R1 helpers unit-tested. All four reuse the proven engine
(`FaceSwapper`: detect → `source_latent` → colour-matched `swap_into`); no new model.

## What ships

### P1 — `--dry-run` / preview
Pick `--face N` without guessing: `plakat faceswap <scene> --dry-run` detects the faces and prints them
largest-first with index, bbox, and score — no swap, no source needed. `--preview <path>` also writes the
scene with **numbered boxes** drawn so you can see which index is which face.

### P2 — batch (a folder of scenes)
`<scene>` may be a **directory** → swap every image in it into the `--out` directory (same source(s),
`--face`/`--all` semantics per image). Skips images with no detected face (reported), so a folder pass is
one command. Mirrors `naturalize`'s batch shape.

### P3 — per-face different sources
`--source` becomes **repeatable**. With one source: today's behaviour (`--face N` or `--all`). With **K
sources**: map the i-th source to the i-th **largest** face (source[0]→largest, source[1]→next, …), so a
group photo gets each person their own identity in one pass. Extra faces beyond K are left untouched
(reported).

### P4 — video face-swap
A **video/animation** input (mp4/mov/webm/mkv/avi/gif) → detect + swap **every frame** and re-encode
(reusing `imaging::video`, the naturalize path). Detection is per-frame (faces move); the same
source-mapping rules as P3 apply per frame. Needs `ffmpeg`. Honest: per-frame independent swaps can
shimmer slightly on fast motion (no temporal smoothing this cut).

### P5 — docs + cut 6.21.0
Tutorial + README; corpus. Cut 6.21.0 (bump Cargo+lock, gate `--test-threads=1`, turbofish on new
`.parse()`, FF main, tag → CI, publish, notes, **verify Windows**, NO Claude coauthor).

## Honest limits
inswapper is 128² identity transfer (not skin-detail); small/occluded/profile faces may miss (SCRFD) or
transfer weakly. Per-face mapping is by **size rank**, not recognition — if two faces swap size order
between frames the identities can swap too (documented; a recognition-matched mapping is a future step).
Video has no temporal smoothing yet. Weights non-commercial (InsightFace).

## Sequencing
**P1** dry-run → **P2** batch → **P3** per-face sources → **P4** video → **P5** cut. P3 (repeatable
`--source`) lands before P4 so video inherits the mapping.
