# Face-swap (`plakat faceswap`)

Swap the face(s) in an image, folder, or video with a **source** face — SCRFD 5-point align → ArcFace
identity → `inswapper_128` → colour-matched paste-back. It edits media **you already have** (no scene
generation). The weights are **non-commercial** (InsightFace) and download on first use (opt-in).

> Identity transfer works best on clear, reasonably-sized faces. Small / occluded / profile faces may be
> missed by the detector or transfer weakly — the honest limit of a 128² identity model.

## 1. Look before you swap

Pick the right face without guessing — this is fast (detector only, no big download):

```bash
plakat faceswap group.jpg --dry-run
#  ✓  5 face(s) in group.jpg:
#    [0] bbox 536,424 → 560,455  score 0.84
#    [1] …
plakat faceswap group.jpg --dry-run --preview boxes.png   # draws colour-coded numbered boxes
```

## 2. Swap

```bash
plakat faceswap photo.jpg --source alice.png                 # the largest face
plakat faceswap photo.jpg --source alice.png --face 1        # the 2nd-largest (see --dry-run)
plakat faceswap group.jpg --source alice.png --all           # every face ← alice
plakat faceswap photo.jpg --source alice.png --restore       # + a light detail pass
```

Output defaults to `<scene>_swapped.<ext>` (`--out` to override).

## 3. Many people, many identities

`--source` is **repeatable**. Each source is matched to its **closest detected face by recognition**
(ArcFace) — so identities follow the face, not its size, even in a crowd:

```bash
plakat faceswap party.jpg --source alice.png --source bob.png --source carol.png
plakat faceswap party.jpg --source alice.png --source bob.png --report   # print who matched whom
```

`--match rank` forces the older size-order mapping; `--source-face N` picks the identity from a *multi-face*
source photo.

## 4. Batch and video

```bash
plakat faceswap ./scenes/ --source alice.png --out ./swapped/   # a whole folder
plakat faceswap clip.mp4  --source alice.png --out clip.mp4     # every frame (needs ffmpeg)
```

## 5. Dialing the blend

- `--feather PX` — paste-back edge softness (default 16).
- `--no-color-match` — paste the raw swap (skip the skin-tone match to the target).
- `--etch` — write plakat provenance into the output (verify later with `plakat doctor --if-plakat`).

## Beyond the CLI

Face-swap is reachable from every surface:

- **Library** — `plakat::api::FaceSwap::new("scene.png", "alice.png").face(0).run().await?` → `Image`.
- **Scripts (Bund)** — `plakat.faceswap ( scene source out -- handle )`.
- **Pipelines (scenario)** — a `type: faceswap` task (`scene` + `source` + optional `face`).
- **Prose (`compile`)** — a `type: faceswap` block (`faceswap-scene:` / `faceswap-source:`).

## Honest limits

128² identity transfer (not high-frequency skin detail; `--restore` helps but can slightly drift
identity). Per-face mapping is by recognition, but a genuinely ambiguous match can still land wrong. Video
has no temporal smoothing yet — fast motion can shimmer. **Weights are non-commercial (InsightFace)** —
personal / research use.
