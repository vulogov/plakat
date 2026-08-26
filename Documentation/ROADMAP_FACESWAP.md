# plakat face-swap — build plan — COMPLETE ✓ (+ 6.20.0 additions)

All 5 phases shipped on the 1.12.0 branch. `multiperson --swap` generates a
coherent scene and face-swaps each persona in by placement region. Identity engine
verified component-by-component vs InsightFace (SCRFD, ArcFace cosine 1.0,
inswapper 1e-5, emap 0.0). Weights hosted: SCRFD + inswapper + ArcFace on HF.

**6.20.0 (RFC FACESWAP-2) additions:**
- **Colour-match paste-back** — `faceswap::colour_match` aligns the swapped crop's mean
  tone to the target face's (clamped ±20) inside `swap_into`, so the swap carries the
  scene's lighting/white-balance not the source photo's. Benefits every caller.
- **Standalone `plakat faceswap` command** — reverses the original "engine-only" call:
  `plakat faceswap <scene> --source <face> [--face N|--all] [--restore]` swaps faces in
  an image you already have (no scene generation). The engine, exposed as a verb.


Goal: place a *specific person* into a generated scene with **strong, reliable
identity** by face-swapping (InsightFace `inswapper_128`), not IP-Adapter region
inpainting (M1, which hit its quality ceiling — diluted/distorted faces).

Pipeline (per target face): SCRFD detect + 5-pt landmarks → align target to the
arcface 128² template → arcface recognition of the **source** face → 512-d
embedding → `inswapper_128(target_128, emap·embedding)` → inverse-warp + blend the
swapped crop back into the scene.

## Phase 1 — reference (DONE)

InsightFace ground truth established: `buffalo_l` (SCRFD det + `w600k_r50` rec) +
`inswapper_128.onnx`. `1.png` identity → `2.png` pose swaps cleanly
(`/tmp/swap_ref_out.png`). Confirms feasibility + the recipe, and gives a
per-stage ground truth to verify the Rust port against.

Key facts learned:
- `inswapper_128.onnx`: `target [1,3,128,128]` + `source [1,512]` → `[1,3,128,128]`.
  StyleGAN-ish: encoder convs → AdaIN bottleneck (12 Gemm inject the 512-d id via
  instance-norm modulation) → decoder (2 Resize upsamples) → Tanh. 20 Conv,
  reflection Pad. `emap` = the `buff2fs` (512,512) initializer; `latent =
  normalize(embedding) @ emap`, then renormalised, is the `source` input.
- recognition: `w600k_r50` (ResNet50 ArcFace, glint360k) → 512-d L2-normed
  embedding from the arcface-aligned 112² crop. inswapper expects THIS embedding
  space — plakat's existing faceid ArcFace must be checked for compatibility;
  likely needs a dedicated `w600k_r50` port.

## Phase 2 — inswapper_128 generator port — DONE ✓

Ported `pipelines::inswapper` + `convert-onnx --arch inswapper-128`. Verified vs
the onnxruntime reference (real aligned face + emap-projected embedding): **max abs
diff 1e-5**. emap (`buff2fs`) is applied externally to the embedding, not in-graph.
Converted weights: 528 MB f32 (→ host as f16 ~264 MB).

### original trace (kept for reference)

Full graph trace (DONE — 20 Conv + 12 Gemm, in order):
```
ENCODER
  C00 Conv 3→128   7x7 s1 (reflect pad 3)   128² in
  C01 Conv 128→256 3x3 s1 pad1
  C02 Conv 256→512 3x3 s2 pad1              → 64²
  C03 Conv 512→1024 3x3 s2 pad1             → 32²
BOTTLENECK  (12× AdaIN res-block: Conv 1024→1024 3x3 reflect-pad1 +
             instance-norm + style from  Gemm source(512)→2048 = [scale1024,bias1024])
  C04..C15 Conv 1024→1024 3x3 s1, each paired with one Gemm[2048,512]
DECODER
  Resize ×2 (32→64), C16 Conv 1024→512 3x3 pad1
  Resize ×2 (64→128), C17 Conv 512→256 3x3 pad1
  C18 Conv 256→128 3x3 pad1
  C19 Conv 128→3 7x7 (reflect pad 3) → Tanh → (x+1)/2
```
Source latent = `normalize(arcface_embedding) @ emap(buff2fs)`, renormalised.

- [ ] `convert-onnx --arch inswapper-128`: map convs + Gemms (FC) to plakat keys.
- [ ] New `pipelines::inswapper` module: forward (encoder → AdaIN-modulated
      bottleneck → decoder → Tanh). Verify output vs onnxruntime on a fixed
      (target,embedding) pair to <1e-3.

## Phase 3 — recognition (arcface w600k_r50) embedding — DONE ✓

Reused plakat's IR-ResNet50 (`face_models.rs`) after **fixing a real bug**: it
omitted the stem PReLU (claimed it folded into conv1.bias — impossible, PReLU is
nonlinear). Added it → embedding matches InsightFace **cosine 1.000000**.
`convert-onnx --arch arcface-w600k` maps numbered convs/prelus positionally +
passes through the pytorch-named BN/fc (byte-identical to a graph-traced ref).
`emap` (buff2fs) now emitted by `convert-onnx --arch inswapper-128`; source latent
`normalize(emb @ emap)` matches reference **diff 0.0**.

## Phase 4 — alignment + paste-back (pure geometry, no GPU)

- [ ] SCRFD 5-pt landmarks → similarity transform to the arcface template
      (112² for rec, 128² for inswapper target). Reuse
      `face_models::align_to_arcface_template`.
- [ ] Inverse-warp the swapped 128² crop back; soft-edge blend (and optional
      colour match) into the scene. Reference: insightface `INSwapper.get`.

## Phase 5 — integration

- [ ] `plakat faceswap <scene> --source <face.png> [--face N]` standalone command.
- [ ] Wire into `multiperson`: generate ONE coherent scene (no per-region inpaint
      needed for identity), SCRFD-detect each face, match to persona by placement
      region, swap with that persona's source. Placement (already perfect) +
      face-swap identity = the real deliverable.
- [ ] corpus showcase + tutorial. Note inswapper's non-commercial license.

## Notes

- Models are large (`inswapper_128.onnx` ~554 MB). Host the converted plakat
  safetensors on HF like the SCRFD default; gate behind explicit opt-in.
- Verify EVERY phase against the Phase-1 reference before moving on — same
  discipline that made the SCRFD port land correctly.
