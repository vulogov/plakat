# Tutorial: put specific people into a scene (`plakat multiperson`)

`plakat multiperson` places **specific people** (each from a photo) into one generated
scene, at relative locations you give in words. There are two identity strategies; pick by
what you need.

## The fastest path

```bash
plakat multiperson \
  "two people at a cafe table by a window, watercolor, upper body, facing the viewer" \
  --person "alice:alice.png" --at "alice:left closer front" \
  --person "bob:bob.png"     --at "bob:right closer front" \
  --swap --pose
```

`--person LABEL:photo` gives each person a reference; `--at LABEL:"<position> <distance>
<facing>"` places them. Axes (order-insensitive): position `left|center-left|center|
center-right|right` · distance `closer|mid|farther` · facing `front|side|back`. Omit `--at`
for a persona and a scene-aware LLM places them.

## Two identity strategies

### `--swap` — face-swap into a coherent scene (the natural-looking path)

Generates one coherent scene, then **face-swaps** each figure with that person's identity
(SCRFD detect → ArcFace identity → `inswapper_128` → blend back). Add **`--pose`** to pin
one synthetic OpenPose skeleton per region, so the model places a figure exactly where each
person goes and the right face lands on the right figure.

The face-swap models auto-download on first use. Override with `PLAKAT_SCRFD_WEIGHTS` /
`PLAKAT_ARCFACE_WEIGHTS` / `PLAKAT_INSWAPPER_WEIGHTS` (or the `_HF` variants).

### `--composite` — exact identity, any model

Generates the scene **background with any text-to-image model**, mattes each person's actual
photo (U2Net, no face model), and places them at their `--at` positions. Identity is
**exact** (it's the real photo) and **model-agnostic**. Add `--harmonize 0.35` to img2img the
result so the placed people share the scene's lighting/style.

```bash
plakat multiperson "a cozy library // oil painting" \
  --person "a:a.png" --at "a:left closer" --person "b:b.png" --at "b:right closer" \
  --composite --harmonize 0.35
```

## What works, honestly

Face-swap gets you a **recognizable person, naturally posed in the scene** — but it swaps the
*inner face* only, and identity strength scales with face size. To get good results:

- **Use photos** — photoreal, roughly frontal, on a light background. Paintings give weak
  identity. Tightly-cropped close-ups are auto-padded so the detector finds the face.
- **Keep figures few and prominent.** Two prominent faces read clearly; a crowd of small
  faces reads faintly. Frame with "upper body" / "head and shoulders".
- **Keep the prompt minimal** — let the swap define the faces. Don't describe each person in
  detail ("an old man with a big beard …") — that look bleeds onto every figure. A light
  gender/age hint to set each figure's *type* is fine ("a man and a woman …").
- **Hair, build, and head shape come from the generated figure**, not the swap. If a person
  is defined by a distinctive hairstyle, set it in the prompt (or use `--composite`).

For **exact** whole-person fidelity regardless of pose, `--composite` is the honest choice —
at the cost of a more "placed-in" look that `--harmonize` softens.

## Convert your own face-swap weights

The defaults are `plakat convert-onnx` conversions of InsightFace's ONNX models. To build
them yourself (e.g. a different SCRFD), the InsightFace packs ship the ONNX:

```bash
plakat convert-onnx det_500m.onnx     scrfd_500m.safetensors     --arch scrfd-500mf
plakat convert-onnx w600k_r50.onnx    arcface_w600k.safetensors  --arch arcface-w600k
plakat convert-onnx inswapper_128.onnx inswapper_128.safetensors --arch inswapper-128
```

> `inswapper_128` is InsightFace's research/non-commercial model — mind its license.
