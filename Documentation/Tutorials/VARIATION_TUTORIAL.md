# Image variation — re-imagine a reference (Stable Cascade)

`--image-variation` conditions generation on a reference image's **CLIP
embedding** (unCLIP-style) rather than a text prompt. It keeps the reference's
*semantics* — subject, composition, mood — while re-composing the pixels: a fresh
"variation on a theme."

This is a **Stable Cascade** feature (its Stage C accepts an image embedding via
a CLIP ViT-L/14 encoder, v0.42).

```bash
# Pure variation — let the reference drive everything (empty prompt):
plakat generate "" --model cascade --image-variation reference.png --out out/

# Steered variation — nudge with a prompt while keeping the reference's gist:
plakat generate "at golden hour, warmer palette" \
  --model cascade --image-variation reference.png --out out/
```

## Pure vs steered

| Prompt | Result |
|---|---|
| empty (`""`) | **Pure** variation — semantics from the reference only. |
| a short phrase | **Steered** — the reference sets the scene, the prompt re-lights / re-styles it. |

## How it differs from neighbours

- **vs `stylize`** — stylize (IP-Adapter) transfers a reference's *look* onto a
  *different subject* you supply. Image-variation re-generates *the same subject*
  from its embedding. See [STYLIZE_TUTORIAL](STYLIZE_TUTORIAL.md).
- **vs `img2img`** — img2img denoises *from the reference's pixels* (structure
  preserved). Image-variation starts from noise conditioned on the *embedding*
  (structure free to change). See [IMG2IMG_TUTORIAL](IMG2IMG_TUTORIAL.md).

## Notes

- Cascade is ungated (~16 GB on Metal). First run downloads the Stage A/B/C
  weights + the Stage C image encoder.
- Use the `--decoder-guidance` / Stage step flags (see
  [CASCADE_TUTORIAL](CASCADE_TUTORIAL.md)) to trade fidelity vs variety.
