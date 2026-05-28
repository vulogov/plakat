# AnimateDiff (v0.26)

`plakat animate --animatediff` renders **motion-coherent N-frame
sequences** from a single prompt using the AnimateDiff V3 motion
adapter spliced into SD 1.5's UNet. Different from the v0.20
`plakat animate` morph mode, which interpolates between two prompts
without temporal-attention coherence.

**Status:** v0.26.0 ships the AnimateDiff **infrastructure**
(motion adapter loader, temporal-attention modules, vendored SD 1.5
UNet with motion splice, motion LoRA composition, CLI surface).
The **inference dispatch** — the actual N-frame scheduler loop —
closes in **v0.26.1**. Calling `--animatediff` in v0.26.0 loads
the full motion stack successfully, then bails with a clean
v0.26.1 deferral message. The phases 1-5 build infrastructure
that v0.26.1 can wire end-to-end in one focused commit.

See [RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md §12](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md)
for the cycle-cut decision tree.

## What works today (v0.26.0)

- ✅ `--animatediff` flag (CLI surface)
- ✅ `--motion-lora SPEC` flag (CLI surface, full Civitai/HF/local resolution)
- ✅ `--format {gif, mp4, webm, frames, all}` (production-ready for
  any animate mode — not just AnimateDiff)
- ✅ Motion adapter V3 weights download + parse (loader + safetensors header inspection)
- ✅ 16 temporal-transformer modules built from real V3 weights
- ✅ Vendored SD 1.5 UNet with motion-module splice at block boundaries
- ✅ Motion LoRA composition (merge into adapter weights via the
  existing `MergeTarget::MOTION_ADAPTER` lora-merge infrastructure)
- ✅ `AnimateDiffPipeline` assembly (motion adapter + modules)
- ⚠️ End-to-end inference loop **(deferred to v0.26.1)**

## Quick start (v0.26.1+)

Once the inference dispatch lands:

```bash
# Watercolor cottage, 16 frames, 8 fps GIF
plakat animate --animatediff --model sd15 \
    --from "a watercolor cottage at dawn" \
    --frames 16 --gif-delay-ms 125

# With a zoom-in motion LoRA + MP4 output
plakat animate --animatediff --model sd15 \
    --from "a knight in a forest" \
    --motion-lora civitai-version:67890:0.8 \
    --format mp4
```

## Architecture

### Motion adapter (`guoyww/animatediff-motion-adapter-v1-5-3`)

Downloaded on first use (~1.4 GB safetensors). Cached afterward
under `$PLAKAT_CACHE_DIR/huggingface/hub/`.

V3 config:
```jsonc
{
  "block_out_channels":              [320, 640, 1280, 1280],  // SD 1.5 channels
  "motion_layers_per_block":         2,
  "motion_max_seq_length":           32,   // hard frame-count cap
  "motion_mid_block_layers_per_block": 1,
  "motion_norm_num_groups":          32,
  "motion_num_attention_heads":      8,
  "use_motion_mid_block":            false  // V3 skips this; V1/V2 use it
}
```

### Per-block motion modules

For V3 + SD 1.5: **16 motion modules total** = 4 down-blocks × 2
layers + 4 up-blocks × 2 layers + 0 mid-block.

Each module is a `TemporalTransformer`:
- `GroupNorm` → `Linear` proj_in
- N transformer blocks, each with:
  - LayerNorm + temporal self-attention (across the F dimension)
  - LayerNorm + cross-attention (identity in V3 — motion is text-agnostic)
  - LayerNorm + GEGLU FFN
- `Linear` proj_out
- Learnable positional embedding (`pe.weight`, shape `(1, 32, channels)`)

Forward: `(B*F, C, H, W) → (B*H*W, F, C) → attention across F → (B*F, C, H, W)` + residual.

### Vendored SD 1.5 UNet (`Sd15MotionUNet`)

Outer UNet structure vendored from candle's stock SD 1.5 UNet
(~580 LOC including tests). Reuses upstream block types
(`CrossAttnDownBlock2D`, etc.) and splices motion modules **at the
output of each down-/up- block**.

**Scope cap**: motion at block output boundaries (not interleaved
per resnet+attn layer like the faithful diffusers
`UNetMotionModel`). The faithful splice would need re-vendoring
the block types themselves (~800 more LOC). v0.26.0 ships the
coarser splice; v0.26.1 evaluates whether quality requires the
full block vendoring.

**Parity property**: `forward_with_motion(..., motion_modules: None, ...)` is
bit-identical to candle's stock SD 1.5 UNet. Verified by the
`empty_motion_modules_behaves_like_none` test in
`src/pipelines/sd15_motion_unet.rs`.

### Motion LoRAs

Motion LoRAs from the community (e.g. `guoyww/animatediff-motion-lora-zoom-in`)
target the motion adapter's attention tensors. They use bare
PEFT-style keys (no `lora_unet_` / `text_encoder.` prefix):

```text
down_blocks.0.motion_modules.0.temporal_transformer
  .transformer_blocks.0.attention_blocks.0.to_q
  .lora.{down,up}.weight
```

Loaded via the existing `MergeTarget::MOTION_ADAPTER` variant of
`merge_loras_into_weights`. The merged safetensors lands in a
detached tempfile (`NamedTempFile::keep()`) that lives for the
`MotionAdapter`'s lifetime; OS reclaims at process exit.

Stacking: pass `--motion-lora SPEC` multiple times. Same LoraSpec
grammar as `--lora`:
- `hf:user/repo:0.7`
- `civitai:NNNNNN:0.5`
- `civitai-version:NNNNNN:0.8`
- `/local/path/file.safetensors:0.6`

## Frame budget

- **Default**: 16 frames at 8 fps (AnimateDiff's training window;
  2-second loop). Per RFC Q3.
- **Hard cap**: 32 frames (V3's `motion_max_seq_length`). The
  positional embedding only has 32 rows; beyond that, the loader
  bails loud.

## Output formats

| `--format` | Effect | Requires |
|---|---|---|
| `frames` (default) | `<out>/frame-NNNN.png` per frame | nothing |
| `gif` | + animated GIF via the `image` crate's GIF encoder | nothing |
| `mp4` | + MP4 via ffmpeg (libx264 + yuv420p + faststart) | ffmpeg on `$PATH` |
| `webm` | + WebM via ffmpeg (libvpx-vp9 + CRF 30) | ffmpeg on `$PATH` |
| `all` | every format above | ffmpeg on `$PATH` |

Install ffmpeg:
- macOS: `brew install ffmpeg`
- Ubuntu: `apt install ffmpeg`
- Windows: `scoop install ffmpeg`

## Limitations

- **SD 1.5 only**. SDXL motion adapters exist
  (`guoyww/animatediff-motion-adapter-sdxl-beta`) but are less
  mature; deferred to v0.27.
- **16-frame training window**. Up to 32 frames work (max_seq_length),
  but quality may degrade past 16 since that's where V3 was trained.
- **Long-form (>32 frames)**: not in scope. Use HotShot-XL or
  similar separate architecture.
- **AnimateDiff + ControlNet**: not wired. Per-frame temporal-coherent
  control signals would need new infrastructure. v0.27+ candidate.
- **Multi-GPU**: single-GPU only. SD 1.5 + 1.4 GB adapter + N-frame
  latent buffer fits in 12 GB without sharding.

## See also

- [`RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md`](RFC_v0.26_ANIMATEDIFF_AND_CARRIES.md)
  — design doc, eight locked decisions, the cycle-cut tree.
- [`Tutorials/ANIMATE_TUTORIAL.md`](Tutorials/ANIMATE_TUTORIAL.md)
  — narrative walkthrough (v0.20 lerp mode + v0.26 AnimateDiff mode).
- [AnimateDiff paper (Guo et al., 2023)](https://arxiv.org/abs/2307.04725)
  — original architecture.
- [`guoyww/animatediff-motion-adapter-v1-5-3`](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3)
  — V3 motion adapter weights.
