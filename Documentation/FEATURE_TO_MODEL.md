# Feature → Model support matrix

Which plakat feature works on which model backbone. This is the authoritative
cross-reference; the per-feature proof drivers live in `corpus/` and are tracked
in [`corpus/COVERAGE.md`](../corpus/COVERAGE.md).

**Legend**

- ✅ — supported and wired.
- ⚠️ — supported but **constrained**: gated weights, a Metal limitation, no
  ungated asset to exercise it, or an SD-core path that works in principle but
  isn't a verified/demo target. See the per-cell note or the model footnotes.
- ❌ — not supported / not applicable for that backbone.

**Models** (text-to-image backbones; aliases in parentheses)

| Alias | Family | Notes |
|---|---|---|
| `sd15` | Stable Diffusion 1.5 | ungated, ~4 GB, 512², the workhorse |
| `sd21` | Stable Diffusion 2.1 | ungated mirror, ~5 GB, 768² v-pred — **secondary/rescued backbone** ([why](#sd-21-notes)) |
| `sdxl` | Stable Diffusion XL | ungated, ~7 GB, 1024² — the richest feature surface |
| `sd35` | Stable Diffusion 3.5 Medium | ⚠️ gated; BF16-native ~16 GB; strong text |
| `flux` | Flux.1 (dev/schnell) | ⚠️ gated (dev) + GGUF broken on Metal ([why](#flux-notes)) |
| `cascade` | Stable Cascade (Würstchen) | ungated, ~16 GB Metal, 3-stage |
| `pixart` | PixArt-Σ (DiT) | ungated, DiT architecture |

---

## Matrix

### Generation core

| Feature | sd15 | sd21 | sdxl | sd35 | flux | cascade | pixart |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Text-to-image** (`generate`) | ✅ | ✅ | ✅ | ✅ | ⚠️¹ | ✅ | ✅ |
| **img2img** (`--from` / `--strength`) | ✅ | ✅ | ✅ | ✅ | ✅¹ | ✅² | ❌ |
| **Inpaint** (`img2img --mask`) | ✅³ | ❌ | ✅³ | ❌ | ✅¹ | ❌ | ❌ |
| **Outpaint** (`outpaint`) | ✅³ | ❌ | ✅³ | ❌ | ❌ | ❌ | ❌ |

### Conditioning & adapters

| Feature | sd15 | sd21 | sdxl | sd35 | flux | cascade | pixart |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **ControlNet** | ✅ | ⚠️⁴ | ✅ | ✅ | ✅¹ | ✅ | ❌ |
| **LoRA / DoRA** (`--lora`, `civitai:`) | ✅ | ⚠️⁴ | ✅ | ✅ | ✅¹ | ⚠️⁵ | ✅ |
| **Textual Inversion** (`--embedding`) | ✅ | ⚠️⁴ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **IP-Adapter / portrait** (`portrait`) | ✅ | ❌ | ✅ | ❌ | ✅¹ ⁶ | ❌ | ❌ |
| **Image variation** (unCLIP-style) | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

### Style

| Feature | sd15 | sd21 | sdxl | sd35 | flux | cascade | pixart |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Looks / Genres** (`--look` / `--genre`) | ❌ | ❌ | ✅⁷ | ❌ | ❌ | ❌ | ❌ |
| **Stylize — ref-variation** (concat IP-Adapter) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Stylize — InstantStyle** (true style) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Style LoRA training** (`style train`) | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **Artefact compositing** (`--artefact` + blend) | ❌ | ❌ | ✅⁸ | ❌ | ❌ | ❌ | ❌ |

### Motion

| Feature | sd15 | sd21 | sdxl | sd35 | flux | cascade | pixart |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **AnimateDiff** (text→short video) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |

### Batch & scripting

| Feature | sd15 | sd21 | sdxl | sd35 | flux | cascade | pixart |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Scenarios** (HJSON batch) | ✅ | ✅ | ✅ | ✅ | ⚠️¹ | ✅ | ✅ |
| **Bund scripting** (`run`) | ✅ | ✅ | ✅ | ✅ | ⚠️¹ | ✅ | ✅ |

### Model-agnostic post-process

These operate on **any image**, regardless of the generator (or an external
image), so they have no per-model column:

| Feature | Support |
|---|---|
| **Transparent / smart cut-out** (`transparent --matte`, U2Net) | ✅ any image |
| **Upscale** (`upscale`, Real-ESRGAN ×2/×4) | ✅ any image |
| **Gallery** (`gallery`) | ✅ indexes any committed images |

---

## Footnotes

1. <a name="flux-notes"></a>**Flux** is feature-rich (t2i, img2img, inpaint/fill,
   ControlNet canny/depth variants, LoRA, IP-Adapter + Redux) but **constrained
   in practice**: the `dev` weights are gated, BF16 is ~33 GB, and the quantized
   **GGUF path is broken on candle's Metal kernel** (garbage output) — so on
   Apple Silicon Flux is effectively limited. Use Cascade or CPU where Flux would
   otherwise fit. Cells are marked ✅ for "wired in code" with this ⚠️ caveat.
2. **Cascade img2img** is `--faithful` img2img plus the Stage-C image-variation
   path (a reference's CLIP embedding), not a generic SD-style `--from` denoise.
3. **Inpaint / outpaint** run through the dedicated inpaint checkpoints
   (`sd15-inpaint`, `sdxl-inpaint`); outpaint pads the canvas + paints the new
   strip (mid-gray fill + binary mask). No SD 2.1 / SD 3.5 / Cascade / PixArt
   inpaint variant is wired.
4. <a name="sd-21-notes"></a>**SD 2.1** is an SD-core member, so the SD-core
   machinery (ControlNet / LoRA / Textual Inversion) *can* run on it, but it's a
   **rescued, secondary backbone** (the alias was repointed off the gated
   stabilityai repo) — these paths aren't bundled with SD 2.1 assets or proven by
   a driver. SD 2.1 is verified for **t2i + img2img**; treat the ⚠️ cells as
   "works via SD-core, unproven." `stylize` explicitly refuses SD 2.1.
5. **Cascade LoRA/DoRA**: the engine is complete (merges into Stage B `decoder.`
   + Stage C `prior.`, kohya + diffusers-PEFT prefixes, DoRA auto-detected, via
   `--lora` and scenarios) — but **there is no ungated Cascade LoRA to demo**
   (Cascade never grew SDXL's ecosystem, and plakat trains only SD1.5/SDXL/SD3.5
   LoRAs). Parked, like Flux assets.
6. **Flux identity** uses Flux's own IP-Adapter / Redux path, not the SD
   `portrait` subcommand (which is SD 1.5 / SDXL IP-Adapter-Plus-Face).
7. **Looks / Genres** are SDXL style/character LoRAs (looks via `--smart-discovery`
   LLM-judged Civitai pool; genres pin a Civitai SDXL LoRA by id), so they're
   SDXL-bound. A Civitai LoRA pulled by id (`civitai:<id>`) otherwise works on
   whatever base it was trained for (typically SDXL or SD 1.5).
8. **Artefact compositing** is **SDXL only** — the integral blend (canvas-relative
   scale + contact shadow + colour harmony + canny-ControlNet re-paint) runs
   through the SD-core img2img pipeline and is demoed on SDXL; the matted cutout
   library is built with `transparent --matte` (model-agnostic).

---

*Generated for plakat 0.47.0. Keep in sync with `corpus/COVERAGE.md` and the
`src/pipelines/` module set (`flux_*`, `sd3_*`, `cascade_*`, `pixart_lora` define
the per-family adapter coverage).*
