# InstantStyle for `plakat stylize` — implementation plan

**Status: CODE-COMPLETE (0.47.0) — GPU verification remaining.** All phases built
+ compiling: vendored-UNet inference, decoupled IP cross-attention, verified IP
K/V loading + style-block install (`up_blocks.0.attentions.1`), and the full
`plakat stylize --instantstyle` integration. What's left is a GPU run to confirm
the actual style transfer + a style-block A/B + (optional) Phase 4 content
subtraction. Goal: make `stylize` transfer **painterly style (texture)**, not
just content/palette (the limit of the default ref-variation path).

## Why the current approach can't do it

`stylize` (and `portrait`) use a **concat approximation** of IP-Adapter, *not*
the real decoupled path — because candle's UNet exposes no attention hooks:

- `stylize.rs:305-313`: the CLIP-H image embedding → `ImageProj` → `image_tokens`,
  scaled by `ref_weight`, then **concatenated into `encoder_hidden_states`** and
  fed to **every** cross-attention layer uniformly via the shared `to_k`/`to_v`.
- There is no per-block control, so the reference's *content* leaks into every
  layer (it conditions layout + content, not only style).

InstantStyle's core trick is **block-specific injection**: feed the style
embedding **only** into the one cross-attn block that governs style
(SDXL `up_blocks.0.attentions.1`), and zero it everywhere else. That requires
per-block hooks the concat path doesn't have.

## The lever we already own

plakat **vendored candle's SD UNet** for style training:
`src/pipelines/sd_train/{attention,blocks,unet,mod}.rs` — "Vendored from
candle_transformers 0.10.2" with LoRA-wired cross-attention
(`CrossAttention { to_q, to_k, to_v, to_out: LoraLinear }`, attention.rs:122).
It predicts ε and runs both SD 1.5 and SDXL. This gives us the per-block hook
InstantStyle needs; we extend the **same** `CrossAttention` with an IP term and
a per-block scale, and drive it for stylize inference.

## Phases

### Phase 0/1 — Vendored UNet as an inference engine  *(FOUNDATION DONE)*
- **Finding (much smaller than feared):** the vendored UNet's `forward` /
  `forward_sdxl` already predict ε — the trainer drives them every step
  (`trainer.rs:195`), which *is* the denoise forward. And the denoise loop
  already exists in `stylize.rs:352-380` (scheduler → `add_noise` → forward →
  `scheduler.step` → `vae.decode`, all via `SdCore`). So the *only* thing that
  changes for InstantStyle is one line — `core.unet.forward` → the vendored
  UNet's `forward`. Not a from-scratch engine.
- **DONE:** `instantstyle::load_vendored_unet` (`src/pipelines/instantstyle.rs`)
  loads the vendored UNet for inference (SD 1.5 / SDXL); the two UNet configs are
  now `pub(crate)`. Compiles clean.
- **Next:** verify forward-parity (vendored `forward` ≈ `core.unet.forward` on
  the same `(latent, t, ehs)`) on GPU, then Phase 2 (per-block IP cross-attn).

### Phase 1+2 — Decoupled IP cross-attention + per-block scale  *(MECHANISM DONE)*
- **DONE** (`sd_train/attention.rs`): `IpInjection { to_k_ip, to_v_ip, scale,
  tokens }` on `CrossAttention` (`ip: Option`, `None` by default → zero behaviour
  change for training + normal inference). Forward becomes
  `out = textπ(q,k,v) + scale · ipπ(q, to_k_ip(tokens), to_v_ip(tokens))` — same
  query, separate K/V over the shared style tokens. `set_ip` attaches it; the
  per-layer `scale` IS the per-block scale (Phase 2), so only the style block
  gets an `IpInjection`. Compiles clean.
- **Remaining (wiring, Phase 2b):**
  - Load the IP-Adapter `to_k_ip`/`to_v_ip` per cross-attn layer
    (`ip-adapter_sd15` / `ip-adapter_sdxl_vit-h` — the tensors the concat path
    ignores) → build `IpInjection`s.
  - **Identify the style block** (SDXL `up_blocks.0.attentions.1`; SD 1.5
    analogous up-block); reach its `attn2` in the vendored UNet to `set_ip`.
  - The shared `tokens` cell = the projected style embedding (reuse `ImageProj`),
    set once before the loop.

### Phase 3 — Stylize InstantStyle inference path
- New path in `stylize.rs`: VAE-encode the subject (`--in`), start an img2img-style
  denoise at a strength (preserve the subject's content), run the vendored UNet
  with CFG + the per-block IP injection of the style ref (`--ref`) tokens, VAE-decode.
- Reuse the existing CLIP-H encode + `ImageProj` for the style tokens.

### Phase 4 — InstantStyle-Plus content subtraction (quality)
- CLIP-encode the **content** (subject) image; subtract `λ · content_embed` from
  the style tokens to suppress residual content leak. Optional, off by default.

### Phase 5 — CLI + verification
- Make InstantStyle the **default** stylize path; keep concat reachable as a
  fallback. Flags: `--style-block {style|all}`, `--style-strength`, `--content-sub`.
- **Verify:** a painterly reference transfers *texture* to the subject **without**
  cloning the reference's content. Judge against the old concat output at full
  size; commit a corpus proof; update COVERAGE/GENERATE docs (stylize graduates
  from "ref-variation" to real style transfer).

## Risks / decisions
- **Biggest effort = Phase 3** (driving the vendored UNet for stylize inference —
  it's train-only today). Phases 0–2 are mechanical once that loop exists.
- **Style-block mapping** (diffusers → vendored indices) needs care; the
  InstantStyle paper's `up_blocks.0.attentions.1` (SDXL) is the target — verify
  empirically (inject one block at a time, see which restyles without cloning).
- **dtype / Metal**: the vendored UNet runs BF16/F32; confirm inference dtype +
  Metal kernels (the matte showed candle Metal gaps exist — CPU fallback ok).
- **Reference-compare** (the lesson from the matte): before suspecting our code,
  diff one denoise step's block outputs against a diffusers InstantStyle run on
  the same latents/seed — catches a wrong block or scale in one pass.
- SD 1.5 vs SDXL: both vendored; different style block + `cross_attn_dim`
  (768 / 2048, already handled in `stylize.rs:50-51`).
