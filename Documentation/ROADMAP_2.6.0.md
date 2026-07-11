# plakat 2.6.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). Theme: **finish the SD UNet SDPA thread (the workhorses)
+ grow the image-quality track**, with high-res/control quality as the 2.6 flavour. Scope with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.5 (finish these first)

- [~] **SD UNet SDPA — step 2.** The hard question is **answered**: added env-gated
      `SdUNet::SdOwn(sd_train::unet)` for SD 1.5, and **`sd15.unet.out` verifies corr 1.00000
      (max_abs 3e-4) vs candle's golden** — plakat's own UNet is output-equivalent, so the switch
      is *proven safe*. BUT the SD 1.5 SDPA speedup is **marginal (~2.5%, 708→690 ms/step)** —
      SD 1.5's head dims are 40/80/160 (only 80 qualifies for the Metal SDPA kernel) and its UNet
      is conv-dominated (attention is a small slice). → **Redirect: SDXL is the real SD-family
      target** (head_dim 64 everywhere → full SDPA; more attention). Wire SDXL through
      `sd_train::unet::forward_sdxl` (sdxl config + add-embeds), verify (`sdxl` unet.out corr 1.0),
      bench. If SDXL wins like the DiTs (~1.2–1.5×), route SDXL through the own UNet; SD 1.5/2.1
      stay on candle (the SDPA win isn't worth switching them). *Own-UNet also decouples from candle.*
- [ ] **SD3 MMDiT PAG** + a **`--pag-scale` CLI flag** (promote PAG off the env knob).
- [ ] **FreeU** + **CFG-rescale** / dynamic thresholding (finish free-quality guidance).

## 2.6 flavour — high-res & control quality

- [ ] **ControlNet-Tile + diffusion tiled-upscale (SUPIR-lite)** — coherent 512→2K/4K with
      hallucinated detail; the missing control kind.
- [ ] **Face/detail restoration** (GFPGAN / CodeFormer) + better upscalers.
- [ ] **Aesthetic scoring** (CLIP predictor → rank generations) — feeds the 3.0 manager's curation.

## House-keeping

- [x] **Open 2.6.0** — branch off `main` (2.5.0 release), version bump `2.5.0 → 2.6.0`.
