# plakat 2.6.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). Theme: **finish the SD UNet SDPA thread (the workhorses)
+ grow the image-quality track**, with high-res/control quality as the 2.6 flavour. Scope with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.5 (finish these first)

- [x] **SD UNet SDPA — step 2. DONE (c54ccc1).** Wired SDXL through `sd_train::unet::forward_sdxl`
      (+ `forward_sdxl_with_residuals` for ControlNet) and **flipped the default**: SD 1.5 + SDXL now
      run plakat's OWN SDPA UNet by default; `PLAKAT_CANDLE_UNET=1` reverts; SD 2.1 stays on candle.
      **Proven safe (own ≡ candle, 3 ways):** `sd15.unet.out` corr 1.00000 (max_abs 3e-4);
      `sdxl.unet.out` corr 1.00000 (max_abs 2e-4); and end-to-end on Metal the own UNet reproduces
      candle's SDXL trajectory *bit-for-bit* — own 0.9347 ≡ candle 0.9343 SSIM (see golden note below).
      SD 1.5 Tier-2 passes 0.9976. **Speed (per-step): SD 1.5 ~2.5% (708→690), SDXL ~8% (4297→3945).**
      Modest vs the DiTs' 1.2–1.9× — the SD UNet is conv-dominated (attention is a small slice even
      with SDXL's full head_dim-64 SDPA coverage). The durable win is **decoupling from candle's
      registry UNet/attention** (aligns with the candle-stability wariness).
      - [x] **The SDXL Tier-2 golden is NOT stale — false alarm, resolved.** The 0.9343 came from
            running Tier-2 on `--device metal` to pair with the Metal bench. Tier-2 is a **CPU-canonical**
            gate (like the Tier-1 diffusers goldens): on `--device cpu` the current pipeline matches the
            HF golden **byte-for-byte (SSIM 1.0000, mean_abs 0.000)**. The Metal 0.9343 is plain
            Metal-vs-CPU fp drift on a flat, low-information det-init smoke — the fixture is *designed*
            flat ("regression gate, not a quality target"), so SSIM is dominated by fine moiré that
            diverges between backends. Confirmed **not** SDPA/the-flip: Metal eager (`PLAKAT_NO_SDPA=1`)
            scored 0.8049 ≈ SDPA's 0.8041 at 20 steps; more steps makes cross-device *worse*, and the
            output stays flat regardless (det-init by design). **No re-author, no HF write.** Lesson:
            don't run Tier-2 on Metal and read the drift as staleness. Added `PLAKAT_VERIFY_AUTHOR_GOLDEN`
            (the "Tier-2 freeze step" the golden.rs error message references) for any *genuine* future
            re-author — run it on CPU.
      - [x] **Follow-up DONE: ControlNet + inpaint through the own UNet default verified** (pre-release
            safety check). SD1.5+CN (Tile upscaler, 9 coherent tiles), SDXL+CN
            (`forward_sdxl_with_residuals` → structure-conditioned forest, no shape errors), and 9-ch
            SD1.5 inpaint (seamless mask region, forest preserved outside) all render coherently through
            the own UNet. The default flip is de-risked across plain/CN/inpaint paths.
- [x] **SD3 MMDiT PAG** + a **`--pag-scale` CLI flag** (2df27e7). PAG threaded through the
      JointBlock trait / all 3 block types / MMDiTCore / new `MMDiT::forward_pag`; x-stream self-attn
      perturbed to identity (output = V), context stream untouched. Applied at the
      `predict_velocity_full` chokepoint. `--pag-scale <f>` promotes it off the raw env knob (sets
      `PLAKAT_PAG_SCALE` → SD3 **and** PixArt honor it). Gotcha fixed: `split_qkv` leaves `v` 4D
      (b,seq,heads,head_dim) while q/k are flattened → the identity output must `flatten_from(2)` to
      match `attn()`'s (b,seq,hidden) layout. **Verified safe**: pag=false is byte-identical
      (`sd35-medium mmdit.block0` corr 1.00000, generation unchanged). pag=true correctness is
      by-construction + mirrors the proven PixArt PAG. **Live render caveat (f9d14e3):** a completed
      PAG-on image showed all-blocks perturbation at scale 3.0 destabilises MMDiT (black + patch-grid)
      — the joint stack is deeper/more fragile than PixArt's. Fixed by restricting PAG to a middle
      layer subset (default single mid block, `PLAKAT_PAG_LAYERS` to tune), diffusers-aligned. SD3 PAG
      is now **opt-in + EXPERIMENTAL** (default off; scale calibration still pending — can't iterate
      sd35 renders on this box due to T5-XXL OOM). PixArt PAG unaffected.
- [x] **FreeU** + **CFG-rescale** / dynamic thresholding (free-quality guidance) — **BUNDLE COMPLETE**.
  - [x] **CFG-rescale** (648217e) — `--guidance-rescale <phi>` (0 = off, ~0.7 sweet spot). Shared
        `guidance::cfg_rescale` wired at every CFG blend in SD (t2i ×3) / PixArt / SD3; env-promoted
        `PLAKAT_CFG_RESCALE` like `--pag-scale`. **Validated** on SD1.5 @ guidance 14 (off = neon
        blow-out + banding + clipped subject; 0.7 = natural exposure, complete portrait). phi=0 no-op.
  - [x] **FreeU** (b8df2ea) — went the full route: wrote a from-scratch 2D DFT (`fft.rs`, DFT-by-matmul
        → any size, not just powers of two; round-trip unit-tested) since candle has no FFT, then
        canonical FreeU in the own SD UNet up-blocks (backbone boost + Fourier skip low-pass on the
        first two up-stages). `--freeu` / `--freeu-params b1,b2,s1,s2` (env `PLAKAT_FREEU`). **Verified**
        off = byte-identical (sd15 unet.out corr 1.0); **validated** on SD1.5 — richer detail/texture/
        contrast vs baseline. Own-UNet-default (v2.6 flip) is what made the up-block hooks editable.
  - [x] **Dynamic thresholding** (724bcc9) — Imagen x0 percentile clamp. candle exposes no x0/alphas,
        so reconstruct x0 from (sample, ε) via the standard SD schedule (matches candle DDIM), threshold,
        re-derive ε before `scheduler.step`. `--dynamic-threshold <pctl>` (env `PLAKAT_DYNTHRESH`).
        Epsilon-SD only (sd15/sdxl; v-pred 2.1 + SD3 guarded out). Off = byte-identical (sd15 Tier-2
        SSIM 1.0); validated on SD1.5 @ g14 (curbs the neon blow-out, recovers the subject).

## 2.6 flavour — high-res & control quality

- [x] **ControlNet-Tile + diffusion tiled-upscale (SUPIR-lite)** (938a9d4) — `upscale --diffusion`.
      Lanczos pre-upscale → tiled img2img refine, each tile conditioned by ControlNet-Tile on its own
      blurry content → feathered overlap blend. New `ControlKind::Tile` (conditioning = the image
      itself, no annotator) + `diffusion_upscale.rs` on `portrait::Pipeline` (own SD UNet default) via
      `blend_latents_one` + per-tile `ControlRequest`. CLI: `--scale/--tile/--overlap/--tile-strength/
      --cn-strength/--steps/--guidance/--prompt`. **Validated** forest 512→1024 (9 tiles): richer
      bark/moss/leaf detail, no seams, structure-faithful. Gotcha: tile CN weights from
      `takuma104/control_v11` (diffusers-safetensors mirror; lllyasviel ships only .bin).
- [ ] **Face/detail restoration** (GFPGAN / CodeFormer) + better upscalers.
- [ ] **Aesthetic scoring** (CLIP predictor → rank generations) — feeds the 3.0 manager's curation.

## House-keeping

- [x] **Open 2.6.0** — branch off `main` (2.5.0 release), version bump `2.5.0 → 2.6.0`.
