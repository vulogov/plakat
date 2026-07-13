# plakat 2.7.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). Theme: **finish the image-quality track and start feeding the
3.x flagship** (a TUI photo/image collection manager). 2.6 delivered the guidance bundle + diffusion
upscaling + the own-UNet flip; 2.7 adds restoration/curation and begins the manager plumbing. Scope
with the user.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Carried from 2.6

- [ ] **Face/detail restoration** (GFPGAN / CodeFormer) + better upscalers. Pairs naturally with the
      new `upscale --diffusion` (restore faces after a diffusion upscale). Detail-restoration nets are
      GAN-based (RRDBNet-adjacent) — reuse the ESRGAN loader pattern.
- [ ] **Aesthetic scoring** (CLIP + a small MLP predictor → rank generations). Feeds the 3.0 manager's
      curation directly, so it doubles as the first piece of manager plumbing. Opt-in `--rank` /
      `plakat rank` over a batch; writes scores to the PNG metadata sidecar.
- [⏸] **SD3 MMDiT PAG — scale calibration.** The mechanics shipped in 2.6 (opt-in, experimental,
      layer-restricted); calibrating a good default `--pag-scale` / `PLAKAT_PAG_LAYERS` needs iterating
      sd35 renders, which the dev box can't hold (T5-XXL OOM). Do on a higher-memory machine or defer.

## 2.7 flavour — curation + the road to 3.0

- [ ] **Manager plumbing (first slice).** Aesthetic scores + existing History semantic search + the
      metadata sidecars are the raw material for the 3.x collection manager. Start shaping the index
      (what a "collection" is, how scores/tags/prompts are stored) — see ROADMAP_TO_3.0.md.
- [ ] Candidate quality items (pick with the user): PAG-for-SDXL-UNet (own UNet makes it editable),
      Kandinsky/Sana support, LCM/Turbo distillation polish, tiled ControlNet-Tile at 4K.

## House-keeping

- [x] **Open 2.7.0** — branch off `main` (2.6.0 release), version bump `2.6.0 → 2.7.0`.
