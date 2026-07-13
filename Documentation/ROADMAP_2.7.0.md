# plakat 2.7.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). **Theme: "curate & finish"** — the pieces that turn raw
generation into a *keepable* collection, bridging 2.6's quality work into 3.0's collection manager.
Filtered hard for load-bearing (fixes a real gap or compounds toward 3.0) over nice-to-have.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Tier 1 — core (highest leverage, feasible on the 24 GB dev box)

- [~] **Aesthetic scoring + `--keep-best K` batch curation.** LAION aesthetic predictor v2 —
      CLIP ViT-L/14 image embed (768-d, L2-normalised) → tiny MLP (768→1024→128→64→16→1, ~4 MB).
      `plakat rank <dir>` scores+sorts; `generate --count N --keep-best K` generates N, keeps the
      top-K, writes the score into the existing PNG metadata sidecar. **Load-bearing:** turns
      "roll the dice N times, sift by hand" into "generate a batch, auto-keep the winners" — AND the
      score is the first sort/filter key the 3.0 manager needs (manager plumbing in disguise). Needs a
      CLIP ViT-L/14 **vision** tower (plakat's CLIP-L is the text encoder; IP-Adapter/Cascade have
      image encoders to reuse or model after). **STARTING HERE.**

- [ ] **PAG for the SDXL / SD 1.5 own UNet** (`--pag-scale` reaches the workhorses). The 2.6
      own-UNet flip made `sd_train::unet`/`attention` the default and editable (that's how FreeU got
      in). Add the identity-attention perturbation to a mid block's self-attention, layer-restricted
      (SD3 lesson), through the existing `--pag-scale` / `PLAKAT_PAG_SCALE` plumbing. **Load-bearing:**
      PAG today is PixArt + SD3-experimental-uncalibratable-here; SDXL is the actual workhorse and
      doesn't OOM → finally calibratable on this hardware. Completes the PAG thread.

## Feature-sync (coverage — no feature is real until it's reachable everywhere)

- [ ] **Scenario-processing feature sync + gap audit.** Sweep every feature added recently (2.5/2.6:
      the guidance bundle `--guidance-rescale`/`--freeu`/`--dynamic-threshold`, `--pag-scale`,
      `upscale --diffusion` / ControlNet-Tile, `ControlKind::Tile`, the own-UNet knob) and confirm each
      is expressible in the scenario HJSON + Tera pipeline. Audit `cli/scenario.rs` for gaps: which CLI
      knobs have no scenario field, which pipeline task types are missing. Close them so a scenario can
      drive the full toolchain, not a subset. **Load-bearing:** scenarios are how batches/pipelines are
      authored; a feature that only exists on the `generate` CLI is invisible to production runs.
- [ ] **`plakat ui` feature sync + gap audit.** Same sweep for the TUI: surface the new quality knobs
      (guidance bundle, PAG) and the diffusion upscaler / restoration in the UI's generate + tools
      panes, plus any 2.x features that never got UI (map, compose/segment, relight, style/embedding
      train). Audit which subsystems the TUI can't reach. **Load-bearing:** the TUI is the 3.0
      flagship's foundation — every gap now is a gap the collection manager inherits.

## Tier 2 — strong, bigger (pick one)

- [ ] **Face/detail restoration (GFPGAN or CodeFormer).** SCRFD detect + ArcFace align (both already
      in plakat) → restore → paste back. Pairs with `upscale --diffusion` + `multiperson`.
      **Load-bearing:** plakat invests in portraits/faceswap; generated/swapped/upscaled faces
      routinely degrade — the standard finishing step is missing. **High effort** (GAN or
      VQ-transformer port + weights); GFPGAN likely simpler than CodeFormer.
- [ ] **sd35 T5-XXL memory relief.** Encode the T5-XXL prompt then free it before the MMDiT denoise
      (or encode on CPU), so ~10 GB of T5 isn't co-resident with MMDiT+VAE on 24 GB Metal.
      **Load-bearing:** the sd35 OOM blocked PAG calibration all of the 2.6 cycle and makes SD3.5
      unusable on 24 GB-class Macs — unblocks a whole model family. Medium–high effort.

## Tier 3 — stability (fold in as you go; the verify DNA)

- [ ] **Regression taps for the 2.6 quality features.** A **T5-free MMDiT-PAG structural tap**
      (finally exercises `forward_pag` on real weights — the check the sd35 OOM blocked) + FreeU /
      CFG-rescale determinism taps. Locks 2.6 in. Low effort.

## Explicitly excluded (nice-to-have, not load-bearing now)

New base models (Kandinsky/Sana) for reach's sake · more schedulers · cosmetic PAG-FreeU combos ·
a full manager UI (that's 3.0, not 2.7).

## House-keeping

- [x] **Open 2.7.0** — branch off `main` (2.6.0 release), version bump `2.6.0 → 2.7.0`.
