# plakat 2.7.0 — roadmap (planning)

On the [road to 3.0](ROADMAP_TO_3.0.md). **Theme: "curate & finish"** — the pieces that turn raw
generation into a *keepable* collection, bridging 2.6's quality work into 3.0's collection manager.
Filtered hard for load-bearing (fixes a real gap or compounds toward 3.0) over nice-to-have.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Tier 1 — core (highest leverage, feasible on the 24 GB dev box)

- [x] **Aesthetic scoring + `--keep-best K` batch curation** (e44c975 / 77d2715). LAION aesthetic
      predictor v2 — reused plakat's `ImageEncoder` (CLIP ViT-L/14 vision, openai/clip-vit-large-patch14)
      → L2-normalised 768-d embed → MLP (768→1024→128→64→16→1, no activations), loaded straight from
      LAION's `.pth` via candle's pickle reader (no upload/conversion). `plakat rank <paths>`
      (`--top`, `--json`) + `generate --count N --keep-best K` (snapshot-diff so only this run's
      outputs are pruned; PNG + `.json` sidecar). **Validated:** ranks coherent renders (6.0–6.25)
      cleanly above degenerate ones (3.9–4.4); keep-best 2-of-4 kept the top two. First real slab of
      the 3.0 manager's curation.
      - [ ] Follow-up: write the score into the generation PNG metadata sidecar at gen time (persistent
            sort key for the manager) — currently computed on-demand by `rank`/`--keep-best`.

- [x] **PAG for the SDXL / SD 1.5 own UNet** (9a14837) — `--pag-scale` reaches the workhorses.
      Two thread-locals (one synchronous denoise thread): the loop marks the perturbed conditional
      forward, the UNet brackets ONLY the mid block so its self-attention → identity (output = V);
      cross-attention (text) untouched. Wired at the main t2i denoise closure (`guided = cfg + pag·(cond
      − cond_pert)`), gated to the own UNet under CFG. **Verified** off = byte-identical (sd15 unet.out
      corr 1.0); **calibrated** on SD1.5 @ g5 — scale 2.0/3.0 both stable (no black/grid — the mid-block
      restriction holds, unlike SD3 all-blocks) and clearly sharpen detail/depth. 2.0–3.0 range.
      Completes the PAG thread on the workhorse that actually calibrates on this box.

## Feature-sync (coverage — no feature is real until it's reachable everywhere)

- [~] **Scenario-processing feature sync + gap audit.** Audited `cli/scenario.rs` vs recent CLI knobs.
      - [x] **Guidance bundle closed** (7b16bec) — scenario-global `pag-scale` / `guidance-rescale` /
            `freeu`(+`freeu-params`) / `dynamic-threshold`, env-promoted after parse like the CLI.
            Dry-run validated. Scenarios can now drive PAG + CFG-rescale + FreeU + dyn-threshold.
      - [ ] **Remaining gaps:** `keep-best` (post-process — add a scenario-global that prunes each
            task's outputs by aesthetic score); **diffusion-upscale as a task kind** (`ControlNet-Tile`
            `upscale --diffusion` isn't a scenario task type yet); per-task guidance overrides (currently
            scenario-global only).
- [~] **`plakat ui` feature sync + gap audit.** Audited the TUI Chat generate (exposed only
      `/steps` + `/cfg`).
      - [x] **Guidance bundle + PAG closed** (627672b) — session slash-commands `/pag`, `/rescale`,
            `/freeu`, `/dynthresh` mirroring `/steps`, applied via the env at dispatch. Help pane updated;
            178 UI tests green. The TUI drives the full guidance toolchain conversationally.
      - [ ] **Remaining UI gaps:** diffusion upscaler + rank/keep-best as UI actions; 2.x subsystems
            never surfaced in the TUI (map, compose/segment, relight, style/embedding train). These feed
            the 3.0 collection-manager flagship.

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
