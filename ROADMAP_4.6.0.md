# plakat 4.6.0 — roadmap: Sana deepening

4.5.0 shipped the **Sana** family (DC-AE + Gemma-2 + Linear-DiT, each verified corr 1.0/0.9998).
4.6.0 deepens it — the follow-ups that turn a verified base-t2i pipeline into a full member of the
family, plus the model variants. Built on already-verified components, so each phase is additive and
the frozen pipelines stay byte-identical.

Ground rules: existing output unchanged; each phase lands with a reference-corr or coherence check;
`Cargo.lock` in sync. Reuses the `tools/reference/sana_*.py` dump tooling + the env-gated corr tests.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

## Phase 1 — DPM++-multistep-flow scheduler (Sana's true default) — DONE

- [x] Implement `DPMSolverMultistepScheduler` with `use_flow_sigmas` + `flow_prediction` (order 2,
      midpoint, `flow_shift 3.0`) — Sana's shipped scheduler, higher quality than the 4.5 FlowMatchEuler.
      Track the previous model output for the 2nd-order step. Make it the Sana default (FlowMatchEuler
      stays available). Self-contained, verifiable, no new download — the cleanest first phase.
- [x] Verify: DONE — both sigma schedules unit-tested exact vs diffusers; the DPM++ **step** (x0-conv +
      1st/2nd-order midpoint) matches a diffusers trajectory (`sana_dpm_dump.py`, fixed velocities, no
      model) over 20 steps, max_abs <1e-3. `--scheduler euler` keeps the 4.5 FlowMatchEuler.

## Phase 2 — Sana img2img (`plakat img2img --model sana`) — DONE

- [x] Add a Sana arm to the dedicated img2img dispatch (`cli/img2img.rs`): DC-AE-encode the init
      (verified corr 1.0), scale by 0.41407, start the flow loop from a **partially-noised** latent per
      `--strength` (`x_σ = (1-σ)·z0 + σ·noise`, sigma schedule trimmed to the strength window).
- [x] Reuse `SanaSched` (DPM++ default / euler); the trimmed start has no multistep history → first
      step is first-order, which is correct.
- [x] Verify: DONE — 256² CPU smoke (townsquare init, strength 0.6) → coherent image retaining the
      init's green-sky character (31k colors, luma_std 92.9). DC-AE encode reused (corr 1.0); Sana
      defaults applied in img2img (20 steps / guidance 4.5); size must be mult-of-32; mask/tiled bail.

## Phase 3 — Sana LoRA — DEFERRED

- [~] Merge-at-load into the Linear-DiT (mirror `sd3_lora`). **Deferred**: no real Sana LoRAs exist yet
      (the one HF `sana-lora` repo is actually a Flux LoRA). Revisit when the ecosystem matures — the
      merge machinery + a synthetic-LoRA corr check is the plan. The `--loras` bail stays for now.

## Phase 4 — variants (parameterize the DiT config + aliases)

- [ ] Lift `sana_dit.rs`'s hardcoded consts (num_layers / hidden / heads / sample_size) into a
      `Config` read from `transformer/config.json`, so non-1.6B variants load. DC-AE + Gemma unchanged.
- [ ] Aliases (all exist on HF): `sana-600m` (`Sana_600M_1024px_diffusers`), `sana-512`
      (`Sana_1600M_512px_diffusers`), `sana-2k` (`Sana_1600M_2Kpx_BF16_diffusers`), `sana-1.5`
      (`SANA1.5_1.6B_1024px_diffusers`). Registry + capability rows + `doctor --capability`.
- [ ] Verify: at least the 600M variant generates a coherent image.

## Phase 5 — docs + release

- [ ] GENERATE tutorial: Sana img2img / LoRA / variants notes. Capability hints. README what's-new.
- [ ] Memory: update [[reference_sana]]. Cut the 4.6.0 release.

## Notes

- The DC-AE / Gemma / DiT-attention are all verified + Metal-portable (4.5 hardening carries over).
- Memory staging (free Gemma after encode, free DiT before decode) applies to every phase.
- Not in scope: inpaint/outpaint (needs mask plumbing), ControlNet, the Sana-Sprint few-step distill.
