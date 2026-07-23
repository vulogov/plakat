# plakat 4.4.0 — roadmap: SDXL few-step (Lightning + Hyper-SD)

**Bring 2–8-step SDXL generation to plakat** — SDXL-Lightning and Hyper-SD-SDXL, so the most-used
family gets the few-step speed that today only Flux/LCM enjoy. Scope (user-chosen): **LoRA presets
_and_ a full-UNet Lightning alias**, targeting **4- and 8-step** tiers.

The infrastructure already exists: `flux_fast.rs` bundles LoRA + steps + guidance + scheduler and
already has a `FastTarget::Sdxl` arm (the shipping `lcm-sdxl` preset). The one genuine gap is
Lightning's required **Euler + `timestep_spacing="trailing"`**, which the scheduler layer can't
express yet (the `Trailing` machinery exists in `extra_schedulers.rs` but is unreachable).

Ground rule: `--fast` presets are opt-in → the **default path stays byte-identical** and
`plakat verify` stays green. New logic lands with unit tests. `Cargo.lock` in sync.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

**Scope update (user call):** ship on the LoRA presets. Phase 3 (full-UNet `sdxl-lightning`
alias) is **dropped** — it needs a composite UNet-override load *and* new alias-baked-sampler-
defaults machinery (no precedent, even `sdxl-turbo` lacks it), a big lift + wider verify surface
for a marginal gain over the working 8-step LoRA preset. Captured as a possible fast-follow.

## Phase 0 — Euler-trailing scheduler (the one new primitive) — DONE

- [x] Add a `SchedulerKind::EulerTrailing` variant (`scheduler.rs:21`), its `FromStr` token
      (`"euler-trailing"`), and a `build()` arm that constructs `EulerSchedulerConfig` with
      `timestep_spacing = TimestepSpacing::Trailing` (`extra_schedulers.rs` already implements the
      trailing branch). Unit test: the built scheduler's timesteps differ from Leading and match the
      expected trailing sequence.

## Phase 1 — Hyper-SD-SDXL presets (LoRA) — DONE

- [x] `PRESETS` rows in `flux_fast.rs` (`target: FastTarget::Sdxl`): `hyper-sdxl-8`, `hyper-sdxl-4` —
      repo `ByteDance/Hyper-SD`, the `Hyper-SDXL-{8,4}steps-lora.safetensors` files, `guidance: 1.0`,
      `scheduler_hint` (Hyper-SD rides the existing `lcm`/TCD-style schedule — no new scheduler).
- [x] Scripting allowlist now DERIVES from `flux_fast::PRESETS` (no hardcoded list to sync).

## Phase 2 — SDXL-Lightning presets (LoRA) — DONE

- [x] `PRESETS` rows: `lightning-sdxl-8`, `lightning-sdxl-4` — repo `ByteDance/SDXL-Lightning`, the
      `sdxl_lightning_{8,4}step_lora.safetensors` files, `guidance: 1.0`,
      `scheduler_hint: "euler-trailing"` (Phase 0). Allowlist updated.

## Phase 3 — Full-UNet Lightning alias — DROPPED (deferred, see scope update)

- [ ] First-class `sdxl-lightning` alias (the fused UNet checkpoint, not base+LoRA): `AliasEntry` in
      `hf/mod.rs` ALIAS_TABLE (repo `ByteDance/SDXL-Lightning`, family SDXL) + `ModelMeta` row in
      `capability.rs` (native_res 1024, F16, tuning hint) + family memory-heuristic keys. Route the
      fused UNet through load/`Variant` so it generates at 4/8 steps with euler-trailing + guidance 1.0
      by default. Decide 4-step vs 8-step default (likely the `_8step_unet` for quality).

## Phase 4 — Verify + docs — DONE (CI gate at release)

- [ ] `plakat verify` (or the `--no-default-features --lib` gate) green — default output byte-identical.
- [x] Live-verified `lightning-sdxl-8` on Metal: LoRA merged 788/788, euler-trailing, coherent 1024² image. HF LoRA filenames all resolve 200.
- [x] Docs: GENERATE tutorial §6 few-step section, `--fast` --help listing, `capability.rs` tuning hints,
      and a **`plakat doctor --capability` few-step-presets section** (user ask) derived from PRESETS.
- [ ] README/what's-new note — at release.

## Notes / risks

- **Exact HF artifact filenames** must be verified at implementation (Lightning: `sdxl_lightning_Nstep_lora.safetensors` / `_unet.safetensors`; Hyper-SD: `Hyper-SDXL-Nsteps-lora.safetensors`).
- Hyper-SD 1-step needs TCD-eta specifics; we target **4/8-step** where the LCM-style schedule is safe.
- Lightning is base-SDXL-only (not a from-scratch model): the LoRA presets require a base `sdxl`; the
  full-UNet alias is self-contained.
