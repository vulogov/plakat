# plakat — stability audit & bugfixing plan

Source-wide bug/stability audit (6 parallel subsystem passes: TUI, CLI/scenario,
SD-family pipelines, transformer/video pipelines, core infra, scripting/map/compile/
training). Each finding was read at the source before reporting; reachability noted.
**Nothing here is fixed yet** — this is the plan. Each fix should begin by *reproducing*
the failure (a regression test), then the change, then re-run the affected suite.

Severities: **Critical** (security / silently-wrong on a mainline path / host crash),
**High** (crash or wrong output on a real feature), **Medium** (degenerate-input crash,
quality regression on a feature intersection, OOM lever), **Low** (latent / narrow / hardening).

Effort: **S** ≤ ~30 min · **M** a few hours · **L** a day+.

> **Audit integrity.** Findings are agent-identified from source reads — treat them as
> leads, not gospel: **reproduce each before fixing.** A re-audit already retracted two
> Cascade findings (3.1 Stage-C-resident, 5.4 single-step) as unverified/refuted and
> re-confirmed one (2.7 img2img `steps=0`). The P0 fixes below were each reproduced +
> tested before shipping.

Cleared by the audit (no action): the new **shared-pipeline scenario reuse**
(`can_reuse_sd_pipeline`) was independently confirmed provably safe — no wrong-output or
double-residency path exists. The TUI has no `block_on` on the event loop; SD3/PixArt/
Cascade/Flux happy paths and the historic "silent noise" surfaces are verified correct.

---

## P0 — Security, data loss, host crash — ✅ DONE (shipped)

| # | Sev | Site | Bug | Fix | Status |
|---|-----|------|-----|-----|--------|
| 0.1 | Critical | `civitai/download.rs` | **Path traversal**: untrusted Civitai `file.name` joined into the cache path — `"/…/.ssh/authorized_keys"` or `"../.."` overwrites arbitrary files. | `safe_basename()` reduces to `Path::file_name`, rejects empty/./../separator; applied before `version_file_path`. + test. | ✅ |
| 0.2 | High | `memwatch.rs` | OOM watchdog's ≥1.5 s window too slow for a fast single-buffer allocation → host can crash first. | Acute fast-path (abort on first Critical + free-below-floor sample); `INTERVAL` 300→100 ms; sustained window kept for ride-out-able pressure. | ✅ |
| 0.3 | High | `memwatch.rs` | Aborts on system-wide pressure it didn't cause; armed for CPU jobs; macOS `Unknown`→discredited free-RAM guard. | `plakat_is_culprit()` attribution gate (RSS share OR free-drop-since-arm); armed **only on Metal**; macOS `Unknown`→no signal. + test. | ✅ |
| 0.4 | Medium | trainer + sd3/pixart/cascade/embedding | **Non-atomic checkpoint write** truncates on a mid-write abort; single-file mode destroys the only artifact. | New `pipelines::atomic_safetensors_save` (temp-then-rename), applied to all 6 trained-artifact saves. + test. | ✅ |

## P1 — Silently-wrong output — 7/8 DONE, 1 deferred

| # | Sev | Site | Bug | Status |
|---|-----|------|-----|--------|
| 1.1 | Critical | `pipelines/animatediff.rs` | **SDXL AnimateDiff CFG scramble** — interleaved embeds vs blocked latents mispaired every frame ≥ 2 → incoherent video. | ✅ blocked layout for hidden/pooled/time_ids (mirrors SD1.5) + layout regression test |
| 1.3 | Medium | `pipelines/t2i.rs` | **Regional SDXL (w,h) vs (h,w)** swap in micro-conditioning ids. | ✅ now matches the main path |
| 1.4 | Medium | `pipelines/lora.rs` | **LoKr alpha** used w1's rows not the decomposition rank. | ✅ rank from whichever factor is decomposed |
| 1.5 | Medium | `pipelines/sdxl_clip.rs` | **argmax EOT pooling** picked a TI trigger (id > EOS) instead of EOS. | ✅ `eot_rows()` finds the explicit EOS id (all 3 sites) + test |
| 1.6 | Medium | `pipelines/flux.rs` | **Flux CLIP-L pooled at BOS** for >77-token prompts. | ✅ `clip_pool_pos()` on pre-resize ids, clamped + test |
| 1.7 | Medium | `compile/emitter.rs` | `model`/`size`/`scheduler` emitted **unquoted** → unloadable scenario. | ✅ routed through `q()` (global + per-task) + hostile-value test |
| 1.8 | Medium | `cli/scenario.rs` / `animate.rs` | **`count:0` / `--frames 0` silent success** (zero output). | ✅ up-front `count≥1` reject; `run_flux` frames guard + test |
| **1.2** | Medium | `pipelines/pixart.rs`, `sd3.rs` | **T5 encoded without a pad attention mask** — pad tokens leak into T5 self-attn + DiT/MMDiT cross-attn; short prompts subtly off (not noise). | **⏸ DEFERRED** — needs the diffusers reference-comparison harness. Pixart/SD3 use candle's `t5::T5EncoderModel` (no mask arg), so a correct fix means switching to the vendored T5 **and** threading a pad mask through every DiT/MMDiT cross-attention block — invasive changes to two **currently-correct** models that can't be verified for correctness on this hardware. Blind-fixing risks regressing a working path for a subtle short-prompt gain. Schedule with real weights + reference dumps. |

## P2 — Crashes / panics from user input — 9/10 DONE, 1 open

| # | Sev | Site | Bug | Status |
|---|-----|------|-----|--------|
| 2.1 | High | `ui/tui/app.rs` | **Soft-lock** — the one drain ignoring a disconnected channel; a panicked render thread wedged `active_gen` forever. | ✅ Empty vs Disconnected split → synthesize Error + clear; test drives a dropped sender |
| 2.2 | High | `pipelines/portrait.rs` | **Portrait inpaint + ControlNet crash** — 9-ch latent → 4-ch CN `conv_in`. | ✅ CN takes the pre-concat 4-ch latent (`cn_latent`); UNet keeps 9-ch |
| 2.3 | High | `pipelines/flux_quantized_inner.rs` / `flux_lora.rs` | **GGUF Flux + LoRA dtype crash** — F32 body vs BF16 override/slots. | ✅ overrides stored F32; slots coerced to activation dtype at forward + test |
| 2.4 | High | `scripting/config.rs` | **`steps=0`** → divide-by-zero in `pick_timesteps`. | ✅ `parse_positive_int` on steps + stage_c/b_steps + test |
| 2.7 | Medium | `pipelines/cascade.rs` | **Cascade img2img `stage_b_steps=0`** empty-Vec index panic. | ✅ `ensure!(stage_c/b ≥ 1)` up front |
| 2.8 | Medium | `map/engine.rs` | **`tile_grid` unclamped** on the `--map-spec`/LLM path → `cols*PX` overflow. | ✅ `clamp(1, 8)` in `from_spec` + 4e9 test |
| 2.9 | Low | `hf/download.rs`, `pipelines/lora.rs` | **Multibyte revision byte-slice** panic. | ✅ `chars().take(8)` both sites |
| 2.10 | Low | `capability.rs` | **Symlink recursion** in `walk_files` → stack overflow. | ✅ `symlink_metadata` decides dir-vs-file + Unix circular-link test |
| **2.5** | High | `prompt/a1111.rs:188` | **Nested `(…)`/`[…]` recursion, no depth cap** → stack overflow from any prompt. | ⬜ OPEN — thread a `depth`, fall back to literal past a cap |
| **2.6** | High | `prompt/wildcards.rs:217` | **Nested `{a\|b}` alternation recursion, no depth cap** → stack overflow. | ⬜ OPEN — same pattern (depth counter in `expand_inline`) |

## P3 — Memory / OOM on 24 GB, and resource races

| # | Sev | Site | Bug | Fix | Eff |
|---|-----|------|-----|-----|-----|
| 3.1 | ~~Medium~~ | `pipelines/cascade.rs` | ~~Stage C prior stays resident → OOM~~ **RETRACTED on re-audit** — Stage C/B/A/CLIP-G total ~12 GB (within 24 GB) and Stage C runs on a fixed 24×24 latent (tiny activations). Not a defect. | — | ✂ |
| 3.2 | Medium | `cli/scenario.rs:2160-2201` | **`style:`+`personas:` SDXL scenario** holds SDXL + stylize(SD1.5) + portrait(SDXL) + shared-CLIP co-resident for the whole run → OOM. | Lazy-load stylize/portrait on first use and drop when leaving that task kind, or preflight-warn. | M |
| 3.3 | Medium | `hf/download.rs:167-186` | **Lock-sweep race** — `clean_stale_locks` unconditionally removes every `.lock`, incl. an in-flight concurrent TUI download's → corrupt blob. | Only remove locks older than an mtime threshold (or whose tmp/blob is absent). | S |
| 3.4 | Medium | `prompt/wildcards.rs:159-181` | **"Billion-laughs"** — `MAX_DEPTH` caps depth, not *width*; a line with ≥2 self-refs grows ×N per pass → GBs before the cap. | Track a cumulative expansion budget (total replacements / output length); bail to literals. | S |
| 3.5 | Medium | `instance_guard.rs:28-74` | **Over-broad instance guard** — any `plakat` process (idle `gallery &`, `doctor`, a zombie) triggers a false "already running" for a heavy run. | Match only *heavy* subcommands in the scanner (inspect the other process's `cmd()`). | S |
| 3.6 | Low | `pipelines/pixart_dit.rs:179` | **PixArt 2K KV-compression fallthrough** — a renamed/forked 2K repo misses the literal-substring match → full O(T²) attention on 16384 tokens → wrong output + ~34 GB OOM. | Auto-detect from the `…kv_proj_conv2d.weight` tensor's presence (like `AdaLnSingleEmb`). | S |

## P4 — Training robustness

| # | Sev | Site | Bug | Fix | Eff |
|---|-----|------|-----|-----|-----|
| 4.1 | Medium | `pipelines/sd_train/trainer.rs:259,554` (+ sd3/pixart/cascade) | **`--resume` discards AdamW state** — fresh `AdamW::new` every run resets moments + step while LR is full → loss spike, degraded LoRA, no warning. | Persist/reload the AdamW moments + step count, or at minimum warm-up/lower LR for the first N resumed steps + document. | M |
| 4.2 | Low | `pipelines/sd_train/unet.rs:415` | Training timestep materialized in **BF16** quantizes large `t` → model conditioned on a slightly-off timestep vs `x_t`'s actual noise level. | Keep `t` in F32 for the embedding path. | S |

## P5 — Low-severity hardening (batch these)

| # | Sev | Site | Bug | Fix |
|---|-----|------|-----|-----|
| 5.1 | Low | `pipelines/lora.rs` (merge store) | F16 merge cast can overflow → inf/NaN → noise patches. | Detect non-finite after cast; warn/skip or keep component in F32. |
| 5.2 | Low | `pipelines/sdxl_unet.rs:734` | `forward_with_additional_residuals` lacks the CN-count guard its motion sibling has → OOB/underflow on a mismatched CN. | Add `if additional.len()!=down.len() { bail! }`. |
| 5.3 | Low | `pipelines/controlnet.rs:1135` | `per_frame_tensors[0]` on an empty pick (frames 0) panics. | `ensure!(n_frames>0)` before indexing. |
| 5.4 | ~~Low~~ | `pipelines/cascade_scheduler.rs:106` | ~~Single-step blow-up~~ **RETRACTED on re-audit** — `mu_scale≈100×` at n=1 is finite in BF16, the `[1e-4,1-1e-4]` clamp keeps endpoints finite; degenerate but not a defect. | — |
| 5.5 | Low | `pipelines/flux.rs:1914` | `animate_frame` silently drops pipeline ControlNets (`&[]` conditioning). | Assert `controlnets.is_empty()` (loud) or thread real conditioning. |
| 5.6 | Low | `pipelines/lora.rs:1032` | compvis→diffusers output-block mapping mis-maps attention-less up-blocks (fail-safe under-apply). | Treat the last sub-index as the upsampler regardless of numbering. |
| 5.7 | Low | `pipelines/stylize.rs:113` | Blurred-ref temp path keyed only by PID → concurrent race. | Add a unique token; remove after encoding. |
| 5.8 | Low | `imaging/io.rs:88` | JSON sidecar collides across formats (`a.png`/`a.webp`→`a.json`). | Include the extension in the sidecar stem. |
| 5.9 | Low | `imaging/sizes.rs:33`, `scripting/config.rs:704`, `words/outpaint.rs:234` | Unbounded aspect/outpaint dims → gigapixel OOM. | Clamp resolved `(w,h)` to a max / reject extreme ratios. |
| 5.10 | Low | `imaging/grid.rs:71` | `cell_w*cols+pad` u32 overflow → truncated canvas. | Compute in u64, validate a ceiling. |
| 5.11 | Low | `cli/scenario.rs:3107` | Dry-run seed-range prints `count-1` underflow on count 0. | Guard / `saturating_sub`. |
| 5.12 | Low | `cli/scenario.rs:2787,5288` | Colliding `safe_name` + equal explicit seed → silent overwrite. | Reject duplicate `safe_name`, or disambiguate by task index. |
| 5.13 | Low | `compile/resolver.rs` (`parse_opt`) | `guidance` accepts `inf`/`NaN` → invalid HJSON number. | Reject non-finite. |
| 5.14 | Low | `civitai/download.rs:146-165` | No download size cap; SHA re-verified only on fresh fetch, not cache hits. | Add a size ceiling; verify SHA on cache reuse. |
| 5.15 | Low | `ui/tui/app.rs:768,809` | History decode thread spawned per tick under fast scroll → decode pile-up. | Skip a new decode while one is in flight (or debounce). |

## Non-bug: contract wording

- `map/` **cross-machine byte-stability**: the in-process guarantee holds (no HashMap-iteration/time/rng leaks), but terrain/coast/hillshade call non-correctly-rounded libm transcendentals (`exp`/`atan2`/`cos`/`powf`) whose last-ULP results vary by platform → the committed `corpus/map/*` PNGs could fail a byte-compare on a different OS/arch. **Either** narrow the doc wording to "byte-stable on a fixed platform" **or** vendor fixed-point implementations of the determinism-critical transcendentals (as was done for the bitmap font). Decision needed before spending effort.

---

## Suggested execution order

1. **P0** (4 fixes, ~½ day): security + host-crash + data-loss. Ship as a patch release.
2. **P1** (8 fixes, ~1 day): every silently-wrong-output bug — highest user-trust impact. The SDXL-AnimateDiff scramble (1.1) needs a reference-comparison regression test.
3. **P2** (10 fixes, ~1 day): input-driven panics — mostly S, several share a fix pattern (multibyte slice, steps=0, recursion caps).
4. **P3 / P4** (8 fixes): memory + training robustness — schedule against real 24 GB runs.
5. **P5** (15 fixes): one hardening sweep; batch by file.
6. Resolve the **map determinism** wording/decision separately.

Each fix: reproduce (add the failing test) → fix → run `cargo test --lib` for the touched
module + the full suite before commit. Keep commits one-bug-each, trailer-clean.
