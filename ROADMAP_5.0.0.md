# plakat 5.0.0 — roadmap: `plakat persona` (RFC PERSONA-1)

The 5.x flagship. Implements [`Documentation/RFC_PERSONA_1.md`](Documentation/RFC_PERSONA_1.md):
controllable synthetic-person composition — a `PersonaSpec` HJSON → deterministic resolver →
per-family conditioning + geometric map + landmark-anchored localized details + identity reference set
+ measurement scorecard + Q/A TUI. **Fully additive** (no existing flag/output changes); the major
bump is earned by scale.

**Decisions (locked, 2026-07-28):** cut line **P0–P8** (extraction P9 + integration P10 → 5.1);
`figure` **IN** v1 scope; the 106-pt landmark aligner is **net-new** (we build/port it); RFC committed
+ gating research is the first work item. **Landmark topology v1 = WFLW-98 (FROZEN)** — port PIPNet-98
(MIT); InsightFace 2d106det rejected (non-commercial license). Face-swap bridge is the PRIMARY
cross-family geometric/identity path (landmark-CN is SD1.5/2.1-only). Converted PIPNet-98 weights host
on the user's HF repo (the one weight hosting the persona track adds).

Ground rules: additive; determinism contract (§5.2) — resolver/geometry/detail-plan/compositing are
pure and byte-stable, corpus-tested, no weights; each phase carries corpus + CI gates; `Cargo.lock` in
sync; no Anthropic/Claude attribution anywhere. Neutrality + age-gate constraints (§7.4, §23) are
binding on every contribution.

Status: `[ ]` open · `[x]` done · `[~]` in progress.

---

## Substrate that already exists (reused, not rebuilt)

| RFC dependency | plakat surface |
|---|---|
| Face detect / identity embed / swap+restore (Tier B, §11.5) | SCRFD + ArcFace + inswapper + `restore-faces` |
| Open-vocab `detect` probe + extraction mark-sweep (§12.1, §16) | **OWL-ViT** (4.10) |
| Region masks, selection, targeted repair (§10.3, §12.4) | SAM + `segment` + `remove`/`replace-bg` (4.9–4.11) |
| U2Net matte, inpaint, regional prompting | `transparent --matte`, img2img `--mask`, `--region` |
| Artefact compositing (basis for the detail pass, §8) | artefact library + masked-img2img blend + colour harmony |
| TI / LoRA / DreamBooth (bake, §11.6) | `embedding train`, `style train` |
| Few-step live preview (§17.5 Tier 2) | lightning/hyper-sdxl (4.4) + step-hook cancel |
| Aesthetic secondary sort (§11.1) | LAION ranker |
| IP-Adapter-Plus-Face / FaceID (Tier A) | `ip_adapter.rs` / `portrait` |
| Depth + pose skeleton conditioning (§10.3/10.4) | Depth-Anything, multiperson pose |
| Prose→spec provider pattern (§17.6) | fractals `--fractal-provider` (LLM + offline keyword fallback) |
| scenario / compile / scripting / library-API / TUI infra (§20) | all shipped |
| Determinism + hand-authored-asset discipline (§5.2, §10.2) | the `map` track (geometry engine + bitmap font precedent) |

---

## Phase G — Gating research + decisions (FIRST; RFC flags as architecture-shaping)

Findings recorded in [`Documentation/PERSONA_GATING.md`](Documentation/PERSONA_GATING.md).

- [x] **§27.3 landmark-ControlNet survey — DONE.** Dedicated face-mesh CN exists ONLY for SD1.5/2.1
      (`CrucibleAI/ControlNetMediaPipeFace`, OpenRAIL-M → optional add-on, not bundled). SDXL/SD3.5/Flux
      have only coarse face keypoints in OpenPose-union CNs; PixArt/Sana/Cascade have none. **DECISION
      (negative result, as the RFC anticipated): the face-swap bridge (SCRFD→ArcFace→inswapper, already
      in plakat) is the PRIMARY cross-family geometric/identity path; the face-landmark-CN map is an
      SD1.5/2.1-only Tier-A bonus.** Layer 2's cross-family value = depth/pose/region-mask/detail-overlay.
- [x] **106-pt aligner (net-new) — DECIDED: port PIPNet-98 (WFLW, MIT, prebuilt ONNX).** InsightFace
      `2d106det` REJECTED (non-commercial weights, license-poisoned). **Topology v1 = WFLW-98, NOT the
      RFC's assumed 106-pt InsightFace set** — feed back into the RFC/lexicon; every named anchor region
      (§8.2/§10.1) + probe is defined on WFLW-98. Port via `convert-onnx` (ResNet-18 + pixel-in-pixel).
      MediaPipe FaceMesh (Apache, 468) is the escape hatch if a dense 3D mesh is later needed.
- [ ] **§2.3 baseline measurement** — harness committed (`tools/reference/persona_baseline.py`;
      InsightFace SCRFD+ArcFace variance + detection-failure + OWL-ViT detail-hit). Measurement runs on
      any image dir today; the **multi-family render is a compute job to schedule.** Commit results as a
      corpus entry; every later phase reports the same stats. (Detail-hit is a noisy OWL-ViT proxy until
      the Phase-1 `local_anomaly` probe replaces it.)
- [ ] **§27.5 Gemma prompt-only ceiling** — does Sana/Gemma hold a persona from one long structured
      paragraph? If yes, casting architecture partially inverts (Gemma = canonical casting renderer).
      (Cheap experiment; run alongside the baselines.)

**Net architectural consequence:** with landmark-CN unavailable cross-family, the swap-bridge is the
backbone and the geometry engine (Layer 2) is primarily a *measurement + region-mask + detail-anchor*
engine (still fully deterministic, no weights) rather than a face-mesh-CN generator. This does not
change the phase order; it changes the emphasis within P2/P5/P6.

## Phase 0 — `PersonaSpec` + resolver (no weights, deterministic)

- [ ] Schema v1 (§6.2, incl. `figure`), lexicon skeleton (§7), pure resolver + manifestation gate (§8.6)
      + detail routing (§8.9) + salience/budget solver (§9.2) + per-encoder-class emitters (§9.3) +
      negative assembly (§9.4) + precedence (§9.5). CLI: `lint`/`show`/`new`/`migrate`/`set`.
- [ ] Structural corpus (byte-stable spec→resolved→per-family prompt) + property tests (§24).
- [ ] **Naming resolution (§4)** adopted: subcommand `persona`, library stays People, scenario
      `personas:` extended to accept spec paths.

## Phase 1 — Scorecard (built BEFORE generation; the central sequencing arg)

- [ ] Probes (§12.1): `landmark`, `region_color`, `local_anomaly`, `region_structure` (net-new) +
      `detect`(OWL-ViT)/`clip_probe`/`identity`(ArcFace)/`aesthetic`(LAION) reusing existing. Scoring
      with the four exclusions + detail sub-score (§12.2). `verify`.
- [ ] Synthetic-ground-truth test for `local_anomaly` (compositor + probe are each other's test, §24).
- [ ] Fold the §2.3 baselines into the verify harness as a tracked number.

## Phase 2 — Geometry engine (no weights; mirrors the map track) — DONE

- [x] **WFLW-98 topology** (Phase-G decision, not the RFC's 106-pt InsightFace) + named anchor regions
      (§10.1), mean template + open-mouth variant, deformation
      basis (§10.2), conditioning maps (landmark/wireframe/depth/pose/dentition/region-mask/detail-overlay,
      §10.3), figure silhouette + skeleton + body anchor sites (§10.4). `geometry`.
- [x] Validity clamping + byte-stable rasteriser + corpus.

**Shipped** in `src/persona/geometry/` (topology/template/basis/raster/figure/from_spec) + the
`plakat persona geometry` subcommand. Pure Rust, no weights, byte-stable (per-layer golden hashes).
P2a topology+mean-template · P2b deformation basis + `resolve()` (10 scalar attrs, anatomical
coupling, seed asymmetry, validity clamp→lint) · P2c rasteriser (mesh/wireframe/depth/skeleton/
region-masks/dentition/detail-overlay; reuses openpose_post LIMB conventions + Depth-Anything
foreground-bright convention) · P2d figure (18-kp skeleton + capsule silhouette + body anchors;
honest weak scope) · P2e CLI + spec→geometry bridge + folded scorecard's duplicate topology onto
`geometry::topology`. `examples/persona/example.hjson` runnable.

## Phase 3 — Detail subsystem (partial weights; unusually valuable early, standalone) — DONE

- [x] Anchor resolution against realised landmarks; procedural overlay generators (mole/scar/birthmark/
      freckle-field) with maturity/relief/light-direction (§8.3, §8.8); **jewelry asset library**
      (original/PD, §10.5) with metal recolour; compositing pass + harmonisation (§8.4); `composite`.
- [x] Determinism (compositing-without-harmonise is byte-stable) + corpus.

**Shipped** in `src/persona/detail/` (overlay/composite) + `plakat persona composite`. P3a procedural
overlay generators (mole/scar w/ maturity ramp + relief/birthmark edge-noise/freckle-field/jewelry
stud-hoop-pendant-bar w/ metal+stone recolour — all pure, byte-stable, license-free, no PNG assets
needed). P3b compositing pass: anchors resolve through the REALISED landmarks (crop→full via new
`FaceMetrics.crop_origin`), z-order, scene-light estimate from face shading, culling reported,
DETAIL_CAP=24. P3c CLI + weights-free corpus golden (synthetic FaceMetrics from mean template) +
optional `--harmonise` (low-strength masked img2img via `api::Img2img`). Jewelry+piercings composite
at face sites; body/hand/glasses culled+reported (§8.5). **Notes/deferred within P3:** jewelry is
*procedural generic shapes* not a bundled PNG library (extensible later); scar `hair_interruption`
`condition` component (§8.8) and dentition mouth-region *inpaint escalation* (§8.7) not wired — the
latter belongs with P7 repair; harmonise wired but not run live (needs model weights).

## Phase 4 — Calibration (inference; committed tables) — DONE (infra + bootstrap; sweep = compute job)

- [x] Prior measurement (§13.1, defines `0.5`), response curves + inverse pre-distort (§13.2),
      harmonisation constants, controllability grades (§13.3, measured not asserted), recalibration
      policy (§13.4). Committed per-family tables.

**Shipped** in `src/persona/calibration/` (fit/table) + `plakat persona calibrate` +
`assets/persona/calibration/{sd15,sdxl}.hjson` + `Documentation/PERSONA_CALIBRATION.md`. P4a table
schema + HJSON loader + `Prior::normalise` + `staleness()` (§13.4) + committed **provisional bootstrap**
tables. P4b pure fit/`predistort`/`grade_from` math (isotonic-monotone, inverse transfer, grades from
slope/monotonicity/variance). P4c wired: `verify --model` SCORES the 3 aligner scalars vs prior (out of
`pending_calibration`), `geometry --calibrate` pre-distorts the deformation, `show` shows grade badges.
P4d `calibrate --bootstrap` (regen from mean-template + lexicon) / `--from <dir>` (measure a
`<attr>__<requested>__<seed>.png` sweep → real priors+curves+grades). **The multi-family render sweep is
the scheduled offline compute job** (like §2.3 baselines); the measurement half runs today. Tables are
provisional until swept.

## Phase 5 — Identity anchor / casting (weights) — DONE

- [x] Casting (§11.1) + multi-view/expression sheets (§11.2) + curation/storage w/ ArcFace coherence
      (§11.3) + Tier A/B/C (§11.4) + rejection sampling (§12.3). `cast`/`render`.

**Shipped** in `src/persona/casting/` + `plakat persona cast` / `persona render` +
`Documentation/PERSONA_CASTING.md`. P5a reference-set storage (`Reference`/`ReferenceSet`, JSON
manifest + images) + pure ArcFace coherence math (cosine/centroid/`compute_coherence`, threshold 0.50,
worst-pair check). P5b `cast`: compile → render N → composite details → `score_render` (calibrated
scalars + eyes.color) → rank (conformance primary, `--aesthetic` LAION secondary) → keep-best →
ArcFace-embed → coherence-validate → store. P5c `render` = Tier-B swap bridge (§11.5): generate →
SCRFD → swap canonical face → restore → detail-composite AFTER swap (hard ordering). P5d Tier A
(IP-Adapter via `api::Portrait`) + `--tier auto|A|B` + rejection sampling (`--min-score`). **Live-
verified** (release, sd15): cast wrote a reference set + coherence correctly flagged 3 unconditioned
renders as different people; render produced a coherent garden scene with the swapped+restored face.
**Deferred:** geometry-CN + multi-view/pose-CN casting (both need `t2i::Request.controls`, not on the
`api` facade); Tier C baking (§11.6 → P6); resident render worker (§22, per-candidate model reload).

## Phase 6 — Cross-model (weights; the "all families" promise) — DONE

- [x] Swap-and-restore bridge wiring (§11.5, detail-composite AFTER swap — hard ordering constraint) +
      region-escalation ladder (face/mouth/hand, §14.1) + multiperson attribution (§14.2) + `bake`
      (§11.6, excludes presentation jewelry by default).

**Shipped.** Swap bridge = P5c `render`. P6a `casting::escalation` (area_fraction + per-region
thresholds + decide + refine_crop) wired into `render` — small face → native-res + stronger restore;
mouth/hand rungs advisory (mouth-inpaint = P7, hand = §8.5 best-effort). P6b `casting::attribution`
(containment+IoU assignment, one-to-one, **refuse below 0.35 → detail absent not misplaced**, the §14.2
catastrophic-failure guard) wired into `render --with` multiperson (left-to-right figure bands, per-
figure detail composite). P6c `persona bake` (Tier C): TI/LoRA from the reference set via the proven
trainers, **excludes worn presentation jewelry by default** (recomposites from stored raw candidates
via `composite_details_opts(false)` unless identity_locked/`--keep-jewelry`), memory-gated
(`memory_preflight` + `MemoryGuard` Training mode), writes a `.bake.json` invalidation record.
Live-verified (release/sd15): escalation branch + multiperson 2-persona scene (attribution + dual
swap + spurious-face refusal). **Deferred:** bake not run live (heavy training); sequential per-figure
refinement for N>2 (§14.2) uses the same swap path.

## Phase 7 — Repair loop (weights) — DONE

- [x] Attribute-targeted repair (§12.4, mask-source preference: region-mask → detect → anchor) +
      surface/detail-incremental re-cast (§6.5). `repair`.

**Shipped.** P7a `persona/edit.rs` + `persona diff` (§6.5): class_of → Structural/Surface/Detail/
Presentation; diff two HJSON specs → classified changed leaves + summary reporting whether the edit
invalidates the reference set (only structural does). P7b `persona repair --attr` (§12.4): class-routed
— detail/presentation → recomposite (deterministic), surface → landmark-region-masked inpaint with the
attribute's phrase re-scored + kept only on improvement (eyes.color ΔE, revert on regression),
structural → reported re-cast-only. `feature_mask` builds the feathered region mask from realised
landmarks (mask source #1). P7c `compile::dentition_prompt` + mouth-region inpaint (§8.7, the deferred
escalation): inner-lip aperture mask + dentition prompt, wired into render's mouth rung + `repair
--attr teeth.*`. Live-verified deterministic branches (diff classification, detail recomposite,
structural report). **Deferred:** surface/dentition inpaints not run live (reuse the proven
img2img-mask path); dentition_hint as a ControlNet cond needs the lower-level t2i path.

## Phase 8 — Composition TUI (no weights for Tier-1 preview) — the 5.0.0 authoring surface — DONE (core)

- [x] Headless interview core (§17.2, pure fns + `--answers` replay + nested collection sub-interview
      stack) + lexicon-driven question graph (§17.3) + widgets incl. `place`/`list`/`tooth` (§17.4) +
      Tier-1 wireframe preview + detail markers + Tier-2 few-step preview (§17.5) + workbench + evolve
      (§17.8–9). Reuses ratatui infra + step-hook + workspace-wizard + retouch-crosshair patterns.

**Shipped.** P8a `persona/interview.rs` — headless engine: question_graph from the lexicon (§17.3,
coarse→fine), next_question/apply/progress, total `when` condition language, Answer (Unknown≠middle,
NoneEmpty), to_partial_spec/spec_from_map; `persona interview <out> [--depth] [--answers]` (§17.12
scriptable replay + headless graph preview). Lexicon gained ask/widget/depth/order/when (derived
defaults). P8b `persona/preview.rs` (braille Tier-1 wireframe, usable over SSH) + `persona/tui.rs`
(feature `ui`) — interactive ratatui view over the engine with a LIVE slider-tracking wireframe;
`persona interview --tui`. P8c cores: `geometry::nearest_anchor` (place-widget drop → nearest named
anchor + offset, anatomical from creation) + `interview::mark_member_questions` (list-widget collection
sub-interview). **Deferred (interactive UI, low verifiability w/o a PTY):** the crosshair place / list /
tooth widgets wired into the TUI loop, Tier-2 debounced diffusion preview (ChannelHook+CancelFlag), and
workbench/evolve (§17.8–9).

## Phase 9 — docs + release (5.0.0 cut) — DOCS DONE; CUT PENDING SAMPLE CORPUS

- [x] Docs deliverables (§21): PERSONA.md, PERSONA_TUTORIAL.md, PERSONA_DETAILS_HOWTO.md,
      PERSONA_LEXICON.md, PERSONA_ANCHORS.md; `doctor` persona section; README what's-new.
- [x] `./corpus`: two authored personas (mira, idris) + `persona_run.sh` feature driver (sd15/sd35) +
      PERSONA_CORPUS.md walkthrough (`persona`-specific; the gallery-index README was restored).
- [ ] **Generate the sample corpus** (`corpus/persona_run.sh`) — the user runs this before the cut.
- [ ] **Cut 5.0.0**: bump Cargo.toml **+ Cargo.lock** in sync; `cargo test --no-default-features --lib`
      green; commit; FF `main`; push `v5.0.0` tag → CI 6-asset release + crates.io; `gh release edit`
      notes. (Release-flow gotchas in the auto-memory.) **Held until the sample corpus lands.**

## Cross-cutting (span all phases)

- [ ] **Resident scoring worker + memory residency** (§22) — 5 scoring models on top of a pipeline;
      generalize the Sana staged-free discipline; probe tiering (`local_anomaly` needs no model beyond
      the aligner); hard memory guards + cost estimates.
- [ ] **Age-gate in the RESOLVER** (§23.1) — no surface bypasses; lint+emit+render redundant; lineage
      re-runs it.
- [ ] **Lexicon neutrality review process** (§7.4/§23.3/§23.5) + jewelry trade-dress review (§10.5).
- [ ] Verify-harness tiers (§24) + corpus scripts from P0 (P0–P3 mostly deterministic → CI without GPUs).

## Deferred to 5.1+ (out of the 5.0.0 cut)

- P9 extraction (photo→spec + mark sweep, §16) + lineage ops (`derive`/`blend`/`vary`, §15).
- P10 full integration parity (scenario/compile/scripting/API/UIs/multiperson/sidecar, §20) — a minimal
      scenario `type: persona` hook may land in 5.0.0 if cheap; full parity is 5.1.
- Future work §29 (body identity, procedural 3D head, learned overlays, etc.).

## Notes / risks (see RFC §26 for the full table)

- R1 landmark-CN availability (Phase G resolves) · R5 scoring residency OOM (resident worker) ·
  R8 schema/topology churn (versioned topology + `lock.hjson`) · R11/R12 composited marks read as
  stickers / harmonisation erases them (maturity+relief+light + `local_anomaly`-calibrated strength).
- This is a multi-cycle major. Intermediate phases are independently useful; whether any ship as 4.x
  point releases before the 5.0.0 cut is a per-phase call with the user.
