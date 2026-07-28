# plakat 5.0.0 — roadmap: `plakat persona` (RFC PERSONA-1)

The 5.x flagship. Implements [`Documentation/RFC_PERSONA_1.md`](Documentation/RFC_PERSONA_1.md):
controllable synthetic-person composition — a `PersonaSpec` HJSON → deterministic resolver →
per-family conditioning + geometric map + landmark-anchored localized details + identity reference set
+ measurement scorecard + Q/A TUI. **Fully additive** (no existing flag/output changes); the major
bump is earned by scale.

**Decisions (locked, 2026-07-28):** cut line **P0–P8** (extraction P9 + integration P10 → 5.1);
`figure` **IN** v1 scope; the 106-pt landmark aligner is **net-new** (we build/port it); RFC committed
+ gating research is the first work item.

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

- [ ] **§2.3 baseline measurement** — the control. Per family: 1 fixed dense person-prompt, N=32 seeds,
      native size → pairwise ArcFace cosine (mean/median/p5) + detection-failure rate + localized-detail
      hit rate (prompted mole/scar/piercing: present? correct side?). Reuses render + ArcFace + OWL-ViT.
      Committed as a corpus entry; every later phase reports the same stats.
- [ ] **§27.3 landmark-ControlNet survey** — which families have usable face-landmark/mesh ControlNets,
      at what resolution + license. **A negative result reshapes the architecture toward the swap bridge
      as primary.** Highest-leverage unknown.
- [ ] **106-pt aligner (net-new)** — scope a permissive dense-landmark aligner (InsightFace 2d106det
      class) portable to candle/onnx; it gates Layer 2 + every `landmark`/`local_anomaly`/`region_*`
      probe. Decide port path (candle native vs `convert-onnx`) + license.
- [ ] **§27.5 Gemma prompt-only ceiling** — does Sana/Gemma hold a persona from one long structured
      paragraph? If yes, casting architecture partially inverts (Gemma = canonical casting renderer).

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

## Phase 2 — Geometry engine (no weights; mirrors the map track)

- [ ] 106-pt topology + named anchor regions (§10.1), mean template + open-mouth variant, deformation
      basis (§10.2), conditioning maps (landmark/wireframe/depth/pose/dentition/region-mask/detail-overlay,
      §10.3), figure silhouette + skeleton + body anchor sites (§10.4). `geometry`.
- [ ] Validity clamping + byte-stable rasteriser + corpus.

## Phase 3 — Detail subsystem (partial weights; unusually valuable early, standalone)

- [ ] Anchor resolution against realised landmarks; procedural overlay generators (mole/scar/birthmark/
      freckle-field) with maturity/relief/light-direction (§8.3, §8.8); **jewelry asset library**
      (original/PD, §10.5) with metal recolour; compositing pass + harmonisation (§8.4); `composite`.
- [ ] Determinism (compositing-without-harmonise is byte-stable) + corpus.

## Phase 4 — Calibration (inference; committed tables)

- [ ] Prior measurement (§13.1, defines `0.5`), response curves + inverse pre-distort (§13.2),
      harmonisation constants, controllability grades (§13.3, measured not asserted), recalibration
      policy (§13.4). Committed per-family tables.

## Phase 5 — Identity anchor / casting (weights)

- [ ] Casting (§11.1) + multi-view/expression sheets (§11.2) + curation/storage w/ ArcFace coherence
      (§11.3) + Tier A/B/C (§11.4) + rejection sampling (§12.3). `cast`/`render`.

## Phase 6 — Cross-model (weights; the "all families" promise)

- [ ] Swap-and-restore bridge wiring (§11.5, detail-composite AFTER swap — hard ordering constraint) +
      region-escalation ladder (face/mouth/hand, §14.1) + multiperson attribution (§14.2) + `bake`
      (§11.6, excludes presentation jewelry by default).

## Phase 7 — Repair loop (weights)

- [ ] Attribute-targeted repair (§12.4, mask-source preference: region-mask → detect → anchor) +
      surface/detail-incremental re-cast (§6.5). `repair`.

## Phase 8 — Composition TUI (no weights for Tier-1 preview) — the 5.0.0 authoring surface

- [ ] Headless interview core (§17.2, pure fns + `--answers` replay + nested collection sub-interview
      stack) + lexicon-driven question graph (§17.3) + widgets incl. `place`/`list`/`tooth` (§17.4) +
      Tier-1 wireframe preview + detail markers + Tier-2 few-step preview (§17.5) + workbench + evolve
      (§17.8–9). Reuses ratatui infra + step-hook + workspace-wizard + retouch-crosshair patterns.

## Phase 9 — docs + release (5.0.0 cut)

- [ ] Docs deliverables (§21): PERSONA_TUTORIAL, PERSONA_DETAILS_HOWTO, PERSONA.md, PERSONA_LEXICON.md,
      PERSONA_ANCHORS.md, capability matrix, `doctor` persona section. README what's-new. Cut 5.0.0.

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
