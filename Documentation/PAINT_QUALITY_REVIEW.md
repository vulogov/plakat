# `plakat paint` quality review & 6.37.0 plan

A complete review of the paint tree (stroke engine, canvas/material, armature/plan/analysis, media,
palette/colour, CLI/HJSON surface) against one question: **why is the output low‑grade, and what fixes it?**
Findings are grounded in `file:line`. Priorities are ranked by quality impact.

## Diagnosis — the four killers

The washed / flat / muddy / mushy / cut‑and‑paste look is not one thing; it's four compounding mechanisms, all
on the *default* `paint from --plan auto` path:

1. **The focal detail gate slams shut.** `--plan auto` sets `sam`/`armature_face`, and `run_from` then builds a
   `face_mask`; `painter.rs:798 focus_on = focal.is_some()` becomes true even though `preserve_face` was never
   set, so `focal_strength = 0` and the focal gate (`painter.rs:931`) skips **almost every detail stroke on the
   whole canvas**. No feature resolution anywhere. `PaintPlan` has no `preserve_face`/`focus_detail` field to set
   the strength it implies.
2. **The analyzer defaults `style: impressionist`** (`plan.rs:76,244`), which has **no detail tier** — so scenes
   never run detail passes at all (mush), and portraits compound with killer 1.
3. **The canvas never dries between passes.** `canvas.wetness` is only raised, never decayed (`stroke.rs:244`);
   after the block‑in the whole canvas reads wet, so every later stroke *picks up* the masses beneath and
   re‑deposits contaminated colour → pervasive mud; detail dissolves.
4. **The deposit throttle `(1 - sat)`** (`stroke.rs:215‑219`) prevents building darks/highlights after the
   block‑in → compressed, washed‑out value range (highlights stay grey, darks won't deepen).

Fixing **1 + 2 together** rescues the default path; **3 + 4** remove the mud and restore value range. Everything
else is density, faces, and medium character.

---

## Bugs (ranked, by tier)

### Tier 0 — the killers (above)
- **Q0.1** focal gate shut at strength 0 when a face mask exists — `painter.rs:791‑798, 824, 931‑936`. Fix:
  `focus_on = focal.is_some() && focal_strength > 0.0`; gate is a no‑op when strength ≤ 0.
- **Q0.2** analyzer defaults `impressionist` (no detail tier) — `plan.rs:76,244`; `run_from` overrides user's
  legible (`cli/paint.rs:1267`). Fix: default from‑photo to `legible`.
- **Q0.3** no inter‑pass drying — `stroke.rs:244`, pass loop `painter.rs:799‑1029`. Fix: decay `canvas.wetness`
  between passes; also gate pickup by brush wetness, not just canvas wetness.
- **Q0.4** deposit throttle `(1-sat)` blocks value building — `stroke.rs:215‑219`. Fix: displacement‑on‑cover in
  `Canvas::deposit` (`conc = conc*(1 - cover*k) + delta`) and throttle by paint **height**, not concentration.

### Tier 1 — density / sparse
- **Q1.1** `run_from` paints at **native resolution** with budget capped at 34k (`cli/paint.rs:1312,1580`;
  `plan.rs:238`) — the working‑resolution normalization that `run_spec` has (`cli/paint.rs:913‑925`, cap 1024) is
  missing → ~1 stroke / 175px² on a big photo. Fix: mirror the working‑res block in `run_from`.
- **Q1.2** plain `paint from` (no `--plan`) defaults budget **1500** + no armature → a 1500‑stroke filter from the
  full‑res photo (`cli/paint.rs:362,1306,1522`). Fix: area‑derived default budget; consider `--plan auto` on.
- **Q1.3** `restate_floor` is an **absolute** RGB distance (`painter.rs:910`) → low‑contrast subjects land within
  0.06 everywhere → nearly all restatement skipped → thin single‑pass wash. Fix: make it adaptive to measured
  contrast / guarantee a minimum restatement density in the subject.
- **Q1.4** block‑in under‑covers the ground (streaky flat brush, one pass) → milky/patchy base the restate passes
  can't repair (`painter.rs:849`, `stroke.rs:116`). Fix: a real "cover" block‑in character (low streak, high
  deposit) and/or a toned imprimatura ground.

### Tier 2 — faces / banding
- **Q2.1** face tier **clones the photo** (`structure_armature` `painter.rs:578‑580`, `r<=3`) then hard‑composites
  over the smeared body (`blend_by_mask` `:549‑560`) → "cut‑and‑paste" face; also resolution‑fragile. Fix: never
  clone the focal tier (always a light bilateral); lower `armature_face` (300 is too fine); blend in value space.
- **Q2.2** per‑channel posterize → skin **hue banding** (`painter.rs:594‑601`); and `armature_levels` default is
  **14**, not the documented 6 (`painter.rs:248` vs `cli/paint.rs:470`), so it's barely posterized (filter‑like).
  Fix: posterize in value/luma keeping chroma; real default ~6‑8; reconcile the help.
- **Q2.3** coherence gate kills detail on smoothly‑lit faces (`painter.rs:920‑925`, threshold 0.14). Fix: exempt
  the focal region / lower the threshold there.
- **Q2.4** `focal_field` uses a centre prior even when a real face/subject mask exists (`painter.rs:1408‑1427`) →
  off‑centre subjects get no detail. Fix: seed the focal field from the mask when present.

### Tier 3 — medium character & material
- **Q3.1** **single‑constant KM overstates black's tinting strength** (`pigment.rs:28‑35,63‑78`): a 50/50
  black+white mixes to near‑black; any picked‑up black reads as sludge. **No per‑pigment tinting strength**, so
  high‑tint pigments (phthalo/alizarin/ultramarine/dioxazine) come out pale → the engine structurally can't reach
  deep saturated colour. Fix: per‑pigment `strength` (K/S scale) → 2‑constant KM later.
- **Q3.2** **chroma range too narrow to separate media** (`medium.rs`; finish `canvas.rs:274‑278`) — oil vs
  watercolour differ by ~16% saturation, the only colour difference at output. Fix: widen (vivid oils/acrylic
  1.2‑1.35, gouache 0.7, watercolour 0.85, pastel 1.25).
- **Q3.3** **sheen imperceptible** where it's a signature (oil 0.15, max ~0.13 specular; `canvas.rs:343,360`).
  Fix: oil 0.3‑0.45.
- **Q3.4** **stage schedule is decorative** — `generate_schedule` derives shadow‑thin/light‑opaque/highlights‑last
  (`medium.rs:461`) but `PassSpec` carries no per‑stage opacity and `with_opacity` is one global
  (`painter.rs:730`), so masses read as a flat wash. Fix: drive per‑stage opacity/charge from the stage.
- **Q3.5** **`--haze` default 0.55** fogs every spec‑path painting (`cli/paint.rs:35`; `run_spec` always has depth
  → veils the background). Fix: default 0.0 (opt‑in), consistent with `from`.
- **Q3.6** **`surface.ground` HJSON field is dead** (`spec.rs:16`; `run_spec` hardcodes the ground
  `cli/paint.rs:1041`). A toned imprimatura is a top tonal‑unity lever. Fix: read `surface.ground`.
- **Q3.7** **tempera `stage_budget = 40`** → ~37 degenerate Colour passes fragment the budget (`medium.rs:273`).
  Fix: ~6.
- **Q3.8** gouache negative `dry_shift` collapses to a fixed grey (`canvas.rs:287‑292`). Fix: luma‑preserving
  desaturation toward the pixel's own value.
- **Q3.9** dry media can't be warm — charcoal/pencil default to `sumi` (black+white), and ivory‑black is slightly
  cool (`palette.rs:69`). Fix: a warm‑graphite palette + warm ivory‑black.
- **Q3.10** bleed smears the whole canvas because nothing is dry (`canvas.rs:148‑183`); compounds Q0.3. Fix:
  gradient‑limited, drying‑aware; avoid the per‑iteration clone.
- **Q3.11** families uses a hardcoded key light 135°/40° (`armature.rs:228`) → mis‑lit subjects get brightened.
  Fix: derive the key direction from the bright‑region centroid.
- **Q3.12** plan **notes misreport** the numbers (`plan.rs:195` says body 104/face 200; returns 210/300). Fix.
- **Q3.13** chroma/broken‑colour done in gamma sRGB (`canvas.rs:275`, `painter.rs:419`), mixer ranks with CIE76
  (`mixer.rs:40`) — lower priority; do when widening chroma. Fix: CIELAB / CIEDE2000 later.

---

## Functions to improve
- **`plan_from`** (`plan.rs:182‑260`) — the crudest analyzer: wrong style default; budget model unrelated to the
  working resolution the painter uses; `armature_face`/`armature_body` too fine; `commit_shadows` a flat constant;
  hardcoded key light; wrong notes. Rebuild around the *working* pixel count + measured facts.
- **`structure_armature`** — drop the `img.clone()` fine‑tier shortcut; radius from an absolute px scale (stable
  across input sizes); posterize in value space; quantize on a shared value ladder so tiers blend seam‑free.
- **`blend_by_mask`** — value/linear blend with a feather, not hard per‑channel RGB (kills the cut‑paste seam).
- **The gate stack** (`painter.rs:900‑949`) — four independent `continue`s with no coordination cap density far
  below budget. Introduce a coverage/density target; make `restate_floor` adaptive; decouple `focus_on`.
- **`Stroke::apply` / `Canvas::deposit`** — throttle by height not conc; pickup keyed to brush wetness + drying;
  a covering/displacement term so opaque media actually cover.
- **`pigment::mix_linear`** — per‑pigment tinting strength (the highest‑leverage realism upgrade for colour).
- **`medium.rs` param tables** — the single highest‑leverage character file; give each medium ≥2 signature params
  pushed to clearly visible magnitude (not the current timid spreads).
- **Wire `--armature` into the spec path** (`run_spec` hardcodes `armature_side = None`).
- **Factor `run_from` / `run_spec`** shared setup so `from` inherits the spec path's working‑res density handling.

---

## New control levers (biggest gaps, prioritized)
- **P0 `ground` / imprimatura** (HJSON `surface.ground` + `--ground`) — field exists but dead; single biggest
  missing "make it a painting" lever for opaque media.
- **P0 `tooth` / surface texture** (`--tooth`, HJSON `surface.texture`) — hardcoded 0.85; cold‑press vs canvas vs
  panel radically changes character.
- **P0 `--dry <0..1>`** (default ~0.5) — inter‑pass drying (fixes Q0.3). Highest‑priority stroke lever.
- **P0 `preserve_face`/`focus_detail` as `PaintPlan` fields** — the missing field behind Q0.1; default ~0.7 when
  `sam`/face present.
- **P1 `--coverage`** (block‑in opacity, Q1.4) · **`--commit-lights`** (mirror of `commit_shadows`, opaque
  highlights) · **`--restate-floor`** (expose Q1.3) · **`--detail-coherence`** (expose Q2.3) ·
  **`--density-target`** (coverage floor that overrides the gates to reach the budget) · **`armature_levels` as a
  plan field** (Q2.2) · **`--work-max`** (working‑res cap).
- **P1 material dials on the spec path** — `--chroma/--dry-shift/--granulate/--sheen/--value-key/--lift` exist on
  `paint from` but not on `plakat paint <spec>` (PaintArgs); add for parity. Add `value_key` to the spec HJSON
  (disable‑able).
- **P1 warm‑light / cool‑shadow temperature** (`--light-temp`/`--shadow-temp`) — per‑family temperature (needs
  `--families`); the classic painter move; big perceived‑quality lever.
- **P2 `--pickup` / `--wet`** (expose the hardcoded brush pickup + per‑stroke wetness — dial mud vs crisp) ·
  **`--saturation`** (perceptual chroma, both paths) · **`--edge-variety`** (lost‑and‑found) ·
  **`--posterize-space value|rgb`** (Q2.2) · **`--pigment-strength`/`--tinting`** (Q3.1).

**Mis‑defaults to fix immediately:** `--haze` 0.55→0.0 on the spec path (Q3.5); tempera `stage_budget` 40→~6
(Q3.7); `armature_levels` 14→~7 + reconcile docs (Q2.2); wire `surface.ground` (Q3.6); fix plan notes (Q3.12).

---

## Recommended 6.37.0 sprint order

1. **Sprint 1 — unbreak the default path (the killers).** Q0.1 (focal gate) + Q0.2 (style default) together, then
   Q0.3 (inter‑pass drying) + Q0.4 (value building). Plus the free mis‑defaults (haze, tempera, levels, notes).
   *Expected: washed/mushy → resolved features + real value range on the default path.*
2. **Sprint 2 — density & coverage.** Q1.1 (working‑res in `run_from`) + Q1.2 (budget default) + Q1.3 (adaptive
   restate) + Q1.4 (cover block‑in). New levers: `--dry`, `--coverage`, `--density-target`, `--restate-floor`,
   `preserve_face` plan field.
   *Expected: no more sparse/washed regardless of subject size or contrast.*
3. **Sprint 3 — faces & seams.** Q2.1 (no clone, value blend) + Q2.2 (value posterize) + Q2.3/Q2.4 (focal detail
   + mask‑seeded focal). Lever: `--posterize-space`, `--detail-coherence`.
   *Expected: natural, non‑pasted faces; off‑centre subjects get detail.*
4. **Sprint 4 — medium character & colour.** Q3.1 (pigment tinting strength) is the big one; Q3.2/Q3.3 (chroma +
   sheen re‑tune), Q3.4 (per‑stage opacity), Q3.6/P0 (ground), P0 (tooth), Q3.8/Q3.9 (gouache/dry warmth). Levers:
   `--ground`, `--tooth`, `--commit-lights`, `--light-temp`/`--shadow-temp`, spec‑path material parity.
   *Expected: media read distinctly; deep saturated colour reachable; tonal unity from a toned ground.*
