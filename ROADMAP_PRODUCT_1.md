# plakat product — roadmap (6.9.0, RFC PRODUCT-1)

Studio product-shots / packshots from a subject (cutout / photo / prompt). The weight-free half
(sweep + contact shadow + reflection + composite) ships value alone; only relight + subject-generation
need a model. Structural sibling of `texture` / `bookart` / `comic`; stands on `relight` (IC-Light) +
`matting` + `compose`.

## G0 — de-risk the one novel weight-free algorithm: grounding

The genuinely new piece is **grounding** — turning a subject alpha into a physically-plausible contact
shadow + floor reflection so the subject sits on the ground instead of floating. Everything else is
composition or reuse. Prove it before building the cycle.

- **G0.1 — grounding probe (`examples/product_grounding_probe.rs`) — PASS.** Alpha → ground projection
  (offset + foreshorten by height, fade with height) → clamp to the floor plane (so the blur can't bleed a
  dark halo above the contact line — the one bug found + fixed) → soft-penumbra box blur; + a floor
  reflection (flip about the foot-line, camera-squash, fade). Composite sweep ← reflection ← shadow ←
  subject. **6/6 measures green**: both shadow models anchored at the base, perspective-cast rakes to the
  side, reflection aligned to the foot-line (top row == ground), **no halo (Δ 0.0)**, subject intact.
  **RFC Q2 SETTLED**: soft-contact = default; perspective-cast = a `shadow: "hard"` / low-camera option.
  → **P1 uses this algorithm** (`src/product/ground.rs`); swap the synthetic bottle for a real alpha matte.

## P1 — spec + canvas/sweep + grounding + composite (weight-free) — **DONE (commit pending)**
`src/product/{spec,lint,ground,compose,render,mod}.rs`: `ProductSpec` (permissive serde) → `compose::
resolve` → canvas + **sweep** (white / grey-sweep / gradient) → **grounding** (`ground.rs`, the G0
algorithm generalized to any canvas: `contact_shadow` soft/hard + floor `reflection`, key-dir offset,
camera squash) → place subject (trim-to-alpha + scale/anchor) → composite sweep←reflection←shadow←subject
→ `shot.png` + `shot.meta.json`. Subject = a supplied **cutout** (photo/prompt = P2, errors with a clear
message). CLI `product new|lint|show|render` (weight-free). 7 unit tests + live smoke (soft grey-sweep +
hard/mirror packshots from a mug cutout, no GPU). **Ships a working packshot pipeline with no GPU.**

## P2 — the model half: subject generation + relight — **DONE (commit pending)**
`src/product/render.rs` now async: **subject from a photo** (`matting::matte` → cutout) or **a prompt**
(`api::Generate` → save → matte → cutout); **relight** the cutout to the `lighting` rig via IC-Light
(`lighting_prompt` compiles rig + `key_dir` + `warmth`; `ic_light::Pipeline::load`+`relight` → re-matte
the relit frame). Relight opt-in per RFC Q3: `--relight` / a `lighting:` block on; `--no-relight` off.
Weight-free path preserved (cutout + no relight → no model). CLI `--relight`/`--no-relight`/`--device`;
report shows subject source + relit. **Live-proven Metal**: (1) mug cutout → three-point warm rig relit +
grounded; (2) prompt "glass perfume bottle, gold cap" → generated → U2Net-matted → grounded = a
professional packshot. **Honest caveat**: IC-Light relights *and* recolors — use `warmth: 0` / a neutral
`lighting.prompt` to preserve the product hue (document in P4).

## P3 — catalog: variants, contact sheet, lighting turntable, scenes

Multi-angle `variants` composited with the **same** rig/ground; `product sheet` (contact sheet, lettered
labels via the 5×7 face like `comic`); **lighting turntable** (one subject, key light rotated across N
frames → sheet or gif); optional **generated scene** backgrounds (`bg: "scene"` → `t2i` / replace-bg
path). Honest non-goal reaffirmed: no object-rotation novel-view (supply angle cutouts).

## P4 — integration + corpus + docs + cut 6.9.0

Parity (scenario `type: product` / compile / Bund `plakat.product.*` / `api::Product` / doctor — the
comic P4 template); a demo (`corpus/product_*` + driver + a white-sweep + a scene shot); `PRODUCT.md`
+ README; **CUT 6.9.0** (bump Cargo+lock, gate `--no-default-features --lib`, **pin turbofish on new
`.parse()`**, FF `git push 6.9.0:main`, tag → 6-asset CI, `cargo publish --locked --allow-dirty
--no-default-features`, `gh release edit` + bg waiter, **verify the Windows leg**, NO Claude/Anthropic
coauthor).

## Sequencing
**G0** (grounding algorithm) → **P1** (weight-free packshot pipeline, ships value) → **P2** (relight +
subject-gen) → **P3** (catalog / sheet / turntable / scenes) → **P4** (cut). Front-load the weight-free
half — a supplied cutout → a sellable white-sweep shot with a real contact shadow, no GPU.

## Scope decision (RFC Q1/Q3) — DECIDED by owner 2026-08-10
- **Q1 scope = FULL phased, MVP-first** — build G0→P4 as written; P1 ships the weight-free packshot
  pipeline first, then P2/P3 deepen, P4 cuts 6.9.0.
- **Q3 relight default = OFF** — `product render` keeps the supplied cutout's own light and stays
  weight-free; `--relight` / a `lighting:` block opts in.
- **Q2 (perspective vs soft contact shadow)** — measured *in* G0 (both variants), decided from the probe.

Ready to start **G0** (the grounding probe).
