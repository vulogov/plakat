# plakat product — roadmap (6.9.0, RFC PRODUCT-1)

Studio product-shots / packshots from a subject (cutout / photo / prompt). The weight-free half
(sweep + contact shadow + reflection + composite) ships value alone; only relight + subject-generation
need a model. Structural sibling of `texture` / `bookart` / `comic`; stands on `relight` (IC-Light) +
`matting` + `compose`.

## G0 — de-risk the one novel weight-free algorithm: grounding

The genuinely new piece is **grounding** — turning a subject alpha into a physically-plausible contact
shadow + floor reflection so the subject sits on the ground instead of floating. Everything else is
composition or reuse. Prove it before building the cycle.

- **G0.1 — grounding probe (`examples/product_grounding_probe.rs`).** Given a subject cutout (alpha) + a
  camera angle + `key_dir`, produce (a) a **contact shadow** — alpha projected to the ground plane,
  offset opposite the key light, blurred with a distance-from-contact penumbra, faded by `falloff`; (b) an
  optional **gloss reflection** — vertical-flip + perspective-squash by angle + fade + slight blur;
  composite both onto a white/grey sweep under the subject. **Measure:** shadow anchored at the subject's
  base (darkest at contact, soft away), reflection aligned to the subject foot-line, no bright halo around
  the cutout, subject unaltered. PASS → P1 uses the algorithm; measure the perspective-shadow variant too
  (RFC Q2).

## P1 — spec + canvas/sweep + grounding + composite (weight-free; front-loaded)

`src/product/{spec,lint,compile,ground,compose,mod}.rs`: `ProductSpec` (permissive serde) → resolve →
canvas + **sweep** (white / grey-sweep / gradient) → **grounding** (from G0) → composite subject (scale/
anchor) → `shot.png` + `shot.meta.json` sidecar. Subject = a supplied **cutout** (photo→matte reuse is
P2). CLI `product new|lint|show|render` (weight-free). **Ships a working packshot pipeline with no GPU.**

## P2 — the model half: subject generation + relight

`src/product/render.rs`: **subject from a photo** (`matting::matte`) or **from a prompt** (`api::Generate`
→ matte); **relight** the cutout to the `lighting` rig via IC-Light (rig + `key_dir` + `warmth` → the
lighting prompt), `--no-relight` / relight-off default (RFC Q3). `product render` full. Verify a rig looks
consistent across two different subjects.

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

## Scope decision (RFC Q1) — pending owner
Recommendation: the full phased build, **MVP-first** (P1 alone is a shippable weight-free packshot tool).
Await the owner's call on scope + Q2 (perspective shadow) + Q3 (relight default) before starting G0.
