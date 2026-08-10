# RFC PRODUCT-1 — `plakat product`: studio product-shots / packshots

**Status:** draft (6.9.0 kickoff) · **Flagship:** the 6.9 cut · **Sibling of:** `texture` / `bookart` /
`comic` (structured-authoring studios) and `relight` / `replace-bg` (the pieces it stands on).

## Summary

`plakat product` turns a **subject** — a cutout PNG, a photograph, or a text prompt — into a clean,
repeatable **studio product-shot**: the subject on a controlled background (white / grey sweep /
gradient / generated scene), lit by a named **lighting rig**, **grounded** with a physically-plausible
contact shadow and/or floor reflection, at a chosen camera angle. It is **fully additive** — no existing
command or output changes.

Like every plakat studio, a product-shot is **structured data**, not a prompt. A packshot is about two
things a text prompt cannot deliver: **repeatability** (the *same* lighting and grounding across a whole
catalog of different products) and **physical plausibility** (a subject that sits on the ground with a
real contact shadow, on a background that is actually pure white). "a product photo of a red sneaker on
white" gives inconsistent light frame-to-frame, a subject that floats, a grey-ish "white", and no way to
reproduce the look on the next SKU. So the shot is authored in a small HJSON document (a `ProductSpec`)
and resolved deterministically: **subject → light → ground → composite**.

## The output contract

`product render spec.hjson --out shot.png` writes a finished packshot PNG (+ a `shot.meta.json` sidecar:
the resolved rig, camera, ground, and a spec-hash for reproducibility). `product sheet` writes a
**contact sheet** of a subject's angle/lighting variants.

## The weight-free thesis

The half that makes a packshot a packshot — the **background sweep**, the **contact shadow**, the **floor
reflection**, and the **composite** — is pure CPU, derived from the subject's **alpha matte**. Only two
steps need a model: **relighting** the subject to a rig (IC-Light) and **generating** a subject/scene when
one isn't supplied. So with a supplied cutout and `--no-relight`, a full catalog renders with **no GPU**.

```
ProductSpec ─▶ subject (cutout | photo→matte | prompt→gen→matte)
            ─▶ [relight to rig]          (IC-Light — optional, model)
            ─▶ ground (contact shadow + reflection from the alpha)   ← the novel weight-free algorithm
            ─▶ composite onto the sweep / scene ─▶ shot.png + sheet
```

## The `ProductSpec`

Permissive HJSON (every field optional; unknown keys ignored; enums carried as strings — lint warns, not
fails), exactly like `TextureSpec` / `ComicSpec`. `product new` scaffolds one.

```hjson
{
  schema: "product/1"

  subject: {
    image:   "sneaker.png"          // a cutout (transparent) — kept pixel-exact (no VAE roundtrip)
    // or  photo:  "sneaker_photo.jpg"   // matte it (U2Net) first
    // or  prompt: "a red running sneaker, product photo"   // generate → matte
    scale:   0.7                    // fraction of the canvas the subject fills
    anchor:  "bottom"               // where the subject sits (bottom = grounded)
  }

  canvas:  { size: "square", px: 1024, bg: "white" }   // white | grey-sweep | gradient:top,bottom | scene
  scene:   { prompt: "a marble kitchen counter, soft daylight" }   // only when bg: "scene"

  lighting: {
    rig:       "three-point"        // three-point | softbox | beauty | rim | hard | flat
    key_dir:   "top-left"           // dominant light direction
    intensity: 1.0
    warmth:    0.0                  // -1 cool … +1 warm
    // prompt: "…"                  // free-text override fed to IC-Light
  }

  camera:  { angle: "three-quarter" }   // eye | hero(low) | top(flatlay) | three-quarter

  ground: {
    shadow:     "soft"              // soft | hard | none  (contact shadow under the subject)
    softness:   0.5
    reflection: "gloss"             // none | gloss | mirror  (floor reflection for glossy products)
    falloff:    0.6
  }

  // a catalog: extra angles composited with the SAME rig/ground (see `product sheet`).
  variants: [ { image: "sneaker_side.png", label: "side" }, { image: "sneaker_top.png", label: "top" } ]

  model: "sdxl"  seed: 7  steps: 30
}
```

## The four pieces

1. **Subject resolution** — a transparent **cutout** is used as-is (composited, pixel-exact — logos and
   labels never distort); a **photo** is matted (`matting::matte` → RGB + alpha); a **prompt** is
   generated (`api::Generate`) then matted. The subject always reduces to *RGB + an alpha matte*.

2. **Relight (model, optional)** — the cutout is re-illuminated to the `lighting` rig via **IC-Light**
   (`pipelines::ic_light`), which plakat already exposes as `relight`. The rig + `key_dir` + `warmth`
   compile to the IC-Light lighting prompt; `--no-relight` skips it and keeps the subject's own light.

3. **Ground — the novel weight-free algorithm.** From the subject **alpha** + the camera angle, derive a
   physically-plausible **contact shadow** (project the alpha to the ground plane, offset by `key_dir`,
   blur ∝ distance-from-contact for a soft penumbra, fade with `falloff`) and an optional **floor
   reflection** (vertical-flip the subject, perspective-squash by camera angle, fade with `falloff`,
   slight blur). This is the piece that grounds the subject so it doesn't float — the equivalent of
   `comic`'s balloon-placement algorithm, and the thing G0 de-risks.

4. **Composite** — sweep/scene background ← reflection ← shadow ← subject, in z-order, on the canvas at
   the subject `scale`/`anchor`. Reuses the `compose` layering primitives.

## CLI

```
plakat product new    <out.hjson>                          # scaffold a spec
plakat product lint   <spec.hjson>                         # schema / vocab / cross-refs (no GPU)
plakat product show   <spec.hjson>                         # resolved rig / camera / ground / canvas
plakat product render <spec.hjson> --out shot.png [--subject <img>] [--no-relight] [--device auto]
plakat product sheet  <spec.hjson> --out sheet.png         # contact sheet of variants / angles
```

`--subject <img>` overrides `subject.image`. With a cutout + `--no-relight`, `render`/`sheet` are
**weight-free**.

## Integration (parity, the flagship template)

Wired exactly like `texture`/`comic`: `plakat::api::Product` builder · scenario `type: "product"` task ·
`plakat compile` (`type: product`) · Bund `plakat.product.*` · `plakat doctor` · `Documentation/PRODUCT.md`
+ a corpus.

## Reuse

`pipelines::ic_light` (relight rigs) · `pipelines::matting::matte` (subject cutout + **alpha**, the
shadow/reflection input) · `api::Generate` / `replace-bg` (generate a subject or a scene bg) · the
`compose` layering primitives (z-order composite) · the studio-flagship template
(spec/lint/render/scenario_task + api/Bund/doctor parity).

## Non-goals (honest, v1)

- **True 3D turntable / novel-view synthesis.** plakat has no NeRF/multi-view. v1 does **lighting
  turntables** (same view, the key light rotated across a set) and accepts **supplied angle cutouts** for
  a real multi-angle catalog. Object-rotation from one image is out of scope.
- **Ray-traced physical accuracy.** The contact shadow and reflection are *plausible approximations* from
  the alpha, not a physically-based render.
- **Packaging text fidelity.** A generated subject may distort logos/labels; supply a **cutout** to keep
  the product pixel-exact (composited, no VAE roundtrip).

## Open questions (for the owner)

- **Q1 — scope:** full studio (multi-rig + lighting turntable + reflections/shadows/sweeps + generated
  scenes + contact sheets) phased over P1–P4, or an MVP first (single subject on a sweep + contact shadow
  + a couple of rigs)? *Recommendation:* the full phased build, **MVP-first** — P1 ships a weight-free
  packshot pipeline that already produces sellable white-sweep shots.
- **Q2 — grounding realism bar:** is a soft alpha-projected contact shadow + gloss reflection enough for
  v1, or do we want a horizon-aware perspective shadow (needs the camera angle to skew the projection)?
  G0 measures both.
- **Q3 — relight default:** relight ON by default (rig-consistent look, but a model step) or OFF (keep the
  supplied cutout's light, fully weight-free)? *Recommendation:* OFF by default; `--relight` / `lighting`
  opts in.
