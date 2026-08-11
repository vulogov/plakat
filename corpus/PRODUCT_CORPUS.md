# `plakat product` — corpus walkthrough

A reproducible demonstration of the PRODUCT-1 feature (plakat 6.9). A subject cutout becomes a **studio
product-shot** — a sweep, a grounded contact shadow, and a floor reflection, all from the subject's alpha
matte; run [`product_run.sh`](product_run.sh) to regenerate the images under `corpus/images/product/`.

```bash
cargo build --release --features metal   # once (only the model step needs it)
corpus/product_run.sh                     # everything (weight-free packshots + a generated subject)
RENDER=0 corpus/product_run.sh            # weight-free only (no GPU)
```

## The subjects

Two committed cutouts — [`subject_bottle.png`](images/product/subject_bottle.png) (front) and
[`subject_bottle_side.png`](images/product/subject_bottle_side.png) (a side angle). Paths in a
`ProductSpec` are relative to the working directory (repo root, where the driver runs), like all plakat
paths.

## What the driver produces

1. **`product lint` / `product show`** — validate the spec and print the resolved plan (canvas, camera,
   ground). *(No GPU.)*
2. **`bottle.png`** — a grounded packshot from [`product_bottle.hjson`](product_bottle.hjson): the bottle
   on a grey studio sweep with a **soft contact shadow** + a **gloss floor reflection**, both derived from
   the cutout's alpha. *(No GPU.)*
3. **`catalog_sheet.png`** — a catalog **contact sheet** from [`product_catalog.hjson`](product_catalog.hjson):
   the front + side angles rendered with the **same rig / grounding**, tiled + labelled. *(No GPU.)*
4. **`generated.png`** — the model half: a subject **generated from a prompt** ("a glass perfume bottle
   with a gold cap"), **matted** (U2Net), and grounded. *(Needs a model.)*

## The idea

A packshot is **structured data**, not a prompt. The two things a prompt can't give you — **repeatability**
(the same lighting + grounding across a whole catalog of different products) and **physical plausibility**
(a subject that sits on the ground with a real contact shadow, on a background that is actually pure
white) — are exactly what a `ProductSpec` resolves deterministically: **subject → light → ground →
composite**. The grounding (contact shadow + reflection from the alpha) is the novel weight-free piece, so
a supplied cutout → a sellable white-sweep shot with no GPU. See
[`Documentation/PRODUCT.md`](../Documentation/PRODUCT.md).
