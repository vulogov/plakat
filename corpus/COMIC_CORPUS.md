# `plakat comic` — corpus walkthrough

A reproducible demonstration of the COMIC-1 feature (plakat 6.8). A `ComicSpec` becomes a **lettered,
multi-panel comic page**; run [`comic_run.sh`](comic_run.sh) to regenerate the images under
`corpus/images/comic/`.

```bash
cargo build --release --features metal   # once
corpus/comic_run.sh                       # the whole corpus (weight-free steps + the model render)
RENDER=0 corpus/comic_run.sh              # weight-free only (no GPU): layout + lettering
```

## The spec

[`comic_noir.hjson`](comic_noir.hjson) authors a six-panel noir page: a `us-letter` grid of
`[[1,1],[1],[1,1,1]]` (two panels, one wide panel, three panels), one recurring `describe:` character
(seed-locked so she looks the same across panels), a shared noir `style`, and per-panel `scene` prompts
with `caption`s and `balloons` (speech · thought · shout).

## What the driver produces

1. **`comic lint` / `comic show`** — validate the script (schema, vocabulary, cast cross-references) and
   print the resolved plan: page pixel size, each panel's rectangle, reading order, cast. *(No GPU.)*
2. **`noir_layout.png`** — the page skeleton: panel frames + placeholders + the `panels.json` sidecar
   (each panel's page rect + reading index). *(No GPU.)*
3. **`noir_lettered_placeholder.png`** — the same page with the **balloons and captions placed and
   lettered** over the placeholders. This is the novel weight-free algorithm on its own: fit → place →
   draw, with the four balloon kinds (speech tail · thought bubble-trail · shout burst · tinted caption).
   *(No GPU.)*
4. **`noir.png`** — the full flagship: each panel's **scene art generated** at the panel aspect, the
   **recurring cast** injected into every panel it appears in, composited, and lettered **face-aware**
   (when a face detector is configured, tails point at the actual face and balloons steer clear of it).
   Per-panel PNGs are kept under `panels/`. *(Needs a model.)*

## Multi-page & reference-lock (6.8.1)

[`comic_series.hjson`](comic_series.hjson) holds the **shared world** — cast, style, engine, page format,
and a named `scenes` library. [`comic_issue.hjson`](comic_issue.hjson) **`extends`** it and supplies only
the **pages**; the cast/style/scenes propagate to every page, and `@alley` recurs across pages by
reference.

5. **`comic lint` / `comic show`** on the issue — the plan is page-aware (two pages, each with its own
   layout + panels; `@scene` refs expanded). *(No GPU.)*
6. **`issue_00.png` / `issue_01.png`** — `comic letter` writes one file per page (here over placeholders,
   weight-free). *(No GPU.)*
7. **`refs/cast_sheet.png`** — `comic cast` renders one canonical portrait per character. *(Needs a
   model.)*
8. **`issue_00.png` / `issue_01.png` (locked)** — `comic render --lock` generates each page's art and
   **face-swaps the reference onto every single-character panel**, so the same face holds across both
   pages (identity that survives beyond description-level drift). *(Needs a model + the face-swap
   weights.)*

## The idea

A comic is **structured data**, not a prompt. "Six noir panels, this woman, these lines, in this order"
is a composition — a page of framed panels, a character who must be the *same* person in panel 5 as in
panel 1, and dialogue that has to land in open space without covering a face. So the page is authored in
a small HJSON document and resolved deterministically: **layout → scene art (persona-consistent) →
composite → letter**. The weight-free half (layout + balloons + composite) runs anywhere; only the
scene art needs a GPU — and you can supply your own panel images (`comic layout/letter --panels <dir>`)
to skip the model entirely. See [`Documentation/COMIC.md`](../Documentation/COMIC.md).
