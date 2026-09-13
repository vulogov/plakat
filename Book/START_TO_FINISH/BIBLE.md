# NIGHT MARKET — poster bible (for the book's authors, not compiled)

This is the shared reference for the example image that *A Poster, Start to
Finish* follows. Every chapter demonstrates a plakat feature by using it on THIS
poster, so the scene, cast, and look must stay consistent across chapters. Keep
the image simple — it is a scaffold for the features, not a masterpiece to
perfect. Use only what a chapter needs.

## Logline
A promotional poster for a fictional weekend event, **NIGHT MARKET** — a
lantern-lit cobbled lane where a food vendor works a wooden cart while a single
customer leans in to order. Warm light, cool fog at the edges, the title set
across the top.

## The image, in one line
> *A lantern-lit cobbled market lane at night; a food vendor at a wooden cart
> hands a paper cup to a customer leaning across the counter; string lights
> overhead, soft fog beyond, warm amber glow.*

## The escalation (so chapters build on each other)
1. **Ch. 1–2** — the empty lane alone: cobbles, lanterns, fog. One subject, no
   people. This is the "first image" and the first prose block.
2. **Ch. 3–6** — the lane described in prose, analyzed, fixed, compiled. Still no
   people yet, or at most one implied figure.
3. **Ch. 7–8** — the single-vendor version, tuned: model, seed, sampler, budget.
4. **Ch. 9** — TWO figures: the vendor (standing at the cart) and the customer
   (leaning in). Introduced with `control-generate`.
5. **Ch. 10** — the **cart** as a connected object (`objects:`), and the
   customer–vendor **relation** (`relate: customer facing vendor`).
6. **Ch. 11** — a consistent **vendor persona** across a poster series; a
   hand-drawn **look**; a **LoRA** for the ink-poster style.
7. **Ch. 12–15** — edit, naturalize, upscale, and export the final print.

## The cast
- **The vendor** — the working figure at the cart; standing, one arm reaching to
  hand over a paper cup. In the persona chapter he becomes a reusable person so a
  *series* of market posters shows the same face.
- **The customer** — a single figure leaning across the cart counter to order;
  seen from behind or three-quarter. Never the focus; the vendor is the hero.

## The object
- **The cart** — a wooden food cart with a canopy and a small chalkboard menu; it
  is a SEPARATE object beside the vendor, not fused into him. This is the
  `objects:` example (the thing the model wants to melt into the figure).

## The look
- **Palette** — warm amber lantern light against a cool foggy blue night; high
  contrast, poster-like. In the style chapter this hardens into a two-colour
  **ink-poster** treatment for the "series" version.
- **Title** — the words NIGHT MARKET across the top. plakat is weak at rendered
  text, so the book is honest: the poster's *type* is added in a layout step
  (`compose`), not asked of the diffusion model. This is a deliberate teaching
  point in the finishing chapters.

## The recurring pain (used for the analyze / fix / smysl chapters)
The scene is easy to *over-stuff*: vendor + customer + cart + string lights +
fog + chalkboard + steam + a crowd + reflections. Left unchecked it hits the
"too complex for this model" wall after a slow render. Chapters 4–5 use
`compile --analyze` to predict that BEFORE rendering, `--analyze --fix` to trim
it, and the **smysl** corpus to record *why* each change was made — the polish
loop's memory. smysl appears ONLY in that prompt-polishing role, never as a
surface the reader authors by hand.

## The through-line
Prose is the surface the designer edits, start to finish. Everything else —
analyze, fix, compile, control-generate, naturalize — serves the prose or the
pixels. The book keeps the reader in prose and reaches up the "control ladder"
only as far as each version of the poster actually needs.
