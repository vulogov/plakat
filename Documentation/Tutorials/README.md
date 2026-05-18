# plakat tutorials

Beginner-friendly, step-by-step walkthroughs of plakat's main
features. No prior text-to-image experience assumed. Each tutorial
explains the *why* alongside the *how*.

For exhaustive flag-by-flag reference material, see the parent
[`Documentation/`](..) directory.

## Recommended reading order

If you're new to plakat, work through these in order:

1. [`GENERATE_TUTORIAL.md`](GENERATE_TUTORIAL.md) — **start here.**
   Your first generation, the flags that matter, seeds for
   reproducibility, and moving from one-off CLI commands to batch
   scenarios. The foundation everything else builds on.

2. [`PORTRAIT_TUTORIAL.md`](PORTRAIT_TUTORIAL.md) — making portraits,
   including identity preservation (rendering a specific person from
   a reference photo), putting portraits into broader scenes via
   scenarios, and multi-persona compositions.

3. [`STYLES_TUTORIAL.md`](STYLES_TUTORIAL.md) — applying art styles
   to your generations: pick by name, detect from a reference photo,
   combine styles with portraits, use styles in scenarios.

4. [`HOW_TO_CREATE_MY_OWN_STYLE.md`](HOW_TO_CREATE_MY_OWN_STYLE.md) —
   build your own style catalog from a folder of images. Covers the
   end-to-end pipeline (organize → init → build → use) and adding
   LoRAs to make detection turn into real style transfer.

5. [`ARTEFACTS_TUTORIAL.md`](ARTEFACTS_TUTORIAL.md) — composite named
   PNG cutouts (trees, sky elements, houses) into named zones of
   your generated images. Useful when you need specific objects in
   specific places, or for consistent visual elements across many
   scenes.

## Specialized portrait techniques

After the foundational portrait tutorial, these dive into specific
creative applications of plakat's weighted multi-reference portrait
feature:

- [`PORTRAIT_HOW_TO_AGE.md`](PORTRAIT_HOW_TO_AGE.md) — interpolate a
  person across ages using photos of the same person at different
  ages and weighted merging. Render plausible portraits at any
  intermediate age.

- [`PORTRAIT_CHILD_PHOTO.md`](PORTRAIT_CHILD_PHOTO.md) — blend two
  parent photos into a plausible child portrait. Combines identity-
  space merging with age-appropriate prompt cues to produce "average
  child" or "looks more like X" variants.

## What each tutorial assumes

| Tutorial | Prerequisites |
|---|---|
| `GENERATE_TUTORIAL.md` | plakat installed; can run `plakat --help`. |
| `PORTRAIT_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. |
| `STYLES_TUTORIAL.md` | Above + finished `PORTRAIT_TUTORIAL.md` (helpful but not required). |
| `HOW_TO_CREATE_MY_OWN_STYLE.md` | Above + finished `STYLES_TUTORIAL.md`. Plus a corpus of images you want to teach plakat. |
| `PORTRAIT_HOW_TO_AGE.md` | Above + finished `PORTRAIT_TUTORIAL.md`. Plus 2-4 photos of the same person at different ages. |
| `PORTRAIT_CHILD_PHOTO.md` | Above + finished `PORTRAIT_TUTORIAL.md`. Plus one head-shot per parent. |
| `ARTEFACTS_TUTORIAL.md` | Above + finished `GENERATE_TUTORIAL.md`. (No external assets required — uses the bundled placeholder set.) |

## When to use a reference manual instead

Tutorials are best when you're learning a feature or want a clear
end-to-end example. When you already know the feature and just need
to look up a specific flag or schema field, read the reference
manuals in [`Documentation/`](..):

- Looking up a specific `plakat generate` flag → [`GENERATE.md`](../GENERATE.md)
- Identity strategies and ArcFace setup → [`PERSONA.md`](../PERSONA.md)
- Style catalog JSON schema → [`STYLES.md`](../STYLES.md)
