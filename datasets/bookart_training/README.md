# bookart training corpus (RFC BOOKART-1 / ROADMAP G0.3)

Public-domain illustration corpus for training the three `plakat bookart`
**origin LoRAs**. Three traditions, one origin each:

| origin key | artist               | tradition | dates        | look                              |
|------------|----------------------|-----------|--------------|-----------------------------------|
| `english`  | Aubrey Beardsley     | English   | 1872–1898    | B/W pen line, high-contrast, Art Nouveau |
| `japanese` | Katsushika Hokusai   | Japanese  | 1760–1849    | B/W sumi sketch, woodblock line   |
| `russian`  | Ivan Bilibin         | Russian   | 1876–1942    | ink outline + flat colour, folk ornament |

## Public-domain rationale

All works used here are **pre-1929** and therefore public domain in the US
(and hosted as PD on Wikimedia Commons):

- **Beardsley** died 1898 — every work is PD worldwide (life + 70+ years).
- **Hokusai** died 1849 — every work is PD worldwide.
- **Bilibin** died 1942. His classic fairy-tale plates (pulled here) are the
  1899–1917 Ekspeditsiya editions, all pre-1929, PD in the US. (Life+70 also
  expired at end of 2012, so PD in most of the world too.)

The fetch script only downloads original media the Commons API reports as
`image/jpeg` / `image/png` at ≥ 400 px on both sides.

## Exact Commons categories used

The fetcher walks the MediaWiki API category members. Primary category first;
fallbacks are only consulted when the primary underperforms the per-artist
target (~40).

- **beardsley**
  - primary:  `Category:Illustrations by Aubrey Beardsley`
  - fallback: `Category:Aubrey Beardsley`
- **hokusai** — *the capitalised `Category:Hokusai Manga` is an empty container;
  the real B/W sumi plates live under the lowercase per-volume categories:*
  - primary:  `Category:Hokusai manga vol01`
  - fallbacks: `Category:Hokusai manga vol03`, `Category:Hokusai manga vol02`,
    `Category:100 Views of Mount Fuji` (Fugaku Hyakkei — also B/W line)
- **bilibin** — *`Category:Illustrations by Ivan Bilibin` does not exist, and the
  broad `Category:Ivan Bilibin` is polluted with grave photos / portraits-of /
  militia-uniform designs. Use the book-illustration subcategories instead:*
  - primary:  `Category:Book illustrations by Ivan Bilibin`
  - fallbacks: `Category:Postcards by Ivan Bilibin`,
    `Category:Magazine illustrations by Ivan Bilibin`

## Counts actually downloaded

| origin key | artist    | images |
|------------|-----------|--------|
| english    | beardsley | 40     |
| japanese   | hokusai   | 40     |
| russian    | bilibin   | 40     |
| **total**  |           | **120**|

Files are named `<artist>_NN.<ext>` (e.g. `beardsley_07.jpg`).

## Curation notes (do this by hand before training)

The fetcher is category-scoped, not content-aware. Before training, eyeball
each folder and prune / preprocess:

- **Prefer clean B/W line plates.** Beardsley and Hokusai folders are almost all
  ideal already (pen line / sumi sketch).
- **Bilibin skews colour.** Desaturate or binarise the colour plates before
  training (the LoRA learns *line/ornament*, not palette). A quick pass:
  greyscale → adaptive threshold / XDoG. Some Bilibin postcards are small
  (~400 px) and heavily bordered — crop the border.
- **Drop non-illustration files** that slip through a category:
  - photographs (of the artist, of prints in a museum case, of a grave, etc.)
  - portraits *of* the artist (e.g. a Somov portrait of Bilibin)
  - text-heavy title pages / covers / prefaces (several Hokusai `MET JIB…
    Preface/Cover` scans are mostly text — drop or crop to the drawn area)
  - the odd design/ephemera item (militia-uniform sheets, etc.)
- **Museum scans** (MET / Rijksmuseum / Yale) often include a grey mat / colour
  bar or a two-page spread — crop to the artwork.
- Hokusai `MET 2013 720 …` and `MET LC-JIB111 …` are multi-page album scans of
  the same *Manga* volume; they are on-style but somewhat redundant — thin the
  near-duplicates so no single spread dominates.

Rough post-curation expectation: ~30 clean plates per artist is plenty for an
SD1.5 style LoRA.

## Re-fetch (idempotent)

From the repo root:

```sh
python3 tools/bookart/fetch_training_corpus.py
```

Pure Python 3 stdlib (no pip installs). Already-downloaded files are skipped, so
re-running only tops up what is missing. Every request carries the required
User-Agent `plakat-bookart-corpus (vulogov@gmail.com)`; the script retries with
exponential backoff on HTTP 429 (Wikimedia rate limit).

## Next step — train the origin LoRAs (G0.4)

One LoRA per origin, triggered by an origin-specific token:

```sh
plakat style train --model sd15 datasets/bookart_training/beardsley --trigger bookart_english
plakat style train --model sd15 datasets/bookart_training/hokusai   --trigger bookart_japanese
plakat style train --model sd15 datasets/bookart_training/bilibin   --trigger bookart_russian
```

The trained `.safetensors` land in `assets/bookart/origins/{english,japanese,russian}.safetensors`
per the ROADMAP module layout.

## Provenance / licensing

Every image is public domain, sourced from Wikimedia Commons. The image bytes
are **not** committed to git (they are large and fully reproducible from this
script + the categories above — see the repo `.gitignore`). Only this README and
the fetch script are tracked.
