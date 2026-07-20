# plakat 3.13.0 — roadmap (in progress)

**Theme — visual search at scale.** 3.12 made CLIP embeddings persist + load fast. 3.13 makes the
*search itself* scale.

## Honest sizing note (read first)

After building the vector store I measured where the time actually goes for a visual search once
embeddings are cached:

- **Query embedding** (CLIP model load + embed the query) — **seconds** the first time (model load),
  then ~ms. This dominates a single search.
- **Corpus scan** (cosine over every vector) — ~**15 ms at 100k images**, ~150 ms at 1M. Not the
  bottleneck until ~1M images.
- **Vector RAM** — 100k×768 f32 = **~300 MB**; the real ceiling before the scan speed is.

So the biggest interactive win is **keeping the CLIP model resident** (fast repeat searches), and the
biggest capacity win is **int8 vectors** (4× less RAM + a faster scan). **HNSW** makes the scan
sub-linear — genuinely valuable at **≥ ~1M images** and for **incremental add**, but it's a tail
optimization below that, and it adds a dependency.

## Tracks (value order)

- [ ] **A — Resident CLIP embedder** — keep the model loaded between searches (like the ML worker), so
      the 2nd+ visual search / lookalike skips the multi-second reload. Dependency-free. *Highest
      interactive win.*
- [x] **B — int8 vector store** — CLIP embeddings are L2-normalized, so they quantize cleanly to i8 +
      a per-vector scale (`visual_search::Embedding` / `quantize` / `qdot`). Cache, the per-album
      `.plakat_clip` (magic v2), and the index `.vec` sidecar (v2) all store i8 — **4× less RAM +
      disk**, a faster int dot product, and `qdot` tracks true cosine within ~0.02 (tested). Search /
      lookalike / `:embed` all go through it.
- [ ] **C — HNSW ANN index** — a pure-Rust HNSW over the (quantized) vectors for sub-linear search +
      cheap incremental add. Persist / rebuild from the sidecar. *Adds a pure-Rust dependency; the win
      shows at very large libraries.*

## Ground rules

- Search stays correct + offline once embedded; the ANN is derived + rebuildable from the vectors.
- Verify-safe: default CLI output byte-identical; new logic lands with unit tests.
