# plakat 3.3.0 — roadmap (planning)

Opening the cycle after 3.2.0 (photos darkroom: layers, masks, adjustments, management ops —
published to crates.io). 3.2 made `plakat photos` a real editor; **3.3 rounds out the darkroom and
tends the loose ends** — more non-AI editing tools, finishing the last Phase-7 items, and clearing
the standing verification/perf debt. Candidate tracks below; not all will land — narrowed with the
owner before build.

Status: `[ ]` open · `[x]` done · `[~]` in progress · `[⏸]` blocked · `[?]` needs a decision.

## Track A — more non-AI editing (edit palette + `:` parity)

- [x] **Vignette** — radial edge darken/lighten (`EditOp::Vignette`). Palette + `:` (c687dce).
- [ ] **Dehaze** — local contrast + saturation lift in low-contrast regions.
- [x] **Curves / Levels** — black/white/gamma (`EditOp::Levels`) with an interactive live-preview
      editor (↑↓ pick handle, ←→ adjust); `:` one-shot `levels B W G` (c687dce).
- [ ] **Hue rotate / selective colour / split-tone** — colour-grading beyond warmth/tint.
- [ ] **Film grain**, **median despeckle** (a stronger denoise than the blur blend).
- [ ] **Redact GPS only** — keep the rest of the EXIF (needs an EXIF read+rewrite, unlike the
      lossless full-strip already shipped).
- [ ] **Batch convert/rename presets** — remembered format/size targets for one-key export sets.

Each stays non-destructive (an `EditOp` or a `scrub`-style file op), searchable in the palette, and
mapped into the closed, album-scoped `:` vocabulary.

## Track B — finish Phase 7 (vision + AI, the manager's own)

- [ ] **Analyze-and-generate** — turn a reference image's analysis into a generation recipe.
- [ ] **Face-scan** — detect/group faces across the library (SCRFD/ArcFace already in-tree).
- [⏸] **CLIP visual search live-verify** — blocked on the external HF cache disk; when reconnected:
      `cargo test --features photos -- --ignored clip_loads_and_embeds`. Compile+unit-verified only.

## Track C — quality / performance / debt

- [ ] **Big-image edit latency** — the tonal/spatial ops re-derive from the pristine original on every
      keypress; a bounded working-resolution preview (bake full-res only on flatten/save) for 24 MP+.
- [ ] **Distribution** — GitHub release with binary assets for 3.2.0 (crates.io done; GH deferred).
- [ ] Carry-throughs from the 2.4 performance pass and the road-to-3.0 quality/stability tracks.

## Ground rules (unchanged)

- Non-destructive by default; per-album `album.hjson` stays authoritative + additive.
- `:` planner stays inside the closed vocabulary — album-scoped, `export`/`convert` the only
  create-only outward writes, **no external read, no exec**.
- Verify-safe: default CLI image output byte-identical; new work lands with unit tests.
