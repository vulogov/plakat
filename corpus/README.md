# plakat proof corpus

A reproducible, self-documenting body of images demonstrating that
plakat's feature surface actually works end to end — and the tools to
regenerate and index it.

🖼️ **[Browse the rendered gallery → `GALLERY.md`](GALLERY.md)** — every
image with its embedded recipe (AnimateDiff clips as animated GIFs).

Two kinds of files live here:

- **Driver definitions** (committed): `*.hjson` scenarios + `*.sh` /
  `*.bund` scripts that render one+ representative image per capability.
- **Output images** (`images/`, committed as the proof): what the
  drivers produce. Each PNG is self-documenting — its full recipe is
  embedded in the `parameters` chunk + a JSON sidecar.

The index ([`README` is regenerated below the line](#corpus-index)) is
built by the `plakat gallery` subcommand straight from that embedded
metadata — no hand-maintenance.

## Workflow

```bash
# 1. Render a category (downloads its model on first run)
plakat scenario corpus/cascade.hjson

# 2. (repeat for other categories you can run — see COVERAGE.md)

# 3. Rebuild the index from every rendered image
plakat gallery corpus/images --recursive --out corpus/GALLERY.md
```

Validate a scenario without generating: `plakat scenario FILE --dry-run`.

## What's here

| Driver | Capability | Model | Runs on Metal? |
|---|---|---|---|
| `cascade.hjson` | t2i variety + canny ControlNet | Stable Cascade | ✅ ungated, ~16 GB |
| `sdxl.hjson` | t2i variety + canny ControlNet | SDXL | ✅ ungated, ~7 GB |
| `pixart.hjson` | t2i variety | PixArt-Σ | ✅ ungated |
| `sd15.hjson` | t2i variety + canny ControlNet | SD 1.5 | ✅ ungated, ~4 GB |

See [`COVERAGE.md`](COVERAGE.md) for the full capability matrix and which
drivers are still to be added.

## Notes

- **Gated models** (Flux-dev, SD3) need a HuggingFace token; **Flux GGUF
  does not work on Apple Metal** (a candle kernel bug — use `--device
  cpu`, non-quantized Flux, or skip on Metal). The corpus marks these.
- Output images are committed so the proof is browsable without running
  anything; rerun the drivers to refresh them.
