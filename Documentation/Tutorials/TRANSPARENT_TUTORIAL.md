# Transparent — smart background removal (U2Net matte)

`plakat transparent` turns an image into a clean **RGBA cut-out** with the
background knocked out. It has two paths:

- **`--matte` (recommended)** — content-aware **U2Net** matting: it finds the
  salient subject and lifts it off **any** background (photo, painted, cluttered),
  no studio backdrop needed.
- **chroma-key (default)** — corner flood-fill: removes a flat, uniform
  background colour. Best only for studio shots on a clean backdrop.

```bash
# Smart matte — works on a real photo with a busy background:
plakat transparent --in apple-on-table.jpg --out apple.png --matte --crop --device cpu
```

The first `--matte` run auto-downloads the weights (`vulogov98/u2net-universal`,
Apache-2.0, ungated) into `~/.cache/plakat/`.

## When to use which

| Input | Use |
|---|---|
| A photoreal / painted subject on **any** background | **`--matte`** |
| A subject on a **flat, uniform** colour (green screen, white sweep) | default chroma-key |

`--matte` is almost always the right choice for real-world images; the chroma
path stays for the clean-studio case where it's faster.

## Flags

| Flag | Meaning |
|---|---|
| `--matte` | Use U2Net content-aware matting (vs corner chroma-key). |
| `--crop` | Tightly crop the output to the subject's bounding box. |
| `--tolerance <N>` | Chroma-key flood-fill tolerance (chroma path only; default 10). |
| `--device cpu` | Run the matte on CPU. Recommended — the U2Net pass is small and avoids the Metal single-buffer limit on large images. |

## Tips

- Pair `--matte --crop` to get a tight, transparent sticker of the subject.
- The matte is **content-aware**, not colour-aware — it won't punch a hole where
  the subject happens to match the background colour (chroma-key would).
- Cut-outs feed the **artefact library**: `transparent --matte` is how
  `corpus/artefact.sh` builds clean cutouts to composite into scenes — see
  [ARTEFACTS_TUTORIAL](ARTEFACTS_TUTORIAL.md).
