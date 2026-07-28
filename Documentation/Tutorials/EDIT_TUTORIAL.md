# Editing images — `remove` and `replace-bg`

Two one-shot commands that wrap plakat's selection + inpaint + matte stack into single verbs:
**`plakat remove`** erases an object and fills the hole; **`plakat replace-bg`** swaps the background
while keeping the subject.

---

## `plakat remove` — erase an object

Select the object, and the region is grown, feathered, and inpainted away while the rest of the image
is preserved. Four ways to select (they can be combined — the selections intersect):

```bash
# By text (OWL-ViT open-vocabulary detection) — just name the object.
plakat remove photo.png --what "the trash can" --prompt "empty pavement"

# Click the object (SAM). Normalised 0–1 unless a value exceeds 1 (then pixels).
plakat remove photo.png --point 0.42,0.71

# A bounding box (top-left, bottom-right).
plakat remove photo.png --box 0.7,0.6,1.0,1.0 --prompt "empty cobblestone street"

# A depth band (near→far), e.g. drop the foreground clutter.
plakat remove photo.png --depth-band 0.0,0.3

# Carve away over-selection with :bg points.
plakat remove photo.png --point 0.5,0.5 --point 0.6,0.4:bg
```

`--what` downloads a small detector (OWL-ViT, ~600 MB, once) and picks the highest-scoring match for
your phrase. Use a plain noun ("a red car", "the dog"); if it finds nothing, fall back to `--point`
or `--box`. The detected region is a rectangle around the object — good enough to inpaint it away.

Useful flags:

| Flag | Default | What it does |
| --- | --- | --- |
| `--prompt` | `""` | What should fill the hole. Empty = a plausible continuation; describing the surrounding scene ("cobblestone street") helps. |
| `--grow PX` | `8` | Grow the mask outward so the object's edge/shadow doesn't survive. |
| `--mask-feather PX` | `8` | Soften the inpaint↔preserve seam. |
| `--model` | `sdxl-inpaint` | Any `--mask` inpaint model (SD 1.5/SDXL inpaint, `flux-fill-dev`, `sana`, or a vanilla UNet). |

Notes:
- Fill quality depends on the inpaint model. SD inpaint is context-biased (it may hallucinate similar
  content rather than empty space) — a descriptive `--prompt` and a tight selection help most.
- A `--point` on the wrong pixel over-selects (SAM grabs the whole surface under the click). Add `:bg`
  points to carve, or use `--box`.
- Everything outside the mask is preserved (to the inpaint model's VAE round-trip floor).

---

## `plakat replace-bg` — swap the background

Mattes the subject off its background (U2Net), gets a new background, and alpha-composites the subject
over it. The **subject is preserved exactly** — it's composited from the original pixels, so there's no
generative round-trip on it.

```bash
# Generate a new background from a prompt.
plakat replace-bg portrait.png --prompt "a sunlit tropical beach, soft bokeh, professional photo"

# Or composite over a supplied background image (resized to the subject's dimensions).
plakat replace-bg product.png --bg-image studio.png
```

Useful flags:

| Flag | Default | What it does |
| --- | --- | --- |
| `--prompt` | `""` | Describes the background to generate (ignored when `--bg-image` is set). |
| `--bg-image PATH` | — | Composite over this image instead of generating one. |
| `--edge-feather PX` | `2` | Soften the subject's matte edge for a clean composite seam. |
| `--model` | `sdxl` | Model for background generation. |

Notes:
- The matte is a salient-object prediction — it works best with a clear single subject.
- Background *generation* quality tracks the model at the subject's resolution (SDXL is happiest near
  1024). For small inputs, `--bg-image` sidesteps generation entirely.

---

Both commands accept `--out DIR`, `--seed`, and `--import <album>` (land the result in a photo album),
like the other image commands.
