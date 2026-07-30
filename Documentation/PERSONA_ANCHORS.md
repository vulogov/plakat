# PERSONA-1 anchor vocabulary (WFLW-98)

Every positional detail (a mark, a piercing, a piece of worn jewelry) carries an **anchor**: a named
region plus a face-normalised offset, not a raw pixel coordinate. Resolution maps the anchor through
the *realised* landmarks, so a mark stays anatomically correct when the geometry deforms or when a
different model renders the face (RFC §8.2, §10.1).

```hjson
marks: [
  {
    kind: "mole"
    anchor: {
      region: "left-nasolabial-upper"   # a named region (below)
      offset: [0.02, -0.03]             # + fractions of face width / height; +x = subject's LEFT
    }
    size: 0.03
  }
]
```

`region` and `landmark` are interchangeable keys drawing from the same vocabulary. `offset` is
optional. "Right"/"left" are the **subject's** — the subject's right eye is on the image left.

The topology is **WFLW-98** (frozen v1; the license-clean PIPNet-98 set, not the RFC's original 106-pt
InsightFace — see [`PERSONA_GATING.md`](PERSONA_GATING.md)). `plakat persona geometry <spec> --map
masks,wireframe` renders the regions so you can see where an anchor lands.

## Face anchor regions

| Group | Names |
|---|---|
| Brows | `right-brow-outer` · `right-brow-inner` · `right-brow-mid` · `left-brow-inner` · `left-brow-outer` · `left-brow-mid` · `glabella` · `forehead-centre` |
| Nose | `nose-bridge` · `nose-tip` · `septum` · `right-nostril` · `left-nostril` |
| Eyes | `right-eye-outer` · `right-eye-inner` · `left-eye-inner` · `left-eye-outer` · `right-under-eye` · `left-under-eye` |
| Cheeks / nasolabial | `right-cheek` · `left-cheek` · `right-cheekbone` · `left-cheekbone` · `right-nasolabial-upper` · `left-nasolabial-upper` |
| Mouth / philtrum | `philtrum` · `upper-lip-centre` · `lower-lip-centre` · `right-mouth-corner` · `left-mouth-corner` |
| Jaw / chin | `right-jaw-mid` · `left-jaw-mid` · `right-jaw-angle` · `left-jaw-angle` · `chin` (`chin-crease`) |
| Ears (approximate) | `right-lobe` · `right-helix` · `left-lobe` · `left-helix` |

A region is either a single landmark or the centroid of a few (a cheek is the pupil↔mouth-corner
midpoint). `lint` rejects a mark anchored to a region the topology does not define; the TUI `place`
widget converts a crosshair drop to the *nearest* region + offset so authored anchors are anatomical
from the moment they are created.

## Piercing / jewelry sites

Worn jewelry and piercings anchor at the **face** sites above (ears/nostrils/septum/brow/lip). Body
sites need the figure skeleton and are best-effort:

| Body site | Notes |
|---|---|
| `throat` · `nape` · `sternum` · `chest` · `navel` | from the pose skeleton (§10.4) |
| `right-shoulder` · `left-shoulder` · `right/left-upper-arm` · `right/left-forearm` · `right/left-thigh` | skeleton-derived |
| `right-wrist` · `left-wrist` · `right-hand` · `left-hand` | **experimental** — hand landmarks are unreliable (§8.5) |

In a face-framed render, body-sited jewelry is **culled and reported**, never mis-placed (§14.2). Put
the hand prominent in-frame and use targeted repair if a ring matters.

## Distributional details

Freckle / pockmark **fields** do not carry a point anchor — they carry a `region` and a `density` and
are realised as a seeded procedural field over the region mask (RFC §8.2):

```hjson
{ kind: "freckles", region: "right-cheek", density: 0.6 }
```
