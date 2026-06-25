# Putting specific people into a scene — `multiperson`

`plakat multiperson` places **specific people** (from reference photos) into one
generated scene, each at a relative location given in words. As of **1.14.0** the
same capability is a first-class **scenario task** and a **scripting word**, so you
can batch people-in-scene compositions from automation — all three surfaces
dispatch the *same* pipeline, so a given spec renders identically everywhere.

- [What works (and the honest ceiling)](#what-works)
- [CLI](#cli)
- [Scenario task (`type: multiperson`)](#scenario)
- [Scripting word (`plakat.multiperson`)](#scripting)
- [Parity](#parity)

Build with a GPU backend for any real run (`--features metal` on Apple Silicon,
`--features cuda` on NVIDIA). SDXL + a few inpaint/swap passes fits 24 GB; keep
`--size` modest.

<a name="what-works"></a>
## What works (and the honest ceiling)

Identity strength scales with **face size**. The reliable recipe (verified):

- Use **photos** — photoreal, frontal, light background. Not paintings. Close-up
  crops are auto-padded so the face detector finds them.
- Keep the prompt **minimal** (`"two people at a cafe table…"`) and **don't
  describe each person** — the swap defines the faces. Describing "an old man with
  a beard" bleeds that look onto *every* figure.
- Keep figures **few and prominent** (closer = larger faces = stronger identity).
  Two prominent figures read as specific people; a crowd of tiny faces swaps faintly.

Two identity paths:

| mode | flag | how | when |
|------|------|-----|------|
| **face-swap** | `--swap` (+ `--pose`) | generate a coherent scene with one OpenPose skeleton pinned per person, then face-swap each figure from the photo | recommended — best identity on prominent frontal faces |
| **composite** | `--composite` | generate the background, then matte each person's actual photo in | exact identity, model-agnostic; reads as a cut-out unless `--harmonize`d |

<a name="cli"></a>
## CLI

```bash
plakat multiperson \
  "an old man on the left and a woman on the right, sitting close together at a \
   small cafe table by a window, soft daylight, watercolor, both facing the viewer" \
  --person "p1:assets/people/1.png" --at "p1:left closer front" \
  --person "p2:assets/people/2.png" --at "p2:right closer front" \
  --model sdxl --swap --pose \
  --size 1024x768 --steps 30 --guidance 7.5 --seed 42 \
  --out ./out
```

`--person LABEL:PATH` declares a person; `--at LABEL:"where"` places them
(`left|center|right` × `closer|farther` × `front|…`). Omit `--at` to auto-place.
`--scale LABEL:0.7` makes a figure shorter (child) for the `--pose` skeleton.
Useful extras: `--composite` / `--harmonize 0.3`, `--restore-faces`.

<a name="scenario"></a>
## Scenario task (`type: multiperson`)

Define each person **once** in the top-level `personas:` list, then reference them
by name from a `type: multiperson` task. One scenario can batch several
compositions (reusing loaded weights).

```hjson
{
  out: ./out/cast

  personas: [
    {
      name: oldman
      photo: assets/people/1.png
    }
    {
      name: woman
      photo: assets/people/2.png
    }
  ]

  tasks: [
    {
      name: cafe
      type: multiperson
      multiperson: {
        scene: "two people at a small cafe table by a window, soft daylight, watercolor, both facing the viewer"
        model: sdxl
        swap: true
        pose: true
        size: 1024x768
        steps: 30
        guidance: 7.5
        people: [
          {
            persona: oldman
            at: "left closer front"
          }
          {
            persona: woman
            at: "right closer front"
          }
        ]
      }
    }
  ]
}
```

Run it:

```bash
plakat scenario cast.hjson
```

Every `multiperson:` field mirrors a CLI flag and serde-defaults, so a minimal
block (just `scene` + `people`) inherits the CLI defaults (`768x768`, `plus-face`,
30 steps, guidance 7.5). `people[].persona` must name a top-level persona;
unknown names are rejected at load, before the model loads. `at`, `prompt`, and
`scale` are per-person; identity mode (`swap` / `composite`, `pose`, `harmonize`,
`restore-faces`) is per-task.

> **HJSON gotcha.** Inside the `people:` array, put **each object field on its own
> line** — the HJSON parser doesn't reliably consume inline commas
> (`{ persona: a, at: "…" }` fails; the multi-line form above works).

<a name="scripting"></a>
## Scripting word (`plakat.multiperson`)

`plakat.multiperson ( spec-path -- handle )` composes a people-in-scene image and
pushes it as an image handle (then `plakat.save` writes it). The spec is a single
self-contained file: the task fields **plus** an inline `personas` table mapping
each name to a reference photo.

`cast.json`:

```json
{
  "scene": "two people at a small cafe table, soft daylight, watercolor",
  "model": "sdxl",
  "swap": true,
  "pose": true,
  "size": "1024x768",
  "people": [
    { "persona": "oldman", "at": "left closer front" },
    { "persona": "woman",  "at": "right closer front" }
  ],
  "personas": {
    "oldman": "assets/people/1.png",
    "woman":  "assets/people/2.png"
  }
}
```

```forth
"cast.json" plakat.multiperson
"cafe.png" plakat.save
```

<a name="parity"></a>
## Parity

All three surfaces build the **same** `MultipersonRequest` and call the same
`pipelines::multiperson::run`. The scenario task and scripting word share one
builder (`multiperson::scenario_task::build_request`) whose defaults mirror the
CLI flags exactly, so an identical spec renders identically on every surface — the
same discipline the map track follows (`map/scenario_task.rs`).
