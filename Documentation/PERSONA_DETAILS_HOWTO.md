# Authoring localized details — marks, jewelry, dentition

Small, positional, high-contrast features — moles, scars, birthmarks, freckles, piercings, jewelry,
individual teeth — defeat text conditioning (they land anywhere, drop under CFG, cost tokens, and move
between renders). PERSONA-1 gives them their own subsystem: **anchored anatomically, realised by
compositing, verified locally** (RFC §8). This is how to author them.

## Marks

A positional mark carries a `kind`, an [anchor](PERSONA_ANCHORS.md), and a `size` (fraction of face
width). Kind-specific fields tune realism — getting these right is the difference between a mark *on
skin* and a *sticker*.

```hjson
marks: [
  # a mole: a soft radial patch. `raised` drives a relief highlight/shadow aligned to the light.
  { kind: "mole", anchor: { region: "left-cheek" }, size: 0.03, raised: 0.7, color: "brown" }

  # a scar: `maturity` is the #1 realism lever — 0 = fresh pink-red + raised, 1 = pale/flat/depressed.
  { kind: "scar", anchor: { region: "right-brow-outer", offset: [0.0, -0.02] },
    form: "linear", length: 0.10, width: 0.012, orientation: 0.4, maturity: 0.7, relief: 0.5 }

  # a birthmark: `edge` (soft | defined | irregular) is what makes the boundary believable.
  { kind: "birthmark", anchor: { region: "left-jaw-mid" }, size: 0.05, edge: "irregular", intensity: 0.6 }

  # a freckle FIELD is distributional — a region + density, not a point (§8.2).
  { kind: "freckles", region: "right-cheek", density: 0.6 }
]
```

`marks: []` asserts *"this person has no marks"* (emits negatives, scored); omitting `marks`
entirely means *unknown*. The two are distinct (§6.4). A mark is **detail** class — adding, moving, or
removing one is a recomposite, never a re-cast.

## Piercings & jewelry — the identity seam (§8.5)

`piercings` are durable *sites* (holes in the body) — **surface** class, cast into the reference set.
`jewelry.items` are *worn objects* bound to a site — **presentation** class, swappable per render:

```hjson
piercings: [ { site: "left-lobe", count: 1 } ]
jewelry: {
  identity_locked: false            # true → promote the items to surface (cast them in)
  items: [
    { kind: "stud", site: "left-lobe", metal: "gold", stone: "ruby", size: 0.05 }
  ]
}
```

- Jewelry is composited from generic procedural shapes (`stud` · `hoop` · `pendant` · `bar`),
  recoloured by `metal` (gold/silver/rose-gold/steel/…) and `stone` (ruby/sapphire/emerald/…). No
  trade-dress (§10.5).
- An empty piercing (a site with no jewelry) still renders faintly as a healed hole, so a persona
  reads the same with and without earrings.
- **Glasses are the exception**: large, salient, reliably prompted — so they use the *prompt path*,
  not compositing, and are culled+reported by `composite`. Set `identity_locked: true` for glasses
  that are inseparable from how the persona reads.
- **Hand/wrist/finger jewelry is experimental** — it needs hand-landmark detection, which is
  unreliable. In a face-framed render it is culled and reported, never mis-placed.

## Dentition (§8.7)

Teeth exist on the person but appear only in open-mouth renders — a *manifesting* attribute. Author
them under `teeth`; they realise through a mouth-region prompt + inpaint when the mouth is open:

```hjson
teeth: { visibility: "visible", alignment: "even", shade: 0.25 }   # shade 0 bright → 1 yellowed
```

`teeth.alignment`/`proportion`/`size` are **structural** (they force a re-cast of any teeth-visible
reference); `shade`/`wear` are **surface**. Individual `teeth.features` (a chip, a gold crown) name a
tooth position and are **detail** class. A persona with authored teeth should cast at least one
open-mouth view (`persona cast … --expressions neutral,smile` — forthcoming) or the dentition is
unanchored; `lint` notes this.

## The compositing pass

`plakat persona composite <spec> --image <render.png>` runs the pass (§8.4): detect + align the face,
resolve each detail's anchor through the **realised** landmarks (so the mole lands below the eye that
*exists*), z-order (fields → areal → linear → piercings → worn jewelry), estimate the scene light,
alpha-composite, and union the affected regions into a mask. `--harmonise` then blends the overlays
into skin with a low-strength masked img2img. The pass runs automatically inside `cast` and `render`
(after the swap — a hard ordering constraint, since the swap would erase a pre-composited mark).

Verify what landed: `plakat persona verify <spec> --image <render.png>` — the `local_anomaly` probe
reports each mark's presence + position error; `detect` (OWL-ViT) confirms beard/glasses.
