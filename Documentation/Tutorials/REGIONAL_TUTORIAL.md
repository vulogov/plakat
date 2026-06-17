# Regional prompting (`--region`)

Put **different prompts in different parts of one image** — "snowy mountains on
the left, tropical jungle on the right" — and have them blend into a single
coherent scene, in one generation. MultiDiffusion: each region's prompt produces
its own prediction, masked to its box and blended over the base prompt (which
fills everywhere else and supplies global coherence).

## Basic

```bash
plakat generate "a vast natural landscape at golden hour, cinematic wide shot" \
  --model sd15 --size 512x512 --seed 42 \
  --region "0.0,0.0,0.5,1.0:snow-capped mountains and glaciers, alpine" \
  --region "0.5,0.0,1.0,1.0:a lush tropical rainforest with a waterfall"
```

- `--region "X0,Y0,X1,Y1:prompt"` — a box in canvas fractions `[0,1]` plus the
  prompt for it. **Repeatable.** The positional prompt is the **base**, applied
  everywhere a region doesn't cover (and for global lighting/coherence).
- Region edges are **feathered**, so neighbouring regions blend instead of seaming.

That's the committed proof (`corpus/regional.sh`): alpine left, tropical right.

## Models

Works on **SD 1.5 / SDXL** (UNet) and **SD 3.5** (MMDiT velocity blend) at native
resolution. It does `(1 + N_regions)` model passes per step, so it's heavier than
a plain generate — SD3.5 with several regions is memory-hungry. Not composed with
`--tiled`, ControlNet, or Flux (those bail with a clear message).

## In scenarios

Every supported model also takes a per-task `regions` key:

```hjson
{ model: sdxl, tasks: [
  { name: split, prompt: "a wide landscape, golden hour", regions: [
      "0,0,0.5,1:snowy alpine peaks"
      "0.5,0,1,1:a tropical jungle waterfall"
  ]}
]}
```

## Tips

- Keep the **base prompt** describing the *whole* scene (lighting, time of day,
  style) — it ties the regions together.
- Boxes can overlap; in the overlap the region prompts average.
- For a hard split with no blend, abut the boxes exactly; for a soft hand-off,
  let them overlap a little.
