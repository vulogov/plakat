# Compose layered scenes (`plakat compose`)

Stack image layers into one scene — a background plus cut-outs placed by a
9-grid (or `x,y`), scaled, and alpha-blended. A layer's pixels come from one of:
**`load:`** an existing image (no GPU — runs anywhere in seconds), **`matte:`** a
subject cut out of a raw photo on the fly (U2Net), or **`generate:`** a layer
rendered inline from a prompt. So a whole scene can be built with nothing on disk.

## The scene file (HJSON)

```hjson
{
  size: "1024x1024"            // canvas WxH
  out:  "scene.png"            // .png/.webp keep alpha; paths relative to this file
  layers: [                    // z-order = array order (first = bottom)
    { load: "background.png" }                               // no `at` ⇒ fills the canvas
    { load: "cottage.png", at: "bottom_center", scale: 0.34 } // placed cut-out
    { load: "balloon.png", at: "top_right", scale: 0.17, opacity: 0.9 }
  ]
}
```

Run it:

```bash
plakat compose scene.hjson
```

## Layer keys

A layer has **exactly one** source key (`load` / `matte` / `generate`), plus
placement:

| Key | Meaning |
|---|---|
| `load` | image file for the layer (resolved relative to the scene file) |
| `matte` | image to matte on the fly (U2Net) into an RGBA cut-out — drop a raw photo's subject in without pre-cutting it |
| `generate` | a text prompt rendered inline (t2i); `model` (default `sd15`), `seed`, `steps`, `gen_size` (`"WxH"`, default `512x512`) tune it |
| `at` | placement — a 9-grid name (`center`, `top_left`, `bottom_right`, …) or `"x,y"` fractions in `[0,1]`. **Omit** to fill the canvas (a background). |
| `scale` | width as a fraction of the canvas width (height keeps aspect) |
| `opacity` | layer opacity in `[0,1]` (default `1`) |

`at` pins the layer's matching point to the canvas's matching point — corners sit
flush, `center` centers — so a placed layer always stays on-canvas.

## A scene from nothing on disk

```hjson
{
  size: "1024x1024"
  out:  "beach.png"
  layers: [
    // Generate the backdrop inline (fills the canvas).
    { generate: "a serene tropical beach at sunset, golden light", model: "sd15", seed: 42 }
    // Matte a raw photo's subject and drop it on the sand.
    { matte: "astronaut.png", at: "bottom_center", scale: 0.6 }
  ]
}
```

That's the committed proof `corpus/compose_generate.sh` →
`corpus/images/compose/beach-generate-matte.png`. It's light (sd15 512² + U2Net),
so it runs even on CPU.

## Tips

- Make cut-outs ahead of time with `plakat transparent --in photo.png --out
  cut.png --matte`, or just use a `matte:` layer to do it inline.
- The committed `load`-only proof is `corpus/compose.sh` (valley + cottage + pine
  + balloon).
