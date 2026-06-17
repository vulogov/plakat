# Compose layered scenes (`plakat compose`)

Stack image layers into one scene — a background plus cut-outs placed by a
9-grid (or `x,y`), scaled, and alpha-blended. **No GPU**: it composes existing
image assets (RGBA cut-outs from `plakat transparent` / the artefact library,
or any image), so it runs anywhere in seconds.

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

| Key | Meaning |
|---|---|
| `load` | image file for the layer (resolved relative to the scene file) |
| `at` | placement — a 9-grid name (`center`, `top_left`, `bottom_right`, …) or `"x,y"` fractions in `[0,1]`. **Omit** to fill the canvas (a background). |
| `scale` | width as a fraction of the canvas width (height keeps aspect) |
| `opacity` | layer opacity in `[0,1]` (default `1`) |

`at` pins the layer's matching point to the canvas's matching point — corners sit
flush, `center` centers — so a placed layer always stays on-canvas.

## Tips

- Make cut-outs with `plakat transparent --in photo.png --out cut.png --matte`
  (content-aware) first, then `load` them as layers.
- The committed proof is `corpus/compose.sh` (valley + cottage + pine + balloon).
- `generate:` (render a layer inline) and inline `matte:` are planned — for now,
  pre-render / pre-matte those layers.
