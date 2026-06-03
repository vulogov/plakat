# plakat gallery

A small showcase of images generated with **plakat**. Every picture
below was produced by the `plakat` CLI on Apple Silicon (Metal),
straight from the prompt — no external editing. Generation parameters
are embedded in each PNG's `parameters` text chunk, so the images are
self-documenting; the prompts and settings are reproduced here for
convenience.

All five were rendered with **Stable Cascade** at 1024×1024 — the
three-stage pipeline that landed across the v0.37–v0.42 cycles.

<table>
  <tr>
    <td width="33%"><img src="1.png" alt="Dalí-style watercolour book illustration"></td>
    <td width="33%"><img src="2.png" alt="Old man with a pipe, vintage photo style"></td>
    <td width="33%"><img src="3.png" alt="Cottage in deep winter snow"></td>
  </tr>
  <tr>
    <td width="33%"><img src="4.png" alt="Wild steppe at orange sunrise"></td>
    <td width="33%"><img src="5.png" alt="Dalí-style watercolour with a woman in an orange coat"></td>
    <td width="33%"></td>
  </tr>
</table>

## Prompts & settings

### 1 — Watercolour book illustration

![1.png](1.png)

> an abstract book illustration in salvador dali style a ring decipering
> an apple grove, night sky, dense forest where trees having blue leaves
> rivers. In the middle of circle portrait of woman holding a cat
> watercolor book illustration style

`stable-cascade` · 1024×1024 · 20 steps · CFG 4 · seed 50 · plakat 0.42.0

### 2 — Old man with a pipe

![2.png](2.png)

> a picture of an old man with tobacco pipe and beret sitting at the
> table, old photo style

`stable-cascade` · 1024×1024 · 20 steps · CFG 4 · seed 49 · plakat 0.42.0

### 3 — Cottage in deep winter

![3.png](3.png)

> a cottage in deep winter snow, photorealistic

`stable-cascade` · img2img (strength 0.6) · 1024×1024 · 30 steps · CFG 4 · seed 42 · plakat 0.41.0

### 4 — Wild steppe at sunrise

![4.png](4.png)

> a serene slightly hilled wild steppe with small white flowers and dark
> blue-green tall grass at bright orange sunrise, greenish-hue sky
> dramatic orange-lit clouds, photorealistic

Negative: `blurry, low quality, watermark`

`stable-cascade` · 1024×1024 · 20 steps · CFG 4 · seed 43 · plakat 0.41.0

### 5 — Watercolour, woman in an orange coat

![5.png](5.png)

> an abstract illustration in salvador dali style a ring on a ring is an
> apple grove, gray-green ocean, dense magical forest where trees having
> blue leaves and flowing rivers rivers. In the middle of circle portrait
> of woman with dark hair dressed in orange coat walking towards us
> watercolor book illustration style

`stable-cascade` · 1024×1024 · 20 steps · CFG 4 · seed 51 · plakat 0.42.0

---

Want to reproduce one? Copy its prompt and settings, e.g.:

```bash
plakat generate "a cottage in deep winter snow, photorealistic" \
    --model stable-cascade --size 1024x1024 \
    --steps 30 --guidance 4.0 --seed 42
```
