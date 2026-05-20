# Using styles — applying an art style to your generations

This tutorial covers using plakat's **style catalog** to render
generations in named art styles: watercolor, oil painting, ukiyo-e,
and others. No prior text-to-image experience assumed.

## What you'll learn

- What a "style catalog" is and what's bundled with plakat
- How to apply a style to a generation by **name** (`--style watercolor`)
- How to apply a style by **showing plakat a reference photo**
  (`--style-ref ./inspiration.jpg`) — plakat figures out which catalog
  style matches
- How to inspect, list, and probe the catalog
- How to use styles in scenarios — both globally and per-task
- How to combine styles with portraits (style + identity together)

## Before you start

- Work through `GENERATE_TUTORIAL.md` first.
- For combining styles with portraits, also work through
  `PORTRAIT_TUTORIAL.md`.
- The bundled catalog ships with plakat. First-time use downloads
  ~2.5 GB of CLIP-H weights (shared with portrait features) and any
  LoRAs the style references (~150 MB each for the styles that have
  LoRAs).

---

## 1. What is a style catalog?

Plakat ships with a curated database of art styles. Each entry in
the catalog has:

- A **name** (id) you can pick by — e.g., `watercolor`, `ukiyo_e`.
- A few **exemplar images** that fingerprint what the style looks
  like. Plakat uses these to *detect* the style from a reference
  photo.
- Optionally, **LoRA references** to small style add-ons hosted on
  HuggingFace. Applying a LoRA actually changes how the underlying
  model paints — the image comes out *in* that style, not just with
  style-related vocabulary in the prompt.
- A **trigger phrase** — words the LoRA was trained to recognize,
  prepended to your prompt.

Run `plakat style list` to see what's bundled:

```
$ plakat style list
ID              Display name     Ex  Bases       Description
──────────────  ──────────────  ───  ──────────  ────────────────────
watercolor      Watercolor        4  sd15        Wet-on-wet pigment washes, ink lineart, visible paper texture.
photorealistic  Photorealistic    4  (none)      Photographic realism; lens characteristics; lighting physicality.
oil_painting    Oil Painting      4  sd15        Classical oil painting; visible brushwork, rich palette, canvas texture.
ukiyo_e         Ukiyo-e           4  sd15        Edo-period Japanese woodblock print; flat color, fine ink lineart, traditional subjects.
art_nouveau     Art Nouveau       4  (none)      Flowing organic lines, decorative borders, Mucha-style portraiture.

5 styles.
```

`Ex` is the exemplar count; `Bases` lists which base models have
LoRAs configured (`sd15`, `sdxl`, `flux`, or `(none)` for trigger-
only styles).

---

## 2. Applying a style by name

The simplest form. You know what you want:

```bash
plakat generate "a fox sitting in tall grass" --style watercolor
```

What happens:

```
  → style: watercolor
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors @ cd8b7d93
      → UNet: 192/192 targets merged (scale 0.80)
 INFO LoRA Arczisan/ink-watercolor/inkwatercolor.safetensors @ cd8b7d93
      → text encoder: 72/72 targets merged (scale 0.80)
→ ./out/plakat-<seed>.png
```

Plakat:

1. Looks up `watercolor` in the catalog.
2. Downloads the watercolor LoRA from HuggingFace (cached after first
   use).
3. Merges the LoRA into the base model.
4. Prepends the style's trigger phrase to your prompt.
5. Adds the style's negative-extras to push generation away from
   competing styles.
6. Generates.

The output is recognizably a watercolor — not just photo-realistic art
of a fox, but a wet-on-wet painted-art-style fox.

### When pick-by-name is the right tool

- You've already used `plakat style list` and know which style fits.
- You're scripting and want deterministic style selection.
- You're combining many shots in the same style (consistency matters).

---

## 3. Applying a style by reference photo

If you have an image you want the style of, but don't know which
catalog entry matches, let plakat detect:

```bash
plakat generate "a fox sitting in tall grass" \
    --style-ref ./inspiration/watercolor_painting.jpg
```

What happens:

```
  → style: watercolor                 # plakat detected the style
 INFO LoRA Arczisan/ink-watercolor/... merged ...
→ ./out/plakat-<seed>.png
```

Plakat:

1. Encodes the reference photo through CLIP-H.
2. Compares it against every exemplar in the catalog.
3. Picks the closest match.
4. Applies that style (same as if you'd typed `--style watercolor`).

The reference photo doesn't have to be *exactly* like the style — just
close enough. A Van Gogh painting matches `oil_painting`; a Hokusai
print matches `ukiyo_e`; an Apollo photograph matches `photorealistic`.

### Preview before committing

If you want to see what plakat would detect without actually
generating:

```bash
plakat style detect ./inspiration/watercolor_painting.jpg
```

Output:

```
Detected: watercolor (0.5037) [picked]

Top 5:
  1. watercolor           0.5037  ✓ picked
  2. oil_painting         0.3142
  3. art_nouveau          0.2898
  4. photorealistic       0.1850
  5. ukiyo_e              0.1733
```

The score is cosine similarity in CLIP-H embedding space. Values
roughly 0.3-0.5 mean confident detection. Below 0.22 (the default
threshold) plakat reports "(none above min_confidence)" and won't
auto-pick.

`[picked]` means a confident match. `[ambiguous]` would mean the top
two scores are too close to call — plakat surfaces both so you can
choose with `--style <id>`.

---

## 4. Inspecting individual styles

Want to know what's behind a style id?

```
$ plakat style show watercolor
ID:              watercolor
Display name:    Watercolor
Description:     Wet-on-wet pigment washes, ink lineart, visible paper texture.
Exemplars:      4 in catalog

Models:
  sd15:
    loras:
      - Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8 (revision: cd8b7d93...)
    trigger:   "colorful inkpainting"
    negative+: "3d render, photo, glossy, photorealistic"
```

This tells you:

- Which LoRAs will be downloaded.
- The trigger phrase that'll get prepended to your prompt.
- The negative-extras appended to your negative prompt.
- The exemplar count (more = more reliable detection).

`(revision: cd8b7d93...)` means the LoRA is pinned to a specific
HuggingFace commit — so your generations stay reproducible even if the
upstream author updates the LoRA.

---

## 5. Adjusting style strength

The default applies the LoRA at its catalog-configured scale (usually
0.8). You can multiply this:

```bash
# Subtler — half the catalog scale
plakat generate "..." --style watercolor --style-strength 0.5

# Default
plakat generate "..." --style watercolor --style-strength 1.0

# Stronger
plakat generate "..." --style watercolor --style-strength 1.5

# Too strong — at ~1.8+ most LoRAs start breaking the prompt
plakat generate "..." --style watercolor --style-strength 2.0
```

What changes:
- Lower → output looks more like base SD 1.5 with watercolor *hints*.
- Higher → output looks unmistakably watercolor at the cost of less
  faithfulness to your text prompt.

There's usually a sweet spot specific to each LoRA. Start with 1.0,
adjust.

---

## 6. Trigger-only styles vs LoRA styles

Some catalog entries have no LoRA — they're "trigger-only":

```
$ plakat style show photorealistic
ID:              photorealistic
...
Models:
  sd15:
    loras:     (none — trigger only)
    trigger:   "photograph, photorealistic, 35mm film, natural lighting"
    negative+: "painting, illustration, cartoon, 3d render, cgi, drawing"
```

These styles don't change the model. They only nudge the prompt
vocabulary toward (or away from) the style's domain. Useful when:

- The base model already does the style well natively (SD 1.5 is
  good at photographs already).
- No quality LoRA exists for the style (the case for `art_nouveau` in
  the bundled catalog).

Trigger-only styles work in exactly the same commands as LoRA styles:

```bash
plakat generate "a fox in tall grass" --style photorealistic
# → plakat prepends "photograph, photorealistic, ..." to your prompt
#   and adds "painting, illustration, ..." to the negative.
#   No LoRA downloaded.
```

---

## 7. Styles with portraits

Combining `--style-ref` (or `--style`) with `--photo` works
seamlessly:

```bash
plakat portrait "a serene expression" \
    --photo ./my_friend.jpg \
    --style watercolor
```

This produces a portrait that:
- Resembles the person in `my_friend.jpg` (identity preservation).
- Renders in watercolor style (style transfer).

The two features are independent: identity controls *who*, style
controls *how*. Both flags can be set on the same command.

Sample output:

```
  → style: watercolor
 INFO LoRA Arczisan/ink-watercolor/... merged ...
→ ./out/plakat-portrait-<seed>.png
```

The watercolor LoRA loads first, then the FaceID/Plus-Face identity
adapter applies on top.

---

## 8. Styles in scenarios — global

Add a style at the scenario level and every task inherits it:

```hjson
{
    model: sd15
    base: 768
    enhancer: deepseek

    # Apply watercolor style to every task in this scenario.
    style: watercolor
    style-strength: 0.9

    # ... scenes / weather / tasks as usual ...
}
```

Sample log:

```
  → style: watercolor
scenario  3 task(s) × 1 image(s) = 3 image(s) to generate
  model:     sd15
  loras:     1 (scale 1)
...
```

The trigger gets prepended to the scenario's `lora-header`; the
negative-extras get appended to the global negative. Every task in the
scenario carries the style.

Alternative — let plakat detect:

```hjson
{
    # ...
    style-ref: ./inspiration/turner_watercolor.jpg
    # ...
}
```

Plakat detects the style once at scenario load, then applies to every
task.

---

## 9. Styles in scenarios — per task

A task can override the global with its own style:

```hjson
{
    # ... no global style — every task picks its own ...
    enhancer: deepseek
    scene: [ ... ]
    weather: [ ... ]
    tasks:
    [
        {
            name: forest_natural
            scene: forest
            weather: dawn
            prompt: "a fox in tall grass"
            style-ref: ./inspiration/photo_inspiration.jpg
        }
        {
            name: forest_painted
            scene: forest
            weather: dawn
            prompt: "the same scene, painted"
            style-ref: ./inspiration/watercolor_inspiration.jpg
        }
    ]
}
```

Plakat detects each task's style separately at task time, sharing one
CLIP-H encoder load across tasks (efficient).

### Limitation: per-task LoRAs don't swap

If task 1 and task 2 resolve to different LoRAs, plakat warns:

```
▶ [2/2] forest_painted (scene=forest, weather=dawn)
  → style: watercolor
  ⚠ per-task style 'watercolor' wants 1 LoRA(s); scenarios share
    one pipeline so only trigger + negative apply (global LoRAs
    stay loaded)
```

Scenarios pre-load the generation pipeline once at start. Swapping
LoRAs mid-batch would require a full pipeline reload (~30 seconds
each task). So per-task style overrides apply only the **trigger**
and **negative-extras**, not the LoRA itself.

This is fine for trigger-only styles (e.g., `photorealistic`) — they
fully apply per task. For LoRA-bearing styles, you have two options:

- **Use one style globally**, accepting that all tasks share it.
- **Split into multiple scenarios**, one per LoRA set.

---

## 10. Disagreeing with detection

Sometimes detection picks a style you didn't want:

```
$ plakat style detect ./my_painting.jpg
Detected: oil_painting (0.4123) [picked]
Top 5:
  1. oil_painting    0.4123  ✓ picked
  2. watercolor      0.3987          # very close!
  ...
```

If you wanted watercolor, override with `--style`:

```bash
plakat generate "..." --style watercolor   # force the style
plakat generate "..." --style watercolor --style-ref ./my_painting.jpg
#                                          # combine: detection runs but --style wins
```

If you regularly get this kind of confusion, the catalog might lack
the exemplars to discriminate well. See `HOW_TO_CREATE_MY_OWN_STYLE.md`
for adding your own style or expanding an existing one's exemplars.

---

## 11. Combining with user-supplied LoRAs

If you pass `--lora` alongside `--style-ref`, the catalog LoRA wins
and your user LoRAs are dropped:

```bash
$ plakat generate "..." --style watercolor --lora some/repo:0.5
  → style: watercolor
  ⚠ --style-ref overrides 1 user-specified LoRA(s); using catalog LoRAs only
```

To stack a user LoRA on top of a style, use `--style <id>` (which
*doesn't* trigger the override warning) and add `--lora` — they
compose. (This is by-design behavior; the `--style-ref` photo
inference path is opinionated about being self-contained.)

---

## 12. Probing the catalog

If you've used the catalog for a while and want to confirm all the
LoRAs still resolve on HuggingFace:

```bash
plakat style probe
```

Sample output:

```
Probing 4 style(s), 3 LoRA(s) total…

  ✓ Arczisan/ink-watercolor#inkwatercolor.safetensors:0.8 (sd15 @ cd8b7d93)
  ✓ Jehugging/oilpaint_lora#S_oilpainting-07.safetensors:0.8 (sd15 @ 957cbf5d)
  ✓ py-img-gen/lora-ukiyo-e-face-blip2-captions:0.8 (sd15 @ 64553e15)

✓ all 3 LoRA(s) resolved
```

A `✗` means a LoRA repo was renamed or deleted on HuggingFace. Worth
running periodically — your generations break silently otherwise.
The command exits with status 1 on any failure, so it's CI-friendly.

---

## 13. Common issues

**`plakat style detect` returns "no style above min_confidence."**
The reference photo doesn't strongly resemble any catalog style.
Either pick by name (`--style <id>`) or build a catalog with more
appropriate exemplars (see `HOW_TO_CREATE_MY_OWN_STYLE.md`).

**LoRA download fails with HTTP 401 or 404.**
The repo was probably renamed or made private. Run `plakat style
probe` to confirm. If the catalog you're using is the bundled one,
file a plakat issue. If it's your own, update the LoRA spec to the
new repo.

**Style applies but the output doesn't look much different.**
- Bump `--style-strength` to 1.2 or 1.5.
- Check `plakat style show <id>` — if the style is `(none — trigger
  only)`, the LoRA isn't doing anything; only the trigger words push
  the prompt. Trigger-only styles are subtler.
- Try a more dramatic prompt; subtle scenes don't show style as much
  as bold compositions.

**Two styles keep getting confused (e.g., watercolor vs. oil).**
The exemplars in the catalog might not be discriminating well between
those two styles in CLIP-H space. Use pick-by-name (`--style ...`) to
force the one you want.

**The trigger phrase shows up literally in the output (e.g., text
spelling out "watercolor" appears in the image).**
Rare but happens. Drop `--style-strength` to 0.5 or add `"text,
letters, words"` to your `--negative` prompt.

**I want to use a style that's not in the bundled catalog.**
You can: see `HOW_TO_CREATE_MY_OWN_STYLE.md` for the
build-your-own-catalog workflow.

---

## Where to next

- **Build your own style catalog from a corpus of images** →
  `HOW_TO_CREATE_MY_OWN_STYLE.md`
- **Combining styles with identity preservation in portraits** →
  `PORTRAIT_TUTORIAL.md` (already covered above; the dual-pass case
  is documented in detail there)
- **The full reference manual for the style system** →
  `Documentation/STYLES.md`
- **General text-to-image + scenarios** → `GENERATE_TUTORIAL.md`
