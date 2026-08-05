# plakat as a library — `plakat::api`

`plakat::api` is the **supported, stable, documented** way to embed plakat in your own Rust
programs. Everything the CLI does *except* the interactive UI is available here as a small set
of ergonomic builder types — no shelling out, no reaching into internals.

> **Stability.** Only `plakat::api` carries a semver promise. The crate's other modules
> (`plakat::pipelines`, `plakat::imaging`, `plakat::scripting`, …) are `pub` so the `plakat`
> binary and tests can share them, but they are `#[doc(hidden)]` and **churn between releases** —
> do not build on them. If `plakat::api` is missing something you need, please open an issue.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts) — async, `Image`, `device()`, model names, temp files
- [Generation](#generation) — `Generate`, `Portrait`
- [Editing](#editing) — `Img2img` (+ inpaint)
- [Style & light](#style--light) — `Stylize`, `Relight`
- [People & masks](#people--masks) — `Multiperson`, `Segment`, `Transparent`
- [Image ops](#image-ops) — `Upscale`
- [Video](#video) — `Animate`
- [Worldbuilding](#worldbuilding) — `Map`
- [Book ornaments](#book-ornaments) — `BookArt`
- [Training](#training) — `StyleTrain`, `EmbeddingTrain`
- [Correctness](#correctness) — `Verify`
- [Re-exported types](#re-exported-types)

---

## Install

plakat is published on crates.io as a lib + bin crate:

```toml
[dependencies]
plakat = "2.3"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

The library never touches Python; model weights are fetched from Hugging Face on first use,
exactly like the CLI.

## Quick start

```rust
use plakat::api::Generate;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let images = Generate::new("sdxl")
        .prompt("a portrait of a red fox in a sunlit forest, detailed fur")
        .negative("blurry, watermark")
        .size(1024, 1024)
        .steps(28)
        .guidance(7.5)
        .seed(42)
        .run()
        .await?;

    images[0].save("fox.png")?;
    Ok(())
}
```

Every builder follows the same shape: **`Type::new(...)` → chain options → `.run().await`**.
Options you don't set take sensible defaults.

---

## Core concepts

### Async

Model load + inference are long-running, so every `.run()` is `async`. Drive it on a Tokio
runtime (`#[tokio::main]`, or an explicit `Runtime`). The builders are `Send`; a `.run()`
future owns its work.

### `Image`

The in-memory result type. RGB8, row-major, `width * height * 3` bytes.

| method | signature | notes |
|---|---|---|
| `width` | `fn(&self) -> u32` | |
| `height` | `fn(&self) -> u32` | |
| `pixels` | `fn(&self) -> &[u8]` | raw RGB8 |
| `save` | `fn(&self, path) -> Result<()>` | container from extension (`.png`, `.jpg`, `.webp`, …) |
| `Image::open` | `fn(path) -> Result<Image>` | load a file back into an `Image` (RGB8) |

Builders that make one image return `Image`; those that make several (`count`, frames, people)
return `Vec<Image>`.

### `device(spec) -> Result<Device>`

```rust
let dev = plakat::api::device("auto")?;   // "auto" | "metal" | "cuda" | "cpu"
```

Most callers can ignore this — every builder defaults to `"auto"` (best available backend) and
takes a `.device("cpu")`-style override.

### Model names

Anywhere a builder takes a model, pass a **plakat alias** (`sd15`, `sd21`, `sdxl`, `pixart`,
`sd35-medium`, `stable-cascade`, `flux-schnell`, …) or any **Hugging Face repo id**. The alias
selects the model family automatically.

### Temporary files

Builders that return `Image`s render into a private temp directory and read the pixels back,
cleaning up afterward — you get pixels, not files. Builders whose output is inherently a file
(alpha cut-outs, masks, composed scenes) write to a path you pass to `.run(path)`.

---

## Generation

### `Generate` — text to image

`Generate::new(model) -> … -> Result<Vec<Image>>`. Works across every family.

| method | default | meaning |
|---|---|---|
| `.prompt(s)` | `""` | positive prompt |
| `.negative(s)` | `""` | negative prompt |
| `.size(w, h)` | `512, 512` | output size (multiples of 8) |
| `.steps(n)` | `20` | denoise steps |
| `.guidance(g)` | `7.5` | CFG scale |
| `.seed(u64)` | random | fix for reproducibility |
| `.count(n)` | `1` | images to make (`seed + i` each) |
| `.clip_skip(n)` | `1` | SD1.5/2.1; `2` = A1111 community default |
| `.scheduler(SchedulerKind)` | default | sampler |
| `.device(spec)` | `"auto"` | backend |
| `.lora(source, scale)` | — | add a LoRA (repeatable) |

```rust
let imgs = Generate::new("sd15")
    .prompt("a cyberpunk city at night, neon")
    .lora("latent-consistency/lcm-lora-sdv1-5", 1.0)
    .steps(8)
    .count(4)
    .run()
    .await?;
for (i, img) in imgs.iter().enumerate() {
    img.save(format!("city-{i}.png"))?;
}
```

### `Portrait` — identity from reference photos

`Portrait::new(model) -> … -> Result<Vec<Image>>`. IP-Adapter identity: carry a face across
renders. Add one or more reference photos and pick the identity variant that matches the family.

| method | default | meaning |
|---|---|---|
| `.prompt(s)` / `.negative(s)` | `""` | prompts |
| `.photo(path, weight)` | — | reference photo (repeatable) |
| `.identity(IdentityKind)` | none | IP-Adapter variant |
| `.size(w, h)` | `512, 512` | |
| `.steps(n)` | `30` | |
| `.guidance(g)` | `7.5` | |
| `.seed(u64)` / `.count(n)` | random / `1` | |
| `.face_strength(f)` | `1.0` | FaceID identity push |
| `.scheduler(...)` / `.device(...)` / `.lora(...)` | | |

```rust
use plakat::api::{Portrait, IdentityKind};
let imgs = Portrait::new("sdxl")
    .prompt("an astronaut on the moon, cinematic")
    .photo("me.jpg", 0.9)
    .identity(IdentityKind::FaceIdSdxl)
    .run()
    .await?;
```

---

## Editing

### `Img2img` — image to image (and inpainting)

`Img2img::new(model, input) -> … -> Result<Vec<Image>>`. Add a `.mask(...)` to inpaint only the
masked region — that's the only difference between img2img and inpainting.

| method | default | meaning |
|---|---|---|
| `.prompt(s)` / `.negative(s)` | `""` | prompts |
| `.strength(f)` | `0.6` | 0 = keep input, 1 = ignore it |
| `.steps(n)` | `20` | |
| `.guidance(g)` | `7.5` | |
| `.seed(u64)` / `.count(n)` | random / `1` | |
| `.scheduler(...)` / `.device(...)` / `.lora(...)` | | |
| `.mask(path)` | none | **inpaint**: regenerate only where white |
| `.mask_feather(px)` | `0` | soften the mask edge |
| `.mask_invert(bool)` | `false` | regenerate the black region instead |

```rust
use plakat::api::Img2img;
// plain img2img
Img2img::new("sd15", "photo.png").prompt("watercolor").strength(0.55).run().await?;
// inpaint (mask an object, describe its replacement)
Img2img::new("sd15", "photo.png")
    .prompt("a vase of flowers")
    .mask("object-mask.png")
    .mask_feather(6)
    .run()
    .await?;
```

---

## Style & light

### `Stylize` — apply a reference's style

`Stylize::new(input, reference) -> … -> Result<Image>`. IP-Adapter / InstantStyle: render
`input` in the artistic style of `reference`.

| method | default | meaning |
|---|---|---|
| `.model(s)` | `"sdxl"` | base model |
| `.strength(f)` | `0.6` | restyle amount |
| `.steps(n)` | `30` | |
| `.seed(u64)` | random | |
| `.ref_blur(f)` | `0.0` | blur the reference (softens transfer) |
| `.ref_weight(f)` | `1.0` | reference conditioning weight |
| `.instantstyle(bool)` | `false` | style-only IP-Adapter targeting |
| `.style_scale(f)` | `1.0` | InstantStyle scale |
| `.device(spec)` | `"auto"` | |

```rust
let img = plakat::api::Stylize::new("photo.jpg", "van-gogh.jpg")
    .instantstyle(true)
    .run()
    .await?;
img.save("photo-vangogh.png")?;
```

### `Relight` — IC-Light re-illumination

`Relight::new(subject) -> … -> Result<Image>`. Re-light a subject (ideally an RGBA cut-out from
[`Transparent`](#people--masks)) under a described lighting condition.

| method | default | meaning |
|---|---|---|
| `.prompt(s)` | `""` | lighting/scene description |
| `.negative(s)` | `""` | |
| `.size(w, h)` | `512, 512` | |
| `.steps(n)` | `20` | |
| `.guidance(g)` | `2.0` | IC-Light works best low |
| `.seed(u64)` | `0` | |
| `.device(spec)` | `"auto"` | |

```rust
let img = plakat::api::Relight::new("subject-cutout.png")
    .prompt("warm sunset light from the left")
    .run()
    .await?;
```

---

## People & masks

### `Multiperson` — several identities in one scene

`Multiperson::new(scene) -> … -> Result<Vec<Image>>`. Add [`Person`](#person)s and describe the
scene.

| method | default | meaning |
|---|---|---|
| `.person(Person)` | — | add a person (repeatable) |
| `.model(s)` | `"sdxl"` | |
| `.identity(IdentityKind)` | `PlusFace` | |
| `.negative(s)` | `""` | |
| `.style(s)` | none | named style preset |
| `.size(w, h)` | `768, 768` | |
| `.steps(n)` | `30` | |
| `.guidance(g)` | `7.0` | |
| `.seed(u64)` / `.count(n)` | random / `1` | |
| `.composite(bool)` | `true` | per-person render → matte → place (best identity) |
| `.relight(bool)` | `false` | match figures to scene light |
| `.pose(bool)` | `false` | pose transfer |
| `.swap(bool)` | `false` | face-swap for extra fidelity |
| `.device(spec)` | `"auto"` | |

#### `Person`

`Person::new(label) -> …`. Bind an identity to a figure in the scene.

| method | meaning |
|---|---|
| `.photo(path, weight)` | reference photo (repeatable) |
| `.prompt(s)` | per-person prompt fragment |
| `.place(Position, Distance, Facing)` | where they stand / how they face |

```rust
use plakat::api::{Multiperson, Person, Position, Distance, Facing, IdentityKind};
let imgs = Multiperson::new("two friends at a sunny cafe")
    .person(Person::new("alice").photo("alice.jpg", 1.0)
        .place(Position::Left, Distance::Mid, Facing::Front))
    .person(Person::new("bob").photo("bob.jpg", 1.0)
        .place(Position::Right, Distance::Mid, Facing::Front))
    .identity(IdentityKind::PlusFace)
    .run()
    .await?;
```

### `Segment` — SAM point → mask

`Segment::new(input) -> … -> run(out_path) -> Result<()>`. Produces a binary mask PNG (255 =
selected) at the input's resolution — feed it to `Img2img::mask`.

| method | default | meaning |
|---|---|---|
| `.point(x, y, foreground)` | — | click point; `true` selects, `false` excludes (repeatable) |
| `.invert(bool)` | `false` | select everything except the subject |
| `.grow(px)` | `0` | grow the mask outward |
| `.feather(px)` | `0` | soften the edge |
| `.device(spec)` | `"auto"` | |

```rust
plakat::api::Segment::new("photo.png")
    .point(256.0, 340.0, true)
    .feather(3)
    .run("mask.png")
    .await?;
```

### `Transparent` — cut out to a transparent background

`Transparent::new(input) -> … -> run(out_path) -> Result<()>`. U2Net matting → RGBA. Because the
result carries alpha, it's written straight to `out_path` (use `.png` / `.webp`).

| method | default | meaning |
|---|---|---|
| `.crop(bool)` | `false` | crop to the subject's bounding box |
| `.device(spec)` | `"auto"` | |

```rust
plakat::api::Transparent::new("photo.jpg").crop(true).run("cutout.png").await?;
```

---

## Image ops

### `Upscale` — classical or Real-ESRGAN

`Upscale::new(input) -> … -> Result<Image>`.

| method | default | meaning |
|---|---|---|
| `.scale(f)` | `2.0` | factor (classical methods only) |
| `.method(UpscaleMethod)` | `Lanczos3` | see [`UpscaleMethod`](#upscalemethod) |
| `.device(spec)` | `"auto"` | Real-ESRGAN only |

```rust
use plakat::api::{Upscale, UpscaleMethod};
Upscale::new("small.png").method(UpscaleMethod::RealEsrganX4).run().await?.save("big.png")?;
Upscale::new("small.png").method(UpscaleMethod::Lanczos3).scale(1.5).run().await?;
```

---

## Video

### `Animate` — CLIP-lerp or AnimateDiff

`Animate::new(model, from, to) -> … -> Result<Vec<Image>>`. A 2-prompt interpolation (SD / SD3 /
Flux) or true motion via AnimateDiff. Returns frames in order; optionally also encodes a
GIF/MP4/WebM.

| method | default | meaning |
|---|---|---|
| `.negative(s)` | `""` | |
| `.frames(n)` | `16` | frame count |
| `.size(w, h)` | `512, 512` | |
| `.steps(n)` | `20` | steps per frame |
| `.guidance(g)` | `7.5` | |
| `.seed(u64)` | random | |
| `.scheduler(...)` | default | |
| `.animatediff(bool)` | `false` | use the AnimateDiff motion module |
| `.format(VideoFormat)` | `Frames` | also encode a container |
| `.out(dir)` | temp | keep frames/video here (else cleaned up after return) |
| `.device(spec)` | `"auto"` | |

```rust
use plakat::api::{Animate, VideoFormat};
let frames = Animate::new("sd15", "a calm sea", "a stormy sea")
    .frames(24)
    .out("./anim")
    .format(VideoFormat::Mp4)   // frames + anim.mp4 under ./anim
    .run()
    .await?;
```

---

## Worldbuilding

### `Map` — fantasy / world maps

Build from an explicit [`MapSpec`](#mapspec) (deterministic) or from a prose description
(LLM-parsed), then `.render()` to an image or `.render_tiles()` to a folder.

| method | default | meaning |
|---|---|---|
| `Map::from_spec(MapSpec)` | — | deterministic; no LLM |
| `Map::from_prose(text)` | — | LLM-parsed (set `.provider`) |
| `.seed(u64)` | `0` | terrain/hydrology seed |
| `.style(name)` | `"parchment"` | `parchment` / `inked` / `blueprint` |
| `.season(name)` | none | `spring` / `summer` / `autumn` / `winter` |
| `.grid(cells)` | none | overlay a coordinate grid |
| `.provider(name)` | `"none"` | LLM provider for `from_prose` |
| `.tier(u8)` | none | scale hint (0–5 geographic, 10–12 urban) |
| `.render() -> Result<Image>` | | single image |
| `.render_tiles(dir, furniture) -> Result<usize>` | | world + per-tile PNGs; returns tile count |

```rust
use plakat::api::{Map, MapSpec};
// deterministic
let img = Map::from_spec(MapSpec::minimal("Aldoria", 8, 6, 3))
    .style("inked").seed(7).render().await?;
img.save("aldoria.png")?;
// from prose (needs a configured LLM provider)
let img = Map::from_prose("a rainy northern archipelago of fjords")
    .provider("...").render().await?;
```

---

## Book ornaments

### `BookArt` — transparent B/W book ornaments

Render a reusable, print-ready, **transparent** black-and-white book ornament (headpiece / border /
vignette / …) from a [`BookArtSpec`](../Documentation/BOOKART.md) — the same render core the
`plakat bookart` CLI, the scenario `type: bookart` task, and the Bund `plakat.bookart.*` words drive.
Build from an HJSON file or an in-memory spec, then `.run()` for an in-memory result.

| method | default | meaning |
|---|---|---|
| `BookArt::load(path)` | — | load a `BookArtSpec` HJSON file |
| `BookArt::from_spec(BookArtSpec)` | — | use an in-memory spec |
| `.model(name)` | `"sd15"` | diffusion base for the diffusion/composite tiers (the origin LoRAs are sd15) |
| `.seed(u64)` | `0` | diffusion seed (also diversifies procedural variants) |
| `.steps(usize)` | `28` | diffusion steps |
| `.svg(bool)` | `false` | also produce born-vector SVG (procedural tier) |
| `.attempts(u32)` | `1` | diffusion rejection sampling — keep the first render that clears the scorecard |
| `.run() -> Result<Rendered>` | | the in-memory result |

`Rendered` carries `page: RgbaImage` (transparent, exactly page-sized), `svg: Option<String>`, the
resolved `plan`, a print/ink `scorecard`, and `pieces`.

```rust
use plakat::api::BookArt;

// a procedural border needs no weights:
let out = BookArt::load("border.hjson")?.svg(true).run().await?;
out.page.save("border.png")?;                         // transparent, exact page size @ DPI
if let Some(svg) = out.svg { std::fs::write("border.svg", svg)?; }
println!("scorecard passes: {}", out.scorecard.passes);
```

---

## Training

Both training builders write a `.safetensors` to `out` and return `Result<()>`. The full knob
set lives on the CLI; the builders expose the common ones with sensible defaults.

> The transformer families (Cascade / PixArt / SD 3.5) are memory-hungry to train (well beyond
> 24 GB). SD 1.5 / 2.1 / SDXL train comfortably.

### `StyleTrain` — style LoRA

`StyleTrain::new(model, images: Vec<PathBuf>, out) -> … -> Result<()>`. Family auto-detected
from the model alias (SD1.5/2.1/SDXL → kohya LoRA; Cascade/PixArt/SD3.5 → PEFT LoRA).

| method | default | meaning |
|---|---|---|
| `.trigger(s)` | `"style"` | trigger word |
| `.rank(n)` | `16` | LoRA rank |
| `.steps(n)` | `800` | |
| `.lr(f)` | `1e-4` | |
| `.size(px)` | `512` | |
| `.log_every(n)` | `25` | |
| `.device(spec)` | `"auto"` | |

```rust
plakat::api::StyleTrain::new("sdxl", vec!["a.png".into(), "b.png".into()], "mystyle.safetensors")
    .trigger("mystyle").steps(1000).run().await?;
```

### `EmbeddingTrain` — Textual Inversion

`EmbeddingTrain::new(model, images, token, out) -> … -> Result<()>`. Supported on SDXL and
SD 3.5.

| method | default | meaning |
|---|---|---|
| `.init_word(s)` | `"object"` | word to initialize the new token from |
| `.steps(n)` | `1000` | |
| `.lr(f)` | `5e-4` | |
| `.size(px)` | `512` | |
| `.log_every(n)` | `25` | |
| `.device(spec)` | `"auto"` | |

```rust
plakat::api::EmbeddingTrain::new("sdxl", vec!["a.png".into()], "<my-token>", "tok.safetensors")
    .init_word("cat").run().await?;
```

---

## Correctness

### `Verify` — the model-correctness harness

`Verify::new() -> … -> Result<()>`. Runs `plakat verify` programmatically; `Ok(())` iff every
check passed. Emits the report to stdout.

| method | default | meaning |
|---|---|---|
| `.tier(u8)` | all | `0` structural, `1` per-module, `2` end-to-end |
| `.model(s)` | all | restrict to one model alias |
| `.golden_dir(dir)` | HF | local golden tensors instead of fetching |
| `.json(bool)` | `false` | emit JSON |
| `.device(spec)` | `"auto"` | |

```rust
plakat::api::Verify::new().tier(1).model("sdxl").run().await?;   // Err if a check fails
```

---

## Re-exported types

`plakat::api` re-exports the value types the builders take, so you never need to import from the
internal modules.

### `SchedulerKind`
The sampler/scheduler enum (e.g. `Ddim`, and the family defaults). `SchedulerKind::default()`
lets the model pick.

### `UpscaleMethod`
`Nearest` · `Bilinear` · `Bicubic` · `Lanczos3` · `RealEsrganX2` · `RealEsrganX4` ·
`RealEsrganAnimeX4`. The `RealEsrgan*` variants are ML (fixed factor, run on the device); the
rest are classical (honor `.scale()`).

### `VideoFormat`
`Frames` · `Gif` · `Mp4` · `Webm` · `All`. Passed to `Animate::format`.

### `IdentityKind`
`PlusFace` · `PlusFaceSdxl` · `FaceId` · `FaceIdSdxl`. Pick the variant matching the model
family (SD1.5 vs SDXL).

### `Placement`, `Position`, `Distance`, `Facing`
Positioning for [`Person`](#person) via `.place(position, distance, facing)`:
- `Position`: `Left` · `CenterLeft` · `Center` · `CenterRight` · `Right`
- `Distance`: `Closer` · `Mid` · `Farther`
- `Facing`: `Front` · `Side` · `Back`

### `MapSpec`
The world description for [`Map`](#worldbuilding). Construct with
`MapSpec::minimal(name, cols, rows, tier)`, or deserialize one from HJSON/JSON. `Serialize` +
`Deserialize` + `Clone`.

---

## Scripting alternative

For batch/pipeline work without writing Rust, the same feature set is available from the **Bund**
scripting language (`plakat run script.bund`). See [`SCRIPTING.md`](SCRIPTING.md). Use the library
API when embedding plakat in a larger Rust program; use Bund for standalone render scripts.
