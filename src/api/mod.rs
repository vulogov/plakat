//! # plakat as a library
//!
//! A small, **stable, ergonomic** API for embedding plakat in your own Rust programs —
//! everything the CLI does (except the interactive UI), without shelling out. This is the
//! *supported* surface: it is semver-stable and documented. The crate's other modules
//! (`pipelines`, `imaging`, `scripting`, …) are implementation detail — powerful but churny,
//! and **not** covered by any stability promise. Build on `plakat::api`.
//!
//! ```no_run
//! # async fn ex() -> anyhow::Result<()> {
//! use plakat::api::Generate;
//!
//! // Text-to-image: build → run → save. Device defaults to auto (Metal/CUDA/CPU).
//! let images = Generate::new("sd15")
//!     .prompt("a portrait of a red fox in a sunlit forest, detailed fur")
//!     .negative("blurry, watermark")
//!     .size(512, 512)
//!     .steps(20)
//!     .guidance(7.5)
//!     .seed(42)
//!     .run()
//!     .await?;
//! images[0].save("fox.png")?;
//! # Ok(()) }
//! ```
//!
//! Model names are the same aliases the CLI accepts (`sd15`, `sd21`, `sdxl`, `pixart`,
//! `sd35-medium`, `stable-cascade`, `flux-schnell`, …) or any Hugging Face repo id.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result};
use candle_core::Device;

pub use crate::imaging::upscale::Method as UpscaleMethod;
pub use crate::imaging::video::Format as VideoFormat;
pub use crate::map::spec::MapSpec;
pub use crate::pipelines::ip_adapter::IdentityKind;
pub use crate::pipelines::multiperson::placement::{Distance, Facing, Position};
pub use crate::pipelines::multiperson::Placement;
pub use crate::pipelines::scheduler::SchedulerKind;

/// Resolve a device spec (`"auto"`, `"metal"`, `"cuda"`, `"cpu"`) to a [`Device`]. `"auto"`
/// picks the best available backend. Most callers can ignore this — builders default to auto.
pub fn device(spec: &str) -> Result<Device> {
    crate::device::select(spec)
}

/// A generated image held in memory as RGB8 (`width * height * 3` bytes, row-major).
#[derive(Clone)]
pub struct Image {
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

impl Image {
    /// Width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Raw RGB8 bytes, row-major (`width * height * 3`).
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    /// Write the image to `path`. The container is chosen from the extension
    /// (`.png`, `.jpg`, `.webp`, …) by the `image` crate.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let img = image::RgbImage::from_raw(self.width, self.height, self.pixels.clone())
            .ok_or_else(|| anyhow::anyhow!("image buffer size mismatch"))?;
        img.save(path).with_context(|| format!("saving image to {}", path.display()))?;
        Ok(())
    }

    /// Load an image file into an [`Image`] (RGB8). Handy for feeding results back into
    /// img2img/upscale, or for tests.
    pub fn open(path: impl AsRef<Path>) -> Result<Image> {
        let path = path.as_ref();
        let img = image::open(path)
            .with_context(|| format!("opening image {}", path.display()))?
            .to_rgb8();
        let (width, height) = img.dimensions();
        Ok(Image { pixels: img.into_raw(), width, height })
    }
}

/// A LoRA to apply during generation: a path/repo + a strength scale.
#[derive(Clone)]
struct Lora {
    source: String,
    scale: f32,
}

/// Resolve the builder's LoRA list into the internal spec type.
fn build_loras(loras: &[Lora]) -> Result<Vec<crate::pipelines::lora::LoraSpec>> {
    loras
        .iter()
        .map(|l| {
            let mut spec = crate::pipelines::lora::LoraSpec::from_str(&l.source)
                .with_context(|| format!("parsing LoRA source {:?}", l.source))?;
            spec.scale = l.scale;
            Ok(spec)
        })
        .collect()
}

/// A private temp dir for one render, unique across concurrent calls.
fn scratch_dir() -> Result<PathBuf> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plakat-api-{}-{}", std::process::id(), uniq));
    std::fs::create_dir_all(&dir).with_context(|| format!("temp dir {}", dir.display()))?;
    Ok(dir)
}

/// Text-to-image generation. Build with [`Generate::new`], chain the options you care about
/// (everything else has a sensible default), then [`run`](Generate::run).
///
/// Works across every plakat model family (SD 1.5/2.1, SDXL, PixArt-Σ, SD 3.5, Stable
/// Cascade, Flux) — the model alias selects the family automatically.
pub struct Generate {
    model: String,
    prompt: String,
    negative: String,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    seed: Option<u64>,
    count: u32,
    clip_skip: usize,
    scheduler: SchedulerKind,
    device: Option<Device>,
    loras: Vec<Lora>,
    controls: Vec<crate::pipelines::controlnet::ControlSpec>,
}

impl Generate {
    /// Start a text-to-image build for `model` (an alias like `"sdxl"` or a HF repo id).
    /// Defaults: 512×512, 20 steps, guidance 7.5, one image, auto device, default scheduler.
    pub fn new(model: impl Into<String>) -> Self {
        Generate {
            model: model.into(),
            prompt: String::new(),
            negative: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5,
            seed: None,
            count: 1,
            clip_skip: 1,
            scheduler: SchedulerKind::default(),
            device: None,
            loras: Vec::new(),
            controls: Vec::new(),
        }
    }

    /// Add a ControlNet conditioning (e.g. a pre-rendered depth map via `ControlSpec.image`). Chainable
    /// for multiple controls. SD 1.5/2.1/SDXL only (the SdCore UNet path).
    pub fn control(mut self, spec: crate::pipelines::controlnet::ControlSpec) -> Self {
        self.controls.push(spec);
        self
    }

    /// The positive prompt.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
    /// The negative prompt (what to avoid). Optional.
    pub fn negative(mut self, negative: impl Into<String>) -> Self {
        self.negative = negative.into();
        self
    }
    /// Output size in pixels (both must be divisible by 8).
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    /// Number of denoise steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Classifier-free guidance scale.
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = guidance;
        self
    }
    /// Fixed RNG seed for reproducibility. Omit for a random seed per run.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// How many images to generate (each gets `seed + i`). Default 1.
    pub fn count(mut self, count: u32) -> Self {
        self.count = count.max(1);
        self
    }
    /// CLIP-skip (SD 1.5/2.1; 1 = default, 2 = penultimate/A1111 community default).
    pub fn clip_skip(mut self, clip_skip: usize) -> Self {
        self.clip_skip = clip_skip.max(1);
        self
    }
    /// Sampler / scheduler. See [`SchedulerKind`].
    pub fn scheduler(mut self, scheduler: SchedulerKind) -> Self {
        self.scheduler = scheduler;
        self
    }
    /// Force a device (`"auto"` default). Errors are deferred to [`run`](Generate::run).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }
    /// Add a LoRA (path or repo id) at `scale`. Chainable for a stack.
    pub fn lora(mut self, source: impl Into<String>, scale: f32) -> Self {
        self.loras.push(Lora { source: source.into(), scale });
        self
    }

    /// Run generation, returning the images in memory. Async because model load + inference
    /// are long-running; drive it on a tokio runtime.
    pub async fn run(self) -> Result<Vec<Image>> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        // Render to a private temp dir, then read the PNGs back into memory. (plakat's render
        // core writes files; the library surface hides that behind an in-memory result.)
        let tmp = scratch_dir()?;
        let loras = build_loras(&self.loras)?;

        let mut req = crate::pipelines::t2i::Request::simple(
            self.prompt,
            self.model,
            self.width,
            self.height,
            self.steps,
            self.seed,
            device,
            tmp.clone(),
        );
        req.negative = self.negative;
        req.guidance = self.guidance;
        req.count = self.count;
        req.clip_skip = self.clip_skip;
        req.scheduler = self.scheduler;
        req.loras = loras;
        req.controls = self.controls;

        let gen_result = crate::pipelines::t2i::run(req).await;
        let images = collect_images(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
        gen_result?; // surface a generation error after cleanup
        images
    }
}

/// Image-to-image (and inpainting). Start with [`Img2img::new`] giving the model + an input
/// image; add a [`mask`](Img2img::mask) to inpaint only the masked region. `strength` controls
/// how far from the input the result may drift (0 = unchanged, 1 = ignore the input).
pub struct Img2img {
    model: String,
    input: PathBuf,
    prompt: String,
    negative: String,
    strength: f32,
    steps: usize,
    guidance: f64,
    seed: Option<u64>,
    count: u32,
    scheduler: SchedulerKind,
    device: Option<Device>,
    mask: Option<PathBuf>,
    mask_feather: u32,
    mask_invert: bool,
    loras: Vec<Lora>,
}

impl Img2img {
    /// Start an img2img build for `model`, transforming `input`.
    /// Defaults: strength 0.6, 20 steps, guidance 7.5, one image, auto device.
    pub fn new(model: impl Into<String>, input: impl Into<PathBuf>) -> Self {
        Img2img {
            model: model.into(),
            input: input.into(),
            prompt: String::new(),
            negative: String::new(),
            strength: 0.6,
            steps: 20,
            guidance: 7.5,
            seed: None,
            count: 1,
            scheduler: SchedulerKind::default(),
            device: None,
            mask: None,
            mask_feather: 0,
            mask_invert: false,
            loras: Vec::new(),
        }
    }

    /// The positive prompt.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
    /// The negative prompt.
    pub fn negative(mut self, negative: impl Into<String>) -> Self {
        self.negative = negative.into();
        self
    }
    /// Denoise strength in `[0, 1]` — how far the output may drift from the input.
    pub fn strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }
    /// Number of denoise steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Classifier-free guidance scale.
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = guidance;
        self
    }
    /// Fixed RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// How many images to generate. Default 1.
    pub fn count(mut self, count: u32) -> Self {
        self.count = count.max(1);
        self
    }
    /// Sampler / scheduler.
    pub fn scheduler(mut self, scheduler: SchedulerKind) -> Self {
        self.scheduler = scheduler;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }
    /// Inpaint: only regenerate where this mask is white. Turns img2img into inpainting.
    pub fn mask(mut self, mask: impl Into<PathBuf>) -> Self {
        self.mask = Some(mask.into());
        self
    }
    /// Feather (soften) the mask edge by this many pixels. Only meaningful with a mask.
    pub fn mask_feather(mut self, px: u32) -> Self {
        self.mask_feather = px;
        self
    }
    /// Invert the mask (regenerate the black region instead of the white).
    pub fn mask_invert(mut self, invert: bool) -> Self {
        self.mask_invert = invert;
        self
    }
    /// Add a LoRA (path or repo id) at `scale`.
    pub fn lora(mut self, source: impl Into<String>, scale: f32) -> Self {
        self.loras.push(Lora { source: source.into(), scale });
        self
    }

    /// Run the transform, returning the images in memory.
    pub async fn run(self) -> Result<Vec<Image>> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let tmp = scratch_dir()?;
        let loras = build_loras(&self.loras)?;
        // Resolve the working size from the input, snapped to /8 (VAE constraint). The image path
        // honours width=0 ("keep input"), but the mask + ControlNet paths need concrete dimensions —
        // passing 0 there yields a 0x0 mask ("empty mask"). Mirror the CLI's `resolve_img2img_size`.
        let (width, height) = match image::image_dimensions(&self.input) {
            Ok((iw, ih)) => (iw / 8 * 8, ih / 8 * 8),
            Err(_) => (0, 0),
        };
        let req = crate::pipelines::img2img::Request {
            prompt: self.prompt,
            negative: self.negative,
            model: self.model,
            device,
            loras,
            lora_scale: 1.0,
            input: self.input,
            mask: self.mask,
            mask_feather: self.mask_feather,
            mask_invert: self.mask_invert,
            width,
            height,
            count: self.count,
            steps: self.steps,
            guidance: self.guidance,
            scheduler: self.scheduler,
            strength: self.strength,
            seed: self.seed,
            out_dir: tmp.clone(),
            controls: Vec::new(),
        };
        let gen_result = crate::pipelines::img2img::run(req).await;
        let images = collect_images(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
        gen_result?;
        images
    }
}

/// Upscale an image — classical (Lanczos/Bicubic/…) or ML (Real-ESRGAN). Real-ESRGAN methods
/// have a fixed factor (×2 / ×4) and run on the device; classical methods honor [`scale`](Upscale::scale).
pub struct Upscale {
    input: PathBuf,
    scale: f32,
    method: UpscaleMethod,
    device: Option<Device>,
}

impl Upscale {
    /// Start an upscale build for `input`. Defaults: ×2, Lanczos3 (classical).
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Upscale { input: input.into(), scale: 2.0, method: UpscaleMethod::Lanczos3, device: None }
    }
    /// Scale factor (classical methods only; Real-ESRGAN methods have a fixed factor).
    pub fn scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }
    /// The upscaling method. See [`UpscaleMethod`].
    pub fn method(mut self, method: UpscaleMethod) -> Self {
        self.method = method;
        self
    }
    /// Force a device (Real-ESRGAN only; `"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run the upscale, returning the result in memory.
    pub async fn run(self) -> Result<Image> {
        let tmp = scratch_dir()?;
        let out = tmp.join("upscaled.png");
        let is_ml = matches!(
            self.method,
            UpscaleMethod::RealEsrganX2 | UpscaleMethod::RealEsrganX4 | UpscaleMethod::RealEsrganAnimeX4
        );
        let result = if is_ml {
            let device = match self.device {
                Some(d) => d,
                None => device("auto")?,
            };
            crate::imaging::upscale::ml_upscale(&self.input, &out, self.method, &device).await
        } else {
            crate::imaging::upscale::upscale(&self.input, &out, self.scale, self.method)
        };
        let image = result.and_then(|_| Image::open(&out));
        let _ = std::fs::remove_dir_all(&tmp);
        image
    }
}

/// IC-Light relighting — re-illuminate a cut-out subject under a described lighting condition.
/// The subject should be an RGBA cut-out (see [`Transparent`]). Returns the relit image.
pub struct Relight {
    subject: PathBuf,
    prompt: String,
    negative: String,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    seed: u64,
    device: Option<Device>,
}

impl Relight {
    /// Relight `subject` (an RGBA cut-out PNG). Defaults: 512×512, 20 steps, guidance 2.0.
    pub fn new(subject: impl Into<PathBuf>) -> Self {
        Relight {
            subject: subject.into(),
            prompt: String::new(),
            negative: String::new(),
            width: 512,
            height: 512,
            steps: 20,
            guidance: 2.0,
            seed: 0,
            device: None,
        }
    }
    /// The lighting/scene prompt (e.g. "warm sunset light from the left").
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
    /// The negative prompt.
    pub fn negative(mut self, negative: impl Into<String>) -> Self {
        self.negative = negative.into();
        self
    }
    /// Output size.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    /// Denoise steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Guidance scale (IC-Light works best low, ~2.0).
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = guidance;
        self
    }
    /// RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run relighting, returning the image in memory.
    pub async fn run(self) -> Result<Image> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let pipe = crate::pipelines::ic_light::Pipeline::load(device)
            .await
            .context("loading IC-Light pipeline")?;
        let (pixels, width, height) = pipe.relight(
            &self.subject,
            &self.prompt,
            &self.negative,
            self.width,
            self.height,
            self.steps,
            self.guidance,
            self.seed,
        )?;
        Ok(Image { pixels, width, height })
    }
}

/// Style transfer — render `input` in the artistic style of a `reference` image (IP-Adapter /
/// InstantStyle). Returns the stylized image.
pub struct Stylize {
    input: PathBuf,
    reference: PathBuf,
    model: String,
    strength: f32,
    steps: usize,
    seed: Option<u64>,
    ref_blur: f32,
    ref_weight: f32,
    instantstyle: bool,
    style_scale: f32,
    device: Option<Device>,
}

impl Stylize {
    /// Style-transfer `input` toward the look of `reference`. Defaults: sdxl, strength 0.6,
    /// 30 steps, InstantStyle off, style_scale 1.0.
    pub fn new(input: impl Into<PathBuf>, reference: impl Into<PathBuf>) -> Self {
        Stylize {
            input: input.into(),
            reference: reference.into(),
            model: "sdxl".into(),
            strength: 0.6,
            steps: 30,
            seed: None,
            ref_blur: 0.0,
            ref_weight: 1.0,
            instantstyle: false,
            style_scale: 1.0,
            device: None,
        }
    }
    /// The base model (default `sdxl`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    /// How strongly to restyle (0 = keep input, 1 = full restyle).
    pub fn strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }
    /// Denoise steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Gaussian blur applied to the reference before encoding (softens style transfer).
    pub fn ref_blur(mut self, ref_blur: f32) -> Self {
        self.ref_blur = ref_blur;
        self
    }
    /// Reference image conditioning weight.
    pub fn ref_weight(mut self, ref_weight: f32) -> Self {
        self.ref_weight = ref_weight;
        self
    }
    /// Enable InstantStyle (style-only IP-Adapter layer targeting).
    pub fn instantstyle(mut self, on: bool) -> Self {
        self.instantstyle = on;
        self
    }
    /// InstantStyle scale.
    pub fn style_scale(mut self, style_scale: f32) -> Self {
        self.style_scale = style_scale;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run style transfer, returning the image in memory.
    pub async fn run(self) -> Result<Image> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let tmp = scratch_dir()?;
        let out = tmp.join("stylized.png");
        let req = crate::pipelines::stylize::Request {
            input: self.input,
            reference: self.reference,
            out: out.clone(),
            strength: self.strength,
            model: self.model,
            steps: self.steps,
            seed: self.seed,
            ref_blur: self.ref_blur,
            ref_weight: self.ref_weight,
            instantstyle: self.instantstyle,
            style_scale: self.style_scale,
            device,
        };
        let result = crate::pipelines::stylize::run(req).await;
        let image = result.and_then(|_| Image::open(&out));
        let _ = std::fs::remove_dir_all(&tmp);
        image
    }
}

/// Cut out the salient subject to a transparent (RGBA) background via U2Net matting. Because
/// the result carries an alpha channel, it is written straight to `out_path` (a `.png`/`.webp`).
pub struct Transparent {
    input: PathBuf,
    crop: bool,
    device: Option<Device>,
}

impl Transparent {
    /// Cut out the subject of `input`. Default: no crop (keep original canvas).
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Transparent { input: input.into(), crop: false, device: None }
    }
    /// Crop the output to the subject's bounding box.
    pub fn crop(mut self, crop: bool) -> Self {
        self.crop = crop;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }
    /// Write the RGBA cut-out to `out_path` (must be `.png` or `.webp` — alpha needs it).
    pub async fn run(self, out_path: impl AsRef<Path>) -> Result<()> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        crate::pipelines::matting::cutout(&self.input, out_path.as_ref(), self.crop, &device).await
    }
}

/// A click point for [`Segment`]: normalized-or-pixel `(x, y)` and whether it marks foreground
/// (include) or background (exclude).
#[derive(Clone, Copy)]
pub struct Point {
    /// X coordinate (pixels).
    pub x: f64,
    /// Y coordinate (pixels).
    pub y: f64,
    /// `true` = foreground (select), `false` = background (exclude).
    pub foreground: bool,
}

/// Segment a subject with SAM/MobileSAM from click points, producing a binary mask PNG
/// (255 = selected) at the input's resolution — ready to feed [`Img2img::mask`].
pub struct Segment {
    input: PathBuf,
    points: Vec<Point>,
    invert: bool,
    grow: u32,
    feather: u32,
    device: Option<Device>,
}

impl Segment {
    /// Segment `input`. Add at least one [`point`](Segment::point).
    pub fn new(input: impl Into<PathBuf>) -> Self {
        Segment { input: input.into(), points: Vec::new(), invert: false, grow: 0, feather: 0, device: None }
    }
    /// Add a click point. `foreground = true` selects, `false` excludes.
    pub fn point(mut self, x: f64, y: f64, foreground: bool) -> Self {
        self.points.push(Point { x, y, foreground });
        self
    }
    /// Invert the mask (select everything except the subject).
    pub fn invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }
    /// Grow the mask outward by this many pixels.
    pub fn grow(mut self, px: u32) -> Self {
        self.grow = px;
        self
    }
    /// Feather (soften) the mask edge by this many pixels.
    pub fn feather(mut self, px: u32) -> Self {
        self.feather = px;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }
    /// Write the mask PNG to `out_path`.
    pub async fn run(self, out_path: impl AsRef<Path>) -> Result<()> {
        anyhow::ensure!(!self.points.is_empty(), "Segment needs at least one point()");
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let points: Vec<crate::pipelines::sam::PointPrompt> = self
            .points
            .iter()
            .map(|p| crate::pipelines::sam::PointPrompt { x: p.x, y: p.y, foreground: p.foreground })
            .collect();
        crate::pipelines::sam::segment(
            &self.input,
            out_path.as_ref(),
            &points,
            self.invert,
            self.grow,
            self.feather,
            &device,
        )
        .await
    }
}

/// Identity-preserving portrait generation from one or more reference photos (IP-Adapter).
/// Add `.photo(...)` + an [`identity`](Portrait::identity) kind to carry a face across renders.
pub struct Portrait {
    model: String,
    prompt: String,
    negative: String,
    photos: Vec<(PathBuf, f32)>,
    identity: Option<IdentityKind>,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    seed: Option<u64>,
    count: u32,
    face_strength: f32,
    scheduler: SchedulerKind,
    device: Option<Device>,
    loras: Vec<Lora>,
}

impl Portrait {
    /// Start a portrait build for `model`. Defaults: 512×512, 30 steps, guidance 7.5,
    /// face_strength 1.0, one image.
    pub fn new(model: impl Into<String>) -> Self {
        Portrait {
            model: model.into(),
            prompt: String::new(),
            negative: String::new(),
            photos: Vec::new(),
            identity: None,
            width: 512,
            height: 512,
            steps: 30,
            guidance: 7.5,
            seed: None,
            count: 1,
            face_strength: 1.0,
            scheduler: SchedulerKind::default(),
            device: None,
            loras: Vec::new(),
        }
    }
    /// The positive prompt.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }
    /// The negative prompt.
    pub fn negative(mut self, negative: impl Into<String>) -> Self {
        self.negative = negative.into();
        self
    }
    /// Add a reference photo at `weight` (relative influence). Chainable for several photos.
    pub fn photo(mut self, path: impl Into<PathBuf>, weight: f32) -> Self {
        self.photos.push((path.into(), weight));
        self
    }
    /// Which IP-Adapter identity variant to use (matches the model family).
    pub fn identity(mut self, kind: IdentityKind) -> Self {
        self.identity = Some(kind);
        self
    }
    /// Output size.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    /// Denoise steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Guidance scale.
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = guidance;
        self
    }
    /// RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// How many images. Default 1.
    pub fn count(mut self, count: u32) -> Self {
        self.count = count.max(1);
        self
    }
    /// FaceID identity strength (how hard to push the reference identity).
    pub fn face_strength(mut self, face_strength: f32) -> Self {
        self.face_strength = face_strength;
        self
    }
    /// Sampler / scheduler.
    pub fn scheduler(mut self, scheduler: SchedulerKind) -> Self {
        self.scheduler = scheduler;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }
    /// Add a LoRA (path or repo id) at `scale`.
    pub fn lora(mut self, source: impl Into<String>, scale: f32) -> Self {
        self.loras.push(Lora { source: source.into(), scale });
        self
    }

    /// Run generation, returning the images in memory.
    pub async fn run(self) -> Result<Vec<Image>> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let tmp = scratch_dir()?;
        let loras = build_loras(&self.loras)?;
        let photos = self
            .photos
            .iter()
            .map(|(p, w)| crate::pipelines::ip_adapter::WeightedPhoto { path: p.clone(), weight: Some(*w) })
            .collect();
        let req = crate::pipelines::portrait::Request {
            prompt: self.prompt,
            negative: self.negative,
            photos,
            model: self.model,
            width: self.width,
            height: self.height,
            count: self.count,
            steps: self.steps,
            guidance: self.guidance,
            seed: self.seed,
            out_dir: tmp.clone(),
            device,
            loras,
            lora_scale: 1.0,
            scheduler: self.scheduler,
            refine: None,
            refine_strength: 0.3,
            face_strength: self.face_strength,
            face_bbox: None,
            face_landmarks: None,
            identity: self.identity,
            shared_clip_h: None,
            controls: Vec::new(),
        };
        let gen_result = crate::pipelines::portrait::run(req).await;
        let images = collect_images(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
        gen_result?;
        images
    }
}

/// One person in a [`Multiperson`] scene: a label, reference photo(s), and optional per-person
/// prompt + placement.
pub struct Person {
    label: String,
    photos: Vec<(PathBuf, f32)>,
    prompt: Option<String>,
    placement: Option<Placement>,
}

impl Person {
    /// A person identified by `label` (used to bind the identity to a figure in the scene).
    pub fn new(label: impl Into<String>) -> Self {
        Person { label: label.into(), photos: Vec::new(), prompt: None, placement: None }
    }
    /// Add a reference photo at `weight`.
    pub fn photo(mut self, path: impl Into<PathBuf>, weight: f32) -> Self {
        self.photos.push((path.into(), weight));
        self
    }
    /// A per-person prompt fragment.
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }
    /// Where this person stands / how they face the camera.
    pub fn place(mut self, position: Position, distance: Distance, facing: Facing) -> Self {
        self.placement = Some(Placement { position, distance, facing });
        self
    }
}

/// Place two or more identities into one coherent scene. Add [`Person`]s, describe the scene,
/// and run. Returns the composed image(s).
pub struct Multiperson {
    scene: String,
    people: Vec<Person>,
    model: String,
    identity: IdentityKind,
    negative: String,
    style: Option<String>,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    seed: Option<u64>,
    count: u32,
    composite: bool,
    relight: bool,
    pose: bool,
    swap: bool,
    device: Option<Device>,
}

impl Multiperson {
    /// Start a multi-person scene described by `scene`. Defaults: sdxl, PlusFace identity,
    /// 768×768, 30 steps, guidance 7.0, composite mode on.
    pub fn new(scene: impl Into<String>) -> Self {
        Multiperson {
            scene: scene.into(),
            people: Vec::new(),
            model: "sdxl".into(),
            identity: IdentityKind::PlusFace,
            negative: String::new(),
            style: None,
            width: 768,
            height: 768,
            steps: 30,
            guidance: 7.0,
            seed: None,
            count: 1,
            composite: true,
            relight: false,
            pose: false,
            swap: false,
            device: None,
        }
    }
    /// Add a [`Person`] to the scene.
    pub fn person(mut self, person: Person) -> Self {
        self.people.push(person);
        self
    }
    /// The base model.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }
    /// IP-Adapter identity variant.
    pub fn identity(mut self, identity: IdentityKind) -> Self {
        self.identity = identity;
        self
    }
    /// Negative prompt.
    pub fn negative(mut self, negative: impl Into<String>) -> Self {
        self.negative = negative.into();
        self
    }
    /// A named style preset to apply.
    pub fn style(mut self, style: impl Into<String>) -> Self {
        self.style = Some(style.into());
        self
    }
    /// Output size.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    /// Denoise steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Guidance scale.
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = guidance;
        self
    }
    /// RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// How many images. Default 1.
    pub fn count(mut self, count: u32) -> Self {
        self.count = count.max(1);
        self
    }
    /// Composite mode (per-person render → matte → place; recommended for strong identity).
    pub fn composite(mut self, on: bool) -> Self {
        self.composite = on;
        self
    }
    /// Relight the composited figures to match the scene lighting.
    pub fn relight(mut self, on: bool) -> Self {
        self.relight = on;
        self
    }
    /// Apply pose transfer.
    pub fn pose(mut self, on: bool) -> Self {
        self.pose = on;
        self
    }
    /// Apply face-swap for extra identity fidelity.
    pub fn swap(mut self, on: bool) -> Self {
        self.swap = on;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run the scene, returning the composed image(s) in memory.
    pub async fn run(self) -> Result<Vec<Image>> {
        anyhow::ensure!(!self.people.is_empty(), "Multiperson needs at least one person()");
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let tmp = scratch_dir()?;
        let people = self
            .people
            .into_iter()
            .map(|p| crate::pipelines::multiperson::Person {
                label: p.label,
                photos: p
                    .photos
                    .iter()
                    .map(|(path, w)| crate::pipelines::ip_adapter::WeightedPhoto {
                        path: path.clone(),
                        weight: Some(*w),
                    })
                    .collect(),
                placement: p.placement,
                bbox: None,
                prompt: p.prompt,
                face_strength: None,
                face_bbox: None,
                face_landmarks: None,
                scale: None,
            })
            .collect();
        let req = crate::pipelines::multiperson::MultipersonRequest {
            scene: self.scene,
            people,
            model: self.model,
            identity: self.identity,
            style: self.style,
            negative: self.negative,
            layout_provider: "none".into(),
            enhancer: None,
            width: self.width,
            height: self.height,
            steps: self.steps,
            guidance: self.guidance,
            seed: self.seed,
            count: self.count,
            out_dir: tmp.clone(),
            scheduler: SchedulerKind::default(),
            device,
            dry_run: false,
            composite: self.composite,
            relight: self.relight,
            harmonize: None,
            pose: self.pose,
            swap: self.swap,
            restore_faces: false,
            refine_faces: false,
            refine_face_strength: 0.85,
            refine_denoise: 0.3,
        };
        let gen_result = crate::pipelines::multiperson::run(req).await;
        let images = collect_images(&tmp);
        let _ = std::fs::remove_dir_all(&tmp);
        gen_result?;
        images
    }
}

/// Render a fantasy/world map. Build from a [`MapSpec`] (deterministic) or from a prose
/// description (LLM-parsed), pick a `style`, and [`render`](Map::render) to an image.
pub struct Map {
    source: MapSource,
    seed: u64,
    style: String,
    season: Option<String>,
    grid: Option<u32>,
    provider: String,
    tier: Option<u8>,
}

enum MapSource {
    Spec(Box<MapSpec>),
    Prose(String),
}

impl Map {
    /// Build from an explicit [`MapSpec`] (no LLM). Default style `parchment`, seed 0.
    pub fn from_spec(spec: MapSpec) -> Self {
        Map {
            source: MapSource::Spec(Box::new(spec)),
            seed: 0,
            style: "parchment".into(),
            season: None,
            grid: None,
            provider: "none".into(),
            tier: None,
        }
    }
    /// Build from a prose world description; the spec is parsed by an LLM `provider`
    /// (set via [`provider`](Map::provider)). Default style `parchment`.
    pub fn from_prose(description: impl Into<String>) -> Self {
        Map {
            source: MapSource::Prose(description.into()),
            seed: 0,
            style: "parchment".into(),
            season: None,
            grid: None,
            provider: "none".into(),
            tier: None,
        }
    }
    /// RNG seed (drives terrain/hydrology generation).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Render style: `parchment`, `inked`, or `blueprint`.
    pub fn style(mut self, style: impl Into<String>) -> Self {
        self.style = style.into();
        self
    }
    /// Season tint: `spring`/`summer`/`autumn`/`winter`.
    pub fn season(mut self, season: impl Into<String>) -> Self {
        self.season = Some(season.into());
        self
    }
    /// Overlay a coordinate grid every N cells.
    pub fn grid(mut self, cells: u32) -> Self {
        self.grid = Some(cells);
        self
    }
    /// LLM provider for `from_prose` (e.g. a configured enhancer provider).
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }
    /// Scale tier hint for prose parsing (0–5 geographic, 10–12 urban).
    pub fn tier(mut self, tier: u8) -> Self {
        self.tier = Some(tier);
        self
    }

    async fn resolve_spec(&self) -> Result<MapSpec> {
        match &self.source {
            MapSource::Spec(s) => Ok((**s).clone()),
            MapSource::Prose(text) => crate::map::parser::parse(
                text,
                &crate::map::parser::ParseOpts {
                    provider: self.provider.clone(),
                    system_override: None,
                    tile_grid: None,
                    scale_tier: self.tier,
                    cache: true,
                },
            )
            .await
            .context("parsing map description"),
        }
    }

    fn build_style(&self) -> Result<crate::map::render::Style> {
        let mut style = crate::map::render::Style::named(&self.style)?;
        if let Some(season) = &self.season {
            style = style.with_season(crate::map::render::Season::parse(season)?);
        }
        if let Some(grid) = self.grid {
            style = style.with_grid(grid);
        }
        Ok(style)
    }

    /// Render the map to a single image in memory.
    pub async fn render(self) -> Result<Image> {
        let spec = self.resolve_spec().await?;
        let style = self.build_style()?;
        let rgb = crate::map::render_map_image(&spec, self.seed, style)?;
        let (width, height) = rgb.dimensions();
        Ok(Image { pixels: rgb.into_raw(), width, height })
    }

    /// Render the world plus per-tile images into `dir`; returns the tile count.
    pub async fn render_tiles(self, dir: impl AsRef<Path>, furniture: bool) -> Result<usize> {
        let spec = self.resolve_spec().await?;
        let style = self.build_style()?;
        crate::map::save_world_tiles(&spec, self.seed, style, dir.as_ref(), furniture)
    }
}

/// Train a **style LoRA** from a handful of images. Works across families (SD 1.5/2.1/SDXL →
/// kohya LoRA; Stable Cascade / PixArt / SD 3.5 → PEFT LoRA). Writes a `.safetensors` to `out`.
///
/// Note: the transformer families (cascade/pixart/sd3) are memory-hungry — training them needs
/// substantially more than 24 GB. SD 1.5/2.1/SDXL train comfortably.
pub struct StyleTrain {
    model: String,
    images: Vec<PathBuf>,
    trigger: String,
    rank: usize,
    steps: usize,
    lr: f64,
    size: u32,
    out: PathBuf,
    log_every: usize,
    device: Option<Device>,
}

impl StyleTrain {
    /// Train a style LoRA for `model` from `images`, writing to `out`.
    /// Defaults: trigger "style", rank 16, 800 steps, lr 1e-4, 512px, log every 25.
    pub fn new(model: impl Into<String>, images: Vec<PathBuf>, out: impl Into<PathBuf>) -> Self {
        StyleTrain {
            model: model.into(),
            images,
            trigger: "style".into(),
            rank: 16,
            steps: 800,
            lr: 1e-4,
            size: 512,
            out: out.into(),
            log_every: 25,
            device: None,
        }
    }
    /// The trigger word that will invoke the style.
    pub fn trigger(mut self, trigger: impl Into<String>) -> Self {
        self.trigger = trigger.into();
        self
    }
    /// LoRA rank.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }
    /// Training steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Learning rate.
    pub fn lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
    /// Training resolution.
    pub fn size(mut self, size: u32) -> Self {
        self.size = size;
        self
    }
    /// Log every N steps.
    pub fn log_every(mut self, n: usize) -> Self {
        self.log_every = n.max(1);
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run training; writes the LoRA to `out`.
    pub async fn run(self) -> Result<()> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        style_train(
            self.model,
            self.images,
            self.trigger,
            self.rank,
            self.steps,
            self.lr,
            self.size,
            self.out,
            self.log_every,
            device,
        )
        .await
    }
}

/// Dispatch a style-LoRA training run to the right family trainer. Shared by
/// [`StyleTrain`] and the `plakat.style.train` Bund word so the family routing lives once.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn style_train(
    model: String,
    images: Vec<PathBuf>,
    trigger: String,
    rank: usize,
    steps: usize,
    lr: f64,
    size: u32,
    out: PathBuf,
    log_every: usize,
    device: Device,
) -> Result<()> {
    let m = model.to_lowercase();
    if m.contains("cascade") || m.contains("stage-c") {
        crate::pipelines::cascade::train_style_lora(crate::pipelines::cascade::StyleTrainRequest {
            repo: model,
            device,
            images,
            trigger,
            rank,
            steps,
            lr,
            size,
            out,
            checkpoint_every: None,
            log_every,
            resume_from: None,
        })
        .await
    } else if m.contains("pixart") {
        crate::pipelines::pixart::train_style_lora(crate::pipelines::pixart::StyleTrainRequest {
            repo: model,
            device,
            images,
            trigger,
            rank,
            steps,
            lr,
            size,
            out,
            checkpoint_every: None,
            log_every,
            resume_from: None,
            class_images: Vec::new(),
            class_prompt: None,
            prior_weight: 1.0,
        })
        .await
    } else if m.contains("sd3") || m.contains("sd35") {
        crate::pipelines::sd3::train_style_lora(crate::pipelines::sd3::StyleTrainRequest {
            variant: crate::pipelines::sd3::Variant::Sd35Medium,
            repo: model,
            device,
            images,
            trigger,
            rank,
            steps,
            lr,
            size,
            out,
            checkpoint_every: None,
            log_every,
            resume_from: None,
            class_images: Vec::new(),
            class_prompt: None,
            prior_weight: 1.0,
        })
        .await
    } else {
        // SD 1.5 / 2.1 / SDXL — the kohya LoRA trainer.
        crate::pipelines::sd_train::trainer::train_style_lora_sd(
            crate::pipelines::sd_train::trainer::SdStyleTrainRequest {
                model,
                device,
                images,
                trigger,
                rank,
                steps,
                lr,
                size,
                out,
                checkpoint_every: None,
                log_every,
                resume_from: None,
                class_images: Vec::new(),
                class_prompt: None,
                prior_weight: 1.0,
            },
        )
        .await
    }
}

/// Train a **Textual Inversion** embedding (a new token) from a few images. Writes a
/// `.safetensors` embedding to `out`. Supported on SDXL and SD 3.5.
pub struct EmbeddingTrain {
    model: String,
    images: Vec<PathBuf>,
    token: String,
    init_word: String,
    steps: usize,
    lr: f64,
    size: u32,
    out: PathBuf,
    log_every: usize,
    device: Option<Device>,
}

impl EmbeddingTrain {
    /// Train a TI embedding for `token` from `images`, writing to `out`.
    /// Defaults: init word "object", 1000 steps, lr 5e-4, 512px, log every 25.
    pub fn new(
        model: impl Into<String>,
        images: Vec<PathBuf>,
        token: impl Into<String>,
        out: impl Into<PathBuf>,
    ) -> Self {
        EmbeddingTrain {
            model: model.into(),
            images,
            token: token.into(),
            init_word: "object".into(),
            steps: 1000,
            lr: 5e-4,
            size: 512,
            out: out.into(),
            log_every: 25,
            device: None,
        }
    }
    /// The word to initialize the new token's embedding from.
    pub fn init_word(mut self, init_word: impl Into<String>) -> Self {
        self.init_word = init_word.into();
        self
    }
    /// Training steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Learning rate.
    pub fn lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }
    /// Training resolution.
    pub fn size(mut self, size: u32) -> Self {
        self.size = size;
        self
    }
    /// Log every N steps.
    pub fn log_every(mut self, n: usize) -> Self {
        self.log_every = n.max(1);
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run training; writes the embedding to `out`.
    pub async fn run(self) -> Result<()> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        crate::pipelines::ti_train::train_textual_inversion(crate::pipelines::ti_train::TiTrainRequest {
            model: self.model,
            device,
            images: self.images,
            token: self.token,
            init_word: self.init_word,
            steps: self.steps,
            lr: self.lr,
            size: self.size,
            out: self.out,
            log_every: self.log_every,
        })
        .await
    }
}

/// Run the model-correctness harness (`plakat verify`) programmatically. Returns `Ok(())` if
/// every check passed, `Err` otherwise. Emits the report to stdout.
pub struct Verify {
    tier: Option<u8>,
    model: Option<String>,
    golden_dir: Option<PathBuf>,
    json: bool,
    device: Option<Device>,
}

impl Default for Verify {
    fn default() -> Self {
        Verify::new()
    }
}

impl Verify {
    /// Verify everything (all tiers, all models) by default.
    pub fn new() -> Self {
        Verify { tier: None, model: None, golden_dir: None, json: false, device: None }
    }
    /// Restrict to a single tier (0 structural, 1 per-module, 2 end-to-end).
    pub fn tier(mut self, tier: u8) -> Self {
        self.tier = Some(tier);
        self
    }
    /// Restrict to a single model alias.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
    /// Use golden tensors from a local directory instead of fetching from Hugging Face.
    pub fn golden_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.golden_dir = Some(dir.into());
        self
    }
    /// Emit the report as JSON.
    pub fn json(mut self, json: bool) -> Self {
        self.json = json;
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run verification. `Ok(())` iff every check passed.
    pub async fn run(self) -> Result<()> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        crate::verify::run(&crate::verify::VerifyConfig {
            tier: self.tier,
            model: self.model,
            golden_dir: self.golden_dir,
            device,
            json: self.json,
        })
        .await
    }
}

/// Generate an animation — a 2-prompt CLIP-lerp (SD/SD3/Flux) or true motion via AnimateDiff.
/// Returns the frames in order; optionally also encodes a GIF/MP4/WebM into `out`.
pub struct Animate {
    model: String,
    from: String,
    to: String,
    negative: String,
    frames: u32,
    width: u32,
    height: u32,
    steps: usize,
    guidance: f64,
    seed: Option<u64>,
    scheduler: SchedulerKind,
    animatediff: bool,
    format: VideoFormat,
    out: Option<PathBuf>,
    device: Option<Device>,
}

impl Animate {
    /// Animate `model` from prompt `from` to prompt `to`. Defaults: 16 frames, 512×512,
    /// 20 steps, guidance 7.5, frames-only output (no video encode).
    pub fn new(model: impl Into<String>, from: impl Into<String>, to: impl Into<String>) -> Self {
        Animate {
            model: model.into(),
            from: from.into(),
            to: to.into(),
            negative: String::new(),
            frames: 16,
            width: 512,
            height: 512,
            steps: 20,
            guidance: 7.5,
            seed: None,
            scheduler: SchedulerKind::default(),
            animatediff: false,
            format: VideoFormat::Frames,
            out: None,
            device: None,
        }
    }
    /// Negative prompt.
    pub fn negative(mut self, negative: impl Into<String>) -> Self {
        self.negative = negative.into();
        self
    }
    /// Number of frames.
    pub fn frames(mut self, frames: u32) -> Self {
        self.frames = frames.max(1);
        self
    }
    /// Frame size.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
    /// Denoise steps per frame.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }
    /// Guidance scale.
    pub fn guidance(mut self, guidance: f64) -> Self {
        self.guidance = guidance;
        self
    }
    /// RNG seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    /// Sampler / scheduler.
    pub fn scheduler(mut self, scheduler: SchedulerKind) -> Self {
        self.scheduler = scheduler;
        self
    }
    /// Use AnimateDiff (true motion module) instead of the 2-prompt CLIP-lerp.
    pub fn animatediff(mut self, on: bool) -> Self {
        self.animatediff = on;
        self
    }
    /// Also encode the frames to this container. See [`VideoFormat`].
    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }
    /// Where to write frames (and any encoded video). Default: a temp dir cleaned up after
    /// the frames are returned. Set this to keep the output.
    pub fn out(mut self, dir: impl Into<PathBuf>) -> Self {
        self.out = Some(dir.into());
        self
    }
    /// Force a device (`"auto"` default).
    pub fn device(mut self, spec: &str) -> Self {
        self.device = device(spec).ok();
        self
    }

    /// Run the animation, returning the frames in order.
    pub async fn run(self) -> Result<Vec<Image>> {
        let device = match self.device {
            Some(d) => d,
            None => device("auto")?,
        };
        let keep = self.out.is_some();
        let out = match self.out {
            Some(d) => {
                std::fs::create_dir_all(&d).with_context(|| format!("out dir {}", d.display()))?;
                d
            }
            None => scratch_dir()?,
        };
        let args = crate::cli::animate::AnimateArgs {
            from: self.from,
            to: self.to,
            frames: self.frames,
            seed: self.seed,
            model: self.model,
            size: format!("{}x{}", self.width, self.height),
            steps: self.steps,
            guidance: self.guidance,
            negative: self.negative,
            scheduler: self.scheduler,
            out: out.clone(),
            gif: matches!(self.format, VideoFormat::Gif),
            gif_delay_ms: 80,
            animatediff: self.animatediff,
            motion_loras: Vec::new(),
            motion_lora_scale: 1.0,
            format: self.format,
            no_metadata: true,
            resume: false,
            control: None,
            control_image: None,
            control_from: None,
            control_strength: 1.0,
            control_specs: Vec::new(),
            lcm: false,
            window_size: 16,
            window_overlap: 4,
            free_noise: false,
        };
        let run_result = crate::cli::animate::run(args, device).await;
        // Frames are written as frame-NNNN.png (sorted name == frame order).
        let frames = collect_images(&out);
        if !keep {
            let _ = std::fs::remove_dir_all(&out);
        }
        run_result?;
        frames
    }
}

/// Read every PNG in `dir` (sorted) into [`Image`]s.
fn collect_images(dir: &Path) -> Result<Vec<Image>> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("reading output dir {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
        .collect();
    paths.sort();
    anyhow::ensure!(!paths.is_empty(), "generation produced no images");
    paths.iter().map(Image::open).collect()
}

/// Compose a transparent, print-ready B/W **book ornament** from a `BookArtSpec` (RFC BOOKART-1).
///
/// Returns the finished ornament in memory — the transparent, page-sized `RgbaImage`, an optional
/// born-vector SVG, the resolved plan, and the print/ink scorecard — via the shared render core, the
/// same one `plakat bookart` drives.
///
/// ```no_run
/// # async fn ex() -> anyhow::Result<()> {
/// use plakat::api::BookArt;
///
/// let out = BookArt::load("border.hjson")?.svg(true).run().await?;
/// out.page.save("border.png")?;                       // transparent, exact page size
/// if let Some(svg) = out.svg { std::fs::write("border.svg", svg)?; }
/// println!("scorecard passes: {}", out.scorecard.passes);
/// # Ok(()) }
/// ```
pub struct BookArt {
    spec: crate::bookart::BookArtSpec,
    opts: crate::bookart::RenderOpts,
}

impl BookArt {
    /// Load a spec from an HJSON file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self { spec: crate::bookart::BookArtSpec::load(path.as_ref())?, opts: Default::default() })
    }

    /// Use an in-memory spec.
    pub fn from_spec(spec: crate::bookart::BookArtSpec) -> Self {
        Self { spec, opts: Default::default() }
    }

    /// Base model for the diffusion/composite tiers (default `sd15` — the origin LoRAs are sd15).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.opts.model = model.into();
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.opts.seed = seed;
        self
    }

    pub fn steps(mut self, steps: usize) -> Self {
        self.opts.steps = steps;
        self
    }

    /// Also produce born-vector SVG (procedural tier).
    pub fn svg(mut self, svg: bool) -> Self {
        self.opts.svg = svg;
        self
    }

    /// Diffusion-tier rejection sampling: try up to N seeds, keep the first that clears the scorecard.
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.opts.attempts = attempts;
        self
    }

    /// Render, returning the in-memory [`Rendered`](crate::bookart::Rendered) result.
    pub async fn run(self) -> Result<crate::bookart::Rendered> {
        crate::bookart::render::render_spec(&self.spec, &self.opts).await
    }
}

/// Seamless PBR material synthesis (RFC TEXTURE-1). Turn a prompt or a photo into a tileable material
/// set (albedo/normal/roughness/metallic/height/AO), written to a directory. Mirrors [`BookArt`].
///
/// ```no_run
/// # async fn f() -> anyhow::Result<()> {
/// use plakat::api::Texture;
/// let scorecard = Texture::from_prompt("mossy cobblestone").upscale("2k").run("mat/").await?;
/// println!("tileable: {}", scorecard.passes);
/// # Ok(()) }
/// ```
pub struct Texture {
    spec: crate::texture::TextureSpec,
    opts: crate::texture::render::RenderOpts,
}

impl Texture {
    /// Text-to-material.
    pub fn from_prompt(material: impl Into<String>) -> Self {
        Self { spec: crate::texture::TextureSpec { material: Some(material.into()), ..Default::default() }, opts: Default::default() }
    }

    /// Image-to-material — a photo → a tileable PBR set.
    pub fn from_image(path: impl Into<String>) -> Self {
        Self { spec: crate::texture::TextureSpec { from_image: Some(path.into()), ..Default::default() }, opts: Default::default() }
    }

    /// Use an in-memory spec.
    pub fn from_spec(spec: crate::texture::TextureSpec) -> Self {
        Self { spec, opts: Default::default() }
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.spec.model = Some(model.into());
        self
    }
    pub fn size(mut self, size: u32) -> Self {
        self.spec.page.get_or_insert_with(Default::default).size = Some(size);
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.spec.seed = Some(seed);
        self
    }
    pub fn steps(mut self, steps: usize) -> Self {
        self.spec.steps = Some(steps);
        self
    }
    /// Tiled upscale: `none` / `2k` / `4k`.
    pub fn upscale(mut self, upscale: impl Into<String>) -> Self {
        self.opts.upscale = Some(upscale.into());
        self
    }
    /// Rejection sampling: try up to N seeds, keep the first that clears the scorecard.
    pub fn attempts(mut self, attempts: u32) -> Self {
        self.opts.attempts = attempts;
        self
    }
    /// Metallic source: `"auto"` (spatially-coherent region mask), `"from-albedo"`, or a scalar `"0.5"`.
    pub fn metallic(mut self, src: impl Into<String>) -> Self {
        self.spec.channels.get_or_insert_with(Default::default).metallic = Some(serde_json::Value::String(src.into()));
        self
    }
    /// Roughness source: `"auto"`, `"from-albedo"`, or a scalar `"0.5"`.
    pub fn roughness(mut self, src: impl Into<String>) -> Self {
        self.spec.channels.get_or_insert_with(Default::default).roughness = Some(serde_json::Value::String(src.into()));
        self
    }
    /// Anisotropy for brushed/grained metals: `strength` in `[0,1]`, `angle_deg` = `None` for auto-detect.
    pub fn anisotropy(mut self, strength: f32, angle_deg: Option<f32>) -> Self {
        let ch = self.spec.channels.get_or_insert_with(Default::default);
        ch.anisotropy = Some(strength);
        ch.anisotropy_angle = angle_deg;
        self
    }
    /// A hand-painted metallic mask PNG to use verbatim (overrides the metallic source).
    pub fn metallic_ref(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.opts.metallic_ref = Some(path.into());
        self
    }
    /// A hand-painted roughness mask PNG to use verbatim (overrides the roughness source).
    pub fn roughness_ref(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.opts.roughness_ref = Some(path.into());
        self
    }

    /// Render the material into `out` (a directory), returning its
    /// [`Scorecard`](crate::texture::Scorecard).
    pub async fn run(self, out: impl AsRef<std::path::Path>) -> Result<crate::texture::Scorecard> {
        crate::texture::render::render_material(&self.spec, out.as_ref(), &self.opts).await
    }
}

/// Blend two material directories through a mask into one PBR set (the `texture blend` op). `mask` is
/// `"mix"` (tileable, default) / `"radial"` / `"x"` / `"y"` / a PNG path. Weight-free.
pub fn texture_blend(
    dir_a: impl AsRef<std::path::Path>,
    dir_b: impl AsRef<std::path::Path>,
    mask: &str,
    out: impl AsRef<std::path::Path>,
) -> Result<crate::texture::Scorecard> {
    use crate::texture::{blend, compile, export, scorecard, Material, TextureSpec};
    let load = |d: &std::path::Path| -> Result<Material> {
        let albedo = image::open(d.join("albedo.png")).with_context(|| format!("albedo.png in {}", d.display()))?.to_rgb8();
        let (w, h) = albedo.dimensions();
        let gray = |n: &str, def: u8| image::open(d.join(n)).ok().map(|i| i.to_luma8()).unwrap_or_else(|| image::GrayImage::from_pixel(w, h, image::Luma([def])));
        let normal = image::open(d.join("normal.png")).ok().map(|i| i.to_rgb8()).unwrap_or_else(|| image::RgbImage::from_pixel(w, h, image::Rgb([128, 128, 255])));
        Ok(Material {
            albedo, normal,
            height: gray("height.png", 128),
            roughness: gray("roughness.png", 153),
            metallic: gray("metallic.png", 0),
            ao: gray("ao.png", 255),
            anisotropy: image::open(d.join("anisotropy.png")).ok().map(|i| i.to_rgb8()),
        })
    };
    let ma = load(dir_a.as_ref())?;
    let mb = load(dir_b.as_ref())?;
    let (w, h) = ma.albedo.dimensions();
    let mask_img = match mask {
        "mix" | "x" | "y" | "radial" | "horizontal" | "vertical" => blend::gradient_mask(w, h, mask),
        path => blend::fit_mask(&image::open(path)?.to_luma8(), w, h),
    };
    let m = blend::blend(&ma, &mb, &mask_img);
    let sc = scorecard::score(&m);
    let plan = compile::resolve(&TextureSpec::default());
    export::write_material(&m, &plan, &sc, out.as_ref())?;
    Ok(sc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_roundtrips_through_a_file() {
        let dir = std::env::temp_dir().join(format!("plakat-api-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.png");
        let img = Image { pixels: vec![10, 20, 30, 40, 50, 60], width: 2, height: 1 };
        img.save(&path).unwrap();
        let back = Image::open(&path).unwrap();
        assert_eq!((back.width(), back.height()), (2, 1));
        assert_eq!(back.pixels(), &[10, 20, 30, 40, 50, 60]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generate_builder_defaults_and_chaining() {
        let g = Generate::new("sdxl").prompt("x").negative("y").size(768, 512).steps(30).guidance(6.0).seed(7).count(3);
        assert_eq!(g.model, "sdxl");
        assert_eq!((g.width, g.height), (768, 512));
        assert_eq!(g.steps, 30);
        assert_eq!(g.count, 3);
        assert_eq!(g.seed, Some(7));
    }

    #[test]
    fn img2img_mask_makes_it_inpaint() {
        let plain = Img2img::new("sd15", "in.png").strength(0.4);
        assert!(plain.mask.is_none());
        let inpaint = Img2img::new("sd15", "in.png").mask("m.png").mask_feather(8).mask_invert(true);
        assert_eq!(inpaint.mask.as_deref(), Some(std::path::Path::new("m.png")));
        assert_eq!(inpaint.mask_feather, 8);
        assert!(inpaint.mask_invert);
        assert_eq!(plain.strength, 0.4);
    }

    #[test]
    fn upscale_builder_selects_method() {
        let u = Upscale::new("in.png").scale(2.0);
        assert!(matches!(u.method, UpscaleMethod::Lanczos3));
        let ml = Upscale::new("in.png").method(UpscaleMethod::RealEsrganX4);
        assert!(matches!(ml.method, UpscaleMethod::RealEsrganX4));
    }

    #[test]
    fn lora_specs_build_from_source_and_scale() {
        let loras = build_loras(&[Lora { source: "latent-consistency/lcm-lora-sdv1-5".into(), scale: 0.8 }]).unwrap();
        assert_eq!(loras.len(), 1);
        assert_eq!(loras[0].scale, 0.8);
    }

    #[test]
    fn portrait_and_multiperson_accept_photos_and_people() {
        let p = Portrait::new("sdxl").prompt("x").photo("a.png", 0.9).identity(IdentityKind::FaceIdSdxl).count(2);
        assert_eq!(p.photos.len(), 1);
        assert_eq!(p.count, 2);
        let mp = Multiperson::new("two friends at a cafe")
            .person(Person::new("alice").photo("a.png", 1.0).place(Position::Left, Distance::Mid, Facing::Front))
            .person(Person::new("bob").photo("b.png", 1.0));
        assert_eq!(mp.people.len(), 2);
        assert!(mp.composite);
    }

    #[test]
    fn training_and_verify_builders_default_sanely() {
        let st = StyleTrain::new("sd15", vec!["a.png".into()], "out.safetensors").trigger("mystyle").rank(8);
        assert_eq!(st.trigger, "mystyle");
        assert_eq!(st.rank, 8);
        let et = EmbeddingTrain::new("sdxl", vec!["a.png".into()], "<tok>", "e.safetensors").init_word("cat");
        assert_eq!(et.init_word, "cat");
        let v = Verify::new().tier(1).model("sdxl");
        assert_eq!(v.tier, Some(1));
        assert_eq!(v.model.as_deref(), Some("sdxl"));
    }

    #[test]
    fn animate_builder_formats_size_and_defaults() {
        let a = Animate::new("sd15", "a fox", "a wolf").frames(24).size(768, 512).animatediff(true);
        assert_eq!(a.frames, 24);
        assert_eq!((a.width, a.height), (768, 512));
        assert!(a.animatediff);
    }
}
