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
        }
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
            width: 0, // 0 = keep the input's dimensions
            height: 0,
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
}
