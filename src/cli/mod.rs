use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod animate;
pub mod artefact;
pub mod civitai;
pub mod clone;
pub mod compile;
#[cfg(feature = "onnx")]
pub mod convert_onnx;
pub mod compose;
pub mod map;
#[cfg(feature = "fractals")]
pub mod fractals;
pub mod bench;
pub mod doctor;
pub mod verify;
pub mod embedding;
pub mod gallery;
pub mod generate;
pub mod img2img;
pub mod import;
pub mod init;
pub mod inspect;
pub mod metadata;
pub mod models;
pub mod motion_adapter;
pub mod outpaint;
pub mod persona;
pub mod bookart;
pub mod texture;
pub mod remove;
pub mod replace_bg;
pub mod portrait;
pub mod multiperson;
pub mod relight;
pub mod run;
pub mod scenario;
pub mod segment;
pub mod style;
pub mod stylize;
pub mod transparent;
pub mod upscale;
pub mod rank;
pub mod restore_faces;
#[cfg(feature = "photos")]
pub mod photos;

#[derive(Parser, Debug)]
#[command(name = "plakat", version, about = "Local text-to-image and style-transfer CLI")]
pub struct Cli {
    /// Increase log verbosity (-v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true, help_heading = "Global options")]
    pub verbose: u8,

    /// Override device: auto | cuda[:N] | metal | cpu.
    #[arg(long, global = true, default_value = "auto", help_heading = "Global options")]
    pub device: String,

    /// Custom cache directory for HuggingFace model downloads.
    /// Takes precedence over PLAKAT_CACHE_DIR / HF_HOME / HUGGINGFACE_HUB_CACHE.
    #[arg(long, global = true, env = "PLAKAT_CACHE_DIR", value_name = "PATH", help_heading = "Global options")]
    pub cache_dir: Option<PathBuf>,

    /// Allow this run even when another plakat instance is already running on
    /// the host. By default a second heavy (model / training) run is refused —
    /// concurrent runs share unified memory and thrash. (env
    /// `PLAKAT_ALLOW_MULTIPLE_INSTANCES=1` does the same.)
    #[arg(long, global = true, help_heading = "Global options")]
    pub enable_multiple_instances: bool,

    /// Write provenance etching (RFC ETCH-1) into images plakat produces. Off by default; silently
    /// ignored by commands that don't write an image.
    #[arg(long, global = true, env = "PLAKAT_ETCH", help_heading = "Provenance (etch)")]
    pub etch: bool,

    /// Key for `EtchId` derivation and carrier PRNG (public constant by default).
    #[arg(long, global = true, env = "PLAKAT_ETCH_KEY", value_name = "KEY", help_heading = "Provenance (etch)")]
    pub etch_key: Option<String>,

    /// Override the derived `EtchId` with an explicit 64-bit hex value.
    #[arg(long, global = true, value_name = "HEX16", help_heading = "Provenance (etch)")]
    pub etch_id: Option<String>,

    /// Comma-list of layers to write: `l0,l1,l2,l3` (default: all applicable).
    #[arg(long, global = true, value_name = "LIST", help_heading = "Provenance (etch)")]
    pub etch_layers: Option<String>,

    /// L1 embedding strength, `0.0..=1.0` (default 0.35).
    #[arg(long, global = true, value_name = "F32", default_value_t = 0.35, help_heading = "Provenance (etch)")]
    pub etch_strength: f32,

    /// L3 fingerprint store (`none` disables L3; default `$PLAKAT_HOME/etchdb`).
    #[arg(long, global = true, value_name = "PATH|none", help_heading = "Provenance (etch)")]
    pub etch_db: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

impl Cli {
    /// Build the runtime [`EtchConfig`](crate::etch::EtchConfig) from the global `--etch*` flags.
    pub fn etch_config(&self) -> crate::etch::EtchConfig {
        let db = match self.etch_db.as_deref() {
            Some("none") => None,
            Some(p) => Some(std::path::PathBuf::from(p)),
            None => crate::etch::default_db(),
        };
        crate::etch::EtchConfig {
            enabled: self.etch,
            key: self.etch_key.clone().unwrap_or_else(|| crate::etch::PUBLIC_KEY.to_string()),
            id_override: self.etch_id.as_deref().and_then(crate::etch::EtchId::parse_hex),
            layers: self.etch_layers.as_deref().map(crate::etch::Layer::parse_list).unwrap_or_default(),
            strength: self.etch_strength.clamp(0.0, 1.0),
            db,
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Generate images from a text prompt.
    Generate(generate::GenerateArgs),
    /// Generate a portrait, optionally from a reference photo.
    Portrait(portrait::PortraitArgs),
    /// Image-to-image: transform an existing image with a prompt.
    /// Supply `--mask` to restrict changes to a region (inpaint).
    Img2img(img2img::Img2ImgArgs),
    /// Outpaint: extend an image past its borders. Pads the canvas,
    /// builds a mask of the new region, hands off to the inpaint
    /// pipeline.
    Outpaint(outpaint::OutpaintArgs),
    /// Apply the style of REF to IN, producing OUT.
    Stylize(stylize::StylizeArgs),
    /// Relight a foreground subject with IC-Light: matte the subject,
    /// composite it onto neutral grey, and re-illuminate it from a text
    /// prompt describing the lighting. SD 1.5-based (widened 8-channel
    /// UNet + IC-Light offset weights).
    Relight(relight::RelightArgs),
    /// Place 2+ specific personas into one generated scene, each at a relative
    /// location (`--at "alice:left closer front"`); un-pinned personas are
    /// placed by a scene-aware LLM. Reference photos give each their identity.
    Multiperson(multiperson::MultipersonArgs),
    /// Make pixels matching the upper-left corner color transparent.
    Transparent(transparent::TransparentArgs),
    /// Segment an object by clicking it (Segment-Anything / MobileSAM):
    /// point prompts → a binary mask PNG that feeds `img2img --mask`
    /// (inpaint) and any other `--mask` consumer. The selection enabler
    /// for compose-and-edit (object removal / replacement).
    Segment(segment::SegmentArgs),
    /// Erase an object and fill the hole seamlessly. Select it with `--point`,
    /// `--box`, or `--depth-band` (SAM) → the region is inpainted away while the
    /// rest is preserved. One-shot wrapper over segment + inpaint.
    Remove(remove::RemoveArgs),
    /// Replace an image's background while keeping the subject. Mattes the
    /// subject (U2Net), generates a new background from `--prompt` (or uses
    /// `--bg-image`), and alpha-composites the subject over it.
    ReplaceBg(replace_bg::ReplaceBgArgs),
    /// Controllable synthetic-person composition (RFC PERSONA-1, the 5.0 flagship).
    /// A `PersonaSpec` HJSON → reproducible person. `new` scaffolds a spec; `lint`
    /// validates it (more subcommands land across the 5.0 phases).
    Persona(persona::PersonaArgs),
    /// Controllable B/W book-ornament composition (RFC BOOKART-1). A `BookArtSpec`
    /// HJSON → transparent, print-sized ornament. `new` scaffolds, `lint` validates,
    /// `show` resolves (render tiers land across later phases).
    Bookart(bookart::BookartArgs),
    /// Seamless PBR material synthesis (RFC TEXTURE-1). A `TextureSpec` HJSON → a
    /// tileable albedo/normal/roughness/metallic/height/AO set. `new` scaffolds,
    /// `lint` validates, `show` resolves (generation + derivation land across later phases).
    Texture(texture::TextureArgs),
    /// Resize an image larger using a classical filter (Lanczos by default).
    Upscale(upscale::UpscaleArgs),
    /// Score images by aesthetic quality (LAION CLIP predictor) and rank them,
    /// best first. Feeds `generate --keep-best` and the collection manager's curation.
    Rank(rank::RankArgs),
    /// Restore degraded faces in existing images (SCRFD-detect → diffusion-refine → composite).
    /// The standalone form of `generate --adetailer`; pairs with `upscale --diffusion`.
    RestoreFaces(restore_faces::RestoreFacesArgs),
    /// Batch-generate images from an HJSON scenario file.
    Scenario(scenario::ScenarioArgs),
    /// Compose a layered scene from an HJSON file: stack image layers
    /// (background + placed cut-outs / artefacts) with z-order, position,
    /// scale, and opacity. No GPU — composes existing image assets.
    Compose(compose::ComposeArgs),
    /// Compile a prose `prompts.txt` into a `scenario` HJSON: each block becomes
    /// a task, prompts are LLM-enhanced (family-aware) with auto-negatives.
    /// `--no-enhance --no-negative` is deterministic (no LLM).
    Compile(compile::CompileArgs),
    /// Generate a fantasy map from a prose world description. MAP-1: prose →
    /// `MapSpec v2` JSON via the LLM (`--map-dump-spec`); geometry + render follow.
    Map(map::MapArgs),
    /// Generate fractals — pure-CPU deterministic render (Track A), optional AI paint
    /// pass (Track B). Escape-time families (Mandelbrot / Julia / Burning Ship) with
    /// Lab-space palettes; the spec is embedded in the PNG for `--fractal-clone`.
    /// RFC FRACTALS-1. Behind `--features fractals`.
    #[cfg(feature = "fractals")]
    Fractals(fractals::FractalsArgs),
    /// Manage the local HuggingFace model cache.
    #[command(subcommand)]
    Models(models::ModelsCmd),
    /// Health-check the environment without downloading or loading
    /// anything: ArcFace / SCRFD identity weights, build vs runtime
    /// device alignment, HF cache disk usage, ffmpeg presence
    /// (v0.30), and HF / Civitai API token presence (v0.30 — never
    /// the value). Add `--json` for a structured report or
    /// `--benchmark` for a synthetic per-op latency measure.
    Doctor(doctor::DoctorArgs),
    /// Verify model correctness (RFC_VERIFY.md). Tier 0 (structural /
    /// determinism) runs offline with no downloads; higher tiers compare
    /// against Hugging Face-hosted golden data. `--json` for CI gating.
    Verify(verify::VerifyArgs),
    /// Benchmark real generation (load / per-step / VAE / peak-mem). Phase 0 of the perf pass.
    Bench(bench::BenchArgs),
    /// Inspect a .safetensors file — list every tensor name, dtype,
    /// and shape. Useful when a weight load fails and you want to see
    /// what's actually in the file vs what the model expected.
    Inspect(inspect::InspectArgs),
    /// Convert an ONNX model into the plakat `.safetensors` layout. ONNX names
    /// weights by graph node, not by the module tree a pipeline loads; this
    /// renames them so plakat can consume the file. Currently supports
    /// `--arch scrfd-500mf` (InsightFace `det_500m.onnx` → the SCRFD face
    /// detector behind `--identity faceid` / `--adetailer` / `multiperson`).
    #[cfg(feature = "onnx")]
    #[command(name = "convert-onnx")]
    ConvertOnnx(convert_onnx::ConvertOnnxArgs),
    /// Interactive terminal UI (RFC TUI-1): conversational generation, models,
    /// scenarios, history, LoRA, people. Needs a graphics-capable terminal.
    #[cfg(feature = "ui")]
    #[command(name = "ui")]
    Ui(crate::ui::tui::UiArgs),
    /// TUI photo & image collection manager (RFC PHOTOS-1): browse → curate → edit → generate over
    /// an image library. The 3.x flagship. Needs a graphics-capable terminal.
    #[cfg(feature = "photos")]
    Photos(photos::PhotosArgs),
    /// Art-style detection from a reference photo.
    #[command(subcommand_value_name = "OP")]
    Style(style::StyleArgs),
    /// Artefact library: cutout PNGs that can be composited into
    /// named zones of a generated image.
    #[command(subcommand_value_name = "OP")]
    Artefact(artefact::ArtefactArgs),
    /// Browse + download Civitai models, LoRAs, and embeddings.
    /// See `plakat civitai --help` for sub-actions.
    #[command(subcommand_value_name = "OP")]
    Civitai(civitai::CivitaiArgs),
    /// Inspect Textual Inversion (embedding) `.safetensors` files.
    /// Runtime injection works via `plakat generate --embedding PATH`
    /// (SD 1.5 / SD 2.1 / SDXL — both CLIP-L-only and dual
    /// CLIP-L+CLIP-G formats).
    #[command(subcommand_value_name = "OP")]
    Embedding(embedding::EmbeddingArgs),
    /// Animate between two prompts via CLIP-embedding lerp — N
    /// frames, fixed seed, optional GIF bundling. SD 1.5 / SD 2.1
    /// only in this release.
    Animate(animate::AnimateArgs),
    /// v0.28 phase 3: inspect AnimateDiff motion adapters.
    /// `plakat motion-adapter info REPO` downloads + dumps the
    /// adapter's config + tensor breakdown; `plakat motion-adapter
    /// list` enumerates the plakat-supported repos.
    #[command(name = "motion-adapter", subcommand_value_name = "OP")]
    MotionAdapter(motion_adapter::MotionAdapterArgs),
    /// v0.18: read back the Auto1111 `parameters` PNG tEXt chunk +
    /// JSON sidecar plakat writes alongside every generation.
    /// Reverse of the metadata write path — recover prompt / seed /
    /// model / LoRAs from a PNG without consulting the shell.
    Metadata(metadata::MetadataArgs),
    /// v0.19: translate a generated PNG's metadata into a
    /// re-runnable `plakat generate` shell command. Pairs with
    /// `metadata` (inspect → translate). JSON sidecar preferred;
    /// falls back to parsing the Auto1111 `parameters` PNG tEXt
    /// chunk (works on Civitai uploads + A1111 Web UI outputs).
    Clone(clone::CloneArgs),
    /// v0.20: bootstrap a starter project — `scenario.hjson`,
    /// `wildcards/` (with a few example files), and a focused
    /// `.gitignore`. Defaults to the current directory; pass DIR
    /// to write somewhere else.
    Init(init::InitArgs),
    /// v0.21: evaluate a Bund script. Host words namespaced under
    /// `plakat.*` drive the same pipelines `plakat generate` /
    /// `img2img` / `portrait` / `upscale` use. See
    /// `Documentation/RFC_v0.21_BUND_SCRIPTING.md` for the design.
    Run(run::RunArgs),
    /// v0.43: build a Markdown gallery index from a directory of
    /// plakat-generated PNGs. Reads each image's embedded generation
    /// metadata (JSON sidecar, else the A1111 `parameters` chunk) and
    /// emits a thumbnail grid + per-image prompt/settings. The
    /// reproducible companion to the `gallery/` proof corpus.
    Gallery(gallery::GalleryArgs),
}

impl Command {
    /// Does this command load image models / do real GPU work — and thus risk
    /// thrashing a concurrent run on a shared (unified-memory) host? Pure
    /// introspection + deterministic utilities (models, doctor, inspect,
    /// metadata, init, gallery, compile, civitai, map, transparent, artefact,
    /// clone) return false so they can always run alongside a busy host; the
    /// single-instance guard only applies to the heavy ones.
    fn is_heavy(&self) -> bool {
        matches!(
            self,
            Command::Generate(_)
                | Command::Portrait(_)
                | Command::Img2img(_)
                | Command::Outpaint(_)
                | Command::Stylize(_)
                | Command::Relight(_)
                | Command::Multiperson(_)
                | Command::Segment(_)
                | Command::Remove(_)
                | Command::ReplaceBg(_)
                | Command::Upscale(_)
                | Command::Rank(_)
                | Command::RestoreFaces(_)
                | Command::Scenario(_)
                | Command::Compose(_)
                | Command::Animate(_)
                | Command::Run(_)
                | Command::Style(_)
                | Command::Embedding(_)
                | Command::Bench(_)
        )
    }
}

pub async fn dispatch(cli: Cli) -> Result<()> {
    if let Some(p) = cli.cache_dir.clone() {
        crate::hf::cache::set_override(p);
    }
    // Install the process-wide etch config from the global `--etch*` flags (RFC ETCH-1). Off by default.
    crate::etch::set_config(cli.etch_config());
    // Single-instance guard: refuse a second heavy run on the host (they share
    // unified memory and thrash). `--enable-multiple-instances` overrides.
    if cli.command.is_heavy() {
        crate::instance_guard::enforce_single_instance(cli.enable_multiple_instances)?;
    }
    match cli.command {
        Command::Generate(args) => {
            let device = crate::device::select(&cli.device)?;
            generate::run(args, device).await
        }
        Command::Portrait(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, portrait::run(args, device)).await
        }
        Command::Img2img(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, img2img::run(args, device)).await
        }
        Command::Outpaint(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, outpaint::run(args, device)).await
        }
        Command::Stylize(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, stylize::run(args, device)).await
        }
        Command::Relight(a) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (a.import.clone(), a.out.clone());
            import::run_with_import(imp, out, relight::run(a, device)).await
        }
        Command::Multiperson(a) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (a.import.clone(), a.out.clone());
            import::run_with_import(imp, out, multiperson::run(a, device)).await
        }
        Command::Transparent(args) => {
            let device = crate::device::select(&cli.device)?;
            transparent::run(args, device).await
        }
        Command::Segment(args) => {
            let device = crate::device::select(&cli.device)?;
            segment::run(args, device).await
        }
        Command::Remove(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, remove::run(args, device)).await
        }
        Command::ReplaceBg(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, replace_bg::run(args, device)).await
        }
        Command::Upscale(args) => {
            let device = crate::device::select(&cli.device)?;
            let (imp, out) = (args.import.clone(), args.out.clone());
            import::run_with_import(imp, out, upscale::run(args, device)).await
        }
        Command::Rank(args) => {
            let device = crate::device::select(&cli.device)?;
            rank::run(args, device).await
        }
        Command::RestoreFaces(args) => {
            let device = crate::device::select(&cli.device)?;
            restore_faces::run(args, device).await
        }
        Command::Scenario(args) => scenario::run(args).await,
        Command::Compile(args) => compile::run(args).await,
        Command::Map(args) => map::run(args, &cli.device).await,
        #[cfg(feature = "fractals")]
        Command::Fractals(args) => fractals::run(args, &cli.device).await,
        Command::Compose(args) => {
            let device = crate::device::select(&cli.device)?;
            compose::run(args, device).await
        }
        Command::Models(cmd) => models::run(cmd).await,
        Command::Doctor(args) => doctor::run(args).await,
        Command::Verify(args) => verify::run(args).await,
        Command::Bench(args) => bench::run(args).await,
        Command::Inspect(args) => inspect::run(args).await,
        Command::Persona(args) => persona::run(args).await,
        Command::Bookart(args) => bookart::run(args).await,
        Command::Texture(args) => texture::run(args).await,
        #[cfg(feature = "onnx")]
        Command::ConvertOnnx(args) => convert_onnx::run(args).await,
        #[cfg(feature = "ui")]
        Command::Ui(args) => crate::ui::tui::run(args),
        #[cfg(feature = "photos")]
        Command::Photos(args) => photos::run(args).await,
        Command::Style(args) => {
            let device = crate::device::select(&cli.device)?;
            style::run(args, device).await
        }
        Command::Artefact(args) => artefact::run(args).await,
        Command::Civitai(args) => civitai::run(args).await,
        Command::Embedding(args) => embedding::run(args).await,
        Command::Animate(args) => {
            let device = crate::device::select(&cli.device)?;
            animate::run(args, device).await
        }
        Command::MotionAdapter(args) => motion_adapter::run(args).await,
        Command::Metadata(args) => metadata::run(args).await,
        Command::Clone(args) => clone::run(args).await,
        Command::Init(args) => init::run(args).await,
        Command::Run(args) => {
            let device = crate::device::select(&cli.device)?;
            run::run(args, device).await
        }
        Command::Gallery(args) => gallery::run(args).await,
    }
}
