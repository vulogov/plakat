use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod animate;
pub mod artefact;
pub mod civitai;
pub mod doctor;
pub mod embedding;
pub mod generate;
pub mod img2img;
pub mod inspect;
pub mod models;
pub mod outpaint;
pub mod portrait;
pub mod scenario;
pub mod style;
pub mod stylize;
pub mod transparent;
pub mod upscale;

#[derive(Parser, Debug)]
#[command(name = "plakat", version, about = "Local text-to-image and style-transfer CLI")]
pub struct Cli {
    /// Increase log verbosity (-v, -vv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Override device: auto | cuda[:N] | metal | cpu.
    #[arg(long, global = true, default_value = "auto")]
    pub device: String,

    /// Custom cache directory for HuggingFace model downloads.
    /// Takes precedence over PLAKAT_CACHE_DIR / HF_HOME / HUGGINGFACE_HUB_CACHE.
    #[arg(long, global = true, env = "PLAKAT_CACHE_DIR", value_name = "PATH")]
    pub cache_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
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
    /// Make pixels matching the upper-left corner color transparent.
    Transparent(transparent::TransparentArgs),
    /// Resize an image larger using a classical filter (Lanczos by default).
    Upscale(upscale::UpscaleArgs),
    /// Batch-generate images from an HJSON scenario file.
    Scenario(scenario::ScenarioArgs),
    /// Manage the local HuggingFace model cache.
    #[command(subcommand)]
    Models(models::ModelsCmd),
    /// Health-check the FaceID / SCRFD / cache configuration without
    /// downloading or loading anything. Run before a long generation
    /// to verify setup.
    Doctor(doctor::DoctorArgs),
    /// Inspect a .safetensors file — list every tensor name, dtype,
    /// and shape. Useful when a weight load fails and you want to see
    /// what's actually in the file vs what the model expected.
    Inspect(inspect::InspectArgs),
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
    /// Currently `info` only — runtime injection into the SD
    /// pipeline lands when candle exposes `clip::Config.vocab_size`.
    #[command(subcommand_value_name = "OP")]
    Embedding(embedding::EmbeddingArgs),
    /// Animate between two prompts via CLIP-embedding lerp — N
    /// frames, fixed seed, optional GIF bundling. SD 1.5 / SD 2.1
    /// only in this release.
    Animate(animate::AnimateArgs),
}

pub async fn dispatch(cli: Cli) -> Result<()> {
    if let Some(p) = cli.cache_dir.clone() {
        crate::hf::cache::set_override(p);
    }
    match cli.command {
        Command::Generate(args) => {
            let device = crate::device::select(&cli.device)?;
            generate::run(args, device).await
        }
        Command::Portrait(args) => {
            let device = crate::device::select(&cli.device)?;
            portrait::run(args, device).await
        }
        Command::Img2img(args) => {
            let device = crate::device::select(&cli.device)?;
            img2img::run(args, device).await
        }
        Command::Outpaint(args) => {
            let device = crate::device::select(&cli.device)?;
            outpaint::run(args, device).await
        }
        Command::Stylize(args) => {
            let device = crate::device::select(&cli.device)?;
            stylize::run(args, device).await
        }
        Command::Transparent(args) => transparent::run(args).await,
        Command::Upscale(args) => {
            let device = crate::device::select(&cli.device)?;
            upscale::run(args, device).await
        }
        Command::Scenario(args) => scenario::run(args).await,
        Command::Models(cmd) => models::run(cmd).await,
        Command::Doctor(args) => doctor::run(args).await,
        Command::Inspect(args) => inspect::run(args).await,
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
    }
}
