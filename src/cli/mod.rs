use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub mod generate;
pub mod models;
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
    /// Apply the style of REF to IN, producing OUT.
    Stylize(stylize::StylizeArgs),
    /// Make pixels matching the upper-left corner color transparent.
    Transparent(transparent::TransparentArgs),
    /// Resize an image larger using a classical filter (Lanczos by default).
    Upscale(upscale::UpscaleArgs),
    /// Manage the local HuggingFace model cache.
    #[command(subcommand)]
    Models(models::ModelsCmd),
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
        Command::Stylize(args) => {
            let device = crate::device::select(&cli.device)?;
            stylize::run(args, device).await
        }
        Command::Transparent(args) => transparent::run(args).await,
        Command::Upscale(args) => upscale::run(args).await,
        Command::Models(cmd) => models::run(cmd).await,
    }
}
