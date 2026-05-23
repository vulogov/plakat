//! v0.16 phase 9: `plakat embedding` subcommand.
//!
//! Single action shipping in v0.16:
//!
//! ```text
//! plakat embedding info PATH
//! ```
//!
//! Inspects a Textual Inversion safetensors file and prints:
//! trigger word (filename-derived), embedding dimensions, vector
//! count, and which CLIP variant the dims match (SD 1.5 / SD 2.1).
//!
//! Runtime injection of TIs into the SD pipeline is gated until
//! the candle CLIP API blocker is resolved (see
//! [`crate::pipelines::embedding`] module docs). `info` works
//! today against the same parser the future runtime path will use.

use anyhow::Result;
use candle_core::Device;
use clap::{Args, Subcommand};
use console::style;

use crate::pipelines::embedding::{
    self, EmbeddingSpec, SD15_EMBED_DIM, SD21_EMBED_DIM, SDXL_G_EMBED_DIM,
};

#[derive(Args, Debug)]
pub struct EmbeddingArgs {
    #[command(subcommand)]
    pub cmd: EmbeddingCmd,
}

#[derive(Subcommand, Debug)]
pub enum EmbeddingCmd {
    /// Inspect a Textual Inversion `.safetensors` file. Prints the
    /// trigger word, vector count, embedding dim, and the matching
    /// SD variant.
    Info(InfoArgs),
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Path to a TI `.safetensors` file. HF repo specs work too
    /// (auto-downloads on first run).
    pub source: String,
    /// Override the trigger word derived from the filename.
    #[arg(long)]
    pub trigger: Option<String>,
}

pub async fn run(args: EmbeddingArgs) -> Result<()> {
    match args.cmd {
        EmbeddingCmd::Info(a) => run_info(a).await,
    }
}

async fn run_info(args: InfoArgs) -> Result<()> {
    let spec = EmbeddingSpec {
        source: args.source.clone(),
        trigger: args.trigger.clone(),
        scale: 1.0,
    };
    let path = embedding::resolve(&spec).await?;
    let resolved = embedding::parse_safetensors(&path, &spec, &Device::Cpu)?;

    let num_tokens = resolved.num_tokens()?;
    let embed_dim = resolved.embed_dim()?;
    let variant_hint = match embed_dim {
        SD15_EMBED_DIM => "SD 1.5 (CLIP-L 768)",
        SD21_EMBED_DIM => "SD 2.1 (OpenCLIP-H 1024)",
        SDXL_G_EMBED_DIM => "SDXL CLIP-G (1280)",
        other => return Err(anyhow::anyhow!(
            "unknown CLIP variant for embed_dim {other} (expected 768 / 1024 / 1280)"
        )),
    };

    println!(
        "{} {}",
        style("file:").dim(),
        path.display()
    );
    println!(
        "{} {}",
        style("trigger:").dim(),
        style(&resolved.trigger).cyan().bold()
    );
    println!(
        "{} {} vector(s) × {} dim   {}",
        style("shape:").dim(),
        num_tokens,
        embed_dim,
        style(format!("[{variant_hint}]")).yellow(),
    );
    println!(
        "{} `{}` once per prompt; use {} times consecutively for the {} additional vector(s).",
        style("usage:").dim(),
        resolved.trigger,
        num_tokens,
        if num_tokens > 1 { num_tokens - 1 } else { 0 },
    );
    if num_tokens > 1 {
        println!(
            "{} multi-vector TIs are rendered as {} consecutive tokens in the prompt — \
             each token gets its own embedding vector at inference time.",
            style("note:").dim(),
            num_tokens
        );
    }
    println!();
    println!(
        "{} runtime injection into `plakat generate` is gated until candle 0.8+ exposes \
         `clip::Config.vocab_size` for mutation. The parser + merger ship; pass \
         `--embedding {}` once that lands.",
        style("status:").dim(),
        args.source
    );
    Ok(())
}
