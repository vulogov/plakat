use anyhow::Result;
use candle_core::Device;
use clap::Args as ClapArgs;
use std::path::PathBuf;

#[derive(ClapArgs, Debug)]
pub struct StylizeArgs {
    /// Input image to transform.
    #[arg(long = "in", value_name = "IN")]
    pub input: PathBuf,

    /// Style reference image.
    #[arg(long = "ref", value_name = "REF")]
    pub reference: PathBuf,

    /// Output image path.
    #[arg(long, value_name = "OUT")]
    pub out: PathBuf,

    /// Strength of style transfer in [0.0, 1.0]. Higher = closer to REF.
    #[arg(long, default_value_t = 0.7)]
    pub strength: f32,

    /// Base diffusion model (alias or HF repo id). Currently SD 1.5 only.
    #[arg(long, default_value = "sd15")]
    pub model: String,

    /// Denoising steps.
    #[arg(long, default_value_t = 30)]
    pub steps: usize,

    /// Random seed.
    #[arg(long)]
    pub seed: Option<u64>,
}

pub async fn run(args: StylizeArgs, device: Device) -> Result<()> {
    crate::pipelines::stylize::run(crate::pipelines::stylize::Request {
        input: args.input,
        reference: args.reference,
        out: args.out,
        strength: args.strength,
        model: args.model,
        steps: args.steps,
        seed: args.seed,
        device,
    })
    .await
}
