//! v0.28 phase 3: `plakat motion-adapter` subcommand.
//!
//! Two sub-actions parallel to `plakat civitai`:
//!
//! ```text
//! plakat motion-adapter info REPO
//! plakat motion-adapter list
//! ```
//!
//! Both are read-only inspection helpers. `info` downloads the
//! adapter's `config.json` + safetensors header (cache-aware via
//! HF Hub) and prints a structured summary. `list` enumerates the
//! known plakat-supported adapter repos with their detected base
//! family (SD 1.5 vs SDXL) and config one-liner.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;

use crate::pipelines::motion_adapter::MotionAdapter;

#[derive(Args, Debug)]
pub struct MotionAdapterArgs {
    #[command(subcommand)]
    pub cmd: MotionAdapterCmd,
}

#[derive(Subcommand, Debug)]
pub enum MotionAdapterCmd {
    /// Download (or cache-hit) the adapter's config + safetensors
    /// and print a config dump + per-block tensor breakdown.
    Info(InfoArgs),
    /// Print the known plakat-supported motion-adapter repos
    /// (V3 SD 1.5, SDXL beta, AnimateLCM) plus community refs.
    List,
}

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// HuggingFace repo id (e.g. `guoyww/animatediff-motion-adapter-v1-5-3`).
    /// First run downloads ~1.4 GB; subsequent runs cache-hit.
    pub repo: String,
}

pub async fn run(args: MotionAdapterArgs) -> Result<()> {
    match args.cmd {
        MotionAdapterCmd::Info(a) => run_info(a).await,
        MotionAdapterCmd::List => {
            print_list();
            Ok(())
        }
    }
}

async fn run_info(args: InfoArgs) -> Result<()> {
    // Reuse the loader path — `load_v3` / `load_sdxl_beta` /
    // `load_animatelcm` all dispatch through `load_from_repo`, which
    // is private. So we mirror its surface via the public per-repo
    // constructors when the repo matches a known one; otherwise we
    // bail with a "use one of the known repos via list" pointer.
    // (A future v0.29 could add a public `MotionAdapter::load_from_any(repo)`
    // if user feedback warrants arbitrary HF repos.)
    let adapter = match args.repo.as_str() {
        "guoyww/animatediff-motion-adapter-v1-5-3" => {
            MotionAdapter::load_v3().await
        }
        "guoyww/animatediff-motion-adapter-sdxl-beta" => {
            MotionAdapter::load_sdxl_beta().await
        }
        "wangfuyun/AnimateLCM" => MotionAdapter::load_animatelcm().await,
        other => {
            anyhow::bail!(
                "plakat motion-adapter info {other:?}: only the three \
                 plakat-built-in repos are supported today. Run \
                 `plakat motion-adapter list` for the full set."
            );
        }
    }
    .with_context(|| format!("loading motion adapter from {}", args.repo))?;

    println!("{}", style(format!("Motion adapter — {}", args.repo)).bold());
    println!("{}", adapter.summary());
    println!("Detected base family: {}", detect_base_family(&adapter));
    Ok(())
}

fn detect_base_family(adapter: &MotionAdapter) -> &'static str {
    // SD 1.5 motion adapters use [320, 640, 1280, 1280] (4 blocks).
    // SDXL motion adapters use [320, 640, 1280] (3 blocks).
    match adapter.config.block_out_channels.as_slice() {
        [320, 640, 1280, 1280] => "SD 1.5 (4-block UNet)",
        [320, 640, 1280] => "SDXL (3-block UNet)",
        _ => "unknown",
    }
}

fn print_list() {
    let head = |s: &str| println!("\n{}", style(s).bold());

    head("plakat-supported motion adapters");
    println!(
        "  {} — V3, SD 1.5, 4-step LCM not native",
        style("guoyww/animatediff-motion-adapter-v1-5-3").cyan()
    );
    println!(
        "    {}",
        style("16 modules, 4 blocks × 2 layers, no mid").dim()
    );
    println!(
        "  {} — SDXL beta, 3-block UNet",
        style("guoyww/animatediff-motion-adapter-sdxl-beta").cyan()
    );
    println!(
        "    {}",
        style("12 modules, 3 blocks × 2 layers, no mid").dim()
    );
    println!(
        "  {} — LCM, SD 1.5 with mid-block motion",
        style("wangfuyun/AnimateLCM").cyan()
    );
    println!(
        "    {}",
        style("17 modules, 4 blocks × 2 layers + 1 mid; pairs with --lcm").dim()
    );

    head("Community refs (untested by plakat; should work via the same loader)");
    println!(
        "  {} — V1 SD 1.5 ({} 8 frames)",
        style("guoyww/animatediff-motion-adapter-v1-5").yellow(),
        style("trained at").dim()
    );
    println!(
        "  {} — V2 SD 1.5 ({} 24 frames)",
        style("guoyww/animatediff-motion-adapter-v1-5-2").yellow(),
        style("trained at").dim()
    );
    println!(
        "  {} — {}",
        style("hotshotco/Hotshot-XL").yellow(),
        style("SDXL Hotshot, different architecture (see RFC v0.27 §10)").dim()
    );

    head("Usage");
    println!(
        "  {}\n  {}\n  {}\n  {}",
        style("plakat motion-adapter info guoyww/animatediff-motion-adapter-v1-5-3").green(),
        style("plakat animate --animatediff --model sd15 --from \"...\"").green(),
        style("plakat animate --animatediff --model sdxl --from \"...\"").green(),
        style("plakat animate --animatediff --model sd15 --lcm --from \"...\"").green(),
    );
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::motion_adapter::MotionAdapterConfig;

    /// detect_base_family classifies V3 SD 1.5 + SDXL beta + AnimateLCM
    /// from their `block_out_channels`. No real adapter needed —
    /// the function only inspects the config field.
    #[test]
    fn detect_base_family_recognises_known_layouts() {
        fn synth(channels: Vec<usize>) -> MotionAdapter {
            MotionAdapter::synthetic_for_test(MotionAdapterConfig {
                class_name: "MotionAdapter".into(),
                diffusers_version: "test".into(),
                block_out_channels: channels,
                motion_layers_per_block: 2,
                motion_max_seq_length: 32,
                motion_mid_block_layers_per_block: 1,
                motion_norm_num_groups: 32,
                motion_num_attention_heads: 8,
                use_motion_mid_block: false,
            })
        }
        assert_eq!(
            detect_base_family(&synth(vec![320, 640, 1280, 1280])),
            "SD 1.5 (4-block UNet)"
        );
        assert_eq!(
            detect_base_family(&synth(vec![320, 640, 1280])),
            "SDXL (3-block UNet)"
        );
        assert_eq!(
            detect_base_family(&synth(vec![64, 128, 256])),
            "unknown"
        );
    }
}
