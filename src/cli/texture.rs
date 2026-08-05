//! `plakat texture` — seamless PBR material synthesis (RFC TEXTURE-1).
//!
//! B0 slice: `new` (scaffold a spec), `lint` (validate without weights), `show` (what a spec resolves
//! to — seamless mode, size, channel sources, export plan, the compiled albedo prompt). The derivation
//! core, scorecard, preview, export, seamless engine, and generation land across later phases
//! (ROADMAP_TEXTURE_1). Fully additive.

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use console::style;
use std::path::PathBuf;

use crate::texture::compile::{self, ChannelSource, HeightSource};
use crate::texture::lint::{self, Level};
use crate::texture::TextureSpec;

#[derive(Args, Debug)]
pub struct TextureArgs {
    #[command(subcommand)]
    pub cmd: TextureCmd,
}

#[derive(Subcommand, Debug)]
pub enum TextureCmd {
    /// Scaffold a new texture spec (a valid, partial `TextureSpec` HJSON you then edit).
    New(NewArgs),
    /// Validate a spec — schema, vocabulary, ranges. No weights, no network. Exits non-zero on any error.
    Lint(LintArgs),
    /// Show what a spec resolves to: seamless mode, size, channel sources, export plan, compiled prompt.
    Show(ShowArgs),
}

#[derive(Args, Debug)]
pub struct NewArgs {
    /// Output path for the new spec (`.hjson`).
    pub out: PathBuf,
    /// The material prompt.
    #[arg(long, default_value = "weathered stone wall")]
    pub material: String,
    /// Native generation size.
    #[arg(long, default_value_t = 1024)]
    pub size: u32,
    /// Diffusion base.
    #[arg(long, default_value = "sdxl")]
    pub model: String,
}

#[derive(Args, Debug)]
pub struct LintArgs {
    pub spec: PathBuf,
}

#[derive(Args, Debug)]
pub struct ShowArgs {
    pub spec: PathBuf,
}

pub async fn run(args: TextureArgs) -> Result<()> {
    match args.cmd {
        TextureCmd::New(a) => run_new(a),
        TextureCmd::Lint(a) => run_lint(a),
        TextureCmd::Show(a) => run_show(a),
    }
}

fn run_new(a: NewArgs) -> Result<()> {
    if a.out.exists() {
        anyhow::bail!("{} already exists — refusing to overwrite", a.out.display());
    }
    let template = format!(
        "{{\n  schema: \"{schema}\"\n  material: \"{material}\"\n\n\
         seamless: {{ mode: \"circular\", axes: \"both\" }}\n\
         channels: {{ height: \"auto\", roughness: 0.6, metallic: 0.0, normal_strength: 1.0, normal_y: \"opengl\" }}\n\
         delight: true\n\
         page: {{ size: {size}, upscale: \"none\", tiling_check: true }}\n\
         export: {{ maps: [\"albedo\", \"normal\", \"roughness\", \"metallic\", \"height\", \"ao\"], orm: true, gltf: false, naming: \"plakat\", preview: true }}\n\n\
         model: \"{model}\"\n  seed: 0\n  steps: 28\n}}\n",
        schema = crate::texture::SCHEMA_VERSION,
        material = a.material,
        size = a.size,
        model = a.model,
    );
    std::fs::write(&a.out, &template).with_context(|| format!("writing {}", a.out.display()))?;
    // Lint the scaffold so `new` never emits something `lint` rejects.
    let spec = TextureSpec::from_hjson(&template).context("scaffold failed to parse")?;
    let errs = lint::lint(&spec).into_iter().filter(|f| f.level == Level::Error).count();
    println!("{} {}  ({} · {}²)", style("wrote").green(), a.out.display(), a.material, a.size);
    if errs > 0 {
        anyhow::bail!("scaffold has {errs} lint error(s) — this is a bug");
    }
    Ok(())
}

fn run_lint(a: LintArgs) -> Result<()> {
    let spec = TextureSpec::load(&a.spec)?;
    let findings = lint::lint(&spec);
    let (mut errors, mut warns) = (0, 0);
    for f in &findings {
        match f.level {
            Level::Error => {
                errors += 1;
                println!("{} {}: {}", style("error").red().bold(), style(&f.path).bold(), f.message);
            }
            Level::Warn => {
                warns += 1;
                println!("{} {}: {}", style("warn").yellow(), style(&f.path).bold(), f.message);
            }
        }
    }
    if errors == 0 {
        println!("{} {} ({} warning(s))", style("✓ lint ok").green(), a.spec.display(), warns);
        Ok(())
    } else {
        anyhow::bail!("{} error(s), {} warning(s)", errors, warns)
    }
}

fn run_show(a: ShowArgs) -> Result<()> {
    let spec = TextureSpec::load(&a.spec)?;
    let p = compile::resolve(&spec);
    let cs = |c: &ChannelSource| match c {
        ChannelSource::Scalar(v) => format!("scalar {v}"),
        ChannelSource::FromAlbedo => "from-albedo".into(),
        ChannelSource::Prompt(s) => format!("prompt “{s}”"),
    };
    let hs = match &p.height {
        HeightSource::Auto => "auto (depth-CN)".to_string(),
        HeightSource::FromAlbedo => "from-albedo".to_string(),
        HeightSource::Prompt(s) => format!("prompt “{s}”"),
    };
    let b = |s: &str| style(s.to_string()).bold();
    println!("{}", b("plakat texture — resolved plan"));
    println!("  material     {}", if p.material.is_empty() { style("(image-to-material)").dim().to_string() } else { p.material.clone() });
    if let Some(img) = &p.from_image {
        println!("  from_image   {img}");
    }
    println!("  seamless     {} · axes {}", p.seamless_mode, p.seamless_axes);
    println!("  size         {}² → upscale {}", p.size, p.upscale);
    println!("  height       {hs}");
    println!("  roughness    {}", cs(&p.roughness));
    println!("  metallic     {}", cs(&p.metallic));
    println!("  normal       strength {} · {}", p.normal_strength, p.normal_y);
    println!("  ao           strength {}", p.ao_strength);
    println!("  delight      {}", p.delight);
    println!("  maps         {}", p.maps.join(", "));
    println!("  export       orm {} · gltf {} · naming {} · preview {}", p.orm, p.gltf, p.naming, p.preview);
    println!("  model        {} · seed {} · steps {}", p.model, p.seed, p.steps);
    if !p.prompt.is_empty() {
        println!("  {} {}", b("albedo prompt:"), style(&p.prompt).dim());
        println!("  {} {}", b("negative:"), style(&p.negative).dim());
    }
    Ok(())
}
