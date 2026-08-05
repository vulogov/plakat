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
    /// Derive the full PBR channel set (normal/roughness/metallic/height/AO) from an existing albedo —
    /// **no weights, no GPU**. `--height` supplies a height map instead of deriving one from luminance.
    Derive(DeriveArgs),
    /// Score a material directory: tileability (x/y edge-wrap), normal validity, albedo flatness,
    /// channel consistency. **No weights.**
    Verify(VerifyArgs),
    /// Re-render the lit PBR preview from a material directory (pure-Rust, **no GPU**).
    Preview(PreviewArgs),
    /// Re-pack a material directory for an engine — naming convention, ORM, glTF (**no weights**).
    Export(ExportArgs),
    /// Render a full seamless PBR material from a spec (generate albedo → delight → derive → export).
    Render(RenderArgs),
}

#[derive(Args, Debug)]
pub struct RenderArgs {
    pub spec: PathBuf,
    /// Output material directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Rejection sampling: try up to N seeds, keep the first that clears the scorecard.
    #[arg(long, default_value_t = 1)]
    pub attempts: u32,
    /// Override the spec's tiled upscale: `none` / `2k` / `4k`.
    #[arg(long)]
    pub upscale: Option<String>,
}

#[derive(Args, Debug)]
pub struct PreviewArgs {
    /// A material directory (contains `albedo.png`, `normal.png`, …).
    pub dir: PathBuf,
    /// Output preview PNG (default `<dir>/preview.png`).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Preview geometry: `sphere` (default) or `plane`.
    #[arg(long, default_value = "sphere")]
    pub shape: String,
    /// Preview resolution.
    #[arg(long, default_value_t = 512)]
    pub size: u32,
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// A material directory to re-pack.
    pub dir: PathBuf,
    /// Output directory (default: in place).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Channel naming: `plakat` (default) | `unity` | `unreal`.
    #[arg(long, default_value = "plakat")]
    pub naming: String,
    /// Also write the packed ORM.
    #[arg(long, default_value_t = true)]
    pub orm: bool,
    /// Also write a glTF 2.0 material.
    #[arg(long, default_value_t = false)]
    pub gltf: bool,
}

#[derive(Args, Debug)]
pub struct DeriveArgs {
    /// The albedo (base color) PNG.
    pub albedo: PathBuf,
    /// Output material directory.
    #[arg(long)]
    pub out: PathBuf,
    /// Supply a height map instead of deriving one from the albedo's luminance.
    #[arg(long)]
    pub height: Option<PathBuf>,
    /// Normal slope gain.
    #[arg(long, default_value_t = 1.0)]
    pub normal_strength: f32,
    /// AO strength.
    #[arg(long, default_value_t = 1.0)]
    pub ao_strength: f32,
    /// Normal Y convention: `opengl` (+Y) or `directx` (-Y).
    #[arg(long, default_value = "opengl")]
    pub normal_y: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    /// A material directory (contains `albedo.png`, `normal.png`, …).
    pub dir: PathBuf,
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
        TextureCmd::Derive(a) => run_derive(a),
        TextureCmd::Verify(a) => run_verify(a),
        TextureCmd::Preview(a) => run_preview(a),
        TextureCmd::Export(a) => run_export(a),
        TextureCmd::Render(a) => run_render(a).await,
    }
}

/// B4: render a full seamless PBR material from a spec (GPU).
async fn run_render(a: RenderArgs) -> Result<()> {
    use crate::texture::render::{render_material, RenderOpts};
    let spec = TextureSpec::load(&a.spec)?;
    let sc = render_material(&spec, &a.out, &RenderOpts { attempts: a.attempts, upscale: a.upscale.clone() }).await?;
    print_scorecard(&sc);
    println!("{} {}  (seamless PBR material)", style("wrote").green(), a.out.display());
    Ok(())
}

/// B2: load a material directory into a `Material` (missing data maps → neutral fill).
fn load_material(dir: &std::path::Path) -> Result<crate::texture::Material> {
    use crate::texture::Material;
    let albedo = image::open(dir.join("albedo.png"))
        .with_context(|| format!("a material dir needs albedo.png ({})", dir.display()))?
        .to_rgb8();
    let (w, h) = albedo.dimensions();
    let gray = |n: &str, d: u8| -> image::GrayImage {
        image::open(dir.join(n)).ok().map(|i| i.to_luma8()).unwrap_or_else(|| image::GrayImage::from_pixel(w, h, image::Luma([d])))
    };
    let normal = image::open(dir.join("normal.png")).ok().map(|i| i.to_rgb8()).unwrap_or_else(|| image::RgbImage::from_pixel(w, h, image::Rgb([128, 128, 255])));
    Ok(Material {
        albedo,
        height: gray("height.png", 128),
        normal,
        roughness: gray("roughness.png", 153),
        metallic: gray("metallic.png", 0),
        ao: gray("ao.png", 255),
    })
}

/// B1/B2: derive the full PBR set from an albedo (weight-free) → score → write the material dir
/// (channels + ORM + preview + manifest).
fn run_derive(a: DeriveArgs) -> Result<()> {
    use crate::texture::{compile, export, scorecard, Material, TextureSpec};
    let albedo = image::open(&a.albedo).with_context(|| format!("opening {}", a.albedo.display()))?.to_rgb8();
    let height = match &a.height {
        Some(p) => Some(image::open(p).with_context(|| format!("opening {}", p.display()))?.to_luma8()),
        None => None,
    };
    // `derive` has no spec → from-albedo channel sources on top of the resolved defaults.
    let mut plan = compile::resolve(&TextureSpec::default());
    plan.roughness = ChannelSource::FromAlbedo;
    plan.normal_y = if a.normal_y.eq_ignore_ascii_case("directx") { "directx".into() } else { "opengl".into() };
    let m = Material::derive(
        albedo,
        height,
        a.normal_strength,
        plan.normal_y == "opengl",
        a.ao_strength,
        &plan.roughness,
        &plan.metallic,
    );
    let sc = scorecard::score(&m);
    export::write_material(&m, &plan, &sc, &a.out)?;
    print_scorecard(&sc);
    println!("{} {}  ({} maps + orm + preview + manifest)", style("wrote").green(), a.out.display(), plan.maps.len());
    Ok(())
}

/// B1: score an existing material directory.
fn run_verify(a: VerifyArgs) -> Result<()> {
    let m = load_material(&a.dir)?;
    let sc = crate::texture::scorecard::score(&m);
    print_scorecard(&sc);
    if sc.passes {
        Ok(())
    } else {
        anyhow::bail!("{} scorecard issue(s)", sc.notes.len())
    }
}

/// B2: re-render the lit preview from a material directory.
fn run_preview(a: PreviewArgs) -> Result<()> {
    use crate::texture::{preview, Shape};
    let m = load_material(&a.dir)?;
    let img = preview::render(&m, Shape::parse(&a.shape), a.size);
    let out = a.out.unwrap_or_else(|| a.dir.join("preview.png"));
    img.save(&out).with_context(|| format!("writing {}", out.display()))?;
    println!("{} {}  ({} · {}²)", style("wrote").green(), out.display(), a.shape, a.size);
    Ok(())
}

/// B2: re-pack a material directory for an engine (naming / ORM / glTF).
fn run_export(a: ExportArgs) -> Result<()> {
    use crate::texture::{compile, export, scorecard, TextureSpec};
    let m = load_material(&a.dir)?;
    let sc = scorecard::score(&m);
    let mut plan = compile::resolve(&TextureSpec::default());
    plan.naming = a.naming.clone();
    plan.orm = a.orm;
    plan.gltf = a.gltf;
    plan.preview = false; // re-packing doesn't re-render the preview
    let out = a.out.unwrap_or_else(|| a.dir.clone());
    export::write_material(&m, &plan, &sc, &out)?;
    println!("{} {}  (naming {} · orm {} · gltf {})", style("packed").green(), out.display(), a.naming, a.orm, a.gltf);
    Ok(())
}

fn print_scorecard(sc: &crate::texture::Scorecard) {
    let mark = |ok: bool| if ok { style("✓").green() } else { style("✗").red() };
    println!("{}", style("scorecard").bold());
    println!("  {} tileability   x {:.2} · y {:.2}", mark(sc.tileability_x <= 1.5 && sc.tileability_y <= 1.5), sc.tileability_x, sc.tileability_y);
    println!("  {} normal-valid  {:.3}", mark(sc.normal_valid >= 0.99), sc.normal_valid);
    println!("  {} consistent    {}", mark(sc.consistent), sc.consistent);
    println!("  {} albedo-flat   {:.3}", mark(sc.albedo_flatness <= 0.14), sc.albedo_flatness);
    for n in &sc.notes {
        println!("    {} {}", style("·").yellow(), n);
    }
    println!("  {}", if sc.passes { style("PASS").green().bold() } else { style("FAIL").red().bold() });
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
