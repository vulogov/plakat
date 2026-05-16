//! `scenario` — batch-generate images from an HJSON file that mixes scenes,
//! weather, and per-task prompts. See README for the schema and an example.

use anyhow::{Context, Result, anyhow, bail};
use clap::Args as ClapArgs;
use console::style;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
use crate::imaging::upscale::{EsrganPipeline, Method as UpscaleMethod};
use crate::pipelines::lora::LoraSpec;
use crate::pipelines::scheduler::SchedulerKind;
use crate::pipelines::t2i::{self, GenRequest, LoadRequest, Pipeline, Variant};

#[derive(ClapArgs, Debug)]
pub struct ScenarioArgs {
    /// Path to the HJSON scenario file.
    pub file: PathBuf,

    /// Validate, print every task's planned prompts, but skip generation.
    /// Does NOT call the enhancer (no API cost).
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct ScenarioFile {
    // ---------- global generation parameters ----------
    model: Option<String>,
    device: Option<String>,
    size: Option<String>,
    aspect: Option<String>,
    base: Option<u32>,
    count: Option<u32>,
    steps: Option<usize>,
    guidance: Option<f64>,
    seed: Option<u64>,
    out: Option<PathBuf>,

    #[serde(default)]
    loras: Vec<String>,
    #[serde(rename = "lora-scale")]
    lora_scale: Option<f32>,

    scheduler: Option<String>,
    refine: Option<usize>,
    #[serde(rename = "refine-strength")]
    refine_strength: Option<f32>,

    /// If true (and model is SDXL/SDXL-Turbo) use the real SDXL refiner
    /// UNet for the last fraction of every task's schedule.
    #[serde(default)]
    refiner: bool,
    #[serde(rename = "refiner-frac")]
    refiner_frac: Option<f32>,

    // ---------- prompt-assembly fragments ----------
    #[serde(rename = "lora-header", default)]
    lora_header: String,
    #[serde(rename = "lora-footer", default)]
    lora_footer: String,
    #[serde(rename = "prompt-header", default)]
    prompt_header: String,
    #[serde(rename = "prompt-footer", default)]
    prompt_footer: String,

    // Accept both correct + the typo-spelling commonly seen in the wild.
    #[serde(alias = "enchancer")]
    enhancer: Option<String>,

    #[serde(default)]
    negative: String,

    // ---------- post-generate options ----------
    #[serde(default)]
    upscale: UpscaleConfig,

    // ---------- catalogs ----------
    #[serde(default)]
    scene: Vec<NamedPrompt>,
    #[serde(default)]
    weather: Vec<NamedPrompt>,
    #[serde(default)]
    tasks: Vec<TaskDef>,
}

/// Top-level `upscale: { ... }` section.
#[derive(Debug, Deserialize)]
struct UpscaleConfig {
    /// Enable the post-generate upscale pass.
    #[serde(default, alias = "enabled")]
    upscale: bool,
    /// Scale factor (2× by default).
    #[serde(default = "default_upscale_scale")]
    scale: f32,
    /// Filter: nearest | bilinear | bicubic | lanczos.
    #[serde(default = "default_upscale_method")]
    method: String,
}

impl Default for UpscaleConfig {
    fn default() -> Self {
        Self {
            upscale: false,
            scale: default_upscale_scale(),
            method: default_upscale_method(),
        }
    }
}

fn default_upscale_scale() -> f32 {
    2.0
}
fn default_upscale_method() -> String {
    "lanczos".to_string()
}

#[derive(Debug, Deserialize)]
struct NamedPrompt {
    name: String,
    prompt: String,
}

#[derive(Debug, Deserialize)]
struct TaskDef {
    name: String,
    scene: String,
    weather: String,
    prompt: String,
    /// Optional path to a style reference image. If set, every generated
    /// image for this task is also run through `stylize` (IP-Adapter) using
    /// this image as REF. Original + styled both land in the task directory.
    #[serde(default)]
    style: Option<PathBuf>,
    /// IP-Adapter strength for the style pass (0..1). Higher = more REF.
    #[serde(rename = "style-strength", default)]
    style_strength: Option<f32>,
}

pub async fn run(args: ScenarioArgs) -> Result<()> {
    let text = std::fs::read_to_string(&args.file)
        .with_context(|| format!("reading {}", args.file.display()))?;
    let s: ScenarioFile = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing HJSON {}", args.file.display()))?;

    // -------- validate structure --------
    if s.tasks.is_empty() {
        bail!("scenario has no `tasks` to run");
    }
    let scenes: HashMap<&str, &str> = s
        .scene
        .iter()
        .map(|p| (p.name.as_str(), p.prompt.as_str()))
        .collect();
    let weathers: HashMap<&str, &str> = s
        .weather
        .iter()
        .map(|p| (p.name.as_str(), p.prompt.as_str()))
        .collect();
    for t in &s.tasks {
        if !scenes.contains_key(t.scene.as_str()) {
            bail!("task {:?} references unknown scene {:?}", t.name, t.scene);
        }
        if !weathers.contains_key(t.weather.as_str()) {
            bail!("task {:?} references unknown weather {:?}", t.name, t.weather);
        }
    }

    let enhancer = s
        .enhancer
        .clone()
        .ok_or_else(|| anyhow!("scenario requires `enhancer` (deepseek | gemini)"))?;
    validate_enhancer_keys(&enhancer)?;

    // Parse the upscale method now so a bad string fails fast.
    let upscale_method: UpscaleMethod = s
        .upscale
        .method
        .parse()
        .with_context(|| format!("upscale.method = {:?}", s.upscale.method))?;

    // -------- resolve global parameters --------
    let model = s.model.clone().unwrap_or_else(|| "sd15".to_string());
    let device = crate::device::select(s.device.as_deref().unwrap_or("auto"))?;
    let base = s.base.unwrap_or(768);
    let count = s.count.unwrap_or(1);
    let steps = s.steps.unwrap_or(28);
    let guidance = s.guidance.unwrap_or(7.5);
    let seed = s.seed.unwrap_or(0);
    let out_root = s.out.clone().unwrap_or_else(|| PathBuf::from("./out"));
    let lora_scale = s.lora_scale.unwrap_or(1.0);
    let refine_strength = s.refine_strength.unwrap_or(0.3);
    let scheduler: SchedulerKind = match s.scheduler.as_deref() {
        Some(x) => x.parse().with_context(|| format!("scheduler {x:?}"))?,
        None => SchedulerKind::Default,
    };

    let size = match s.size.as_deref() {
        Some(s) => Some(s.parse::<Size>().with_context(|| format!("size {s:?}"))?),
        None => None,
    };
    let (width, height) = crate::imaging::sizes::resolve(size, s.aspect.as_deref(), base)?;

    let loras: Vec<LoraSpec> = s
        .loras
        .iter()
        .map(|x| x.parse::<LoraSpec>())
        .collect::<Result<Vec<_>>>()?;

    // -------- execution plan summary --------
    let total_images = (s.tasks.len() as u32) * count;
    println!(
        "{}  {} task(s) × {} image(s) = {} image(s) to generate",
        style("scenario").yellow().bold(),
        s.tasks.len(),
        count,
        total_images,
    );
    println!("  model:     {model}");
    println!("  size:      {width}×{height}");
    println!("  steps:     {steps}  guidance: {guidance}  scheduler: {scheduler:?}");
    println!("  out:       {}", out_root.display());
    println!("  enhancer:  {enhancer}");
    if !loras.is_empty() {
        println!("  loras:     {} (scale {lora_scale})", loras.len());
    }
    if let Some(r) = s.refine {
        println!("  refine:    {r} steps × strength {refine_strength}");
    }
    if s.refiner {
        let frac = s.refiner_frac.unwrap_or(0.8);
        println!(
            "  refiner:   on (switch at {:.0}% of schedule, SDXL only)",
            frac * 100.0
        );
    }
    if s.upscale.upscale {
        let shown = upscale_method.native_scale().unwrap_or(s.upscale.scale);
        println!(
            "  upscale:   {:.2}× {} (post-stylize if `style` is set, else original)",
            shown, s.upscale.method
        );
    }

    if !args.dry_run {
        std::fs::create_dir_all(&out_root)?;
    }

    // -------- preload the Real-ESRGAN model if used --------
    // Without this, every task would re-download + re-build the model.
    let esrgan: Option<EsrganPipeline> =
        if !args.dry_run && s.upscale.upscale && upscale_method.is_ml() {
            Some(EsrganPipeline::load(upscale_method, &device).await?)
        } else {
            None
        };

    // -------- load pipeline once (skipped for dry-run and for Flux) --------
    // For Flux scenarios we still call t2i::run per task; the SD Pipeline
    // optimization doesn't apply to Flux's separate pipeline module.
    let pipeline: Option<Pipeline> = if args.dry_run || Variant::detect(&model).is_flux() {
        None
    } else {
        Some(
            Pipeline::load(LoadRequest {
                model: model.clone(),
                device: device.clone(),
                loras: loras.clone(),
                lora_scale,
                use_refiner: s.refiner,
            })
            .await?,
        )
    };

    // -------- main loop --------
    let mut seed_offset: u64 = 0;
    for (idx, task) in s.tasks.iter().enumerate() {
        let scene_prompt = scenes[task.scene.as_str()];
        let weather_prompt = weathers[task.weather.as_str()];

        let pre_refine = join_parts(&[
            &s.prompt_header,
            scene_prompt,
            weather_prompt,
            &task.prompt,
            &s.prompt_footer,
        ]);

        crate::ui::progress::println(&format!(
            "\n{} [{}/{}] {} (scene={}, weather={})",
            style("▶").cyan().bold(),
            idx + 1,
            s.tasks.len(),
            style(&task.name).bold(),
            task.scene,
            task.weather,
        ));
        crate::ui::progress::println(&wrap_label("pre-enhance", &pre_refine));

        let enhanced = if args.dry_run {
            format!("(dry-run; {enhancer} not called)")
        } else {
            crate::prompt::enhance(&enhancer, &pre_refine)
                .await
                .with_context(|| format!("enhancing prompt for task {:?}", task.name))?
        };
        crate::ui::progress::println(&wrap_label("enhanced", &enhanced));

        let final_prompt = join_parts(&[&s.lora_header, &enhanced, &s.lora_footer]);
        crate::ui::progress::println(&wrap_label("final", &final_prompt));

        if args.dry_run {
            crate::ui::progress::println(&format!(
                "  {} would generate {} image(s) with seeds {}..{}",
                style("(dry-run)").dim(),
                count,
                seed + seed_offset,
                seed + seed_offset + count as u64 - 1,
            ));
            if let Some(style_ref) = &task.style {
                let strength = task.style_strength.unwrap_or(0.6);
                let exists = if style_ref.exists() { "ok" } else { "MISSING" };
                crate::ui::progress::println(&format!(
                    "  {} would stylize each with REF {} (strength {:.2}, {})",
                    style("(dry-run)").dim(),
                    style_ref.display(),
                    strength,
                    exists,
                ));
            }
            if s.upscale.upscale {
                let target = if task.style.is_some() {
                    "styled"
                } else {
                    "original"
                };
                crate::ui::progress::println(&format!(
                    "  {} would upscale the {} image(s) at {:.2}× ({:?})",
                    style("(dry-run)").dim(),
                    target,
                    s.upscale.scale,
                    upscale_method,
                ));
            }
            seed_offset += count as u64;
            continue;
        }

        let task_out = out_root.join(safe_name(&task.name));
        let refiner_frac_val = s.refiner_frac.unwrap_or(0.8);
        let gen_req = GenRequest {
            prompt: final_prompt,
            negative: s.negative.clone(),
            width,
            height,
            count,
            steps,
            guidance,
            seed: Some(seed + seed_offset),
            out_dir: task_out,
            scheduler,
            refine: s.refine,
            refine_strength,
            refiner_frac: if s.refiner { Some(refiner_frac_val) } else { None },
        };
        match &pipeline {
            // SD: reuse the loaded weights across tasks.
            Some(p) => p.generate(&gen_req)?,
            // Flux (or any case we couldn't preload): per-task t2i::run.
            None => {
                let req = t2i::Request {
                    prompt: gen_req.prompt.clone(),
                    negative: gen_req.negative.clone(),
                    model: model.clone(),
                    width: gen_req.width,
                    height: gen_req.height,
                    count: gen_req.count,
                    steps: gen_req.steps,
                    guidance: gen_req.guidance,
                    seed: gen_req.seed,
                    out_dir: gen_req.out_dir.clone(),
                    device: device.clone(),
                    loras: loras.clone(),
                    lora_scale,
                    scheduler: gen_req.scheduler,
                    refine: gen_req.refine,
                    refine_strength: gen_req.refine_strength,
                    use_refiner: s.refiner,
                    refiner_frac: refiner_frac_val,
                };
                t2i::run(req).await?;
            }
        }

        // Optional post-generate style pass.
        let style_attempted = task.style.is_some();
        if let Some(style_ref) = &task.style {
            if !style_ref.exists() {
                crate::ui::progress::println(&format!(
                    "  {} style reference not found: {} — skipping",
                    style("warn:").yellow().bold(),
                    style_ref.display(),
                ));
            } else {
                run_style_pass(
                    style_ref,
                    task.style_strength.unwrap_or(0.6),
                    &gen_req.out_dir,
                    seed + seed_offset,
                    count,
                    Variant::detect(&model).is_flux(),
                    &device,
                )
                .await;
            }
        }

        // Optional post-generate upscale pass.
        // Targets the stylized image when stylize was requested, otherwise the
        // original. Falls back to the original (with a warning) if the styled
        // file isn't on disk (e.g. stylize failed).
        if s.upscale.upscale {
            run_upscale_pass(
                &gen_req.out_dir,
                seed + seed_offset,
                count,
                Variant::detect(&model).is_flux(),
                style_attempted,
                s.upscale.scale,
                upscale_method,
                &device,
                esrgan.as_ref(),
            )
            .await;
        }

        seed_offset += count as u64;
    }

    println!(
        "\n{} {} task(s), {} image(s) → {}",
        style("✓ done").green().bold(),
        s.tasks.len(),
        total_images,
        out_root.display()
    );
    Ok(())
}

fn validate_enhancer_keys(enhancer: &str) -> Result<()> {
    let cfg = crate::config::Config::load()?;
    match enhancer.to_lowercase().as_str() {
        "deepseek" => {
            if cfg.deepseek_api_key.is_none() {
                bail!(
                    "scenario uses `enhancer: deepseek` but DEEPSEEK_API_KEY \
                     is not set in the environment or ~/.config/plakat/config.toml"
                );
            }
        }
        "gemini" => {
            if cfg.gemini_api_key.is_none() {
                bail!(
                    "scenario uses `enhancer: gemini` but GEMINI_API_KEY \
                     is not set in the environment or ~/.config/plakat/config.toml"
                );
            }
        }
        other => bail!("unknown enhancer {other:?} (expected: deepseek | gemini)"),
    }
    Ok(())
}

/// Strip leading/trailing whitespace and commas from each part so users can
/// write fragments like `"fantasy art,"` or `, masterpiece` without producing
/// double-commas in the final prompt.
fn join_parts(parts: &[&str]) -> String {
    parts
        .iter()
        .map(|s| s.trim().trim_matches(|c: char| c == ',' || c.is_whitespace()))
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Run the IP-Adapter stylize pass on every image produced by a task. The
/// original `plakat-<seed>.png` (or `plakat-flux-<seed>.png`) is preserved;
/// the styled version is written next to it as `…-styled.png`.
///
/// Failures on individual images are logged but don't abort the scenario —
/// you keep the original even if stylization fails (e.g. SDXL output with
/// dims stylize can't handle).
async fn run_style_pass(
    ref_path: &std::path::Path,
    strength: f32,
    out_dir: &std::path::Path,
    seed_start: u64,
    count: u32,
    is_flux: bool,
    device: &candle_core::Device,
) {
    let prefix = if is_flux { "plakat-flux" } else { "plakat" };
    for i in 0..count {
        let seed = (seed_start + i as u64) & (u32::MAX as u64);
        let in_path = out_dir.join(format!("{prefix}-{seed}.png"));
        let out_path = out_dir.join(format!("{prefix}-{seed}-styled.png"));

        if !in_path.exists() {
            crate::ui::progress::println(&format!(
                "  {} expected {} not on disk — stylize skipped",
                style("warn:").yellow().bold(),
                in_path.display(),
            ));
            continue;
        }

        crate::ui::progress::println(&format!(
            "  {} {} (REF {}, strength {:.2})",
            style("stylize").cyan().bold(),
            in_path.display(),
            ref_path.display(),
            strength,
        ));

        let req = crate::pipelines::stylize::Request {
            input: in_path,
            reference: ref_path.to_path_buf(),
            out: out_path,
            strength,
            model: "sd15".to_string(),
            steps: 30,
            seed: Some(seed),
            device: device.clone(),
        };
        if let Err(e) = crate::pipelines::stylize::run(req).await {
            crate::ui::progress::println(&format!(
                "  {} stylize failed: {e}",
                style("warn:").yellow().bold(),
            ));
        }
    }
}

/// Run the classical upscaler on every image produced by a task.
///
/// Target file:
///   - If `style_attempted` and `<task>/plakat-<seed>-styled.png` exists → upscale it.
///   - Else → upscale `<task>/plakat-<seed>.png`.
///
/// Output is written next to the source with `-upscaled` appended.
async fn run_upscale_pass(
    out_dir: &std::path::Path,
    seed_start: u64,
    count: u32,
    is_flux: bool,
    style_attempted: bool,
    scale: f32,
    method: UpscaleMethod,
    device: &candle_core::Device,
    esrgan: Option<&EsrganPipeline>,
) {
    let prefix = if is_flux { "plakat-flux" } else { "plakat" };
    for i in 0..count {
        let seed = (seed_start + i as u64) & (u32::MAX as u64);
        let styled = out_dir.join(format!("{prefix}-{seed}-styled.png"));
        let orig = out_dir.join(format!("{prefix}-{seed}.png"));

        // Pick source per the rule.
        let (source, suffix) = if style_attempted && styled.exists() {
            (styled, "styled-upscaled")
        } else {
            if style_attempted {
                crate::ui::progress::println(&format!(
                    "  {} styled image missing; upscaling original instead",
                    style("warn:").yellow().bold(),
                ));
            }
            (orig, "upscaled")
        };
        if !source.exists() {
            crate::ui::progress::println(&format!(
                "  {} {} not on disk — upscale skipped",
                style("warn:").yellow().bold(),
                source.display(),
            ));
            continue;
        }
        let dest = out_dir.join(format!("{prefix}-{seed}-{suffix}.png"));

        let result = match (method.is_ml(), esrgan) {
            // Cached ESRGAN model — no per-image build cost.
            (true, Some(p)) => p.upscale_file(&source, &dest),
            // ML method but no preloaded pipeline (shouldn't happen in normal
            // scenario flow; fall back to the one-shot path).
            (true, None) => {
                crate::imaging::upscale::ml_upscale(&source, &dest, method, device).await
            }
            (false, _) => crate::imaging::upscale::upscale(&source, &dest, scale, method),
        };
        match result {
            Ok((w, h, nw, nh)) => {
                let shown = method.native_scale().unwrap_or(scale);
                crate::ui::progress::println(&format!(
                    "  {} {} ({}×{} → {}×{}, {:.2}×, {:?})",
                    style("upscale").cyan().bold(),
                    dest.display(),
                    w,
                    h,
                    nw,
                    nh,
                    shown,
                    method,
                ));
            }
            Err(e) => crate::ui::progress::println(&format!(
                "  {} upscale failed for {}: {e}",
                style("warn:").yellow().bold(),
                source.display(),
            )),
        }
    }
}

/// Word-wrap `text` under a labeled line. Continuation lines are indented
/// to line up after the `"  <label>: "` prefix so the result reads as one
/// logical entry. Existing newlines in `text` are treated as whitespace
/// (HJSON multi-line strings carry editor-formatting newlines that aren't
/// semantically meaningful to SD).
///
/// Format:
///     "  pre-enhance: first line of wrapped text up to terminal width"
///     "               second line continues at the same column"
///     "               third line ..."
fn wrap_label(label: &str, text: &str) -> String {
    let cols = terminal_width();
    let prefix_len = 2 + label.chars().count() + 2; // "  " + label + ": "
    let avail = cols.saturating_sub(prefix_len).max(40);
    let indent = " ".repeat(prefix_len);

    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.chars().count() + 1 + word.chars().count() <= avail {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }

    let label_styled = style(label).dim();
    let mut out = format!("  {label_styled}: {}", lines[0]);
    for line in &lines[1..] {
        out.push('\n');
        out.push_str(&indent);
        out.push_str(line);
    }
    out
}

fn terminal_width() -> usize {
    console::Term::stdout()
        .size_checked()
        .map(|(_, c)| c as usize)
        .unwrap_or(100)
}

/// Sanitize a task name for use as a directory.
fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}
