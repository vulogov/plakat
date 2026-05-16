//! `scenario` — batch-generate images from an HJSON file that mixes scenes,
//! weather, and per-task prompts. See README for the schema and an example.

use anyhow::{Context, Result, anyhow, bail};
use clap::Args as ClapArgs;
use console::style;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::imaging::sizes::Size;
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

    // ---------- catalogs ----------
    #[serde(default)]
    scene: Vec<NamedPrompt>,
    #[serde(default)]
    weather: Vec<NamedPrompt>,
    #[serde(default)]
    tasks: Vec<TaskDef>,
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

    if !args.dry_run {
        std::fs::create_dir_all(&out_root)?;
    }

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
        crate::ui::progress::println(&format!(
            "  {}: {}",
            style("pre-enhance").dim(),
            short(&pre_refine, 160)
        ));

        let enhanced = if args.dry_run {
            format!("(dry-run; {enhancer} not called)")
        } else {
            crate::prompt::enhance(&enhancer, &pre_refine)
                .await
                .with_context(|| format!("enhancing prompt for task {:?}", task.name))?
        };
        crate::ui::progress::println(&format!(
            "  {}: {}",
            style("enhanced").dim(),
            short(&enhanced, 160)
        ));

        let final_prompt = join_parts(&[&s.lora_header, &enhanced, &s.lora_footer]);
        crate::ui::progress::println(&format!(
            "  {}: {}",
            style("final").dim(),
            short(&final_prompt, 160)
        ));

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
            seed_offset += count as u64;
            continue;
        }

        let task_out = out_root.join(safe_name(&task.name));
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
                };
                t2i::run(req).await?;
            }
        }

        // Optional post-generate style pass.
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

fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let taken: String = s.chars().take(n).collect();
        format!("{taken}…")
    }
}

/// Sanitize a task name for use as a directory.
fn safe_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}
