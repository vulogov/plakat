//! `plakat style` subcommand — detect art style from a reference photo.
//!
//! Spike scope: only the `detect` operation. The `list` / `show` /
//! `probe` operations sketched in the design land once the catalog is
//! filled in with real LoRA mappings and trigger phrases.

use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device};
use clap::{Args as ClapArgs, Subcommand, ValueEnum};
use console::style;

use crate::pipelines::ip_adapter::{ImageEncoder, IPA_REPO};
use crate::pipelines::lora::{LoraSource, LoraSpec};
use crate::style::{detect_style, encode_reference_photo, BaseModel, LoraEntry, StyleCatalog};

#[derive(ClapArgs, Debug)]
pub struct StyleArgs {
    #[command(subcommand)]
    pub op: StyleOp,

    /// Override the bundled style catalog directory.
    #[arg(long, value_name = "DIR", global = true)]
    pub catalog: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum StyleOp {
    /// Detect art style from a photo. Prints top-K matches; doesn't generate.
    Detect(DetectArgs),
    /// List every style in the catalog with one-line descriptions.
    List(ListArgs),
    /// Show full info for one style: description, exemplar count, LoRAs, triggers.
    Show(ShowArgs),
    /// Probe HuggingFace: HEAD-request every LoRA in the catalog and
    /// report which still resolve. Network-dependent. Suitable for
    /// periodic CI to catch upstream repo deletions or renames before
    /// users hit them.
    Probe(ProbeArgs),
    /// Scan a directory of images and emit a starter catalog HJSON.
    /// One subdirectory per style; the subdir name becomes the style id
    /// (slugified). Doesn't build the catalog itself — the next step is
    /// `cargo run --bin build_catalog -- --sources <HJSON> --out <DIR>`.
    Init(InitArgs),
    /// Train a style LoRA from a folder of images (creation, not
    /// detection). Phase 1: SD3.5 only. Writes a diffusers-PEFT
    /// `.safetensors` loadable via `--lora`.
    Train(TrainArgs),
}

#[derive(ClapArgs, Debug)]
pub struct TrainArgs {
    /// Folder of style images to train on (3+ recommended; jpg/png).
    #[arg(long, value_name = "DIR")]
    pub from_dir: PathBuf,
    /// Base model: `sd15` (SD 1.5) or `sd35` (SD3.5 Medium). sdxl in progress.
    #[arg(long, default_value = "sd35")]
    pub base: String,
    /// Trigger phrase woven into training — include it in your prompts
    /// at inference to invoke the style.
    #[arg(long, default_value = "in this style")]
    pub trigger: String,
    /// Output `.safetensors` path.
    #[arg(long, value_name = "FILE")]
    pub out: PathBuf,
    /// Training steps.
    #[arg(long, default_value_t = 90)]
    pub steps: usize,
    /// LoRA rank.
    #[arg(long, default_value_t = 16)]
    pub rank: usize,
    /// Training resolution (256 fits 24 GB; higher needs more memory).
    #[arg(long, default_value_t = 256)]
    pub size: u32,
    /// Learning rate.
    #[arg(long, default_value_t = 1.5e-4)]
    pub lr: f64,
}

#[derive(ClapArgs, Debug)]
pub struct DetectArgs {
    /// Reference photo to detect style from.
    pub photo: PathBuf,

    /// Number of top matches to show.
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,
}

#[derive(ClapArgs, Debug)]
pub struct ListArgs {
    /// Filter: only styles with LoRA mappings for the given base model.
    /// Without this flag, every catalog style is listed (including
    /// detection-only styles with empty `models`).
    #[arg(long, value_enum)]
    pub base: Option<BaseFilter>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,
}

#[derive(ClapArgs, Debug)]
pub struct ShowArgs {
    /// Style id (run `plakat style list` for available ids).
    pub id: String,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,
}

#[derive(ClapArgs, Debug)]
pub struct InitArgs {
    /// Directory containing one subdirectory per style. Each subdirectory's
    /// .jpg / .jpeg / .png files become that style's exemplars; the
    /// subdir name becomes the style id (slugified — lowercased,
    /// non-alphanumerics replaced with `_`).
    ///
    /// `holdout/` and hidden directories (starting with `.`) are skipped.
    #[arg(long, value_name = "DIR")]
    pub from_dir: PathBuf,

    /// Output HJSON path. Defaults to `<from-dir>/catalog.hjson`.
    /// Exemplar paths in the emitted HJSON are resolved relative to
    /// this file's directory.
    #[arg(long, value_name = "PATH")]
    pub out: Option<PathBuf>,

    /// Overwrite the output file if it already exists.
    #[arg(long)]
    pub force: bool,
}

#[derive(ClapArgs, Debug)]
pub struct ProbeArgs {
    /// Probe only the LoRAs for this style id. Default: every style.
    #[arg(long, value_name = "ID")]
    pub id: Option<String>,

    /// Output format.
    #[arg(long, value_enum, default_value_t = OutFormat::Text)]
    pub format: OutFormat,

    /// Network timeout per request, in seconds.
    #[arg(long, default_value_t = 10)]
    pub timeout: u64,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum BaseFilter {
    Sd15,
    Sdxl,
    Flux,
}

impl BaseFilter {
    fn to_base(self) -> BaseModel {
        match self {
            Self::Sd15 => BaseModel::Sd15,
            Self::Sdxl => BaseModel::Sdxl,
            Self::Flux => BaseModel::Flux,
        }
    }
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum OutFormat {
    Text,
    Json,
}

pub async fn run(args: StyleArgs, device: Device) -> Result<()> {
    let catalog_dir = args
        .catalog
        .clone()
        .unwrap_or_else(|| PathBuf::from("assets/style_catalog"));

    match args.op {
        StyleOp::Detect(a) => detect_cmd(a, &catalog_dir, device).await,
        StyleOp::List(a) => list_cmd(a, &catalog_dir, &device),
        StyleOp::Show(a) => show_cmd(a, &catalog_dir, &device),
        StyleOp::Probe(a) => probe_cmd(a, &catalog_dir, &device).await,
        StyleOp::Init(a) => init_cmd(a),
        StyleOp::Train(a) => train_cmd(a, device).await,
    }
}

/// Train a style LoRA from a folder of images (Phase 1: SD3.5).
async fn train_cmd(args: TrainArgs, device: Device) -> Result<()> {
    let mut images: Vec<PathBuf> = std::fs::read_dir(&args.from_dir)
        .with_context(|| format!("reading {}", args.from_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|x| x.to_str()),
                Some("jpg") | Some("jpeg") | Some("png")
            )
        })
        .collect();
    images.sort();
    if images.len() < 3 {
        anyhow::bail!(
            "style train: need 3+ images in {}, found {}",
            args.from_dir.display(),
            images.len()
        );
    }
    println!(
        "Training {} style LoRA on {} images ({} steps, rank {}, {}²) → {}",
        args.base,
        images.len(),
        args.steps,
        args.rank,
        args.size,
        args.out.display()
    );

    match args.base.as_str() {
        "sd15" | "sd1.5" | "stable-diffusion-v1-5" => {
            use crate::pipelines::sd_train::trainer;
            trainer::train_style_lora_sd(trainer::SdStyleTrainRequest {
                model: "sd15".to_string(),
                device,
                images,
                trigger: args.trigger,
                rank: args.rank,
                steps: args.steps,
                lr: args.lr,
                size: args.size,
                out: args.out,
            })
            .await
        }
        "sd35" | "sd35-medium" | "sd3.5-medium" => {
            use crate::pipelines::sd3;
            sd3::train_style_lora(sd3::StyleTrainRequest {
                variant: sd3::Variant::Sd35Medium,
                repo: "stabilityai/stable-diffusion-3.5-medium".to_string(),
                device,
                images,
                trigger: args.trigger,
                rank: args.rank,
                steps: args.steps,
                lr: args.lr,
                size: args.size,
                out: args.out,
            })
            .await
        }
        other => anyhow::bail!(
            "style train: base '{other}' not supported — use sd15 or sd35 \
             (sdxl in progress)"
        ),
    }
}

async fn detect_cmd(args: DetectArgs, catalog_dir: &Path, device: Device) -> Result<()> {
    let catalog = StyleCatalog::load(catalog_dir, &device)?;
    catalog.assert_encoder("clip-h-laion2b")?;

    let weights =
        crate::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors").await?;
    let encoder = ImageEncoder::load(&weights, &device, DType::F32)?;

    let emb = encode_reference_photo(&encoder, &args.photo, &device)?;
    let result = detect_style(&catalog, &emb, args.top_k)?;

    match args.format {
        OutFormat::Text => print_text(&result),
        OutFormat::Json => print_json(&result)?,
    }
    Ok(())
}

fn print_text(result: &crate::style::DetectionResult) {
    let picked = result.picked.as_deref();

    match (picked, result.ambiguous) {
        (Some(id), false) => {
            let top = &result.top[0];
            println!(
                "Detected: {} ({:.4}) {}",
                style(id).bold().cyan(),
                top.score,
                style("[picked]").green()
            );
        }
        (Some(id), true) => {
            let top = &result.top[0];
            println!(
                "Detected: {} ({:.4}) {}",
                style(id).bold().yellow(),
                top.score,
                style("[ambiguous]").yellow()
            );
            if let Some(runner_up) = result.top.get(1) {
                println!(
                    "Runner-up: {} ({:.4})",
                    style(&runner_up.style_id).bold(),
                    runner_up.score
                );
            }
        }
        (None, _) => {
            println!(
                "{}",
                style("Detected: (none above min_confidence)").red().bold()
            );
            if let Some(top) = result.top.first() {
                println!(
                    "Closest: {} ({:.4})",
                    style(&top.style_id).bold(),
                    top.score
                );
            }
        }
    }

    println!();
    println!("Top {}:", result.top.len());
    for (i, m) in result.top.iter().enumerate() {
        let marker = if Some(m.style_id.as_str()) == picked {
            style("✓ picked").green().to_string()
        } else {
            String::new()
        };
        println!(
            "  {}. {:<20} {:.4}  {}",
            i + 1,
            m.style_id,
            m.score,
            marker
        );
    }
}

fn print_json(result: &crate::style::DetectionResult) -> Result<()> {
    let value = serde_json::json!({
        "picked": result.picked,
        "ambiguous": result.ambiguous,
        "top": result.top.iter().map(|m| serde_json::json!({
            "style_id": m.style_id,
            "display_name": m.display_name,
            "score": m.score,
        })).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

// =========================================================================
// list
// =========================================================================

fn list_cmd(args: ListArgs, catalog_dir: &Path, device: &Device) -> Result<()> {
    let catalog = StyleCatalog::load(catalog_dir, device)?;

    let entries: Vec<ListEntry> = catalog
        .order
        .iter()
        .filter_map(|id| {
            let style = catalog.styles.get(id)?;
            if let Some(bf) = args.base {
                if !style.models.contains_key(&bf.to_base()) {
                    return None;
                }
            }
            let mut bases: Vec<&'static str> = style.models.keys().map(|b| b.slug()).collect();
            bases.sort();
            Some(ListEntry {
                id: style.id.clone(),
                display_name: style.display_name.clone(),
                description: style.description.clone(),
                bases,
                exemplar_count: style
                    .exemplars
                    .dim(0)
                    .unwrap_or(0),
            })
        })
        .collect();

    match args.format {
        OutFormat::Text => print_list_text(&entries, args.base),
        OutFormat::Json => print_list_json(&entries)?,
    }
    Ok(())
}

struct ListEntry {
    id: String,
    display_name: String,
    description: String,
    bases: Vec<&'static str>,
    exemplar_count: usize,
}

fn print_list_text(entries: &[ListEntry], filter: Option<BaseFilter>) {
    if entries.is_empty() {
        match filter {
            Some(f) => println!(
                "No styles have LoRA mappings for {:?}",
                f
            ),
            None => println!("Catalog is empty."),
        }
        return;
    }

    // Column widths.
    let id_w = entries.iter().map(|e| e.id.len()).max().unwrap_or(2).max(2);
    let name_w = entries.iter().map(|e| e.display_name.len()).max().unwrap_or(12).max(12);

    println!(
        "{:<id_w$}  {:<name_w$}  {:>3}  {:<10}  {}",
        "ID", "Display name", "Ex", "Bases", "Description",
        id_w = id_w,
        name_w = name_w
    );
    println!(
        "{}  {}  {}  {}  {}",
        "─".repeat(id_w),
        "─".repeat(name_w),
        "───",
        "─".repeat(10),
        "─".repeat(20),
    );
    for e in entries {
        let bases = if e.bases.is_empty() {
            String::from("(none)")
        } else {
            e.bases.join(",")
        };
        println!(
            "{:<id_w$}  {:<name_w$}  {:>3}  {:<10}  {}",
            e.id,
            e.display_name,
            e.exemplar_count,
            bases,
            e.description,
            id_w = id_w,
            name_w = name_w
        );
    }
    println!();
    println!("{} styles{}.", entries.len(), match filter {
        Some(f) => format!(" with LoRAs for {:?}", f).to_lowercase(),
        None => String::new(),
    });
}

fn print_list_json(entries: &[ListEntry]) -> Result<()> {
    let value = serde_json::json!(
        entries.iter().map(|e| serde_json::json!({
            "id": e.id,
            "display_name": e.display_name,
            "description": e.description,
            "bases": e.bases,
            "exemplar_count": e.exemplar_count,
        })).collect::<Vec<_>>()
    );
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

// =========================================================================
// show
// =========================================================================

fn show_cmd(args: ShowArgs, catalog_dir: &Path, device: &Device) -> Result<()> {
    let catalog = StyleCatalog::load(catalog_dir, device)?;
    let s = catalog
        .styles
        .get(&args.id)
        .ok_or_else(|| anyhow!("unknown style id '{}' (try `plakat style list`)", args.id))?;

    match args.format {
        OutFormat::Text => print_show_text(s),
        OutFormat::Json => print_show_json(s)?,
    }
    Ok(())
}

fn print_show_text(s: &crate::style::LoadedStyle) {
    println!("{}:              {}", style("ID").bold(), s.id);
    println!("{}:    {}", style("Display name").bold(), s.display_name);
    println!("{}:     {}", style("Description").bold(), s.description);
    println!(
        "{}:      {} in catalog",
        style("Exemplars").bold(),
        s.exemplars.dim(0).unwrap_or(0)
    );
    println!();

    if s.models.is_empty() {
        println!(
            "{} this style is detection-only — no LoRAs / triggers configured.",
            style("Note:").yellow().bold()
        );
        return;
    }

    println!("{}:", style("Models").bold());
    let mut bases: Vec<_> = s.models.iter().collect();
    bases.sort_by_key(|(b, _)| b.slug());
    for (base, entry) in bases {
        println!("  {}:", style(base.slug()).bold().cyan());
        if entry.loras.is_empty() {
            println!("    loras:     (none — trigger only)");
        } else {
            println!("    loras:");
            for l in &entry.loras {
                let rev = l.revision().map(|r| format!(" (revision: {})", r)).unwrap_or_default();
                println!("      - {}{}", l.spec(), rev);
            }
        }
        if !entry.trigger.is_empty() {
            println!("    trigger:   \"{}\"", entry.trigger);
        }
        if !entry.negative_extras.is_empty() {
            println!("    negative+: \"{}\"", entry.negative_extras);
        }
    }
}

fn print_show_json(s: &crate::style::LoadedStyle) -> Result<()> {
    let models_json: serde_json::Map<String, serde_json::Value> = s
        .models
        .iter()
        .map(|(base, entry)| {
            let loras: Vec<serde_json::Value> = entry
                .loras
                .iter()
                .map(|l| {
                    serde_json::json!({
                        "spec": l.spec(),
                        "revision": l.revision(),
                    })
                })
                .collect();
            (
                base.slug().to_string(),
                serde_json::json!({
                    "loras": loras,
                    "trigger": entry.trigger,
                    "negative_extras": entry.negative_extras,
                }),
            )
        })
        .collect();

    let value = serde_json::json!({
        "id": s.id,
        "display_name": s.display_name,
        "description": s.description,
        "exemplar_count": s.exemplars.dim(0).unwrap_or(0),
        "models": models_json,
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

// =========================================================================
// probe
// =========================================================================

#[derive(Debug)]
enum ProbeOutcome {
    /// HEAD returned 2xx — file or repo exists.
    Ok { url: String },
    /// Local file exists on disk.
    LocalOk { path: PathBuf },
    /// HEAD returned a non-2xx; flagged as broken.
    NotFound { url: String, status: u16 },
    /// Local file is missing.
    LocalMissing { path: PathBuf },
    /// HF request failed with a network/transport error.
    NetworkError { url: String, error: String },
    /// Spec couldn't be parsed.
    BadSpec { spec: String, error: String },
}

impl ProbeOutcome {
    fn is_ok(&self) -> bool {
        matches!(self, Self::Ok { .. } | Self::LocalOk { .. })
    }
}

#[derive(Debug)]
struct ProbeRow {
    style_id: String,
    base: BaseModel,
    spec: String,
    revision: Option<String>,
    outcome: ProbeOutcome,
}

async fn probe_cmd(args: ProbeArgs, catalog_dir: &Path, device: &Device) -> Result<()> {
    let catalog = StyleCatalog::load(catalog_dir, device)?;

    // Build the work list.
    let mut work: Vec<(String, BaseModel, LoraEntry)> = Vec::new();
    for sid in &catalog.order {
        if let Some(filter) = args.id.as_deref() {
            if sid != filter {
                continue;
            }
        }
        let style = &catalog.styles[sid];
        for (base, entry) in &style.models {
            for lora in &entry.loras {
                work.push((sid.clone(), *base, lora.clone()));
            }
        }
    }

    if args.id.is_some() && work.is_empty() && !catalog.styles.contains_key(args.id.as_deref().unwrap()) {
        return Err(anyhow!(
            "unknown style id '{}' (try `plakat style list`)",
            args.id.as_deref().unwrap()
        ));
    }

    if !matches!(args.format, OutFormat::Json) {
        let n_styles = if args.id.is_some() {
            1
        } else {
            work.iter().map(|(s, _, _)| s.as_str()).collect::<std::collections::HashSet<_>>().len()
        };
        println!("Probing {n_styles} style(s), {} LoRA(s) total…", work.len());
        println!();
    }

    let client = reqwest::Client::builder()
        .user_agent("plakat-style-probe/0.1")
        .timeout(std::time::Duration::from_secs(args.timeout))
        .build()?;

    let mut rows: Vec<ProbeRow> = Vec::with_capacity(work.len());
    for (style_id, base, lora) in work {
        let outcome = probe_one(&client, &lora).await;
        rows.push(ProbeRow {
            style_id,
            base,
            spec: lora.spec().to_string(),
            revision: lora.revision().map(str::to_owned),
            outcome,
        });
    }

    match args.format {
        OutFormat::Text => print_probe_text(&rows),
        OutFormat::Json => print_probe_json(&rows)?,
    }

    let failures = rows.iter().filter(|r| !r.outcome.is_ok()).count();
    if failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}

async fn probe_one(client: &reqwest::Client, entry: &LoraEntry) -> ProbeOutcome {
    let spec_str = entry.spec();
    let spec: LoraSpec = match LoraSpec::from_str(spec_str) {
        Ok(s) => s,
        Err(e) => {
            return ProbeOutcome::BadSpec {
                spec: spec_str.to_owned(),
                error: e.to_string(),
            };
        }
    };

    let revision = entry.revision().unwrap_or("main");

    match spec.source {
        LoraSource::Local(path) => {
            if path.exists() {
                ProbeOutcome::LocalOk { path }
            } else {
                ProbeOutcome::LocalMissing { path }
            }
        }
        LoraSource::Hub { repo, file, .. } => {
            let repo_resolved = crate::hf::resolve_alias(&repo).to_string();
            let url = match file {
                Some(f) => format!(
                    "https://huggingface.co/{}/resolve/{}/{}",
                    repo_resolved, revision, f
                ),
                None => {
                    // No explicit file — verify the repo itself exists via the
                    // models API. Discovery happens at download time.
                    format!("https://huggingface.co/api/models/{}", repo_resolved)
                }
            };
            match client.head(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        ProbeOutcome::Ok { url }
                    } else {
                        ProbeOutcome::NotFound {
                            url,
                            status: status.as_u16(),
                        }
                    }
                }
                Err(e) => ProbeOutcome::NetworkError {
                    url,
                    error: e.to_string(),
                },
            }
        }
        // v0.17 phase G: Civitai LoRA specs probe via the
        // `civitai.com/api/v1/models/<id>` endpoint. The exact
        // URL is informational only — the probe doesn't follow
        // the API redirect to the file CDN, just reports
        // reachability.
        LoraSource::Civitai { id_kind, .. } => {
            let url = match id_kind {
                crate::pipelines::lora::CivitaiIdKind::Model(m) => {
                    format!("https://civitai.com/api/v1/models/{m}")
                }
                crate::pipelines::lora::CivitaiIdKind::Version(v) => {
                    format!("https://civitai.com/api/v1/model-versions/{v}")
                }
            };
            match client.head(&url).send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        ProbeOutcome::Ok { url }
                    } else {
                        ProbeOutcome::NotFound {
                            url,
                            status: status.as_u16(),
                        }
                    }
                }
                Err(e) => ProbeOutcome::NetworkError {
                    url,
                    error: e.to_string(),
                },
            }
        }
    }
}

fn print_probe_text(rows: &[ProbeRow]) {
    for r in rows {
        let (marker, marker_color) = match &r.outcome {
            ProbeOutcome::Ok { .. } | ProbeOutcome::LocalOk { .. } => ("✓", "green"),
            ProbeOutcome::NotFound { .. } | ProbeOutcome::LocalMissing { .. } => ("✗", "red"),
            ProbeOutcome::NetworkError { .. } => ("⚠", "yellow"),
            ProbeOutcome::BadSpec { .. } => ("✗", "red"),
        };
        let marker = match marker_color {
            "green" => style(marker).green().to_string(),
            "red" => style(marker).red().to_string(),
            "yellow" => style(marker).yellow().to_string(),
            _ => marker.to_string(),
        };

        let detail = match &r.outcome {
            ProbeOutcome::Ok { .. } => String::new(),
            ProbeOutcome::LocalOk { .. } => String::from(" (local)"),
            ProbeOutcome::NotFound { status, .. } => format!(" HTTP {}", status),
            ProbeOutcome::LocalMissing { path } => format!(" missing: {}", path.display()),
            ProbeOutcome::NetworkError { error, .. } => format!(" network error: {}", error),
            ProbeOutcome::BadSpec { error, .. } => format!(" bad spec: {}", error),
        };

        let rev_note = match &r.revision {
            Some(rev) if rev != "main" => format!(" @ {}", &rev[..rev.len().min(8)]),
            _ => String::new(),
        };

        println!(
            "  {} {} ({}{}){}",
            marker,
            r.spec,
            r.base.slug(),
            rev_note,
            detail,
        );
    }

    println!();
    let total = rows.len();
    let failures = rows.iter().filter(|r| !r.outcome.is_ok()).count();
    if failures == 0 {
        println!("{} all {} LoRA(s) resolved", style("✓").green().bold(), total);
    } else {
        println!(
            "{} {} / {} LoRA(s) failed to resolve",
            style("✗").red().bold(),
            failures,
            total
        );
    }
}

fn print_probe_json(rows: &[ProbeRow]) -> Result<()> {
    let rows_json: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            let (status, detail) = match &r.outcome {
                ProbeOutcome::Ok { url } => ("ok", serde_json::json!({ "url": url })),
                ProbeOutcome::LocalOk { path } => {
                    ("local_ok", serde_json::json!({ "path": path }))
                }
                ProbeOutcome::NotFound { url, status } => (
                    "not_found",
                    serde_json::json!({ "url": url, "http_status": status }),
                ),
                ProbeOutcome::LocalMissing { path } => {
                    ("local_missing", serde_json::json!({ "path": path }))
                }
                ProbeOutcome::NetworkError { url, error } => (
                    "network_error",
                    serde_json::json!({ "url": url, "error": error }),
                ),
                ProbeOutcome::BadSpec { spec, error } => (
                    "bad_spec",
                    serde_json::json!({ "spec": spec, "error": error }),
                ),
            };
            serde_json::json!({
                "style_id": r.style_id,
                "base": r.base.slug(),
                "spec": r.spec,
                "revision": r.revision,
                "status": status,
                "detail": detail,
            })
        })
        .collect();

    let total = rows.len();
    let failures = rows.iter().filter(|r| !r.outcome.is_ok()).count();
    let value = serde_json::json!({
        "probed": total,
        "failures": failures,
        "results": rows_json,
    });
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

// =========================================================================
// init — scan a corpus directory and emit a starter catalog HJSON
// =========================================================================

fn init_cmd(args: InitArgs) -> Result<()> {
    if !args.from_dir.is_dir() {
        return Err(anyhow!(
            "--from-dir is not a directory: {}",
            args.from_dir.display()
        ));
    }

    let out_path = args
        .out
        .clone()
        .unwrap_or_else(|| args.from_dir.join("catalog.hjson"));

    if out_path.exists() && !args.force {
        return Err(anyhow!(
            "{} already exists. Pass --force to overwrite, or --out <PATH> to write elsewhere.",
            out_path.display()
        ));
    }

    let out_parent = out_path.parent().unwrap_or(Path::new("."));
    let scanned = scan_corpus(&args.from_dir)?;

    if scanned.is_empty() {
        return Err(anyhow!(
            "no usable style subdirectories found under {}. Expected layout: \
             <from-dir>/<style-name>/*.jpg (and similar for png). \
             `holdout/` and hidden directories are skipped.",
            args.from_dir.display()
        ));
    }

    // ----- Emit summary BEFORE writing — so the user sees the scan
    // result whether or not the write succeeds.
    println!("==> scanning {}", args.from_dir.display());
    for s in &scanned {
        let warn = if s.exemplars.len() < 3 {
            format!("  {}", style(format!("⚠ <3 exemplars")).yellow())
        } else {
            String::new()
        };
        println!(
            "    {:<20} {:>3} images{}",
            s.id,
            s.exemplars.len(),
            warn
        );
    }
    println!();

    let hjson = render_hjson(&scanned, out_parent);
    std::fs::write(&out_path, hjson)
        .with_context(|| format!("writing {}", out_path.display()))?;

    println!(
        "{} wrote {} with {} style(s)",
        style("✓").green().bold(),
        out_path.display(),
        scanned.len()
    );
    println!();
    println!("Next steps:");
    println!(
        "  1. Edit {} to fill in descriptions, triggers, and LoRA pins.",
        out_path.display()
    );
    println!("  2. Build the catalog:");
    println!(
        "       cargo run --release --bin build_catalog -- \\\n         --sources {} \\\n         --out     {}/built",
        out_path.display(),
        out_parent.display()
    );
    println!("  3. Use it:");
    println!(
        "       plakat style detect <PHOTO> --catalog {}/built",
        out_parent.display()
    );

    Ok(())
}

struct ScannedStyle {
    id: String,
    display_name: String,
    /// Paths RELATIVE TO `out_parent`. The HJSON's exemplars: [...] list
    /// resolves paths from the HJSON's directory, so we strip the
    /// common prefix here.
    exemplars: Vec<PathBuf>,
}

fn scan_corpus(from_dir: &Path) -> Result<Vec<ScannedStyle>> {
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(from_dir)
        .with_context(|| format!("reading {}", from_dir.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        // Skip hidden + reserved.
        if name.starts_with('.') || name == "holdout" {
            continue;
        }

        let mut images: Vec<PathBuf> = std::fs::read_dir(&path)
            .with_context(|| format!("reading {}", path.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                let ext = p
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_ascii_lowercase());
                matches!(ext.as_deref(), Some("jpg" | "jpeg" | "png"))
            })
            .collect();
        images.sort();

        if images.is_empty() {
            eprintln!(
                "{} skipping {}: no .jpg/.jpeg/.png files inside",
                style("⚠").yellow(),
                path.display()
            );
            continue;
        }

        out.push(ScannedStyle {
            id: slugify(name),
            display_name: humanize(name),
            exemplars: images,
        });
    }
    Ok(out)
}

/// Lowercased, non-alphanumerics → `_`, no leading/trailing/repeated `_`.
fn slugify(s: &str) -> String {
    let mut acc = String::with_capacity(s.len());
    let mut prev_underscore = true;
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            acc.push(ch.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            acc.push('_');
            prev_underscore = true;
        }
    }
    while acc.ends_with('_') {
        acc.pop();
    }
    if acc.is_empty() {
        acc.push_str("style");
    }
    acc
}

/// Title-case-ish display name derived from a directory name. Best-effort —
/// the user is expected to edit the emitted HJSON anyway.
fn humanize(s: &str) -> String {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().chain(chars.flat_map(|c| c.to_lowercase())).collect::<String>()
                }
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn render_hjson(styles: &[ScannedStyle], hjson_parent: &Path) -> String {
    let mut out = String::new();
    out.push_str(
        "{\n    # Auto-generated by `plakat style init`. Edit to fill in\n    \
         # descriptions, trigger phrases, and LoRA references. See Documentation/STYLES.md\n    \
         # for the schema and `tools/style_sources/catalog.hjson` for an\n    \
         # example with LoRAs configured.\n    \
         #\n    \
         # Build the catalog with:\n    \
         #     cargo run --release --bin build_catalog -- \\\n    \
         #         --sources <this-file> --out <output-dir>\n\n",
    );
    out.push_str("    schema_version: 1\n\n");
    out.push_str("    encoder:\n    {\n        id: clip-h-laion2b\n        embed_dim: 1024\n    }\n\n");
    out.push_str(
        "    detection:\n    {\n        aggregation: top3-mean\n        \
         min_confidence: 0.22\n        margin_over_runner_up: 0.02\n    }\n\n",
    );
    out.push_str("    styles:\n    [\n");

    for (i, s) in styles.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str("        {\n");
        out.push_str(&format!("            id: {}\n", s.id));
        out.push_str(&format!("            display_name: \"{}\"\n", s.display_name));
        out.push_str("            description: \"\"\n\n");
        out.push_str("            exemplars:\n            [\n");
        for img in &s.exemplars {
            let rel = relpath(img, hjson_parent);
            out.push_str(&format!("                {}\n", rel.display()));
        }
        out.push_str("            ]\n\n");
        out.push_str(
            "            # Add per-base-model LoRA references + trigger phrases below.\n            \
             # Leave `models: {}` for a detection-only style (no transfer).\n",
        );
        out.push_str("            models: {}\n");
        out.push_str("        }\n");
    }

    out.push_str("    ]\n}\n");
    out
}

/// Compute a relative path from `base` to `target`. Falls back to the
/// absolute target path if they don't share a usable prefix — keeps the
/// emitted HJSON portable when the user moves it later.
fn relpath(target: &Path, base: &Path) -> PathBuf {
    let target = target.canonicalize().unwrap_or_else(|_| target.to_owned());
    let base = base.canonicalize().unwrap_or_else(|_| base.to_owned());

    let mut t_components: Vec<_> = target.components().collect();
    let mut b_components: Vec<_> = base.components().collect();

    // Drop the common prefix.
    let mut common = 0;
    while common < t_components.len()
        && common < b_components.len()
        && t_components[common] == b_components[common]
    {
        common += 1;
    }
    if common == 0 {
        // No shared prefix (different roots) — fall back to absolute.
        return target;
    }
    t_components.drain(..common);
    b_components.drain(..common);

    let mut rel = PathBuf::new();
    for _ in &b_components {
        rel.push("..");
    }
    for c in &t_components {
        rel.push(c.as_os_str());
    }
    if rel.as_os_str().is_empty() {
        rel.push(".");
    }
    rel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_normalizes_corpus_names() {
        assert_eq!(slugify("Watercolor"), "watercolor");
        assert_eq!(slugify("Oil Painting"), "oil_painting");
        assert_eq!(slugify("art-nouveau"), "art_nouveau");
        assert_eq!(slugify("__leading"), "leading");
        assert_eq!(slugify("trailing__"), "trailing");
        assert_eq!(slugify("Mucha 1896!"), "mucha_1896");
        assert_eq!(slugify("!!!"), "style"); // fallback for empty result
    }

    #[test]
    fn humanize_titlecases_words() {
        assert_eq!(humanize("oil_painting"), "Oil Painting");
        assert_eq!(humanize("art-nouveau"), "Art Nouveau");
        assert_eq!(humanize("ukiyo_e"), "Ukiyo E");
    }
}
