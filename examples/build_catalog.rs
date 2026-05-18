//! Build the plakat style catalog from a curator-authored HJSON config.
//!
//! Replaces the simpler `examples/spike_catalog.rs` (which hardcodes
//! routing metadata for the two spike styles).
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example build_catalog -- \
//!     --sources tools/style_sources/catalog.hjson \
//!     --out     assets/style_catalog
//! ```
//!
//! Optional flags:
//!
//! * `--device {auto|cuda[:N]|metal|cpu}` — encoder device.
//! * `--probe-hf` — HEAD-check every catalog LoRA on HuggingFace
//!   before emitting the catalog. Recommended in CI.
//!
//! Outputs (all written into `--out`):
//!
//! * `catalog.json`              — routing metadata
//! * `exemplars.safetensors`     — CLIP-H pooled embeddings, f16,
//!                                 L2-normalized, keyed `<style>/<idx>`
//! * `LICENSES.md`               — sidecar listing each LoRA's license
//! * `provenance.json`           — exemplar_key → source path mapping,
//!                                 so rebuilds can be verified against the
//!                                 same source images

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Tensor};
use clap::Parser;
use serde::Deserialize;
use serde_json::json;

use plakat::pipelines::ip_adapter::{ImageEncoder, IPA_REPO};
use plakat::pipelines::lora::{LoraSource, LoraSpec};

#[derive(Parser, Debug)]
#[command(about = "Build the plakat style catalog from a curator's HJSON config.")]
struct Args {
    /// Curator-authored HJSON file. See tools/style_sources/catalog.hjson
    /// for the schema; exemplar paths are resolved relative to this file.
    #[arg(long, value_name = "PATH")]
    sources: PathBuf,

    /// Output directory. Created if missing; existing files are overwritten.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,

    /// Encoder device (auto | cuda[:N] | metal | cpu).
    #[arg(long, default_value = "auto")]
    device: String,

    /// HEAD-check every LoRA URL on HuggingFace before writing the
    /// catalog. Fails the build if any LoRA doesn't resolve.
    #[arg(long)]
    probe_hf: bool,
}

// ----- Curator-facing schema (what they write in catalog.hjson) -----

#[derive(Debug, Deserialize)]
struct SourceFile {
    schema_version: u32,
    encoder: SourceEncoder,
    #[serde(default)]
    detection: Option<SourceDetection>,
    styles: Vec<SourceStyle>,
}

#[derive(Debug, Deserialize)]
struct SourceEncoder {
    id: String,
    embed_dim: usize,
}

#[derive(Debug, Deserialize, Default)]
struct SourceDetection {
    #[serde(default)]
    aggregation: Option<String>,
    #[serde(default)]
    min_confidence: Option<f32>,
    #[serde(default)]
    margin_over_runner_up: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SourceStyle {
    id: String,
    display_name: String,
    #[serde(default)]
    description: String,
    exemplars: Vec<PathBuf>,
    #[serde(default)]
    models: HashMap<String, SourceModelEntry>,
}

#[derive(Debug, Deserialize)]
struct SourceModelEntry {
    #[serde(default)]
    loras: Vec<SourceLora>,
    #[serde(default)]
    trigger: String,
    #[serde(rename = "negative_extras", alias = "negative-extras", default)]
    negative_extras: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SourceLora {
    Shorthand(String),
    Full {
        spec: String,
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        license: Option<String>,
        #[serde(default)]
        license_url: Option<String>,
    },
}

impl SourceLora {
    fn spec(&self) -> &str {
        match self {
            Self::Shorthand(s) => s,
            Self::Full { spec, .. } => spec,
        }
    }
    fn revision(&self) -> Option<&str> {
        match self {
            Self::Shorthand(_) => None,
            Self::Full { revision, .. } => revision.as_deref(),
        }
    }
    fn license(&self) -> Option<&str> {
        match self {
            Self::Shorthand(_) => None,
            Self::Full { license, .. } => license.as_deref(),
        }
    }
    fn license_url(&self) -> Option<&str> {
        match self {
            Self::Shorthand(_) => None,
            Self::Full { license_url, .. } => license_url.as_deref(),
        }
    }
}

const KNOWN_BASES: &[&str] = &["sd15", "sdxl", "flux"];

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let device = plakat::device::select(&args.device)?;
    let sources_dir = args
        .sources
        .parent()
        .ok_or_else(|| anyhow!("--sources has no parent directory: {}", args.sources.display()))?;

    // ----- Parse + validate -----
    let text = std::fs::read_to_string(&args.sources)
        .with_context(|| format!("reading {}", args.sources.display()))?;
    let src: SourceFile = deser_hjson::from_str(&text)
        .with_context(|| format!("parsing HJSON {}", args.sources.display()))?;
    validate(&src, sources_dir)?;

    std::fs::create_dir_all(&args.out)
        .with_context(|| format!("creating {}", args.out.display()))?;

    // ----- Optional HF availability check before any encoding -----
    // Runs first so curators don't sit through a 60-second encode pass
    // just to discover their LoRA references are broken.
    if args.probe_hf {
        eprintln!("==> probing {} LoRA reference(s) on HuggingFace", count_loras(&src));
        probe_hf(&src).await?;
    }

    // ----- Load encoder + encode every exemplar -----
    eprintln!("==> loading CLIP-H image encoder");
    let weights =
        plakat::hf::download::get_file(IPA_REPO, "models/image_encoder/model.safetensors").await?;
    let encoder = ImageEncoder::load(&weights, &device, DType::F32)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut provenance: HashMap<String, String> = HashMap::new();
    let mut catalog_styles: Vec<serde_json::Value> = Vec::with_capacity(src.styles.len());

    for style in &src.styles {
        eprintln!("==> {} ({} exemplars)", style.id, style.exemplars.len());

        let mut keys = Vec::with_capacity(style.exemplars.len());
        for (idx, rel_path) in style.exemplars.iter().enumerate() {
            let abs = sources_dir.join(rel_path);
            let key = format!("{}/{:02}", style.id, idx);
            eprintln!("    {} ← {}", key, rel_path.display());

            let emb = plakat::style::encode_reference_photo(&encoder, &abs, &device)
                .with_context(|| format!("encoding {}", abs.display()))?;
            let stored = emb.to_dtype(DType::F16)?;
            tensors.insert(key.clone(), stored);
            provenance.insert(key.clone(), rel_path.display().to_string());
            keys.push(key);
        }

        catalog_styles.push(emit_style(style, keys)?);
    }

    // ----- Emit outputs -----
    let exemplars_path = args.out.join("exemplars.safetensors");
    candle_core::safetensors::save(&tensors, &exemplars_path)
        .with_context(|| format!("writing {}", exemplars_path.display()))?;
    eprintln!("==> wrote {} ({} tensors)", exemplars_path.display(), tensors.len());

    let catalog_path = args.out.join("catalog.json");
    write_catalog_json(&catalog_path, &src, catalog_styles)?;
    eprintln!("==> wrote {}", catalog_path.display());

    let licenses_path = args.out.join("LICENSES.md");
    write_licenses_md(&licenses_path, &src)?;
    eprintln!("==> wrote {}", licenses_path.display());

    let provenance_path = args.out.join("provenance.json");
    write_provenance_json(&provenance_path, &src, &provenance)?;
    eprintln!("==> wrote {}", provenance_path.display());

    eprintln!();
    eprintln!(
        "✓ built catalog: {} style(s), {} exemplar embedding(s), {} LoRA reference(s)",
        src.styles.len(),
        tensors.len(),
        count_loras(&src)
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate(src: &SourceFile, sources_dir: &Path) -> Result<()> {
    if src.schema_version != 1 {
        bail!(
            "schema_version={}, expected 1 (this builder supports v1 catalogs only)",
            src.schema_version
        );
    }
    if src.styles.is_empty() {
        bail!("no styles in catalog");
    }

    // Duplicate id check.
    let mut seen: HashMap<&str, ()> = HashMap::new();
    for s in &src.styles {
        if seen.insert(s.id.as_str(), ()).is_some() {
            bail!("duplicate style id '{}'", s.id);
        }
    }

    for style in &src.styles {
        if style.exemplars.is_empty() {
            bail!("style '{}' has zero exemplars", style.id);
        }
        if style.exemplars.len() < 3 {
            eprintln!(
                "⚠ style '{}' has only {} exemplars (< 3 is sparse; detection may be unreliable)",
                style.id,
                style.exemplars.len()
            );
        }

        // Verify every exemplar path resolves to an existing file.
        for rel in &style.exemplars {
            let abs = sources_dir.join(rel);
            if !abs.exists() {
                bail!(
                    "style '{}': exemplar not found at {} (resolved from {})",
                    style.id,
                    abs.display(),
                    rel.display()
                );
            }
        }

        // Unknown base-model slot names.
        for base in style.models.keys() {
            if !KNOWN_BASES.contains(&base.as_str()) {
                bail!(
                    "style '{}': unknown base-model slot '{}' (valid: {:?})",
                    style.id,
                    base,
                    KNOWN_BASES
                );
            }
        }

        // Every LoRA spec parses through plakat's grammar.
        for (base, entry) in &style.models {
            for lora in &entry.loras {
                LoraSpec::from_str(lora.spec()).with_context(|| {
                    format!(
                        "style '{}', base '{}': invalid LoRA spec '{}'",
                        style.id,
                        base,
                        lora.spec()
                    )
                })?;
            }
            // Quiet authoring hint: detection-only style (no LoRAs and no
            // trigger) is fine but probably unintentional in a per-base
            // entry that DOES exist.
            if entry.loras.is_empty() && entry.trigger.is_empty() {
                eprintln!(
                    "⚠ style '{}' base '{}': both `loras` and `trigger` are empty — \
                     declaring this base does nothing (consider removing it)",
                    style.id, base
                );
            }
        }
    }

    Ok(())
}

fn count_loras(src: &SourceFile) -> usize {
    src.styles
        .iter()
        .flat_map(|s| s.models.values())
        .map(|m| m.loras.len())
        .sum()
}

// ---------------------------------------------------------------------------
// HF probe
// ---------------------------------------------------------------------------

async fn probe_hf(src: &SourceFile) -> Result<()> {
    let client = reqwest::Client::builder()
        .user_agent("plakat-build-catalog/0.1")
        .timeout(Duration::from_secs(15))
        .build()?;

    let mut failures = 0usize;
    for style in &src.styles {
        for (base, entry) in &style.models {
            for lora in &entry.loras {
                let spec_str = lora.spec();
                let parsed = match LoraSpec::from_str(spec_str) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("  ✗ {} ({}): bad spec — {}", spec_str, base, e);
                        failures += 1;
                        continue;
                    }
                };
                let LoraSource::Hub { repo, file, .. } = parsed.source else {
                    eprintln!("  ✓ {} ({}): local (not probed)", spec_str, base);
                    continue;
                };
                let revision = lora.revision().unwrap_or("main");
                let repo_resolved = plakat::hf::resolve_alias(&repo).to_string();
                let url = match file {
                    Some(f) => format!(
                        "https://huggingface.co/{}/resolve/{}/{}",
                        repo_resolved, revision, f
                    ),
                    None => format!("https://huggingface.co/api/models/{}", repo_resolved),
                };
                match client.head(&url).send().await {
                    Ok(resp) if resp.status().is_success() => {
                        let rev_note = if revision != "main" {
                            format!(" @ {}", &revision[..revision.len().min(8)])
                        } else {
                            String::new()
                        };
                        eprintln!("  ✓ {} ({}{})", spec_str, base, rev_note);
                    }
                    Ok(resp) => {
                        eprintln!("  ✗ {} ({}): HTTP {}", spec_str, base, resp.status().as_u16());
                        failures += 1;
                    }
                    Err(e) => {
                        eprintln!("  ✗ {} ({}): network error — {}", spec_str, base, e);
                        failures += 1;
                    }
                }
            }
        }
    }
    if failures > 0 {
        bail!(
            "{} LoRA reference(s) failed to resolve on HuggingFace",
            failures
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Output emission
// ---------------------------------------------------------------------------

fn emit_style(style: &SourceStyle, exemplar_keys: Vec<String>) -> Result<serde_json::Value> {
    let models_json: serde_json::Map<String, serde_json::Value> = style
        .models
        .iter()
        .map(|(base, entry)| {
            let loras: Vec<serde_json::Value> = entry
                .loras
                .iter()
                .map(|l| {
                    // Always emit the `Full` shape — keeps the runtime
                    // schema uniform regardless of how curators wrote it.
                    json!({
                        "spec": l.spec(),
                        "revision": l.revision(),
                        "license": l.license(),
                        "license_url": l.license_url(),
                    })
                })
                .collect();
            (
                base.clone(),
                json!({
                    "loras": loras,
                    "trigger": entry.trigger,
                    "negative_extras": entry.negative_extras,
                }),
            )
        })
        .collect();

    Ok(json!({
        "id": style.id,
        "display_name": style.display_name,
        "description": style.description,
        "exemplar_keys": exemplar_keys,
        "models": models_json,
    }))
}

fn write_catalog_json(
    path: &Path,
    src: &SourceFile,
    catalog_styles: Vec<serde_json::Value>,
) -> Result<()> {
    let detection = src.detection.as_ref();
    let catalog = json!({
        "schema_version": src.schema_version,
        "encoder": {
            "id": src.encoder.id,
            "embed_dim": src.encoder.embed_dim,
            "exemplars_file": "exemplars.safetensors",
            "preprocess": "clip-standard-224",
        },
        "detection": {
            "aggregation": detection.and_then(|d| d.aggregation.clone()).unwrap_or_else(|| "top3-mean".to_string()),
            "min_confidence": detection.and_then(|d| d.min_confidence).unwrap_or(0.22),
            "margin_over_runner_up": detection.and_then(|d| d.margin_over_runner_up).unwrap_or(0.02),
        },
        "styles": catalog_styles,
    });
    std::fs::write(path, serde_json::to_string_pretty(&catalog)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn write_licenses_md(path: &Path, src: &SourceFile) -> Result<()> {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "# Style catalog — LoRA licenses").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "Plakat does not redistribute LoRA weights. Each LoRA below is \
         pinned in the catalog as a reference; users download on demand \
         from HuggingFace at their own discretion. Verify each LoRA's \
         license terms apply to your use case before depending on this \
         catalog."
    )
    .unwrap();
    writeln!(out).unwrap();

    for style in &src.styles {
        if style.models.values().all(|m| m.loras.is_empty()) {
            continue;
        }
        writeln!(out, "## {} (`{}`)", style.display_name, style.id).unwrap();
        writeln!(out).unwrap();
        let mut bases: Vec<_> = style.models.iter().collect();
        bases.sort_by_key(|(b, _)| b.as_str());
        for (base, entry) in bases {
            if entry.loras.is_empty() {
                continue;
            }
            writeln!(out, "### `{}` base", base).unwrap();
            writeln!(out).unwrap();
            writeln!(out, "| Spec | Revision | License | URL |").unwrap();
            writeln!(out, "|---|---|---|---|").unwrap();
            for lora in &entry.loras {
                let rev = lora.revision().map(|r| &r[..r.len().min(8)]).unwrap_or("main");
                let lic = lora.license().unwrap_or("(not declared)");
                let url = lora.license_url().unwrap_or("");
                writeln!(out, "| `{}` | `{}` | {} | {} |", lora.spec(), rev, lic, url).unwrap();
            }
            writeln!(out).unwrap();
        }
    }

    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn write_provenance_json(
    path: &Path,
    src: &SourceFile,
    exemplar_sources: &HashMap<String, String>,
) -> Result<()> {
    let mut keys: Vec<&String> = exemplar_sources.keys().collect();
    keys.sort();
    let exemplars_json: serde_json::Map<String, serde_json::Value> = keys
        .into_iter()
        .map(|k| (k.clone(), json!({ "source": exemplar_sources[k] })))
        .collect();

    let value = json!({
        "schema_version": 1,
        "encoder_id": src.encoder.id,
        "exemplars": exemplars_json,
        "note": "Records which source image produced each exemplar embedding. \
                 Re-running build_catalog against the same sources should produce \
                 the same `exemplars.safetensors`, modulo encoder version.",
    });
    std::fs::write(path, serde_json::to_string_pretty(&value)?)
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
