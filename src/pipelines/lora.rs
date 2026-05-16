//! LoRA (Low-Rank Adaptation) support for Stable Diffusion UNets.
//!
//! Implementation: weight merging at load time. For each LoRA target,
//!     W_new = W_orig + (alpha / rank) * scale * (B @ A)
//! where A is `lora_down.weight` (r × k), B is `lora_up.weight` (d × r),
//! and `alpha` is the per-layer scaling stored alongside.
//!
//! Format: kohya-ss (`lora_unet_*` keys, dots → underscores in layer path).
//! This is what civitai ships. Diffusers-format LoRAs (`lora_A` / `lora_B`)
//! are not parsed here.
//!
//! Text-encoder LoRA (`lora_te_*`) is recognised and ignored for now — adding
//! it requires a parallel temp-file dance for the CLIP weights.

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Where a LoRA's weights live.
#[derive(Debug, Clone)]
pub enum LoraSource {
    /// Path to a local `.safetensors`.
    Local(PathBuf),
    /// HuggingFace repo + optional explicit filename inside it. When
    /// `file` is `None`, `discover_lora_file` picks one.
    Hub { repo: String, file: Option<String> },
}

/// Unresolved LoRA spec from the CLI. `resolve()` turns it into a
/// concrete local path (downloading from HF if needed).
#[derive(Debug, Clone)]
pub struct LoraSpec {
    pub source: LoraSource,
    pub scale: f32,
}

/// A LoRA spec with its on-disk file located (downloaded if it was a hub spec).
#[derive(Debug, Clone)]
pub struct ResolvedLora {
    pub path: PathBuf,
    pub scale: f32,
    pub display: String,
}

impl FromStr for LoraSpec {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        // Strip optional trailing :SCALE.
        let (head, scale) = match s.rsplit_once(':') {
            Some((h, sc)) => match sc.parse::<f32>() {
                Ok(v) => (h, v),
                Err(_) => (s, 1.0),
            },
            None => (s, 1.0),
        };

        // Split optional #filename for explicit HF file selection.
        let (path_part, file) = match head.split_once('#') {
            Some((p, f)) => (p, Some(f.to_string())),
            None => (head, None),
        };

        // 1. If it points at an existing file, treat as local.
        let p = PathBuf::from(path_part);
        if p.exists() {
            return Ok(Self {
                source: LoraSource::Local(p),
                scale,
            });
        }

        // 2. Heuristic for HF repo: "org/name" — exactly one '/', not a
        //    local-ish path, no trailing slash.
        let is_hub_shape = !path_part.starts_with('/')
            && !path_part.starts_with('.')
            && !path_part.starts_with('~')
            && path_part.chars().filter(|&c| c == '/').count() == 1
            && !path_part.ends_with('/')
            && !path_part.is_empty();

        if is_hub_shape {
            return Ok(Self {
                source: LoraSource::Hub {
                    repo: path_part.to_string(),
                    file,
                },
                scale,
            });
        }

        // 3. Otherwise, assume local — error will surface at resolve time
        //    with a clearer message than clap's parse error.
        Ok(Self {
            source: LoraSource::Local(p),
            scale,
        })
    }
}

impl LoraSpec {
    pub async fn resolve(&self) -> Result<ResolvedLora> {
        match &self.source {
            LoraSource::Local(p) => {
                if !p.exists() {
                    return Err(anyhow!(
                        "LoRA file not found: {} (path doesn't exist, and doesn't look like an \
                         org/name HF repo — did you mean a hub repo?)",
                        p.display()
                    ));
                }
                Ok(ResolvedLora {
                    path: p.clone(),
                    scale: self.scale,
                    display: p.display().to_string(),
                })
            }
            LoraSource::Hub { repo, file } => {
                let filename = match file {
                    Some(f) => f.clone(),
                    None => discover_lora_file(repo).await?,
                };
                let path = crate::hf::download::get_file(repo, &filename)
                    .await
                    .with_context(|| format!("downloading LoRA {repo}/{filename}"))?;
                Ok(ResolvedLora {
                    path,
                    scale: self.scale,
                    display: format!("{repo}/{filename}"),
                })
            }
        }
    }
}

#[derive(Deserialize)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
}

/// Pick a `.safetensors` file inside an HF repo. Prefers canonical names;
/// falls back to the single .safetensors if there's only one, else the
/// largest with a warning.
async fn discover_lora_file(repo: &str) -> Result<String> {
    let resolved = crate::hf::resolve_alias(repo);
    let url = reqwest::Url::parse_with_params(
        &format!("https://huggingface.co/api/models/{resolved}/tree/main"),
        &[("recursive", "true")],
    )?;
    let resp = reqwest::Client::builder()
        .user_agent("plakat/0.1")
        .build()?
        .get(url)
        .send()
        .await?
        .error_for_status()
        .with_context(|| format!("listing tree of {resolved}"))?;
    let entries: Vec<TreeEntry> = resp.json().await?;

    let candidates: Vec<&TreeEntry> = entries
        .iter()
        .filter(|e| e.kind == "file" && e.path.ends_with(".safetensors"))
        .collect();

    if candidates.is_empty() {
        return Err(anyhow!(
            "no .safetensors found in {resolved}; specify the file with \
             `{repo}#path/to/lora.safetensors`"
        ));
    }

    // Canonical names first.
    const CANONICAL: &[&str] = &[
        "pytorch_lora_weights.safetensors",
        "lora.safetensors",
        "adapter_model.safetensors",
    ];
    for name in CANONICAL {
        if let Some(e) = candidates.iter().find(|e| e.path == *name) {
            return Ok(e.path.clone());
        }
    }
    if candidates.len() == 1 {
        return Ok(candidates[0].path.clone());
    }
    let largest = candidates.iter().max_by_key(|e| e.size).unwrap();
    tracing::warn!(
        target: "plakat",
        "{resolved} has {} .safetensors files; picking largest ({}). \
         Use `{repo}#path/to/file.safetensors` to choose.",
        candidates.len(),
        largest.path
    );
    Ok(largest.path.clone())
}

/// Merge one or more kohya LoRAs into a UNet, write the result to `out_path`.
/// Returns (modified_keys, total_lora_targets_in_files) — the second number
/// helps diagnose LoRAs whose targets don't match this base model.
pub fn merge_loras_into_unet(
    base_path: &Path,
    out_path: &Path,
    loras: &[ResolvedLora],
    default_scale: f32,
    device: &Device,
) -> Result<(usize, usize)> {
    // Load all UNet tensors. ~3.4 GB peak RAM for SD 1.5, ~10 GB for SDXL.
    let mut merged: HashMap<String, Tensor> = candle_core::safetensors::load(base_path, device)
        .with_context(|| format!("loading base UNet {}", base_path.display()))?;
    let kohya_to_diffusers = build_kohya_map(&merged);

    let mut modified = 0usize;
    let mut seen_targets = 0usize;
    for lora in loras {
        let lora_tensors: HashMap<String, Tensor> =
            candle_core::safetensors::load(&lora.path, device)
                .with_context(|| format!("loading LoRA {}", lora.display))?;
        let effective_scale = lora.scale * default_scale;
        let (n_mod, n_targets) =
            apply_one_lora(&mut merged, &lora_tensors, &kohya_to_diffusers, effective_scale)?;
        modified += n_mod;
        seen_targets += n_targets;
        tracing::info!(
            target: "plakat",
            "LoRA {}: {n_mod}/{n_targets} targets merged (scale {:.2})",
            lora.display,
            effective_scale
        );
    }

    candle_core::safetensors::save(&merged, out_path)
        .with_context(|| format!("writing merged UNet to {}", out_path.display()))?;
    Ok((modified, seen_targets))
}

/// Build "lora_unet_..." → "down_blocks.0.attentions...weight" map from the
/// base UNet's actual key set. Inverts the kohya convention by enumerating.
fn build_kohya_map(base: &HashMap<String, Tensor>) -> HashMap<String, String> {
    let mut map = HashMap::with_capacity(base.len());
    for key in base.keys() {
        let Some(stem) = key.strip_suffix(".weight") else {
            continue;
        };
        let kohya = format!("lora_unet_{}", stem.replace('.', "_"));
        map.insert(kohya, key.clone());
    }
    map
}

#[derive(Default)]
struct LoraGroup {
    down: Option<Tensor>, // lora_down.weight: (rank, in)  or 4D (rank, in, 1, 1)
    up: Option<Tensor>,   // lora_up.weight:   (out, rank) or 4D (out, rank, 1, 1)
    alpha: Option<Tensor>,
}

/// Guess what base model the LoRA was made for by looking at a delta's
/// "in" dim (last axis), which on cross-attention layers equals the base
/// model's `cross_attention_dim`.
fn guess_base_mismatch_hint(base_dims: &[usize], delta_dims: &[usize]) -> String {
    let base_in = base_dims.get(1).copied().unwrap_or(0);
    let delta_in = delta_dims.last().copied().unwrap_or(0);

    let name = |d: usize| match d {
        768 => Some("SD 1.5 (--model sd15)"),
        1024 => Some("SD 2.1 (--model sd21)"),
        2048 => Some("SDXL (--model sdxl or sdxl-turbo)"),
        _ => None,
    };
    match (name(delta_in), name(base_in)) {
        (Some(want), Some(have)) => {
            format!("LoRA looks trained for {want}; you're running with {have}. Re-run with the matching --model.")
        }
        (Some(want), None) => format!("LoRA looks trained for {want}. Re-run with the matching --model."),
        _ => "Try a different --model that matches this LoRA's training base.".to_string(),
    }
}

fn squeeze_trailing_1x1(t: &Tensor) -> Result<Tensor> {
    let dims = t.dims();
    if dims.len() == 4 && dims[2] == 1 && dims[3] == 1 {
        Ok(t.squeeze(3)?.squeeze(2)?)
    } else {
        Ok(t.clone())
    }
}

/// Reshape `(down, up)` so that `up_2d @ down_2d` produces the delta with
/// out-channels in dim 0 and in-channels*kh*kw flattened in dim 1.
fn normalize_lora_pair(down: &Tensor, up: &Tensor) -> Result<(Tensor, Tensor)> {
    // up is (out, rank) or (out, rank, 1, 1).
    let up_2d = squeeze_trailing_1x1(up)?;

    // down is (rank, in) — pass-through; or (rank, in, kh, kw) — flatten.
    let down_dims = down.dims();
    let down_2d = if down_dims.len() == 4 {
        let (r, ic, kh, kw) = (down_dims[0], down_dims[1], down_dims[2], down_dims[3]);
        if kh == 1 && kw == 1 {
            squeeze_trailing_1x1(down)?
        } else {
            down.reshape((r, ic * kh * kw))?
        }
    } else {
        down.clone()
    };

    Ok((down_2d, up_2d))
}

/// Suffixes that identify the three roles of a LoRA tensor.
/// First match wins; longer / more-specific suffixes appear first.
const DOWN_SUFFIXES: &[&str] = &[
    ".lora_A.default.weight",
    ".lora_down.weight",
    ".lora_A.weight",
    ".lora.down.weight",
];
const UP_SUFFIXES: &[&str] = &[
    ".lora_B.default.weight",
    ".lora_up.weight",
    ".lora_B.weight",
    ".lora.up.weight",
];
const ALPHA_SUFFIXES: &[&str] = &[".alpha"];

/// Prefixes that wrap the actual layer path. Strip them before matching.
const STRIP_PREFIXES: &[&str] = &[
    "base_model.model.",
    "diffusion_model.",
];

fn strip_known_prefix(k: &str) -> &str {
    for p in STRIP_PREFIXES {
        if let Some(rest) = k.strip_prefix(p) {
            return rest;
        }
    }
    k
}

fn match_suffix<'a>(k: &'a str, suffixes: &[&str]) -> Option<&'a str> {
    for s in suffixes {
        if let Some(base) = k.strip_suffix(s) {
            return Some(base);
        }
    }
    None
}

/// Resolve a parsed LoRA base path to an actual base UNet key (one of `merged`'s).
/// Handles both kohya (underscored, `lora_unet_` prefix) and diffusers (dotted,
/// optionally `unet.` prefix) styles.
fn resolve_lora_base(
    lora_base: &str,
    kohya_map: &HashMap<String, String>,
    base_keys: &HashSet<String>,
) -> Option<String> {
    // 1. kohya: full key starts with "lora_unet_..."
    if let Some(k) = kohya_map.get(lora_base) {
        return Some(k.clone());
    }
    // 2. diffusers-style with explicit unet. prefix.
    if let Some(rest) = lora_base.strip_prefix("unet.") {
        let with_weight = format!("{rest}.weight");
        if base_keys.contains(&with_weight) {
            return Some(with_weight);
        }
    }
    // 3. diffusers-style without prefix.
    let direct = format!("{lora_base}.weight");
    if base_keys.contains(&direct) {
        return Some(direct);
    }
    None
}

/// Group LoRA-style keys into (down, up, alpha) triples, then apply each to its
/// base UNet weight. Supports kohya and diffusers/PEFT formats.
fn apply_one_lora(
    merged: &mut HashMap<String, Tensor>,
    lora: &HashMap<String, Tensor>,
    kohya_to_diffusers: &HashMap<String, String>,
    scale: f32,
) -> Result<(usize, usize)> {
    let base_keys: HashSet<String> = merged.keys().cloned().collect();

    let mut groups: BTreeMap<String, LoraGroup> = BTreeMap::new();
    let mut total_keys = 0usize;
    let mut sample_unknown: Option<String> = None;
    for (k, t) in lora.iter() {
        total_keys += 1;
        let normalized = strip_known_prefix(k);
        if let Some(base) = match_suffix(normalized, DOWN_SUFFIXES) {
            groups.entry(base.to_string()).or_default().down = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, UP_SUFFIXES) {
            groups.entry(base.to_string()).or_default().up = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, ALPHA_SUFFIXES) {
            groups.entry(base.to_string()).or_default().alpha = Some(t.clone());
        } else if sample_unknown.is_none() {
            sample_unknown = Some(k.clone());
        }
    }

    let total_targets = groups.len();
    if total_targets == 0 {
        if let Some(sample) = sample_unknown {
            tracing::warn!(
                target: "plakat",
                "no LoRA-style keys recognized in {total_keys} entries (sample: {sample}). \
                 Unsupported format? plakat understands kohya (.lora_down/.lora_up/.alpha) \
                 and diffusers (.lora_A/.lora_B[.default].weight)."
            );
        }
        return Ok((0, 0));
    }

    let mut count = 0usize;
    let mut sample_unmatched: Option<String> = None;
    let mut shape_mismatches = 0usize;
    let mut sample_shape_mismatch: Option<(String, Vec<usize>, Vec<usize>)> = None;

    for (lora_base, g) in groups {
        // Text-encoder targets aren't merged into the UNet here.
        if lora_base.starts_with("lora_te_") || lora_base.starts_with("text_encoder.") {
            continue;
        }
        let Some(diffusers_key) = resolve_lora_base(&lora_base, kohya_to_diffusers, &base_keys)
        else {
            if sample_unmatched.is_none() {
                sample_unmatched = Some(lora_base);
            }
            continue;
        };
        let (down, up) = match (g.down, g.up) {
            (Some(d), Some(u)) => (d, u),
            _ => continue,
        };

        let rank = down.dim(0).unwrap_or(1) as f32;
        let alpha = match &g.alpha {
            Some(a) => {
                let a_f32 = a.to_dtype(DType::F32)?;
                match a_f32.dims().len() {
                    0 => a_f32.to_scalar::<f32>()?,
                    _ => a_f32
                        .flatten_all()?
                        .to_vec1::<f32>()
                        .ok()
                        .and_then(|v| v.first().copied())
                        .unwrap_or(rank),
                }
            }
            None => rank,
        };
        let coeff = (alpha / rank) * scale;

        // Three LoRA-target shapes ship in real safetensors:
        //   - Linear:   down (rank, in),     up (out, rank)
        //   - Conv 1×1: down (rank, in,1,1), up (out, rank,1,1)
        //   - Conv 3×3: down (rank, in,3,3), up (out, rank,1,1)
        // Normalize all of them to a 2D matmul: down_2d = (rank, in*kh*kw),
        // up_2d = (out, rank). Then reshape the delta to match the base.
        let (down_2d, up_2d) = normalize_lora_pair(&down, &up)?;

        // Compute delta in F32 for precision.
        let down_f32 = down_2d.to_dtype(DType::F32)?;
        let up_f32 = up_2d.to_dtype(DType::F32)?;
        let delta_flat = (up_f32.matmul(&down_f32)? * coeff as f64)?;

        let base = merged
            .get(&diffusers_key)
            .ok_or_else(|| anyhow!("base key vanished: {diffusers_key}"))?
            .clone();
        let base_dtype = base.dtype();
        let base_f32 = base.to_dtype(DType::F32)?;

        // delta_flat has shape (out, in*kh*kw). Reshape only if the base is
        // a 4D conv weight AND the element counts match. If the LoRA was
        // trained for a different base (e.g. SDXL vs SD 1.5), shapes won't
        // line up — skip with a count instead of crashing.
        let base_shape = base.dims().to_vec();
        let delta_elems = delta_flat.elem_count();
        let base_elems = base.elem_count();
        if delta_elems != base_elems {
            shape_mismatches += 1;
            if sample_shape_mismatch.is_none() {
                sample_shape_mismatch =
                    Some((diffusers_key.clone(), base_shape, delta_flat.dims().to_vec()));
            }
            continue;
        }
        let delta_shaped = if base.dims().len() == 4 {
            let (oc, ic, kh, kw) = base.dims4()?;
            delta_flat.reshape((oc, ic, kh, kw))?
        } else {
            delta_flat
        };

        let new_base = (base_f32 + delta_shaped)?.to_dtype(base_dtype)?;
        merged.insert(diffusers_key, new_base);
        count += 1;
    }
    if let Some(s) = sample_unmatched {
        tracing::debug!(target: "plakat", "first unmatched LoRA target: {s}");
    }
    if shape_mismatches > 0 {
        let (key, base_dims, delta_dims) = sample_shape_mismatch.unwrap();
        let hint = guess_base_mismatch_hint(&base_dims, &delta_dims);
        if count == 0 {
            return Err(anyhow!(
                "all {shape_mismatches} LoRA target(s) have shape mismatch with the base model — \
                 this LoRA was trained for a different SD variant.\n\
                 example: {key} base={base_dims:?} vs delta={delta_dims:?}\n\
                 {hint}"
            ));
        } else {
            tracing::warn!(
                target: "plakat",
                "{shape_mismatches} target(s) skipped due to shape mismatch \
                 (example: {key} base={base_dims:?} delta={delta_dims:?}). {hint}"
            );
        }
    }
    Ok((count, total_targets))
}
