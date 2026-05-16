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
    // Standard LoRA / LoCon
    down: Option<Tensor>, // (rank, in) or 4D (rank, in, kh, kw)
    up: Option<Tensor>,   // (out, rank) or 4D (out, rank, 1, 1)
    alpha: Option<Tensor>,

    // LyCORIS LoHa
    hada_w1_a: Option<Tensor>,
    hada_w1_b: Option<Tensor>,
    hada_w2_a: Option<Tensor>,
    hada_w2_b: Option<Tensor>,

    // DoRA — extra magnitude vector on top of LoRA delta
    dora_scale: Option<Tensor>,

    // LyCORIS LoKr — Kronecker product factors (each may be full OR low-rank)
    lokr_w1: Option<Tensor>,
    lokr_w1_a: Option<Tensor>,
    lokr_w1_b: Option<Tensor>,
    lokr_w2: Option<Tensor>,
    lokr_w2_a: Option<Tensor>,
    lokr_w2_b: Option<Tensor>,

    // Tensor-LoHa (Tucker decomposition) — detected only, not yet applied.
    hada_t1: Option<Tensor>,
    hada_t2: Option<Tensor>,
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

/// Read alpha as a scalar f32; falls back to `rank` if missing or malformed.
fn extract_alpha(alpha: Option<&Tensor>, rank: f32) -> Result<f32> {
    match alpha {
        Some(a) => {
            let a_f32 = a.to_dtype(DType::F32)?;
            Ok(match a_f32.dims().len() {
                0 => a_f32.to_scalar::<f32>()?,
                _ => a_f32
                    .flatten_all()?
                    .to_vec1::<f32>()
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or(rank),
            })
        }
        None => Ok(rank),
    }
}

/// Standard LoRA delta in F32, flattened to (out, in*kh*kw).
fn build_lora_delta(down: &Tensor, up: &Tensor, coeff: f32) -> Result<Tensor> {
    let (down_2d, up_2d) = normalize_lora_pair(down, up)?;
    let down_f32 = down_2d.to_dtype(DType::F32)?;
    let up_f32 = up_2d.to_dtype(DType::F32)?;
    let delta = up_f32.matmul(&down_f32)?;
    Ok((delta * coeff as f64)?)
}

/// LyCORIS LoHa delta: (W1_b @ W1_a) ⊙ (W2_b @ W2_a) · coeff.
/// All four inputs follow the same shape convention as LoRA's down/up.
fn build_loha_delta(
    w1_a: &Tensor,
    w1_b: &Tensor,
    w2_a: &Tensor,
    w2_b: &Tensor,
    coeff: f32,
) -> Result<Tensor> {
    let (w1a_2d, w1b_2d) = normalize_lora_pair(w1_a, w1_b)?;
    let (w2a_2d, w2b_2d) = normalize_lora_pair(w2_a, w2_b)?;
    let w1 = w1b_2d
        .to_dtype(DType::F32)?
        .matmul(&w1a_2d.to_dtype(DType::F32)?)?;
    let w2 = w2b_2d
        .to_dtype(DType::F32)?
        .matmul(&w2a_2d.to_dtype(DType::F32)?)?;
    let delta = (w1 * w2)?;
    Ok((delta * coeff as f64)?)
}

/// Reconstruct one LoKr factor (w1 or w2). Each factor is either stored
/// fully, or as a rank-decomposed pair (a, b) where the full form is
/// `b · a` with shapes `(out, rank) @ (rank, in)`.
fn build_lokr_factor(
    full: Option<&Tensor>,
    a: Option<&Tensor>,
    b: Option<&Tensor>,
    name: &str,
) -> Result<Tensor> {
    if let Some(w) = full {
        return Ok(w.to_dtype(DType::F32)?);
    }
    match (a, b) {
        (Some(a), Some(b)) => {
            let a_f32 = a.to_dtype(DType::F32)?;
            let b_f32 = b.to_dtype(DType::F32)?;
            Ok(b_f32.matmul(&a_f32)?)
        }
        _ => Err(anyhow!(
            "LoKr factor {name} missing — needs lokr_{name} (full) OR both lokr_{name}_a + lokr_{name}_b"
        )),
    }
}

/// LoKr delta in F32, flattened to (oc, ic*kh*kw). w1 is 2D; w2 may be 2D
/// (Linear targets) or 4D conv-style — flattened along trailing dims for the
/// Kronecker reshape, the outer reshape to base 4D happens at the call site.
///
/// Kronecker: A (m1, n1) ⊗ B (m2, n2) → (m1·m2, n1·n2) where
/// `result[i·m2 + p, j·n2 + q] = A[i, j] · B[p, q]`. The 4D-broadcast +
/// reshape trick below produces that layout for free in candle.
fn build_lokr_delta(w1: &Tensor, w2: &Tensor, coeff: f32) -> Result<Tensor> {
    // Flatten w2 to 2D if conv-style: (oc2, ic2, kh, kw) → (oc2, ic2·kh·kw).
    let w2_2d = match w2.dims().len() {
        2 => w2.clone(),
        4 => {
            let (oc, ic, kh, kw) = w2.dims4()?;
            w2.reshape((oc, ic * kh * kw))?
        }
        n => anyhow::bail!("LoKr w2 has unsupported rank {n}"),
    };
    if w1.dims().len() != 2 {
        anyhow::bail!("LoKr w1 must be 2D (got rank {})", w1.dims().len());
    }

    let (m1, n1) = (w1.dim(0)?, w1.dim(1)?);
    let (m2, n2) = (w2_2d.dim(0)?, w2_2d.dim(1)?);
    let a4 = w1.reshape((m1, 1, n1, 1))?;
    let b4 = w2_2d.reshape((1, m2, 1, n2))?;
    let prod = a4.broadcast_mul(&b4)?; // (m1, m2, n1, n2)
    let kron = prod.reshape((m1 * m2, n1 * n2))?;
    Ok((kron * coeff as f64)?)
}

/// Rebuild one Tucker LoHa half from its core tensor + two factor matrices.
///
/// Math:
///   result[o, i, kh, kw] = sum_{r1, r2} t[r1, r2, kh, kw] · w_b[o, r1] · w_a[r2, i]
///
/// Shapes:
///   t:   (r1, r2, kh, kw)          — Tucker core (both ranks usually = lora_dim)
///   w_a: (r2, in)                  — LoRA-style "down"
///   w_b: (out, r1)                 — LoRA-style "up"
///   →    (out, in, kh, kw)
///
/// Implemented as two matmuls:
///   step 1: temp[o, r2, kh, kw] = w_b ⨂_(r1) t
///   step 2: out[o, in, kh, kw]  = temp ⨂_(r2) w_a    (via permute+reshape)
fn tucker_factor(t: &Tensor, w_a: &Tensor, w_b: &Tensor) -> Result<Tensor> {
    let t_f32 = t.to_dtype(DType::F32)?;
    let wa_f32 = w_a.to_dtype(DType::F32)?;
    let wb_f32 = w_b.to_dtype(DType::F32)?;

    if t_f32.dims().len() != 4 {
        anyhow::bail!(
            "Tucker LoHa core must be 4D (got rank {}); plakat doesn't support \
             Linear-Tucker (which LyCORIS itself doesn't emit)",
            t_f32.dims().len()
        );
    }
    let (r1, r2, kh, kw) = t_f32.dims4()?;
    let (out, r1_check) = (wb_f32.dim(0)?, wb_f32.dim(1)?);
    let (r2_check, in_dim) = (wa_f32.dim(0)?, wa_f32.dim(1)?);
    if r1 != r1_check {
        anyhow::bail!("Tucker rank mismatch: t.dim(0)={r1} but w_b.dim(1)={r1_check}");
    }
    if r2 != r2_check {
        anyhow::bail!("Tucker rank mismatch: t.dim(1)={r2} but w_a.dim(0)={r2_check}");
    }

    // Step 1: contract r1 — w_b (out, r1) @ t.reshape((r1, r2·kh·kw))
    let t_flat = t_f32.reshape((r1, r2 * kh * kw))?;
    let temp_flat = wb_f32.matmul(&t_flat)?; // (out, r2·kh·kw)
    let temp = temp_flat.reshape((out, r2, kh, kw))?;

    // Step 2: contract r2. Rearrange temp so r2 is the inner matmul dim:
    //   (out, r2, kh, kw) → permute (0, 2, 3, 1) → (out, kh, kw, r2)
    //   flatten leading dims → (out·kh·kw, r2)
    //   matmul w_a → (out·kh·kw, in)
    //   reshape → (out, kh, kw, in), permute → (out, in, kh, kw)
    let temp_p = temp.permute((0, 2, 3, 1))?.contiguous()?;
    let temp_2d = temp_p.reshape((out * kh * kw, r2))?;
    let out_2d = temp_2d.matmul(&wa_f32)?; // (out·kh·kw, in)
    let out_4d_perm = out_2d.reshape((out, kh, kw, in_dim))?;
    let result = out_4d_perm.permute((0, 3, 1, 2))?.contiguous()?;
    Ok(result)
}

/// Tucker LoHa delta: `(W1 ⊙ W2) · coeff` where W1 and W2 are each
/// reconstructed via `tucker_factor`. Returns the flattened
/// `(out, in·kh·kw)` shape so it slots into the same downstream reshape
/// path as standard LoRA/LoHa deltas.
#[allow(clippy::too_many_arguments)]
fn build_tucker_loha_delta(
    t1: &Tensor,
    w1_a: &Tensor,
    w1_b: &Tensor,
    t2: &Tensor,
    w2_a: &Tensor,
    w2_b: &Tensor,
    coeff: f32,
) -> Result<Tensor> {
    let w1 = tucker_factor(t1, w1_a, w1_b)?; // (out, in, kh, kw) F32
    let w2 = tucker_factor(t2, w2_a, w2_b)?; // (out, in, kh, kw) F32
    let delta = (w1 * w2)?; // Hadamard
    let (out, in_dim, kh, kw) = delta.dims4()?;
    let flat = delta.reshape((out, in_dim * kh * kw))?;
    Ok((flat * coeff as f64)?)
}

/// DoRA merge: W_new = scale · (W + ΔW) / row_L2_norm(W + ΔW).
/// `dora_scale` is a length-`out_dim` vector; `direction` (= base + delta)
/// can be 2D or 4D. Output matches the shape of `direction`.
fn apply_dora(base_f32: &Tensor, delta: &Tensor, dora_scale: &Tensor) -> Result<Tensor> {
    let direction = (base_f32 + delta)?;
    let dims = direction.dims().to_vec();
    let oc = dims[0];

    // Flatten everything past dim 0, compute per-row L2 norm.
    let flat = direction.flatten_from(1)?;
    let sq = flat.sqr()?;
    let sum = sq.sum_keepdim(1)?;
    let norm = sum.sqrt()?;
    // Avoid division by zero on dead rows.
    let eps = Tensor::full(1e-8_f32, norm.shape(), norm.device())?;
    let norm = norm.maximum(&eps)?;

    let normalized_flat = flat.broadcast_div(&norm)?;
    let normalized = normalized_flat.reshape(dims.as_slice())?;

    // Reshape dora_scale (oc,) → (oc, 1) for 2D base or (oc, 1, 1, 1) for 4D.
    let scale_f32 = dora_scale.to_dtype(DType::F32)?.flatten_all()?;
    let mut shape = Vec::with_capacity(dims.len());
    shape.push(oc);
    for _ in 1..dims.len() {
        shape.push(1);
    }
    let scale = scale_f32.reshape(shape)?;

    Ok(normalized.broadcast_mul(&scale)?)
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

/// Suffixes that identify the role of a LoRA-family tensor.
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

// LyCORIS LoHa — Hadamard product of two low-rank pairs.
const HADA_W1_A_SUFFIXES: &[&str] = &[".hada_w1_a"];
const HADA_W1_B_SUFFIXES: &[&str] = &[".hada_w1_b"];
const HADA_W2_A_SUFFIXES: &[&str] = &[".hada_w2_a"];
const HADA_W2_B_SUFFIXES: &[&str] = &[".hada_w2_b"];

// DoRA — magnitude rescaling on top of standard LoRA delta.
const DORA_SCALE_SUFFIXES: &[&str] = &[".dora_scale"];

// LyCORIS LoKr — Kronecker product. Each factor (w1, w2) is either stored
// fully OR rank-decomposed via {_a, _b}.
const LOKR_W1_SUFFIXES: &[&str] = &[".lokr_w1"];
const LOKR_W1_A_SUFFIXES: &[&str] = &[".lokr_w1_a"];
const LOKR_W1_B_SUFFIXES: &[&str] = &[".lokr_w1_b"];
const LOKR_W2_SUFFIXES: &[&str] = &[".lokr_w2"];
const LOKR_W2_A_SUFFIXES: &[&str] = &[".lokr_w2_a"];
const LOKR_W2_B_SUFFIXES: &[&str] = &[".lokr_w2_b"];

// Tensor-LoHa (Tucker form). Detection-only for now — the einsum
// `"i j k l, j r, i p -> p r k l"` isn't implemented.
const HADA_T1_SUFFIXES: &[&str] = &[".hada_t1"];
const HADA_T2_SUFFIXES: &[&str] = &[".hada_t2"];

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
        } else if let Some(base) = match_suffix(normalized, HADA_W1_A_SUFFIXES) {
            groups.entry(base.to_string()).or_default().hada_w1_a = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, HADA_W1_B_SUFFIXES) {
            groups.entry(base.to_string()).or_default().hada_w1_b = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, HADA_W2_A_SUFFIXES) {
            groups.entry(base.to_string()).or_default().hada_w2_a = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, HADA_W2_B_SUFFIXES) {
            groups.entry(base.to_string()).or_default().hada_w2_b = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, DORA_SCALE_SUFFIXES) {
            groups.entry(base.to_string()).or_default().dora_scale = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, LOKR_W1_SUFFIXES) {
            groups.entry(base.to_string()).or_default().lokr_w1 = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, LOKR_W1_A_SUFFIXES) {
            groups.entry(base.to_string()).or_default().lokr_w1_a = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, LOKR_W1_B_SUFFIXES) {
            groups.entry(base.to_string()).or_default().lokr_w1_b = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, LOKR_W2_SUFFIXES) {
            groups.entry(base.to_string()).or_default().lokr_w2 = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, LOKR_W2_A_SUFFIXES) {
            groups.entry(base.to_string()).or_default().lokr_w2_a = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, LOKR_W2_B_SUFFIXES) {
            groups.entry(base.to_string()).or_default().lokr_w2_b = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, HADA_T1_SUFFIXES) {
            groups.entry(base.to_string()).or_default().hada_t1 = Some(t.clone());
        } else if let Some(base) = match_suffix(normalized, HADA_T2_SUFFIXES) {
            groups.entry(base.to_string()).or_default().hada_t2 = Some(t.clone());
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
                 Unsupported format? plakat understands LoRA / LoCon / DyLoRA \
                 (.lora_down/.lora_up + alpha), PEFT (.lora_A/.lora_B[.default]), \
                 DoRA (+ .dora_scale), LyCORIS LoHa (.hada_w{{1,2}}_a/b, with optional \
                 .hada_t1/_t2 for Tucker), and LyCORIS LoKr (.lokr_w1[/_a/_b], \
                 .lokr_w2[/_a/_b])."
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

        let base = merged
            .get(&diffusers_key)
            .ok_or_else(|| anyhow!("base key vanished: {diffusers_key}"))?
            .clone();
        let base_dtype = base.dtype();
        let base_shape = base.dims().to_vec();

        // Classify the group and compute delta_flat (2D, shape (oc, ic*kh*kw)).
        let delta_flat = match &g {
            // Tucker LoHa — Tucker decomposition of both halves before Hadamard.
            // Requires all 6 keys: t1+w1_a+w1_b and t2+w2_a+w2_b.
            LoraGroup {
                hada_t1: Some(t1),
                hada_w1_a: Some(w1a),
                hada_w1_b: Some(w1b),
                hada_t2: Some(t2),
                hada_w2_a: Some(w2a),
                hada_w2_b: Some(w2b),
                alpha,
                ..
            } => {
                let rank = w1a.dim(0).unwrap_or(1) as f32;
                let alpha_val = extract_alpha(alpha.as_ref(), rank)?;
                let coeff = (alpha_val / rank) * scale;
                build_tucker_loha_delta(t1, w1a, w1b, t2, w2a, w2b, coeff)?
            }
            // LyCORIS LoKr — Kronecker product of two factors. Detection
            // priority: take this arm if any lokr_* key is present.
            LoraGroup {
                lokr_w1,
                lokr_w1_a,
                lokr_w1_b,
                lokr_w2,
                lokr_w2_a,
                lokr_w2_b,
                alpha,
                ..
            } if lokr_w1.is_some()
                || lokr_w1_a.is_some()
                || lokr_w2.is_some()
                || lokr_w2_a.is_some() =>
            {
                let w1 = build_lokr_factor(
                    lokr_w1.as_ref(),
                    lokr_w1_a.as_ref(),
                    lokr_w1_b.as_ref(),
                    "w1",
                )?;
                let w2 = build_lokr_factor(
                    lokr_w2.as_ref(),
                    lokr_w2_a.as_ref(),
                    lokr_w2_b.as_ref(),
                    "w2",
                )?;
                // The "dim" for LoKr alpha scaling is the rank of the inner
                // decomposition. When w1 is full, use its number of rows; when
                // decomposed, use w1_b.dim(1) (= w1_a.dim(0)) which is the rank.
                let dim = match (lokr_w1.is_some(), lokr_w1_b.as_ref()) {
                    (true, _) => w1.dim(0).unwrap_or(1) as f32,
                    (false, Some(b)) => b.dim(1).unwrap_or(1) as f32,
                    _ => w1.dim(0).unwrap_or(1) as f32,
                };
                let alpha_val = extract_alpha(alpha.as_ref(), dim)?;
                let coeff = (alpha_val / dim) * scale;
                build_lokr_delta(&w1, &w2, coeff)?
            }
            // LyCORIS LoHa: (W1_b @ W1_a) ⊙ (W2_b @ W2_a)
            LoraGroup {
                hada_w1_a: Some(w1a),
                hada_w1_b: Some(w1b),
                hada_w2_a: Some(w2a),
                hada_w2_b: Some(w2b),
                alpha,
                ..
            } => {
                let rank = w1a.dim(0).unwrap_or(1) as f32;
                let alpha_val = extract_alpha(alpha.as_ref(), rank)?;
                let coeff = (alpha_val / rank) * scale;
                build_loha_delta(w1a, w1b, w2a, w2b, coeff)?
            }
            // Standard LoRA / LoCon, possibly with DoRA scaling.
            LoraGroup {
                down: Some(down),
                up: Some(up),
                alpha,
                ..
            } => {
                let rank = down.dim(0).unwrap_or(1) as f32;
                let alpha_val = extract_alpha(alpha.as_ref(), rank)?;
                let coeff = (alpha_val / rank) * scale;
                build_lora_delta(down, up, coeff)?
            }
            _ => continue, // incomplete group
        };

        if delta_flat.elem_count() != base.elem_count() {
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

        let base_f32 = base.to_dtype(DType::F32)?;
        let merged_weight = match &g.dora_scale {
            Some(dora) => apply_dora(&base_f32, &delta_shaped, dora)?,
            None => (base_f32 + delta_shaped)?,
        };
        merged.insert(diffusers_key, merged_weight.to_dtype(base_dtype)?);
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
