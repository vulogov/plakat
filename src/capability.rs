//! `plakat doctor --capability` — which supported models can run on the
//! detected hardware, at a glance.
//!
//! Sizes are **derived, not tabulated** (per the design decision): for a
//! model already in the HF cache we sum its on-disk weight blobs (exactly
//! what plakat loaded); otherwise we query the HF repo's file sizes over the
//! API without downloading. Resident need ≈ weights + a small activation
//! overhead, judged against the conservative budget from [`crate::hw`].
//!
//! Per-model *metadata* (native resolution, default dtype, the tuning lever)
//! is static — it doesn't drift the way GB figures do.
use crate::hw::HardwareReport;
use serde::Serialize;

/// Rough activation + framework overhead added on top of resident weights.
const OVERHEAD_GB: f64 = 2.0;

/// Static per-model metadata. The GB figure is NOT here — it's derived.
struct ModelMeta {
    /// `--model` alias; also the key for repo + `gated` lookup.
    alias: &'static str,
    native_res: u32,
    /// Default dtype on a GPU backend (CPU always falls back to F32).
    dtype: &'static str,
    /// The one lever that most helps when this model is tight / won't fit.
    tuning: &'static str,
    /// Blocked on Metal regardless of memory (e.g. candle's GGUF matmul bug).
    metal_blocked: bool,
}

const MODELS: &[ModelMeta] = &[
    ModelMeta { alias: "sd15",          native_res: 512,  dtype: "F16",  tuning: "", metal_blocked: false },
    ModelMeta { alias: "sd21",          native_res: 768,  dtype: "F16",  tuning: "", metal_blocked: false },
    ModelMeta { alias: "sdxl",          native_res: 1024, dtype: "F16",  tuning: "smaller --size, drop --refiner, or --fast lcm-sdxl", metal_blocked: false },
    ModelMeta { alias: "sdxl-turbo",    native_res: 1024, dtype: "F16",  tuning: "1–4 steps, no CFG", metal_blocked: false },
    ModelMeta { alias: "pony",          native_res: 1024, dtype: "F16",  tuning: "smaller --size or --fast lcm-sdxl", metal_blocked: false },
    ModelMeta { alias: "sd35-medium",   native_res: 1024, dtype: "BF16", tuning: "smaller --size; needs HF_TOKEN", metal_blocked: false },
    ModelMeta { alias: "sd35-large",    native_res: 1024, dtype: "BF16", tuning: "needs ≥32 GB; else use sd35-medium", metal_blocked: false },
    ModelMeta { alias: "pixart",        native_res: 1024, dtype: "BF16", tuning: "T5-XXL is the hog; else use pixart-512", metal_blocked: false },
    ModelMeta { alias: "pixart-512",    native_res: 512,  dtype: "BF16", tuning: "", metal_blocked: false },
    ModelMeta { alias: "stable-cascade", native_res: 1024, dtype: "BF16", tuning: "--decoder-guidance / smaller --size", metal_blocked: false },
    ModelMeta { alias: "flux-dev",      native_res: 1024, dtype: "BF16", tuning: "→ flux-dev-gguf --quant-level Q4_K_S (~7 GB) + --quantize-t5; gated", metal_blocked: false },
    ModelMeta { alias: "flux-schnell",  native_res: 1024, dtype: "BF16", tuning: "→ flux-schnell-gguf Q4 + --quantize-t5; 4-step", metal_blocked: false },
];

/// The native (training) square resolution for a model alias, e.g. sd15 → 512,
/// sd21 → 768, sdxl → 1024. Used by the TUI to generate at a Metal-safe size for
/// the loaded model. Unknown aliases default to 768.
pub fn native_res(alias: &str) -> u32 {
    MODELS
        .iter()
        .find(|m| m.alias == alias)
        .map(|m| m.native_res)
        .unwrap_or(768)
}

/// An estimate of a model's resident memory footprint (weights + runtime overhead).
pub struct ResidentEstimate {
    /// Estimated GB the model occupies once loaded.
    pub gb: f64,
    /// `true` when derived from the on-disk cached snapshot (exact); `false` when it's
    /// a coarse family/dtype guess because the model isn't cached yet.
    pub exact: bool,
}

/// Estimate what `alias` will cost in RAM once loaded — fast + synchronous (no
/// network). If the model is already cached we sum its snapshot weights (exact);
/// otherwise we fall back to a coarse per-family guess so the TUI can still warn
/// before a multi-GB download. Both add `OVERHEAD_GB` of runtime headroom.
pub fn resident_estimate(alias: &str) -> ResidentEstimate {
    let repo = crate::hf::resolve_alias(alias);
    if let Some(gb) = cached_repo_gb(repo) {
        return ResidentEstimate { gb: gb + OVERHEAD_GB, exact: true };
    }
    let weight = MODELS.iter().find(|m| m.alias == alias).map(rough_weight_gb).unwrap_or(8.0);
    ResidentEstimate { gb: weight + OVERHEAD_GB, exact: false }
}

/// A coarse weight-size guess (GB) for an uncached model, by family. Only used when
/// the exact on-disk size is unavailable; deliberately conservative (rounds up).
fn rough_weight_gb(m: &ModelMeta) -> f64 {
    match m.alias {
        "sd15" | "sd21" => 4.0,
        "sdxl" | "sdxl-turbo" | "pony" => 7.0,
        "sd35-medium" => 12.0,
        "sd35-large" => 20.0,
        a if a.starts_with("pixart") => 12.0, // T5-XXL dominates
        "stable-cascade" => 14.0,
        a if a.starts_with("flux") => 24.0,
        _ => 8.0,
    }
}

/// One model's feasibility on the probed hardware.
#[derive(Debug, Clone, Serialize)]
pub struct ModelCapability {
    pub model: String,
    pub family: String,
    pub repo: String,
    pub gated: bool,
    /// Derived total weight size (GB), or None if it couldn't be determined.
    pub weight_gb: Option<f64>,
    /// Where `weight_gb` came from: `cache` | `hf-api` | `unknown`.
    pub size_source: String,
    /// Estimated resident need (GB) = weights + overhead.
    pub resident_gb: Option<f64>,
    pub native_res: u32,
    pub dtype: String,
    /// `runs` | `tight` | `wont-fit` | `blocked` | `unknown`.
    pub verdict: String,
    /// The lever that helps (only when not `runs`), or None.
    pub tuning: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CapabilityReport {
    pub hardware: HardwareReport,
    pub models: Vec<ModelCapability>,
}

/// Build the capability report. When `want_network` is true, models not in
/// the local cache have their size queried from the HF API (no download);
/// otherwise they're left `unknown`.
pub async fn build(hardware: HardwareReport, want_network: bool) -> CapabilityReport {
    let token = std::env::var("HF_TOKEN")
        .ok()
        .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok());

    let mut models = Vec::with_capacity(MODELS.len());
    for m in MODELS {
        let entry = crate::hf::entry_for_alias(m.alias);
        let repo = entry.map(|e| e.repo).unwrap_or(m.alias).to_string();
        let family = entry.map(|e| e.family).unwrap_or("").to_string();
        let gated = entry.map(|e| e.gated).unwrap_or(false);

        let (weight_gb, size_source) = derive_size(&repo, want_network, token.as_deref()).await;
        let resident_gb = weight_gb.map(|w| w + OVERHEAD_GB);
        let verdict = verdict_for(resident_gb, &hardware, m, gated, token.is_some(), size_source);
        let tuning = (verdict != "runs" && !m.tuning.is_empty()).then(|| m.tuning.to_string());

        models.push(ModelCapability {
            model: m.alias.to_string(),
            family,
            repo,
            gated,
            weight_gb,
            size_source: size_source.to_string(),
            resident_gb,
            native_res: m.native_res,
            dtype: if hardware.backend == "cpu" { "F32".into() } else { m.dtype.to_string() },
            verdict: verdict.to_string(),
            tuning,
        });
    }
    CapabilityReport { hardware, models }
}

fn verdict_for(
    resident_gb: Option<f64>,
    hw: &HardwareReport,
    meta: &ModelMeta,
    gated: bool,
    have_token: bool,
    size_source: &str,
) -> &'static str {
    if meta.metal_blocked && hw.backend == "metal" {
        return "blocked";
    }
    if gated && !have_token {
        // Can't even fetch the size; surface the gate rather than a fit guess.
        if resident_gb.is_none() {
            return "gated";
        }
    }
    // "runs" if it fits the conservative budget; "tight" if it still fits
    // total RAM (over budget but workable, maybe slow); else won't fit.
    // A cache size is exact; an HF-API size is a loose UPPER BOUND (the repo
    // often holds several precisions/checkpoints) — it can confirm "runs",
    // but an over-RAM upper bound is inconclusive, not a "won't fit".
    match resident_gb {
        None => "unknown",
        Some(r) if r < hw.budget_gb => "runs",
        Some(r) if r < hw.total_ram_gb => "tight",
        Some(_) if size_source == "cache" => "wont-fit",
        Some(_) => "unknown",
    }
}

/// Derive a repo's total weight size (GB): on-disk cache first (accurate —
/// it's exactly what plakat loaded), then the HF API (no download).
async fn derive_size(repo: &str, want_network: bool, token: Option<&str>) -> (Option<f64>, &'static str) {
    if let Some(gb) = cached_repo_gb(repo) {
        return (Some(gb), "cache");
    }
    if want_network {
        if let Ok(gb) = hf_repo_gb(repo, token).await {
            if gb > 0.0 {
                return (Some(gb), "hf-api");
            }
        }
    }
    (None, "unknown")
}

/// Sum the weight files of a repo's current cached snapshot (None if not
/// cached). Walks the latest `snapshots/<rev>/` — which symlinks to exactly
/// the files plakat fetched — rather than the raw `blobs/` dir (which also
/// holds stale revisions and both precisions), and applies the same
/// precision filter as the HF path so cache and API agree.
fn cached_repo_gb(repo: &str) -> Option<f64> {
    let root = crate::hf::cache::hf_cache_root();
    let snapshots = root
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    // Pick the most-recently-modified snapshot (the current revision).
    let snap = std::fs::read_dir(&snapshots)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())?;
    let mut files = Vec::new();
    walk_files(&snap, "", &mut files);
    let gb = weight_total_gb(&files);
    (gb > 0.0).then_some(gb)
}

/// Recursively collect `(relative-path, size)` for every file under `dir`.
/// `std::fs::metadata` follows symlinks, so cache snapshot entries resolve
/// to their blob sizes.
fn walk_files(dir: &std::path::Path, prefix: &str, out: &mut Vec<(String, u64)>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let rel = if prefix.is_empty() { name } else { format!("{prefix}/{name}") };
        match std::fs::metadata(e.path()) {
            Ok(m) if m.is_dir() => walk_files(&e.path(), &rel, out),
            Ok(m) => out.push((rel, m.len())),
            Err(_) => {}
        }
    }
}

/// Sum the model weight files, de-duplicating **per component**: prefer
/// `.safetensors` over `.bin`/`.ckpt`, and skip a full-precision file only
/// when its own `.fp16.` twin is also present (not whenever *any* fp16 file
/// exists — that wrongly dropped BF16 single-file checkpoints like SD 3.5).
/// Approximates what plakat actually loads.
fn weight_total_gb(files: &[(String, u64)]) -> f64 {
    let paths: std::collections::HashSet<&str> = files.iter().map(|(p, _)| p.as_str()).collect();
    let has_st = files.iter().any(|(p, _)| p.ends_with(".safetensors"));
    let mut total = 0u64;
    for (path, size) in files {
        let is_weight = path.ends_with(".safetensors")
            || path.ends_with(".gguf")
            || path.ends_with(".bin")
            || path.ends_with(".ckpt");
        if !is_weight {
            continue;
        }
        // Prefer safetensors over the pytorch .bin/.ckpt twins.
        if has_st && (path.ends_with(".bin") || path.ends_with(".ckpt")) {
            continue;
        }
        // Prefer this file's own fp16 twin, if one exists.
        if path.ends_with(".safetensors") && !path.contains("fp16") {
            let twin = path.replace(".safetensors", ".fp16.safetensors");
            if paths.contains(twin.as_str()) {
                continue;
            }
        }
        total += size;
    }
    total as f64 / 1e9
}

/// Query the HF repo's file tree (no download) and sum its weight files with
/// the same filter as the cache path. A coarse estimate for models not yet
/// downloaded.
async fn hf_repo_gb(repo: &str, token: Option<&str>) -> anyhow::Result<f64> {
    let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
    let client = reqwest::Client::new();
    let mut req = client.get(&url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let entries: Vec<serde_json::Value> = req.send().await?.error_for_status()?.json().await?;

    let files: Vec<(String, u64)> = entries
        .iter()
        .map(|f| {
            let path = f.get("path").and_then(|p| p.as_str()).unwrap_or("").to_string();
            let size = f
                .get("size")
                .and_then(|s| s.as_u64())
                .or_else(|| f.get("lfs").and_then(|l| l.get("size")).and_then(|s| s.as_u64()))
                .unwrap_or(0);
            (path, size)
        })
        .collect();
    Ok(weight_total_gb(&files))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::HardwareReport;

    #[test]
    fn native_res_per_alias() {
        assert_eq!(native_res("sd15"), 512);
        assert_eq!(native_res("sd21"), 768);
        assert_eq!(native_res("sdxl"), 1024);
        assert_eq!(native_res("totally-unknown"), 768); // safe default
    }

    #[test]
    fn resident_estimate_always_adds_overhead_and_scales_by_family() {
        // Every estimate includes runtime overhead over the bare weights.
        let sd15 = resident_estimate("sd15");
        assert!(sd15.gb >= OVERHEAD_GB, "estimate includes overhead");
        // Larger families estimate larger footprints than SD1.5 (when uncached, the
        // coarse guess drives this; when cached, the real snapshot does).
        let flux = resident_estimate("flux-dev");
        assert!(flux.gb > sd15.gb, "flux is estimated heavier than sd15");
        // An unknown alias still yields a finite, overhead-inclusive guess.
        let unknown = resident_estimate("totally-unknown");
        assert!(unknown.gb > OVERHEAD_GB && !unknown.exact);
    }

    fn hw(budget: f64) -> HardwareReport {
        HardwareReport {
            backend: "metal".into(),
            features: vec!["metal".into()],
            total_ram_gb: budget / 0.75,
            budget_gb: budget,
            budget_is_proxy: false,
            cpu_cores: 8,
            os: "macos".into(),
            arch: "arm64".into(),
            tier: "16 GB".into(),
        }
    }

    fn meta() -> ModelMeta {
        ModelMeta { alias: "x", native_res: 1024, dtype: "BF16", tuning: "t", metal_blocked: false }
    }

    #[test]
    fn verdict_buckets() {
        let h = hw(18.0); // budget 18, total RAM 24
        // 5 GB resident on an 18 GB budget → runs.
        assert_eq!(verdict_for(Some(5.0), &h, &meta(), false, false, "cache"), "runs");
        // 19 GB → over budget but under 24 GB RAM → tight.
        assert_eq!(verdict_for(Some(19.0), &h, &meta(), false, false, "cache"), "tight");
        // 30 GB exact (cache) → won't fit.
        assert_eq!(verdict_for(Some(30.0), &h, &meta(), false, false, "cache"), "wont-fit");
        // 30 GB as an HF upper bound → inconclusive, not "won't fit".
        assert_eq!(verdict_for(Some(30.0), &h, &meta(), false, false, "hf-api"), "unknown");
        // unknown size → unknown.
        assert_eq!(verdict_for(None, &h, &meta(), false, false, "unknown"), "unknown");
    }

    #[test]
    fn metal_block_and_gate() {
        let h = hw(64.0);
        let blocked = ModelMeta { metal_blocked: true, ..meta() };
        assert_eq!(verdict_for(Some(4.0), &h, &blocked, false, false, "cache"), "blocked");
        // gated + no token + unknown size → gated.
        assert_eq!(verdict_for(None, &h, &meta(), true, false, "unknown"), "gated");
        // gated + token present → judged normally.
        assert_eq!(verdict_for(Some(4.0), &h, &meta(), true, true, "cache"), "runs");
    }
}
