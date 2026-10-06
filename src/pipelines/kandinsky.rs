//! Kandinsky 5.0 T2I Lite — plakat's eighth model family (RFC KANDINSKY-1).
//!
//! **Through P2.** Phase 0 registered the surface (alias, variant and dispatch, capability row, the
//! native resolution buckets, the family-scoped flags). P1 adds the conditioning and the VAE
//! (`kandinsky_text`): `generate --model kandinsky5` now loads the text encoders, encodes the prompt
//! and the negative, releases the encoders, and — given an image — round-trips it through the VAE. P2
//! adds the 6B flow-matching DiT (`kandinsky_dit`), verified against the reference stage by stage but
//! not yet driven from here: the denoise loop is P3, and until then the run ends at [`not_yet`].

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::kandinsky_text::{Embeds, TextEncoders, Vae};

/// The seven native resolution buckets (W×H) the model was trained at (RFC §4.3).
pub const BUCKETS: [(u32, u32); 7] = [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)];

/// Pipeline defaults (RFC §4.3).
pub const DEFAULT_STEPS: usize = 50;
pub const DEFAULT_GUIDANCE: f64 = 3.5;
/// Prompt tokens kept after the template (RFC §7.1); the reference requires `< 1024`.
pub const DEFAULT_MAX_SEQ: usize = 512;
pub const MAX_SEQ_CAP: usize = 1023;
/// Peak resident GB at BF16 with staged residency — encode, release the encoders, denoise (RFC §10.1).
pub const STAGED_PEAK_GB: f64 = 15.0;

/// Whether the family's pipeline exists yet. Phase 0 = no: `doctor --capability` says `pending` rather
/// than promising a fit, and every path into the family ends at [`not_yet`].
pub const PIPELINE_READY: bool = false;

/// Whether a `--model` alias or repo id names this family. The substring is unambiguous across
/// plakat's alias and repo surface; a short alias that does not carry it (`k5`) is resolved to its
/// repo first.
pub fn is_kandinsky(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("kandinsky") || crate::hf::resolve_alias(&m).to_lowercase().contains("kandinsky")
}

/// Snap a requested size to the nearest native bucket: by aspect ratio first (compared in log space,
/// so 2:1 and 1:2 are equally far from square), then by area.
pub fn snap_bucket(w: u32, h: u32) -> (u32, u32) {
    let want = (w.max(1) as f64 / h.max(1) as f64).ln();
    let area = w as f64 * h as f64;
    let mut best = BUCKETS[0];
    let mut best_key = (f64::MAX, f64::MAX);
    for &(bw, bh) in &BUCKETS {
        let d_ratio = ((bw as f64 / bh as f64).ln() - want).abs();
        let d_area = ((bw as f64 * bh as f64) - area).abs();
        if d_ratio < best_key.0 - 1e-9 || ((d_ratio - best_key.0).abs() <= 1e-9 && d_area < best_key.1) {
            best = (bw, bh);
            best_key = (d_ratio, d_area);
        }
    }
    best
}

/// `--size-exact`: an off-bucket size must still divide by 16 (the 8× VAE and the 2×2 patch).
pub fn check_exact(w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 || w % 16 != 0 || h % 16 != 0 {
        anyhow::bail!("--size-exact needs both dimensions divisible by 16 (the 8× VAE and the DiT's 2×2 patch); got {w}x{h}");
    }
    Ok(())
}

/// Where every path into the family ends until the denoise loop (P3) lands.
pub fn not_yet() -> anyhow::Error {
    anyhow::anyhow!(
        "Kandinsky 5 cannot generate yet (RFC KANDINSKY-1): the surface (P0), the text encoders + VAE (P1) \
         and the DiT (P2) are in; the denoise loop lands in P3."
    )
}

/// The sigma schedule the family samples on: diffusers' `FlowMatchEulerDiscreteScheduler` at
/// `shift = 5.0` (the repo's `scheduler_config.json`), `steps + 1` values ending at 0. The DiT's
/// timestep is `sigma · 1000`.
pub const SCHEDULER_SHIFT: f64 = 5.0;
pub fn sigmas(steps: usize) -> Vec<f64> {
    super::sana::flow_sigmas(steps, SCHEDULER_SHIFT)
}

fn stats(t: &Tensor) -> Result<(f32, f32)> {
    let f = t.to_dtype(DType::F32)?.flatten_all()?;
    let mean = f.mean_all()?.to_scalar::<f32>()?;
    let var = f.broadcast_sub(&f.mean_all()?)?.sqr()?.mean_all()?.to_scalar::<f32>()?;
    Ok((mean, var.sqrt()))
}

fn report(name: &str, e: &Embeds) -> Result<()> {
    let (m, s) = stats(&e.qwen)?;
    let (pm, ps) = stats(&e.pooled)?;
    crate::ui::progress::println(&format!(
        "kandinsky5: {name}: {} tokens in → Qwen hidden {:?} (mean {m:+.4}, std {s:.4}) · CLIP pooled {:?} (mean {pm:+.4}, std {ps:.4})",
        e.ids.len(),
        e.qwen.dims(),
        e.pooled.dims()
    ));
    if let Some(d) = &e.dropped {
        crate::ui::progress::println(&format!("kandinsky5: {name}: the prompt ran past --max-seq; dropped: “{}”", d.trim()));
    }
    Ok(())
}

/// P1's run: load the text encoders, encode the prompt and the negative, RELEASE the encoders (staged
/// residency, RFC §10.1), optionally round-trip an image through the VAE, then stop at [`not_yet`].
///
/// * `PLAKAT_K5_DUMP_DIR=<dir>` writes `p1_plakat.safetensors` (the token ids, the Qwen hidden states and
///   the CLIP pooled vectors, positive and negative) for comparison against a reference dump.
/// * `PLAKAT_K5_VAE_IMAGE=<file>` encodes that image at the run's size and decodes it again, reports the
///   round-trip PSNR and writes `kandinsky_p1_vae_roundtrip.png` into the output directory.
pub async fn run_p1(model: &str, prompt: &str, negative: &str, max_seq: usize, size: (u32, u32), device: &Device, out: &std::path::Path) -> Result<()> {
    let repo = crate::hf::resolve_alias(model).to_string();
    // `PLAKAT_K5_STAGE=vae` runs the VAE check alone (no 15 GB text tower): for a quick look at the VAE.
    let vae_only = std::env::var("PLAKAT_K5_STAGE").ok().as_deref() == Some("vae");
    if !vae_only {
    let t0 = std::time::Instant::now();
    let spin = crate::ui::progress::spinner("Loading the Kandinsky 5 text encoders (Qwen2.5-VL text tower + CLIP-L)");
    let mut enc = TextEncoders::load(&repo, device).await.context("loading the Kandinsky 5 text encoders")?;
    spin.finish_with_message(format!("✓ text encoders loaded in {:.1}s (template `{}`)", t0.elapsed().as_secs_f64(), enc.template.name));
    let t1 = std::time::Instant::now();
    let pos = enc.encode(prompt, max_seq)?;
    let neg = enc.encode(negative, max_seq)?;
    report("prompt", &pos)?;
    report("negative", &neg)?;
    crate::ui::progress::println(&format!("kandinsky5: encoded both in {:.1}s", t1.elapsed().as_secs_f64()));
    if let Some(dir) = std::env::var_os("PLAKAT_K5_DUMP_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        let ids = |e: &Embeds| -> Result<Tensor> { Ok(Tensor::new(e.ids.iter().map(|&i| i as i64).collect::<Vec<_>>().as_slice(), &Device::Cpu)?) };
        let cpu = |t: &Tensor| -> Result<Tensor> { Ok(t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?) };
        let mut m = std::collections::HashMap::new();
        m.insert("qwen_ids_pos".to_string(), ids(&pos)?);
        m.insert("qwen_ids_neg".to_string(), ids(&neg)?);
        m.insert("qwen_hidden_pos".to_string(), cpu(&pos.qwen)?);
        m.insert("qwen_hidden_neg".to_string(), cpu(&neg.qwen)?);
        m.insert("clip_pooled_pos".to_string(), cpu(&pos.pooled)?);
        m.insert("clip_pooled_neg".to_string(), cpu(&neg.pooled)?);
        let path = dir.join("p1_plakat.safetensors");
        candle_core::safetensors::save(&m, &path)?;
        crate::ui::progress::println(&format!("kandinsky5: wrote {}", path.display()));
    }
    // Staged residency: the encoders are gone before anything else is loaded.
    drop((pos, neg));
    drop(enc);
    }

    if let Some(img_path) = std::env::var_os("PLAKAT_K5_VAE_IMAGE") {
        let vae = Vae::load(&repo, device).await?;
        let (w, h) = size;
        let img = image::open(&img_path).with_context(|| format!("reading {}", std::path::Path::new(&img_path).display()))?.resize_exact(w, h, image::imageops::FilterType::Lanczos3).to_rgb8();
        let data: Vec<f32> = (0..3).flat_map(|c| img.pixels().map(move |p| p.0[c] as f32 / 127.5 - 1.0).collect::<Vec<_>>()).collect();
        let x = Tensor::from_vec(data, (1, 3, h as usize, w as usize), device)?;
        let z = vae.encode(&x)?;
        let y = vae.decode(&z)?.clamp(-1.0, 1.0)?;
        let mse = (&y - &x)?.sqr()?.mean_all()?.to_scalar::<f32>()? / 4.0; // on a [0,1] scale
        let psnr = -10.0 * mse.max(1e-12).log10();
        let (zm, zs) = stats(&z)?;
        let px: Vec<f32> = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?;
        let plane = (w * h) as usize;
        let outimg = image::RgbImage::from_fn(w, h, |xx, yy| {
            let i = (yy * w + xx) as usize;
            image::Rgb([0, 1, 2].map(|c| ((px[c * plane + i] + 1.0) * 127.5).round().clamp(0.0, 255.0) as u8))
        });
        std::fs::create_dir_all(out).ok();
        let path = out.join("kandinsky_p1_vae_roundtrip.png");
        outimg.save(&path)?;
        crate::ui::progress::println(&format!("kandinsky5: VAE round-trip at {w}x{h}: latent {:?} (mean {zm:+.3}, std {zs:.3}) · PSNR {psnr:.1} dB → {}", z.dims(), path.display()));
    }
    Err(not_yet())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipelines::t2i::Variant;

    #[test]
    fn the_aliases_resolve_to_the_family() {
        for a in ["kandinsky5", "kandinsky", "k5", "kandinsky5-lite", "kandinsky5-pretrain"] {
            let repo = crate::hf::resolve_alias(a);
            assert!(repo.starts_with("kandinskylab/Kandinsky-5.0-T2I-Lite-"), "{a} → {repo}");
            assert_eq!(Variant::detect(repo), Variant::Kandinsky5, "{a}");
            // The dispatch detects from what the user typed, not the resolved repo — `k5` included.
            assert_eq!(Variant::detect(a), Variant::Kandinsky5, "{a}");
            assert!(is_kandinsky(a) && is_kandinsky(&a.to_uppercase()), "{a}");
        }
        assert_eq!(Variant::detect("kandinsky5"), Variant::Kandinsky5);
        // No other family is caught by the substring.
        for other in ["sd15", "sdxl", "flux-dev", "sd35-medium", "pixart", "stable-cascade", "sana"] {
            assert!(!Variant::detect(other).is_kandinsky(), "{other}");
            assert!(!is_kandinsky(crate::hf::resolve_alias(other)), "{other}");
        }
    }

    #[test]
    fn a_size_snaps_to_its_nearest_bucket() {
        assert_eq!(snap_bucket(1200, 800), (1280, 768)); // the RFC's example
        assert_eq!(snap_bucket(512, 512), (1024, 1024));
        assert_eq!(snap_bucket(720, 1280), (768, 1280));
        assert_eq!(snap_bucket(1920, 800), (1408, 640));
        assert_eq!(snap_bucket(1000, 1250), (896, 1152));
        for &(w, h) in &BUCKETS {
            assert_eq!(snap_bucket(w, h), (w, h), "a bucket is its own nearest");
            assert!(check_exact(w, h).is_ok());
            // The largest bucket stays inside the 128-position RoPE table (RFC §6.2).
            assert!(w / 16 <= 128 && h / 16 <= 128);
        }
    }

    #[test]
    fn the_sigma_schedule_runs_from_one_to_zero() {
        let s = sigmas(DEFAULT_STEPS);
        assert_eq!(s.len(), DEFAULT_STEPS + 1);
        assert!((s[0] - 1.0).abs() < 1e-9, "starts at pure noise: {}", s[0]);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[1] < w[0]), "strictly decreasing");
        // shift 5 keeps the schedule in the high-noise region longer than the unshifted line.
        assert!(s[DEFAULT_STEPS / 2] > 0.5);
    }

    #[test]
    fn an_exact_size_must_divide_by_sixteen() {
        assert!(check_exact(1040, 720).is_ok());
        assert!(check_exact(1000, 720).is_err());
        assert!(check_exact(0, 720).is_err());
    }
}
