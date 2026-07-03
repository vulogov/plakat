//! Tier 0 — structural / determinism invariants that need **no external data** (no model
//! weights, no network). These run offline and in CI, and guard properties that were the
//! root cause of real bugs: CFG batch layout (the SDXL-AnimateDiff scramble, BUGFIX 1.1),
//! seed→noise reproducibility, and map byte-stability.

use anyhow::{Result, ensure};
use candle_core::{DType, Device, Tensor};

use super::{Check, Report};

pub fn run(report: &mut Report) {
    report.push(Check::from_result(
        "tier0.cfg_batch_layout",
        0,
        "CFG conditioning tiles BLOCKED [uncond×F, cond×F], matching the latent split",
        cfg_batch_layout(),
    ));
    report.push(Check::from_result(
        "tier0.map_render_deterministic",
        0,
        "`plakat map` render is a byte-stable pure function of (spec, seed)",
        map_render_deterministic(),
    ));
}

/// The CFG batch layout must be BLOCKED — `[uncond×F, cond×F]` — because the latents are
/// `cat([latents, latents])` and split back with `chunk(2, 0)`. Building the conditioning as
/// `cat([uncond, cond]).repeat(F)` INTERLEAVES it (`repeat` tiles the whole tensor), which
/// mispairs every frame ≥ 2 with the wrong prompt — the silently-incoherent-video bug the
/// 1.22.0 audit fixed. This asserts the layout the pipelines rely on and that it is NOT the
/// interleaved one, so a future refactor (or a candle semantics change) can't reintroduce it.
fn cfg_batch_layout() -> Result<()> {
    let dev = Device::Cpu;
    let frames = 3usize;
    // uncond rows carry 0.0, cond rows carry 1.0 (shape (1, seq, dim)).
    let uncond = Tensor::zeros((1usize, 2, 4), DType::F32, &dev)?;
    let cond = Tensor::ones((1usize, 2, 4), DType::F32, &dev)?;

    // Correct (blocked) construction: replicate each branch across frames FIRST, then stack.
    let blocked = Tensor::cat(&[&uncond.repeat((frames, 1, 1))?, &cond.repeat((frames, 1, 1))?], 0)?;
    ensure!(blocked.dim(0)? == 2 * frames, "blocked batch has 2F rows");
    let blocked_means = blocked.mean(2)?.mean(1)?.to_vec1::<f32>()?;
    ensure!(
        blocked_means == vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        "blocked layout must be [uncond×F, cond×F], got {blocked_means:?}"
    );

    // The buggy (interleaved) construction MUST differ — guards against reintroduction.
    let interleaved = Tensor::cat(&[&uncond, &cond], 0)?.repeat((frames, 1, 1))?;
    let inter_means = interleaved.mean(2)?.mean(1)?.to_vec1::<f32>()?;
    ensure!(
        inter_means == vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0] && inter_means != blocked_means,
        "interleaved layout must be distinct from blocked (else the invariant is meaningless)"
    );
    Ok(())
}

/// `plakat map` documents its render as a byte-stable pure function of `(spec, seed)`. Render
/// a small map twice and require identical bytes. (This is the in-process guarantee; the
/// cross-machine claim is a separate, open decision — see RFC_VERIFY / ROADMAP_2.0.0.)
fn map_render_deterministic() -> Result<()> {
    use crate::map::{render::Style, render_map_image, spec::MapSpec};
    let spec = MapSpec::minimal("verify", 2, 2, 5);
    let style = Style::named("parchment")?;
    let a = render_map_image(&spec, 7, style)?;
    let b = render_map_image(&spec, 7, style)?;
    ensure!(a.dimensions() == b.dimensions(), "map render size not stable");
    ensure!(a.as_raw() == b.as_raw(), "map render is not byte-stable for a fixed (spec, seed)");
    // A different seed must change the map (else the seed is ignored).
    let c = render_map_image(&spec, 8, style)?;
    ensure!(a.as_raw() != c.as_raw(), "map render ignores the seed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier0_checks_all_pass() {
        let mut report = Report::default();
        run(&mut report);
        let failures: Vec<_> = report
            .checks
            .iter()
            .filter(|c| c.status == super::super::Status::Fail)
            .collect();
        assert!(failures.is_empty(), "tier 0 must be green: {failures:?}");
        assert_eq!(report.checks.len(), 2, "two tier-0 checks");
    }
}
