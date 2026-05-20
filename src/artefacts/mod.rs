//! Artefact compositing — place named PNG cutouts into named zones of
//! a generated image.
//!
//! The pipeline is:
//!
//! 1. Curator builds a library (`assets/artefact_library/library.json`
//!    + PNGs). Each artefact declares its **natural zone**, **size %
//!    of zone**, and **anchor** — defaults the user can override per
//!    invocation.
//! 2. User specifies one or more artefacts on `plakat generate` /
//!    `plakat portrait` (`--artefact NAME[@ZONE[:SCALE]]`) or per task
//!    in a scenario (`artefacts: [...]`).
//! 3. After generation but BEFORE stylize / upscale, the compositing
//!    hook resolves each spec to a [`runtime::ResolvedArtefact`] and
//!    alpha-blends the PNG onto the generated image (with auto chroma-
//!    key fallback when the input PNG lacks an alpha channel).
//!
//! The rigid 4×3 zone grid is the v1 baseline. v3 plans replace it
//! with depth/segmentation-aware zones derived from the actual
//! generated image; the runtime abstraction
//! ([`zones::ZoneRef::resolve`] → [`zones::Rect`]) is the boundary
//! v3 will plug into.
//!
//! v2 plans add a masked img2img blending pass after the alpha
//! composite, reusing the multi-persona mask machinery in
//! `pipelines::portrait`.

pub mod anchor;
pub mod compositing;
pub mod library;
pub mod runtime;
pub mod smart_zones;
pub mod zones;

pub use anchor::Anchor;
pub use compositing::composite_resolved;
pub use library::{Artefact, ArtefactLibrary};
pub use runtime::{
    resolve_specs, ArtefactSpec, ArtefactSpecEntry, ArtefactSpecObject, ResolvedArtefact,
};
pub use zones::{Depth, Horizontal, Rect, ZoneOverrides, ZoneRef};

use anyhow::Context;
use std::path::{Path, PathBuf};

use crate::pipelines::depth::DepthPipeline;

/// Convenience: load library, resolve specs at a known canvas size,
/// then composite onto each output file in place (read PNG → composite
/// → write back). Used by both `plakat generate` and `plakat portrait`
/// CLI hooks (and scenarios, per task).
///
/// Empty `specs` short-circuits with no I/O — the caller doesn't need
/// to guard against the no-artefact case.
///
/// When `smart` is `Some`, each file's zone extents are recomputed
/// from the image's own depth map + luminance (v3). When `None`, the
/// rigid grid (with the provided `overrides`) is used for every file
/// (v1/v2 behaviour).
pub fn composite_onto_files(
    specs: &[ArtefactSpec],
    library_dir: &Path,
    files: &[PathBuf],
    canvas_w: u32,
    canvas_h: u32,
    overrides: &ZoneOverrides,
    smart: Option<&DepthPipeline>,
) -> anyhow::Result<()> {
    if specs.is_empty() {
        return Ok(());
    }
    let lib = ArtefactLibrary::load(library_dir)
        .with_context(|| format!("loading artefact library {}", library_dir.display()))?;

    let smart_tag = if smart.is_some() { " (smart zones)" } else { "" };
    crate::ui::progress::println(&format!(
        "  {} compositing {} artefact(s) onto {} image(s){smart_tag}",
        console::style("◆").cyan().bold(),
        specs.len(),
        files.len()
    ));

    for path in files {
        let effective = resolve_overrides_for(path, canvas_w, canvas_h, overrides, smart);
        let resolved = resolve_specs(specs, &lib, canvas_w, canvas_h, &effective)?;
        let mut img = image::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        compositing::composite_resolved(&mut img, &resolved)
            .with_context(|| format!("compositing onto {}", path.display()))?;
        img.save(path)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// Helper for the most common caller pattern: composite onto every
/// `plakat-<seed>.png` (or `plakat-portrait-<seed>.png`, etc.) in
/// `out_dir` for the range of seeds a single generation produced.
pub fn composite_onto_seed_range(
    specs: &[ArtefactSpec],
    library_dir: &Path,
    out_dir: &Path,
    base_seed: Option<u64>,
    count: u32,
    file_prefix: &str,
    canvas_w: u32,
    canvas_h: u32,
    overrides: &ZoneOverrides,
    smart: Option<&DepthPipeline>,
) -> anyhow::Result<()> {
    if specs.is_empty() {
        return Ok(());
    }
    let start = base_seed.unwrap_or(0);
    let mut files: Vec<PathBuf> = Vec::with_capacity(count as usize);
    for i in 0..count {
        // Match t2i's default naming: `<prefix>-<seed>.png`.
        let seed = start.wrapping_add(i as u64);
        files.push(out_dir.join(format!("{file_prefix}-{seed}.png")));
    }
    // Only composite onto files that actually exist (the pipeline may
    // not have written every expected name if a seed override changed
    // the layout). Skip missing rather than error.
    let present: Vec<PathBuf> = files.into_iter().filter(|p| p.exists()).collect();
    composite_onto_files(specs, library_dir, &present, canvas_w, canvas_h, overrides, smart)
}

/// Compute the effective zone overrides for one image. When `smart`
/// is supplied, depth + luminance inference produces per-image
/// overrides; on inference failure we warn once-per-image and fall
/// back to the caller's static `base`.
pub fn resolve_overrides_for(
    path: &Path,
    canvas_w: u32,
    canvas_h: u32,
    base: &ZoneOverrides,
    smart: Option<&DepthPipeline>,
) -> ZoneOverrides {
    let Some(pipeline) = smart else {
        return base.clone();
    };
    match smart_zones::smart_zones_from_image(path, canvas_w, canvas_h, pipeline) {
        Ok(ov) => merge_overrides(base, &ov),
        Err(e) => {
            crate::ui::progress::println(&format!(
                "  {} smart zones failed on {}: {e}. Using rigid grid.",
                console::style("warn:").yellow().bold(),
                path.display(),
            ));
            base.clone()
        }
    }
}

/// Merge smart-zone overrides over the user-supplied base. Smart
/// values win where present; base values fill any band the smart
/// signal couldn't resolve (degenerate depth, flat image, etc).
fn merge_overrides(base: &ZoneOverrides, smart: &ZoneOverrides) -> ZoneOverrides {
    ZoneOverrides {
        sky: smart.sky.or(base.sky),
        far_plan: smart.far_plan.or(base.far_plan),
        middle_plan: smart.middle_plan.or(base.middle_plan),
        close_plan: smart.close_plan.or(base.close_plan),
        left: smart.left.or(base.left),
        center: smart.center.or(base.center),
        right: smart.right.or(base.right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_smart_wins_over_base_when_both_set() {
        let base = ZoneOverrides {
            sky: Some([0.0, 0.30]),
            ..Default::default()
        };
        let smart = ZoneOverrides {
            sky: Some([0.0, 0.45]),
            ..Default::default()
        };
        let merged = merge_overrides(&base, &smart);
        assert_eq!(merged.sky, Some([0.0, 0.45]));
    }

    #[test]
    fn merge_base_fills_gaps_left_by_smart() {
        let base = ZoneOverrides {
            sky: Some([0.0, 0.30]),
            close_plan: Some([0.80, 1.0]),
            ..Default::default()
        };
        let smart = ZoneOverrides {
            // Smart only resolved the middle/far bands.
            middle_plan: Some([0.5, 0.7]),
            far_plan: Some([0.3, 0.5]),
            ..Default::default()
        };
        let merged = merge_overrides(&base, &smart);
        // Smart filled middle + far.
        assert_eq!(merged.middle_plan, Some([0.5, 0.7]));
        assert_eq!(merged.far_plan, Some([0.3, 0.5]));
        // Base retained for the bands smart didn't resolve.
        assert_eq!(merged.sky, Some([0.0, 0.30]));
        assert_eq!(merged.close_plan, Some([0.80, 1.0]));
    }

    #[test]
    fn resolve_overrides_without_smart_clones_base() {
        let base = ZoneOverrides {
            sky: Some([0.0, 0.40]),
            ..Default::default()
        };
        let r = resolve_overrides_for(
            std::path::Path::new("/does/not/exist.png"),
            512,
            512,
            &base,
            None,
        );
        assert_eq!(r.sky, Some([0.0, 0.40]));
        assert!(r.middle_plan.is_none());
    }
}
