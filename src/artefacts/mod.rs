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

/// Convenience: load library, resolve specs at a known canvas size,
/// then composite onto each output file in place (read PNG → composite
/// → write back). Used by both `plakat generate` and `plakat portrait`
/// CLI hooks (and scenarios, per task).
///
/// Empty `specs` short-circuits with no I/O — the caller doesn't need
/// to guard against the no-artefact case.
pub fn composite_onto_files(
    specs: &[ArtefactSpec],
    library_dir: &Path,
    files: &[PathBuf],
    canvas_w: u32,
    canvas_h: u32,
    overrides: &ZoneOverrides,
) -> anyhow::Result<()> {
    if specs.is_empty() {
        return Ok(());
    }
    let lib = ArtefactLibrary::load(library_dir)
        .with_context(|| format!("loading artefact library {}", library_dir.display()))?;
    let resolved = resolve_specs(specs, &lib, canvas_w, canvas_h, overrides)?;

    crate::ui::progress::println(&format!(
        "  {} compositing {} artefact(s) onto {} image(s)",
        console::style("◆").cyan().bold(),
        resolved.len(),
        files.len()
    ));

    for path in files {
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
    composite_onto_files(specs, library_dir, &present, canvas_w, canvas_h, overrides)
}
