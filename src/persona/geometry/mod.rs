//! Layer 2 — the face (and figure) geometry engine (RFC §10).
//!
//! Converts a `ResolvedSpec` into an actual conditioning image and resolves every detail anchor into a
//! position. Pure Rust: no GPU, no network, no model weights; a pure function of `(spec, seed)`,
//! byte-stable on-box, and independently useful before any generative step exists — mirroring the
//! `map` track's geometry engine (§10, §5.2).
//!
//! Phase-G reshaped the emphasis (see `Documentation/PERSONA_GATING.md`): with a dedicated
//! face-landmark ControlNet available for SD1.5/2.1 **only**, this engine's cross-family value is its
//! *measurement + region-mask + detail-anchor* outputs (depth / pose / masks / overlay), not a
//! face-mesh-CN generator. The mesh map remains an SD1.5/2.1 Tier-A bonus.
//!
//! Build order (ROADMAP_5.0.0 P2): topology + mean template (this) → deformation basis + `resolve`
//! → conditioning-map rasteriser → figure geometry → `geometry` CLI + corpus.

pub mod basis;
pub mod figure;
pub mod from_spec;
pub mod raster;
pub mod template;
pub mod topology;

pub use from_spec::{figure_params, geometry_values, open_mouth};

pub use basis::{anchor_point, identity, nearest_anchor, resolve, Deformed, GeoWarning, GEOMETRIC_ATTRS};
pub use figure::{
    figure_anchor, figure_skeleton_map, resolve_figure, silhouette_mask, Build, Figure, FigureParams,
    BODY_ANCHOR_VOCAB,
};
pub use raster::{
    dentition_hint, depth_proxy, detail_overlay, face_skeleton, mesh_map, region_mask, wireframe,
    MeshStyle, Region,
};
pub use template::{mean_template, Point, Template};
pub use topology::{is_named_region, named_region, ANCHOR_VOCAB, NUM_LANDMARKS};
