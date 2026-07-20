//! `plakat photos` — TUI photo & image collection manager (RFC PHOTOS-1, the 3.x flagship).
//!
//! Phase 1: walk + classify the library ([`library`]), the per-album HJSON store ([`hjson`]), image
//! + RAW decode / EXIF / thumbnail cache ([`loader`], [`exif`]), and a three-pane shell (status bar ·
//! tree | album grid · command) with a navigable Tree and a lazily-rendered thumbnail grid. Later
//! phases (RFC §29) add the image view, editing, browse, and vision features.

pub mod analysis;
pub mod dedup;
pub mod edit;
pub mod exif;
pub mod export;
pub mod hjson;
pub mod import;
pub mod layers;
pub mod lut;
pub mod cull;
pub mod faces;
pub mod homography;
pub mod mledit;
pub mod mlworker;
pub mod multishot;
pub mod quality;
pub mod nl;
pub mod portfolio;
pub mod rename;
pub mod scrub;
pub mod stitch;
pub mod versions;
pub mod vision;
pub mod visual_search;
pub mod library;
pub mod loader;
pub mod watcher;

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::stdout;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::ExecutableCommand;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::{Resize, StatefulImage};

use library::{LibraryNode, NodeKind};

use std::sync::mpsc::Receiver;
use tokio::runtime::Handle;
use crate::pipelines::gen_channel::{CancelFlag, GenMessage};

/// Which pane has focus (RFC §6).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Tree,
    Album,
}

/// Album pane sub-mode (RFC §6): thumbnail grid, full-pane image view, the culling loupe, or the
/// side-by-side survey/compare of a small selection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AlbumMode {
    Grid,
    Image,
    Cull,
    Compare,
}

/// A tree/curation action awaiting text or confirmation in the command pane (RFC §11).
enum PendingCmd {
    NewFolder { parent: PathBuf },
    NewAlbum { parent: PathBuf },
    Rename { path: PathBuf },
    Delete { path: PathBuf, is_album: bool },
    /// Edit a free-text metadata field of a single image (RFC §8.5: notes/caption). Carries the
    /// image path so the write can be routed to the right album even in a smart-album view.
    EditMeta { path: PathBuf, field: EditField },
    /// Save the current filter as a named library-wide smart album (root folder.hjson).
    SaveSmart { query: String },
    /// Delete a named smart album from the root folder.hjson.
    DeleteSmart { name: String },
    /// Run a library-wide metadata (TF-IDF) semantic search for the entered free-text query.
    Search,
    /// Run a library-wide CLIP *visual* search for the entered free-text query.
    VisualSearch,
    /// Aesthetic auto-cull: rank the album and keep the entered top-N (flag keepers, reject the rest).
    AutoCull,
    /// AI generate (txt2img) from the entered prompt → a new album image.
    Generate,
    /// AI portrait from the entered prompt (+ the cursor image as an optional identity face).
    Portrait,
    /// AI multiperson scene from the entered prompt (the selected images become the people).
    Multiperson,
    /// Multi-shot composite over the selection: `hdr` (exposure fusion) or `focus` (focus stacking).
    Multishot,
    /// Move (or copy, when `copy`) the target images into the album at the entered path.
    MoveToAlbum { copy: bool },
    /// Confirm permanently emptying the trash.
    EmptyTrash,
    /// Collect a prompt for a T2 ML edit (`relight` when true, else `img2img`), then queue the job.
    MlPrompt { relight: bool },
    /// Export the current targets to the entered `DIR [MAXPX]`.
    Export,
    /// Batch-rename the current targets with the entered pattern (`#` runs = numbers).
    BatchRename,
    /// Export the current targets as a portfolio to the entered `DIR [MAXPX] [| watermark text]`.
    Portfolio,
    /// Crop the cursor image to the entered exact `WxH` pixels (centered).
    CropExact,
    /// Resize the cursor image to fit the entered `WxH` (or single `N`) pixels.
    ResizeExact,
    /// Add the entered image (album filename or path) as a new top layer in layer mode.
    AddLayer,
    /// Set the active layer's mask to the entered grayscale matte (album filename or path).
    MaskImage,
    /// Save the cursor image's edit stack as a named preset.
    SavePreset,
    /// Straighten the cursor image by the entered degrees.
    Straighten,
    /// Confirm stripping EXIF/GPS metadata from the target images.
    StripExif,
    /// Confirm redacting only GPS from the target images (keeps the rest of the EXIF).
    RedactGps,
    /// Convert the targets to the entered `fmt [Npx | NkB]`.
    Convert,
    /// Burn a watermark/caption (`TEXT` or `TEXT | font.ttf`) onto the targets.
    Watermark,
    /// Apply the `.cube` LUT at the entered path to the targets.
    Lut,
    /// Edit an album-level metadata field (from the tree info editor / tag keys).
    AlbumEdit { path: PathBuf, field: AlbumFieldKind },
    /// Export an album/folder's images (tree `e`) to the entered `DIR [MAXPX]`.
    ExportAlbum { path: PathBuf, recursive: bool },
    /// Export + convert an album/folder's images (tree `E`) to the entered `FMT DIR [MAXPX]`.
    ExportConvertAlbum { path: PathBuf, recursive: bool },
    /// Materialize a smart album (saved search) into a portable album of file **copies** at the
    /// entered directory — no symlinks; curation travels in a fresh `album.hjson`.
    MaterializeSmart { name: String, query: String },
    /// Stitch the selection into a panorama in the entered direction (`h` / `v` / `grid`).
    Panorama,
    /// A natural-language command from the `:` pane — parse (deterministic, else LLM) then confirm.
    NlCommand,
    /// Confirm and run the pending parsed command plan.
    ConfirmPlan,
}

/// A heavy per-image job the event loop drains one at a time (TUI-suspended). Enables batches
/// (e.g. "upscale every photo in this album") over the same run functions the menus use.
enum Job {
    Ml(mledit::MlJob),
    Vision(vision::VisionOp, PathBuf),
}

/// Where the image-view info panel sits: hidden, on the right (`i`), or across the bottom (`I`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum InfoPos {
    Off,
    Right,
    Bottom,
}

/// Diagnostic overlay baked into the image view (`o` cycles it).
#[derive(Clone, Copy, PartialEq, Eq)]
enum OverlayMode {
    Off,
    /// "Zebras": blown highlights → red, crushed shadows → blue.
    Clipping,
    /// Focus peaking: high-gradient (in-focus) edges → green.
    FocusPeak,
}

/// Which quickhelp overlay is showing (opened via the `Ctrl-B` leader).
#[derive(Clone, Copy, PartialEq, Eq)]
enum HelpKind {
    /// Key chords (the keyboard shortcuts).
    Chords,
    /// Named commands / actions.
    Commands,
}

/// A command in the Edit palette (a pixel op, undo/redo/revert, or the interactive free-form crop).
#[derive(Clone)]
enum EditCmd {
    Op(edit::EditOp),
    /// Open the interactive +/- slider for a scalar adjustment (the op carries the template value).
    Adjust(edit::EditOp),
    FreeCrop,
    CropExact,  // prompts for WxH pixels
    ResizeExact, // prompts for WxH (or N) pixels
    Layers,     // interactive layer compositing
    Levels,     // interactive black/white/gamma editor
    Curve,      // interactive tone-curve editor
    History,    // step through / trim the edit stack
    Duplicate,  // duplicate the target image(s) within the album
    Look(usize), // apply a built-in look preset (index into look_presets)
    CopyEdits,   // copy this image's edit stack
    PasteEdits,  // paste the copied edits onto the targets
    SavePreset,  // save this image's edits as a named preset
    ApplyPreset, // pick a preset and apply to the targets
    Straighten, // prompts for degrees
    StripExif,  // strip file metadata (confirm)
    RedactGps,  // remove only GPS, keep the rest of the EXIF (confirm)
    Convert,    // prompts for format / size
    Watermark,  // prompts for text (+ optional font)
    Lut,        // prompts for a .cube path
    FindDups,   // group near-duplicates (perceptual dHash) → a review smart view
    QualityCull, // reject soft / badly-exposed frames (non-AI)
    MoveToAlbum { copy: bool }, // move / copy the targets into another album (prompts for the path)
    Trash,       // soft-delete the targets into .trash
    OpenTrash,   // browse .trash
    RestoreTrash, // restore everything from .trash
    EmptyTrash,  // permanently purge .trash (confirms)
    ClearThumbs, // purge the whole persistent thumbnail cache
    Retouch(PickOp), // enter the crosshair pick-mode for a retouch op
    Meta(EditField), // edit a per-image metadata field (title / author / copyright / date / geotag)
    FacePolish, // AI-detect faces (SCRFD), then open the 0–100% skin-smoothing slider
    // AI "create" ops (prompt-driven generation → a new album image). These load a model.
    Generate,     // txt2img from a prompt
    Img2imgCreate, // img2img: transform the cursor image under a prompt (resident worker)
    Portrait,     // portrait from a prompt (+ the cursor image as an optional identity face)
    Multiperson,  // multi-person scene from a prompt + the selected images as people
    Undo,
    Redo,
    Revert,
}

/// The Edit palette's command list: `(searchable label, action)`.
/// The category keys for the `Ctrl-B` edit chords (the first key after the leader, in image view).
/// Each avoids the global leader keys (h/H/t/v/p/l/L).
fn chord_categories() -> [(char, &'static str); 11] {
    [
        ('g', "geometry"),
        ('c', "crop"),
        ('a', "adjust (light/tone)"),
        ('k', "colour"),
        ('x', "effects / detail"),
        ('e', "edit stack"),
        ('m', "manage"),
        ('s', "stylize (looks & filters)"),
        ('n', "AI create (generate / portrait / scene)"),
        ('r', "retouch (heal / clone / dodge / perspective)"),
        ('d', "metadata (title / author / copyright / date / geotag)"),
    ]
}

/// Built-in "look" presets — a named, fixed sequence of edits applied in one keystroke.
fn look_presets() -> Vec<(&'static str, &'static str, Vec<edit::EditOp>)> {
    use edit::EditOp::*;
    vec![
        ("look: vintage / faded film", "sv", vec![
            Curve { pts: [24, 72, 128, 188, 232] }, Warmth(15), Saturation(-15), Vignette(22), Grain(14),
        ]),
        ("look: lomo", "sl", vec![Saturation(32), Contrast(16), Vignette(45), Warmth(10)]),
        ("look: cross-process", "sc", vec![SplitTone(32), Saturation(22), Contrast(14), HueRotate(-8)]),
        ("look: noir (b&w)", "sn", vec![Grayscale, Contrast(26), Vignette(34)]),
        ("look: pop-art", "sp", vec![Posterize(62), Saturation(55), Contrast(20)]),
        ("look: golden hour", "sd", vec![Warmth(30), Brilliance(16), Saturation(12), Vignette(14)]),
        ("look: old photo", "sa", vec![Sepia, Contrast(-8), Curve { pts: [30, 78, 130, 186, 226] }, Vignette(28), Grain(22)]),
        ("look: daguerreotype", "sq", vec![Grayscale, Clahe(40), Contrast(18), Vignette(40), Grain(16)]),
        // Apple Photos–style looks (palette-only — search "vivid", "dramatic", "mono", "noir").
        ("look: vivid", "", vec![Saturation(32), Contrast(12), Brilliance(10)]),
        ("look: vivid warm", "", vec![Saturation(32), Contrast(12), Brilliance(10), Warmth(18)]),
        ("look: vivid cool", "", vec![Saturation(32), Contrast(12), Brilliance(10), Warmth(-18)]),
        ("look: dramatic", "", vec![Contrast(30), Saturation(-10), Highlights(-16), Shadows(-10), Definition(20)]),
        ("look: dramatic warm", "", vec![Contrast(30), Saturation(-8), Highlights(-16), Shadows(-10), Definition(20), Warmth(16)]),
        ("look: dramatic cool", "", vec![Contrast(30), Saturation(-8), Highlights(-16), Shadows(-10), Definition(20), Warmth(-16)]),
        ("look: mono", "", vec![Grayscale, Contrast(10)]),
        ("look: silvertone", "", vec![Grayscale, Contrast(8), Brilliance(14), Warmth(8)]),
        // iOS / Apple Photos classic filter names (palette-only — search "fade", "chrome", …).
        ("look: fade", "", vec![Contrast(-14), Saturation(-22), Blackpoint(-24), Warmth(8)]),
        ("look: chrome", "", vec![Brilliance(14), Contrast(14), Saturation(14), Highlights(-8)]),
        ("look: process", "", vec![SplitTone(24), Saturation(10), Contrast(10), HueRotate(6)]),
        ("look: transfer", "", vec![Warmth(24), Saturation(20), Contrast(12), Vignette(12)]),
        ("look: instant", "", vec![Warmth(14), Tint(-10), Blackpoint(-18), Contrast(-8), Vignette(20)]),
        ("look: tonal (soft b&w)", "", vec![Grayscale, Contrast(-6), Brilliance(10)]),
        // Cinematic extras.
        ("look: bleach bypass", "", vec![Saturation(-45), Contrast(28), Brilliance(-8), Definition(16)]),
        ("look: teal & orange", "", vec![SplitTone(-26), Warmth(10), Saturation(16), Contrast(12)]),
    ]
}

/// The full edit command table: `(label, chord, command)`. The 2-char `chord` is `Ctrl-B <cat><item>`
/// (image view); it's the single source of truth for the palette, the chord dispatch, and KEYMAP.md.
fn edit_commands() -> Vec<(&'static str, &'static str, EditCmd)> {
    use edit::EditOp::*;
    let mut v = vec![
        // Geometry (g)
        ("rotate clockwise ⟳", "gr", EditCmd::Op(RotateCw)),
        ("rotate counter-clockwise ⟲", "gl", EditCmd::Op(RotateCcw)),
        ("rotate 180°", "g2", EditCmd::Op(Rotate180)),
        ("flip horizontal", "gh", EditCmd::Op(FlipH)),
        ("flip vertical", "gv", EditCmd::Op(FlipV)),
        ("grayscale / desaturate", "gg", EditCmd::Op(Grayscale)),
        ("auto-enhance (auto levels + colour)", "ga", EditCmd::Op(AutoEnhance)),
        ("straighten (rotate by degrees)", "gs", EditCmd::Straighten),
        ("keystone vertical (fix verticals)…", "gk", EditCmd::Adjust(Keystone { axis: 0, amount: 0 })),
        ("keystone horizontal…", "gK", EditCmd::Adjust(Keystone { axis: 1, amount: 0 })),
        ("lens distortion (barrel / pincushion)…", "gd", EditCmd::Adjust(LensDistort(0))),
        // Crop (c)
        ("crop free-form (interactive)", "cf", EditCmd::FreeCrop),
        ("crop to exact size (WxH px)", "cx", EditCmd::CropExact),
        ("resize to exact size (WxH or N px)", "cz", EditCmd::ResizeExact),
        ("crop to square 1:1", "cs", EditCmd::Op(CropSquare)),
        ("crop 4:5 (portrait)", "c4", EditCmd::Op(CropAspect { w: 4, h: 5 })),
        ("crop 5:4", "c5", EditCmd::Op(CropAspect { w: 5, h: 4 })),
        ("crop 3:2 (photo)", "c3", EditCmd::Op(CropAspect { w: 3, h: 2 })),
        ("crop 2:3 (portrait)", "c2", EditCmd::Op(CropAspect { w: 2, h: 3 })),
        ("crop 16:9 (wide)", "cw", EditCmd::Op(CropAspect { w: 16, h: 9 })),
        ("crop 9:16 (tall)", "ct", EditCmd::Op(CropAspect { w: 9, h: 16 })),
        ("border (white frame)", "cb", EditCmd::Op(Border { aspect_w: 0, aspect_h: 0, mode: 1 })),
        ("border (black frame)", "", EditCmd::Op(Border { aspect_w: 0, aspect_h: 0, mode: 0 })),
        ("letterbox 1:1 (black)", "", EditCmd::Op(Border { aspect_w: 1, aspect_h: 1, mode: 0 })),
        ("letterbox 16:9 (black)", "", EditCmd::Op(Border { aspect_w: 16, aspect_h: 9, mode: 0 })),
        ("letterbox 4:5 (blurred bg)", "", EditCmd::Op(Border { aspect_w: 4, aspect_h: 5, mode: 2 })),
        ("circle crop (black)", "co", EditCmd::Op(CropCircle(0))),
        ("circle crop (white)", "", EditCmd::Op(CropCircle(1))),
        // Adjust — light/tone (a); scalar sliders
        ("brightness…", "ab", EditCmd::Adjust(Brightness(0))),
        ("contrast…", "ac", EditCmd::Adjust(Contrast(0))),
        ("exposure…", "ae", EditCmd::Adjust(Exposure(0))),
        ("brilliance…", "ar", EditCmd::Adjust(Brilliance(0))),
        ("highlights…", "ah", EditCmd::Adjust(Highlights(0))),
        ("midrange…", "am", EditCmd::Adjust(Midrange(0))),
        ("shadows…", "as", EditCmd::Adjust(Shadows(0))),
        ("black point…", "ak", EditCmd::Adjust(Blackpoint(0))),
        ("levels (black / white / gamma)…", "al", EditCmd::Levels),
        ("curves (tone curve)…", "au", EditCmd::Curve),
        ("CLAHE (adaptive contrast)…", "aq", EditCmd::Adjust(Clahe(0))),
        // Local (masked) adjustments — the base op applied through a linear / radial mask; slider = amount.
        ("local exposure — graduated (top)…", "ag", EditCmd::Adjust(LocalAdjust { adjust: 0, amount: 0, shape: 0, dir: 0 })),
        ("local exposure — graduated (bottom)…", "", EditCmd::Adjust(LocalAdjust { adjust: 0, amount: 0, shape: 0, dir: 1 })),
        ("local saturation — graduated (top)…", "", EditCmd::Adjust(LocalAdjust { adjust: 3, amount: 0, shape: 0, dir: 0 })),
        ("local warmth — graduated (top)…", "", EditCmd::Adjust(LocalAdjust { adjust: 4, amount: 0, shape: 0, dir: 0 })),
        ("local exposure — radial (centre)…", "ai", EditCmd::Adjust(LocalAdjust { adjust: 0, amount: 0, shape: 1, dir: 0 })),
        ("local saturation — radial (centre)…", "", EditCmd::Adjust(LocalAdjust { adjust: 3, amount: 0, shape: 1, dir: 0 })),
        ("local exposure — radial (edges)…", "", EditCmd::Adjust(LocalAdjust { adjust: 0, amount: 0, shape: 1, dir: 1 })),
        ("local blur — radial (edges → focus)…", "", EditCmd::Adjust(LocalAdjust { adjust: 7, amount: 0, shape: 1, dir: 1 })),
        // Colour (k)
        ("saturation…", "ks", EditCmd::Adjust(Saturation(0))),
        ("vibrance…", "kv", EditCmd::Adjust(Vibrance(0))),
        ("warmth (warm / cool)…", "kw", EditCmd::Adjust(Warmth(0))),
        ("Kelvin white balance…", "kk", EditCmd::Adjust(Kelvin(0))),
        ("auto white balance (gray-world)…", "ka", EditCmd::Adjust(AutoWhiteBalance(100))),
        ("gray-point white balance (sample centre)", "ke", EditCmd::Op(GrayPointWB { x: 500, y: 500 })),
        ("tint (magenta / green)…", "kt", EditCmd::Adjust(Tint(0))),
        ("hue rotate…", "kh", EditCmd::Adjust(HueRotate(0))),
        ("split-tone…", "kp", EditCmd::Adjust(SplitTone(0))),
        ("selective colour: boost reds", "kr", EditCmd::Op(SelectiveColor { hue: 0, sat: 45 })),
        ("selective colour: mute reds", "kR", EditCmd::Op(SelectiveColor { hue: 0, sat: -55 })),
        ("selective colour: boost greens", "kg", EditCmd::Op(SelectiveColor { hue: 120, sat: 45 })),
        ("selective colour: mute greens", "kG", EditCmd::Op(SelectiveColor { hue: 120, sat: -55 })),
        ("selective colour: boost blues", "kb", EditCmd::Op(SelectiveColor { hue: 240, sat: 45 })),
        ("selective colour: mute blues", "kB", EditCmd::Op(SelectiveColor { hue: 240, sat: -55 })),
        // HSL per-band: the other five bands (palette-only) + per-band lightness.
        ("selective colour: boost oranges", "", EditCmd::Op(SelectiveColor { hue: 30, sat: 45 })),
        ("selective colour: boost yellows", "", EditCmd::Op(SelectiveColor { hue: 60, sat: 45 })),
        ("selective colour: boost aquas", "", EditCmd::Op(SelectiveColor { hue: 180, sat: 45 })),
        ("selective colour: boost purples", "", EditCmd::Op(SelectiveColor { hue: 270, sat: 45 })),
        ("selective colour: boost magentas", "", EditCmd::Op(SelectiveColor { hue: 300, sat: 45 })),
        ("HSL: brighten reds", "", EditCmd::Op(SelectiveLum { hue: 0, lum: 40 })),
        ("HSL: darken reds", "", EditCmd::Op(SelectiveLum { hue: 0, lum: -40 })),
        ("HSL: brighten blues (sky)", "", EditCmd::Op(SelectiveLum { hue: 240, lum: 40 })),
        ("HSL: darken blues (sky)", "kd", EditCmd::Op(SelectiveLum { hue: 240, lum: -40 })),
        ("HSL: darken greens (foliage)", "", EditCmd::Op(SelectiveLum { hue: 120, lum: -35 })),
        // Effects / detail (x)
        ("definition (clarity)…", "xd", EditCmd::Adjust(Definition(0))),
        ("blur (soft focus)…", "xf", EditCmd::Adjust(Blur(50))),
        ("bloom / glow…", "xo", EditCmd::Adjust(Bloom(60))),
        ("sharpen / soften…", "xs", EditCmd::Adjust(Sharpen(0))),
        ("noise reduction…", "xn", EditCmd::Adjust(NoiseReduction(0))),
        ("bilateral denoise (edge-preserving)…", "xj", EditCmd::Adjust(Bilateral(80))),
        ("chromatic aberration removal (defringe)…", "xA", EditCmd::Adjust(ChromaticAberration(60))),
        ("tilt-shift / miniature…", "xm", EditCmd::Adjust(TiltShift(70))),
        ("motion blur — horizontal…", "xC", EditCmd::Adjust(MotionBlur { angle: 0, strength: 60 })),
        ("motion blur — vertical…", "", EditCmd::Adjust(MotionBlur { angle: 90, strength: 60 })),
        ("motion blur — diagonal…", "", EditCmd::Adjust(MotionBlur { angle: 45, strength: 60 })),
        ("zoom blur (radial)…", "xw", EditCmd::Adjust(ZoomBlur(60))),
        ("spin blur (rotational)…", "xq", EditCmd::Adjust(SpinBlur(60))),
        ("film grain…", "xg", EditCmd::Adjust(Grain(0))),
        ("despeckle (median)…", "xk", EditCmd::Adjust(Despeckle(0))),
        ("dehaze…", "xz", EditCmd::Adjust(Dehaze(0))),
        ("enhance sky (auto-mask polarizer)…", "xy", EditCmd::Adjust(EnhanceSky(100))),
        ("face polish (auto-detect + smooth)…", "xc", EditCmd::FacePolish),
        ("vignette…", "xv", EditCmd::Adjust(Vignette(0))),
        ("radial dodge / burn…", "xr", EditCmd::Adjust(Radial(0))),
        ("graduated ND (from top)…", "xt", EditCmd::Adjust(GradND { dir: 0, strength: 0 })),
        ("graduated ND (from bottom)…", "xb", EditCmd::Adjust(GradND { dir: 1, strength: 0 })),
        ("graduated ND (from left)…", "xl", EditCmd::Adjust(GradND { dir: 2, strength: 0 })),
        ("graduated ND (from right)…", "xR", EditCmd::Adjust(GradND { dir: 3, strength: 0 })),
        ("invert (negative)", "xi", EditCmd::Op(Invert)),
        ("film negative → positive (remove orange mask)", "xN", EditCmd::Op(FilmNegative)),
        ("B&W: red filter (dramatic sky)", "xB", EditCmd::Op(ChannelMixerBW { r: 75, g: 20, b: 5 })),
        ("B&W: green filter (skin / foliage)", "", EditCmd::Op(ChannelMixerBW { r: 25, g: 65, b: 10 })),
        ("B&W: blue filter (airy)", "", EditCmd::Op(ChannelMixerBW { r: 10, g: 25, b: 65 })),
        ("B&W: orange filter (portrait)", "", EditCmd::Op(ChannelMixerBW { r: 55, g: 40, b: 5 })),
        ("B&W: luminance (neutral)", "", EditCmd::Op(ChannelMixerBW { r: 30, g: 59, b: 11 })),
        ("sepia", "xe", EditCmd::Op(Sepia)),
        ("duotone", "xu", EditCmd::Op(Duotone)),
        ("posterize…", "xp", EditCmd::Adjust(Posterize(0))),
        ("solarize…", "xa", EditCmd::Adjust(Solarize(0))),
        ("threshold (black & white)", "xh", EditCmd::Op(Threshold(128))),
        // Edit stack (e)
        ("layers — overlay / compose images", "ey", EditCmd::Layers),
        ("edit history (step / trim)…", "eh", EditCmd::History),
        ("duplicate image in album", "ed", EditCmd::Duplicate),
        ("copy edits (from this image)", "ec", EditCmd::CopyEdits),
        ("paste edits (to selection / cursor)", "ev", EditCmd::PasteEdits),
        ("save edits as preset…", "es", EditCmd::SavePreset),
        ("apply preset…", "ea", EditCmd::ApplyPreset),
        ("undo", "eu", EditCmd::Undo),
        ("redo", "eo", EditCmd::Redo),
        ("revert to original", "e0", EditCmd::Revert),
        // Manage (m)
        ("strip metadata (EXIF / GPS)", "mm", EditCmd::StripExif),
        ("redact GPS only (keep other EXIF)", "mg", EditCmd::RedactGps),
        ("convert format / resize (jpg·png·webp)", "mc", EditCmd::Convert),
        ("watermark / caption (burn in text)…", "mw", EditCmd::Watermark),
        ("apply LUT (.cube colour grade)…", "mu", EditCmd::Lut),
        ("find near-duplicates (perceptual hash)", "mf", EditCmd::FindDups),
        ("cull soft / badly-exposed (non-AI)", "mq", EditCmd::QualityCull),
        ("move to album (path)…", "mo", EditCmd::MoveToAlbum { copy: false }),
        ("copy to album (path)…", "mp", EditCmd::MoveToAlbum { copy: true }),
        ("move to trash (soft-delete)", "mt", EditCmd::Trash),
        ("browse trash", "mb", EditCmd::OpenTrash),
        ("restore all from trash", "", EditCmd::RestoreTrash),
        ("empty trash (permanent)…", "", EditCmd::EmptyTrash),
        ("clear thumbnail cache (all — reclaims disk)", "", EditCmd::ClearThumbs),
        // Stylize — algorithmic filters (s). All open the slider (strength 0..100); numbered variants
        // are palette-only (empty chord).
        ("pencil sketch…", "sk", EditCmd::Adjust(PencilSketch(100))),
        ("charcoal…", "sg", EditCmd::Adjust(Charcoal(100))),
        ("cartoon / comic…", "st", EditCmd::Adjust(Cartoon(100))),
        ("emboss…", "se", EditCmd::Adjust(Emboss(100))),
        ("halftone / newsprint…", "sf", EditCmd::Adjust(Halftone(100))),
        ("pixelate / mosaic…", "sx", EditCmd::Adjust(Pixelate(0))),
        ("cross-hatch…", "sy", EditCmd::Adjust(Crosshatch(100))),
        ("crystallize (Voronoi / low-poly)…", "sz", EditCmd::Adjust(Crystallize(100))),
        ("gradient map: warm…", "sm", EditCmd::Adjust(GradientMap { style: 1, strength: 100 })),
        ("gradient map: cyanotype…", "", EditCmd::Adjust(GradientMap { style: 2, strength: 100 })),
        ("gradient map: fire…", "", EditCmd::Adjust(GradientMap { style: 3, strength: 100 })),
        ("gradient map: teal-orange…", "", EditCmd::Adjust(GradientMap { style: 4, strength: 100 })),
        ("false colour: thermal…", "sb", EditCmd::Adjust(FalseColor { style: 1, strength: 100 })),
        ("false colour: infrared…", "", EditCmd::Adjust(FalseColor { style: 2, strength: 100 })),
        ("false colour: night-vision…", "", EditCmd::Adjust(FalseColor { style: 3, strength: 100 })),
        ("ink: European…", "si", EditCmd::Adjust(Ink { style: 1, strength: 100 })),
        ("ink: Japanese sumi-e…", "sj", EditCmd::Adjust(Ink { style: 2, strength: 100 })),
        ("ink: Chinese wash…", "sh", EditCmd::Adjust(Ink { style: 3, strength: 100 })),
        ("ink: Russian icon (tempera)…", "sr", EditCmd::Adjust(Ink { style: 4, strength: 100 })),
        ("oil paint 1…", "", EditCmd::Adjust(OilPaint { style: 1, strength: 100 })),
        ("oil paint 2…", "", EditCmd::Adjust(OilPaint { style: 2, strength: 100 })),
        ("oil paint 3…", "so", EditCmd::Adjust(OilPaint { style: 3, strength: 100 })),
        ("oil paint 4…", "", EditCmd::Adjust(OilPaint { style: 4, strength: 100 })),
        ("oil paint 5…", "", EditCmd::Adjust(OilPaint { style: 5, strength: 100 })),
        ("oil paint 6…", "", EditCmd::Adjust(OilPaint { style: 6, strength: 100 })),
        ("oil paint 7…", "", EditCmd::Adjust(OilPaint { style: 7, strength: 100 })),
        ("oil paint 8…", "", EditCmd::Adjust(OilPaint { style: 8, strength: 100 })),
        ("oil paint 9…", "", EditCmd::Adjust(OilPaint { style: 9, strength: 100 })),
        ("oil paint 10…", "", EditCmd::Adjust(OilPaint { style: 10, strength: 100 })),
        ("watercolour 1…", "", EditCmd::Adjust(Watercolor { style: 1, strength: 100 })),
        ("watercolour 2…", "", EditCmd::Adjust(Watercolor { style: 2, strength: 100 })),
        ("watercolour 3…", "", EditCmd::Adjust(Watercolor { style: 3, strength: 100 })),
        ("watercolour 4…", "", EditCmd::Adjust(Watercolor { style: 4, strength: 100 })),
        ("watercolour 5…", "sw", EditCmd::Adjust(Watercolor { style: 5, strength: 100 })),
        ("watercolour 6…", "", EditCmd::Adjust(Watercolor { style: 6, strength: 100 })),
        ("watercolour 7…", "", EditCmd::Adjust(Watercolor { style: 7, strength: 100 })),
        ("watercolour 8…", "", EditCmd::Adjust(Watercolor { style: 8, strength: 100 })),
        ("watercolour 9…", "", EditCmd::Adjust(Watercolor { style: 9, strength: 100 })),
        ("watercolour 10…", "", EditCmd::Adjust(Watercolor { style: 10, strength: 100 })),
        // AI create (n) — prompt-driven generation, landing a new image in the album.
        ("AI generate (txt2img)…", "ng", EditCmd::Generate),
        ("AI img2img (transform this image)…", "ni", EditCmd::Img2imgCreate),
        ("AI portrait (this image as the face)…", "np", EditCmd::Portrait),
        ("AI multiperson scene (selected = people)…", "nm", EditCmd::Multiperson),
        // Retouch (r) — interactive crosshair pick-mode.
        ("spot heal (remove blemish / dust)…", "rh", EditCmd::Retouch(PickOp::SpotHeal)),
        ("clone stamp (copy a region)…", "rc", EditCmd::Retouch(PickOp::Clone)),
        ("red-eye removal…", "re", EditCmd::Retouch(PickOp::RedEye)),
        ("dodge (lighten) brush…", "rd", EditCmd::Retouch(PickOp::Dodge)),
        ("burn (darken) brush…", "rb", EditCmd::Retouch(PickOp::Burn)),
        ("perspective rectify (pick 4 corners)…", "rp", EditCmd::Retouch(PickOp::Perspective)),
        // Metadata (d) — per-image, stored non-destructively in the album record.
        ("set title…", "dt", EditCmd::Meta(EditField::Title)),
        ("set author / creator…", "da", EditCmd::Meta(EditField::Author)),
        ("set copyright…", "dc", EditCmd::Meta(EditField::Copyright)),
        ("set capture date…", "dd", EditCmd::Meta(EditField::DateTaken)),
        ("set geotag (lat, lon)…", "dg", EditCmd::Meta(EditField::Geotag)),
        ("set caption…", "de", EditCmd::Meta(EditField::Caption)),
    ];
    // Stylize (s): built-in look presets appended after the filters.
    for (i, (label, chord, _)) in look_presets().into_iter().enumerate() {
        v.push((label, chord, EditCmd::Look(i)));
    }
    v
}

/// Look up an edit command by its 2-char chord (empty chords are palette-only, never matched).
fn edit_cmd_for_chord(chord: &str) -> Option<EditCmd> {
    edit_commands().into_iter().find(|(_, c, _)| !c.is_empty() && *c == chord).map(|(_, _, cmd)| cmd)
}

/// Edit commands whose label or chord contains `query` (case-insensitive; empty query = all).
fn filtered_edit_commands(query: &str) -> Vec<(&'static str, &'static str, EditCmd)> {
    let q = query.trim().to_lowercase();
    edit_commands()
        .into_iter()
        .filter(|(l, c, _)| q.is_empty() || l.to_lowercase().contains(&q) || c.contains(&q))
        .collect()
}

/// A free-text per-image metadata field editable from the command pane.
#[derive(Clone, Copy)]
enum EditField {
    Caption,
    Notes,
    Title,
    Tags,
    Author,
    Copyright,
    DateTaken,
    Geotag,
}

/// An album-level metadata field editable from the tree (info editor / `t`·`T`).
#[derive(Clone, Copy)]
enum AlbumFieldKind {
    Name,
    Description,
    TagsAdd,
    TagsReplace,
    Cover,
    Sort,
}

/// Sort orders for the album grid (album.hjson `sort`), cycled with `s`.
const SORT_ORDER: [&str; 6] =
    ["name-asc", "name-desc", "date-desc", "date-asc", "rating-desc", "score-desc"];

/// A flattened, displayable tree row.
struct Row {
    path: PathBuf,
    name: String,
    kind: NodeKind,
    count: usize,
    depth: usize,
    expanded: bool,
    has_children: bool,
}

struct App {
    root: LibraryNode,
    root_dir: PathBuf,
    expanded: HashSet<PathBuf>,
    tree_cursor: usize,
    focus: Focus,
    // Tree-name incremental filter (`/`): when active, the tree shows only matching nodes + their
    // ancestor folders.
    tree_filter: String,
    tree_filter_active: bool,
    // Album info panel (`i`, read-only modal) + info editor menu (`I`); both target `info_target`.
    album_info: Option<(String, Vec<String>)>,
    info_editor: bool,
    info_target: Option<PathBuf>,

    // Album grid.
    album_dir: Option<PathBuf>,
    album_meta: hjson::AlbumMeta,
    album_paths: Vec<PathBuf>,
    /// Filtered view: indices into `album_paths` that pass `filter` (all when empty). `album_cursor`
    /// indexes into this, so navigation/rendering operate on the filtered set.
    view: Vec<usize>,
    filter: String,
    filter_active: bool,
    album_cursor: usize,
    /// Multi-selection stored as `album_paths` indices (survives filter changes).
    selected: HashSet<usize>,
    mode: AlbumMode,
    info: InfoPos,
    overlay: OverlayMode,
    /// Image-view zoom (1.0 = fit; center crop-zoom). Resets to fit on navigation.
    zoom: f32,
    // View analysis (RFC §Phase 6): histogram + exposure/focus stats panel in the image view (`H`).
    show_analysis: bool,
    analysis: Option<analysis::Analysis>,
    view_proto: Option<StatefulProtocol>,
    view_exif: Option<hjson::ExifRecord>,
    /// A compact luma-histogram sparkline of the currently-displayed image, for the top bar (updates
    /// live under the edit previews).
    view_spark: Option<String>,
    /// Cached decoded working-resolution base for the image view, keyed by `(path, bound)`, so the
    /// interactive edit modes re-apply the op to a cached image instead of re-decoding the (possibly
    /// huge) file on every keypress.
    view_base: Option<(PathBuf, u32, image::DynamicImage)>,
    thumbs: HashMap<PathBuf, StatefulProtocol>,
    cols: usize,
    thumb_px: u32,

    // Smart-album view (RFC §8: library-wide saved searches). When `smart` is set, the grid holds
    // images collected from many albums; records/writes route through the path-keyed maps below
    // instead of `album_meta`.
    smart_albums: Vec<hjson::SmartAlbum>,
    smart: Option<String>,       // active smart-view label (None = a real album is open)
    smart_query: String,         // the active view's query (filter grammar, or NL search text)
    smart_is_search: bool,       // true = relevance-ranked semantic search; false = filter query
    lookalike_of: Option<PathBuf>, // when Some, the smart view is a "similar-to" ranking
    lookalike_clip: bool,          // true = CLIP semantic (snapshot, no auto-refresh); false = perceptual
    pending_clip_lookalike: Option<PathBuf>, // queued CLIP lookalike (event loop runs it, TUI-suspended)
    smart_src: HashMap<PathBuf, PathBuf>, // image path → its source album dir (for write routing)
    smart_rec: HashMap<PathBuf, hjson::ImageRecord>, // image path → its record (badges/filter/sort)

    // T1 pixel-edit palette (RFC §Phase 3) — a searchable/scrollable modal over the cursor image.
    edit_menu: bool,
    edit_query: String,
    edit_cursor: usize,
    edit_visible: usize, // rows the palette can show (set at draw; used for PageUp/Down)
    // Interactive free-form crop (from the Edit palette): rect in [0,1] fractions (x, y, w, h).
    crop_mode: bool,
    crop_rect: (f32, f32, f32, f32),
    // Interactive retouch pick-mode (heal / clone / red-eye / dodge-burn / perspective).
    pick: Option<PickState>,
    // Interactive scalar-adjustment slider (brightness/contrast/…): the op carries the live value,
    // shown on a +/- bar with a live preview.
    adjust_mode: bool,
    adjust_op: Option<edit::EditOp>,
    // Before/after: show the pristine original (backup) instead of the edited file (`\`).
    show_original: bool,
    // Edit-history scrubber: step through / trim the edit stack over a decoded pristine original.
    history_mode: bool,
    history_ops: Vec<hjson::EditEntry>,
    history_pos: usize,
    history_orig: Option<image::DynamicImage>,
    // Interactive curves editor: 5 output points (input 0/64/128/192/255) + the selected point.
    curve_mode: bool,
    curve_pts: [i32; 5],
    curve_sel: usize,
    // Interactive levels editor (from the Edit palette): black/white input points (0..255) + midtone
    // gamma (×100), with a live preview. `lv_sel` = which handle the ←/→ keys adjust (0/1/2).
    levels_mode: bool,
    lv_black: i32,
    lv_white: i32,
    lv_gamma: i32,
    lv_sel: usize,
    // Interactive layer compositing (Phase 8): an overlay stack over the cursor image. The stack is
    // persisted on the image's record (`layers`); `layer_active` is the selected layer.
    layer_mode: bool,
    layers: Vec<layers::Layer>,
    layer_active: usize,
    /// Mask-adjust sub-mode: arrows reposition the active layer's shape mask (rather than the layer).
    mask_adjust: bool,
    // T2 ML-edit menu (RFC §Phase 4) + a queued job the event loop runs with the TUI suspended.
    ml_menu: bool,
    // Vision + AI menu (RFC §Phase 7).
    ai_menu: bool,
    // Heavy jobs (ML edits + vision) drained one per tick — a batch of one from a menu, or many
    // from a natural-language pipeline (`:`).
    jobs: VecDeque<Job>,
    // CLIP visual search (RFC §Phase 7): a queued text query + the in-session embedding cache.
    pending_visual: Option<String>,
    clip_cache: visual_search::Cache,
    // Track A AI jobs, run TUI-suspended by the event loop: aesthetic auto-cull (keep top N),
    // analyze-and-generate (describe → img2img), and face-scan (detect + group).
    pending_cull: Option<usize>,
    pending_analyze: Option<PathBuf>,
    pending_face_scan: bool,
    // Face-polish: detect faces on the cursor image, then open the skin-smoothing slider.
    pending_face_polish: bool,
    // AI "create" op (generate / portrait / multiperson), run TUI-suspended → a new album image.
    pending_create: Option<(CreateOp, PathBuf)>,
    // Natural-language command pipeline (`:`): a raw query awaiting the LLM planner, and a parsed
    // plan awaiting the user's y/N confirmation.
    pending_nl: Option<String>,
    pending_plan: Option<nl::CommandPlan>,

    // Survey / compare (RFC §Phase 5): a small set of images decoded side-by-side, with a focus.
    compare: Vec<(PathBuf, StatefulProtocol)>,
    compare_cursor: usize,

    // Curation undo/redo: snapshots of `album_meta` before each metadata mutation (normal-album
    // mode only). Coarse but general — covers ratings/flags/reject/colour/tags/captions.
    undo_stack: Vec<hjson::AlbumMeta>,
    redo_stack: Vec<hjson::AlbumMeta>,
    /// Redo stack for the cursor image's T1 pixel edits (cleared on a new edit / navigation).
    edit_redo: Vec<edit::EditOp>,

    // Stacking (RFC §Phase 5): when on, derivative variants collapse under their base in the grid.
    stack_view: bool,
    // Timeline (RFC §Phase 5): a modal list of date buckets over the current view.
    timeline: bool,
    tl_buckets: Vec<(String, usize, usize)>, // (label, first view-position, count)
    tl_cursor: usize,

    // `Ctrl-B` leader prefix (tmux-style) + the quickhelp overlay it opens.
    leader: bool,
    // Edit-chord second stage (image view): after `Ctrl-B <category>`, this holds the category key
    // while we wait for the item key.
    chord_prefix: Option<char>,
    help: Option<HelpKind>,
    // Copy/paste edits + presets: the copied edit stack, and the apply-preset picker.
    edit_clipboard: Vec<hjson::EditEntry>,
    preset_browser: bool,
    presets_list: Vec<hjson::EditPreset>,
    preset_cursor: usize,
    // Tag browser (Ctrl-B t): pick a tag from the album to filter by.
    tag_browser: bool,
    tags_list: Vec<(String, usize)>,
    tag_cursor: usize,
    // Version browser (Ctrl-B v): save/restore per-image snapshots for the cursor image.
    version_browser: bool,
    versions_list: Vec<u32>,
    version_cursor: usize,
    version_target: Option<(PathBuf, String, PathBuf)>, // (album dir, filename, image path)

    // Command pane input (RFC §11).
    cmd_active: bool,
    cmd_prompt: String,
    cmd_buffer: String,
    pending: Option<PendingCmd>,

    picker: Picker,
    status: String,

    // Live filesystem watch (RFC §23) + debounce.
    watch: Option<watcher::Watch>,
    dirty_since: Option<Instant>,

    // AI-track Phase A: a background resident ML worker + the in-flight job the event loop drains
    // inline. `rt` is the manager's tokio handle (captured in `run_with`) so the worker can block_on
    // async pipeline loads. The worker is spawned lazily on the first ML op.
    rt: Option<Handle>,
    ml_worker: Option<mlworker::MlWorker>,
    active_ml: Option<ActiveMl>,

    // Memory status indicator (throttled ~1 Hz): the kernel pressure level (trustworthy on macOS)
    // plus free/total RAM, shown in the top bar to steer the user away from AI-heavy ops when low.
    mem_pressure: crate::hw::Pressure,
    mem_free_gb: f64,
    mem_total_gb: f64,
    mem_checked: Option<Instant>,
}

/// A prompt-driven AI "create" op — generation that lands a **new** image in the album (as opposed to
/// the [`mledit`] ops that transform an existing one). Run TUI-suspended via the `crate::api` builders.
#[derive(Clone)]
enum CreateOp {
    /// txt2img from a prompt.
    Generate { prompt: String, model: String, size: Option<(u32, u32)> },
    /// Portrait from a prompt, optionally using `photo` (the cursor image) as the identity face.
    Portrait { prompt: String, model: String, size: Option<(u32, u32)>, photo: Option<PathBuf> },
    /// Multi-person scene from a prompt; `people` are the selected images' identity faces.
    Multiperson { prompt: String, model: String, size: Option<(u32, u32)>, people: Vec<PathBuf> },
}

/// Parse an AI-create prompt spec: `prompt [| model [WxH | N]]`. The part after `|` picks the model
/// (an SD-family alias) and, optionally, the output size. Defaults: model `sdxl`, pipeline-default
/// size. So `a red fox | sd15 512` → ("a red fox", "sd15", 640×… no — (512,512)).
fn parse_gen_spec(arg: &str) -> (String, String, Option<(u32, u32)>) {
    match arg.split_once('|') {
        Some((p, rest)) => {
            let mut it = rest.split_whitespace();
            let model = it.next().map(|m| m.to_lowercase()).unwrap_or_else(|| "sdxl".into());
            let size = it.next().and_then(parse_gen_size_tok);
            (p.trim().to_string(), model, size)
        }
        None => (arg.trim().to_string(), "sdxl".into(), None),
    }
}

/// `WxH` or a single `N` (square) → pixel size, clamped to a sane range.
fn parse_gen_size_tok(tok: &str) -> Option<(u32, u32)> {
    let clamp = |n: u32| n.clamp(256, 2048);
    if let Some((w, h)) = tok.split_once(['x', 'X', '×']) {
        Some((clamp(w.trim().parse().ok()?), clamp(h.trim().parse().ok()?)))
    } else {
        let n = clamp(tok.trim().parse().ok()?);
        Some((n, n))
    }
}

/// The IP-Adapter identity strategy that matches a model family (SDXL vs SD 1.5).
fn identity_for_model(model: &str) -> crate::api::IdentityKind {
    if model.contains("xl") {
        crate::api::IdentityKind::PlusFaceSdxl
    } else {
        crate::api::IdentityKind::PlusFace
    }
}

/// A retouch operation driven by the interactive crosshair pick-mode.
#[derive(Clone, Copy, PartialEq)]
enum PickOp {
    SpotHeal,
    Clone,
    RedEye,
    Dodge,
    Burn,
    Perspective,
}

impl PickOp {
    /// How many points the user confirms before the op applies.
    fn points_needed(self) -> usize {
        match self {
            PickOp::Clone => 2,       // source, then destination
            PickOp::Perspective => 4, // TL, TR, BR, BL
            _ => 1,
        }
    }
    fn uses_radius(self) -> bool {
        self != PickOp::Perspective
    }
    fn label(self) -> &'static str {
        match self {
            PickOp::SpotHeal => "spot heal",
            PickOp::Clone => "clone stamp",
            PickOp::RedEye => "red-eye",
            PickOp::Dodge => "dodge (lighten)",
            PickOp::Burn => "burn (darken)",
            PickOp::Perspective => "perspective rectify",
        }
    }
}

/// Interactive pick-mode state: the live crosshair, confirmed points, and the brush radius.
struct PickState {
    op: PickOp,
    pos: (f32, f32),            // crosshair in [0,1]
    points: Vec<(f32, f32)>,    // confirmed points
    radius: f32,                // brush radius as a fraction of the min dimension
}

/// An ML edit running on the resident worker, watched by the event loop.
struct ActiveMl {
    label: String,
    src: PathBuf,
    rx: Receiver<GenMessage>,
    cancel: CancelFlag,
}

impl App {
    fn new(root: LibraryNode, root_dir: PathBuf, picker: Picker, thumb_px: u32) -> Self {
        let mut expanded = HashSet::new();
        expanded.insert(root.path.clone());
        let smart_albums = hjson::read_folder(&root_dir).unwrap_or_default().smart_albums;
        Self {
            root,
            root_dir,
            expanded,
            tree_cursor: 0,
            focus: Focus::Tree,
            tree_filter: String::new(),
            tree_filter_active: false,
            album_info: None,
            info_editor: false,
            info_target: None,
            album_dir: None,
            album_meta: hjson::AlbumMeta::default(),
            album_paths: Vec::new(),
            view: Vec::new(),
            filter: String::new(),
            filter_active: false,
            album_cursor: 0,
            selected: HashSet::new(),
            mode: AlbumMode::Grid,
            info: InfoPos::Off,
            overlay: OverlayMode::Off,
            zoom: 1.0,
            show_analysis: false,
            analysis: None,
            view_proto: None,
            view_exif: None,
            view_spark: None,
            view_base: None,
            thumbs: HashMap::new(),
            cols: 4,
            thumb_px,
            smart_albums,
            smart: None,
            smart_query: String::new(),
            smart_is_search: false,
            lookalike_of: None,
            lookalike_clip: false,
            pending_clip_lookalike: None,
            smart_src: HashMap::new(),
            smart_rec: HashMap::new(),
            edit_menu: false,
            edit_query: String::new(),
            edit_cursor: 0,
            edit_visible: 10,
            crop_mode: false,
            crop_rect: (0.1, 0.1, 0.8, 0.8),
            pick: None,
            adjust_mode: false,
            adjust_op: None,
            show_original: false,
            history_mode: false,
            history_ops: Vec::new(),
            history_pos: 0,
            history_orig: None,
            curve_mode: false,
            curve_pts: [0, 64, 128, 192, 255],
            curve_sel: 0,
            levels_mode: false,
            lv_black: 0,
            lv_white: 255,
            lv_gamma: 100,
            lv_sel: 0,
            layer_mode: false,
            layers: Vec::new(),
            layer_active: 0,
            mask_adjust: false,
            ml_menu: false,
            ai_menu: false,
            jobs: VecDeque::new(),
            pending_visual: None,
            pending_cull: None,
            pending_analyze: None,
            pending_face_scan: false,
            pending_face_polish: false,
            pending_create: None,
            clip_cache: HashMap::new(),
            pending_nl: None,
            pending_plan: None,
            compare: Vec::new(),
            compare_cursor: 0,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            edit_redo: Vec::new(),
            stack_view: false,
            timeline: false,
            tl_buckets: Vec::new(),
            tl_cursor: 0,
            leader: false,
            chord_prefix: None,
            help: None,
            edit_clipboard: Vec::new(),
            preset_browser: false,
            presets_list: Vec::new(),
            preset_cursor: 0,
            tag_browser: false,
            tags_list: Vec::new(),
            tag_cursor: 0,
            version_browser: false,
            versions_list: Vec::new(),
            version_cursor: 0,
            version_target: None,
            cmd_active: false,
            cmd_prompt: String::new(),
            cmd_buffer: String::new(),
            pending: None,
            picker,
            status: String::from("↑/↓ move · →/Enter open · h collapse · Tab pane · q quit"),
            watch: None,
            dirty_since: None,
            rt: None,
            ml_worker: None,
            active_ml: None,
            mem_pressure: crate::hw::Pressure::Unknown,
            mem_free_gb: 0.0,
            mem_total_gb: 0.0,
            mem_checked: None,
        }
    }

    /// Re-walk the library and reload the open album (after a filesystem change). Keeps the album
    /// cursor in range; new thumbnails decode lazily on the next ticks.
    fn rescan(&mut self) {
        if let Ok(root) = library::walk(&self.root_dir) {
            self.root = root;
        }
        // A lookalike view re-ranks by similarity (not a text query) — handle before the smart branch.
        if let Some(qpath) = self.lookalike_of.clone() {
            // CLIP lookalike is expensive (embeddings) — treat it as a snapshot, don't re-run on a
            // filesystem change. Perceptual lookalike is cheap → re-rank.
            if self.lookalike_clip {
                return;
            }
            let cur_path = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned());
            self.open_lookalike(qpath);
            if let Some(cp) = cur_path {
                if let Some(pos) = self.view.iter().position(|&pi| self.album_paths.get(pi) == Some(&cp)) {
                    self.album_cursor = pos;
                }
            }
            return;
        }
        // A smart-album / search view spans the whole library — re-evaluate it, preserving the cursor.
        if let Some(name) = self.smart.clone() {
            let query = self.smart_query.clone();
            let cur_path = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned());
            if self.smart_is_search {
                self.open_search(query);
            } else {
                self.open_smart(name, query);
            }
            if let Some(cp) = cur_path {
                if let Some(pos) = self.view.iter().position(|&pi| self.album_paths.get(pi) == Some(&cp)) {
                    self.album_cursor = pos;
                }
            }
            return;
        }
        if let Some(dir) = self.album_dir.clone() {
            if dir.is_dir() {
                let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.is_file() && library::is_image(p))
                    .collect();
                paths.sort();
                self.thumbs.retain(|k, _| paths.contains(k)); // drop thumbs for removed files
                self.album_paths = paths;
                self.sort_album(); // honour the album's persisted sort order
                self.rebuild_view();
            }
        }
    }

    fn rows(&self) -> Vec<Row> {
        let mut out = Vec::new();
        let filter = self.tree_filter.trim().to_lowercase();
        // Library-wide smart albums come first, as ★ rows at depth 0 (sentinel paths — never
        // touched on disk; the tree handler routes them by name). Hidden when a filter excludes them.
        for sa in &self.smart_albums {
            if filter.is_empty() || sa.name.to_lowercase().contains(&filter) {
                out.push(Row {
                    path: PathBuf::from(format!("\u{1}smart:{}", sa.name)),
                    name: sa.name.clone(),
                    kind: NodeKind::SmartAlbum,
                    count: 0,
                    depth: 0,
                    expanded: false,
                    has_children: false,
                });
            }
        }
        // Whether a subtree contains a name match (so ancestor folders of a match stay visible).
        fn matches(node: &LibraryNode, f: &str) -> bool {
            node.name.to_lowercase().contains(f) || node.children.iter().any(|c| matches(c, f))
        }
        fn rec(node: &LibraryNode, depth: usize, exp: &HashSet<PathBuf>, f: &str, out: &mut Vec<Row>) {
            // When filtering, a node shows if it or a descendant matches, and we descend regardless
            // of the expanded state so matches surface without manual expansion.
            if !f.is_empty() && !matches(node, f) {
                return;
            }
            let is_open = f.is_empty() && exp.contains(&node.path);
            out.push(Row {
                path: node.path.clone(),
                name: node.name.clone(),
                kind: node.kind,
                // An album's badge = its OWN images (what opening it shows); a folder has no images
                // of its own, so it shows the recursive total beneath it. This keeps the badge honest:
                // a mixed album (images + sub-albums) reads `[2]`, not the recursive `[5]`, and its
                // sub-albums appear as expandable children with their own counts.
                count: if node.kind == NodeKind::Album { node.image_count } else { node.total_images() },
                depth,
                expanded: is_open,
                has_children: !node.children.is_empty(),
            });
            if is_open || (!f.is_empty() && !node.children.is_empty()) {
                for c in &node.children {
                    rec(c, depth + 1, exp, f, out);
                }
            }
        }
        rec(&self.root, 0, &self.expanded, &filter, &mut out);
        out
    }

    // ---- Tree album operations (RFC §7.4, extended) ----------------------------------------------

    /// The display name of whatever's open in the album pane — the smart-view label, the open album's
    /// metadata name (fallback to its dir), else the library root.
    fn current_view_name(&self) -> String {
        if let Some(name) = &self.smart {
            return format!("{} {}", if self.smart_is_search { "🔎" } else { "★" }, name);
        }
        if let Some(dir) = &self.album_dir {
            return self
                .album_meta
                .name
                .clone()
                .filter(|n| !n.trim().is_empty())
                .unwrap_or_else(|| dir.file_name().and_then(|n| n.to_str()).unwrap_or("album").to_string());
        }
        self.root_dir.file_name().and_then(|n| n.to_str()).unwrap_or("library").to_string()
    }

    /// The tree cursor's `(path, kind, name)`.
    fn cur_tree_node(&self) -> Option<(PathBuf, NodeKind, String)> {
        self.rows().get(self.tree_cursor).map(|r| (r.path.clone(), r.kind, r.name.clone()))
    }

    /// Collapse an expanded folder, else move the cursor up to the parent row (Left / `h`).
    fn goto_tree_parent(&mut self) {
        let rows = self.rows();
        let Some(cur) = rows.get(self.tree_cursor) else { return };
        if cur.expanded {
            let p = cur.path.clone();
            self.expanded.remove(&p);
            return;
        }
        if let Some(parent) = cur.path.parent().map(|p| p.to_path_buf()) {
            if let Some(i) = rows.iter().position(|r| r.path == parent) {
                self.tree_cursor = i;
            }
        }
    }

    /// An album's metadata (the open album's live copy if it matches, else read from disk).
    fn album_meta_at(&self, dir: &Path) -> hjson::AlbumMeta {
        if self.album_dir.as_deref() == Some(dir) {
            self.album_meta.clone()
        } else {
            hjson::read_album(dir).unwrap_or_default()
        }
    }

    /// Mutate an album's metadata on disk (and in memory if it's the open album).
    fn edit_album_meta_at(&mut self, dir: &Path, f: impl FnOnce(&mut hjson::AlbumMeta)) {
        let mut meta = self.album_meta_at(dir);
        f(&mut meta);
        if let Err(e) = hjson::write_album(dir, &meta) {
            self.status = format!("save failed: {e}");
            return;
        }
        if self.album_dir.as_deref() == Some(dir) {
            self.album_meta = meta;
        }
    }

    /// Image files directly in `dir` (album), or all images under it (folder, when `recursive`).
    fn gather_image_files(dir: &Path, recursive: bool) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_file() && library::is_image(&p) {
                    out.push(p);
                } else if recursive && p.is_dir() {
                    let hidden = p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with('.'));
                    if !hidden {
                        out.extend(Self::gather_image_files(&p, true));
                    }
                }
            }
        }
        out.sort();
        out
    }

    /// Regenerate thumbnails for the tree cursor's album/folder (drop cached + in-memory thumbs).
    fn regen_thumbs_tree(&mut self) {
        let Some((path, kind, _)) = self.cur_tree_node() else { return };
        if kind == NodeKind::SmartAlbum {
            self.status = "not an album".into();
            return;
        }
        let files = Self::gather_image_files(&path, kind == NodeKind::Folder);
        let mut cleared = 0;
        for f in &files {
            for size in [self.thumb_px, 384] {
                if std::fs::remove_file(loader::thumb_cache_path(f, size)).is_ok() {
                    cleared += 1;
                }
            }
            self.thumbs.remove(f);
        }
        self.status =
            format!("regenerating thumbnails: cleared {cleared} cached for {} image(s)", files.len());
    }

    /// Export the tree cursor's images to `dest`.
    fn export_album_files(&mut self, path: &Path, recursive: bool, dest: &str) {
        let files = Self::gather_image_files(path, recursive);
        if files.is_empty() {
            self.status = "no images to export".into();
            return;
        }
        // A trailing integer is a longest-side cap.
        let (dir, max_px) = match dest.trim().rsplit_once(char::is_whitespace) {
            Some((h, last)) if last.parse::<u32>().is_ok() => (h.trim().to_string(), last.parse().ok()),
            _ => (dest.trim().to_string(), None),
        };
        let d = expand_tilde(&dir);
        match export::export(&files, &d, max_px) {
            Ok(n) => self.status = format!("exported {n} image(s) → {}", d.display()),
            Err(e) => self.status = format!("export failed: {e:#}"),
        }
    }

    /// Export + convert the tree cursor's images: `FMT DIR [MAXPX]`.
    fn export_convert_files(&mut self, path: &Path, recursive: bool, arg: &str) {
        let Some((fmt, rest)) = arg.trim().split_once(char::is_whitespace) else {
            self.status = "enter: FMT DIR [maxpx]  (e.g. jpg ~/out 2048)".into();
            return;
        };
        let (dir, max_px) = match rest.trim().rsplit_once(char::is_whitespace) {
            Some((h, last)) if last.parse::<u32>().is_ok() => (h.trim().to_string(), last.parse().ok()),
            _ => (rest.trim().to_string(), None),
        };
        let dest = expand_tilde(&dir);
        if let Err(e) = std::fs::create_dir_all(&dest) {
            self.status = format!("mkdir failed: {e}");
            return;
        }
        let size = max_px.map(scrub::ConvertSize::MaxPx).unwrap_or(scrub::ConvertSize::Keep);
        let files = Self::gather_image_files(path, recursive);
        let (mut ok, mut err) = (0, 0);
        for f in &files {
            match scrub::convert(f, &dest, fmt, size) {
                Ok(_) => ok += 1,
                Err(_) => err += 1,
            }
        }
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status = format!("exported+converted {ok} → {} ({fmt}){tail}", dest.display());
    }

    /// Materialize a smart album (saved search) into a **portable** album at `dest`: copy every
    /// matching image (real file copies — no symlinks/hardlinks) and write a fresh `album.hjson`
    /// carrying each image's curation, so the result is self-contained and movable. Filenames are
    /// de-duplicated across source albums; the replay-only fields (edits/variants/layers, which point
    /// at files that aren't copied) are cleared so each copy is a clean baseline.
    fn materialize_smart(&mut self, name: &str, query: &str, dest: &str) {
        let items: Vec<_> = self
            .collect_library()
            .into_iter()
            .filter(|(p, _, rec)| {
                let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                matches_filter(fname, rec.as_ref(), query)
            })
            .collect();
        if items.is_empty() {
            self.status = "no matches to materialize".into();
            return;
        }
        let d = expand_tilde(dest);
        if let Err(e) = std::fs::create_dir_all(&d) {
            self.status = format!("mkdir failed: {e}");
            return;
        }
        let mut meta = hjson::AlbumMeta {
            name: Some(name.to_string()),
            description: Some(format!("Portable smart album — query: {query}")),
            ..Default::default()
        };
        let mut used: HashSet<String> = HashSet::new();
        let (mut ok, mut err) = (0u32, 0u32);
        for (src, _dir, rec) in &items {
            let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("");
            let make = |s: &str| if ext.is_empty() { s.to_string() } else { format!("{s}.{ext}") };
            let mut fname = make(stem);
            let mut i = 2;
            while used.contains(&fname) || d.join(&fname).exists() {
                fname = make(&format!("{stem}-{i}"));
                i += 1;
            }
            if std::fs::copy(src, d.join(&fname)).is_err() {
                err += 1;
                continue;
            }
            used.insert(fname.clone());
            let dstem = std::path::Path::new(&fname).file_stem().and_then(|s| s.to_str()).unwrap_or(stem);
            // Carry the generation sidecar (<stem>.json) alongside, renamed to the copy's stem.
            let sidecar = src.with_extension("json");
            if sidecar.exists() {
                let _ = std::fs::copy(&sidecar, d.join(format!("{dstem}.json")));
            }
            if let Some(r) = rec {
                let mut r = r.clone();
                r.edits.clear(); // the copy already reflects them
                r.variants.clear(); // derivative files aren't copied
                r.layers.clear();
                meta.images.insert(fname, r);
            }
            ok += 1;
        }
        if let Err(e) = hjson::write_album(&d, &meta) {
            self.status = format!("write album.hjson failed: {e}");
            return;
        }
        // If the destination lands inside the library, surface it in the tree.
        if d.starts_with(&self.root_dir) {
            if let Ok(root) = library::walk(&self.root_dir) {
                self.root = root;
            }
        }
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status = format!("materialized ★ {name} → {} · {ok} copies (no links){tail}", d.display());
    }

    /// Open the read-only album/folder info panel for the tree cursor.
    fn open_album_info(&mut self) {
        let Some((path, kind, name)) = self.cur_tree_node() else { return };
        if kind == NodeKind::SmartAlbum {
            let q = self.smart_albums.iter().find(|s| s.name == name).map(|s| s.query.clone());
            self.album_info = Some((
                format!("★ {name}"),
                vec!["smart album (saved search)".into(), format!("query: {}", q.unwrap_or_default())],
            ));
            return;
        }
        let meta = self.album_meta_at(&path);
        let files = Self::gather_image_files(&path, kind == NodeKind::Folder);
        let total: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
        let flagged = meta.images.values().filter(|r| r.flagged).count();
        let rejected = meta.images.values().filter(|r| r.rejected).count();
        let rated = meta.images.values().filter(|r| r.rating > 0).count();
        let mut lines = vec![
            format!("path:   {}", path.display()),
            format!("kind:   {}", if kind == NodeKind::Folder { "folder" } else { "album" }),
        ];
        if let Some(d) = meta.description.as_deref().filter(|d| !d.is_empty()) {
            lines.push(format!("desc:   {d}"));
        }
        if !meta.tags.is_empty() {
            lines.push(format!("tags:   {}", meta.tags.join(", ")));
        }
        lines.push(format!("images: {}", files.len()));
        lines.push(format!("size:   {}", human_size(total)));
        lines.push(format!("rated {rated}  ·  ⚑ {flagged}  ·  ✗ {rejected}"));
        if let Some(c) = &meta.cover {
            lines.push(format!("cover:  {c}"));
        }
        lines.push(format!("sort:   {}", meta.sort.as_deref().unwrap_or("name-asc")));
        let title = meta.name.clone().unwrap_or(name);
        self.album_info = Some((format!("ⓘ {title}"), lines));
    }

    /// Open the album info editor (a field menu) for the tree cursor.
    fn open_info_editor(&mut self) {
        let Some((path, kind, _)) = self.cur_tree_node() else { return };
        if kind == NodeKind::SmartAlbum {
            self.status = "can't edit a smart album's info (D deletes it)".into();
            return;
        }
        self.info_target = Some(path);
        self.info_editor = true;
        self.status = "album info: n name · d desc · t tags · a add-tag · c cover · s sort · Esc".into();
    }

    /// Prompt to edit `field` on the album at `path` (prefilled with the current value).
    fn prompt_album_field(&mut self, path: PathBuf, field: AlbumFieldKind) {
        let m = self.album_meta_at(&path);
        let (label, prefill) = match field {
            AlbumFieldKind::Name => ("album name: ", m.name.unwrap_or_default()),
            AlbumFieldKind::Description => ("album description: ", m.description.unwrap_or_default()),
            AlbumFieldKind::TagsReplace => ("album tags (comma-sep): ", m.tags.join(", ")),
            AlbumFieldKind::TagsAdd => ("add album tags (comma-sep): ", String::new()),
            AlbumFieldKind::Cover => ("cover image filename: ", m.cover.unwrap_or_default()),
            AlbumFieldKind::Sort => ("sort (name-asc|date-desc|rating-desc|…): ", m.sort.unwrap_or_default()),
        };
        self.prompt(label, prefill, PendingCmd::AlbumEdit { path, field });
    }

    fn album_count(&self) -> usize {
        fn rec(n: &LibraryNode) -> usize {
            (n.kind == NodeKind::Album) as usize + n.children.iter().map(rec).sum::<usize>()
        }
        rec(&self.root)
    }

    /// Open the album at `dir`: list its images (non-recursive), reset the grid, switch focus.
    fn open_album(&mut self, dir: PathBuf) {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && library::is_image(p))
            .collect();
        paths.sort();
        self.status = format!("{}  ·  {} images", dir.display(), paths.len());
        self.smart = None; // leaving any smart-album / search view
        self.smart_is_search = false;
        self.lookalike_of = None;
        self.smart_src.clear();
        self.smart_rec.clear();
        self.album_meta = hjson::read_album(&dir).unwrap_or_default();
        self.album_dir = Some(dir);
        self.album_paths = paths;
        self.sort_album(); // honour the album's persisted sort order
        self.album_cursor = 0;
        self.selected.clear();
        self.filter.clear();
        self.rebuild_view();
        self.mode = AlbumMode::Grid;
        self.view_proto = None;
        self.thumbs.clear();
        self.focus = Focus::Album;
    }

    /// Every image in the library with its source album dir + record (each album.hjson read once).
    /// The shared collection step behind smart albums and metadata search.
    fn collect_library(&self) -> Vec<(PathBuf, PathBuf, Option<hjson::ImageRecord>)> {
        let mut dirs = Vec::new();
        collect_album_dirs(&self.root, &mut dirs);
        let mut out = Vec::new();
        for dir in dirs {
            let meta = hjson::read_album(&dir).unwrap_or_default();
            let mut imgs: Vec<PathBuf> = std::fs::read_dir(&dir)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.is_file() && library::is_image(p))
                .collect();
            imgs.sort();
            for p in imgs {
                let rec = p.file_name().and_then(|n| n.to_str()).and_then(|n| meta.images.get(n)).cloned();
                out.push((p, dir.clone(), rec));
            }
        }
        out
    }

    /// Flatten browse: show **every** image beneath `dir` (across its sub-albums) in one grid — the
    /// recursive view a mixed album / folder can't show directly. Each image's curation still routes
    /// to its own source album (via the smart-view's per-image source map).
    fn open_recursive(&mut self, dir: PathBuf) {
        let paths = Self::gather_image_files(&dir, true);
        if paths.is_empty() {
            self.status = "no images under here".into();
            return;
        }
        let mut meta_cache: HashMap<PathBuf, hjson::AlbumMeta> = HashMap::new();
        let mut items = Vec::with_capacity(paths.len());
        for p in paths {
            let Some(parent) = p.parent().map(|x| x.to_path_buf()) else { continue };
            let meta = meta_cache
                .entry(parent.clone())
                .or_insert_with(|| hjson::read_album(&parent).unwrap_or_default());
            let rec = p.file_name().and_then(|n| n.to_str()).and_then(|n| meta.images.get(n)).cloned();
            items.push((p, parent, rec));
        }
        let count = items.len();
        let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("subtree").to_string();
        self.enter_smart_view(format!("all under {name}"), String::new(), false, items, false);
        self.status = format!("▤ {count} images under '{name}' (recursive) · curation routes to each album");
    }

    /// Commit a collected result set as the active smart view (shared tail of smart-album / search).
    /// `paths` is already in display order; pass `presorted` to keep it (search ranks by relevance).
    fn enter_smart_view(
        &mut self,
        label: String,
        query: String,
        is_search: bool,
        items: Vec<(PathBuf, PathBuf, Option<hjson::ImageRecord>)>,
        presorted: bool,
    ) {
        let mut paths = Vec::with_capacity(items.len());
        let mut src = HashMap::new();
        let mut recs = HashMap::new();
        for (p, dir, rec) in items {
            if let Some(r) = rec {
                recs.insert(p.clone(), r);
            }
            src.insert(p.clone(), dir);
            paths.push(p);
        }
        self.smart = Some(label);
        self.smart_query = query;
        self.smart_is_search = is_search;
        self.lookalike_of = None; // a plain smart/search view (open_lookalike re-sets this after)
        self.lookalike_clip = false;
        self.smart_src = src;
        self.smart_rec = recs;
        self.album_dir = None;
        // Transient meta; only `sort` is read. Search results are pre-ranked → `relevance` (no-op sort).
        self.album_meta = hjson::AlbumMeta {
            sort: presorted.then(|| "relevance".to_string()),
            ..Default::default()
        };
        self.album_paths = paths;
        if !presorted {
            self.sort_album();
        }
        self.album_cursor = 0;
        self.selected.clear();
        self.filter.clear();
        self.rebuild_view();
        self.mode = AlbumMode::Grid;
        self.view_proto = None;
        self.thumbs.clear();
        self.focus = Focus::Album;
    }

    /// Open a smart album: evaluate `query` (filter grammar) against every album, collecting matches
    /// into one grid. Curation writes route back to each image's own album (see [`edit_record_at`]).
    fn open_smart(&mut self, name: String, query: String) {
        let items: Vec<_> = self
            .collect_library()
            .into_iter()
            .filter(|(p, _, rec)| {
                let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                matches_filter(fname, rec.as_ref(), &query)
            })
            .collect();
        let count = items.len();
        self.enter_smart_view(name.clone(), query.clone(), false, items, false);
        self.status =
            format!("★ {name}  ·  {count} images  ·  [{query}]  (curation writes to each source album)");
    }

    /// Metadata semantic search: rank every image in the library by TF-IDF relevance of `query`
    /// against its text metadata (filename, title, caption, notes, tags, and the `--import` prompt /
    /// model), and show the matches best-first. A relevance-ranked smart view — curation still routes
    /// back to each source album.
    fn open_search(&mut self, query: String) {
        let items = self.collect_library();
        let docs: Vec<String> = items.iter().map(|(p, _, rec)| doc_for(p, rec.as_ref())).collect();
        let ranked = crate::textsearch::rank(&query, &docs); // (index, score) desc; empty on no hit
        let ordered: Vec<_> = ranked.into_iter().map(|(i, _)| items[i].clone()).collect();
        let count = ordered.len();
        self.enter_smart_view(format!("search: {query}"), query.clone(), true, ordered, true);
        self.status = format!("🔎 '{query}'  ·  {count} matches by relevance");
    }

    /// Perceptual lookalike: rank the whole library by dHash similarity to `query_path` (nearest
    /// first). Fully offline — no model, no network. A relevance-style smart view.
    fn open_lookalike(&mut self, query_path: PathBuf) {
        let Some(qhash) = loader::thumbnail(&query_path, 64).ok().map(|img| dedup::dhash(&img)) else {
            self.status = "couldn't read the image".into();
            return;
        };
        let mut ranked: Vec<(PathBuf, PathBuf, Option<hjson::ImageRecord>, u32)> = self
            .collect_library()
            .into_iter()
            .filter_map(|(p, dir, rec)| {
                loader::thumbnail(&p, 64)
                    .ok()
                    .map(|img| (p, dir, rec, dedup::hamming(qhash, dedup::dhash(&img))))
            })
            .collect();
        ranked.sort_by_key(|x| x.3); // nearest (query itself is distance 0, stays first)
        let ordered: Vec<_> = ranked.into_iter().take(120).map(|(p, d, r, _)| (p, d, r)).collect();
        let count = ordered.len();
        let name = query_path.file_name().and_then(|n| n.to_str()).unwrap_or("image").to_string();
        self.enter_smart_view(format!("similar: {name}"), String::new(), true, ordered, true);
        self.lookalike_of = Some(query_path); // so a rescan re-ranks rather than filtering
        self.status = format!("🔍 {count} most-similar to {name} (perceptual hash)");
    }

    /// The `album_paths` index at the cursor (through the filtered view).
    fn cur_idx(&self) -> Option<usize> {
        self.view.get(self.album_cursor).copied()
    }

    /// `album_paths` indices the next curation op applies to: the multi-selection if any, else the
    /// cursor image.
    fn targets(&self) -> Vec<usize> {
        if self.selected.is_empty() {
            self.cur_idx().into_iter().collect()
        } else {
            let mut v: Vec<usize> = self.selected.iter().copied().collect();
            v.sort_unstable();
            v
        }
    }

    /// The curation record for `path`: from the source-album maps in a smart-album view, else from
    /// the open album's `album_meta` (keyed by filename).
    fn record(&self, path: &Path) -> Option<&hjson::ImageRecord> {
        if self.smart.is_some() {
            self.smart_rec.get(path)
        } else {
            path.file_name().and_then(|n| n.to_str()).and_then(|n| self.album_meta.images.get(n))
        }
    }

    /// Open the tag browser: gather every tag used in the album with its count, most-used first.
    fn open_tag_browser(&mut self) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let recs: Box<dyn Iterator<Item = &hjson::ImageRecord>> = if self.smart.is_some() {
            Box::new(self.smart_rec.values())
        } else {
            Box::new(self.album_meta.images.values())
        };
        for r in recs {
            for t in &r.tags {
                *counts.entry(t.clone()).or_insert(0) += 1;
            }
        }
        let mut list: Vec<(String, usize)> = counts.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if list.is_empty() {
            self.status = "no tags in this album yet (t to add, A→t to autotag)".into();
            return;
        }
        self.tags_list = list;
        self.tag_cursor = 0;
        self.tag_browser = true;
    }

    /// Open the version browser for the cursor image (save current / restore a snapshot).
    fn open_version_browser(&mut self) {
        let Some((dir, filename)) = self.cur_source() else {
            self.status = "open an album first".into();
            return;
        };
        let path = dir.join(&filename);
        self.versions_list = versions::list(&dir, &filename);
        self.version_target = Some((dir, filename, path));
        self.version_cursor = 0; // 0 = "save current"
        self.version_browser = true;
    }

    /// Version-browser action: row 0 snapshots the current image; a version row restores it.
    fn version_action(&mut self) {
        let Some((dir, filename, path)) = self.version_target.clone() else {
            self.version_browser = false;
            return;
        };
        if self.version_cursor == 0 {
            match versions::snapshot(&dir, &filename) {
                Ok(n) => {
                    self.versions_list = versions::list(&dir, &filename);
                    self.status = format!("saved v{n} of {filename}");
                }
                Err(e) => self.status = format!("snapshot failed: {e:#}"),
            }
        } else if let Some(&n) = self.versions_list.get(self.version_cursor - 1) {
            match versions::restore(&dir, &filename, n) {
                Ok(()) => {
                    // Make the restored version the new pristine baseline so the T1 edit log (if any)
                    // doesn't re-derive over it: point the backup at the restored file + clear edits.
                    let bak = edit::backup_path(&dir, &filename);
                    if bak.exists() {
                        let _ = std::fs::copy(&path, &bak);
                    }
                    self.edit_record_at(&path, |rec| rec.edits.clear());
                    self.thumbs.remove(&path);
                    if self.mode == AlbumMode::Image {
                        self.load_view();
                    }
                    self.status = format!("restored v{n} of {filename}");
                    self.version_browser = false;
                }
                Err(e) => self.status = format!("restore failed: {e:#}"),
            }
        }
    }

    /// Filter the view to the selected tag and close the browser.
    fn pick_tag(&mut self) {
        if let Some((tag, _)) = self.tags_list.get(self.tag_cursor).cloned() {
            self.filter = format!("tag:{tag}");
            self.rebuild_view();
            self.status = format!("filtered to tag:{tag} (/ to clear)");
        }
        self.tag_browser = false;
    }

    /// All filenames referenced as a `variant` by some record (the derivatives, for stacking).
    fn all_variant_names(&self) -> HashSet<String> {
        let mut s = HashSet::new();
        let recs: Box<dyn Iterator<Item = &hjson::ImageRecord>> = if self.smart.is_some() {
            Box::new(self.smart_rec.values())
        } else {
            Box::new(self.album_meta.images.values())
        };
        for r in recs {
            for v in &r.variants {
                s.insert(v.clone());
            }
        }
        s
    }

    /// Rebuild the filtered view from `filter` over `album_paths` + the curation records. When
    /// stacking is on, images that are someone's derivative `variant` are collapsed out (their base
    /// carries a `⧉N` badge).
    fn rebuild_view(&mut self) {
        let f = self.filter.trim().to_string();
        let variants = if self.stack_view { self.all_variant_names() } else { HashSet::new() };
        self.view = (0..self.album_paths.len())
            .filter(|&i| {
                let path = &self.album_paths[i];
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if self.stack_view && variants.contains(name) {
                    return false;
                }
                matches_filter(name, self.record(path), &f)
            })
            .collect();
        if self.view.is_empty() {
            self.album_cursor = 0;
        } else {
            self.album_cursor = self.album_cursor.min(self.view.len() - 1);
        }
    }

    /// Snapshot `album_meta` onto the undo stack before a curation mutation (normal-album mode only;
    /// smart/search views edit source albums directly and aren't covered). Clears the redo stack.
    fn snapshot_meta(&mut self) {
        if self.smart.is_some() || self.album_dir.is_none() {
            return;
        }
        self.undo_stack.push(self.album_meta.clone());
        if self.undo_stack.len() > 50 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Undo the last curation change (restore the previous `album_meta`).
    fn undo_curation(&mut self) {
        if self.smart.is_some() {
            self.status = "undo works in a regular album, not a smart/search view".into();
            return;
        }
        match self.undo_stack.pop() {
            Some(prev) => {
                self.redo_stack.push(std::mem::replace(&mut self.album_meta, prev));
                self.save_album();
                self.rebuild_view();
                self.status = format!("undo ({} more)", self.undo_stack.len());
            }
            None => self.status = "nothing to undo".into(),
        }
    }

    /// Redo the last undone curation change.
    fn redo_curation(&mut self) {
        match self.redo_stack.pop() {
            Some(next) => {
                self.undo_stack.push(std::mem::replace(&mut self.album_meta, next));
                self.save_album();
                self.rebuild_view();
                self.status = format!("redo ({} more)", self.redo_stack.len());
            }
            None => self.status = "nothing to redo".into(),
        }
    }

    /// Mutate each target image's record, then persist. In a smart-album view each write routes to
    /// the image's own source album; otherwise all targets share the open album (one write).
    fn edit_targets(&mut self, mut f: impl FnMut(&mut hjson::ImageRecord)) {
        self.snapshot_meta(); // for undo (normal-album curation)
        let paths: Vec<PathBuf> =
            self.targets().into_iter().filter_map(|i| self.album_paths.get(i).cloned()).collect();
        if self.smart.is_some() {
            for p in paths {
                self.edit_record_at(&p, |rec| f(rec));
            }
        } else {
            for p in &paths {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    f(self.album_meta.images.entry(name.to_string()).or_default());
                }
            }
            self.save_album();
        }
    }

    /// Apply `f` to a single image's record and persist it, routing to the right album whether or
    /// not a smart-album view is active. In smart mode, refreshes the cached record for the badges.
    fn edit_record_at(&mut self, path: &Path, f: impl FnOnce(&mut hjson::ImageRecord)) {
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(String::from) else { return };
        if self.smart.is_some() {
            let Some(dir) = self.smart_src.get(path).cloned() else { return };
            let mut m = hjson::read_album(&dir).unwrap_or_default();
            f(m.images.entry(name.clone()).or_default());
            if let Err(e) = hjson::write_album(&dir, &m) {
                self.status = format!("save failed: {e}");
                return;
            }
            if let Some(r) = m.images.get(&name) {
                self.smart_rec.insert(path.to_path_buf(), r.clone());
            }
        } else {
            f(self.album_meta.images.entry(name).or_default());
            self.save_album();
        }
    }

    /// The cursor image's source album dir + filename (works in a normal or smart-album view).
    fn cur_source(&self) -> Option<(PathBuf, String)> {
        let p = self.cur_idx().and_then(|i| self.album_paths.get(i))?.clone();
        let filename = p.file_name().and_then(|n| n.to_str())?.to_string();
        let dir = if self.smart.is_some() {
            self.smart_src.get(&p)?.clone()
        } else {
            self.album_dir.clone()?
        };
        Some((dir, filename))
    }

    /// The cursor image's current edit ops (parsed from its record's edit log).
    fn cur_edit_ops(&self) -> Vec<edit::EditOp> {
        self.cur_idx()
            .and_then(|i| self.album_paths.get(i))
            .and_then(|p| self.record(p))
            .map(|r| r.edits.iter().filter_map(edit::EditOp::from_entry).collect())
            .unwrap_or_default()
    }

    /// Open the Edit palette (reset its search + cursor).
    fn open_edit_menu(&mut self) {
        self.edit_menu = true;
        self.edit_query.clear();
        self.edit_cursor = 0;
    }

    /// Run a selected Edit-palette command. The palette stays open so edits can be chained.
    fn run_edit_cmd(&mut self, cmd: EditCmd) {
        match cmd {
            EditCmd::Op(op) => self.apply_edit(op),
            EditCmd::Adjust(op) => self.enter_adjust(op),
            EditCmd::FreeCrop => self.enter_crop(),
            EditCmd::CropExact => {
                self.edit_menu = false;
                self.prompt("crop to size (WxH px): ", "", PendingCmd::CropExact);
            }
            EditCmd::ResizeExact => {
                self.edit_menu = false;
                self.prompt("resize to (WxH or N px): ", "", PendingCmd::ResizeExact);
            }
            EditCmd::Layers => self.enter_layers(),
            EditCmd::Levels => self.enter_levels(),
            EditCmd::Curve => self.enter_curve(),
            EditCmd::History => self.enter_history(),
            EditCmd::Duplicate => self.duplicate_in_album(),
            EditCmd::Look(i) => self.apply_look(i),
            EditCmd::CopyEdits => self.copy_edits(),
            EditCmd::PasteEdits => self.paste_edits(),
            EditCmd::SavePreset => {
                self.edit_menu = false;
                self.prompt("save edits as preset (name): ", "", PendingCmd::SavePreset);
            }
            EditCmd::ApplyPreset => {
                self.edit_menu = false;
                self.open_preset_browser();
            }
            EditCmd::Straighten => {
                self.edit_menu = false;
                self.prompt("straighten by degrees (e.g. 3 or -2.5): ", "", PendingCmd::Straighten);
            }
            EditCmd::StripExif => {
                self.edit_menu = false;
                let n = self.targets().len();
                self.prompt(
                    format!("strip EXIF/GPS from {n} image(s)? [y/N]: "),
                    "",
                    PendingCmd::StripExif,
                );
            }
            EditCmd::RedactGps => {
                self.edit_menu = false;
                let n = self.targets().len();
                self.prompt(
                    format!("redact GPS from {n} image(s)? [y/N]: "),
                    "",
                    PendingCmd::RedactGps,
                );
            }
            EditCmd::Convert => {
                self.edit_menu = false;
                self.prompt("convert to (fmt [Npx | NkB]): ", "", PendingCmd::Convert);
            }
            EditCmd::Watermark => {
                self.edit_menu = false;
                self.prompt("watermark text  (or: TEXT | /path/font.ttf): ", "", PendingCmd::Watermark);
            }
            EditCmd::Lut => {
                self.edit_menu = false;
                self.prompt("apply LUT — path to a .cube file: ", "", PendingCmd::Lut);
            }
            EditCmd::FindDups => {
                self.edit_menu = false;
                self.dedup_scan();
            }
            EditCmd::QualityCull => {
                self.edit_menu = false;
                self.quality_cull();
            }
            EditCmd::Retouch(op) => self.enter_pick(op),
            EditCmd::Meta(field) => {
                self.edit_menu = false;
                self.begin_edit(field);
            }
            EditCmd::MoveToAlbum { copy } => {
                self.edit_menu = false;
                let verb = if copy { "copy" } else { "move" };
                self.prompt(format!("{verb} to album — path: "), "", PendingCmd::MoveToAlbum { copy });
            }
            EditCmd::Trash => {
                self.edit_menu = false;
                self.trash_targets();
            }
            EditCmd::OpenTrash => {
                self.edit_menu = false;
                self.open_trash();
            }
            EditCmd::RestoreTrash => {
                self.edit_menu = false;
                self.restore_trash();
            }
            EditCmd::EmptyTrash => {
                self.edit_menu = false;
                self.prompt("empty trash — permanently delete all? [y/N]: ", "", PendingCmd::EmptyTrash);
            }
            EditCmd::ClearThumbs => {
                self.edit_menu = false;
                self.clear_thumb_cache();
            }
            EditCmd::FacePolish => {
                self.edit_menu = false;
                if self.cur_source().is_none() {
                    self.status = "open an image first".into();
                } else if self.mem_low() {
                    self.status = "⚠ memory low — free RAM before face detection".into();
                } else {
                    self.pending_face_polish = true;
                    self.status = "face polish — detecting faces … (the UI will pause)".into();
                }
            }
            EditCmd::Generate => {
                self.edit_menu = false;
                self.prompt("AI generate — prompt (| model [WxH], e.g. | sd15 512): ", "", PendingCmd::Generate);
            }
            EditCmd::Img2imgCreate => {
                self.edit_menu = false;
                if self.cur_source().is_none() {
                    self.status = "open an image first".into();
                } else {
                    self.prompt("AI img2img — prompt: ", "", PendingCmd::MlPrompt { relight: false });
                }
            }
            EditCmd::Portrait => {
                self.edit_menu = false;
                self.prompt("AI portrait — prompt (| model [WxH]): ", "", PendingCmd::Portrait);
            }
            EditCmd::Multiperson => {
                self.edit_menu = false;
                self.prompt("AI multiperson — scene (| model [WxH]): ", "", PendingCmd::Multiperson);
            }
            EditCmd::Undo => self.undo_edit(),
            EditCmd::Redo => self.redo_edit(),
            EditCmd::Revert => self.revert_edits(),
        }
    }

    // ---- Interactive scalar-adjustment slider ----------------------------------------------------

    /// Enter the +/- slider for a scalar adjustment on the cursor image (live preview).
    fn enter_adjust(&mut self, op: edit::EditOp) {
        if self.cur_source().is_none() {
            self.status = "open an image first".into();
            return;
        }
        self.edit_menu = false;
        // Start at the op's own value: tonal adjustments pass 0 (neutral); filters pass their full
        // strength so the slider opens on the full effect and can be dialled down.
        self.adjust_op = Some(op);
        self.adjust_mode = true;
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_adjust_status();
    }

    /// Change the slider value by `delta`, clamped to the op's range; refresh the live preview.
    fn nudge_adjust(&mut self, delta: i32) {
        let Some(op) = self.adjust_op.clone() else { return };
        let (min, max, _) = op.scalar_range();
        let v = (op.scalar().unwrap_or(0) + delta).clamp(min, max);
        self.adjust_op = Some(op.with_scalar(v));
        self.load_view();
        self.set_adjust_status();
    }

    fn set_adjust_status(&mut self) {
        if let Some(op) = &self.adjust_op {
            self.status = format!(
                "{} {:+} · ←/→ fine ±1 · [/] jump ±{} · Enter apply · Esc cancel",
                op.label(),
                op.scalar().unwrap_or(0),
                op.scalar_range().2,
            );
        }
    }

    /// Commit the slider value as an edit (a no-op at 0); leave the slider.
    fn apply_adjust(&mut self) {
        let Some(op) = self.adjust_op.take() else { return };
        self.adjust_mode = false;
        if op.scalar() == Some(0) {
            self.load_view();
            self.status = "no change".into();
            return;
        }
        self.apply_edit(op);
    }

    /// Leave the slider without applying.
    fn cancel_adjust(&mut self) {
        self.adjust_mode = false;
        self.adjust_op = None;
        self.load_view();
        self.status = "adjustment cancelled".into();
    }

    // ---- Before/after --------------------------------------------------------------------------

    /// Toggle showing the pristine original (the edit backup) vs the edited file.
    fn toggle_before_after(&mut self) {
        self.show_original = !self.show_original;
        self.status = if self.show_original {
            "showing ORIGINAL (before) · \\ back to edited".into()
        } else {
            "showing edited (after)".into()
        };
        self.load_view();
    }

    // ---- Interactive tone-curve editor ---------------------------------------------------------

    fn enter_curve(&mut self) {
        if self.cur_source().is_none() {
            self.status = "open an image first".into();
            return;
        }
        self.edit_menu = false;
        self.curve_pts = [0, 64, 128, 192, 255];
        self.curve_sel = 0;
        self.curve_mode = true;
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_curve_status();
    }

    fn curve_op(&self) -> edit::EditOp {
        edit::EditOp::Curve { pts: self.curve_pts }
    }

    /// Move the selected curve point's output value; refresh the live preview.
    fn adjust_curve(&mut self, d: i32) {
        self.curve_pts[self.curve_sel] = (self.curve_pts[self.curve_sel] + d).clamp(0, 255);
        self.load_view();
        self.set_curve_status();
    }

    /// Select the previous/next curve point (5 points, no wrap).
    fn select_curve(&mut self, delta: i32) {
        self.curve_sel = (self.curve_sel as i32 + delta).clamp(0, 4) as usize;
        self.set_curve_status();
    }

    fn set_curve_status(&mut self) {
        const IN: [i32; 5] = [0, 64, 128, 192, 255];
        self.status = format!(
            "curve · point {}/5 (in {}) → out {} · ←/→ pick · ↑/↓ move · [/] fine · Enter apply · Esc",
            self.curve_sel + 1,
            IN[self.curve_sel],
            self.curve_pts[self.curve_sel]
        );
    }

    fn apply_curve(&mut self) {
        let op = self.curve_op();
        self.curve_mode = false;
        if self.curve_pts == [0, 64, 128, 192, 255] {
            self.load_view();
            self.status = "curve: no change".into();
            return;
        }
        self.apply_edit(op);
    }

    fn cancel_curve(&mut self) {
        self.curve_mode = false;
        self.load_view();
        self.status = "curve cancelled".into();
    }

    // ---- Edit-history scrubber -----------------------------------------------------------------

    fn enter_history(&mut self) {
        let Some((dir, filename)) = self.cur_source() else {
            self.status = "open an image first".into();
            return;
        };
        let ops = self.cur_edit_entries();
        if ops.is_empty() {
            self.status = "this image has no edit history".into();
            return;
        }
        // The scrubber replays over the pristine original (the edit backup, else the file itself).
        let bak = edit::backup_path(&dir, &filename);
        let src = if bak.exists() { bak } else { dir.join(&filename) };
        let Some(orig) = loader::thumbnail(&src, 1400).ok() else {
            self.status = "couldn't load the original".into();
            return;
        };
        self.edit_menu = false;
        self.history_orig = Some(orig);
        self.history_ops = ops;
        self.history_pos = self.history_ops.len(); // start on the final result
        self.history_mode = true;
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_history_status();
    }

    /// The image at the current history position: replay the first `history_pos` ops on the original.
    fn history_preview(&self) -> Option<image::DynamicImage> {
        let base = self.history_orig.as_ref()?;
        let ops: Vec<edit::EditOp> = self.history_ops[..self.history_pos.min(self.history_ops.len())]
            .iter()
            .filter_map(edit::EditOp::from_entry)
            .collect();
        Some(edit::replay(base, &ops))
    }

    fn move_history(&mut self, delta: i32) {
        let n = self.history_ops.len() as i32;
        self.history_pos = (self.history_pos as i32 + delta).clamp(0, n) as usize;
        self.load_view();
        self.set_history_status();
    }

    /// Delete the op currently shown (at `history_pos - 1`), rebuilding the preview.
    fn delete_history_op(&mut self) {
        if self.history_pos == 0 || self.history_ops.is_empty() {
            self.status = "at the original — nothing to delete here".into();
            return;
        }
        self.history_ops.remove(self.history_pos - 1);
        self.history_pos -= 1;
        self.load_view();
        self.set_history_status();
    }

    fn set_history_status(&mut self) {
        let n = self.history_ops.len();
        if self.history_pos == 0 {
            self.status = format!("history 0/{n}: original · →/↑ step forward · Enter apply · Esc");
        } else {
            let label = self
                .history_ops
                .get(self.history_pos - 1)
                .and_then(edit::EditOp::from_entry)
                .map(|o| o.label())
                .unwrap_or_default();
            self.status = format!(
                "history {}/{n}: {label} · ←/→ step · d delete this edit · Enter apply · Esc",
                self.history_pos
            );
        }
    }

    /// Commit the (possibly trimmed) history to the record and rebuild the file.
    fn apply_history(&mut self) {
        let Some((dir, filename)) = self.cur_source() else {
            self.history_mode = false;
            return;
        };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else {
            self.history_mode = false;
            return;
        };
        let ops = std::mem::take(&mut self.history_ops);
        self.history_mode = false;
        self.history_orig = None;
        let full: Vec<edit::EditOp> = ops.iter().filter_map(edit::EditOp::from_entry).collect();
        self.edit_record_at(&path, move |rec| rec.edits = ops);
        match edit::rebuild_file(&dir, &filename, &full) {
            Ok(()) => {
                self.status = format!("history applied · {} edit(s)", full.len());
                self.refresh_after_edit(&path);
            }
            Err(e) => self.status = format!("apply failed: {e:#}"),
        }
    }

    fn cancel_history(&mut self) {
        self.history_mode = false;
        self.history_orig = None;
        self.load_view();
        self.status = "history closed (no change)".into();
    }

    /// Enter interactive free-form crop on the cursor image (a dimmed live preview in the image pane).
    fn enter_crop(&mut self) {
        if self.cur_source().is_none() {
            self.status = "open an image first".into();
            return;
        }
        self.edit_menu = false;
        self.crop_mode = true;
        self.crop_rect = (0.1, 0.1, 0.8, 0.8);
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_crop_status();
    }

    /// Nudge the crop rect (deltas in fractions), keeping it inside the image, then refresh preview.
    fn adjust_crop(&mut self, dx: f32, dy: f32, dw: f32, dh: f32) {
        let (mut x, mut y, mut w, mut h) = self.crop_rect;
        w = (w + dw).clamp(0.05, 1.0);
        h = (h + dh).clamp(0.05, 1.0);
        x = (x + dx).clamp(0.0, 1.0 - w);
        y = (y + dy).clamp(0.0, 1.0 - h);
        self.crop_rect = (x, y, w, h);
        self.load_view();
        self.set_crop_status();
    }

    fn set_crop_status(&mut self) {
        let (_, _, w, h) = self.crop_rect;
        self.status = format!(
            "crop {:.0}%×{:.0}% · arrows move · +/- size · [ ] width · , . height · Enter apply · Esc",
            w * 100.0,
            h * 100.0
        );
    }

    /// Grow (`d`>0) or shrink the crop box, keeping it centered.
    fn grow_crop(&mut self, d: f32) {
        let (x, y, w, h) = self.crop_rect;
        let (nw, nh) = ((w + d).clamp(0.05, 1.0), (h + d).clamp(0.05, 1.0));
        self.crop_rect = (
            (x - (nw - w) / 2.0).clamp(0.0, 1.0 - nw),
            (y - (nh - h) / 2.0).clamp(0.0, 1.0 - nh),
            nw,
            nh,
        );
        self.load_view();
        self.set_crop_status();
    }

    /// Commit the free-form crop as an edit; leave crop mode.
    fn apply_crop(&mut self) {
        let (x, y, w, h) = self.crop_rect;
        self.crop_mode = false;
        self.apply_edit(edit::EditOp::Crop { x, y, w, h });
    }

    /// Leave crop mode without cropping.
    fn cancel_crop(&mut self) {
        self.crop_mode = false;
        self.load_view();
        self.status = "crop cancelled".into();
    }

    // ---- Interactive retouch pick-mode -------------------------------------------------------------

    /// Enter the crosshair pick-mode for a retouch op (heal / clone / red-eye / dodge-burn / perspective).
    fn enter_pick(&mut self, op: PickOp) {
        if self.cur_source().is_none() {
            self.status = "open an image first".into();
            return;
        }
        self.edit_menu = false;
        self.pick = Some(PickState { op, pos: (0.5, 0.5), points: Vec::new(), radius: 0.06 });
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_pick_status();
    }

    fn set_pick_status(&mut self) {
        let Some(p) = &self.pick else { return };
        let step = p.points.len() + 1;
        let total = p.op.points_needed();
        let hint = match p.op {
            PickOp::Clone if p.points.is_empty() => " (source)",
            PickOp::Clone => " (destination)",
            PickOp::Perspective => " (corner TL→TR→BR→BL)",
            _ => "",
        };
        let rad = if p.op.uses_radius() { " · +/- size" } else { "" };
        self.status = format!(
            "{} · point {step}/{total}{hint} · arrows move{rad} · Enter set · Esc cancel",
            p.op.label()
        );
    }

    /// Move the crosshair by `(dx, dy)` fractions, clamped to the image.
    fn pick_move(&mut self, dx: f32, dy: f32) {
        if let Some(p) = &mut self.pick {
            p.pos.0 = (p.pos.0 + dx).clamp(0.0, 1.0);
            p.pos.1 = (p.pos.1 + dy).clamp(0.0, 1.0);
            self.load_view();
            self.set_pick_status();
        }
    }

    /// Grow / shrink the brush radius.
    fn pick_radius(&mut self, d: f32) {
        if let Some(p) = &mut self.pick {
            if p.op.uses_radius() {
                p.radius = (p.radius + d).clamp(0.01, 0.4);
                self.load_view();
                self.set_pick_status();
            }
        }
    }

    /// Confirm the crosshair as the next point; apply the op once enough points are collected.
    fn pick_confirm(&mut self) {
        let Some(p) = &mut self.pick else { return };
        p.points.push(p.pos);
        if p.points.len() < p.op.points_needed() {
            self.load_view();
            self.set_pick_status();
            return;
        }
        // Enough points — build the replayable edit and leave pick-mode.
        let PickState { op, points, radius, .. } = self.pick.take().unwrap();
        let pm = |f: f32| (f * 1000.0).round() as i32;
        let rad = pm(radius);
        let edit = match op {
            PickOp::SpotHeal => edit::EditOp::SpotHeal { x: pm(points[0].0), y: pm(points[0].1), radius: rad },
            PickOp::Clone => edit::EditOp::Clone {
                sx: pm(points[0].0), sy: pm(points[0].1),
                dx: pm(points[1].0), dy: pm(points[1].1), radius: rad,
            },
            PickOp::RedEye => edit::EditOp::RedEye { x: pm(points[0].0), y: pm(points[0].1), radius: rad },
            PickOp::Dodge => edit::EditOp::DodgeBurn { x: pm(points[0].0), y: pm(points[0].1), radius: rad, amount: 35 },
            PickOp::Burn => edit::EditOp::DodgeBurn { x: pm(points[0].0), y: pm(points[0].1), radius: rad, amount: -35 },
            PickOp::Perspective => {
                let mut pts = [[0i32; 2]; 4];
                for (i, pt) in points.iter().enumerate().take(4) {
                    pts[i] = [pm(pt.0), pm(pt.1)];
                }
                edit::EditOp::Perspective4 { pts }
            }
        };
        self.apply_edit(edit);
    }

    /// Leave pick-mode without applying.
    fn cancel_pick(&mut self) {
        self.pick = None;
        self.load_view();
        self.status = "retouch cancelled".into();
    }

    // ---- Interactive levels editor -----------------------------------------------------------------

    /// Enter the levels editor on the cursor image (live preview; ↑/↓ pick a handle, ←/→ adjust).
    fn enter_levels(&mut self) {
        if self.cur_source().is_none() {
            self.status = "open an image first".into();
            return;
        }
        self.edit_menu = false;
        self.levels_mode = true;
        self.lv_black = 0;
        self.lv_white = 255;
        self.lv_gamma = 100;
        self.lv_sel = 0;
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_levels_status();
    }

    /// The working levels as an `EditOp` (for the preview + commit).
    fn levels_op(&self) -> edit::EditOp {
        edit::EditOp::Levels { black: self.lv_black, white: self.lv_white, gamma: self.lv_gamma }
    }

    /// Adjust the selected handle by `d` (black/white in 0..255, gamma in hundredths), clamped so
    /// black < white and gamma stays sane.
    fn adjust_levels(&mut self, d: i32) {
        match self.lv_sel {
            0 => self.lv_black = (self.lv_black + d).clamp(0, self.lv_white - 1),
            1 => self.lv_white = (self.lv_white + d).clamp(self.lv_black + 1, 255),
            _ => self.lv_gamma = (self.lv_gamma + d).clamp(20, 500),
        }
        self.load_view();
        self.set_levels_status();
    }

    /// Cycle which handle ←/→ adjusts (black → white → gamma).
    fn select_levels(&mut self, delta: i32) {
        self.lv_sel = (((self.lv_sel as i32 + delta) % 3 + 3) % 3) as usize;
        self.set_levels_status();
    }

    fn set_levels_status(&mut self) {
        let mark = |i: usize, s: String| if self.lv_sel == i { format!("[{s}]") } else { s };
        self.status = format!(
            "levels · {} {} {} · ↑↓ pick · ←→ adjust · Enter apply · Esc",
            mark(0, format!("black {}", self.lv_black)),
            mark(1, format!("white {}", self.lv_white)),
            mark(2, format!("γ {:.2}", self.lv_gamma as f32 / 100.0)),
        );
    }

    /// Commit the levels as a pixel edit; leave the editor.
    fn apply_levels(&mut self) {
        let op = self.levels_op();
        self.levels_mode = false;
        if op == (edit::EditOp::Levels { black: 0, white: 255, gamma: 100 }) {
            self.load_view();
            self.status = "levels: no change".into();
            return;
        }
        self.apply_edit(op);
    }

    /// Leave the levels editor without applying.
    fn cancel_levels(&mut self) {
        self.levels_mode = false;
        self.load_view();
        self.status = "levels cancelled".into();
    }

    // ---- Layer compositing (Phase 8) -------------------------------------------------------------

    /// Enter interactive layer compositing on the cursor image, loading any persisted stack.
    fn enter_layers(&mut self) {
        let Some((dir, _)) = self.cur_source() else {
            self.status = "open an image first".into();
            return;
        };
        self.edit_menu = false;
        self.layers = self
            .cur_idx()
            .and_then(|i| self.album_paths.get(i))
            .and_then(|p| self.record(p))
            .map(|r| r.layers.iter().map(|e| layers::Layer::from_entry(e, &dir)).collect())
            .unwrap_or_default();
        self.layer_active = self.layers.len().saturating_sub(1);
        self.layer_mode = true;
        self.mask_adjust = false;
        self.mode = AlbumMode::Image;
        self.load_view();
        self.set_layer_status();
    }

    /// Prompt for a source image to add as a new top layer (an album filename or a filesystem path).
    fn layer_add_prompt(&mut self) {
        self.prompt("add layer (album file or path): ", "", PendingCmd::AddLayer);
    }

    /// Resolve `spec` (an in-album filename first, else a path) and push it as a new top layer.
    fn add_layer(&mut self, spec: &str) {
        let Some((dir, _)) = self.cur_source() else { return };
        let spec = spec.trim();
        if spec.is_empty() {
            return;
        }
        let in_album = dir.join(spec);
        let path = if in_album.exists() {
            in_album
        } else {
            let p = expand_tilde(spec);
            if p.exists() {
                p
            } else {
                self.status = format!("no such image: {spec}");
                return;
            }
        };
        self.layers.push(layers::Layer::new(path));
        self.layer_active = self.layers.len() - 1;
        self.after_layer_change();
    }

    /// Move the active layer by `(dx, dy)` fractions of the base.
    fn nudge_layer(&mut self, dx: f32, dy: f32) {
        let Some(l) = self.layers.get_mut(self.layer_active) else { return };
        l.x = (l.x + dx).clamp(-0.9, 0.99);
        l.y = (l.y + dy).clamp(-0.9, 0.99);
        self.after_layer_change();
    }

    /// Grow/shrink the active layer (scale = fraction of base width).
    fn scale_layer(&mut self, d: f32) {
        let Some(l) = self.layers.get_mut(self.layer_active) else { return };
        l.scale = (l.scale + d).clamp(0.02, 4.0);
        self.after_layer_change();
    }

    /// Adjust the active layer's opacity.
    fn opacity_layer(&mut self, d: f32) {
        let Some(l) = self.layers.get_mut(self.layer_active) else { return };
        l.opacity = (l.opacity + d).clamp(0.0, 1.0);
        self.after_layer_change();
    }

    /// Cycle the active layer's blend mode.
    fn cycle_layer_blend(&mut self) {
        let Some(l) = self.layers.get_mut(self.layer_active) else { return };
        l.blend = l.blend.cycle();
        self.after_layer_change();
    }

    /// Cycle the active layer's mask: none → ellipse → rectangle → none (image mattes use `k`).
    fn cycle_layer_mask(&mut self) {
        use layers::{Mask, ShapeKind};
        let Some(l) = self.layers.get_mut(self.layer_active) else { return };
        l.mask = match &l.mask {
            Mask::None => Mask::centered_shape(ShapeKind::Ellipse),
            Mask::Shape { kind: ShapeKind::Ellipse, x, y, w, h, feather, invert } => Mask::Shape {
                kind: ShapeKind::Rect,
                x: *x,
                y: *y,
                w: *w,
                h: *h,
                feather: *feather,
                invert: *invert,
            },
            Mask::Shape { kind: ShapeKind::Rect, .. } | Mask::Image { .. } => Mask::None,
        };
        self.after_layer_change();
    }

    /// Prompt for a grayscale matte image to use as the active layer's mask.
    fn layer_mask_prompt(&mut self) {
        if self.layers.get(self.layer_active).is_none() {
            self.status = "add a layer first (a)".into();
            return;
        }
        self.prompt("mask matte (grayscale album file or path): ", "", PendingCmd::MaskImage);
    }

    /// Set the active layer's mask to an image matte resolved from `spec` (album file or path).
    fn set_layer_mask_image(&mut self, spec: &str) {
        let Some((dir, _)) = self.cur_source() else { return };
        let spec = spec.trim();
        if spec.is_empty() {
            return;
        }
        let in_album = dir.join(spec);
        let path = if in_album.exists() {
            in_album
        } else {
            let p = expand_tilde(spec);
            if p.exists() {
                p
            } else {
                self.status = format!("no such image: {spec}");
                return;
            }
        };
        if let Some(l) = self.layers.get_mut(self.layer_active) {
            l.mask = layers::Mask::Image { src: path, invert: false };
        }
        self.after_layer_change();
    }

    /// Grow/shrink a shape mask around its own centre (fraction of the layer), so it stays put.
    fn resize_mask(&mut self, d: f32) {
        if let Some(layers::Mask::Shape { x, y, w, h, .. }) =
            self.layers.get_mut(self.layer_active).map(|l| &mut l.mask)
        {
            let (cx, cy) = (*x + *w / 2.0, *y + *h / 2.0);
            let s = (*w + d).clamp(0.05, 1.0);
            *w = s;
            *h = s;
            *x = cx - s / 2.0;
            *y = cy - s / 2.0;
            self.after_layer_change();
        }
    }

    /// Enter the mask-adjust sub-mode (arrows reposition the mask). Creates a default ellipse if the
    /// active layer has no shape mask; image mattes fill the layer, so there's nothing to position.
    fn enter_mask_adjust(&mut self) {
        use layers::{Mask, ShapeKind};
        let Some(l) = self.layers.get_mut(self.layer_active) else {
            self.status = "add a layer first (a)".into();
            return;
        };
        match &l.mask {
            Mask::Shape { .. } => {}
            Mask::Image { .. } => {
                self.status = "image mattes fill the layer — move the layer to reposition".into();
                return;
            }
            Mask::None => l.mask = Mask::centered_shape(ShapeKind::Ellipse),
        }
        self.mask_adjust = true;
        self.after_layer_change();
    }

    /// Leave the mask-adjust sub-mode, back to moving the layer.
    fn exit_mask_adjust(&mut self) {
        self.mask_adjust = false;
        self.set_layer_status();
    }

    /// Move the active layer's shape mask by `(dx, dy)` fractions of the layer (may sit partly off
    /// the layer for edge masks; kept at least ~quarter-overlapping).
    fn move_mask(&mut self, dx: f32, dy: f32) {
        if let Some(layers::Mask::Shape { x, y, w, h, .. }) =
            self.layers.get_mut(self.layer_active).map(|l| &mut l.mask)
        {
            *x = (*x + dx).clamp(-*w * 0.75, 1.0 - *w * 0.25);
            *y = (*y + dy).clamp(-*h * 0.75, 1.0 - *h * 0.25);
            self.after_layer_change();
        }
    }

    /// Soften/harden a shape mask's edge.
    fn feather_mask(&mut self, d: f32) {
        if let Some(layers::Mask::Shape { feather, .. }) =
            self.layers.get_mut(self.layer_active).map(|l| &mut l.mask)
        {
            *feather = (*feather + d).clamp(0.0, 0.9);
            self.after_layer_change();
        }
    }

    /// Invert the active layer's mask (show ↔ hide).
    fn invert_mask(&mut self) {
        match self.layers.get_mut(self.layer_active).map(|l| &mut l.mask) {
            Some(layers::Mask::Shape { invert, .. }) | Some(layers::Mask::Image { invert, .. }) => {
                *invert = !*invert;
                self.after_layer_change();
            }
            _ => {}
        }
    }

    /// Select the next/previous layer (wraps).
    fn select_layer(&mut self, delta: i32) {
        if self.layers.is_empty() {
            return;
        }
        let n = self.layers.len() as i32;
        self.layer_active = (((self.layer_active as i32 + delta) % n + n) % n) as usize;
        self.mask_adjust = false; // the newly-selected layer may have no shape mask
        self.load_view();
        self.set_layer_status();
    }

    /// Move the active layer up (`up`=toward the top of the stack) or down in z-order.
    fn reorder_layer(&mut self, up: bool) {
        let i = self.layer_active;
        if up && i + 1 < self.layers.len() {
            self.layers.swap(i, i + 1);
            self.layer_active = i + 1;
        } else if !up && i > 0 {
            self.layers.swap(i, i - 1);
            self.layer_active = i - 1;
        } else {
            return;
        }
        self.after_layer_change();
    }

    /// Delete the active layer.
    fn delete_layer(&mut self) {
        if self.layer_active >= self.layers.len() {
            return;
        }
        self.layers.remove(self.layer_active);
        self.layer_active = self.layer_active.min(self.layers.len().saturating_sub(1));
        self.mask_adjust = false;
        self.after_layer_change();
    }

    /// Persist the stack + refresh the composited preview + status after any change.
    fn after_layer_change(&mut self) {
        self.persist_layers();
        self.load_view();
        self.set_layer_status();
    }

    /// Write the working stack onto the cursor image's record.
    fn persist_layers(&mut self) {
        let Some((dir, _)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        let entries: Vec<hjson::LayerEntry> = self.layers.iter().map(|l| l.to_entry(&dir)).collect();
        self.edit_record_at(&path, |rec| rec.layers = entries);
    }

    fn set_layer_status(&mut self) {
        if self.mask_adjust {
            self.status =
                "position mask · arrows move · [ ] size · , . feather · / invert · Enter/Esc done".into();
            return;
        }
        if let Some(l) = self.layers.get(self.layer_active) {
            self.status = format!(
                "layer {}/{}: {} · arrows move · +/- size · < > opacity · b blend · m mask · {{ }} order · x del · a add · Enter flatten · Esc",
                self.layer_active + 1,
                self.layers.len(),
                l.label()
            );
        } else {
            self.status = "layers · a add an image · Esc leave".into();
        }
    }

    /// Bake the stack into a new `_layered.png` variant, record it, and return to the grid on it.
    fn flatten_layers(&mut self) {
        let Some((dir, _)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        if self.layers.is_empty() {
            self.status = "no layers to flatten — press a to add one".into();
            return;
        }
        match layers::flatten(&path, &dir, &self.layers) {
            Ok(name) => {
                self.record_variant(&path, &name);
                self.layer_mode = false;
                self.mask_adjust = false;
                self.mode = AlbumMode::Grid;
                self.rescan();
                self.select_by_name(&name);
                self.status = format!("flattened {} layer(s) → {name}", self.layers.len());
            }
            Err(e) => self.status = format!("flatten failed: {e:#}"),
        }
    }

    /// Leave layer mode (the stack stays persisted for next time).
    fn cancel_layers(&mut self) {
        self.layer_mode = false;
        self.mask_adjust = false;
        self.load_view();
        self.status = "layers closed (stack saved)".into();
    }

    /// Append a pixel edit to the cursor image: back up the pristine original (once), record the op
    /// in `album.hjson`, and re-derive the visible file from original + the full edit list.
    fn apply_edit(&mut self, op: edit::EditOp) {
        let Some((dir, filename)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        if let Err(e) = edit::ensure_backup(&dir, &filename) {
            self.status = format!("edit failed: {e:#}");
            return;
        }
        self.edit_redo.clear(); // a fresh edit invalidates the redo chain
        self.edit_record_at(&path, |rec| rec.edits.push(op.clone().to_entry()));
        let ops = self.cur_edit_ops();
        match edit::rebuild_file(&dir, &filename, &ops) {
            Ok(()) => {
                self.status = format!("{} · {} edit(s) · u undo · U redo · 0 revert", op.label(), ops.len());
                self.refresh_after_edit(&path);
            }
            Err(e) => self.status = format!("edit failed: {e:#}"),
        }
    }

    /// Undo the cursor image's last edit (rebuild from the remaining ops; restores the original when
    /// none remain). The undone op moves onto the redo stack.
    fn undo_edit(&mut self) {
        let Some((dir, filename)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        let mut ops = self.cur_edit_ops();
        let Some(undone) = ops.pop() else {
            self.status = "nothing to undo".into();
            return;
        };
        self.edit_redo.push(undone);
        self.edit_record_at(&path, |rec| {
            rec.edits.pop();
        });
        if let Err(e) = edit::rebuild_file(&dir, &filename, &ops) {
            self.status = format!("undo failed: {e:#}");
            return;
        }
        self.status = format!("undo · {} edit(s) · U redo", ops.len());
        self.refresh_after_edit(&path);
    }

    /// Redo the last undone pixel edit on the cursor image.
    fn redo_edit(&mut self) {
        let Some((dir, filename)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        let Some(op) = self.edit_redo.pop() else {
            self.status = "nothing to redo".into();
            return;
        };
        self.edit_record_at(&path, |rec| rec.edits.push(op.clone().to_entry()));
        let ops = self.cur_edit_ops();
        if let Err(e) = edit::rebuild_file(&dir, &filename, &ops) {
            self.status = format!("redo failed: {e:#}");
            return;
        }
        self.status = format!("redo {} · {} edit(s)", op.label(), ops.len());
        self.refresh_after_edit(&path);
    }

    /// Discard all edits on the cursor image and restore the pristine original.
    fn revert_edits(&mut self) {
        let Some((dir, filename)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        if self.cur_edit_ops().is_empty() {
            self.status = "no edits to revert".into();
            return;
        }
        self.edit_record_at(&path, |rec| rec.edits.clear());
        if let Err(e) = edit::rebuild_file(&dir, &filename, &[]) {
            self.status = format!("revert failed: {e:#}");
            return;
        }
        self.status = "reverted to original".into();
        self.refresh_after_edit(&path);
    }

    /// Queue a T2 ML edit on the cursor image (the event loop hands it to the resident worker).
    fn queue_ml(&mut self, op: mledit::MlOp) {
        self.ml_menu = false;
        let Some((album, filename)) = self.cur_source() else {
            self.status = "open an album first".into();
            return;
        };
        let input = album.join(&filename);
        let warn = if self.mem_low() { "  ⚠ memory low — this may be refused" } else { "" };
        self.status = format!("queued {} …{warn}", op.label());
        self.jobs.push_back(Job::Ml(mledit::MlJob { op, input, album }));
    }

    /// AI generate (txt2img): `arg` is `prompt [| model [WxH]]`.
    fn ai_generate(&mut self, arg: &str) {
        let (prompt, model, size) = parse_gen_spec(arg);
        if prompt.is_empty() {
            self.status = "enter a prompt".into();
            return;
        }
        self.queue_create(CreateOp::Generate { prompt, model, size });
    }

    /// AI portrait: `arg` is `prompt [| model [WxH]]`; the cursor image becomes the identity face.
    fn ai_portrait(&mut self, arg: &str) {
        let (prompt, model, size) = parse_gen_spec(arg);
        if prompt.is_empty() {
            self.status = "enter a prompt".into();
            return;
        }
        let photo = self.cur_source().map(|(d, f)| d.join(f));
        self.queue_create(CreateOp::Portrait { prompt, model, size, photo });
    }

    /// AI multiperson: `arg` is `scene [| model [WxH]]`; the selected images become the people.
    fn ai_multiperson(&mut self, arg: &str) {
        let (prompt, model, size) = parse_gen_spec(arg);
        if prompt.is_empty() {
            self.status = "enter a scene prompt".into();
            return;
        }
        let people: Vec<PathBuf> = self.target_sources().into_iter().map(|(_, p)| p).collect();
        self.queue_create(CreateOp::Multiperson { prompt, model, size, people });
    }

    /// Queue an AI "create" op (generate / portrait / multiperson) — the event loop runs it
    /// TUI-suspended, landing a new image in the current album (or the cursor's source album).
    fn queue_create(&mut self, op: CreateOp) {
        let Some(dir) = self.album_dir.clone().or_else(|| self.cur_source().map(|(d, _)| d)) else {
            self.status = "open an album first".into();
            return;
        };
        if self.mem_low() {
            self.status = "⚠ memory low — free RAM before generating".into();
            return;
        }
        let (label, model) = match &op {
            CreateOp::Generate { model, .. } => ("generate", model.clone()),
            CreateOp::Portrait { model, .. } => ("portrait", model.clone()),
            CreateOp::Multiperson { model, .. } => ("multiperson", model.clone()),
        };
        self.pending_create = Some((op, dir));
        self.status = format!("AI {label} ({model}) — loading model … (the UI will pause)");
    }

    /// Analyze-and-generate (Track A): queue the cursor image to be described by the configured LLM,
    /// then img2img'd from that description into a fresh "reimagined" variant (event loop runs it).
    fn analyze_and_generate(&mut self) {
        let Some((album, filename)) = self.cur_source() else {
            self.status = "open an album first".into();
            return;
        };
        self.pending_analyze = Some(album.join(&filename));
        self.status = "analyze & generate — describing the reference … (the UI will pause)".into();
    }

    /// Face-scan (Track A): queue a library-wide face detect + group (event loop runs it suspended).
    fn request_face_scan(&mut self) {
        if self.mem_low() {
            self.status = "⚠ memory low — free RAM before a face scan".into();
            return;
        }
        self.pending_face_scan = true;
        self.status = "face-scan — detecting faces across the library … (the UI will pause)".into();
    }

    /// Hand a queued ML edit to the background resident worker (spawned lazily on first use), and set
    /// it as the in-flight job the event loop drains inline. Non-blocking: the manager keeps drawing.
    fn dispatch_ml(&mut self, job: mledit::MlJob) {
        if self.ml_worker.is_none() {
            let Some(rt) = self.rt.clone() else {
                self.status = "ML worker unavailable (no async runtime)".into();
                return;
            };
            self.ml_worker = Some(mlworker::MlWorker::spawn(rt));
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = CancelFlag::new();
        let label = job.op.label();
        let src = job.input.clone();
        let wjob = mlworker::Job { op: job.op, input: job.input, album: job.album, tx, cancel: cancel.clone() };
        if self.ml_worker.as_ref().unwrap().submit(wjob) {
            self.status = format!("⟳ {label} — loading model… · Esc cancels");
            self.active_ml = Some(ActiveMl { label, src, rx, cancel });
        } else {
            self.status = format!("✗ {label}: worker unavailable");
            self.ml_worker = None; // let the next attempt re-spawn
        }
    }

    /// Drain the in-flight ML job's progress channel (called each tick while one is active). Updates
    /// the status line from `Progress`, and on `Done`/`Error` finalizes: link the new variant, rescan,
    /// select it, and clear the in-flight slot.
    fn poll_active_ml(&mut self) {
        enum Outcome {
            None,
            Done { output: PathBuf, cancelled: bool },
            Failed(String),
        }
        let (status, outcome, label, src) = {
            let Some(active) = self.active_ml.as_ref() else { return };
            let label = active.label.clone();
            let src = active.src.clone();
            let mut status: Option<String> = None;
            let mut outcome = Outcome::None;
            loop {
                match active.rx.try_recv() {
                    Ok(GenMessage::Progress { step, total, steps_per_sec, .. }) => {
                        status = Some(if total > 1 {
                            format!(
                                "⟳ {label} — step {}/{} ({steps_per_sec:.1} it/s) · Esc cancels",
                                step + 1,
                                total
                            )
                        } else {
                            format!("⟳ {label} — working… · Esc cancels")
                        });
                    }
                    Ok(GenMessage::Preview { .. }) | Ok(GenMessage::Enhanced { .. }) => {}
                    Ok(GenMessage::Done { output, cancelled }) => {
                        outcome = Outcome::Done { output, cancelled };
                        break;
                    }
                    Ok(GenMessage::Error { message }) => {
                        outcome = Outcome::Failed(message);
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        outcome = Outcome::Failed("worker stopped unexpectedly".into());
                        break;
                    }
                }
            }
            (status, outcome, label, src)
        };
        if let Some(s) = status {
            self.status = s;
        }
        match outcome {
            Outcome::None => {}
            Outcome::Done { output, cancelled } => {
                self.active_ml = None;
                if cancelled {
                    self.status = format!("⚠ {label} cancelled");
                    self.rescan();
                } else {
                    let name = output.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                    self.record_variant(&src, &name); // link the derivative before the rescan
                    self.rescan();
                    self.select_by_name(&name);
                    self.status = format!("✓ {label} → {name}");
                }
            }
            Outcome::Failed(e) => {
                self.active_ml = None;
                self.status = format!("✗ {label} failed: {e}");
            }
        }
    }

    /// Refresh the cached memory-status figures (self-throttled to ~1 Hz — `available_ram_gb` builds
    /// and refreshes a `sysinfo::System`, too heavy for every 80 ms draw tick). Called each event-loop
    /// tick; the top bar reads the cached values.
    fn refresh_mem(&mut self) {
        let now = Instant::now();
        if self.mem_checked.map(|t| now.duration_since(t) < Duration::from_millis(1000)).unwrap_or(false) {
            return;
        }
        self.mem_checked = Some(now);
        self.mem_pressure = crate::hw::mem_pressure();
        self.mem_free_gb = crate::hw::available_ram_gb();
        if self.mem_total_gb == 0.0 {
            self.mem_total_gb = crate::hw::total_ram_gb();
        }
    }

    /// Whether AI-heavy ops are risky right now — the kernel pressure signal on macOS, else a
    /// free-RAM floor (accurate off-macOS, where pressure is `Unknown`). Used to warn before load.
    fn mem_low(&self) -> bool {
        use crate::hw::Pressure;
        match self.mem_pressure {
            Pressure::Warning | Pressure::Critical => true,
            Pressure::Unknown => self.mem_free_gb > 0.0 && self.mem_free_gb < 3.0,
            Pressure::Normal => false,
        }
    }

    /// The top-bar memory indicator: `(text, colour)`. On macOS the kernel pressure level is the
    /// honest signal (free-RAM under-reports reclaimable pages), so it drives the words; off-macOS we
    /// show free GB. Red/amber carry the "avoid AI" guidance.
    fn mem_hint(&self) -> (String, Color) {
        use crate::hw::Pressure;
        match self.mem_pressure {
            Pressure::Critical => ("  ⛔ memory low — avoid AI ops  ".into(), Color::Red),
            Pressure::Warning => ("  ⚠ memory tight — heavy AI risky  ".into(), Color::Yellow),
            Pressure::Normal => (format!("  ● memory ok · {:.0} GB  ", self.mem_total_gb), Color::Green),
            Pressure::Unknown => {
                let col = if self.mem_free_gb <= 0.0 {
                    Color::DarkGray
                } else if self.mem_free_gb < 3.0 {
                    Color::Red
                } else if self.mem_free_gb < 6.0 {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let tail = if self.mem_low() { " — avoid AI ops" } else { "" };
                (format!("  ● {:.0}/{:.0} GB free{tail}  ", self.mem_free_gb, self.mem_total_gb), col)
            }
        }
    }

    /// Request cancellation of the in-flight ML job (Esc). img2img stops at the next denoise step;
    /// relight/upscale have no mid-run hook, so they run to completion (the flag still prevents the
    /// result from being adopted).
    fn cancel_active_ml(&mut self) -> bool {
        if let Some(active) = self.active_ml.as_ref() {
            active.cancel.cancel();
            self.status = format!("cancelling {} …", active.label);
            true
        } else {
            false
        }
    }

    /// Offline auto-tag: for every browse target that has a generation recipe (AI-made / `--import`ed),
    /// merge recipe-derived tags (`ai` + model + prompt keywords). No network.
    fn ai_tag_from_recipe(&mut self) {
        self.ai_menu = false;
        self.snapshot_meta();
        let paths: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        let mut n = 0;
        for p in paths {
            let derived = self.record(&p).and_then(|r| r.generation.as_ref()).map(import::ai_tags);
            if let Some(tags) = derived {
                self.edit_record_at(&p, |rec| {
                    for t in tags {
                        if !rec.tags.contains(&t) {
                            rec.tags.push(t);
                        }
                    }
                });
                n += 1;
            }
        }
        self.rebuild_view();
        self.status = format!("recipe-tagged {n} AI image(s)");
    }

    /// Queue a vision request on the cursor image (the event loop runs it).
    fn queue_vision(&mut self, op: vision::VisionOp) {
        self.ai_menu = false;
        let Some((album, filename)) = self.cur_source() else {
            self.status = "open an album first".into();
            return;
        };
        self.jobs.push_back(Job::Vision(op, album.join(filename)));
    }

    /// The source album dir + filename for an arbitrary image `path` (normal or smart/search view).
    fn source_of(&self, path: &Path) -> Option<(PathBuf, String)> {
        let filename = path.file_name().and_then(|n| n.to_str())?.to_string();
        let dir = if self.smart.is_some() {
            self.smart_src.get(path)?.clone()
        } else {
            self.album_dir.clone()?
        };
        Some((dir, filename))
    }

    // --- Natural-language command pipeline (`:`) — RFC PHOTOS-1 §NL. ---

    /// Stash a parsed plan and open a y/N confirmation with its summary.
    fn confirm_plan(&mut self, plan: nl::CommandPlan) {
        let summary = plan.summary();
        self.pending_plan = Some(plan);
        self.prompt(format!("run: {summary}?  [y/N] "), "", PendingCmd::ConfirmPlan);
    }

    /// Execute a command plan: resolve the selector into the working set, then run the pipeline in
    /// order over the existing primitives.
    fn run_plan(&mut self, plan: nl::CommandPlan) {
        if let Some(sel) = plan.select.as_deref() {
            match sel.trim() {
                "selected" | "selection" => {} // keep the current selection
                "all" | "" => self.selected = self.view.iter().copied().collect(),
                filter => {
                    // Treat as a filter expression: narrow the view + select every match.
                    self.filter = filter.to_string();
                    self.rebuild_view();
                    self.selected = self.view.iter().copied().collect();
                }
            }
        }
        for action in plan.actions {
            self.run_action(action);
        }
    }

    fn run_action(&mut self, action: nl::Action) {
        use nl::Action;
        match action {
            Action::Rate { stars } => self.edit_targets(|r| r.rating = stars.min(5)),
            Action::Flag => self.edit_targets(|r| r.flagged = true),
            Action::Reject => self.edit_targets(|r| r.rejected = true),
            Action::Color { label } => self.edit_targets(|r| r.color_label = Some(label.clone())),
            Action::Tag { tags } => self.edit_targets(|r| {
                for t in &tags {
                    if !r.tags.contains(t) {
                        r.tags.push(t.clone());
                    }
                }
            }),
            Action::Autotag => self.enqueue_vision_over_targets(vision::VisionOp::Autotag),
            Action::Describe => self.enqueue_vision_over_targets(vision::VisionOp::Describe),
            Action::Upscale => self.enqueue_ml_over_targets(mledit::MlOp::Upscale),
            Action::Img2img { prompt } => self.enqueue_ml_over_targets(mledit::MlOp::Img2img { prompt }),
            Action::Relight { prompt } => self.enqueue_ml_over_targets(mledit::MlOp::Relight { prompt }),
            Action::Generate { prompt } => self.ai_generate(&prompt),
            Action::Portrait { prompt } => self.ai_portrait(&prompt),
            Action::Multiperson { prompt } => self.ai_multiperson(&prompt),
            Action::Edit { op } => match edit::EditOp::from_tag(&op) {
                Some(eop) => self.batch_edit(eop),
                None => self.status = format!("unknown edit op: {op}"),
            },
            Action::Export { dir, max_px } => self.export_targets(&dir, max_px),
            Action::Rename { pattern } => {
                self.batch_rename(&pattern);
                self.rescan();
            }
            Action::Sort { by } => {
                self.album_meta.sort = Some(by);
                self.save_album();
                self.sort_album();
                self.rebuild_view();
            }
            Action::Dedup => self.dedup_scan(),
            Action::Stack => self.toggle_stack(),
            Action::StripMeta => self.strip_metadata_targets(),
            Action::RedactGps => self.redact_gps_targets(),
            Action::Take => self.take_photo(),
            Action::PutBack => self.promote_to_parent(),
            Action::Duplicate => self.duplicate_in_album(),
            Action::Panorama { mode } => self.stitch_panorama(stitch::PanoMode::from_i32(mode)),
            Action::Collage => self.make_collage(),
            Action::Mosaic => self.make_mosaic(),
            Action::Hdr => self.make_multishot(false),
            Action::FocusStack => self.make_multishot(true),
            Action::QualityCull => self.quality_cull(),
            Action::Flatten => {
                let dir = self.album_dir.clone().unwrap_or_else(|| self.root.path.clone());
                self.open_recursive(dir);
            }
            Action::Trash => self.trash_targets(),
            Action::RestoreTrash => self.restore_trash(),
            Action::EmptyTrash => self.empty_trash(),
            Action::OpenTrash => self.open_trash(),
            Action::Convert { fmt, max_px } => {
                let size = max_px.map(scrub::ConvertSize::MaxPx).unwrap_or(scrub::ConvertSize::Keep);
                self.convert_targets(&fmt, size);
            }
            Action::SmartAlbum { name } => {
                let query = self.filter.trim().to_string();
                if query.is_empty() {
                    self.status = "smart album needs a filter/selector".into();
                } else {
                    let mut fm = hjson::read_folder(&self.root_dir).unwrap_or_default();
                    fm.smart_albums.retain(|s| s.name != name);
                    fm.smart_albums.push(hjson::SmartAlbum { name: name.clone(), query });
                    let _ = hjson::write_folder(&self.root_dir, &fm);
                    self.smart_albums = fm.smart_albums;
                    self.status = format!("saved smart album '{name}'");
                }
            }
        }
    }

    /// Apply a T1 pixel edit to every browse target (selection, else the whole view).
    fn batch_edit(&mut self, op: edit::EditOp) {
        let paths: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        let mut n = 0;
        for p in paths {
            let Some((dir, filename)) = self.source_of(&p) else { continue };
            let _ = edit::ensure_backup(&dir, &filename);
            self.edit_record_at(&p, |r| r.edits.push(op.clone().to_entry()));
            let ops: Vec<edit::EditOp> = self
                .record(&p)
                .map(|r| r.edits.iter().filter_map(edit::EditOp::from_entry).collect())
                .unwrap_or_default();
            if edit::rebuild_file(&dir, &filename, &ops).is_ok() {
                self.thumbs.remove(&p);
                n += 1;
            }
        }
        self.status = format!("{} · {n} image(s)", op.label());
    }

    /// Enqueue an ML job per browse target (batch upscale / img2img / relight).
    fn enqueue_ml_over_targets(&mut self, op: mledit::MlOp) {
        let paths: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        let mut n = 0;
        for p in paths {
            if let Some((dir, _)) = self.source_of(&p) {
                self.jobs.push_back(Job::Ml(mledit::MlJob { op: op.clone(), input: p, album: dir }));
                n += 1;
            }
        }
        let warn = if self.mem_low() { "  ⚠ memory low — may be refused" } else { "" };
        self.status = format!("queued {n} × {} — runs in the background{warn}", op.label());
    }

    /// Enqueue a vision job per browse target (batch autotag / describe).
    fn enqueue_vision_over_targets(&mut self, op: vision::VisionOp) {
        let paths: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        let mut n = 0;
        for p in paths {
            self.jobs.push_back(Job::Vision(op, p));
            n += 1;
        }
        self.status = format!("queued {n} × {}", op.label());
    }

    /// The album_paths indices the next *browse* op applies to: the multi-selection if any, else the
    /// whole current view (unlike curation's `targets`, which falls back to the single cursor image).
    fn browse_targets(&self) -> Vec<usize> {
        if self.selected.is_empty() {
            self.view.clone()
        } else {
            let mut v: Vec<usize> = self.selected.iter().copied().collect();
            v.sort_unstable();
            v
        }
    }

    /// Scan the current view for near-duplicates (perceptual dHash), tag every duplicate-of-a-kept
    /// image `dup`, and narrow the view to `tag:dup` so they can be reviewed / culled. The best image
    /// per group (highest rating, then score, then first) is kept untagged.
    fn dedup_scan(&mut self) {
        let paths: Vec<PathBuf> =
            self.view.iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        if paths.len() < 2 {
            self.status = "need at least 2 images to scan for duplicates".into();
            return;
        }
        self.status = format!("hashing {} images…", paths.len());
        let hashes: Vec<(PathBuf, u64)> = paths
            .iter()
            .filter_map(|p| loader::thumbnail(p, 64).ok().map(|img| (p.clone(), dedup::dhash(&img))))
            .collect();
        let groups = dedup::find_duplicates(&hashes, 5);
        let mut dups = 0;
        for group in &groups {
            // Keep the best; tag the rest `dup`.
            let best = group
                .iter()
                .max_by(|a, b| {
                    let ra = self.record(a);
                    let rb = self.record(b);
                    let key = |r: Option<&hjson::ImageRecord>| {
                        (r.map_or(0, |x| x.rating), r.and_then(|x| x.score).unwrap_or(f64::MIN))
                    };
                    key(ra).partial_cmp(&key(rb)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .cloned();
            for p in group {
                if Some(p) != best.as_ref() {
                    self.edit_record_at(p, |rec| {
                        if !rec.tags.iter().any(|t| t == "dup") {
                            rec.tags.push("dup".into());
                        }
                    });
                    dups += 1;
                }
            }
        }
        if dups == 0 {
            self.status = "no near-duplicates found".into();
        } else {
            self.filter = "tag:dup".into();
            self.rebuild_view();
            self.status = format!("{dups} duplicate(s) in {} group(s) tagged `dup` (C to cull)", groups.len());
        }
    }

    /// Export the browse targets (selection, else the whole view) to `dir`, optionally downscaling to
    /// a `max_px` longer side.
    fn export_targets(&mut self, dir: &str, max_px: Option<u32>) {
        let files: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        if files.is_empty() {
            self.status = "nothing to export".into();
            return;
        }
        let dest = expand_tilde(dir);
        match export::export(&files, &dest, max_px) {
            Ok(n) => self.status = format!("exported {n} image(s) → {}", dest.display()),
            Err(e) => self.status = format!("export failed: {e:#}"),
        }
    }

    /// The target images as `(source album dir, full path)` pairs — the selection, else the cursor.
    /// Routes to each image's own album even in a smart-album view.
    fn target_sources(&self) -> Vec<(PathBuf, PathBuf)> {
        self.targets()
            .iter()
            .filter_map(|&i| {
                let p = self.album_paths.get(i)?.clone();
                let dir = if self.smart.is_some() {
                    self.smart_src.get(&p)?.clone()
                } else {
                    self.album_dir.clone()?
                };
                Some((dir, p))
            })
            .collect()
    }

    /// Strip EXIF/XMP/IPTC/GPS metadata from the target files (in place; JPEG/PNG lossless).
    fn strip_metadata_targets(&mut self) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "nothing to strip".into();
            return;
        }
        let (mut ok, mut reencoded, mut err) = (0, 0, 0);
        for (_dir, path) in &files {
            match scrub::strip_metadata(path) {
                Ok(true) => ok += 1,
                Ok(false) => {
                    ok += 1;
                    reencoded += 1;
                }
                Err(_) => err += 1,
            }
            self.thumbs.remove(path);
        }
        if self.mode == AlbumMode::Image {
            self.load_view();
        }
        let extra = if reencoded > 0 { format!(" ({reencoded} re-encoded)") } else { String::new() };
        self.status = if err > 0 {
            format!("stripped {ok} · {err} failed{extra}")
        } else {
            format!("stripped metadata from {ok} image(s){extra}")
        };
    }

    /// Redact only the GPS location from the target files (in place; keeps the rest of the EXIF).
    fn redact_gps_targets(&mut self) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "nothing to redact".into();
            return;
        }
        let (mut redacted, mut none, mut err) = (0, 0, 0);
        for (_dir, path) in &files {
            match scrub::redact_gps(path) {
                Ok(true) => redacted += 1,
                Ok(false) => none += 1,
                Err(_) => err += 1,
            }
        }
        self.status = if err > 0 {
            format!("GPS redacted from {redacted} · {none} had none · {err} unsupported")
        } else {
            format!("GPS redacted from {redacted} image(s) · {none} had none")
        };
    }

    /// Stitch the selected image(s) into a panorama (horizontal / vertical / grid) → a new
    /// `panorama.png` in the album. Order follows the album's sort.
    fn stitch_panorama(&mut self, mode: stitch::PanoMode) {
        let files = self.target_sources();
        if files.len() < 2 {
            self.status = "select 2+ images (Space) for a panorama".into();
            return;
        }
        let Some(dir) = self.album_dir.clone().or_else(|| files.first().map(|(d, _)| d.clone())) else {
            self.status = "no album".into();
            return;
        };
        let imgs: Vec<image::RgbImage> =
            files.iter().filter_map(|(_, p)| loader::load(p).ok().map(|i| i.to_rgb8())).collect();
        if imgs.len() < 2 {
            self.status = "couldn't load enough images".into();
            return;
        }
        match stitch::panorama(&imgs, mode) {
            Ok(pano) => {
                let dest = variant_path(&dir, "panorama", mode.label(), "png");
                if pano.save(&dest).is_ok() {
                    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    self.rescan();
                    self.select_by_name(&name);
                    self.status = format!("stitched {} images → {} panorama '{name}'", imgs.len(), mode.label());
                } else {
                    self.status = "couldn't save the panorama".into();
                }
            }
            Err(e) => self.status = format!("panorama failed: {e:#}"),
        }
    }

    /// Build a collage from the workbench: the selected images, or the whole album if none are
    /// selected. Auto-columned padded grid → a new `collage.png` in the album.
    fn make_collage(&mut self) {
        let paths: Vec<PathBuf> = if self.selected.is_empty() {
            self.album_paths.clone()
        } else {
            self.target_sources().into_iter().map(|(_, p)| p).collect()
        };
        if paths.len() < 2 {
            self.status = "need 2+ images for a collage".into();
            return;
        }
        let Some(dir) = self.album_dir.clone() else {
            self.status = "open an album first".into();
            return;
        };
        let imgs: Vec<image::RgbImage> =
            paths.iter().filter_map(|p| loader::load(p).ok().map(|i| i.to_rgb8())).collect();
        match stitch::collage(&imgs, 10) {
            Ok(c) => {
                let dest = variant_path(&dir, "collage", "grid", "png");
                if c.save(&dest).is_ok() {
                    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    self.rescan();
                    self.select_by_name(&name);
                    self.status = format!("collaged {} images → '{name}'", imgs.len());
                } else {
                    self.status = "couldn't save the collage".into();
                }
            }
            Err(e) => self.status = format!("collage failed: {e:#}"),
        }
    }

    /// Build a **mosaic / scrapbook** collage (justified rows — varied cell sizes) from the selected
    /// images, or the whole album if none are selected → a new `mosaic.png` in the album.
    fn make_mosaic(&mut self) {
        let paths: Vec<PathBuf> = if self.selected.is_empty() {
            self.album_paths.clone()
        } else {
            self.target_sources().into_iter().map(|(_, p)| p).collect()
        };
        if paths.len() < 2 {
            self.status = "need 2+ images for a mosaic".into();
            return;
        }
        let Some(dir) = self.album_dir.clone() else {
            self.status = "open an album first".into();
            return;
        };
        let imgs: Vec<image::RgbImage> =
            paths.iter().filter_map(|p| loader::load(p).ok().map(|i| i.to_rgb8())).collect();
        match stitch::mosaic(&imgs, 1600, 340, 10) {
            Ok(c) => {
                let dest = variant_path(&dir, "mosaic", "scrapbook", "png");
                if c.save(&dest).is_ok() {
                    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    self.rescan();
                    self.select_by_name(&name);
                    self.status = format!("mosaic of {} images → '{name}'", imgs.len());
                } else {
                    self.status = "couldn't save the mosaic".into();
                }
            }
            Err(e) => self.status = format!("mosaic failed: {e:#}"),
        }
    }

    /// Multi-shot composite over the selection (or whole album): `hdr` = exposure fusion, `focus` =
    /// focus stacking → a new `hdr.png` / `focus.png` in the album. Best for tripod bursts/brackets.
    fn make_multishot(&mut self, focus: bool) {
        let paths: Vec<PathBuf> = if self.selected.is_empty() {
            self.album_paths.clone()
        } else {
            self.target_sources().into_iter().map(|(_, p)| p).collect()
        };
        if paths.len() < 2 {
            self.status = "need 2+ frames (select a burst / bracket with Space)".into();
            return;
        }
        let Some(dir) = self.album_dir.clone() else {
            self.status = "open an album first".into();
            return;
        };
        let imgs: Vec<image::RgbImage> =
            paths.iter().filter_map(|p| loader::load(p).ok().map(|i| i.to_rgb8())).collect();
        let (res, kind) = if focus {
            (multishot::focus_stack(&imgs), "focus")
        } else {
            (multishot::exposure_fusion(&imgs), "hdr")
        };
        match res {
            Ok(c) => {
                let dest = variant_path(&dir, kind, "stack", "png");
                if c.save(&dest).is_ok() {
                    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
                    self.rescan();
                    self.select_by_name(&name);
                    self.status = format!("{kind} of {} frames → '{name}'", imgs.len());
                } else {
                    self.status = format!("couldn't save the {kind} result");
                }
            }
            Err(e) => self.status = format!("{kind} failed: {e:#}"),
        }
    }

    /// Non-AI quality cull: score every image in the current view for **sharpness** (Laplacian
    /// variance) and **exposure** (mean brightness), then reject the soft / badly-exposed ones
    /// (metadata only, undoable via the curation snapshot).
    fn quality_cull(&mut self) {
        let paths: Vec<PathBuf> = self.album_paths.clone();
        if paths.is_empty() {
            self.status = "no images to check".into();
            return;
        }
        // Adaptive sharpness floor: 35 % of the median sharpness (so it flags the genuinely-soft
        // outliers for this shoot, not a fixed absolute that mis-fires on low-detail scenes).
        let mut scores: Vec<(PathBuf, f32, f32)> = Vec::new();
        for p in &paths {
            if let Ok(img) = loader::load(p) {
                scores.push((p.clone(), quality::sharpness(&img), quality::brightness(&img)));
            }
        }
        if scores.is_empty() {
            self.status = "couldn't read the images".into();
            return;
        }
        let mut sharps: Vec<f32> = scores.iter().map(|s| s.1).collect();
        sharps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sharps[sharps.len() / 2];
        let floor = (median * 0.35).max(8.0);

        self.snapshot_meta();
        let (mut soft, mut under, mut over) = (0u32, 0u32, 0u32);
        for (p, sharp, bright) in &scores {
            match quality::judge(*sharp, *bright, floor) {
                quality::Cull::Keep => {}
                verdict => {
                    match verdict {
                        quality::Cull::Soft => soft += 1,
                        quality::Cull::Underexposed => under += 1,
                        quality::Cull::Overexposed => over += 1,
                        quality::Cull::Keep => {}
                    }
                    self.edit_record_at(p, |rec| rec.rejected = true);
                }
            }
        }
        self.rebuild_view();
        let total = soft + under + over;
        self.status = format!(
            "✓ quality cull: rejected {total} ({soft} soft · {under} dark · {over} bright) of {} — filter 'rejected' to review · undo with u",
            scores.len()
        );
    }

    /// Move (or copy) the target images into the album at `dest_arg` — the file, its `.json` sidecar,
    /// and its curation record all go along; on a move the record is removed from the source album.
    /// (Edit-undo history in `.plakat_edits/` is not carried — the moved file is its current state.)
    fn move_targets(&mut self, dest_arg: &str, copy: bool) {
        let dest = expand_tilde(dest_arg.trim());
        if !dest.is_dir() {
            self.status = format!("no album directory at {}", dest.display());
            return;
        }
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "select image(s) first".into();
            return;
        }
        let verb = if copy { "copied" } else { "moved" };
        let (mut ok, mut err) = (0u32, 0u32);
        let mut last = None;
        for (src_dir, path) in &files {
            if src_dir == &dest {
                err += 1; // no-op: already in the destination album
                continue;
            }
            let Some(fname) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                err += 1;
                continue;
            };
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let ext = path.extension().and_then(|e| e.to_str());
            let make = |s: &str| match ext {
                Some(e) => format!("{s}.{e}"),
                None => s.to_string(),
            };
            let mut name = make(stem);
            let mut i = 2;
            while dest.join(&name).exists() {
                name = make(&format!("{stem}-{i}"));
                i += 1;
            }
            let dest_path = dest.join(&name);
            let moved = if copy {
                std::fs::copy(path, &dest_path).is_ok()
            } else {
                std::fs::rename(path, &dest_path).is_ok()
                    || (std::fs::copy(path, &dest_path).is_ok() && std::fs::remove_file(path).is_ok())
            };
            if !moved {
                err += 1;
                continue;
            }
            let sc = path.with_extension("json");
            if sc.exists() {
                let dsc = dest_path.with_extension("json");
                if copy {
                    let _ = std::fs::copy(&sc, &dsc);
                } else if std::fs::rename(&sc, &dsc).is_err() && std::fs::copy(&sc, &dsc).is_ok() {
                    let _ = std::fs::remove_file(&sc);
                }
            }
            let rec = self.record(path).cloned();
            if let Some(r) = rec {
                let key = name.clone();
                self.edit_album_meta_at(&dest, |m| {
                    m.images.insert(key, r);
                });
            }
            if !copy {
                self.edit_album_meta_at(src_dir, |m| {
                    m.images.remove(&fname);
                });
            }
            last = Some(name);
            ok += 1;
        }
        self.selected.clear();
        self.rescan();
        if let Some(n) = &last {
            self.select_by_name(n);
        }
        let dname = dest.file_name().and_then(|n| n.to_str()).unwrap_or("album");
        let tail = if err > 0 { format!(", {err} skipped/failed") } else { String::new() };
        self.status = format!("{verb} {ok} image(s) → '{dname}'{tail}");
    }

    /// Purge the **entire** persistent thumbnail cache (XDG `<cache>/plakat/photos/thumbs`) — every
    /// thumb is keyed by `sha256(path + size + mtime + byte-len)`, so this only reclaims disk; the
    /// thumbs rebuild lazily on the next browse. (Per-album regeneration is the tree's `regen`.)
    fn clear_thumb_cache(&mut self) {
        let dir = loader::thumb_cache_dir();
        self.thumbs.clear();
        match std::fs::remove_dir_all(&dir) {
            Ok(_) => self.status = "🧹 thumbnail cache cleared (rebuilds on browse)".into(),
            Err(_) if !dir.exists() => self.status = "thumbnail cache already empty".into(),
            Err(e) => self.status = format!("couldn't clear thumbnail cache: {e}"),
        }
    }

    /// The library's hidden recycle bin (`<root>/.trash`). Skipped by the tree walk (dot-dir).
    fn trash_dir(&self) -> PathBuf {
        self.root.path.join(".trash")
    }

    /// Soft-delete: move the target images (+ sidecar) into `.trash`, recording each one's origin in
    /// `.trash/.manifest` so it can be restored. The curation record is removed from the source album.
    fn trash_targets(&mut self) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "select image(s) to trash".into();
            return;
        }
        let trash = self.trash_dir();
        if std::fs::create_dir_all(&trash).is_err() {
            self.status = "couldn't create the trash folder".into();
            return;
        }
        let manifest = trash.join(".manifest");
        let mut log = std::fs::read_to_string(&manifest).unwrap_or_default();
        let (mut ok, mut err) = (0u32, 0u32);
        for (src_dir, path) in &files {
            let Some(fname) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else {
                err += 1;
                continue;
            };
            let mut name = fname.clone();
            let mut i = 2;
            while trash.join(&name).exists() {
                name = format!("{i}_{fname}");
                i += 1;
            }
            let dest = trash.join(&name);
            let moved = std::fs::rename(path, &dest).is_ok()
                || (std::fs::copy(path, &dest).is_ok() && std::fs::remove_file(path).is_ok());
            if !moved {
                err += 1;
                continue;
            }
            let sc = path.with_extension("json");
            if sc.exists() {
                let dsc = dest.with_extension("json");
                if std::fs::rename(&sc, &dsc).is_err() && std::fs::copy(&sc, &dsc).is_ok() {
                    let _ = std::fs::remove_file(&sc);
                }
            }
            // Manifest: trashed_name \t original_dir \t original_filename.
            log.push_str(&format!("{name}\t{}\t{fname}\n", src_dir.display()));
            self.edit_album_meta_at(src_dir, |m| {
                m.images.remove(&fname);
            });
            ok += 1;
        }
        let _ = std::fs::write(&manifest, log);
        self.selected.clear();
        self.rescan();
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status = format!("🗑 trashed {ok} image(s){tail} · ':restore' / ':empty trash'");
    }

    /// Restore every image in `.trash` to its recorded origin (deduped if the name is taken).
    fn restore_trash(&mut self) {
        let trash = self.trash_dir();
        let manifest = trash.join(".manifest");
        let Ok(content) = std::fs::read_to_string(&manifest) else {
            self.status = "trash is empty".into();
            return;
        };
        let (mut ok, mut err) = (0u32, 0u32);
        let mut remaining = String::new();
        for line in content.lines() {
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            let [name, odir, ofname] = parts[..] else { continue };
            let src = trash.join(name);
            if !src.exists() {
                continue; // manifest entry with no file — drop it
            }
            let odir = PathBuf::from(odir);
            let _ = std::fs::create_dir_all(&odir);
            let (stem, ext) = (
                std::path::Path::new(ofname).file_stem().and_then(|s| s.to_str()).unwrap_or("image"),
                std::path::Path::new(ofname).extension().and_then(|e| e.to_str()),
            );
            let make = |s: &str| match ext {
                Some(e) => format!("{s}.{e}"),
                None => s.to_string(),
            };
            let mut target = make(stem);
            let mut i = 2;
            while odir.join(&target).exists() {
                target = make(&format!("{stem}-{i}"));
                i += 1;
            }
            let dest = odir.join(&target);
            if std::fs::rename(&src, &dest).is_ok()
                || (std::fs::copy(&src, &dest).is_ok() && std::fs::remove_file(&src).is_ok())
            {
                let sc = src.with_extension("json");
                if sc.exists() {
                    let _ = std::fs::rename(&sc, dest.with_extension("json"));
                }
                ok += 1;
            } else {
                remaining.push_str(line);
                remaining.push('\n');
                err += 1;
            }
        }
        let _ = std::fs::write(&manifest, remaining);
        self.rescan();
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status = format!("♻ restored {ok} image(s) from trash{tail}");
    }

    /// Permanently delete everything in `.trash`.
    fn empty_trash(&mut self) {
        let trash = self.trash_dir();
        match std::fs::remove_dir_all(&trash) {
            Ok(_) | Err(_) if !trash.exists() => self.status = "🗑 trash emptied".into(),
            _ => self.status = "couldn't empty the trash".into(),
        }
    }

    /// Browse the trash (the hidden `.trash` folder the tree skips).
    fn open_trash(&mut self) {
        let trash = self.trash_dir();
        if !trash.is_dir() || Self::gather_image_files(&trash, false).is_empty() {
            self.status = "trash is empty".into();
            return;
        }
        self.open_recursive(trash);
        self.status = format!("{} · 🗑 trash · ':restore' to put back · ':empty trash' to purge", self.view.len());
    }

    /// "Take photo for processing": copy the highest-resolution version of each target image into a
    /// new **nested sub-album** (named from the image), so destructive edits + artefacts stay
    /// isolated from the source album. Multi-select aware.
    fn take_photo(&mut self) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "select image(s) first (Space to multi-select), then take".into();
            return;
        }
        let Some(parent) = self.album_dir.clone().or_else(|| files.first().map(|(d, _)| d.clone())) else {
            self.status = "no album to take from".into();
            return;
        };
        // Sub-album name derived from the (first) image.
        let stem = files[0].1.file_stem().and_then(|s| s.to_str()).unwrap_or("photo");
        let base = if files.len() == 1 { stem.to_string() } else { format!("{stem}+{}", files.len() - 1) };
        let subname = sanitize_name(&base);
        let mut subdir = parent.join(&subname);
        let mut i = 2;
        while subdir.exists() {
            subdir = parent.join(format!("{subname}-{i}"));
            i += 1;
        }
        if let Err(e) = std::fs::create_dir_all(&subdir) {
            self.status = format!("couldn't create sub-album: {e}");
            return;
        }
        let mut meta = hjson::AlbumMeta {
            name: Some(base.clone()),
            description: Some(format!(
                "Working copies taken from {}",
                parent.file_name().and_then(|n| n.to_str()).unwrap_or("album")
            )),
            ..Default::default()
        };
        let (mut ok, mut err) = (0u32, 0u32);
        for (dir, path) in &files {
            let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("photo").to_string();
            // Candidates: the file, its pristine backup, and any variants — pick the largest by area.
            let mut cands: Vec<PathBuf> = vec![path.clone()];
            let bak = edit::backup_path(dir, &filename);
            if bak.exists() {
                cands.push(bak);
            }
            if let Some(rec) = self.record(path) {
                for v in &rec.variants {
                    let vp = dir.join(v);
                    if vp.exists() {
                        cands.push(vp);
                    }
                }
            }
            let best = cands
                .iter()
                .max_by_key(|p| image::image_dimensions(p).map(|(w, h)| w as u64 * h as u64).unwrap_or(0))
                .cloned()
                .unwrap_or_else(|| path.clone());
            if std::fs::copy(&best, subdir.join(&filename)).is_err() {
                err += 1;
                continue;
            }
            let sc = path.with_extension("json");
            if sc.exists() {
                let _ = std::fs::copy(&sc, subdir.join(&filename).with_extension("json"));
            }
            // Carry curation, but clear the replay state so the copy is a clean high-res baseline.
            if let Some(rec) = self.record(path) {
                let mut r = rec.clone();
                r.edits.clear();
                r.variants.clear();
                r.layers.clear();
                r.generation = None;
                meta.images.insert(filename, r);
            }
            ok += 1;
        }
        let _ = hjson::write_album(&subdir, &meta);
        if let Ok(root) = library::walk(&self.root_dir) {
            self.root = root;
        }
        self.expanded.insert(parent);
        if let Some(pos) = self.rows().iter().position(|r| r.path == subdir) {
            self.tree_cursor = pos;
        }
        // Switch straight into the new workbench album so you can start editing there.
        self.open_album(subdir);
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status =
            format!("took {ok} photo(s) → workbench '{subname}'{tail} — edit here, source stays clean");
    }

    /// Duplicate the target image(s) **within their album** (a `<stem>_copy` next to the original),
    /// so you can try an alternate edit without touching the original. Curation is carried; the edit
    /// baseline starts clean. Multi-select aware.
    fn duplicate_in_album(&mut self) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "select image(s) to duplicate".into();
            return;
        }
        let (mut ok, mut err) = (0u32, 0u32);
        let mut last = None;
        for (dir, path) in &files {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("image");
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
            let mut name = format!("{stem}_copy.{ext}");
            let mut i = 2;
            while dir.join(&name).exists() {
                name = format!("{stem}_copy-{i}.{ext}");
                i += 1;
            }
            if std::fs::copy(path, dir.join(&name)).is_err() {
                err += 1;
                continue;
            }
            let sc = path.with_extension("json");
            if sc.exists() {
                let _ = std::fs::copy(&sc, dir.join(&name).with_extension("json"));
            }
            let rec = self.record(path).map(|r| {
                let mut r = r.clone();
                r.edits.clear();
                r.variants.clear();
                r.layers.clear();
                r
            });
            let key = name.clone();
            self.edit_album_meta_at(dir, |m| {
                if let Some(r) = rec {
                    m.images.insert(key, r);
                }
            });
            last = Some(name);
            ok += 1;
        }
        self.rescan();
        if let Some(n) = &last {
            self.select_by_name(n);
        }
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status = format!("duplicated {ok} image(s){tail}");
    }

    /// "Put back": copy the selected finished image(s) from a workbench sub-album up to its **parent
    /// album** (deduped, curation carried, replay state cleared), so the sub-album can then be
    /// deleted — keeping only the results you want.
    fn promote_to_parent(&mut self) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "select image(s) to put back".into();
            return;
        }
        let Some(sub) = self.album_dir.clone() else {
            self.status = "open the workbench sub-album first".into();
            return;
        };
        let Some(parent) = sub.parent().filter(|p| p.is_dir() && *p != self.root_dir).map(|p| p.to_path_buf())
        else {
            self.status = "no parent album to put back into".into();
            return;
        };
        let mut pmeta = hjson::read_album(&parent).unwrap_or_default();
        let (mut ok, mut err) = (0u32, 0u32);
        for (_dir, path) in &files {
            let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("photo").to_string();
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("photo");
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("png");
            // Don't clobber the original (same name); the promoted, edited copy is deduped.
            let mut dest = filename.clone();
            let mut i = 2;
            while parent.join(&dest).exists() {
                dest = format!("{stem}-{i}.{ext}");
                i += 1;
            }
            if std::fs::copy(path, parent.join(&dest)).is_err() {
                err += 1;
                continue;
            }
            let sc = path.with_extension("json");
            if sc.exists() {
                let _ = std::fs::copy(&sc, parent.join(&dest).with_extension("json"));
            }
            if let Some(rec) = self.record(path) {
                let mut r = rec.clone();
                r.edits.clear();
                r.variants.clear();
                r.layers.clear();
                pmeta.images.insert(dest, r);
            }
            ok += 1;
        }
        let _ = hjson::write_album(&parent, &pmeta);
        if let Ok(root) = library::walk(&self.root_dir) {
            self.root = root;
        }
        let tail = if err > 0 { format!(", {err} failed") } else { String::new() };
        self.status =
            format!("put back {ok} image(s) → parent album{tail} — safe to delete this sub-album now");
    }

    /// Add a watermark / caption as a **replayable** edit on the target image(s). `spec` = `TEXT` or
    /// `TEXT | /path/font.ttf` to pick a font. Non-destructive (rebuilds from the pristine backup),
    /// undoable, and re-rendered on every replay.
    fn watermark_edit(&mut self, spec: &str) {
        let (text, font) = match spec.split_once('|') {
            Some((t, f)) => {
                let p = expand_tilde(f.trim());
                (t.trim().to_string(), p.exists().then(|| p.to_string_lossy().into_owned()))
            }
            None => (spec.trim().to_string(), None),
        };
        if text.is_empty() {
            self.status = "enter watermark text  (optionally: TEXT | /path/font.ttf)".into();
            return;
        }
        let entry = edit::EditOp::Watermark { text, font }.to_entry();
        self.apply_edit_ops_to_targets(&[entry], "watermarked");
    }

    /// Add a `.cube` 3D LUT grade as a **replayable** edit on the target image(s). Non-destructive;
    /// a missing/invalid file replays as a no-op so the stack never breaks.
    fn lut_edit(&mut self, cube: &str) {
        let path = expand_tilde(cube.trim());
        if !path.exists() {
            self.status = format!("no LUT file at {}", path.display());
            return;
        }
        // Validate up front so a typo is reported now rather than silently no-op'ing later.
        if let Err(e) = lut::load_cube(&path) {
            self.status = format!("LUT load failed: {e:#}");
            return;
        }
        let entry = edit::EditOp::Lut { path: path.to_string_lossy().into_owned() }.to_entry();
        self.apply_edit_ops_to_targets(&[entry], "graded (LUT)");
    }

    /// Convert the target images to `fmt`/`size`, landing a new file per source (deduped variant).
    fn convert_targets(&mut self, fmt: &str, size: scrub::ConvertSize) {
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "nothing to convert".into();
            return;
        }
        let (mut ok, mut err, mut last) = (0, 0, None);
        for (dir, path) in &files {
            match scrub::convert(path, dir, fmt, size) {
                Ok(name) => {
                    self.record_variant(path, &name);
                    last = Some(name);
                    ok += 1;
                }
                Err(e) => {
                    err += 1;
                    self.status = format!("convert failed: {e:#}");
                }
            }
        }
        if ok > 0 {
            self.rescan();
            if let Some(n) = &last {
                self.select_by_name(n);
            }
            self.status = if err > 0 {
                format!("converted {ok} · {err} failed")
            } else {
                format!("converted {ok} image(s) → {fmt}")
            };
        }
    }

    /// Export the browse targets as a portfolio (watermarked/resized copies + a contact sheet).
    fn export_portfolio(&mut self, dir: &str, max_px: Option<u32>, mark: Option<&str>) {
        let files: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        if files.is_empty() {
            self.status = "nothing to export".into();
            return;
        }
        let dest = expand_tilde(dir);
        match portfolio::export(&files, &dest, mark, max_px) {
            Ok(n) => self.status = format!("portfolio: {n} image(s) + contact sheet → {}", dest.display()),
            Err(e) => self.status = format!("portfolio failed: {e:#}"),
        }
    }

    /// Batch-rename the browse targets in the open album with `pattern` (`#` runs → sequence number).
    /// Album-local only (not in a smart/search view). Two-phase (→ hidden temp → final) so intra-set
    /// name swaps can't clobber, and each image's `album.hjson` record + edit backup migrate with it.
    fn batch_rename(&mut self, pattern: &str) {
        let Some(dir) = self.album_dir.clone() else {
            self.status = "open a real album to batch-rename".into();
            return;
        };
        let files: Vec<PathBuf> =
            self.browse_targets().iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        if files.is_empty() {
            self.status = "nothing to rename".into();
            return;
        }
        let plan = rename::plan(&files, pattern);
        let n = rename::apply(&dir, plan, &mut self.album_meta);
        self.save_album();
        self.status = format!("renamed {n} image(s)");
    }

    /// Link `out_name` as a derivative variant of the source image at `src` (deduped).
    fn record_variant(&mut self, src: &Path, out_name: &str) {
        let out = out_name.to_string();
        self.edit_record_at(src, |rec| {
            if !rec.variants.iter().any(|v| v == &out) {
                rec.variants.push(out);
            }
        });
    }

    /// Move the album cursor onto the image named `name`, if it's in the current view.
    fn select_by_name(&mut self, name: &str) {
        let Some(pi) = self
            .album_paths
            .iter()
            .position(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
        else {
            return;
        };
        if let Some(vpos) = self.view.iter().position(|&i| i == pi) {
            self.album_cursor = vpos;
        }
    }

    // ---- Copy/paste edits + presets --------------------------------------------------------------

    /// The cursor image's raw edit entries (for copy / save-preset).
    fn cur_edit_entries(&self) -> Vec<hjson::EditEntry> {
        self.cur_idx()
            .and_then(|i| self.album_paths.get(i))
            .and_then(|p| self.record(p))
            .map(|r| r.edits.clone())
            .unwrap_or_default()
    }

    /// Apply a built-in look preset (a fixed edit sequence) to the target image(s).
    fn apply_look(&mut self, i: usize) {
        let Some((label, _, ops)) = look_presets().into_iter().nth(i) else { return };
        let entries: Vec<hjson::EditEntry> = ops.into_iter().map(|o| o.to_entry()).collect();
        self.apply_edit_ops_to_targets(&entries, "applied");
        // Overwrite the generic status with the look name.
        if !self.status.starts_with("no ") {
            self.status = format!("applied {label}");
        }
    }

    /// Copy the cursor image's edit stack to the in-session clipboard.
    fn copy_edits(&mut self) {
        let ops = self.cur_edit_entries();
        if ops.is_empty() {
            self.status = "this image has no edits to copy".into();
            return;
        }
        self.status = format!("copied {} edit(s) — paste onto a selection", ops.len());
        self.edit_clipboard = ops;
    }

    /// Paste the copied edit stack onto the target images.
    fn paste_edits(&mut self) {
        if self.edit_clipboard.is_empty() {
            self.status = "clipboard empty — copy edits first".into();
            return;
        }
        let ops = self.edit_clipboard.clone();
        self.apply_edit_ops_to_targets(&ops, "pasted");
    }

    /// Append `ops` to every target image's edit log and rebuild each from its pristine original.
    fn apply_edit_ops_to_targets(&mut self, ops: &[hjson::EditEntry], verb: &str) {
        let valid = ops.iter().filter(|e| edit::EditOp::from_entry(e).is_some()).count();
        if valid == 0 {
            self.status = "no applicable edits".into();
            return;
        }
        let files = self.target_sources();
        if files.is_empty() {
            self.status = "no target image".into();
            return;
        }
        let mut n = 0;
        for (dir, path) in &files {
            let Some(filename) = path.file_name().and_then(|f| f.to_str()).map(|s| s.to_string()) else {
                continue;
            };
            if edit::ensure_backup(dir, &filename).is_err() {
                continue;
            }
            self.edit_record_at(path, |rec| rec.edits.extend(ops.iter().cloned()));
            let full: Vec<edit::EditOp> = self
                .record(path)
                .map(|r| r.edits.iter().filter_map(edit::EditOp::from_entry).collect())
                .unwrap_or_default();
            if edit::rebuild_file(dir, &filename, &full).is_ok() {
                self.thumbs.remove(path);
                n += 1;
            }
        }
        self.edit_redo.clear();
        if self.mode == AlbumMode::Image {
            self.load_view();
        }
        self.status = format!("{verb} {valid} edit(s) → {n} image(s)");
    }

    /// Save the cursor image's edit stack as a named preset in the root `folder.hjson`.
    fn save_preset(&mut self, name: &str) {
        let ops = self.cur_edit_entries();
        if ops.is_empty() {
            self.status = "this image has no edits to save".into();
            return;
        }
        let mut fm = hjson::read_folder(&self.root_dir).unwrap_or_default();
        fm.edit_presets.retain(|p| p.name != name);
        let count = ops.len();
        fm.edit_presets.push(hjson::EditPreset { name: name.to_string(), ops });
        if let Err(e) = hjson::write_folder(&self.root_dir, &fm) {
            self.status = format!("save failed: {e}");
            return;
        }
        self.status = format!("saved preset '{name}' ({count} edits)");
    }

    /// Open the preset picker (apply-preset modal).
    fn open_preset_browser(&mut self) {
        let fm = hjson::read_folder(&self.root_dir).unwrap_or_default();
        if fm.edit_presets.is_empty() {
            self.status = "no presets saved yet — 'save edits as preset' first".into();
            return;
        }
        self.presets_list = fm.edit_presets;
        self.preset_cursor = 0;
        self.preset_browser = true;
    }

    /// After a file's pixels change: drop its cached thumbnail (so it re-decodes) and reload the
    /// full-pane image if it's on screen.
    fn refresh_after_edit(&mut self, path: &Path) {
        self.thumbs.remove(path);
        self.view_base = None; // the file changed on disk — re-decode the working base
        if self.mode == AlbumMode::Image {
            self.load_view();
        }
    }

    fn save_album(&mut self) {
        if let Some(dir) = &self.album_dir {
            if let Err(e) = hjson::write_album(dir, &self.album_meta) {
                self.status = format!("save failed: {e}");
            }
        }
    }

    /// Re-order `album_paths` per the album's persisted `sort` (default `name-asc`). Rating/score
    /// come from the smart-view maps in a smart album, else from `album_meta`.
    fn sort_album(&mut self) {
        let mode = self.album_meta.sort.clone().unwrap_or_else(|| "name-asc".into());
        let mut paths = std::mem::take(&mut self.album_paths);
        let rating = |p: &Path| self.record(p).map_or(0, |r| r.rating);
        let score = |p: &Path| self.record(p).and_then(|r| r.score).unwrap_or(f64::MIN);
        sort_paths(&mut paths, &mode, rating, score);
        self.album_paths = paths;
    }

    /// Cycle to the next sort order, persist it, re-sort, and keep the cursor on the same image.
    fn cycle_sort(&mut self) {
        let cur = self.album_meta.sort.as_deref().unwrap_or("name-asc");
        let i = SORT_ORDER.iter().position(|m| *m == cur).unwrap_or(0);
        let next = SORT_ORDER[(i + 1) % SORT_ORDER.len()];
        self.album_meta.sort = Some(next.to_string());
        self.save_album();
        let cur_path = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned());
        self.selected.clear(); // indices are about to be reshuffled
        self.sort_album();
        self.rebuild_view();
        if let Some(cp) = cur_path {
            if let Some(pos) = self.view.iter().position(|&pi| self.album_paths.get(pi) == Some(&cp)) {
                self.album_cursor = pos;
            }
        }
        self.status = format!("sort: {next}  (s to cycle)");
    }

    /// Open the command pane to edit a free-text metadata field of the cursor image, prefilled with
    /// its current value.
    fn begin_edit(&mut self, field: EditField) {
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        let rec = self.record(&path);
        let (label, prefill) = match field {
            EditField::Caption => ("caption: ", rec.and_then(|r| r.caption.clone()).unwrap_or_default()),
            EditField::Notes => ("notes: ", rec.and_then(|r| r.notes.clone()).unwrap_or_default()),
            EditField::Title => ("title: ", rec.and_then(|r| r.title.clone()).unwrap_or_default()),
            EditField::Tags => ("tags (comma-sep): ", rec.map(|r| r.tags.join(", ")).unwrap_or_default()),
            EditField::Author => ("author / creator: ", rec.and_then(|r| r.author.clone()).unwrap_or_default()),
            EditField::Copyright => ("copyright: ", rec.and_then(|r| r.copyright.clone()).unwrap_or_default()),
            EditField::DateTaken => (
                "capture date (YYYY-MM-DD or YYYY:MM:DD HH:MM:SS): ",
                rec.and_then(|r| r.exif.as_ref()).and_then(|e| e.date_taken.clone()).unwrap_or_default(),
            ),
            EditField::Geotag => (
                "geotag — lat, lon (decimal): ",
                rec.and_then(|r| r.exif.as_ref())
                    .and_then(|e| e.gps_lat.zip(e.gps_lon))
                    .map(|(la, lo)| format!("{la}, {lo}"))
                    .unwrap_or_default(),
            ),
        };
        self.prompt(label, prefill, PendingCmd::EditMeta { path, field });
    }

    /// Open the command pane for a pending action (RFC §11).
    fn prompt(&mut self, prompt: impl Into<String>, prefill: impl Into<String>, pending: PendingCmd) {
        self.cmd_prompt = prompt.into();
        self.cmd_buffer = prefill.into();
        self.pending = Some(pending);
        self.cmd_active = true;
    }

    /// Execute the pending command with `cmd_buffer` as the argument. Filesystem mutations trigger a
    /// full rescan; a metadata edit only rebuilds the filtered view (tags can change filter matches).
    fn commit_cmd(&mut self) {
        let arg = self.cmd_buffer.trim().to_string();
        let mut fs_changed = false;
        let mut meta_changed = false;
        let result: Result<()> = (|| {
            match self.pending.take() {
                Some(PendingCmd::NewFolder { parent }) if !arg.is_empty() => {
                    let dir = parent.join(&arg);
                    std::fs::create_dir_all(&dir)?;
                    hjson::write_folder(&dir, &hjson::FolderMeta::default())?;
                    fs_changed = true;
                }
                Some(PendingCmd::NewAlbum { parent }) if !arg.is_empty() => {
                    let dir = parent.join(&arg);
                    std::fs::create_dir_all(&dir)?;
                    hjson::write_album(&dir, &hjson::AlbumMeta::default())?;
                    fs_changed = true;
                }
                Some(PendingCmd::Rename { path }) if !arg.is_empty() => {
                    if let Some(parent) = path.parent() {
                        std::fs::rename(&path, parent.join(&arg))?;
                    }
                    fs_changed = true;
                }
                Some(PendingCmd::Delete { path, is_album }) if arg.eq_ignore_ascii_case("y") => {
                    if is_album {
                        std::fs::remove_dir_all(&path)?;
                    } else {
                        std::fs::remove_dir(&path)?; // folders: only if empty
                    }
                    fs_changed = true;
                }
                Some(PendingCmd::EditMeta { path, field }) => {
                    self.snapshot_meta();
                    let val = (!arg.is_empty()).then(|| arg.clone());
                    self.edit_record_at(&path, |rec| match field {
                        EditField::Caption => rec.caption = val,
                        EditField::Notes => rec.notes = val,
                        EditField::Title => rec.title = val,
                        EditField::Tags => rec.tags = parse_tags(&arg),
                        EditField::Author => rec.author = val,
                        EditField::Copyright => rec.copyright = val,
                        EditField::DateTaken => rec.exif.get_or_insert_with(Default::default).date_taken = val,
                        EditField::Geotag => {
                            let e = rec.exif.get_or_insert_with(Default::default);
                            if arg.trim().is_empty() {
                                e.gps_lat = None;
                                e.gps_lon = None;
                            } else {
                                let nums: Vec<f64> =
                                    arg.split([',', ' ']).filter_map(|t| t.trim().parse().ok()).collect();
                                if nums.len() >= 2 {
                                    e.gps_lat = Some(nums[0]);
                                    e.gps_lon = Some(nums[1]);
                                }
                            }
                        }
                    });
                    meta_changed = true;
                }
                Some(PendingCmd::SaveSmart { query }) if !arg.is_empty() => {
                    let mut fm = hjson::read_folder(&self.root_dir).unwrap_or_default();
                    fm.smart_albums.retain(|s| s.name != arg); // upsert by name
                    fm.smart_albums.push(hjson::SmartAlbum { name: arg.clone(), query });
                    hjson::write_folder(&self.root_dir, &fm)?;
                    self.smart_albums = fm.smart_albums;
                    self.status = format!("saved smart album '{arg}'");
                }
                Some(PendingCmd::Search) if !arg.is_empty() => {
                    self.open_search(arg.clone());
                }
                Some(PendingCmd::VisualSearch) if !arg.is_empty() => {
                    self.pending_visual = Some(arg.clone());
                    self.status = format!("visual search '{arg}' … (the UI will pause)");
                }
                Some(PendingCmd::AutoCull) => {
                    let n: usize = arg.trim().parse().unwrap_or(10).max(1);
                    self.pending_cull = Some(n);
                    self.status = format!("aesthetic auto-cull — keep top {n} … (the UI will pause)");
                }
                Some(PendingCmd::Generate) if !arg.is_empty() => self.ai_generate(&arg),
                Some(PendingCmd::Portrait) if !arg.is_empty() => self.ai_portrait(&arg),
                Some(PendingCmd::Multiperson) if !arg.is_empty() => self.ai_multiperson(&arg),
                Some(PendingCmd::MlPrompt { relight }) if !arg.is_empty() => {
                    let op = if relight {
                        mledit::MlOp::Relight { prompt: arg.clone() }
                    } else {
                        mledit::MlOp::Img2img { prompt: arg.clone() }
                    };
                    self.queue_ml(op);
                }
                Some(PendingCmd::Export) if !arg.is_empty() => {
                    // `DIR [MAXPX]` — a trailing integer sets a longer-side cap.
                    let mut it = arg.rsplitn(2, char::is_whitespace);
                    let last = it.next().unwrap_or("");
                    let (dir, max_px) = match (last.parse::<u32>().ok(), it.next()) {
                        (Some(px), Some(d)) => (d.trim().to_string(), Some(px)),
                        _ => (arg.clone(), None),
                    };
                    self.export_targets(&dir, max_px);
                }
                Some(PendingCmd::BatchRename) if !arg.is_empty() => {
                    self.batch_rename(&arg);
                    fs_changed = true; // files moved on disk
                }
                Some(PendingCmd::CropExact) if !arg.is_empty() => {
                    match parse_dims(&arg) {
                        Some((w, h)) => {
                            self.apply_edit(edit::EditOp::CropPx { w, h });
                            meta_changed = true;
                        }
                        None => self.status = "enter WxH pixels, e.g. 1200x800".into(),
                    }
                }
                Some(PendingCmd::ResizeExact) if !arg.is_empty() => {
                    match parse_dims(&arg) {
                        Some((w, h)) => {
                            self.apply_edit(edit::EditOp::Resize { w, h });
                            meta_changed = true;
                        }
                        None => self.status = "enter WxH, or a single number for the longer side".into(),
                    }
                }
                Some(PendingCmd::AddLayer) if !arg.is_empty() => {
                    self.add_layer(&arg);
                }
                Some(PendingCmd::MaskImage) if !arg.is_empty() => {
                    self.set_layer_mask_image(&arg);
                }
                Some(PendingCmd::AlbumEdit { path, field }) => {
                    let v = arg.trim().to_string();
                    let some = (!v.is_empty()).then(|| v.clone());
                    self.edit_album_meta_at(&path, |m| match field {
                        AlbumFieldKind::Name => m.name = some,
                        AlbumFieldKind::Description => m.description = some,
                        AlbumFieldKind::Cover => m.cover = some,
                        AlbumFieldKind::Sort => m.sort = some,
                        AlbumFieldKind::TagsReplace => {
                            m.tags =
                                v.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
                        }
                        AlbumFieldKind::TagsAdd => {
                            for t in v.split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
                                if !m.tags.contains(&t) {
                                    m.tags.push(t);
                                }
                            }
                        }
                    });
                    // Name changes affect the tree/grid label → re-walk so it shows immediately.
                    if let Ok(root) = library::walk(&self.root_dir) {
                        self.root = root;
                    }
                    self.status = "album metadata updated".into();
                }
                Some(PendingCmd::ExportAlbum { path, recursive }) if !arg.is_empty() => {
                    self.export_album_files(&path, recursive, &arg);
                }
                Some(PendingCmd::ExportConvertAlbum { path, recursive }) if !arg.is_empty() => {
                    self.export_convert_files(&path, recursive, &arg);
                }
                Some(PendingCmd::MaterializeSmart { name, query }) if !arg.is_empty() => {
                    self.materialize_smart(&name, &query, &arg);
                    fs_changed = true;
                }
                Some(PendingCmd::SavePreset) if !arg.is_empty() => {
                    self.save_preset(arg.trim());
                }
                Some(PendingCmd::Straighten) if !arg.is_empty() => match arg.trim().parse::<f32>() {
                    Ok(deg) => {
                        self.apply_edit(edit::EditOp::Straighten((deg * 10.0).round() as i32));
                        meta_changed = true;
                    }
                    Err(_) => self.status = "enter a number of degrees, e.g. 3 or -2.5".into(),
                },
                Some(PendingCmd::StripExif) if arg.eq_ignore_ascii_case("y") => {
                    self.strip_metadata_targets();
                    fs_changed = true;
                }
                Some(PendingCmd::RedactGps) if arg.eq_ignore_ascii_case("y") => {
                    self.redact_gps_targets();
                    fs_changed = true;
                }
                Some(PendingCmd::Panorama) if !arg.is_empty() => {
                    // `ha` / `va` = aligned (overlap registration); `h` / `v` = edge-to-edge; `g` = grid.
                    let a = arg.trim().to_ascii_lowercase();
                    let mode = match a.as_str() {
                        "ha" => stitch::PanoMode::HorizontalAligned,
                        "va" => stitch::PanoMode::VerticalAligned,
                        "hm" | "homography" => stitch::PanoMode::Homography,
                        _ => match a.chars().next() {
                            Some('v') => stitch::PanoMode::Vertical,
                            Some('g') => stitch::PanoMode::Grid,
                            _ => stitch::PanoMode::Horizontal,
                        },
                    };
                    self.stitch_panorama(mode);
                    fs_changed = true;
                }
                Some(PendingCmd::Multishot) if !arg.is_empty() => {
                    let focus = arg.trim().to_ascii_lowercase().starts_with('f');
                    self.make_multishot(focus);
                    fs_changed = true;
                }
                Some(PendingCmd::MoveToAlbum { copy }) if !arg.is_empty() => {
                    self.move_targets(&arg, copy);
                    fs_changed = true;
                }
                Some(PendingCmd::EmptyTrash) if arg.eq_ignore_ascii_case("y") => {
                    self.empty_trash();
                    fs_changed = true;
                }
                Some(PendingCmd::Watermark) if !arg.is_empty() => {
                    self.watermark_edit(&arg);
                    fs_changed = true;
                }
                Some(PendingCmd::Lut) if !arg.is_empty() => {
                    self.lut_edit(&arg);
                    fs_changed = true;
                }
                Some(PendingCmd::Convert) if !arg.is_empty() => match parse_convert(&arg) {
                    Some((fmt, size)) => {
                        self.convert_targets(&fmt, size);
                        fs_changed = true;
                    }
                    None => {
                        self.status = "enter e.g. 'jpg 2048', 'jpg 500kb', or 'png'".into()
                    }
                },
                Some(PendingCmd::Portfolio) if !arg.is_empty() => {
                    // `DIR [MAXPX] | watermark text` — `|` splits off the optional watermark.
                    let (left, mark) = match arg.split_once('|') {
                        Some((l, r)) => {
                            let t = r.trim();
                            (l.trim().to_string(), (!t.is_empty()).then(|| t.to_string()))
                        }
                        None => (arg.clone(), None),
                    };
                    let (dir, max_px) = match left.trim().rsplit_once(char::is_whitespace) {
                        Some((h, last)) if last.parse::<u32>().is_ok() => {
                            (h.trim().to_string(), last.parse::<u32>().ok())
                        }
                        _ => (left.trim().to_string(), None),
                    };
                    self.export_portfolio(&dir, max_px, mark.as_deref());
                }
                Some(PendingCmd::NlCommand) if !arg.is_empty() => {
                    // Deterministic fast-path first; otherwise hand off to the LLM planner.
                    match nl::parse_deterministic(&arg) {
                        Some(plan) => self.confirm_plan(plan),
                        None => {
                            self.pending_nl = Some(arg.clone());
                            self.status = "planning with the LLM…".into();
                        }
                    }
                }
                Some(PendingCmd::ConfirmPlan) if arg.eq_ignore_ascii_case("y") => {
                    if let Some(plan) = self.pending_plan.take() {
                        self.run_plan(plan);
                    }
                }
                Some(PendingCmd::DeleteSmart { name }) if arg.eq_ignore_ascii_case("y") => {
                    let mut fm = hjson::read_folder(&self.root_dir).unwrap_or_default();
                    fm.smart_albums.retain(|s| s.name != name);
                    hjson::write_folder(&self.root_dir, &fm)?;
                    self.smart_albums = fm.smart_albums;
                    if self.smart.as_deref() == Some(name.as_str()) {
                        self.smart = None; // the open view was just deleted
                    }
                    self.status = format!("deleted smart album '{name}'");
                }
                _ => {}
            }
            Ok(())
        })();
        if let Err(e) = result {
            self.status = format!("error: {e}");
        }
        self.cmd_active = false;
        self.cmd_buffer.clear();
        if fs_changed {
            self.rescan();
        } else if meta_changed {
            self.rebuild_view();
        }
    }

    fn enter_image_view(&mut self) {
        self.mode = AlbumMode::Image;
        self.zoom = 1.0;
        self.load_view();
    }

    /// Enter the survey/compare view of the current selection (2–4 images, decoded side-by-side).
    /// Falls back to the cursor image + its neighbours when nothing is selected.
    fn enter_compare(&mut self) {
        let mut idxs: Vec<usize> = if self.selected.is_empty() {
            // Cursor + up to the next 3 in the view.
            self.view.iter().skip(self.album_cursor).take(4).copied().collect()
        } else {
            let mut v: Vec<usize> = self.selected.iter().copied().collect();
            v.sort_unstable();
            v.truncate(4);
            v
        };
        idxs.dedup();
        let paths: Vec<PathBuf> =
            idxs.iter().filter_map(|&i| self.album_paths.get(i).cloned()).collect();
        if paths.len() < 2 {
            self.status = "select 2–4 images (Space) to compare".into();
            return;
        }
        self.compare = paths
            .into_iter()
            .filter_map(|p| {
                loader::thumbnail(&p, 1400).ok().map(|img| (p, self.picker.new_resize_protocol(img)))
            })
            .collect();
        self.compare_cursor = 0;
        self.mode = AlbumMode::Compare;
        self.status = "compare · ←/→ focus · 1–5/f/x rate the focused · Esc".into();
    }

    /// Apply a curation edit to the focused compare image (routes to its source album).
    fn curate_compare(&mut self, f: impl FnOnce(&mut hjson::ImageRecord)) {
        let Some(path) = self.compare.get(self.compare_cursor).map(|(p, _)| p.clone()) else { return };
        self.edit_record_at(&path, f);
    }

    /// Toggle stacking: collapse/expand derivative variants under their base image.
    fn toggle_stack(&mut self) {
        self.stack_view = !self.stack_view;
        let cur_path = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned());
        self.rebuild_view();
        if let Some(cp) = cur_path {
            if let Some(pos) = self.view.iter().position(|&pi| self.album_paths.get(pi) == Some(&cp)) {
                self.album_cursor = pos;
            }
        }
        self.status = if self.stack_view {
            "stacked: variants collapsed under their base (⧉N) · S to expand".into()
        } else {
            "stacks expanded".into()
        };
    }

    /// Open the timeline: date-sort the view, then bucket it by capture month so you can jump.
    fn enter_timeline(&mut self) {
        if self.view.is_empty() {
            self.status = "no images to place on a timeline".into();
            return;
        }
        // Date-sort so buckets are contiguous, then group by YYYY-MM.
        self.album_meta.sort = Some("date-desc".into());
        self.sort_album();
        self.rebuild_view();
        let mut buckets: Vec<(String, usize, usize)> = Vec::new();
        for (vpos, &pi) in self.view.iter().enumerate() {
            let label = self
                .album_paths
                .get(pi)
                .map(|p| date_bucket(p, self.record(p)))
                .unwrap_or_else(|| "undated".into());
            match buckets.last_mut() {
                Some((l, _, count)) if *l == label => *count += 1,
                _ => buckets.push((label, vpos, 1)),
            }
        }
        self.tl_buckets = buckets;
        self.tl_cursor = 0;
        self.timeline = true;
        self.status = "timeline · ↑/↓ pick a month · Enter jump · Esc".into();
    }

    /// Jump the grid cursor to the selected timeline bucket's first image, then close the timeline.
    fn jump_timeline(&mut self) {
        if let Some((_, vpos, _)) = self.tl_buckets.get(self.tl_cursor) {
            self.album_cursor = (*vpos).min(self.view.len().saturating_sub(1));
        }
        self.timeline = false;
    }

    /// Decode the cursor image (bounded to ~1600 px) into the full-pane view protocol + its EXIF,
    /// and (when the analysis panel is on) its histogram/exposure/focus stats. Honours the current
    /// zoom by centre-cropping the source before it's fit to the pane.
    /// A clone of the decoded working-resolution base for `(path, bound)` — cached so the interactive
    /// edit modes don't re-decode the file on every keypress.
    fn working_base(&mut self, path: &Path, bound: u32) -> Option<image::DynamicImage> {
        if let Some((p, b, img)) = &self.view_base {
            if p == path && *b == bound {
                return Some(img.clone());
            }
        }
        let img = loader::thumbnail(path, bound).ok()?;
        self.view_base = Some((path.to_path_buf(), bound, img.clone()));
        Some(img)
    }

    fn load_view(&mut self) {
        let path = self.cur_idx().and_then(|i| self.album_paths.get(i)).cloned();
        let zoom = self.zoom;
        // History scrubber: the view is a replay over the decoded pristine original, not the file.
        if self.history_mode {
            let prev = self.history_preview();
            self.view_spark = prev.as_ref().map(spark_of);
            self.view_proto = prev.map(|img| self.picker.new_resize_protocol(img));
            self.view_exif = None;
            self.analysis = None;
            return;
        }
        // Interactive edit modes re-render on every keypress, so use a smaller working resolution for
        // responsiveness (the final edit still applies at full resolution). Static viewing follows
        // the zoom, capped so a big image still fills the pane sharply.
        let interactive = self.adjust_mode
            || self.curve_mode
            || self.levels_mode
            || self.layer_mode
            || self.crop_mode
            || self.pick.is_some();
        let bound = if interactive { 1100 } else { (1600.0 * zoom).min(4096.0) as u32 };
        // Before/after: decode the pristine backup (if any) instead of the edited file.
        let decode = if self.show_original {
            self.cur_source()
                .map(|(d, f)| edit::backup_path(&d, &f))
                .filter(|b| b.exists())
                .or_else(|| path.clone())
        } else {
            path.clone()
        };
        // Reuse the cached decoded base when the path + bound match (no re-decode while dragging).
        let base = decode.as_ref().and_then(|p| self.working_base(p, bound));
        self.view_proto = base
            .map(|img| {
                let img = if self.curve_mode {
                    self.curve_op().apply(img) // live curve preview
                } else if self.adjust_mode {
                    self.adjust_op.clone().map(|op| op.apply(img.clone())).unwrap_or(img) // live slider preview
                } else if self.levels_mode {
                    self.levels_op().apply(img) // live levels preview
                } else if self.layer_mode {
                    layers::composite(&img, &self.layers, Some(self.layer_active)) // live composite
                } else if self.crop_mode {
                    crop_preview(&img, self.crop_rect) // full image, dimmed outside the crop rect
                } else if let Some(pk) = &self.pick {
                    pick_preview(&img, pk) // crosshair + confirmed points + brush circle
                } else if zoom > 1.01 {
                    apply_overlay(crop_center(&img, zoom), self.overlay)
                } else {
                    apply_overlay(img, self.overlay)
                };
                self.view_spark = Some(spark_of(&img)); // live histogram for the top bar
                self.picker.new_resize_protocol(img)
            });
        if path.is_none() {
            self.view_spark = None;
        }
        self.view_exif = path.as_ref().and_then(|p| exif::read_exif(p).ok());
        self.analysis = None;
        if self.show_analysis {
            self.compute_analysis();
        }
    }

    /// Zoom the image view by `factor` (>1 in, <1 out), clamped to 1.0–8.0, and re-decode.
    fn zoom_by(&mut self, factor: f32) {
        self.zoom = (self.zoom * factor).clamp(1.0, 8.0);
        self.load_view();
        self.status = if self.zoom > 1.01 {
            format!("zoom {:.1}× · z out · Z in", self.zoom)
        } else {
            "fit".into()
        };
    }

    /// Compute view-analysis stats for the cursor image (from a small decode, for speed).
    fn compute_analysis(&mut self) {
        self.analysis = self
            .cur_idx()
            .and_then(|i| self.album_paths.get(i))
            .and_then(|p| loader::thumbnail(p, 384).ok())
            .map(|img| analysis::analyze(&img));
    }

    /// Lazily decode a few thumbnails per tick for images near the cursor that aren't built yet.
    fn build_thumbs(&mut self, budget: usize) {
        let mut built = 0;
        // Prioritise the visible view from the cursor forward, then wrap.
        let n = self.view.len();
        for off in 0..n {
            if built >= budget {
                break;
            }
            let vi = (self.album_cursor + off) % n.max(1);
            let Some(path) = self.view.get(vi).and_then(|&pi| self.album_paths.get(pi)).cloned()
            else {
                break;
            };
            if self.thumbs.contains_key(&path) {
                continue;
            }
            match loader::get_or_render_thumb(&path, self.thumb_px).and_then(|c| {
                Ok(image::open(&c)?)
            }) {
                Ok(img) => {
                    self.thumbs.insert(path, self.picker.new_resize_protocol(img));
                    built += 1;
                }
                Err(_) => {
                    // Insert a 1×1 placeholder protocol so we don't retry a broken file every tick.
                    let ph = image::DynamicImage::new_rgb8(1, 1);
                    self.thumbs.insert(path, self.picker.new_resize_protocol(ph));
                }
            }
        }
    }
}

/// Entry point. Walks `root_dir`, then runs the TUI until `q`.
pub async fn run_with(root_dir: PathBuf, thumb_px: u32) -> Result<()> {
    anyhow::ensure!(root_dir.is_dir(), "photo root {} is not a directory", root_dir.display());
    let picker = Picker::from_query_stdio().map_err(|_| {
        anyhow::anyhow!(
            "no graphics-capable terminal detected. `plakat photos` needs a terminal with pixel \
             image support (Kitty, Ghostty, WezTerm, iTerm2, or Sixel)."
        )
    })?;
    let root = library::walk(&root_dir)?;
    let mut app = App::new(root, root_dir.clone(), picker, thumb_px);
    // Capture the async runtime handle so the resident ML worker can block_on pipeline loads.
    app.rt = Handle::try_current().ok();
    // Live watch (best-effort — the manager still works statically if it can't be started).
    app.watch = watcher::spawn(&root_dir).ok();

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    // If the OOM watchdog (armed by the ML worker) has to hard-exit, it skips Drop — so restore the
    // terminal here too, or the shell is left in raw mode / the alt screen. Best-effort; runs on the
    // watchdog thread just before the process exits.
    crate::memwatch::set_abort_hook(|| {
        let _ = disable_raw_mode();
        let _ = stdout().execute(LeaveAlternateScreen);
    });
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let res = event_loop(&mut terminal, &mut app);
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    res
}

/// Back-compat entry (default thumb size).
pub async fn run(root_dir: PathBuf) -> Result<()> {
    run_with(root_dir, 128).await
}

fn event_loop(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;
        // Poll with a timeout so thumbnails keep decoding even without input.
        if event::poll(Duration::from_millis(80))? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                if handle_key(app, k) {
                    break;
                }
            }
        }
        // An ML edit in flight on the resident worker: drain its progress inline (no TUI suspend) and
        // finalize on Done. No new heavy work starts until it finishes.
        if app.active_ml.is_some() {
            app.poll_active_ml();
        } else if let Some(job) = app.jobs.pop_front() {
            // Drain one heavy job per tick. ML edits go to the background worker (inline progress);
            // vision is a quick blocking call. A batch pipeline (`:`) enqueues many — they sequence.
            match job {
                Job::Ml(j) => app.dispatch_ml(j),
                Job::Vision(op, path) => run_vision_job(terminal, app, op, path)?,
            }
            continue;
        } else if let Some(query) = app.pending_visual.take() {
            // A queued CLIP visual search (Phase 7): heavy (model load + embed), run TUI-suspended.
            run_visual_search(terminal, app, query)?;
            continue;
        } else if let Some(qpath) = app.pending_clip_lookalike.take() {
            // A queued CLIP semantic lookalike (Phase 7): heavy (embeddings), run TUI-suspended.
            run_clip_lookalike(terminal, app, qpath)?;
            continue;
        } else if let Some(keep) = app.pending_cull.take() {
            // Aesthetic auto-cull (Track A): rank the album, keep top N.
            run_auto_cull(terminal, app, keep)?;
            continue;
        } else if let Some(path) = app.pending_analyze.take() {
            // Analyze-and-generate (Track A): describe the reference → img2img from the description.
            run_analyze_generate(terminal, app, path)?;
            continue;
        } else if std::mem::take(&mut app.pending_face_scan) {
            // Face-scan (Track A): detect + (when ArcFace is available) group faces across the library.
            run_face_scan(terminal, app)?;
            continue;
        } else if std::mem::take(&mut app.pending_face_polish) {
            // Face-polish: detect faces on the cursor image, then open the skin-smoothing slider.
            run_face_polish(terminal, app)?;
            continue;
        } else if let Some((op, dir)) = app.pending_create.take() {
            // AI create (generate / portrait / multiperson): prompt-driven generation → new image.
            run_create_job(terminal, app, op, dir)?;
            continue;
        } else if let Some(text) = app.pending_nl.take() {
            // A natural-language command the deterministic parser couldn't handle → LLM planner.
            run_nl_planner(terminal, app, text)?;
            continue;
        }
        app.refresh_mem();
        if app.focus == Focus::Album && !app.album_paths.is_empty() {
            app.build_thumbs(2);
        }
        // Filesystem watch: coalesce change signals, then rescan after a 500 ms quiet period.
        if let Some(w) = &app.watch {
            let mut changed = false;
            while w.rx.try_recv().is_ok() {
                changed = true;
            }
            if changed {
                app.dirty_since = Some(Instant::now());
            }
        }
        if let Some(t) = app.dirty_since {
            if t.elapsed() >= Duration::from_millis(500) {
                app.rescan();
                app.dirty_since = None;
            }
        }
    }
    Ok(())
}


/// Ask the configured LLM to turn `text` into a command plan, grounded with the album's HJSON, then
/// open the y/N confirmation. Runs off a dedicated-runtime thread (a quick network call), drawing a
/// "planning…" frame first — reuses the same provider pipeline as prompt enhancement.
fn run_nl_planner(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    text: String,
) -> Result<()> {
    app.status = "planning with the LLM…".into();
    terminal.draw(|f| draw(f, app))?;

    // Ground the planner with the album HJSON + a one-line context summary.
    let grounding = {
        let album = serde_json::to_string_pretty(&app.album_meta).unwrap_or_default();
        let name = app
            .album_dir
            .as_ref()
            .and_then(|d| d.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("(album)");
        format!(
            "album: {name}\nvisible: {} images\nselected: {}\n{album}",
            app.view.len(),
            app.selected.len(),
        )
    };
    let provider = crate::prompt::resolve_provider_label("auto");
    let result = std::thread::spawn(move || -> Result<nl::CommandPlan> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(nl::plan_llm(&provider, &text, &grounding))
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("planner thread panicked")));

    match result {
        Ok(plan) if plan.actions.is_empty() => {
            app.status = "couldn't turn that into a command — try rephrasing".into();
        }
        Ok(plan) => app.confirm_plan(plan),
        Err(e) => app.status = format!("planner failed: {e:#}"),
    }
    Ok(())
}

/// Aesthetic auto-cull (Track A), TUI-suspended: rank every image in the current view with the LAION
/// aesthetic scorer, then flag the top `keep` and reject the rest (metadata only, undoable). Scores
/// are also written to each image's `.json` sidecar.
fn run_auto_cull(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    keep: usize,
) -> Result<()> {
    use std::io::Write as _;
    let paths: Vec<PathBuf> = app.album_paths.clone();
    if paths.is_empty() {
        app.status = "no images to rank".into();
        return Ok(());
    }
    if crate::hw::mem_pressure() == crate::hw::Pressure::Critical {
        app.status = "⛔ memory low — free RAM before ranking".into();
        return Ok(());
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ aesthetic auto-cull: scoring {} images (loading CLIP + LAION predictor…)\n", paths.len());

    let result = std::thread::spawn(move || -> Result<Vec<(PathBuf, f32)>> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async {
            let device = crate::device::select("auto")?;
            let _guard = crate::memwatch::MemoryGuard::start(&device, "plakat photos cull");
            cull::rank(&device, paths, |done, tot| {
                if done % 10 == 0 || done == tot {
                    print!("\r  scoring {done}/{tot}…   ");
                    let _ = std::io::stdout().flush();
                }
            })
            .await
        })
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("auto-cull thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok(ranked) => {
            let total = ranked.len();
            let keep = keep.min(total);
            app.snapshot_meta();
            for (i, (p, score)) in ranked.iter().enumerate() {
                let is_keeper = i < keep;
                app.edit_record_at(p, |rec| {
                    rec.flagged = is_keeper;
                    rec.rejected = !is_keeper;
                });
                let _ = crate::imaging::io::patch_sidecar_score(p, *score as f64);
            }
            app.rebuild_view();
            app.status = format!(
                "✓ auto-cull: kept top {keep} of {total} (⚑ flagged); {} rejected — filter 'rejected' to review/delete · undo with u",
                total - keep
            );
        }
        Err(e) => app.status = format!("✗ auto-cull failed: {e:#}"),
    }
    Ok(())
}

/// Analyze-and-generate (Track A), TUI-suspended: describe the reference image with the configured
/// LLM vision provider, then queue an img2img from that description — a fresh "reimagined" variant.
fn run_analyze_generate(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    path: PathBuf,
) -> Result<()> {
    let provider = crate::prompt::vision::resolve_vision_provider("auto");
    app.status = format!("analyze & generate — describing with {provider} …");
    terminal.draw(|f| draw(f, app))?;

    const INSTR: &str = "Describe this image as a concise text-to-image prompt: the subject, style, \
        lighting, colour palette and mood, in one comma-separated line. No preamble.";
    let job_path = path.clone();
    let prov = provider.clone();
    let result = std::thread::spawn(move || -> Result<String> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(crate::prompt::vision::describe_image(&prov, &job_path, INSTR))
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("describe thread panicked")));

    match result {
        Ok(prompt) => {
            let prompt = prompt.trim().to_string();
            if prompt.is_empty() {
                app.status = "analyze & generate: the description came back empty".into();
                return Ok(());
            }
            // Record the derived prompt as the image's caption (useful even on its own), then queue
            // the img2img on the *known* reference path — the resident worker runs it inline.
            app.edit_record_at(&path, |rec| rec.caption = Some(prompt.clone()));
            app.status = format!("analyzed → “{prompt}” · generating…");
            if let Some(album) = path.parent() {
                app.jobs.push_back(Job::Ml(mledit::MlJob {
                    op: mledit::MlOp::Img2img { prompt },
                    input: path.clone(),
                    album: album.to_path_buf(),
                }));
            }
        }
        Err(e) => app.status = format!("✗ analyze & generate failed: {e:#}"),
    }
    Ok(())
}

/// Face-scan (Track A), TUI-suspended: detect faces across the library with SCRFD and, when ArcFace
/// weights are available, group them into `person-N` clusters — tagging each image so the existing
/// tag filter / smart albums surface people. Detection-only (face counts) when ArcFace isn't present.
fn run_face_scan(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, app: &mut App) -> Result<()> {
    use std::io::Write as _;
    let lib = app.collect_library();
    if lib.is_empty() {
        app.status = "no images to scan".into();
        return Ok(());
    }
    let items: Vec<PathBuf> = lib.iter().map(|(p, _, _)| p.clone()).collect();

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ face-scan: detecting faces in {} images (loading SCRFD…)\n", items.len());

    let result = std::thread::spawn(move || -> Result<faces::ScanResult> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async {
            let device = crate::device::select("auto")?;
            let _guard = crate::memwatch::MemoryGuard::start(&device, "plakat photos face-scan");
            faces::scan(&device, items, |done, tot| {
                if done % 5 == 0 || done == tot {
                    print!("\r  scanning {done}/{tot}…   ");
                    let _ = std::io::stdout().flush();
                }
            })
            .await
        })
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("face-scan thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok(scan) => {
            app.snapshot_meta();
            for (path, tags) in &scan.tags {
                app.edit_record_at(path, |rec| {
                    for t in tags {
                        if !rec.tags.contains(t) {
                            rec.tags.push(t.clone());
                        }
                    }
                });
            }
            app.rebuild_view();
            app.status = scan.summary();
        }
        Err(e) => app.status = format!("✗ face-scan failed: {e:#}"),
    }
    Ok(())
}

/// Face-polish, TUI-suspended: detect faces on the cursor image with SCRFD, then open the 0–100 %
/// skin-smoothing slider pre-loaded with a `FacePolish` op carrying the detected face ellipses (so
/// the AI cost is paid once — the slider preview + replay are pure geometry, no model).
fn run_face_polish(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, app: &mut App) -> Result<()> {
    let Some((album, filename)) = app.cur_source() else {
        app.status = "open an image first".into();
        return Ok(());
    };
    let path = album.join(&filename);
    if crate::hw::mem_pressure() == crate::hw::Pressure::Critical {
        app.status = "⛔ memory low — free RAM before face detection".into();
        return Ok(());
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ face polish: detecting faces (loading SCRFD…)\n");

    let job_path = path.clone();
    let result = std::thread::spawn(move || -> Result<Vec<[i32; 4]>> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async {
            let device = crate::device::select("auto")?;
            let _guard = crate::memwatch::MemoryGuard::start(&device, "plakat photos face-polish");
            faces::detect_ellipses(&device, &job_path, 6).await
        })
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("face-detect thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok(ellipses) if ellipses.is_empty() => {
            app.status = "face polish: no faces detected".into();
        }
        Ok(ellipses) => {
            let n = ellipses.len().min(6);
            let mut faces = [[0i32; 4]; 6];
            faces[..n].copy_from_slice(&ellipses[..n]);
            // Open the standard slider on the pre-built op (starts at full strength; dial down).
            app.enter_adjust(edit::EditOp::FacePolish { strength: 100, faces, n: n as i32 });
            app.status =
                format!("face polish · {n} face(s) · ←/→ fine · [/] jump · Enter apply · Esc cancel");
        }
        Err(e) => app.status = format!("✗ face polish failed: {e:#}"),
    }
    Ok(())
}

/// Run a queued AI "create" op (generate / portrait / multiperson) TUI-suspended, via the stable
/// `crate::api` builders, landing a new image in `dir`. Mirrors the other heavy-job runners (own
/// runtime thread, OOM guard, critical-pressure refuse); the source images are never modified.
fn run_create_job(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    op: CreateOp,
    dir: PathBuf,
) -> Result<()> {
    let label = match &op {
        CreateOp::Generate { .. } => "generate",
        CreateOp::Portrait { .. } => "portrait",
        CreateOp::Multiperson { .. } => "multiperson",
    };
    if crate::hw::mem_pressure() == crate::hw::Pressure::Critical {
        app.status = "⛔ memory low — free RAM before generating".into();
        return Ok(());
    }

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ AI {label}: loading model — this can take a while…\n");

    let result = std::thread::spawn(move || -> Result<crate::api::Image> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(async {
            let device = crate::api::device("auto")?;
            let _guard = crate::memwatch::MemoryGuard::start(&device, "plakat photos create");
            let imgs = match op {
                CreateOp::Generate { prompt, model, size } => {
                    let mut b = crate::api::Generate::new(model).prompt(prompt).device("auto");
                    if let Some((w, h)) = size {
                        b = b.size(w, h);
                    }
                    b.run().await?
                }
                CreateOp::Portrait { prompt, model, size, photo } => {
                    let id = identity_for_model(&model);
                    let mut b = crate::api::Portrait::new(model).prompt(prompt).device("auto");
                    if let Some((w, h)) = size {
                        b = b.size(w, h);
                    }
                    if let Some(p) = photo {
                        b = b.photo(p, 1.0).identity(id);
                    }
                    b.run().await?
                }
                CreateOp::Multiperson { prompt, model, size, people } => {
                    if people.is_empty() {
                        anyhow::bail!("select the people images first (Space to select in the grid)");
                    }
                    let id = identity_for_model(&model);
                    let mut b = crate::api::Multiperson::new(prompt).model(model).identity(id).device("auto");
                    if let Some((w, h)) = size {
                        b = b.size(w, h);
                    }
                    for (i, p) in people.into_iter().enumerate() {
                        b = b.person(crate::api::Person::new(format!("person{}", i + 1)).photo(p, 1.0));
                    }
                    b.run().await?
                }
            };
            imgs.into_iter().next().ok_or_else(|| anyhow::anyhow!("no image produced"))
        })
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("create thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok(img) => {
            let dest = variant_path(&dir, "ai", label, "png");
            if img.save(&dest).is_ok() {
                let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
                app.rescan();
                app.select_by_name(&name);
                app.status = format!("✓ AI {label} → {name}");
            } else {
                app.status = format!("AI {label}: couldn't save the image");
            }
        }
        Err(e) => app.status = format!("✗ AI {label} failed: {e:#}"),
    }
    Ok(())
}

/// Run a queued Gemini-vision request and merge the result into the image's record. Quick network
/// call (no alt-screen suspend) — draws a "querying…" frame, then blocks on a dedicated-runtime
/// thread (same no-`block_on`-on-event-loop reasoning as [`run_ml_job`]).
fn run_vision_job(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    op: vision::VisionOp,
    path: PathBuf,
) -> Result<()> {
    let provider = crate::prompt::vision::resolve_vision_provider("auto");
    app.status = format!("querying {provider} ({})…", op.label());
    terminal.draw(|f| draw(f, app))?; // show the status before we block

    let job_path = path.clone();
    let result = std::thread::spawn(move || -> Result<vision::VisionOutcome> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(vision::run(op, &job_path, &provider))
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("vision thread panicked")));

    match result {
        Ok(vision::VisionOutcome::Tags(tags)) => {
            let added = tags.len();
            app.edit_record_at(&path, |rec| {
                for t in tags {
                    if !rec.tags.contains(&t) {
                        rec.tags.push(t);
                    }
                }
            });
            app.rebuild_view(); // tags change filter matches
            app.status = format!("✓ autotag: +{added} tag(s)");
        }
        Ok(vision::VisionOutcome::Caption(caption)) => {
            app.edit_record_at(&path, |rec| rec.caption = Some(caption.clone()));
            app.status = format!("✓ caption: {caption}");
        }
        Err(e) => app.status = format!("✗ vision failed: {e:#}"),
    }
    Ok(())
}

/// Run a queued CLIP visual search with the TUI suspended (model load + per-image embed is heavy).
/// Reuses/updates the in-session embedding cache, then shows the top matches as a relevance view.
fn run_visual_search(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    query: String,
) -> Result<()> {
    use std::io::Write as _;
    let lib = app.collect_library();
    if lib.is_empty() {
        app.status = "no images to search".into();
        return Ok(());
    }
    let items: Vec<(PathBuf, PathBuf)> = lib.iter().map(|(p, d, _)| (p.clone(), d.clone())).collect();

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ CLIP visual search: \"{query}\"  (loading model + embedding {} images…)\n", items.len());

    // Seed from the persisted per-album cache (fast) before embedding; the in-session cache wins.
    let mut cache = std::mem::take(&mut app.clip_cache);
    let dirs: Vec<PathBuf> = {
        let mut d: Vec<PathBuf> = lib.iter().map(|(_, dir, _)| dir.clone()).collect();
        d.sort();
        d.dedup();
        d
    };
    for (k, v) in visual_search::load_cache(&dirs) {
        cache.entry(k).or_insert(v);
    }

    let q = query.clone();
    let result = std::thread::spawn(
        move || -> Result<(Vec<(PathBuf, PathBuf, f32)>, visual_search::Cache)> {
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            let out = rt.block_on(async {
                let device = crate::device::select("auto")?;
                visual_search::search(&device, items, &q, cache, |done, tot| {
                    if done % 10 == 0 || done == tot {
                        print!("\r  embedding {done}/{tot}…   ");
                        let _ = std::io::stdout().flush();
                    }
                })
                .await
            })?;
            // Persist embeddings to disk so the next session's search is fast.
            visual_search::save_cache(&out.1);
            Ok(out)
        },
    )
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("visual-search thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok((ranked, cache)) => {
            app.clip_cache = cache;
            let lookup: HashMap<PathBuf, Option<hjson::ImageRecord>> =
                lib.into_iter().map(|(p, _, r)| (p, r)).collect();
            let ordered: Vec<(PathBuf, PathBuf, Option<hjson::ImageRecord>)> = ranked
                .into_iter()
                .take(200)
                .map(|(p, d, _)| {
                    let r = lookup.get(&p).cloned().flatten();
                    (p, d, r)
                })
                .collect();
            let count = ordered.len();
            app.enter_smart_view(format!("visual: {query}"), query.clone(), true, ordered, true);
            app.status = format!("🔍 visual '{query}' · top {count} by CLIP similarity");
        }
        Err(e) => app.status = format!("✗ visual search failed: {e:#}"),
    }
    Ok(())
}

/// Run a queued CLIP semantic lookalike (image→image) with the TUI suspended. Reuses the persistent
/// embedding cache with a lazy model load, so it runs offline when everything's already embedded.
fn run_clip_lookalike(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    query_path: PathBuf,
) -> Result<()> {
    use std::io::Write as _;
    let lib = app.collect_library();
    if lib.is_empty() {
        app.status = "no images to search".into();
        return Ok(());
    }
    let items: Vec<(PathBuf, PathBuf)> = lib.iter().map(|(p, d, _)| (p.clone(), d.clone())).collect();
    let qname = query_path.file_name().and_then(|n| n.to_str()).unwrap_or("image").to_string();

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ CLIP lookalike: images like \"{qname}\"  (embedding {} images; cached ones are instant)…\n", items.len());

    let mut cache = std::mem::take(&mut app.clip_cache);
    let dirs: Vec<PathBuf> = {
        let mut d: Vec<PathBuf> = lib.iter().map(|(_, dir, _)| dir.clone()).collect();
        d.sort();
        d.dedup();
        d
    };
    for (k, v) in visual_search::load_cache(&dirs) {
        cache.entry(k).or_insert(v);
    }

    let qp = query_path.clone();
    let result = std::thread::spawn(
        move || -> Result<(Vec<(PathBuf, PathBuf, f32)>, visual_search::Cache)> {
            let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
            let out = rt.block_on(async {
                let device = crate::device::select("auto")?;
                visual_search::search_by_image(&device, items, &qp, cache, |done, tot| {
                    if done % 10 == 0 || done == tot {
                        print!("\r  embedding {done}/{tot}…   ");
                        let _ = std::io::stdout().flush();
                    }
                })
                .await
            })?;
            visual_search::save_cache(&out.1);
            Ok(out)
        },
    )
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("lookalike thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok((ranked, cache)) => {
            app.clip_cache = cache;
            let lookup: HashMap<PathBuf, Option<hjson::ImageRecord>> =
                lib.into_iter().map(|(p, _, r)| (p, r)).collect();
            let ordered: Vec<(PathBuf, PathBuf, Option<hjson::ImageRecord>)> = ranked
                .into_iter()
                .take(120)
                .map(|(p, d, _)| (p.clone(), d, lookup.get(&p).cloned().flatten()))
                .collect();
            let count = ordered.len();
            app.enter_smart_view(format!("clip-similar: {qname}"), String::new(), true, ordered, true);
            app.lookalike_of = Some(query_path); // snapshot; rescan won't re-embed
            app.lookalike_clip = true;
            app.status = format!("🔍 {count} most-similar to {qname} (CLIP)");
        }
        Err(e) => app.status = format!("✗ lookalike failed: {e:#}"),
    }
    Ok(())
}

/// Returns true to quit.
fn handle_key(app: &mut App, k: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyModifiers;
    // While an ML edit runs on the worker, Esc requests cancellation and `q` cancels-then-quits;
    // every other key is ignored so a stray keypress can't disturb the in-flight job. (img2img stops
    // at the next denoise step; relight/upscale have no mid-run hook, so quit may wait for them.)
    if app.active_ml.is_some() {
        match k.code {
            KeyCode::Esc => {
                app.cancel_active_ml();
                return false;
            }
            KeyCode::Char('q') => {
                app.cancel_active_ml();
                return true;
            }
            _ => return false,
        }
    }
    // A quickhelp overlay swallows the next key (dismiss).
    if app.help.is_some() {
        app.help = None;
        return false;
    }
    // Second stage of an edit chord: `Ctrl-B <category>` is set; this key is the item.
    if let Some(cat) = app.chord_prefix.take() {
        match k.code {
            KeyCode::Esc => app.status = "chord cancelled".into(),
            KeyCode::Char(item) => {
                let chord = format!("{cat}{item}");
                match edit_cmd_for_chord(&chord) {
                    Some(cmd) => app.run_edit_cmd(cmd),
                    None => app.status = format!("no edit chord Ctrl-B {chord}"),
                }
            }
            _ => {}
        }
        return false;
    }
    // `Ctrl-B` leader: the next key is a leader command (h = chords help, H = commands help). In the
    // image view, a category key (g/c/a/k/x/e/m) begins a two-key edit chord.
    if app.leader {
        app.leader = false;
        let img_view = app.focus == Focus::Album && app.mode == AlbumMode::Image;
        if img_view {
            if let KeyCode::Char(c) = k.code {
                if chord_categories().iter().any(|(cat, _)| *cat == c) {
                    app.chord_prefix = Some(c);
                    let items: Vec<String> = edit_commands()
                        .iter()
                        .filter(|(_, ch, _)| ch.starts_with(c))
                        .map(|(_, ch, _)| ch[1..].to_string())
                        .collect();
                    let name = chord_categories().iter().find(|(cat, _)| *cat == c).map(|(_, n)| *n).unwrap_or("");
                    app.status = format!("⌃B {c} · {name}: {} · Esc", items.join(" "));
                    return false;
                }
            }
        }
        match k.code {
            KeyCode::Char('h') => app.help = Some(HelpKind::Chords),
            KeyCode::Char('H') => app.help = Some(HelpKind::Commands),
            KeyCode::Char('t') => app.open_tag_browser(),
            KeyCode::Char('l') => {
                if let Some(p) = app.cur_idx().and_then(|i| app.album_paths.get(i).cloned()) {
                    app.open_lookalike(p); // perceptual (offline, fast)
                }
            }
            KeyCode::Char('L') => {
                if let Some(p) = app.cur_idx().and_then(|i| app.album_paths.get(i).cloned()) {
                    app.pending_clip_lookalike = Some(p); // CLIP semantic (event loop runs it)
                    app.status = "CLIP lookalike … (the UI will pause)".into();
                }
            }
            KeyCode::Char('v') => app.open_version_browser(),
            KeyCode::Char('p') => {
                app.prompt("portfolio to (DIR [MAXPX] | watermark): ", "", PendingCmd::Portfolio)
            }
            _ => {}
        }
        return false;
    }
    if app.tag_browser {
        handle_tag_key(app, k.code);
        return false;
    }
    if app.version_browser {
        handle_version_key(app, k.code);
        return false;
    }
    if app.preset_browser {
        handle_preset_key(app, k.code);
        return false;
    }
    if k.modifiers.contains(KeyModifiers::CONTROL) && matches!(k.code, KeyCode::Char('b')) {
        app.leader = true;
        let edits = if app.focus == Focus::Album && app.mode == AlbumMode::Image {
            " · edits g/c/a/k/x/e/m"
        } else {
            ""
        };
        app.status =
            format!("Ctrl-B · h/H help · t tags · v versions · p portfolio · l/L lookalike{edits}");
        return false;
    }
    if app.cmd_active {
        handle_cmd_key(app, k.code);
        return false;
    }
    if app.filter_active {
        handle_filter_key(app, k.code);
        return false;
    }
    if app.album_info.is_some() {
        app.album_info = None; // any key dismisses the info panel
        return false;
    }
    if app.info_editor {
        handle_info_editor_key(app, k.code);
        return false;
    }
    if app.tree_filter_active {
        handle_tree_filter_key(app, k.code);
        return false;
    }
    if app.curve_mode {
        handle_curve_key(app, k.code);
        return false;
    }
    if app.history_mode {
        handle_history_key(app, k.code);
        return false;
    }
    if app.adjust_mode {
        handle_adjust_key(app, k.code);
        return false;
    }
    if app.levels_mode {
        handle_levels_key(app, k.code);
        return false;
    }
    if app.layer_mode {
        handle_layer_key(app, k.code);
        return false;
    }
    if app.crop_mode {
        handle_crop_key(app, k.code);
        return false;
    }
    if app.pick.is_some() {
        handle_pick_key(app, k.code);
        return false;
    }
    if app.edit_menu {
        handle_edit_key(app, k.code);
        return false;
    }
    if app.ml_menu {
        handle_ml_key(app, k.code);
        return false;
    }
    if app.ai_menu {
        handle_ai_key(app, k.code);
        return false;
    }
    if app.timeline {
        handle_timeline_key(app, k.code);
        return false;
    }
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match app.focus {
        Focus::Tree => handle_tree_key(app, k.code),
        Focus::Album if app.mode == AlbumMode::Image => handle_image_key(app, k.code),
        Focus::Album if app.mode == AlbumMode::Cull => handle_cull_key(app, k.code),
        Focus::Album if app.mode == AlbumMode::Compare => handle_compare_key(app, k.code),
        Focus::Album => handle_grid_key(app, k.code, ctrl),
    }
}

/// T1 pixel-edit menu (Phase 3): a modal key layer over the cursor image. Each op applies and keeps
/// the menu open (chain edits); `u` undoes, `0` reverts, Esc/`E` closes.
/// Interactive free-form crop: adjust the crop rectangle over the dimmed preview; Enter applies.
fn handle_crop_key(app: &mut App, code: KeyCode) {
    let s = 0.02;
    match code {
        KeyCode::Esc => app.cancel_crop(),
        KeyCode::Enter => app.apply_crop(),
        KeyCode::Left => app.adjust_crop(-s, 0.0, 0.0, 0.0),
        KeyCode::Right => app.adjust_crop(s, 0.0, 0.0, 0.0),
        KeyCode::Up => app.adjust_crop(0.0, -s, 0.0, 0.0),
        KeyCode::Down => app.adjust_crop(0.0, s, 0.0, 0.0),
        KeyCode::Char('+') | KeyCode::Char('=') => app.grow_crop(0.04),
        KeyCode::Char('-') | KeyCode::Char('_') => app.grow_crop(-0.04),
        KeyCode::Char('[') => app.adjust_crop(0.0, 0.0, -s, 0.0),
        KeyCode::Char(']') => app.adjust_crop(0.0, 0.0, s, 0.0),
        KeyCode::Char(',') => app.adjust_crop(0.0, 0.0, 0.0, -s),
        KeyCode::Char('.') => app.adjust_crop(0.0, 0.0, 0.0, s),
        _ => {}
    }
}

/// Retouch pick-mode: arrows move the crosshair (fine), h/j/k/l coarse, +/- brush size, Enter sets
/// the next point (applies once enough are collected), Esc cancels.
fn handle_pick_key(app: &mut App, code: KeyCode) {
    let (fine, coarse) = (0.004, 0.03);
    match code {
        KeyCode::Esc => app.cancel_pick(),
        KeyCode::Enter => app.pick_confirm(),
        KeyCode::Left => app.pick_move(-fine, 0.0),
        KeyCode::Right => app.pick_move(fine, 0.0),
        KeyCode::Up => app.pick_move(0.0, -fine),
        KeyCode::Down => app.pick_move(0.0, fine),
        KeyCode::Char('h') => app.pick_move(-coarse, 0.0),
        KeyCode::Char('l') => app.pick_move(coarse, 0.0),
        KeyCode::Char('k') => app.pick_move(0.0, -coarse),
        KeyCode::Char('j') => app.pick_move(0.0, coarse),
        KeyCode::Char('+') | KeyCode::Char('=') => app.pick_radius(0.01),
        KeyCode::Char('-') | KeyCode::Char('_') => app.pick_radius(-0.01),
        _ => {}
    }
}

/// Interactive scalar-adjustment slider: ←/→ change the value by the op's step, `[`/`]` fine ±1,
/// live preview, Enter commits, Esc cancels.
fn handle_adjust_key(app: &mut App, code: KeyCode) {
    let step = app.adjust_op.as_ref().map(|o| o.scalar_range().2).unwrap_or(5);
    match code {
        KeyCode::Esc => app.cancel_adjust(),
        KeyCode::Enter => app.apply_adjust(),
        // Arrows = fine ±1 (the intuitive default); brackets / -+ / PageKeys = the coarse jump.
        KeyCode::Left => app.nudge_adjust(-1),
        KeyCode::Right => app.nudge_adjust(1),
        KeyCode::Char('[') | KeyCode::Char('-') | KeyCode::Char('_') | KeyCode::PageDown => {
            app.nudge_adjust(-step)
        }
        KeyCode::Char(']') | KeyCode::Char('+') | KeyCode::Char('=') | KeyCode::PageUp => {
            app.nudge_adjust(step)
        }
        _ => {}
    }
}

/// Interactive tone-curve editor: ←/→ pick one of the 5 points, ↑/↓ move its output (live preview),
/// `[`/`]` fine, Enter applies, Esc cancels.
fn handle_curve_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.cancel_curve(),
        KeyCode::Enter => app.apply_curve(),
        KeyCode::Left => app.select_curve(-1),
        KeyCode::Right => app.select_curve(1),
        KeyCode::Up => app.adjust_curve(4),
        KeyCode::Down => app.adjust_curve(-4),
        KeyCode::Char('[') => app.adjust_curve(-1),
        KeyCode::Char(']') => app.adjust_curve(1),
        _ => {}
    }
}

/// Edit-history scrubber: ←/→ step through the stack, `d` deletes the shown edit, Enter applies the
/// (possibly trimmed) stack, Esc cancels.
fn handle_history_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => app.cancel_history(),
        KeyCode::Enter => app.apply_history(),
        KeyCode::Left | KeyCode::Down => app.move_history(-1),
        KeyCode::Right | KeyCode::Up => app.move_history(1),
        KeyCode::Char('d') | KeyCode::Char('x') => app.delete_history_op(),
        _ => {}
    }
}

/// Interactive levels editor: ↑/↓ pick a handle (black/white/gamma), ←/→ adjust with a live preview,
/// Enter commits the edit, Esc cancels. `[`/`]` and `,`/`.` mirror ←/→ for the current handle.
fn handle_levels_key(app: &mut App, code: KeyCode) {
    let step = if app.lv_sel == 2 { 5 } else { 3 }; // gamma in hundredths, points in 0..255
    match code {
        KeyCode::Esc => app.cancel_levels(),
        KeyCode::Enter => app.apply_levels(),
        KeyCode::Up => app.select_levels(-1),
        KeyCode::Down => app.select_levels(1),
        KeyCode::Left | KeyCode::Char('[') | KeyCode::Char(',') => app.adjust_levels(-step),
        KeyCode::Right | KeyCode::Char(']') | KeyCode::Char('.') => app.adjust_levels(step),
        _ => {}
    }
}

/// Interactive layer compositing (Phase 8): arrows move the active layer over a live composited
/// preview; the stack persists on the record; Enter flattens to a new `_layered.png` variant.
fn handle_layer_key(app: &mut App, code: KeyCode) {
    let m = 0.02;
    // Mask-adjust sub-mode: arrows reposition the active layer's mask; Esc/Enter/M leave it. Other
    // keys fall through to the normal handlers below (size/feather/blend/… behave the same).
    if app.mask_adjust {
        match code {
            KeyCode::Esc | KeyCode::Enter | KeyCode::Char('M') => {
                app.exit_mask_adjust();
                return;
            }
            KeyCode::Left => {
                app.move_mask(-m, 0.0);
                return;
            }
            KeyCode::Right => {
                app.move_mask(m, 0.0);
                return;
            }
            KeyCode::Up => {
                app.move_mask(0.0, -m);
                return;
            }
            KeyCode::Down => {
                app.move_mask(0.0, m);
                return;
            }
            _ => {}
        }
    }
    match code {
        KeyCode::Esc => app.cancel_layers(),
        KeyCode::Enter => app.flatten_layers(),
        KeyCode::Char('a') => app.layer_add_prompt(),
        KeyCode::Char('M') => app.enter_mask_adjust(),
        KeyCode::Left => app.nudge_layer(-m, 0.0),
        KeyCode::Right => app.nudge_layer(m, 0.0),
        KeyCode::Up => app.nudge_layer(0.0, -m),
        KeyCode::Down => app.nudge_layer(0.0, m),
        KeyCode::Char('+') | KeyCode::Char('=') => app.scale_layer(0.05),
        KeyCode::Char('-') | KeyCode::Char('_') => app.scale_layer(-0.05),
        KeyCode::Char('<') => app.opacity_layer(-0.05),
        KeyCode::Char('>') => app.opacity_layer(0.05),
        KeyCode::Char('b') => app.cycle_layer_blend(),
        KeyCode::Char('{') => app.reorder_layer(false),
        KeyCode::Char('}') => app.reorder_layer(true),
        KeyCode::Char('n') | KeyCode::Tab => app.select_layer(1),
        KeyCode::Char('p') | KeyCode::BackTab => app.select_layer(-1),
        KeyCode::Char('x') => app.delete_layer(),
        // Mask (active layer): m cycle shape · k image matte · [ ] size · , . feather · / invert.
        KeyCode::Char('m') => app.cycle_layer_mask(),
        KeyCode::Char('k') => app.layer_mask_prompt(),
        KeyCode::Char('[') => app.resize_mask(-0.05),
        KeyCode::Char(']') => app.resize_mask(0.05),
        KeyCode::Char(',') => app.feather_mask(-0.03),
        KeyCode::Char('.') => app.feather_mask(0.03),
        KeyCode::Char('/') => app.invert_mask(),
        _ => {}
    }
}

/// Edit command palette (`E`): a searchable, scrollable list. Type to filter; ↑/↓ · PgUp/PgDn ·
/// Home/End to navigate; Enter runs the selected command (palette stays open to chain edits); Esc
/// closes. Keys still fall through to running commands — but selection is by search + Enter now.
fn handle_edit_key(app: &mut App, code: KeyCode) {
    let n = filtered_edit_commands(&app.edit_query).len();
    let last = n.saturating_sub(1);
    let page = app.edit_visible.max(1);
    match code {
        KeyCode::Esc => app.edit_menu = false,
        KeyCode::Up => app.edit_cursor = app.edit_cursor.saturating_sub(1),
        KeyCode::Down => app.edit_cursor = (app.edit_cursor + 1).min(last),
        KeyCode::PageUp => app.edit_cursor = app.edit_cursor.saturating_sub(page),
        KeyCode::PageDown => app.edit_cursor = (app.edit_cursor + page).min(last),
        KeyCode::Home => app.edit_cursor = 0,
        KeyCode::End => app.edit_cursor = last,
        KeyCode::Enter => {
            if let Some((_, _, cmd)) = filtered_edit_commands(&app.edit_query).get(app.edit_cursor).cloned() {
                app.run_edit_cmd(cmd);
            }
        }
        KeyCode::Backspace => {
            app.edit_query.pop();
            app.edit_cursor = 0;
        }
        KeyCode::Char(c) => {
            app.edit_query.push(c);
            app.edit_cursor = 0;
        }
        _ => {}
    }
}

/// Vision + AI menu (Phase 7): describe/tag the cursor image with Gemini vision.
fn handle_ai_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('A') | KeyCode::Char('q') => app.ai_menu = false,
        KeyCode::Char('t') => app.queue_vision(vision::VisionOp::Autotag),
        KeyCode::Char('d') => app.queue_vision(vision::VisionOp::Describe),
        KeyCode::Char('g') => app.ai_tag_from_recipe(),
        KeyCode::Char('r') => {
            app.ai_menu = false;
            app.prompt("aesthetic auto-cull — keep top N: ", "10", PendingCmd::AutoCull);
        }
        KeyCode::Char('n') => {
            app.ai_menu = false;
            app.analyze_and_generate();
        }
        KeyCode::Char('f') => {
            app.ai_menu = false;
            app.request_face_scan();
        }
        _ => {}
    }
}

/// Tag browser (Ctrl-B t): pick a tag to filter by.
fn handle_tag_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.tag_browser = false,
        KeyCode::Up | KeyCode::Char('k') => app.tag_cursor = app.tag_cursor.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            app.tag_cursor = (app.tag_cursor + 1).min(app.tags_list.len().saturating_sub(1));
        }
        KeyCode::Enter | KeyCode::Char('l') => app.pick_tag(),
        _ => {}
    }
}

/// Preset picker (Edit → apply preset): pick a saved edit stack and apply it to the targets.
fn handle_preset_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.preset_browser = false,
        KeyCode::Up | KeyCode::Char('k') => app.preset_cursor = app.preset_cursor.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            app.preset_cursor = (app.preset_cursor + 1).min(app.presets_list.len().saturating_sub(1));
        }
        KeyCode::Enter | KeyCode::Char('l') => {
            if let Some(p) = app.presets_list.get(app.preset_cursor) {
                let ops = p.ops.clone();
                app.preset_browser = false;
                app.apply_edit_ops_to_targets(&ops, "applied preset:");
            }
        }
        _ => {}
    }
}

/// Version browser (Ctrl-B v): row 0 saves the current image; rows below restore a snapshot.
fn handle_version_key(app: &mut App, code: KeyCode) {
    let rows = app.versions_list.len() + 1; // +1 for the "save current" row
    match code {
        KeyCode::Esc | KeyCode::Char('q') => app.version_browser = false,
        KeyCode::Up | KeyCode::Char('k') => app.version_cursor = app.version_cursor.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            app.version_cursor = (app.version_cursor + 1).min(rows.saturating_sub(1));
        }
        KeyCode::Enter | KeyCode::Char('l') => app.version_action(),
        _ => {}
    }
}

/// Timeline modal (Phase 5): pick a capture-month bucket and jump the grid to it.
fn handle_timeline_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('@') => app.timeline = false,
        KeyCode::Up | KeyCode::Char('k') => app.tl_cursor = app.tl_cursor.saturating_sub(1),
        KeyCode::Down | KeyCode::Char('j') => {
            app.tl_cursor = (app.tl_cursor + 1).min(app.tl_buckets.len().saturating_sub(1));
        }
        KeyCode::Enter | KeyCode::Char('l') => app.jump_timeline(),
        _ => {}
    }
}

/// Survey / compare mode (Phase 5): side-by-side images; move the focus and rate/flag/reject the
/// focused one without leaving the comparison.
fn handle_compare_key(app: &mut App, code: KeyCode) -> bool {
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc => {
            app.mode = AlbumMode::Grid;
            app.compare.clear();
        }
        KeyCode::Left | KeyCode::Char('h') => {
            app.compare_cursor = app.compare_cursor.saturating_sub(1);
        }
        KeyCode::Right | KeyCode::Char('l') => {
            if app.compare_cursor + 1 < app.compare.len() {
                app.compare_cursor += 1;
            }
        }
        KeyCode::Char(d @ '0'..='5') => {
            let r = d.to_digit(10).unwrap() as u8;
            app.curate_compare(|rec| rec.rating = r);
        }
        KeyCode::Char('f') => app.curate_compare(|rec| rec.flagged = !rec.flagged),
        KeyCode::Char('x') => app.curate_compare(|rec| rec.rejected = !rec.rejected),
        _ => {}
    }
    false
}

/// T2 ML-edit menu (Phase 4): pick an operation to run on the cursor image via an existing pipeline.
/// Prompt-driven ops open the command pane; upscale queues immediately. The event loop runs the job
/// with the TUI suspended.
fn handle_ml_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('M') | KeyCode::Char('q') => app.ml_menu = false,
        KeyCode::Char('u') => app.queue_ml(mledit::MlOp::Upscale),
        KeyCode::Char('i') => {
            app.ml_menu = false;
            app.prompt("img2img prompt: ", "", PendingCmd::MlPrompt { relight: false });
        }
        KeyCode::Char('l') => {
            app.ml_menu = false;
            app.prompt("relight prompt: ", "", PendingCmd::MlPrompt { relight: true });
        }
        KeyCode::Char('g') => {
            app.ml_menu = false;
            app.prompt("AI generate — prompt (| model [WxH], e.g. | sd15 512): ", "", PendingCmd::Generate);
        }
        KeyCode::Char('p') => {
            app.ml_menu = false;
            app.prompt("AI portrait — prompt (| model [WxH]): ", "", PendingCmd::Portrait);
        }
        KeyCode::Char('m') => {
            app.ml_menu = false;
            app.prompt("AI multiperson — scene (| model [WxH]): ", "", PendingCmd::Multiperson);
        }
        _ => {}
    }
}

/// Filter-bar input: type a filter expression, Enter applies, Esc clears.
/// Incremental tree-name filter: type to narrow, Enter keeps it applied, Esc clears it.
fn handle_tree_filter_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.tree_filter.clear();
            app.tree_filter_active = false;
        }
        KeyCode::Enter => app.tree_filter_active = false,
        KeyCode::Backspace => {
            app.tree_filter.pop();
        }
        KeyCode::Char(c) => app.tree_filter.push(c),
        _ => {}
    }
    app.tree_cursor = app.tree_cursor.min(app.rows().len().saturating_sub(1));
}

/// Album info editor menu (`I`): a key picks a field to edit via the command pane.
fn handle_info_editor_key(app: &mut App, code: KeyCode) {
    let Some(path) = app.info_target.clone() else {
        app.info_editor = false;
        return;
    };
    let field = match code {
        KeyCode::Esc => {
            app.info_editor = false;
            return;
        }
        KeyCode::Char('n') => AlbumFieldKind::Name,
        KeyCode::Char('d') => AlbumFieldKind::Description,
        KeyCode::Char('t') => AlbumFieldKind::TagsReplace,
        KeyCode::Char('a') => AlbumFieldKind::TagsAdd,
        KeyCode::Char('c') => AlbumFieldKind::Cover,
        KeyCode::Char('s') => AlbumFieldKind::Sort,
        _ => return,
    };
    app.info_editor = false;
    app.prompt_album_field(path, field);
}

fn handle_filter_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.filter_active = false;
            app.filter.clear();
            app.rebuild_view();
        }
        KeyCode::Enter => {
            app.filter_active = false;
            app.rebuild_view();
        }
        KeyCode::Backspace => {
            app.filter.pop();
            app.rebuild_view();
        }
        KeyCode::Char(c) => {
            app.filter.push(c);
            app.rebuild_view();
        }
        _ => {}
    }
}

/// Command-pane text input (RFC §11): type the argument, Enter commits, Esc cancels.
fn handle_cmd_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc => {
            app.cmd_active = false;
            app.pending = None;
            app.cmd_buffer.clear();
        }
        KeyCode::Enter => app.commit_cmd(),
        KeyCode::Backspace => {
            app.cmd_buffer.pop();
        }
        KeyCode::Char(c) => app.cmd_buffer.push(c),
        _ => {}
    }
}

fn handle_tree_key(app: &mut App, code: KeyCode) -> bool {
    let rows = app.rows();
    let cur = rows.get(app.tree_cursor).map(|r| (r.path.clone(), r.kind, r.has_children, r.name.clone()));
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('j') | KeyCode::Down => {
            app.tree_cursor = (app.tree_cursor + 1).min(rows.len().saturating_sub(1));
        }
        KeyCode::Char('k') | KeyCode::Up => app.tree_cursor = app.tree_cursor.saturating_sub(1),
        KeyCode::PageDown => {
            app.tree_cursor = (app.tree_cursor + 10).min(rows.len().saturating_sub(1));
        }
        KeyCode::PageUp => app.tree_cursor = app.tree_cursor.saturating_sub(10),
        KeyCode::Char('g') | KeyCode::Home => app.tree_cursor = 0,
        KeyCode::Char('G') | KeyCode::End => app.tree_cursor = rows.len().saturating_sub(1),
        // → / `l` reveal children first: a folder OR a *nested* album (images + sub-dirs) expands;
        // only a leaf album (no children) opens the grid here.
        KeyCode::Char('l') | KeyCode::Right => {
            if let Some((path, kind, has_children, name)) = cur.clone() {
                if kind == NodeKind::SmartAlbum {
                    if let Some(q) = app.smart_albums.iter().find(|s| s.name == name).map(|s| s.query.clone()) {
                        app.open_smart(name, q);
                    }
                } else if has_children {
                    app.expanded.insert(path);
                } else if kind == NodeKind::Album {
                    app.open_album(path);
                }
            }
        }
        // Enter always *opens* an album's images (a nested album's own pictures), else expands.
        KeyCode::Enter => {
            if let Some((path, kind, has_children, name)) = cur.clone() {
                if kind == NodeKind::SmartAlbum {
                    if let Some(q) = app.smart_albums.iter().find(|s| s.name == name).map(|s| s.query.clone()) {
                        app.open_smart(name, q);
                    }
                } else if kind == NodeKind::Album {
                    app.open_album(path);
                } else if has_children {
                    app.expanded.insert(path);
                }
            }
        }
        KeyCode::Char('h') | KeyCode::Left => app.goto_tree_parent(),
        // Incremental tree-name filter.
        KeyCode::Char('/') => {
            app.tree_filter_active = true;
            app.status = "filter tree: type to match names · Enter keep · Esc clear".into();
        }
        // Flatten browse: every image beneath this folder/album in one grid.
        KeyCode::Char('*') => {
            if let Some((path, kind, ..)) = cur.clone() {
                if kind != NodeKind::SmartAlbum {
                    app.open_recursive(path);
                }
            }
        }
        // Album operations on the cursor node.
        KeyCode::Char('t') => {
            if let Some((path, kind, ..)) = cur.clone() {
                if kind != NodeKind::SmartAlbum {
                    app.prompt_album_field(path, AlbumFieldKind::TagsAdd);
                }
            }
        }
        KeyCode::Char('T') => {
            if let Some((path, kind, ..)) = cur.clone() {
                if kind != NodeKind::SmartAlbum {
                    app.prompt_album_field(path, AlbumFieldKind::TagsReplace);
                }
            }
        }
        KeyCode::Char('i') => app.open_album_info(),
        KeyCode::Char('I') => app.open_info_editor(),
        KeyCode::Char('r') => app.regen_thumbs_tree(),
        KeyCode::Char('e') => {
            if let Some((path, kind, _, name)) = cur.clone() {
                if kind == NodeKind::SmartAlbum {
                    // Materialize the saved search into a portable album of copies.
                    if let Some(q) = app.smart_albums.iter().find(|s| s.name == name).map(|s| s.query.clone()) {
                        app.prompt(
                            "materialize smart album to (DIR): ",
                            "",
                            PendingCmd::MaterializeSmart { name, query: q },
                        );
                    }
                } else {
                    let recursive = kind == NodeKind::Folder;
                    app.prompt("export to (DIR [maxpx]): ", "", PendingCmd::ExportAlbum { path, recursive });
                }
            }
        }
        KeyCode::Char('E') => {
            if let Some((path, kind, ..)) = cur.clone() {
                if kind != NodeKind::SmartAlbum {
                    let recursive = kind == NodeKind::Folder;
                    app.prompt(
                        "export + convert (FMT DIR [maxpx]): ",
                        "",
                        PendingCmd::ExportConvertAlbum { path, recursive },
                    );
                }
            }
        }
        // Library-wide metadata / visual search from anywhere.
        KeyCode::Char('?') => app.prompt("search metadata: ", "", PendingCmd::Search),
        KeyCode::Char('V') => app.prompt("visual search: ", "", PendingCmd::VisualSearch),
        KeyCode::Char(':') => app.prompt("command: ", "", PendingCmd::NlCommand),
        KeyCode::Tab => {
            if app.album_dir.is_some() || app.smart.is_some() {
                app.focus = Focus::Album;
            }
        }
        // Mutations (RFC §7.4) → command pane. Smart-album (★) rows are virtual — only Delete
        // applies; new-folder/new-album/rename are skipped for them.
        KeyCode::Char('n') => {
            if let Some((path, kind, ..)) = cur {
                if kind != NodeKind::SmartAlbum {
                    app.prompt("new folder: ", "", PendingCmd::NewFolder { parent: path });
                }
            }
        }
        KeyCode::Char('a') | KeyCode::Char('+') => {
            if let Some((path, kind, ..)) = cur {
                if kind != NodeKind::SmartAlbum {
                    app.prompt("new album: ", "", PendingCmd::NewAlbum { parent: path });
                }
            }
        }
        KeyCode::Char('R') => {
            if let Some((path, kind, _, _)) = cur {
                if kind != NodeKind::SmartAlbum {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                    app.prompt("rename to: ", name, PendingCmd::Rename { path });
                }
            }
        }
        KeyCode::Char('D') | KeyCode::Char('-') => {
            if let Some((path, kind, _, name)) = cur {
                if kind == NodeKind::SmartAlbum {
                    app.prompt(
                        format!("delete smart album \"{name}\"? [y/N] "),
                        "",
                        PendingCmd::DeleteSmart { name },
                    );
                } else {
                    let n = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                    app.prompt(
                        format!("delete \"{n}\"? [y/N] "),
                        "",
                        PendingCmd::Delete { path, is_album: kind == NodeKind::Album },
                    );
                }
            }
        }
        _ => {}
    }
    false
}

fn handle_grid_key(app: &mut App, code: KeyCode, ctrl: bool) -> bool {
    let n = app.view.len();
    let cols = app.cols.max(1);
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc | KeyCode::Tab => app.focus = Focus::Tree,
        KeyCode::Char('l') | KeyCode::Right => {
            if app.album_cursor + 1 < n {
                app.album_cursor += 1;
            }
        }
        KeyCode::Char('h') | KeyCode::Left => app.album_cursor = app.album_cursor.saturating_sub(1),
        KeyCode::Char('j') | KeyCode::Down => {
            if app.album_cursor + cols < n {
                app.album_cursor += cols;
            }
        }
        KeyCode::Char('k') | KeyCode::Up => app.album_cursor = app.album_cursor.saturating_sub(cols),
        KeyCode::Char('g') => app.album_cursor = 0,
        KeyCode::Char('G') => app.album_cursor = n.saturating_sub(1),
        KeyCode::Char('[') => app.cols = app.cols.saturating_sub(1).max(1),
        KeyCode::Char(']') => app.cols = (app.cols + 1).min(12),
        KeyCode::Char('s') => app.cycle_sort(),
        KeyCode::Enter => {
            if n > 0 {
                app.enter_image_view();
            }
        }
        // Filter bar (`/`) + culling loupe (`C`).
        KeyCode::Char('/') => app.filter_active = true,
        // Save the current filter as a library-wide smart album.
        KeyCode::Char('F') => {
            let q = app.filter.trim().to_string();
            if q.is_empty() {
                app.status = "type a filter (/) first, then F to save it as a smart album".into();
            } else {
                app.prompt("smart album name: ", "", PendingCmd::SaveSmart { query: q });
            }
        }
        // Library-wide metadata semantic search (prompts / captions / notes / tags).
        KeyCode::Char('?') => app.prompt("search metadata: ", "", PendingCmd::Search),
        // Library-wide CLIP visual search ("find images that look like…").
        KeyCode::Char('V') => app.prompt("visual search: ", "", PendingCmd::VisualSearch),
        // Natural-language command pipeline (e.g. "find rating>=4 then upscale then export to ~/x").
        KeyCode::Char(':') => app.prompt("command: ", "", PendingCmd::NlCommand),
        // Pixel-edit menu on the cursor image.
        KeyCode::Char('E') => {
            if n > 0 {
                app.open_edit_menu();
            }
        }
        // ML-edit menu (upscale / img2img / relight) on the cursor image.
        KeyCode::Char('M') => {
            if n > 0 {
                app.ml_menu = true;
            }
        }
        // Survey / compare the selection (2–4 images side by side).
        KeyCode::Char('=') => app.enter_compare(),
        // Vision + AI menu (Gemini autotag / describe) on the cursor image.
        KeyCode::Char('A') => {
            if n > 0 {
                app.ai_menu = true;
            }
        }
        // Stacking: collapse/expand derivative variants under their base.
        KeyCode::Char('S') => app.toggle_stack(),
        // Timeline: jump to a capture-month bucket.
        KeyCode::Char('@') => app.enter_timeline(),
        // Browse: scan for near-duplicates (#) · export (X) · batch-rename (r).
        KeyCode::Char('#') => app.dedup_scan(),
        // Take photo(s) for processing → a fresh nested working sub-album (keeps the source clean).
        KeyCode::Char('P') => app.take_photo(),
        // Put back: promote the selected finished image(s) up to the parent album.
        KeyCode::Char('p') => app.promote_to_parent(),
        // Duplicate the selected image(s) within the album (a <stem>_copy).
        KeyCode::Char('d') if !ctrl => app.duplicate_in_album(),
        // Collage from the selection (or the whole album); mosaic = varied-cell justified rows;
        // panorama prompts for a direction.
        // Flatten browse: show every image beneath the open album (incl. its sub-albums).
        KeyCode::Char('*') => {
            if let Some(dir) = app.album_dir.clone() {
                app.open_recursive(dir);
            }
        }
        KeyCode::Char('w') => app.make_collage(),
        KeyCode::Char('W') => app.make_mosaic(),
        KeyCode::Char('H') => app.prompt("multi-shot (hdr = exposure blend · focus = focus stack): ", "hdr", PendingCmd::Multishot),
        KeyCode::Char('b') => app.prompt("panorama (h / v / grid · ha / va aligned · hm homography): ", "h", PendingCmd::Panorama),
        KeyCode::Char('X') => {
            app.prompt("export to (DIR [MAXPX]): ", "", PendingCmd::Export);
        }
        KeyCode::Char('r') => {
            app.prompt("rename pattern (# = number): ", "", PendingCmd::BatchRename);
        }
        KeyCode::Char('C') => {
            if n > 0 {
                app.selected.clear(); // cull operates one image at a time
                app.mode = AlbumMode::Cull;
                app.load_view();
            }
        }
        // Selection (RFC §8.4) — indices are into `album_paths`, restricted to the visible view.
        KeyCode::Char(' ') => {
            if let Some(i) = app.cur_idx() {
                if !app.selected.insert(i) {
                    app.selected.remove(&i);
                }
            }
        }
        KeyCode::Char('a') if ctrl => app.selected = app.view.iter().copied().collect(),
        KeyCode::Char('d') if ctrl => app.selected.clear(),
        KeyCode::Char('i') if ctrl => {
            let visible: HashSet<usize> = app.view.iter().copied().collect();
            app.selected = visible.difference(&app.selected).copied().collect();
        }
        // Curation (RFC §8.5) — applies to selection or cursor; persisted to album.hjson.
        _ => return apply_curation(app, code),
    }
    false
}

/// Culling loupe (RFC §21.1): one image at a time, keep / reject / rate, advancing.
fn handle_cull_key(app: &mut App, code: KeyCode) -> bool {
    let advance = |app: &mut App| {
        if app.album_cursor + 1 < app.view.len() {
            app.album_cursor += 1;
            app.load_view();
        } else {
            app.mode = AlbumMode::Grid; // end of album → back to grid
        }
    };
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc => app.mode = AlbumMode::Grid,
        KeyCode::Right | KeyCode::Char(' ') => advance(app), // keep + advance
        KeyCode::Left => {
            if app.album_cursor > 0 {
                app.album_cursor -= 1;
                app.load_view();
            }
        }
        KeyCode::Char('x') => {
            app.edit_targets(|rec| rec.rejected = true);
            advance(app);
        }
        KeyCode::Char('f') => {
            app.edit_targets(|rec| rec.flagged = true);
            advance(app);
        }
        KeyCode::Char(d @ '1'..='5') => {
            let r = d.to_digit(10).unwrap() as u8;
            app.edit_targets(|rec| rec.rating = r);
            advance(app);
        }
        KeyCode::Char('i') => {
            app.info = if app.info == InfoPos::Right { InfoPos::Off } else { InfoPos::Right };
        }
        _ => {}
    }
    false
}

fn handle_image_key(app: &mut App, code: KeyCode) -> bool {
    let n = app.view.len();
    match code {
        KeyCode::Char('q') => return true,
        KeyCode::Esc => {
            app.mode = AlbumMode::Grid;
            app.view_proto = None;
        }
        KeyCode::Char('l') | KeyCode::Right => {
            if app.album_cursor + 1 < n {
                app.album_cursor += 1;
                app.zoom = 1.0; // each image starts fit
                app.edit_redo.clear(); // redo chain is per-image
                app.show_original = false;
                app.load_view();
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            if app.album_cursor > 0 {
                app.album_cursor -= 1;
                app.zoom = 1.0;
                app.edit_redo.clear();
                app.show_original = false;
                app.load_view();
            }
        }
        // Info panel: i = right, I (Shift-i) = bottom (toggle each).
        KeyCode::Char('i') => {
            app.info = if app.info == InfoPos::Right { InfoPos::Off } else { InfoPos::Right };
        }
        KeyCode::Char('I') => {
            app.info = if app.info == InfoPos::Bottom { InfoPos::Off } else { InfoPos::Bottom };
        }
        // Zoom: Z (Shift-z) in, z out.
        KeyCode::Char('Z') => app.zoom_by(1.5),
        KeyCode::Char('z') => app.zoom_by(1.0 / 1.5),
        // Before/after: show the pristine original vs the edited file.
        KeyCode::Char('\\') => app.toggle_before_after(),
        KeyCode::Char('H') => {
            app.show_analysis = !app.show_analysis;
            if app.show_analysis {
                app.compute_analysis();
            }
        }
        // Cycle the diagnostic overlay: off → clipping zebras → focus peaking.
        KeyCode::Char('o') => {
            app.overlay = match app.overlay {
                OverlayMode::Off => OverlayMode::Clipping,
                OverlayMode::Clipping => OverlayMode::FocusPeak,
                OverlayMode::FocusPeak => OverlayMode::Off,
            };
            app.status = match app.overlay {
                OverlayMode::Off => "overlay: off".into(),
                OverlayMode::Clipping => "overlay: clipping (red=blown · blue=crushed)".into(),
                OverlayMode::FocusPeak => "overlay: focus peaking (green=in-focus edges)".into(),
            };
            app.load_view();
        }
        KeyCode::Char('E') => app.open_edit_menu(),
        KeyCode::Char('M') => app.ml_menu = true,   // open the ML-edit menu
        KeyCode::Char('A') => app.ai_menu = true,   // open the vision/AI menu
        _ => return apply_curation(app, code),
    }
    false
}

/// Rating (1–5, 0 clears) / flag / reject / color-label, on the selection or cursor.
fn apply_curation(app: &mut App, code: KeyCode) -> bool {
    match code {
        KeyCode::Char(d @ '0'..='5') => {
            let r = d.to_digit(10).unwrap() as u8;
            app.edit_targets(|rec| rec.rating = r);
        }
        KeyCode::Char('f') => app.edit_targets(|rec| rec.flagged = !rec.flagged),
        KeyCode::Char('x') => app.edit_targets(|rec| rec.rejected = !rec.rejected),
        KeyCode::Char('c') => {
            const LABELS: [&str; 5] = ["red", "yellow", "green", "blue", "purple"];
            app.edit_targets(|rec| {
                let next = match rec.color_label.as_deref() {
                    None => Some(LABELS[0]),
                    Some(l) => LABELS.iter().position(|x| *x == l).and_then(|i| LABELS.get(i + 1)).copied(),
                };
                rec.color_label = next.map(String::from);
            });
        }
        // Free-text metadata editing (RFC §8.5) — opens the command pane on the cursor image.
        KeyCode::Char('t') => app.begin_edit(EditField::Tags),
        KeyCode::Char('e') => app.begin_edit(EditField::Caption),
        KeyCode::Char('N') => app.begin_edit(EditField::Notes),
        KeyCode::Char('T') => app.begin_edit(EditField::Title),
        // Curation undo / redo (u / U).
        KeyCode::Char('u') => app.undo_curation(),
        KeyCode::Char('U') => app.redo_curation(),
        _ => {}
    }
    false
}

fn draw(f: &mut Frame, app: &mut App) {
    let [status_bar, body, cmd_pane] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1), Constraint::Length(3)])
            .areas(f.area());

    // Top bar: the open album's name + library counts + a live histogram of the current image
    // (updates under the edit previews). Transient status/hints go to the command pane below.
    let base = Style::default().fg(Color::Black).bg(Color::Gray);
    let mut top: Vec<Span> = vec![
        Span::styled(" plakat photos ", base.add_modifier(Modifier::DIM)),
        Span::styled(format!(" {} ", app.current_view_name()), base.add_modifier(Modifier::BOLD)),
        Span::styled(
            format!(" {} albums · {} images ", app.album_count(), app.root.total_images()),
            base.add_modifier(Modifier::DIM),
        ),
    ];
    if app.mode == AlbumMode::Image {
        if let Some(sp) = &app.view_spark {
            top.push(Span::styled(format!("  {sp}  "), base.fg(Color::DarkGray)));
        }
    }
    // Memory-status indicator: steers the user away from AI-heavy ops when memory is low.
    let (mem_txt, mem_col) = app.mem_hint();
    top.push(Span::styled(mem_txt, base.fg(mem_col).add_modifier(Modifier::BOLD)));
    f.render_widget(Paragraph::new(Line::from(top)).style(base), status_bar);

    let [tree_col, album_col] =
        Layout::horizontal([Constraint::Percentage(30), Constraint::Fill(1)]).areas(body);

    draw_tree(f, app, tree_col);
    draw_album(f, app, album_col);
    if app.timeline {
        draw_timeline(f, app, album_col);
    }
    if app.tag_browser {
        draw_tag_browser(f, app, album_col);
    }
    if app.version_browser {
        draw_version_browser(f, app, album_col);
    }
    if app.preset_browser {
        draw_preset_browser(f, app, album_col);
    }
    if app.edit_menu {
        draw_edit_palette(f, app, album_col);
    }
    if app.ml_menu {
        draw_menu_palette(f, "ML edit (loads a model)", Color::Magenta, &ml_palette(), album_col);
    }
    if app.ai_menu {
        draw_menu_palette(f, "AI vision", Color::Green, &ai_palette(), album_col);
    }
    if app.adjust_mode {
        draw_adjust_bar(f, app, album_col);
    }
    if app.curve_mode {
        draw_curve_editor(f, app, album_col);
    }
    if app.history_mode {
        draw_history_bar(f, app, album_col);
    }
    if app.info_editor {
        let rows = [
            prow("n", "edit name"),
            prow("d", "edit description"),
            prow("t", "set tags (replace)"),
            prow("a", "add tags"),
            prow("c", "set cover image"),
            prow("s", "set sort order"),
        ];
        draw_menu_palette(f, "Album info editor", Color::Yellow, &rows, tree_col);
    }
    if let Some((title, lines)) = &app.album_info {
        draw_info_panel(f, title, lines, body);
    }
    if let Some(cat) = app.chord_prefix {
        draw_chord_overlay(f, cat, album_col);
    }
    if let Some(kind) = app.help {
        draw_help(f, kind, app, body);
    }

    // Command pane: menu hint, active text input, or a passive prompt. (Menus render as a palette
    // overlay — see draw_menu_palette — so the pane just nudges toward it.)
    let (cmd, cmd_style) = if app.edit_menu || app.ml_menu || app.ai_menu {
        (" ▸ pick from the palette · Esc to close ".to_string(), Style::default().fg(Color::DarkGray))
    } else if app.cmd_active {
        (format!(" {}{}_", app.cmd_prompt, app.cmd_buffer), Style::default().fg(Color::Yellow))
    } else if !app.status.is_empty() {
        // The transient status / contextual key-hints live here now (not the top bar).
        (format!(" {} ", app.status), Style::default().fg(Color::Gray))
    } else {
        (" CMD ▶ ".to_string(), Style::default())
    };
    f.render_widget(
        Paragraph::new(cmd).style(cmd_style).block(Block::default().borders(Borders::ALL)),
        cmd_pane,
    );
}

/// Interactive scalar-adjustment slider overlay: the op name, a `─┼───●──` bar (center tick at 0,
/// dot at the value), the numeric value, and the keys.
fn draw_adjust_bar(f: &mut Frame, app: &App, area: Rect) {
    let Some(op) = app.adjust_op.as_ref() else { return };
    let (min, max, _) = op.scalar_range();
    let v = op.scalar().unwrap_or(0);
    let width = 40usize.min(area.width.saturating_sub(6).max(10) as usize).max(10);
    let pos = |x: i32| {
        (((x - min) as f32 / (max - min).max(1) as f32) * (width as f32 - 1.0)).round() as usize
    };
    let mut bar: Vec<char> = vec!['─'; width];
    let zp = pos(0);
    if zp < width {
        bar[zp] = '┼'; // the neutral (0) tick
    }
    let vp = pos(v).min(width - 1);
    bar[vp] = '●';
    let bar: String = bar.into_iter().collect();
    let lines = vec![
        Line::from(vec![
            Span::styled(op.label(), Style::new().add_modifier(Modifier::BOLD)),
            Span::styled(format!("  {v:+}   [{min}..{max}]", ), Style::new().fg(Color::DarkGray)),
        ]),
        Line::from(Span::styled(bar, Style::new().fg(Color::Cyan))),
        Line::from(Span::styled(
            "←/→ fine ±1 · [/] jump · Enter apply · Esc",
            Style::new().fg(Color::DarkGray),
        )),
    ];
    let w = (width as u16 + 4).min(area.width);
    let h = 5.min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + area.height.saturating_sub(h + 1),
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Interactive tone-curve editor overlay: a small graph of the 5-point curve with the selected point
/// marked, plus the keys.
fn draw_curve_editor(f: &mut Frame, app: &App, area: Rect) {
    const IN: [i32; 5] = [0, 64, 128, 192, 255];
    let pts = app.curve_pts;
    // Build the 256-LUT (same math as edit::adjust::curve) then sample into a graph grid.
    let xs = [0i32, 64, 128, 192, 255];
    let lut = |i: i32| -> i32 {
        let mut seg = 0;
        while seg < 3 && i > xs[seg + 1] {
            seg += 1;
        }
        let (x0, x1) = (xs[seg], xs[seg + 1]);
        let (y0, y1) = (pts[seg], pts[seg + 1]);
        let t = if x1 > x0 { (i - x0) as f32 / (x1 - x0) as f32 } else { 0.0 };
        (y0 as f32 + t * (y1 - y0) as f32).round() as i32
    };
    const GW: usize = 24;
    const GH: usize = 8;
    let mut grid = vec![vec![' '; GW]; GH];
    for gx in 0..GW {
        let inp = (gx * 255 / (GW - 1)) as i32;
        let out = lut(inp).clamp(0, 255);
        let gy = (GH - 1) - (out as usize * (GH - 1) / 255);
        grid[gy][gx] = '·';
    }
    // Mark the 5 control points ('#', the selected one '●').
    for (i, &inp) in IN.iter().enumerate() {
        let gx = (inp as usize * (GW - 1) / 255).min(GW - 1);
        let gy = (GH - 1) - (pts[i].clamp(0, 255) as usize * (GH - 1) / 255);
        grid[gy][gx] = if i == app.curve_sel { '●' } else { '#' };
    }
    let mut lines: Vec<Line> = grid
        .into_iter()
        .map(|row| Line::from(Span::styled(row.into_iter().collect::<String>(), Style::new().fg(Color::Cyan))))
        .collect();
    lines.push(Line::from(Span::styled(
        format!("pt {}/5  in {}  out {}", app.curve_sel + 1, IN[app.curve_sel], pts[app.curve_sel]),
        Style::new().add_modifier(Modifier::BOLD),
    )));
    lines.push(Line::from(Span::styled(
        "←/→ point · ↑/↓ move · Enter apply · Esc",
        Style::new().fg(Color::DarkGray),
    )));
    let w = (GW as u16 + 4).min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default().borders(Borders::ALL).title(" Curves ").border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Edit-history scrubber overlay: a position bar over the edit stack + the current edit label.
fn draw_history_bar(f: &mut Frame, app: &App, area: Rect) {
    let n = app.history_ops.len();
    let label = if app.history_pos == 0 {
        "original".to_string()
    } else {
        app.history_ops
            .get(app.history_pos - 1)
            .and_then(edit::EditOp::from_entry)
            .map(|o| o.label())
            .unwrap_or_default()
    };
    // A dot per step: '○' original, '●' applied up to pos, '·' not-yet.
    let mut bar = String::new();
    for i in 0..=n {
        bar.push(if i == app.history_pos {
            '◉'
        } else if i < app.history_pos {
            '●'
        } else {
            '·'
        });
    }
    let lines = vec![
        Line::from(vec![
            Span::styled(format!("history {}/{n}  ", app.history_pos), Style::new().add_modifier(Modifier::BOLD)),
            Span::styled(label, Style::new().fg(Color::Cyan)),
        ]),
        Line::from(Span::styled(bar, Style::new().fg(Color::Cyan))),
        Line::from(Span::styled(
            "←/→ step · d delete this edit · Enter apply · Esc",
            Style::new().fg(Color::DarkGray),
        )),
    ];
    let w = (lines.iter().map(|l| l.width()).max().unwrap_or(30) as u16 + 4).min(area.width);
    let h = 5.min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + area.height.saturating_sub(h + 1),
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default().borders(Borders::ALL).border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Centered read-only info panel (album/folder `i`): a titled card, any key dismisses it.
fn draw_info_panel(f: &mut Frame, title: &str, lines: &[String], area: Rect) {
    let body: Vec<Line> = lines.iter().map(|l| Line::from(format!(" {l}"))).collect();
    let w = (body.iter().map(|l| l.width()).max().unwrap_or(30).clamp(24, 76) as u16 + 3).min(area.width);
    let h = (body.len() as u16 + 2).min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(body).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" {title} · any key "))
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

fn draw_tree(f: &mut Frame, app: &App, area: Rect) {
    let active = app.focus == Focus::Tree;
    let title = if app.tree_filter_active || !app.tree_filter.is_empty() {
        format!(" Library · /{}{} ", app.tree_filter, if app.tree_filter_active { "_" } else { "" })
    } else {
        " Library ".to_string()
    };
    let lines: Vec<Line> = app
        .rows()
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let icon = match (r.kind, r.has_children, r.expanded) {
                (NodeKind::SmartAlbum, ..) => "★ ",
                (NodeKind::Album, false, _) => "│ ",
                (_, true, true) => "▼ ",
                (_, true, false) => "▶ ",
                _ => "  ",
            };
            // Smart albums are virtual saved searches — no on-disk image count.
            let text = if r.kind == NodeKind::SmartAlbum {
                format!("{}{}", icon, r.name)
            } else {
                format!("{}{}{}  [{}]", "  ".repeat(r.depth), icon, r.name, r.count)
            };
            let mut style = Style::default();
            if r.kind == NodeKind::SmartAlbum {
                style = style.fg(Color::Magenta);
            }
            if i == app.tree_cursor {
                style = style.add_modifier(Modifier::REVERSED);
            }
            Line::from(Span::styled(text, style))
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(if active { Color::Yellow } else { Color::DarkGray })),
        ),
        area,
    );
}

fn draw_album(f: &mut Frame, app: &mut App, area: Rect) {
    let active = app.focus == Focus::Album;
    let sort = app.album_meta.sort.as_deref().unwrap_or("name-asc");
    // Count flagged images in the current view so the pick state is reflected in the header.
    let flagged = app
        .view
        .iter()
        .filter(|&&pi| app.album_paths.get(pi).and_then(|p| app.record(p)).map(|r| r.flagged).unwrap_or(false))
        .count();
    let flag_note = if flagged > 0 { format!("  ·  ⚑ {flagged}") } else { String::new() };
    let title = match (&app.smart, &app.album_dir) {
        (Some(name), _) if app.smart_is_search => format!(" 🔎 {name}  ·  ↕ {sort}{flag_note} "),
        (Some(name), _) => format!(" ★ {name}  ·  ↕ {sort}{flag_note} "),
        (None, Some(d)) => format!(
            " {}  ·  ↕ {}{} ",
            app.album_meta
                .name
                .as_deref()
                .filter(|n| !n.trim().is_empty())
                .unwrap_or_else(|| d.file_name().and_then(|n| n.to_str()).unwrap_or("album")),
            sort,
            flag_note,
        ),
        (None, None) => " Album ".to_string(),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(title)
        .border_style(Style::default().fg(if active { Color::Yellow } else { Color::DarkGray }));
    let inner = block.inner(area);
    f.render_widget(block, area);

    if app.album_paths.is_empty() {
        f.render_widget(
            Paragraph::new("\n  Select an album in the tree (→ / Enter).")
                .style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    }

    if app.mode == AlbumMode::Image || app.mode == AlbumMode::Cull {
        draw_image_view(f, app, inner);
        return;
    }
    if app.mode == AlbumMode::Compare {
        draw_compare(f, app, inner);
        return;
    }

    // Optional filter bar at the top of the grid (shown while typing or when a filter is set).
    let grid_area = if app.filter_active || !app.filter.is_empty() {
        let [bar, grid] = Layout::vertical([Constraint::Length(1), Constraint::Fill(1)]).areas(inner);
        let cursor = if app.filter_active { "_" } else { "" };
        let txt = format!(" filter ▸ {}{}   ({} match)", app.filter, cursor, app.view.len());
        f.render_widget(Paragraph::new(txt).style(Style::new().fg(Color::Yellow)), bar);
        grid
    } else {
        inner
    };

    if app.view.is_empty() {
        f.render_widget(
            Paragraph::new("  no images match the filter").style(Style::new().fg(Color::DarkGray)),
            grid_area,
        );
        return;
    }

    let cols = app.cols.max(1);
    let rows = grid_area.height as usize / 8; // ~8 text rows per thumbnail cell
    let per_page = (cols * rows).max(1);
    let page = app.album_cursor / per_page;
    let start = page * per_page;

    let row_rects = Layout::vertical(vec![Constraint::Ratio(1, rows.max(1) as u32); rows.max(1)]).split(grid_area);
    for r in 0..rows.max(1) {
        let col_rects =
            Layout::horizontal(vec![Constraint::Ratio(1, cols as u32); cols]).split(row_rects[r]);
        for c in 0..cols {
            let vi = start + r * cols + c;
            let Some(&pi) = app.view.get(vi) else { continue };
            let Some(path) = app.album_paths.get(pi).cloned() else { continue };
            let cell = col_rects[c];
            let is_cursor = vi == app.album_cursor;
            let is_sel = app.selected.contains(&pi);
            let rec = app.record(&path);
            let flagged = rec.map(|r| r.flagged).unwrap_or(false);
            let rejected = rec.map(|r| r.rejected).unwrap_or(false);
            let name: String = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let cap: String = {
                let cs: Vec<char> = name.chars().collect();
                if cs.len() > 14 { cs[cs.len() - 14..].iter().collect() } else { name.clone() }
            };
            // Cursor/selection take border priority; otherwise flagged→gold, rejected→dim red so the
            // pick/reject state is visible at a glance (not just the small badge).
            let border = if is_cursor {
                Color::Cyan
            } else if is_sel {
                Color::Green
            } else if flagged {
                Color::LightYellow
            } else if rejected {
                Color::Rgb(120, 60, 60)
            } else {
                Color::DarkGray
            };
            // Prefix a ⚑ on the flagged cell's title too.
            let title = match (is_cursor, flagged) {
                (true, _) => format!("▶{cap}"),
                (false, true) => format!("⚑{cap}"),
                (false, false) => cap,
            };
            let cb = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(border))
                .title(title);
            let ci = cb.inner(cell);
            f.render_widget(cb, cell);
            // Curation badge (computed before the mutable thumbs borrow).
            let badge = curation_badge(app.record(&path));
            let [thumb_area, badge_area] =
                Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(ci);
            match app.thumbs.get_mut(&path) {
                Some(proto) => f.render_stateful_widget(StatefulImage::new(), thumb_area, proto),
                None => f.render_widget(
                    Paragraph::new("…").style(Style::new().fg(Color::DarkGray)),
                    thumb_area,
                ),
            }
            f.render_widget(Paragraph::new(badge), badge_area);
        }
    }
}

/// The `YYYY-MM` capture-month bucket for the timeline, from the record's EXIF `date_taken`
/// (`"YYYY:MM:DD …"`). Images without a scanned capture date bucket as `undated`.
fn date_bucket(_path: &Path, rec: Option<&hjson::ImageRecord>) -> String {
    rec.and_then(|r| r.exif.as_ref())
        .and_then(|e| e.date_taken.as_ref())
        .and_then(|d| (d.len() >= 7).then(|| d[..7].replace(':', "-")))
        .unwrap_or_else(|| "undated".into())
}

/// Free-form crop preview: the full image with everything OUTSIDE the crop `rect` (x,y,w,h fractions)
/// dimmed, so the kept region stands out. Baked into the pixels (no graphics-protocol overlay needed).
fn crop_preview(img: &image::DynamicImage, rect: (f32, f32, f32, f32)) -> image::DynamicImage {
    let mut rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as f32, rgb.height() as f32);
    let (x0, y0) = (rect.0 * w, rect.1 * h);
    let (x1, y1) = ((rect.0 + rect.2) * w, (rect.1 + rect.3) * h);
    // Rule-of-thirds guide lines inside the crop rect (±1px tolerance so they show at any scale).
    let vt = [x0 + (x1 - x0) / 3.0, x0 + 2.0 * (x1 - x0) / 3.0];
    let hz = [y0 + (y1 - y0) / 3.0, y0 + 2.0 * (y1 - y0) / 3.0];
    for (px, py, p) in rgb.enumerate_pixels_mut() {
        let (fx, fy) = (px as f32, py as f32);
        let inside = fx >= x0 && fx < x1 && fy >= y0 && fy < y1;
        if !inside {
            p.0 = [p.0[0] / 3, p.0[1] / 3, p.0[2] / 3];
        } else if (vt.iter().any(|&v| (fx - v).abs() < 1.0) && fy >= y0 && fy < y1)
            || (hz.iter().any(|&v| (fy - v).abs() < 1.0) && fx >= x0 && fx < x1)
        {
            // Subtle thirds guide: lighten toward white for composition.
            p.0 = [p.0[0].max(200), p.0[1].max(200), p.0[2].max(200)];
        }
    }
    image::DynamicImage::ImageRgb8(rgb)
}

/// Draw the retouch pick-mode overlay: confirmed points (amber ×), the live crosshair (cyan), and —
/// for brush ops — the brush circle at the crosshair, all baked into the displayed image.
fn pick_preview(img: &image::DynamicImage, pk: &PickState) -> image::DynamicImage {
    let mut rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as i32, rgb.height() as i32);
    let put = |rgb: &mut image::RgbImage, x: i32, y: i32, c: [u8; 3]| {
        if x >= 0 && y >= 0 && x < w && y < h {
            rgb.put_pixel(x as u32, y as u32, image::Rgb(c));
        }
    };
    // Confirmed points → amber ×.
    for &(fx, fy) in &pk.points {
        let (cx, cy) = ((fx * w as f32) as i32, (fy * h as f32) as i32);
        for d in -4..=4 {
            put(&mut rgb, cx + d, cy + d, [255, 190, 40]);
            put(&mut rgb, cx + d, cy - d, [255, 190, 40]);
        }
    }
    // Live crosshair → cyan full-width/height lines.
    let (cx, cy) = ((pk.pos.0 * w as f32) as i32, (pk.pos.1 * h as f32) as i32);
    for x in 0..w {
        put(&mut rgb, x, cy, [0, 230, 230]);
    }
    for y in 0..h {
        put(&mut rgb, cx, y, [0, 230, 230]);
    }
    // Brush circle at the crosshair (radius as a fraction of the min dimension).
    if pk.op.uses_radius() {
        let r = (pk.radius * w.min(h) as f32).max(1.0);
        let steps = ((2.0 * std::f32::consts::PI * r) as i32).clamp(24, 720);
        for i in 0..steps {
            let a = i as f32 / steps as f32 * std::f32::consts::TAU;
            put(&mut rgb, cx + (r * a.cos()) as i32, cy + (r * a.sin()) as i32, [255, 255, 0]);
        }
    }
    image::DynamicImage::ImageRgb8(rgb)
}

/// Centre-crop `img` to `1/zoom` of its size (so fitting the crop to a pane magnifies by `zoom`).
/// Bake a diagnostic overlay into the displayed image: clipping "zebras" (blown → red, crushed →
/// blue) or focus peaking (high-gradient edges → green). `Off` returns the image unchanged.
fn apply_overlay(img: image::DynamicImage, mode: OverlayMode) -> image::DynamicImage {
    use image::Rgb;
    let lum = |p: &[u8; 3]| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32;
    match mode {
        OverlayMode::Off => img,
        OverlayMode::Clipping => {
            let mut rgb = img.to_rgb8();
            for p in rgb.pixels_mut() {
                let y = lum(&p.0);
                if y >= 250.0 {
                    p.0 = [255, 0, 0];
                } else if y <= 5.0 {
                    p.0 = [0, 0, 255];
                }
            }
            image::DynamicImage::ImageRgb8(rgb)
        }
        OverlayMode::FocusPeak => {
            let rgb = img.to_rgb8();
            let (w, h) = (rgb.width(), rgb.height());
            if w < 3 || h < 3 {
                return image::DynamicImage::ImageRgb8(rgb);
            }
            let at = |x: u32, y: u32| lum(&rgb.get_pixel(x, y).0);
            let mut out = rgb.clone();
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let gx = at(x + 1, y) - at(x - 1, y);
                    let gy = at(x, y + 1) - at(x, y - 1);
                    if (gx * gx + gy * gy).sqrt() > 40.0 {
                        out.put_pixel(x, y, Rgb([0, 255, 0]));
                    }
                }
            }
            image::DynamicImage::ImageRgb8(out)
        }
    }
}

fn crop_center(img: &image::DynamicImage, zoom: f32) -> image::DynamicImage {
    let (w, h) = (img.width(), img.height());
    let cw = ((w as f32 / zoom) as u32).max(1);
    let ch = ((h as f32 / zoom) as u32).max(1);
    img.crop_imm((w - cw) / 2, (h - ch) / 2, cw, ch)
}

/// File modified time, `UNIX_EPOCH` when unavailable (used as the `date-*` sort key — always
/// present, unlike EXIF capture time).
fn mtime(p: &Path) -> std::time::SystemTime {
    std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH)
}

/// Order `paths` in place per `mode` (album.hjson `sort`). `rating`/`score` look up each path's
/// curation values (source varies: the open album, or the smart-view maps). Unknown modes fall back
/// to `name-asc`. Ties break on filename so the order is deterministic.
fn sort_paths(
    paths: &mut [PathBuf],
    mode: &str,
    rating: impl Fn(&Path) -> u8,
    score: impl Fn(&Path) -> f64,
) {
    let name = |p: &Path| p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    match mode {
        "relevance" => {} // search results arrive pre-ranked — keep the given order
        "name-desc" => paths.sort_by(|a, b| name(b).cmp(&name(a))),
        "date-asc" => paths.sort_by_key(|p| mtime(p)),
        "date-desc" => paths.sort_by(|a, b| mtime(b).cmp(&mtime(a))),
        "rating-desc" => {
            paths.sort_by(|a, b| rating(b).cmp(&rating(a)).then_with(|| name(a).cmp(&name(b))))
        }
        "score-desc" => paths.sort_by(|a, b| {
            score(b).partial_cmp(&score(a)).unwrap_or(std::cmp::Ordering::Equal).then_with(|| name(a).cmp(&name(b)))
        }),
        _ => paths.sort_by(|a, b| name(a).cmp(&name(b))), // name-asc (default)
    }
}

/// Parse a dimensions string: `"1200x800"` / `"1200×800"` / `"1200 800"` → `(1200, 800)`; a single
/// `"1200"` → `(1200, 1200)`.
/// `<dir>/<stem>_<suffix>.<ext>`, suffixing `-2`, `-3`, … on a collision.
fn variant_path(dir: &Path, stem: &str, suffix: &str, ext: &str) -> PathBuf {
    let cand = dir.join(format!("{stem}_{suffix}.{ext}"));
    if !cand.exists() {
        return cand;
    }
    for i in 2..10_000 {
        let c = dir.join(format!("{stem}_{suffix}-{i}.{ext}"));
        if !c.exists() {
            return c;
        }
    }
    dir.join(format!("{stem}_{suffix}-dup.{ext}"))
}

/// Filename-safe version of a derived album name (keeps alphanumerics + `- _ . space`).
fn sanitize_name(s: &str) -> String {
    let out: String = s
        .chars()
        .map(|c| if c.is_alphanumeric() || "-_. ".contains(c) { c } else { '_' })
        .collect();
    let out = out.trim().to_string();
    if out.is_empty() { "photo".to_string() } else { out }
}

/// Human-readable byte size (GB/MB/KB/B).
fn human_size(b: u64) -> String {
    let f = b as f64;
    if f >= 1e9 {
        format!("{:.1} GB", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.0} KB", f / 1e3)
    } else {
        format!("{b} B")
    }
}

/// Parse a convert spec `fmt [Npx | NkB]` → `(fmt, ConvertSize)`. A bare integer = max longest side;
/// a `kb`/`k` suffix = target JPEG size; nothing = keep dimensions.
fn parse_convert(arg: &str) -> Option<(String, scrub::ConvertSize)> {
    let mut it = arg.split_whitespace();
    let fmt = it.next()?.to_string();
    let size = match it.next() {
        None => scrub::ConvertSize::Keep,
        Some(tok) => {
            let low = tok.to_ascii_lowercase();
            let low = low.strip_suffix("px").unwrap_or(&low);
            if let Some(kb) = low.strip_suffix("kb").or_else(|| low.strip_suffix('k')) {
                scrub::ConvertSize::MaxKb(kb.parse().ok()?)
            } else {
                scrub::ConvertSize::MaxPx(low.parse().ok()?)
            }
        }
    };
    Some((fmt, size))
}

fn parse_dims(s: &str) -> Option<(u32, u32)> {
    let parts: Vec<&str> = s
        .trim()
        .split(|c| matches!(c, 'x' | 'X' | '×' | ' ' | ','))
        .filter(|p| !p.is_empty())
        .collect();
    match parts.as_slice() {
        [n] => {
            let v: u32 = n.parse().ok()?;
            Some((v, v))
        }
        [w, h] => Some((w.parse().ok()?, h.parse().ok()?)),
        _ => None,
    }
}

/// Expand a leading `~` to `$HOME` for an export destination the user typed.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    PathBuf::from(path)
}

/// Collect every album directory under `node` (depth-first) — the search space for a smart album.
fn collect_album_dirs(node: &LibraryNode, out: &mut Vec<PathBuf>) {
    if node.kind == NodeKind::Album {
        out.push(node.path.clone());
    }
    for c in &node.children {
        collect_album_dirs(c, out);
    }
}

/// Build the searchable document for an image from its text metadata: filename, title, caption,
/// notes, tags, and — for `--import`ed images — the generation prompt + model. Fed to the TF-IDF
/// ranker for metadata search.
fn doc_for(path: &Path, rec: Option<&hjson::ImageRecord>) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(n) = path.file_name().and_then(|n| n.to_str()) {
        parts.push(n.to_string());
    }
    if let Some(r) = rec {
        if let Some(t) = &r.title {
            parts.push(t.clone());
        }
        if let Some(c) = &r.caption {
            parts.push(c.clone());
        }
        if let Some(n) = &r.notes {
            parts.push(n.clone());
        }
        if !r.tags.is_empty() {
            parts.push(r.tags.join(" "));
        }
        if let Some(g) = &r.generation {
            parts.push(g.prompt.clone());
            parts.push(g.model.clone());
        }
    }
    parts.join(" ")
}

/// Parse a comma-separated tag string into a trimmed, de-duplicated, order-preserving list.
fn parse_tags(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in s.split(',') {
        let t = t.trim();
        if !t.is_empty() && !out.iter().any(|x| x == t) {
            out.push(t.to_string());
        }
    }
    out
}

/// Filter grammar (RFC §16 subset): whitespace-separated predicates, ALL must match. Supports
/// `rating>=N` / `rating>N` / `rating=N` / `unrated`, `flag` / `-flag`, `rejected` / `-rejected`,
/// `ai` (scored), `tag:X` / `-tag:X`, and free text (filename contains).
fn matches_filter(name: &str, rec: Option<&hjson::ImageRecord>, filter: &str) -> bool {
    let f = filter.trim();
    if f.is_empty() {
        return true;
    }
    f.split_whitespace().all(|tok| tok_match(tok, name, rec))
}

fn tok_match(tok: &str, name: &str, rec: Option<&hjson::ImageRecord>) -> bool {
    let rating = rec.map(|r| r.rating).unwrap_or(0);
    let flagged = rec.map(|r| r.flagged).unwrap_or(false);
    let rejected = rec.map(|r| r.rejected).unwrap_or(false);
    let has_tag = |t: &str| rec.map(|r| r.tags.iter().any(|g| g.eq_ignore_ascii_case(t))).unwrap_or(false);
    let exif = rec.and_then(|r| r.exif.as_ref());
    match tok {
        "unrated" => rating == 0,
        "flag" | "flagged" => flagged,
        "-flag" => !flagged,
        "rejected" => rejected,
        "-rejected" => !rejected,
        "ai" | "scored" => rec.map(|r| r.score.is_some()).unwrap_or(false),
        _ if tok.starts_with("rating>=") => tok[8..].parse::<u8>().map(|n| rating >= n).unwrap_or(true),
        _ if tok.starts_with("rating>") => tok[7..].parse::<u8>().map(|n| rating > n).unwrap_or(true),
        _ if tok.starts_with("rating=") => tok[7..].parse::<u8>().map(|n| rating == n).unwrap_or(true),
        _ if tok.starts_with("tag:") => has_tag(&tok[4..]),
        _ if tok.starts_with("-tag:") => !has_tag(&tok[5..]),
        // EXIF / metadata predicates (read from the cached record — no file access).
        "has-gps" | "geotagged" => exif.is_some_and(|e| e.gps_lat.is_some() && e.gps_lon.is_some()),
        "-has-gps" => !exif.is_some_and(|e| e.gps_lat.is_some() && e.gps_lon.is_some()),
        _ if tok.starts_with("iso") => cmp_num(exif.and_then(|e| e.iso).map(|v| v as f64), &tok[3..]),
        _ if tok.starts_with("focal") => cmp_num(exif.and_then(|e| e.focal_length_mm), &tok[5..]),
        _ if tok.starts_with("date") => cmp_date(exif.and_then(|e| e.date_taken.as_deref()), &tok[4..]),
        _ if tok.starts_with("camera:") => exif.is_some_and(|e| {
            format!("{} {}", e.camera_make.clone().unwrap_or_default(), e.camera_model.clone().unwrap_or_default())
                .to_lowercase()
                .contains(&tok[7..].to_lowercase())
        }),
        _ if tok.starts_with("lens:") => text_has(exif.and_then(|e| e.lens_model.as_ref()), &tok[5..]),
        _ if tok.starts_with("author:") => text_has(rec.and_then(|r| r.author.as_ref()), &tok[7..]),
        _ if tok.starts_with("copyright:") => text_has(rec.and_then(|r| r.copyright.as_ref()), &tok[10..]),
        _ if tok.starts_with("title:") => text_has(rec.and_then(|r| r.title.as_ref()), &tok[6..]),
        _ => name.to_lowercase().contains(&tok.to_lowercase()),
    }
}

/// Case-insensitive "field contains query" (missing field → no match).
fn text_has(field: Option<&String>, q: &str) -> bool {
    field.is_some_and(|v| v.to_lowercase().contains(&q.to_lowercase()))
}

/// Numeric predicate `>N` / `>=N` / `<N` / `<=N` / `=N` / bare `N` (treated as `>=N`).
fn cmp_num(val: Option<f64>, spec: &str) -> bool {
    let Some(v) = val else { return false };
    let (op, num) = if let Some(r) = spec.strip_prefix(">=") {
        (">=", r)
    } else if let Some(r) = spec.strip_prefix("<=") {
        ("<=", r)
    } else if let Some(r) = spec.strip_prefix('>') {
        (">", r)
    } else if let Some(r) = spec.strip_prefix('<') {
        ("<", r)
    } else if let Some(r) = spec.strip_prefix('=') {
        ("=", r)
    } else {
        (">=", spec)
    };
    let Ok(n) = num.trim().parse::<f64>() else { return false };
    match op {
        ">" => v > n,
        ">=" => v >= n,
        "<" => v < n,
        "<=" => v <= n,
        _ => (v - n).abs() < 0.5,
    }
}

/// Date predicate against the capture year: `>YYYY` / `<YYYY` / `=YYYY` / bare `YYYY`, or `:text`
/// (the date string contains text).
fn cmp_date(val: Option<&str>, spec: &str) -> bool {
    let Some(d) = val else { return false };
    if let Some(q) = spec.strip_prefix(':') {
        return d.contains(q);
    }
    // EXIF dates are `YYYY:MM:DD …` or `YYYY-MM-DD` — the year is the first four chars.
    let Some(year) = d.get(0..4).and_then(|s| s.parse::<i32>().ok()) else { return false };
    cmp_num(Some(year as f64), spec)
}

fn label_color(l: &str) -> Color {
    match l {
        "red" => Color::Red,
        "yellow" => Color::Yellow,
        "green" => Color::Green,
        "blue" => Color::Blue,
        "purple" => Color::Magenta,
        _ => Color::White,
    }
}

/// Rating stars + flag / reject / color-label, from an image's curation record.
fn curation_badge(rec: Option<&hjson::ImageRecord>) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    if let Some(r) = rec {
        if r.rating > 0 {
            let stars: String = (0..5).map(|i| if i < r.rating { '★' } else { '☆' }).collect();
            spans.push(Span::styled(stars, Style::new().fg(Color::Yellow)));
        }
        if r.flagged {
            spans.push(Span::styled(" ⚑", Style::new().fg(Color::Yellow)));
        }
        if r.rejected {
            spans.push(Span::styled(" ✗", Style::new().fg(Color::Red)));
        }
        if let Some(l) = &r.color_label {
            spans.push(Span::styled(" ●", Style::new().fg(label_color(l))));
        }
        if !r.variants.is_empty() {
            // Stack badge: this image has derivative variants (edits / ML outputs).
            spans.push(Span::styled(format!(" ⧉{}", r.variants.len()), Style::new().fg(Color::Cyan)));
        }
    }
    Line::from(spans)
}

/// Full-pane image view (RFC §9): the image, optionally with an EXIF + curation side panel (`i`).
/// The context the quickhelp is shown in — drives which bindings/commands are listed.
enum HelpCtx {
    Tree,
    Grid,
    Image,
    Cull,
    Compare,
    Crop,
    Layers,
    Levels,
    Adjust,
    Curve,
    History,
}

fn help_ctx(app: &App) -> HelpCtx {
    if app.curve_mode {
        HelpCtx::Curve
    } else if app.history_mode {
        HelpCtx::History
    } else if app.adjust_mode {
        HelpCtx::Adjust
    } else if app.levels_mode {
        HelpCtx::Levels
    } else if app.layer_mode {
        HelpCtx::Layers
    } else if app.crop_mode {
        HelpCtx::Crop
    } else if app.focus == Focus::Tree {
        HelpCtx::Tree
    } else {
        match app.mode {
            AlbumMode::Grid => HelpCtx::Grid,
            AlbumMode::Image => HelpCtx::Image,
            AlbumMode::Cull => HelpCtx::Cull,
            AlbumMode::Compare => HelpCtx::Compare,
        }
    }
}

fn ctx_name(ctx: &HelpCtx) -> &'static str {
    match ctx {
        HelpCtx::Tree => "Tree",
        HelpCtx::Grid => "Album grid",
        HelpCtx::Image => "Image view",
        HelpCtx::Cull => "Cull loupe",
        HelpCtx::Compare => "Compare",
        HelpCtx::Crop => "Free-form crop",
        HelpCtx::Layers => "Layers",
        HelpCtx::Levels => "Levels",
        HelpCtx::Adjust => "Adjust",
        HelpCtx::Curve => "Curves",
        HelpCtx::History => "Edit history",
    }
}

/// Quickhelp overlay (`Ctrl-B h` chords / `Ctrl-B H` commands): a centered card **built for the
/// current pane/mode** and seeded with live state. Any key dismisses it.
fn draw_help(f: &mut Frame, kind: HelpKind, app: &App, area: Rect) {
    let ctx = help_ctx(app);
    let (label, lines): (&str, Vec<Line>) = match kind {
        HelpKind::Chords => ("chords", chords_help(app, &ctx)),
        HelpKind::Commands => ("commands", commands_help(app, &ctx)),
    };
    let title = format!(" {} · {} · any key to close ", ctx_name(&ctx), label);
    let w = (lines.iter().map(|l| l.width()).max().unwrap_or(40).clamp(24, 76) as u16 + 4).min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

fn hd(s: &str) -> Line<'static> {
    Line::from(Span::styled(s.to_string(), Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD)))
}
fn kv(k: &str, v: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {k:<10}"), Style::new().fg(Color::Cyan)),
        Span::raw(v.into()),
    ])
}
/// A dim live-state line (current sort, selection count, …).
fn state(v: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(format!("  {}", v.into()), Style::new().fg(Color::DarkGray)))
}

/// Live one-line status of the album view (sort, selection, filter, stacking) for the help header.
fn album_state_lines(app: &App) -> Vec<Line<'static>> {
    let mut out = vec![state(format!(
        "{} images · sort {} · {} selected",
        app.view.len(),
        app.album_meta.sort.as_deref().unwrap_or("name-asc"),
        app.selected.len(),
    ))];
    if !app.filter.trim().is_empty() {
        out.push(state(format!("filter: {}", app.filter)));
    }
    if app.stack_view {
        out.push(state("stacked (variants collapsed)"));
    }
    out
}

/// Contextual key chords for the current pane/mode, seeded with live state.
fn chords_help(app: &App, ctx: &HelpCtx) -> Vec<Line<'static>> {
    let mut l: Vec<Line> = Vec::new();
    match ctx {
        HelpCtx::Tree => {
            l.push(hd("Tree"));
            l.push(kv("j k / ↑↓", "move · PgUp/PgDn page · Home/End · g/G first/last"));
            l.push(kv("l → Enter", "open album / expand folder"));
            l.push(kv("h ←", "collapse folder / up one level"));
            l.push(kv("n a/+ R D/-", "new folder · new album · rename · delete"));
            l.push(hd("Album"));
            l.push(kv("i / I", "info panel / info editor (name·desc·cover·sort)"));
            l.push(kv("t / T", "add album tags / edit album tags"));
            l.push(kv("e / E", "export album / export + convert (fmt)"));
            l.push(kv("r /", "regenerate thumbnails · filter tree by name"));
            l.push(kv("Tab", "focus the album grid"));
            l.push(kv("? V", "metadata search · visual search"));
        }
        HelpCtx::Grid => {
            l.extend(album_state_lines(app));
            l.push(hd("Move"));
            l.push(kv("h j k l", "move · g/G first-last · [ ] columns · Enter open"));
            l.push(hd("Curate (image/selection)"));
            l.push(kv("1–5 0", "rate/clear · f flag · x reject · c colour"));
            l.push(kv("t e N T", "tags · caption · notes · title · s sort"));
            l.push(kv("u / U", "undo / redo curation"));
            l.push(hd("Select"));
            l.push(kv("Space", "toggle · Ctrl-A all · Ctrl-D none · Ctrl-I invert"));
            l.push(hd("Do"));
            l.push(kv("/ C =", "filter · cull · compare selection"));
            l.push(kv("E M A", "edit · ML-edit · AI-vision menus"));
            l.push(kv("# X r", "duplicates · export · batch-rename"));
            l.push(kv("P / p", "take → working sub-album  ·  put back → parent album"));
            l.push(kv("d w b", "duplicate · collage · panorama (h/v/grid)"));
            l.push(kv("S @ F", "stack · timeline · save smart album"));
            l.push(kv("? V", "search metadata · visual (CLIP)"));
        }
        HelpCtx::Image => {
            l.push(hd("Image view"));
            l.push(kv("← →", "previous / next · Esc back to grid"));
            l.push(kv("Z / z", format!("zoom in / out ({:.1}×)", app.zoom)));
            l.push(kv("i / I", "info panel: right / bottom"));
            l.push(kv("H", format!("analysis panel{}", if app.show_analysis { " (on)" } else { "" })));
            l.push(kv("o", "overlay: clipping zebras → focus peaking → off"));
            l.push(hd("Edit"));
            l.push(kv("E", "edit palette (search + Enter)"));
            l.push(kv("Ctrl-B", "edit chord → category g/c/a/k/x/e/m → item (see KEYMAP.md)"));
            l.push(hd("Curate"));
            l.push(kv("1–5 0", "rate/clear · f flag · x reject · c colour"));
            l.push(kv("t e N T", "tags · caption · notes · title"));
            l.push(kv("E M A", "edit · ML-edit · AI-vision menus"));
        }
        HelpCtx::Cull => {
            l.push(hd("Cull loupe"));
            l.push(kv("→ Space", "keep + advance · ← back"));
            l.push(kv("x f", "reject + advance · flag + advance"));
            l.push(kv("1–5", "rate + advance · i EXIF · Esc leave"));
        }
        HelpCtx::Compare => {
            l.push(hd("Compare"));
            l.push(kv("← →", "move focus between images"));
            l.push(kv("1–5 f x", "rate / flag / reject the focused one"));
            l.push(kv("Esc", "back to grid"));
        }
        HelpCtx::Crop => {
            let (_, _, w, h) = app.crop_rect;
            l.push(hd("Free-form crop"));
            l.push(state(format!("current box: {:.0}% × {:.0}% of the image", w * 100.0, h * 100.0)));
            l.push(hd("Change the box"));
            l.push(kv("+ / -", "grow / shrink both sides (centred)"));
            l.push(kv("[ / ]", "width  narrower / wider"));
            l.push(kv(", / .", "height  shorter / taller"));
            l.push(kv("← ↑ → ↓", "move the box"));
            l.push(hd("Finish"));
            l.push(kv("Enter", "apply the crop · Esc cancel"));
            l.push(state("(for an exact size, use the Edit palette → crop/resize to exact size)"));
            return l; // crop mode has no global chords
        }
        HelpCtx::Layers => {
            l.push(hd("Layers"));
            if let Some(la) = app.layers.get(app.layer_active) {
                l.push(state(format!("active {}/{}: {}", app.layer_active + 1, app.layers.len(), la.label())));
            } else {
                l.push(state("stack is empty — press a to add an image"));
            }
            l.push(hd("Build the stack"));
            l.push(kv("a", "add an image (album file or path) as a new top layer"));
            l.push(kv("n / p", "select next / previous layer (Tab / Shift-Tab)"));
            l.push(kv("{ / }", "move the active layer down / up in z-order"));
            l.push(kv("x", "delete the active layer"));
            l.push(hd("Transform the active layer"));
            l.push(kv("← ↑ → ↓", "move · + / - grow / shrink"));
            l.push(kv("< / >", "opacity down / up · b cycle blend mode"));
            l.push(hd("Mask the active layer"));
            l.push(kv("m", "cycle mask: none → ellipse → rectangle"));
            l.push(kv("k", "image matte — a grayscale file (white shows, black hides)"));
            l.push(kv("M", "position the mask (arrows move it) — Enter/Esc when done"));
            l.push(kv("[ / ]", "mask smaller / larger · , / . feather less / more"));
            l.push(kv("/", "invert the mask (show ↔ hide)"));
            l.push(hd("Finish"));
            l.push(kv("Enter", "flatten → a new _layered.png · Esc leave (stack saved)"));
            return l; // layer mode has no global chords
        }
        HelpCtx::Levels => {
            l.push(hd("Levels"));
            l.push(state(format!(
                "black {} · white {} · γ {:.2}",
                app.lv_black,
                app.lv_white,
                app.lv_gamma as f32 / 100.0
            )));
            l.push(hd("Adjust"));
            l.push(kv("↑ / ↓", "pick a handle: black point / white point / gamma"));
            l.push(kv("← / →", "decrease / increase it ([ ] and , . also work)"));
            l.push(state("black/white clip the input range; gamma > 1 brightens the mid-tones"));
            l.push(hd("Finish"));
            l.push(kv("Enter", "apply as an edit · Esc cancel"));
            return l; // levels mode has no global chords
        }
        HelpCtx::Adjust => {
            l.push(hd("Adjust"));
            if let Some(op) = &app.adjust_op {
                let (min, max, step) = op.scalar_range();
                l.push(state(format!("{}: {:+}  [{min}..{max}, step {step}]", op.label(), op.scalar().unwrap_or(0))));
            }
            l.push(kv("← / →", "fine ±1"));
            l.push(kv("[ / ]", "jump by the coarse step  ·  − / + · PgUp/PgDn also jump"));
            l.push(kv("Enter", "apply as an edit · Esc cancel"));
            l.push(state("the bar's centre tick is 0 (no change); the dot is the current value"));
            return l; // adjust mode has no global chords
        }
        HelpCtx::Curve => {
            l.push(hd("Curves"));
            l.push(state("5 points: input 0 · 64 · 128 · 192 · 255, each with an output value"));
            l.push(kv("← / →", "pick a point"));
            l.push(kv("↑ / ↓", "move its output · [ / ] fine"));
            l.push(kv("Enter", "apply as an edit · Esc cancel"));
            return l;
        }
        HelpCtx::History => {
            l.push(hd("Edit history"));
            l.push(state("replays the edit stack over the pristine original"));
            l.push(kv("← / →", "step back / forward through the edits"));
            l.push(kv("d", "delete the edit shown at this step"));
            l.push(kv("Enter", "apply the (trimmed) stack · Esc cancel"));
            return l;
        }
    }
    l.push(hd("Anywhere"));
    l.push(kv("Ctrl-B", "h/H help · t tags · v versions · l/L lookalike · q quit"));
    l
}

/// Contextual named commands for the current pane/mode — the vocabulary a natural-language command
/// would map onto.
fn commands_help(app: &App, ctx: &HelpCtx) -> Vec<Line<'static>> {
    let mut l: Vec<Line> = Vec::new();
    match ctx {
        HelpCtx::Tree => {
            l.push(hd("Organise"));
            l.push(kv("new", "create a folder (n) or album (a) here"));
            l.push(kv("rename", "rename this folder/album (R)"));
            l.push(kv("delete", "delete this folder/album (D)"));
            l.push(kv("open", "open the album (→) — then Ctrl-B H for its commands"));
        }
        HelpCtx::Grid | HelpCtx::Image | HelpCtx::Cull | HelpCtx::Compare | HelpCtx::Crop
        | HelpCtx::Layers | HelpCtx::Levels | HelpCtx::Adjust | HelpCtx::Curve | HelpCtx::History => {
            let provider = crate::prompt::vision::resolve_vision_provider("auto");
            let target = if app.selected.is_empty() { "the view" } else { "the selection" };
            l.extend(album_state_lines(app));
            l.push(hd("Curate"));
            l.push(kv("rate", format!("stars / flag / reject / colour on {target}")));
            l.push(kv("tag caption", "edit tags / caption / notes / title"));
            l.push(kv("autotag", format!("AI vision → tags/caption (via {provider})")));
            l.push(hd("Find"));
            l.push(kv("filter", "rating/flag/tag/ai grammar (/)"));
            l.push(kv("search", "metadata (?) · visual CLIP (V) · smart album (F)"));
            l.push(kv("duplicates", "perceptual near-dupes (#)"));
            l.push(kv("take", "copy hi-res into a working sub-album (P)"));
            l.push(hd("Produce"));
            l.push(kv("upscale", "ML ×4 (M → u)"));
            l.push(kv("img2img", "transform / relight with a prompt (M → i/l)"));
            l.push(kv("edit", "rotate/flip/crop/bright/contrast (E)"));
            l.push(kv("layers", "overlay/compose images → flatten (E → layers)"));
            l.push(kv("export", format!("copy {target} out, optional resize (X)")));
            l.push(kv("portfolio", "Ctrl-B p: watermarked copies + contact sheet"));
            l.push(kv("rename", "batch-rename with a #-pattern (r)"));
            l.push(hd("Natural language  (:)"));
            l.push(kv(":", "album-scoped command — pipe with 'then'"));
            l.push(state("find rating>=4 then upscale then export to ~/best 2000"));
            l.push(state("all photos then autotag   ·   take flag then rate 5"));
        }
    }
    l
}

/// A modal command palette for a menu: `key — description` rows, centred, keys still act underneath.
fn draw_menu_palette(f: &mut Frame, title: &str, color: Color, rows: &[(String, String)], area: Rect) {
    let keyw = rows.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(3);
    let width = rows
        .iter()
        .map(|(_k, d)| keyw + 2 + d.chars().count())
        .max()
        .unwrap_or(30)
        .clamp(24, 60) as u16
        + 4;
    let w = width.min(area.width);
    let h = (rows.len() as u16 + 2).min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    let lines: Vec<Line> = rows
        .iter()
        .map(|(k, d)| {
            Line::from(vec![
                Span::styled(format!(" {k:>keyw$}  "), Style::new().fg(color).add_modifier(Modifier::BOLD)),
                Span::raw(d.clone()),
            ])
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" {title} · any listed key · Esc "))
                .border_style(Style::default().fg(color)),
        ),
        popup,
    );
}

fn prow(k: &str, d: &str) -> (String, String) {
    (k.to_string(), d.to_string())
}

/// Searchable/scrollable Edit command palette (`E`): a search line + a windowed command list with a
/// highlighted cursor. Type to filter, arrows/PgUp-Dn/Home/End to move, Enter to run.
fn draw_edit_palette(f: &mut Frame, app: &mut App, area: Rect) {
    let cmds = filtered_edit_commands(&app.edit_query);
    let w = 54u16.min(area.width);
    let h = (cmds.len() as u16 + 4).clamp(6, area.height); // search line + list + borders
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);

    let list_h = popup.height.saturating_sub(3) as usize; // minus borders (2) + search line (1)
    app.edit_visible = list_h.max(1);
    let cursor = app.edit_cursor.min(cmds.len().saturating_sub(1));
    let start = if cursor >= list_h { cursor + 1 - list_h } else { 0 };

    let mut lines: Vec<Line> = vec![Line::from(vec![
        Span::styled("🔎 ", Style::new().fg(Color::DarkGray)),
        Span::styled(app.edit_query.clone(), Style::new().fg(Color::Yellow)),
        Span::styled("_", Style::new().fg(Color::DarkGray)),
    ])];
    if cmds.is_empty() {
        lines.push(Line::from(Span::styled("  no match", Style::new().fg(Color::DarkGray))));
    }
    let labw = (w as usize).saturating_sub(12); // room for the chord column on the right
    for (i, (label, chord, _)) in cmds.iter().enumerate().skip(start).take(list_h) {
        let sel = i == cursor;
        let base = if sel { Style::default().add_modifier(Modifier::REVERSED) } else { Style::default() };
        let name = format!(" {label:<w$}", w = labw.saturating_sub(1));
        let chord_span = if chord.is_empty() {
            Span::styled("       ".to_string(), base)
        } else {
            Span::styled(format!("⌃B {chord} "), if sel { base } else { Style::new().fg(Color::DarkGray) })
        };
        lines.push(Line::from(vec![Span::styled(name, base), chord_span]));
    }
    let title = format!(" Edit palette  {}/{} · Enter run · ⌃B chord · Esc ", (cursor + 1).min(cmds.len().max(1)), cmds.len());
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default().borders(Borders::ALL).title(title).border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

fn ml_palette() -> Vec<(String, String)> {
    vec![
        prow("u", "ML upscale ×4 (Real-ESRGAN)"),
        prow("i", "img2img — transform with a prompt"),
        prow("l", "relight — re-illuminate with a prompt"),
        prow("g", "generate — txt2img from a prompt"),
        prow("p", "portrait — this image as the face + a prompt"),
        prow("m", "multiperson — selected images as people + a scene"),
        prow("Esc", "close  (runs a model; Esc cancels the inline ops)"),
    ]
}

fn ai_palette() -> Vec<(String, String)> {
    vec![
        prow("t", "autotag — LLM vision → tags"),
        prow("d", "describe — LLM vision → caption"),
        prow("g", "recipe-tag AI images (offline)"),
        prow("r", "aesthetic auto-cull — rank + keep top N"),
        prow("n", "analyze & generate — describe → img2img"),
        prow("f", "face-scan — detect + group faces"),
        prow("Esc", "close"),
    ]
}

/// Version browser popup (Ctrl-B v): save the current image as a snapshot, or restore an earlier one.
fn draw_version_browser(f: &mut Frame, app: &App, area: Rect) {
    let name = app.version_target.as_ref().map(|(_, f, _)| f.as_str()).unwrap_or("");
    let w = area.width.min(34).max(20);
    let rows = (app.versions_list.len() as u16 + 3).clamp(3, area.height);
    let popup = Rect { x: area.x + area.width.saturating_sub(w), y: area.y, width: w, height: rows };
    f.render_widget(Clear, popup);
    let mut items: Vec<Line> = vec![row_line("＋ save current version", app.version_cursor == 0)];
    for (i, n) in app.versions_list.iter().enumerate() {
        items.push(row_line(&format!("v{n}  (restore)"), app.version_cursor == i + 1));
    }
    f.render_widget(
        Paragraph::new(items).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Versions · {name} "))
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

fn row_line(text: &str, selected: bool) -> Line<'static> {
    let mut style = Style::default();
    if selected {
        style = style.add_modifier(Modifier::REVERSED);
    }
    Line::from(Span::styled(format!(" {text}"), style))
}

/// Which-key overlay after `Ctrl-B <category>` in image view: the category's edit chords + labels.
fn draw_chord_overlay(f: &mut Frame, cat: char, area: Rect) {
    let name = chord_categories().iter().find(|(c, _)| *c == cat).map(|(_, n)| *n).unwrap_or("");
    let rows: Vec<(String, String)> = edit_commands()
        .iter()
        .filter(|(_, chord, _)| chord.starts_with(cat))
        .map(|(label, chord, _)| (chord[1..].to_string(), label.to_string()))
        .collect();
    let keyw = rows.iter().map(|(k, _)| k.chars().count()).max().unwrap_or(1);
    let width = rows.iter().map(|(_, d)| keyw + 2 + d.chars().count()).max().unwrap_or(24).clamp(24, 60) as u16 + 4;
    let w = width.min(area.width);
    let h = (rows.len() as u16 + 2).min(area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    let lines: Vec<Line> = rows
        .iter()
        .map(|(k, d)| {
            Line::from(vec![
                Span::styled(format!(" {k:>keyw$}  "), Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(d.clone()),
            ])
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" ⌃B {cat} · {name} · Esc "))
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Preset picker popup (Edit → apply preset): saved edit stacks; Enter applies to the targets.
fn draw_preset_browser(f: &mut Frame, app: &App, area: Rect) {
    let rows: Vec<String> = app
        .presets_list
        .iter()
        .map(|p| format!("{}  ({} edits)", p.name, p.ops.len()))
        .collect();
    let w = (rows.iter().map(|r| r.len()).max().unwrap_or(20) as u16 + 4).clamp(24, 60).min(area.width);
    let h = (rows.len() as u16 + 2).clamp(3, area.height);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(w)) / 2,
        y: area.y + (area.height.saturating_sub(h)) / 2,
        width: w,
        height: h,
    };
    f.render_widget(Clear, popup);
    let inner_h = popup.height.saturating_sub(2) as usize;
    let start = app.preset_cursor.saturating_sub(inner_h.saturating_sub(1));
    let lines: Vec<Line> = rows
        .iter()
        .enumerate()
        .skip(start)
        .take(inner_h)
        .map(|(i, r)| row_line(r, i == app.preset_cursor))
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Apply preset · Enter · Esc ")
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Tag browser popup (Ctrl-B t): the album's tags with counts; Enter filters by the chosen tag.
fn draw_tag_browser(f: &mut Frame, app: &App, area: Rect) {
    let w = area.width.min(30).max(18);
    let rows = (app.tags_list.len() as u16 + 2).clamp(3, area.height);
    let popup = Rect {
        x: area.x + area.width.saturating_sub(w),
        y: area.y,
        width: w,
        height: rows,
    };
    f.render_widget(Clear, popup);
    let inner_h = popup.height.saturating_sub(2) as usize;
    let start = app.tag_cursor.saturating_sub(inner_h.saturating_sub(1));
    let lines: Vec<Line> = app
        .tags_list
        .iter()
        .enumerate()
        .skip(start)
        .take(inner_h)
        .map(|(i, (tag, count))| {
            let text = format!(" {tag}  ({count})");
            let mut style = Style::default();
            if i == app.tag_cursor {
                style = style.add_modifier(Modifier::REVERSED);
            }
            Line::from(Span::styled(text, style))
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Tags · Enter filters ")
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Timeline popup (Phase 5): a scrollable list of capture-month buckets with counts, over the right
/// edge of the album pane.
fn draw_timeline(f: &mut Frame, app: &App, area: Rect) {
    let w = area.width.min(28).max(16);
    let rows = (app.tl_buckets.len() as u16 + 2).min(area.height);
    let popup = Rect {
        x: area.x + area.width.saturating_sub(w),
        y: area.y,
        width: w,
        height: rows.max(3),
    };
    f.render_widget(Clear, popup);
    let inner_h = popup.height.saturating_sub(2) as usize;
    // Keep the cursor visible within the window.
    let start = app.tl_cursor.saturating_sub(inner_h.saturating_sub(1));
    let lines: Vec<Line> = app
        .tl_buckets
        .iter()
        .enumerate()
        .skip(start)
        .take(inner_h)
        .map(|(i, (label, _, count))| {
            let text = format!(" {label}  ({count})");
            let mut style = Style::default();
            if i == app.tl_cursor {
                style = style.add_modifier(Modifier::REVERSED);
            }
            Line::from(Span::styled(text, style))
        })
        .collect();
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(" Timeline ")
                .border_style(Style::default().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// Side-by-side survey/compare (Phase 5): N images in equal columns, the focused one cyan-bordered,
/// each with its filename + curation badge.
fn draw_compare(f: &mut Frame, app: &mut App, area: Rect) {
    let n = app.compare.len();
    if n == 0 {
        return;
    }
    // Precompute labels + badges (immutable borrow) before the mutable protocol render.
    let meta: Vec<(String, Line, bool)> = app
        .compare
        .iter()
        .enumerate()
        .map(|(i, (p, _))| {
            let name = p.file_name().and_then(|x| x.to_str()).unwrap_or("").to_string();
            (name, curation_badge(app.record(p)), i == app.compare_cursor)
        })
        .collect();
    let cells = Layout::horizontal(vec![Constraint::Ratio(1, n as u32); n]).split(area);
    for (i, (_, proto)) in app.compare.iter_mut().enumerate() {
        let (name, badge, focused) = &meta[i];
        let block = Block::default()
            .borders(Borders::ALL)
            .title(if *focused { format!("▶ {name}") } else { name.clone() })
            .border_style(Style::default().fg(if *focused { Color::Cyan } else { Color::DarkGray }));
        let inner = block.inner(cells[i]);
        f.render_widget(block, cells[i]);
        let [img_area, badge_area] =
            Layout::vertical([Constraint::Fill(1), Constraint::Length(1)]).areas(inner);
        f.render_stateful_widget(StatefulImage::new(), img_area, proto);
        f.render_widget(Paragraph::new(badge.clone()), badge_area);
    }
}

/// Layer-mode HUD (Phase 8): the stack listed top-of-stack first, the active layer highlighted.
fn draw_layers_hud(f: &mut Frame, app: &App, area: Rect) {
    let header = if app.mask_adjust {
        " Layers · ◆ MASK MOVE · arrows position · Enter/Esc done "
    } else {
        " Layers · a add · b blend · m mask · M move-mask · Enter flatten · Esc "
    };
    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        header,
        Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
    ))];
    if app.layers.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (empty) — press a to add an image",
            Style::new().fg(Color::DarkGray),
        )));
    }
    // Top of the list = top of the stack (last composited).
    for (i, l) in app.layers.iter().enumerate().rev() {
        let sel = i == app.layer_active;
        let mut st = Style::default();
        if sel {
            st = st.fg(Color::Yellow).add_modifier(Modifier::BOLD);
        }
        lines.push(Line::from(Span::styled(
            format!("{} {}", if sel { "▶" } else { " " }, l.label()),
            st,
        )));
    }
    let w = (lines.iter().map(|l| l.width()).max().unwrap_or(20) as u16 + 2).min(area.width);
    let h = (lines.len() as u16 + 2).min(area.height);
    let popup = Rect { x: area.x, y: area.y, width: w, height: h };
    f.render_widget(Clear, popup);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default().borders(Borders::ALL).border_style(Style::new().fg(Color::Cyan)),
        ),
        popup,
    );
}

/// View-analysis panel (Phase 6): a luma histogram bar chart + exposure/focus stats.
fn draw_analysis_panel(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(" Analysis (H) ");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let Some(a) = &app.analysis else {
        f.render_widget(Paragraph::new(" computing…").style(Style::new().fg(Color::DarkGray)), inner);
        return;
    };

    // Histogram as an 8-row bar chart across `BINS` buckets, scaled to the panel width.
    const ROWS: usize = 8;
    const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let width = (inner.width as usize).clamp(1, analysis::BINS);
    let max = a.hist.iter().copied().max().unwrap_or(1).max(1) as f32;
    // Down-sample the 64 buckets to the panel width.
    let cols: Vec<f32> = (0..width)
        .map(|c| {
            let lo = c * analysis::BINS / width;
            let hi = ((c + 1) * analysis::BINS / width).max(lo + 1);
            let s: u32 = a.hist[lo..hi.min(analysis::BINS)].iter().sum();
            (s as f32 / (hi - lo) as f32) / max // 0..1
        })
        .collect();
    let mut lines: Vec<Line> = Vec::with_capacity(ROWS + 6);
    for row in (0..ROWS).rev() {
        let s: String = cols
            .iter()
            .map(|&v| {
                let cell = (v * ROWS as f32) - row as f32; // portion of this cell filled in this row
                BLOCKS[(cell.clamp(0.0, 1.0) * 8.0).round() as usize]
            })
            .collect();
        lines.push(Line::from(Span::styled(s, Style::new().fg(Color::Cyan))));
    }
    lines.push(Line::from(Span::styled("0──────────────255 luma", Style::new().fg(Color::DarkGray))));
    lines.push(Line::from(""));
    lines.push(Line::from(format!("size     {}×{}", a.width, a.height)));
    lines.push(Line::from(format!("mean     {:.0}/255", a.mean)));
    let clip = |label: &str, frac: f32, warn: bool| {
        let c = if warn && frac > 0.02 { Color::Red } else { Color::Reset };
        Line::from(Span::styled(format!("{label}{:.1}%", frac * 100.0), Style::new().fg(c)))
    };
    lines.push(clip("clip hi  ", a.clip_high, true));
    lines.push(clip("clip lo  ", a.clip_low, true));
    lines.push(Line::from(format!("focus    {:.0}", a.sharpness)));

    // Colour balance — mean R/G/B as coloured bars (a right-leaning bar = a colour cast).
    let barw = (inner.width as usize).saturating_sub(12).clamp(6, 32);
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Colour balance", Style::new().fg(Color::Yellow))));
    for (i, (lbl, col)) in [("R", Color::Red), ("G", Color::Green), ("B", Color::Blue)].iter().enumerate() {
        let v = a.channel_mean[i];
        lines.push(Line::from(vec![
            Span::raw(format!("{lbl} ")),
            Span::styled(hbar(v / 255.0, barw), Style::new().fg(*col)),
            Span::styled(format!(" {v:.0}"), Style::new().fg(Color::DarkGray)),
        ]));
    }

    // Lighting balance — share of pixels in shadows / midtones / highlights.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Lighting", Style::new().fg(Color::Yellow))));
    for (i, lbl) in ["shadow", "mid", "highlt"].iter().enumerate() {
        let v = a.zones[i];
        lines.push(Line::from(vec![
            Span::raw(format!("{lbl} ")),
            Span::styled(hbar(v, barw), Style::new().fg(Color::Cyan)),
            Span::styled(format!(" {:.0}%", v * 100.0), Style::new().fg(Color::DarkGray)),
        ]));
    }

    // Per-channel (RGB) histograms as compact coloured sparklines.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("RGB histogram", Style::new().fg(Color::Yellow))));
    let sw = (inner.width as usize).saturating_sub(2).clamp(8, analysis::BINS);
    for (i, col) in [Color::Red, Color::Green, Color::Blue].iter().enumerate() {
        lines.push(Line::from(Span::styled(sparkline(&a.hist_rgb[i], sw), Style::new().fg(*col))));
    }

    // Dominant colours — swatches of the most common hues.
    if !a.dominant.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Dominant colours", Style::new().fg(Color::Yellow))));
        let sw: Vec<Span> = a
            .dominant
            .iter()
            .map(|c| Span::styled("██", Style::new().fg(Color::Rgb(c[0], c[1], c[2]))))
            .collect();
        lines.push(Line::from(sw));
    }

    // Luma waveform (a per-column tonal scope) — brightest row at the top.
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled("Waveform", Style::new().fg(Color::Yellow))));
    let wmax = a.waveform.iter().flatten().copied().max().unwrap_or(1).max(1) as f32;
    for row in &a.waveform {
        lines.push(Line::from(Span::styled(scope_row(row, wmax), Style::new().fg(Color::Gray))));
    }

    // RGB parade — the three channel waveforms side by side (R | G | B).
    lines.push(Line::from(Span::styled("RGB parade", Style::new().fg(Color::Yellow))));
    let pmax = a.parade.iter().flatten().flatten().copied().max().unwrap_or(1).max(1) as f32;
    let third = analysis::WCOLS / 3;
    let cols = [Color::Red, Color::Green, Color::Blue];
    for r in 0..analysis::WROWS {
        let spans: Vec<Span> = (0..3)
            .map(|c| Span::styled(scope_row(&a.parade[c][r][..third], pmax), Style::new().fg(cols[c])))
            .collect();
        lines.push(Line::from(spans));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

/// One waveform row → a string of block-shaded cells scaled to `max`.
fn scope_row(row: &[u16], max: f32) -> String {
    const B: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    row.iter().map(|&v| B[((v as f32 / max) * 8.0).round().clamp(0.0, 8.0) as usize]).collect()
}

/// A compact luma-histogram sparkline of `img` (sampled, 28 columns) for the top bar.
fn spark_of(img: &image::DynamicImage) -> String {
    let rgb = img.to_rgb8();
    let total = (rgb.width() as u64 * rgb.height() as u64).max(1);
    let stride = (total / 8000).max(1) as usize; // cap ~8k samples
    let mut hist = [0u32; 28];
    for (i, p) in rgb.pixels().enumerate() {
        if i % stride != 0 {
            continue;
        }
        let y = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as usize;
        hist[(y * 28 / 256).min(27)] += 1;
    }
    sparkline(&hist, 28)
}

/// A one-row sparkline of `hist` down-sampled to `width` columns (8 block levels).
fn sparkline(hist: &[u32], width: usize) -> String {
    const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let n = hist.len().max(1);
    let max = hist.iter().copied().max().unwrap_or(1).max(1) as f32;
    (0..width)
        .map(|c| {
            let lo = c * n / width;
            let hi = ((c + 1) * n / width).max(lo + 1).min(n);
            let s: u32 = hist[lo..hi].iter().sum();
            let v = (s as f32 / (hi - lo) as f32) / max; // 0..1
            BLOCKS[(v * 8.0).round().clamp(0.0, 8.0) as usize]
        })
        .collect()
}

/// A horizontal bar of `width` cells filled to `frac` (0..1) with block chars.
fn hbar(frac: f32, width: usize) -> String {
    let fill = (frac.clamp(0.0, 1.0) * width as f32).round() as usize;
    let mut s = String::with_capacity(width);
    for i in 0..width {
        s.push(if i < fill { '█' } else { '░' });
    }
    s
}

fn draw_image_view(f: &mut Frame, app: &mut App, area: Rect) {
    // Analysis (H) forces a right panel; otherwise the info panel sits per `info` (i=right, I=bottom).
    let placement = if app.show_analysis { InfoPos::Right } else { app.info };
    let (img_area, panel) = match placement {
        InfoPos::Off => (area, None),
        InfoPos::Right => {
            let [a, b] = Layout::horizontal([Constraint::Percentage(60), Constraint::Fill(1)]).areas(area);
            (a, Some(b))
        }
        InfoPos::Bottom => {
            let [a, b] = Layout::vertical([Constraint::Fill(1), Constraint::Length(8)]).areas(area);
            (a, Some(b))
        }
    };

    match app.view_proto.as_mut() {
        // `Scale` (unlike the default `Fit`) also upscales, so the image always fills the pane —
        // needed for zoom, where the cropped centre may be smaller than the render area.
        Some(proto) => {
            f.render_stateful_widget(StatefulImage::new().resize(Resize::Scale(None)), img_area, proto)
        }
        None => f.render_widget(Paragraph::new("  decoding…").style(Style::new().fg(Color::DarkGray)), img_area),
    }

    // Layer mode: a compact stack HUD over the top-left of the image.
    if app.layer_mode {
        draw_layers_hud(f, app, img_area);
    }

    // The analysis panel (H) takes precedence over EXIF (i) when both are on.
    if app.show_analysis {
        if let Some(panel) = panel {
            draw_analysis_panel(f, app, panel);
        }
        return;
    }

    if let Some(panel) = panel {
        let path = app.cur_idx().and_then(|i| app.album_paths.get(i).cloned());
        let name = path.as_ref().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("").to_string();
        let mut lines: Vec<Line> = vec![Line::from(Span::styled(name.clone(), Style::new().add_modifier(Modifier::BOLD)))];
        let rec_ref = path.as_ref().and_then(|p| app.record(p));
        // The record's EXIF carries any manual edits (date / geotag); fall back to the file's EXIF.
        let exif_display = rec_ref.and_then(|r| r.exif.as_ref()).or(app.view_exif.as_ref());
        if let Some(e) = exif_display {
            let mut kv = |k: &str, v: Option<String>| {
                if let Some(v) = v {
                    lines.push(Line::from(format!("{k:<9}{v}")));
                }
            };
            kv("Date", e.date_taken.clone());
            kv("Camera", match (&e.camera_make, &e.camera_model) {
                (Some(m), Some(md)) => Some(format!("{m} {md}")),
                (_, Some(md)) => Some(md.clone()),
                _ => None,
            });
            kv("Lens", e.lens_model.clone());
            kv("Focal", e.focal_length_mm.map(|f| format!("{f:.0}mm")));
            kv("Aperture", e.aperture.clone());
            kv("Shutter", e.shutter.clone());
            kv("ISO", e.iso.map(|i| i.to_string()));
            kv("Size", match (e.width_px, e.height_px) {
                (Some(w), Some(h)) => Some(format!("{w}×{h}")),
                _ => None,
            });
            kv("GPS", match (e.gps_lat, e.gps_lon) {
                (Some(la), Some(lo)) => Some(format!("{la:.4}, {lo:.4}")),
                _ => None,
            });
        }
        // Editable metadata (title / author / copyright) from the record.
        if let Some(r) = rec_ref {
            let mut kv = |k: &str, v: Option<String>| {
                if let Some(v) = v.filter(|s| !s.is_empty()) {
                    lines.push(Line::from(format!("{k:<9}{v}")));
                }
            };
            kv("Title", r.title.clone());
            kv("Author", r.author.clone());
            kv("©", r.copyright.clone());
        }
        if let Some(r) = path.as_ref().and_then(|p| app.record(p)) {
            lines.push(section("curation"));
            lines.push(curation_badge(Some(r)));
            if let Some(s) = r.score {
                lines.push(Line::from(format!("score    {s:.2}")));
            }
            if let Some(t) = &r.title {
                lines.push(Line::from(format!("title    {t}")));
            }
            if !r.tags.is_empty() {
                lines.push(Line::from(format!("tags     {}", r.tags.join(", "))));
            }
            if let Some(c) = &r.caption {
                lines.push(Line::from(format!("caption  {c}")));
            }
            if let Some(no) = &r.notes {
                lines.push(Line::from(format!("notes    {no}")));
            }
            if !r.edits.is_empty() {
                lines.push(section("edits"));
                lines.push(Line::from(Span::styled(
                    format!("{} edit(s):  {}", r.edits.len(), edit_summary(&r.edits)),
                    Style::new().fg(Color::Cyan),
                )));
            }
            // Generation recipe for plakat-made images (`--import`).
            if let Some(g) = &r.generation {
                lines.push(section("plakat"));
                lines.push(Line::from(format!("model    {}", g.model)));
                lines.push(Line::from(format!("seed     {}", g.seed)));
                lines.push(Line::from(format!("steps    {}  cfg {}", g.steps, g.guidance)));
            }
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::default().borders(Borders::ALL).title(" Info ")),
            panel,
        );
    }
}

/// A styled section divider for the info panel.
fn section(name: &str) -> Line<'static> {
    Line::from(Span::styled(format!("── {name} ──"), Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD)))
}

/// Collapse a raw edit log into a readable summary: group by label (first-seen order) with counts,
/// e.g. "shadows ×3 · sharpen ×4 · warmth". Avoids the repetitive "shadows, shadows, shadows, …".
fn edit_summary(edits: &[hjson::EditEntry]) -> String {
    let mut order: Vec<String> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for op in edits.iter().filter_map(edit::EditOp::from_entry) {
        let l = op.label();
        if !counts.contains_key(&l) {
            order.push(l.clone());
        }
        *counts.entry(l).or_insert(0) += 1;
    }
    order
        .iter()
        .map(|l| {
            let c = counts[l];
            if c > 1 { format!("{l} ×{c}") } else { l.clone() }
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

#[cfg(test)]
mod tree_ops_tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn exif_and_metadata_filter_predicates() {
        let mut rec = hjson::ImageRecord::default();
        rec.author = Some("Jane Roe".into());
        rec.exif = Some(hjson::ExifRecord {
            date_taken: Some("2024:07:14 12:00:00".into()),
            camera_make: Some("Canon".into()),
            camera_model: Some("EOS R5".into()),
            focal_length_mm: Some(50.0),
            iso: Some(6400),
            gps_lat: Some(37.7),
            gps_lon: Some(-122.4),
            ..Default::default()
        });
        let r = Some(&rec);
        assert!(matches_filter("x.jpg", r, "iso>3200"));
        assert!(!matches_filter("x.jpg", r, "iso<3200"));
        assert!(matches_filter("x.jpg", r, "focal=50"));
        assert!(matches_filter("x.jpg", r, "camera:canon"));
        assert!(matches_filter("x.jpg", r, "date>2023"));
        assert!(!matches_filter("x.jpg", r, "date<2020"));
        assert!(matches_filter("x.jpg", r, "date:2024"));
        assert!(matches_filter("x.jpg", r, "has-gps"));
        assert!(matches_filter("x.jpg", r, "author:jane"));
        // ALL predicates must hold.
        assert!(matches_filter("x.jpg", r, "iso>=6400 camera:canon has-gps"));
        // No EXIF → the exif predicates fail (and -has-gps holds).
        let empty = hjson::ImageRecord::default();
        assert!(!matches_filter("y.jpg", Some(&empty), "iso>100"));
        assert!(matches_filter("y.jpg", Some(&empty), "-has-gps"));
    }

    #[test]
    fn gen_spec_parses_model_and_size() {
        // No spec → default model, pipeline-default size.
        assert_eq!(parse_gen_spec("a red fox in snow"), ("a red fox in snow".into(), "sdxl".into(), None));
        // model only.
        assert_eq!(parse_gen_spec("a knight | sd15"), ("a knight".into(), "sd15".into(), None));
        // model + square size.
        assert_eq!(parse_gen_spec("a fox | sd15 512"), ("a fox".into(), "sd15".into(), Some((512, 512))));
        // model + WxH, case/×-insensitive; clamped.
        assert_eq!(parse_gen_spec("x | sdxl 1024x768"), ("x".into(), "sdxl".into(), Some((1024, 768))));
        assert_eq!(parse_gen_size_tok("99"), Some((256, 256))); // clamped up
        // Identity strategy tracks the family.
        assert!(matches!(identity_for_model("sdxl"), crate::api::IdentityKind::PlusFaceSdxl));
        assert!(matches!(identity_for_model("sd15"), crate::api::IdentityKind::PlusFace));
    }

    #[test]
    fn gather_image_files_album_and_recursive() {
        let dir = std::env::temp_dir().join(format!("plakat-tree-{}", std::process::id()));
        let sub = dir.join("sub");
        std::fs::create_dir_all(&sub).unwrap();
        let img = image::DynamicImage::ImageRgb8(ImageBuffer::from_pixel(4, 4, Rgb([1u8, 2, 3])));
        img.save(dir.join("a.png")).unwrap();
        img.save(dir.join("b.jpg")).unwrap();
        img.save(sub.join("c.png")).unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap(); // not an image

        // Non-recursive: only the two images directly in `dir`.
        assert_eq!(App::gather_image_files(&dir, false).len(), 2);
        // Recursive: includes the sub-album image.
        assert_eq!(App::gather_image_files(&dir, true).len(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_presets_roundtrip_through_folder_hjson() {
        let dir = std::env::temp_dir().join(format!("plakat-preset-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ops = vec![
            edit::EditOp::Warmth(20).to_entry(),
            edit::EditOp::Vignette(30).to_entry(),
        ];
        let fm = hjson::FolderMeta {
            edit_presets: vec![hjson::EditPreset { name: "warm vignette".into(), ops }],
            ..Default::default()
        };
        hjson::write_folder(&dir, &fm).unwrap();
        let back = hjson::read_folder(&dir).unwrap();
        assert_eq!(back.edit_presets.len(), 1);
        assert_eq!(back.edit_presets[0].name, "warm vignette");
        // The stored ops parse back into real edits.
        let parsed: Vec<_> =
            back.edit_presets[0].ops.iter().filter_map(edit::EditOp::from_entry).collect();
        assert_eq!(parsed, vec![edit::EditOp::Warmth(20), edit::EditOp::Vignette(30)]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edit_chords_are_unique_two_char_and_categorised() {
        let cats: Vec<char> = chord_categories().iter().map(|(c, _)| *c).collect();
        // Categories must not collide with the global Ctrl-B leader keys.
        for g in ['h', 'H', 't', 'v', 'p', 'l', 'L', 'q'] {
            assert!(!cats.contains(&g), "category '{g}' collides with a global leader key");
        }
        let mut seen = std::collections::HashSet::new();
        for (_, chord, _) in edit_commands() {
            if chord.is_empty() {
                continue; // palette-only entry (e.g. a numbered filter variant)
            }
            assert_eq!(chord.chars().count(), 2, "chord '{chord}' must be 2 chars");
            let cat = chord.chars().next().unwrap();
            assert!(cats.contains(&cat), "chord '{chord}' category '{cat}' not registered");
            assert!(seen.insert(chord), "duplicate chord '{chord}'");
            assert!(edit_cmd_for_chord(chord).is_some(), "chord '{chord}' does not resolve");
        }
    }

    #[test]
    fn edit_summary_groups_repeats_in_order() {
        let edits = vec![
            edit::EditOp::Shadows(15).to_entry(),
            edit::EditOp::Shadows(15).to_entry(),
            edit::EditOp::Shadows(15).to_entry(),
            edit::EditOp::Sharpen(22).to_entry(),
            edit::EditOp::Warmth(10).to_entry(),
            edit::EditOp::Sharpen(22).to_entry(),
        ];
        // Grouped by label, first-seen order, with counts — not "shadows, shadows, shadows, …".
        assert_eq!(edit_summary(&edits), "shadows ×3 · sharpen ×2 · warmth");
    }

    #[test]
    fn sanitize_name_is_filename_safe() {
        assert_eq!(sanitize_name("IMG_1234"), "IMG_1234");
        assert_eq!(sanitize_name("beach/day 2"), "beach_day 2");
        assert_eq!(sanitize_name("a:b*c?"), "a_b_c_");
        assert_eq!(sanitize_name("   "), "photo");
    }

    #[test]
    fn human_size_scales() {
        assert_eq!(human_size(512), "512 B");
        assert_eq!(human_size(2_000), "2 KB");
        assert_eq!(human_size(3_500_000), "3.5 MB");
        assert_eq!(human_size(2_000_000_000), "2.0 GB");
    }

    #[test]
    fn album_tags_roundtrip_through_hjson() {
        let dir = std::env::temp_dir().join(format!("plakat-atags-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = hjson::AlbumMeta { tags: vec!["trip".into(), "2026".into()], ..Default::default() };
        m.description = Some("summer".into());
        hjson::write_album(&dir, &m).unwrap();
        let back = hjson::read_album(&dir).unwrap();
        assert_eq!(back.tags, vec!["trip".to_string(), "2026".into()]);
        assert_eq!(back.description.as_deref(), Some("summer"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use crate::photos::hjson::ImageRecord;

    #[test]
    fn filter_predicates() {
        let mut rec = ImageRecord { rating: 4, flagged: true, ..Default::default() };
        rec.tags = vec!["waterfall".into()];
        let r = Some(&rec);
        assert!(matches_filter("a.jpg", r, ""));            // empty → all
        assert!(matches_filter("a.jpg", r, "rating>=4"));
        assert!(!matches_filter("a.jpg", r, "rating>=5"));
        assert!(matches_filter("a.jpg", r, "flag tag:waterfall"));  // AND
        assert!(!matches_filter("a.jpg", r, "-flag"));
        assert!(matches_filter("a.jpg", r, "-rejected"));
        assert!(matches_filter("IMG_1.jpg", r, "img"));     // free text (case-insensitive)
        assert!(!matches_filter("a.jpg", None, "rating>=1")); // no record → rating 0
        assert!(matches_filter("a.jpg", None, "unrated"));
    }

    fn p(s: &str) -> PathBuf {
        PathBuf::from(s)
    }

    #[test]
    fn sort_orders() {
        let mut meta = hjson::AlbumMeta::default();
        meta.images.insert("b.png".into(), ImageRecord { rating: 5, score: Some(2.0), ..Default::default() });
        meta.images.insert("a.png".into(), ImageRecord { rating: 1, score: Some(9.0), ..Default::default() });
        // c.png has no record → rating 0, no score.
        let names = |v: &[PathBuf]| v.iter().map(|p| p.file_name().unwrap().to_str().unwrap().to_string()).collect::<Vec<_>>();

        let rec = |p: &Path| p.file_name().and_then(|n| n.to_str()).and_then(|n| meta.images.get(n));
        let rating = |p: &Path| rec(p).map_or(0, |r| r.rating);
        let score = |p: &Path| rec(p).and_then(|r| r.score).unwrap_or(f64::MIN);

        let mut v = vec![p("c.png"), p("a.png"), p("b.png")];
        sort_paths(&mut v, "name-asc", rating, score);
        assert_eq!(names(&v), ["a.png", "b.png", "c.png"]);
        sort_paths(&mut v, "name-desc", rating, score);
        assert_eq!(names(&v), ["c.png", "b.png", "a.png"]);
        sort_paths(&mut v, "rating-desc", rating, score);
        assert_eq!(names(&v), ["b.png", "a.png", "c.png"]); // 5, 1, 0
        sort_paths(&mut v, "score-desc", rating, score);
        assert_eq!(names(&v), ["a.png", "b.png", "c.png"]); // 9.0, 2.0, none(MIN)
        sort_paths(&mut v, "bogus-mode", rating, score); // unknown → name-asc
        assert_eq!(names(&v), ["a.png", "b.png", "c.png"]);
    }

    #[test]
    fn date_bucket_from_exif_month() {
        let mut rec = ImageRecord::default();
        rec.exif = Some(crate::photos::hjson::ExifRecord {
            date_taken: Some("2024:07:15 12:00:00".into()),
            ..Default::default()
        });
        assert_eq!(date_bucket(Path::new("a.jpg"), Some(&rec)), "2024-07");
        assert_eq!(date_bucket(Path::new("a.jpg"), None), "undated");
        assert_eq!(date_bucket(Path::new("a.jpg"), Some(&ImageRecord::default())), "undated");
    }

    #[test]
    fn tags_parse_trim_dedup() {
        assert_eq!(parse_tags("sunset, beach ,sunset,, iceland"), ["sunset", "beach", "iceland"]);
        assert!(parse_tags("   ,  ,").is_empty());
        assert_eq!(parse_tags("solo"), ["solo"]);
    }

    #[test]
    fn doc_for_gathers_text_metadata() {
        use crate::imaging::metadata::GenerationMetadata;
        let rec = ImageRecord {
            title: Some("Foxes".into()),
            caption: Some("a red fox in snow".into()),
            notes: Some("golden hour".into()),
            tags: vec!["wildlife".into(), "winter".into()],
            generation: Some(GenerationMetadata::new(
                "majestic fox portrait", "sdxl", 1, 20, 7.0, "ddim", 512, 512,
            )),
            ..Default::default()
        };
        let d = doc_for(Path::new("/a/plakat-7.png"), Some(&rec)).to_lowercase();
        for term in ["plakat-7", "foxes", "fox", "snow", "golden", "wildlife", "winter", "majestic", "sdxl"] {
            assert!(d.contains(term), "doc missing '{term}': {d}");
        }
        // No record → just the filename.
        assert_eq!(doc_for(Path::new("/a/b.png"), None), "b.png");
    }

    #[test]
    fn collect_album_dirs_finds_only_albums() {
        let base = std::env::temp_dir().join(format!("plakat-collect-{}", std::process::id()));
        let iceland = base.join("2024/Iceland");
        let faroes = base.join("2024/Faroes");
        std::fs::create_dir_all(&iceland).unwrap();
        std::fs::create_dir_all(&faroes).unwrap();
        for d in [&iceland, &faroes] {
            image::DynamicImage::ImageRgb8(image::ImageBuffer::from_pixel(2, 2, image::Rgb([0, 0, 0])))
                .save(d.join("x.png"))
                .unwrap();
        }
        let root = library::walk(&base).unwrap();
        let mut dirs = Vec::new();
        collect_album_dirs(&root, &mut dirs);
        assert!(dirs.contains(&iceland));
        assert!(dirs.contains(&faroes));
        assert!(!dirs.contains(&base.join("2024"))); // a folder of sub-dirs is not an album
        let _ = std::fs::remove_dir_all(&base);
    }
}
