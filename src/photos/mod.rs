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
pub mod mledit;
pub mod nl;
pub mod rename;
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
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;
use ratatui_image::StatefulImage;

use library::{LibraryNode, NodeKind};

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
    /// Collect a prompt for a T2 ML edit (`relight` when true, else `img2img`), then queue the job.
    MlPrompt { relight: bool },
    /// Export the current targets to the entered `DIR [MAXPX]`.
    Export,
    /// Batch-rename the current targets with the entered pattern (`#` runs = numbers).
    BatchRename,
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

/// Which quickhelp overlay is showing (opened via the `Ctrl-B` leader).
#[derive(Clone, Copy, PartialEq, Eq)]
enum HelpKind {
    /// Key chords (the keyboard shortcuts).
    Chords,
    /// Named commands / actions.
    Commands,
}

/// A free-text per-image metadata field editable from the command pane.
#[derive(Clone, Copy)]
enum EditField {
    Caption,
    Notes,
    Title,
    Tags,
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
    show_exif: bool,
    // View analysis (RFC §Phase 6): histogram + exposure/focus stats panel in the image view (`H`).
    show_analysis: bool,
    analysis: Option<analysis::Analysis>,
    view_proto: Option<StatefulProtocol>,
    view_exif: Option<hjson::ExifRecord>,
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
    smart_src: HashMap<PathBuf, PathBuf>, // image path → its source album dir (for write routing)
    smart_rec: HashMap<PathBuf, hjson::ImageRecord>, // image path → its record (badges/filter/sort)

    // T1 pixel-edit menu (RFC §Phase 3) — a modal key layer over the cursor image.
    edit_menu: bool,
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
    // Natural-language command pipeline (`:`): a raw query awaiting the LLM planner, and a parsed
    // plan awaiting the user's y/N confirmation.
    pending_nl: Option<String>,
    pending_plan: Option<nl::CommandPlan>,

    // Survey / compare (RFC §Phase 5): a small set of images decoded side-by-side, with a focus.
    compare: Vec<(PathBuf, StatefulProtocol)>,
    compare_cursor: usize,

    // Stacking (RFC §Phase 5): when on, derivative variants collapse under their base in the grid.
    stack_view: bool,
    // Timeline (RFC §Phase 5): a modal list of date buckets over the current view.
    timeline: bool,
    tl_buckets: Vec<(String, usize, usize)>, // (label, first view-position, count)
    tl_cursor: usize,

    // `Ctrl-B` leader prefix (tmux-style) + the quickhelp overlay it opens.
    leader: bool,
    help: Option<HelpKind>,

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
            album_dir: None,
            album_meta: hjson::AlbumMeta::default(),
            album_paths: Vec::new(),
            view: Vec::new(),
            filter: String::new(),
            filter_active: false,
            album_cursor: 0,
            selected: HashSet::new(),
            mode: AlbumMode::Grid,
            show_exif: false,
            show_analysis: false,
            analysis: None,
            view_proto: None,
            view_exif: None,
            thumbs: HashMap::new(),
            cols: 4,
            thumb_px,
            smart_albums,
            smart: None,
            smart_query: String::new(),
            smart_is_search: false,
            smart_src: HashMap::new(),
            smart_rec: HashMap::new(),
            edit_menu: false,
            ml_menu: false,
            ai_menu: false,
            jobs: VecDeque::new(),
            pending_visual: None,
            clip_cache: HashMap::new(),
            pending_nl: None,
            pending_plan: None,
            compare: Vec::new(),
            compare_cursor: 0,
            stack_view: false,
            timeline: false,
            tl_buckets: Vec::new(),
            tl_cursor: 0,
            leader: false,
            help: None,
            cmd_active: false,
            cmd_prompt: String::new(),
            cmd_buffer: String::new(),
            pending: None,
            picker,
            status: String::from("↑/↓ move · →/Enter open · h collapse · Tab pane · q quit"),
            watch: None,
            dirty_since: None,
        }
    }

    /// Re-walk the library and reload the open album (after a filesystem change). Keeps the album
    /// cursor in range; new thumbnails decode lazily on the next ticks.
    fn rescan(&mut self) {
        if let Ok(root) = library::walk(&self.root_dir) {
            self.root = root;
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
        // Library-wide smart albums come first, as ★ rows at depth 0 (sentinel paths — never
        // touched on disk; the tree handler routes them by name).
        for sa in &self.smart_albums {
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
        fn rec(node: &LibraryNode, depth: usize, exp: &HashSet<PathBuf>, out: &mut Vec<Row>) {
            let is_open = exp.contains(&node.path);
            out.push(Row {
                path: node.path.clone(),
                name: node.name.clone(),
                kind: node.kind,
                count: node.total_images(),
                depth,
                expanded: is_open,
                has_children: !node.children.is_empty(),
            });
            if is_open {
                for c in &node.children {
                    rec(c, depth + 1, exp, out);
                }
            }
        }
        rec(&self.root, 0, &self.expanded, &mut out);
        out
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

    /// Mutate each target image's record, then persist. In a smart-album view each write routes to
    /// the image's own source album; otherwise all targets share the open album (one write).
    fn edit_targets(&mut self, mut f: impl FnMut(&mut hjson::ImageRecord)) {
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

    /// Append a pixel edit to the cursor image: back up the pristine original (once), record the op
    /// in `album.hjson`, and re-derive the visible file from original + the full edit list.
    fn apply_edit(&mut self, op: edit::EditOp) {
        let Some((dir, filename)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        if let Err(e) = edit::ensure_backup(&dir, &filename) {
            self.status = format!("edit failed: {e:#}");
            return;
        }
        self.edit_record_at(&path, |rec| rec.edits.push(op.to_entry()));
        let ops = self.cur_edit_ops();
        match edit::rebuild_file(&dir, &filename, &ops) {
            Ok(()) => {
                self.status = format!("{} · {} edit(s) · u undo · 0 revert", op.label(), ops.len());
                self.refresh_after_edit(&path);
            }
            Err(e) => self.status = format!("edit failed: {e:#}"),
        }
    }

    /// Undo the cursor image's last edit (rebuild from the remaining ops; restores the original when
    /// none remain).
    fn undo_edit(&mut self) {
        let Some((dir, filename)) = self.cur_source() else { return };
        let Some(path) = self.cur_idx().and_then(|i| self.album_paths.get(i).cloned()) else { return };
        if self.cur_edit_ops().is_empty() {
            self.status = "nothing to undo".into();
            return;
        }
        self.edit_record_at(&path, |rec| {
            rec.edits.pop();
        });
        let ops = self.cur_edit_ops();
        if let Err(e) = edit::rebuild_file(&dir, &filename, &ops) {
            self.status = format!("undo failed: {e:#}");
            return;
        }
        self.status = format!("undo · {} edit(s) remain", ops.len());
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

    /// Queue a T2 ML edit on the cursor image (the event loop runs it with the TUI suspended).
    fn queue_ml(&mut self, op: mledit::MlOp) {
        self.ml_menu = false;
        let Some((album, filename)) = self.cur_source() else {
            self.status = "open an album first".into();
            return;
        };
        let input = album.join(&filename);
        self.status = format!("running {} … (the UI will pause)", op.label());
        self.jobs.push_back(Job::Ml(mledit::MlJob { op, input, album }));
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
            Action::Edit { op } => match edit::EditOp::from_tag(&op) {
                Some(eop) => self.batch_edit(eop),
                None => self.status = format!("unknown edit op: {op}"),
            },
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
            self.edit_record_at(&p, |r| r.edits.push(op.to_entry()));
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
        self.status = format!("queued {n} × {} (the UI will pause per image)", op.label());
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

    /// After a file's pixels change: drop its cached thumbnail (so it re-decodes) and reload the
    /// full-pane image if it's on screen.
    fn refresh_after_edit(&mut self, path: &Path) {
        self.thumbs.remove(path);
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
                    let val = (!arg.is_empty()).then(|| arg.clone());
                    self.edit_record_at(&path, |rec| match field {
                        EditField::Caption => rec.caption = val,
                        EditField::Notes => rec.notes = val,
                        EditField::Title => rec.title = val,
                        EditField::Tags => rec.tags = parse_tags(&arg),
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
    /// and (when the analysis panel is on) its histogram/exposure/focus stats.
    fn load_view(&mut self) {
        let path = self.cur_idx().and_then(|i| self.album_paths.get(i)).cloned();
        self.view_proto = path
            .as_ref()
            .and_then(|p| loader::thumbnail(p, 1600).ok())
            .map(|img| self.picker.new_resize_protocol(img));
        self.view_exif = path.as_ref().and_then(|p| exif::read_exif(p).ok());
        self.analysis = None;
        if self.show_analysis {
            self.compute_analysis();
        }
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
    // Live watch (best-effort — the manager still works statically if it can't be started).
    app.watch = watcher::spawn(&root_dir).ok();

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
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
        // Drain one heavy job per tick (ML edits suspend the TUI; vision is a quick blocking call).
        // A batch pipeline (`:`) enqueues many — they run in sequence, redrawing between each.
        if let Some(job) = app.jobs.pop_front() {
            match job {
                Job::Ml(j) => run_ml_job(terminal, app, j)?,
                Job::Vision(op, path) => run_vision_job(terminal, app, op, path)?,
            }
            continue;
        }
        // A queued CLIP visual search (Phase 7): heavy (model load + embed), run TUI-suspended.
        if let Some(query) = app.pending_visual.take() {
            run_visual_search(terminal, app, query)?;
            continue;
        }
        // A natural-language command the deterministic parser couldn't handle → LLM planner.
        if let Some(text) = app.pending_nl.take() {
            run_nl_planner(terminal, app, text)?;
            continue;
        }
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

/// Run a queued T2 ML edit with the TUI suspended, then resume. The job runs on a dedicated thread
/// with its own runtime (avoiding a `block_on` on the async event-loop thread); the `join` blocks
/// here on purpose while the manager is paused and the pipeline's progress shows on the real screen.
fn run_ml_job(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
    job: mledit::MlJob,
) -> Result<()> {
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    println!("\n▶ {}\n   {}\n", job.op.label(), job.input.display());
    let label = job.op.label();
    let src = job.input.clone();
    let result = std::thread::spawn(move || -> Result<PathBuf> {
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
        rt.block_on(job.run())
    })
    .join()
    .unwrap_or_else(|_| Err(anyhow::anyhow!("ML edit thread panicked")));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()?;
    match result {
        Ok(out) => {
            let name = out.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            app.record_variant(&src, &name); // link the derivative before the rescan
            app.rescan();
            app.select_by_name(&name);
            app.status = format!("✓ {label} → {name}");
        }
        Err(e) => app.status = format!("✗ ML edit failed: {e:#}"),
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

/// Returns true to quit.
fn handle_key(app: &mut App, k: crossterm::event::KeyEvent) -> bool {
    use crossterm::event::KeyModifiers;
    // A quickhelp overlay swallows the next key (dismiss).
    if app.help.is_some() {
        app.help = None;
        return false;
    }
    // `Ctrl-B` leader: the next key is a leader command (h = chords help, H = commands help).
    if app.leader {
        app.leader = false;
        match k.code {
            KeyCode::Char('h') => app.help = Some(HelpKind::Chords),
            KeyCode::Char('H') => app.help = Some(HelpKind::Commands),
            _ => {}
        }
        return false;
    }
    if k.modifiers.contains(KeyModifiers::CONTROL) && matches!(k.code, KeyCode::Char('b')) {
        app.leader = true;
        app.status = "Ctrl-B · h: key chords · H: commands".into();
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
fn handle_edit_key(app: &mut App, code: KeyCode) {
    use edit::EditOp;
    match code {
        KeyCode::Esc | KeyCode::Char('E') | KeyCode::Char('q') => app.edit_menu = false,
        KeyCode::Char('r') => app.apply_edit(EditOp::RotateCw),
        KeyCode::Char('R') => app.apply_edit(EditOp::RotateCcw),
        KeyCode::Char('t') => app.apply_edit(EditOp::Rotate180),
        KeyCode::Char('h') => app.apply_edit(EditOp::FlipH),
        KeyCode::Char('v') => app.apply_edit(EditOp::FlipV),
        KeyCode::Char('g') => app.apply_edit(EditOp::Grayscale),
        KeyCode::Char('s') => app.apply_edit(EditOp::CropSquare),
        KeyCode::Char('+') | KeyCode::Char('=') => app.apply_edit(EditOp::Brightness(15)),
        KeyCode::Char('-') | KeyCode::Char('_') => app.apply_edit(EditOp::Brightness(-15)),
        KeyCode::Char('>') | KeyCode::Char('.') => app.apply_edit(EditOp::Contrast(12)),
        KeyCode::Char('<') | KeyCode::Char(',') => app.apply_edit(EditOp::Contrast(-12)),
        KeyCode::Char('u') => app.undo_edit(),
        KeyCode::Char('0') => app.revert_edits(),
        _ => {}
    }
}

/// Vision + AI menu (Phase 7): describe/tag the cursor image with Gemini vision.
fn handle_ai_key(app: &mut App, code: KeyCode) {
    match code {
        KeyCode::Esc | KeyCode::Char('A') | KeyCode::Char('q') => app.ai_menu = false,
        KeyCode::Char('t') => app.queue_vision(vision::VisionOp::Autotag),
        KeyCode::Char('d') => app.queue_vision(vision::VisionOp::Describe),
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
        _ => {}
    }
}

/// Filter-bar input: type a filter expression, Enter applies, Esc clears.
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
        KeyCode::Char('g') => app.tree_cursor = 0,
        KeyCode::Char('G') => app.tree_cursor = rows.len().saturating_sub(1),
        KeyCode::Char('l') | KeyCode::Right | KeyCode::Enter => {
            if let Some((path, kind, has_children, name)) = cur {
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
        KeyCode::Char('h') | KeyCode::Left => {
            if let Some((path, ..)) = cur {
                app.expanded.remove(&path);
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
        KeyCode::Char('a') => {
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
        KeyCode::Char('D') => {
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
                app.edit_menu = true;
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
        KeyCode::Char('i') => app.show_exif = !app.show_exif,
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
                app.load_view();
            }
        }
        KeyCode::Char('h') | KeyCode::Left => {
            if app.album_cursor > 0 {
                app.album_cursor -= 1;
                app.load_view();
            }
        }
        KeyCode::Char('i') => app.show_exif = !app.show_exif,
        KeyCode::Char('H') => {
            app.show_analysis = !app.show_analysis;
            if app.show_analysis {
                app.compute_analysis();
            }
        }
        KeyCode::Char('E') => app.edit_menu = true, // open the pixel-edit menu
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
        _ => {}
    }
    false
}

fn draw(f: &mut Frame, app: &mut App) {
    let [status_bar, body, cmd_pane] =
        Layout::vertical([Constraint::Length(1), Constraint::Fill(1), Constraint::Length(3)])
            .areas(f.area());

    let status = format!(
        " plakat photos  {}   {} albums · {} images   {}",
        app.root_dir.display(),
        app.album_count(),
        app.root.total_images(),
        app.status,
    );
    f.render_widget(Paragraph::new(status).style(Style::default().fg(Color::Black).bg(Color::Gray)), status_bar);

    let [tree_col, album_col] =
        Layout::horizontal([Constraint::Percentage(30), Constraint::Fill(1)]).areas(body);

    draw_tree(f, app, tree_col);
    draw_album(f, app, album_col);
    if app.timeline {
        draw_timeline(f, app, album_col);
    }
    if let Some(kind) = app.help {
        draw_help(f, kind, app, body);
    }

    // Command pane: edit/ML menu, active text input, or a passive hint.
    let (cmd, cmd_style) = if app.edit_menu {
        (
            " EDIT  r/R rotate · t 180° · h/v flip · g gray · s crop1:1 · -/+ bright · </> contrast · u undo · 0 revert · Esc"
                .to_string(),
            Style::default().fg(Color::Cyan),
        )
    } else if app.ml_menu {
        (
            " ML EDIT  u upscale ×4 · i img2img (prompt) · l relight (prompt) · Esc   — runs a model; the UI pauses"
                .to_string(),
            Style::default().fg(Color::Magenta),
        )
    } else if app.ai_menu {
        (
            " AI vision  t autotag · d describe · Esc   — uses your configured LLM (Gemini / DeepSeek)"
                .to_string(),
            Style::default().fg(Color::Green),
        )
    } else if app.cmd_active {
        (format!(" {}{}_", app.cmd_prompt, app.cmd_buffer), Style::default().fg(Color::Yellow))
    } else {
        (" CMD ▶ ".to_string(), Style::default())
    };
    f.render_widget(
        Paragraph::new(cmd).style(cmd_style).block(Block::default().borders(Borders::ALL)),
        cmd_pane,
    );
}

fn draw_tree(f: &mut Frame, app: &App, area: Rect) {
    let active = app.focus == Focus::Tree;
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
                .title(" Library ")
                .border_style(Style::default().fg(if active { Color::Yellow } else { Color::DarkGray })),
        ),
        area,
    );
}

fn draw_album(f: &mut Frame, app: &mut App, area: Rect) {
    let active = app.focus == Focus::Album;
    let sort = app.album_meta.sort.as_deref().unwrap_or("name-asc");
    let title = match (&app.smart, &app.album_dir) {
        (Some(name), _) if app.smart_is_search => format!(" 🔎 {name}  ·  ↕ {sort} "),
        (Some(name), _) => format!(" ★ {name}  ·  ↕ {sort} "),
        (None, Some(d)) => format!(
            " {}  ·  ↕ {} ",
            d.file_name().and_then(|n| n.to_str()).unwrap_or("album"),
            sort,
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
            let name: String = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
            let cap: String = {
                let cs: Vec<char> = name.chars().collect();
                if cs.len() > 14 { cs[cs.len() - 14..].iter().collect() } else { name.clone() }
            };
            let border = if is_cursor {
                Color::Cyan
            } else if is_sel {
                Color::Green
            } else {
                Color::DarkGray
            };
            let cb = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(border))
                .title(if is_cursor { format!("▶{cap}") } else { cap });
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
        _ => name.to_lowercase().contains(&tok.to_lowercase()),
    }
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
}

fn help_ctx(app: &App) -> HelpCtx {
    if app.focus == Focus::Tree {
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
            l.push(kv("j k / ↑↓", "move · g/G first/last"));
            l.push(kv("l → Enter", "open album / expand folder · h collapse"));
            l.push(kv("n a R D", "new folder · new album · rename · delete"));
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
            l.push(hd("Select"));
            l.push(kv("Space", "toggle · Ctrl-A all · Ctrl-D none · Ctrl-I invert"));
            l.push(hd("Do"));
            l.push(kv("/ C =", "filter · cull · compare selection"));
            l.push(kv("E M A", "edit · ML-edit · AI-vision menus"));
            l.push(kv("# X r", "duplicates · export · batch-rename"));
            l.push(kv("S @ F", "stack · timeline · save smart album"));
            l.push(kv("? V", "search metadata · visual (CLIP)"));
        }
        HelpCtx::Image => {
            l.push(hd("Image view"));
            l.push(kv("← →", "previous / next · Esc back to grid"));
            l.push(kv("i H", format!(
                "EXIF panel · analysis panel{}",
                if app.show_analysis { " (on)" } else { "" }
            )));
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
    }
    l.push(hd("Anywhere"));
    l.push(kv("Ctrl-B", "leader (h chords · H commands) · q quit"));
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
        HelpCtx::Grid | HelpCtx::Image | HelpCtx::Cull | HelpCtx::Compare => {
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
            l.push(hd("Produce"));
            l.push(kv("upscale", "ML ×4 (M → u)"));
            l.push(kv("img2img", "transform / relight with a prompt (M → i/l)"));
            l.push(kv("edit", "rotate/flip/crop/bright/contrast (E)"));
            l.push(kv("export", format!("copy {target} out, optional resize (X)")));
            l.push(kv("rename", "batch-rename with a #-pattern (r)"));
            l.push(hd("Natural language  (:)"));
            l.push(kv(":", "album-scoped command — pipe with 'then'"));
            l.push(state("find rating>=4 then tag as portfolio then upscale"));
            l.push(state("all photos then autotag   ·   take flag then rate 5"));
        }
    }
    l
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
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_image_view(f: &mut Frame, app: &mut App, area: Rect) {
    let (img_area, panel) = if app.show_exif || app.show_analysis {
        let [a, b] = Layout::horizontal([Constraint::Percentage(60), Constraint::Fill(1)]).areas(area);
        (a, Some(b))
    } else {
        (area, None)
    };

    match app.view_proto.as_mut() {
        Some(proto) => f.render_stateful_widget(StatefulImage::new(), img_area, proto),
        None => f.render_widget(Paragraph::new("  decoding…").style(Style::new().fg(Color::DarkGray)), img_area),
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
        if let Some(e) = &app.view_exif {
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
        if let Some(r) = path.as_ref().and_then(|p| app.record(p)) {
            lines.push(Line::from("─ curation ─"));
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
                let ops: Vec<&str> =
                    r.edits.iter().filter_map(edit::EditOp::from_entry).map(|o| o.label()).collect();
                lines.push(Line::from(Span::styled(
                    format!("edits    {} ({})", r.edits.len(), ops.join(", ")),
                    Style::new().fg(Color::Cyan),
                )));
            }
            // Generation recipe for plakat-made images (`--import`).
            if let Some(g) = &r.generation {
                lines.push(Line::from("─ plakat ─"));
                lines.push(Line::from(format!("model    {}", g.model)));
                lines.push(Line::from(format!("seed     {}", g.seed)));
                lines.push(Line::from(format!("steps    {}  cfg {}", g.steps, g.guidance)));
            }
        }
        f.render_widget(
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(" EXIF ")),
            panel,
        );
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
