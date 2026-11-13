//! GridTalk egui shell: the spreadsheet grid and the lifted graph,
//! with two-way selection sync, beside a full-height chat column. This is
//! the Phase 2/4 front end; it renders GridTalk snapshots directly
//! (the Gridcraft `ui-egui` ribbon remains the long-term editor
//! surface, see docs/decisions/0001).

mod eval_worker;
mod find;
mod history;
mod layout;

use egui::{Key, KeyboardShortcut, Modifiers};
use eval_worker::{Lookup, ValueCache};
use gridtalk_adapter::{
    adapters_root, cap_prompt, list_adapters, load_adapter, prompt_text, sidecar_rules,
    valid_adapter_name, write_atomic_bytes,
};
use gridtalk_graph::{
    build_graph_cached, load_path, CellEdit, CellValue, Graph, GraphCache, IncrementalSnapshot,
    NodeKind, RangeRef,
};
use gridtalk_index::HybridIndex as Index;
use gridtalk_infer::config::Config;
use gridtalk_infer::{backend_from_config, cleartext_url_warning, cleartext_warning, ChatRequest};
use history::History;
use layout::{spawn_layout, LayoutWorker, Pos};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ViewMode {
    #[default]
    Split,
    GridFull,
    GraphFull,
}

impl ViewMode {
    pub fn label(&self) -> &'static str {
        match self {
            ViewMode::Split => "Split",
            ViewMode::GridFull => "Grid",
            ViewMode::GraphFull => "Graph",
        }
    }
}

/// Size of a graph node button.
const GRAPH_NODE_SIZE: egui::Vec2 = egui::vec2(120.0, 18.0);
/// Initial and minimum width of the chat column on the right.
const CHAT_PANEL_DEFAULT_WIDTH: f32 = 380.0;
const CHAT_PANEL_MIN_WIDTH: f32 = 240.0;
/// The data side (grid, graph) never narrows below this, however
/// wide chat gets.
const DATA_MIN_WIDTH: f32 = 320.0;
/// Minimum height of the graph under the grid in Split view.
const GRAPH_PANEL_MIN_HEIGHT: f32 = 120.0;
/// The grid never shrinks below this, however tall the graph gets.
const GRID_MIN_HEIGHT: f32 = 120.0;
/// The whole UI's scale: where it starts, its bounds, and one step of
/// Ctrl+= / Ctrl+-.
pub const DEFAULT_ZOOM: f32 = 1.25;
const MIN_ZOOM: f32 = 0.5;
const MAX_ZOOM: f32 = 3.0;
const ZOOM_STEP: f32 = 0.1;
/// The chat input's id, so Ctrl+L can focus it.
const CHAT_INPUT_ID: &str = "chat-input";

/// `zoom` rounded to a 5% step and kept within bounds; anything that is
/// not a number is the default.
pub fn clamp_zoom(zoom: f32) -> f32 {
    if !zoom.is_finite() {
        return DEFAULT_ZOOM;
    }
    ((zoom * 20.0).round() / 20.0).clamp(MIN_ZOOM, MAX_ZOOM)
}
/// Bounds for the llama.cpp settings in the Settings window.
const LLM_MAX_GPU_LAYERS: u32 = 999;
const LLM_MIN_N_CTX: u32 = 512;
const LLM_MAX_N_CTX: u32 = 65_536;
const LLM_DEFAULT_N_CTX: u32 = 8_192;

/// Per-workbook UI preferences, persisted as JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UiPrefs {
    pub mode_by_workbook: BTreeMap<String, ViewMode>,
    #[serde(default)]
    pub first_run_done: bool,
    #[serde(default)]
    pub config_path: String,
    /// Active adapter name ("" = none).
    #[serde(default)]
    pub active_adapter: String,
    /// Show formula text in formula cells instead of their results.
    #[serde(default)]
    pub show_formulas: bool,
    /// Read-only mode: cells cannot be edited and saving is off.
    #[serde(default)]
    pub read_only: bool,
    /// The UI scale; `None` = [`DEFAULT_ZOOM`].
    #[serde(default)]
    pub zoom: Option<f32>,
    /// Where `save` writes, when not the usual [`UiPrefs::path`]
    /// (tests use a scratch file).
    #[serde(skip)]
    pub file: Option<std::path::PathBuf>,
}

/// GridTalk's config directory: `$GRIDTALK_CONFIG_DIR` when set,
/// otherwise the platform config dir (e.g. `%APPDATA%` on Windows,
/// `~/.config` on Linux) with a `gridtalk` component.
pub fn config_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("GRIDTALK_CONFIG_DIR") {
        if !d.is_empty() {
            return std::path::PathBuf::from(d);
        }
    }
    if let Some(d) = dirs::config_dir() {
        return d.join("gridtalk");
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => std::path::PathBuf::from(home).join(".config/gridtalk"),
        _ => std::path::PathBuf::from(".config/gridtalk"),
    }
}

/// Mirror of gridtalk-app's `gridcraft_binary` resolution.
fn gridcraft_binary_ui() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIDTALK_GRIDCRAFT_BIN") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let exe_name = if cfg!(windows) {
        "gridcraft.exe"
    } else {
        "gridcraft"
    };
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for profile in ["release", "debug"] {
        let candidate = root.join(format!("vendor/gridcraft/target/{profile}/{exe_name}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Parsed-value equality for edit-buffer pruning: a formula buffer
/// matches the stored cell when the expression matches, ignoring
/// the stored cached value.
fn edit_buffer_matches(
    current: &gridtalk_graph::CellValue,
    parsed: &gridtalk_graph::CellValue,
) -> bool {
    use gridtalk_graph::CellValue;
    match (current, parsed) {
        (CellValue::Formula { expr: a, .. }, CellValue::Formula { expr: b, .. }) => {
            a.trim() == b.trim()
        }
        _ => current == parsed,
    }
}

impl UiPrefs {
    /// The saved UI scale, or the default.
    pub fn zoom(&self) -> f32 {
        clamp_zoom(self.zoom.unwrap_or(DEFAULT_ZOOM))
    }
    pub fn path() -> std::path::PathBuf {
        config_dir().join("ui-state.json")
    }
    pub fn load() -> Self {
        match std::fs::read_to_string(Self::path()) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                // Salvage valid keys instead of discarding every
                // preference over one bad value.
                eprintln!("gridtalk-ui: warning: bad ui-state.json ({e}); salvaging valid keys");
                let value: serde_json::Value =
                    serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
                let mut prefs = UiPrefs::default();
                if let Some(map) = value.get("mode_by_workbook") {
                    if let Ok(m) = serde_json::from_value(map.clone()) {
                        prefs.mode_by_workbook = m;
                    }
                }
                prefs.first_run_done = value
                    .get("first_run_done")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                prefs.config_path = value
                    .get("config_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                prefs.active_adapter = value
                    .get("active_adapter")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                prefs.show_formulas = value
                    .get("show_formulas")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                prefs.read_only = value
                    .get("read_only")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                prefs.zoom = value.get("zoom").and_then(|v| v.as_f64()).map(|z| z as f32);
                prefs
            }),
            Err(_) => UiPrefs::default(),
        }
    }
    pub fn save(&self) {
        let p = self.file.clone().unwrap_or_else(Self::path);
        if let Some(parent) = p.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                eprintln!(
                    "gridtalk-ui: warning: cannot create {}: {e}",
                    parent.display()
                );
                return;
            }
        }
        match serde_json::to_string_pretty(self) {
            Ok(text) => {
                if let Err(e) = write_atomic_bytes(&p, text.as_bytes()) {
                    eprintln!("gridtalk-ui: warning: cannot save {}: {e}", p.display());
                }
            }
            Err(e) => eprintln!("gridtalk-ui: warning: cannot serialise prefs: {e}"),
        }
    }
}

/// Two-way sync state: the single selection both panes honour.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SyncState {
    pub selected_range: Option<RangeRef>,
    pub highlighted_node: Option<usize>,
    pub mode: ViewMode,
}

impl SyncState {
    pub fn select_node(&mut self, graph: &Graph, node_id: usize) {
        self.highlighted_node = Some(node_id);
        if let Some(n) = graph.nodes.iter().find(|n| n.id == node_id) {
            self.selected_range = n.range.clone();
        }
    }
    /// Select a whole range, highlighting the node of its first cell.
    pub fn select_range(&mut self, graph: &Graph, range: &RangeRef) {
        self.highlighted_node = graph
            .node_for_cell(&range.sheet, range.r0, range.c0)
            .map(|n| n.id);
        self.selected_range = Some(range.clone());
    }
    pub fn select_cell(&mut self, graph: &Graph, sheet: &str, row: usize, col: usize) {
        self.selected_range = Some(RangeRef {
            sheet: sheet.into(),
            r0: row,
            c0: col,
            r1: row,
            c1: col,
        });
        self.highlighted_node = graph.node_for_cell(sheet, row, col).map(|n| n.id);
    }
}

/// A graph rebuild running on a worker thread.
struct RebuildJob {
    revision: u64,
    /// Whether the job updates an existing index (else it builds one
    /// from scratch, and a failure is not retried).
    incremental: bool,
    rx: std::sync::mpsc::Receiver<RebuildResult>,
}

/// A rebuilt graph, its invariant violations, the index, and the
/// graph cache for the next rebuild.
type RebuildResult = (Graph, Vec<String>, Option<Index>, GraphCache);

/// Build the graph for `snapshot` and bring `index` up to date with
/// it: incrementally from the edited `dirty` ranges when there is an
/// index to update and a cache to reuse, else from scratch. A graph
/// with invariant violations gets no index (never serve a stale one).
fn compute_rebuild(
    snapshot: &gridtalk_graph::WorkbookSnapshot,
    index: Option<Index>,
    mut cache: GraphCache,
    dirty: Option<&[RangeRef]>,
) -> RebuildResult {
    let (graph, violations) = build_graph_cached(snapshot, &mut cache, dirty);
    if !violations.is_empty() {
        return (graph, violations, None, cache);
    }
    let index = match (index, dirty) {
        (Some(mut index), Some(dirty)) => {
            index.update_dirty(snapshot, &graph, dirty);
            index
        }
        _ => Index::build(snapshot, &graph),
    };
    (graph, violations, Some(index), cache)
}

/// The node a selection highlights: the node whose range it is (a
/// graph-side pick), else the most specific node holding its cell.
fn node_for_selection(graph: &Graph, sel: &RangeRef) -> Option<usize> {
    graph
        .nodes
        .iter()
        .find(|n| n.range.as_ref() == Some(sel))
        .or_else(|| graph.node_for_cell(&sel.sheet, sel.r0, sel.c0))
        .map(|n| n.id)
}

struct ChatMsg {
    role: String,
    text: String,
    /// The ranges an answer was given as context (its "Sources:").
    sources: Vec<RangeRef>,
}

/// The conversation so far as question/answer turns, oldest first:
/// each "you" message paired with the reply that followed it. A
/// question that never got a reply (still waiting, or followed
/// straight by another question) pairs with nothing and is left
/// out, rather than handed to the model with a blank answer.
fn chat_history(chat: &[ChatMsg]) -> Vec<gridtalk_index::ChatTurn> {
    let mut turns = Vec::new();
    let mut pending: Option<String> = None;
    for msg in chat {
        if msg.role == "you" {
            pending = Some(msg.text.clone());
        } else if let Some(q) = pending.take() {
            turns.push(gridtalk_index::ChatTurn {
                question: q,
                answer: msg.text.clone(),
            });
        }
    }
    turns
}

/// What a question about the selection stands on: the selected
/// node's graph neighbours (what it reads, what reads it) and
/// the selected cells themselves, bounded so a whole-column
/// selection cannot flood the prompt.
fn selection_context(
    graph: &Graph,
    wb: &gridtalk_graph::WorkbookSnapshot,
    sel: &RangeRef,
) -> String {
    let mut out = format!("selected {}:\n", sel.a1());
    let header_len = out.len();
    // The node the whole selection sits in (as the `expand` tool
    // picks it): its exact node, else the smallest region or
    // column covering it, not merely the one holding its first
    // cell, so B1:C3 reads the region rather than column B alone.
    let covering = graph
        .nodes
        .iter()
        .find(|n| n.range.as_ref() == Some(sel))
        .or_else(|| {
            graph
                .nodes
                .iter()
                .filter(|n| matches!(n.kind, NodeKind::Region | NodeKind::Column))
                .filter(|n| {
                    n.range.as_ref().is_some_and(|r| {
                        gridtalk_graph::sheet_name_eq(&r.sheet, &sel.sheet)
                            && r.contains(sel.r0, sel.c0)
                            && r.contains(sel.r1, sel.c1)
                    })
                })
                .min_by_key(|n| {
                    n.range
                        .as_ref()
                        .map_or(u128::MAX, gridtalk_graph::range_area)
                })
        })
        .map(|n| n.id);
    if let Some(id) = covering {
        if let Some(node) = graph.nodes.iter().find(|n| n.id == id) {
            out.push_str(&format!("node {} ({})\n", node.id, node.cited_label()));
            let mut shown = 0usize;
            for e in &graph.edges {
                let (other, arrow) = if e.from == id {
                    (e.to, "->")
                } else if e.to == id {
                    (e.from, "<-")
                } else {
                    continue;
                };
                if shown >= 50 {
                    out.push_str(" ... (more edges)\n");
                    break;
                }
                if let Some(t) = graph.nodes.iter().find(|n| n.id == other) {
                    out.push_str(&format!(" {}{} {}\n", arrow, e.kind, t.cited_label()));
                    shown += 1;
                }
            }
        }
    }
    if let Some(sheet) = wb
        .sheets
        .iter()
        .find(|s| gridtalk_graph::sheet_name_eq(&s.name, &sel.sheet))
    {
        out.push_str("cells:\n");
        let mut cells = 0usize;
        let last_row = sel.r1.min(sel.r0 + 99);
        let mut cut = false;
        'rows: for r in sel.r0..=last_row {
            let mut line = Vec::new();
            for c in sel.c0..=sel.c1 {
                if cells >= 300 {
                    // Keep the part of the row already read.
                    if !line.is_empty() {
                        out.push_str(&line.join(" | "));
                        out.push('\n');
                    }
                    cut = true;
                    break 'rows;
                }
                line.push(sheet.get(r, c).display());
                cells += 1;
            }
            out.push_str(&line.join(" | "));
            out.push('\n');
        }
        if cut || last_row < sel.r1 {
            // Say so, so a cut selection never reads as complete.
            out.push_str("... (selection cut at 100 rows / 300 cells)\n");
        }
    }
    // Nothing behind the header (no node, no such sheet): empty,
    // so the caller treats it as no context at all.
    if out.len() == header_len {
        return String::new();
    }
    out
}

pub struct GridTalkApp {
    inc: Option<IncrementalSnapshot>,
    graph: Graph,
    index: Option<Index>,
    sync: SyncState,
    prefs: UiPrefs,
    workbook_name: String,
    positions: BTreeMap<usize, Pos>,
    layout_worker: Option<LayoutWorker>,
    /// The graph and index rebuild after an edit, running off the UI
    /// thread (on a large workbook it takes a noticeable fraction of
    /// a second). The index is moved into it and comes back updated.
    rebuild_job: Option<RebuildJob>,
    /// Ranges edited since the index was last updated.
    pending_dirty: Vec<RangeRef>,
    /// Edits landed that no rebuild has picked up yet.
    rebuild_wanted: bool,
    /// Region and formula-precedent work kept between graph rebuilds
    /// (travels with a rebuild job, like the index).
    graph_cache: GraphCache,
    layout_revision: u64,
    chat_input: String,
    chat: Vec<ChatMsg>,
    status: String,
    show_settings: bool,
    /// llama.cpp settings as edited in the Settings window; `None` =
    /// the backend default (all layers / the model's context).
    llm_gpu_layers: Option<u32>,
    llm_n_ctx: Option<u32>,
    /// The cleartext warning for the config in use, shown in Settings.
    llm_cleartext: Option<String>,
    /// Whether the two fields above reflect the config file yet
    /// (re-read each time Settings opens).
    llm_settings_loaded: bool,
    config_path: String,
    wizard_step: u8,
    wizard_backend: String,
    wizard_model_path: String,
    wizard_base_url: String,
    current_path: Option<std::path::PathBuf>,
    saved_revision: u64,
    selected_sheet: usize,
    edit_buffers: BTreeMap<(usize, usize, usize), String>,
    frame_sheet: Option<gridtalk_graph::SheetSnapshot>,
    frame_sheet_idx: usize,
    /// Computed formula results, evaluated on a worker thread.
    values: ValueCache,
    /// Read-only forced by `--read-only`: the toggle is disabled.
    read_only_locked: bool,
    frame_revision: u64,
    scrolled_for: Option<String>,
    show_adapters: bool,
    adapter_cache: Vec<(String, String)>,
    adapter_card_cache: String,
    active_adapter: String,
    layout_size: (f32, f32),
    chat_cancel: Option<Arc<AtomicBool>>,
    chat_workbook: String,
    pending_open: Option<std::path::PathBuf>,
    close_armed: bool,
    graph_version: u64,
    chat_rx: Option<std::sync::mpsc::Receiver<(String, String, String)>>,
    chat_busy: bool,
    chat_started: Option<std::time::Instant>,
    /// The context ranges of the answer being generated.
    chat_sources: Vec<RangeRef>,
    /// The range the last ask() selected itself (its top hit). While
    /// it is still the selection the user has not pointed at it, so
    /// "is it right?" does not read it as the user's selection.
    ask_selected: Option<RangeRef>,
    /// Chat messages removed to keep the log short (shown as a note).
    chat_dropped: usize,
    /// Grid edits that undo and redo walk through.
    history: History,
    /// The grid cell editor that had the keyboard last frame, if any.
    grid_editor_focus: Option<egui::Id>,
    /// The find bar (Ctrl+F): open, its query, the matches for that
    /// query at `find_revision`, and the match shown.
    find_open: bool,
    find_focus: bool,
    /// Focus the chat input next frame (Ctrl+L).
    chat_focus: bool,
    find_query: String,
    find_matches: Vec<find::Match>,
    find_for: Option<(String, u64)>,
    find_pos: usize,
    /// The backend as edited in Settings (loaded with the model
    /// settings).
    set_backend: String,
    set_model_path: String,
    set_base_url: String,
    set_model: String,
    set_api_key_env: String,
}

impl Default for GridTalkApp {
    fn default() -> Self {
        GridTalkApp::with_prefs(UiPrefs::load())
    }
}

impl GridTalkApp {
    /// Build the app from explicit preferences. `Default` loads the
    /// user's saved prefs; tests pass their own so they never depend
    /// on (or write to) the developer's real `ui-state.json`.
    pub fn with_prefs(mut prefs: UiPrefs) -> Self {
        // GRIDTALK_CONFIG beats the persisted prefs value (the CLI
        // checks the env first; a wizard-saved path must not
        // silently shadow it).
        let config_path = match std::env::var("GRIDTALK_CONFIG") {
            Ok(p) if !p.is_empty() => p,
            _ if !prefs.config_path.is_empty() => prefs.config_path.clone(),
            _ => String::new(),
        };
        let saved_adapter = prefs.active_adapter.clone();
        // A deleted/renamed adapter must not stay silently active,
        // and a hand-edited prefs value must never escape the
        // adapters root (path traversal via prefs).
        let saved_adapter = if !saved_adapter.is_empty()
            && (!valid_adapter_name(&saved_adapter)
                || !adapters_root().join(&saved_adapter).is_dir())
        {
            eprintln!(
                "gridtalk-ui: warning: active adapter {saved_adapter:?} no longer exists; clearing it"
            );
            String::new()
        } else {
            saved_adapter
        };
        // Persist the sanitised value so a later prefs.save() cannot
        // resurrect the cleared (deleted/traversal) name.
        if prefs.active_adapter != saved_adapter {
            prefs.active_adapter = saved_adapter.clone();
            prefs.save();
        }
        GridTalkApp {
            inc: None,
            graph: Graph::default(),
            index: None,
            sync: SyncState::default(),
            prefs,
            workbook_name: String::new(),
            positions: BTreeMap::new(),
            layout_worker: None,
            layout_revision: u64::MAX,
            chat_input: String::new(),
            chat: Vec::new(),
            status: "Open a workbook (File > Open, or pass a path as argv[1]).".into(),
            show_settings: false,
            llm_gpu_layers: None,
            llm_n_ctx: None,
            llm_cleartext: None,
            llm_settings_loaded: false,
            config_path,
            show_adapters: false,
            adapter_cache: Vec::new(),
            adapter_card_cache: String::new(),
            active_adapter: saved_adapter,
            current_path: None,
            saved_revision: 0,
            selected_sheet: 0,
            edit_buffers: BTreeMap::new(),
            frame_sheet: None,
            frame_sheet_idx: usize::MAX,
            values: ValueCache::default(),
            rebuild_job: None,
            pending_dirty: Vec::new(),
            rebuild_wanted: false,
            graph_cache: GraphCache::default(),
            read_only_locked: false,
            frame_revision: u64::MAX,
            scrolled_for: None,
            layout_size: (0.0, 0.0),
            chat_cancel: None,
            chat_workbook: String::new(),
            pending_open: None,
            close_armed: false,
            graph_version: 0,
            chat_rx: None,
            chat_busy: false,
            chat_started: None,
            chat_sources: Vec::new(),
            ask_selected: None,
            chat_dropped: 0,
            history: History::default(),
            grid_editor_focus: None,
            find_open: false,
            find_focus: false,
            chat_focus: false,
            find_query: String::new(),
            find_matches: Vec::new(),
            find_for: None,
            find_pos: 0,
            set_backend: String::new(),
            set_model_path: String::new(),
            set_base_url: String::new(),
            set_model: String::new(),
            set_api_key_env: String::new(),
            wizard_step: 0,
            wizard_backend: "local".into(),
            wizard_model_path: String::new(),
            wizard_base_url: "http://127.0.0.1:11434/v1".into(),
        }
    }
}

/// The config file the platform default resolves to
/// (`<config dir>/config.toml`), as the CLI discovers it.
fn default_config_file() -> std::path::PathBuf {
    config_dir().join("config.toml")
}

/// The config file the app reads: the configured path, else the
/// platform default.
fn active_config_file(config_path: &str) -> std::path::PathBuf {
    if config_path.is_empty() {
        default_config_file()
    } else {
        std::path::PathBuf::from(config_path)
    }
}

/// A TOML basic string literal for `s`.
fn toml_string(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace(['\r', '\n'], " ");
    format!("\"{escaped}\"")
}

/// One `[inference]` key: `Some(TOML value text)` sets it, `None`
/// removes it.
type ConfigUpdate = (&'static str, Option<String>);

/// Apply `updates` to the `[inference]` table of a config file's
/// text. Everything else — other keys, other tables, comments — is
/// kept as written.
fn merge_inference_config(existing: &str, updates: &[ConfigUpdate]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut in_inference = false;
    let mut seen_section = false;
    let mut applied: Vec<bool> = vec![false; updates.len()];
    let flush = |out: &mut Vec<String>, applied: &mut Vec<bool>| {
        for (i, (k, v)) in updates.iter().enumerate() {
            if !applied[i] {
                if let Some(v) = v {
                    out.push(format!("{k} = {v}"));
                }
                applied[i] = true;
            }
        }
    };
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if in_inference {
                // Leaving [inference]: add the keys it did not have.
                flush(&mut out, &mut applied);
            }
            in_inference = trimmed.split('#').next().map(|h| {
                h.trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim()
            }) == Some("inference");
            seen_section |= in_inference;
        } else if in_inference {
            if let Some((key, _)) = trimmed.split_once('=') {
                let key = key.trim();
                if let Some(i) = updates.iter().position(|(k, _)| *k == key) {
                    if let (Some(v), false) = (&updates[i].1, applied[i]) {
                        out.push(format!("{key} = {v}"));
                    }
                    applied[i] = true;
                    continue;
                }
            }
        }
        out.push(line.to_string());
    }
    if !seen_section && updates.iter().any(|(_, v)| v.is_some()) {
        out.push("[inference]".into());
    }
    flush(&mut out, &mut applied);
    out.join("\n") + "\n"
}

/// Merge `updates` into the config file at `path` (created if
/// missing), written atomically.
fn write_inference_config(path: &std::path::Path, updates: &[ConfigUpdate]) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let existing = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        // Never replace a config we could not read with a fresh one.
        Err(e) => return Err(e),
    };
    let text = merge_inference_config(&existing, updates);
    write_atomic_bytes(path, text.as_bytes()).map_err(std::io::Error::other)
}

/// Write the wizard's choices to the GridTalk config file.
fn write_wizard_config(
    backend: &str,
    model_path: &str,
    base_url: &str,
) -> std::io::Result<std::path::PathBuf> {
    let path = default_config_file();
    let mut updates: Vec<ConfigUpdate> = vec![
        ("backend", Some(toml_string(backend))),
        ("engine", Some(toml_string("llamacpp"))),
    ];
    if !model_path.is_empty() {
        updates.push(("model_path", Some(toml_string(model_path))));
    }
    if backend == "openai-compatible" {
        updates.push(("base_url", Some(toml_string(base_url))));
    }
    write_inference_config(&path, &updates)?;
    Ok(path)
}

/// Validate wizard inputs before Finish may persist anything.
fn validate_wizard(backend: &str, model_path: &str, base_url: &str) -> Result<(), String> {
    match backend {
        "local" => {
            if model_path.is_empty() {
                return Err("choose a local .gguf model file first".into());
            }
            let path = std::path::Path::new(model_path);
            if !path.is_file() {
                return Err(format!("model file does not exist: {model_path}"));
            }
            if !model_path.to_ascii_lowercase().ends_with(".gguf") {
                return Err(format!("model file must be a .gguf file: {model_path}"));
            }
            Ok(())
        }
        "openai-compatible" => {
            if !base_url.starts_with("http://") && !base_url.starts_with("https://") {
                return Err(format!(
                    "base URL must start with http:// or https://: {base_url}"
                ));
            }
            Ok(())
        }
        other => Err(format!("unknown backend {other}")),
    }
}

/// Chat messages kept; older ones are dropped, and the log says so.
const MAX_CHAT: usize = 200;

impl GridTalkApp {
    fn push_chat(&mut self, role: &str, text: String) {
        self.push_chat_with_sources(role, text, Vec::new());
    }

    fn push_chat_with_sources(&mut self, role: &str, text: String, sources: Vec<RangeRef>) {
        self.chat.push(ChatMsg {
            role: role.into(),
            text,
            sources,
        });
        // Cap history so per-frame layout cost stays bounded; the log
        // says how many messages went.
        if self.chat.len() > MAX_CHAT {
            let drop = self.chat.len() - MAX_CHAT;
            self.chat.drain(..drop);
            self.chat_dropped += drop;
        }
    }

    fn open(&mut self, path: &std::path::Path) {
        // Never silently discard unsaved edits: the first Open click
        // arms the discard, the second (same path) confirms it.
        if self.dirty() && self.pending_open.as_deref() != Some(path) {
            self.pending_open = Some(path.to_path_buf());
            self.status = "Unsaved edits would be discarded — click Open again to confirm.".into();
            return;
        }
        self.pending_open = None;
        match load_path(path) {
            Ok(wb) => {
                self.workbook_name = wb.name.clone();
                let warnings = wb.warnings.join(" ");
                self.sync.mode = self
                    .prefs
                    .mode_by_workbook
                    .get(&self.workbook_name)
                    .copied()
                    .unwrap_or_default();
                let inc = IncrementalSnapshot::new(wb);
                self.saved_revision = inc.revision;
                // A rebuild in flight belongs to the old workbook.
                self.rebuild_job = None;
                self.pending_dirty.clear();
                self.rebuild_wanted = false;
                self.graph_cache = GraphCache::default();
                let violations = self.rebuild(inc.snapshot(), None);
                self.inc = Some(inc);
                self.current_path = Some(path.to_path_buf());
                self.selected_sheet = 0;
                self.edit_buffers.clear();
                self.frame_revision = u64::MAX;
                // A new snapshot restarts at revision 0: never reuse
                // the previous workbook's computed results.
                self.values.clear();
                // Chat (and any in-flight answer) belongs to the old
                // workbook; never land its reply in the new one.
                if let Some(cancel) = &self.chat_cancel {
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                self.chat_cancel = None;
                self.chat.clear();
                self.chat_dropped = 0;
                self.chat_sources.clear();
                self.history.clear();
                self.find_matches.clear();
                self.find_for = None;
                self.chat_rx = None;
                self.chat_busy = false;
                self.chat_started = None;
                // rebuild() reports invariant violations explicitly;
                // surface them instead of papering over the status.
                if violations.is_empty() && warnings.is_empty() {
                    self.status = format!("Opened {}", path.display());
                } else if violations.is_empty() {
                    self.status = format!("Opened {}. Warning: {warnings}", path.display());
                } else {
                    self.status = format!(
                        "Opened {} — but graph invariants violated: {violations:?}; search index disabled",
                        path.display()
                    );
                    if !warnings.is_empty() {
                        self.status.push_str(&format!(". Warning: {warnings}"));
                    }
                }
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    /// Rebuild the graph and refresh the index from `snapshot`, on
    /// this thread. `dirty` is `None` for a newly opened workbook
    /// (fresh index, selection dropped) and the edited ranges after an
    /// edit (see [`apply_rebuild`](Self::apply_rebuild)).
    fn rebuild(
        &mut self,
        snapshot: &gridtalk_graph::WorkbookSnapshot,
        dirty: Option<Vec<RangeRef>>,
    ) -> Vec<String> {
        let after_edit = dirty.is_some();
        let (index, cache) = if after_edit {
            (self.index.take(), std::mem::take(&mut self.graph_cache))
        } else {
            self.index = None;
            (None, GraphCache::default())
        };
        let (graph, violations, index, cache) =
            compute_rebuild(snapshot, index, cache, dirty.as_deref());
        self.graph_cache = cache;
        self.apply_rebuild(graph, violations, index, after_edit)
    }

    /// Install a rebuilt graph and index. After an edit
    /// (`after_edit`) the selection is a cell range that still means
    /// the same cells, so it is kept and its highlighted node
    /// re-found; and when the edit left the graph's shape alone (the
    /// usual case: a value changed inside a region) the layout is kept
    /// too, instead of the graph view jumping to a fresh one.
    fn apply_rebuild(
        &mut self,
        graph: Graph,
        violations: Vec<String>,
        index: Option<Index>,
        after_edit: bool,
    ) -> Vec<String> {
        let same_shape =
            after_edit && graph.nodes == self.graph.nodes && graph.edges == self.graph.edges;
        if !same_shape {
            self.graph_version += 1;
            self.positions.clear();
            if let Some(worker) = &self.layout_worker {
                worker.cancel();
            }
            self.layout_worker = None;
            self.layout_revision = u64::MAX; // force fresh layout for the new graph
        }
        if after_edit {
            self.sync.highlighted_node = self
                .sync
                .selected_range
                .as_ref()
                .and_then(|sel| node_for_selection(&graph, sel));
        } else {
            // Selection refers to the previous workbook; drop it.
            self.sync.selected_range = None;
            self.sync.highlighted_node = None;
            self.scrolled_for = None;
        }
        self.index = index;
        self.graph = graph;
        violations
    }

    /// Start rebuilding for the current snapshot on a worker thread,
    /// with the edits gathered in `pending_dirty`.
    fn start_rebuild(&mut self, ctx: Option<egui::Context>) {
        let Some(inc) = &self.inc else {
            return;
        };
        let revision = inc.revision;
        let snapshot = Arc::new(inc.snapshot().clone());
        // The formula worker needs the same copy for this revision.
        self.values.share(revision, snapshot.clone());
        let dirty = std::mem::take(&mut self.pending_dirty);
        let index = self.index.take();
        let cache = std::mem::take(&mut self.graph_cache);
        let incremental = index.is_some();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(compute_rebuild(&snapshot, index, cache, Some(&dirty)));
            if let Some(ctx) = ctx {
                ctx.request_repaint();
            }
        });
        self.rebuild_job = Some(RebuildJob {
            revision,
            incremental,
            rx,
        });
        self.rebuild_wanted = false;
    }

    /// Install a finished rebuild, waiting for it when `block`.
    /// Returns true when one landed.
    fn poll_rebuild(&mut self, block: bool) -> bool {
        let Some(job) = &self.rebuild_job else {
            return false;
        };
        let result = if block {
            job.rx.recv().ok()
        } else {
            match job.rx.try_recv() {
                Ok(result) => Some(result),
                Err(std::sync::mpsc::TryRecvError::Empty) => return false,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
            }
        };
        let job = self.rebuild_job.take().expect("job checked above");
        match result {
            Some((graph, violations, index, cache)) => {
                self.graph_cache = cache;
                let violations = self.apply_rebuild(graph, violations, index, true);
                self.layout_revision = job.revision;
                if !violations.is_empty() {
                    self.status = format!("Graph invariants violated: {violations:?}");
                }
            }
            None => {
                // The worker panicked and took the index with it. An
                // incremental update is retried once as a fresh build;
                // a failed fresh build is not retried on every frame.
                self.index = None;
                if job.incremental {
                    self.rebuild_wanted = true;
                    self.status = "Rebuilding the graph after an edit failed; retrying.".into();
                } else {
                    self.status = "Rebuilding the graph after an edit failed.".into();
                }
            }
        }
        true
    }

    /// Adapter prompt text for chat, built by the same
    /// `gridtalk_adapter` code the CLI uses: rules, trigger-matched
    /// skills, then the per-workbook `.gridtalk.md` sidecar (which
    /// applies with or without an active adapter). Returns the
    /// capped text and the adapter's `top_k`.
    fn active_adapter_prompt(&self, question: &str) -> (String, usize) {
        let sidecar = self.current_path.as_deref().and_then(sidecar_rules);
        let adapter = (!self.active_adapter.is_empty())
            .then(|| load_adapter(&adapters_root(), &self.active_adapter));
        let top_k = adapter.as_ref().map_or(3, |a| a.top_k.clamp(1, 25));
        let text = prompt_text(adapter.as_ref(), question, sidecar.as_deref());
        (cap_prompt(text), top_k)
    }

    /// Refresh the adapter picker cache (filesystem reads happen
    /// on open/selection change, never per frame).
    fn refresh_adapter_cache(&mut self) {
        let root = adapters_root();
        self.adapter_cache = list_adapters(&root)
            .into_iter()
            .map(|name| {
                let version = load_adapter(&root, &name).version.to_string();
                (name, version)
            })
            .collect();
        self.adapter_card_cache = if self.active_adapter.is_empty() {
            String::new()
        } else {
            std::fs::read_to_string(root.join(&self.active_adapter).join("CARD.md"))
                .unwrap_or_else(|_| "(no eval record yet)".into())
        };
    }

    fn ask(&mut self) {
        let question = self.chat_input.trim().to_string();
        if question.is_empty() || self.chat_busy {
            return;
        }
        self.chat_input.clear();
        // What was said before this question, and what is selected
        // as it is asked: a follow-up ("and what reads its Hrs
        // column?") and a pointed question ("what feeds this?")
        // both stand on them. Captured before the search below
        // moves the selection to its top hit.
        let history = chat_history(&self.chat);
        let standalone = gridtalk_index::standalone_query(&question, &history);
        let asked_selection = if gridtalk_index::references_selection(&question)
            && self.sync.selected_range != self.ask_selected
        {
            self.sync.selected_range.clone()
        } else {
            None
        };
        self.push_chat("you", question.clone());
        // Search the index as of the latest edit: land the rebuild in
        // flight, then run one more for edits that arrived during it.
        self.poll_rebuild(true);
        if self.rebuild_wanted {
            self.start_rebuild(None);
            self.poll_rebuild(true);
        }
        let Some(index) = &self.index else {
            self.push_chat("gridtalk", "Open a workbook first.".into());
            return;
        };
        let (adapter_prompt, adapter_top_k) = self.active_adapter_prompt(&standalone);
        let hits = index.search(&standalone, adapter_top_k);
        // A question about the selection stands on the selection
        // even when its own words match no region.
        let selection_section = match (&asked_selection, &self.inc) {
            (Some(sel), Some(inc)) => selection_context(&self.graph, inc.snapshot(), sel),
            _ => String::new(),
        };
        if hits.is_empty() && selection_section.is_empty() {
            self.push_chat(
                "gridtalk",
                format!("No matching region found for \"{question}\"."),
            );
            return;
        }
        // A question about the selection keeps it selected, so a
        // follow-up ("and what reads it?") still points at it.
        if let Some(first) = hits.first().filter(|_| asked_selection.is_none()) {
            if let Some(node) =
                self.graph
                    .node_for_cell(&first.range.sheet, first.range.r0, first.range.c0)
            {
                let id = node.id;
                self.sync.select_node(&self.graph, id);
                self.ask_selected = self.sync.selected_range.clone();
            }
        }
        self.chat_sources = hits.iter().map(|h| h.range.clone()).collect();
        if let Some(sel) = &asked_selection {
            if !selection_section.is_empty() && !self.chat_sources.contains(sel) {
                self.chat_sources.insert(0, sel.clone());
            }
        }
        let mut context = hits
            .iter()
            .map(|h| format!("[{}]\n{}", h.range.a1(), h.passage))
            .collect::<Vec<_>>()
            .join("\n---\n");
        // The selection leads the context, ahead of the hits.
        if !selection_section.is_empty() {
            context = if context.is_empty() {
                selection_section
            } else {
                format!("{selection_section}\n---\n{context}")
            };
        }
        // The conversation so far, for the prompt: recent turns,
        // each cut so one long turn cannot crowd out the rest.
        let history_text = gridtalk_index::history_prompt(&history);
        let searched_as = if standalone != question {
            format!("(searched as: {standalone})\n")
        } else {
            String::new()
        };
        let config_path = self.config_path.clone();
        let cancel: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
        self.chat_cancel = Some(cancel.clone());
        self.chat_workbook = self.workbook_name.clone();
        let workbook_tag = self.workbook_name.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        self.chat_rx = Some(rx);
        self.chat_busy = true;
        self.chat_started = Some(std::time::Instant::now());
        // Generation (potentially seconds on a local model) runs on
        // a worker thread; the UI polls the channel each frame.
        std::thread::spawn(move || {
            let cfg = if config_path.is_empty() {
                // Blank means the platform default the docs
                // promise (<config dir>/gridtalk/config.toml),
                // matching the CLI's discovery.
                let platform = config_dir().join("config.toml");
                if platform.exists() {
                    Config::load(&platform)
                } else {
                    Config::default()
                }
            } else {
                Config::load(std::path::Path::new(&config_path))
            };
            let backend = backend_from_config(&cfg.inference);
            let cleartext = cleartext_warning(&cfg.inference);
            let req = ChatRequest {
                system: format!(
                    "You answer questions about a spreadsheet using only the cited context. Always cite the range.{adapter_prompt}"
                ),
                user: format!("{history_text}Question: {question}\n{searched_as}\nContext:\n{context}"),
                max_tokens: 512,
            };
            let mut out = String::new();
            let mut msg = match backend.stream(&req, &cancel, &mut |tok| out.push_str(tok)) {
                Ok(()) => (workbook_tag.clone(), backend.name().to_string(), out),
                Err(e) => (workbook_tag.clone(), "error".to_string(), e.to_string()),
            };
            // Every answer that travelled over cleartext says so.
            if let Some(warning) = cleartext {
                msg.2 = format!("Warning: {warning}\n\n{}", msg.2);
            }
            let _ = tx.send(msg);
        });
    }

    /// Pick a file to open (the Open button and Ctrl+O).
    fn open_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Spreadsheets", &open_extensions())
            .pick_file()
        {
            self.open(&path);
        } else {
            self.status = "Open cancelled.".into();
        }
    }

    /// Keyboard shortcuts: Ctrl+S save, Ctrl+O open, Ctrl+F find,
    /// Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo (Cmd on macOS), Escape
    /// closes the find bar. Undo and redo act on grid edits unless a
    /// text field is being typed in, so a cell editor keeps egui's own
    /// text undo while it is being typed in; a grid cell that was only
    /// clicked into (not typed in yet) has no text to undo.
    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        let pressed = |modifiers: Modifiers, key: Key| {
            ctx.input_mut(|i| i.consume_shortcut(&KeyboardShortcut::new(modifiers, key)))
        };
        if pressed(Modifiers::COMMAND, Key::S) {
            self.save();
        }
        if pressed(Modifiers::COMMAND, Key::O) {
            self.open_dialog();
        }
        if pressed(Modifiers::COMMAND, Key::F) {
            self.find_open = true;
            self.find_focus = true;
        }
        if self.find_open && ctx.input(|i| i.key_pressed(Key::Escape)) {
            self.find_open = false;
        }
        if pressed(Modifiers::COMMAND, Key::L) {
            self.chat_focus = true;
        }
        // Zoom works while typing too, as in a browser. Ctrl+Plus is
        // Ctrl+Shift+= on many layouts; the extra Shift is ignored.
        if pressed(Modifiers::COMMAND, Key::Plus) || pressed(Modifiers::COMMAND, Key::Equals) {
            self.set_zoom(ctx, ctx.zoom_factor() + ZOOM_STEP);
        }
        if pressed(Modifiers::COMMAND, Key::Minus) {
            self.set_zoom(ctx, ctx.zoom_factor() - ZOOM_STEP);
        }
        if pressed(Modifiers::COMMAND, Key::Num0) {
            self.set_zoom(ctx, DEFAULT_ZOOM);
        }
        let typing_in_field = ctx
            .memory(|m| m.focused())
            .is_some_and(|id| Some(id) != self.grid_editor_focus || self.history.typing());
        if typing_in_field {
            return;
        }
        // Redo first: Ctrl+Z would also match Ctrl+Shift+Z.
        if pressed(Modifiers::COMMAND | Modifiers::SHIFT, Key::Z)
            || pressed(Modifiers::COMMAND, Key::Y)
        {
            self.redo();
        } else if pressed(Modifiers::COMMAND, Key::Z) {
            self.undo();
        }
    }

    /// Scale the whole UI to `zoom` (clamped) and remember it.
    fn set_zoom(&mut self, ctx: &egui::Context, zoom: f32) {
        let zoom = clamp_zoom(zoom);
        ctx.set_zoom_factor(zoom);
        if self.prefs.zoom != Some(zoom) {
            self.prefs.zoom = Some(zoom);
            self.prefs.save();
        }
    }

    fn undo(&mut self) {
        self.step_history(true);
    }

    fn redo(&mut self) {
        self.step_history(false);
    }

    /// Undo (`back`) or redo one grid edit.
    fn step_history(&mut self, back: bool) {
        if self.is_read_only() {
            let verb = if back { "undone" } else { "redone" };
            self.status = format!("Read-only mode is on: nothing is {verb}.");
            return;
        }
        let change = if back {
            self.history.undo()
        } else {
            self.history.redo()
        };
        let Some(change) = change else {
            self.status = format!("Nothing to {}.", if back { "undo" } else { "redo" });
            return;
        };
        let cell = gridtalk_graph::CellRef {
            sheet: change.sheet.clone(),
            row: change.row,
            col: change.col,
        }
        .a1();
        let value = if back { change.before } else { change.after };
        self.apply_change(&change.sheet, change.row, change.col, value);
        self.status = if back {
            format!("Undid the edit of {cell} (Ctrl+Y redoes it).")
        } else {
            format!("Redid the edit of {cell}.")
        };
    }

    /// Set a cell for undo or redo: record the edit (not in the
    /// history), show the value in its editor, and select the cell.
    fn apply_change(
        &mut self,
        sheet: &str,
        row: usize,
        col: usize,
        value: gridtalk_graph::CellValue,
    ) {
        let Some(inc) = &mut self.inc else {
            return;
        };
        let Some(sheet_idx) = inc
            .snapshot()
            .sheets
            .iter()
            .position(|s| gridtalk_graph::sheet_name_eq(&s.name, sheet))
        else {
            return;
        };
        self.edit_buffers
            .insert((sheet_idx, row, col), cell_input_text(&value));
        inc.record_edit(
            CellEdit {
                sheet: sheet.to_string(),
                row,
                col,
                value,
            },
            std::time::Instant::now(),
        );
        self.select_on_sheet(
            sheet_idx,
            &RangeRef {
                sheet: sheet.to_string(),
                r0: row,
                c0: col,
                r1: row,
                c1: col,
            },
        );
    }

    /// Show `range`'s sheet and select the range (the grid scrolls to
    /// it).
    fn select_on_sheet(&mut self, sheet_idx: usize, range: &RangeRef) {
        self.selected_sheet = sheet_idx;
        self.sync.select_range(&self.graph, range);
    }

    /// Select a range named by sheet (a chat answer's source).
    fn select_source(&mut self, range: &RangeRef) {
        let idx = self.inc.as_ref().and_then(|inc| {
            inc.snapshot()
                .sheets
                .iter()
                .position(|s| gridtalk_graph::sheet_name_eq(&s.name, &range.sheet))
        });
        match idx {
            Some(idx) => self.select_on_sheet(idx, range),
            None => self.status = format!("{} is not in this workbook any more.", range.a1()),
        }
    }

    /// The find matches for the current query and workbook revision.
    fn refresh_find(&mut self) {
        let Some(inc) = &self.inc else {
            self.find_matches.clear();
            return;
        };
        let key = (self.find_query.clone(), inc.revision);
        if self.find_for.as_ref() != Some(&key) {
            let snapshot = inc.snapshot();
            self.find_matches = find::find_cells(snapshot, &self.find_query);
            // An edit re-runs the search: carry on from the selected
            // match, if it still is one, instead of the first.
            let selected = self.sync.selected_range.as_ref().and_then(|sel| {
                let s = snapshot
                    .sheets
                    .iter()
                    .position(|sh| gridtalk_graph::sheet_name_eq(&sh.name, &sel.sheet))?;
                (sel.r0 == sel.r1 && sel.c0 == sel.c1).then_some((s, sel.r0, sel.c0))
            });
            self.find_pos = selected
                .and_then(|m| self.find_matches.binary_search(&m).ok())
                .unwrap_or(0);
            self.find_for = Some(key);
        }
    }

    /// Go to the next (`forward`) or previous match.
    fn find_step(&mut self, forward: bool) {
        self.refresh_find();
        let n = self.find_matches.len();
        if n == 0 {
            self.status = format!("No cell contains \"{}\".", self.find_query.trim());
            return;
        }
        // The first step lands on the first match.
        let at_start = self.sync.selected_range.is_none()
            || self
                .find_matches
                .get(self.find_pos)
                .is_none_or(|&(s, r, c)| {
                    let name = self
                        .inc
                        .as_ref()
                        .map(|i| i.snapshot().sheets[s].name.clone());
                    self.sync.selected_range.as_ref()
                        != Some(&RangeRef {
                            sheet: name.unwrap_or_default(),
                            r0: r,
                            c0: c,
                            r1: r,
                            c1: c,
                        })
                });
        if !at_start {
            self.find_pos = if forward {
                (self.find_pos + 1) % n
            } else {
                (self.find_pos + n - 1) % n
            };
        }
        let (s, r, c) = self.find_matches[self.find_pos];
        let Some(name) = self
            .inc
            .as_ref()
            .map(|i| i.snapshot().sheets[s].name.clone())
        else {
            return;
        };
        self.select_on_sheet(
            s,
            &RangeRef {
                sheet: name,
                r0: r,
                c0: c,
                r1: r,
                c1: c,
            },
        );
        let more = if n >= find::MAX_MATCHES { "+" } else { "" };
        self.status = format!("Match {} of {n}{more}.", self.find_pos + 1);
    }

    /// The find bar above the grid.
    fn find_bar(&mut self, ui: &mut egui::Ui) {
        let mut step: Option<bool> = None;
        ui.horizontal(|ui| {
            ui.label("Find:");
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.find_query)
                    .hint_text("text or formula")
                    .desired_width(220.0),
            );
            if std::mem::take(&mut self.find_focus) {
                resp.request_focus();
            }
            if resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                step = Some(!ui.input(|i| i.modifiers.shift));
                // Keep typing-and-Enter going.
                resp.request_focus();
            }
            if ui.button("Previous").clicked() {
                step = Some(false);
            }
            if ui.button("Next").clicked() {
                step = Some(true);
            }
            let found = self
                .find_for
                .as_ref()
                .is_some_and(|(q, _)| *q == self.find_query);
            if found && !self.find_query.trim().is_empty() {
                let n = self.find_matches.len();
                let more = if n >= find::MAX_MATCHES { "+" } else { "" };
                ui.weak(format!("{n}{more} found"));
            }
            if ui.button("✕").on_hover_text("Close (Escape)").clicked() {
                self.find_open = false;
            }
        });
        if let Some(forward) = step {
            self.find_step(forward);
        }
    }

    /// Settings' backend section: which backend answers, and how to
    /// reach it.
    fn backend_settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Inference backend");
        ui.radio_value(
            &mut self.set_backend,
            "local".to_string(),
            "Local GGUF model (private)",
        );
        if self.set_backend == "local" {
            ui.horizontal(|ui| {
                ui.label("Model .gguf:");
                ui.text_edit_singleline(&mut self.set_model_path);
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("GGUF model", &["gguf"])
                        .pick_file()
                    {
                        self.set_model_path = path.display().to_string();
                    }
                }
            });
        }
        ui.radio_value(
            &mut self.set_backend,
            "openai-compatible".to_string(),
            "OpenAI-compatible endpoint (cell values are sent verbatim)",
        );
        if self.set_backend == "openai-compatible" {
            egui::Grid::new("backend-settings")
                .num_columns(2)
                .show(ui, |ui| {
                    ui.label("Base URL:");
                    ui.text_edit_singleline(&mut self.set_base_url);
                    ui.end_row();
                    ui.label("Model name:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.set_model)
                            .hint_text("as the endpoint names it"),
                    );
                    ui.end_row();
                    ui.label("API key variable:");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.set_api_key_env)
                            .hint_text("environment variable holding the key"),
                    );
                    ui.end_row();
                });
            if let Some(warning) = cleartext_url_warning(&self.set_base_url) {
                ui.colored_label(ui.visuals().warn_fg_color, warning);
            }
            ui.weak("The key itself is never written to the config file.");
        }
        if ui.button("Save backend").clicked() {
            self.save_backend_settings();
        }
    }

    /// Validate and write Settings' backend to the config file in use.
    fn save_backend_settings(&mut self) {
        if let Err(e) = validate_wizard(
            &self.set_backend,
            self.set_model_path.trim(),
            self.set_base_url.trim(),
        ) {
            self.status = format!("Backend not saved: {e}");
            return;
        }
        let path = active_config_file(&self.config_path);
        let optional = |v: &str| Some(v.trim()).filter(|v| !v.is_empty()).map(toml_string);
        let updates: Vec<ConfigUpdate> = if self.set_backend == "local" {
            vec![
                ("backend", Some(toml_string("local"))),
                ("engine", Some(toml_string("llamacpp"))),
                ("model_path", optional(&self.set_model_path)),
            ]
        } else {
            vec![
                ("backend", Some(toml_string("openai-compatible"))),
                ("base_url", optional(&self.set_base_url)),
                ("model", optional(&self.set_model)),
                ("api_key_env", optional(&self.set_api_key_env)),
            ]
        };
        self.status = match write_inference_config(&path, &updates) {
            Ok(()) => {
                // The cleartext warning follows the saved backend.
                // Only it is reloaded: unsaved GPU-layer and context
                // edits in the section below must survive this save.
                self.llm_cleartext = cleartext_warning(&Config::load(&path).inference);
                format!(
                    "Backend saved to {}; it answers the next question.",
                    path.display()
                )
            }
            Err(e) => format!("Could not save the backend to {}: {e}", path.display()),
        };
    }

    /// Load the llama.cpp settings shown in the Settings window from
    /// the config file in use.
    fn load_llm_settings(&mut self) {
        let path = active_config_file(&self.config_path);
        let cfg = Config::load(&path);
        // u32::MAX is how the backend spells "all layers".
        self.llm_gpu_layers = cfg.inference.n_gpu_layers.filter(|&n| n != u32::MAX);
        self.llm_n_ctx = cfg.inference.n_ctx;
        self.llm_cleartext = cleartext_warning(&cfg.inference);
        self.set_backend = if cfg.inference.backend == "openai-compatible" {
            "openai-compatible".into()
        } else {
            "local".into()
        };
        self.set_model_path = cfg.inference.model_path.clone().unwrap_or_default();
        self.set_base_url = cfg
            .inference
            .base_url
            .clone()
            .unwrap_or_else(|| gridtalk_infer::DEFAULT_BASE_URL.to_string());
        self.set_model = cfg.inference.model.clone().unwrap_or_default();
        self.set_api_key_env = cfg.inference.api_key_env.clone().unwrap_or_default();
        self.llm_settings_loaded = true;
    }

    /// GPU-layer and context-size controls for the local backend.
    fn llm_settings_ui(&mut self, ui: &mut egui::Ui) {
        ui.strong("Local model (llama.cpp)");
        egui::Grid::new("llm-settings")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("GPU layers:");
                ui.horizontal(|ui| {
                    let mut all = self.llm_gpu_layers.is_none();
                    if ui
                        .checkbox(&mut all, "All")
                        .on_hover_text("Offload every layer to the GPU (Vulkan).")
                        .changed()
                    {
                        self.llm_gpu_layers = if all { None } else { Some(0) };
                    }
                    if let Some(n) = &mut self.llm_gpu_layers {
                        ui.add(egui::DragValue::new(n).range(0..=LLM_MAX_GPU_LAYERS))
                            .on_hover_text("0 runs on the CPU only.");
                    }
                });
                ui.end_row();
                ui.label("Context limit:");
                ui.horizontal(|ui| {
                    let mut model = self.llm_n_ctx.is_none();
                    if ui
                        .checkbox(&mut model, "Model default")
                        .on_hover_text("Use the model's training length (up to 64K tokens).")
                        .changed()
                    {
                        self.llm_n_ctx = if model { None } else { Some(LLM_DEFAULT_N_CTX) };
                    }
                    if let Some(n) = &mut self.llm_n_ctx {
                        ui.add(
                            egui::DragValue::new(n)
                                .range(LLM_MIN_N_CTX..=LLM_MAX_N_CTX)
                                .speed(256)
                                .suffix(" tokens"),
                        );
                    }
                });
                ui.end_row();
            });
        ui.weak("Lower these if the model does not fit in GPU memory.");
        if ui.button("Save model settings").clicked() {
            let path = active_config_file(&self.config_path);
            let updates: [ConfigUpdate; 2] = [
                ("n_gpu_layers", self.llm_gpu_layers.map(|n| n.to_string())),
                ("n_ctx", self.llm_n_ctx.map(|n| n.to_string())),
            ];
            self.status = match write_inference_config(&path, &updates) {
                Ok(()) => format!(
                    "Model settings saved to {}; they apply to the next question.",
                    path.display()
                ),
                Err(e) => format!("Could not save model settings to {}: {e}", path.display()),
            };
        }
    }

    /// Read-only from prefs, or locked on by `--read-only`.
    fn is_read_only(&self) -> bool {
        self.prefs.read_only || self.read_only_locked
    }

    fn save(&mut self) {
        if self.is_read_only() {
            self.status = "Read-only mode is on: saving is disabled.".into();
            return;
        }
        let Some(path) = self.current_path.clone() else {
            self.status = "Nothing to save: open a workbook first, or use Save As.".into();
            return;
        };
        self.save_to(&path, true);
    }

    fn save_as(&mut self, path: &std::path::Path) {
        if self.is_read_only() {
            self.status = "Read-only mode is on: saving is disabled.".into();
            return;
        }
        // If the chosen name has no usable extension, default to
        // .xlsx rather than failing in save_to.
        let path = if path.extension().is_none() {
            path.with_extension("xlsx")
        } else {
            path.to_path_buf()
        };
        if self.save_to(&path, false) {
            self.current_path = Some(path);
        }
    }

    /// Flush pending debounced edits first, then write. Returns
    /// true on success.
    fn save_to(&mut self, path: &std::path::Path, _is_current: bool) -> bool {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        // A rebuild in flight holds the index; let it land first.
        self.poll_rebuild(true);
        // An edit typed <500ms ago must be in the saved file.
        let mut dropped_note = None;
        let mut applied = 0;
        if let Some(inc) = &mut self.inc {
            let before = inc.dropped_edits();
            applied = inc.flush();
            dropped_note = dropped_edit_note(inc, before);
        }
        let Some(inc) = &mut self.inc else {
            return false;
        };
        // The flush empties the pending queue, so poll() will not see
        // these edits: rebuild the graph and index here instead.
        self.pending_dirty.extend(inc.take_dirty_ranges());
        let snapshot = inc.snapshot().clone();
        let revision = inc.revision;
        if applied > 0 {
            let dirty = std::mem::take(&mut self.pending_dirty);
            self.rebuild_wanted = false;
            let violations = self.rebuild(&snapshot, Some(dirty));
            self.layout_revision = revision;
            if !violations.is_empty() {
                self.status = format!("Graph invariants violated: {violations:?}");
            }
        }
        // Only .xlsm and .xltm hold macros: say when a save leaves them out.
        let macros_note = (gridtalk_graph::has_macros(&snapshot)
            && matches!(ext.as_str(), "xlsx" | "xltx"))
        .then_some("The workbook's macros were not saved: only .xlsm and .xltm hold them.");
        let result: Result<(), String> = match ext.as_str() {
            e if gridtalk_graph::WRITABLE_PACKAGE_EXTENSIONS.contains(&e) => {
                gridtalk_graph::to_package_bytes(&snapshot, e)
                    .map_err(|e| e.to_string())
                    .and_then(|bytes| write_atomic_bytes(path, &bytes))
            }
            "csv" | "tsv" => {
                if snapshot.sheets.len() > 1 {
                    Err(format!(
                        "workbook has {} sheets but .{ext} holds one sheet; save as .xlsx to keep them all",
                        snapshot.sheets.len()
                    ))
                } else {
                    let delim = if ext == "tsv" { '\t' } else { ',' };
                    let text = snapshot_to_delimited(&snapshot, delim);
                    write_atomic_bytes(path, text.as_bytes())
                }
            }
            e if gridtalk_graph::PACKAGE_EXTENSIONS.contains(&e) => Err(format!(
                ".{ext} is read-only in GridTalk; use Save As to write .xlsx or .csv"
            )),
            _ => Err(format!("cannot save .{ext}; use .xlsx or .csv")),
        };
        match result {
            Ok(()) => {
                self.saved_revision = revision;
                let notes: Vec<&str> = [dropped_note.as_deref(), macros_note]
                    .into_iter()
                    .flatten()
                    .collect();
                self.status = if notes.is_empty() {
                    format!("Saved {}", path.display())
                } else {
                    format!("Saved {}. {}", path.display(), notes.join(" "))
                };
                true
            }
            Err(e) => {
                self.status = format!("Save failed: {e}");
                false
            }
        }
    }

    fn dirty(&self) -> bool {
        self.inc
            .as_ref()
            .map(|inc| inc.revision != self.saved_revision || inc.pending_edits() > 0)
            .unwrap_or(false)
    }
}

/// What the Open dialog offers: CSV/TSV and every format
/// `gridtalk_graph` reads through Gridcraft.
fn open_extensions() -> Vec<&'static str> {
    ["csv", "tsv"]
        .into_iter()
        .chain(gridtalk_graph::PACKAGE_EXTENSIONS.iter().copied())
        .collect()
}

/// Hover text for the save buttons: what an .xlsx save keeps.
const SAVE_HOVER: &str = "Saving .xlsx keeps what was read from the opened file (formatting, merges, tables, charts), with your edits applied; anything it could not read was listed when it opened. .csv and .tsv hold one sheet's cells. Macros are kept only in .xlsm and .xltm; .xls, .xlsb and .ods files are read-only";

/// A status line for edits dropped since `before` (the dropped-edit
/// count before a flush), or `None` when every edit was applied.
fn dropped_edit_note(inc: &IncrementalSnapshot, before: usize) -> Option<String> {
    let n = inc.dropped_edits().saturating_sub(before);
    if n == 0 {
        return None;
    }
    let what = if n == 1 { "edit was" } else { "edits were" };
    let reason = inc.last_drop_reason().unwrap_or("unknown reason");
    Some(format!("{n} {what} not applied: {reason}"))
}

/// A non-editable grid cell (read-only mode, or a formula result):
/// painted like an editor, clickable to select.
fn display_cell(
    ui: &mut egui::Ui,
    size: egui::Vec2,
    text: &str,
    selected: bool,
    is_formula: bool,
    error: bool,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let visuals = ui.visuals();
    let fill = if selected {
        egui::Color32::from_rgb(74, 62, 0)
    } else {
        visuals.extreme_bg_color
    };
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, fill);
    if resp.hovered() {
        painter.rect_stroke(
            rect,
            2.0,
            visuals.widgets.hovered.bg_stroke,
            egui::StrokeKind::Inside,
        );
    }
    let colour = if error {
        egui::Color32::from_rgb(230, 110, 110)
    } else if selected {
        egui::Color32::YELLOW
    } else if is_formula {
        // Computed results read slightly differently from inputs.
        egui::Color32::from_rgb(150, 200, 255)
    } else {
        egui::Color32::LIGHT_GRAY
    };
    painter.text(
        rect.left_center() + egui::vec2(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        text,
        egui::TextStyle::Body.resolve(ui.style()),
        colour,
    );
    resp
}

fn cell_input_text(cell: &gridtalk_graph::CellValue) -> String {
    match cell {
        gridtalk_graph::CellValue::Formula { expr, .. } => format!("={expr}"),
        other => other.display(),
    }
}

fn snapshot_to_delimited(snapshot: &gridtalk_graph::WorkbookSnapshot, delim: char) -> String {
    let Some(sheet) = snapshot.sheets.first() else {
        return String::new();
    };
    let mut out = String::new();
    for row in &sheet.cells {
        let fields: Vec<String> = row
            .iter()
            .map(|c| {
                let text = cell_input_text(c);
                // Quote anything the reader would split or trim:
                // delimiters, quotes, CR/LF, and edge whitespace.
                if text.contains(delim)
                    || text.contains('"')
                    || text.contains('\n')
                    || text.contains('\r')
                    || text.trim() != text
                {
                    format!("\"{}\"", text.replace('"', "\"\""))
                } else {
                    text
                }
            })
            .collect();
        out.push_str(&fields.join(&delim.to_string()));
        out.push('\n');
    }
    out
}

impl eframe::App for GridTalkApp {
    fn ui(&mut self, ui_root: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui_root.ctx().clone();
        self.handle_shortcuts(&ctx);
        // Closing with unsaved edits needs a second, deliberate
        // close. The guard re-arms whenever the workbook is clean
        // again (saved/opened), so it works more than once.
        if !self.dirty() {
            self.close_armed = false;
            // A stale discard-confirmation must not survive a save:
            // the next Open click would otherwise discard silently.
            self.pending_open = None;
        }
        if ctx.input(|i| i.viewport().close_requested()) && self.dirty() && !self.close_armed {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_armed = true;
            self.status = "Unsaved edits — close again to exit without saving.".into();
        }
        // Apply debounced cell edits; rebuild graph+index on revision change.
        // Applied edits are rebuilt into the graph and index on a
        // worker thread, one rebuild at a time; edits landing while one
        // runs are gathered for the next.
        let mut note = None;
        if let Some(inc) = &mut self.inc {
            let before = inc.dropped_edits();
            if inc.poll(std::time::Instant::now()) {
                note = dropped_edit_note(inc, before);
                self.pending_dirty.extend(inc.take_dirty_ranges());
                self.rebuild_wanted = true;
                ctx.request_repaint_after(std::time::Duration::from_millis(120));
            } else if inc.pending_edits() > 0 {
                ctx.request_repaint_after(std::time::Duration::from_millis(120));
            }
        }
        if let Some(note) = note {
            self.status = note;
        }
        self.poll_rebuild(false);
        if self.rebuild_wanted && self.rebuild_job.is_none() {
            self.start_rebuild(Some(ctx.clone()));
        }
        // Watchdog: a worker that never sends (hung backend, lost
        // cancel flag) must not wedge chat forever.
        if self.chat_busy
            && self
                .chat_started
                .is_some_and(|t| t.elapsed() > std::time::Duration::from_secs(600))
        {
            if let Some(cancel) = &self.chat_cancel {
                cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            self.chat_rx = None;
            self.chat_busy = false;
            self.chat_started = None;
            self.chat_cancel = None;
            self.status = "Generation timed out after 10 minutes; chat is ready again.".into();
        }
        // Collect finished chat generations (worker thread).
        if let Some(rx) = &self.chat_rx {
            match rx.try_recv() {
                Ok((wb_tag, role, text)) => {
                    // Drop answers generated for a workbook that is
                    // no longer open.
                    if wb_tag == self.chat_workbook && wb_tag == self.workbook_name {
                        let sources = if role == "error" {
                            Vec::new()
                        } else {
                            std::mem::take(&mut self.chat_sources)
                        };
                        self.push_chat_with_sources(&role, text, sources);
                    }
                    self.chat_rx = None;
                    self.chat_busy = false;
                    self.chat_started = None;
                    self.chat_cancel = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.push_chat(
                        "error",
                        "Generation stopped: the worker thread disconnected.".into(),
                    );
                    self.status = "Chat worker disconnected.".into();
                    self.chat_rx = None;
                    self.chat_busy = false;
                    self.chat_started = None;
                    self.chat_cancel = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(100));
                }
            }
        }
        // After a flush, drop edit buffers that now match the cell.
        if let Some(inc) = &self.inc {
            if inc.pending_edits() == 0 && !self.edit_buffers.is_empty() {
                let snapshot = inc.snapshot();
                self.edit_buffers.retain(|(sheet_idx, r, c), buffer| {
                    // Drop buffers whose *parsed value* equals the
                    // stored cell ("99.0" for 99), not only exact
                    // text matches.
                    let current = snapshot
                        .sheets
                        .get(*sheet_idx)
                        .map(|s| s.get(*r, *c))
                        .unwrap_or(gridtalk_graph::CellValue::Empty);
                    !edit_buffer_matches(&current, &parse_cell_input(buffer))
                });
            }
        }
        // Bound the edit-buffer map: scrolled-to cells insert an
        // entry each, so prune value-equal entries when it grows.
        if self.edit_buffers.len() > 1024 {
            if let Some(inc) = &self.inc {
                let snapshot = inc.snapshot();
                self.edit_buffers.retain(|(sheet_idx, r, c), buffer| {
                    let current = snapshot
                        .sheets
                        .get(*sheet_idx)
                        .map(|s| s.get(*r, *c))
                        .unwrap_or(gridtalk_graph::CellValue::Empty);
                    !edit_buffer_matches(&current, &parse_cell_input(buffer))
                });
            }
        }

        // Cache only the displayed sheet, refreshed when the
        // revision or selected sheet changes — never a whole-workbook
        // clone per frame.
        if let Some(inc) = &self.inc {
            // A graph-side selection may name another sheet: follow
            // it (sheet names compare case-insensitively, as the
            // rest of the app treats them).
            if let Some(sel) = &self.sync.selected_range {
                if let Some(idx) = inc
                    .snapshot()
                    .sheets
                    .iter()
                    .position(|sh| sh.name.eq_ignore_ascii_case(&sel.sheet))
                {
                    self.selected_sheet = idx;
                }
            }
            if inc.revision != self.frame_revision
                || self.selected_sheet != self.frame_sheet_idx
                || self.frame_sheet.is_none()
            {
                self.frame_sheet = inc.snapshot().sheets.get(self.selected_sheet).cloned();
                self.frame_sheet_idx = self.selected_sheet;
                self.frame_revision = inc.revision;
            }
        }
        // Collect finished formula results.
        if self.values.poll() {
            ctx.request_repaint();
        }
        // Collect finished layouts.
        if let Some(worker) = &self.layout_worker {
            let expected = worker.revision;
            match worker.rx.try_recv() {
                Ok((rev, positions)) => {
                    if rev == expected {
                        self.positions = positions;
                    }
                    self.layout_worker = None;
                    ctx.request_repaint();
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.layout_worker = None;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }

        egui::Panel::top("menu").show(ui_root, |ui| {
            // Wrapped, so a narrow or zoomed-in window keeps every
            // button reachable.
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open…").on_hover_text("Ctrl+O").clicked() {
                    self.open_dialog();
                }
                let save_label = if self.dirty() { "Save ●" } else { "Save" };
                if ui
                    .add_enabled(!self.is_read_only(), egui::Button::new(save_label))
                    .on_hover_text(format!("{SAVE_HOVER} (Ctrl+S)"))
                    .on_disabled_hover_text("Read-only mode is on")
                    .clicked()
                {
                    self.save();
                }
                if ui
                    .add_enabled(!self.is_read_only(), egui::Button::new("Save As…"))
                    .on_hover_text(SAVE_HOVER)
                    .on_disabled_hover_text("Read-only mode is on")
                    .clicked()
                {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Excel workbook", &["xlsx"])
                        .add_filter("Excel macro-enabled workbook", &["xlsm"])
                        .add_filter("CSV", &["csv"])
                        .add_filter("TSV", &["tsv"])
                        .save_file()
                    {
                        self.save_as(&path);
                    }
                }
                // Gridcraft is a full editor that can save over the
                // file, so it is unavailable in read-only mode.
                if ui
                    .add_enabled(!self.is_read_only(), egui::Button::new("Open in Gridcraft"))
                    .on_disabled_hover_text("Read-only mode is on: Gridcraft could edit the file")
                    .clicked()
                {
                    match &self.current_path {
                        Some(path) if self.dirty() => {
                            self.status =
                                "Save first: Gridcraft would open the stale on-disk file, not your unsaved edits.".into();
                        }
                        Some(path) => {
                            // Mirror the CLI's gridcraft_binary
                            // resolution: checked env var, beside the
                            // exe, PATH, vendor build tree.
                            let bin = gridcraft_binary_ui();
                            match bin {
                                Some(bin) => {
                                    match std::process::Command::new(&bin).arg(path).spawn() {
                                        Ok(_) => {
                                            self.status =
                                                format!("Launched Gridcraft on {}", path.display())
                                        }
                                        Err(e) => {
                                            self.status = format!(
                                                "Could not launch Gridcraft ({}): {e}",
                                                bin.display()
                                            )
                                        }
                                    }
                                }
                                None => {
                                    self.status = "Gridcraft binary not found. Build it once with: cargo build --manifest-path vendor/gridcraft/Cargo.toml -p gridcraft (or set GRIDTALK_GRIDCRAFT_BIN)".into()
                                }
                            }
                        }
                        None => self.status = "Open a workbook first.".into(),
                    }
                }
                ui.separator();
                for mode in [ViewMode::Split, ViewMode::GridFull, ViewMode::GraphFull] {
                    if ui
                        .selectable_label(self.sync.mode == mode, mode.label())
                        .clicked()
                    {
                        self.sync.mode = mode;
                        if !self.workbook_name.is_empty() {
                            self.prefs
                                .mode_by_workbook
                                .insert(self.workbook_name.clone(), mode);
                            self.prefs.save();
                        }
                    }
                }
                ui.separator();
                // Values (computed results) vs Formulas (formula text).
                let view_label = if self.prefs.show_formulas {
                    "Showing: Formulas"
                } else {
                    "Showing: Values"
                };
                if ui
                    .button(view_label)
                    .on_hover_text("Switch formula cells between computed values and formula text")
                    .clicked()
                {
                    self.prefs.show_formulas = !self.prefs.show_formulas;
                    self.prefs.save();
                }
                let mut read_only = self.is_read_only();
                let toggle = ui
                    .add_enabled(
                        !self.read_only_locked,
                        egui::Checkbox::new(&mut read_only, "Read-only"),
                    )
                    .on_hover_text("Cells cannot be edited and saving is off")
                    .on_disabled_hover_text("Locked on by --read-only");
                if toggle.changed() {
                    self.prefs.read_only = read_only;
                    self.prefs.save();
                }
                ui.separator();
                if ui.button("Settings").clicked() {
                    self.show_settings = !self.show_settings;
                }
                let adapter_label = if self.active_adapter.is_empty() {
                    "Adapter: none".to_string()
                } else {
                    format!("Adapter: {}", self.active_adapter)
                };
                if ui.button(adapter_label).clicked() {
                    self.show_adapters = !self.show_adapters;
                    if self.show_adapters {
                        self.refresh_adapter_cache();
                    }
                }
                ui.separator();
                let zoom = ctx.zoom_factor();
                if ui.button("−").on_hover_text("Zoom out (Ctrl+-)").clicked() {
                    self.set_zoom(&ctx, zoom - ZOOM_STEP);
                }
                if ui
                    .button(format!("{:.0}%", zoom * 100.0))
                    .on_hover_text(format!(
                        "Reset zoom to {:.0}% (Ctrl+0)",
                        DEFAULT_ZOOM * 100.0
                    ))
                    .clicked()
                {
                    self.set_zoom(&ctx, DEFAULT_ZOOM);
                }
                if ui.button("+").on_hover_text("Zoom in (Ctrl+=)").clicked() {
                    self.set_zoom(&ctx, zoom + ZOOM_STEP);
                }
            });
        });
        egui::Panel::bottom("status").show(ui_root, |ui| {
            ui.add(egui::Label::new(&self.status).truncate())
                .on_hover_text(&self.status);
        });

        if !self.prefs.first_run_done {
            let mut finish = false;
            let mut skip = false;
            egui::Window::new("Welcome to GridTalk")
                .collapsible(false)
                .show(&ctx, |ui| {
                    match self.wizard_step {
                        0 => {
                            ui.heading("Talk to your spreadsheet — locally.");
                            ui.label("GridTalk answers questions about your workbook with cell citations.");
                            ui.label("Local model: nothing leaves this PC.");
                            ui.label("External endpoint: cell values are sent verbatim (no redaction in GridTalk).");
                            if ui.button("Next").clicked() {
                                self.wizard_step = 1;
                            }
                        }
                        1 => {
                            ui.heading("Choose inference");
                            ui.radio_value(&mut self.wizard_backend, "local".to_string(), "Local GGUF model (private)");
                            if self.wizard_backend == "local" {
                                ui.label("Model .gguf path:");
                                ui.horizontal(|ui| {
                                    ui.text_edit_singleline(&mut self.wizard_model_path);
                                    if ui.button("Browse…").clicked() {
                                        if let Some(path) = rfd::FileDialog::new()
                                            .add_filter("GGUF model", &["gguf"])
                                            .pick_file()
                                        {
                                            self.wizard_model_path =
                                                path.display().to_string();
                                        }
                                    }
                                });
                                ui.label("Import models with: gridtalk models import <file.gguf>");
                            }
                            ui.radio_value(&mut self.wizard_backend, "openai-compatible".to_string(), "OpenAI-compatible endpoint");
                            if self.wizard_backend == "openai-compatible" {
                                ui.label("Base URL (a redacting proxy can live here):");
                                ui.text_edit_singleline(&mut self.wizard_base_url);
                                if let Some(warning) = cleartext_url_warning(&self.wizard_base_url) {
                                    ui.colored_label(ui.visuals().warn_fg_color, warning);
                                }
                            }
                            ui.horizontal(|ui| {
                                if ui.button("Back").clicked() {
                                    self.wizard_step = 0;
                                }
                                if ui.button("Next").clicked() {
                                    self.wizard_step = 2;
                                }
                            });
                        }
                        _ => {
                            ui.heading("Ready");
                            ui.label("Open a workbook from the top-left, then ask a question in the chat pane.");
                            ui.label("Tip: click a graph node to jump to its cells; click a cell to find its graph node.");
                            ui.horizontal(|ui| {
                                if ui.button("Back").clicked() {
                                    self.wizard_step = 1;
                                }
                                if ui.button("Finish").clicked() {
                                    finish = true;
                                }
                            });
                        }
                    }
                    ui.separator();
                    if ui.button("Skip setup").clicked() {
                        skip = true;
                    }
                });
            if skip {
                // Skipping is a completed first run: never trap the
                // user in the wizard on every launch.
                self.prefs.first_run_done = true;
                self.prefs.save();
                self.status = "Setup skipped — no inference backend configured yet.".into();
            }
            if finish {
                // Validation failure must not abort the frame: set
                // the status and keep rendering the rest of the UI.
                match validate_wizard(
                    &self.wizard_backend,
                    &self.wizard_model_path,
                    &self.wizard_base_url,
                ) {
                    Err(e) => {
                        self.status = format!("Setup not saved: {e}");
                    }
                    Ok(()) => match write_wizard_config(
                        &self.wizard_backend,
                        &self.wizard_model_path,
                        &self.wizard_base_url,
                    ) {
                        Ok(p) => {
                            self.config_path = p.display().to_string();
                            self.prefs.config_path = self.config_path.clone();
                            self.status = format!("Setup saved to {}", p.display());
                            // Only a successfully saved setup counts as done.
                            self.prefs.first_run_done = true;
                            self.prefs.save();
                        }
                        Err(e) => {
                            self.status = format!(
                                "Could not save setup: {e} — fix the path and click Finish again."
                            )
                        }
                    },
                }
            }
        }

        if self.show_adapters {
            let root = adapters_root();
            let mut chosen: Option<String> = None;
            let active = self.active_adapter.clone();
            let cache = self.adapter_cache.clone();
            let card = self.adapter_card_cache.clone();
            let mut open = self.show_adapters;
            egui::Window::new("Adapters")
                .open(&mut open)
                .show(&ctx, |ui| {
                    ui.label(format!("Adapters in {}", root.display()));
                    if ui.selectable_label(active.is_empty(), "(none)").clicked() {
                        chosen = Some(String::new());
                    }
                    for (name, version) in &cache {
                        if ui
                            .selectable_label(active == *name, format!("{name} (v{version})"))
                            .clicked()
                        {
                            chosen = Some(name.clone());
                        }
                    }
                    if !active.is_empty() {
                        ui.separator();
                        ui.label("CARD.md:");
                        egui::ScrollArea::vertical()
                            .max_height(220.0)
                            .show(ui, |ui| {
                                ui.label(&card);
                            });
                    }
                    ui.label("Optimise from the CLI: gridtalk optimise <workbook>");
                });
            self.show_adapters = open;
            if let Some(name) = chosen {
                self.active_adapter = name;
                self.prefs.active_adapter = self.active_adapter.clone();
                self.prefs.save();
                self.refresh_adapter_cache();
            }
        }

        if self.show_settings {
            if !self.llm_settings_loaded {
                self.load_llm_settings();
            }
            egui::Window::new("Settings").show(&ctx, |ui| {
                ui.label("Config file (blank = platform config dir, else local echo backend):");
                let resp = ui.text_edit_singleline(&mut self.config_path);
                // Persist on Enter/focus loss, not on every keystroke,
                // so a half-typed path is never saved mid-edit.
                if resp.lost_focus() {
                    self.prefs.config_path = self.config_path.clone();
                    self.prefs.save();
                    // Show the model settings of the file now in use.
                    self.llm_settings_loaded = false;
                }
                ui.label("External endpoints receive cell values verbatim.");
                ui.label("For sensitive workbooks, use a local model.");
                if let Some(warning) = &self.llm_cleartext {
                    ui.colored_label(ui.visuals().warn_fg_color, format!("Warning: {warning}"));
                }
                ui.separator();
                self.backend_settings_ui(ui);
                ui.separator();
                self.llm_settings_ui(ui);
                ui.separator();
                if ui.button("Close").clicked() {
                    self.prefs.config_path = self.config_path.clone();
                    self.prefs.save();
                    self.show_settings = false;
                    self.llm_settings_loaded = false;
                }
            });
        }

        // Chat is the app's main way in, so it is a full-height column
        // in every view. egui reserves its space first, so the grid
        // and graph fill the rest and can never push it off screen.
        let max_chat = (ui_root.available_width() - DATA_MIN_WIDTH).max(CHAT_PANEL_MIN_WIDTH);
        egui::Panel::right("chat")
            .resizable(true)
            .default_size(CHAT_PANEL_DEFAULT_WIDTH.min(max_chat))
            .min_size(CHAT_PANEL_MIN_WIDTH)
            .max_size(max_chat)
            .show(ui_root, |ui| self.chat_section(ui));
        egui::CentralPanel::default().show(ui_root, |ui| {
            // Above both panes, so find works in the Graph view too
            // (a match selects its node there).
            if self.find_open {
                self.find_bar(ui);
            }
            self.data_panes(ui);
        });
    }
}

impl GridTalkApp {
    /// The grid, the graph, or both, as the view mode says.
    fn data_panes(&mut self, ui: &mut egui::Ui) {
        match self.sync.mode {
            ViewMode::GridFull => self.grid_pane(ui),
            ViewMode::GraphFull => self.graph_pane(ui),
            ViewMode::Split => {
                // The graph sits under the grid, resizable, and never
                // squeezes the grid below its minimum.
                let max_graph =
                    (ui.available_height() - GRID_MIN_HEIGHT).max(GRAPH_PANEL_MIN_HEIGHT);
                egui::Panel::bottom("graph")
                    .resizable(true)
                    .default_size(
                        (ui.available_height() * 0.4).clamp(GRAPH_PANEL_MIN_HEIGHT, max_graph),
                    )
                    .min_size(GRAPH_PANEL_MIN_HEIGHT)
                    .max_size(max_graph)
                    .show(ui, |ui| self.graph_pane(ui));
                self.grid_pane(ui);
            }
        }
    }

    fn grid_pane(&mut self, ui: &mut egui::Ui) {
        // Collect edits/clicks while rendering; apply after.
        let mut pending_edit: Option<CellEdit> = None;
        // An editor that lost focus ends its typing session for undo.
        let mut typing_ended = false;
        let mut focused_editor: Option<egui::Id> = None;
        let mut clicked_cell: Option<(String, usize, usize)> = None;
        // Sheet tabs need the sheet list; take names from the cached
        // sheet plus the snapshot sheet count via inc (names only).
        let sheet_names: Vec<String> = self
            .inc
            .as_ref()
            .map(|inc| {
                inc.snapshot()
                    .sheets
                    .iter()
                    .map(|s| s.name.clone())
                    .collect()
            })
            .unwrap_or_default();
        if sheet_names.len() > 1 {
            ui.horizontal(|ui| {
                for (i, name) in sheet_names.iter().enumerate() {
                    if ui
                        .selectable_label(self.selected_sheet == i, name)
                        .clicked()
                        && self.selected_sheet != i
                    {
                        self.selected_sheet = i;
                        // Selection belongs to the previous sheet.
                        self.sync.selected_range = None;
                        self.sync.highlighted_node = None;
                    }
                }
            });
        }
        if let Some(sheet) = &self.frame_sheet {
            let sheet_idx = self.frame_sheet_idx;
            let sheet_name = sheet.name.clone();
            ui.heading(&sheet_name);
            let selected = self.sync.selected_range.clone();
            let max_col = sheet.max_col();
            let display_cols = max_col + 2;
            // A few editable rows/columns beyond the data rectangle
            // so users can append data instead of being fenced in.
            let display_rows = sheet.cells.len() + 3;
            let row_height = 22.0;
            // Header + rows live inside the same scrolling area so
            // column letters stay aligned. Both axes are
            // virtualised: only the visible row/column window is
            // built, with spacers preserving the scroll extent.
            let col_w = 90.0f32;
            let read_only = self.is_read_only();
            let show_formulas = self.prefs.show_formulas;
            // Visible formula cells without a result for this revision
            // are sent to the evaluation worker after this frame; until
            // then they keep the previous result (or "…").
            let revision = self.frame_revision;
            let values = &self.values;
            let mut value_misses: Vec<(usize, usize)> = Vec::new();
            let total_w = 34.0 + display_cols as f32 * col_w;
            let total_h = (display_rows + 1) as f32 * row_height;
            if selected.is_none() {
                self.scrolled_for = None;
            }
            let scroll_target = selected.clone().filter(|sel| {
                sel.sheet.eq_ignore_ascii_case(&sheet_name)
                    && self.scrolled_for.as_deref() != Some(sel.a1().as_str())
            });
            egui::ScrollArea::both()
                .id_salt("grid")
                .auto_shrink([false, false])
                .show_viewport(ui, |ui, viewport| {
                    ui.set_min_size(egui::vec2(total_w, total_h));
                    let first = ((viewport.min.y / row_height).floor().max(0.0) as usize)
                        .min(display_rows + 1);
                    let last = ((viewport.max.y / row_height).ceil().max(0.0) as usize)
                        .min(display_rows + 1);
                    let c_first = (((viewport.min.x - 34.0).max(0.0) / col_w).floor() as usize)
                        .min(display_cols);
                    let c_last = (((viewport.max.x - 34.0).max(0.0) / col_w).ceil() as usize + 1)
                        .min(display_cols);
                    if first > 0 {
                        ui.add_space(first as f32 * row_height);
                    }
                    for r_plus in first..last {
                        if r_plus == 0 {
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [34.0, row_height],
                                    egui::Label::new(egui::RichText::new("").strong()),
                                );
                                if c_first > 0 {
                                    ui.allocate_space(egui::vec2(
                                        c_first as f32 * col_w,
                                        row_height,
                                    ));
                                }
                                for c in c_first..c_last {
                                    ui.add_sized(
                                        [col_w, row_height],
                                        egui::Label::new(
                                            egui::RichText::new(gridtalk_graph::col_name(c))
                                                .strong(),
                                        ),
                                    );
                                }
                            });
                            continue;
                        }
                        let r = r_plus - 1;
                        ui.horizontal(|ui| {
                            ui.add_sized(
                                [34.0, row_height],
                                egui::Label::new(
                                    egui::RichText::new(format!("{}", r + 1)).strong(),
                                ),
                            );
                            if c_first > 0 {
                                ui.allocate_space(egui::vec2(c_first as f32 * col_w, row_height));
                            }
                            for c in c_first..c_last {
                                let cell = sheet.get(r, c);
                                let is_selected = selected
                                    .as_ref()
                                    .map(|s| {
                                        s.sheet.eq_ignore_ascii_case(&sheet_name)
                                            && s.contains(r, c)
                                    })
                                    .unwrap_or(false);
                                let is_formula =
                                    matches!(cell, gridtalk_graph::CellValue::Formula { .. });
                                // Index key: no sheet-name String
                                // allocated per visible cell per frame.
                                let key = (sheet_idx, r, c);
                                let editor_id = egui::Id::unique(("grid-cell", key));
                                // A cell being typed into stays an
                                // editor even once its debounced edit
                                // lands as a formula.
                                let editing = ui.memory(|m| m.has_focus(editor_id));
                                if editing {
                                    focused_editor = Some(editor_id);
                                }
                                let show_value = is_formula && !show_formulas && !editing;
                                if read_only || show_value {
                                    // Not editable here: a clickable,
                                    // selectable display cell.
                                    let (text, error): (Cow<str>, Option<&str>) = if show_value {
                                        let result = match values.get(key, revision) {
                                            Lookup::Fresh(v) => Some(v),
                                            Lookup::Stale(v) => {
                                                value_misses.push((r, c));
                                                Some(v)
                                            }
                                            Lookup::Missing => {
                                                value_misses.push((r, c));
                                                None
                                            }
                                        };
                                        match result {
                                            Some(Ok(v)) => (Cow::Borrowed(v.as_str()), None),
                                            Some(Err(e)) => {
                                                (Cow::Borrowed("#ERR"), Some(e.as_str()))
                                            }
                                            None => (Cow::Borrowed("…"), None),
                                        }
                                    } else {
                                        (Cow::Owned(cell_input_text(&cell)), None)
                                    };
                                    let resp = display_cell(
                                        ui,
                                        egui::vec2(col_w, row_height),
                                        &text,
                                        is_selected,
                                        is_formula,
                                        error.is_some(),
                                    );
                                    let resp = match error {
                                        Some(e) => resp.on_hover_text(e),
                                        None if show_value && !read_only => resp.on_hover_text(
                                            "Formula result. Switch to \"Showing: Formulas\" to edit.",
                                        ),
                                        None => resp,
                                    };
                                    if resp.clicked() {
                                        clicked_cell = Some((sheet_name.clone(), r, c));
                                    }
                                    continue;
                                }
                                let buffer = self
                                    .edit_buffers
                                    .entry(key)
                                    .or_insert_with(|| cell_input_text(&cell));
                                let editor =
                                    egui::TextEdit::singleline(buffer).id(editor_id).text_color(if is_selected {
                                        egui::Color32::YELLOW
                                    } else {
                                        egui::Color32::LIGHT_GRAY
                                    });
                                let resp = if is_selected {
                                    // A filled background so an
                                    // empty selected cell is visible.
                                    let (rect, _) = ui.allocate_exact_size(
                                        egui::vec2(col_w, row_height),
                                        egui::Sense::hover(),
                                    );
                                    ui.painter().rect_filled(
                                        rect,
                                        2.0,
                                        egui::Color32::from_rgb(74, 62, 0),
                                    );
                                    ui.put(rect, editor)
                                } else {
                                    ui.add_sized([col_w, row_height], editor)
                                };
                                if resp.clicked() {
                                    clicked_cell = Some((sheet_name.clone(), r, c));
                                }
                                if resp.lost_focus() {
                                    typing_ended = true;
                                }
                                if resp.changed() {
                                    pending_edit = Some(CellEdit {
                                        sheet: sheet_name.clone(),
                                        row: r,
                                        col: c,
                                        value: parse_cell_input(buffer),
                                    });
                                }
                            }
                        });
                    }
                    let remaining = display_rows + 1 - last;
                    if remaining > 0 {
                        ui.add_space(remaining as f32 * row_height);
                    }
                    if let Some(sel) = &scroll_target {
                        let origin = ui.min_rect().min;
                        let rect = egui::Rect::from_min_size(
                            origin
                                + egui::vec2(
                                    34.0 + sel.c0 as f32 * col_w,
                                    (sel.r0 as f32 + 1.0) * row_height,
                                ),
                            egui::vec2(col_w, row_height),
                        );
                        ui.scroll_to_rect(rect, Some(egui::Align::Center));
                    }
                });
            if scroll_target.is_some() {
                self.scrolled_for = scroll_target.as_ref().map(|sel| sel.a1());
            }
            // `frame_sheet` is refreshed whenever the revision changes,
            // so the snapshot here is at `revision`.
            if let Some(inc) = self.inc.as_ref().filter(|i| i.revision == revision) {
                self.values.request(
                    inc.snapshot(),
                    revision,
                    sheet_idx,
                    value_misses,
                    Some(ui.ctx().clone()),
                );
            }
        }
        if let Some((sheet, r, c)) = clicked_cell {
            self.sync.select_cell(&self.graph, &sheet, r, c);
        }
        // Read-only cells are not editors, but never record an edit
        // in read-only mode whatever the source.
        let pending_edit = pending_edit.filter(|_| !self.is_read_only());
        if let Some(edit) = pending_edit {
            self.record_grid_edit(edit);
        }
        if typing_ended {
            self.history.end_typing();
        }
        self.grid_editor_focus = focused_editor;
    }

    /// An edit typed into the grid: recorded for undo with the value
    /// it replaces (pending edits included), then queued.
    fn record_grid_edit(&mut self, edit: CellEdit) {
        let Some(inc) = &mut self.inc else {
            return;
        };
        let before = inc.current_value(&edit.sheet, edit.row, edit.col);
        self.history
            .record(&edit.sheet, edit.row, edit.col, before, edit.value.clone());
        inc.record_edit(edit, std::time::Instant::now());
    }

    /// Chat log and input, drawn in the bottom panel under the grid.
    fn chat_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Chat");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(
                        !self.chat.is_empty() && !self.chat_busy,
                        egui::Button::new("Clear"),
                    )
                    .on_hover_text("Remove every message")
                    .clicked()
                {
                    self.chat.clear();
                    self.chat_dropped = 0;
                }
            });
        });
        if !self.workbook_name.is_empty() {
            ui.weak(format!("Ask about {}", self.workbook_name));
        }
        ui.separator();
        let mut clicked_source: Option<RangeRef> = None;
        // Input pinned to the bottom, messages filling the rest: the
        // content always uses the column's full height.
        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
            ui.weak("Enter asks · Shift+Enter adds a line · Ctrl+L focuses");
            let input_id = egui::Id::unique(CHAT_INPUT_ID);
            let ask_enabled = !self.chat_input.trim().is_empty() && !self.chat_busy;
            // Plain Enter asks; take it before the text box would add
            // a line break.
            let enter = ui.memory(|m| m.has_focus(input_id))
                && ui.input_mut(|i| {
                    !i.modifiers.shift && i.consume_key(Modifiers::NONE, Key::Enter)
                });
            let mut ask = false;
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Max), |ui| {
                    if ui
                        .add_enabled(ask_enabled, egui::Button::new("Ask"))
                        .clicked()
                    {
                        ask = true;
                    }
                    let resp = ui.add(
                        egui::TextEdit::multiline(&mut self.chat_input)
                            .id(input_id)
                            .hint_text("Ask a question about this workbook…")
                            .desired_rows(3)
                            .desired_width(f32::INFINITY),
                    );
                    if std::mem::take(&mut self.chat_focus) {
                        resp.request_focus();
                    }
                });
            });
            if (ask || enter) && ask_enabled {
                self.ask();
                ui.memory_mut(|m| m.request_focus(input_id));
            }
            egui::ScrollArea::vertical()
                .id_salt("chat")
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                        self.chat_messages(ui, &mut clicked_source);
                    });
                });
        });
        if let Some(range) = clicked_source {
            self.select_source(&range);
        }
    }

    /// The chat log: each message under who sent it, with links to
    /// the cells an answer cites.
    fn chat_messages(&mut self, ui: &mut egui::Ui, clicked_source: &mut Option<RangeRef>) {
        if self.chat.is_empty() && !self.chat_busy {
            ui.add_space(8.0);
            ui.weak(if self.workbook_name.is_empty() {
                "Open a workbook, then ask about it here."
            } else {
                "Ask about the data, for example \"What is the total of Qty?\". \
                 Answers cite the cells they come from; click one to select it."
            });
        }
        if self.chat_dropped > 0 {
            ui.weak(format!(
                "{} earlier message{} removed: the chat keeps the last {MAX_CHAT}.",
                self.chat_dropped,
                if self.chat_dropped == 1 {
                    " was"
                } else {
                    "s were"
                }
            ));
        }
        for msg in &self.chat {
            let visuals = ui.visuals();
            let (who, colour, fill) = match msg.role.as_str() {
                "you" => ("You", visuals.strong_text_color(), visuals.faint_bg_color),
                "error" => ("Error", visuals.error_fg_color, egui::Color32::TRANSPARENT),
                "gridtalk" => (
                    "GridTalk",
                    visuals.hyperlink_color,
                    egui::Color32::TRANSPARENT,
                ),
                // An answer names the backend that wrote it.
                backend => (backend, visuals.hyperlink_color, egui::Color32::TRANSPARENT),
            };
            egui::Frame::new()
                .fill(fill)
                .corner_radius(4.0)
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(egui::RichText::new(who).strong().color(colour));
                    ui.label(&msg.text);
                    if !msg.sources.is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            ui.weak("Sources:");
                            for range in &msg.sources {
                                if ui
                                    .link(range.a1())
                                    .on_hover_text("Select these cells")
                                    .clicked()
                                {
                                    *clicked_source = Some(range.clone());
                                }
                            }
                        });
                    }
                });
            ui.add_space(4.0);
        }
        if self.chat_busy {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Generating…");
                if ui.button("Cancel").clicked() {
                    if let Some(cancel) = &self.chat_cancel {
                        cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                    self.status = "Cancelling generation…".into();
                }
            });
        }
    }

    fn graph_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph");
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
        // The layout keeps node centres a small margin inside its
        // area, but node buttons are wider than that margin; lay out
        // in an inset area so edge nodes are not cut off.
        let inset = (GRAPH_NODE_SIZE.x / 2.0 - layout::EDGE_MARGIN).max(0.0);
        let layout_rect = rect.shrink2(egui::vec2(inset, 0.0));
        // (Re)spawn layout whenever the graph changed (rebuild
        // clears positions) or on first paint.
        // Respawn when the graph changed OR the pane was resized
        // (positions are absolute to the spawn-time rect).
        let size_changed = (self.layout_size.0 - layout_rect.width()).abs() > 1.0
            || (self.layout_size.1 - layout_rect.height()).abs() > 1.0;
        if (self.positions.len() != self.graph.nodes.len() || size_changed)
            && self.layout_worker.is_none()
        {
            self.layout_size = (layout_rect.width(), layout_rect.height());
            self.layout_worker = Some(spawn_layout(
                self.graph.clone(),
                self.graph_version,
                layout_rect.width(),
                layout_rect.height(),
                Some(ui.ctx().clone()),
            ));
        }
        let painter = ui.painter_at(rect);
        // Node widgets live in a child UI clipped to the canvas. Placed
        // directly with `ui.put`, nodes near the edge extend the
        // parent's used area, and a resizable side panel then grows
        // to fit them every frame (squeezing the grid in Split mode).
        let mut canvas = ui.new_child(egui::UiBuilder::new().max_rect(rect));
        canvas.set_clip_rect(rect.intersect(ui.clip_rect()));
        let origin = layout_rect.min;
        for e in &self.graph.edges {
            if let (Some(a), Some(b)) = (self.positions.get(&e.from), self.positions.get(&e.to)) {
                // Cull edges with both ends off the viewport.
                let pa = egui::pos2(origin.x + a.x, origin.y + a.y);
                let pb = egui::pos2(origin.x + b.x, origin.y + b.y);
                if !rect.contains(pa) && !rect.contains(pb) {
                    continue;
                }
                painter.line_segment(
                    [
                        egui::pos2(origin.x + a.x, origin.y + a.y),
                        egui::pos2(origin.x + b.x, origin.y + b.y),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::DARK_GRAY),
                );
            }
        }
        // Nodes are real widgets: clickable, hoverable (tooltip with
        // the range), and reachable by keyboard (Left/Right cycles
        // nodes when the graph pane is hovered).
        let mut clicked_node: Option<usize> = None;
        if response.hovered() {
            let (next, prev) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::ArrowRight),
                    i.key_pressed(egui::Key::ArrowLeft),
                )
            });
            if (next || prev) && !self.graph.nodes.is_empty() {
                let len = self.graph.nodes.len();
                let current = self
                    .sync
                    .highlighted_node
                    .and_then(|id| self.graph.nodes.iter().position(|n| n.id == id));
                // From nothing highlighted: Right lands on the
                // FIRST node, Left on the last (not index 1 / len-2).
                let idx = match (current, next) {
                    (None, true) => 0,
                    (None, false) => len - 1,
                    (Some(cur), true) => (cur + 1) % len,
                    (Some(cur), false) => (cur + len - 1) % len,
                };
                clicked_node = Some(self.graph.nodes[idx].id);
            }
        }
        for n in &self.graph.nodes {
            if let Some(p) = self.positions.get(&n.id) {
                let centre = egui::pos2(origin.x + p.x, origin.y + p.y);
                let highlighted = self.sync.highlighted_node == Some(n.id);
                let colour = match n.kind {
                    NodeKind::Workbook => egui::Color32::GOLD,
                    NodeKind::Sheet => egui::Color32::LIGHT_BLUE,
                    NodeKind::Region => egui::Color32::LIGHT_GREEN,
                    NodeKind::Column => egui::Color32::GRAY,
                };
                painter.circle_filled(centre, if highlighted { 10.0 } else { 7.0 }, colour);
                if !rect.expand(70.0).contains(centre) {
                    // Off-viewport: no widget this frame.
                    continue;
                }
                let node_rect = egui::Rect::from_center_size(centre, GRAPH_NODE_SIZE);
                let label = if n.label.chars().count() > 16 {
                    format!("{}…", n.label.chars().take(16).collect::<String>())
                } else {
                    n.label.clone()
                };
                let resp = canvas
                    .put(
                        node_rect,
                        egui::Button::new(
                            egui::RichText::new(label)
                                .size(11.0)
                                .color(egui::Color32::BLACK),
                        )
                        .fill(colour)
                        .stroke(if highlighted {
                            egui::Stroke::new(2.0, egui::Color32::YELLOW)
                        } else {
                            egui::Stroke::NONE
                        }),
                    )
                    .on_hover_text(format!(
                        "{:?}: {}{}",
                        n.kind,
                        n.label,
                        n.range
                            .as_ref()
                            .map(|r| format!(" ({})", r.a1()))
                            .unwrap_or_default()
                    ));
                if resp.clicked() {
                    clicked_node = Some(n.id);
                }
            }
        }
        if let Some(id) = clicked_node {
            self.sync.select_node(&self.graph, id);
        }
    }
}

fn parse_cell_input(text: &str) -> CellValue {
    let t = text.trim();
    if t.is_empty() {
        CellValue::Empty
    } else if let Some(expr) = t.strip_prefix('=') {
        CellValue::Formula {
            expr: expr.to_string(),
            cached: None,
        }
    } else if t.eq_ignore_ascii_case("true") {
        CellValue::Bool(true)
    } else if t.eq_ignore_ascii_case("false") {
        CellValue::Bool(false)
    } else if let Ok(n) = t.parse::<f64>() {
        // NaN/inf parse as f64 but are not cell numbers (ingest
        // treats them as text); mirror that.
        if n.is_finite() {
            CellValue::Number(n)
        } else {
            CellValue::Text(t.to_string())
        }
    } else {
        CellValue::Text(t.to_string())
    }
}

fn main() -> eframe::Result<()> {
    let mut app = GridTalkApp::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--read-only") {
        // Locked on for this session only (never written to prefs): an agent or
        // kiosk launch cannot be switched back to editing in the UI.
        app.read_only_locked = true;
    }
    for flag in args
        .iter()
        .filter(|a| a.starts_with("--") && *a != "--read-only")
    {
        eprintln!("gridtalk-ui: warning: ignoring unknown option {flag}");
    }
    if let Some(path) = args.iter().find(|a| !a.starts_with("--")) {
        app.open(std::path::Path::new(path));
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 900.0]),
        ..Default::default()
    };
    let zoom = app.prefs.zoom();
    eframe::run_native(
        "GridTalk",
        options,
        Box::new(move |cc| {
            // GridTalk handles the zoom keys itself, so the zoom is
            // saved and Ctrl+0 returns to GridTalk's default.
            cc.egui_ctx.options_mut(|o| o.zoom_with_keyboard = false);
            cc.egui_ctx.set_zoom_factor(zoom);
            Ok(Box::new(app))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, RawInput};
    use gridtalk_graph::parse_delimited;

    /// An app with `text` open as `in.csv` in a fresh scratch folder.
    fn app_with(tag: &str, text: &str) -> (GridTalkApp, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("gt-ui-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("in.csv");
        std::fs::write(&source, text).unwrap();
        // Prefs go to the scratch folder, never the real ui-state.json.
        let mut app = GridTalkApp::with_prefs(UiPrefs {
            file: Some(dir.join("ui-state.json")),
            ..UiPrefs::default()
        });
        app.open(&source);
        (app, dir)
    }

    /// Press `key` with `modifiers` and let the app handle shortcuts.
    fn press(app: &mut GridTalkApp, ctx: &egui::Context, modifiers: Modifiers, key: Key) {
        ctx.begin_pass(RawInput {
            events: vec![
                Event::ModifiersChanged(modifiers),
                Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers,
                },
            ],
            ..RawInput::default()
        });
        app.handle_shortcuts(ctx);
        // No renderer here: drop the font texture egui uploads.
        ctx.end_pass().textures_delta.clear();
    }

    fn b2(app: &GridTalkApp) -> CellValue {
        app.inc.as_ref().unwrap().current_value("Sheet1", 1, 1)
    }

    fn typed(value: f64) -> CellEdit {
        CellEdit {
            sheet: "Sheet1".into(),
            row: 1,
            col: 1,
            value: CellValue::Number(value),
        }
    }

    #[test]
    fn undo_and_redo_restore_grid_edits() {
        let (mut app, dir) = app_with("undo", "Item,Qty\nApples,2\nPears,4\n");
        // Typing "50" into B2: two keystrokes, one change.
        app.record_grid_edit(typed(5.0));
        app.record_grid_edit(typed(50.0));
        app.history.end_typing();
        app.sync.selected_range = None;
        app.undo();
        assert_eq!(b2(&app), CellValue::Number(2.0));
        assert_eq!(
            app.edit_buffers.get(&(0, 1, 1)).map(String::as_str),
            Some("2")
        );
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!B2:B2"
        );
        assert!(app.status.contains("Sheet1!B2"), "{}", app.status);
        app.redo();
        assert_eq!(b2(&app), CellValue::Number(50.0));
        app.undo();
        app.undo();
        assert_eq!(app.status, "Nothing to undo.");
        // Read-only: undo changes nothing.
        app.redo();
        app.prefs.read_only = true;
        app.undo();
        assert_eq!(b2(&app), CellValue::Number(50.0));
        assert!(app.status.contains("Read-only"), "{}", app.status);
        // A new workbook starts with no history.
        app.prefs.read_only = false;
        // The second Open confirms discarding the unsaved edits.
        app.open(&dir.join("in.csv"));
        app.open(&dir.join("in.csv"));
        app.undo();
        assert_eq!(app.status, "Nothing to undo.");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn keyboard_shortcuts() {
        let (mut app, dir) = app_with("keys", "Item,Qty\nApples,2\n");
        let ctx = egui::Context::default();
        app.record_grid_edit(typed(7.0));
        app.history.end_typing();
        press(&mut app, &ctx, Modifiers::COMMAND, Key::Z);
        assert_eq!(b2(&app), CellValue::Number(2.0));
        press(
            &mut app,
            &ctx,
            Modifiers::COMMAND | Modifiers::SHIFT,
            Key::Z,
        );
        assert_eq!(b2(&app), CellValue::Number(7.0));
        press(&mut app, &ctx, Modifiers::COMMAND, Key::Z);
        press(&mut app, &ctx, Modifiers::COMMAND, Key::Y);
        assert_eq!(b2(&app), CellValue::Number(7.0));
        // Ctrl+F opens the find bar, Escape closes it.
        press(&mut app, &ctx, Modifiers::COMMAND, Key::F);
        assert!(app.find_open && app.find_focus);
        press(&mut app, &ctx, Modifiers::NONE, Key::Escape);
        assert!(!app.find_open);
        // Ctrl+S saves the open file.
        press(&mut app, &ctx, Modifiers::COMMAND, Key::S);
        assert!(app.status.starts_with("Saved"), "{}", app.status);
        let saved = std::fs::read_to_string(dir.join("in.csv")).unwrap();
        assert!(saved.contains("Apples,7"), "{saved}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_steps_through_matches_across_sheets() {
        let (mut app, dir) = app_with("find", "Item,Qty\nApples,2\nPears,4\nApple pie,1\n");
        app.find_query = "apple".into();
        app.find_step(true);
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!A2:A2"
        );
        assert_eq!(app.status, "Match 1 of 2.");
        app.find_step(true);
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!A4:A4"
        );
        app.find_step(true);
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!A2:A2"
        );
        app.find_step(false);
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!A4:A4"
        );
        app.find_query = "zebra".into();
        app.find_step(true);
        assert_eq!(app.status, "No cell contains \"zebra\".");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_carries_on_from_the_selected_match_after_an_edit() {
        let (mut app, dir) = app_with("find-edit", "Item,Qty\nfig,1\nfig,2\nfig,3\n");
        app.find_query = "fig".into();
        app.find_step(true);
        app.find_step(true);
        assert_eq!(app.status, "Match 2 of 3.");
        // An edit elsewhere re-runs the search; Next goes on to the third.
        let inc = app.inc.as_mut().unwrap();
        inc.record_edit(typed(9.0), std::time::Instant::now());
        inc.flush();
        app.find_step(true);
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!A4:A4"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn chat_history_pairs_questions_with_their_answers() {
        let msg = |role: &str, text: &str| ChatMsg {
            role: role.into(),
            text: text.into(),
            sources: Vec::new(),
        };
        let chat = vec![
            msg("you", "What does Sheet1 hold?"),
            msg("echo-local", "Items and totals."),
            msg("you", "and the Total column?"),
            msg("error", "backend down"),
            msg("you", "still waiting"),
        ];
        let turns = chat_history(&chat);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].question, "What does Sheet1 hold?");
        assert_eq!(turns[0].answer, "Items and totals.");
        assert_eq!(turns[1].answer, "backend down");
        // A question followed straight by another (its reply never
        // came) is left out, not paired with a blank answer.
        let unanswered = vec![
            msg("you", "timed out"),
            msg("you", "What is Total?"),
            msg("echo-local", "Qty times Price."),
        ];
        let kept = chat_history(&unanswered);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].question, "What is Total?");
        // And the follow-up condenses against them.
        assert!(
            gridtalk_index::standalone_query("and what reads it?", &turns)
                .contains("What does Sheet1 hold?")
        );
    }

    #[test]
    fn selection_context_names_neighbours_and_cells() {
        let wb = gridtalk_graph::parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\n",
            ',',
        );
        let (graph, violations) = gridtalk_graph::build_graph(&wb);
        assert!(violations.is_empty());
        let sel = RangeRef {
            sheet: "Sheet1".into(),
            r0: 0,
            c0: 3,
            r1: 1,
            c1: 3,
        };
        let text = selection_context(&graph, &wb, &sel);
        assert!(text.contains("selected Sheet1!D1:D2"), "{text}");
        assert!(text.contains("depends_on"), "{text}");
        assert!(text.contains("Qty"), "{text}");
        assert!(text.contains("Total"), "{text}");
        assert!(!text.contains("cut"), "{text}");
        // A selection on no sheet and no node is no context at all.
        let gone = RangeRef {
            sheet: "Gone".into(),
            ..sel.clone()
        };
        assert_eq!(selection_context(&graph, &wb, &gone), "");
        // A selection longer than the 100-row cap says it was cut.
        let rows: String = (0..250).map(|i| format!("{i}\n")).collect();
        let tall = gridtalk_graph::parse_delimited("tall", "Sheet1", &format!("N\n{rows}"), ',');
        let (graph, _) = gridtalk_graph::build_graph(&tall);
        let sel = RangeRef {
            sheet: "Sheet1".into(),
            r0: 0,
            c0: 0,
            r1: 250,
            c1: 0,
        };
        let text = selection_context(&graph, &tall, &sel);
        assert!(text.contains("selection cut"), "{text}");
        // And one cut mid-row by the 300-cell cap keeps that row's
        // cells read so far.
        let wide: String = (0..7)
            .map(|c| format!("c{c}"))
            .collect::<Vec<_>>()
            .join(",");
        let body: String = (0..50)
            .map(|_| format!("{}\n", ["1"; 7].join(",")))
            .collect();
        let wide_wb =
            gridtalk_graph::parse_delimited("wide", "Sheet1", &format!("{wide}\n{body}"), ',');
        let (graph, _) = gridtalk_graph::build_graph(&wide_wb);
        let sel = RangeRef {
            sheet: "Sheet1".into(),
            r0: 0,
            c0: 0,
            r1: 50,
            c1: 6,
        };
        let text = selection_context(&graph, &wide_wb, &sel);
        assert!(text.contains("selection cut"), "{text}");
        // 300 cells = 42 full rows of 7 plus 6 cells of the 43rd.
        let cell_lines: Vec<&str> = text
            .lines()
            .skip_while(|l| *l != "cells:")
            .skip(1)
            .filter(|l| !l.starts_with("..."))
            .collect();
        assert_eq!(cell_lines.len(), 43, "{text}");
        assert_eq!(cell_lines[42].split(" | ").count(), 6, "{text}");
    }

    #[test]
    fn chat_sources_select_their_range_and_dropped_messages_are_counted() {
        let (mut app, dir) = app_with("chat", "Item,Qty\nApples,2\n");
        let range = RangeRef {
            sheet: "sheet1".into(),
            r0: 0,
            c0: 0,
            r1: 1,
            c1: 1,
        };
        app.push_chat_with_sources("echo", "B2 is 2".into(), vec![range.clone()]);
        assert_eq!(app.chat[0].sources, vec![range.clone()]);
        app.select_source(&range);
        assert_eq!(app.sync.selected_range.as_ref(), Some(&range));
        assert!(app.sync.highlighted_node.is_some());
        app.select_source(&RangeRef {
            sheet: "Gone".into(),
            ..range
        });
        assert!(
            app.status.contains("not in this workbook"),
            "{}",
            app.status
        );
        for i in 0..MAX_CHAT + 4 {
            app.push_chat("you", format!("q{i}"));
        }
        assert_eq!(app.chat.len(), MAX_CHAT);
        assert_eq!(app.chat_dropped, 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_save_the_backend_to_the_config_in_use() {
        let (mut app, dir) = app_with("backend", "A\n1\n");
        let config = dir.join("config.toml");
        std::fs::write(
            &config,
            "[inference]\nn_ctx = 4096\n\n[ui]\ntheme = \"dark\"\n",
        )
        .unwrap();
        app.config_path = config.display().to_string();
        app.load_llm_settings();
        assert_eq!(app.set_backend, "local");
        app.set_backend = "openai-compatible".into();
        app.set_base_url = "http://203.0.113.5/v1".into();
        app.set_model = "qwen".into();
        app.set_api_key_env = "MY_KEY".into();
        // An unsaved context-size edit in the model section.
        app.llm_n_ctx = Some(8192);
        app.save_backend_settings();
        assert!(app.status.starts_with("Backend saved"), "{}", app.status);
        let text = std::fs::read_to_string(&config).unwrap();
        for line in [
            "backend = \"openai-compatible\"",
            "base_url = \"http://203.0.113.5/v1\"",
            "model = \"qwen\"",
            "api_key_env = \"MY_KEY\"",
            "n_ctx = 4096",
            "theme = \"dark\"",
        ] {
            assert!(text.contains(line), "{line} missing:\n{text}");
        }
        // The plain-http warning follows the saved backend; unsaved
        // model settings are kept.
        assert!(app.llm_cleartext.is_some());
        assert_eq!(app.llm_n_ctx, Some(8192));
        // A local backend needs a .gguf file that exists.
        app.set_backend = "local".into();
        app.set_model_path = dir.join("missing.gguf").display().to_string();
        app.save_backend_settings();
        assert!(
            app.status.starts_with("Backend not saved"),
            "{}",
            app.status
        );
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        app.set_model_path = model.display().to_string();
        app.save_backend_settings();
        let text = std::fs::read_to_string(&config).unwrap();
        assert!(text.contains("backend = \"local\""), "{text}");
        assert!(text.contains("model_path = "), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ui_two_way_sync_and_mode_persistence() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,2\n", ',');
        let (g, _) = gridtalk_graph::build_graph(&wb);
        let mut sync = SyncState::default();
        let region = g.nodes.iter().find(|n| n.kind == NodeKind::Region).unwrap();
        sync.select_node(&g, region.id);
        assert_eq!(sync.selected_range.as_ref().unwrap().a1(), "Sheet1!A1:B2");
        sync.select_cell(&g, "Sheet1", 1, 1);
        assert!(sync.highlighted_node.is_some());

        let mut prefs = UiPrefs::default();
        prefs
            .mode_by_workbook
            .insert("d".into(), ViewMode::GraphFull);
        let json = serde_json::to_string(&prefs).unwrap();
        let back: UiPrefs = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mode_by_workbook.get("d"), Some(&ViewMode::GraphFull));
    }

    #[test]
    fn wizard_config_round_trips_through_parser() {
        let text = "[inference]\nbackend = \"local\"\nengine = \"llamacpp\"\nmodel_path = \"/models/m.gguf\"\n";
        let cfg = gridtalk_infer::config::Config::parse(text);
        assert_eq!(cfg.inference.backend, "local");
        assert_eq!(cfg.inference.model_path.as_deref(), Some("/models/m.gguf"));
        // First-run default: wizard not yet done.
        assert!(!UiPrefs::default().first_run_done);
    }

    #[test]
    fn regression_formula_cells_edit_as_formulas_and_save_round_trips() {
        // Editing must present =B2*C2, never the cached value.
        let formula = gridtalk_graph::CellValue::Formula {
            expr: "B2*C2".into(),
            cached: Some(Box::new(gridtalk_graph::CellValue::Number(6.0))),
        };
        assert_eq!(cell_input_text(&formula), "=B2*C2");
        // Save serialisation keeps the formula for re-open.
        let wb = parse_delimited("d", "Sheet1", "A,B\n2,=A2*3\n", ',');
        let csv = snapshot_to_delimited(&wb, ',');
        assert!(csv.contains("=A2*3"), "saved CSV must keep formulas: {csv}");
        let back = parse_delimited("d", "Sheet1", &csv, ',');
        assert!(matches!(
            back.sheets[0].get(1, 1),
            gridtalk_graph::CellValue::Formula { .. }
        ));
        // CR and edge whitespace survive a CSV round trip.
        let wb = gridtalk_graph::WorkbookSnapshot {
            name: "d".into(),
            sheets: vec![gridtalk_graph::SheetSnapshot::new(
                "Sheet1",
                vec![vec![
                    gridtalk_graph::CellValue::Text("a\rb".into()),
                    gridtalk_graph::CellValue::Text("  pad ".into()),
                ]],
            )],
            names: Vec::new(),
            warnings: Vec::new(),
            source: None,
        };
        let back = parse_delimited("d", "Sheet1", &snapshot_to_delimited(&wb, ','), ',');
        assert_eq!(back.sheets[0].cells, wb.sheets[0].cells);
    }

    #[test]
    fn wizard_validation_and_save_flush() {
        assert!(validate_wizard("local", "", "").is_err());
        assert!(validate_wizard("local", "/definitely/missing.gguf", "").is_err());
        assert!(validate_wizard("openai-compatible", "", "ftp://x").is_err());
        assert!(validate_wizard("openai-compatible", "", "https://api.example.com/v1").is_ok());
        assert!(validate_wizard("openai-compatible", "", "http://127.0.0.1:1/v1").is_ok());

        // Save must flush a pending (not yet debounced) edit.
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,2\n", ',');
        let mut app = GridTalkApp::with_prefs(UiPrefs::default());
        let mut inc = IncrementalSnapshot::new(wb);
        inc.record_edit(
            CellEdit {
                sheet: "Sheet1".into(),
                row: 1,
                col: 0,
                value: CellValue::Number(99.0),
            },
            std::time::Instant::now(),
        );
        let dir = std::env::temp_dir().join(format!("gt-ui-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.csv");
        app.inc = Some(inc);
        app.current_path = Some(path.clone());
        app.save();
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("99"), "pending edit must be saved: {saved}");
        // Multi-sheet CSV save is an explicit error, never silent loss.
        let mut wb2 = parse_delimited("d", "Sheet1", "A\n1\n", ',');
        wb2.sheets.push(gridtalk_graph::SheetSnapshot::new(
            "Second",
            vec![vec![CellValue::Number(7.0)]],
        ));
        app.inc = Some(IncrementalSnapshot::new(wb2));
        app.current_path = Some(path.clone());
        app.save();
        assert!(app.status.contains("sheets"), "status: {}", app.status);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cell_input_parsing_matches_ingest_extended() {
        use gridtalk_graph::CellValue;
        assert_eq!(parse_cell_input("true"), CellValue::Bool(true));
        assert_eq!(parse_cell_input("FALSE"), CellValue::Bool(false));
        assert!(matches!(parse_cell_input("NaN"), CellValue::Text(_)));
        assert!(matches!(parse_cell_input("inf"), CellValue::Text(_)));
        assert!(matches!(parse_cell_input("-infinity"), CellValue::Text(_)));
    }

    #[test]
    fn cell_input_parsing_matches_ingest() {
        assert_eq!(parse_cell_input(""), CellValue::Empty);
        assert_eq!(parse_cell_input("42"), CellValue::Number(42.0));
        assert!(matches!(
            parse_cell_input("=B2*C2"),
            CellValue::Formula { .. }
        ));
        assert_eq!(parse_cell_input("hi"), CellValue::Text("hi".into()));
    }

    #[test]
    fn inference_config_merge_sets_removes_and_keeps_the_rest() {
        let existing =
            "# mine\n[inference]\nbackend = \"local\"\nn_gpu_layers = 10\n\n[other]\nx = 1\n";
        let merged = merge_inference_config(
            existing,
            &[("n_gpu_layers", None), ("n_ctx", Some("8192".into()))],
        );
        let cfg = Config::parse(&merged);
        assert_eq!(cfg.inference.backend, "local");
        assert_eq!(cfg.inference.n_gpu_layers, None);
        assert_eq!(cfg.inference.n_ctx, Some(8192));
        assert!(merged.starts_with("# mine\n") && merged.contains("[other]\nx = 1"));
        // Numbers stay numbers: a quoted value would fail to parse.
        assert!(merged.contains("n_ctx = 8192\n"));
        // No [inference] table yet: one is added.
        let fresh =
            merge_inference_config("", &[("model_path", Some(toml_string("C:\\m \"q\".gguf")))]);
        let cfg = Config::parse(&fresh);
        assert_eq!(
            cfg.inference.model_path.as_deref(),
            Some("C:\\m \"q\".gguf")
        );
        // Only removals on an empty file write nothing.
        assert_eq!(merge_inference_config("", &[("n_ctx", None)]), "\n");
    }

    #[test]
    fn zoom_keys_scale_the_ui_and_the_zoom_is_saved() {
        let (mut app, dir) = app_with("zoom", "Item,Qty\nApples,2\n");
        let ctx = egui::Context::default();
        // A new zoom takes effect when the next pass begins.
        let zoom = |ctx: &egui::Context| {
            ctx.begin_pass(RawInput::default());
            ctx.end_pass().textures_delta.clear();
            ctx.zoom_factor()
        };
        ctx.set_zoom_factor(app.prefs.zoom());
        assert_eq!(zoom(&ctx), DEFAULT_ZOOM);
        press(&mut app, &ctx, Modifiers::COMMAND, Key::Equals);
        assert_eq!(zoom(&ctx), 1.35);
        // Ctrl+Plus arrives with Shift on many layouts.
        press(
            &mut app,
            &ctx,
            Modifiers::COMMAND | Modifiers::SHIFT,
            Key::Plus,
        );
        assert_eq!(zoom(&ctx), 1.45);
        press(&mut app, &ctx, Modifiers::COMMAND, Key::Minus);
        assert_eq!(zoom(&ctx), 1.35);
        let saved: UiPrefs =
            serde_json::from_str(&std::fs::read_to_string(dir.join("ui-state.json")).unwrap())
                .unwrap();
        assert_eq!(saved.zoom(), 1.35);
        // Ctrl+0 returns to GridTalk's default, not egui's 100%.
        press(&mut app, &ctx, Modifiers::COMMAND, Key::Num0);
        assert_eq!(zoom(&ctx), DEFAULT_ZOOM);
        // Zoom stays within bounds.
        for _ in 0..40 {
            press(&mut app, &ctx, Modifiers::COMMAND, Key::Minus);
            zoom(&ctx);
        }
        assert_eq!(zoom(&ctx), MIN_ZOOM);
        // Ctrl+L asks for the chat input's focus.
        press(&mut app, &ctx, Modifiers::COMMAND, Key::L);
        assert!(app.chat_focus);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn saved_zoom_is_clamped_and_defaults_when_missing() {
        let old: UiPrefs = serde_json::from_str(r#"{"mode_by_workbook":{}}"#).unwrap();
        assert_eq!(old.zoom(), DEFAULT_ZOOM);
        let huge: UiPrefs = serde_json::from_str(r#"{"mode_by_workbook":{},"zoom":40}"#).unwrap();
        assert_eq!(huge.zoom(), MAX_ZOOM);
        assert_eq!(clamp_zoom(f32::NAN), DEFAULT_ZOOM);
        assert_eq!(clamp_zoom(1.26), 1.25);
    }

    #[test]
    fn view_and_read_only_prefs_persist_and_default_off() {
        // Older prefs files without the new keys load with both off.
        let old: UiPrefs = serde_json::from_str(r#"{"mode_by_workbook":{}}"#).unwrap();
        assert!(!old.show_formulas && !old.read_only);
        let prefs = UiPrefs {
            show_formulas: true,
            read_only: true,
            ..UiPrefs::default()
        };
        let back: UiPrefs = serde_json::from_str(&serde_json::to_string(&prefs).unwrap()).unwrap();
        assert!(back.show_formulas && back.read_only);
    }

    #[test]
    fn dropped_edits_get_a_status_note() {
        let mut inc = IncrementalSnapshot::new(parse_delimited("d", "Sheet1", "A\n1\n", ','));
        let now = std::time::Instant::now();
        inc.record_edit(
            CellEdit {
                sheet: "Sheet1".into(),
                row: 0,
                col: 0,
                value: CellValue::Number(2.0),
            },
            now,
        );
        assert_eq!(dropped_edit_note(&inc, inc.dropped_edits()), None);
        inc.record_edit(
            CellEdit {
                sheet: "Gone".into(),
                row: 0,
                col: 0,
                value: CellValue::Number(2.0),
            },
            now,
        );
        let before = inc.dropped_edits();
        inc.flush();
        let note = dropped_edit_note(&inc, before).unwrap();
        assert!(note.starts_with("1 edit was not applied"), "{note}");
        assert!(note.contains("unknown sheet Gone"), "{note}");
    }

    #[test]
    fn saving_an_xlsx_sourced_workbook_writes_at_once() {
        let dir = std::env::temp_dir().join(format!("gt-ui-xlsx-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("in.xlsx");
        let wb = parse_delimited("in", "Sheet1", "A\n1\n", ',');
        std::fs::write(&source, gridtalk_graph::to_xlsx_bytes(&wb).unwrap()).unwrap();
        let mut app = GridTalkApp::with_prefs(UiPrefs::default());
        app.open(&source);
        let out = dir.join("out.xlsx");
        app.save_as(&out);
        assert!(out.exists(), "status: {}", app.status);
        assert_eq!(app.current_path.as_deref(), Some(out.as_path()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edits_are_rebuilt_on_a_worker_and_reach_the_index() {
        let dir = std::env::temp_dir().join(format!("gt-ui-worker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("in.csv");
        std::fs::write(&source, "Item,Qty\nApples,2\nPears,4\n").unwrap();
        let mut app = GridTalkApp::with_prefs(UiPrefs::default());
        app.open(&source);
        app.sync.select_cell(&app.graph, "Sheet1", 1, 1);
        let inc = app.inc.as_mut().unwrap();
        inc.record_edit(
            CellEdit {
                sheet: "Sheet1".into(),
                row: 2,
                col: 0,
                value: CellValue::Text("Zebrafruit".into()),
            },
            std::time::Instant::now(),
        );
        inc.flush();
        app.pending_dirty.extend(inc.take_dirty_ranges());
        app.start_rebuild(None);
        assert!(app.index.is_none(), "the index travels with the job");
        assert!(app.poll_rebuild(true));
        assert!(app.rebuild_job.is_none() && app.pending_dirty.is_empty());
        let hits = app.index.as_ref().unwrap().search("Zebrafruit", 3);
        assert!(!hits.is_empty() && !hits[0].vector_only);
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!B2:B2"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_edit_keeps_the_selection_and_an_unchanged_graph_keeps_its_layout() {
        let dir = std::env::temp_dir().join(format!("gt-ui-keep-sel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("in.csv");
        std::fs::write(&source, "Item,Qty\nApples,2\nPears,4\n").unwrap();
        let mut app = GridTalkApp::with_prefs(UiPrefs::default());
        app.open(&source);
        app.sync.select_cell(&app.graph, "Sheet1", 1, 1);
        let highlighted = app.sync.highlighted_node;
        assert!(highlighted.is_some());
        app.positions = app
            .graph
            .nodes
            .iter()
            .map(|n| (n.id, layout::Pos { x: 1.0, y: 2.0 }))
            .collect();
        let version = app.graph_version;
        let edit = |app: &mut GridTalkApp, row, col, value| {
            let inc = app.inc.as_mut().unwrap();
            inc.record_edit(
                CellEdit {
                    sheet: "Sheet1".into(),
                    row,
                    col,
                    value,
                },
                std::time::Instant::now(),
            );
            inc.flush();
            let mut inc = app.inc.take().unwrap();
            let dirty = inc.take_dirty_ranges();
            app.rebuild(inc.snapshot(), Some(dirty));
            app.inc = Some(inc);
        };
        // A value inside the region: same graph, same layout.
        edit(&mut app, 1, 1, CellValue::Number(9.0));
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!B2:B2"
        );
        assert_eq!(app.sync.highlighted_node, highlighted);
        assert_eq!(app.graph_version, version);
        assert_eq!(app.positions.len(), app.graph.nodes.len());
        // A renamed header changes the graph: fresh layout, but the
        // selection stays.
        edit(&mut app, 0, 1, CellValue::Text("Count".into()));
        assert_ne!(app.graph_version, version);
        assert!(app.positions.is_empty());
        assert_eq!(
            app.sync.selected_range.as_ref().unwrap().a1(),
            "Sheet1!B2:B2"
        );
        assert!(app.sync.highlighted_node.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_only_blocks_saving_and_edits() {
        let wb = parse_delimited("d", "Sheet1", "A\n1\n", ',');
        let mut app = GridTalkApp::with_prefs(UiPrefs::default());
        // --read-only locks the mode for the session without
        // touching the persisted preference.
        app.read_only_locked = true;
        assert!(app.is_read_only());
        assert!(!app.prefs.read_only);
        let dir = std::env::temp_dir().join(format!("gt-ui-ro-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.csv");
        app.inc = Some(IncrementalSnapshot::new(wb));
        app.current_path = Some(path.clone());
        app.save();
        app.save_as(&dir.join("copy.csv"));
        assert!(!path.exists() && !dir.join("copy.csv").exists());
        assert!(app.status.contains("Read-only"), "status: {}", app.status);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
