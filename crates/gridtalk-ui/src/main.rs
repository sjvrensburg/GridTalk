//! GridTalk egui shell: spreadsheet grid + chat on one side, the
//! lifted graph on the other, with two-way selection sync. This is
//! the Phase 2/4 front end; it renders GridTalk snapshots directly
//! (the Gridcraft `ui-egui` ribbon remains the long-term editor
//! surface, see docs/decisions/0001).

mod eval_worker;
mod layout;

use eval_worker::{Lookup, ValueCache};
use gridtalk_adapter::{
    adapters_root, cap_prompt, list_adapters, load_adapter, prompt_text, sidecar_rules,
    valid_adapter_name,
};
use gridtalk_graph::{
    build_graph, load_path, CellEdit, CellValue, Graph, IncrementalSnapshot, NodeKind, RangeRef,
};
use gridtalk_index::HybridIndex as Index;
use gridtalk_infer::config::Config;
use gridtalk_infer::{backend_from_config, ChatRequest};
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
/// Initial and minimum height of the chat panel under the grid.
const CHAT_PANEL_DEFAULT_HEIGHT: f32 = 200.0;
const CHAT_PANEL_MIN_HEIGHT: f32 = 90.0;
/// The grid never shrinks below this, however tall chat gets.
const GRID_MIN_HEIGHT: f32 = 120.0;
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

/// Write bytes atomically (unique temp + rename), like prefs save.
fn write_atomic_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write as _;
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    let write = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)?;
        #[cfg(unix)]
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    })();
    match write {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e.to_string())
        }
    }
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
                prefs
            }),
            Err(_) => UiPrefs::default(),
        }
    }
    pub fn save(&self) {
        let p = Self::path();
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
                // Atomic + durable: unique temp, fsync, rename, fsync
                // the directory; clean the temp up on failure.
                let tmp = p.with_extension(format!("json.tmp-{}", std::process::id()));
                let result = (|| -> std::io::Result<()> {
                    use std::io::Write as _;
                    let mut f = std::fs::File::create(&tmp)?;
                    f.write_all(text.as_bytes())?;
                    f.sync_all()?;
                    drop(f);
                    std::fs::rename(&tmp, &p)?;
                    #[cfg(unix)]
                    if let Some(parent) = p.parent() {
                        if let Ok(dir) = std::fs::File::open(parent) {
                            let _ = dir.sync_all();
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = result {
                    let _ = std::fs::remove_file(&tmp);
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

/// A rebuilt graph, its invariant violations, and the index.
type RebuildResult = (Graph, Vec<String>, Option<Index>);

/// Build the graph for `snapshot` and bring `index` up to date with
/// it: incrementally from the edited `dirty` ranges when there is an
/// index to update, else from scratch. A graph with invariant
/// violations gets no index (never serve a stale one).
fn compute_rebuild(
    snapshot: &gridtalk_graph::WorkbookSnapshot,
    index: Option<Index>,
    dirty: Option<&[RangeRef]>,
) -> RebuildResult {
    let (graph, violations) = build_graph(snapshot);
    if !violations.is_empty() {
        return (graph, violations, None);
    }
    let index = match (index, dirty) {
        (Some(mut index), Some(dirty)) => {
            index.update_dirty(snapshot, &graph, dirty);
            index
        }
        _ => Index::build(snapshot, &graph),
    };
    (graph, violations, Some(index))
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
    layout_revision: u64,
    chat_input: String,
    chat: Vec<ChatMsg>,
    status: String,
    show_settings: bool,
    /// llama.cpp settings as edited in the Settings window; `None` =
    /// the backend default (all layers / the model's context).
    llm_gpu_layers: Option<u32>,
    llm_n_ctx: Option<u32>,
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

impl GridTalkApp {
    fn push_chat(&mut self, role: &str, text: String) {
        self.chat.push(ChatMsg {
            role: role.into(),
            text,
        });
        // Cap history so per-frame layout cost stays bounded.
        const MAX_CHAT: usize = 200;
        if self.chat.len() > MAX_CHAT {
            let drop = self.chat.len() - MAX_CHAT;
            self.chat.drain(..drop);
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
        let index = if after_edit {
            self.index.take()
        } else {
            self.index = None;
            None
        };
        let (graph, violations, index) = compute_rebuild(snapshot, index, dirty.as_deref());
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
        let incremental = index.is_some();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(compute_rebuild(&snapshot, index, Some(&dirty)));
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
            Some((graph, violations, index)) => {
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
        let (adapter_prompt, adapter_top_k) = self.active_adapter_prompt(&question);
        let hits = index.search(&question, adapter_top_k);
        if hits.is_empty() {
            self.push_chat(
                "gridtalk",
                format!("No matching region found for \"{question}\"."),
            );
            return;
        }
        if let Some(first) = hits.first() {
            if let Some(node) =
                self.graph
                    .node_for_cell(&first.range.sheet, first.range.r0, first.range.c0)
            {
                let id = node.id;
                self.sync.select_node(&self.graph, id);
            }
        }
        let context = hits
            .iter()
            .map(|h| format!("[{}]\n{}", h.range.a1(), h.passage))
            .collect::<Vec<_>>()
            .join("\n---\n");
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
            let req = ChatRequest {
                system: format!(
                    "You answer questions about a spreadsheet using only the cited context. Always cite the range.{adapter_prompt}"
                ),
                user: format!("Question: {question}\n\nContext:\n{context}"),
                max_tokens: 512,
            };
            let mut out = String::new();
            let msg = match backend.stream(&req, &cancel, &mut |tok| out.push_str(tok)) {
                Ok(()) => (workbook_tag.clone(), backend.name().to_string(), out),
                Err(e) => (workbook_tag.clone(), "error".to_string(), e.to_string()),
            };
            let _ = tx.send(msg);
        });
    }

    /// Load the llama.cpp settings shown in the Settings window from
    /// the config file in use.
    fn load_llm_settings(&mut self) {
        let path = active_config_file(&self.config_path);
        let cfg = Config::load(&path);
        // u32::MAX is how the backend spells "all layers".
        self.llm_gpu_layers = cfg.inference.n_gpu_layers.filter(|&n| n != u32::MAX);
        self.llm_n_ctx = cfg.inference.n_ctx;
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
        let path = if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("xlsx") | Some("csv") | Some("tsv")
        ) {
            path.to_path_buf()
        } else if path.extension().is_none() {
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
        let result: Result<(), String> = match ext.as_str() {
            "xlsx" => gridtalk_graph::to_xlsx_bytes(&snapshot)
                .map_err(|e| e.to_string())
                .and_then(|bytes| write_atomic_bytes(path, &bytes)),
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
            "ods" => Err("ODS is read-only in GridTalk; use Save As to write .xlsx or .csv".into()),
            _ => Err(format!("cannot save .{ext}; use .xlsx or .csv")),
        };
        match result {
            Ok(()) => {
                self.saved_revision = revision;
                self.status = match dropped_note {
                    Some(note) => format!("Saved {}. {note}", path.display()),
                    None => format!("Saved {}", path.display()),
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

/// Hover text for the save buttons: what an .xlsx save keeps.
const SAVE_HOVER: &str = "Saving .xlsx keeps what was read from the opened file (formatting, merges, tables, charts), with your edits applied; anything it could not read was listed when it opened. .csv and .tsv hold one sheet's cells";

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
                        self.push_chat(&role, text);
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
                let snapshot = inc.snapshot().clone();
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
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Spreadsheets", &["csv", "tsv", "xlsx", "ods"])
                        .pick_file()
                    {
                        self.open(&path);
                    } else {
                        self.status = "Open cancelled.".into();
                    }
                }
                let save_label = if self.dirty() { "Save ●" } else { "Save" };
                if ui
                    .add_enabled(!self.is_read_only(), egui::Button::new(save_label))
                    .on_hover_text(SAVE_HOVER)
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
                ui.label(egui::RichText::new(&self.status).small())
                    .on_hover_text(&self.status);
            });
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

        match self.sync.mode {
            ViewMode::GridFull => {
                egui::CentralPanel::default().show(ui_root, |ui| {
                    self.grid_and_chat(ui);
                });
            }
            ViewMode::GraphFull => {
                egui::CentralPanel::default().show(ui_root, |ui| {
                    self.graph_pane(ui);
                });
            }
            ViewMode::Split => {
                egui::Panel::right("graph")
                    .resizable(true)
                    .default_size(420.0)
                    .show(ui_root, |ui| {
                        self.graph_pane(ui);
                    });
                egui::CentralPanel::default().show(ui_root, |ui| {
                    self.grid_and_chat(ui);
                });
            }
        }
    }
}

impl GridTalkApp {
    fn grid_and_chat(&mut self, ui: &mut egui::Ui) {
        // Chat is a resizable bottom panel; egui reserves its space
        // before the grid lays out, so the grid fills the rest and
        // can never push chat off screen.
        let max_chat = (ui.available_height() - GRID_MIN_HEIGHT).max(CHAT_PANEL_MIN_HEIGHT);
        egui::Panel::bottom("chat")
            .resizable(true)
            .default_size(CHAT_PANEL_DEFAULT_HEIGHT.min(max_chat))
            .min_size(CHAT_PANEL_MIN_HEIGHT)
            .max_size(max_chat)
            .show(ui, |ui| self.chat_section(ui));
        // Collect edits/clicks while rendering; apply after.
        let mut pending_edit: Option<CellEdit> = None;
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
        if let (Some(edit), Some(inc)) = (pending_edit, &mut self.inc) {
            inc.record_edit(edit, std::time::Instant::now());
        }
    }

    /// Chat log and input, drawn in the bottom panel under the grid.
    fn chat_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Chat");
        // Input row pinned to the bottom, log filling the rest: the
        // content always uses the panel's full height, so egui never
        // shrinks the panel to fit it.
        ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
            ui.horizontal(|ui| {
                let resp = ui.text_edit_singleline(&mut self.chat_input);
                let ask_enabled = !self.chat_input.is_empty() && !self.chat_busy;
                if (ui
                    .add_enabled(ask_enabled, egui::Button::new("Ask"))
                    .clicked()
                    || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))))
                    && ask_enabled
                {
                    self.ask();
                }
            });
            egui::ScrollArea::vertical()
                .id_salt("chat")
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &self.chat {
                        ui.label(format!("{}: {}", msg.role, msg.text));
                    }
                    if self.chat_busy {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Generating… (on a worker thread; the grid stays live)");
                            if ui.button("Cancel").clicked() {
                                if let Some(cancel) = &self.chat_cancel {
                                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                self.status = "Cancelling generation…".into();
                            }
                        });
                    }
                });
        });
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
        viewport: egui::ViewportBuilder::default().with_inner_size([1180.0, 760.0]),
        ..Default::default()
    };
    eframe::run_native("GridTalk", options, Box::new(|_cc| Ok(Box::new(app))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::parse_delimited;

    #[test]
    fn ui_two_way_sync_and_mode_persistence() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,2\n", ',');
        let (g, _) = build_graph(&wb);
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
            sheets: vec![gridtalk_graph::SheetSnapshot {
                name: "Sheet1".into(),
                cells: vec![vec![
                    gridtalk_graph::CellValue::Text("a\rb".into()),
                    gridtalk_graph::CellValue::Text("  pad ".into()),
                ]],
            }],
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
        wb2.sheets.push(gridtalk_graph::SheetSnapshot {
            name: "Second".into(),
            cells: vec![vec![CellValue::Number(7.0)]],
        });
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
