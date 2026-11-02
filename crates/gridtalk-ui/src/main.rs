//! GridTalk egui shell: spreadsheet grid + chat on one side, the
//! lifted graph on the other, with two-way selection sync. This is
//! the Phase 2/4 front end; it renders GridTalk snapshots directly
//! (the Gridcraft `ui-egui` ribbon remains the long-term editor
//! surface, see docs/decisions/0001).

mod layout;

use gridtalk_graph::{
    build_graph, load_path, CellEdit, CellValue, Graph, IncrementalSnapshot, NodeKind, RangeRef,
};
use gridtalk_index::HybridIndex as Index;
use gridtalk_infer::config::Config;
use gridtalk_infer::{backend_from_config, ChatRequest};
use layout::{spawn_layout, LayoutWorker, Pos};
use serde::{Deserialize, Serialize};
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

/// Per-workbook UI preferences, persisted as JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UiPrefs {
    pub mode_by_workbook: BTreeMap<String, ViewMode>,
    #[serde(default)]
    pub first_run_done: bool,
    #[serde(default)]
    pub config_path: String,
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

impl UiPrefs {
    pub fn path() -> std::path::PathBuf {
        config_dir().join("ui-state.json")
    }
    pub fn load() -> Self {
        match std::fs::read_to_string(Self::path()) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|e| {
                eprintln!("gridtalk-ui: warning: bad ui-state.json ({e}); using defaults");
                UiPrefs::default()
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
                if let Err(e) = std::fs::write(&p, text) {
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
    layout_revision: u64,
    chat_input: String,
    chat: Vec<ChatMsg>,
    status: String,
    show_settings: bool,
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
    frame_revision: u64,
    chat_cancel: Option<Arc<AtomicBool>>,
    chat_workbook: String,
    pending_open: Option<std::path::PathBuf>,
    close_armed: bool,
    graph_version: u64,
    chat_rx: Option<std::sync::mpsc::Receiver<(String, String, String)>>,
    chat_busy: bool,
}

impl Default for GridTalkApp {
    fn default() -> Self {
        let prefs = UiPrefs::load();
        let config_path = if !prefs.config_path.is_empty() {
            prefs.config_path.clone()
        } else {
            std::env::var("GRIDTALK_CONFIG").unwrap_or_default()
        };
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
            config_path,
            current_path: None,
            saved_revision: 0,
            selected_sheet: 0,
            edit_buffers: BTreeMap::new(),
            frame_sheet: None,
            frame_sheet_idx: usize::MAX,
            frame_revision: u64::MAX,
            chat_cancel: None,
            chat_workbook: String::new(),
            pending_open: None,
            close_armed: false,
            graph_version: 0,
            chat_rx: None,
            chat_busy: false,
            wizard_step: 0,
            wizard_backend: "local".into(),
            wizard_model_path: String::new(),
            wizard_base_url: "http://127.0.0.1:11434/v1".into(),
        }
    }
}

/// Write the wizard's choices to the GridTalk config file.
fn write_wizard_config(
    backend: &str,
    model_path: &str,
    base_url: &str,
) -> std::io::Result<std::path::PathBuf> {
    let dir = UiPrefs::path()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("config.toml");
    let esc = |s: &str| {
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace(['\r', '\n'], " ")
    };
    // Merge into any existing config instead of overwriting it:
    // unrelated keys (model, api_key_env, comments) survive.
    let mut updates: Vec<(String, String)> = vec![
        ("backend".into(), esc(backend)),
        ("engine".into(), "llamacpp".into()),
    ];
    if !model_path.is_empty() {
        updates.push(("model_path".into(), esc(model_path)));
    }
    if backend == "openai-compatible" {
        updates.push(("base_url".into(), esc(base_url)));
    }
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut out: Vec<String> = Vec::new();
    let mut in_inference = false;
    let mut seen_section = false;
    let mut applied: Vec<bool> = vec![false; updates.len()];
    for line in existing.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            if in_inference {
                // Leaving [inference]: flush unapplied updates first.
                for (i, (k, v)) in updates.iter().enumerate() {
                    if !applied[i] {
                        out.push(format!("{k} = \"{v}\""));
                        applied[i] = true;
                    }
                }
            }
            in_inference = trimmed == "[inference]";
            seen_section |= in_inference;
        }
        if in_inference {
            if let Some((key, _)) = trimmed.split_once('=') {
                let key = key.trim();
                if let Some(i) = updates.iter().position(|(k, _)| k == key) {
                    out.push(format!("{key} = \"{}\"", updates[i].1));
                    applied[i] = true;
                    continue;
                }
            }
        }
        out.push(line.to_string());
    }
    if !seen_section {
        out.push("[inference]".into());
    }
    for (i, (k, v)) in updates.iter().enumerate() {
        if !applied[i] {
            out.push(format!("{k} = \"{v}\""));
        }
    }
    std::fs::write(&path, out.join("\n") + "\n")?;
    Ok(path)
}

/// Validate wizard inputs before Finish may persist anything.
fn validate_wizard(backend: &str, model_path: &str, base_url: &str) -> Result<(), String> {
    match backend {
        "local" => {
            if model_path.is_empty() {
                return Err("choose a local .gguf model file first".into());
            }
            if !std::path::Path::new(model_path).exists() {
                return Err(format!("model file does not exist: {model_path}"));
            }
            Ok(())
        }
        "openai-compatible" => {
            if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
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
                self.sync.mode = self
                    .prefs
                    .mode_by_workbook
                    .get(&self.workbook_name)
                    .copied()
                    .unwrap_or_default();
                let inc = IncrementalSnapshot::new(wb);
                self.saved_revision = inc.revision;
                self.rebuild(inc.snapshot().clone());
                self.inc = Some(inc);
                self.current_path = Some(path.to_path_buf());
                self.selected_sheet = 0;
                self.edit_buffers.clear();
                self.frame_revision = u64::MAX;
                // Chat (and any in-flight answer) belongs to the old
                // workbook; never land its reply in the new one.
                if let Some(cancel) = &self.chat_cancel {
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                self.chat_cancel = None;
                self.chat.clear();
                self.chat_rx = None;
                self.chat_busy = false;
                self.status = format!("Opened {}", path.display());
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    fn rebuild(&mut self, snapshot: gridtalk_graph::WorkbookSnapshot) {
        let (graph, violations) = build_graph(&snapshot);
        self.graph_version += 1;
        self.positions.clear();
        if let Some(worker) = &self.layout_worker {
            worker.cancel();
        }
        self.layout_worker = None;
        // Selection refers to the previous graph; drop it.
        self.sync.selected_range = None;
        self.sync.highlighted_node = None;
        if violations.is_empty() {
            // Refresh incrementally when possible (unchanged regions
            // keep their embedding vectors).
            match &mut self.index {
                Some(index) => index.update(&snapshot, &graph),
                None => self.index = Some(Index::build(&snapshot, &graph)),
            }
        } else {
            // Never keep serving a stale index for a broken graph.
            self.index = None;
            self.status = format!("Graph invariants violated: {violations:?}");
        }
        self.graph = graph;
        self.layout_revision = u64::MAX; // force fresh layout for the new graph
    }

    fn ask(&mut self) {
        let question = self.chat_input.trim().to_string();
        if question.is_empty() || self.chat_busy {
            return;
        }
        self.chat_input.clear();
        self.push_chat("you", question.clone());
        let Some(index) = &self.index else {
            self.push_chat("gridtalk", "Open a workbook first.".into());
            return;
        };
        let hits = index.search(&question, 3);
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
        // Generation (potentially seconds on a local model) runs on
        // a worker thread; the UI polls the channel each frame.
        std::thread::spawn(move || {
            let cfg = if config_path.is_empty() {
                Config::default()
            } else {
                Config::load(std::path::Path::new(&config_path))
            };
            let backend = backend_from_config(&cfg.inference);
            let req = ChatRequest {
                system: "You answer questions about a spreadsheet using only the cited context. Always cite the range.".into(),
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

    fn save(&mut self) {
        let Some(path) = self.current_path.clone() else {
            self.status = "Nothing to save: open a workbook first, or use Save As.".into();
            return;
        };
        self.save_to(&path, true);
    }

    fn save_as(&mut self, path: &std::path::Path) {
        if self.save_to(path, false) {
            self.current_path = Some(path.to_path_buf());
        }
    }

    /// Flush pending debounced edits first, then write. Returns
    /// true on success.
    fn save_to(&mut self, path: &std::path::Path, _is_current: bool) -> bool {
        // An edit typed <500ms ago must be in the saved file.
        if let Some(inc) = &mut self.inc {
            inc.flush();
        }
        let Some(inc) = &self.inc else { return false };
        let snapshot = inc.snapshot().clone();
        let revision = inc.revision;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let result: Result<(), String> = match ext.as_str() {
            "xlsx" => gridtalk_graph::to_xlsx_bytes(&snapshot)
                .map_err(|e| e.to_string())
                .and_then(|bytes| std::fs::write(path, bytes).map_err(|e| e.to_string())),
            "csv" | "tsv" => {
                if snapshot.sheets.len() > 1 {
                    Err(format!(
                        "workbook has {} sheets but .{ext} holds one sheet; save as .xlsx to keep them all",
                        snapshot.sheets.len()
                    ))
                } else {
                    let delim = if ext == "tsv" { '\t' } else { ',' };
                    let text = snapshot_to_delimited(&snapshot, delim);
                    std::fs::write(path, text).map_err(|e| e.to_string())
                }
            }
            "ods" => Err("ODS is read-only in GridTalk; use Save As to write .xlsx or .csv".into()),
            _ => Err(format!("cannot save .{ext}; use .xlsx or .csv")),
        };
        match result {
            Ok(()) => {
                self.saved_revision = revision;
                self.status = format!("Saved {}", path.display());
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
                if text.contains(delim) || text.contains('"') || text.contains('\n') {
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
        // Closing with unsaved edits needs a second, deliberate close.
        if ctx.input(|i| i.viewport().close_requested()) && self.dirty() && !self.close_armed {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.close_armed = true;
            self.status = "Unsaved edits — close again to exit without saving.".into();
        }
        // Apply debounced cell edits; rebuild graph+index on revision change.
        if let Some(inc) = &mut self.inc {
            if inc.poll(std::time::Instant::now()) {
                let snapshot = inc.snapshot().clone();
                let revision = inc.revision;
                self.rebuild(snapshot);
                self.layout_revision = revision;
                ctx.request_repaint_after(std::time::Duration::from_millis(120));
            } else if inc.pending_edits() > 0 {
                ctx.request_repaint_after(std::time::Duration::from_millis(120));
            }
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
                    let current = snapshot
                        .sheets
                        .get(*sheet_idx)
                        .map(|s| cell_input_text(&s.get(*r, *c)))
                        .unwrap_or_default();
                    &current != buffer
                });
            }
        }
        // Cache only the displayed sheet, refreshed when the
        // revision or selected sheet changes — never a whole-workbook
        // clone per frame.
        if let Some(inc) = &self.inc {
            if inc.revision != self.frame_revision
                || self.selected_sheet != self.frame_sheet_idx
                || self.frame_sheet.is_none()
            {
                self.frame_sheet = inc.snapshot().sheets.get(self.selected_sheet).cloned();
                self.frame_sheet_idx = self.selected_sheet;
                self.frame_revision = inc.revision;
            }
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
                if ui.button(save_label).clicked() {
                    self.save();
                }
                if ui.button("Save As…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .add_filter("Excel workbook", &["xlsx"])
                        .add_filter("CSV", &["csv"])
                        .save_file()
                    {
                        self.save_as(&path);
                    }
                }
                if ui.button("Open in Gridcraft").clicked() {
                    match &self.current_path {
                        Some(path) => {
                            self.status =
                                "Use the CLI: gridtalk open-in-gridcraft <file> (Gridcraft is a companion app; see decision 0003).".into();
                            let _ = path;
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
                if ui.button("Settings").clicked() {
                    self.show_settings = !self.show_settings;
                }
                ui.label(egui::RichText::new(&self.status).small())
                    .on_hover_text(&self.status);
            });
        });

        if !self.prefs.first_run_done {
            let mut finish = false;
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
                                ui.label("Model .gguf path (tokenizer.json must sit beside it):");
                                ui.text_edit_singleline(&mut self.wizard_model_path);
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
                            if ui.button("Finish").clicked() {
                                finish = true;
                            }
                        }
                    }
                });
            if finish {
                if let Err(e) = validate_wizard(
                    &self.wizard_backend,
                    &self.wizard_model_path,
                    &self.wizard_base_url,
                ) {
                    self.status = format!("Setup not saved: {e}");
                    // Stay in the wizard (do not mark first run done).
                    return;
                }
                match write_wizard_config(
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
                }
            }
        }

        if self.show_settings {
            egui::Window::new("Settings").show(&ctx, |ui| {
                ui.label("Config file (blank = local echo backend):");
                let resp = ui.text_edit_singleline(&mut self.config_path);
                // Persist on Enter/focus loss, not on every keystroke,
                // so a half-typed path is never saved mid-edit.
                if resp.lost_focus() {
                    self.prefs.config_path = self.config_path.clone();
                    self.prefs.save();
                }
                ui.label("External endpoints receive cell values verbatim.");
                ui.label("For sensitive workbooks, use a local model.");
                if ui.button("Close").clicked() {
                    self.prefs.config_path = self.config_path.clone();
                    self.prefs.save();
                    self.show_settings = false;
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
            // Header + rows live inside the same horizontally
            // scrolling area so column letters stay aligned.
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show_rows(ui, row_height, display_rows + 1, |ui, row_range| {
                    for r_plus in row_range {
                        if r_plus == 0 {
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [34.0, row_height],
                                    egui::Label::new(egui::RichText::new("").strong()),
                                );
                                for c in 0..display_cols {
                                    ui.add_sized(
                                        [90.0, row_height],
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
                            for c in 0..display_cols {
                                let cell = sheet.get(r, c);
                                let is_selected = selected
                                    .as_ref()
                                    .map(|s| s.sheet == sheet_name && s.contains(r, c))
                                    .unwrap_or(false);
                                // Index key: no sheet-name String
                                // allocated per visible cell per frame.
                                let key = (sheet_idx, r, c);
                                let buffer = self
                                    .edit_buffers
                                    .entry(key)
                                    .or_insert_with(|| cell_input_text(&cell));
                                let resp = ui.add_sized(
                                    [90.0, row_height],
                                    egui::TextEdit::singleline(buffer).text_color(if is_selected {
                                        egui::Color32::YELLOW
                                    } else {
                                        egui::Color32::LIGHT_GRAY
                                    }),
                                );
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
                });
        }
        if let Some((sheet, r, c)) = clicked_cell {
            self.sync.select_cell(&self.graph, &sheet, r, c);
        }
        if let (Some(edit), Some(inc)) = (pending_edit, &mut self.inc) {
            inc.record_edit(edit, std::time::Instant::now());
        }

        ui.separator();
        ui.heading("Chat");
        egui::ScrollArea::vertical()
            .max_height(220.0)
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
    }

    fn graph_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph");
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
        // (Re)spawn layout whenever the graph changed (rebuild
        // clears positions) or on first paint.
        if self.positions.len() != self.graph.nodes.len() && self.layout_worker.is_none() {
            self.layout_worker = Some(spawn_layout(
                self.graph.clone(),
                self.graph_version,
                rect.width(),
                rect.height(),
                Some(ui.ctx().clone()),
            ));
        }
        let painter = ui.painter_at(rect);
        let origin = rect.min;
        for e in &self.graph.edges {
            if let (Some(a), Some(b)) = (self.positions.get(&e.from), self.positions.get(&e.to)) {
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
                let cur = self
                    .sync
                    .highlighted_node
                    .and_then(|id| self.graph.nodes.iter().position(|n| n.id == id))
                    .unwrap_or(0);
                let len = self.graph.nodes.len();
                let idx = if next {
                    (cur + 1) % len
                } else {
                    (cur + len - 1) % len
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
                let node_rect = egui::Rect::from_center_size(centre, egui::vec2(120.0, 18.0));
                let label = if n.label.len() > 16 {
                    format!("{}…", &n.label[..16])
                } else {
                    n.label.clone()
                };
                let resp = ui
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
    } else if let Ok(n) = t.parse::<f64>() {
        CellValue::Number(n)
    } else {
        CellValue::Text(t.to_string())
    }
}

fn main() -> eframe::Result<()> {
    let mut app = GridTalkApp::default();
    if let Some(path) = std::env::args().nth(1) {
        app.open(std::path::Path::new(&path));
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
    }

    #[test]
    fn wizard_validation_and_save_flush() {
        assert!(validate_wizard("local", "", "").is_err());
        assert!(validate_wizard("local", "/definitely/missing.gguf", "").is_err());
        assert!(validate_wizard("openai-compatible", "", "ftp://x").is_err());
        assert!(validate_wizard("openai-compatible", "", "http://127.0.0.1:1/v1").is_ok());

        // Save must flush a pending (not yet debounced) edit.
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,2\n", ',');
        let mut app = GridTalkApp::default();
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
    fn cell_input_parsing_matches_ingest() {
        assert_eq!(parse_cell_input(""), CellValue::Empty);
        assert_eq!(parse_cell_input("42"), CellValue::Number(42.0));
        assert!(matches!(
            parse_cell_input("=B2*C2"),
            CellValue::Formula { .. }
        ));
        assert_eq!(parse_cell_input("hi"), CellValue::Text("hi".into()));
    }
}
