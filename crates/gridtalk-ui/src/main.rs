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

impl UiPrefs {
    pub fn path() -> std::path::PathBuf {
        let base =
            std::env::var("GRIDTALK_CONFIG_DIR").unwrap_or_else(|_| match std::env::var("HOME") {
                Ok(home) if !home.is_empty() => format!("{home}/.config/gridtalk"),
                _ => ".config/gridtalk".to_string(),
            });
        std::path::PathBuf::from(base).join("ui-state.json")
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
    edit_buffers: BTreeMap<(String, usize, usize), String>,
    frame_snapshot: Option<gridtalk_graph::WorkbookSnapshot>,
    frame_revision: u64,
    graph_version: u64,
    chat_rx: Option<std::sync::mpsc::Receiver<(String, String)>>,
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
            frame_snapshot: None,
            frame_revision: u64::MAX,
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
    let mut text = String::from("[inference]\n");
    text.push_str(&format!("backend = \"{}\"\n", esc(backend)));
    text.push_str("engine = \"llamacpp\"\n");
    if !model_path.is_empty() {
        text.push_str(&format!("model_path = \"{}\"\n", esc(model_path)));
    }
    if backend == "openai-compatible" {
        text.push_str(&format!("base_url = \"{}\"\n", esc(base_url)));
    }
    std::fs::write(&path, text)?;
    Ok(path)
}

impl GridTalkApp {
    fn open(&mut self, path: &std::path::Path) {
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
                self.status = format!("Opened {}", path.display());
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    fn rebuild(&mut self, snapshot: gridtalk_graph::WorkbookSnapshot) {
        let (graph, violations) = build_graph(&snapshot);
        self.graph_version += 1;
        self.positions.clear();
        self.layout_worker = None;
        if violations.is_empty() {
            self.index = Some(Index::build(&snapshot, &graph));
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
        self.chat.push(ChatMsg {
            role: "you".into(),
            text: question.clone(),
        });
        let Some(index) = &self.index else {
            self.chat.push(ChatMsg {
                role: "gridtalk".into(),
                text: "Open a workbook first.".into(),
            });
            return;
        };
        let hits = index.search(&question, 3);
        if hits.is_empty() {
            self.chat.push(ChatMsg {
                role: "gridtalk".into(),
                text: format!("No matching region found for \"{question}\"."),
            });
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
            let cancel = Arc::new(AtomicBool::new(false));
            let mut out = String::new();
            let msg = match backend.stream(&req, &cancel, &mut |tok| out.push_str(tok)) {
                Ok(()) => (backend.name().to_string(), out),
                Err(e) => ("error".to_string(), e.to_string()),
            };
            let _ = tx.send(msg);
        });
    }

    fn save(&mut self) {
        let Some(path) = self.current_path.clone() else {
            self.status = "Nothing to save: open a workbook first.".into();
            return;
        };
        let Some(inc) = &self.inc else { return };
        let snapshot = inc.snapshot().clone();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let result: Result<(), String> = match ext.as_str() {
            "xlsx" => gridtalk_graph::to_xlsx_bytes(&snapshot)
                .map_err(|e| e.to_string())
                .and_then(|bytes| std::fs::write(&path, bytes).map_err(|e| e.to_string())),
            "csv" | "tsv" => {
                let delim = if ext == "tsv" { '\t' } else { ',' };
                let text = snapshot_to_delimited(&snapshot, delim);
                std::fs::write(&path, text).map_err(|e| e.to_string())
            }
            "ods" => Err("ODS is read-only in GridTalk; save as .xlsx or .csv".into()),
            _ => Err(format!("cannot save .{ext}; use .xlsx or .csv")),
        };
        match result {
            Ok(()) => {
                self.saved_revision = inc.revision;
                self.status = format!("Saved {}", path.display());
            }
            Err(e) => self.status = format!("Save failed: {e}"),
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
                Ok((role, text)) => {
                    self.chat.push(ChatMsg { role, text });
                    self.chat_rx = None;
                    self.chat_busy = false;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.chat_rx = None;
                    self.chat_busy = false;
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
                self.edit_buffers.retain(|(sheet, r, c), buffer| {
                    let current = snapshot
                        .sheets
                        .iter()
                        .find(|s| &s.name == sheet)
                        .map(|s| cell_input_text(&s.get(*r, *c)))
                        .unwrap_or_default();
                    &current != buffer
                });
            }
        }
        // Cache the frame snapshot only when the revision changes,
        // instead of cloning the workbook every frame.
        if let Some(inc) = &self.inc {
            if inc.revision != self.frame_revision || self.frame_snapshot.is_none() {
                self.frame_snapshot = Some(inc.snapshot().clone());
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
                if ui.text_edit_singleline(&mut self.config_path).changed() {
                    self.prefs.config_path = self.config_path.clone();
                    self.prefs.save();
                }
                ui.label("External endpoints receive cell values verbatim.");
                ui.label("For sensitive workbooks, use a local model.");
                if ui.button("Close").clicked() {
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
        // Collect edits/clicks while rendering from the cached frame
        // snapshot; apply them after the borrow ends.
        let mut pending_edit: Option<CellEdit> = None;
        let mut clicked_cell: Option<(String, usize, usize)> = None;
        if let Some(snapshot) = self.frame_snapshot.clone() {
            // Sheet tabs.
            if snapshot.sheets.len() > 1 {
                ui.horizontal(|ui| {
                    for (i, sheet) in snapshot.sheets.iter().enumerate() {
                        if ui
                            .selectable_label(self.selected_sheet == i, &sheet.name)
                            .clicked()
                        {
                            self.selected_sheet = i;
                        }
                    }
                });
            }
            let sheet_idx = self
                .selected_sheet
                .min(snapshot.sheets.len().saturating_sub(1));
            if let Some(sheet) = snapshot.sheets.get(sheet_idx) {
                ui.heading(&sheet.name);
                let selected = self.sync.selected_range.clone();
                let max_col = sheet.max_col();
                // Header row.
                ui.horizontal(|ui| {
                    ui.add_space(38.0);
                    for c in 0..max_col {
                        ui.add_sized(
                            [90.0, 18.0],
                            egui::Label::new(
                                egui::RichText::new(gridtalk_graph::col_name(c)).strong(),
                            ),
                        );
                    }
                });
                // Virtualised rows: only the visible window is built.
                let row_height = 22.0;
                egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show_rows(ui, row_height, sheet.cells.len(), |ui, row_range| {
                        for r in row_range {
                            let row = &sheet.cells[r];
                            ui.horizontal(|ui| {
                                ui.add_sized(
                                    [34.0, row_height],
                                    egui::Label::new(
                                        egui::RichText::new(format!("{}", r + 1)).strong(),
                                    ),
                                );
                                for (c, cell) in row.iter().enumerate() {
                                    let is_selected = selected
                                        .as_ref()
                                        .map(|s| s.sheet == sheet.name && s.contains(r, c))
                                        .unwrap_or(false);
                                    let key = (sheet.name.clone(), r, c);
                                    // The editor shows formulas as
                                    // formulas (=B2*C2), never the
                                    // cached display value, and the
                                    // buffer survives frames until
                                    // the debounce flush lands.
                                    let buffer = self
                                        .edit_buffers
                                        .entry(key)
                                        .or_insert_with(|| cell_input_text(cell));
                                    let resp = ui.add_sized(
                                        [90.0, row_height],
                                        egui::TextEdit::singleline(buffer).text_color(
                                            if is_selected {
                                                egui::Color32::YELLOW
                                            } else {
                                                egui::Color32::LIGHT_GRAY
                                            },
                                        ),
                                    );
                                    if resp.clicked() {
                                        clicked_cell = Some((sheet.name.clone(), r, c));
                                    }
                                    if resp.changed() {
                                        pending_edit = Some(CellEdit {
                                            sheet: sheet.name.clone(),
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
        let mut clicked_node: Option<usize> = None;
        let mut clicked_dist = f32::MAX;
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
                painter.text(
                    centre + egui::vec2(10.0, -6.0),
                    egui::Align2::LEFT_TOP,
                    &n.label,
                    egui::FontId::proportional(11.0),
                    egui::Color32::WHITE,
                );
                if response.clicked() {
                    if let Some(click) = response.interact_pointer_pos() {
                        let d = centre.distance(click);
                        if d <= 12.0 && d < clicked_dist {
                            clicked_dist = d;
                            clicked_node = Some(n.id);
                        }
                    }
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
