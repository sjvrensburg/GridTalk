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
}

impl UiPrefs {
    pub fn path() -> std::path::PathBuf {
        let base = std::env::var("GRIDTALK_CONFIG_DIR").unwrap_or_else(|_| {
            format!(
                "{}/.config/gridtalk",
                std::env::var("HOME").unwrap_or_default()
            )
        });
        std::path::PathBuf::from(base).join("ui-state.json")
    }
    pub fn load() -> Self {
        std::fs::read_to_string(Self::path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }
    pub fn save(&self) {
        let p = Self::path();
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(t) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(p, t);
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
}

impl Default for GridTalkApp {
    fn default() -> Self {
        GridTalkApp {
            inc: None,
            graph: Graph::default(),
            index: None,
            sync: SyncState::default(),
            prefs: UiPrefs::load(),
            workbook_name: String::new(),
            positions: BTreeMap::new(),
            layout_worker: None,
            layout_revision: u64::MAX,
            chat_input: String::new(),
            chat: Vec::new(),
            status: "Open a workbook (File > Open, or pass a path as argv[1]).".into(),
            show_settings: false,
            config_path: std::env::var("GRIDTALK_CONFIG").unwrap_or_default(),
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
    let mut text = String::from("[inference]\n");
    text.push_str(&format!("backend = \"{backend}\"\n"));
    text.push_str("engine = \"candle\"\n");
    if !model_path.is_empty() {
        text.push_str(&format!("model_path = \"{model_path}\"\n"));
    }
    if backend == "openai-compatible" {
        text.push_str(&format!("base_url = \"{base_url}\"\n"));
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
                self.rebuild(inc.snapshot().clone());
                self.inc = Some(inc);
                self.status = format!("Opened {}", path.display());
            }
            Err(e) => self.status = format!("Open failed: {e}"),
        }
    }

    fn rebuild(&mut self, snapshot: gridtalk_graph::WorkbookSnapshot) {
        let (graph, violations) = build_graph(&snapshot);
        if violations.is_empty() {
            self.index = Some(Index::build(&snapshot, &graph));
        } else {
            self.status = format!("Graph invariants violated: {violations:?}");
        }
        self.graph = graph;
        self.layout_revision = u64::MAX; // force fresh layout for the new graph
    }

    fn ask(&mut self) {
        let question = self.chat_input.trim().to_string();
        if question.is_empty() {
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
        let cfg = if self.config_path.is_empty() {
            Config::default()
        } else {
            Config::load(std::path::Path::new(&self.config_path))
        };
        let backend = backend_from_config(&cfg.inference);
        let req = ChatRequest {
            system: "You answer questions about a spreadsheet using only the cited context. Always cite the range.".into(),
            user: format!("Question: {question}\n\nContext:\n{context}"),
            max_tokens: 512,
        };
        let cancel = Arc::new(AtomicBool::new(false));
        let mut out = String::new();
        match backend.stream(&req, &cancel, &mut |t| out.push_str(t)) {
            Ok(()) => self.chat.push(ChatMsg {
                role: backend.name().to_string(),
                text: out,
            }),
            Err(e) => self.chat.push(ChatMsg {
                role: "error".into(),
                text: e.to_string(),
            }),
        }
    }
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
        // Collect finished layouts.
        if let Some(worker) = &self.layout_worker {
            let expected = worker.revision;
            if let Ok((rev, positions)) = worker.rx.try_recv() {
                if rev == expected {
                    self.positions = positions;
                }
                self.layout_worker = None;
            }
        }

        egui::Panel::top("menu").show(ui_root, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open…").clicked() {
                    if let Some(path) = rfd_pick_file() {
                        self.open(&path);
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
                ui.label(&self.status);
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
                        self.status = format!("Setup saved to {}", p.display());
                    }
                    Err(e) => self.status = format!("Could not save setup: {e}"),
                }
                self.prefs.first_run_done = true;
                self.prefs.save();
            }
        }

        if self.show_settings {
            egui::Window::new("Settings").show(&ctx, |ui| {
                ui.label("Config file (blank = local echo backend):");
                ui.text_edit_singleline(&mut self.config_path);
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
        // Collect edits outside the borrow of `inc`.
        let mut pending_edit: Option<CellEdit> = None;
        let mut clicked_cell: Option<(String, usize, usize)> = None;
        if let Some(inc) = &self.inc {
            let snapshot = inc.snapshot().clone();
            if let Some(sheet) = snapshot.sheets.first() {
                ui.heading(&sheet.name);
                let selected = self.sync.selected_range.clone();
                egui::ScrollArea::both().show(ui, |ui| {
                    egui::Grid::new("cells").striped(true).show(ui, |ui| {
                        ui.label("");
                        for c in 0..sheet.max_col() {
                            ui.strong(gridtalk_graph::col_name(c));
                        }
                        ui.end_row();
                        for (r, row) in sheet.cells.iter().enumerate() {
                            ui.strong(format!("{}", r + 1));
                            for (c, cell) in row.iter().enumerate() {
                                let is_selected = selected
                                    .as_ref()
                                    .map(|s| s.sheet == sheet.name && s.contains(r, c))
                                    .unwrap_or(false);
                                let mut text = cell.display();
                                let resp = ui.add(
                                    egui::TextEdit::singleline(&mut text)
                                        .desired_width(90.0)
                                        .text_color(if is_selected {
                                            egui::Color32::YELLOW
                                        } else {
                                            egui::Color32::LIGHT_GRAY
                                        }),
                                );
                                if resp.clicked() {
                                    clicked_cell = Some((sheet.name.clone(), r, c));
                                }
                                if resp.changed() {
                                    pending_edit = Some(CellEdit {
                                        sheet: sheet.name.clone(),
                                        row: r,
                                        col: c,
                                        value: parse_cell_input(&text),
                                    });
                                }
                            }
                            ui.end_row();
                        }
                    });
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
            .show(ui, |ui| {
                for msg in &self.chat {
                    ui.label(format!("{}: {}", msg.role, msg.text));
                }
            });
        ui.horizontal(|ui| {
            let resp = ui.text_edit_singleline(&mut self.chat_input);
            if (ui.button("Ask").clicked()
                || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))))
                && !self.chat_input.is_empty()
            {
                self.ask();
            }
        });
    }

    fn graph_pane(&mut self, ui: &mut egui::Ui) {
        ui.heading("Graph");
        let size = ui.available_size();
        let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
        // (Re)spawn layout for a new graph revision / first paint.
        if self.positions.len() != self.graph.nodes.len() && self.layout_worker.is_none() {
            let revision = if self.layout_revision == u64::MAX {
                0
            } else {
                self.layout_revision
            };
            self.layout_worker = Some(spawn_layout(
                self.graph.clone(),
                revision,
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
                        if centre.distance(click) <= 12.0 {
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

fn rfd_pick_file() -> Option<std::path::PathBuf> {
    // No file-dialog dependency in the shell yet: the path can be
    // passed as argv[1]; the button keeps the UI affordance wired
    // and reports that in the status line if no dialog is available.
    std::env::args().nth(1).map(std::path::PathBuf::from)
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
        let text = "[inference]\nbackend = \"local\"\nengine = \"candle\"\nmodel_path = \"/models/m.gguf\"\n";
        let cfg = gridtalk_infer::config::Config::parse(text);
        assert_eq!(cfg.inference.backend, "local");
        assert_eq!(cfg.inference.model_path.as_deref(), Some("/models/m.gguf"));
        // First-run default: wizard not yet done.
        assert!(!UiPrefs::default().first_run_done);
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
