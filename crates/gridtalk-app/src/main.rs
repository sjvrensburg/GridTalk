//! GridTalk CLI: the headless shell for the dual-view app. It proves
//! the full pipeline (ingest -> graph -> index -> cited answer) and
//! the grid<->graph two-way sync state machine that the egui front
//! end (Phase 4, on Gridcraft's ui-egui) will render.

use gridtalk_graph::{build_graph, load_path, Graph, NodeKind, RangeRef, WorkbookSnapshot};
use gridtalk_index::Index;
use gridtalk_infer::config::Config;
use gridtalk_infer::{backend_from_config, ChatRequest};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Shared selection state: the single source of truth both views read.
/// Selecting a graph node selects its range in the grid; selecting
/// cells in the grid highlights the containing graph node.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ViewState {
    pub selected_range: Option<RangeRef>,
    pub highlighted_node: Option<usize>,
    pub mode: ViewMode,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    #[default]
    Split,
    GridFull,
    GraphFull,
}

impl ViewState {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args) {
        eprintln!("gridtalk: {e}");
        std::process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "ask" => {
            let file = args
                .get(2)
                .ok_or("usage: gridtalk ask <file.csv> <question...>")?;
            let question = args[3..].join(" ");
            if question.is_empty() {
                return Err("usage: gridtalk ask <file.csv> <question...>".into());
            }
            let wb = load_workbook(file)?;
            let (graph, violations) = build_graph(&wb);
            if !violations.is_empty() {
                return Err(format!("graph invariants violated: {violations:?}"));
            }
            let index = Index::build(&wb, &graph);
            let hits = index.search(&question, 3);
            if hits.is_empty() {
                println!("No matching region found for \"{question}\".");
                return Ok(());
            }
            for h in &hits {
                println!("[{}] score {:.2}", h.range.a1(), h.score);
            }
            let context = hits
                .iter()
                .map(|h| h.passage.as_str())
                .collect::<Vec<_>>()
                .join("\n---\n");
            let cfg = config_from_env();
            let backend = backend_from_config(&cfg.inference);
            let req = ChatRequest {
                system: "You answer questions about a spreadsheet using only the cited context. Always cite the range.".into(),
                user: format!("Question: {question}\n\nContext:\n{context}"),
                max_tokens: 512,
            };
            let cancel = Arc::new(AtomicBool::new(false));
            let mut first = true;
            backend
                .stream(&req, &cancel, &mut |t| {
                    if first {
                        print!("\n--- {} ---\n", backend.name());
                        first = false;
                    }
                    print!("{t}");
                })
                .map_err(|e| e.to_string())?;
            println!();
            Ok(())
        }
        "graph" => {
            let file = args.get(2).ok_or("usage: gridtalk graph <file.csv>")?;
            let wb = load_workbook(file)?;
            let (graph, _) = build_graph(&wb);
            for n in &graph.nodes {
                let kind = match n.kind {
                    NodeKind::Workbook => "workbook",
                    NodeKind::Sheet => "sheet",
                    NodeKind::Region => "region",
                    NodeKind::Column => "column",
                };
                println!("{} [{}] {}", n.id, kind, n.label);
            }
            for e in &graph.edges {
                println!("{} -{}-> {}", e.from, e.kind, e.to);
            }
            Ok(())
        }
        "view" => {
            // Terminal rendering of the dual view: grid left, graph right.
            let file = args.get(2).ok_or("usage: gridtalk view <file.csv>")?;
            let wb = load_workbook(file)?;
            let (graph, _) = build_graph(&wb);
            let mut state = ViewState::default();
            if let Some(sheet) = wb.sheets.first() {
                println!("Sheet: {}", sheet.name);
                for (r, row) in sheet.cells.iter().enumerate() {
                    let cells: Vec<String> = row
                        .iter()
                        .map(|c| format!("{:>8}", truncate(&c.display(), 8)))
                        .collect();
                    println!("{:>3} {}", r + 1, cells.join(" "));
                }
            }
            println!("\nGraph nodes:");
            for n in &graph.nodes {
                println!("  {} {}", n.id, n.label);
            }
            // Demonstrate two-way sync.
            if let Some(region) = graph.nodes.iter().find(|n| n.kind == NodeKind::Region) {
                state.select_node(&graph, region.id);
                println!(
                    "\nselect_node({}) -> selected_range {:?}",
                    region.id,
                    state.selected_range.as_ref().map(|r| r.a1())
                );
            }
            state.select_cell(&graph, &wb.sheets[0].name, 1, 1);
            println!(
                "select_cell(B2) -> highlighted_node {:?}",
                state.highlighted_node
            );
            Ok(())
        }
        _ => {
            println!("GridTalk — talk to your spreadsheet, locally.\n\nUsage:\n  gridtalk ask <file.csv|tsv> <question...>\n  gridtalk graph <file.csv|tsv>\n  gridtalk view <file.csv|tsv>\n\nConfig: GRIDTALK_CONFIG=/path/to/config.toml (default: local echo backend)");
            Ok(())
        }
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

fn load_workbook(file: &str) -> Result<WorkbookSnapshot, String> {
    load_path(std::path::Path::new(file)).map_err(|e| e.to_string())
}

fn config_from_env() -> Config {
    match std::env::var("GRIDTALK_CONFIG") {
        Ok(p) => Config::load(std::path::Path::new(&p)),
        Err(_) => Config::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::parse_delimited;

    #[test]
    fn two_way_sync_round_trips() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,2\n", ',');
        let (g, _) = build_graph(&wb);
        let mut state = ViewState::default();
        let region = g.nodes.iter().find(|n| n.kind == NodeKind::Region).unwrap();
        state.select_node(&g, region.id);
        assert_eq!(state.selected_range.as_ref().unwrap().a1(), "Sheet1!A1:B2");
        state.select_cell(&g, "Sheet1", 1, 1);
        assert!(state.highlighted_node.is_some());
    }
}
