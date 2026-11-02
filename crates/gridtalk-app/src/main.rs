//! GridTalk CLI: the headless shell for the dual-view app. It proves
//! the full pipeline (ingest -> graph -> index -> cited answer) and
//! the grid<->graph two-way sync state machine that the egui front
//! end (Phase 4, on Gridcraft's ui-egui) will render.

mod investigate;

use gridtalk_graph::{build_graph, load_path, Graph, NodeKind, RangeRef, WorkbookSnapshot};
use gridtalk_index::{HybridIndex, Index};
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
        "investigate" => {
            let file = args
                .get(2)
                .ok_or("usage: gridtalk investigate <file> <question...>")?;
            let question = args[3..].join(" ");
            if question.is_empty() {
                return Err("usage: gridtalk investigate <file> <question...>".into());
            }
            let wb = load_workbook(file)?;
            let (graph, violations) = build_graph(&wb);
            if !violations.is_empty() {
                return Err(format!("graph invariants violated: {violations:?}"));
            }
            let hybrid = HybridIndex::build(&wb, &graph);
            let tools = investigate::Tools {
                wb: &wb,
                graph: &graph,
                hybrid: &hybrid,
            };
            let cfg = config_from_env();
            let backend = backend_from_config(&cfg.inference);
            let inv = investigate::investigate(&question, &tools, backend.as_ref(), 8);
            for step in &inv.trail {
                println!(
                    "[tool] {} ({}) -> {}",
                    step.tool,
                    step.args,
                    step.result.replace('\n', " | ")
                );
            }
            println!("\ncitations: {}", inv.citations.join(", "));
            println!("\n--- answer ({}) ---\n{}", backend.name(), inv.answer);
            Ok(())
        }
        "open-in-gridcraft" => {
            let file = args
                .get(2)
                .ok_or("usage: gridtalk open-in-gridcraft <file>")?;
            let bin = gridcraft_binary().ok_or(
                "Gridcraft binary not found. Build it once with: cargo build --manifest-path vendor/gridcraft/Cargo.toml -p gridcraft (or set GRIDTALK_GRIDCRAFT_BIN)",
            )?;
            println!("Launching Gridcraft ({}) on {file}", bin.display());
            println!("Edit and save there, then re-open the file in GridTalk to re-snapshot it.");
            std::process::Command::new(&bin)
                .arg(file)
                .spawn()
                .map_err(|e| format!("failed to launch Gridcraft: {e}"))?;
            Ok(())
        }
        "generate" => {
            #[cfg(feature = "candle-gen")]
            {
                let model = args
                    .get(2)
                    .ok_or("usage: gridtalk generate <model.gguf> <tokenizer.json> <prompt...>")?;
                let tokenizer = args.get(3).ok_or("missing tokenizer.json path")?;
                let prompt = args[4..].join(" ");
                let backend = gridtalk_infer::generate::CandleLlamaBackend {
                    model_path: std::path::PathBuf::from(model),
                    tokenizer_path: std::path::PathBuf::from(tokenizer),
                    seed: 42,
                    temperature: 0.7,
                };
                let req = ChatRequest {
                    system: "You are a helpful spreadsheet assistant.".into(),
                    user: prompt,
                    max_tokens: 48,
                };
                let cancel = Arc::new(AtomicBool::new(false));
                let stats = backend
                    .generate_with_stats(&req, &cancel, &mut |t| print!("{t}"))
                    .map_err(|e| e.to_string())?;
                println!(
                    "\n\n[bench] prompt_tokens={} generated={} first_token_ms={:?} total_ms={:.0} tok/s={:.1}",
                    stats.prompt_tokens,
                    stats.generated_tokens,
                    stats.first_token_ms.map(|v| (v * 10.0).round() / 10.0),
                    stats.total_ms,
                    stats.tokens_per_sec
                );
                Ok(())
            }
            #[cfg(not(feature = "candle-gen"))]
            {
                Err(
                    "generate requires a candle-gen build: cargo run -p gridtalk-app --features candle-gen -- generate ..."
                        .into(),
                )
            }
        }
        "spike" => {
            let file = args.get(2).ok_or("usage: gridtalk spike <model.gguf>")?;
            for report in gridtalk_infer::probe::probe_ladder(std::path::Path::new(file)) {
                println!(
                    "{:?}: compiled_in={} loads={} arch={:?} tensors={:?} — {}",
                    report.engine,
                    report.compiled_in,
                    report.loads,
                    report.gguf_arch,
                    report.tensor_count,
                    report.detail
                );
            }
            Ok(())
        }
        "models" => {
            let mgr = gridtalk_infer::models::ModelManager::default_cache();
            match args.get(2).map(|s| s.as_str()) {
                Some("import") => {
                    let src = args
                        .get(3)
                        .ok_or("usage: gridtalk models import <file.gguf>")?;
                    let entry = mgr
                        .import(std::path::Path::new(src))
                        .map_err(|e| e.to_string())?;
                    println!(
                        "imported {} ({} bytes) -> {}",
                        entry.name,
                        entry.bytes,
                        entry.path.display()
                    );
                    Ok(())
                }
                Some("remove") => {
                    let name = args.get(3).ok_or("usage: gridtalk models remove <name>")?;
                    mgr.remove(name).map_err(|e| e.to_string())?;
                    println!("removed {name}");
                    Ok(())
                }
                _ => {
                    let entries = mgr.list();
                    if entries.is_empty() {
                        println!("no models in {}", mgr.dir.display());
                    }
                    for e in entries {
                        println!(
                            "{} ({} bytes) arch={:?} name={:?}",
                            e.name,
                            e.bytes,
                            e.info.as_ref().and_then(|i| i.architecture.clone()),
                            e.info.as_ref().and_then(|i| i.name.clone())
                        );
                    }
                    Ok(())
                }
            }
        }
        _ => {
            println!("GridTalk — talk to your spreadsheet, locally.\n\nUsage:\n  gridtalk ask <file.csv|tsv|xlsx|ods> <question...>\n  gridtalk graph <file>\n  gridtalk view <file>\n  gridtalk spike <model.gguf>\n  gridtalk models [import <file.gguf>|remove <name>]\n\nConfig: GRIDTALK_CONFIG=/path/to/config.toml (default: local echo backend)");
            Ok(())
        }
    }
}

/// Locate the Gridcraft desktop binary (decision 0003).
pub fn gridcraft_binary() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GRIDTALK_GRIDCRAFT_BIN") {
        let path = std::path::PathBuf::from(p);
        if path.exists() {
            return Some(path);
        }
    }
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for profile in ["release", "debug"] {
        let candidate = root.join(format!("vendor/gridcraft/target/{profile}/gridcraft"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
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
