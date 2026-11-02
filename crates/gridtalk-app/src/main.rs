//! GridTalk CLI: the headless shell for the dual-view app. It proves
//! the full pipeline (ingest -> graph -> index -> cited answer) and
//! the grid<->graph two-way sync state machine that the egui front
//! end (Phase 4, on Gridcraft's ui-egui) will render.

mod bench;
mod investigate;
mod mcp;
mod tools;

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

const COMMANDS: &[&str] = &[
    "ask",
    "bench",
    "investigate",
    "graph",
    "view",
    "spike",
    "tools",
    "mcp",
    "generate",
    "models",
    "open-in-gridcraft",
    "help",
    "--help",
    "-h",
];

fn run(args: &[String]) -> Result<(), String> {
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    if !COMMANDS.contains(&cmd) {
        return Err(format!(
            "unknown command {cmd:?}; run `gridtalk help` for usage"
        ));
    }
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
            warn_if_external(&cfg);
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
            let (graph, violations) = build_graph(&wb);
            if !violations.is_empty() {
                eprintln!("warning: graph invariants violated: {violations:?}");
            }
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
            if wb.sheets.is_empty() {
                return Err("workbook has no sheets".into());
            }
            let (graph, violations) = build_graph(&wb);
            if !violations.is_empty() {
                eprintln!("warning: graph invariants violated: {violations:?}");
            }
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
            state.select_cell(
                &graph,
                wb.sheets.first().map(|s| s.name.as_str()).unwrap_or(""),
                1,
                1,
            );
            println!(
                "select_cell(B2) -> highlighted_node {:?}",
                state.highlighted_node
            );
            Ok(())
        }
        "bench" => {
            // gridtalk bench generate <file> [--out tasks.json] [--max 100]
            // gridtalk bench run <file> [--tasks tasks.json] [--split train|dev|all] [--card CARD.md]
            let flag = |name: &str| -> Option<String> {
                args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
            };
            match args.get(2).map(|s| s.as_str()) {
                Some("generate") => {
                    let file = args.get(3).ok_or(
                        "usage: gridtalk bench generate <file> [--out tasks.json] [--max 100]",
                    )?;
                    let max: usize = flag("--max").and_then(|v| v.parse().ok()).unwrap_or(100);
                    let wb = load_workbook(file)?;
                    let (_, violations) = build_graph(&wb);
                    if !violations.is_empty() {
                        return Err(format!("graph invariants violated: {violations:?}"));
                    }
                    let tasks = bench::generate_tasks(&wb, max);
                    let json = serde_json::to_string_pretty(&tasks).map_err(|e| e.to_string())?;
                    match flag("--out") {
                        Some(out) => {
                            std::fs::write(&out, &json).map_err(|e| e.to_string())?;
                            eprintln!(
                                "wrote {} tasks ({} train / {} dev) to {out}",
                                tasks.len(),
                                tasks.iter().filter(|t| t.split == "train").count(),
                                tasks.iter().filter(|t| t.split == "dev").count()
                            );
                        }
                        None => println!("{json}"),
                    }
                    Ok(())
                }
                Some("run") => {
                    let file = args
                        .get(3)
                        .ok_or("usage: gridtalk bench run <file> [--tasks tasks.json] [--split train|dev|all] [--card CARD.md]")?;
                    let wb = load_workbook(file)?;
                    let (graph, violations) = build_graph(&wb);
                    if !violations.is_empty() {
                        return Err(format!("graph invariants violated: {violations:?}"));
                    }
                    let hybrid = HybridIndex::build(&wb, &graph);
                    let toolset = tools::Tools {
                        wb: &wb,
                        graph: &graph,
                        hybrid: &hybrid,
                    };
                    let tasks: Vec<bench::BenchTask> = match flag("--tasks") {
                        Some(path) => {
                            let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                            serde_json::from_str(&text)
                                .map_err(|e| format!("invalid tasks file {path}: {e}"))?
                        }
                        None => bench::generate_tasks(&wb, 100),
                    };
                    let split = flag("--split").unwrap_or_else(|| "all".into());
                    if !matches!(split.as_str(), "train" | "dev" | "all") {
                        return Err(format!("unknown split {split}; expected train, dev or all"));
                    }
                    let cfg = config_from_env();
                    let backend = backend_from_config(&cfg.inference);
                    warn_if_external(&cfg);
                    let report = bench::run_bench(&tasks, &toolset, backend.as_ref(), &split, 8)
                        .map_err(|e| format!("bench run failed: {e}"))?;
                    let card = bench::render_card(&report, tasks.len());
                    match flag("--card") {
                        Some(path) => {
                            std::fs::write(&path, &card).map_err(|e| e.to_string())?;
                            println!(
                                "bench: {} tasks, accuracy {:.1}% -> {path}",
                                report.results.len(),
                                report.accuracy("all").unwrap_or(0.0) * 100.0
                            );
                        }
                        None => println!("{card}"),
                    }
                    Ok(())
                }
                _ => {
                    Err("usage: gridtalk bench generate <file> | gridtalk bench run <file>".into())
                }
            }
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
            let tools = tools::Tools {
                wb: &wb,
                graph: &graph,
                hybrid: &hybrid,
            };
            let cfg = config_from_env();
            let backend = backend_from_config(&cfg.inference);
            warn_if_external(&cfg);
            let inv = investigate::investigate(&question, &tools, backend.as_ref(), 8)
                .map_err(|e| format!("backend failed: {e}"))?;
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
            let model = args
                .get(2)
                .ok_or("usage: gridtalk generate <model.gguf> <prompt...>")?;
            let prompt = args[3..].join(" ");
            if prompt.trim().is_empty() {
                return Err(
                    "usage: gridtalk generate <model.gguf> <prompt...> (prompt must not be empty)"
                        .into(),
                );
            }
            let backend = gridtalk_infer::llama_backend::LlamaCppBackend {
                model_path: std::path::PathBuf::from(model),
                ..gridtalk_infer::llama_backend::LlamaCppBackend::default()
            };
            let req = ChatRequest {
                system: "You are a helpful spreadsheet assistant.".into(),
                user: prompt,
                max_tokens: 128,
            };
            let cancel = Arc::new(AtomicBool::new(false));
            let stats = backend
                .generate_with_stats(&req, &cancel, &mut |tok| print!("{tok}"))
                .map_err(|e| e.to_string())?;
            println!(
                "\n\n[bench] llama.cpp/Vulkan prompt_tokens={} generated={} first_token_ms={:?} total_ms={:.0} tok/s={:.1}",
                stats.prompt_tokens,
                stats.generated_tokens,
                stats.first_token_ms.map(|v| (v * 10.0).round() / 10.0),
                stats.total_ms,
                stats.tokens_per_sec
            );
            Ok(())
        }
        "mcp" => {
            // MCP server over stdio for external agent harnesses.
            // --allow-write is accepted for forward compatibility;
            // there are no write tools yet either way (what_if is
            // simulation-only), so it currently changes nothing.
            // Anything else is a typo and must not be silently ignored.
            for extra in &args[2..] {
                if extra != "--allow-write" {
                    return Err(format!(
                        "unknown argument {extra} for `gridtalk mcp` (only --allow-write is recognised)"
                    ));
                }
            }
            mcp::serve_stdio()
        }
        "tools" => {
            // gridtalk tools list
            // gridtalk tools call <file> '<json ToolRequest>'
            match args.get(2).map(|s| s.as_str()) {
                Some("list") => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&tools::ToolRequest::schemas()).unwrap()
                    );
                    Ok(())
                }
                Some("call") => {
                    let file = args
                        .get(3)
                        .ok_or("usage: gridtalk tools call <file> '<json>'")?;
                    let json = args
                        .get(4)
                        .ok_or("usage: gridtalk tools call <file> '<json>'")?;
                    let wb = load_workbook(file)?;
                    let (graph, violations) = build_graph(&wb);
                    if !violations.is_empty() {
                        return Err(format!("graph invariants violated: {violations:?}"));
                    }
                    let hybrid = HybridIndex::build(&wb, &graph);
                    let toolset = tools::Tools {
                        wb: &wb,
                        graph: &graph,
                        hybrid: &hybrid,
                    };
                    let req: tools::ToolRequest = serde_json::from_str(json)
                        .map_err(|e| format!("invalid tool request JSON: {e}"))?;
                    let result = toolset.call(&req);
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
                    );
                    if result.ok {
                        Ok(())
                    } else {
                        Err(result.text)
                    }
                }
                _ => Err("usage: gridtalk tools list | gridtalk tools call <file> '<json>'".into()),
            }
        }
        "spike" => {
            let file = args.get(2).ok_or("usage: gridtalk spike <model.gguf>")?;
            let report = gridtalk_infer::probe::probe_model(std::path::Path::new(file));
            println!(
                "{}: header_ok={} arch={:?} tensors={:?} — {}",
                report.engine,
                report.header_ok,
                report.gguf_arch,
                report.tensor_count,
                report.detail
            );
            if report.header_ok {
                Ok(())
            } else {
                Err("model header did not validate".into())
            }
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
                Some(other) => Err(format!(
                    "unknown models subcommand {other:?}; usage: gridtalk models [import <file.gguf>|remove <name>]"
                )),
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
        "help" | "--help" | "-h" => {
            println!("GridTalk — talk to your spreadsheet, locally.\n\nUsage:\n  gridtalk ask <file.csv|tsv|xlsx|ods> <question...>\n  gridtalk investigate <file> <question...>\n  gridtalk bench generate <file> [--out tasks.json] | bench run <file> [--card CARD.md]\n  gridtalk graph <file>\n  gridtalk view <file>\n  gridtalk spike <model.gguf>\n  gridtalk tools list | tools call <file> '<json>'\n  gridtalk mcp  (MCP server over stdio)\n  gridtalk generate <model.gguf> <prompt...>\n  gridtalk models [import <file.gguf>|remove <name>]\n  gridtalk open-in-gridcraft <file>\n\nConfig: GRIDTALK_CONFIG=/path/to/config.toml (default: local echo backend)");
            Ok(())
        }
        _ => unreachable!("COMMANDS guard rejects unknown commands before dispatch"),
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
    let exe_name = if cfg!(windows) {
        "gridcraft.exe"
    } else {
        "gridcraft"
    };
    // Beside the running executable (installed layouts).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // PATH lookup.
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(exe_name);
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }
    // Build tree (development).
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    for profile in ["release", "debug"] {
        let candidate = root.join(format!("vendor/gridcraft/target/{profile}/{exe_name}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Warn (once per invocation) before passages leave the machine.
fn warn_if_external(cfg: &Config) {
    if cfg.inference.backend == "openai-compatible" {
        let url = cfg.inference.base_url.as_deref().unwrap_or("");
        eprintln!(
            "gridtalk: note: external backend {url} will receive spreadsheet passages (including cell values) verbatim. Use a local model or a redacting proxy if that is not acceptable."
        );
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
