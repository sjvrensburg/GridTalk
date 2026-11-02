//! GridTalk CLI: the headless shell for the dual-view app. It proves
//! the full pipeline (ingest -> graph -> index -> cited answer) and
//! the grid<->graph two-way sync state machine that the egui front
//! end (Phase 4, on Gridcraft's ui-egui) will render.

mod adapter;
mod bench;
mod investigate;
mod mcp;
mod optimise;
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
    "adapter",
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
    "optimise",
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
            if args.len() > 3 {
                return Err("usage: gridtalk graph <file.csv> (no extra arguments)".into());
            }
            let wb = load_workbook(file)?;
            let (graph, violations) = build_graph(&wb);
            if !violations.is_empty() {
                return Err(format!("graph invariants violated: {violations:?}"));
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
            if args.len() > 3 {
                return Err("usage: gridtalk view <file.csv> (no extra arguments)".into());
            }
            let wb = load_workbook(file)?;
            if wb.sheets.is_empty() {
                return Err("workbook has no sheets".into());
            }
            let (graph, violations) = build_graph(&wb);
            if !violations.is_empty() {
                return Err(format!("graph invariants violated: {violations:?}"));
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
        "adapter" => {
            let root = adapter::adapters_root();
            match args.get(2).map(|s| s.as_str()) {
                Some("list") => {
                    let names = adapter::list_adapters(&root);
                    if names.is_empty() {
                        println!("no adapters in {}", root.display());
                    }
                    for name in &names {
                        let a = adapter::load_adapter(&root, name);
                        println!(
                            "{name}: v{} top_k={} max_steps={}",
                            a.version, a.top_k, a.max_steps
                        );
                    }
                    Ok(())
                }
                Some("show") => {
                    let name = args.get(3).ok_or("usage: gridtalk adapter show <name>")?;
                    // The CARD.md read below joins `name` onto the
                    // root: a traversal name must never reach it.
                    if !adapter::valid_adapter_name(name) {
                        return Err(format!("invalid adapter name {name:?}"));
                    }
                    let a = adapter::load_adapter(&root, name);
                    println!(
                        "adapter {name} v{} top_k={} max_steps={} ref_context={}",
                        a.version, a.top_k, a.max_steps, a.ref_context
                    );
                    println!("--- ADAPTER.md ---");
                    println!("{}", a.rules);
                    let card = root.join(name).join("CARD.md");
                    if card.exists() {
                        println!("--- CARD.md ---");
                        println!(
                            "{}",
                            std::fs::read_to_string(&card).map_err(|e| e.to_string())?
                        );
                    }
                    Ok(())
                }
                Some("export") => {
                    // gridtalk adapter export <name> [--out AGENTS.md]
                    let name = args
                        .get(3)
                        .filter(|n| !n.starts_with("--"))
                        .ok_or("usage: gridtalk adapter export <name> [--out AGENTS.md]")?;
                    if !adapter::valid_adapter_name(name) || !root.join(name).is_dir() {
                        return Err(format!("adapter {name:?} not found"));
                    }
                    // Strict flag parsing: at most one --out with a
                    // non-flag value, nothing else.
                    let mut out_path: Option<&String> = None;
                    let mut i = 4;
                    while i < args.len() {
                        match args[i].as_str() {
                            "--out" if out_path.is_none() => {
                                let value = args
                                    .get(i + 1)
                                    .filter(|v| !v.starts_with("--"))
                                    .ok_or("export flag --out needs a value")?;
                                out_path = Some(value);
                                i += 2;
                            }
                            other => {
                                return Err(format!("unknown or duplicate export flag {other}"));
                            }
                        }
                    }
                    let loaded = adapter::load_adapter(&root, name);
                    let md = adapter::export_agents_md(&loaded);
                    match out_path {
                        Some(path) => {
                            std::fs::write(path, &md).map_err(|e| e.to_string())?;
                            println!("wrote {} (adapter {name} v{})", path, loaded.version);
                        }
                        None => print!("{md}"),
                    }
                    Ok(())
                }
                Some("rollback") => {
                    let name = args
                        .get(3)
                        .ok_or("usage: gridtalk adapter rollback <name>")?;
                    let v = adapter::rollback(&root, name)?;
                    println!("adapter {name} rolled back to v{v}");
                    Ok(())
                }
                _ => Err(
                    "usage: gridtalk adapter list|show <name>|rollback <name>|export <name> [--out AGENTS.md]"
                        .into(),
                ),
            }
        }
        "optimise" => {
            // gridtalk optimise <file> [--adapter <name>] [--rounds 3]
            let file = args
                .get(2)
                .ok_or("usage: gridtalk optimise <file> [--adapter <name>] [--rounds 3]")?;
            let mut name = "default".to_string();
            let mut rounds = 3usize;
            {
                // Strict like bench: no duplicates, no unknowns,
                // no flag-shaped values, rounds capped at 10.
                let mut seen: Vec<&str> = Vec::new();
                let mut i = 3;
                while i < args.len() {
                    let flag_name = args[i].as_str();
                    if seen.contains(&flag_name) {
                        return Err(format!("duplicate optimise flag {flag_name}"));
                    }
                    seen.push(flag_name);
                    let value = args
                        .get(i + 1)
                        .filter(|v| !v.starts_with("--"))
                        .ok_or_else(|| format!("optimise flag {flag_name} needs a value"))?;
                    match flag_name {
                        "--adapter" => name = value.clone(),
                        "--rounds" => {
                            rounds = value
                                .parse::<usize>()
                                .ok()
                                .filter(|n| (1..=10).contains(n))
                                .ok_or("--rounds must be an integer in 1..=10")?
                        }
                        other => return Err(format!("unknown optimise flag {other}")),
                    }
                    i += 2;
                }
            }
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
            let gen = bench::generate_tasks_ex(&wb, 100);
            let cfg = config_from_env();
            let backend = backend_from_config(&cfg.inference);
            warn_if_external(&cfg);
            let root = adapter::adapters_root();
            let mut current = load_named_adapter(&name);
            println!(
                "optimise adapter {name} (v{}) on {} tasks with backend {}",
                current.version,
                gen.tasks.len(),
                backend.name()
            );
            for round in 1..=rounds {
                let outcome = optimise::optimise_round(
                    &root,
                    &gen.tasks,
                    &toolset,
                    backend.as_ref(),
                    &current,
                    round,
                )?;
                if let Some(note) = &outcome.note {
                    println!("round {round}: {note}; stopping");
                    break;
                }
                match &outcome.accepted {
                    Some((label, train, dev)) => {
                        println!(
                            "round {round}: ACCEPTED {label} (train {:.1}%, dev {:.1}%)",
                            train * 100.0,
                            dev * 100.0
                        );
                        current = adapter::load_adapter(&root, &name);
                    }
                    None => {
                        println!(
                            "round {round}: no candidate beat train {:.1}% (dev {:.1}%); stopping",
                            outcome.baseline_train * 100.0,
                            outcome.baseline_dev * 100.0
                        );
                        break;
                    }
                }
            }
            println!("adapter {name} is now v{}", current.version);
            Ok(())
        }
        "bench" => {
            // gridtalk bench generate <file> [--out tasks.json] [--max 100]
            // gridtalk bench run <file> [--tasks tasks.json] [--split train|dev|all] [--card CARD.md]
            // Strict flag parsing: unknown flags, duplicates and
            // missing values are usage errors, never silent defaults.
            let mut flags: std::collections::BTreeMap<String, String> =
                std::collections::BTreeMap::new();
            {
                let known = [
                    "--out",
                    "--max",
                    "--tasks",
                    "--split",
                    "--card",
                    "--adapter",
                ];
                let mut i = 4;
                while i < args.len() {
                    let name = &args[i];
                    if !known.contains(&name.as_str()) {
                        return Err(format!(
                            "unknown bench flag {name}; expected one of {known:?} with a value"
                        ));
                    }
                    let value = args
                        .get(i + 1)
                        .filter(|v| !v.starts_with("--"))
                        .ok_or_else(|| format!("bench flag {name} needs a value"))?;
                    if flags.insert(name.clone(), value.clone()).is_some() {
                        return Err(format!("duplicate bench flag {name}"));
                    }
                    i += 2;
                }
            }
            let flag = |name: &str| -> Option<String> { flags.get(name).cloned() };
            match args.get(2).map(|s| s.as_str()) {
                Some("generate") => {
                    let file = args.get(3).ok_or(
                        "usage: gridtalk bench generate <file> [--out tasks.json] [--max 100]",
                    )?;
                    let max: usize = match flag("--max") {
                        Some(v) => {
                            v.parse::<usize>().ok().filter(|n| *n >= 1).ok_or_else(|| {
                                format!("--max must be a positive integer, got {v:?}")
                            })?
                        }
                        None => 100,
                    };
                    let wb = load_workbook(file)?;
                    let (_, violations) = build_graph(&wb);
                    if !violations.is_empty() {
                        return Err(format!("graph invariants violated: {violations:?}"));
                    }
                    let gen = bench::generate_tasks_ex(&wb, max);
                    let tasks = gen.tasks;
                    if gen.skipped_formulas > 0 {
                        eprintln!(
                            "note: {} formula cells skipped (erroring/non-finite)",
                            gen.skipped_formulas
                        );
                    }
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
                    let mut skipped_formulas = 0usize;
                    let tasks: Vec<bench::BenchTask> = match flag("--tasks") {
                        Some(path) => {
                            let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                            serde_json::from_str(&text)
                                .map_err(|e| format!("invalid tasks file {path}: {e}"))?
                        }
                        None => {
                            let gen = bench::generate_tasks_ex(&wb, 100);
                            skipped_formulas = gen.skipped_formulas;
                            gen.tasks
                        }
                    };
                    {
                        // Tasks files are trusted local input, but
                        // nonsense should be loud, not silent.
                        let mut seen = std::collections::BTreeSet::new();
                        for t in &tasks {
                            if !matches!(t.split.as_str(), "train" | "dev") {
                                eprintln!(
                                    "gridtalk: warning: task {} has unknown split {:?}",
                                    t.id, t.split
                                );
                            }
                            if !matches!(
                                t.kind.as_str(),
                                "cell_value" | "formula_result" | "what_if"
                            ) {
                                eprintln!(
                                    "gridtalk: warning: task {} has unknown kind {:?}",
                                    t.id, t.kind
                                );
                            }
                            if !seen.insert(t.id.clone()) {
                                eprintln!("gridtalk: warning: duplicate task id {}", t.id);
                            }
                        }
                    }
                    if flag("--split").is_none() {
                        eprintln!(
                            "gridtalk: note: scoring all splits (dev included); use --split train to exclude dev"
                        );
                    }
                    let split = flag("--split").unwrap_or_else(|| "all".into());
                    if !matches!(split.as_str(), "train" | "dev" | "all") {
                        return Err(format!("unknown split {split}; expected train, dev or all"));
                    }
                    let cfg = config_from_env();
                    let backend = backend_from_config(&cfg.inference);
                    warn_if_external(&cfg);
                    let bench_adapter = flag("--adapter").map(|name| load_named_adapter(&name));
                    let report = bench::run_bench_with(
                        &tasks,
                        &toolset,
                        backend.as_ref(),
                        &split,
                        bench_adapter.as_ref().map(|a| a.max_steps).unwrap_or(8),
                        bench_adapter.as_ref(),
                    )
                    .map_err(|e| format!("bench run failed: {e}"))?;
                    let card =
                        bench::render_card_with_skipped(&report, tasks.len(), skipped_formulas);
                    match flag("--card") {
                        Some(path) => {
                            std::fs::write(&path, &card).map_err(|e| e.to_string())?;
                            match report.accuracy("all") {
                                Some(acc) => println!(
                                    "bench: {} tasks, accuracy {:.1}% -> {path}",
                                    report.results.len(),
                                    acc * 100.0
                                ),
                                None => println!("bench: no tasks scored for this split -> {path}"),
                            }
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
                .ok_or("usage: gridtalk investigate <file> [--adapter <name>] <question...>")?;
            // Extract an optional --adapter <name> pair before the
            // question words are joined.
            let mut adapter_name: Option<String> = None;
            let mut question_words: Vec<String> = Vec::new();
            {
                // --adapter is only a flag before the question
                // starts (or after a `--` separator handling);
                // inside the question it is literal text.
                let mut i = 3;
                let mut in_flags = true;
                while i < args.len() {
                    if in_flags && args[i] == "--adapter" {
                        if adapter_name.is_some() {
                            return Err("duplicate --adapter flag".into());
                        }
                        adapter_name = Some(
                            args.get(i + 1)
                                .filter(|v| !v.starts_with("--"))
                                .ok_or("--adapter needs a value")?
                                .clone(),
                        );
                        i += 2;
                    } else {
                        in_flags = false;
                        question_words.push(args[i].clone());
                        i += 1;
                    }
                }
            }
            let question = question_words.join(" ");
            if question.is_empty() {
                return Err(
                    "usage: gridtalk investigate <file> [--adapter <name>] <question...>".into(),
                );
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
            let cli_adapter = adapter_name.as_ref().map(|n| load_named_adapter(n));
            let sidecar = adapter::sidecar_rules(std::path::Path::new(file));
            let inv = investigate::investigate_with(
                &question,
                &tools,
                backend.as_ref(),
                cli_adapter.as_ref().map(|a| a.max_steps).unwrap_or(8),
                cli_adapter.as_ref(),
                sidecar.as_deref(),
            )
            .map_err(|e| format!("backend failed: {e}"))?;
            for step in &inv.trail {
                println!(
                    "[tool] {} ({}) -> {}",
                    step.tool,
                    // Args carry the raw question: flatten newlines
                    // so they cannot forge citations/answer markers.
                    step.args.replace('\n', " | "),
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
                if extra == "--allow-write" {
                    eprintln!(
                        "gridtalk: warning: --allow-write is a no-op (no write tools exist yet)"
                    );
                } else {
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
                    if args.len() > 3 {
                        return Err("usage: gridtalk tools list (no extra arguments)".into());
                    }
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&tools::ToolRequest::schemas()).unwrap()
                    );
                    Ok(())
                }
                Some("call") => {
                    if args.len() > 5 {
                        return Err("usage: gridtalk tools call <file> '<json>'".into());
                    }
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
                    // Print the result exactly once; a failed call
                    // exits 1 without duplicating the error text.
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
                    );
                    if result.ok {
                        Ok(())
                    } else {
                        std::process::exit(1);
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
            println!("GridTalk — talk to your spreadsheet, locally.\n\nUsage:\n  gridtalk ask <file.csv|tsv|xlsx|ods> <question...>\n  gridtalk investigate <file> <question...>\n  gridtalk bench generate <file> [--out tasks.json] | bench run <file> [--card CARD.md] [--adapter <name>]\n  gridtalk optimise <file> [--adapter <name>] [--rounds 3]\n  gridtalk adapter list|show <name>|rollback <name>|export <name> [--out AGENTS.md]\n  gridtalk graph <file>\n  gridtalk view <file>\n  gridtalk spike <model.gguf>\n  gridtalk tools list | tools call <file> '<json>'\n  gridtalk mcp  (MCP server over stdio)\n  gridtalk generate <model.gguf> <prompt...>\n  gridtalk models [import <file.gguf>|remove <name>]\n  gridtalk open-in-gridcraft <file>\n\nConfig: GRIDTALK_CONFIG=/path/to/config.toml (default: local echo backend)");
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

/// Load an explicitly named adapter, warning when it does not
/// exist (a typo must not silently run unadapted).
fn load_named_adapter(name: &str) -> adapter::Adapter {
    let root = adapter::adapters_root();
    if !adapter::valid_adapter_name(name) {
        eprintln!("gridtalk: warning: invalid adapter name {name:?}; running unadapted");
        return adapter::Adapter::empty(name);
    }
    if !root.join(name).is_dir() {
        eprintln!(
            "gridtalk: warning: adapter {name:?} not found in {}; running unadapted",
            root.display()
        );
    }
    adapter::load_adapter(&root, name)
}

fn load_workbook(file: &str) -> Result<WorkbookSnapshot, String> {
    load_path(std::path::Path::new(file)).map_err(|e| e.to_string())
}

fn config_from_env() -> Config {
    match std::env::var("GRIDTALK_CONFIG") {
        Ok(p) => Config::load(std::path::Path::new(&p)),
        Err(_) => {
            // Discover the platform default the docs promise:
            // <config dir>/gridtalk/config.toml (GRIDTALK_CONFIG_DIR
            // honoured via the UI's shared resolution).
            let dir = match std::env::var("GRIDTALK_CONFIG_DIR") {
                Ok(d) if !d.is_empty() => std::path::PathBuf::from(d),
                _ => gridtalk_app_config_dir(),
            };
            let path = dir.join("config.toml");
            if path.exists() {
                Config::load(&path)
            } else {
                Config::default()
            }
        }
    }
}

fn gridtalk_app_config_dir() -> std::path::PathBuf {
    // Mirrors dirs::config_dir (the UI's resolution): Roaming
    // APPDATA on Windows, XDG_CONFIG_HOME on Linux.
    if let Ok(appdata) = std::env::var("APPDATA") {
        if !appdata.is_empty() {
            return std::path::PathBuf::from(appdata).join("gridtalk");
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return std::path::PathBuf::from(xdg).join("gridtalk");
        }
    }
    if let Ok(appdata) = std::env::var("LOCALAPPDATA") {
        if !appdata.is_empty() {
            return std::path::PathBuf::from(appdata).join("gridtalk");
        }
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => std::path::PathBuf::from(home).join(".config/gridtalk"),
        _ => std::path::PathBuf::from(".config/gridtalk"),
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
