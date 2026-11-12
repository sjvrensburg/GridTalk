//! GridTalk CLI: the headless shell for the dual-view app. It proves
//! the full pipeline (ingest -> graph -> index -> cited answer) and
//! the grid<->graph two-way sync state machine that the egui front
//! end (Phase 4, on Gridcraft's ui-egui) will render.

mod bench;
mod cli;
mod investigate;
mod mcp;
mod optimise;
mod tools;

use cli::CliError;
use gridtalk_adapter as adapter;
use gridtalk_graph::{build_graph, load_path, Graph, NodeKind, RangeRef, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
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
        if let Some(message) = e.message() {
            eprintln!("gridtalk: {message}");
        }
        std::process::exit(e.exit_code());
    }
}

/// The file argument and the free text after it (`ask`, `generate`).
fn file_and_text(
    cmd: &str,
    rest: &[String],
    first: &str,
    what: &str,
) -> Result<(String, String), CliError> {
    // No flags: one before the text is a typo, not the start of it.
    let parsed = cli::parse_with(cmd, rest, &[], Some(1))?;
    let file = parsed.positionals(cmd, &[&format!("<{first}>")], 0)?[0].clone();
    let text = parsed.text();
    if text.trim().is_empty() {
        return Err(cli::usage(cmd, format!("missing the {what}")));
    }
    Ok((file, text))
}

/// The workbook a command reads, or an input error.
fn workbook_and_graph(file: &str) -> Result<(WorkbookSnapshot, Graph), CliError> {
    let wb = load_workbook(file)?;
    let (graph, violations) = build_graph(&wb);
    if !violations.is_empty() {
        return Err(CliError::Other(format!(
            "graph invariants violated: {violations:?}"
        )));
    }
    Ok((wb, graph))
}

fn run(args: &[String]) -> Result<(), CliError> {
    let Some(cmd) = args.get(1).map(String::as_str) else {
        println!("{}", cli::top_help());
        return Ok(());
    };
    let cmd = cli::canonical(cmd);
    let rest = &args[2..];
    match cmd {
        "help" | "--help" | "-h" => {
            return match rest {
                [] => {
                    println!("{}", cli::top_help());
                    Ok(())
                }
                [flag] if cli::is_help(flag) => {
                    println!("{}", cli::top_help());
                    Ok(())
                }
                [name] => match cli::command(name) {
                    Some(c) => {
                        println!("{}", cli::command_help(c));
                        Ok(())
                    }
                    None => Err(cli::usage("help", format!("unknown command {name:?}"))),
                },
                [_, extra, ..] => Err(cli::usage("help", format!("unexpected argument {extra:?}"))),
            };
        }
        "version" | "--version" | "-V" => {
            if let ([flag], Some(c)) = (rest, cli::command("version")) {
                if cli::is_help(flag) {
                    println!("{}", cli::command_help(c));
                    return Ok(());
                }
            }
            if let Some(extra) = rest.first() {
                return Err(cli::usage(
                    "version",
                    format!("unexpected argument {extra:?}"),
                ));
            }
            println!("{}", cli::version());
            return Ok(());
        }
        _ => {}
    }
    let Some(command) = cli::command(cmd) else {
        return Err(CliError::Usage(format!(
            "unknown command {cmd:?}; run `gridtalk help` for the commands"
        )));
    };
    let asks_help = rest.first().is_some_and(|a| cli::is_help(a))
        || (cli::WITH_SUBCOMMANDS.contains(&cmd) && rest.get(1).is_some_and(|a| cli::is_help(a)));
    if asks_help {
        println!("{}", cli::command_help(command));
        return Ok(());
    }
    match cmd {
        "ask" => {
            let (file, question) = file_and_text(cmd, rest, "file", "question")?;
            let (wb, graph) = workbook_and_graph(&file)?;
            // The same hybrid (lexical + vector) retrieval as chat,
            // investigate and MCP.
            let index = HybridIndex::build(&wb, &graph);
            let hits = index.search(&question, 3);
            if hits.is_empty() {
                println!("No matching region found for \"{question}\".");
                return Ok(());
            }
            for h in &hits {
                let note = if h.vector_only {
                    " (vector match only: no query word appears in it)"
                } else {
                    ""
                };
                println!("[{}] score {:.3}{note}", h.range.a1(), h.score);
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
            let mut sink = cli::TokenSink::new(&cancel);
            let mut first = true;
            let header = format!("\n--- {} ---\n", backend.name());
            let streamed = backend.stream(&req, &cancel, &mut |t| {
                if first {
                    sink.write(&header);
                    first = false;
                }
                sink.write(t);
            });
            // A reader that went away (`| head`) is not a failure.
            if sink.closed {
                return Ok(());
            }
            streamed.map_err(|e| CliError::Backend(e.to_string()))?;
            sink.write("\n");
            Ok(())
        }
        "graph" => {
            let parsed = cli::parse(cmd, rest, &[])?;
            let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
            let (_, graph) = workbook_and_graph(file)?;
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
            let parsed = cli::parse(cmd, rest, &[])?;
            let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
            let (wb, graph) = workbook_and_graph(file)?;
            if wb.sheets.is_empty() {
                return Err(CliError::Input("workbook has no sheets".into()));
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
            let sub = rest.first().map(String::as_str);
            let sub_args = rest.get(1..).unwrap_or_default();
            match sub {
                Some("list") => {
                    cli::parse(cmd, sub_args, &[])?.positionals(cmd, &[], 0)?;
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
                    let parsed = cli::parse(cmd, sub_args, &[])?;
                    let name = &parsed.positionals(cmd, &["<name>"], 0)?[0];
                    // The CARD.md read below joins `name` onto the
                    // root: a traversal name must never reach it.
                    valid_adapter_arg(cmd, name)?;
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
                            std::fs::read_to_string(&card)
                                .map_err(|e| CliError::Input(format!("{}: {e}", card.display())))?
                        );
                    }
                    Ok(())
                }
                Some("export") => {
                    let parsed = cli::parse(cmd, sub_args, &["--out"])?;
                    let name = &parsed.positionals(cmd, &["<name>"], 0)?[0];
                    existing_adapter(cmd, &root, name)?;
                    let loaded = adapter::load_adapter(&root, name);
                    let md = adapter::export_agents_md(&loaded);
                    match parsed.flag("--out") {
                        Some(path) => {
                            std::fs::write(path, &md).map_err(|e| format!("{path}: {e}"))?;
                            println!("wrote {} (adapter {name} v{})", path, loaded.version);
                        }
                        None => print!("{md}"),
                    }
                    Ok(())
                }
                Some("rollback") => {
                    let parsed = cli::parse(cmd, sub_args, &[])?;
                    let name = &parsed.positionals(cmd, &["<name>"], 0)?[0];
                    existing_adapter(cmd, &root, name)?;
                    let v = adapter::rollback(&root, name)?;
                    println!("adapter {name} rolled back to v{v}");
                    Ok(())
                }
                Some(other) => Err(cli::usage(cmd, format!("unknown subcommand {other:?}"))),
                None => Err(cli::usage(cmd, "missing subcommand")),
            }
        }
        "optimise" => {
            let parsed = cli::parse(cmd, rest, &["--adapter", "--rounds"])?;
            let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
            let name = parsed.flag("--adapter").unwrap_or("default").to_string();
            // Checked before the workbook loads and the rounds score:
            // an invalid name would only fail at the first commit.
            valid_adapter_arg(cmd, &name)?;
            let rounds = match parsed.flag("--rounds") {
                Some(v) => v
                    .parse::<usize>()
                    .ok()
                    .filter(|n| (1..=10).contains(n))
                    .ok_or_else(|| {
                        cli::usage(cmd, format!("--rounds must be 1 to 10, got {v:?}"))
                    })?,
                None => 3,
            };
            let (wb, graph) = workbook_and_graph(file)?;
            let hybrid = HybridIndex::build(&wb, &graph);
            let toolset = tools::Tools {
                wb: &wb,
                graph: &graph,
                hybrid: &hybrid,
                calc: &gridtalk_graph::eval::EvalCache::new(),
            };
            let gen = bench::generate_tasks_ex(&wb, 100, toolset.calc);
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
                )
                .map_err(|e| match e {
                    optimise::OptimiseError::Scoring(_) => CliError::Backend(e.to_string()),
                    optimise::OptimiseError::Commit(_) => CliError::Other(e.to_string()),
                })?;
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
            let sub_args = rest.get(1..).unwrap_or_default();
            match rest.first().map(String::as_str) {
                Some("generate") => {
                    let parsed = cli::parse(cmd, sub_args, &["--out", "--max"])?;
                    let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
                    let max: usize = match parsed.flag("--max") {
                        Some(v) => {
                            v.parse::<usize>().ok().filter(|n| *n >= 1).ok_or_else(|| {
                                cli::usage(
                                    cmd,
                                    format!("--max must be a positive integer, got {v:?}"),
                                )
                            })?
                        }
                        None => 100,
                    };
                    let (wb, _) = workbook_and_graph(file)?;
                    let gen =
                        bench::generate_tasks_ex(&wb, max, &gridtalk_graph::eval::EvalCache::new());
                    let tasks = gen.tasks;
                    if gen.skipped_formulas > 0 {
                        eprintln!(
                            "note: {} formula cells skipped (erroring/non-finite)",
                            gen.skipped_formulas
                        );
                    }
                    let json = serde_json::to_string_pretty(&tasks).map_err(|e| e.to_string())?;
                    match parsed.flag("--out") {
                        Some(out) => {
                            std::fs::write(out, &json).map_err(|e| format!("{out}: {e}"))?;
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
                    let parsed = cli::parse(
                        cmd,
                        sub_args,
                        &["--tasks", "--split", "--card", "--adapter"],
                    )?;
                    let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
                    // Checked before the workbook loads: a typo must
                    // not cost a full load first.
                    let split = parsed.flag("--split").unwrap_or("all").to_string();
                    if !matches!(split.as_str(), "train" | "dev" | "all") {
                        return Err(cli::usage(
                            cmd,
                            format!("unknown split {split:?}; expected train, dev or all"),
                        ));
                    }
                    let (wb, graph) = workbook_and_graph(file)?;
                    let hybrid = HybridIndex::build(&wb, &graph);
                    let toolset = tools::Tools {
                        wb: &wb,
                        graph: &graph,
                        hybrid: &hybrid,
                        calc: &gridtalk_graph::eval::EvalCache::new(),
                    };
                    let mut skipped_formulas = 0usize;
                    let tasks: Vec<bench::BenchTask> = match parsed.flag("--tasks") {
                        Some(path) => {
                            let text = std::fs::read_to_string(path)
                                .map_err(|e| CliError::Input(format!("{path}: {e}")))?;
                            serde_json::from_str(&text).map_err(|e| {
                                CliError::Input(format!("invalid tasks file {path}: {e}"))
                            })?
                        }
                        None => {
                            let gen = bench::generate_tasks_ex(&wb, 100, toolset.calc);
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
                            // `kind` is a free string: hand-written
                            // tasks (check_agrees, …) name their own.
                            // But a task needs ground truth: with no
                            // non-blank want/expected_text it is judged
                            // against `expected`, which defaults to 0
                            // when the file omits it — a truncated
                            // hand-written task would score as a
                            // 0-task instead of being caught here.
                            let blank_want =
                                !t.want.is_empty() && t.want.iter().all(|w| w.trim().is_empty());
                            if blank_want {
                                eprintln!(
                                    "gridtalk: warning: task {} has only blank want strings; they are ignored",
                                    t.id
                                );
                            }
                            let generated_kind = matches!(
                                t.kind.as_str(),
                                "cell_value" | "formula_result" | "what_if"
                            );
                            let no_truth = !generated_kind
                                && t.want.iter().all(|w| w.trim().is_empty())
                                && t.expected_text
                                    .as_deref()
                                    .is_none_or(|x| x.trim().is_empty())
                                && t.expected == 0.0;
                            if no_truth {
                                eprintln!(
                                    "gridtalk: warning: task {} has no non-blank want or expected_text; it is judged against expected = 0 (the default when the file omits it)",
                                    t.id
                                );
                            }
                            if !seen.insert(t.id.clone()) {
                                eprintln!("gridtalk: warning: duplicate task id {}", t.id);
                            }
                        }
                    }
                    if parsed.flag("--split").is_none() {
                        eprintln!(
                            "gridtalk: note: scoring all splits (dev included); use --split train to exclude dev"
                        );
                    }
                    let cfg = config_from_env();
                    let backend = backend_from_config(&cfg.inference);
                    warn_if_external(&cfg);
                    let bench_adapter = parsed.flag("--adapter").map(load_named_adapter);
                    let report = bench::run_bench_with(
                        &tasks,
                        &toolset,
                        backend.as_ref(),
                        &split,
                        bench_adapter.as_ref().map(|a| a.max_steps).unwrap_or(8),
                        bench_adapter.as_ref(),
                    )
                    .map_err(|e| CliError::Backend(format!("bench run failed: {e}")))?;
                    let card =
                        bench::render_card_with_skipped(&report, tasks.len(), skipped_formulas);
                    match parsed.flag("--card") {
                        Some(path) => {
                            std::fs::write(path, &card).map_err(|e| format!("{path}: {e}"))?;
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
                Some(other) => Err(cli::usage(cmd, format!("unknown subcommand {other:?}"))),
                None => Err(cli::usage(cmd, "missing subcommand")),
            }
        }
        "investigate" => {
            // Flags are flags only before the question starts;
            // inside the question they are literal text.
            let parsed = cli::parse_with(
                cmd,
                rest,
                &[
                    cli::Flag::Value("--adapter"),
                    cli::Flag::Value("--selection"),
                    cli::Flag::Value("--history-file"),
                ],
                Some(1),
            )?;
            let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
            let adapter_name = parsed.flag("--adapter");
            if let Some(name) = adapter_name {
                valid_adapter_arg(cmd, name)?;
            }
            let selection = parsed.flag("--selection");
            if let Some(sel) = selection {
                if tools::parse_range_uncapped(sel).is_none() {
                    return Err(cli::usage(
                        cmd,
                        format!("--selection must be a range like Sheet1!A1:B2, got {sel:?}"),
                    ));
                }
            }
            let history: Vec<gridtalk_index::ChatTurn> = match parsed.flag("--history-file") {
                Some(path) => {
                    let text = std::fs::read_to_string(path)
                        .map_err(|e| CliError::Input(format!("{path}: {e}")))?;
                    serde_json::from_str(&text).map_err(|e| {
                        CliError::Input(format!(
                            "invalid history file {path} (a JSON array of {{\"question\", \"answer\"}} turns): {e}"
                        ))
                    })?
                }
                None => Vec::new(),
            };
            let question = parsed.text();
            if question.trim().is_empty() {
                return Err(cli::usage(cmd, "missing the question"));
            }
            let (wb, graph) = workbook_and_graph(file)?;
            let hybrid = HybridIndex::build(&wb, &graph);
            let tools = tools::Tools {
                wb: &wb,
                graph: &graph,
                hybrid: &hybrid,
                calc: &gridtalk_graph::eval::EvalCache::new(),
            };
            let cfg = config_from_env();
            let backend = backend_from_config(&cfg.inference);
            warn_if_external(&cfg);
            let cli_adapter = adapter_name.map(load_named_adapter);
            let sidecar = adapter::sidecar_rules(std::path::Path::new(file));
            let inv = investigate::investigate_with_context(
                &question,
                &tools,
                backend.as_ref(),
                cli_adapter.as_ref().map(|a| a.max_steps).unwrap_or(8),
                cli_adapter.as_ref(),
                sidecar.as_deref(),
                &history,
                selection,
            )
            .map_err(|e| CliError::Backend(format!("backend failed: {e}")))?;
            for step in &inv.trail {
                println!(
                    "[tool] {} ({}) -> {}",
                    step.tool,
                    // Args carry the raw question: flatten newlines
                    // so they cannot forge citations/answer markers,
                    // and keep " -> " out so a line's first " -> "
                    // always starts its result (hermit's judge
                    // counts only results as carried).
                    step.args.replace('\n', " | ").replace(" -> ", " - > "),
                    step.result.replace('\n', " | ")
                );
            }
            println!("\ncitations: {}", inv.citations.join(", "));
            println!("\n--- answer ({}) ---\n{}", backend.name(), inv.answer);
            Ok(())
        }
        "open-in-gridcraft" => {
            let parsed = cli::parse(cmd, rest, &[])?;
            let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
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
            let (model, prompt) = file_and_text(cmd, rest, "model.gguf", "prompt")?;
            // A model that is not there is an input error (exit 3),
            // not a backend failure.
            if !std::path::Path::new(&model).is_file() {
                return Err(CliError::Input(format!("{model}: no such model file")));
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
            let mut sink = cli::TokenSink::new(&cancel);
            let generated = backend.generate_with_stats(&req, &cancel, &mut |tok| sink.write(tok));
            if sink.closed {
                return Ok(());
            }
            let stats = generated.map_err(|e| CliError::Backend(e.to_string()))?;
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
            let parsed = cli::parse_with(
                cmd,
                rest,
                &[
                    cli::Flag::Repeated("--root"),
                    cli::Flag::Value("--timeout"),
                    cli::Flag::Value("--memory-budget"),
                    cli::Flag::Switch("--allow-write"),
                ],
                None,
            )?;
            parsed.positionals(cmd, &[], 0)?;
            let mut options = mcp::McpOptions::default();
            if parsed.switch("--allow-write") {
                eprintln!("gridtalk: warning: --allow-write is a no-op (no write tools exist yet)");
            }
            for root in parsed.values("--root") {
                options
                    .add_root(std::path::Path::new(root))
                    .map_err(|e| cli::usage(cmd, e))?;
            }
            if let Some(secs) = parsed.flag("--timeout") {
                let secs: u64 = secs
                    .parse()
                    .ok()
                    .filter(|s| (1..=86_400).contains(s))
                    .ok_or_else(|| {
                        cli::usage(
                            cmd,
                            format!("--timeout must be 1 to 86400 seconds, got {secs}"),
                        )
                    })?;
                options.call_timeout = std::time::Duration::from_secs(secs);
            }
            if let Some(mib) = parsed.flag("--memory-budget") {
                let mib: u64 = mib
                    .parse()
                    .ok()
                    .filter(|m| (1..=1 << 30).contains(m))
                    .ok_or_else(|| {
                        cli::usage(
                            cmd,
                            format!("--memory-budget must be a positive number of MiB, got {mib}"),
                        )
                    })?;
                options.memory_budget = mib << 20;
            }
            Ok(mcp::serve_stdio(options)?)
        }
        "tools" => {
            let sub_args = rest.get(1..).unwrap_or_default();
            match rest.first().map(String::as_str) {
                Some("list") => {
                    cli::parse(cmd, sub_args, &[])?.positionals(cmd, &[], 0)?;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&tools::ToolRequest::schemas()).unwrap()
                    );
                    Ok(())
                }
                Some("call") => {
                    let parsed = cli::parse(cmd, sub_args, &[])?;
                    let positional = parsed.positionals(cmd, &["<file>", "'<json>'"], 0)?;
                    let (file, json) = (&positional[0], &positional[1]);
                    // The request is checked before the workbook loads.
                    let req: tools::ToolRequest = serde_json::from_str(json)
                        .map_err(|e| cli::usage(cmd, format!("invalid tool request JSON: {e}")))?;
                    let (wb, graph) = workbook_and_graph(file)?;
                    let hybrid = HybridIndex::build(&wb, &graph);
                    let toolset = tools::Tools {
                        wb: &wb,
                        graph: &graph,
                        hybrid: &hybrid,
                        calc: &gridtalk_graph::eval::EvalCache::new(),
                    };
                    let result = toolset.call(&req);
                    // Print the result exactly once; a failed call
                    // exits 5 without repeating the error text.
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
                    );
                    if result.ok {
                        Ok(())
                    } else {
                        Err(CliError::ToolFailed)
                    }
                }
                Some(other) => Err(cli::usage(cmd, format!("unknown subcommand {other:?}"))),
                None => Err(cli::usage(cmd, "missing subcommand")),
            }
        }
        "spike" => {
            let parsed = cli::parse(cmd, rest, &[])?;
            let file = &parsed.positionals(cmd, &["<file>"], 0)?[0];
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
                Err(CliError::Input("model header did not validate".into()))
            }
        }
        "models" => {
            let mgr = gridtalk_infer::models::ModelManager::default_cache();
            let sub_args = rest.get(1..).unwrap_or_default();
            match rest.first().map(String::as_str) {
                Some("import") => {
                    let parsed = cli::parse(cmd, sub_args, &[])?;
                    let src = &parsed.positionals(cmd, &["<file.gguf>"], 0)?[0];
                    let entry = mgr
                        .import(std::path::Path::new(src))
                        .map_err(|e| CliError::Input(e.to_string()))?;
                    println!(
                        "imported {} ({} bytes) -> {}",
                        entry.name,
                        entry.bytes,
                        entry.path.display()
                    );
                    Ok(())
                }
                Some("remove") => {
                    let parsed = cli::parse(cmd, sub_args, &[])?;
                    let name = &parsed.positionals(cmd, &["<name>"], 0)?[0];
                    mgr.remove(name).map_err(|e| e.to_string())?;
                    println!("removed {name}");
                    Ok(())
                }
                Some("download") => {
                    let parsed = cli::parse(cmd, sub_args, &[])?;
                    let positional = parsed.positionals(cmd, &["<url>"], 1)?;
                    let url = &positional[0];
                    // Default name: the URL's last path segment.
                    let name = match positional.get(1) {
                        Some(n) => n.clone(),
                        None => url
                            .split(['?', '#'])
                            .next()
                            .and_then(|u| u.rsplit('/').next())
                            .filter(|n| !n.is_empty())
                            .ok_or_else(|| {
                                cli::usage(cmd, "cannot derive a model name from the URL; pass one")
                            })?
                            .to_string(),
                    };
                    let entry = mgr.download_http(url, &name).map_err(|e| e.to_string())?;
                    println!(
                        "downloaded {} ({} bytes) -> {}",
                        entry.name,
                        entry.bytes,
                        entry.path.display()
                    );
                    Ok(())
                }
                Some(other) => Err(cli::usage(cmd, format!("unknown subcommand {other:?}"))),
                None => {
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
        // The command table and this match list the same commands
        // (help and version are answered above).
        other => Err(CliError::Other(format!(
            "command {other:?} is in the help table but not implemented"
        ))),
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
        let url = cfg
            .inference
            .base_url
            .as_deref()
            .unwrap_or(gridtalk_infer::DEFAULT_BASE_URL);
        eprintln!(
            "gridtalk: note: external backend {url} will receive spreadsheet passages (including cell values) verbatim. Use a local model or a redacting proxy if that is not acceptable."
        );
    }
    if let Some(warning) = gridtalk_infer::cleartext_warning(&cfg.inference) {
        eprintln!("gridtalk: warning: {warning}");
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

/// An adapter name on the command line must be a valid one (usage
/// error otherwise): it is joined onto the adapters root.
fn valid_adapter_arg(cmd: &str, name: &str) -> Result<(), CliError> {
    if adapter::valid_adapter_name(name) {
        Ok(())
    } else {
        Err(cli::usage(cmd, format!("invalid adapter name {name:?}")))
    }
}

/// A valid adapter name that exists under `root` (input error when
/// it does not).
fn existing_adapter(cmd: &str, root: &std::path::Path, name: &str) -> Result<(), CliError> {
    valid_adapter_arg(cmd, name)?;
    if root.join(name).is_dir() {
        Ok(())
    } else {
        Err(CliError::Input(format!("adapter {name:?} not found")))
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

/// Load a workbook, printing the reader's warnings (parts it skipped
/// or could not import) to stderr so a partial import is never silent.
fn load_workbook(file: &str) -> Result<WorkbookSnapshot, CliError> {
    let wb = load_path(std::path::Path::new(file))
        .map_err(|e| CliError::Input(format!("{file}: {e}")))?;
    for w in &wb.warnings {
        eprintln!("gridtalk: warning: {file}: {w}");
    }
    Ok(wb)
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
