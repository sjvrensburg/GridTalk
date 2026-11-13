# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

GridTalk is a local-first "talk to your spreadsheet" app in Rust: ingest CSV/TSV/XLSX/XLSM/XLS/XLSB/ODS → lift a property graph → hybrid search → answers that cite specific cell ranges, with inference running locally (llama.cpp over Vulkan) by default.

## Commands

CI runs exactly these three (on Ubuntu and Windows). All must pass:

```bash
cargo fmt --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

- Single crate: `cargo test --locked -p gridtalk-graph`
- Single test: `cargo test --locked -p gridtalk-graph <test_name_substring>`
- MSRV is 1.95 (`rust-version` in the workspace); CI also runs `cargo test` on that toolchain, so don't use newer std/language features.
- CI also runs the Python mirror tests: `python -m pytest scripts/hermit -q` (only pytest needed; `conftest.py` stubs `hermit-harness` when it isn't installed).
- Build prerequisites: the `vendor/gridcraft` submodule must be checked out (`git submodule update --init`), because `gridtalk-graph` path-depends on it. `gridtalk-infer` builds llama.cpp with Vulkan, so it needs the Vulkan SDK (`glslc`, headers, SPIRV-Headers) and CMake. The first build is slow.
- Run the CLI: `cargo run -p gridtalk-app -- <command>` (binary `gridtalk`; `gridtalk help` lists the commands, such as `ask`, `investigate`, `graph`, `tools`, `mcp`, `bench`, `optimise`, `adapter`, `models` and `generate`, with their usage and the exit codes; `gridtalk help <cmd>` shows one). Commands are declared once in `crates/gridtalk-app/src/cli.rs` (`COMMANDS`), which also holds the strict flag parser and `CliError` (exit codes 1–5). Desktop UI: `cargo run -p gridtalk-ui`. Sample inputs are in `tests/fixtures/`.
- `.mcp.json` launches `gridtalk mcp`, so it needs a `gridtalk` binary on `PATH`. Install it with `cargo install --locked --path crates/gridtalk-app`. Without `--locked`, `cargo install` ignores `Cargo.lock` and pulls a newer `llama-cpp-sys-2` than the one paired with the pinned `llama-cpp-2`.

## Architecture

The workspace crates form a strict pipeline. Dependencies only point down this list:

1. **`gridtalk-graph`** is the core data layer. `load_path` turns a file into a `WorkbookSnapshot`. CSV/TSV are parsed in-crate; every other format goes through the vendored Gridcraft `xlsx`/`model`/`core` crates, with the reader chosen by `gridcraft_xlsx::sniff` (content, not extension). `.xlsb` and `.ods` are values only (Gridcraft imports their formulas as saved results; `values_only_warning` comes first), and `to_package_bytes` writes the `.xlsx`-family kind the save path names. Defined names are kept in the snapshot. `detect_regions` and `build_graph` lift the snapshot into a `Graph` (workbook → sheet → region → column; a labelled block's value column is split into one single-cell column node per row, named by its label) and return invariant violations next to it. `depends_on` edges come from Gridcraft's precedent analysis (`lineage.rs`: 3D refs, defined names, self-edges for running totals). Callers treat a non-empty violation list as an error. `render_passage` produces the text that gets indexed. The functions in `eval.rs` (which back the tools, the UI and what-if) convert the snapshot with `to_gridcraft_workbook` and evaluate it with Gridcraft's `Calc` engine (`calc.rs`). GridTalk has no evaluator of its own, so if a formula is wrong, look in Gridcraft's function library. Downstream code works on snapshots and never touches live grid cells. A snapshot read through Gridcraft also keeps the Gridcraft workbook it was read from (`WorkbookSnapshot::source`, decision 0005): `to_gridcraft_workbook` overlays the snapshot's changed cells on it, so evaluation and Save keep styles, merges, tables, array formulas, error values and the date system. Edit the snapshot, never `source`. Repeated evaluation of one snapshot goes through `eval::EvalCache` (one recalculation, reused; what-if runs on a clone), which `Tools`, MCP sessions, bench and the UI worker all hold.
2. **`gridtalk-index`** indexes regions (not cells) with a linear-scan lexical index (`Index`). `HybridIndex` fuses it with an n-gram embedding ranker (`embed.rs`) using weighted reciprocal rank fusion. Hits return an `excerpt` capped at `MAX_PASSAGE_CHARS` (the header plus rows matching the query), because a whole large region would overflow an agent's or a local model's context. The vector ranker is morphological, not semantic: `Hit::vector_only` marks hits that only it found.
3. **`gridtalk-infer`** defines the `ChatBackend` trait (streaming with an `AtomicBool` cancel flag). It has three backends: the local llama.cpp backend (`llama_backend.rs`), an OpenAI-compatible external client over its own HTTP/rustls `transport.rs`, and a deterministic echo backend that is used when no model is configured. `backend_from_config` chooses the backend. Config is TOML (`config::Config`, see `config.example.toml`), found through `GRIDTALK_CONFIG` / `GRIDTALK_CONFIG_DIR`. `models.rs` handles GGUF import/download into the model cache.
4. **`gridtalk-adapter`** holds versioned, human-readable harness config: `ADAPTER.md` rules, `adapter.toml` knobs, `CARD.md` eval record, `skills/*.md`, and `vN/` snapshots. `gridtalk optimise` evolves these files without touching model weights. Prompt layering at investigate time: built-in prompt → adapter rules → matched skills → per-workbook `<workbook>.gridtalk.md` sidecar.
5. **`gridtalk-app`** is the `gridtalk` CLI (headless shell). The key design is `tools.rs`: one serde-typed `ToolRequest`/response implementation (`#[serde(tag = "tool")]`, `deny_unknown_fields`) drives three consumers: the `investigate.rs` planner loop, `gridtalk tools call` JSON, and the stdio MCP server in `mcp.rs`. Add or change a tool there, never per consumer. Every tool is read-only, `what_if` included. `bench.rs`/`optimise.rs` generate and score tasks for adapter optimisation.
6. **`gridtalk-ui`** is the egui/eframe desktop app (crates.io egui 0.37). It has a dual view (grid + graph) with shared selection sync, a chat pane, formula evaluation on a worker thread (`eval_worker.rs`), and the graph and index rebuild after an edit on another (`RebuildJob`; the graph (`GraphCache`) and the index update incrementally from the dirty ranges).

## Constraints and decisions (see `docs/decisions/`)

- **No redaction layer** (0002). The local backend keeps everything on the machine. The external backend sends passages, cell values included, verbatim. Don't add redaction inside GridTalk; that belongs in an external proxy.
- **One local engine** (0004): llama.cpp via `llama-cpp-2` (pinned `=0.1.160`, `vulkan` feature). Old `engine = "candle"|"mistralrs"` config values are accepted and mapped to llama.cpp. Targets are Linux and Windows x86_64 only; macOS is intentionally unsupported.
- **Gridcraft is vendored, not forked** (0001). `vendor/gridcraft` is a submodule excluded from the workspace and pinned in `UPSTREAM.md` (update that pin when syncing). Depend only on its public model/calc surface and never edit engine internals.
- **Keep the Gridcraft workbook** (0005): a snapshot read through Gridcraft carries its source workbook, and conversion back overlays only the changed cells.
- **UI composition, not embedding** (0003). Gridcraft's egui fork is incompatible with GridTalk's egui, so heavy editing happens by launching Gridcraft (`open-in-gridcraft`, `GRIDTALK_GRIDCRAFT_BIN`) and re-snapshotting the file afterwards.
- The MCP server reserves stdout for JSON-RPC frames, so log to stderr (`docs/mcp.md`). Only the stdio thread writes frames: tool work runs as a `Job` on a worker thread and comes back as a `Done` that `McpServer::finish` applies to the sessions (`Arc<OpenWorkbook>`), so session state changes only on that thread.
- `docs/plan.md` is the phased plan with honest checkboxes. Keep it accurate when you land or descope work.

## Conventions

- Put `use` declarations at module scope and mark public API items `pub`.
- Snapshot rows are `Arc`-shared (`gridtalk_graph::Row`) so copies are cheap: read cells as usual, write through `SheetSnapshot::cell_mut` (or `Arc::make_mut`), and build sheets with `SheetSnapshot::new`.
