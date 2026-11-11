# GridTalk implementation plan

Vision: "Talk to your spreadsheet without data ever leaving your PC." A
lightweight, local-first evolution of ExcelGRAG, built on Gridcraft.

## Privacy model

No redaction layer in GridTalk. Local backend = nothing leaves the PC.
External OpenAI-compatible backend = passage (with cell values) sent
verbatim. A redacting proxy is a separate future project; GridTalk
supports it unchanged by pointing `base_url` at the proxy.

## Phase 1: Architecture & scaffolding — DONE (scaffold)

- [x] Public repo `sjvrensburg/GridTalk`
- [x] Gridcraft vendored as submodule at `vendor/gridcraft`, pinned in `UPSTREAM.md`
- [x] Workspace crates: `gridtalk-adapter`, `gridtalk-graph`, `gridtalk-index`, `gridtalk-infer`, `gridtalk-app`, `gridtalk-ui`
- [x] Decision records 0001 (fork-vs-embed), 0002 (no-redaction)
- [x] Settings skeleton (`gridtalk-infer::config`, `config.example.toml`)
- [x] CI: fmt, clippy, test
- [ ] Rebase cadence: monthly upstream sync (half-day budget)

## Phase 2: Data ingestion & grid UI — IN PROGRESS

- [x] Snapshot interchange type (`WorkbookSnapshot` / `SheetSnapshot`)
- [x] CSV/TSV ingestion incl. quoted fields, type inference, formulas
- [x] Typed `UnsupportedFormat` errors for XLSB/XLS/XLSM (explicit, not silent)
- [x] XLSX/ODS via Gridcraft `xlsx` crate (`load_gridcraft_bytes`, XLSX round-trip + ODS fixture tests). Known limits: ODS is data-only (Gridcraft imports formula results as constants, so what-if on `.ods` changes nothing; the reader's warning says so), and XLSX save writes the source Gridcraft workbook with the edits overlaid, so it keeps what Gridcraft read (decision 0005)
- [x] Incremental dirty-range re-snapshot (`IncrementalSnapshot`: 500 ms debounce, dirty-range list, revision counter). Known limits: the UI never consumes the dirty ranges (each edit rebuilds the whole graph and index), and an edit that cannot be applied is reported in the UI status
- [x] egui grid+chat shell (`gridtalk-ui`: editable grid, chat pane, settings window) — on egui/eframe directly; Gridcraft `ui-egui` ribbon reuse remains a later swap

## Phase 3: Graph-RAG & inference — IN PROGRESS

- [x] Region/header detection, lifted graph (workbook→sheet→region→column)
- [x] Lifted-edge invariant check re-derived from raw formulas
- [x] Lexical TF-IDF index with cited passages (tantivy swap keeps this API)
- [x] `ChatBackend` trait: streaming + cancellation
- [x] OpenAI-compatible external backend (verbatim send over http or certificate-verified https via rustls; extra CAs from `GRIDTALK_CA_FILE`)
- [x] Deterministic echo backend for tests/demos
- [x] Inference: single llama.cpp-over-Vulkan engine (decision 0004); `gridtalk spike <model.gguf>` validates a GGUF header (header-only, not a load test)
- [x] Local embeddings + hybrid fusion: offline n-gram `EmbeddingModel` (bge-small ONNX remains an adapter swap behind the same trait), cosine vector index, Reciprocal Rank Fusion of lexical+vector rankings (`HybridIndex`, used by UI chat, investigate, MCP and CLI `ask`)
- [x] Model management: `~/.cache/gridtalk/models` (or `GRIDTALK_MODEL_CACHE`), GGUF metadata parser, `gridtalk models import|download|remove|list`, http/https download following up to 5 redirects (absolute or relative; never https→http), chunked or Content-Length bodies with truncation detected either way
- [x] Investigate tool-loop harness (Phase 3b): `gridtalk investigate` — search → expand → get_cells → list_formulas → what_if tools with a recorded trail and citations; formula evaluation and what-if recomputation through Gridcraft's calc engine (`gridcraft-calc`: its full function library incl. VLOOKUP/INDEX/MATCH/PV/SUMPRODUCT, defined names, 3D references, Excel rounding), so a what-if change flows through lookup tables instead of stopping at cached values; unparseable formulas are typed parse errors, and so are the cells that depend on them. Gridcraft is the only evaluator
- [x] Bounded search passages: regions are indexed whole, but a hit returns at most 4,000 characters — header plus the rows matching the query, labelled with their sheet rows, flagged `truncated`, with a note counting rows shown, matching and total and naming a bounded range to read next

## Phase 4: Dual-view & polish — IN PROGRESS

- [x] `ViewState` two-way sync state machine (node→range, cell→node) with tests
- [x] Terminal dual-view rendering (`gridtalk view`)
- [x] egui graph pane: force layout computed in a worker thread, cached per revision, click-to-select nodes; formula results evaluated on a worker thread (previous result shown until the new one lands); chat in a resizable bottom panel
- [x] Split / Graph-full / Grid-full modes persisted per workbook (platform config dir: `~/.config/gridtalk/ui-state.json` on Linux, `%APPDATA%\gridtalk\ui-state.json` on Windows)
- [x] Settings window (config path, verbatim-send notice, llama.cpp GPU layers and context limit)
- [x] First-run wizard (privacy posture → backend/model → ready; writes `config.toml`, persisted `first_run_done`)
- [x] Installers: tag-triggered release workflow (Linux/Windows archives attached to the GitHub Release; macOS is not a target, decision 0004) + `scripts/install.sh`; release workflow is exercised on the first `v*` tag

- [x] Real-model generation: llama.cpp over Vulkan generates from real GGUFs incl. Qwen3-MoE (see Single-engine pivot below)
- [x] Gridcraft ribbon reuse: companion-app composition (`gridtalk open-in-gridcraft`), decision 0003 — embedding rejected (forked egui + `SheetApp` ownership)

## Validation

`cargo test --workspace`, `cargo clippy --workspace --all-targets`,
`cargo fmt --check` on every push. Synthetic fixtures only — never
commit a real spreadsheet.

## Code review (multi-agent, adversarially interrogated)

A 6-reviewer / 6-skeptic workflow produced 130 surviving findings
(7 refuted, including the standard-IDF and provably-safe unwrap
claims). All high and medium findings, and the low findings with
code-level fixes, were addressed in the review-fix commit:
formula-ref range expansion and literal skipping, cross-sheet
lifting, stream CSV parsing, evaluator `$` refs / trailing tokens
/ aggregate semantics / depth and area caps, HTTP status +
chunked + timeouts + key-over-cleartext refusal, GGUF depth and
overflow guards, header-only model I/O with atomic renames, EOS
by token id with incremental BPE decode, hybrid `vector_only`
flagging, async (worker-thread) chat in the UI, formula-preserving
cell editing with save-to-disk, sheet tabs, virtualised rows, a
real file dialog, and CI/release hardening (least-privilege
permissions, `--locked`, CI jobs for all crates, create-once releases
with a version gate). Remaining known limits: the release
workflow is first exercised on a `v*` tag, and Windows Vulkan
runtime behaviour is verified only by inspection (no Windows test
hardware).

## Single-engine pivot (feat/llama-cpp-vulkan, decision 0004)

- [x] Local inference is llama.cpp over Vulkan only; candle and
  mistral.rs backends and the engine ladder are removed
- [x] Qwen3-30B-A3B MoE generates locally (~9.7 tok/s incl. load)
- [x] Platforms: Linux + Windows; macOS dropped from CI/release
- [x] llama.cpp GPU-layer/context settings exposed in the UI (Settings window; `n_gpu_layers` / `n_ctx` in `[inference]`)

## Codebase review follow-ups (2026-11-09)

From `docs/codebase-review.md`, after verifying its claims (see its
"Verification" section). Ordered by priority.

Phase 0 — silent data loss and wrong numbers:

- [x] Edits anywhere in Excel's grid are applied unless they would push the workbook past `MAX_SNAPSHOT_CELLS`; dropped edits and their reason are shown in the UI status; the UI drains dirty ranges after each rebuild
- [x] A formula Gridcraft cannot parse gets a `#NAME?` value before recalculation, so its dependents fail instead of using its cached value; first direct `calc.rs` tests
- [x] One capped reader (`read_workbook_bytes`: regular files only, at most cap + 1 bytes) for `load_path`, MCP `load` and `hash_of`
- [x] Gridcraft `ReadReport` warnings kept in `WorkbookSnapshot::warnings` and surfaced in `open_workbook`, CLI stderr and the UI status (incl. ODS "data-only import")
- [x] UI asks once per opened .xlsx/.ods workbook before an .xlsx save drops formatting, merges, charts and tables (until the next item lands); the save buttons say so on hover

Phase 1 — structural:

- [x] Keep the original Gridcraft workbook alongside the snapshot (decision 0005): `WorkbookSnapshot::source`; `to_gridcraft_workbook` overlays only changed cells (keeping their style), so styles, merges, tables/structured refs, array formulas, the date system and error cells survive save and eval. The UI's "XLSX save loses formatting" prompt is gone. Editing a table header renames the column and its structured references. Known limit: a snapshot whose sheets were renamed, added or removed falls back to snapshot-only conversion
- [x] One cached calc engine per workbook revision (`eval::EvalCache`/`Evaluator`) for MCP sessions, `Tools` (investigate, `tools call`), bench task generation and the UI eval worker; what-if runs on a clone, recalculating only the override's dependents. Measured on 5,000 `SUM(A:A)` rows: the first evaluation still costs the full recalculation (~0.9 s), the next 100 cell reads take 17 µs together. A what-if whose override every row depends on still recalculates every row (Gridcraft's cost)
- [x] UI keeps the selection after an edit, and the graph layout when the edit leaves the graph's nodes and edges unchanged. Each rebuild hands its worker a copy of the snapshot that shares every row (the formula worker shares that copy)
- The incremental-rebuild item was split on 2026-11-10 after measuring (release build, one cell edited; test workbooks of 30k, 300k and 1.2M cells across 1–4 sheets). Before Phase 1b, each edit cost 29 ms, 0.58 s and 2.7 s, all on the UI thread, so a large workbook froze for seconds per edit. `check_invariants` was ~60% of `build_graph`, and the index update cost as much as a full index build

Phase 1b — edit latency (independent of the lineage rules):

- [x] Rebuild the graph and index on a worker thread after an edit (`RebuildJob`): the UI never blocks; chat and Save wait for a rebuild in flight; the rebuild's snapshot copy is shared with the formula worker
- [x] Cheaper graph build and invariant check: `NodeLocator` groups region/column nodes by sheet, so a cell lookup is an integer scan of one sheet's nodes instead of a case-folded name comparison against every node. `build_graph` per edit: 17→4.6 ms, 413→54 ms, 1.73→0.33 s
- [x] Incremental index from dirty ranges (`HybridIndex::update_dirty`): a region that kept its range and that no edit touched keeps its lexical document and vectors without being re-rendered. Index update per edit: 162→43 ms (300k cells), 0.96→0.29 s (1.2M). Total per edit, now off the UI thread: 18 ms, 97 ms, 0.63 s
- [x] No full snapshot copy on the UI thread per edit: `SheetSnapshot::cells` rows are `Arc`-shared (`Row`), so copying a snapshot for the rebuild and formula workers copies one pointer per row, and an edit copies only its row (`SheetSnapshot::cell_mut`). Snapshot copy at 1.2M cells: 33 ms → 1.5 ms (300k: 5.9 → 0.46 ms); a flush that has to copy a row still held by a worker takes ~1 ms

Phase 2 — agent-facing correctness:

- [x] `get_cells` returns the evaluated value plus the typed cached value (formula cells carry `result`, `cached_result` and `error`; one calc batch per call); cell parsing rejects refs outside the sheet grid, and the error says so
- [x] CLI `ask` uses `HybridIndex` (vector-only hits are marked); investigate's what-if discovery finds text and boolean dependents, from one override copy (`Evaluator::with_override`) instead of a copy per candidate
- [x] Graph lineage from Gridcraft's dependency analysis (`lineage.rs`, `gridcraft_calc::eval::precedents`): 3D refs (`Jan:Mar!A2` reads A2 on each sheet in the span; it was misread as columns JAN:MAR), defined names (resolved through their formulas, sheet scope first, nested names to depth 8), range-end sheet qualifiers checked (`Sheet2!A1:Sheet1!B2`, which Gridcraft rejects, keeps its two cells instead of inventing a `Sheet2!A1:B2` rectangle), and running totals (a column reading itself keeps its self-edge). Areas map to the column nodes they touch, exactly rather than by sampling large ranges; the invariant check maps the same areas by an independent coverage rule. A formula Gridcraft cannot parse falls back to the lexical extractor. Structured references still lift no edges (Gridcraft flags them dynamic). Cost: `build_graph` per edit 4.6→6.3 ms, 54→72 ms, 0.33→0.43 s (30k/300k/1.2M cells)
- [x] Incremental graph rebuild from dirty ranges (`GraphCache`, `build_graph_cached`; the UI's rebuild job carries the cache like the index): only formulas inside the edited ranges are re-parsed and only the sheets they touch re-detect their regions; the cache resets when the sheets or defined names change. Lifting and the invariant check still run over the whole graph from the cached areas, so edges into a sheet whose regions shifted stay right (a test compares cached and fresh builds after each of a sequence of edits). Each cached rebuild also re-parses a rotating window of 256 reused entries from the cells' current text (`CACHE_AUDIT_SAMPLE`); a mismatch, which would mean the edit tracking missed a change, drops the cache and rebuilds from scratch with a warning on stderr, so the invariant check's shared areas cannot hide a stale cache for long. `build_graph` per edit: 6.3→2.8 ms, 72→38 ms, 0.43→0.20 s (30k/300k/1.2M cells). Not done: an edited region is still re-indexed whole, which dominates the remaining 0.29 s at 1.2M cells
- [x] `workbook_info` tool (sheets, defined names, regions, load warnings, limits); MCP annotations (`readOnlyHint`, `destructiveHint: false`) on every tool; A1 examples, the quoting rule and limits in tool descriptions. MCP's tool list is now derived from `ToolRequest::schemas()`, so a tool is defined once
- [x] One typed error envelope (`ErrorKind`): tool failures are `isError` results with `{"error": {"kind", "message"}}`, protocol errors carry `data.kind`; `expand` capped at 200 edges with `total_edges`/`truncated`; search reports `limit`/`limit_clamped`; re-opening a path (canonicalised) reuses its session. MCP builds each request through `ToolRequest`'s serde
- [x] Reader warnings reach the agent after a silent reload: when `fresh()` reloads a changed file, its new `warnings` are reported on the next tool response, not only by `open_workbook` (`reloaded: {"warnings": [...]}`)
- [x] Invariant check's coverage rule without the quadratic scan: `AreaNodes` indexes each sheet's nodes by the columns they span, so `nodes_by_coverage` tests a column against the few nodes on it instead of every node on the sheet (a test checks it gives the old scan's answer for every small area over three layouts). A 250-column region with 2,000 row-total formulas: `build_graph` 175 → 16 ms
- [x] Bench what-if tasks from Gridcraft lineage and typed values: `bench.rs` tries precedents from `lineage::Lineage` (names and 3D refs included, clipped to the cells a sheet has), and any typed change (`Value::differs_from`) makes a task. Text and boolean results become tasks with `expected_text`, judged by `judge_text` (the last value stated, as `judge_number` wants the final number: only citations may follow its last whole-word occurrence) in GridTalk and by the mirrored `_text_present` in `scripts/hermit`; numeric questions keep their wording, so their splits do not move. No bench scores are committed, so there is no baseline to redo in the repo; re-run any local CARDs, since the task set changed

Phase 3 — hardening (done 2026-11-10):

- [x] MCP `--root <dir>` allowlist (canonicalised, rechecked on reload; a missing file outside the roots reads as not allowed); `open_workbook` and tool calls run on worker threads (at most 4) with a `--timeout` (default 120 s) answered as a `timeout` error; a `--memory-budget` (default 4 GiB) next to `MAX_WORKBOOKS`, estimated as 16 × `WorkbookSnapshot::approx_bytes` (measured 12–25× on 5k–1.2M-cell workbooks). A timed-out call cannot be interrupted: it holds its slot until Gridcraft returns. Workbooks load one at a time and the budget is charged once a load finishes, so the peak can exceed it by one workbook being built; a reload that alone exceeds the budget closes its session with `over_budget`
- [x] Unpacked-size pre-scan of XLSX/ODS packages before Gridcraft parses them: the zip directory may declare at most 512 MiB. It reads only the directory, so a package that understates its sizes still passes; Gridcraft's reader then counts what it really unpacks (512 MiB per part, 1 GiB in all)
- [x] `create_new` temp files (adapter files and version directories, UI prefs, config and saves); 64 GiB model import cap, also for a source that grows while copied; cleartext-`http` warning to remote hosts even without an API key (CLI, every UI chat answer, Settings and the first-run wizard); GGUF header checked while a download streams in (the magic and version on the first 8 bytes, the full header from 64 KiB, at each doubling)

Phase 4 — tests and CI (done 2026-11-11):

- [x] Subprocess MCP stdio test (`crates/gridtalk-app/tests/mcp_stdio.rs`): runs the `gridtalk` binary to check calls refused before `initialize`, the handshake, a notification getting no reply, `tools/list`, an open/`get_cells`/reopen/close session under `--root`, tool errors (`unknown_workbook`, `path_not_allowed`) versus protocol errors (`unknown_tool`, a bad argument, an unknown method, a frame that is not JSON), a final request without a newline, exit 0 at end of input with nothing extra on stdout, and exit 1 with a `gridtalk:` message and no frames for bad flags
- [x] Synthetic `.xlsx` files generated in tests with Gridcraft's writer (`crates/gridtalk-graph/tests/limits.rs`): values and formulas round-trip; the 4M-cell cap at and one row past the boundary, at Excel's last cell, and summed across sheets; the 32 MiB and 256 MiB file caps at and one byte past the cap (CSV, XLSX, ODS); empty workbooks (one or several empty sheets, an empty CSV, an empty sheet next to a filled one) load, build an empty graph and save
- [x] CI runs `pytest scripts/hermit` (a `conftest.py` stands in for `hermit-harness`, so only a pinned pytest is installed), and the MSRV job runs `cargo test --workspace` on 1.95 instead of `cargo check`

Phase 5 — features and UX:

- [ ] Accept `.xlsm`, `.xls` and `.xlsb` (the vendored reader supports them); detect encrypted files with `check_encryption`
- [ ] CLI: per-subcommand `--help`, `--version`, distinct exit codes, strict argument parsing everywhere, flush after streamed tokens
- [ ] UI: Ctrl+S/O/F, undo, backend/model editing in Settings, a "Sources:" line on chat answers, notice when old chat messages are dropped

Deprioritised: macOS CI (decision 0004), the Tantivy swap, the HTTP MCP
transport, ODS formula import (needs upstream Gridcraft work), an MCP
write path and a model-driven agent loop (product decisions, not fixes).
