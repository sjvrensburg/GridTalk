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
- [x] Local embeddings + hybrid fusion: offline n-gram `EmbeddingModel` (bge-small ONNX remains an adapter swap behind the same trait), cosine vector index, Reciprocal Rank Fusion of lexical+vector rankings (`HybridIndex`, used by UI chat, investigate and MCP; CLI `ask` still builds the lexical-only `Index`)
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
- [x] UI keeps the selection after an edit, and the graph layout when the edit leaves the graph's nodes and edges unchanged. Since Phase 1b each rebuild copies the snapshot once on the UI thread for its worker (the formula worker shares that copy)
- The incremental-rebuild item was split on 2026-11-10 after measuring (release build, one cell edited; test workbooks of 30k, 300k and 1.2M cells across 1–4 sheets). Before Phase 1b, each edit cost 29 ms, 0.58 s and 2.7 s, all on the UI thread, so a large workbook froze for seconds per edit. `check_invariants` was ~60% of `build_graph`, and the index update cost as much as a full index build

Phase 1b — edit latency (independent of the lineage rules):

- [x] Rebuild the graph and index on a worker thread after an edit (`RebuildJob`): the UI never blocks; chat and Save wait for a rebuild in flight; the rebuild's snapshot copy is shared with the formula worker
- [x] Cheaper graph build and invariant check: `NodeLocator` groups region/column nodes by sheet, so a cell lookup is an integer scan of one sheet's nodes instead of a case-folded name comparison against every node. `build_graph` per edit: 17→4.6 ms, 413→54 ms, 1.73→0.33 s
- [x] Incremental index from dirty ranges (`HybridIndex::update_dirty`): a region that kept its range and that no edit touched keeps its lexical document and vectors without being re-rendered. Index update per edit: 162→43 ms (300k cells), 0.96→0.29 s (1.2M). Total per edit, now off the UI thread: 18 ms, 97 ms, 0.63 s
- [ ] No full snapshot copy on the UI thread per edit: `IncrementalSnapshot` holds its snapshot (or each sheet) behind an `Arc` and edits through `Arc::make_mut`, so a rebuild or formula job shares the current revision instead of copying it (today ~33 ms per edit at 1.2M cells, ~6 ms at 300k)

Phase 2 — agent-facing correctness:

- [ ] `get_cells` returns the evaluated value plus the typed cached value; cell parsing rejects refs outside the sheet grid
- [ ] CLI `ask` uses `HybridIndex`; investigate's what-if discovery finds text dependents
- [ ] Graph lineage: 3D refs (`Jan:Mar!A2` is currently misread as columns JAN:MAR), defined names, checked range-end sheet qualifiers, running totals within a column
- [ ] Incremental graph rebuild from dirty ranges, after the lineage item (it depends on the final edge rules): re-detect regions and re-lift edges only for the sheets an edit touched, and check invariants for those sheets. The edited region is still re-indexed whole, which matters when one region spans most of a sheet
- [ ] `list_sheets`/`workbook_info` tool (sheets, defined names, load warnings); `readOnlyHint` annotations; A1 examples in tool descriptions
- [ ] One typed MCP error envelope; `expand` capped with truncation fields; search limit clamp disclosed; re-opening a path reuses its session
- [ ] Reader warnings reach the agent after a silent reload: when `fresh()` reloads a changed file, its new `warnings` are reported on the next tool response, not only by `open_workbook`

Phase 3 — hardening:

- [ ] MCP `--root <dir>` allowlist (canonicalised); tool calls off the stdio thread with a time budget; a byte budget next to `MAX_WORKBOOKS`
- [ ] Uncompressed-size pre-scan of XLSX packages before Gridcraft parses them
- [ ] `create_new` temp files; model import size cap; cleartext-`http` warning to remote hosts even without an API key (UI and CLI); early GGUF header check during download

Phase 4 — tests and CI:

- [ ] Subprocess MCP stdio test (handshake, open, error shapes, exit codes)
- [ ] Synthetic `.xlsx` fixtures generated in tests; tests for the 4M-cell, 32 MiB and 256 MiB caps and empty workbooks
- [ ] pytest step for `scripts/hermit`; MSRV `cargo test` if CI time allows

Phase 5 — features and UX:

- [ ] Accept `.xlsm`, `.xls` and `.xlsb` (the vendored reader supports them); detect encrypted files with `check_encryption`
- [ ] CLI: per-subcommand `--help`, `--version`, distinct exit codes, strict argument parsing everywhere, flush after streamed tokens
- [ ] UI: Ctrl+S/O/F, undo, backend/model editing in Settings, a "Sources:" line on chat answers, notice when old chat messages are dropped

Deprioritised: macOS CI (decision 0004), the Tantivy swap, the HTTP MCP
transport, ODS formula import (needs upstream Gridcraft work), an MCP
write path and a model-driven agent loop (product decisions, not fixes).
