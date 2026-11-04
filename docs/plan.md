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
- [x] XLSX/ODS via Gridcraft `xlsx` crate (`load_gridcraft_bytes`, XLSX round-trip + ODS fixture tests)
- [x] Incremental dirty-range re-snapshot (`IncrementalSnapshot`: 500 ms debounce, dirty-range list, revision counter)
- [x] egui grid+chat shell (`gridtalk-ui`: editable grid, chat pane, settings window) — on egui/eframe directly; Gridcraft `ui-egui` ribbon reuse remains a later swap

## Phase 3: Graph-RAG & inference — IN PROGRESS

- [x] Region/header detection, lifted graph (workbook→sheet→region→column)
- [x] Lifted-edge invariant check re-derived from raw formulas
- [x] Lexical TF-IDF index with cited passages (tantivy swap keeps this API)
- [x] `ChatBackend` trait: streaming + cancellation
- [x] OpenAI-compatible external backend (verbatim send over http or certificate-verified https via rustls; extra CAs from `GRIDTALK_CA_FILE`)
- [x] Deterministic echo backend for tests/demos
- [x] Inference: single llama.cpp-over-Vulkan engine (decision 0004); `gridtalk spike <model.gguf>` validates a GGUF header (header-only, not a load test)
- [x] Local embeddings + hybrid fusion: offline n-gram `EmbeddingModel` (bge-small ONNX remains an adapter swap behind the same trait), cosine vector index, Reciprocal Rank Fusion of lexical+vector rankings (`HybridIndex`, used by chat + investigate)
- [x] Model management: `~/.cache/gridtalk/models` (or `GRIDTALK_MODEL_CACHE`), GGUF metadata parser, `gridtalk models import|download|remove|list`, http/https download following up to 5 redirects (absolute or relative; never https→http), chunked or Content-Length bodies with truncation detected either way
- [x] Investigate tool-loop harness (Phase 3b): `gridtalk investigate` — search → expand → get_cells → list_formulas → what_if tools with a recorded trail and citations; formula evaluation and what-if recomputation through Gridcraft's calc engine (`gridcraft-calc`: its full function library incl. VLOOKUP/INDEX/MATCH/PV/SUMPRODUCT, defined names, 3D references, Excel rounding), so a what-if change flows through lookup tables instead of stopping at cached values; unparseable formulas are typed parse errors. GridTalk's small built-in evaluator remains only for the single-sheet helpers and builds without the `gridcraft` feature
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
