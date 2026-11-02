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
- [x] Workspace crates: `gridtalk-graph`, `gridtalk-index`, `gridtalk-infer`, `gridtalk-app`
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
- [x] OpenAI-compatible external backend (verbatim send, http; https = typed error until TLS lands)
- [x] Deterministic echo backend for tests/demos
- [x] Inference spike harness: `gridtalk spike` probes candle → mistral.rs → llama-cpp FFI (candle loads GGUF today; other rungs report why not — see `docs/spikes/inference-engine.md`)
- [x] Local embeddings + hybrid fusion: offline n-gram `EmbeddingModel` (bge-small ONNX remains an adapter swap behind the same trait), cosine vector index, Reciprocal Rank Fusion of lexical+vector rankings (`HybridIndex`, used by chat + investigate)
- [x] Model management: `~/.cache/gridtalk/models` (or `GRIDTALK_MODEL_CACHE`), GGUF metadata parser, `gridtalk models import|remove|list`, http download (https = typed error)
- [x] Investigate tool-loop harness (Phase 3b): `gridtalk investigate` — search → expand → cells → formulas → what_if tools with a recorded trail and citations; formula evaluator with what-if recomputation

## Phase 4: Dual-view & polish — IN PROGRESS

- [x] `ViewState` two-way sync state machine (node→range, cell→node) with tests
- [x] Terminal dual-view rendering (`gridtalk view`)
- [x] egui graph pane: force layout computed in a worker thread, cached per revision, click-to-select nodes
- [x] Split / Graph-full / Grid-full modes persisted per workbook (`~/.config/gridtalk/ui-state.json`)
- [x] Settings window (config path, verbatim-send notice)
- [x] First-run wizard (privacy posture → backend/model → ready; writes `config.toml`, persisted `first_run_done`)
- [x] Installers: tag-triggered release workflow (Linux/macOS/Windows archives attached to the GitHub Release) + `scripts/install.sh`; release workflow is exercised on the first `v*` tag

- [x] Real-model generation: candle (`candle-gen`) generates from a real llama-family GGUF — SmolLM2-135M Q4_K_M at 31.9 tok/s, 541 ms first token (CPU, release). See `docs/spikes/inference-engine.md`
- [x] Gridcraft ribbon reuse: companion-app composition (`gridtalk open-in-gridcraft`), decision 0003 — embedding rejected (forked egui + `SheetApp` ownership)

## Validation

`cargo test --workspace`, `cargo clippy --workspace --all-targets`,
`cargo fmt --check` on every push. Synthetic fixtures only — never
commit a real spreadsheet.
