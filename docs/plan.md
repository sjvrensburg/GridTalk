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
- [ ] XLSX/ODS via Gridcraft `xlsx` crate (replaces the typed stub error)
- [ ] Incremental dirty-range re-snapshot on engine command history (500 ms debounce)
- [ ] egui grid shell on Gridcraft `ui-egui` with chat pane placeholder

## Phase 3: Graph-RAG & inference — IN PROGRESS

- [x] Region/header detection, lifted graph (workbook→sheet→region→column)
- [x] Lifted-edge invariant check re-derived from raw formulas
- [x] Lexical TF-IDF index with cited passages (tantivy swap keeps this API)
- [x] `ChatBackend` trait: streaming + cancellation
- [x] OpenAI-compatible external backend (verbatim send, http; https = typed error until TLS lands)
- [x] Deterministic echo backend for tests/demos
- [ ] Inference spike: candle → mistral.rs → llama-cpp FFI (see `docs/spikes/inference-engine.md`)
- [ ] Local embeddings (bge-small via fastembed/ONNX) + hybrid fusion
- [ ] Model download/management (`hf-hub`, `~/.cache/gridtalk/models`)
- [ ] Investigate tool-loop harness (Phase 3b, behind a toggle)

## Phase 4: Dual-view & polish — IN PROGRESS

- [x] `ViewState` two-way sync state machine (node→range, cell→node) with tests
- [x] Terminal dual-view rendering (`gridtalk view`)
- [ ] egui graph pane: force layout in worker thread, cached subgraphs
- [ ] Split / Graph-full / Grid-full modes persisted per workbook
- [ ] Settings dialog, first-run wizard, installers (Linux/macOS/Windows)

## Validation

`cargo test --workspace`, `cargo clippy --workspace --all-targets`,
`cargo fmt --check` on every push. Synthetic fixtures only — never
commit a real spreadsheet.
