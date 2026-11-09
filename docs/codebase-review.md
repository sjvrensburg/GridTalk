# GridTalk codebase review

Date: 2026-11-09
Scope: read-only review of the entire codebase. No files were changed during the review, and no tests were run.
Method: 8 parallel reviewers plus an adversarial critic and a synthesis agent. Findings are code-inspected, not runtime-reproduced.

## Verification (2026-11-09)

Every claim was then checked against the code at `dc978b5`. The high-severity claims that can be run were probed with a scratch crate built against `gridtalk-graph` (release build, synthetic data). Inline notes marked **Verified:** record corrections. Claims without a note were confirmed as written.

- **Wrong as written:** the bad-precedent example (`=NOSUCHFN(1)` correctly yields `#NAME?`; the bug is real only for formulas Gridcraft cannot parse), the whole-column graph-build hang (5,000 × `=SUM(A:A)` builds in 22 ms), the macOS CI recommendation (macOS is intentionally unsupported, decision 0004), ping before initialize (the MCP spec allows it) and the help text for `view`/`spike` (both are listed).
- **Confirmed by running:** ODS formulas load as constants; one single-cell eval of the 5,000-row `SUM(A:A)` sheet takes about 0.86 s because each call rebuilds and fully recalculates.
- **Overstated:** CSV parser panic (unreachable from shipped code), merged cells (values survive, only the merge structure is lost), `get_cells` (only formula cells come back as untyped strings, with the formula text alongside), unreachable graph results (reachable via `expand`/`search`), the incremental-index claim (the vector side really is incremental), unflushed streaming (output is line-buffered, not held to the end), malformed initialize (still initializes), header inference (a blank second row splits the region, so the cause differs).
- **Worse than stated:** 3D refs are not just missing edges. `Jan:Mar!A2` is parsed as the whole-column range JAN:MAR on the current sheet, so edges are invented.
- **Missed by the review:** `take_dirty_ranges()` is never called outside tests, so the dirty list grows for the whole session, and every debounced edit clears the UI selection and graph layout.
- **Test count:** 107 `#[test]` functions (adapter 7, app 15, graph 32, index 17, infer 21, ui 15). The 132 figure was not reproduced.

The prioritised plan is in `docs/plan.md` under "Codebase review follow-ups".

## Top finding

The highest risks are silent data loss and wrong numbers, not crashes. Edits past row 10,000 are silently dropped while the UI reports clean, XLSX save destroys formatting and charts, `get_cells` returns stale cached values, and an unparseable precedent formula can silently contribute its cached value to a calculation.

## Bugs

### High severity

- **Silent edit loss** — `crates/gridtalk-graph/src/lib.rs:1531`, `IncrementalSnapshot::flush`: drops edits with row > 10,000 or col > 1,000 with no revision bump. The UI then reports clean and `dropped_edits()` is never read. An edit at row 20,000 appears to work and is never saved. **Verified:** reachable, since the UI grid shows every row (`display_rows = sheet.cells.len() + 3`) and `dirty()` compares only revision and pending count.
- **Wrong totals from bad formulas** — `crates/gridtalk-graph/src/calc.rs:46`, `Engine::value`: the parse guard checks only the directly queried cell. A precedent like `=NOSUCHFN(1)` cached as 42 silently feeds 42 into a `SUM` instead of raising a parse or `#NAME?` error. **Verified:** the example is wrong, because an unknown function parses and yields `#NAME?`. A formula Gridcraft cannot parse (`=1+(` cached as 42) does leak: `=A2*1` returns 42.
- **Unbounded file read before size caps** — `crates/gridtalk-app/src/mcp.rs:75` and `crates/gridtalk-graph/src/lib.rs:355`: the whole file is read before the 32 MiB / 256 MiB caps apply. A FIFO, `/dev/zero`, or oversized path can hang or OOM the MCP server. `hash_of` at `crates/gridtalk-app/src/mcp.rs:69` repeats the unbounded read.
- **Whole-column formula hang** — `crates/gridtalk-graph/src/lib.rs:720`, `push_range_refs`: each range expands to up to 100,000 refs, and graph build plus invariant checks repeat this three times with linear scans. A workbook full of `=SUM(A:A)` can effectively hang on open. No timeout. **Verified:** not reproduced. 5,000 × `=SUM(A:A)` builds the graph in 22 ms. Evaluating that sheet is the slow part (see the next two items).
- **ODS formulas treated as constants** — `vendor/gridcraft/crates/xlsx/src/ods.rs:47`, `read_ods`: ODS formulas import as cached values only. What-if and eval on `.ods` files silently treat formula cells as static numbers. **Verified by running:** a LibreOffice file with `=[.A2]*10` loads as `Number(20.0)`. This is Gridcraft's documented data-only ODS import, and its warning is discarded with `_report`.
- **Every evaluation is a full clone and recalc** — `crates/gridtalk-graph/src/calc.rs:27`, `Engine::new`: single-cell eval, `bench` (`crates/gridtalk-app/src/bench.rs:89`) and `investigate` (up to 200 trial what-ifs at `crates/gridtalk-app/src/investigate.rs:163`) each rebuild the whole workbook engine. Cost is formulas × attempts × workbook size. **Verified:** one eval of a 5,000-row sheet of `=SUM(A:A)` takes about 0.86 s.
- **One-cell edit rebuilds everything** — `crates/gridtalk-ui/src/main.rs:1059` and `crates/gridtalk-ui/src/main.rs:653`: each debounced edit clones the entire snapshot and runs a full graph and index rebuild. Dirty ranges at `crates/gridtalk-graph/src/lib.rs:1593` are never passed to the rebuild. **Verified:** they are never taken outside tests either, so the list grows for the session, and each rebuild clears the selection and graph layout.

### Medium severity

- **`ask` uses weaker search** — `crates/gridtalk-app/src/main.rs:105`: CLI `ask` builds a lexical-only index while `investigate` and the UI use `HybridIndex`. Same question, different retrieval quality by entry point. Contradicts `docs/plan.md:41`.
- **`get_cells` never evaluates** — `crates/gridtalk-app/src/tools.rs:548`, `cell_json`: returns the stale cached value as an untyped display string. Only `list_formulas` (`crates/gridtalk-app/src/tools.rs:291`) evaluates. Agents reading a stale XLSX get wrong numbers. **Verified:** overstated. Constant cells are typed; only a formula cell's cached value is a display string, and its formula text is returned alongside.
- **Inconsistent MCP error shapes** — `crates/gridtalk-app/src/mcp.rs:376`: failed open returns `isError` text, unknown `workbook_id` returns JSON-RPC `-32602`, a deleted workbook returns `-32000`, and tool errors collapse to an untyped `{"error": string}`. Clients cannot distinguish error kinds without string matching.
- **Unbounded / unreachable graph results** — `crates/gridtalk-app/src/tools.rs:221`, `Expand`: returns unbounded edges with no truncation fields. `GraphOverview` at `crates/gridtalk-app/src/tools.rs:393` truncates at 500 nodes / 2,000 edges with no cursor, so capped nodes are unreachable. **Verified:** overstated. It discloses `total_nodes`/`total_edges`/`truncated`, and capped nodes are reachable through `expand` and `search`.
- **Public CSV parser panics** — `crates/gridtalk-graph/src/lib.rs:539`, `parse_delimited`: panics via `.expect` on malformed CSV while `parse_delimited_strict` returns a typed `IngestError`. **Verified:** only tests call it; shipped loading uses the strict version.
- **No date or error cell type** — `crates/gridtalk-graph/src/lib.rs:18`, `CellValue`: dates render as serial numbers such as `45292`, and `#DIV/0!` is stored as literal text at `crates/gridtalk-graph/src/lib.rs:440` and indexed as text.
- **Graph lineage gaps** — `crates/gridtalk-graph/src/lib.rs:887`: 3D refs (`Jan:Mar!A2`), defined names (`crates/gridtalk-graph/src/lib.rs:1180`), and intra-column running totals (`D3=D2+B3`, suppressed at `crates/gridtalk-graph/src/lib.rs:1184`) produce missing or invented `depends_on` edges. **Verified:** `Jan:Mar!A2` is parsed as the whole-column range JAN:MAR on the current sheet plus an unqualified `A2`, so its edges are invented. The range-end sheet qualifier is stripped unchecked at `crates/gridtalk-graph/src/lib.rs:956`, so `Sheet2!A1:Sheet1!B2` becomes a `Sheet2!A1:B2` edge for a formula eval rejects.
- **Overlapping regions** — `crates/gridtalk-graph/src/lib.rs:1092`, `detect_regions`: the flood-filled component is reported as its full bounding rectangle, so regions overlap, passages duplicate cells, and membership is arbitrated by smallest area rather than true membership.
- **Sparse sheet rejected** — `crates/gridtalk-graph/src/lib.rs:402`, `MAX_SNAPSHOT_CELLS`: a sparse sheet with one far cell (e.g. `XFD100000`) is rejected because the densified bounding box exceeds 4M cells.
- **What-if discovery skips text dependents** — `crates/gridtalk-app/src/investigate.rs:178`: auto what-if uses numeric-only eval with `if let Ok`, so a text dependent such as `=A2&" x"&B2` is skipped and investigate reports "no formula depends on the override" incorrectly.
- **No grid-bounds check in tools** — `crates/gridtalk-app/src/tools.rs:598`, `parse_cell_on_sheet`: `A1048577` and `ZZZ1` parse successfully, producing phantom empty cells and misleading later errors instead of a parse rejection.
- **Zip-bomb window** — `vendor/gridcraft/crates/xlsx/src/package.rs:11`: decompressed limits of 512 MiB per part / 1 GiB total sit behind GridTalk's 256 MiB compressed cap, and parsing completes before the 4M-cell snapshot cap runs. A crafted XLSX/ODS expands to roughly 1 GiB in RAM on open.
- **Index cost on large regions** — `crates/gridtalk-index/src/lib.rs:242`: the whole passage is materialised and tokenised before the 4,000-char cap, search linear-scans all docs and builds an excerpt for every hit before truncating, and `HybridIndex::update` at `crates/gridtalk-index/src/lib.rs:415` fully rebuilds the lexical index despite the "incremental" claim. **Verified:** its doc comment says the lexical side is rebuilt, and the vector side really is incremental (per-line cache).

### Low severity

- **CSV row shift** — `crates/gridtalk-graph/src/lib.rs:555`: leading blank CSV rows are skipped, so a value on file row 3 becomes sheet row 1, breaking A1 citations.
- **Header inference is shallow** — `crates/gridtalk-graph/src/lib.rs:1073`: only row `r0+1` is inspected, so a table with a blank second row gets column-letter labels instead of real headers. **Verified:** a fully blank second row actually splits the flood fill, giving the same result for a different reason. The shallow check still bites when row 2 is text-only, such as a units row.
- **Sampled range refs** — `crates/gridtalk-graph/src/lib.rs:728`: ranges over 100k cells emit only an 8×8 interior sample plus the high corner, so precedents between samples get no edge, and the invariant check mirrors the same sampling at `crates/gridtalk-graph/src/lib.rs:1363` and cannot detect the miss.
- **Validation gaps** — `crates/gridtalk-app/src/mcp.rs:329` (empty path misreported), `crates/gridtalk-app/src/mcp.rs:402` (search limit silently clamped to 25), `crates/gridtalk-app/src/mcp.rs:243` (ping answered before initialize; a malformed initialize consumes the one-shot init). **Verified:** the MCP spec allows ping at any time, and an initialize without params still initializes.
- **Predictable temp files** — `crates/gridtalk-adapter/src/lib.rs:363`, `write_atomic`: temp name is predictable and `File::create` follows a pre-planted symlink. Same pattern at `crates/gridtalk-ui/src/main.rs:136`.
- **Unbounded bench tasks file** — `crates/gridtalk-app/src/main.rs:495`: `--tasks` is read with unbounded `read_to_string` and parsed with no size or task-count cap.
- **Sheet-name matching mismatch** — `crates/gridtalk-ui/src/main.rs:1157`: UI sheet following uses `eq_ignore_ascii_case` while the rest of the app uses Unicode `sheet_name_eq`, so `Café` vs `CAFÉ` fails to switch sheets.
- **Silent vector sampling** — `crates/gridtalk-index/src/embed.rs:236`: regions over 64 lines are stride-sampled with no signal in search output, so vector-only hits can silently miss unsampled lines.

### Security-adjacent

- **Arbitrary host path open** — `crates/gridtalk-app/src/mcp.rs:328`, `open_workbook`: any MCP client (or a prompt-injected caller) can open any readable spreadsheet on the host and exfiltrate cells via `get_cells`/`search`. No allowlist, canonicalize, or root restriction. Severity depends on whether the MCP client is trusted.
- **Unbounded model import** — `crates/gridtalk-infer/src/models.rs:468`, `ModelManager::import`: copies the source with no size cap, unlike the 64 GiB cap on downloads. A huge or growing source fills the cache disk.
- **Download SSRF / disk exhaustion** — `crates/gridtalk-infer/src/models.rs:522`, `download_http`: 64 GiB cap over a 2-hour deadline, redirects followed to any host (only https→http downgrade refused). GGUF validation runs only after download.
- **Cleartext external backend** — `crates/gridtalk-infer/src/lib.rs:435`, `OpenAiCompatibleBackend::stream`: cleartext `http` is refused only when an API key is present. Without a key, spreadsheet passages are POSTed unencrypted to any remote `http` base URL. The CLI warns (`crates/gridtalk-app/src/main.rs:912`); the UI chat path at `crates/gridtalk-ui/src/main.rs:773` has no equivalent check.
- **No global memory budget** — `crates/gridtalk-app/src/mcp.rs:135`, `MAX_WORKBOOKS`: up to 16 workbooks retained at once, each allowed 256 MiB compressed / ~1 GiB decompressed / 4M cells.
- **No eval timeout** — `crates/gridtalk-graph/src/calc.rs:27`: every eval constructs an engine and recalcs with no time or step budget, invoked synchronously on the single-threaded stdio loop at `crates/gridtalk-app/src/mcp.rs:433`. A pathological formula graph blocks all clients indefinitely.

Positives verified and not counted as findings: `what_if` is genuinely read-only (fresh engine from a cloned snapshot, no filesystem write on that path); no `unsafe` in `crates/`; MCP stdio frame cap of 1 MiB; `get_cells`/`list_formulas` 10k-cell cap; GGUF traversal rejection; adapter name validation.

## Missing features

### High impact

- **No write path** — `crates/gridtalk-app/src/mcp.rs:167`, `tool_definitions`: only 8 read-only MCP tools. `mcp --allow-write` at `crates/gridtalk-app/src/main.rs:704` is an accepted no-op. The XLSX writer exists at `crates/gridtalk-graph/src/lib.rs:533` (`to_xlsx_bytes`) but no tool exposes it. Agents can analyse but never apply or export a fix.
- **Save destroys workbook content** — `crates/gridtalk-graph/src/lib.rs:404`, `from_gridcraft_workbook`: the snapshot keeps only values, formulas and defined names. Merges (`vendor/gridcraft/crates/model/src/sheet.rs:113`), charts, images, tables, conditional formats and styles are dropped, and `to_gridcraft_workbook` at `crates/gridtalk-graph/src/lib.rs:478` rebuilds only what was kept. UI Save to `.xlsx` (`crates/gridtalk-ui/src/main.rs:916`) silently destroys them.
- **No semantic search** — `crates/gridtalk-index/src/embed.rs:5`, `NGramEmbedding`: the shipped ranker is character-trigram hashing; the bge-small ONNX adapter is pending at `crates/gridtalk-index/src/embed.rs:141`. No synonym or semantic matching despite the "hybrid lexical+embedding" framing in `README.md:24` and `docs/mcp.md:40`.
- **No agentic loop** — `crates/gridtalk-app/src/investigate.rs:28`, `investigate_with`: the planner is deterministic (always search → expand → get_cells → list_formulas, what-if only on keyword), never model-driven tool selection. The LLM tool-calling agent promised in `docs/plans/agentic-interfaces-and-optimisation.md:14` is not delivered.
- **No undo/redo** — `crates/gridtalk-graph/src/lib.rs:1460`, `IncrementalSnapshot`: no history stack; flush overwrites destructively. One mistyped cell means discarding all unsaved edits.
- **Legacy formats rejected** — `crates/gridtalk-graph/src/lib.rs:336`, `check_ingestible`: `.xls`, `.xlsb` and `.xlsm` return `UnsupportedFormat` although the vendored reader supports them (`vendor/gridcraft/crates/xlsx/src/lib.rs:54`). Legacy and macro workbooks must be re-saved before any use.
- **No integration or end-to-end tests** — `tests/fixtures/demo.csv:1`: `tests/` holds only `demo.csv` and `demo.ods`, with zero `tests/*.rs`. The closest MCP test at `crates/gridtalk-app/src/mcp.rs:545` calls the handler in-process. Binaries, stdio transport, argv dispatch and exit codes are never exercised as subprocesses. The 2,238-line UI binary at `crates/gridtalk-ui/src/main.rs:1040` has 9 tests.

### Medium impact

- **No sheet discovery tool** — `crates/gridtalk-app/src/mcp.rs:167`: no `list_sheets` or workbook-info tool. Sheet names are returned only once at open and lost after eviction. Defined names stored at `crates/gridtalk-graph/src/lib.rs:85` are never surfaced by any tool.
- **Single-cell numeric what-if only** — `crates/gridtalk-app/src/tools.rs:335`: the override is one cell with an `f64` value and one target re-evaluated. Text, bool, date, formula and multi-cell scenarios are impossible.
- **CSV import is narrow** — `crates/gridtalk-graph/src/lib.rs:370`: strict UTF-8 only, comma/tab only with no semicolon sniffing, always a single `Sheet1`. `infer_value` at `crates/gridtalk-graph/src/lib.rs:644` does no date, percent or currency inference, so `2024-01-01`, `12%` and `$1,200` stay as non-numeric text. UTF-16 and European CSVs fail or mis-parse.
- **Encrypted XLSX unsupported** — `crates/gridtalk-graph/src/lib.rs:387`: no encryption check or password path although the vendored library exports them. Such files fail with a generic parse error.
- **Unimplemented Excel functions** — `vendor/gridcraft/crates/functions/src/lib.rs:277`: function coverage is broad (400+), but `IMAGE`, `TRANSLATE`, `DETECTLANGUAGE`, `GETPIVOTDATA` and `RTD` have no implementation, and `WEBSERVICE`/`FILTERXML`/CUBE functions always return errors. Modern sheets using them get `#NAME?` or silent error values.
- **ODS is read-only on save** — `crates/gridtalk-ui/src/main.rs:931`: editing an advertised `.ods` forces a format conversion, and multi-sheet CSV save is rejected at `crates/gridtalk-ui/src/main.rs:920`.
- **Adapter tuning knobs missing** — `crates/gridtalk-adapter/src/lib.rs:32`: only `top_k`, `max_steps` and `ref_context` are honoured. The temperature, tool-order and few-shot knobs promised in `docs/plans/agentic-interfaces-and-optimisation.md:110` do not exist.
- **No redacting proxy** — `README.md:12` and `docs/decisions/0002-no-redaction.md:13`: redaction for the external backend is a deferred separate project. Only `base_url` passthrough exists at `crates/gridtalk-infer/src/lib.rs:133`.
- **No embedded ribbon** — `docs/decisions/0003-gridcraft-ribbon-reuse.md:19`: heavy editing launches an external Gridcraft binary (`crates/gridtalk-app/src/main.rs:655`) and re-snapshots. In-UI rich editing is unshipped.
- **Merged cells silently dropped** — `crates/gridtalk-graph/src/lib.rs:411`: conversion never reads merged ranges, with no test and no warning. Merged secondary cells and values vanish. **Verified:** overstated. Cell values, secondary cells included, are kept; only the merge structure is lost. Whether this also affects eval results was not probed.
- **No direct calc tests** — `crates/gridtalk-graph/src/calc.rs:21`: `calc.rs` has zero tests. Engine error variants are only indirectly touched via `eval.rs`.
- **CI gaps** — `.github/workflows/ci.yml:75`: no macOS job (**Verified:** not a gap, since macOS is intentionally unsupported per decision 0004), no coverage reporting, the MSRV job runs `cargo check` only, and there is no benchmark or perf-regression gate. The 4 Python mirror tests in `scripts/hermit/test_gridtalk_task.py:17` never run in CI, so Rust/Python drift is unguarded.
- **Empty and huge workbook edges untested** — `crates/gridtalk-graph/src/lib.rs:404`: no test builds an empty snapshot or hits the 4M-cell, 32 MiB or 256 MiB caps. Malformed-file coverage is thin: only invalid UTF-8 and unsupported `.xlsb` are tested, there is no `.xlsx` fixture, and XLSX is only self-round-tripped, so reader bugs symmetric with the writer are invisible.

### Low impact

- **Stdio-only MCP transport** — `crates/gridtalk-app/src/mcp.rs:444`: the streamable-HTTP loopback transport in the plan has no implementation, limiting remote and multi-client harness use.
- **Linear-scan lexical index** — `crates/gridtalk-index/src/lib.rs:6`: the Tantivy-backed index is a future swap. The shipped index has no posting-list scaling for large workbooks.
- **Minimal export** — `crates/gridtalk-graph/src/lib.rs:531`: XLSX only, plus single-sheet delimited export from the UI. No ODS, PDF, HTML or `.xls` export.
- **No `--version`** — `crates/gridtalk-app/src/main.rs:65`: neither binary supports `--version`/`-V`. Only MCP `initialize` returns the crate version.
- **Other gaps** — no Dependabot, CODEOWNERS, issue/PR templates or `SECURITY.md` in `.github/`; full `hermit train` has never been run (`docs/hermit.md:79`); the Gridcraft upstream rebase cadence has no automation.

## UX issues

### Highest impact

- **Echo backend looks like a real answer** — `crates/gridtalk-infer/src/lib.rs:90`, `EchoBackend`: with no model configured, the default backend replies with the question and workbook context parroted back as if it were an answer. The CLI prints only a late `--- echo-local ---` header at `crates/gridtalk-app/src/main.rs:132` and never warns that the answer is non-generative. The README admits this; `--help` does not. **Verified:** help does end with "(default: local echo backend)", though it does not say the reply is non-generative.

### Medium impact

- **No per-subcommand help** — `crates/gridtalk-app/src/main.rs:861`: only top-level `help` exists. `gridtalk ask --help` treats `--help` as a file path, and `gridtalk bench --help` falls through to a generic usage error. Flags such as `bench --split`/`--tasks`, plus the commands `spike` and `view` (**Verified:** wrong for `spike` and `view`, which help lists), are undiscoverable except by reading source or docs. Help has no examples, no exit-code contract, and incomplete env-var documentation.
- **Exit codes carry no meaning** — `crates/gridtalk-app/src/main.rs:57`: every failure exits 1 and every success exits 0. `ask` with zero hits prints "No matching region found" and exits 0 (`crates/gridtalk-app/src/main.rs:107`). Harnesses cannot distinguish empty results, usage errors and runtime failures.
- **Raw internal errors** — `crates/gridtalk-app/src/main.rs:103`: invariant failures surface as `graph invariants violated: {violations:?}` Rust Debug dumps with no actionable cause. Load errors leak raw `e.to_string()` via `crates/gridtalk-app/src/main.rs:947` with no remediation hint.
- **Inconsistent argument strictness** — `crates/gridtalk-app/src/main.rs:796`: `graph`, `view`, `bench` and `optimise` reject unknown flags with specific errors, while `models import`, `spike`, `open-in-gridcraft` and `models remove` silently ignore trailing arguments. A mistyped flag is a silent no-op there.
- **Undocumented tool surface** — `crates/gridtalk-app/src/mcp.rs:167`: MCP tool descriptions are one-liners with no A1 range example (e.g. `Sheet1!A1:D4`), no sheet-name quoting rule, and no error-shape documentation. Agents must guess syntax until `range_error` tells them only that it failed. No MCP annotations (`readOnlyHint`) or `outputSchema` either, so clients cannot tell `what_if` is safe or validate results.
- **Limits discovered by erroring** — `crates/gridtalk-app/src/tools.rs:122`: search `limit` is silently clamped to 1..=25 with no field saying so. The 10,000-cell cap at `crates/gridtalk-app/src/tools.rs:556` is disclosed only after failure. `graph_overview` shows the better pattern by disclosing its 500/2000 caps upfront at `crates/gridtalk-app/src/tools.rs:418`. Re-opening the same path also mints duplicate sessions, and with `MAX_WORKBOOKS=16` a held ID silently dies on eviction, learned only via an "unknown workbook_id" error on next use.
- **UI chat hides progress and citations** — `crates/gridtalk-ui/src/main.rs:781`: the worker buffers the full backend stream and shows only a spinner (watchdog allows 600 s at `crates/gridtalk-ui/src/main.rs:1073`). Citations are only prompted for, never enforced or appended, unlike `investigate` at `crates/gridtalk-app/src/investigate.rs:245` — so UI answers can silently violate the cited-answer promise.
- **Settings are nearly empty** — `crates/gridtalk-ui/src/main.rs:802`: Settings exposes only `n_gpu_layers` and `n_ctx`. Backend, model path, base URL, API key env, temperature and seed are editable only in the one-time wizard (`crates/gridtalk-ui/src/main.rs:1334`) or by hand-editing TOML. There is no UI path to switch model or endpoint after first run.
- **Mouse-only desktop app** — `crates/gridtalk-ui/src/main.rs:1198`: Open, Save, Save As and Settings are mouse-only buttons. No Ctrl+S / Ctrl+O / Ctrl+F shortcuts and no grid find function.
- **README gives no on-ramp** — `README.md:20`: the 35-line README has no installation, build, usage or example section. The only install paths are a Linux source build in `scripts/install.sh` (requiring cargo/CMake/glslc/C++) and an unlinked release zip mentioned only in that script's comment.
- **Dropped workbook parts are never reported** — `crates/gridtalk-graph/src/lib.rs:387`: the Gridcraft `ReadReport` is bound to `_report` and discarded, although the vendored reader reports skipped unsupported or broken parts as warnings. Users are never told parts of their workbook were dropped.
- **Unflushed streaming output** — `crates/gridtalk-app/src/main.rs:130`: streamed tokens use `print!` with no flush anywhere in `main.rs`, so piped output appears all at once and long local inference looks hung. **Verified:** Rust's stdout is line-buffered, so output arrives a line at a time. Only partial lines are delayed.

### Low impact

- **Contradictory binary behaviour** — `crates/gridtalk-ui/src/main.rs:2030`: the desktop binary has no `--help` and warns "ignoring unknown option" then launches anyway, while the CLI errors on unknown commands.
- **No progress on long runs** — `crates/gridtalk-app/src/main.rs:632`: `investigate` prints its tool trail only at the end, and `bench` prints only the final card. Only `models download` reports progress, coarsely every 256 MiB.
- **Naming and error-channel nits** — `crates/gridtalk-app/src/main.rs:309`: British `optimise` with no `optimize` alias; `spike` for a model-header probe; `tools call` requires single-quoted JSON documented only in the schemas payload. On `ok:false`, `tools call` prints pretty JSON to stdout then exits 1 at `crates/gridtalk-app/src/main.rs:763`, bypassing the stderr error convention.
- **`spike` overpromises** — `crates/gridtalk-infer/src/probe.rs:14`: header-only check that "does not load tensors or prove engine can run file" (`crates/gridtalk-infer/src/probe.rs:28`). Success can mislead users into thinking a model is validated.
- **Silent chat truncation** — `crates/gridtalk-ui/src/main.rs:591`: chat history is drained past 200 messages with no notice or export, so long investigations lose earlier cited answers.
- **Unbounded terminal view** — `crates/gridtalk-app/src/main.rs:182`: `view` prints every row of the first sheet with no bound or paging. A 32 MiB CSV permits roughly 1M rows of terminal flooding.
- **Unproven installers and stale docs** — `docs/plan.md:54`: the tag-triggered release workflow is admittedly unexercised and Windows Vulkan is verified only by inspection. The module doc at `crates/gridtalk-infer/src/lib.rs:1` still says "three local engine rungs" after decision 0004 removed two of them.

## Performance summary

The dominant pattern is repeated whole-workbook work with no caching or incremental path:

- Every evaluation clones the workbook and recalcs in full (`crates/gridtalk-graph/src/calc.rs:27`); `bench` and `investigate` multiply this by up to 64–200 attempts per formula.
- Graph build pays 3× formula parsing and range expansion via invariant re-derivation (`crates/gridtalk-graph/src/lib.rs:1197`), with up to 100k refs per range formula (`crates/gridtalk-graph/src/lib.rs:720`) and linear node lookup per referenced cell (`crates/gridtalk-graph/src/lib.rs:242`).
- Search is a linear scan over all docs with excerpts built for every hit before truncation (`crates/gridtalk-index/src/lib.rs:270`); vector search is brute-force cosine with norms recomputed per pair (`crates/gridtalk-index/src/embed.rs:336`).
- Graph layout repulsion is O(n²) per iteration at up to 140 iterations — roughly 17.5M pair steps at the 500-node cap (`crates/gridtalk-ui/src/layout.rs:78`) — and nodes beyond the 500 stride sample receive no repulsion at all.
- MCP freshness checks re-read and hash the entire file on any stat change, synchronously on the single-threaded stdio loop (`crates/gridtalk-app/src/mcp.rs:99`), blocking every other workbook.

No benchmarks were run; these are asymptotic and allocational claims from inspected code, not measured timings.

## Coverage limits of this review

- The adversarial critic received no prior claims to falsify (an orchestration gap in the run), so it contributed first-hand findings instead of verification. Treat individual findings as inspected-but-unverified rather than confirmed.
- Vendored Gridcraft internals (calc engine, formula parser, cycle and coercion semantics) were only spot-checked. Gridcraft-internal wrong-result bugs are unreviewed, and whether Gridcraft's `recalc_all` itself errors on unparseable precedents could not be verified without reading vendored code or running a probe.
- No `cargo test`, clippy or pytest run was performed by any reviewer (read-only scope), so the suite's pass/fail state is unknown. Test counts also differ between reviewers (132 vs 107 Rust `#[test]` functions). **Verified:** 107.
- GUI and terminal behaviour, Windows Vulkan, release archives, a live MCP client handshake, and the reachability of the >10,000-row dropped-edit path were source-inspected only, never run.
- Whether the arbitrary-path MCP finding is high severity depends on an untrusted or prompt-injected deployment, which the code does not answer. Performance and exhaustion claims are code-inspection asymptotics, not reproduced with crafted files.
- `llama-cpp-2` FFI and C++ model parsing inside the dependency were out of scope.
