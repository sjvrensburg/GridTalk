# Driving GridTalk from an agent harness (MCP)

GridTalk ships an MCP server over stdio so Claude Code, Codex,
Cursor, or any MCP client can drive a workbook without the UI.

## Start it

```bash
gridtalk mcp [--root <dir>]... [--timeout <seconds>] [--memory-budget <MiB>]
```

Stdout is reserved for JSON-RPC frames; logs go to stderr. The
server only opens paths your harness passes to `open_workbook`.
All investigation tools are read-only (`what_if` simulates; it
never writes the file). There is no write tool.

- `--root <dir>` (repeatable) confines `open_workbook` to files under
  these directories. Paths are canonicalised first, so `..` and
  symlinks that lead out of a root are refused (`path_not_allowed`),
  and the check runs again whenever a session reloads its file. A
  missing file is reported as missing only when its directory is
  inside a root. Without `--root` any readable path may be opened,
  so pass one when the client is not fully trusted (a prompt-injected
  agent can ask to open anything).
- `--timeout <seconds>` (default 120) is how long one call may run.
  Calls run on worker threads, so `ping` and other calls are answered
  while one is busy; a call past the limit is answered with `timeout`.
  Gridcraft cannot be interrupted, so the timed-out call keeps running
  until it returns, and its result is dropped (a file reload it did
  is kept).
- `--memory-budget <MiB>` (default 4096) caps the estimated memory of
  the open sessions together; see Sessions. Workbooks load one at a
  time (other calls keep running), and a load is charged once it
  finishes, so the peak can pass the budget by one workbook being
  built. A load waiting its turn counts toward `--timeout`.

## Claude Code (`.mcp.json` in your project)

```json
{
  "mcpServers": {
    "gridtalk": {
      "command": "gridtalk",
      "args": ["mcp"]
    }
  }
}
```

## Codex (`~/.codex/config.toml`)

```toml
[mcp_servers.gridtalk]
command = "gridtalk"
args = ["mcp"]
```

## Tools

Every tool is listed with MCP annotations (`readOnlyHint: true` for
all but `close_workbook`; none is destructive) and a description
with an A1 example. Ranges carry their sheet: `Sheet1!A1:D4`,
`Sheet1!B2`, or `'Sheet Name'!A1:B2` (quote names with spaces or
punctuation, doubling any quote inside). Sheet names match
case-insensitively, and cells must lie inside the sheet grid
(`A1`..`XFD1048576`).

- `open_workbook`: returns `workbook_id`, `sheets`, `regions` and
  `warnings` (what the file reader skipped or could not import, e.g.
  the formulas of an `.xlsb` or ODS file, which are imported as their
  saved results: the first warning of such a file says what that means
  for the graph and the tools). Files
  are read by their content, so an `.xls` that is really an XLSX
  package opens, with a warning; an encrypted workbook is refused with
  `open_failed`. `.xls` files keep their formulas; `.xlsm`/`.xltm`
  macros are kept but never run.
  Opening a path that is already open, however it is spelled,
  returns that session (`reused: true`) instead of a second one.
- `close_workbook`.
- `workbook_info`: sheets with their sizes and `visibility`
  (`visible`, `hidden` or `veryHidden`, as the file records it —
  also on `open_workbook`'s sheets and on the sheet nodes of
  `graph_overview`), defined names, regions,
  load warnings and the tools' limits (`range_cells` 10,000,
  `search_limit` 25, `expand_edges` 200). Call it first.
- `search`: hybrid lexical + embedding, ranges with `vector_only`
  flags. A passage longer than 4,000 characters is cut to the header
  plus the rows matching the query, labelled with their sheet rows,
  and flagged `truncated`; a closing note counts rows shown, matching
  and total, and names a bounded range to read next. `limit` is 1-25:
  the response's `limit` is the one used, and `limit_clamped` says
  when a larger request was cut down.
- `expand`: graph neighbours of the region or column containing a
  range, at most 200 edges; `total_edges` and `truncated` tell when
  there are more. When no node contains the range (a column run
  across two sections of a table, or a table made of several
  regions), the nodes it overlaps are expanded one after the other:
  `contained: false`, `overlapping` (how many), and `nodes`, up to
  four, each shaped like a single expansion. Within one column those
  are column nodes, otherwise region nodes. A range that overlaps no
  node is still `no_graph_node`.
- `get_cells`: typed values; a formula cell carries `result`, what it
  computes, and `cached_result`, what the file cached, both typed,
  plus `error` when evaluation failed. Every cell carries its
  `address` (`Labour!D2`), and the result names the `sheet`, the
  `first_row` and the `columns`, so a cell can be cited without
  counting columns. At most 10,000 cells per call.
- `list_formulas`: formulas with evaluated values, each with its
  `cell` (`F2`) and its sheet-qualified `address` (`Labour!F2`).
  `columns` describes each column of the range that holds three or
  more formulas: how many share its commonest `pattern` (compared in
  R1C1, so a formula filled down is one pattern), an `example`, the
  `other_formulas` that differ, and `typed_values`, the cells
  holding a typed value where the column otherwise holds formulas
  (a formula that has been typed over).
- `what_if`: before and after values of a target when one cell is
  set to a number, evaluated on a copy.
- `filter`: the rows of a table whose columns meet every condition
  in `where` (`[{"column": "Region", "op": "=", "value": "EMEA"}]`;
  `op` is `=`, `!=`, `>`, `>=`, `<`, `<=` or `contains`). `range` is
  the table (a detected region) or any cell or column in it; a range
  of several rows checks only those rows. A column is named by its
  heading (ignoring case) or its letter. Numbers compare as numbers
  when the value reads as one (`5000`, `12.5%`), text compares
  ignoring case, and a number is never ordered against text; formula
  cells compare by what they compute, and one that fails matches
  nothing. `value` may be sent as a string or a number. `columns`
  limits the columns shown. Each row carries its `row` number, its
  `range` and its `cells`, typed like `get_cells` values with their
  `address` and `column` heading. `limit` is 1-200 (default 50);
  `matched` counts every matching row, `rows_checked` the non-empty
  rows, and `truncated` says when more matched than are shown.
- `aggregate`: figures over the rows of a table that meet `where`
  (the same conditions as `filter`; none keeps every row). With a
  `column`: `rows`, `numbers`, `text` (with `distinct_text`),
  `blanks`, `errors` (formulas that fail; never counted as zero),
  `sum`, `mean`, and `min`/`max` each with the `address` of its cell.
  With `group_by` too: the same per value of that column (`groups`,
  merged ignoring case and surrounding spaces, in the order they first
  appear; at most 100, `groups_total` counts all). `group_by` alone
  counts rows per group; neither profiles every column (`columns`).
  Columns are named by heading or letter, as in `filter`.
- `issues`: problems in the workbook's formulas, errors first. Each
  issue has a `kind`, a `severity` (`error` or `warning`), the
  `range` of its cells (or the defined `name`), a `detail`, a
  `message`, a `count` and up to 20 `cells` (`address`, `formula`,
  typed `result` or `error`, and the value the file `saved`). Errors:
  `formula_error` (evaluates to an error that starts there),
  `missing_sheet`, `deleted_reference` (`#REF!` in the formula) and
  `broken_name`. Warnings: `error_from_precedent` (passes on an error
  from a cell it reads), `saved_value_differs` (the file's saved
  value disagrees with Gridcraft's recalculation; volatile functions
  such as `NOW()` are skipped), `external_reference` (reads another
  workbook) and `constant_in_formula` (a number other than 0 or 1
  written into the formula's arithmetic, `=B5*1.05`). Cells of one
  column with the same problem and detail are one issue. An optional
  `range` checks one sheet's range and leaves defined names out. At
  most 40 issues are listed; `errors`, `warnings` and `total` count
  all of them, and `truncated` says when more exist.
- `graph_overview`: nodes and edges, capped at 500 / 2,000.

Formula results are typed like `get_cells` values:
`list_formulas` entries carry `result` and `what_if` carries
`before_result`/`after_result`, each `{"type": "number" | "text" |
"bool", "value": ...}` (`null` when evaluation failed; `error` then
says why). The older numeric fields (`value`, `before`, `after`)
stay for compatibility: booleans there are 1/0 and text results are
`null`, so read the typed fields to see text.

## Errors

Every failure names its kind, so a client branches on `kind`
rather than on message text.

- Protocol errors are JSON-RPC errors: `-32600` for a malformed
  frame, `-32601` for an unknown method, `-32002` before
  `initialize`, and `-32602` for a `tools/call` with an unknown tool
  or arguments that break its schema. A `tools/call` error carries
  `data.kind`: `unknown_tool` or `invalid_arguments`.
- Failures while running a tool are results with `isError: true`
  and `structuredContent` `{"error": {"kind": ..., "message": ...}}`.
  Kinds: `unknown_workbook`, `workbook_unavailable` (the file was
  deleted or became unreadable), `open_failed`, `invalid_range`
  (unparseable, over the cell limit, or outside the grid),
  `unknown_sheet` (the message names the closest sheet when one is
  near, "did you mean …?", and lists the sheets), `unknown_column`
  (`filter`; the same, for the table's columns), `no_graph_node`,
  `evaluation_failed`,
  `invalid_arguments` (e.g. a non-finite what-if value),
  `path_not_allowed` (outside every `--root`), `over_budget` (one
  workbook's estimate is over `--memory-budget` on its own; when an
  open session's file grows past it, that session is closed),
  `timeout`, `busy` (four calls are already running, timed-out ones
  included; retry later) and `internal` (a GridTalk bug).
  `gridtalk tools call` prints the same payload.

## Sessions

A session reloads when its file changes (stat first, then a content
hash), so edits saved from the GridTalk UI or Gridcraft are picked
up on the next call. That call's response then carries
`reloaded: {"warnings": [...]}`, the new file's reader warnings, so
a reload is never silent. Sessions are capped at 16 open workbooks
and at the memory budget together; the oldest are evicted to make
room, and `open_workbook` lists them in `evicted_workbook_ids`. A
session's memory is estimated as 16 times its snapshot's cells and
text: on generated workbooks of 5k to 1.2M cells, the session (the
Gridcraft workbook, graph, index and recalculated engine) held 12 to
25 times that, so the estimate is a middle, not a bound. Only
regular files are read, and never more than the ingestion cap (32
MiB for CSV/TSV, 256 MiB for the other formats); a zip package whose
zip directory declares more than 512 MiB unpacked is refused before
it is unpacked.

## Troubleshooting

- **`gridtalk` not found / MCP server fails to start:** the
  shipped `.mcp.json` assumes a `gridtalk` binary on your `PATH`.
  Install one first (`scripts/install.sh` from a checkout, or a
  release archive), or point `command` at the binary's absolute
  path, e.g. `/home/you/.local/bin/gridtalk`.
- **Protocol errors:** the server requires `initialize` before
  `tools/call`, negotiates `protocolVersion` down to a supported
  version (2025-03-26 / 2024-11-05), and answers malformed frames
  with JSON-RPC errors rather than dropping them (see Errors).

## Without MCP: JSON CLI

```bash
gridtalk tools list
gridtalk tools call tests/fixtures/demo.csv '{"tool":"what_if","cell":"B2","value":10,"target":"Sheet1!D2"}'
```

Same typed core. `tools call` prints the result once and exits 5 when it is `ok:false`; a bad request is a usage error (exit 2) and an unreadable workbook exit 3 (see `gridtalk help`).
