# Driving GridTalk from an agent harness (MCP)

GridTalk ships an MCP server over stdio so Claude Code, Codex,
Cursor, or any MCP client can drive a workbook without the UI.

## Start it

```bash
gridtalk mcp
```

Stdout is reserved for JSON-RPC frames; logs go to stderr. The
server only opens paths your harness passes to `open_workbook`.
All investigation tools are read-only (`what_if` simulates; it
never writes the file). There is no write tool.

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

`open_workbook` · `close_workbook` · `search` (hybrid lexical +
embedding, ranges with `vector_only` flags) · `expand` ·
`get_cells` (typed values) · `list_formulas` (with evaluated
values) · `what_if` · `graph_overview`.

Formula results are typed like `get_cells` values:
`list_formulas` entries carry `result` and `what_if` carries
`before_result`/`after_result`, each `{"type": "number" | "text" |
"bool", "value": ...}` (`null` when evaluation failed; `error` then
says why). The older numeric fields (`value`, `before`, `after`)
stay for compatibility: booleans there are 1/0 and text results are
`null`, so read the typed fields to see text.

Sessions are invalidated when the file's (mtime, length) stamp
changes, so edits saved from the GridTalk UI or Gridcraft are
picked up on the next call. Sessions are capped at 16 open
workbooks (oldest evicted).

## Troubleshooting

- **`gridtalk` not found / MCP server fails to start:** the
  shipped `.mcp.json` assumes a `gridtalk` binary on your `PATH`.
  Install one first (`scripts/install.sh` from a checkout, or a
  release archive), or point `command` at the binary's absolute
  path, e.g. `/home/you/.local/bin/gridtalk`.
- **Protocol errors:** the server requires `initialize` before
  `tools/call`, negotiates `protocolVersion` down to a supported
  version (2025-03-26 / 2024-11-05), and answers malformed frames
  with JSON-RPC `-32600`/`-32602` rather than dropping them.

## Without MCP: JSON CLI

```bash
gridtalk tools list
gridtalk tools call tests/fixtures/demo.csv '{"tool":"what_if","cell":"B2","value":10,"target":"Sheet1!D2"}'
```

Same typed core, stable exit codes (non-zero on `ok:false`).
