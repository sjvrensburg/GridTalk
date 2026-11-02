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

Sessions are invalidated when the file's mtime changes, so edits
saved from the GridTalk UI or Gridcraft are picked up on the
next call.

## Without MCP: JSON CLI

```bash
gridtalk tools list
gridtalk tools call tests/fixtures/demo.csv '{"tool":"what_if","cell":"B2","value":10,"target":"Sheet1!D2"}'
```

Same typed core, stable exit codes (non-zero on `ok:false`).
