//! MCP server (Phase B of the agentic-interfaces plan): GridTalk
//! tools over Model Context Protocol JSON-RPC on stdio, so Claude
//! Code / Codex / any MCP client can drive a workbook without the
//! UI. Implements the MCP subset GridTalk needs — `initialize`,
//! `ping`, `tools/list`, `tools/call` — directly on the spec's
//! JSON-RPC framing (one JSON object per line on stdio), with the
//! typed tool core from `tools.rs` doing the work.
//!
//! Privacy/safety: the server only opens paths the client passes
//! to `open_workbook`. Every investigation tool is read-only
//! (`what_if` simulates; nothing writes the file). There is no
//! write tool in the list at all.

use crate::tools::{ToolRequest, Tools};
use gridtalk_graph::{build_graph, load_path, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::SystemTime;

struct OpenWorkbook {
    path: PathBuf,
    /// (mtime, len) captured *before* reading, so a write during the
    /// load cannot stamp new metadata onto old content.
    stamp: Option<(SystemTime, u64)>,
    wb: WorkbookSnapshot,
    graph: gridtalk_graph::Graph,
    hybrid: HybridIndex,
}

impl OpenWorkbook {
    fn stamp_of(path: &PathBuf) -> Option<(SystemTime, u64)> {
        std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok().map(|t| (t, m.len())))
    }

    fn load(path: &PathBuf) -> Result<Self, String> {
        let stamp = Self::stamp_of(path);
        let wb = load_path(path).map_err(|e| e.to_string())?;
        let (graph, violations) = build_graph(&wb);
        if !violations.is_empty() {
            return Err(format!("graph invariants violated: {violations:?}"));
        }
        let hybrid = HybridIndex::build(&wb, &graph);
        Ok(OpenWorkbook {
            path: path.clone(),
            stamp,
            wb,
            graph,
            hybrid,
        })
    }

    fn fresh(&mut self) -> Result<(), String> {
        // mtime + length: a same-mtime replacement with a different
        // size is still picked up.
        if Self::stamp_of(&self.path) != self.stamp {
            *self = OpenWorkbook::load(&self.path.clone())?;
        }
        Ok(())
    }

    fn tools(&self) -> Tools<'_> {
        Tools {
            wb: &self.wb,
            graph: &self.graph,
            hybrid: &self.hybrid,
        }
    }
}

pub const MAX_WORKBOOKS: usize = 16;
const SUPPORTED_PROTOCOLS: [&str; 2] = ["2025-03-26", "2024-11-05"];

pub struct McpServer {
    workbooks: HashMap<String, OpenWorkbook>,
    order: std::collections::VecDeque<String>,
    next_id: u64,
    initialized: bool,
}

impl Default for McpServer {
    fn default() -> Self {
        McpServer {
            workbooks: HashMap::new(),
            order: std::collections::VecDeque::new(),
            next_id: 1,
            initialized: false,
        }
    }
}

fn req_str<'a>(args: &'a Value, field: &str) -> Result<&'a str, (i64, String)> {
    args.get(field)
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .ok_or((-32602, format!("missing or non-string argument: {field}")))
}

fn tool_definitions() -> Value {
    json!([
        {"name": "open_workbook", "description": "Open a spreadsheet (csv/tsv/xlsx/ods) and return its session id, sheets and regions", "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
        {"name": "close_workbook", "description": "Close an open workbook session", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}}, "required": ["workbook_id"]}},
        {"name": "search", "description": "Hybrid lexical+embedding search over workbook regions", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}, "query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["workbook_id", "query"]}},
        {"name": "expand", "description": "Graph neighbours of the region containing a range", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}, "range": {"type": "string"}}, "required": ["workbook_id", "range"]}},
        {"name": "get_cells", "description": "Typed cell values and formulas in a range", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}, "range": {"type": "string"}}, "required": ["workbook_id", "range"]}},
        {"name": "list_formulas", "description": "Formulas in a range with evaluated values", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}, "range": {"type": "string"}}, "required": ["workbook_id", "range"]}},
        {"name": "what_if", "description": "Simulate overriding one cell and re-evaluating a target formula (read-only, never writes)", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}, "cell": {"type": "string"}, "value": {"type": "number"}, "target": {"type": "string"}}, "required": ["workbook_id", "cell", "value", "target"]}},
        {"name": "graph_overview", "description": "All graph nodes and edges for an open workbook", "inputSchema": {"type": "object", "properties": {"workbook_id": {"type": "string"}}, "required": ["workbook_id"]}}
    ])
}

impl McpServer {
    /// Handle one JSON-RPC request object; returns the response
    /// object, or None for notifications.
    pub fn handle(&mut self, request: &Value) -> Option<Value> {
        // Non-object frames (batches, scalars) are Invalid Request,
        // never silently dropped.
        if !request.is_object() {
            return Some(
                json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "invalid request: expected a JSON-RPC object"}}),
            );
        }
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        // Notifications get no response per JSON-RPC.
        request.get("id")?;
        if let Some(v) = request.get("jsonrpc") {
            if v.as_str() != Some("2.0") {
                return Some(
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32600, "message": "invalid request: jsonrpc must be \"2.0\""}}),
                );
            }
        }
        let method = match request.get("method").and_then(|m| m.as_str()) {
            Some(m) if !m.is_empty() => m,
            _ => {
                return Some(
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32600, "message": "invalid request: missing method"}}),
                );
            }
        };
        if method == "notifications/initialized" {
            self.initialized = true;
            return None;
        }
        let result = match method {
            "initialize" => {
                self.initialized = true;
                let client_version = request
                    .pointer("/params/protocolVersion")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                // Negotiate: echo the client's version only when we
                // support it, otherwise offer our newest supported.
                let negotiated = if SUPPORTED_PROTOCOLS.contains(&client_version) {
                    client_version
                } else {
                    SUPPORTED_PROTOCOLS[0]
                };
                Ok(json!({
                    "protocolVersion": negotiated,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "gridtalk", "version": env!("CARGO_PKG_VERSION")}
                }))
            }
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tool_definitions()})),
            "tools/call" => {
                if !self.initialized {
                    Err((
                        -32002,
                        "server not initialized: send initialize first".to_string(),
                    ))
                } else {
                    self.call_tool(request.pointer("/params").cloned().unwrap_or(json!({})))
                }
            }
            other => Err((-32601, format!("method not found: {other}"))),
        };
        Some(match result {
            Ok(value) => json!({"jsonrpc": "2.0", "id": id, "result": value}),
            Err((code, message)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
            }
        })
    }

    fn call_tool(&mut self, params: Value) -> Result<Value, (i64, String)> {
        let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let text_result = |text: String, is_error: bool| {
            Ok(json!({
                "content": [{"type": "text", "text": text}],
                "isError": is_error
            }))
        };
        match name {
            "open_workbook" => {
                let path = args
                    .get("path")
                    .and_then(|p| p.as_str())
                    .ok_or((-32602, "missing string argument: path".to_string()))?;
                match OpenWorkbook::load(&PathBuf::from(path)) {
                    Ok(open) => {
                        // Unpredictable session ids + a hard cap with
                        // oldest-first eviction.
                        let nanos = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_nanos())
                            .unwrap_or(0);
                        let id = format!(
                            "wb-{:x}-{}",
                            nanos ^ (std::process::id() as u128) << 32,
                            self.next_id
                        );
                        self.next_id += 1;
                        while self.workbooks.len() >= MAX_WORKBOOKS {
                            match self.order.pop_front() {
                                Some(old) => {
                                    self.workbooks.remove(&old);
                                }
                                None => break,
                            }
                        }
                        self.order.push_back(id.clone());
                        let sheets: Vec<Value> = open
                            .wb
                            .sheets
                            .iter()
                            .map(|s| json!({"name": s.name, "rows": s.cells.len(), "cols": s.max_col()}))
                            .collect();
                        let regions: Vec<Value> = open
                            .graph
                            .regions
                            .iter()
                            .map(|r| json!({"range": r.range.a1(), "headers": r.headers}))
                            .collect();
                        self.workbooks.insert(id.clone(), open);
                        text_result(
                            json!({"workbook_id": id, "sheets": sheets, "regions": regions})
                                .to_string(),
                            false,
                        )
                    }
                    Err(e) => text_result(format!("open failed: {e}"), true),
                }
            }
            "close_workbook" => {
                let id = req_str(&args, "workbook_id")?;
                if self.workbooks.remove(id).is_some() {
                    self.order.retain(|x| x != id);
                    text_result(format!("closed {id}"), false)
                } else {
                    Err((-32602, format!("unknown workbook_id {id}")))
                }
            }
            _ => {
                let id = args
                    .get("workbook_id")
                    .and_then(|p| p.as_str())
                    .ok_or((-32602, "missing string argument: workbook_id".to_string()))?;
                let open = self.workbooks.get_mut(id).ok_or((
                    -32602,
                    format!("unknown workbook_id {id}; call open_workbook first"),
                ))?;
                open.fresh().map_err(|e| (-32000, e))?;
                let tools = open.tools();
                let req = match name {
                    "search" => ToolRequest::Search {
                        query: req_str(&args, "query")?.into(),
                        limit: match args.get("limit") {
                            None => 3,
                            Some(v) => v
                                .as_u64()
                                .ok_or((-32602, "limit must be a positive integer".to_string()))?
                                as usize,
                        },
                    },
                    "expand" => ToolRequest::Expand {
                        range: req_str(&args, "range")?.into(),
                    },
                    "get_cells" => ToolRequest::Cells {
                        range: req_str(&args, "range")?.into(),
                    },
                    "list_formulas" => ToolRequest::Formulas {
                        range: req_str(&args, "range")?.into(),
                    },
                    "what_if" => ToolRequest::WhatIf {
                        cell: req_str(&args, "cell")?.into(),
                        value: args.get("value").and_then(|q| q.as_f64()).ok_or((
                            -32602,
                            "missing or non-numeric argument: value".to_string(),
                        ))?,
                        target: req_str(&args, "target")?.into(),
                    },
                    "graph_overview" => ToolRequest::GraphOverview,
                    other => {
                        return text_result(format!("unknown tool {other}"), true);
                    }
                };
                let result = tools.call(&req);
                // Machines get the structured payload as JSON text.
                text_result(result.data.to_string(), !result.ok)
            }
        }
    }
}

/// Serve MCP on stdio until EOF. Logs go to stderr; stdout is
/// reserved for JSON-RPC frames.
pub fn serve_stdio() -> Result<(), String> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    let mut server = McpServer::default();
    eprintln!(
        "gridtalk mcp: serving on stdio ({} tools)",
        tool_definitions().as_array().map(|a| a.len()).unwrap_or(0)
    );
    // Cap each frame at 1 MiB: an unbounded line would grow memory
    // until OOM before any parse error could be reported.
    const MAX_LINE: usize = 1024 * 1024;
    let mut reader = std::io::BufReader::new(stdin.lock());
    loop {
        let mut raw: Vec<u8> = Vec::new();
        loop {
            let mut chunk = Vec::new();
            let n = reader
                .read_until(b'\n', &mut chunk)
                .map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            let done = chunk.ends_with(b"\n");
            raw.extend_from_slice(&chunk);
            if raw.len() > MAX_LINE {
                let resp = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "request line exceeds the 1 MiB cap"}});
                writeln!(stdout, "{resp}").map_err(|e| e.to_string())?;
                stdout.flush().map_err(|e| e.to_string())?;
                raw.clear();
                if done {
                    break;
                }
                // Keep discarding until the newline.
                continue;
            }
            if done {
                break;
            }
        }
        if raw.is_empty() {
            break;
        }
        let line = String::from_utf8_lossy(&raw).into_owned();
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let resp = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}});
                writeln!(stdout, "{resp}").map_err(|e| e.to_string())?;
                stdout.flush().map_err(|e| e.to_string())?;
                continue;
            }
        };
        if let Some(response) = server.handle(&request) {
            writeln!(stdout, "{response}").map_err(|e| e.to_string())?;
            stdout.flush().map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rpc(server: &mut McpServer, value: Value) -> Value {
        server.handle(&value).expect("requests get responses")
    }

    #[test]
    fn protocol_and_tool_flow_over_jsonrpc() {
        let mut server = McpServer::default();
        let init = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}}),
        );
        assert_eq!(init["result"]["serverInfo"]["name"], "gridtalk");
        assert_eq!(init["result"]["protocolVersion"], "2024-11-05");
        // Notifications get no response.
        assert!(server
            .handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))
            .is_none());

        let list = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        );
        let tools = list["result"]["tools"].as_array().unwrap();
        assert!(tools.len() >= 8);
        assert!(tools.iter().any(|t| t["name"] == "what_if"));

        // Open the demo workbook through the protocol.
        let demo = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/demo.csv")
            .display()
            .to_string();
        let opened = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "open_workbook", "arguments": {"path": demo}}}),
        );
        let text = opened["result"]["content"][0]["text"].as_str().unwrap();
        let payload: Value = serde_json::from_str(text).unwrap();
        let wb_id = payload["workbook_id"].as_str().unwrap().to_string();
        assert!(!payload["regions"].as_array().unwrap().is_empty());

        let formulas = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "list_formulas", "arguments": {"workbook_id": wb_id, "range": "Sheet1!A1:D4"}}}),
        );
        let text = formulas["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("B2*C2"), "structured payload: {text}");

        let whatif = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "what_if", "arguments": {"workbook_id": wb_id, "cell": "B2", "value": 10, "target": "Sheet1!D2"}}}),
        );
        let text = whatif["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"after\":30.0"), "what-if payload: {text}");

        // Unknown workbook is a protocol-level invalid-params error.
        let bad = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "search", "arguments": {"workbook_id": "wb-999", "query": "x"}}}),
        );
        assert_eq!(bad["error"]["code"], -32602);
        // Missing required args are -32602, never silent defaults.
        let missing = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "search", "arguments": {"workbook_id": wb_id}}}),
        );
        assert_eq!(missing["error"]["code"], -32602);
        // Non-object frames get -32600, not silence.
        let batch = server
            .handle(&json!([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]))
            .unwrap();
        assert_eq!(batch["error"]["code"], -32600);
        // Unsupported protocol versions are negotiated down.
        let mut fresh = McpServer::default();
        let init = rpc(
            &mut fresh,
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
    }
}
