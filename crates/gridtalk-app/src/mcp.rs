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

use crate::tools::{error_payload, ErrorKind, ToolRequest, Tools};
use gridtalk_graph::eval::EvalCache;
use gridtalk_graph::{build_graph, load_bytes, read_workbook_bytes, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::SystemTime;

struct OpenWorkbook {
    path: PathBuf,
    /// (mtime, len, content hash): stat first on every call;
    /// the file is only re-read/hashed when stat changed, and
    /// only reloaded when the hash also changed.
    stamp: Option<(SystemTime, u64, i64, u64)>,
    wb: WorkbookSnapshot,
    graph: gridtalk_graph::Graph,
    hybrid: HybridIndex,
    /// Recalculated once per loaded revision of the file: a reload
    /// replaces the whole session, cache included.
    calc: EvalCache,
}

impl OpenWorkbook {
    fn stat_of(path: &PathBuf) -> Option<(SystemTime, u64, i64)> {
        std::fs::metadata(path).ok().and_then(|m| {
            // A change stamp catches same-mtime/same-length
            // replacements the pair would alias: ctime+nsec on
            // Unix, the creation time on Windows (`file_index` is
            // still unstable there).
            #[cfg(unix)]
            let ctime = {
                use std::os::unix::fs::MetadataExt as _;
                m.ctime()
                    .wrapping_mul(1_000_000_000)
                    .wrapping_add(m.ctime_nsec())
            };
            #[cfg(windows)]
            let ctime = {
                use std::os::windows::fs::MetadataExt as _;
                m.creation_time() as i64
            };
            #[cfg(not(any(unix, windows)))]
            let ctime = 0i64;
            m.modified().ok().map(|t| (t, m.len(), ctime))
        })
    }

    /// FNV-1a over the content: cheap and deterministic.
    fn hash_bytes(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    fn hash_of(path: &std::path::Path) -> Option<u64> {
        read_workbook_bytes(path)
            .ok()
            .map(|bytes| Self::hash_bytes(&bytes))
    }

    fn load(path: &PathBuf) -> Result<Self, String> {
        // Stat before reading, so a write landing between the two
        // leaves a stale stamp (forcing a reload), never a fresh
        // stamp on stale content. The file is read once: the same
        // bytes are hashed and parsed.
        let stat = Self::stat_of(path);
        let bytes = read_workbook_bytes(path).map_err(|e| e.to_string())?;
        let stamp = stat.map(|(mtime, len, ctime)| (mtime, len, ctime, Self::hash_bytes(&bytes)));
        let wb = load_bytes(path, &bytes).map_err(|e| e.to_string())?;
        drop(bytes);
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
            calc: EvalCache::new(),
        })
    }

    /// Reload the workbook if its file changed. Returns whether it
    /// reloaded, so the caller can report the new file's warnings.
    fn fresh(&mut self) -> Result<bool, String> {
        // Stat first (cheap): unchanged stat means no re-read at
        // all. Changed stat re-hashes; only a changed hash (real
        // content change) triggers the expensive reload.
        let Some((mtime, len, ctime)) = Self::stat_of(&self.path) else {
            return Err(format!(
                "workbook {} was deleted or became unreadable",
                self.path.display()
            ));
        };
        if let Some((old_mtime, old_len, old_ctime, old_hash)) = self.stamp {
            // Windows has no change counter in std metadata, so
            // (mtime, len, file_index) cannot detect an in-place
            // same-length edit: always fall through to the hash.
            if !cfg!(windows) && (mtime, len, ctime) == (old_mtime, old_len, old_ctime) {
                return Ok(false);
            }
            if Self::hash_of(&self.path) == Some(old_hash) {
                // Content-preserving touch: refresh the stamp.
                self.stamp = Some((mtime, len, ctime, old_hash));
                return Ok(false);
            }
        }
        *self = OpenWorkbook::load(&self.path.clone())?;
        Ok(true)
    }

    fn tools(&self) -> Tools<'_> {
        Tools {
            wb: &self.wb,
            graph: &self.graph,
            hybrid: &self.hybrid,
            calc: &self.calc,
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
    /// True only once an `initialize` *request* has succeeded;
    /// a bare notifications/initialized never unlocks the server.
    initialize_seen: bool,
}

impl Default for McpServer {
    fn default() -> Self {
        McpServer {
            workbooks: HashMap::new(),
            order: std::collections::VecDeque::new(),
            next_id: 1,
            initialized: false,
            initialize_seen: false,
        }
    }
}

/// A JSON-RPC error: code, message, and the typed kind (sent as
/// `data.kind`) when it comes from a tool call.
type RpcError = (i64, String, Option<ErrorKind>);

/// An invalid-arguments protocol error (-32602).
fn invalid(message: String) -> RpcError {
    (-32602, message, Some(ErrorKind::InvalidArguments))
}

fn req_str<'a>(args: &'a Value, field: &str) -> Result<&'a str, RpcError> {
    args.get(field)
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid(format!("missing or non-string argument: {field}")))
}

/// MCP's tools: the two session tools, then every shared tool from
/// `ToolRequest::schemas()` with a required `workbook_id` added. One
/// definition per tool, shared with `gridtalk tools`.
fn tool_definitions() -> &'static Value {
    static DEFINITIONS: OnceLock<Value> = OnceLock::new();
    DEFINITIONS.get_or_init(build_tool_definitions)
}

fn build_tool_definitions() -> Value {
    let annotations = |read_only: bool| json!({"readOnlyHint": read_only, "destructiveHint": false, "idempotentHint": read_only, "openWorldHint": false});
    let mut tools = vec![
        json!({"name": "open_workbook", "description": "Open a spreadsheet (csv/tsv/xlsx/ods; path on this machine) and return its session id, sheets, regions and load warnings. Opening a path that is already open returns its existing session. Reads the file; never writes it.", "inputSchema": {"type": "object", "additionalProperties": false, "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": annotations(true)}),
        json!({"name": "close_workbook", "description": "Close an open workbook session", "inputSchema": {"type": "object", "additionalProperties": false, "properties": {"workbook_id": {"type": "string"}}, "required": ["workbook_id"]}, "annotations": annotations(false)}),
    ];
    let shared = ToolRequest::schemas();
    for t in shared["tools"].as_array().into_iter().flatten() {
        let mut t = t.clone();
        if let Some(obj) = t.as_object_mut() {
            obj.remove("tool");
        }
        if let Some(props) = t
            .pointer_mut("/inputSchema/properties")
            .and_then(|p| p.as_object_mut())
        {
            props.insert(
                "workbook_id".into(),
                json!({"type": "string", "description": "The id open_workbook returned"}),
            );
        }
        if let Some(required) = t
            .pointer_mut("/inputSchema/required")
            .and_then(|r| r.as_array_mut())
        {
            required.insert(0, json!("workbook_id"));
        }
        tools.push(t);
    }
    Value::Array(tools)
}

/// The argument names a tool's input schema allows.
fn allowed_arguments(definitions: &Value, name: &str) -> Option<Vec<String>> {
    definitions
        .as_array()?
        .iter()
        .find(|t| t["name"] == name)?
        .pointer("/inputSchema/properties")?
        .as_object()
        .map(|p| p.keys().cloned().collect())
}

/// A tool failure as a tool result: `isError` with the typed
/// `{"error": {"kind", "message"}}` payload.
fn tool_error(kind: ErrorKind, message: String) -> Result<Value, RpcError> {
    let data = error_payload(kind, &message);
    Ok(json!({
        "content": [{"type": "text", "text": data.to_string()}],
        "structuredContent": data,
        "isError": true
    }))
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
        // Notifications (no id) get no response; handle them before
        // any id-based early return so notifications/initialized
        // can actually land.
        if request.get("id").is_none() {
            // A bare notifications/initialized never unlocks the
            // server: only a completed initialize request does.
            if request.get("method").and_then(|m| m.as_str()) == Some("notifications/initialized")
                && self.initialize_seen
            {
                self.initialized = true;
            }
            return None;
        }
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        if request.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
            return Some(
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32600, "message": "invalid request: jsonrpc must be \"2.0\""}}),
            );
        }
        let method = match request.get("method").and_then(|m| m.as_str()) {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => {
                return Some(
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32600, "message": "invalid request: missing method"}}),
                );
            }
        };
        let result = match method.as_str() {
            "initialize" => {
                if self.initialize_seen {
                    Err((-32602, "already initialized".to_string(), None))
                } else {
                    self.initialize_seen = true;
                    self.initialized = true;
                    let client_version = request
                        .pointer("/params/protocolVersion")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    // Negotiate: echo the client's version only when
                    // we support it, otherwise offer our newest.
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
            }
            "ping" => Ok(json!({})),
            "tools/list" => {
                if !self.initialized {
                    Err((
                        -32002,
                        "server not initialized: send initialize first".to_string(),
                        None,
                    ))
                } else {
                    Ok(json!({"tools": tool_definitions()}))
                }
            }
            "tools/call" => {
                if !self.initialized {
                    Err((
                        -32002,
                        "server not initialized: send initialize first".to_string(),
                        None,
                    ))
                } else {
                    self.call_tool(request.pointer("/params").cloned().unwrap_or(json!({})))
                }
            }
            other => Err((-32601, format!("method not found: {other}"), None)),
        };
        Some(match result {
            Ok(value) => json!({"jsonrpc": "2.0", "id": id, "result": value}),
            Err((code, message, None)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
            }
            Err((code, message, Some(kind))) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": {"kind": kind}}})
            }
        })
    }

    /// Run one `tools/call`. Protocol errors (an unknown tool,
    /// arguments that break the tool's schema) are JSON-RPC -32602
    /// errors with `data.kind`; failures while running a tool (an
    /// unknown or vanished workbook, an unreadable file, a bad range)
    /// are `isError` results with `{"error": {"kind", "message"}}`.
    fn call_tool(&mut self, params: Value) -> Result<Value, RpcError> {
        let name = params
            .get("name")
            .and_then(|n| n.as_str())
            .filter(|n| !n.is_empty())
            .ok_or_else(|| invalid("missing or non-string argument: name".to_string()))?
            .to_string();
        let name = name.as_str();
        let Some(allowed) = allowed_arguments(tool_definitions(), name) else {
            return Err((
                -32602,
                format!("unknown tool: {name}"),
                Some(ErrorKind::UnknownTool),
            ));
        };
        let args = params.get("arguments").cloned().unwrap_or(json!({}));
        let Some(obj) = args.as_object() else {
            return Err(invalid("arguments must be an object".to_string()));
        };
        // Schemas declare additionalProperties: false.
        if let Some(key) = obj.keys().find(|k| !allowed.contains(k)) {
            return Err(invalid(format!("unknown argument: {key}")));
        }
        let structured_result = |data: Value, is_error: bool| {
            Ok(json!({
                "content": [{"type": "text", "text": data.to_string()}],
                "structuredContent": data,
                "isError": is_error
            }))
        };
        match name {
            "open_workbook" => {
                let path = PathBuf::from(req_str(&args, "path")?);
                // One session per file: reopening a path (however it is
                // spelled) returns the open session, refreshed.
                let key = std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                if let Some(id) = self.order.iter().find(|id| {
                    self.workbooks.get(*id).is_some_and(|w| {
                        std::fs::canonicalize(&w.path).unwrap_or_else(|_| w.path.clone()) == key
                    })
                }) {
                    let id = id.clone();
                    // Reopening counts as a fresh open for eviction.
                    self.order.retain(|x| x != &id);
                    self.order.push_back(id.clone());
                    let open = self.workbooks.get_mut(&id).expect("listed session");
                    if let Err(e) = open.fresh() {
                        return tool_error(ErrorKind::WorkbookUnavailable, e);
                    }
                    let mut payload = open_payload(&id, open);
                    payload["reused"] = json!(true);
                    return structured_result(payload, false);
                }
                match OpenWorkbook::load(&path) {
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
                        let mut evicted: Vec<String> = Vec::new();
                        while self.workbooks.len() >= MAX_WORKBOOKS {
                            match self.order.pop_front() {
                                Some(old) => {
                                    self.workbooks.remove(&old);
                                    evicted.push(old);
                                }
                                None => break,
                            }
                        }
                        self.order.push_back(id.clone());
                        let mut payload = open_payload(&id, &open);
                        payload["reused"] = json!(false);
                        payload["evicted_workbook_ids"] = json!(evicted);
                        self.workbooks.insert(id, open);
                        structured_result(payload, false)
                    }
                    Err(e) => tool_error(ErrorKind::OpenFailed, format!("open failed: {e}")),
                }
            }
            "close_workbook" => {
                let id = req_str(&args, "workbook_id")?;
                if self.workbooks.remove(id).is_some() {
                    self.order.retain(|x| x != id);
                    structured_result(json!({"closed": id}), false)
                } else {
                    tool_error(
                        ErrorKind::UnknownWorkbook,
                        format!("unknown workbook_id {id}"),
                    )
                }
            }
            _ => {
                let id = req_str(&args, "workbook_id")?.to_string();
                // The shared request is the arguments minus the session
                // id, tagged with the tool: ToolRequest's own serde
                // checks every field and type.
                let mut fields = obj.clone();
                fields.remove("workbook_id");
                if let Some(limit) = fields.get("limit") {
                    if limit.as_u64().is_none_or(|n| n < 1) {
                        return Err(invalid("limit must be a positive integer".to_string()));
                    }
                }
                fields.insert("tool".into(), json!(name));
                let req: ToolRequest = serde_json::from_value(Value::Object(fields))
                    .map_err(|e| invalid(format!("invalid arguments for {name}: {e}")))?;
                let Some(open) = self.workbooks.get_mut(&id) else {
                    return tool_error(
                        ErrorKind::UnknownWorkbook,
                        format!("unknown workbook_id {id}; call open_workbook first"),
                    );
                };
                let reloaded = match open.fresh() {
                    Ok(reloaded) => reloaded,
                    Err(e) => return tool_error(ErrorKind::WorkbookUnavailable, e),
                };
                let result = open.tools().call(&req);
                let mut data = result.data;
                // A silent reload replaced the workbook under this
                // session: say so, with the new file's warnings.
                if reloaded {
                    if let Some(obj) = data.as_object_mut() {
                        obj.insert("reloaded".into(), json!({"warnings": open.wb.warnings}));
                    }
                }
                // Machines get structuredContent; the text content
                // carries the same payload for text-only clients.
                structured_result(data, !result.ok)
            }
        }
    }
}

/// What `open_workbook` reports about a session.
fn open_payload(id: &str, open: &OpenWorkbook) -> Value {
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
    json!({"workbook_id": id, "sheets": sheets, "regions": regions, "warnings": open.wb.warnings})
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
    let mut discarding = false;
    loop {
        let mut raw: Vec<u8> = Vec::new();
        loop {
            // Read in bounded 8 KiB chunks so memory stays capped
            // even before the 1 MiB frame check can fire.
            let mut chunk = Vec::new();
            let n = {
                let available = reader.fill_buf().map_err(|e| e.to_string())?;
                if available.is_empty() {
                    0
                } else {
                    let take_n = available
                        .iter()
                        .position(|b| *b == b'\n')
                        .map(|i| i + 1)
                        .unwrap_or(available.len())
                        .min(8192);
                    chunk.extend_from_slice(&available[..take_n]);
                    reader.consume(take_n);
                    take_n
                }
            };
            if n == 0 {
                break;
            }
            let done = chunk.ends_with(b"\n");
            raw.extend_from_slice(&chunk);
            if raw.len() > MAX_LINE {
                if !discarding {
                    // One error per oversized frame, then discard
                    // through the newline without parsing the tail.
                    discarding = true;
                    let resp = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "request line exceeds the 1 MiB cap"}});
                    writeln!(stdout, "{resp}").map_err(|e| e.to_string())?;
                    stdout.flush().map_err(|e| e.to_string())?;
                }
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
        if discarding {
            // The oversized frame ended at its newline; resume
            // normal framing with the next line.
            discarding = false;
            continue;
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

    /// Tool names are one vocabulary: MCP's list is the shared tool
    /// list plus its session tools, and each shared name round-trips
    /// through `ToolRequest` and back out of `ToolRequest::name`.
    #[test]
    fn mcp_and_cli_share_tool_names() {
        let names = |tools: &Value| -> Vec<String> {
            tools
                .as_array()
                .expect("tool list")
                .iter()
                .map(|t| t["name"].as_str().expect("tool name").to_string())
                .collect()
        };
        let shared = names(&ToolRequest::schemas()["tools"]);
        let mcp: Vec<String> = names(tool_definitions())
            .into_iter()
            .filter(|n| n != "open_workbook" && n != "close_workbook")
            .collect();
        assert_eq!(mcp, shared);
        for (tool, args) in [
            ("search", json!({"query": "q"})),
            ("expand", json!({"range": "S!A1"})),
            ("get_cells", json!({"range": "S!A1"})),
            ("list_formulas", json!({"range": "S!A1"})),
            (
                "what_if",
                json!({"cell": "A1", "value": 1.0, "target": "S!A2"}),
            ),
            ("graph_overview", json!({})),
        ] {
            assert!(shared.iter().any(|n| n == tool), "{tool} not listed");
            let mut request = args;
            request["tool"] = json!(tool);
            let parsed: ToolRequest = serde_json::from_value(request).expect(tool);
            assert_eq!(parsed.name(), tool);
        }
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

        // An unknown workbook is a tool failure with a typed kind.
        let bad = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "search", "arguments": {"workbook_id": "wb-999", "query": "x"}}}),
        );
        assert_eq!(bad["result"]["isError"], true);
        assert_eq!(
            bad["result"]["structuredContent"]["error"]["kind"],
            "unknown_workbook"
        );
        // Missing required args are -32602, never silent defaults.
        let missing = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "search", "arguments": {"workbook_id": wb_id}}}),
        );
        assert_eq!(missing["error"]["code"], -32602);
        assert_eq!(missing["error"]["data"]["kind"], "invalid_arguments");
        // Non-object frames get -32600, not silence.
        let batch = server
            .handle(&json!([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]))
            .unwrap();
        assert_eq!(batch["error"]["code"], -32600);
        // A bare notifications/initialized must NOT unlock the
        // server (no initialize request was ever answered).
        let mut fresh2 = McpServer::default();
        let _ = fresh2.handle(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        let locked = rpc(
            &mut fresh2,
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        );
        assert_eq!(locked["error"]["code"], -32002);
        // Unsupported protocol versions are negotiated down.
        let mut fresh = McpServer::default();
        let init = rpc(
            &mut fresh,
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}),
        );
        assert_eq!(init["result"]["protocolVersion"], "2025-03-26");
    }

    #[test]
    fn typed_errors_annotations_session_reuse_and_reload_warnings() {
        let mut server = McpServer::default();
        rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}),
        );
        let list = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        );
        let tools = list["result"]["tools"].as_array().unwrap();
        for t in tools {
            assert_eq!(t["annotations"]["destructiveHint"], false, "{}", t["name"]);
        }
        let info = tools.iter().find(|t| t["name"] == "workbook_info").unwrap();
        assert_eq!(info["inputSchema"]["required"], json!(["workbook_id"]));
        let get_cells = tools.iter().find(|t| t["name"] == "get_cells").unwrap();
        assert!(get_cells["description"]
            .as_str()
            .unwrap()
            .contains("Sheet1!A1:D4"));
        assert_eq!(get_cells["annotations"]["readOnlyHint"], true);

        let dir = std::env::temp_dir().join(format!("gt-mcp-reuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("book.csv");
        std::fs::write(&path, "Item,Qty\nApples,2\n").unwrap();
        let open = |server: &mut McpServer, p: &std::path::Path| {
            rpc(
                server,
                json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "open_workbook", "arguments": {"path": p.to_str().unwrap()}}}),
            )["result"]["structuredContent"]
                .clone()
        };
        let first = open(&mut server, &path);
        assert_eq!(first["reused"], false);
        // The same file under another spelling reuses the session.
        let again = open(&mut server, &dir.join(".").join("book.csv"));
        assert_eq!(again["reused"], true);
        assert_eq!(again["workbook_id"], first["workbook_id"]);
        let id = first["workbook_id"].clone();

        let call = |server: &mut McpServer, name: &str, args: Value| {
            let mut args = args;
            args["workbook_id"] = id.clone();
            rpc(
                server,
                json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": name, "arguments": args}}),
            )
        };
        let info = call(&mut server, "workbook_info", json!({}));
        assert_eq!(
            info["result"]["structuredContent"]["sheets"][0]["name"],
            "Sheet1"
        );
        let bad_range = call(&mut server, "get_cells", json!({"range": "Sheet1!A0"}));
        assert_eq!(bad_range["result"]["isError"], true);
        assert_eq!(
            bad_range["result"]["structuredContent"]["error"]["kind"],
            "invalid_range"
        );
        let no_sheet = call(&mut server, "get_cells", json!({"range": "Nope!A1"}));
        assert_eq!(
            no_sheet["result"]["structuredContent"]["error"]["kind"],
            "unknown_sheet"
        );
        let wrong_type = call(
            &mut server,
            "what_if",
            json!({"cell": "B2", "value": "ten", "target": "Sheet1!B2"}),
        );
        assert_eq!(wrong_type["error"]["code"], -32602);
        let unknown = rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}),
        );
        assert_eq!(unknown["error"]["data"]["kind"], "unknown_tool");
        let clamped = call(
            &mut server,
            "search",
            json!({"query": "apples", "limit": 99}),
        );
        assert_eq!(clamped["result"]["structuredContent"]["limit"], 25);
        assert_eq!(
            clamped["result"]["structuredContent"]["limit_clamped"],
            true
        );

        // A changed file reloads silently; the next response says so.
        std::fs::write(&path, "Item,Qty,Note\nApples,2,x\nPears,4,y\n").unwrap();
        let after = call(&mut server, "workbook_info", json!({}));
        let data = &after["result"]["structuredContent"];
        assert!(data["reloaded"].is_object(), "{data}");
        assert_eq!(data["sheets"][0]["cols"], 3);
        let quiet = call(&mut server, "workbook_info", json!({}));
        assert!(quiet["result"]["structuredContent"]["reloaded"].is_null());

        std::fs::remove_file(&path).unwrap();
        let gone = call(&mut server, "workbook_info", json!({}));
        assert_eq!(
            gone["result"]["structuredContent"]["error"]["kind"],
            "workbook_unavailable"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
