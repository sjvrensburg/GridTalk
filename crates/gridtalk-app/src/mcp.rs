//! MCP server (Phase B of the agentic-interfaces plan): GridTalk
//! tools over Model Context Protocol JSON-RPC on stdio, so Claude
//! Code / Codex / any MCP client can drive a workbook without the
//! UI. Implements the MCP subset GridTalk needs — `initialize`,
//! `ping`, `tools/list`, `tools/call` — directly on the spec's
//! JSON-RPC framing (one JSON object per line on stdio), with the
//! typed tool core from `tools.rs` doing the work.
//!
//! Privacy/safety: the server only opens paths the client passes
//! to `open_workbook`, and with `--root` only paths under those
//! directories. Every investigation tool is read-only (`what_if`
//! simulates; nothing writes the file). There is no write tool in
//! the list at all.
//!
//! Threads: the stdio thread reads frames and writes every response.
//! `open_workbook` and the workbook tools run on worker threads (at
//! most [`MAX_RUNNING_CALLS`]), so `ping` and other calls are answered
//! while one is busy, and a call that runs past the time limit is
//! answered with a `timeout` error. Gridcraft cannot be interrupted, so
//! a timed-out call keeps its thread (and its slot) until it finishes;
//! its result is then dropped.

use crate::tools::{error_payload, ErrorKind, ToolRequest, Tools};
use gridtalk_graph::eval::EvalCache;
use gridtalk_graph::{build_graph, load_bytes, read_workbook_bytes, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Write};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

/// (mtime, len, change stamp, content hash).
type Stamp = (SystemTime, u64, i64, u64);

struct OpenWorkbook {
    /// Canonical path, checked against the roots at open and on each
    /// reload.
    path: PathBuf,
    /// Stat first on every call; the file is only re-read/hashed
    /// when stat changed, and only reloaded when the hash also
    /// changed. A content-preserving touch refreshes it in place,
    /// hence the lock: sessions are shared with worker threads.
    stamp: Mutex<Option<Stamp>>,
    wb: WorkbookSnapshot,
    graph: gridtalk_graph::Graph,
    hybrid: HybridIndex,
    /// Recalculated once per loaded revision of the file: a reload
    /// replaces the whole session, cache included.
    calc: EvalCache,
    /// Estimated memory, charged against the server's budget.
    bytes: u64,
}

/// A session's resident memory per byte of its snapshot estimate
/// ([`WorkbookSnapshot::approx_bytes`]). Measured on generated
/// workbooks of 5k to 1.2M cells: the Gridcraft source workbook,
/// graph, index and recalculated engine together held 12 to 25 times
/// the estimate, so 16 is a middle, not a bound.
const SESSION_BYTES_PER_SNAPSHOT_BYTE: u64 = 16;

/// Held while a workbook loads, so one load runs at a time however
/// many calls do: the budget is charged only once a load finishes, and
/// this keeps the peak above it to one workbook being built.
static LOADING: Mutex<()> = Mutex::new(());

impl OpenWorkbook {
    fn stat_of(path: &Path) -> Option<(SystemTime, u64, i64)> {
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

    fn hash_of(path: &Path) -> Option<u64> {
        read_workbook_bytes(path)
            .ok()
            .map(|bytes| Self::hash_bytes(&bytes))
    }

    /// Load the file at `path`, which the caller has canonicalised
    /// and checked against the roots.
    fn load(path: &Path) -> Result<Self, String> {
        let _loading = LOADING.lock().unwrap_or_else(|e| e.into_inner());
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
        let bytes = wb
            .approx_bytes()
            .saturating_mul(SESSION_BYTES_PER_SNAPSHOT_BYTE);
        Ok(OpenWorkbook {
            path: path.to_path_buf(),
            stamp: Mutex::new(stamp),
            wb,
            graph,
            hybrid,
            calc: EvalCache::new(),
            bytes,
        })
    }

    /// The session's file reloaded, if it changed since it was read
    /// (`None` when it did not). The path must still resolve inside
    /// the roots: a directory swapped for a symlink after the open is
    /// caught here.
    fn fresh(&self, roots: &[PathBuf]) -> Result<Option<Box<OpenWorkbook>>, (ErrorKind, String)> {
        let unavailable = || {
            (
                ErrorKind::WorkbookUnavailable,
                format!(
                    "workbook {} was deleted or became unreadable",
                    self.path.display()
                ),
            )
        };
        let resolved = std::fs::canonicalize(&self.path).map_err(|_| unavailable())?;
        if !path_allowed(roots, &resolved) {
            return Err((
                ErrorKind::PathNotAllowed,
                format!(
                    "workbook {} now resolves to {}, outside the server's --root directories",
                    self.path.display(),
                    resolved.display()
                ),
            ));
        }
        // Stat first (cheap): unchanged stat means no re-read at
        // all. Changed stat re-hashes; only a changed hash (real
        // content change) triggers the expensive reload.
        let (mtime, len, ctime) = Self::stat_of(&resolved).ok_or_else(unavailable)?;
        let old = *self.stamp.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((old_mtime, old_len, old_ctime, old_hash)) = old {
            // Windows has no change counter in std metadata, so
            // (mtime, len, file_index) cannot detect an in-place
            // same-length edit: always fall through to the hash.
            if !cfg!(windows) && (mtime, len, ctime) == (old_mtime, old_len, old_ctime) {
                return Ok(None);
            }
            if Self::hash_of(&resolved) == Some(old_hash) {
                // Content-preserving touch: refresh the stamp.
                *self.stamp.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((mtime, len, ctime, old_hash));
                return Ok(None);
            }
        }
        let mut reloaded =
            OpenWorkbook::load(&resolved).map_err(|e| (ErrorKind::WorkbookUnavailable, e))?;
        reloaded.path = self.path.clone();
        Ok(Some(Box::new(reloaded)))
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

/// Whether a canonical path lies under one of the canonical roots
/// (any path, when there are none).
fn path_allowed(roots: &[PathBuf], canonical: &Path) -> bool {
    roots.is_empty() || roots.iter().any(|root| canonical.starts_with(root))
}

pub const MAX_WORKBOOKS: usize = 16;
/// Most calls running on worker threads at once, timed-out ones
/// included (they run on until Gridcraft returns).
pub const MAX_RUNNING_CALLS: usize = 4;
const SUPPORTED_PROTOCOLS: [&str; 2] = ["2025-03-26", "2024-11-05"];

/// How `gridtalk mcp` was started.
#[derive(Clone, Debug)]
pub struct McpOptions {
    /// Canonical directories `open_workbook` may open files under;
    /// empty allows any path the process can read.
    pub roots: Vec<PathBuf>,
    /// How long the stdio thread waits for one call before it answers
    /// `timeout`.
    pub call_timeout: Duration,
    /// Estimated memory the open sessions may hold together; the
    /// oldest are evicted to make room.
    pub memory_budget: u64,
}

impl Default for McpOptions {
    fn default() -> Self {
        McpOptions {
            roots: Vec::new(),
            call_timeout: Duration::from_secs(120),
            memory_budget: 4 * 1024 * 1024 * 1024,
        }
    }
}

impl McpOptions {
    /// Allow files under `dir` (canonicalised, so a symlinked root is
    /// its target).
    pub fn add_root(&mut self, dir: &Path) -> Result<(), String> {
        let canonical =
            std::fs::canonicalize(dir).map_err(|e| format!("--root {}: {e}", dir.display()))?;
        if !canonical.is_dir() {
            return Err(format!("--root {} is not a directory", dir.display()));
        }
        self.roots.push(canonical);
        Ok(())
    }
}

pub struct McpServer {
    options: McpOptions,
    workbooks: HashMap<String, Arc<OpenWorkbook>>,
    order: VecDeque<String>,
    next_id: u64,
    initialized: bool,
    /// True only once an `initialize` *request* has succeeded;
    /// a bare notifications/initialized never unlocks the server.
    initialize_seen: bool,
    /// Tests: how long every job sleeps first, to make one slow.
    #[cfg(test)]
    job_delay: Option<Duration>,
}

impl Default for McpServer {
    fn default() -> Self {
        McpServer::new(McpOptions::default())
    }
}

/// A JSON-RPC error: code, message, and the typed kind (sent as
/// `data.kind`) when it comes from a tool call.
type RpcError = (i64, String, Option<ErrorKind>);

/// What a call does off the stdio thread.
type Job = Box<dyn FnOnce() -> Done + Send + 'static>;

/// A finished job, applied to the server on the stdio thread.
enum Done {
    /// `open_workbook` loaded (or failed to load) a new file.
    Opened(Result<Box<OpenWorkbook>, String>),
    /// A call on session `id`, with the session's replacement when
    /// its file changed.
    Session {
        id: String,
        /// The session the call started from: a reload replaces the
        /// session only while it is still this one, so a slow (or
        /// timed-out) call never installs an older revision over a
        /// newer reload.
        base: Arc<OpenWorkbook>,
        reload: Option<Box<OpenWorkbook>>,
        result: Result<Value, RpcError>,
    },
    /// A call that failed before touching any session.
    Failed(Result<Value, RpcError>),
}

/// What dispatching one request produced.
enum Step {
    /// The response (`None` for a notification).
    Reply(Option<Value>),
    /// Work to run; its response goes to request `id`.
    Run { id: Value, job: Job },
}

/// The JSON-RPC response to request `id`.
fn response(id: Value, result: Result<Value, RpcError>) -> Value {
    match result {
        Ok(value) => json!({"jsonrpc": "2.0", "id": id, "result": value}),
        Err((code, message, None)) => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
        }
        Err((code, message, Some(kind))) => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message, "data": {"kind": kind}}})
        }
    }
}

/// Run a job, turning a panic into an `internal` tool error, so one
/// bad call neither kills the server nor leaks its running slot.
fn run_job(job: Job) -> Done {
    std::panic::catch_unwind(AssertUnwindSafe(job)).unwrap_or_else(|_| {
        Done::Failed(tool_error(
            ErrorKind::Internal,
            "the call panicked inside GridTalk; this is a bug".to_string(),
        ))
    })
}

/// A tool result: the payload as `structuredContent`, and as text for
/// text-only clients.
fn structured_result(data: Value, is_error: bool) -> Result<Value, RpcError> {
    Ok(json!({
        "content": [{"type": "text", "text": data.to_string()}],
        "structuredContent": data,
        "isError": is_error
    }))
}

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
        json!({"name": "open_workbook", "description": "Open a spreadsheet (csv/tsv/xlsx/xlsm/xltx/xltm/xls/xlsb/ods; path on this machine; .xlsb and .ods formulas come in as their saved values, see warnings) and return its session id, sheets, regions and load warnings. Opening a path that is already open returns its existing session. Reads the file; never writes it.", "inputSchema": {"type": "object", "additionalProperties": false, "properties": {"path": {"type": "string"}}, "required": ["path"]}, "annotations": annotations(true)}),
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

/// A `tools/call` answered now, or work for a worker thread.
enum Call {
    Now(Result<Value, RpcError>),
    Later(Job),
}

impl McpServer {
    pub fn new(options: McpOptions) -> Self {
        McpServer {
            options,
            workbooks: HashMap::new(),
            order: VecDeque::new(),
            next_id: 1,
            initialized: false,
            initialize_seen: false,
            #[cfg(test)]
            job_delay: None,
        }
    }

    /// Handle one JSON-RPC request object on this thread, tool work
    /// included; returns the response object, or None for
    /// notifications.
    #[cfg(test)]
    pub fn handle(&mut self, request: &Value) -> Option<Value> {
        match self.dispatch(request) {
            Step::Reply(response) => response,
            Step::Run { id, job } => {
                let done = run_job(job);
                Some(response(id, self.finish(done)))
            }
        }
    }

    /// Answer a request, or hand back the work a `tools/call` needs.
    fn dispatch(&mut self, request: &Value) -> Step {
        // Non-object frames (batches, scalars) are Invalid Request,
        // never silently dropped.
        if !request.is_object() {
            return Step::Reply(Some(
                json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "invalid request: expected a JSON-RPC object"}}),
            ));
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
            return Step::Reply(None);
        }
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        if request.get("jsonrpc").and_then(|v| v.as_str()) != Some("2.0") {
            return Step::Reply(Some(
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32600, "message": "invalid request: jsonrpc must be \"2.0\""}}),
            ));
        }
        let method = match request.get("method").and_then(|m| m.as_str()) {
            Some(m) if !m.is_empty() => m.to_string(),
            _ => {
                return Step::Reply(Some(
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32600, "message": "invalid request: missing method"}}),
                ));
            }
        };
        let not_initialized = || {
            Err((
                -32002,
                "server not initialized: send initialize first".to_string(),
                None,
            ))
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
            "tools/list" if !self.initialized => not_initialized(),
            "tools/list" => Ok(json!({"tools": tool_definitions()})),
            "tools/call" if !self.initialized => not_initialized(),
            "tools/call" => {
                match self.call_tool(request.pointer("/params").cloned().unwrap_or(json!({}))) {
                    Ok(Call::Now(result)) => result,
                    Ok(Call::Later(job)) => {
                        #[cfg(test)]
                        let job: Job = match self.job_delay {
                            Some(delay) => Box::new(move || {
                                std::thread::sleep(delay);
                                job()
                            }),
                            None => job,
                        };
                        return Step::Run { id, job };
                    }
                    Err(e) => Err(e),
                }
            }
            other => Err((-32601, format!("method not found: {other}"), None)),
        };
        Step::Reply(Some(response(id, result)))
    }

    /// Check one `tools/call` and prepare its work. Protocol errors
    /// (an unknown tool, arguments that break the tool's schema) are
    /// JSON-RPC -32602 errors with `data.kind`; failures while running
    /// a tool (an unknown or vanished workbook, an unreadable file, a
    /// bad range) are `isError` results with `{"error": {"kind",
    /// "message"}}`.
    fn call_tool(&mut self, params: Value) -> Result<Call, RpcError> {
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
        match name {
            "open_workbook" => {
                let requested = PathBuf::from(req_str(&args, "path")?);
                // Canonical: one session per file however it is
                // spelled, and the root check sees through symlinks.
                // A path that does not resolve is only reported as
                // missing when its directory is inside the roots, so
                // the error never says what exists outside them.
                let resolved = std::fs::canonicalize(&requested).map_err(|e| {
                    // Without roots every path is allowed, so a missing
                    // one is simply missing. A bare file name's parent
                    // is "", which means the current directory.
                    let parent_inside = self.options.roots.is_empty()
                        || requested
                            .parent()
                            .map(|p| {
                                if p.as_os_str().is_empty() {
                                    Path::new(".")
                                } else {
                                    p
                                }
                            })
                            .and_then(|p| std::fs::canonicalize(p).ok())
                            .is_some_and(|p| path_allowed(&self.options.roots, &p));
                    (e, parent_inside)
                });
                let path = match resolved {
                    Ok(path) if path_allowed(&self.options.roots, &path) => path,
                    Err((e, true)) => {
                        return Ok(Call::Now(tool_error(
                            ErrorKind::OpenFailed,
                            format!("open failed: {}: {e}", requested.display()),
                        )))
                    }
                    _ => {
                        let roots: Vec<String> = self
                            .options
                            .roots
                            .iter()
                            .map(|r| r.display().to_string())
                            .collect();
                        return Ok(Call::Now(tool_error(
                            ErrorKind::PathNotAllowed,
                            format!(
                                "{} is outside the server's --root directories: {}",
                                requested.display(),
                                roots.join(", ")
                            ),
                        )));
                    }
                };
                // Reopening a path returns the open session, refreshed.
                if let Some(id) = self.session_for(&path) {
                    // Reopening counts as a fresh open for eviction.
                    self.order.retain(|x| x != &id);
                    self.order.push_back(id.clone());
                    let open = Arc::clone(&self.workbooks[&id]);
                    let roots = self.options.roots.clone();
                    return Ok(Call::Later(Box::new(move || match open.fresh(&roots) {
                        Err((kind, e)) => Done::Failed(tool_error(kind, e)),
                        Ok(reload) => {
                            let mut payload = open_payload(&id, reload.as_deref().unwrap_or(&open));
                            payload["reused"] = json!(true);
                            Done::Session {
                                id,
                                base: open,
                                reload,
                                result: structured_result(payload, false),
                            }
                        }
                    })));
                }
                Ok(Call::Later(Box::new(move || {
                    Done::Opened(OpenWorkbook::load(&path).map(Box::new))
                })))
            }
            "close_workbook" => {
                let id = req_str(&args, "workbook_id")?;
                Ok(Call::Now(if self.workbooks.remove(id).is_some() {
                    self.order.retain(|x| x != id);
                    structured_result(json!({"closed": id}), false)
                } else {
                    tool_error(
                        ErrorKind::UnknownWorkbook,
                        format!("unknown workbook_id {id}"),
                    )
                }))
            }
            _ => {
                let id = req_str(&args, "workbook_id")?.to_string();
                // The shared request is the arguments minus the session
                // id, tagged with the tool: ToolRequest's own serde
                // checks every field and type.
                let mut fields = obj.clone();
                fields.remove("workbook_id");
                // Like req_str: an empty string is a missing argument,
                // never a silent empty query or range.
                if let Some(key) = fields.iter().find(|(_, v)| v.as_str() == Some("")) {
                    return Err(invalid(format!(
                        "missing or non-string argument: {}",
                        key.0
                    )));
                }
                if let Some(limit) = fields.get("limit") {
                    if limit.as_u64().is_none_or(|n| n < 1) {
                        return Err(invalid("limit must be a positive integer".to_string()));
                    }
                }
                fields.insert("tool".into(), json!(name));
                let req: ToolRequest = serde_json::from_value(Value::Object(fields))
                    .map_err(|e| invalid(format!("invalid arguments for {name}: {e}")))?;
                let Some(open) = self.workbooks.get(&id).cloned() else {
                    return Ok(Call::Now(tool_error(
                        ErrorKind::UnknownWorkbook,
                        format!("unknown workbook_id {id}; call open_workbook first"),
                    )));
                };
                let roots = self.options.roots.clone();
                Ok(Call::Later(Box::new(move || {
                    let reload = match open.fresh(&roots) {
                        Ok(reload) => reload,
                        Err((kind, e)) => return Done::Failed(tool_error(kind, e)),
                    };
                    let current = reload.as_deref().unwrap_or(&open);
                    let result = current.tools().call(&req);
                    let mut data = result.data;
                    // A silent reload replaced the workbook under this
                    // session: say so, with the new file's warnings.
                    if reload.is_some() {
                        if let Some(obj) = data.as_object_mut() {
                            obj.insert("reloaded".into(), json!({"warnings": current.wb.warnings}));
                        }
                    }
                    Done::Session {
                        id,
                        base: open,
                        reload,
                        result: structured_result(data, !result.ok),
                    }
                })))
            }
        }
    }

    /// The open session for a canonical path.
    fn session_for(&self, path: &Path) -> Option<String> {
        self.order
            .iter()
            .find(|id| self.workbooks.get(*id).is_some_and(|w| w.path == path))
            .cloned()
    }

    /// Apply a finished job to the sessions and return its result.
    fn finish(&mut self, done: Done) -> Result<Value, RpcError> {
        match done {
            Done::Failed(result) => result,
            Done::Session {
                id,
                base,
                reload,
                result,
            } => {
                // A session closed or evicted while the call ran stays
                // gone, and one another call already replaced keeps
                // its newer revision; the call still gets its answer.
                if let Some(new) = reload {
                    if self
                        .workbooks
                        .get(&id)
                        .is_some_and(|current| Arc::ptr_eq(current, &base))
                    {
                        // A file that grew past the whole budget is
                        // refused like a new open: the old revision is
                        // stale, so the session closes.
                        let budget = self.options.memory_budget;
                        if new.bytes > budget {
                            self.workbooks.remove(&id);
                            self.order.retain(|x| x != &id);
                            return tool_error(
                                ErrorKind::OverBudget,
                                format!(
                                    "{} changed and now needs an estimated {} MiB, over the server's {} MiB memory budget (gridtalk mcp --memory-budget); session {id} was closed",
                                    new.path.display(),
                                    new.bytes.div_ceil(1 << 20),
                                    budget >> 20
                                ),
                            );
                        }
                        self.workbooks.insert(id.clone(), Arc::from(new));
                        let evicted = self.make_room(0, 0, Some(&id));
                        if !evicted.is_empty() {
                            eprintln!(
                                "gridtalk mcp: the reloaded {id} grew; evicted {} to stay in the memory budget",
                                evicted.join(", ")
                            );
                        }
                    }
                }
                result
            }
            Done::Opened(Err(e)) => tool_error(ErrorKind::OpenFailed, format!("open failed: {e}")),
            Done::Opened(Ok(open)) => {
                // Two opens of one file raced: keep the first session.
                if let Some(id) = self.session_for(&open.path) {
                    let mut payload = open_payload(&id, &self.workbooks[&id]);
                    payload["reused"] = json!(true);
                    return structured_result(payload, false);
                }
                let budget = self.options.memory_budget;
                if open.bytes > budget {
                    return tool_error(
                        ErrorKind::OverBudget,
                        format!(
                            "{} needs an estimated {} MiB, over the server's {} MiB memory budget (gridtalk mcp --memory-budget)",
                            open.path.display(),
                            open.bytes.div_ceil(1 << 20),
                            budget >> 20
                        ),
                    );
                }
                // Unpredictable session ids + a hard cap with
                // oldest-first eviction.
                let nanos = SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let id = format!(
                    "wb-{:x}-{}",
                    nanos ^ (std::process::id() as u128) << 32,
                    self.next_id
                );
                self.next_id += 1;
                let evicted = self.make_room(open.bytes, 1, None);
                self.order.push_back(id.clone());
                let mut payload = open_payload(&id, &open);
                payload["reused"] = json!(false);
                payload["evicted_workbook_ids"] = json!(evicted);
                self.workbooks.insert(id, Arc::from(open));
                structured_result(payload, false)
            }
        }
    }

    /// Apply a job that finished after its call timed out: a reload
    /// still replaces its session, but a new workbook is dropped (the
    /// client never got its id).
    fn finish_late(&mut self, done: Done) {
        if !matches!(done, Done::Opened(_)) {
            let _ = self.finish(done);
        }
    }

    /// Evict the oldest sessions (never `keep`) until `sessions` more
    /// sessions and `bytes` more estimated memory fit. Returns the
    /// evicted ids.
    fn make_room(&mut self, bytes: u64, sessions: usize, keep: Option<&str>) -> Vec<String> {
        let mut evicted = Vec::new();
        loop {
            let held: u64 = self.workbooks.values().map(|w| w.bytes).sum();
            let over = self.workbooks.len() + sessions > MAX_WORKBOOKS
                || held.saturating_add(bytes) > self.options.memory_budget;
            if !over {
                break;
            }
            let Some(at) = self.order.iter().position(|id| Some(id.as_str()) != keep) else {
                break;
            };
            if let Some(old) = self.order.remove(at) {
                self.workbooks.remove(&old);
                evicted.push(old);
            }
        }
        evicted
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

/// What the stdio thread waits on: frames from the reader thread and
/// finished jobs from the workers.
enum Event {
    Frame(Vec<u8>),
    /// A frame over the 1 MiB cap (already discarded).
    Oversized,
    /// End of input, or the read error that ended it.
    Closed(Option<String>),
    Finished(u64, Done),
}

/// Cap on one request frame: an unbounded line would grow memory
/// until OOM before any parse error could be reported.
const MAX_LINE: usize = 1024 * 1024;

/// Read newline-framed requests from stdin and pass them on, until EOF
/// or a read error. Runs on its own thread so the stdio thread can
/// answer while it waits for workers.
fn read_frames(input: impl std::io::Read, events: mpsc::Sender<Event>) {
    let mut reader = std::io::BufReader::new(input);
    let mut raw: Vec<u8> = Vec::new();
    let mut discarding = false;
    loop {
        // Read in bounded 8 KiB chunks so memory stays capped even
        // before the 1 MiB frame check can fire.
        let (chunk_len, done) = match reader.fill_buf() {
            Err(e) => {
                let _ = events.send(Event::Closed(Some(e.to_string())));
                return;
            }
            Ok([]) => {
                // EOF: a final unterminated frame still counts.
                if !raw.is_empty() && !discarding {
                    let _ = events.send(Event::Frame(std::mem::take(&mut raw)));
                }
                let _ = events.send(Event::Closed(None));
                return;
            }
            Ok(available) => {
                let take_n = available
                    .iter()
                    .position(|b| *b == b'\n')
                    .map(|i| i + 1)
                    .unwrap_or(available.len())
                    .min(8192);
                let done = available[take_n - 1] == b'\n';
                if !discarding {
                    raw.extend_from_slice(&available[..take_n]);
                }
                (take_n, done)
            }
        };
        reader.consume(chunk_len);
        if !discarding && raw.len() > MAX_LINE {
            // One error per oversized frame, then discard through the
            // newline without parsing the tail.
            discarding = true;
            raw.clear();
            if events.send(Event::Oversized).is_err() {
                return;
            }
        }
        if done {
            if discarding {
                discarding = false;
            } else if events.send(Event::Frame(std::mem::take(&mut raw))).is_err() {
                return;
            }
        }
    }
}

fn write_frame(stdout: &mut dyn Write, frame: &Value) -> Result<(), String> {
    writeln!(stdout, "{frame}").map_err(|e| e.to_string())?;
    stdout.flush().map_err(|e| e.to_string())
}

/// Serve MCP on stdio until EOF. Logs go to stderr; stdout is
/// reserved for JSON-RPC frames, written only by this thread.
pub fn serve_stdio(options: McpOptions) -> Result<(), String> {
    let mut stdout = std::io::stdout();
    let roots: Vec<String> = options
        .roots
        .iter()
        .map(|r| r.display().to_string())
        .collect();
    eprintln!(
        "gridtalk mcp: serving on stdio ({} tools; roots: {}; call timeout {}s; memory budget {} MiB)",
        tool_definitions().as_array().map(|a| a.len()).unwrap_or(0),
        if roots.is_empty() {
            "any path".to_string()
        } else {
            roots.join(", ")
        },
        options.call_timeout.as_secs(),
        options.memory_budget >> 20
    );
    serve(McpServer::new(options), std::io::stdin(), &mut stdout)
}

/// The serve loop over any input and output (stdio in production).
fn serve(
    mut server: McpServer,
    input: impl std::io::Read + Send + 'static,
    stdout: &mut dyn Write,
) -> Result<(), String> {
    let timeout = server.options.call_timeout;
    let (events, inbox) = mpsc::channel::<Event>();
    {
        let events = events.clone();
        std::thread::Builder::new()
            .name("mcp-stdin".into())
            .spawn(move || read_frames(input, events))
            .map_err(|e| e.to_string())?;
    }
    // Calls awaiting an answer: ticket → (request id, deadline).
    let mut waiting: HashMap<u64, (Value, Instant)> = HashMap::new();
    // Worker threads alive, timed-out ones included.
    let mut running = 0usize;
    let mut next_ticket = 0u64;
    let mut closed: Option<Option<String>> = None;
    loop {
        if closed.is_some() && waiting.is_empty() {
            break;
        }
        let event = match waiting.values().map(|(_, deadline)| *deadline).min() {
            Some(deadline) => {
                inbox.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            }
            None => inbox
                .recv()
                .map_err(|_| mpsc::RecvTimeoutError::Disconnected),
        };
        match event {
            Ok(Event::Frame(raw)) => {
                let line = String::from_utf8_lossy(&raw);
                if line.trim().is_empty() {
                    continue;
                }
                let request: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(e) => {
                        let resp = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": format!("parse error: {e}")}});
                        write_frame(stdout, &resp)?;
                        continue;
                    }
                };
                match server.dispatch(&request) {
                    Step::Reply(Some(resp)) => write_frame(stdout, &resp)?,
                    Step::Reply(None) => {}
                    Step::Run { id, job: _ } if running >= MAX_RUNNING_CALLS => {
                        let busy = tool_error(
                            ErrorKind::Busy,
                            format!("{running} calls are still running (calls that timed out run until they finish); retry later"),
                        );
                        write_frame(stdout, &response(id, busy))?;
                    }
                    Step::Run { id, job } => {
                        let ticket = next_ticket;
                        next_ticket += 1;
                        let events = events.clone();
                        let spawned = std::thread::Builder::new()
                            .name(format!("mcp-call-{ticket}"))
                            .spawn(move || {
                                let _ = events.send(Event::Finished(ticket, run_job(job)));
                            });
                        match spawned {
                            Ok(_) => {
                                running += 1;
                                waiting.insert(ticket, (id, Instant::now() + timeout));
                            }
                            Err(e) => {
                                let failed = tool_error(
                                    ErrorKind::Internal,
                                    format!("could not start a worker thread: {e}"),
                                );
                                write_frame(stdout, &response(id, failed))?;
                            }
                        }
                    }
                }
            }
            Ok(Event::Oversized) => {
                let resp = json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32600, "message": "request line exceeds the 1 MiB cap"}});
                write_frame(stdout, &resp)?;
            }
            Ok(Event::Closed(error)) => closed = Some(error),
            Ok(Event::Finished(ticket, done)) => {
                running -= 1;
                match waiting.remove(&ticket) {
                    Some((id, _)) => write_frame(stdout, &response(id, server.finish(done)))?,
                    None => server.finish_late(done),
                }
            }
            // Expired calls are answered below, after every event.
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // The server holds a sender, so the channel never closes.
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
        // Sweep on every pass, not only when the inbox goes quiet: a
        // steady stream of frames would otherwise hold back timeouts.
        let now = Instant::now();
        let expired: Vec<u64> = waiting
            .iter()
            .filter(|(_, (_, deadline))| *deadline <= now)
            .map(|(ticket, _)| *ticket)
            .collect();
        for ticket in expired {
            if let Some((id, _)) = waiting.remove(&ticket) {
                let late = tool_error(
                    ErrorKind::Timeout,
                    format!(
                        "the call did not finish within {}s (gridtalk mcp --timeout); it runs on and its result will be dropped. Narrow the request, e.g. a smaller range",
                        timeout.as_secs()
                    ),
                );
                write_frame(stdout, &response(id, late))?;
            }
        }
    }
    match closed {
        Some(Some(error)) => Err(error),
        _ => Ok(()),
    }
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

    /// An initialized server with these options.
    fn started(options: McpOptions) -> McpServer {
        let mut server = McpServer::new(options);
        rpc(
            &mut server,
            json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}),
        );
        server
    }

    fn open_call(id: u64, path: &Path) -> Value {
        json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": "open_workbook", "arguments": {"path": path.to_str().unwrap()}}})
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gt-mcp-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Canonical, so the roots compare equal on macOS-like /tmp
        // symlinks and Windows verbatim paths.
        std::fs::canonicalize(&dir).unwrap()
    }

    #[test]
    fn roots_confine_open_workbook() {
        let dir = scratch("roots");
        let inside = dir.join("in");
        let outside = dir.join("out");
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(inside.join("a.csv"), "Item,Qty\nApples,2\n").unwrap();
        std::fs::write(outside.join("b.csv"), "Item,Qty\nPears,4\n").unwrap();
        let mut options = McpOptions::default();
        options.add_root(&inside).unwrap();
        assert!(options.add_root(&inside.join("a.csv")).is_err());
        let mut server = started(options);
        let ok = rpc(&mut server, open_call(1, &inside.join("a.csv")));
        assert_eq!(ok["result"]["isError"], false, "{ok}");
        // Outside the root, and a `..` that climbs out of it.
        for path in [
            outside.join("b.csv"),
            inside.join("..").join("out").join("b.csv"),
        ] {
            let denied = rpc(&mut server, open_call(2, &path));
            assert_eq!(denied["result"]["isError"], true);
            assert_eq!(
                denied["result"]["structuredContent"]["error"]["kind"], "path_not_allowed",
                "{denied}"
            );
        }
        // A missing file is reported as missing inside the root only.
        let missing = rpc(&mut server, open_call(4, &inside.join("nope.csv")));
        assert_eq!(
            missing["result"]["structuredContent"]["error"]["kind"],
            "open_failed"
        );
        let hidden = rpc(&mut server, open_call(5, &outside.join("nope.csv")));
        assert_eq!(
            hidden["result"]["structuredContent"]["error"]["kind"],
            "path_not_allowed"
        );
        // A symlink inside the root that points out of it.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.join("b.csv"), inside.join("link.csv")).unwrap();
            let denied = rpc(&mut server, open_call(3, &inside.join("link.csv")));
            assert_eq!(
                denied["result"]["structuredContent"]["error"]["kind"], "path_not_allowed",
                "{denied}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Without `--root` a missing file is missing (`open_failed`),
    /// even when its directory is missing too or it is a bare name.
    #[test]
    fn missing_paths_without_roots_are_open_failed() {
        let dir = scratch("noroots");
        let mut server = started(McpOptions::default());
        for path in [
            dir.join("no-such-dir").join("a.csv"),
            PathBuf::from("gt-no-such-file-for-mcp-test.csv"),
        ] {
            let missing = rpc(&mut server, open_call(1, &path));
            assert_eq!(
                missing["result"]["structuredContent"]["error"]["kind"], "open_failed",
                "{missing}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_budget_refuses_and_evicts() {
        let dir = scratch("budget");
        let a = dir.join("a.csv");
        let b = dir.join("b.csv");
        std::fs::write(&a, "Item,Qty\nApples,2\nPears,4\n").unwrap();
        std::fs::write(&b, "Item,Qty\nPlums,3\nFigs,5\n").unwrap();
        // A budget below one session refuses the open.
        let mut server = started(McpOptions {
            memory_budget: 1,
            ..McpOptions::default()
        });
        let refused = rpc(&mut server, open_call(1, &a));
        assert_eq!(
            refused["result"]["structuredContent"]["error"]["kind"], "over_budget",
            "{refused}"
        );
        // A budget of one session's estimate evicts the first for the
        // second.
        let one = OpenWorkbook::load(&a).unwrap().bytes;
        let other = OpenWorkbook::load(&b).unwrap().bytes;
        let mut server = started(McpOptions {
            memory_budget: one.max(other),
            ..McpOptions::default()
        });
        let first = rpc(&mut server, open_call(1, &a));
        let first_id = first["result"]["structuredContent"]["workbook_id"].clone();
        let second = rpc(&mut server, open_call(2, &b));
        assert_eq!(
            second["result"]["structuredContent"]["evicted_workbook_ids"],
            json!([first_id]),
            "{second}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A session whose file grows past the whole budget is closed with
    /// `over_budget` on its next call, and other sessions are kept.
    #[test]
    fn reload_over_budget_closes_the_session() {
        let dir = scratch("regrow");
        let a = dir.join("a.csv");
        let b = dir.join("b.csv");
        std::fs::write(&a, "Item,Qty\nApples,2\n").unwrap();
        std::fs::write(&b, "Item,Qty\nPlums,3\n").unwrap();
        let small = OpenWorkbook::load(&a).unwrap().bytes + OpenWorkbook::load(&b).unwrap().bytes;
        let mut server = started(McpOptions {
            memory_budget: small,
            ..McpOptions::default()
        });
        let id_a = rpc(&mut server, open_call(1, &a))["result"]["structuredContent"]["workbook_id"]
            .clone();
        let id_b = rpc(&mut server, open_call(2, &b))["result"]["structuredContent"]["workbook_id"]
            .clone();
        let rows: String = (0..200).map(|i| format!("Item{i},{i}\n")).collect();
        std::fs::write(&a, format!("Item,Qty\n{rows}")).unwrap();
        let call = |id: u64, wb: &Value| json!({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": "graph_overview", "arguments": {"workbook_id": wb}}});
        let grown = rpc(&mut server, call(3, &id_a));
        assert_eq!(
            grown["result"]["structuredContent"]["error"]["kind"], "over_budget",
            "{grown}"
        );
        let gone = rpc(&mut server, call(4, &id_a));
        assert_eq!(
            gone["result"]["structuredContent"]["error"]["kind"],
            "unknown_workbook"
        );
        let kept = rpc(&mut server, call(5, &id_b));
        assert_eq!(kept["result"]["isError"], false, "{kept}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Run the serve loop over `input` and return its response frames.
    fn serve_lines(server: McpServer, input: &str) -> Vec<Value> {
        let mut out: Vec<u8> = Vec::new();
        serve(
            server,
            std::io::Cursor::new(input.to_string().into_bytes()),
            &mut out,
        )
        .unwrap();
        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    #[test]
    fn serve_answers_while_a_call_runs_and_times_it_out() {
        let dir = scratch("timeout");
        let path = dir.join("a.csv");
        std::fs::write(&path, "Item,Qty\nApples,2\n").unwrap();
        let mut server = started(McpOptions {
            call_timeout: Duration::from_millis(100),
            ..McpOptions::default()
        });
        server.job_delay = Some(Duration::from_millis(600));
        let mut input = String::new();
        for id in 1..=5 {
            input.push_str(&format!("{}\n", open_call(id, &path)));
        }
        input.push_str("{\"jsonrpc\":\"2.0\",\"id\":9,\"method\":\"ping\"}\n");
        let frames = serve_lines(server, &input);
        let kind = |f: &Value| f["result"]["structuredContent"]["error"]["kind"].clone();
        // The fifth call finds every slot taken; ping is answered at
        // once; the four slow opens then time out, after both.
        assert_eq!(frames[0]["id"], 5);
        assert_eq!(kind(&frames[0]), "busy", "{}", frames[0]);
        assert_eq!(frames[1]["id"], 9);
        assert_eq!(frames[1]["result"], json!({}));
        let mut timed_out: Vec<u64> = frames[2..]
            .iter()
            .inspect(|f| assert_eq!(kind(f), "timeout", "{f}"))
            .map(|f| f["id"].as_u64().unwrap())
            .collect();
        timed_out.sort();
        assert_eq!(timed_out, vec![1, 2, 3, 4]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn serve_runs_calls_on_workers() {
        let dir = scratch("serve");
        let path = dir.join("a.csv");
        std::fs::write(&path, "Item,Qty\nApples,2\n").unwrap();
        // An open, then an unterminated last frame.
        let input = format!(
            "{}\n\n{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{{\"name\":\"close_workbook\",\"arguments\":{{\"workbook_id\":\"none\"}}}}}}",
            open_call(1, &path)
        );
        let frames = serve_lines(started(McpOptions::default()), &input);
        assert_eq!(frames.len(), 2, "{frames:?}");
        let opened = frames.iter().find(|f| f["id"] == 1).unwrap();
        assert_eq!(opened["result"]["isError"], false, "{opened}");
        let closed = frames.iter().find(|f| f["id"] == 2).unwrap();
        assert_eq!(
            closed["result"]["structuredContent"]["error"]["kind"],
            "unknown_workbook"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
