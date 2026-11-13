//! Inference backends. One trait, three local engine rungs plus an
//! OpenAI-compatible external client. No redaction lives here: the
//! external client sends the passage verbatim (see
//! docs/decisions/0002-no-redaction.md).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub first_token_ms: Option<f64>,
    pub total_ms: f64,
    /// Model load time, reported separately so generation timings
    /// (first token, tokens/sec) are not conflated with it.
    pub load_ms: f64,
    pub tokens_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub system: String,
    pub user: String,
    pub max_tokens: usize,
}

#[derive(Debug)]
pub enum InferError {
    Cancelled,
    EngineUnavailable(String),
    Http(String),
    Io(std::io::Error),
}

impl std::fmt::Display for InferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferError::Cancelled => write!(f, "cancelled"),
            InferError::EngineUnavailable(m) => write!(f, "engine unavailable: {m}"),
            InferError::Http(m) => write!(f, "http error: {m}"),
            InferError::Io(e) => write!(f, "io error: {e}"),
        }
    }
}
impl std::error::Error for InferError {}
impl From<std::io::Error> for InferError {
    fn from(e: std::io::Error) -> Self {
        InferError::Io(e)
    }
}

pub type CancelToken = Arc<AtomicBool>;

pub mod models;
pub mod probe;
pub mod transport;
pub mod turns;

pub use turns::{
    Message, Role, ToolCall, ToolChoice, ToolSession, ToolSpec, Turn, TurnRequest, Usage,
};

pub mod llama_backend;

pub trait ChatBackend: Send {
    fn name(&self) -> &str;
    /// Stream tokens through `on_token`. Implementations must poll
    /// `cancel` between tokens and stop promptly when it is set.
    fn stream(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(), InferError>;

    /// Whether the backend can take tool turns
    /// ([`ChatBackend::with_tool_session`]). The echo backend cannot:
    /// with no model there is nothing to choose a call.
    fn supports_tools(&self) -> bool {
        false
    }

    /// Open a conversation in which the model may call tools and hand
    /// it to `run`, which takes as many turns as it needs. The session
    /// ends when `run` returns, and `run`'s error is passed through.
    fn with_tool_session(
        &self,
        run: &mut dyn FnMut(&mut dyn ToolSession) -> Result<(), InferError>,
    ) -> Result<(), InferError> {
        let _ = run;
        Err(InferError::EngineUnavailable(format!(
            "the {} backend cannot call tools",
            self.name()
        )))
    }
}

/// Deterministic offline backend: echoes the cited context back in a
/// fixed template. Proves streaming + cancellation without a model,
/// and is what the test suite and first-run demo use.
pub struct EchoBackend;

impl ChatBackend for EchoBackend {
    fn name(&self) -> &str {
        "echo-local"
    }
    fn stream(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(), InferError> {
        let reply = format!(
            "Based on the workbook context:\n{}\n(end of context)",
            req.user
        );
        for word in reply.split_inclusive(' ') {
            if cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            on_token(word);
        }
        Ok(())
    }
}

/// Backend for an engine that was requested but is not compiled
/// into this build. Surfaces a typed error instead of silently
/// falling back to echo, so misconfiguration is visible.
pub struct UnavailableBackend {
    pub engine: String,
    pub reason: String,
}

impl ChatBackend for UnavailableBackend {
    fn name(&self) -> &str {
        "unavailable"
    }
    fn stream(
        &self,
        _req: &ChatRequest,
        _cancel: &CancelToken,
        _on_token: &mut dyn FnMut(&str),
    ) -> Result<(), InferError> {
        Err(InferError::EngineUnavailable(format!(
            "engine '{}' requested but not available: {}",
            self.engine, self.reason
        )))
    }
}

/// OpenAI-compatible chat-completions client (external backend and
/// future redacting-proxy endpoint). Sends the request verbatim over
/// `https` (certificate-verified TLS) or plain `http`. An API key is
/// only sent over cleartext http to a loopback host.
pub struct OpenAiCompatibleBackend {
    pub base_url: String, // e.g. http://127.0.0.1:11434/v1
    pub model: String,
    pub api_key: Option<String>,
    /// How many tokens a tool conversation may grow to (the
    /// config's `n_ctx`); `None` when the endpoint's window is not
    /// known.
    pub context_tokens: Option<usize>,
    /// How tool turns go over the wire (the config's `protocol`).
    pub protocol: Protocol,
}

/// How an OpenAI-compatible backend takes tool turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Protocol {
    /// Function definitions and `tool_calls` (the default).
    #[default]
    Native,
    /// The text protocol the local backend speaks (#49): the tools
    /// described in the system message, a call as one JSON object, a
    /// result as a user message, plain system/user/assistant messages
    /// and no `tools` field. A scripted endpoint then sees exactly
    /// what the local backend sends a model, less the chat template;
    /// it also lets the loop run against an endpoint without tool
    /// calling. Unlike the local backend, nothing holds the reply to
    /// the protocol's grammar.
    Text,
}

impl Protocol {
    /// The config's `protocol` value: `"text"`, or `"native"` (and
    /// when unset). Anything else is native, with a warning.
    pub fn from_config(value: Option<&str>) -> Protocol {
        match value.map(str::trim) {
            None | Some("") | Some("native") => Protocol::Native,
            Some("text") => Protocol::Text,
            Some(other) => {
                eprintln!(
                    "gridtalk: warning: unknown protocol {other:?}; expected \"native\" or \"text\" — using native"
                );
                Protocol::Native
            }
        }
    }
}

fn reject_crlf(field: &str, value: &str) -> Result<(), InferError> {
    if value.contains(['\r', '\n']) {
        return Err(InferError::Http(format!("{field} contains CR/LF")));
    }
    Ok(())
}

fn is_loopback_host(host_port: &str) -> bool {
    // Parse the host properly: a DNS name that merely starts with
    // "127." (e.g. 127.attacker.example) is NOT loopback.
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host_port
            .rsplit_once(':')
            .map(|(h, _)| h)
            .unwrap_or(host_port)
    };
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// The base URL the external backend uses when none is configured.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8080/v1";

/// A warning when the configured external backend talks plain `http`
/// to a host that is not loopback: spreadsheet passages, cell values
/// included, then cross the network unencrypted, whether or not an API
/// key is set (a key is refused outright there). `None` for the local
/// backend, `https` and loopback hosts.
pub fn cleartext_warning(cfg: &config::InferenceConfig) -> Option<String> {
    if cfg.backend != "openai-compatible" {
        return None;
    }
    cleartext_url_warning(cfg.base_url.as_deref().unwrap_or(DEFAULT_BASE_URL))
}

/// [`cleartext_warning`] for one base URL (the first-run wizard checks
/// the URL as it is typed). A URL that does not parse gets no warning:
/// the backend reports it as an error instead.
pub fn cleartext_url_warning(base_url: &str) -> Option<String> {
    let url = transport::parse_url(base_url).ok()?;
    (!url.tls && !is_loopback_host(&url.authority)).then(|| {
        format!(
            "{base_url} is plain http to a remote host: spreadsheet passages (cell values included) cross the network unencrypted. Use https, or a proxy on this machine."
        )
    })
}

/// True when `body` contains a complete chunked message: walks
/// real chunk boundaries (size lines at chunk offsets) and
/// accepts a terminal 0-chunk with optional trailers.
fn chunked_complete(body: &[u8]) -> bool {
    let mut rest = body;
    loop {
        let Some(line_end) = rest.windows(2).position(|w| w == b"\r\n") else {
            return false;
        };
        let size_str = String::from_utf8_lossy(&rest[..line_end]);
        let Ok(size) = usize::from_str_radix(size_str.split(';').next().unwrap_or("").trim(), 16)
        else {
            return false;
        };
        let after = &rest[line_end + 2..];
        if size == 0 {
            // Terminal chunk: complete once the trailer section
            // ends with an empty line.
            return after == b"\r\n" || after.windows(4).any(|w| w == b"\r\n\r\n");
        }
        let Some(need) = size.checked_add(2) else {
            return false;
        };
        if after.len() < need {
            return false;
        }
        rest = &after[need..];
    }
}

/// Transfer-Encoding contains a `chunked` token (any spacing/case,
/// possibly `gzip, chunked`).
fn is_chunked(head: &str) -> bool {
    head.lines().any(|l| {
        let Some((name, value)) = l.split_once(':') else {
            return false;
        };
        name.trim().eq_ignore_ascii_case("transfer-encoding")
            && value
                .split(',')
                .any(|tok| tok.trim().eq_ignore_ascii_case("chunked"))
    })
}

fn decode_chunked(body: &[u8]) -> Result<Vec<u8>, InferError> {
    let mut out = Vec::new();
    let mut rest = body;
    loop {
        let line_end = rest
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or_else(|| InferError::Http("bad chunked encoding (size line)".into()))?;
        let size_str = String::from_utf8_lossy(&rest[..line_end])
            .trim()
            .to_string();
        let size = usize::from_str_radix(size_str.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| InferError::Http(format!("bad chunk size {size_str:?}")))?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        let need = size
            .checked_add(2)
            .filter(|n| *n <= 16 * 1024 * 1024 + 2)
            .ok_or_else(|| InferError::Http(format!("chunk size {size} exceeds cap")))?;
        if rest.len() < need {
            return Err(InferError::Http("truncated chunk".into()));
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[need..];
        if out.len() > 16 * 1024 * 1024 {
            return Err(InferError::Http("response exceeds 16 MiB cap".into()));
        }
    }
}

/// Read an HTTP/1.1 response (status checked, chunked decoded,
/// 16 MiB cap) and return the body text.
fn read_http_response(
    stream: &mut transport::Conn,
    cancel: Option<&CancelToken>,
) -> Result<String, InferError> {
    use std::io::Read as _;
    // Short read timeout so cancellation is polled during the
    // blocking exchange instead of only before/after it.
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(2)))
        .ok();
    // Total deadline, and a stall budget that resets only on
    // >=4 KiB of new progress (a slow drip defeats neither).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    let mut last_progress = 0usize;
    let mut last_progress_at = std::time::Instant::now();
    let mut content_length: Option<usize> = None;
    let mut header_end: Option<usize> = None;
    let mut chunked = false;
    let mut header_search_from = 0usize;
    loop {
        if std::time::Instant::now() > deadline {
            return Err(InferError::Http(
                "endpoint did not complete the response within 300 seconds".into(),
            ));
        }
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                // Poll cancellation on EVERY read, and count a
                // stall budget that only resets on meaningful
                // progress toward a complete response — a slow
                // drip must not defeat either.
                if let Some(c) = cancel {
                    if c.load(Ordering::Relaxed) {
                        return Err(InferError::Cancelled);
                    }
                }
                raw.extend_from_slice(&buf[..n]);
                if raw.len() > 16 * 1024 * 1024 {
                    return Err(InferError::Http("response exceeds 16 MiB cap".into()));
                }
                // Time-based stall: no >=4 KiB of progress for
                // 10 s is a drip, however many reads it was spread
                // across (read counts are meaningless under Nagle /
                // buffering).
                if raw.len() - last_progress >= 4096 {
                    last_progress = raw.len();
                    last_progress_at = std::time::Instant::now();
                } else if (last_progress > 0 || header_end.is_some())
                    && last_progress_at.elapsed() > std::time::Duration::from_secs(30)
                {
                    // Only after first progress: a slow FIRST byte
                    // (a non-streaming endpoint thinking for 10-30 s)
                    // is governed by the WouldBlock arm and the 300 s
                    // deadline, not the drip check.
                    return Err(InferError::Http(
                        "endpoint is slow-dripping the response".into(),
                    ));
                }
                // Complete as soon as Content-Length bytes of body
                // have arrived; never wait for connection close.
                if header_end.is_none() {
                    // Incremental search: only new bytes + overlap.
                    let search_from = header_search_from.min(raw.len());
                    let found = raw[search_from..]
                        .windows(4)
                        .position(|w| w == b"\r\n\r\n")
                        .map(|rel| search_from + rel);
                    header_search_from = raw.len().saturating_sub(3);
                    if let Some(pos) = found {
                        header_end = Some(pos + 4);
                        // The stall window measures from the last
                        // arrival, not from connect.
                        last_progress_at = std::time::Instant::now();
                        last_progress = raw.len();
                        let head = String::from_utf8_lossy(&raw[..pos]);
                        content_length = head.lines().find_map(|l| {
                            let (name, value) = l.split_once(':')?;
                            if name.trim().eq_ignore_ascii_case("content-length") {
                                value.trim().parse().ok()
                            } else {
                                None
                            }
                        });
                        chunked = is_chunked(&head);
                    }
                }
                if let (Some(he), Some(cl)) = (header_end, content_length) {
                    if raw.len() - he >= cl {
                        break;
                    }
                }
                // Chunked bodies: a chunk-aligned terminal 0-chunk
                // (with optional trailers) completes in-loop.
                if let Some(he) = header_end {
                    if chunked && chunked_complete(&raw[he..]) {
                        break;
                    }
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                if let Some(c) = cancel {
                    if c.load(Ordering::Relaxed) {
                        return Err(InferError::Cancelled);
                    }
                }
                // Pre-first-byte waits are governed by the 300 s
                // deadline, not the stall budget.
                if (last_progress > 0 || header_end.is_some())
                    && last_progress_at.elapsed() > std::time::Duration::from_secs(30)
                {
                    return Err(InferError::Http(
                        "endpoint stalled for over 30 seconds".into(),
                    ));
                }
            }
            Err(e) => return Err(e.into()),
        }
    }
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| InferError::Http("malformed HTTP response".into()))?;
    let head = String::from_utf8_lossy(&raw[..split]).into_owned();
    let status: u32 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| InferError::Http("missing HTTP status".into()))?;
    if (300..400).contains(&status) {
        let loc = head
            .lines()
            .find_map(|l| {
                // Match the header name case-insensitively but keep
                // the value's original case.
                let (name, value) = l.split_once(':')?;
                if name.trim().eq_ignore_ascii_case("location") {
                    Some(value.trim().to_string())
                } else {
                    None
                }
            })
            .unwrap_or_default();
        return Err(InferError::Http(format!(
            "endpoint redirected (HTTP {status} -> {loc}); redirects are not followed, configure the final URL"
        )));
    }
    if !(200..300).contains(&status) {
        return Err(InferError::Http(format!(
            "endpoint returned HTTP {status}: {}",
            String::from_utf8_lossy(&raw[split + 4..])
                .chars()
                .take(200)
                .collect::<String>()
        )));
    }
    let body = &raw[split + 4..];
    let decoded = if is_chunked(&head) {
        decode_chunked(body)?
    } else {
        body.to_vec()
    };
    String::from_utf8(decoded).map_err(|e| InferError::Http(format!("response is not UTF-8: {e}")))
}

impl OpenAiCompatibleBackend {
    /// POST `body` to `chat/completions` and return the parsed JSON
    /// reply.
    fn post_chat(
        &self,
        body: &serde_json::Value,
        cancel: &CancelToken,
    ) -> Result<serde_json::Value, InferError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(InferError::Cancelled);
        }
        reject_crlf("base_url", &self.base_url)?;
        reject_crlf("model", &self.model)?;
        if let Some(k) = &self.api_key {
            reject_crlf("api_key", k)?;
        }
        let url = transport::parse_url(&self.base_url)
            .map_err(|e| InferError::Http(format!("base_url: {e}")))?;
        let host_port = url.authority.as_str();
        if self.api_key.is_some() && !url.tls && !is_loopback_host(host_port) {
            return Err(InferError::Http(format!(
                "refusing to send an API key over cleartext http to non-loopback host {host_port}; use https or a loopback redacting proxy"
            )));
        }
        let path_prefix = url.path.as_str();
        let path = format!("{}/chat/completions", path_prefix.trim_end_matches('/'));
        let body = body.to_string();
        let mut stream = transport::connect(
            &url,
            std::time::Duration::from_secs(5),
            std::time::Duration::from_secs(30),
        )?;
        let mut req_text = format!(
            "POST {path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
            body.len()
        );
        if let Some(k) = &self.api_key {
            req_text.push_str(&format!("Authorization: Bearer {k}\r\n"));
        }
        req_text.push_str("\r\n");
        stream.write_all(req_text.as_bytes())?;
        stream.write_all(body.as_bytes())?;
        let raw = read_http_response(&mut stream, Some(cancel))?;
        serde_json::from_str(&raw).map_err(|e| InferError::Http(format!("bad JSON response: {e}")))
    }
}

/// A reply message's `content`: a string, or an array of text parts
/// (some compatible servers). `None` when it is neither.
fn message_content(content: &serde_json::Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        Some(text.to_string())
    } else {
        content.as_array().map(|parts| {
            parts
                .iter()
                .filter_map(|part| part["text"].as_str())
                .collect()
        })
    }
}

impl ChatBackend for OpenAiCompatibleBackend {
    fn name(&self) -> &str {
        "openai-compatible"
    }
    fn stream(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(), InferError> {
        let body = serde_json::json!({
            "model": self.model,
            "stream": false,
            "max_tokens": req.max_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user}
            ]
        });
        // Note: stream:false, so tokens are replayed word-by-word
        // locally; cancellation is honoured between words.
        let value = self.post_chat(&body, cancel)?;
        let content = message_content(&value["choices"][0]["message"]["content"])
            .ok_or_else(|| InferError::Http("response has no choices[0].message.content".into()))?;
        for word in content.split_inclusive(' ') {
            if cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            on_token(word);
        }
        Ok(())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn with_tool_session(
        &self,
        run: &mut dyn FnMut(&mut dyn ToolSession) -> Result<(), InferError>,
    ) -> Result<(), InferError> {
        run(&mut OpenAiSession {
            backend: self,
            required_refused: false,
            calls: 0,
        })
    }
}

/// Tool turns over the OpenAI wire format: the tools go out as
/// function definitions and the calls come back in `tool_calls`.
/// Each turn sends the whole conversation, verbatim (decision 0002).
struct OpenAiSession<'a> {
    backend: &'a OpenAiCompatibleBackend,
    /// The endpoint refused `tool_choice: "required"`; later turns
    /// ask with `"auto"` and leave the check to the caller.
    required_refused: bool,
    /// Turns taken in the text protocol, for its calls' ids.
    calls: usize,
}

/// One message in the OpenAI wire format.
fn wire_message(message: &Message) -> serde_json::Value {
    match message.role {
        Role::Assistant if !message.tool_calls.is_empty() => serde_json::json!({
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls.iter().map(|call| serde_json::json!({
                "id": call.id, "type": "function",
                "function": {"name": call.name, "arguments": match &call.arguments {
                    // What the model wrote when it was not an object.
                    serde_json::Value::String(raw) => raw.clone(),
                    arguments => arguments.to_string(),
                }}
            })).collect::<Vec<_>>()
        }),
        Role::Tool => serde_json::json!({
            "role": "tool", "tool_call_id": message.tool_call_id, "content": message.content
        }),
        role => serde_json::json!({"role": role.as_str(), "content": message.content}),
    }
}

impl ToolSession for OpenAiSession<'_> {
    fn turn(
        &mut self,
        req: &TurnRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Turn, InferError> {
        if self.backend.protocol == Protocol::Text {
            return self.text_turn(req, cancel, on_token);
        }
        let body = |choice: &str| {
            let mut body = serde_json::json!({
                "model": self.backend.model,
                "stream": false,
                "max_tokens": req.max_tokens,
                "messages": req.messages.iter().map(wire_message).collect::<Vec<_>>(),
            });
            if !req.tools.is_empty() {
                body["tools"] = req
                    .tools
                    .iter()
                    .map(|tool| {
                        serde_json::json!({"type": "function", "function": {
                            "name": tool.name, "description": tool.description,
                            "parameters": tool.parameters
                        }})
                    })
                    .collect();
                body["tool_choice"] = choice.into();
            }
            body
        };
        let choice = match req.choice {
            ToolChoice::Required if !self.required_refused => "required",
            ToolChoice::None => "none",
            _ => "auto",
        };
        let value = match self.backend.post_chat(&body(choice), cancel) {
            // Not every compatible server takes "required": ask again
            // with "auto", and do so from here on.
            Err(InferError::Http(message))
                if choice == "required" && message.starts_with("endpoint returned HTTP 4") =>
            {
                // Remembered only once "auto" got through: a 4xx
                // for another reason (a bad key, a rate limit) fails
                // both ways and says nothing about "required".
                let value = self.backend.post_chat(&body("auto"), cancel)?;
                self.required_refused = true;
                value
            }
            other => other?,
        };
        let message = &value["choices"][0]["message"];
        if !message.is_object() {
            return Err(InferError::Http(
                "response has no choices[0].message".into(),
            ));
        }
        let calls: Vec<ToolCall> = message["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .enumerate()
                    .map(|(i, call)| {
                        let raw = &call["function"]["arguments"];
                        // The arguments arrive as a JSON string; a
                        // few servers send the object itself.
                        let arguments = match raw.as_str() {
                            Some(text) => match serde_json::from_str::<serde_json::Value>(text) {
                                Ok(parsed @ serde_json::Value::Object(_)) => parsed,
                                _ if text.trim().is_empty() => serde_json::json!({}),
                                _ => serde_json::Value::String(text.to_string()),
                            },
                            None if raw.is_object() => raw.clone(),
                            None => serde_json::json!({}),
                        };
                        ToolCall {
                            id: call["id"]
                                .as_str()
                                .map(String::from)
                                .unwrap_or_else(|| format!("call_{i}")),
                            name: call["function"]["name"].as_str().unwrap_or("").to_string(),
                            arguments,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        let text = message_content(&message["content"]).unwrap_or_default();
        let text = turns::strip_answer_mark(&text).to_string();
        if calls.is_empty() && text.is_empty() && value["choices"][0]["finish_reason"] == "length" {
            // A reasoning model can spend the whole turn thinking.
            return Err(InferError::Http(format!(
                "the model used all {} tokens of its turn before replying",
                req.max_tokens
            )));
        }
        if calls.is_empty() {
            for word in text.split_inclusive(' ') {
                if cancel.load(Ordering::Relaxed) {
                    return Err(InferError::Cancelled);
                }
                on_token(word);
            }
        }
        let count = |key: &str| value["usage"][key].as_u64().map(|n| n as usize);
        let usage = count("prompt_tokens").map(|prompt_tokens| Usage {
            prompt_tokens,
            reused_tokens: value["usage"]["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0) as usize,
            generated_tokens: count("completion_tokens").unwrap_or(0),
        });
        Ok(Turn { text, calls, usage })
    }

    fn context_tokens(&self) -> Option<usize> {
        self.backend.context_tokens
    }
}

impl OpenAiSession<'_> {
    /// A turn in the text protocol (see [`Protocol::Text`]): the
    /// messages the local backend would build, sent as plain chat
    /// messages, and the reply read as the local backend reads it.
    fn text_turn(
        &mut self,
        req: &TurnRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<Turn, InferError> {
        let messages: Vec<serde_json::Value> = turns::text_messages(req.messages, req.tools)
            .into_iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect();
        let body = serde_json::json!({
            "model": self.backend.model,
            "stream": false,
            "max_tokens": req.max_tokens,
            "messages": messages,
        });
        let value = self.backend.post_chat(&body, cancel)?;
        let message = &value["choices"][0]["message"];
        if !message.is_object() {
            return Err(InferError::Http(
                "response has no choices[0].message".into(),
            ));
        }
        let text = message_content(&message["content"]).unwrap_or_default();
        if text.trim().is_empty() && value["choices"][0]["finish_reason"] == "length" {
            return Err(InferError::Http(format!(
                "the model used all {} tokens of its turn before replying",
                req.max_tokens
            )));
        }
        self.calls += 1;
        let mut turn = turns::parse_text_turn(&text, &format!("call_{}", self.calls));
        if turn.calls.is_empty() {
            for word in turn.text.split_inclusive(' ') {
                if cancel.load(Ordering::Relaxed) {
                    return Err(InferError::Cancelled);
                }
                on_token(word);
            }
        }
        let count = |key: &str| value["usage"][key].as_u64().map(|n| n as usize);
        turn.usage = count("prompt_tokens").map(|prompt_tokens| Usage {
            prompt_tokens,
            reused_tokens: value["usage"]["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .unwrap_or(0) as usize,
            generated_tokens: count("completion_tokens").unwrap_or(0),
        });
        Ok(turn)
    }
}

/// Resolve a backend from settings. `local` with no compiled engine
/// falls back to EchoBackend so the app always runs; the UI surfaces
/// that no real model is loaded.
pub fn backend_from_config(cfg: &crate::config::InferenceConfig) -> Box<dyn ChatBackend + Send> {
    if !matches!(cfg.backend.as_str(), "local" | "openai-compatible" | "") {
        eprintln!(
            "gridtalk: warning: unknown backend {:?}; expected \"local\" or \"openai-compatible\" — falling back to local",
            cfg.backend
        );
    }
    match cfg.backend.as_str() {
        "openai-compatible" => Box::new(OpenAiCompatibleBackend {
            base_url: cfg
                .base_url
                .clone()
                .unwrap_or_else(|| DEFAULT_BASE_URL.into()),
            model: cfg.model.clone().unwrap_or_else(|| "default".into()),
            api_key: cfg
                .api_key_env
                .as_ref()
                .and_then(|env| std::env::var(env).ok()),
            context_tokens: cfg.n_ctx.map(|n| n as usize),
            protocol: Protocol::from_config(cfg.protocol.as_deref()),
        }),
        _ => {
            if !matches!(
                cfg.engine.as_str(),
                "llamacpp" | "llama-cpp" | "llama.cpp" | ""
            ) {
                eprintln!(
                    "gridtalk: note: engine = {:?} is ignored; the only local engine is llama.cpp (Vulkan)",
                    cfg.engine
                );
            }
            match crate::llama_backend::llama_backend_from_config(cfg) {
                Some(backend) if backend.model_path.exists() => Box::new(backend),
                Some(backend) => {
                    eprintln!(
                        "gridtalk: warning: configured model {} does not exist; using the echo backend until a real model is configured",
                        backend.model_path.display()
                    );
                    Box::new(EchoBackend)
                }
                _ => Box::new(EchoBackend),
            }
        }
    }
}

pub mod config {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InferenceConfig {
        /// "local" (default) or "openai-compatible"
        #[serde(default = "default_backend")]
        pub backend: String,
        /// "candle" | "mistralrs" | "llamacpp"
        #[serde(default = "default_engine")]
        pub engine: String,
        #[serde(default)]
        pub model: Option<String>,
        #[serde(default)]
        pub model_path: Option<String>,
        #[serde(default)]
        pub base_url: Option<String>,
        /// Name of the env var holding the API key. The key itself is
        /// never persisted.
        #[serde(default)]
        pub api_key_env: Option<String>,
        /// Local-backend sampling/layer settings (llama.cpp only).
        #[serde(default)]
        pub temperature: Option<f32>,
        #[serde(default)]
        pub seed: Option<u32>,
        /// Layers to offload to the GPU; unset = all of them.
        #[serde(default)]
        pub n_gpu_layers: Option<u32>,
        /// Upper bound on the context window in tokens; unset = the
        /// model's own training length (capped at 64K).
        #[serde(default)]
        pub n_ctx: Option<u32>,
        /// OpenAI-compatible backend: how tool turns go over the
        /// wire, `"native"` (function definitions, the default) or
        /// `"text"` (the local backend's text protocol; #49).
        #[serde(default)]
        pub protocol: Option<String>,
    }

    fn default_backend() -> String {
        "local".into()
    }
    fn default_engine() -> String {
        "llamacpp".into()
    }

    impl Default for InferenceConfig {
        fn default() -> Self {
            InferenceConfig {
                backend: default_backend(),
                engine: default_engine(),
                model: None,
                model_path: None,
                base_url: None,
                api_key_env: None,
                temperature: None,
                seed: None,
                n_gpu_layers: None,
                n_ctx: None,
                protocol: None,
            }
        }
    }

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct Config {
        #[serde(default)]
        pub inference: InferenceConfig,
    }

    impl Config {
        /// Parse with a real TOML parser. Malformed config warns on
        /// stderr and yields defaults rather than silently
        /// misparsing individual keys.
        pub fn parse(text: &str) -> Config {
            match toml::from_str::<Config>(text) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("gridtalk: warning: malformed config ({e}); using defaults");
                    Config::default()
                }
            }
        }
        pub fn load(path: &std::path::Path) -> Config {
            match std::fs::read_to_string(path) {
                Ok(text) => Config::parse(&text),
                Err(e) => {
                    if path.exists() {
                        eprintln!(
                            "gridtalk: warning: cannot read config {} ({e}); using defaults",
                            path.display()
                        );
                    }
                    Config::default()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn echo_streams_and_cancels() {
        let req = ChatRequest {
            system: "s".into(),
            user: "hello".into(),
            max_tokens: 8,
        };
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        let mut out = String::new();
        EchoBackend
            .stream(&req, &cancel, &mut |t| out.push_str(t))
            .unwrap();
        assert!(out.contains("hello"));
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(
            EchoBackend.stream(&req, &cancel, &mut |_| {}),
            Err(InferError::Cancelled)
        ));
    }

    #[test]
    fn cleartext_warning_covers_remote_http_only() {
        let cfg = |backend: &str, url: Option<&str>| config::InferenceConfig {
            backend: backend.into(),
            base_url: url.map(str::to_string),
            ..Default::default()
        };
        assert!(
            cleartext_warning(&cfg("openai-compatible", Some("http://203.0.113.9/v1")))
                .is_some_and(|w| w.contains("unencrypted"))
        );
        assert!(
            cleartext_warning(&cfg("openai-compatible", Some("https://203.0.113.9/v1"))).is_none()
        );
        assert!(
            cleartext_warning(&cfg("openai-compatible", Some("http://localhost:11434/v1")))
                .is_none()
        );
        assert!(
            cleartext_warning(&cfg("openai-compatible", Some("http://[::1]:8080/v1"))).is_none()
        );
        assert!(cleartext_warning(&cfg("openai-compatible", None)).is_none());
        assert!(cleartext_warning(&cfg("local", Some("http://203.0.113.9/v1"))).is_none());
        assert!(cleartext_url_warning("http://127.attacker.example/v1").is_some());
        assert!(cleartext_url_warning("not a url").is_none());
    }

    #[test]
    fn loopback_and_chunk_extensions() {
        assert!(is_loopback_host("127.0.0.1:8080"));
        assert!(is_loopback_host("127.5.6.7:8080"));
        assert!(is_loopback_host("localhost:8080"));
        assert!(is_loopback_host("[::1]:8080"));
        assert!(!is_loopback_host("127.attacker.example:8080"));
        assert!(!is_loopback_host("example.com:8080"));
        assert_eq!(
            decode_chunked(b"2; ext=val\r\nhi\r\n0\r\n\r\n").unwrap(),
            b"hi".to_vec()
        );
    }

    #[test]
    fn config_parses_inference_table() {
        let cfg = config::Config::parse("[inference]\nbackend = \"openai-compatible\"\nengine = \"mistralrs\"\napi_key_env = \"GRIDTALK_KEY\"\n");
        assert_eq!(cfg.inference.backend, "openai-compatible");
        assert_eq!(cfg.inference.engine, "mistralrs");
        assert_eq!(cfg.inference.api_key_env.as_deref(), Some("GRIDTALK_KEY"));
    }

    #[test]
    fn external_backend_sends_passage_verbatim() {
        // Mock OpenAI-compatible server: captures the request body and
        // replies with a canned completion. Guards decision 0002 —
        // no redaction may creep into the external path.
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            // Read headers, then exactly Content-Length body bytes.
            let body_len = loop {
                let n = sock.read(&mut buf).unwrap();
                assert!(n > 0);
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                if let Some(pos) = text.find("\r\n\r\n") {
                    let len: usize = text[..pos]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse().ok())
                        })
                        .unwrap_or(0);
                    if raw.len() >= pos + 4 + len {
                        break len;
                    }
                }
            };
            let _ = body_len;
            let text = String::from_utf8_lossy(&raw).to_string();
            let body = r#"{"choices":[{"message":{"content":"answer citing Sheet1!A1"}}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            sock.write_all(resp.as_bytes()).unwrap();
            text
        });
        let b = OpenAiCompatibleBackend {
            base_url: format!("http://127.0.0.1:{port}/v1"),
            model: "m".into(),
            api_key: None,
            context_tokens: None,
            protocol: Protocol::Native,
        };
        let req = ChatRequest {
            system: "s".into(),
            user: "SecretCellValue123".into(),
            max_tokens: 4,
        };
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        let mut out = String::new();
        b.stream(&req, &cancel, &mut |t| out.push_str(t)).unwrap();
        let sent = server.join().unwrap();
        assert!(
            sent.contains("SecretCellValue123"),
            "passage must be sent verbatim"
        );
        assert!(out.contains("Sheet1!A1"));
    }

    /// A mock chat-completions server: replies to each connection in
    /// turn with the next (status, body) and returns the request
    /// bodies it was sent.
    fn mock_chat_server(
        replies: Vec<(u16, String)>,
    ) -> (u16, std::thread::JoinHandle<Vec<serde_json::Value>>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let mut bodies = Vec::new();
            for (status, reply) in replies {
                let (mut sock, _) = listener.accept().unwrap();
                let mut raw = Vec::new();
                let mut buf = [0u8; 4096];
                let body = loop {
                    let n = sock.read(&mut buf).unwrap();
                    assert!(n > 0);
                    raw.extend_from_slice(&buf[..n]);
                    let Some(pos) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
                        continue;
                    };
                    let head = String::from_utf8_lossy(&raw[..pos]).to_ascii_lowercase();
                    let len: usize = head
                        .lines()
                        .find_map(|l| l.strip_prefix("content-length:")?.trim().parse().ok())
                        .unwrap_or(0);
                    if raw.len() >= pos + 4 + len {
                        break raw[pos + 4..pos + 4 + len].to_vec();
                    }
                };
                bodies.push(serde_json::from_slice(&body).unwrap());
                let resp = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{reply}",
                    reply.len()
                );
                sock.write_all(resp.as_bytes()).unwrap();
            }
            bodies
        });
        (port, server)
    }

    /// `protocol = "text"`: an OpenAI-compatible endpoint is sent the
    /// local backend's text protocol, the messages `text_messages`
    /// builds and no `tools` field, and its replies are read as the
    /// local backend reads them (#49).
    #[test]
    fn external_backend_can_speak_the_text_protocol() {
        let reply = |content: &str| {
            serde_json::json!({"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 7}})
            .to_string()
        };
        let (port, server) = mock_chat_server(vec![
            (
                200,
                reply("{\"tool\":\"get_cells\",\"range\":\"Sheet1!A1:B2\"}"),
            ),
            (200, reply("ANSWER: It is 6 (Sheet1!B2).")),
        ]);
        let cfg = config::Config::parse(&format!(
            "[inference]\nbackend = \"openai-compatible\"\nbase_url = \"http://127.0.0.1:{port}/v1\"\nmodel = \"m\"\nprotocol = \"text\"\n"
        ));
        assert_eq!(cfg.inference.protocol.as_deref(), Some("text"));
        let backend = backend_from_config(&cfg.inference);
        let tools = vec![ToolSpec {
            name: "get_cells".into(),
            description: "Read cells.".into(),
            parameters: serde_json::json!({
                "type": "object", "properties": {"range": {"type": "string"}}, "required": ["range"]
            }),
        }];
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        let mut streamed = String::new();
        let mut sent_messages = Vec::new();
        backend
            .with_tool_session(&mut |session| {
                let mut messages = vec![
                    Message::system("Answer from the workbook."),
                    Message::user("What is B2?"),
                ];
                let first = session.turn(
                    &TurnRequest {
                        messages: &messages,
                        tools: &tools,
                        choice: ToolChoice::Required,
                        max_tokens: 64,
                    },
                    &cancel,
                    &mut |t| streamed.push_str(t),
                )?;
                assert_eq!(
                    first.calls,
                    vec![ToolCall {
                        id: "call_1".into(),
                        name: "get_cells".into(),
                        arguments: serde_json::json!({"range": "Sheet1!A1:B2"}),
                    }]
                );
                assert_eq!(first.usage.map(|u| u.prompt_tokens), Some(50));
                sent_messages.push(turns::text_messages(&messages, &tools));
                messages.push(Message::assistant("", first.calls.clone()));
                messages.push(Message::tool(&first.calls[0], "Sheet1!B2: 6"));
                let second = session.turn(
                    &TurnRequest {
                        messages: &messages,
                        tools: &tools,
                        choice: ToolChoice::Auto,
                        max_tokens: 64,
                    },
                    &cancel,
                    &mut |t| streamed.push_str(t),
                )?;
                assert_eq!(second.text, "It is 6 (Sheet1!B2).");
                assert!(second.calls.is_empty());
                sent_messages.push(turns::text_messages(&messages, &tools));
                Ok(())
            })
            .unwrap();
        assert_eq!(streamed, "It is 6 (Sheet1!B2).");
        let bodies = server.join().unwrap();
        for (body, expected) in bodies.iter().zip(&sent_messages) {
            assert!(body.get("tools").is_none(), "{body}");
            assert!(body.get("tool_choice").is_none(), "{body}");
            let expected: Vec<serde_json::Value> = expected
                .iter()
                .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
                .collect();
            assert_eq!(body["messages"], serde_json::Value::from(expected));
        }
        // The second request carries the call as the assistant's JSON
        // and the result as a user message, as the local backend
        // builds them.
        let roles: Vec<&str> = bodies[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "user"]);
        assert!(bodies[0]["messages"][0]["content"]
            .as_str()
            .unwrap()
            .contains("get_cells"));
        assert_eq!(Protocol::from_config(Some("native")), Protocol::Native);
        assert_eq!(Protocol::from_config(None), Protocol::Native);
        assert_eq!(Protocol::from_config(Some("bogus")), Protocol::Native);
    }

    #[test]
    fn external_backend_takes_tool_turns_on_the_openai_wire_format() {
        let call_reply = serde_json::json!({"choices": [{"message": {
            "content": null,
            "tool_calls": [{"id": "abc", "type": "function", "function": {
                "name": "get_cells", "arguments": "{\"range\":\"Sheet1!A1:B2\"}"
            }}]
        }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 9,
                      "prompt_tokens_details": {"cached_tokens": 100}}})
        .to_string();
        let answer_reply = serde_json::json!({"choices": [{"message": {
            "content": "ANSWER: It is 6 (Sheet1!B2)."
        }}]})
        .to_string();
        let (port, server) = mock_chat_server(vec![
            // The endpoint refuses "required"; the turn is asked again.
            (
                400,
                "{\"error\":\"tool_choice required is not supported\"}".into(),
            ),
            (200, call_reply),
            (200, answer_reply),
            (
                200,
                "{\"choices\":[{\"message\":{\"content\":\"\"},\"finish_reason\":\"length\"}]}"
                    .into(),
            ),
        ]);
        let backend = OpenAiCompatibleBackend {
            base_url: format!("http://127.0.0.1:{port}/v1"),
            model: "m".into(),
            api_key: None,
            context_tokens: Some(8_192),
            protocol: Protocol::Native,
        };
        assert!(backend.supports_tools());
        let tools = vec![ToolSpec {
            name: "get_cells".into(),
            description: "Read cells.".into(),
            parameters: serde_json::json!({
                "type": "object", "properties": {"range": {"type": "string"}}, "required": ["range"]
            }),
        }];
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        let mut streamed = String::new();
        backend
            .with_tool_session(&mut |session| {
                assert_eq!(session.context_tokens(), Some(8_192));
                let mut messages = vec![
                    Message::system("Answer from the workbook."),
                    Message::user("What is B2? SecretCellValue123"),
                ];
                let first = session.turn(
                    &TurnRequest {
                        messages: &messages,
                        tools: &tools,
                        choice: ToolChoice::Required,
                        max_tokens: 64,
                    },
                    &cancel,
                    &mut |t| streamed.push_str(t),
                )?;
                assert_eq!(first.text, "");
                assert_eq!(
                    first.calls,
                    vec![ToolCall {
                        id: "abc".into(),
                        name: "get_cells".into(),
                        arguments: serde_json::json!({"range": "Sheet1!A1:B2"}),
                    }]
                );
                assert_eq!(
                    first.usage,
                    Some(Usage {
                        prompt_tokens: 120,
                        reused_tokens: 100,
                        generated_tokens: 9
                    })
                );
                // A call streams nothing.
                assert!(streamed.is_empty());
                messages.push(Message::assistant("", first.calls.clone()));
                messages.push(Message::tool(&first.calls[0], "Sheet1!B2: 6"));
                // "required" was refused once, so it is not sent again.
                let second = session.turn(
                    &TurnRequest {
                        messages: &messages,
                        tools: &tools,
                        choice: ToolChoice::Required,
                        max_tokens: 64,
                    },
                    &cancel,
                    &mut |t| streamed.push_str(t),
                )?;
                assert!(second.calls.is_empty());
                assert_eq!(second.text, "It is 6 (Sheet1!B2).");
                assert_eq!(streamed, "It is 6 (Sheet1!B2).");
                // A turn spent before any reply is an error, not an
                // empty answer.
                let spent = session.turn(
                    &TurnRequest {
                        messages: &messages,
                        tools: &tools,
                        choice: ToolChoice::None,
                        max_tokens: 64,
                    },
                    &cancel,
                    &mut |_| {},
                );
                assert!(
                    matches!(&spent, Err(InferError::Http(m)) if m.contains("all 64 tokens")),
                    "{spent:?}"
                );
                Ok(())
            })
            .unwrap();
        let sent = server.join().unwrap();
        assert_eq!(sent.len(), 4);
        assert_eq!(sent[0]["tool_choice"], "required");
        assert_eq!(sent[1]["tool_choice"], "auto");
        assert_eq!(sent[2]["tool_choice"], "auto");
        assert_eq!(sent[3]["tool_choice"], "none");
        assert_eq!(
            sent[0]["tools"],
            serde_json::json!([{"type": "function", "function": {
                "name": "get_cells", "description": "Read cells.",
                "parameters": {"type": "object", "properties": {"range": {"type": "string"}}, "required": ["range"]}
            }}])
        );
        // The conversation goes out verbatim (decision 0002), with the
        // call and its result in the wire format's own shape.
        assert_eq!(
            sent[0]["messages"][1]["content"],
            "What is B2? SecretCellValue123"
        );
        assert_eq!(
            sent[2]["messages"][2],
            serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{
                "id": "abc", "type": "function",
                "function": {"name": "get_cells", "arguments": "{\"range\":\"Sheet1!A1:B2\"}"}
            }]})
        );
        assert_eq!(
            sent[2]["messages"][3],
            serde_json::json!({"role": "tool", "tool_call_id": "abc", "content": "Sheet1!B2: 6"})
        );
        // The echo backend takes no tool turns.
        assert!(!EchoBackend.supports_tools());
        assert!(EchoBackend.with_tool_session(&mut |_| Ok(())).is_err());
    }

    #[test]
    fn regression_http_status_chunked_and_key_guards() {
        use std::io::{Read, Write};
        // Chunked + non-200 handling against a mock server.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            // Read the full request (headers + Content-Length body)
            // before responding, or the client sees a reset.
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = sock.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&raw).to_string();
                if let Some(pos) = text.find("\r\n\r\n") {
                    let len: usize = text[..pos]
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse().ok())
                        })
                        .unwrap_or(0);
                    if raw.len() >= pos + 4 + len {
                        break;
                    }
                }
            }
            let body = "{\"choices\":[{\"message\":{\"content\":\"chunked ok\"}}]}";
            let half = body.len() / 2;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n{:x}\r\n{}\r\n0\r\n\r\n",
                half,
                &body[..half],
                body.len() - half,
                &body[half..]
            );
            sock.write_all(resp.as_bytes()).unwrap();
        });
        let b = OpenAiCompatibleBackend {
            base_url: format!("http://127.0.0.1:{port}/v1"),
            model: "m".into(),
            api_key: None,
            context_tokens: None,
            protocol: Protocol::Native,
        };
        let req = ChatRequest {
            system: "s".into(),
            user: "u".into(),
            max_tokens: 4,
        };
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        let mut out = String::new();
        b.stream(&req, &cancel, &mut |t| out.push_str(t)).unwrap();
        assert!(out.contains("chunked ok"));
        server.join().unwrap();
        // API key must never go over cleartext http to a remote host.
        let remote = OpenAiCompatibleBackend {
            base_url: "http://203.0.113.9/v1".into(),
            model: "m".into(),
            api_key: Some("secret".into()),
            context_tokens: None,
            protocol: Protocol::Native,
        };
        assert!(matches!(
            remote.stream(&req, &cancel, &mut |_| {}),
            Err(InferError::Http(_))
        ));
        // CRLF in config fields is rejected, not injected.
        let bad = OpenAiCompatibleBackend {
            base_url: "http://127.0.0.1:9/v1\r\nX-Injected: 1".into(),
            model: "m".into(),
            api_key: None,
            context_tokens: None,
            protocol: Protocol::Native,
        };
        assert!(bad.stream(&req, &cancel, &mut |_| {}).is_err());
    }

    #[test]
    fn external_backend_speaks_https_and_sends_key_over_tls() {
        let body = r#"{"choices":[{"message":{"content":"over tls"}}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let (port, cert, requests) =
            transport::test_support::serve_tls(vec![response.into_bytes()]);
        transport::TEST_TLS_CONFIG.with(|c| {
            *c.borrow_mut() = Some(transport::test_support::client_config_trusting(&cert))
        });
        // A key is fine over TLS, even though the check for
        // cleartext http would refuse a non-loopback host.
        let b = OpenAiCompatibleBackend {
            base_url: format!("https://localhost:{port}/v1"),
            model: "m".into(),
            api_key: Some("secret".into()),
            context_tokens: None,
            protocol: Protocol::Native,
        };
        let req = ChatRequest {
            system: "s".into(),
            user: "u".into(),
            max_tokens: 1,
        };
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        let mut out = String::new();
        b.stream(&req, &cancel, &mut |t| out.push_str(t)).unwrap();
        transport::TEST_TLS_CONFIG.with(|c| *c.borrow_mut() = None);
        assert_eq!(out, "over tls");
        let sent = requests.recv().unwrap();
        assert!(
            sent.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "{sent}"
        );
        assert!(sent.contains("Authorization: Bearer secret"));
    }
}
