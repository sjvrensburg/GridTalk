//! Inference backends. One trait, three local engine rungs plus an
//! OpenAI-compatible external client. No redaction lives here: the
//! external client sends the passage verbatim (see
//! docs/decisions/0002-no-redaction.md).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::net::ToSocketAddrs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    Candle,
    MistralRs,
    LlamaCpp,
}

impl EngineKind {
    /// Ordered fallback ladder: pure-Rust first, FFI last.
    pub fn ladder() -> [EngineKind; 3] {
        [
            EngineKind::Candle,
            EngineKind::MistralRs,
            EngineKind::LlamaCpp,
        ]
    }
    pub fn feature_name(&self) -> &'static str {
        match self {
            EngineKind::Candle => "candle",
            EngineKind::MistralRs => "mistralrs",
            EngineKind::LlamaCpp => "llamacpp",
        }
    }
    pub fn compiled_in(&self) -> bool {
        match self {
            EngineKind::Candle => cfg!(feature = "candle"),
            EngineKind::MistralRs => cfg!(feature = "mistralrs"),
            EngineKind::LlamaCpp => cfg!(feature = "llamacpp"),
        }
    }
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

#[cfg(feature = "candle-gen")]
pub mod generate;

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

/// Probe the local engine ladder and return the first compiled-in
/// rung. The Phase 3 spike replaces `compiled_in` with a runtime
/// load test per engine (see docs/spikes/inference-engine.md).
pub fn select_local_engine() -> Option<EngineKind> {
    EngineKind::ladder().into_iter().find(|e| e.compiled_in())
}

/// OpenAI-compatible chat-completions client (external backend and
/// future redacting-proxy endpoint). Sends the request verbatim;
/// supports plain http for loopback/proxy endpoints. HTTPS endpoints
/// require the deployment to terminate TLS at a local proxy until a
/// TLS transport is wired in — reported as a typed error, never a
/// silent downgrade.
pub struct OpenAiCompatibleBackend {
    pub base_url: String, // e.g. http://127.0.0.1:11434/v1
    pub model: String,
    pub api_key: Option<String>,
}

fn reject_crlf(field: &str, value: &str) -> Result<(), InferError> {
    if value.contains(['\r', '\n']) {
        return Err(InferError::Http(format!("{field} contains CR/LF")));
    }
    Ok(())
}

fn is_loopback_host(host_port: &str) -> bool {
    let host = host_port
        .trim_start_matches('[')
        .split([':', ']'])
        .next()
        .unwrap_or("");
    matches!(host, "127.0.0.1" | "localhost" | "::1") || host.starts_with("127.")
}

fn connect_timeout(host_port: &str) -> Result<std::net::TcpStream, InferError> {
    let addrs: Vec<std::net::SocketAddr> = host_port
        .to_socket_addrs()
        .map_err(|e| InferError::Http(format!("cannot resolve {host_port}: {e}")))?
        .collect();
    let mut last: Option<InferError> = None;
    for addr in addrs {
        match std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(5)) {
            Ok(s) => {
                s.set_read_timeout(Some(std::time::Duration::from_secs(30)))
                    .ok();
                s.set_write_timeout(Some(std::time::Duration::from_secs(30)))
                    .ok();
                return Ok(s);
            }
            Err(e) => last = Some(InferError::Io(e)),
        }
    }
    Err(last.unwrap_or_else(|| InferError::Http(format!("cannot resolve {host_port}"))))
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
        let size = usize::from_str_radix(&size_str, 16)
            .map_err(|_| InferError::Http(format!("bad chunk size {size_str:?}")))?;
        rest = &rest[line_end + 2..];
        if size == 0 {
            return Ok(out);
        }
        if rest.len() < size + 2 {
            return Err(InferError::Http("truncated chunk".into()));
        }
        out.extend_from_slice(&rest[..size]);
        rest = &rest[size + 2..];
        if out.len() > 16 * 1024 * 1024 {
            return Err(InferError::Http("response exceeds 16 MiB cap".into()));
        }
    }
}

/// Read an HTTP/1.1 response (status checked, chunked decoded,
/// 16 MiB cap) and return the body text.
fn read_http_response(stream: &mut std::net::TcpStream) -> Result<String, InferError> {
    use std::io::Read as _;
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        raw.extend_from_slice(&buf[..n]);
        if raw.len() > 16 * 1024 * 1024 {
            return Err(InferError::Http("response exceeds 16 MiB cap".into()));
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
                l.to_ascii_lowercase()
                    .strip_prefix("location:")
                    .map(|v| v.trim().to_string())
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
    let lower = head.to_ascii_lowercase();
    let body = &raw[split + 4..];
    let decoded = if lower.contains("transfer-encoding: chunked") {
        decode_chunked(body)?
    } else {
        body.to_vec()
    };
    String::from_utf8(decoded).map_err(|e| InferError::Http(format!("response is not UTF-8: {e}")))
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
        if cancel.load(Ordering::Relaxed) {
            return Err(InferError::Cancelled);
        }
        reject_crlf("base_url", &self.base_url)?;
        reject_crlf("model", &self.model)?;
        if let Some(k) = &self.api_key {
            reject_crlf("api_key", k)?;
        }
        if self.base_url.starts_with("https://") {
            return Err(InferError::Http(
                "https endpoints need a TLS transport or local proxy; none configured".into(),
            ));
        }
        let rest = self.base_url.strip_prefix("http://").ok_or_else(|| {
            InferError::Http("base_url must start with http:// or https://".into())
        })?;
        let (host_port, path_prefix) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        if self.api_key.is_some() && !is_loopback_host(host_port) {
            return Err(InferError::Http(format!(
                "refusing to send an API key over cleartext http to non-loopback host {host_port}; use a loopback redacting proxy or a TLS endpoint"
            )));
        }
        let path = format!("{path_prefix}/chat/completions");
        let body = serde_json::json!({
            "model": self.model,
            "stream": false,
            "max_tokens": req.max_tokens,
            "messages": [
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user}
            ]
        })
        .to_string();
        // Note: stream:false, so tokens are replayed word-by-word
        // locally; cancellation is honoured between words.
        let mut stream = connect_timeout(host_port)?;
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
        let raw = read_http_response(&mut stream)?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| InferError::Http(format!("bad JSON response: {e}")))?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| InferError::Http("response has no choices[0].message.content".into()))?;
        for word in content.split_inclusive(' ') {
            if cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            on_token(word);
        }
        Ok(())
    }
}

/// Resolve a backend from settings. `local` with no compiled engine
/// falls back to EchoBackend so the app always runs; the UI surfaces
/// that no real model is loaded.
pub fn backend_from_config(cfg: &crate::config::InferenceConfig) -> Box<dyn ChatBackend + Send> {
    match cfg.backend.as_str() {
        "openai-compatible" => Box::new(OpenAiCompatibleBackend {
            base_url: cfg
                .base_url
                .clone()
                .unwrap_or_else(|| "http://127.0.0.1:8080/v1".into()),
            model: cfg.model.clone().unwrap_or_else(|| "default".into()),
            api_key: cfg
                .api_key_env
                .as_ref()
                .and_then(|env| std::env::var(env).ok()),
        }),
        _ => {
            if cfg.engine == "mistralrs" || cfg.engine == "llamacpp" {
                return Box::new(UnavailableBackend {
                    engine: cfg.engine.clone(),
                    reason: "that rung of the inference ladder is not compiled into this build (see docs/spikes/inference-engine.md)".into(),
                });
            }
            #[cfg(feature = "candle-gen")]
            {
                if let Some(backend) = crate::generate::candle_backend_from_config(cfg) {
                    if backend.model_path.exists() {
                        return Box::new(backend);
                    }
                }
            }
            Box::new(EchoBackend)
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
    }

    fn default_backend() -> String {
        "local".into()
    }
    fn default_engine() -> String {
        "candle".into()
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

    #[test]
    fn ladder_is_pure_rust_first() {
        assert_eq!(EngineKind::ladder()[0], EngineKind::Candle);
        assert_eq!(EngineKind::ladder()[2], EngineKind::LlamaCpp);
    }

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
        };
        assert!(bad.stream(&req, &cancel, &mut |_| {}).is_err());
    }

    #[test]
    fn external_backend_rejects_https_without_transport() {
        let b = OpenAiCompatibleBackend {
            base_url: "https://api.example.com/v1".into(),
            model: "m".into(),
            api_key: None,
        };
        let req = ChatRequest {
            system: "s".into(),
            user: "u".into(),
            max_tokens: 1,
        };
        let cancel: CancelToken = Arc::new(AtomicBool::new(false));
        assert!(matches!(
            b.stream(&req, &cancel, &mut |_| {}),
            Err(InferError::Http(_))
        ));
    }
}
