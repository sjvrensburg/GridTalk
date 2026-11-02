//! Inference backends. One trait, three local engine rungs plus an
//! OpenAI-compatible external client. No redaction lives here: the
//! external client sends the passage verbatim (see
//! docs/decisions/0002-no-redaction.md).

use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
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
        let mut stream = std::net::TcpStream::connect(host_port)?;
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
        let mut raw = String::new();
        stream.read_to_string(&mut raw)?;
        let json_start = raw
            .find("\r\n\r\n")
            .map(|i| i + 4)
            .ok_or_else(|| InferError::Http("malformed HTTP response".into()))?;
        let value: serde_json::Value = serde_json::from_str(&raw[json_start..])
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
pub fn backend_from_config(cfg: &crate::config::InferenceConfig) -> Box<dyn ChatBackend> {
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
        /// Minimal TOML reader for the flat [inference] table we write.
        /// (Avoids a toml dependency for a six-key file.)
        pub fn parse(text: &str) -> Config {
            let mut cfg = Config::default();
            let mut in_inference = false;
            for line in text.lines() {
                let line = line.trim();
                if line.starts_with('[') {
                    in_inference = line == "[inference]";
                    continue;
                }
                if !in_inference {
                    continue;
                }
                if let Some((k, v)) = line.split_once('=') {
                    let v = v.trim().trim_matches('"').to_string();
                    match k.trim() {
                        "backend" => cfg.inference.backend = v,
                        "engine" => cfg.inference.engine = v,
                        "model" => cfg.inference.model = Some(v),
                        "model_path" => cfg.inference.model_path = Some(v),
                        "base_url" => cfg.inference.base_url = Some(v),
                        "api_key_env" => cfg.inference.api_key_env = Some(v),
                        _ => {}
                    }
                }
            }
            cfg
        }
        pub fn load(path: &std::path::Path) -> Config {
            std::fs::read_to_string(path)
                .map(|t| Config::parse(&t))
                .unwrap_or_default()
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
