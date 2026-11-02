//! Model probe: can the single local engine (llama.cpp/Vulkan)
//! use this GGUF? The probe parses the GGUF header and reports
//! architecture/tensors; actual load + generation is exercised by
//! `LlamaCppBackend` (and `gridtalk generate`).

use crate::models::{parse_gguf_file, ModelError};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineReport {
    pub engine: &'static str,
    pub gguf_arch: Option<String>,
    pub tensor_count: Option<usize>,
    /// True when the GGUF *header* parses. This does not load
    /// tensors or prove the engine can run the file.
    pub header_ok: bool,
    pub detail: String,
}

pub fn probe_model(model_path: &Path) -> EngineReport {
    match parse_gguf_file(model_path) {
        Ok(info) => EngineReport {
            engine: "llama-cpp-vulkan",
            gguf_arch: info.architecture.clone(),
            tensor_count: Some(info.tensor_count as usize),
            header_ok: true,
            detail: format!(
                "GGUF header OK (arch {:?}, {} tensors). Header-only check: the engine has NOT loaded this file; run `gridtalk generate` for a real load test",
                info.architecture, info.tensor_count
            ),
        },
        Err(ModelError::NotGguf(m)) | Err(ModelError::Truncated(m)) => EngineReport {
            engine: "llama-cpp-vulkan",
            gguf_arch: None,
            tensor_count: None,
            header_ok: false,
            detail: format!("GGUF header rejected: {m}"),
        },
        Err(e) => EngineReport {
            engine: "llama-cpp-vulkan",
            gguf_arch: None,
            tensor_count: None,
            header_ok: false,
            detail: e.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::synthetic_gguf;

    #[test]
    fn probe_reports_arch_for_any_gguf() {
        let dir = std::env::temp_dir().join(format!("gt-probe2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.gguf");
        std::fs::write(&good, synthetic_gguf("qwen3moe")).unwrap();
        let report = probe_model(&good);
        assert!(report.header_ok);
        assert_eq!(report.gguf_arch.as_deref(), Some("qwen3moe"));
        let bad = dir.join("bad.gguf");
        std::fs::write(&bad, b"not a model").unwrap();
        assert!(!probe_model(&bad).header_ok);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
