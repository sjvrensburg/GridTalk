//! Model probe: can the single local engine (llama.cpp/Vulkan)
//! use this GGUF? The probe parses the GGUF header and reports
//! architecture/tensors; actual load + generation is exercised by
//! `LlamaCppBackend` (and `gridtalk generate`).

use crate::models::{parse_gguf, ModelError};
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineReport {
    pub engine: &'static str,
    pub gguf_arch: Option<String>,
    pub tensor_count: Option<usize>,
    pub loads: bool,
    pub detail: String,
}

pub fn probe_model(model_path: &Path) -> EngineReport {
    let bytes = match std::fs::read(model_path) {
        Ok(b) => b,
        Err(e) => {
            return EngineReport {
                engine: "llama-cpp-vulkan",
                gguf_arch: None,
                tensor_count: None,
                loads: false,
                detail: format!("cannot read model file: {e}"),
            }
        }
    };
    match parse_gguf(&bytes) {
        Ok(info) => EngineReport {
            engine: "llama-cpp-vulkan",
            gguf_arch: info.architecture.clone(),
            tensor_count: Some(info.tensor_count as usize),
            loads: true,
            detail: format!(
                "GGUF header OK (arch {:?}, {} tensors); llama.cpp loads every GGUF architecture, with all layers offloaded to Vulkan when a driver is present",
                info.architecture, info.tensor_count
            ),
        },
        Err(ModelError::NotGguf(m)) | Err(ModelError::Truncated(m)) => EngineReport {
            engine: "llama-cpp-vulkan",
            gguf_arch: None,
            tensor_count: None,
            loads: false,
            detail: format!("GGUF header rejected: {m}"),
        },
        Err(e) => EngineReport {
            engine: "llama-cpp-vulkan",
            gguf_arch: None,
            tensor_count: None,
            loads: false,
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
        assert!(report.loads);
        assert_eq!(report.gguf_arch.as_deref(), Some("qwen3moe"));
        let bad = dir.join("bad.gguf");
        std::fs::write(&bad, b"not a model").unwrap();
        assert!(!probe_model(&bad).loads);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
