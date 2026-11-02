//! Inference-engine spike probe: can this rung load this GGUF?
//! Exit criteria per rung (see docs/spikes/inference-engine.md):
//! header parses, engine library reads the file, tensors enumerated.

use crate::models::{parse_gguf, ModelError};
use crate::EngineKind;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineReport {
    pub engine: EngineKind,
    pub compiled_in: bool,
    pub gguf_arch: Option<String>,
    pub tensor_count: Option<usize>,
    pub loads: bool,
    pub detail: String,
}

pub fn probe_engine(engine: EngineKind, model_path: &Path) -> EngineReport {
    let bytes = match std::fs::read(model_path) {
        Ok(b) => b,
        Err(e) => {
            return EngineReport {
                engine,
                compiled_in: engine.compiled_in(),
                gguf_arch: None,
                tensor_count: None,
                loads: false,
                detail: format!("cannot read model file: {e}"),
            }
        }
    };
    let info = match parse_gguf(&bytes) {
        Ok(i) => i,
        Err(ModelError::NotGguf(m)) | Err(ModelError::Truncated(m)) => {
            return EngineReport {
                engine,
                compiled_in: engine.compiled_in(),
                gguf_arch: None,
                tensor_count: None,
                loads: false,
                detail: format!("GGUF header rejected: {m}"),
            }
        }
        Err(e) => {
            return EngineReport {
                engine,
                compiled_in: engine.compiled_in(),
                gguf_arch: None,
                tensor_count: None,
                loads: false,
                detail: e.to_string(),
            }
        }
    };
    let arch = info.architecture.clone();
    match engine {
        EngineKind::Candle => probe_candle(engine, model_path, arch),
        EngineKind::MistralRs => EngineReport {
            engine,
            compiled_in: false,
            gguf_arch: arch,
            tensor_count: None,
            loads: false,
            detail: "mistral.rs adapter not compiled in this build; spike rung 2 is evaluated by enabling its crate behind this probe (same EngineReport shape)".into(),
        },
        EngineKind::LlamaCpp => EngineReport {
            engine,
            compiled_in: false,
            gguf_arch: arch,
            tensor_count: None,
            loads: false,
            detail: "llama-cpp FFI adapter not compiled in this build; enabling it requires the native llama.cpp build (feature `llamacpp`), spike rung 3".into(),
        },
    }
}

#[cfg(feature = "candle")]
fn probe_candle(engine: EngineKind, model_path: &Path, arch: Option<String>) -> EngineReport {
    let mut file = match std::fs::File::open(model_path) {
        Ok(f) => f,
        Err(e) => {
            return EngineReport {
                engine,
                compiled_in: true,
                gguf_arch: arch,
                tensor_count: None,
                loads: false,
                detail: format!("open failed: {e}"),
            }
        }
    };
    match candle_core::quantized::gguf_file::Content::read(&mut file) {
        Ok(content) => EngineReport {
            engine,
            compiled_in: true,
            gguf_arch: arch,
            tensor_count: Some(content.tensor_infos.len()),
            loads: true,
            detail: format!(
                "candle-core read GGUF: {} tensors, {} metadata keys",
                content.tensor_infos.len(),
                content.metadata.len()
            ),
        },
        Err(e) => EngineReport {
            engine,
            compiled_in: true,
            gguf_arch: arch,
            tensor_count: None,
            loads: false,
            detail: format!("candle-core rejected GGUF: {e}"),
        },
    }
}

#[cfg(not(feature = "candle"))]
fn probe_candle(engine: EngineKind, _model_path: &Path, arch: Option<String>) -> EngineReport {
    EngineReport {
        engine,
        compiled_in: false,
        gguf_arch: arch,
        tensor_count: None,
        loads: false,
        detail: "candle adapter not compiled in; rebuild with --features candle".into(),
    }
}

/// Run the ladder top-down and return every rung's report.
pub fn probe_ladder(model_path: &Path) -> Vec<EngineReport> {
    EngineKind::ladder()
        .into_iter()
        .map(|e| probe_engine(e, model_path))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::synthetic_gguf;

    #[test]
    fn ladder_reports_all_rungs_and_rejects_bad_files() {
        let dir = std::env::temp_dir().join(format!("gt-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let good = dir.join("good.gguf");
        std::fs::write(&good, synthetic_gguf("qwen3moe")).unwrap();
        let reports = probe_ladder(&good);
        assert_eq!(reports.len(), 3);
        assert_eq!(reports[0].engine, EngineKind::Candle);
        assert_eq!(reports[0].gguf_arch.as_deref(), Some("qwen3moe"));
        let bad = dir.join("bad.gguf");
        std::fs::write(&bad, b"not a model").unwrap();
        assert!(!probe_engine(EngineKind::Candle, &bad).loads);
        assert!(!probe_engine(EngineKind::Candle, Path::new("/nonexistent.gguf")).loads);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
