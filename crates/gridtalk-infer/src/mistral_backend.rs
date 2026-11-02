//! mistral.rs in-process backend (feature `mistralrs`).
//!
//! Chosen on the `feat/mistralrs-engine` branch per user decision:
//! mistral.rs first, in-process library, engine-first across GGUF
//! architectures (Qwen3-MoE is the stretch target, not the gate).
//! The async mistral.rs runtime is bridged to the synchronous
//! `ChatBackend` trait with a per-call Tokio runtime; the model is
//! loaded per call today (same caveat as the candle backend — a
//! pooled model cache is follow-up work).

use crate::{CancelToken, ChatBackend, ChatRequest, InferError};
use std::path::PathBuf;
use std::sync::atomic::Ordering;

pub struct MistralRsBackend {
    pub model_path: PathBuf,
    pub max_tokens: usize,
}

impl MistralRsBackend {
    async fn run(&self, req: &ChatRequest) -> Result<String, InferError> {
        // Pre-flight: candle's MoE forward is not implemented on
        // CPU ("indexed_moe_forward is not implemented in this
        // platform" panics inside mistral.rs). Detect MoE GGUFs and
        // fail with an actionable typed error instead.
        let info = crate::models::parse_gguf_file(&self.model_path)
            .map_err(|e| InferError::EngineUnavailable(e.to_string()))?;
        let arch = info.architecture.clone().unwrap_or_default();
        if arch.to_ascii_lowercase().contains("moe") {
            return Err(InferError::EngineUnavailable(format!(
                "mistral.rs (candle) cannot run MoE architecture '{arch}' on CPU: indexed_moe_forward is GPU-only. Options: use a dense GGUF with mistral.rs, build mistral.rs with a GPU feature (cuda/metal) on supported hardware, or use the llama-cpp rung"
            )));
        }
        let dir = self
            .model_path
            .parent()
            .ok_or_else(|| {
                InferError::EngineUnavailable("model path has no parent directory".into())
            })?
            .to_path_buf();
        let file = self
            .model_path
            .file_name()
            .ok_or_else(|| InferError::EngineUnavailable("model path has no file name".into()))?
            .to_string_lossy()
            .into_owned();
        let builder = mistralrs::GgufModelBuilder::new(dir.to_string_lossy(), vec![file]);
        // A tokenizer.json beside the model wins; otherwise mistral.rs
        // uses the tokenizer embedded in / resolved for the GGUF.
        let tokenizer_json = dir.join("tokenizer.json");
        let builder = if tokenizer_json.exists() {
            builder.with_tokenizer_json(tokenizer_json.to_string_lossy())
        } else {
            builder
        };
        let model = builder
            .build()
            .await
            .map_err(|e| InferError::EngineUnavailable(format!("mistral.rs load: {e}")))?;
        let messages = mistralrs::TextMessages::new()
            .add_message(mistralrs::TextMessageRole::System, &req.system)
            .add_message(mistralrs::TextMessageRole::User, &req.user);
        let response = model
            .send_chat_request(messages)
            .await
            .map_err(|e| InferError::EngineUnavailable(format!("mistral.rs generate: {e}")))?;
        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .ok_or_else(|| {
                InferError::EngineUnavailable("mistral.rs returned no content".into())
            })?;
        Ok(content)
    }
}

impl ChatBackend for MistralRsBackend {
    fn name(&self) -> &str {
        "mistralrs-local"
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
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| InferError::EngineUnavailable(format!("tokio runtime: {e}")))?;
        // The request is one blocking generation; cancellation is
        // honoured before it starts (mid-generation abort is a
        // mistral.rs streaming follow-up).
        let text = runtime.block_on(self.run(req))?;
        if cancel.load(Ordering::Relaxed) {
            return Err(InferError::Cancelled);
        }
        for word in text.split_inclusive(' ') {
            if cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            on_token(word);
        }
        Ok(())
    }
}

/// Build the backend from config when engine = "mistralrs" and a
/// model path is configured and present.
pub fn mistral_backend_from_config(
    cfg: &crate::config::InferenceConfig,
) -> Option<MistralRsBackend> {
    if cfg.engine != "mistralrs" {
        return None;
    }
    let model_path = cfg.model_path.as_ref().map(PathBuf::from)?;
    Some(MistralRsBackend {
        model_path,
        max_tokens: 512,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_models_fail_with_actionable_typed_error_not_a_panic() {
        let dir = std::env::temp_dir().join(format!("gt-moe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("moe.gguf");
        std::fs::write(&path, crate::models::synthetic_gguf("qwen3moe")).unwrap();
        let backend = MistralRsBackend {
            model_path: path,
            max_tokens: 8,
        };
        let req = ChatRequest {
            system: "s".into(),
            user: "u".into(),
            max_tokens: 8,
        };
        let cancel: CancelToken = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let err = backend.stream(&req, &cancel, &mut |_| {}).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("MoE") && msg.contains("llama-cpp"),
            "got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn config_gating_requires_engine_and_model() {
        let mut cfg = crate::config::InferenceConfig::default();
        assert!(mistral_backend_from_config(&cfg).is_none());
        cfg.engine = "mistralrs".into();
        assert!(mistral_backend_from_config(&cfg).is_none());
        cfg.model_path = Some("/models/m.gguf".into());
        let backend = mistral_backend_from_config(&cfg).unwrap();
        assert_eq!(backend.name(), "mistralrs-local");
    }
}
