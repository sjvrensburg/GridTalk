//! Real GGUF generation on candle (feature `candle-gen`).
//! Supports llama-family GGUF architectures via
//! `candle-transformers`' quantized llama weights — the same loader
//! family as the candle examples. Other architectures return a
//! typed `EngineUnavailable` naming the architecture, which is
//! exactly the signal the spike ladder needs to fall through to
//! mistral.rs / llama-cpp.

use crate::{CancelToken, ChatBackend, ChatRequest, EngineKind, InferError};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub first_token_ms: Option<f64>,
    pub total_ms: f64,
    pub tokens_per_sec: f64,
}

pub struct CandleLlamaBackend {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub seed: u64,
    pub temperature: f64,
}

impl CandleLlamaBackend {
    pub fn generate_with_stats(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenerationStats, InferError> {
        let started = Instant::now();
        // Architecture gate from GGUF metadata.
        let bytes = std::fs::read(&self.model_path)?;
        let info = crate::models::parse_gguf(&bytes)
            .map_err(|e| InferError::EngineUnavailable(e.to_string()))?;
        let arch = info
            .architecture
            .clone()
            .unwrap_or_else(|| "unknown".into());
        if arch != "llama" && arch != "qwen2" && !arch.starts_with("llama") {
            return Err(InferError::EngineUnavailable(format!(
                "candle-gen supports llama-family GGUF today; this model is arch={arch}. \
                 Try the mistral.rs or llama-cpp rung of the ladder"
            )));
        }

        let device = candle_core::Device::Cpu;
        let mut file = std::fs::File::open(&self.model_path)?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| InferError::EngineUnavailable(format!("candle GGUF read: {e}")))?;
        let mut model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
            content, &mut file, &device,
        )
        .map_err(|e| InferError::EngineUnavailable(format!("candle weight load: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&self.tokenizer_path).map_err(|e| {
            InferError::EngineUnavailable(format!(
                "tokenizer load {}: {e}",
                self.tokenizer_path.display()
            ))
        })?;

        let prompt = format!("{}\n\n{}\n\nAssistant:", req.system, req.user);
        let encoding = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| InferError::EngineUnavailable(format!("tokenize: {e}")))?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() {
            return Err(InferError::EngineUnavailable("empty prompt".into()));
        }
        const MAX_CONTEXT: usize = 2048;
        if prompt_tokens.len() + req.max_tokens > MAX_CONTEXT {
            return Err(InferError::EngineUnavailable(format!(
                "prompt ({} tokens) + max_tokens ({}) exceeds the {MAX_CONTEXT}-token context; shorten the context",
                prompt_tokens.len(),
                req.max_tokens
            )));
        }
        let eos_id = tokenizer
            .token_to_id("</s>")
            .or_else(|| tokenizer.token_to_id("<|end_of_text|>"));

        let mut logits_processor = candle_transformers::generation::LogitsProcessor::new(
            self.seed,
            Some(self.temperature),
            None,
        );

        let mut tokens: Vec<u32> = prompt_tokens.clone();
        let mut generated = 0usize;

        // Prefill.
        let input = candle_core::Tensor::new(tokens.as_slice(), &device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| InferError::EngineUnavailable(e.to_string()))?;
        let logits = model
            .forward(&input, 0)
            .map_err(|e| InferError::EngineUnavailable(format!("prefill: {e}")))?;
        let logits = logits
            .squeeze(0)
            .map_err(|e| InferError::EngineUnavailable(e.to_string()))?;
        let mut next = logits_processor
            .sample(&logits)
            .map_err(|e| InferError::EngineUnavailable(format!("sample: {e}")))?;
        let first_token_ms = started.elapsed().as_secs_f64() * 1000.0;
        tokens.push(next);
        generated += 1;
        let mut emitted = String::new();
        if Some(next) != eos_id {
            if let Ok(text) = tokenizer.decode(&tokens[prompt_tokens.len()..], true) {
                if let Some(delta) = text.strip_prefix(&emitted) {
                    on_token(delta);
                    emitted = text;
                }
            }
        }

        while generated < req.max_tokens {
            if cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            let input = candle_core::Tensor::new(&[next], &device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| InferError::EngineUnavailable(e.to_string()))?;
            let logits = model
                .forward(&input, tokens.len() - 1)
                .map_err(|e| InferError::EngineUnavailable(format!("decode: {e}")))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| InferError::EngineUnavailable(e.to_string()))?;
            next = logits_processor
                .sample(&logits)
                .map_err(|e| InferError::EngineUnavailable(format!("sample: {e}")))?;
            tokens.push(next);
            generated += 1;
            if Some(next) == eos_id {
                break;
            }
            // Decode the whole generated suffix and emit only the
            // new text: per-token decode corrupts BPE merges and
            // multi-byte UTF-8 split across tokens.
            if let Ok(text) = tokenizer.decode(&tokens[prompt_tokens.len()..], true) {
                if let Some(delta) = text.strip_prefix(&emitted) {
                    on_token(delta);
                    emitted = text;
                }
            }
        }

        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok(GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            first_token_ms: Some(first_token_ms),
            total_ms,
            tokens_per_sec: if total_ms > 0.0 {
                generated as f64 / (total_ms / 1000.0)
            } else {
                0.0
            },
        })
    }
}

impl ChatBackend for CandleLlamaBackend {
    fn name(&self) -> &str {
        "candle-local"
    }
    fn stream(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<(), InferError> {
        self.generate_with_stats(req, cancel, on_token)?;
        Ok(())
    }
}

pub fn candle_backend_from_config(
    cfg: &crate::config::InferenceConfig,
) -> Option<CandleLlamaBackend> {
    if cfg.engine != EngineKind::Candle.feature_name() {
        return None;
    }
    let model_path = cfg.model_path.as_ref().map(PathBuf::from)?;
    let tokenizer_path = model_path.with_file_name("tokenizer.json");
    Some(CandleLlamaBackend {
        model_path,
        tokenizer_path,
        seed: 42,
        temperature: 0.7,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::synthetic_gguf;

    #[test]
    fn non_llama_arch_is_typed_unavailable() {
        let dir = std::env::temp_dir().join(format!("gt-gen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m.gguf");
        std::fs::write(&path, synthetic_gguf("qwen3moe")).unwrap();
        // qwen3moe arch string starts with qwen3, not qwen2/llama -> unavailable.
        let backend = CandleLlamaBackend {
            model_path: path,
            tokenizer_path: dir.join("tokenizer.json"),
            seed: 1,
            temperature: 0.7,
        };
        let req = ChatRequest {
            system: "s".into(),
            user: "u".into(),
            max_tokens: 2,
        };
        let cancel: CancelToken = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let err = backend.stream(&req, &cancel, &mut |_| {}).unwrap_err();
        assert!(
            format!("{err}").contains("arch=qwen3moe"),
            "must name the unsupported architecture: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
