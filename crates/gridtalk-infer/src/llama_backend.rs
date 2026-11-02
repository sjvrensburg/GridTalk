//! The single local inference backend: llama.cpp over Vulkan
//! (via `llama-cpp-2`). One engine for every GGUF architecture —
//! including Qwen3-MoE, which candle cannot run on CPU — on every
//! platform with a Vulkan driver. macOS is not a target (no test
//! hardware; Vulkan is unavailable there in practice).
//!
//! All layers are offloaded to the GPU (`with_n_gpu_layers(u32::MAX)`);
//! on a Vulkan-less machine llama.cpp falls back to CPU and the
//! backend still works, just slower.

use crate::{CancelToken, ChatBackend, ChatRequest, GenerationStats, InferError};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub const MAX_CONTEXT: usize = 4096;

pub struct LlamaCppBackend {
    pub model_path: PathBuf,
    pub seed: u32,
    pub temperature: f32,
    /// GPU layers to offload; u32::MAX = everything (Vulkan).
    pub n_gpu_layers: u32,
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        LlamaCppBackend {
            model_path: PathBuf::new(),
            seed: 42,
            temperature: 0.7,
            n_gpu_layers: u32::MAX,
        }
    }
}

/// Process-wide backend + loaded-model cache. Only successes are
/// cached (a transient Vulkan init failure can be retried), and the
/// model is reloaded only when the path changes. The guards are held
/// for the whole generation, which serialises concurrent requests —
/// llama.cpp contexts on one model are not safe to share anyway.
struct CachedState {
    backend: LlamaBackend,
    model: Option<(PathBuf, LlamaModel)>,
}

fn cached_state() -> &'static std::sync::Mutex<Option<CachedState>> {
    static STATE: std::sync::OnceLock<std::sync::Mutex<Option<CachedState>>> =
        std::sync::OnceLock::new();
    STATE.get_or_init(|| std::sync::Mutex::new(None))
}

impl LlamaCppBackend {
    pub fn generate_with_stats(
        &self,
        req: &ChatRequest,
        cancel: &CancelToken,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenerationStats, InferError> {
        let started = Instant::now();
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(InferError::EngineUnavailable(format!(
                "temperature {} is outside 0.0..=2.0",
                self.temperature
            )));
        }
        if req.max_tokens == 0 {
            return Ok(GenerationStats {
                prompt_tokens: 0,
                generated_tokens: 0,
                first_token_ms: None,
                total_ms: started.elapsed().as_secs_f64() * 1000.0,
                load_ms: 0.0,
                tokens_per_sec: 0.0,
            });
        }
        let load_started = Instant::now();
        let mut guard = cached_state()
            .lock()
            .map_err(|_| InferError::EngineUnavailable("model cache poisoned".into()))?;
        if guard.is_none() {
            let backend = LlamaBackend::init().map_err(|e| {
                InferError::EngineUnavailable(format!("llama.cpp backend init failed: {e}"))
            })?;
            *guard = Some(CachedState {
                backend,
                model: None,
            });
        }
        let state = guard.as_mut().unwrap();
        let needs_load = state
            .model
            .as_ref()
            .is_none_or(|(path, _)| path != &self.model_path);
        if needs_load {
            let model_params = LlamaModelParams::default().with_n_gpu_layers(self.n_gpu_layers);
            let loaded =
                LlamaModel::load_from_file(&state.backend, &self.model_path, &model_params)
                    .map_err(|e| {
                        InferError::EngineUnavailable(format!(
                            "llama.cpp could not load {}: {e}",
                            self.model_path.display()
                        ))
                    })?;
            state.model = Some((self.model_path.clone(), loaded));
        }
        let backend = &state.backend;
        let model = &state.model.as_ref().unwrap().1;
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

        // Prompt via the model's own chat template when it has one.
        let prompt = match model.chat_template(None) {
            Ok(tmpl) => {
                let messages = vec![
                    LlamaChatMessage::new("system".into(), req.system.clone())
                        .map_err(|e| InferError::EngineUnavailable(e.to_string()))?,
                    LlamaChatMessage::new("user".into(), req.user.clone())
                        .map_err(|e| InferError::EngineUnavailable(e.to_string()))?,
                ];
                model
                    .apply_chat_template(&tmpl, &messages, true)
                    .map_err(|e| InferError::EngineUnavailable(format!("chat template: {e}")))?
            }
            Err(_) => format!(
                "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                req.system, req.user
            ),
        };

        let vocab = model.vocab();
        let prompt_tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if prompt_tokens.is_empty() {
            return Err(InferError::EngineUnavailable("empty prompt".into()));
        }
        if prompt_tokens
            .len()
            .checked_add(req.max_tokens)
            .is_none_or(|total| total > MAX_CONTEXT)
        {
            return Err(InferError::EngineUnavailable(format!(
                "prompt ({} tokens) + max_tokens ({}) exceeds the {MAX_CONTEXT}-token context",
                prompt_tokens.len(),
                req.max_tokens
            )));
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(MAX_CONTEXT as u32))
            .with_n_batch(512);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| InferError::EngineUnavailable(format!("llama.cpp context: {e}")))?;

        let mut sampler = if self.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(self.temperature),
                LlamaSampler::dist(self.seed),
            ])
        };

        // Prefill in <=512-token chunks (the batch capacity), with
        // logits requested only for the final prompt token.
        let gen_started = Instant::now();
        for chunk_start in (0..prompt_tokens.len()).step_by(512) {
            let chunk = &prompt_tokens[chunk_start..(chunk_start + 512).min(prompt_tokens.len())];
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (j, token) in chunk.iter().enumerate() {
                let last = chunk_start + j + 1 == prompt_tokens.len();
                batch
                    .add(*token, (chunk_start + j) as i32, &[0], last)
                    .map_err(|e| InferError::EngineUnavailable(format!("batch: {e}")))?;
            }
            ctx.decode(&mut batch)
                .map_err(|e| InferError::EngineUnavailable(format!("prefill decode: {e}")))?;
        }

        let mut first_token_ms: Option<f64> = None;
        let mut generated = 0usize;
        let mut text_bytes: Vec<u8> = Vec::new();
        let mut emitted = String::new();
        let mut n_cur = prompt_tokens.len() as i32;

        while generated < req.max_tokens {
            if cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);
            if first_token_ms.is_none() {
                first_token_ms = Some(gen_started.elapsed().as_secs_f64() * 1000.0);
            }
            // Stop on EOG, and on end-of-turn (<|im_end|> etc.) which
            // some GGUFs do not register as EOG.
            if vocab.is_eog(token) || token == vocab.eot() || token == vocab.eos() {
                break;
            }
            let piece = vocab.token_to_piece(token, true, None);
            text_bytes.extend_from_slice(&piece);
            // Stop markers can be split across BPE pieces, so check
            // the cumulative text and truncate at the marker.
            let cumulative = String::from_utf8_lossy(&text_bytes).into_owned();
            // Any "<|...|>" special in the *text* is a stop marker
            // (possibly split across pieces, e.g. "<|im_end|").
            for marker in ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>"] {
                if let Some(at) = cumulative.find(marker) {
                    text_bytes.truncate(at);
                    if let Ok(text) = std::str::from_utf8(&text_bytes) {
                        if let Some(delta) = text.strip_prefix(emitted.as_str()) {
                            if !delta.is_empty() {
                                on_token(delta);
                            }
                        }
                    }
                    generated += 1;
                    break;
                }
            }
            let _ = &cumulative;
            generated += 1;
            // Emit only complete UTF-8 so far (pieces can split a
            // multi-byte character across tokens), and hold back a
            // trailing "<|..." fragment that may complete into a
            // stop marker on the next token.
            if let Ok(text) = std::str::from_utf8(&text_bytes) {
                let safe_end = match text.rfind('<') {
                    Some(at)
                        if text[at..].chars().all(|c| {
                            c == '<' || c == '|' || c.is_ascii_alphanumeric() || c == '_'
                        }) =>
                    {
                        at
                    }
                    _ => text.len(),
                };
                let safe = &text[..safe_end];
                if let Some(delta) = safe.strip_prefix(emitted.as_str()) {
                    if !delta.is_empty() {
                        on_token(delta);
                        emitted = safe.to_string();
                    }
                }
            }
            let mut next_batch = LlamaBatch::new(1, 1);
            next_batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| InferError::EngineUnavailable(format!("batch: {e}")))?;
            n_cur += 1;
            ctx.decode(&mut next_batch)
                .map_err(|e| InferError::EngineUnavailable(format!("decode: {e}")))?;
        }

        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        let gen_ms = gen_started.elapsed().as_secs_f64() * 1000.0;
        Ok(GenerationStats {
            prompt_tokens: prompt_tokens.len(),
            generated_tokens: generated,
            first_token_ms,
            total_ms,
            load_ms,
            tokens_per_sec: if gen_ms > 0.0 {
                generated as f64 / (gen_ms / 1000.0)
            } else {
                0.0
            },
        })
    }
}

impl ChatBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        "llama-cpp-vulkan"
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

/// Build from config: local backend is always llama.cpp. The old
/// `engine = "candle"|"mistralrs"` values are accepted for config
/// compatibility but mapped onto llama.cpp with a stderr note.
pub fn llama_backend_from_config(cfg: &crate::config::InferenceConfig) -> Option<LlamaCppBackend> {
    let model_path = cfg.model_path.as_ref().map(PathBuf::from)?;
    Some(LlamaCppBackend {
        model_path,
        ..LlamaCppBackend::default()
    })
}
