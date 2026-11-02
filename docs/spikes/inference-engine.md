# Spike: local inference engine (Phase 3, week 1)

## Question

Which engine runs the target quantized Qwen3-MoE GGUF acceptably on
a CPU-only laptop and a mid-range GPU: candle, mistral.rs, or
llama-cpp via FFI?

## Ladder (highest rung that passes wins)

| Rung | Engine | Why this order | Cost if chosen |
|---|---|---|---|
| 1 | candle (`candle-core` + `candle-transformers`) | Pure Rust, no C toolchain, smallest binary | Narrowest GGUF arch coverage |
| 2 | mistral.rs | Rust inference, GGUF support, OpenAI-compatible serving | Larger dependency; may prefer sidecar process |
| 3 | llama-cpp FFI (`llama-cpp-rs` / bindgen over llama.cpp) | Broadest GGUF/arch coverage | C/C++ build, `unsafe` surface, per-platform native packaging |

All three sit behind `gridtalk_infer::ChatBackend` and are Cargo
features (`candle`, `mistralrs`, `llamacpp`) so the FFI build cost is
never paid unless that rung is selected.

## Exit criteria (fill in during the spike)

- [ ] Loads the target GGUF and streams tokens
- [ ] Cancels cleanly via the shared `CancelToken`
- [ ] First-token latency (CPU / GPU): ___ / ___
- [ ] Throughput tok/s (CPU / GPU): ___ / ___
- [ ] Decision + date: ___

Until the spike runs, `select_local_engine()` returns the first
compiled-in rung and the app falls back to the deterministic echo
backend, so the full pipeline is demoable without a model.
