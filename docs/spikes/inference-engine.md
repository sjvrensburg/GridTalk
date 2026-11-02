# Spike: local inference engine (Phase 3)

## Question

Which engine runs the target quantized Qwen3-MoE GGUF acceptably:
candle, mistral.rs, or llama-cpp via FFI?

## Ladder (highest rung that passes wins)

| Rung | Engine | Why this order | Cost if chosen |
|---|---|---|---|
| 1 | candle (`candle-core` quantized GGUF) | Pure Rust, no C toolchain | Narrowest GGUF arch coverage |
| 2 | mistral.rs | Rust inference, GGUF support | Larger dependency |
| 3 | llama-cpp FFI (`llama-cpp-rs` / bindgen) | Broadest GGUF coverage | C/C++ build, `unsafe`, native packaging |

All rungs sit behind `gridtalk_infer::probe::probe_engine` returning
an `EngineReport`, and behind Cargo features (`candle`, `mistralrs`,
`llamacpp`) so the FFI build cost is never paid unless selected.
Run it with: `gridtalk spike <model.gguf>`.

## Results (2026-11-02, synthetic GGUF, arch `qwen3moe`)

- [x] GGUF header parser: reads v2/v3, architecture/name metadata,
      rejects bad magic and truncated files (unit tests).
- [x] **candle rung: LOADS.** With `--features candle`,
      `candle-core`'s GGUF reader opens the file and enumerates
      tensors/metadata (`gridtalk spike` reports
      `Candle: compiled_in=true loads=true`). Generation
      (token streaming for a specific architecture) is the next
      spike step and is *not* yet proven.
- [ ] mistral.rs rung: probe reports `compiled_in=false` with the
      reason. Its crate has not been enabled yet; the probe shape
      is ready for it.
- [ ] llama-cpp FFI rung: probe reports `compiled_in=false`.
      Enabling it requires the native llama.cpp build; kept last
      deliberately.
- [ ] Real-model latency/throughput (first-token, tok/s, CPU/GPU):
      not measured — requires downloading a real GGUF
      (`gridtalk models import <file.gguf>`), then re-running
      `gridtalk --features candle spike <model>`.

## Interim decision

candle is the default rung (it loads GGUF today). mistral.rs is
next if candle's architecture coverage blocks the target model.
llama-cpp FFI remains the fallback of last resort. Until a real
model is wired, local chat uses the deterministic echo backend so
the full pipeline stays demoable.
