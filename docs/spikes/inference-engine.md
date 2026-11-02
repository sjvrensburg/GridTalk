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
- [x] **Real-model generation + benchmark (2026-11-02).**
      Model: SmolLM2-135M-Instruct Q4_K_M GGUF (101 MB, arch
      `llama`) + its `tokenizer.json`, run through
      `gridtalk --features candle-gen generate` on an AMD Ryzen AI
      MAX+ 395 (CPU only, release build):
      prompt 24 tokens, generated 48 tokens,
      **first token 541 ms** (includes weight load),
      **31.9 tok/s**. Debug build: 7.1 s / 2.2 tok/s — release
      builds are required for interactive use. The target
      Qwen3-MoE model remains unproven on candle (its arch is
      rejected with a typed error naming `qwen3moe`); that is the
      trigger to evaluate rung 2 (mistral.rs).

## Decision

candle is the default rung: it loads GGUF and *generates* today
for llama-family models (feature `candle-gen`; configured via
`model_path` in config, with `tokenizer.json` beside the model).
Users whose model architecture candle rejects (e.g. Qwen3-MoE)
get a typed error naming the architecture — evaluating mistral.rs
(rung 2) is the next step for those models, and llama-cpp FFI
remains the fallback of last resort. With no model configured,
local chat uses the deterministic echo backend so the full
pipeline stays demoable.

## mistral.rs rung (branch feat/mistralrs-engine, 2026-11-02)

User decisions: mistral.rs first, in-process library, engine-first
across GGUFs (Qwen3-MoE = stretch, not the gate).

Measured on this machine (Ryzen AI Max+ 395, CPU, release):

| Model | Arch | Result |
|---|---|---|
| SmolLM2-135M-Instruct Q4_K_M | llama | Generates (debug build, 30 s incl. load) |
| Qwen3-4B-Instruct-2507 Q4_K_M (2.4 GB, on disk) | qwen3 | **Generates: 11.4 s total incl. load** — an arch candle-gen rejects |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M (18 GB, on disk) | qwen3moe | Loads, then candle panics `indexed_moe_forward is not implemented in this platform` — **MoE forward is GPU-only in candle** |

Consequences, implemented in `gridtalk-infer` (feature `mistralrs`):

- `MistralRsBackend` generates in-process for dense GGUFs of any
  architecture mistral.rs supports; wired into
  `backend_from_config` (engine = "mistralrs" + model_path).
- MoE GGUFs are detected from the GGUF header before loading and
  fail with a typed, actionable error (dense model / GPU build /
  llama-cpp rung) instead of a channel-closed panic.
- `gridtalk spike` reports this MoE caveat per model.

llama-cpp FFI remains not compiled in. On this ROCm machine it is
also the only rung that could use the GPU (llama.cpp + ROCm, per
~/Projects/halo-prep benchmarks at ~50 tok/s for 35B-A3B), which
makes it the natural next rung if MoE-on-this-hardware becomes a
requirement. The existing loopback llama-server stack on the
EVO-X3 (:8092/:8088) is already reachable today through the
OpenAI-compatible backend with no new code.
