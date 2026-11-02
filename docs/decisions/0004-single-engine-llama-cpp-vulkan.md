# 0004: One local engine — llama.cpp over Vulkan

## Decision

GridTalk has exactly **one** local inference engine: llama.cpp
(via `llama-cpp-2`) with Vulkan GPU offload. The candle backend
and the mistral.rs experiment are removed. There is no engine
ladder, no engine config switch; `engine = "candle"|"mistralrs"`
in old configs is accepted, noted on stderr, and mapped onto
llama.cpp.

## Why

- One maintained generation path instead of three partial ones.
- llama.cpp runs **every** GGUF architecture, including
  Qwen3-MoE (`qwen3moe`), which candle rejects outright and
  whose MoE forward candle only implements on GPU backends
  GridTalk was not using.
- Vulkan works on Linux and Windows across AMD/Intel/NVIDIA.
  Measured here (Ryzen AI Max+ 395, Radeon 8060S): Qwen3-30B-A3B
  Q4_K_M generates at ~9.7 tok/s end-to-end (incl. load) and
  Qwen3-4B at ~31.9 tok/s through `gridtalk generate`.

## Platforms

Linux and Windows only. **macOS is not a target**: Vulkan is
unavailable there in practice and there is no Apple hardware to
test on. CI and release matrices drop macOS accordingly.

## Consequences

- The native build needs a C++ toolchain, CMake, and Vulkan
  headers/shader compiler (`glslc`) at build time; CI installs
  the Vulkan SDK. llama.cpp falls back to CPU when no Vulkan
  device is present, so Vulkan-less machines still run.
- The external OpenAI-compatible backend is unchanged (it is a
  client, not a local engine), as is the echo backend used when
  no model is configured.
