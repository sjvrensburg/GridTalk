# Third-party licences shipped with GridTalk binaries

- `llama.cpp-LICENSE.txt` — llama.cpp and ggml (MIT), statically
  linked into `gridtalk` / `gridtalk-ui` via the `llama-cpp-2` crate.
- Gridcraft is vendored under `vendor/gridcraft` (MIT/Apache-2.0);
  its `LICENSE-MIT`, `LICENSE-APACHE` and `NOTICE` are copied into
  release packages as `Gridcraft-LICENSE-MIT`,
  `Gridcraft-LICENSE-APACHE` and `Gridcraft-NOTICE`.
- Remaining crates.io dependencies are listed in `Cargo.lock`; their
  licences are the standard crates.io MIT/Apache-2.0 terms recorded
  in each crate's manifest.
- The Windows package also bundles `vulkan-1.dll` (the Khronos
  Vulkan loader, Apache-2.0) from the LunarG SDK so the binaries
  start on machines without a Vulkan driver installed; llama.cpp
  then falls back to CPU.
