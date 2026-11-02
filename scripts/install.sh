#!/bin/sh
# GridTalk installer (Linux): builds from the checkout containing
# this script and installs `gridtalk` and `gridtalk-ui` into
# ~/.local/bin (override with GRIDTALK_INSTALL_DIR). Nothing is
# downloaded except crates.io dependencies during the build.
# Windows users: use the release .zip instead (no install script).
# Prerequisites: cargo, CMake, a C++ toolchain and glslc (Vulkan
# SDK) — llama-cpp-sys-2 needs all of them to build.
set -eu
for tool in cargo cmake glslc cc c++; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing prerequisite: $tool (install Rust, CMake and the Vulkan SDK/glslc first)" >&2
    exit 1
  fi
done
# Resolve symlinks so invoking through a linked script still
# builds the real checkout.
SCRIPT_PATH=$(readlink -f "$0" 2>/dev/null || echo "$0")
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
cd "$REPO_ROOT"
DIR="${GRIDTALK_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$DIR"
for bin in gridtalk gridtalk-ui; do
  if [ -e "$DIR/$bin" ] && [ "${GRIDTALK_FORCE:-}" != "1" ]; then
    echo "refusing to overwrite existing $DIR/$bin (set GRIDTALK_FORCE=1 to replace)" >&2
    exit 1
  fi
done
cargo build --locked --release -p gridtalk-app -p gridtalk-ui
# Install atomically: stage BOTH binaries beside their targets
# first, then rename both, so a failure cannot leave a mixed pair.
trap 'rm -f "$DIR/.gridtalk.tmp-$$" "$DIR/.gridtalk-ui.tmp-$$"' EXIT
for bin in gridtalk gridtalk-ui; do
  cp "target/release/$bin" "$DIR/.$bin.tmp-$$"
  chmod +x "$DIR/.$bin.tmp-$$"
done
# Recheck BOTH targets before the first rename, so a binary that
# appeared during the build cannot leave a mixed pair installed.
for bin in gridtalk gridtalk-ui; do
  if [ -e "$DIR/$bin" ] && [ "${GRIDTALK_FORCE:-}" != "1" ]; then
    echo "refusing to overwrite existing $DIR/$bin (set GRIDTALK_FORCE=1 to replace)" >&2
    exit 1
  fi
done
for bin in gridtalk gridtalk-ui; do
  mv "$DIR/.$bin.tmp-$$" "$DIR/$bin"
done
trap - EXIT
echo "Installed gridtalk and gridtalk-ui into $DIR"
echo "First run: gridtalk-ui  (a short setup wizard will appear)"
