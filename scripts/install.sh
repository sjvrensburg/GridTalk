#!/bin/sh
# GridTalk installer: builds from the checkout containing this
# script and installs `gridtalk` and `gridtalk-ui` into
# ~/.local/bin (override with GRIDTALK_INSTALL_DIR). Nothing is
# downloaded except crates.io dependencies during the build.
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(dirname -- "$SCRIPT_DIR")
cd "$REPO_ROOT"
DIR="${GRIDTALK_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$DIR"
for bin in gridtalk gridtalk-ui; do
  if [ -e "$DIR/$bin" ] && [ -z "${GRIDTALK_FORCE:-}" ]; then
    echo "refusing to overwrite existing $DIR/$bin (set GRIDTALK_FORCE=1 to replace)" >&2
    exit 1
  fi
done
cargo build --release -p gridtalk-app -p gridtalk-ui
cp target/release/gridtalk target/release/gridtalk-ui "$DIR/"
echo "Installed gridtalk and gridtalk-ui into $DIR"
echo "First run: gridtalk-ui  (a short setup wizard will appear)"
