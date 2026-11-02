#!/bin/sh
# GridTalk installer: builds from this checkout and installs the
# `gridtalk` CLI and `gridtalk-ui` desktop app into ~/.local/bin
# (override with GRIDTALK_INSTALL_DIR). For prebuilt archives, see
# the GitHub Releases page.
set -eu
DIR="${GRIDTALK_INSTALL_DIR:-$HOME/.local/bin}"
mkdir -p "$DIR"
cargo build --release -p gridtalk-app -p gridtalk-ui
cp target/release/gridtalk target/release/gridtalk-ui "$DIR/"
echo "Installed gridtalk and gridtalk-ui into $DIR"
echo "First run: gridtalk-ui  (a short setup wizard will appear)"
