# 0003: Gridcraft ribbon reuse — companion app, not embedding

## Context

Gridcraft's `gridcraft-ui-egui` compiles standalone (verified:
`cargo check -p gridcraft-ui-egui` in the vendored workspace). Its
ribbon, however, is not a widget library: `SheetApp` owns the
window, the engine session, dialogs, and the control channel, and
the crate links storytold's *forked* egui (git rev `6d518fc`),
while GridTalk's shell uses crates.io egui 0.37. Two incompatible
egui versions cannot render into one window, and adopting
`SheetApp` wholesale would replace — not extend — the GridTalk
shell, chat pane, and graph pane.

## Decision

Reuse the ribbon by *composition*, not embedding:

- `gridtalk open-in-gridcraft <file>` launches the Gridcraft
  desktop app (full ribbon, calc engine, XLSX fidelity) on the
  same file. The GridTalk UI has an equivalent button.
- The user edits in Gridcraft, saves, and GridTalk re-opens /
  re-snapshots the file (the incremental snapshot layer makes
  this cheap). GridTalk owns chat, graph, and citations;
  Gridcraft owns heavy editing.

## Rejected

- **Embed `gridcraft-ui-egui` as a library:** requires migrating
  GridTalk to Gridcraft's forked egui and engine session model;
  re-evaluate if upstream publishes the UI on crates.io with a
  stable egui.
- **Re-implement ribbon fidelity in GridTalk:** exactly the
  complexity GridTalk exists to avoid.

## Consequences

 Locating the binary: `GRIDTALK_GRIDCRAFT_BIN`, else a built
 `vendor/gridcraft/target/{release,debug}/gridcraft`, else a
 typed error telling the user the one build command. First launch
 may compile Gridcraft once; after that it is instant.
