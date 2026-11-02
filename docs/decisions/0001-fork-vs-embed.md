# 0001: Fork Gridcraft as a vendored base, extend in new crates

## Decision

Vendor Gridcraft as a git submodule (`vendor/gridcraft`, pinned in
`UPSTREAM.md`) and put all GridTalk code in new workspace crates.
GridTalk crates depend only on Gridcraft's public model/calc surface
once XLSX/ODS integration lands. Engine internals are never forked.

## Rejected

- **Gridcraft as a crates-io dependency:** its crates are
  workspace-internal with no stable published API.
- **Greenfield egui grid:** re-implements formula calculation and
  XLSX fidelity, the hardest and least differentiating part.

## Consequences

Monthly upstream rebases (half-day budget). The submodule is excluded
from the GridTalk workspace so upstream's build never slows ours.

## Update (Phase 2, landed)

GridTalk now path-depends on Gridcraft's `xlsx`, `model`, and
`core` crates for XLSX/ODS ingestion (see `load_gridcraft_bytes`).
The UI did *not* adopt `gridcraft-ui-egui`; see decision 0003 for
the companion-app composition that replaced direct embedding.
