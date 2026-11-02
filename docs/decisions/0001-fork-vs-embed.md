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
