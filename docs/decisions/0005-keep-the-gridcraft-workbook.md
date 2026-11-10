# 0005: Keep the Gridcraft workbook alongside the snapshot

Decided 2026-11-04; landed 2026-11-10.

## Decision

A snapshot read from XLSX or ODS keeps the Gridcraft `Workbook` it
was converted from, in `WorkbookSnapshot::source`. GridTalk does
not extend the snapshot type field by field to carry more of the
file.

`to_gridcraft_workbook` starts from that workbook and rewrites only
the cells whose snapshot value differs from it. A rewritten cell
keeps its style, and a new cell takes its row or column style. If
the snapshot's sheets no longer match the source's (count or
names), the workbook is built from the snapshot alone, as before.

An edited table header renames its column, as in Excel and
Gridcraft's engine: an empty or repeated header gets a unique
generated name, and structured references follow the rename.
Without that, Gridcraft's writer would restore the old column name
on save. The snapshot's own formula text keeps the old name; only
the Gridcraft workbook is rewritten.

## Why

- The snapshot models values, formulas, cached values and defined
  names. A file also has styles, number formats, merges, tables and
  structured references, legacy and dynamic array formulas, error
  values, the 1904 date system, charts, pivots and pass-through
  parts. Before this decision, evaluation and Save saw none of
  those: a constant `#N/A` evaluated as the text "#N/A", and an
  XLSX save dropped every format.
- Anything Gridcraft supports keeps working without more
  conversion code in GridTalk, which is the point of vendoring it
  (0001).

## Consequences

- The CLAUDE.md rule "downstream code works on snapshots only" is
  relaxed. Code that needs file features may read `source`; code
  that edits a workbook still edits the snapshot, and the overlay
  carries the edit into Gridcraft.
- The source workbook stays in memory for as long as the snapshot
  does. It is not serialised (`#[serde(skip)]`).
- Saving an XLSX-sourced workbook as XLSX keeps what Gridcraft read.
  What it could not read is reported by its reader warnings when the
  file opens, so the UI no longer asks before an XLSX save.
- ODS stays data-only: Gridcraft's ODS reader imports formula
  results as constants, so the source has no formulas to keep.
