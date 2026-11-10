//! Typed tool core (Phase A of the agentic-interfaces plan).
//! One serde-typed implementation powers the investigate planner,
//! the `gridtalk tools call --json` CLI, and the MCP server —
//! external harnesses never scrape human-formatted strings.

use gridtalk_graph::eval::{EvalCache, Value};
use gridtalk_graph::{quote_sheet, sheet_name_eq, CellValue, Graph, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case", deny_unknown_fields)]
pub enum ToolRequest {
    Search {
        query: String,
        #[serde(default = "default_limit")]
        limit: usize,
    },
    Expand {
        range: String,
    },
    GetCells {
        range: String,
    },
    ListFormulas {
        range: String,
    },
    WhatIf {
        cell: String,
        value: f64,
        target: String,
    },
    GraphOverview,
    WorkbookInfo,
}

/// What went wrong in a failed tool call, so a client can branch on
/// the kind instead of matching message text. Every failure carries
/// `{"error": {"kind": ..., "message": ...}}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// An argument has the wrong type or value (also MCP-level
    /// schema violations).
    InvalidArguments,
    /// A range or cell reference does not parse, is over the cell
    /// limit, or lies outside the sheet grid.
    InvalidRange,
    /// No sheet of that name (sheet names match case-insensitively).
    UnknownSheet,
    /// No graph node covers the range.
    NoGraphNode,
    /// The calc engine could not evaluate the request.
    EvaluationFailed,
    /// MCP: no open workbook has that id.
    UnknownWorkbook,
    /// MCP: the workbook's file was deleted or became unreadable.
    WorkbookUnavailable,
    /// MCP: the file could not be opened or parsed.
    OpenFailed,
    /// MCP: no tool of that name.
    UnknownTool,
}

/// The `{"error": {"kind", "message"}}` payload of a failed call.
pub fn error_payload(kind: ErrorKind, message: &str) -> serde_json::Value {
    serde_json::json!({"error": {"kind": kind, "message": message}})
}

/// Largest `search` limit; larger requests are clamped (and say so).
pub const MAX_SEARCH_LIMIT: usize = 25;
/// Most edges `expand` returns; the rest are counted, not listed.
pub const MAX_EXPAND_EDGES: usize = 200;

fn default_limit() -> usize {
    3
}

impl ToolRequest {
    #[allow(dead_code)] // used by external tool-list consumers
    pub fn name(&self) -> &'static str {
        match self {
            ToolRequest::Search { .. } => "search",
            ToolRequest::Expand { .. } => "expand",
            ToolRequest::GetCells { .. } => "get_cells",
            ToolRequest::ListFormulas { .. } => "list_formulas",
            ToolRequest::WhatIf { .. } => "what_if",
            ToolRequest::GraphOverview => "graph_overview",
            ToolRequest::WorkbookInfo => "workbook_info",
        }
    }

    /// JSON Schemas for every tool, for MCP/tool-list consumers.
    /// Each entry carries a real JSON Schema (`inputSchema`), the
    /// `tool` discriminator value a `ToolRequest` JSON must use, and
    /// MCP annotations (every tool is read-only: `what_if` evaluates
    /// on a copy and never writes).
    pub fn schemas() -> serde_json::Value {
        let schema = |props: serde_json::Value, required: &[&str]| serde_json::json!({"type": "object", "properties": props, "required": required, "additionalProperties": false});
        let range = serde_json::json!({"type": "string", "description": RANGE_SYNTAX});
        let tool = |name: &str, description: String, input: serde_json::Value| {
            serde_json::json!({
                "name": name, "tool": name, "description": description, "inputSchema": input,
                "annotations": {"readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false}
            })
        };
        serde_json::json!({
            "request_format": "A ToolRequest is a JSON object with a \"tool\" discriminator, e.g. {\"tool\": \"search\", \"query\": \"...\", \"limit\": 3}. Unknown fields are rejected. A failed call returns {\"error\": {\"kind\": ..., \"message\": ...}}.",
            "tools": [
                tool("workbook_info", "Sheets (name, rows, cols), defined names, detected regions, load warnings and the tools' limits. Call it first to learn sheet names and ranges.".into(), schema(serde_json::json!({}), &[])),
                tool("search", format!("Hybrid lexical+embedding search over workbook regions. Returns ranked regions with a cited passage each. limit: 1-{MAX_SEARCH_LIMIT} (default 3; larger values are clamped and the response says so)."), schema(serde_json::json!({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "default": 3}}), &["query"])),
                tool("expand", format!("Graph neighbours (contains/depends_on edges) of the region or column containing a range, e.g. Sheet1!A1:D4. At most {MAX_EXPAND_EDGES} edges; total_edges and truncated tell when more exist."), schema(serde_json::json!({"range": range}), &["range"])),
                tool("get_cells", format!("Typed cell values in a range, e.g. Sheet1!A1:D4 or 'My Sheet'!B2. Formula cells carry their computed result and the file's cached value. At most {MAX_RANGE_CELLS} cells per call."), schema(serde_json::json!({"range": range}), &["range"])),
                tool("list_formulas", format!("Formulas in a range (e.g. Sheet1!A1:D40) with their evaluated values. At most {MAX_RANGE_CELLS} cells per call."), schema(serde_json::json!({"range": range}), &["range"])),
                tool("what_if", "Simulate setting one cell to a number and report a target cell before and after, e.g. cell Sheet1!B2, value 10, target Sheet1!D2. A cell without a sheet uses the target's sheet. Read-only: evaluates on a copy, never writes.".into(), schema(serde_json::json!({"cell": {"type": "string", "description": "The cell to override, e.g. Sheet1!B2"}, "value": {"type": "number"}, "target": {"type": "string", "description": "The cell to report, e.g. Sheet1!D2"}}), &["cell", "value", "target"])),
                tool("graph_overview", "Graph nodes and edges (capped at 500 nodes / 2000 edges; see total_nodes/total_edges/truncated)".into(), schema(serde_json::json!({}), &[]))
            ]
        })
    }
}

/// How ranges are written, for tool descriptions.
const RANGE_SYNTAX: &str = "An A1 range with its sheet: Sheet1!A1:D4, Sheet1!B2 (one cell), or 'Sheet Name'!A1:B2 (quote names with spaces or punctuation; double a quote inside: 'Bob''s'!A1). Sheet names match case-insensitively; cells run A1..XFD1048576.";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool: String,
    pub ok: bool,
    /// Structured payload for machines.
    pub data: serde_json::Value,
    /// Human rendering (CLI text mode, planner context).
    pub text: String,
}

impl ToolResult {
    fn ok(tool: &str, data: serde_json::Value, text: String) -> Self {
        ToolResult {
            tool: tool.into(),
            ok: true,
            data,
            text,
        }
    }
    fn err(tool: &str, kind: ErrorKind, message: String) -> Self {
        ToolResult {
            tool: tool.into(),
            ok: false,
            data: error_payload(kind, &message),
            text: message,
        }
    }
}

pub struct Tools<'a> {
    pub wb: &'a WorkbookSnapshot,
    pub graph: &'a Graph,
    pub hybrid: &'a HybridIndex,
    /// The workbook's recalculated engine, built on the first
    /// evaluating call and shared by the rest. It must belong to `wb`.
    pub calc: &'a EvalCache,
}

impl<'a> Tools<'a> {
    pub fn sheet_for(&self, name: &str) -> Option<&gridtalk_graph::SheetSnapshot> {
        self.sheet(name)
    }

    fn sheet(&self, name: &str) -> Option<&gridtalk_graph::SheetSnapshot> {
        // Caller spellings are case-insensitive (expand already
        // canonicalises sheet1!a1:b2; get_cells/list_formulas/what_if agree).
        self.wb.sheets.iter().find(|s| sheet_name_eq(&s.name, name))
    }

    /// The one typed entry point every surface calls.
    pub fn call(&self, req: &ToolRequest) -> ToolResult {
        match req {
            ToolRequest::Search {
                query,
                limit: requested,
            } => {
                let limit = (*requested).clamp(1, MAX_SEARCH_LIMIT);
                let clamped = limit != *requested;
                let limit_note = if clamped {
                    format!(
                        "\n(limit {requested} clamped to {limit}; allowed 1-{MAX_SEARCH_LIMIT})"
                    )
                } else {
                    String::new()
                };
                let hits = self.hybrid.search(query, limit);
                if hits.is_empty() {
                    return ToolResult::ok(
                        "search",
                        serde_json::json!({"hits": [], "limit": limit, "limit_clamped": clamped}),
                        format!("no regions matched \"{query}\"{limit_note}"),
                    );
                }
                let data_hits: Vec<serde_json::Value> = hits
                    .iter()
                    .map(|h| {
                        serde_json::json!({
                            "range": h.range.a1(),
                            "score": h.score,
                            "vector_only": h.vector_only,
                            "matched_terms": h.matched_terms,
                            "passage": h.passage,
                            "truncated": h.truncated,
                        })
                    })
                    .collect();
                let text = hits
                    .iter()
                    .map(|h| format!("[{}] score {:.3}", h.range.a1(), h.score))
                    .collect::<Vec<_>>()
                    .join("\n")
                    + &limit_note;
                ToolResult::ok(
                    "search",
                    serde_json::json!({"hits": data_hits, "limit": limit, "limit_clamped": clamped}),
                    text,
                )
            }
            ToolRequest::Expand { range } => {
                // Accept caller spellings (sheet1!a1:b2, $A$1) by
                // canonicalising through the range parser first.
                let canonical = parse_range_uncapped(range).map(|(sheet, r0, c0, r1, c1)| {
                    // Use the workbook's own sheet spelling so the
                    // exact finds below match node ranges.
                    let sheet = self
                        .sheet(&sheet)
                        .map(|sh| sh.name.clone())
                        .unwrap_or(sheet);
                    gridtalk_graph::RangeRef {
                        sheet,
                        r0,
                        c0,
                        r1,
                        c1,
                    }
                    .a1()
                });
                let wanted = canonical.as_deref().unwrap_or(range.as_str());
                let nodes_by_id: std::collections::HashMap<usize, &gridtalk_graph::Node> =
                    self.graph.nodes.iter().map(|n| (n.id, n)).collect();
                let node =
                    self.graph
                        .nodes
                        .iter()
                        .filter(|n| {
                            matches!(
                                n.kind,
                                gridtalk_graph::NodeKind::Region | gridtalk_graph::NodeKind::Column
                            )
                        })
                        .find(|n| n.range.as_ref().map(|r| r.a1()) == Some(wanted.to_string()))
                        .or_else(|| {
                            self.graph.nodes.iter().find(|n| {
                                n.range.as_ref().map(|r| r.a1()) == Some(wanted.to_string())
                            })
                        })
                        .or_else(|| {
                            // "Region containing this range": smallest
                            // Region/Column whose range covers it.
                            let (sheet, r0, c0, r1, c1) = parse_range_uncapped(range)?;
                            self.graph
                                .nodes
                                .iter()
                                .filter(|n| {
                                    matches!(
                                        n.kind,
                                        gridtalk_graph::NodeKind::Region
                                            | gridtalk_graph::NodeKind::Column
                                    )
                                })
                                .filter(|n| {
                                    n.range.as_ref().is_some_and(|r| {
                                        sheet_name_eq(&r.sheet, &sheet)
                                            && r.r0 <= r0
                                            && r.c0 <= c0
                                            && r.r1 >= r1
                                            && r.c1 >= c1
                                    })
                                })
                                .min_by_key(|n| {
                                    n.range
                                        .as_ref()
                                        .map(|r| (r.r1 - r.r0 + 1) * (r.c1 - r.c0 + 1))
                                        .unwrap_or(usize::MAX)
                                })
                        });
                match node {
                    Some(n) => {
                        let mut edges = Vec::new();
                        let mut lines = vec![format!("node {} ({})", n.id, n.label)];
                        // Counted per listed entry: a self-edge (a
                        // running total) is listed both out and in, and
                        // an edge to a missing node is not listed.
                        let mut total_edges = 0usize;
                        for e in &self.graph.edges {
                            let sides = [
                                (e.from == n.id, e.to, "out", "-", "->"),
                                (e.to == n.id, e.from, "in", "<-", "-"),
                            ];
                            for (touches, other, direction, left, right) in sides {
                                if !touches {
                                    continue;
                                }
                                let Some(t) = nodes_by_id.get(&other) else {
                                    continue;
                                };
                                total_edges += 1;
                                if edges.len() >= MAX_EXPAND_EDGES {
                                    continue;
                                }
                                lines.push(format!(" {left}{}{right} {}", e.kind, t.label));
                                edges.push(serde_json::json!({
                                    "direction": direction, "kind": e.kind, "label": t.label,
                                    "range": t.range.as_ref().map(|r| r.a1())
                                }));
                            }
                        }
                        let truncated = total_edges > edges.len();
                        if truncated {
                            lines.push(format!(
                                "({} of {total_edges} edges shown; the cap is {MAX_EXPAND_EDGES})",
                                edges.len()
                            ));
                        }
                        ToolResult::ok(
                            "expand",
                            serde_json::json!({"node_id": n.id, "label": n.label, "range": n.range.as_ref().map(|r| r.a1()), "edges": edges, "total_edges": total_edges, "truncated": truncated}),
                            lines.join("\n"),
                        )
                    }
                    None => ToolResult::err(
                        "expand",
                        ErrorKind::NoGraphNode,
                        format!("no graph node for range {range}"),
                    ),
                }
            }
            ToolRequest::GetCells { range } => match parse_range(range) {
                Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                    Some(sheet) => {
                        // Formula cells report what they compute, not
                        // only the file's cached value: one batch
                        // through the workbook's calc engine.
                        let formulas: Vec<(usize, usize)> = (r0..=r1)
                            .flat_map(|r| (c0..=c1).map(move |c| (r, c)))
                            .filter(|(r, c)| {
                                matches!(sheet.get_ref(*r, *c), CellValue::Formula { .. })
                            })
                            .collect();
                        let mut results = if formulas.is_empty() {
                            Vec::new()
                        } else {
                            self.calc.get(self.wb).values(&sheet.name, &formulas)
                        }
                        .into_iter();
                        let mut rows = Vec::new();
                        let mut text = format!("Range {range}\n");
                        for r in r0..=r1 {
                            let mut row = Vec::new();
                            for c in c0..=c1 {
                                let cell = sheet.get(r, c);
                                match &cell {
                                    CellValue::Formula { expr, cached } => {
                                        let result =
                                            results.next().expect("one result per formula");
                                        text.push_str(&match &result {
                                            Ok(v) => v.display(),
                                            Err(e) => format!("#ERROR({e})"),
                                        });
                                        row.push(formula_json(expr, cached.as_deref(), &result));
                                    }
                                    _ => {
                                        text.push_str(&cell.display());
                                        row.push(cell_json(&cell));
                                    }
                                }
                                if c < c1 {
                                    text.push_str(" | ");
                                }
                            }
                            text.push('\n');
                            rows.push(serde_json::Value::Array(row));
                        }
                        ToolResult::ok(
                            "get_cells",
                            serde_json::json!({"range": range, "rows": rows}),
                            text,
                        )
                    }
                    None => ToolResult::err(
                        "get_cells",
                        ErrorKind::UnknownSheet,
                        format!("no sheet named {sheet_name}"),
                    ),
                },
                None => ToolResult::err("get_cells", ErrorKind::InvalidRange, range_error(range)),
            },
            ToolRequest::ListFormulas { range } => match parse_range(range) {
                Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                    Some(sheet) => {
                        // One shared evaluator memo across the
                        // whole range, not a fresh one per cell.
                        let coords: Vec<(usize, usize)> = (r0..=r1)
                            .flat_map(|r| (c0..=c1).map(move |c| (r, c)))
                            .filter(|(r, c)| matches!(sheet.get(*r, *c), CellValue::Formula { .. }))
                            .collect();
                        // One recalculation for the whole batch, not
                        // one per cell.
                        let evaluated = self.calc.get(self.wb).values(&sheet.name, &coords);
                        let mut eval_iter = coords.iter().zip(evaluated.iter());
                        let mut formulas = Vec::new();
                        let mut lines = Vec::new();
                        for r in r0..=r1 {
                            for c in c0..=c1 {
                                if let CellValue::Formula { expr, .. } = sheet.get(r, c) {
                                    let (_, result) = eval_iter.next().unwrap();
                                    let cell = format!("{}{}", gridtalk_graph::col_name(c), r + 1);
                                    let (value, typed, error) = match result {
                                        Ok(v) => (legacy_number(v), Some(value_json(v)), None),
                                        Err(e) => (None, None, Some(e.to_string())),
                                    };
                                    lines.push(match result {
                                        Ok(v) => format!("{cell} = {expr} -> {}", v.display()),
                                        Err(e) => format!("{cell} = {expr} -> error: {e}"),
                                    });
                                    formulas.push(serde_json::json!({
                                        "cell": cell, "formula": expr, "value": value,
                                        "result": typed, "error": error
                                    }));
                                }
                            }
                        }
                        if formulas.is_empty() {
                            ToolResult::ok(
                                "list_formulas",
                                serde_json::json!({"formulas": []}),
                                "no formulas in range".into(),
                            )
                        } else {
                            ToolResult::ok(
                                "list_formulas",
                                serde_json::json!({"formulas": formulas}),
                                lines.join("\n"),
                            )
                        }
                    }
                    None => ToolResult::err(
                        "list_formulas",
                        ErrorKind::UnknownSheet,
                        format!("no sheet named {sheet_name}"),
                    ),
                },
                None => {
                    ToolResult::err("list_formulas", ErrorKind::InvalidRange, range_error(range))
                }
            },
            ToolRequest::WhatIf {
                cell,
                value,
                target,
            } => {
                if !value.is_finite() {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::InvalidArguments,
                        "what-if value must be finite".into(),
                    );
                }
                let Some((target_sheet, tr, tc)) = parse_cell_ref(target) else {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::InvalidRange,
                        cell_error("target", target),
                    );
                };
                // The override may name its own sheet (Other!B2);
                // it must never silently land on the target sheet.
                let qualified = if cell.contains('!') {
                    cell.clone()
                } else {
                    format!("{}!{cell}", gridtalk_graph::quote_sheet(&target_sheet))
                };
                let Some((override_sheet, orow, ocol)) = parse_cell_ref(&qualified) else {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::InvalidRange,
                        cell_error("cell", cell),
                    );
                };
                if self.sheet(&override_sheet).is_none() {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::UnknownSheet,
                        format!("no sheet named {override_sheet} for override cell {cell}"),
                    );
                }
                if self.sheet(&target_sheet).is_none() {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::UnknownSheet,
                        format!("no sheet named {target_sheet}"),
                    );
                }
                // Canonicalise to the workbook's own sheet names:
                // callers may write sheet1!a1 in any case, and the
                // evaluator matches sheet names exactly.
                let override_sheet = self.sheet(&override_sheet).unwrap().name.clone();
                let target_sheet = self.sheet(&target_sheet).unwrap().name.clone();
                match self.calc.get(self.wb).what_if(
                    (&override_sheet, orow, ocol),
                    *value,
                    (&target_sheet, tr, tc),
                ) {
                    Ok((before, after)) => ToolResult::ok(
                        "what_if",
                        serde_json::json!({
                            "cell": cell, "value": value, "target": target,
                            "before": legacy_number(&before), "after": legacy_number(&after),
                            "before_result": value_json(&before),
                            "after_result": value_json(&after)
                        }),
                        format!(
                            "what-if {cell}={value}: {target} changes {} -> {}",
                            before.display(),
                            after.display()
                        ),
                    ),
                    Err(e) => ToolResult::err(
                        "what_if",
                        ErrorKind::EvaluationFailed,
                        format!("what-if failed: {e}"),
                    ),
                }
            }
            ToolRequest::WorkbookInfo => {
                let sheets: Vec<serde_json::Value> = self
                    .wb
                    .sheets
                    .iter()
                    .map(|s| serde_json::json!({"name": s.name, "rows": s.cells.len(), "cols": s.max_col()}))
                    .collect();
                let names: Vec<serde_json::Value> = self
                    .wb
                    .names
                    .iter()
                    .map(|n| serde_json::json!({"name": n.name, "scope": n.scope, "refers_to": n.formula}))
                    .collect();
                let regions: Vec<serde_json::Value> = self
                    .graph
                    .regions
                    .iter()
                    .map(|r| serde_json::json!({"range": r.range.a1(), "headers": r.headers}))
                    .collect();
                let mut lines = vec![format!("workbook {}", self.wb.name)];
                for s in &self.wb.sheets {
                    lines.push(format!(
                        "sheet {} ({} rows x {} cols)",
                        quote_sheet(&s.name),
                        s.cells.len(),
                        s.max_col()
                    ));
                }
                for n in &self.wb.names {
                    lines.push(format!("name {} = {}", n.name, n.formula));
                }
                for r in &self.graph.regions {
                    lines.push(format!("region {}: {}", r.range.a1(), r.headers.join(", ")));
                }
                for w in &self.wb.warnings {
                    lines.push(format!("warning: {w}"));
                }
                ToolResult::ok(
                    "workbook_info",
                    serde_json::json!({
                        "name": self.wb.name, "sheets": sheets, "defined_names": names,
                        "regions": regions, "warnings": self.wb.warnings,
                        "limits": {"range_cells": MAX_RANGE_CELLS, "search_limit": MAX_SEARCH_LIMIT, "expand_edges": MAX_EXPAND_EDGES, "grid_end": "XFD1048576"}
                    }),
                    lines.join("\n"),
                )
            }
            ToolRequest::GraphOverview => {
                const MAX_OVERVIEW_NODES: usize = 500;
                const MAX_OVERVIEW_EDGES: usize = 2000;
                let total_nodes = self.graph.nodes.len();
                let total_edges = self.graph.edges.len();
                let nodes: Vec<serde_json::Value> = self
                    .graph
                    .nodes
                    .iter()
                    .take(MAX_OVERVIEW_NODES)
                    .map(|n| {
                        serde_json::json!({
                            "id": n.id, "kind": format!("{:?}", n.kind), "label": n.label,
                            "range": n.range.as_ref().map(|r| r.a1())
                        })
                    })
                    .collect();
                let edges: Vec<serde_json::Value> = self
                    .graph
                    .edges
                    .iter()
                    .take(MAX_OVERVIEW_EDGES)
                    .map(|e| serde_json::json!({"from": e.from, "to": e.to, "kind": e.kind}))
                    .collect();
                let truncated = total_nodes > nodes.len() || total_edges > edges.len();
                let text = if truncated {
                    format!(
                        "{total_nodes} nodes, {total_edges} edges (showing first {} nodes, {} edges)",
                        nodes.len(),
                        edges.len()
                    )
                } else {
                    format!("{} nodes, {} edges", nodes.len(), edges.len())
                };
                ToolResult::ok(
                    "graph_overview",
                    serde_json::json!({"nodes": nodes, "edges": edges, "total_nodes": total_nodes, "total_edges": total_edges, "truncated": truncated}),
                    text,
                )
            }
        }
    }

    /// Legacy string interface kept for the planner's text context:
    /// parses `key=value;key=value` and renders the typed result.
    pub fn execute(&self, tool: &str, args: &str) -> String {
        let arg = |key: &str| -> Option<String> {
            split_legacy_args(args).iter().find_map(|kv| {
                let (k, v) = kv.split_once('=')?;
                if k.trim() == key {
                    Some(v.trim().to_string())
                } else {
                    None
                }
            })
        };
        let req = match tool {
            "search" => {
                let q = args
                    .strip_prefix("q=")
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| arg("q").unwrap_or_default());
                ToolRequest::Search {
                    query: q,
                    limit: 3,
                }
            }
            "expand" => ToolRequest::Expand {
                range: arg("range").unwrap_or_default(),
            },
            "get_cells" => ToolRequest::GetCells {
                range: arg("range").unwrap_or_default(),
            },
            "list_formulas" => ToolRequest::ListFormulas {
                range: arg("range").unwrap_or_default(),
            },
            "what_if" => {
                let (Some(cell), Some(value), Some(target)) = (
                    arg("cell"),
                    arg("value").and_then(|v| v.parse::<f64>().ok()),
                    arg("target"),
                ) else {
                    return "usage: cell=B2;value=10;target=Sheet1!D4".into();
                };
                ToolRequest::WhatIf {
                    cell,
                    value,
                    target,
                }
            }
            "graph_overview" => ToolRequest::GraphOverview,
            "workbook_info" => ToolRequest::WorkbookInfo,
            other => {
                return format!(
                    "unknown tool {other}; available: workbook_info, search, expand, get_cells, list_formulas, what_if, graph_overview"
                )
            }
        };
        self.call(&req).text
    }
}

/// Legacy `key=value;key=value` splitting that does not split on
/// semicolons inside single-quoted sheet names.
fn split_legacy_args(args: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    let mut chars = args.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\'' {
            if in_quote && chars.peek() == Some(&'\'') {
                cur.push(c);
                cur.push(chars.next().unwrap());
                continue;
            }
            in_quote = !in_quote;
            cur.push(c);
        } else if c == ';' && !in_quote {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

/// A formula result in the same typed shape `get_cells` uses.
fn value_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Number(n) => serde_json::json!({"type": "number", "value": n}),
        Value::Text(s) => serde_json::json!({"type": "text", "value": s}),
        Value::Bool(b) => serde_json::json!({"type": "bool", "value": b}),
        Value::Empty => serde_json::json!({"type": "empty"}),
    }
}

/// The numeric field (`list_formulas` `value`, `what_if`
/// `before`/`after`) these tools have always reported: booleans as
/// 1/0, text as no number (the typed `*result` field carries it).
fn legacy_number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => Some(*n),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::Empty => Some(0.0),
        Value::Text(_) => None,
    }
}

/// A formula cell for `get_cells`: its text, what it computes
/// (`result`, typed like `list_formulas`; `null` with `error` when
/// evaluation failed), and the value the file cached (`cached_result`,
/// typed; `cached` keeps the display string older callers read).
fn formula_json(
    expr: &str,
    cached: Option<&CellValue>,
    result: &Result<Value, gridtalk_graph::eval::EvalError>,
) -> serde_json::Value {
    let (typed, error) = match result {
        Ok(v) => (Some(value_json(v)), None),
        Err(e) => (None, Some(e.to_string())),
    };
    serde_json::json!({
        "type": "formula", "formula": expr,
        "result": typed, "error": error,
        "cached": cached.map(|c| c.display()),
        "cached_result": cached.map(cell_json),
    })
}

fn cell_json(cell: &CellValue) -> serde_json::Value {
    match cell {
        CellValue::Empty => serde_json::json!({"type": "empty"}),
        CellValue::Text(s) => serde_json::json!({"type": "text", "value": s}),
        CellValue::Number(n) => serde_json::json!({"type": "number", "value": n}),
        CellValue::Bool(b) => serde_json::json!({"type": "bool", "value": b}),
        CellValue::Formula { expr, cached } => serde_json::json!({
            "type": "formula", "formula": expr,
            "cached": cached.as_ref().map(|c| c.display())
        }),
    }
}

/// Largest range a GetCells/ListFormulas call may materialise.
pub const MAX_RANGE_CELLS: usize = 10_000;

/// Split `Sheet!rest` or `'My Sheet'!rest` into (sheet, rest),
/// honouring the doubled-quote escape inside quoted names.
/// Distinguish "too large" from "unparseable" for range errors.
fn range_error(range: &str) -> String {
    if parse_range_uncapped(range).is_some() {
        format!("range {range} exceeds the {MAX_RANGE_CELLS}-cell limit; narrow the range")
    } else if parse_range_inner(range, false, false).is_some() {
        format!("range {range} is outside the sheet grid, which ends at XFD1048576")
    } else {
        format!("cannot parse range {range}")
    }
}

/// Why a what-if cell or target did not parse.
fn cell_error(what: &str, a1: &str) -> String {
    let outside = split_sheet_prefix(a1)
        .or_else(|| Some((String::new(), a1)))
        .and_then(|(sheet, cell)| parse_a1(&sheet, cell))
        .is_some();
    if outside {
        format!("{what} {a1} is outside the sheet grid, which ends at XFD1048576")
    } else {
        format!("cannot parse {what} {a1}")
    }
}

fn split_sheet_prefix(a1: &str) -> Option<(String, &str)> {
    if let Some(rest) = a1.strip_prefix('\'') {
        let mut name = String::new();
        let mut idx = 0;
        let bytes = rest.as_bytes();
        while idx < bytes.len() {
            if bytes[idx] == b'\'' {
                if bytes.get(idx + 1) == Some(&b'\'') {
                    name.push('\'');
                    idx += 2;
                    continue;
                }
                // Closing quote must be followed by '!'.
                let after = &rest[idx + 1..];
                return after.strip_prefix('!').map(|r| (name, r));
            }
            // Push one char (byte-walk is safe: we only split on
            // ASCII quote/bang boundaries).
            let ch_len = rest[idx..].chars().next()?.len_utf8();
            name.push_str(&rest[idx..idx + ch_len]);
            idx += ch_len;
        }
        None
    } else {
        let (sheet, rest) = a1.split_once('!')?;
        Some((sheet.to_string(), rest))
    }
}

/// An A1 cell as 0-based (row, col), anywhere inside the sheet grid
/// (`A1`..`XFD1048576`).
fn parse_cell_on_sheet(sheet: &str, cell: &str) -> Option<(usize, usize)> {
    parse_a1(sheet, cell)
        .filter(|&(r, c)| r < gridtalk_graph::EXCEL_MAX_ROWS && c < gridtalk_graph::EXCEL_MAX_COLS)
}

/// An A1 cell as 0-based (row, col), with no grid bound.
fn parse_a1(sheet: &str, cell: &str) -> Option<(usize, usize)> {
    let cell_owned = cell.replace('$', "");
    let cell = cell_owned.as_str();
    let split = cell.find(|c: char| c.is_ascii_digit())?;
    let (letters, digits) = cell.split_at(split);
    let row: usize = digits.parse().ok()?;
    if row == 0 || letters.is_empty() {
        return None;
    }
    let mut col = 0usize;
    for ch in letters.to_ascii_uppercase().chars() {
        if !ch.is_ascii_alphabetic() {
            return None;
        }
        col = col
            .checked_mul(26)?
            .checked_add(ch as usize - 'A' as usize + 1)?;
    }
    let _ = sheet;
    Some((row - 1, col - 1))
}

/// `Sheet1!A1:D4` -> (sheet, r0, c0, r1, c1), 0-based inclusive.
/// A single cell is a 1x1 range; reversed ranges and ranges over
/// MAX_RANGE_CELLS are rejected. Quoted sheet names and a
/// qualified range end (`Sheet1!A1:Sheet1!B2`) are accepted.
pub fn parse_range(a1: &str) -> Option<(String, usize, usize, usize, usize)> {
    parse_range_inner(a1, true, true)
}

/// Range parsing without the materialisation cap (for Expand,
/// which never materialises cells).
pub fn parse_range_uncapped(a1: &str) -> Option<(String, usize, usize, usize, usize)> {
    parse_range_inner(a1, false, true)
}

/// `capped`: at most MAX_RANGE_CELLS. `grid`: inside the sheet grid
/// (off only to tell an out-of-grid range from an unparseable one).
fn parse_range_inner(
    a1: &str,
    capped: bool,
    grid: bool,
) -> Option<(String, usize, usize, usize, usize)> {
    let parse_cell_on_sheet = if grid { parse_cell_on_sheet } else { parse_a1 };
    let (sheet, rest) = split_sheet_prefix(a1)?;
    if let Some((a, b)) = rest.split_once(':') {
        let (r0, c0) = parse_cell_on_sheet(&sheet, a)?;
        // The end may repeat the sheet qualifier.
        let b_cell = if b.contains('!') {
            let (end_sheet, cell) = split_sheet_prefix(b)?;
            if !sheet_name_eq(&end_sheet, &sheet) {
                return None;
            }
            cell
        } else {
            b
        };
        let (r1, c1) = parse_cell_on_sheet(&sheet, b_cell)?;
        if r1 < r0 || c1 < c0 {
            return None;
        }
        if capped && (r1 - r0 + 1).saturating_mul(c1 - c0 + 1) > MAX_RANGE_CELLS {
            return None;
        }
        Some((sheet, r0, c0, r1, c1))
    } else {
        let (r, c) = parse_cell_on_sheet(&sheet, rest)?;
        Some((sheet, r, c, r, c))
    }
}

/// `Sheet1!B2` -> (sheet, row, col), 0-based.
pub fn parse_cell_ref(a1: &str) -> Option<(String, usize, usize)> {
    let (sheet, cell) = split_sheet_prefix(a1)?;
    let (r, c) = parse_cell_on_sheet(&sheet, cell)?;
    Some((sheet, r, c))
}
#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::{build_graph, parse_delimited};

    /// The range a truncation note suggests must work as a `get_cells`
    /// call, including on a region over the tool's range limit and a
    /// sheet name that needs quoting.
    #[test]
    fn truncation_note_range_works_as_a_get_cells_call() {
        let mut csv = String::from("Account,Territory,Balance\n");
        for i in 0..5000 {
            csv.push_str(&format!("BG-{i},Ironwood,{i}.5\n"));
        }
        let wb = parse_delimited("big", "Debtors Ledger", &csv, ',');
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let whole = tools.call(&ToolRequest::GetCells {
            range: g.regions[0].range.a1(),
        });
        assert!(!whole.ok, "the region itself is over the range limit");

        let hit = &hybrid.search("ironwood", 1)[0];
        let (_, next) = hit
            .passage
            .split_once("next: get_cells on ")
            .expect("note names a next range");
        let next = next.trim_end().trim_end_matches(']');
        let cells = tools.call(&ToolRequest::GetCells { range: next.into() });
        assert!(cells.ok, "{}", cells.text);
        let rows = cells.data["rows"].as_array().expect("rows");
        assert_eq!(rows.len(), 100);
        // Picks up straight after the last row the excerpt showed.
        let last_shown = hit
            .passage
            .lines()
            .rfind(|l| l.starts_with("row "))
            .expect("rows shown");
        let number: usize = last_shown
            .trim_start_matches("row ")
            .split(':')
            .next()
            .and_then(|n| n.parse().ok())
            .expect("row label");
        assert_eq!(rows[0][0]["value"], format!("BG-{}", number - 1));
    }

    fn fixture() -> (WorkbookSnapshot, Graph, HybridIndex) {
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\nProvision,1,1200,=B4*C4\n",
            ',',
        );
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty());
        let hybrid = HybridIndex::build(&wb, &g);
        (wb, g, hybrid)
    }

    #[test]
    fn text_and_boolean_results_are_typed() {
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Label,Big\nApples,2,=A2&\" x\"&B2,=B2>5\nPears,=1/0,,\n",
            ',',
        );
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let res = tools.call(&ToolRequest::ListFormulas {
            range: "Sheet1!A1:D3".into(),
        });
        let f = &res.data["formulas"];
        // C2: text, no legacy number and no error.
        assert_eq!(f[0]["cell"], "C2");
        assert!(f[0]["value"].is_null());
        assert!(f[0]["error"].is_null());
        assert_eq!(f[0]["result"]["type"], "text");
        assert_eq!(f[0]["result"]["value"], "Apples x2");
        // D2: boolean, legacy 0.
        assert_eq!(f[1]["value"], 0.0);
        assert_eq!(f[1]["result"]["type"], "bool");
        assert_eq!(f[1]["result"]["value"], false);
        // B3: a genuine error has no result.
        assert_eq!(f[2]["cell"], "B3");
        assert!(f[2]["result"].is_null());
        assert!(f[2]["error"].as_str().unwrap().contains("division"));
        assert!(
            res.text.contains("C2 = A2&\" x\"&B2 -> Apples x2"),
            "{}",
            res.text
        );
        assert!(res.text.contains("D2 = B2>5 -> FALSE"), "{}", res.text);

        let whatif = tools.call(&ToolRequest::WhatIf {
            cell: "B2".into(),
            value: 7.0,
            target: "Sheet1!C2".into(),
        });
        assert!(whatif.ok, "{}", whatif.text);
        assert!(whatif.data["before"].is_null());
        assert_eq!(whatif.data["before_result"]["value"], "Apples x2");
        assert_eq!(whatif.data["after_result"]["value"], "Apples x7");
        assert!(
            whatif.text.contains("Apples x2 -> Apples x7"),
            "{}",
            whatif.text
        );
        let whatif = tools.call(&ToolRequest::WhatIf {
            cell: "B2".into(),
            value: 7.0,
            target: "Sheet1!D2".into(),
        });
        assert_eq!(whatif.data["before"], 0.0);
        assert_eq!(whatif.data["after"], 1.0);
        assert_eq!(whatif.data["after_result"]["value"], true);
    }

    #[test]
    fn get_cells_reports_computed_and_cached_values_inside_the_grid() {
        // D2's cached 99 is stale: it computes 2*3.
        let mut wb = parse_delimited(
            "d",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\n",
            ',',
        );
        *wb.sheets[0].cell_mut(1, 3) = CellValue::Formula {
            expr: "B2*C2".into(),
            cached: Some(Box::new(CellValue::Number(99.0))),
        };
        *wb.sheets[0].cell_mut(1, 4) = CellValue::Formula {
            expr: "1/0".into(),
            cached: None,
        };
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let out = tools.call(&ToolRequest::GetCells {
            range: "Sheet1!B2:E2".into(),
        });
        assert!(out.ok, "{}", out.text);
        let d2 = &out.data["rows"][0][2];
        assert_eq!(d2["type"], "formula");
        assert_eq!(
            d2["result"],
            serde_json::json!({"type": "number", "value": 6.0})
        );
        assert_eq!(
            d2["cached_result"],
            serde_json::json!({"type": "number", "value": 99.0})
        );
        assert_eq!(d2["cached"], "99");
        assert!(d2["error"].is_null());
        let e2 = &out.data["rows"][0][3];
        assert!(e2["result"].is_null() && e2["error"].is_string());
        assert!(e2["cached_result"].is_null());
        assert!(out.text.contains("2 | 3 | 6 | #ERROR("), "{}", out.text);
        // The grid ends at XFD1048576.
        assert!(
            tools
                .call(&ToolRequest::GetCells {
                    range: "Sheet1!XFD1048576".into()
                })
                .ok
        );
        for range in ["Sheet1!XFE1", "Sheet1!A1048577", "Sheet1!A1:A1048577"] {
            let bad = tools.call(&ToolRequest::GetCells {
                range: range.into(),
            });
            assert!(!bad.ok);
            assert!(
                bad.text.contains("outside the sheet grid"),
                "{range}: {}",
                bad.text
            );
        }
        assert!(tools
            .call(&ToolRequest::GetCells {
                range: "Sheet1!1A".into()
            })
            .text
            .contains("cannot parse"));
        let what_if = tools.call(&ToolRequest::WhatIf {
            cell: "Sheet1!ZZZZ1".into(),
            value: 1.0,
            target: "Sheet1!D2".into(),
        });
        assert!(
            what_if.text.contains("outside the sheet grid"),
            "{}",
            what_if.text
        );
    }

    #[test]
    fn expand_is_capped_and_says_so() {
        let header: Vec<String> = (0..250).map(|i| format!("h{i}")).collect();
        let row: Vec<String> = (0..250).map(|i| i.to_string()).collect();
        let csv = format!("{}\n{}\n", header.join(","), row.join(","));
        let wb = parse_delimited("w", "Sheet1", &csv, ',');
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let region = &g.regions[0].range.a1();
        let out = tools.call(&ToolRequest::Expand {
            range: region.clone(),
        });
        assert!(out.ok, "{}", out.text);
        assert_eq!(
            out.data["edges"].as_array().unwrap().len(),
            MAX_EXPAND_EDGES
        );
        assert_eq!(out.data["total_edges"], 251); // 250 columns + its sheet
        assert_eq!(out.data["truncated"], true);
        assert!(out.text.contains("200 of 251 edges shown"), "{}", out.text);
        // A running total's self-edge is listed out and in, and both
        // count toward total_edges.
        let wb2 = parse_delimited("r", "Sheet1", "Amount,Running\n5,=A2\n7,=B2+A3\n", ',');
        let (g2, v2) = build_graph(&wb2);
        assert!(v2.is_empty(), "{v2:?}");
        let hybrid2 = HybridIndex::build(&wb2, &g2);
        let tools2 = Tools {
            wb: &wb2,
            graph: &g2,
            hybrid: &hybrid2,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let running = g2.nodes.iter().find(|n| n.label == "Running").unwrap();
        let out2 = tools2.call(&ToolRequest::Expand {
            range: running.range.as_ref().unwrap().a1(),
        });
        let listed = out2.data["edges"].as_array().unwrap().len();
        assert_eq!(out2.data["total_edges"], listed);
        assert_eq!(out2.data["truncated"], false);
        let info = tools.call(&ToolRequest::WorkbookInfo);
        assert_eq!(info.data["sheets"][0]["cols"], 250);
        assert_eq!(info.data["limits"]["expand_edges"], MAX_EXPAND_EDGES);
    }

    #[test]
    fn typed_calls_return_structured_data() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let search = tools.call(&ToolRequest::Search {
            query: "provision".into(),
            limit: 3,
        });
        assert!(search.ok);
        assert_eq!(search.data["hits"][0]["range"], "Sheet1!A1:D3");
        let formulas = tools.call(&ToolRequest::ListFormulas {
            range: "Sheet1!A1:D3".into(),
        });
        assert_eq!(formulas.data["formulas"][0]["value"], 6.0);
        assert_eq!(formulas.data["formulas"][0]["result"]["type"], "number");
        assert_eq!(formulas.data["formulas"][0]["result"]["value"], 6.0);
        let whatif = tools.call(&ToolRequest::WhatIf {
            cell: "B2".into(),
            value: 10.0,
            target: "Sheet1!D2".into(),
        });
        assert_eq!(whatif.data["before"], 6.0);
        assert_eq!(whatif.data["after"], 30.0);
        assert_eq!(whatif.data["after_result"]["type"], "number");
        assert_eq!(whatif.data["after_result"]["value"], 30.0);
        // Legacy wrapper agrees with the typed path.
        assert!(tools.execute("search", "q=provision").contains("Sheet1!"));
        // An override naming another sheet is honoured (and a
        // nonexistent override sheet is an error, not a silent
        // landing on the target sheet).
        let bad = tools.call(&ToolRequest::WhatIf {
            cell: "Nope!B2".into(),
            value: 1.0,
            target: "Sheet1!D2".into(),
        });
        assert!(!bad.ok);
        assert!(bad.text.contains("Nope"), "got: {}", bad.text);
        assert!(
            parse_range("Sheet1!A1:ZZ100000").is_none(),
            "huge ranges are capped"
        );
        let schemas = ToolRequest::schemas();
        assert_eq!(schemas["tools"][0]["inputSchema"]["type"], "object");
        assert!(schemas["request_format"]
            .as_str()
            .unwrap()
            .contains("\"tool\""));

        // Bad input is an ok:false result, never a panic.
        assert!(
            !tools
                .call(&ToolRequest::GetCells {
                    range: "nonsense".into()
                })
                .ok
        );
    }
}
