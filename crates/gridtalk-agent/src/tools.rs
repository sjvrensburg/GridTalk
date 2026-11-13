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
    Issues {
        /// Check only this range (one sheet); the whole workbook,
        /// defined names included, when absent.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        range: Option<String>,
    },
    Filter {
        /// A table's range, or any cell or column in it.
        range: String,
        /// Conditions every returned row meets; none returns every
        /// row.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        r#where: Vec<Condition>,
        /// The columns to show, by heading or letter; all when empty.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        columns: Vec<String>,
        #[serde(default = "default_filter_limit")]
        limit: usize,
    },
    Aggregate {
        /// A table's range, or any cell or column in it.
        range: String,
        /// The column to add up; every column when absent (and no
        /// group_by).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        column: Option<String>,
        /// The column whose values form the groups.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        group_by: Option<String>,
        /// Only the rows meeting every condition, as in `filter`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        r#where: Vec<Condition>,
    },
}

/// One `filter` condition: a column (by heading, or by letter), a
/// comparison, and the value to compare with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Condition {
    pub column: String,
    pub op: FilterOp,
    /// Written as text; a number or boolean is accepted and read as
    /// its text.
    #[serde(deserialize_with = "text_or_scalar")]
    pub value: String,
}

/// How a `filter` condition compares. Numbers compare as numbers
/// when both sides are numbers; anything else compares as text,
/// ignoring case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    #[serde(rename = "=", alias = "==")]
    Eq,
    #[serde(rename = "!=", alias = "<>")]
    Ne,
    #[serde(rename = ">")]
    Gt,
    #[serde(rename = ">=")]
    Ge,
    #[serde(rename = "<")]
    Lt,
    #[serde(rename = "<=")]
    Le,
    #[serde(rename = "contains")]
    Contains,
}

impl FilterOp {
    pub const ALL: [FilterOp; 7] = [
        FilterOp::Eq,
        FilterOp::Ne,
        FilterOp::Gt,
        FilterOp::Ge,
        FilterOp::Lt,
        FilterOp::Le,
        FilterOp::Contains,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            FilterOp::Eq => "=",
            FilterOp::Ne => "!=",
            FilterOp::Gt => ">",
            FilterOp::Ge => ">=",
            FilterOp::Lt => "<",
            FilterOp::Le => "<=",
            FilterOp::Contains => "contains",
        }
    }
}

/// A condition value given as a string, number or boolean, as text.
fn text_or_scalar<'de, D: serde::Deserializer<'de>>(de: D) -> Result<String, D::Error> {
    match serde_json::Value::deserialize(de)? {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Number(n) => Ok(n.to_string()),
        serde_json::Value::Bool(b) => Ok(if b { "TRUE" } else { "FALSE" }.to_string()),
        other => Err(serde::de::Error::custom(format!(
            "value must be text or a number, not {other}"
        ))),
    }
}

fn default_filter_limit() -> usize {
    DEFAULT_FILTER_ROWS
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
    /// No column of the table has that heading or letter.
    UnknownColumn,
    /// MCP: no open workbook has that id.
    UnknownWorkbook,
    /// MCP: the workbook's file was deleted or became unreadable.
    WorkbookUnavailable,
    /// MCP: the file could not be opened or parsed.
    OpenFailed,
    /// MCP: no tool of that name.
    UnknownTool,
    /// MCP: the path lies outside every `--root` directory.
    PathNotAllowed,
    /// MCP: the workbook's estimated memory is over the server's
    /// budget on its own.
    OverBudget,
    /// MCP: the call did not finish within the server's time limit.
    Timeout,
    /// MCP: too many calls are still running; retry later.
    Busy,
    /// MCP: the call failed inside GridTalk (a bug, not the input).
    Internal,
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
    pub fn name(&self) -> &'static str {
        match self {
            ToolRequest::Search { .. } => "search",
            ToolRequest::Expand { .. } => "expand",
            ToolRequest::GetCells { .. } => "get_cells",
            ToolRequest::ListFormulas { .. } => "list_formulas",
            ToolRequest::WhatIf { .. } => "what_if",
            ToolRequest::GraphOverview => "graph_overview",
            ToolRequest::WorkbookInfo => "workbook_info",
            ToolRequest::Issues { .. } => "issues",
            ToolRequest::Filter { .. } => "filter",
            ToolRequest::Aggregate { .. } => "aggregate",
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
                tool("workbook_info", "Sheets (name, rows, cols, visibility: visible/hidden/veryHidden), defined names, detected regions, load warnings and the tools' limits. Call it first to learn sheet names and ranges.".into(), schema(serde_json::json!({}), &[])),
                tool("search", format!("Hybrid lexical+embedding search over workbook regions. Returns ranked regions with a cited passage each. limit: 1-{MAX_SEARCH_LIMIT} (default 3; larger values are clamped and the response says so)."), schema(serde_json::json!({"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "default": 3}}), &["query"])),
                tool("expand", format!("Graph neighbours (contains/depends_on edges) of the region or column containing a range, e.g. Sheet1!A1:D4. At most {MAX_EXPAND_EDGES} edges; total_edges and truncated tell when more exist."), schema(serde_json::json!({"range": range}), &["range"])),
                tool("get_cells", format!("Typed cell values in a range, e.g. Sheet1!A1:D4 or 'My Sheet'!B2, each with its address. Formula cells carry their computed result and the file's cached value. At most {MAX_RANGE_CELLS} cells per call."), schema(serde_json::json!({"range": range}), &["range"])),
                tool("list_formulas", format!("Formulas in a range (e.g. Sheet1!A1:D40) with their evaluated values. For each column it says how far one formula is filled down, which cells hold a different formula, and which hold a typed value where the column has formulas. At most {MAX_RANGE_CELLS} cells per call."), schema(serde_json::json!({"range": range}), &["range"])),
                tool("filter", format!("Rows of a table whose columns meet every condition in where, e.g. {{\"column\": \"Region\", \"op\": \"=\", \"value\": \"EMEA\"}}. range: the table or any cell in it (several rows: only those rows). column: a heading or letter. Numbers compare as numbers, text ignoring case; formula cells by their result. Rows carry cell addresses. limit: 1-{MAX_FILTER_ROWS} (default {DEFAULT_FILTER_ROWS})"), schema(serde_json::json!({
                    "range": range,
                    "where": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string", "enum": FilterOp::ALL.iter().map(|op| op.as_str()).collect::<Vec<_>>()},
                            "value": {"type": "string"}
                        },
                        "required": ["column", "op", "value"],
                        "additionalProperties": false
                    }},
                    "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to show; all when absent"},
                    "limit": {"type": "integer", "minimum": 1, "default": DEFAULT_FILTER_ROWS}
                }), &["range"])),
                tool("aggregate", format!("Sum, mean, min, max (with their cells) and counts of a table's column, over the rows meeting where (as in filter). group_by gives the figures per value of another column (groups ignore case); group_by without column counts rows per group; neither profiles every column. range: the table or any cell in it. At most {MAX_AGGREGATE_GROUPS} groups."), schema(serde_json::json!({
                    "range": range,
                    "column": {"type": "string", "description": "Heading or letter of the column to add up"},
                    "group_by": {"type": "string", "description": "Heading or letter of the column to group by"},
                    "where": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op": {"type": "string", "enum": FilterOp::ALL.iter().map(|op| op.as_str()).collect::<Vec<_>>()},
                            "value": {"type": "string"}
                        },
                        "required": ["column", "op", "value"],
                        "additionalProperties": false
                    }},
                }), &["range"])),
                tool("issues", format!("Problems in the workbook's formulas, errors first: formulas that evaluate to an error (and those that only pass on an error from a cell they read), references to a missing sheet, #REF! left by deleted cells, broken defined names, values saved in the file that recalculation disagrees with, links to other workbooks, and numbers written into a formula's arithmetic (=B5*1.05). Cells with the same problem in one column are one issue. range: check one sheet's range only (e.g. Sheet1!A1:F200); absent checks the whole workbook. At most {MAX_ISSUES_LISTED} issues are listed; the counts cover all."), schema(serde_json::json!({"range": range}), &[])),
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

    /// "no sheet named X", with the closest sheet name when one is
    /// near and the sheets there are.
    fn no_sheet(&self, name: &str) -> String {
        let names: Vec<&str> = self.wb.sheets.iter().map(|s| s.name.as_str()).collect();
        let mut text = format!("no sheet named {name}");
        if let Some(near) = did_you_mean(name, &names) {
            text.push_str(&format!(" (did you mean {}?)", quote_sheet(near)));
        }
        let quoted: Vec<String> = names.iter().map(|n| quote_sheet(n)).collect();
        text.push_str(&format!("; the sheets are {}", listed(&quoted, 20)));
        text
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
                // Each hit's range and score, then the top of its
                // passage (the heading and the rows that matched), so
                // a reader can tell the hits apart before reading one.
                let text = hits
                    .iter()
                    .map(|h| {
                        let mut entry = format!("[{}] score {:.3}", h.range.a1(), h.score);
                        let mut used = 0usize;
                        for line in h.passage.lines() {
                            let len = line.chars().count();
                            if used + len > MAX_SEARCH_TEXT_PASSAGE {
                                entry.push_str("\n  ...");
                                break;
                            }
                            used += len;
                            entry.push_str("\n  ");
                            entry.push_str(line);
                        }
                        entry
                    })
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
                        let mut lines = vec![format!("node {} ({})", n.id, n.cited_label())];
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
                                lines.push(format!(" {left}{}{right} {}", e.kind, t.cited_label()));
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
            ToolRequest::GetCells { range } => self.get_cells(range),
            ToolRequest::Issues { range } => self.issues(range.as_deref()),
            ToolRequest::Filter {
                range,
                r#where,
                columns,
                limit,
            } => self.filter(range, r#where, columns, *limit),
            ToolRequest::Aggregate {
                range,
                column,
                group_by,
                r#where,
            } => self.aggregate(range, column.as_deref(), group_by.as_deref(), r#where),
            ToolRequest::ListFormulas { range } => self.list_formulas(range),
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
                        cell_error("target", target, target),
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
                        cell_error("cell", cell, &qualified),
                    );
                };
                if self.sheet(&override_sheet).is_none() {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::UnknownSheet,
                        format!(
                            "{} (for override cell {cell})",
                            self.no_sheet(&override_sheet)
                        ),
                    );
                }
                if self.sheet(&target_sheet).is_none() {
                    return ToolResult::err(
                        "what_if",
                        ErrorKind::UnknownSheet,
                        self.no_sheet(&target_sheet),
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
                    .map(|s| serde_json::json!({"name": s.name, "rows": s.cells.len(), "cols": s.max_col(), "visibility": s.visibility.as_str()}))
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
                    let hidden = if s.visibility == gridtalk_graph::SheetVisibility::Visible {
                        String::new()
                    } else {
                        format!(", {}", s.visibility.as_str())
                    };
                    lines.push(format!(
                        "sheet {} ({} rows x {} cols{hidden})",
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
                            "range": n.range.as_ref().map(|r| r.a1()),
                            "visibility": n.visibility.map(|v| v.as_str())
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

    /// `get_cells`: every cell carries its address, in the JSON
    /// (`address`) and in the text, so a reader can cite a cell
    /// without counting columns. A range of up to
    /// [`MAX_ADDRESSED_CELLS`] cells prints one line per cell
    /// (`Labour!D2: E/012`); a larger one prints one line per sheet
    /// row with each value after its column letter.
    fn get_cells(&self, range: &str) -> ToolResult {
        let Some((sheet_name, r0, c0, r1, c1)) = parse_range(range) else {
            return ToolResult::err("get_cells", ErrorKind::InvalidRange, range_error(range));
        };
        let Some(sheet) = self.sheet(&sheet_name) else {
            return ToolResult::err(
                "get_cells",
                ErrorKind::UnknownSheet,
                self.no_sheet(&sheet_name),
            );
        };
        // Formula cells report what they compute, not only the
        // file's cached value: one batch through the workbook's
        // calc engine.
        let formulas: Vec<(usize, usize)> = (r0..=r1)
            .flat_map(|r| (c0..=c1).map(move |c| (r, c)))
            .filter(|(r, c)| matches!(sheet.get_ref(*r, *c), CellValue::Formula { .. }))
            .collect();
        let mut results = if formulas.is_empty() {
            Vec::new()
        } else {
            self.calc.get(self.wb).values(&sheet.name, &formulas)
        }
        .into_iter();
        let quoted = quote_sheet(&sheet.name);
        let per_cell = (r1 - r0 + 1) * (c1 - c0 + 1) <= MAX_ADDRESSED_CELLS;
        let mut rows = Vec::new();
        let mut lines = Vec::new();
        let mut empty = 0usize;
        for r in r0..=r1 {
            let mut row = Vec::new();
            let mut shown: Vec<String> = Vec::new();
            for c in c0..=c1 {
                let column = gridtalk_graph::col_name(c);
                let address = format!("{quoted}!{column}{}", r + 1);
                let cell = sheet.get(r, c);
                let (mut json, display, formula) = match &cell {
                    CellValue::Formula { expr, cached } => {
                        let result = results.next().expect("one result per formula");
                        let display = match &result {
                            Ok(v) => v.display(),
                            Err(e) => format!("#ERROR({e})"),
                        };
                        (
                            formula_json(expr, cached.as_deref(), &result),
                            display,
                            Some(expr.clone()),
                        )
                    }
                    _ => (cell_json(&cell), cell.display(), None),
                };
                json["address"] = serde_json::Value::String(address.clone());
                row.push(json);
                if matches!(cell, CellValue::Empty) {
                    empty += 1;
                } else if per_cell {
                    lines.push(match formula {
                        Some(expr) => format!("{address}: {display}  [={expr}]"),
                        None => format!("{address}: {display}"),
                    });
                } else {
                    shown.push(format!("{column}: {display}"));
                }
            }
            if !per_cell {
                lines.push(if shown.is_empty() {
                    format!("{} | (empty)", r + 1)
                } else {
                    format!("{} | {}", r + 1, shown.join(" | "))
                });
            }
            rows.push(serde_json::Value::Array(row));
        }
        let mut text = if per_cell {
            format!("Range {range}\n")
        } else {
            format!(
                "Range {range}\nOne line per sheet row; each value follows its column letter, so row {}'s {} value is {quoted}!{}{}.\n",
                r0 + 1,
                gridtalk_graph::col_name(c0),
                gridtalk_graph::col_name(c0),
                r0 + 1
            )
        };
        for line in &lines {
            text.push_str(line);
            text.push('\n');
        }
        if per_cell && empty > 0 {
            text.push_str(&if lines.is_empty() {
                "(every cell in the range is empty)\n".to_string()
            } else {
                format!("({empty} empty cells not listed)\n")
            });
        }
        let columns: Vec<String> = (c0..=c1).map(gridtalk_graph::col_name).collect();
        ToolResult::ok(
            "get_cells",
            serde_json::json!({
                "range": range, "sheet": sheet.name, "first_row": r0 + 1,
                "columns": columns, "rows": rows
            }),
            text,
        )
    }

    /// `list_formulas`: each formula with its sheet-qualified cell,
    /// after one line per column that says how far one formula is
    /// filled down and which cells break the pattern (see
    /// [`Tools::column_patterns`]).
    fn list_formulas(&self, range: &str) -> ToolResult {
        let Some((sheet_name, r0, c0, r1, c1)) = parse_range(range) else {
            return ToolResult::err("list_formulas", ErrorKind::InvalidRange, range_error(range));
        };
        let Some(sheet) = self.sheet(&sheet_name) else {
            return ToolResult::err(
                "list_formulas",
                ErrorKind::UnknownSheet,
                self.no_sheet(&sheet_name),
            );
        };
        let coords: Vec<(usize, usize)> = (r0..=r1)
            .flat_map(|r| (c0..=c1).map(move |c| (r, c)))
            .filter(|(r, c)| matches!(sheet.get_ref(*r, *c), CellValue::Formula { .. }))
            .collect();
        if coords.is_empty() {
            return ToolResult::ok(
                "list_formulas",
                serde_json::json!({"formulas": [], "columns": []}),
                "no formulas in range".into(),
            );
        }
        // One recalculation for the whole batch, not one per cell.
        let evaluated = self.calc.get(self.wb).values(&sheet.name, &coords);
        let quoted = quote_sheet(&sheet.name);
        let mut formulas = Vec::new();
        let mut lines = Vec::new();
        for (&(r, c), result) in coords.iter().zip(evaluated.iter()) {
            let CellValue::Formula { expr, .. } = sheet.get_ref(r, c) else {
                continue;
            };
            let cell = format!("{}{}", gridtalk_graph::col_name(c), r + 1);
            let address = format!("{quoted}!{cell}");
            let (value, typed, error) = match result {
                Ok(v) => (legacy_number(v), Some(value_json(v)), None),
                Err(e) => (None, None, Some(e.to_string())),
            };
            lines.push(match result {
                Ok(v) => format!("{address} = {expr} -> {}", v.display()),
                Err(e) => format!("{address} = {expr} -> error: {e}"),
            });
            formulas.push(serde_json::json!({
                "cell": cell, "address": address, "formula": expr, "value": value,
                "result": typed, "error": error
            }));
        }
        let (columns, mut text) = self.column_patterns(sheet, (r0, c0, r1, c1));
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&lines.join("\n"));
        ToolResult::ok(
            "list_formulas",
            serde_json::json!({"formulas": formulas, "columns": columns}),
            text,
        )
    }

    /// One entry per column of the range that holds at least
    /// [`MIN_PATTERN_FORMULAS`] formulas: how many share the column's
    /// commonest shape (compared in R1C1, so a formula filled down
    /// counts as one shape), the formula cells with another shape,
    /// and the cells holding a typed value where the column
    /// otherwise holds formulas. The second kind is what "has a
    /// formula been typed over?" asks for, and reading it off
    /// hundreds of formula lines is where a model goes wrong.
    fn column_patterns(
        &self,
        sheet: &gridtalk_graph::SheetSnapshot,
        (r0, c0, r1, c1): (usize, usize, usize, usize),
    ) -> (Vec<serde_json::Value>, String) {
        let quoted = quote_sheet(&sheet.name);
        let address =
            |r: usize, c: usize| format!("{quoted}!{}{}", gridtalk_graph::col_name(c), r + 1);
        let mut columns = Vec::new();
        let mut lines = Vec::new();
        for c in c0..=c1 {
            // (row, formula, shape) for each formula cell, top down.
            let cells: Vec<(usize, &str, String)> = (r0..=r1)
                .filter_map(|r| match sheet.get_ref(r, c) {
                    CellValue::Formula { expr, .. } => Some((
                        r,
                        expr.as_str(),
                        gridtalk_graph::lineage::formula_pattern(expr, r, c)
                            .unwrap_or_else(|| expr.clone()),
                    )),
                    _ => None,
                })
                .collect();
            if cells.len() < MIN_PATTERN_FORMULAS {
                continue;
            }
            let mut counts: Vec<(&str, usize)> = Vec::new();
            for (_, _, shape) in &cells {
                match counts.iter_mut().find(|(s, _)| s == shape) {
                    Some(entry) => entry.1 += 1,
                    None => counts.push((shape, 1)),
                }
            }
            // The commonest shape; the first seen wins a tie.
            let mut best = counts[0];
            for &entry in &counts[1..] {
                if entry.1 > best.1 {
                    best = entry;
                }
            }
            let (pattern, same) = best;
            let example = cells.iter().find(|(_, _, s)| s == pattern).unwrap();
            let other: Vec<String> = cells
                .iter()
                .filter(|(_, _, s)| s != pattern)
                .map(|(r, _, _)| address(*r, c))
                .collect();
            // The rows the column's formulas run over, widened to the
            // graph column they sit in (below its heading), so a value
            // typed over the first or last formula is still inside.
            let (first, last) = (cells[0].0, cells[cells.len() - 1].0);
            let (mut top, mut bottom) = (first, last);
            if let Some(node) = self.graph.nodes.iter().find_map(|n| {
                let r = n.range.as_ref()?;
                (n.kind == gridtalk_graph::NodeKind::Column
                    && sheet_name_eq(&r.sheet, &sheet.name)
                    && r.c0 == c
                    && r.c1 == c
                    && r.r0 <= first
                    && r.r1 >= last
                    && r.r1 > r.r0)
                    .then_some(r)
            }) {
                let heading = matches!(sheet.get_ref(node.r0, c), CellValue::Text(_));
                top = top.min(node.r0 + usize::from(heading)).max(r0);
                bottom = bottom.max(node.r1).min(r1);
            }
            let typed: Vec<(usize, String)> = (top..=bottom)
                .filter_map(|r| match sheet.get_ref(r, c) {
                    CellValue::Empty | CellValue::Formula { .. } => None,
                    value => Some((r, value.display())),
                })
                .collect();
            // A column that is mostly typed values is mixed by
            // design, not typed over.
            let typed = if typed.len() < cells.len() {
                typed
            } else {
                Vec::new()
            };
            let span = gridtalk_graph::RangeRef {
                sheet: sheet.name.clone(),
                r0: top,
                c0: c,
                r1: bottom,
                c1: c,
            }
            .a1();
            let mut line = format!(
                "column {} ({span}): {} formulas, {} like {} = {}",
                gridtalk_graph::col_name(c),
                cells.len(),
                if same == cells.len() {
                    "all".to_string()
                } else {
                    same.to_string()
                },
                address(example.0, c),
                example.1
            );
            if !other.is_empty() {
                line.push_str(&format!(
                    "; a different formula in {}",
                    listed(&other, MAX_PATTERN_EXCEPTIONS)
                ));
            }
            if !typed.is_empty() {
                let cells: Vec<String> = typed
                    .iter()
                    .map(|(r, v)| format!("{} ({v})", address(*r, c)))
                    .collect();
                line.push_str(&format!(
                    "; a typed value, not a formula, in {}",
                    listed(&cells, MAX_PATTERN_EXCEPTIONS)
                ));
            }
            if other.is_empty() && typed.is_empty() {
                line.push_str("; no cell breaks the pattern");
            }
            lines.push(line);
            columns.push(serde_json::json!({
                "column": gridtalk_graph::col_name(c), "range": span,
                "formulas": cells.len(), "same_pattern": same, "pattern": pattern,
                "example": {"address": address(example.0, c), "formula": example.1},
                "other_formulas": other,
                "typed_values": typed.iter().map(|(r, v)| serde_json::json!({"address": address(*r, c), "value": v})).collect::<Vec<_>>(),
            }));
        }
        (columns, lines.join("\n"))
    }

    /// The table at `range` with its data rows and `conditions`
    /// resolved, for `filter` and `aggregate`; or the error to return.
    fn table_rows<'t>(
        &'t self,
        tool: &str,
        range: &str,
        conditions: &[Condition],
    ) -> Result<TableRows<'t>, ToolResult> {
        let fail = |kind, message: String| Err(ToolResult::err(tool, kind, message));
        let Some((sheet_name, r0, c0, r1, c1)) = parse_range_uncapped(range) else {
            return fail(ErrorKind::InvalidRange, range_error(range));
        };
        let Some(sheet) = self.sheet(&sheet_name) else {
            return fail(ErrorKind::UnknownSheet, self.no_sheet(&sheet_name));
        };
        // The smallest table holding the range's first cell, else the
        // first one it overlaps.
        let on_sheet = || {
            self.graph
                .regions
                .iter()
                .filter(|g| sheet_name_eq(&g.range.sheet, &sheet.name))
        };
        let Some(region) = on_sheet()
            .filter(|g| {
                let t = &g.range;
                t.r0 <= r0 && r0 <= t.r1 && t.c0 <= c0 && c0 <= t.c1
            })
            .min_by_key(|g| (g.range.r1 - g.range.r0 + 1) * (g.range.c1 - g.range.c0 + 1))
            .or_else(|| {
                on_sheet().find(|g| {
                    let t = &g.range;
                    t.r0 <= r1 && r0 <= t.r1 && t.c0 <= c1 && c0 <= t.c1
                })
            })
        else {
            return fail(
                ErrorKind::NoGraphNode,
                format!("no table at {range}; workbook_info lists the tables (regions)"),
            );
        };
        let table = &region.range;
        let first_data = region.header_row.map_or(table.r0, |h| h + 1).max(table.r0);
        // A range of several rows checks only those rows; a cell or
        // a column from the heading down checks the whole table.
        let (top, bottom) = if r1 > r0 {
            (first_data.max(r0), table.r1.min(r1))
        } else {
            (first_data, table.r1)
        };
        let mut rows = TableRows {
            sheet,
            region,
            first_data,
            top,
            bottom,
            computed: Default::default(),
            tests: Vec::new(),
        };
        for condition in conditions {
            if let Some(message) = operator_in_value(condition) {
                return fail(ErrorKind::InvalidArguments, message);
            }
            match rows.resolve(&condition.column) {
                Ok(c) => rows.tests.push((c, condition.clone())),
                Err(e) => return fail(ErrorKind::UnknownColumn, e),
            }
        }
        // What each formula cell of the rows computes, in one batch.
        let formulas: Vec<(usize, usize)> = (top..=bottom)
            .flat_map(|r| (table.c0..=table.c1).map(move |c| (r, c)))
            .filter(|(r, c)| matches!(sheet.get_ref(*r, *c), CellValue::Formula { .. }))
            .collect();
        if !formulas.is_empty() {
            let values = self.calc.get(self.wb).values(&sheet.name, &formulas);
            rows.computed = formulas
                .into_iter()
                .zip(values)
                .map(|(at, v)| (at, v.map_err(|e| e.to_string())))
                .collect();
        }
        Ok(rows)
    }

    /// `filter`: the data rows of the table at `range` whose cells
    /// meet every condition, with their addresses.
    fn filter(
        &self,
        range: &str,
        conditions: &[Condition],
        shown: &[String],
        requested: usize,
    ) -> ToolResult {
        let t = match self.table_rows("filter", range, conditions) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let table = &t.region.range;
        let mut show = Vec::new();
        for name in shown {
            match t.resolve(name) {
                Ok(c) if !show.contains(&c) => show.push(c),
                Ok(_) => {}
                Err(e) => return ToolResult::err("filter", ErrorKind::UnknownColumn, e),
            }
        }
        if show.is_empty() {
            show = (table.c0..=table.c1).collect();
        }
        let limit = requested.clamp(1, MAX_FILTER_ROWS);
        let quoted = quote_sheet(&t.sheet.name);
        let (matching, checked) = t.matching_rows();
        let mut rows = Vec::new();
        let mut lines = Vec::new();
        for &r in matching.iter().take(limit) {
            let mut cells = Vec::new();
            let mut parts = Vec::new();
            for &c in &show {
                let address = format!("{}{}", gridtalk_graph::col_name(c), r + 1);
                let (mut json, display) = match &t.value_at(r, c) {
                    Ok(v) => (value_json(v), v.display()),
                    Err(e) => (
                        serde_json::json!({"type": "error", "error": e}),
                        format!("#ERROR({e})"),
                    ),
                };
                json["address"] = serde_json::json!(format!("{quoted}!{address}"));
                json["column"] = serde_json::json!(t.heading_of(c));
                cells.push(json);
                parts.push(match t.heading_of(c) {
                    Some(h) => format!("{address} {h} = {display}"),
                    None => format!("{address} = {display}"),
                });
            }
            let row_range = gridtalk_graph::RangeRef {
                sheet: t.sheet.name.clone(),
                r0: r,
                c0: *show.iter().min().unwrap_or(&table.c0),
                r1: r,
                c1: *show.iter().max().unwrap_or(&table.c1),
            }
            .a1();
            lines.push(format!("{quoted} row {}: {}", r + 1, parts.join(" | ")));
            rows.push(serde_json::json!({"row": r + 1, "range": row_range, "cells": cells}));
        }
        let matched = matching.len();
        let mut text = t.heading_line();
        text.push_str(&format!(": {matched} of {checked} rows match"));
        if matched > rows.len() {
            text.push_str(&format!(
                " (the first {} shown; raise limit, up to {MAX_FILTER_ROWS}, or add a condition)",
                rows.len()
            ));
        }
        if requested != limit {
            text.push_str(&format!(" (limit {requested} clamped to {limit})"));
        }
        for line in &lines {
            text.push('\n');
            text.push_str(line);
        }
        let columns_json: Vec<serde_json::Value> = show
            .iter()
            .map(|&c| serde_json::json!({"name": t.heading_of(c), "letter": gridtalk_graph::col_name(c)}))
            .collect();
        ToolResult::ok(
            "filter",
            serde_json::json!({
                "table": table.a1(), "checked": t.checked_range(), "where": t.where_json(),
                "columns": columns_json, "matched": matched, "rows_checked": checked,
                "shown": rows.len(), "limit": limit, "limit_clamped": requested != limit,
                "truncated": matched > rows.len(), "rows": rows
            }),
            text,
        )
    }

    /// `aggregate`: figures over the matching rows of a table: of one
    /// column, of one column per group, rows per group, or (with
    /// neither) of every column.
    fn aggregate(
        &self,
        range: &str,
        column: Option<&str>,
        group_by: Option<&str>,
        conditions: &[Condition],
    ) -> ToolResult {
        let t = match self.table_rows("aggregate", range, conditions) {
            Ok(t) => t,
            Err(e) => return e,
        };
        let resolve = |name: Option<&str>| match name.map(str::trim).filter(|n| !n.is_empty()) {
            None => Ok(None),
            Some(name) => t.resolve(name).map(Some),
        };
        let (column, group_by) = match (resolve(column), resolve(group_by)) {
            (Ok(c), Ok(g)) => (c, g),
            (Err(e), _) | (_, Err(e)) => {
                return ToolResult::err("aggregate", ErrorKind::UnknownColumn, e)
            }
        };
        let table = &t.region.range;
        let (matching, checked) = t.matching_rows();
        let mut text = t.heading_line();
        text.push_str(&format!(": {} of {checked} rows match", matching.len()));
        let mut data = serde_json::json!({
            "table": table.a1(), "checked": t.checked_range(), "where": t.where_json(),
            "rows_matched": matching.len(), "rows_checked": checked,
        });
        match group_by {
            None => {
                let columns: Vec<usize> = match column {
                    Some(c) => vec![c],
                    None => (table.c0..=table.c1).collect(),
                };
                let mut stats_json = Vec::new();
                for c in columns {
                    let stats = Stats::of(&t, c, &matching);
                    text.push('\n');
                    text.push_str(&stats.line(&t.sheet.name, &t.name_of(c), &t.column_range(c)));
                    let mut json = stats.json(&t.sheet.name);
                    json["column"] = serde_json::json!(t.name_of(c));
                    json["letter"] = serde_json::json!(gridtalk_graph::col_name(c));
                    json["range"] = serde_json::json!(t.column_range(c));
                    stats_json.push(json);
                }
                data["columns"] = serde_json::Value::Array(stats_json);
            }
            Some(g) => {
                // Groups in the order they first appear, merged
                // ignoring case and surrounding spaces; the label is
                // the first spelling seen.
                let mut groups: Vec<(String, String, Vec<usize>)> = Vec::new();
                for &r in &matching {
                    let label = match t.value_at(r, g) {
                        Ok(v) => v.display().trim().to_string(),
                        Err(e) => format!("#ERROR({e})"),
                    };
                    let key = label.to_lowercase();
                    match groups.iter_mut().find(|(k, _, _)| *k == key) {
                        Some(group) => group.2.push(r),
                        None => groups.push((key, label, vec![r])),
                    }
                }
                let total = groups.len();
                let what = match column {
                    Some(c) => format!("{} by {}", t.name_of(c), t.name_of(g)),
                    None => format!("rows by {}", t.name_of(g)),
                };
                text.push_str(&format!("\n{total} groups of {what}"));
                if total > MAX_AGGREGATE_GROUPS {
                    text.push_str(&format!(" (the first {MAX_AGGREGATE_GROUPS} shown, in the order they appear; add a condition to narrow)"));
                }
                let mut groups_json = Vec::new();
                for (_, label, rows) in groups.iter().take(MAX_AGGREGATE_GROUPS) {
                    let shown = if label.is_empty() {
                        "(blank)"
                    } else {
                        label.as_str()
                    };
                    let mut json = serde_json::json!({"group": label, "rows": rows.len()});
                    match column {
                        Some(c) => {
                            let stats = Stats::of(&t, c, rows);
                            text.push('\n');
                            text.push_str(&stats.line(
                                &t.sheet.name,
                                shown,
                                &plural(rows.len(), "row"),
                            ));
                            let figures = stats.json(&t.sheet.name);
                            if let (Some(json), Some(figures)) =
                                (json.as_object_mut(), figures.as_object())
                            {
                                json.extend(figures.clone());
                            }
                        }
                        None => text.push_str(&format!("\n{shown}: {}", plural(rows.len(), "row"))),
                    }
                    groups_json.push(json);
                }
                data["group_by"] = serde_json::json!({"column": t.name_of(g), "letter": gridtalk_graph::col_name(g)});
                if let Some(c) = column {
                    data["column"] = serde_json::json!({"column": t.name_of(c), "letter": gridtalk_graph::col_name(c), "range": t.column_range(c)});
                }
                data["groups_total"] = serde_json::json!(total);
                data["truncated"] = serde_json::json!(total > MAX_AGGREGATE_GROUPS);
                data["groups"] = serde_json::Value::Array(groups_json);
            }
        }
        ToolResult::ok("aggregate", data, text)
    }

    /// `issues`: the workbook's problems (see
    /// [`gridtalk_graph::issues`]), errors first, at most
    /// [`MAX_ISSUES_LISTED`] of them with their first cells.
    fn issues(&self, range: Option<&str>) -> ToolResult {
        use gridtalk_graph::issues::{find_issues, IssueKind, Severity};
        let scope = match range {
            None => None,
            Some(range) => {
                let Some((sheet_name, r0, c0, r1, c1)) = parse_range_uncapped(range) else {
                    return ToolResult::err("issues", ErrorKind::InvalidRange, range_error(range));
                };
                let Some(index) = self
                    .wb
                    .sheets
                    .iter()
                    .position(|s| sheet_name_eq(&s.name, &sheet_name))
                else {
                    return ToolResult::err(
                        "issues",
                        ErrorKind::UnknownSheet,
                        self.no_sheet(&sheet_name),
                    );
                };
                Some((index, (r0, c0, r1, c1)))
            }
        };
        let found = find_issues(self.wb, self.calc.get(self.wb), scope);
        let errors = found
            .iter()
            .filter(|i| i.severity() == Severity::Error)
            .count();
        let warnings = found.len() - errors;
        let place = range.map_or("the workbook".to_string(), |r| r.to_string());
        if found.is_empty() {
            return ToolResult::ok(
                "issues",
                serde_json::json!({"range": range, "errors": 0, "warnings": 0, "total": 0, "listed": 0, "truncated": false, "issues": []}),
                format!("no issues found in {place}"),
            );
        }
        let mut lines = vec![format!(
            "{errors} {} and {warnings} {} in {place}{}",
            if errors == 1 { "error" } else { "errors" },
            if warnings == 1 { "warning" } else { "warnings" },
            if found.len() > MAX_ISSUES_LISTED {
                format!(" (the first {MAX_ISSUES_LISTED} of {} listed)", found.len())
            } else {
                String::new()
            }
        )];
        let mut listed_issues = Vec::new();
        let mut saved_note = false;
        for issue in found.iter().take(MAX_ISSUES_LISTED) {
            let sheet = issue.sheet.as_deref().unwrap_or_default();
            let address = |row: usize| {
                issue
                    .col
                    .map(|col| gridtalk_graph::issues::cell_range(sheet, row, col, row))
                    .unwrap_or_default()
            };
            let cells: Vec<serde_json::Value> = issue
                .cells
                .iter()
                .take(MAX_ISSUE_CELLS_JSON)
                .map(|cell| {
                    let (result, error) = match &cell.result {
                        Ok(v) => (Some(value_json(v)), None),
                        Err(e) => (None, Some(e.to_string())),
                    };
                    serde_json::json!({
                        "address": address(cell.row), "formula": cell.formula,
                        "result": result, "error": error, "saved": cell.saved
                    })
                })
                .collect();
            let message = match issue.kind {
                IssueKind::FormulaError => format!("evaluates to {}", issue.detail),
                IssueKind::ErrorFromPrecedent => {
                    format!("passes on an error from a cell it reads: {}", issue.detail)
                }
                IssueKind::MissingSheet => {
                    format!("refers to a sheet the workbook lacks: {}", issue.detail)
                }
                IssueKind::DeletedReference => {
                    "holds #REF!, a reference whose cells were deleted".to_string()
                }
                IssueKind::BrokenName => issue.detail.clone(),
                IssueKind::SavedValueDiffers => {
                    saved_note = true;
                    "the value saved in the file differs from the recalculated one".to_string()
                }
                IssueKind::ExternalReference => format!(
                    "reads another workbook ({}), which GridTalk cannot open; the value shown is the one the file saved",
                    issue.detail
                ),
                IssueKind::ConstantInFormula => format!(
                    "writes {} into the formula's arithmetic; an assumption written there cannot be changed or traced from a cell",
                    issue.detail
                ),
            };
            let place = match (&issue.name, issue.range()) {
                (Some(name), _) => format!("name {name}"),
                (None, Some(range)) if issue.cells.len() > 1 => {
                    format!("{range} ({} cells)", issue.cells.len())
                }
                (None, Some(range)) => range,
                (None, None) => String::new(),
            };
            let mut line = format!(
                "{} {} {place}: {message}",
                issue.severity().as_str(),
                issue.kind.as_str()
            );
            let examples: Vec<String> = issue
                .cells
                .iter()
                .map(|cell| {
                    let mut text = format!("{} = {}", address(cell.row), cell.formula);
                    if issue.kind == IssueKind::SavedValueDiffers {
                        let now = match &cell.result {
                            Ok(v) => v.display(),
                            Err(e) => e.to_string(),
                        };
                        text.push_str(&format!(
                            " saved {}, recalculates to {now}",
                            cell.saved.as_deref().unwrap_or("nothing")
                        ));
                    } else if issue.kind == IssueKind::ExternalReference {
                        if let Some(saved) = &cell.saved {
                            text.push_str(&format!(" (saved {saved})"));
                        }
                    }
                    text
                })
                .collect();
            if !examples.is_empty() {
                line.push_str(&format!("; {}", listed(&examples, MAX_ISSUE_CELLS_TEXT)));
            }
            lines.push(line);
            listed_issues.push(serde_json::json!({
                "kind": issue.kind, "severity": issue.severity(), "sheet": issue.sheet,
                "range": issue.range(), "name": issue.name, "detail": issue.detail,
                "message": message, "count": issue.cells.len(), "cells": cells,
                "cells_truncated": issue.cells.len() > MAX_ISSUE_CELLS_JSON,
            }));
        }
        if saved_note {
            lines.push("(a saved value differs when the file was saved without recalculating, a cell it reads changed since, or Gridcraft computes a function differently from the program that saved it)".into());
        }
        ToolResult::ok(
            "issues",
            serde_json::json!({
                "range": range, "errors": errors, "warnings": warnings, "total": found.len(),
                "listed": listed_issues.len(), "truncated": found.len() > MAX_ISSUES_LISTED,
                "issues": listed_issues
            }),
            lines.join("\n"),
        )
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
            "filter" | "aggregate" => {
                return "filter and aggregate take a JSON request: {\"tool\": \"filter\", \"range\": ..., \"where\": [...]}".into()
            }
            "issues" => ToolRequest::Issues {
                range: arg("range").filter(|r| !r.is_empty()),
            },
            other => {
                return format!(
                    "unknown tool {other}; available: workbook_info, search, expand, get_cells, list_formulas, filter, aggregate, what_if, issues, graph_overview"
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

/// Largest range `get_cells` prints one line per cell for; a larger
/// one prints one line per sheet row.
pub const MAX_ADDRESSED_CELLS: usize = 60;
/// Fewest formulas in a column before `list_formulas` describes the
/// column's pattern.
pub const MIN_PATTERN_FORMULAS: usize = 3;
/// Most cells named per kind of exception in a column's pattern line
/// (the JSON lists them all).
pub const MAX_PATTERN_EXCEPTIONS: usize = 12;
/// Most characters of a hit's passage in `search`'s text (the JSON
/// carries the whole passage).
pub const MAX_SEARCH_TEXT_PASSAGE: usize = 600;

/// `a, b, c` for up to `max` items, then how many more there are.
fn listed(items: &[String], max: usize) -> String {
    let mut out = items
        .iter()
        .take(max)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    if items.len() > max {
        out.push_str(&format!(" and {} more", items.len() - max));
    }
    out
}

/// A table's data rows, as `filter` and `aggregate` read them: the
/// rows checked, what each formula cell computes, and the conditions
/// resolved to columns.
struct TableRows<'t> {
    sheet: &'t gridtalk_graph::SheetSnapshot,
    region: &'t gridtalk_graph::Region,
    /// The first row below the heading.
    first_data: usize,
    /// The rows checked, inclusive.
    top: usize,
    bottom: usize,
    computed: std::collections::HashMap<(usize, usize), Result<Value, String>>,
    tests: Vec<(usize, Condition)>,
}

impl TableRows<'_> {
    /// A column's heading; `None` without a heading row (the headings
    /// are then the letters themselves) or for an empty heading.
    fn heading_of(&self, c: usize) -> Option<&str> {
        let h = self.region.headers.get(c - self.region.range.c0)?.trim();
        (!h.is_empty() && self.region.header_row.is_some()).then_some(h)
    }

    /// A column's heading, or its letter.
    fn name_of(&self, c: usize) -> String {
        self.heading_of(c)
            .map(str::to_string)
            .unwrap_or_else(|| gridtalk_graph::col_name(c))
    }

    /// A column named by heading (ignoring case) or letter, or what
    /// to say about a name that is neither.
    fn resolve(&self, name: &str) -> Result<usize, String> {
        let table = &self.region.range;
        let wanted = name.trim();
        if let Some(c) = (table.c0..=table.c1).find(|&c| {
            self.heading_of(c)
                .is_some_and(|h| h.eq_ignore_ascii_case(wanted))
        }) {
            return Ok(c);
        }
        if let Some(c) = column_letters(wanted) {
            if (table.c0..=table.c1).contains(&c) {
                return Ok(c);
            }
        }
        let named: Vec<&str> = (table.c0..=table.c1)
            .filter_map(|c| self.heading_of(c))
            .collect();
        let mut text = format!("no column {wanted} in {}", table.a1());
        if let Some(near) = did_you_mean(wanted, &named) {
            text.push_str(&format!(" (did you mean {near}?)"));
        }
        let all: Vec<String> = (table.c0..=table.c1)
            .map(|c| match self.heading_of(c) {
                Some(h) => format!("{h} ({})", gridtalk_graph::col_name(c)),
                None => gridtalk_graph::col_name(c),
            })
            .collect();
        text.push_str(&format!("; its columns are {}", listed(&all, 40)));
        Err(text)
    }

    /// What a cell holds, or what its formula computes.
    fn value_at(&self, r: usize, c: usize) -> Result<Value, String> {
        match self.sheet.get_ref(r, c) {
            CellValue::Formula { .. } => self
                .computed
                .get(&(r, c))
                .cloned()
                .unwrap_or(Ok(Value::Empty)),
            CellValue::Empty => Ok(Value::Empty),
            CellValue::Text(t) => Ok(Value::Text(t.clone())),
            CellValue::Number(n) => Ok(Value::Number(*n)),
            CellValue::Bool(b) => Ok(Value::Bool(*b)),
        }
    }

    /// The rows that meet every condition, top down, and how many
    /// non-empty rows were checked.
    fn matching_rows(&self) -> (Vec<usize>, usize) {
        let table = &self.region.range;
        let mut checked = 0;
        let rows = (self.top..=self.bottom)
            .filter(|&r| {
                !(table.c0..=table.c1).all(|c| matches!(self.sheet.get_ref(r, c), CellValue::Empty))
            })
            .inspect(|_| checked += 1)
            .filter(|&r| {
                self.tests
                    .iter()
                    .all(|(c, condition)| condition_holds(&self.value_at(r, *c), condition))
            })
            .collect();
        (rows, checked)
    }

    /// The rows checked, as a range over the whole table's width.
    fn checked_range(&self) -> String {
        let table = &self.region.range;
        gridtalk_graph::RangeRef {
            sheet: self.sheet.name.clone(),
            r0: self.top,
            c0: table.c0,
            r1: self.bottom.max(self.top),
            c1: table.c1,
        }
        .a1()
    }

    /// One column over the rows checked.
    fn column_range(&self, c: usize) -> String {
        gridtalk_graph::RangeRef {
            sheet: self.sheet.name.clone(),
            r0: self.top,
            c0: c,
            r1: self.bottom.max(self.top),
            c1: c,
        }
        .a1()
    }

    /// `Table Sheet!A1:D9[, rows ...][ where ...]`.
    fn heading_line(&self) -> String {
        let table = &self.region.range;
        let mut text = format!("Table {}", table.a1());
        if (self.top, self.bottom) != (self.first_data, table.r1) {
            text.push_str(&format!(", rows {}", self.checked_range()));
        }
        if !self.tests.is_empty() {
            let described: Vec<String> = self
                .tests
                .iter()
                .map(|(c, cond)| {
                    format!("{} {} {}", self.name_of(*c), cond.op.as_str(), cond.value)
                })
                .collect();
            text.push_str(&format!(" where {}", described.join(" and ")));
        }
        text
    }

    fn where_json(&self) -> Vec<serde_json::Value> {
        self.tests
            .iter()
            .map(|(c, cond)| {
                serde_json::json!({
                    "column": cond.column, "letter": gridtalk_graph::col_name(*c),
                    "op": cond.op, "value": cond.value
                })
            })
            .collect()
    }
}

/// `aggregate`'s figures for one column over some rows.
struct Stats {
    rows: usize,
    blanks: usize,
    numbers: usize,
    texts: usize,
    errors: usize,
    /// Distinct non-empty values that are not numbers, ignoring case.
    distinct_text: usize,
    sum: f64,
    /// (value, 0-based row) of the smallest and largest number.
    min: Option<(f64, usize)>,
    max: Option<(f64, usize)>,
    column: usize,
}

impl Stats {
    fn of(t: &TableRows, c: usize, rows: &[usize]) -> Stats {
        let mut stats = Stats {
            rows: rows.len(),
            blanks: 0,
            numbers: 0,
            texts: 0,
            errors: 0,
            distinct_text: 0,
            sum: 0.0,
            min: None,
            max: None,
            column: c,
        };
        let mut seen: std::collections::HashSet<String> = Default::default();
        for &r in rows {
            match t.value_at(r, c) {
                Ok(Value::Number(n)) => {
                    stats.numbers += 1;
                    stats.sum += n;
                    if stats.min.is_none_or(|(m, _)| n < m) {
                        stats.min = Some((n, r));
                    }
                    if stats.max.is_none_or(|(m, _)| n > m) {
                        stats.max = Some((n, r));
                    }
                }
                Ok(Value::Empty) => stats.blanks += 1,
                Ok(Value::Text(s)) if s.trim().is_empty() => stats.blanks += 1,
                Ok(other) => {
                    stats.texts += 1;
                    seen.insert(other.display().trim().to_lowercase());
                }
                Err(_) => stats.errors += 1,
            }
        }
        stats.distinct_text = seen.len();
        stats
    }

    fn mean(&self) -> Option<f64> {
        (self.numbers > 0).then(|| self.sum / self.numbers as f64)
    }

    fn address(&self, sheet: &str, row: usize) -> String {
        format!(
            "{}!{}{}",
            quote_sheet(sheet),
            gridtalk_graph::col_name(self.column),
            row + 1
        )
    }

    /// `label (where): 12 numbers, sum 1200, mean 100, min 5 (Sheet!C4),
    /// max 300 (Sheet!C9); 2 blank; 1 text (1 distinct); 1 error`.
    fn line(&self, sheet: &str, label: &str, place: &str) -> String {
        let n = |x: f64| CellValue::Number(significant(x, 12)).display();
        let mut parts = Vec::new();
        if self.numbers > 0 {
            let mut figures = format!("{}, sum {}", plural(self.numbers, "number"), n(self.sum));
            if let Some(mean) = self.mean() {
                figures.push_str(&format!(", mean {}", n(mean)));
            }
            if let (Some((lo, lo_row)), Some((hi, hi_row))) = (self.min, self.max) {
                figures.push_str(&format!(
                    ", min {} ({}), max {} ({})",
                    n(lo),
                    self.address(sheet, lo_row),
                    n(hi),
                    self.address(sheet, hi_row)
                ));
            }
            parts.push(figures);
        }
        if self.texts > 0 {
            parts.push(format!(
                "{} text ({} distinct)",
                self.texts, self.distinct_text
            ));
        }
        if self.blanks > 0 {
            parts.push(format!("{} blank", self.blanks));
        }
        if self.errors > 0 {
            parts.push(plural(self.errors, "error"));
        }
        if parts.is_empty() {
            parts.push("no rows".into());
        }
        format!("{label} ({place}): {}", parts.join("; "))
    }

    fn json(&self, sheet: &str) -> serde_json::Value {
        let at = |m: Option<(f64, usize)>| {
            m.map(|(v, r)| serde_json::json!({"value": v, "address": self.address(sheet, r)}))
        };
        serde_json::json!({
            "rows": self.rows, "numbers": self.numbers, "text": self.texts,
            "distinct_text": self.distinct_text, "blanks": self.blanks, "errors": self.errors,
            "sum": (self.numbers > 0).then_some(self.sum), "mean": self.mean(),
            "min": at(self.min), "max": at(self.max),
        })
    }
}

/// `x` rounded to `digits` significant digits, so a sum of many
/// decimals reads `447088.07`, not `447088.0700000005`. The JSON keeps
/// the exact value.
fn significant(x: f64, digits: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let decimals = (digits - 1 - x.abs().log10().floor() as i32).clamp(0, 15);
    let scale = 10f64.powi(decimals);
    (x * scale).round() / scale
}

/// `1 row`, `3 rows`.
fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("1 {word}")
    } else {
        format!("{n} {word}s")
    }
}

/// Most groups `aggregate` lists; `groups_total` counts them all.
pub const MAX_AGGREGATE_GROUPS: usize = 100;

/// Rows `filter` returns by default.
pub const DEFAULT_FILTER_ROWS: usize = 50;
/// Most rows one `filter` call returns; `matched` counts them all.
pub const MAX_FILTER_ROWS: usize = 200;

/// Whether a cell's value meets a `filter` condition. Numbers compare
/// as numbers when the condition's value reads as one (`5000`,
/// `12.5%`); otherwise both sides compare as text, ignoring case and
/// surrounding spaces. A number is never ordered against text. A cell
/// whose formula fails meets no condition.
fn condition_holds(value: &Result<Value, String>, condition: &Condition) -> bool {
    let Ok(value) = value else {
        return false;
    };
    let wanted = condition.value.trim();
    let wanted_number = wanted
        .strip_suffix('%')
        .and_then(|n| n.trim().parse::<f64>().ok().map(|n| n / 100.0))
        .or_else(|| wanted.parse::<f64>().ok())
        .filter(|n| n.is_finite());
    let display = value.display();
    let text = display.trim().to_lowercase();
    let wanted_text = wanted.to_lowercase();
    if condition.op == FilterOp::Contains {
        return text.contains(&wanted_text);
    }
    let ordering = match (value, wanted_number) {
        (Value::Number(n), Some(w)) => {
            if (n - w).abs() <= 1e-9 * n.abs().max(w.abs()).max(1.0) {
                std::cmp::Ordering::Equal
            } else if *n < w {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        }
        (Value::Number(_), None) | (Value::Text(_), Some(_))
            if !matches!(condition.op, FilterOp::Eq | FilterOp::Ne) =>
        {
            return false;
        }
        _ => text.cmp(&wanted_text),
    };
    match condition.op {
        FilterOp::Eq => ordering.is_eq(),
        FilterOp::Ne => ordering.is_ne(),
        FilterOp::Gt => ordering.is_gt(),
        FilterOp::Ge => ordering.is_ge(),
        FilterOp::Lt => ordering.is_lt(),
        FilterOp::Le => ordering.is_le(),
        FilterOp::Contains => text.contains(&wanted_text),
    }
}

/// What to say about a condition whose value starts with a
/// comparison (`{"op": ">", "value": "> 44000"}`): compared as text,
/// it would match nothing, and an empty result does not tell a model
/// what it got wrong. `None` for an ordinary value.
fn operator_in_value(condition: &Condition) -> Option<String> {
    let value = condition.value.trim_start();
    let op = [">=", "<=", "!=", "<>", "==", ">", "<", "="]
        .into_iter()
        .find(|op| value.starts_with(op))?;
    let rest = value[op.len()..].trim();
    let op = match op {
        "<>" => "!=",
        "==" => "=",
        other => other,
    };
    Some(format!(
        "the value of the condition on {} starts with {op:?}; put the comparison in op and only the value in value: {}",
        condition.column,
        serde_json::json!({"column": condition.column, "op": op, "value": rest})
    ))
}

/// A column given by its letters (`D`, `aa`), 0-based, within
/// Excel's columns.
fn column_letters(text: &str) -> Option<usize> {
    if text.is_empty() || text.len() > 3 || !text.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let col = text
        .to_ascii_uppercase()
        .bytes()
        .fold(0usize, |acc, b| acc * 26 + usize::from(b - b'A' + 1));
    (col <= gridtalk_graph::EXCEL_MAX_COLS).then(|| col - 1)
}

/// The candidate closest to `name` when it is near enough to be what
/// was meant, ignoring case: within one edit for names of up to five
/// characters, two beyond; failing that, the one candidate that starts
/// with `name` (or that `name` starts with), or else the one that
/// holds `name` as a word, when only one does.
pub fn did_you_mean<'a>(name: &str, candidates: &[&'a str]) -> Option<&'a str> {
    let wanted = name.trim().to_lowercase();
    if wanted.is_empty() {
        return None;
    }
    let allowed = if wanted.chars().count() <= 5 { 1 } else { 2 };
    let best = candidates
        .iter()
        .map(|c| (edit_distance(&wanted, &c.to_lowercase()), *c))
        .filter(|(d, _)| *d <= allowed)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c);
    if best.is_some() {
        return best;
    }
    let prefixed: Vec<&str> = candidates
        .iter()
        .filter(|c| {
            let c = c.to_lowercase();
            wanted.chars().count() >= 3 && (c.starts_with(&wanted) || wanted.starts_with(&c))
        })
        .copied()
        .collect();
    if prefixed.len() == 1 {
        return Some(prefixed[0]);
    }
    // `Type` for `Debt Type`: the one name holding it as a word.
    let worded: Vec<&str> = candidates
        .iter()
        .filter(|c| {
            c.to_lowercase()
                .split(|ch: char| !ch.is_alphanumeric())
                .any(|word| word == wanted)
        })
        .copied()
        .collect();
    (worded.len() == 1).then(|| worded[0])
}

/// Levenshtein distance over chars.
pub fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(ca != cb))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Most issues `issues` lists; the counts cover every issue.
pub const MAX_ISSUES_LISTED: usize = 40;
/// Cells an issue shows in its text line.
const MAX_ISSUE_CELLS_TEXT: usize = 3;
/// Cells an issue carries in its JSON.
const MAX_ISSUE_CELLS_JSON: usize = 20;

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

/// Why a what-if cell or target did not parse. `shown` is the
/// caller's spelling; `qualified` the sheet-qualified reference that
/// was parsed (a reference with no sheet is unparseable, not outside
/// the grid).
fn cell_error(what: &str, shown: &str, qualified: &str) -> String {
    let outside = split_sheet_prefix(qualified)
        .and_then(|(sheet, cell)| parse_a1(&sheet, cell))
        .is_some();
    if outside {
        format!("{what} {shown} is outside the sheet grid, which ends at XFD1048576")
    } else {
        format!("cannot parse {what} {shown}")
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

    /// `issues` over a CSV: an error that starts in one cell and is
    /// passed on, a number written into a column of formulas, the
    /// typed request with and without a range, and the legacy form.
    #[test]
    fn issues_lists_errors_first_and_groups_a_column() {
        let wb = parse_delimited(
            "demo",
            "Calc",
            "Qty,Price,Net,Share\n2,0,=A2*1.15,=A2/B2\n4,5,=A3*1.15,=D2+1\n6,7,=A4*1.15,=A4/B4\n",
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
        let res = tools.call(&ToolRequest::Issues { range: None });
        assert!(res.ok, "{}", res.text);
        assert_eq!(res.data["errors"], 1, "{}", res.text);
        assert_eq!(res.data["warnings"], 2, "{}", res.text);
        let issues = res.data["issues"].as_array().unwrap();
        assert_eq!(issues[0]["kind"], "formula_error");
        assert_eq!(issues[0]["range"], "Calc!D2");
        assert_eq!(issues[0]["cells"][0]["address"], "Calc!D2");
        let net = issues
            .iter()
            .find(|i| i["kind"] == "constant_in_formula")
            .expect("the written-in 1.15");
        assert_eq!(net["range"], "Calc!C2:C4");
        assert_eq!(net["count"], 3);
        assert_eq!(net["detail"], "1.15");
        assert!(issues
            .iter()
            .any(|i| i["kind"] == "error_from_precedent" && i["range"] == "Calc!D3"));
        let lines: Vec<&str> = res.text.lines().collect();
        assert_eq!(lines[0], "1 error and 2 warnings in the workbook");
        assert!(
            lines[1].starts_with("error formula_error Calc!D2: evaluates to #DIV/0!"),
            "{}",
            res.text
        );
        assert!(
            res.text
                .contains("warning constant_in_formula Calc!C2:C4 (3 cells): writes 1.15"),
            "{}",
            res.text
        );

        // A range limits the check; a column without problems has none.
        let res = tools.call(&ToolRequest::Issues {
            range: Some("calc!A1:B4".into()),
        });
        assert_eq!(res.text, "no issues found in calc!A1:B4");
        assert_eq!(res.data["total"], 0);
        let res = tools.call(&ToolRequest::Issues {
            range: Some("Nope!A1".into()),
        });
        assert_eq!(res.data["error"]["kind"], "unknown_sheet");

        // The JSON request: `range` is optional, unknown fields are not.
        let req: ToolRequest = serde_json::from_str(r#"{"tool":"issues"}"#).unwrap();
        assert_eq!(req, ToolRequest::Issues { range: None });
        assert!(
            serde_json::from_str::<ToolRequest>(r#"{"tool":"issues","sheet":"Calc"}"#).is_err()
        );
        assert!(tools
            .execute("issues", "")
            .starts_with("1 error and 2 warnings"));
        assert!(tools
            .execute("issues", "range=Calc!D2:D4")
            .starts_with("1 error and 1 warning in Calc!D2:D4"));
    }

    /// `filter` on a table with a heading row: text compares ignoring
    /// case, numbers as numbers, formula cells by what they compute;
    /// rows come with addresses; a near-miss column or sheet name is
    /// answered with the name meant.
    #[test]
    fn filter_finds_rows_by_value_with_addresses() {
        let wb = parse_delimited(
            "demo",
            "Sales",
            "Account,Region,Amount,Net\nA-1,EMEA,7000,=C2*0.9\nA-2,APAC,9000,=C3*0.9\nA-3,emea,4000,=C4*0.9\nA-4,EMEA North,12000,=C5*0.9\nA-5,EMEA,abc,=1/0\n",
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
        let call = |json: serde_json::Value| {
            let req: ToolRequest = serde_json::from_value(json).unwrap();
            tools.call(&req)
        };
        let rows = |res: &ToolResult| -> Vec<u64> {
            res.data["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["row"].as_u64().unwrap())
                .collect()
        };

        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!B3",
            "where": [{"column": "region", "op": "=", "value": "EMEA"}]}));
        assert!(res.ok, "{}", res.text);
        assert_eq!(rows(&res), vec![2, 4, 6]);
        assert_eq!(res.data["matched"], 3);
        assert_eq!(res.data["rows_checked"], 5);
        let lines: Vec<&str> = res.text.lines().collect();
        assert_eq!(
            lines[0],
            "Table Sales!A1:D6 where Region = EMEA: 3 of 5 rows match"
        );
        assert_eq!(
            lines[1],
            "Sales row 2: A2 Account = A-1 | B2 Region = EMEA | C2 Amount = 7000 | D2 Net = 6300"
        );
        assert_eq!(res.data["rows"][0]["range"], "Sales!A2:D2");
        assert_eq!(res.data["rows"][0]["cells"][3]["address"], "Sales!D2");
        assert_eq!(res.data["rows"][0]["cells"][3]["value"], 6300.0);
        // A formula that fails shows its error and matches nothing on
        // that column.
        assert_eq!(res.data["rows"][2]["cells"][3]["type"], "error");

        // Numbers as numbers (a value given as a JSON number too), on
        // a formula column, by letter, with two conditions and a
        // subset of columns.
        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!A1:D6",
            "where": [{"column": "D", "op": ">=", "value": 6300}, {"column": "Region", "op": "contains", "value": "emea"}],
            "columns": ["Account", "Net"]}));
        assert_eq!(rows(&res), vec![2, 5], "{}", res.text);
        assert_eq!(
            res.text.lines().nth(2).unwrap(),
            "Sales row 5: A5 Account = A-4 | D5 Net = 10800"
        );
        assert_eq!(res.data["rows"][1]["range"], "Sales!A5:D5");
        // Text is never ordered against a number.
        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!A1",
            "where": [{"column": "Amount", "op": "<", "value": "5000"}]}));
        assert_eq!(rows(&res), vec![4]);
        // A range of several rows checks only those rows.
        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!A4:D6"}));
        assert_eq!(rows(&res), vec![4, 5, 6]);
        assert!(
            res.text
                .starts_with("Table Sales!A1:D6, rows Sales!A4:D6: 3 of 3 rows match"),
            "{}",
            res.text
        );
        // The limit cuts the rows shown, not the count.
        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!A1", "limit": 2}));
        assert_eq!(rows(&res), vec![2, 3]);
        assert_eq!(res.data["matched"], 5);
        assert_eq!(res.data["truncated"], true);

        // A comparison written into the value is an error that says
        // where it goes, for filter and aggregate alike.
        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!A1",
            "where": [{"column": "Amount", "op": ">", "value": "> 5000"}]}));
        assert_eq!(res.data["error"]["kind"], "invalid_arguments");
        assert_eq!(
            res.text,
            r#"the value of the condition on Amount starts with ">"; put the comparison in op and only the value in value: {"column":"Amount","op":">","value":"5000"}"#
        );
        let res = call(
            serde_json::json!({"tool": "aggregate", "range": "Sales!A1", "column": "Amount",
            "where": [{"column": "Region", "op": "=", "value": "<>APAC"}]}),
        );
        assert!(
            res.text
                .contains(r#"{"column":"Region","op":"!=","value":"APAC"}"#),
            "{}",
            res.text
        );

        // Names a model gets nearly right.
        let res = call(serde_json::json!({"tool": "filter", "range": "Sales!A1",
            "where": [{"column": "Regoin", "op": "=", "value": "EMEA"}]}));
        assert_eq!(res.data["error"]["kind"], "unknown_column");
        assert_eq!(
            res.text,
            "no column Regoin in Sales!A1:D6 (did you mean Region?); its columns are Account (A), Region (B), Amount (C), Net (D)"
        );
        let res = call(serde_json::json!({"tool": "filter", "range": "Sale!A1"}));
        assert_eq!(
            res.text,
            "no sheet named Sale (did you mean Sales?); the sheets are Sales"
        );
        let res = tools.call(&ToolRequest::GetCells {
            range: "sles!A1".into(),
        });
        assert!(res.text.contains("(did you mean Sales?)"), "{}", res.text);
        assert!(serde_json::from_value::<ToolRequest>(
            serde_json::json!({"tool": "filter", "range": "Sales!A1",
            "where": [{"column": "Region", "op": "~", "value": "x"}]})
        )
        .is_err());
    }

    /// `aggregate` over the same table as the `filter` test: one
    /// column, per group (merged ignoring case), rows per group, every
    /// column, and only the rows a condition keeps.
    #[test]
    fn aggregate_adds_up_columns_and_groups() {
        let wb = parse_delimited(
            "demo",
            "Sales",
            "Account,Region,Amount,Net\nA-1,EMEA,7000,=C2*0.9\nA-2,APAC,9000,=C3*0.9\nA-3,emea,4000,=C4*0.9\nA-4,EMEA North,12000,=C5*0.9\nA-5,EMEA,abc,=1/0\n",
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
        let call = |json: serde_json::Value| {
            let req: ToolRequest = serde_json::from_value(json).unwrap();
            tools.call(&req)
        };

        let res =
            call(serde_json::json!({"tool": "aggregate", "range": "Sales!A1", "column": "amount"}));
        assert!(res.ok, "{}", res.text);
        assert_eq!(
            res.text,
            "Table Sales!A1:D6: 5 of 5 rows match\nAmount (Sales!C2:C6): 4 numbers, sum 32000, mean 8000, min 4000 (Sales!C4), max 12000 (Sales!C5); 1 text (1 distinct)"
        );
        let amount = &res.data["columns"][0];
        assert_eq!(amount["sum"], 32000.0);
        assert_eq!(amount["min"]["address"], "Sales!C4");
        assert_eq!(amount["text"], 1);

        // Per group, on a formula column: the failed formula is an
        // error, not a zero.
        let res = call(
            serde_json::json!({"tool": "aggregate", "range": "Sales!A1:D6", "column": "Net", "group_by": "B"}),
        );
        let lines: Vec<&str> = res.text.lines().collect();
        assert_eq!(lines[1], "3 groups of Net by Region", "{}", res.text);
        assert_eq!(
            lines[2],
            "EMEA (3 rows): 2 numbers, sum 9900, mean 4950, min 3600 (Sales!D4), max 6300 (Sales!D2); 1 error"
        );
        assert!(
            lines[3].starts_with("APAC (1 row): 1 number, sum 8100"),
            "{}",
            res.text
        );
        assert!(lines[4].starts_with("EMEA North (1 row)"), "{}", res.text);
        assert_eq!(res.data["groups"][0]["group"], "EMEA");
        assert_eq!(res.data["groups"][0]["sum"], 9900.0);
        assert_eq!(res.data["groups"][0]["errors"], 1);
        assert_eq!(res.data["groups_total"], 3);

        // Rows per group; a profile of every column; a condition.
        let res = call(
            serde_json::json!({"tool": "aggregate", "range": "Sales!A1", "group_by": "Region"}),
        );
        assert!(
            res.text
                .ends_with("\nEMEA: 3 rows\nAPAC: 1 row\nEMEA North: 1 row"),
            "{}",
            res.text
        );
        let res = call(serde_json::json!({"tool": "aggregate", "range": "Sales!B2"}));
        assert_eq!(res.data["columns"].as_array().unwrap().len(), 4);
        assert!(
            res.text
                .contains("\nAccount (Sales!A2:A6): 5 text (5 distinct)"),
            "{}",
            res.text
        );
        assert!(
            res.text
                .contains("\nRegion (Sales!B2:B6): 5 text (3 distinct)"),
            "{}",
            res.text
        );
        let res = call(
            serde_json::json!({"tool": "aggregate", "range": "Sales!A1", "column": "Amount",
            "where": [{"column": "Region", "op": "contains", "value": "emea"}]}),
        );
        assert!(res.text.starts_with("Table Sales!A1:D6 where Region contains emea: 4 of 5 rows match\nAmount (Sales!C2:C6): 3 numbers, sum 23000"), "{}", res.text);
        let res = call(
            serde_json::json!({"tool": "aggregate", "range": "Sales!A1", "group_by": "Regoin"}),
        );
        assert_eq!(res.data["error"]["kind"], "unknown_column");
        assert!(res.text.contains("(did you mean Region?)"), "{}", res.text);
    }

    #[test]
    fn did_you_mean_takes_near_names_only() {
        let names = ["Labour", "Rates", "Summary 2024"];
        assert_eq!(did_you_mean("labor", &names), Some("Labour"));
        assert_eq!(did_you_mean("Rats", &names), Some("Rates"));
        assert_eq!(did_you_mean("Summary", &names), Some("Summary 2024"));
        assert_eq!(did_you_mean("Debtors", &names), None);
        assert_eq!(
            did_you_mean("type", &["Debt Type", "Account"]),
            Some("Debt Type")
        );
        assert_eq!(
            did_you_mean("type", &["Debt Type", "Type Code"]),
            Some("Type Code")
        );
        assert_eq!(did_you_mean("rate", &["Discount Rate", "Tax Rate"]), None);
        assert_eq!(did_you_mean("", &names), None);
        assert_eq!(significant(447_088.070_000_000_5, 12), 447_088.07);
        assert_eq!(significant(931.433_479_166_667_7, 12), 931.433_479_167);
        assert_eq!(significant(1e20, 12), 1e20);
        assert_eq!(column_letters("d"), Some(3));
        assert_eq!(column_letters("XFD"), Some(16_383));
        assert_eq!(column_letters("XFE"), None);
        assert_eq!(column_letters("Region"), None);
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
            res.text.contains("Sheet1!C2 = A2&\" x\"&B2 -> Apples x2"),
            "{}",
            res.text
        );
        assert!(
            res.text.contains("Sheet1!D2 = B2>5 -> FALSE"),
            "{}",
            res.text
        );
        assert_eq!(f[0]["address"], "Sheet1!C2");

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
        assert_eq!(d2["address"], "Sheet1!D2");
        assert!(
            out.text
                .contains("Sheet1!B2: 2\nSheet1!C2: 3\nSheet1!D2: 6  [=B2*C2]\nSheet1!E2: #ERROR("),
            "{}",
            out.text
        );
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
        // A target with no sheet does not parse; it is not off-grid.
        let unqualified = tools.call(&ToolRequest::WhatIf {
            cell: "B2".into(),
            value: 1.0,
            target: "D2".into(),
        });
        assert!(
            unqualified.text.contains("cannot parse target D2"),
            "{}",
            unqualified.text
        );
    }
    /// #44: a reader can cite any cell either tool shows without
    /// counting columns, and a column's formulas are summed up so a
    /// typed-over cell is named instead of left among the rows.
    #[test]
    fn cells_and_formulas_carry_their_addresses() {
        let mut csv = String::from("Item,Qty,Price,Total\n");
        for r in 2..=30 {
            // Row 9 was typed over; row 12 holds another formula.
            let total = match r {
                9 => "77".to_string(),
                12 => format!("=B{r}+C{r}"),
                _ => format!("=B{r}*C{r}"),
            };
            csv.push_str(&format!("item{r},{r},2,{total}\n"));
        }
        let wb = parse_delimited("d", "My Data", &csv, ',');
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        // A few cells: one line each, the sheet name quoted as a
        // citation needs it.
        let small = tools.call(&ToolRequest::GetCells {
            range: "'my data'!A1:D2".into(),
        });
        assert!(
            small.text.contains("'My Data'!D1: Total\n"),
            "{}",
            small.text
        );
        assert!(
            small.text.contains("'My Data'!D2: 4  [=B2*C2]\n"),
            "{}",
            small.text
        );
        assert_eq!(small.data["rows"][1][3]["address"], "'My Data'!D2");
        assert_eq!(
            small.data["columns"],
            serde_json::json!(["A", "B", "C", "D"])
        );
        assert_eq!(small.data["first_row"], 1);
        // Many cells: one line per sheet row, each value after its
        // column letter.
        let tall = tools.call(&ToolRequest::GetCells {
            range: "'My Data'!A1:D30".into(),
        });
        assert!(
            tall.text.contains("\n9 | A: item9 | B: 9 | C: 2 | D: 77\n"),
            "{}",
            tall.text
        );
        assert!(tall.text.contains("'My Data'!A1"), "{}", tall.text);
        // Empty cells are counted, not printed.
        let sparse = tools.call(&ToolRequest::GetCells {
            range: "'My Data'!D30:E31".into(),
        });
        assert!(
            sparse.text.contains("(3 empty cells not listed)"),
            "{}",
            sparse.text
        );

        let formulas = tools.call(&ToolRequest::ListFormulas {
            range: "'My Data'!A1:D30".into(),
        });
        let first = formulas.text.lines().next().unwrap();
        assert_eq!(
            first,
            "column D ('My Data'!D2:D30): 28 formulas, 27 like 'My Data'!D2 = B2*C2; \
             a different formula in 'My Data'!D12; \
             a typed value, not a formula, in 'My Data'!D9 (77)"
        );
        assert!(
            formulas.text.contains("\n'My Data'!D3 = B3*C3 -> 6\n"),
            "{}",
            formulas.text
        );
        let column = &formulas.data["columns"][0];
        assert_eq!(column["pattern"], "RC[-2]*RC[-1]");
        assert_eq!(
            column["other_formulas"],
            serde_json::json!(["'My Data'!D12"])
        );
        assert_eq!(column["typed_values"][0]["address"], "'My Data'!D9");
        // A value typed over the last formula is still inside the
        // column, and a column with no exception says so.
        let mut wb2 = wb.clone();
        *wb2.sheets[0].cell_mut(29, 3) = CellValue::Number(5.0);
        *wb2.sheets[0].cell_mut(8, 3) = CellValue::Formula {
            expr: "B9*C9".into(),
            cached: None,
        };
        *wb2.sheets[0].cell_mut(11, 3) = CellValue::Formula {
            expr: "B12*C12".into(),
            cached: None,
        };
        let (g2, _) = build_graph(&wb2);
        let hybrid2 = HybridIndex::build(&wb2, &g2);
        let tools2 = Tools {
            wb: &wb2,
            graph: &g2,
            hybrid: &hybrid2,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let last = tools2.call(&ToolRequest::ListFormulas {
            range: "'My Data'!D1:D30".into(),
        });
        assert!(
            last.text
                .starts_with("column D ('My Data'!D2:D30): 28 formulas, all like"),
            "{}",
            last.text
        );
        assert!(
            last.text
                .contains("a typed value, not a formula, in 'My Data'!D30 (5)"),
            "{}",
            last.text
        );
        *wb2.sheets[0].cell_mut(29, 3) = CellValue::Formula {
            expr: "B30*C30".into(),
            cached: None,
        };
        let (g3, _) = build_graph(&wb2);
        let hybrid3 = HybridIndex::build(&wb2, &g3);
        let tools3 = Tools {
            wb: &wb2,
            graph: &g3,
            hybrid: &hybrid3,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let clean = tools3.call(&ToolRequest::ListFormulas {
            range: "'My Data'!D1:D30".into(),
        });
        assert!(
            clean
                .text
                .lines()
                .next()
                .unwrap()
                .ends_with("; no cell breaks the pattern"),
            "{}",
            clean.text
        );
        // A search hit shows the top of its passage.
        let hit = tools.call(&ToolRequest::Search {
            query: "item9 total".into(),
            limit: 1,
        });
        assert!(
            hit.text.contains("\n  Item | Qty | Price | Total"),
            "{}",
            hit.text
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
        // The text names every neighbour with its range, so the model
        // can cite what the column reads (#39).
        let amount = g2.nodes.iter().find(|n| n.label == "Amount").unwrap();
        let amount_a1 = amount.range.as_ref().unwrap().a1();
        let running_a1 = running.range.as_ref().unwrap().a1();
        assert!(
            out2.text
                .contains(&format!("node {} (Running  {running_a1})", running.id)),
            "{}",
            out2.text
        );
        assert!(
            out2.text
                .contains(&format!("-depends_on-> Amount  {amount_a1}")),
            "{}",
            out2.text
        );
        assert!(
            out2.text
                .contains(&format!("-depends_on-> Running  {running_a1}")),
            "{}",
            out2.text
        );
        assert_eq!(out2.data["truncated"], false);
        let info = tools.call(&ToolRequest::WorkbookInfo);
        assert_eq!(info.data["sheets"][0]["cols"], 250);
        assert_eq!(info.data["limits"]["expand_edges"], MAX_EXPAND_EDGES);
    }

    #[test]
    fn workbook_info_and_graph_overview_say_which_sheets_are_hidden() {
        let (mut wb, _, _) = fixture();
        wb.sheets[0].visibility = gridtalk_graph::SheetVisibility::Hidden;
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty());
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let info = tools.call(&ToolRequest::WorkbookInfo);
        assert_eq!(info.data["sheets"][0]["visibility"], "hidden");
        assert!(info.text.contains("hidden"), "{}", info.text);
        let overview = tools.call(&ToolRequest::GraphOverview);
        let sheet = overview.data["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["kind"] == "Sheet")
            .expect("a sheet node");
        assert_eq!(sheet["visibility"], "hidden");
        // Snapshot and graph JSON written before the field existed
        // still load: sheets as visible, nodes with no visibility.
        let mut old = serde_json::to_value(&wb.sheets[0]).unwrap();
        old.as_object_mut().unwrap().remove("visibility");
        let old: gridtalk_graph::SheetSnapshot = serde_json::from_value(old).unwrap();
        assert_eq!(old.visibility, gridtalk_graph::SheetVisibility::Visible);
        let mut node = serde_json::to_value(&g.nodes[0]).unwrap();
        node.as_object_mut().unwrap().remove("visibility");
        let node: gridtalk_graph::Node = serde_json::from_value(node).unwrap();
        assert_eq!(node.visibility, None);
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
