//! Typed tool core (Phase A of the agentic-interfaces plan).
//! One serde-typed implementation powers the investigate planner,
//! the `gridtalk tools call --json` CLI, and the MCP server —
//! external harnesses never scrape human-formatted strings.

use gridtalk_graph::eval::{eval_many_in_workbook, what_if_in_workbook};
use gridtalk_graph::{CellValue, Graph, WorkbookSnapshot};
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
    Cells {
        range: String,
    },
    Formulas {
        range: String,
    },
    WhatIf {
        cell: String,
        value: f64,
        target: String,
    },
    GraphOverview,
}

fn default_limit() -> usize {
    3
}

impl ToolRequest {
    #[allow(dead_code)] // used by external tool-list consumers
    pub fn name(&self) -> &'static str {
        match self {
            ToolRequest::Search { .. } => "search",
            ToolRequest::Expand { .. } => "expand",
            ToolRequest::Cells { .. } => "cells",
            ToolRequest::Formulas { .. } => "formulas",
            ToolRequest::WhatIf { .. } => "what_if",
            ToolRequest::GraphOverview => "graph_overview",
        }
    }

    /// JSON Schemas for every tool, for MCP/tool-list consumers.
    /// Each entry carries a real JSON Schema (`inputSchema`) plus the
    /// `tool` discriminator value a `ToolRequest` JSON must use.
    pub fn schemas() -> serde_json::Value {
        let schema = |props: serde_json::Value, required: &[&str]| serde_json::json!({"type": "object", "properties": props, "required": required, "additionalProperties": false});
        serde_json::json!({
            "request_format": "A ToolRequest is a JSON object with a \"tool\" discriminator, e.g. {\"tool\": \"search\", \"query\": \"...\", \"limit\": 3}. Unknown fields are rejected.",
            "tools": [
                {"name": "search", "tool": "search", "description": "Hybrid lexical+embedding search over workbook regions", "inputSchema": schema(serde_json::json!({"query": {"type": "string"}, "limit": {"type": "integer", "default": 3}}), &["query"])},
                {"name": "expand", "tool": "expand", "description": "Graph neighbours of the region containing a range", "inputSchema": schema(serde_json::json!({"range": {"type": "string"}}), &["range"])},
                {"name": "cells", "tool": "cells", "description": "Typed cell values and formulas in a range", "inputSchema": schema(serde_json::json!({"range": {"type": "string"}}), &["range"])},
                {"name": "formulas", "tool": "formulas", "description": "Formulas in a range with evaluated values", "inputSchema": schema(serde_json::json!({"range": {"type": "string"}}), &["range"])},
                {"name": "what_if", "tool": "what_if", "description": "Simulate overriding one cell and re-evaluating a target formula (read-only)", "inputSchema": schema(serde_json::json!({"cell": {"type": "string"}, "value": {"type": "number"}, "target": {"type": "string"}}), &["cell", "value", "target"])},
                {"name": "graph_overview", "tool": "graph_overview", "description": "Graph nodes and edges (capped at 500 nodes / 2000 edges; see total_nodes/total_edges/truncated)", "inputSchema": schema(serde_json::json!({}), &[])}
            ]
        })
    }
}

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
    fn err(tool: &str, message: String) -> Self {
        ToolResult {
            tool: tool.into(),
            ok: false,
            data: serde_json::json!({"error": message}),
            text: message,
        }
    }
}

pub struct Tools<'a> {
    pub wb: &'a WorkbookSnapshot,
    pub graph: &'a Graph,
    pub hybrid: &'a HybridIndex,
}

impl<'a> Tools<'a> {
    pub fn sheet_for(&self, name: &str) -> Option<&gridtalk_graph::SheetSnapshot> {
        self.sheet(name)
    }

    fn sheet(&self, name: &str) -> Option<&gridtalk_graph::SheetSnapshot> {
        // Caller spellings are case-insensitive (expand already
        // canonicalises sheet1!a1:b2; cells/formulas/what_if agree).
        self.wb
            .sheets
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
    }

    /// The one typed entry point every surface calls.
    pub fn call(&self, req: &ToolRequest) -> ToolResult {
        match req {
            ToolRequest::Search { query, limit } => {
                let limit = (*limit).clamp(1, 25);
                let hits = self.hybrid.search(query, limit);
                if hits.is_empty() {
                    return ToolResult::ok(
                        "search",
                        serde_json::json!({"hits": []}),
                        format!("no regions matched \"{query}\""),
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
                        })
                    })
                    .collect();
                let text = hits
                    .iter()
                    .map(|h| format!("[{}] score {:.3}", h.range.a1(), h.score))
                    .collect::<Vec<_>>()
                    .join("\n");
                ToolResult::ok("search", serde_json::json!({"hits": data_hits}), text)
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
                                        r.sheet.eq_ignore_ascii_case(&sheet)
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
                        for e in &self.graph.edges {
                            if e.from == n.id {
                                if let Some(t) = nodes_by_id.get(&e.to) {
                                    lines.push(format!(" -{}-> {}", e.kind, t.label));
                                    edges.push(serde_json::json!({
                                        "direction": "out", "kind": e.kind, "label": t.label,
                                        "range": t.range.as_ref().map(|r| r.a1())
                                    }));
                                }
                            }
                            if e.to == n.id {
                                if let Some(t) = nodes_by_id.get(&e.from) {
                                    lines.push(format!(" <-{}- {}", e.kind, t.label));
                                    edges.push(serde_json::json!({
                                        "direction": "in", "kind": e.kind, "label": t.label,
                                        "range": t.range.as_ref().map(|r| r.a1())
                                    }));
                                }
                            }
                        }
                        ToolResult::ok(
                            "expand",
                            serde_json::json!({"node_id": n.id, "label": n.label, "range": n.range.as_ref().map(|r| r.a1()), "edges": edges}),
                            lines.join("\n"),
                        )
                    }
                    None => ToolResult::err("expand", format!("no graph node for range {range}")),
                }
            }
            ToolRequest::Cells { range } => match parse_range(range) {
                Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                    Some(sheet) => {
                        let mut rows = Vec::new();
                        let mut text = format!("Range {range}\n");
                        for r in r0..=r1 {
                            let mut row = Vec::new();
                            for c in c0..=c1 {
                                let cell = sheet.get(r, c);
                                row.push(cell_json(&cell));
                                text.push_str(&cell.display());
                                if c < c1 {
                                    text.push_str(" | ");
                                }
                            }
                            text.push('\n');
                            rows.push(serde_json::Value::Array(row));
                        }
                        ToolResult::ok(
                            "cells",
                            serde_json::json!({"range": range, "rows": rows}),
                            text,
                        )
                    }
                    None => ToolResult::err("cells", format!("no sheet named {sheet_name}")),
                },
                None => ToolResult::err("cells", range_error(range)),
            },
            ToolRequest::Formulas { range } => match parse_range(range) {
                Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                    Some(sheet) => {
                        // One shared evaluator memo across the
                        // whole range, not a fresh one per cell.
                        let coords: Vec<(usize, usize)> = (r0..=r1)
                            .flat_map(|r| (c0..=c1).map(move |c| (r, c)))
                            .filter(|(r, c)| matches!(sheet.get(*r, *c), CellValue::Formula { .. }))
                            .collect();
                        // Workbook-aware evaluation with one shared
                        // memo (not a fresh EvalState per cell).
                        let evaluated = eval_many_in_workbook(self.wb, &sheet.name, &coords);
                        let mut eval_iter = coords.iter().zip(evaluated.iter());
                        let mut formulas = Vec::new();
                        let mut lines = Vec::new();
                        for r in r0..=r1 {
                            for c in c0..=c1 {
                                if let CellValue::Formula { expr, .. } = sheet.get(r, c) {
                                    let (_, result) = eval_iter.next().unwrap();
                                    let (value, error) = match result {
                                        Ok(v) => (Some(*v), None),
                                        Err(e) => (None, Some(e.to_string())),
                                    };
                                    let cell = format!("{}{}", gridtalk_graph::col_name(c), r + 1);
                                    lines.push(format!(
                                        "{cell} = {expr} -> {}",
                                        value.map(|v| v.to_string()).unwrap_or_else(|| format!(
                                            "error: {}",
                                            error.clone().unwrap_or_default()
                                        ))
                                    ));
                                    formulas.push(serde_json::json!({
                                        "cell": cell, "formula": expr, "value": value, "error": error
                                    }));
                                }
                            }
                        }
                        if formulas.is_empty() {
                            ToolResult::ok(
                                "formulas",
                                serde_json::json!({"formulas": []}),
                                "no formulas in range".into(),
                            )
                        } else {
                            ToolResult::ok(
                                "formulas",
                                serde_json::json!({"formulas": formulas}),
                                lines.join("\n"),
                            )
                        }
                    }
                    None => ToolResult::err("formulas", format!("no sheet named {sheet_name}")),
                },
                None => ToolResult::err("formulas", range_error(range)),
            },
            ToolRequest::WhatIf {
                cell,
                value,
                target,
            } => {
                if !value.is_finite() {
                    return ToolResult::err("what_if", "what-if value must be finite".into());
                }
                let Some((target_sheet, tr, tc)) = parse_cell_ref(target) else {
                    return ToolResult::err("what_if", format!("cannot parse target {target}"));
                };
                // The override may name its own sheet (Other!B2);
                // it must never silently land on the target sheet.
                let qualified = if cell.contains('!') {
                    cell.clone()
                } else {
                    format!("{}!{cell}", gridtalk_graph::quote_sheet(&target_sheet))
                };
                let Some((override_sheet, orow, ocol)) = parse_cell_ref(&qualified) else {
                    return ToolResult::err("what_if", format!("cannot parse cell {cell}"));
                };
                if self.sheet(&override_sheet).is_none() {
                    return ToolResult::err(
                        "what_if",
                        format!("no sheet named {override_sheet} for override cell {cell}"),
                    );
                }
                if self.sheet(&target_sheet).is_none() {
                    return ToolResult::err("what_if", format!("no sheet named {target_sheet}"));
                }
                // Canonicalise to the workbook's own sheet names:
                // callers may write sheet1!a1 in any case, and the
                // evaluator matches sheet names exactly.
                let override_sheet = self.sheet(&override_sheet).unwrap().name.clone();
                let target_sheet = self.sheet(&target_sheet).unwrap().name.clone();
                match what_if_in_workbook(
                    self.wb,
                    (&override_sheet, orow, ocol),
                    *value,
                    (&target_sheet, tr, tc),
                ) {
                    Ok((before, after)) => ToolResult::ok(
                        "what_if",
                        serde_json::json!({"cell": cell, "value": value, "target": target, "before": before, "after": after}),
                        format!("what-if {cell}={value}: {target} changes {before} -> {after}"),
                    ),
                    Err(e) => ToolResult::err("what_if", format!("what-if failed: {e}")),
                }
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
            "cells" => ToolRequest::Cells {
                range: arg("range").unwrap_or_default(),
            },
            "formulas" => ToolRequest::Formulas {
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
            other => {
                return format!(
                    "unknown tool {other}; available: search, expand, cells, formulas, what_if, graph_overview"
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

/// Largest range a Cells/Formulas call may materialise.
pub const MAX_RANGE_CELLS: usize = 10_000;

/// Split `Sheet!rest` or `'My Sheet'!rest` into (sheet, rest),
/// honouring the doubled-quote escape inside quoted names.
/// Distinguish "too large" from "unparseable" for range errors.
fn range_error(range: &str) -> String {
    if parse_range_uncapped(range).is_some() {
        format!("range {range} exceeds the {MAX_RANGE_CELLS}-cell limit; narrow the range")
    } else {
        format!("cannot parse range {range}")
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

fn parse_cell_on_sheet(sheet: &str, cell: &str) -> Option<(usize, usize)> {
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
    parse_range_inner(a1, true)
}

/// Range parsing without the materialisation cap (for Expand,
/// which never materialises cells).
pub fn parse_range_uncapped(a1: &str) -> Option<(String, usize, usize, usize, usize)> {
    parse_range_inner(a1, false)
}

fn parse_range_inner(a1: &str, capped: bool) -> Option<(String, usize, usize, usize, usize)> {
    let (sheet, rest) = split_sheet_prefix(a1)?;
    if let Some((a, b)) = rest.split_once(':') {
        let (r0, c0) = parse_cell_on_sheet(&sheet, a)?;
        // The end may repeat the sheet qualifier.
        let b_cell = if b.contains('!') {
            let (end_sheet, cell) = split_sheet_prefix(b)?;
            if !end_sheet.eq_ignore_ascii_case(&sheet) {
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
    fn typed_calls_return_structured_data() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
        };
        let search = tools.call(&ToolRequest::Search {
            query: "provision".into(),
            limit: 3,
        });
        assert!(search.ok);
        assert_eq!(search.data["hits"][0]["range"], "Sheet1!A1:D3");
        let formulas = tools.call(&ToolRequest::Formulas {
            range: "Sheet1!A1:D3".into(),
        });
        assert_eq!(formulas.data["formulas"][0]["value"], 6.0);
        let whatif = tools.call(&ToolRequest::WhatIf {
            cell: "B2".into(),
            value: 10.0,
            target: "Sheet1!D2".into(),
        });
        assert_eq!(whatif.data["before"], 6.0);
        assert_eq!(whatif.data["after"], 30.0);
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
                .call(&ToolRequest::Cells {
                    range: "nonsense".into()
                })
                .ok
        );
    }
}
