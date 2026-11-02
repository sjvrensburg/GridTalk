//! Typed tool core (Phase A of the agentic-interfaces plan).
//! One serde-typed implementation powers the investigate planner,
//! the `gridtalk tools call --json` CLI, and the MCP server —
//! external harnesses never scrape human-formatted strings.

use gridtalk_graph::eval::{eval_cell, what_if};
use gridtalk_graph::{CellValue, Graph, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tool", rename_all = "snake_case")]
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
    pub fn schemas() -> serde_json::Value {
        serde_json::json!([
            {"name": "search", "description": "Hybrid lexical+embedding search over workbook regions", "input": {"query": "string", "limit": "usize (default 3)"}},
            {"name": "expand", "description": "Graph neighbours of the region containing a range", "input": {"range": "Sheet1!A1:D4"}},
            {"name": "cells", "description": "Typed cell values and formulas in a range", "input": {"range": "Sheet1!A1:D4"}},
            {"name": "formulas", "description": "Formulas in a range with evaluated values", "input": {"range": "Sheet1!A1:D4"}},
            {"name": "what_if", "description": "Simulate overriding one cell and re-evaluating a target formula (read-only)", "input": {"cell": "B2 or Sheet1!B2", "value": 10.0, "target": "Sheet1!D4"}},
            {"name": "graph_overview", "description": "All graph nodes and edges", "input": {}}
        ])
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
        self.wb.sheets.iter().find(|s| s.name == name)
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
                let node = self
                    .graph
                    .nodes
                    .iter()
                    .filter(|n| {
                        matches!(
                            n.kind,
                            gridtalk_graph::NodeKind::Region | gridtalk_graph::NodeKind::Column
                        )
                    })
                    .find(|n| n.range.as_ref().map(|r| r.a1()) == Some(range.clone()))
                    .or_else(|| {
                        self.graph
                            .nodes
                            .iter()
                            .find(|n| n.range.as_ref().map(|r| r.a1()) == Some(range.clone()))
                    });
                match node {
                    Some(n) => {
                        let mut edges = Vec::new();
                        let mut lines = vec![format!("node {} ({})", n.id, n.label)];
                        for e in &self.graph.edges {
                            if e.from == n.id {
                                if let Some(t) = self.graph.nodes.iter().find(|x| x.id == e.to) {
                                    lines.push(format!(" -{}-> {}", e.kind, t.label));
                                    edges.push(serde_json::json!({
                                        "direction": "out", "kind": e.kind, "label": t.label,
                                        "range": t.range.as_ref().map(|r| r.a1())
                                    }));
                                }
                            }
                            if e.to == n.id {
                                if let Some(t) = self.graph.nodes.iter().find(|x| x.id == e.from) {
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
                None => ToolResult::err("cells", format!("cannot parse range {range}")),
            },
            ToolRequest::Formulas { range } => match parse_range(range) {
                Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                    Some(sheet) => {
                        let mut formulas = Vec::new();
                        let mut lines = Vec::new();
                        for r in r0..=r1 {
                            for c in c0..=c1 {
                                if let CellValue::Formula { expr, .. } = sheet.get(r, c) {
                                    let (value, error) = match eval_cell(sheet, r, c) {
                                        Ok(v) => (Some(v), None),
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
                None => ToolResult::err("formulas", format!("cannot parse range {range}")),
            },
            ToolRequest::WhatIf {
                cell,
                value,
                target,
            } => {
                if !value.is_finite() {
                    return ToolResult::err("what_if", "what-if value must be finite".into());
                }
                let Some((sheet_name, tr, tc)) = parse_cell_ref(target) else {
                    return ToolResult::err("what_if", format!("cannot parse target {target}"));
                };
                let qualified = if cell.contains('!') {
                    cell.clone()
                } else {
                    format!("{sheet_name}!{cell}")
                };
                let Some((_, orow, ocol)) = parse_cell_ref(&qualified) else {
                    return ToolResult::err("what_if", format!("cannot parse cell {cell}"));
                };
                match self.sheet(&sheet_name) {
                    Some(sheet) => match what_if(sheet, (orow, ocol), *value, (tr, tc)) {
                        Ok((before, after)) => ToolResult::ok(
                            "what_if",
                            serde_json::json!({"cell": cell, "value": value, "target": target, "before": before, "after": after}),
                            format!("what-if {cell}={value}: {target} changes {before} -> {after}"),
                        ),
                        Err(e) => ToolResult::err("what_if", format!("what-if failed: {e}")),
                    },
                    None => ToolResult::err("what_if", format!("no sheet named {sheet_name}")),
                }
            }
            ToolRequest::GraphOverview => {
                let nodes: Vec<serde_json::Value> = self
                    .graph
                    .nodes
                    .iter()
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
                    .map(|e| serde_json::json!({"from": e.from, "to": e.to, "kind": e.kind}))
                    .collect();
                let text = format!("{} nodes, {} edges", nodes.len(), edges.len());
                ToolResult::ok(
                    "graph_overview",
                    serde_json::json!({"nodes": nodes, "edges": edges}),
                    text,
                )
            }
        }
    }

    /// Legacy string interface kept for the planner's text context:
    /// parses `key=value;key=value` and renders the typed result.
    pub fn execute(&self, tool: &str, args: &str) -> String {
        let arg = |key: &str| -> Option<String> {
            args.split(';').find_map(|kv| {
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

/// `Sheet1!A1:D4` -> (sheet, r0, c0, r1, c1), 0-based inclusive.
/// A single cell is a 1x1 range; reversed ranges are rejected.
pub fn parse_range(a1: &str) -> Option<(String, usize, usize, usize, usize)> {
    let (sheet, rest) = a1.split_once('!')?;
    if let Some((a, b)) = rest.split_once(':') {
        let (_, r0, c0) = parse_cell_ref(&format!("{sheet}!{a}"))?;
        let (_, r1, c1) = parse_cell_ref(&format!("{sheet}!{b}"))?;
        if r1 < r0 || c1 < c0 {
            return None;
        }
        Some((sheet.to_string(), r0, c0, r1, c1))
    } else {
        let (_, r, c) = parse_cell_ref(a1)?;
        Some((sheet.to_string(), r, c, r, c))
    }
}

/// `Sheet1!B2` -> (sheet, row, col), 0-based.
pub fn parse_cell_ref(a1: &str) -> Option<(String, usize, usize)> {
    let (sheet, cell) = a1.split_once('!')?;
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
    Some((sheet.to_string(), row - 1, col - 1))
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
