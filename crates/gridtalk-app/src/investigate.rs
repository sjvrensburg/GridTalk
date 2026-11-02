//! Investigate harness (Phase 3b): the model — or, with no model,
//! a deterministic planner — drives tools instead of receiving a
//! fixed pipeline's context. Every step is recorded in a trail and
//! the final answer carries its citations. Tools:
//! `search`, `expand`, `cells`, `formulas`, `what_if`.

use gridtalk_graph::eval::{eval_cell, what_if};
use gridtalk_graph::{CellValue, Graph, WorkbookSnapshot};
use gridtalk_index::HybridIndex;
use gridtalk_infer::{ChatBackend, ChatRequest};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct TrailStep {
    pub tool: String,
    pub args: String,
    pub result: String,
}

#[derive(Debug)]
pub struct Investigation {
    pub answer: String,
    pub trail: Vec<TrailStep>,
    pub citations: Vec<String>,
}

pub struct Tools<'a> {
    pub wb: &'a WorkbookSnapshot,
    pub graph: &'a Graph,
    pub hybrid: &'a HybridIndex,
}

impl<'a> Tools<'a> {
    fn sheet(&self, name: &str) -> Option<&gridtalk_graph::SheetSnapshot> {
        self.wb.sheets.iter().find(|s| s.name == name)
    }

    /// Execute one tool by name. Args are `key=value` pairs separated
    /// by `;` (e.g. `q=provision`, `range=Sheet1!A1:D4`,
    /// `cell=B2;value=10;target=D4`).
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
        match tool {
            "search" => {
                let q = arg("q").unwrap_or_default();
                let hits = self.hybrid.search(&q, 3);
                if hits.is_empty() {
                    format!("no regions matched \"{q}\"")
                } else {
                    hits.iter()
                        .map(|h| format!("[{}] score {:.3}", h.range.a1(), h.score))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            "expand" => {
                let range = arg("range").unwrap_or_default();
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
                        let mut out = vec![format!("node {} ({})", n.id, n.label)];
                        for e in &self.graph.edges {
                            if e.from == n.id {
                                if let Some(t) = self.graph.nodes.iter().find(|x| x.id == e.to) {
                                    out.push(format!(" -{}-> {}", e.kind, t.label));
                                }
                            }
                            if e.to == n.id {
                                if let Some(t) = self.graph.nodes.iter().find(|x| x.id == e.from) {
                                    out.push(format!(" <-{}- {}", e.kind, t.label));
                                }
                            }
                        }
                        out.join("\n")
                    }
                    None => format!("no graph node for range {range}"),
                }
            }
            "cells" => {
                let range = arg("range").unwrap_or_default();
                match parse_range(&range) {
                    Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                        Some(sheet) => {
                            let mut out = format!("Range {range}\n");
                            for r in r0..=r1 {
                                let cells: Vec<String> =
                                    (c0..=c1).map(|c| sheet.get(r, c).display()).collect();
                                out.push_str(&cells.join(" | "));
                                out.push('\n');
                            }
                            out
                        }
                        None => format!("no sheet named {sheet_name}"),
                    },
                    None => format!("cannot parse range {range}"),
                }
            }
            "formulas" => {
                let range = arg("range").unwrap_or_default();
                match parse_range(&range) {
                    Some((sheet_name, r0, c0, r1, c1)) => match self.sheet(&sheet_name) {
                        Some(sheet) => {
                            let mut out = Vec::new();
                            for r in r0..=r1 {
                                for c in c0..=c1 {
                                    if let CellValue::Formula { expr, .. } = sheet.get(r, c) {
                                        let value = eval_cell(sheet, r, c)
                                            .map(|v| v.to_string())
                                            .unwrap_or_else(|e| format!("error: {e}"));
                                        out.push(format!(
                                            "{}{} = {expr} -> {value}",
                                            gridtalk_graph::col_name(c),
                                            r + 1
                                        ));
                                    }
                                }
                            }
                            if out.is_empty() {
                                "no formulas in range".into()
                            } else {
                                out.join("\n")
                            }
                        }
                        None => format!("no sheet named {sheet_name}"),
                    },
                    None => format!("cannot parse range {range}"),
                }
            }
            "what_if" => {
                let (Some(cell), Some(value), Some(target)) = (
                    arg("cell"),
                    arg("value").and_then(|v| v.parse::<f64>().ok()),
                    arg("target"),
                ) else {
                    return "usage: cell=B2;value=10;target=Sheet1!D4".into();
                };
                let Some((sheet_name, tr, tc)) = parse_cell_ref(&target) else {
                    return format!("cannot parse target {target}");
                };
                // The override cell is on the same sheet as the target.
                let Some((_, orow, ocol)) = parse_cell_ref(&format!("{sheet_name}!{cell}")) else {
                    return format!("cannot parse cell {cell}");
                };
                match self.sheet(&sheet_name) {
                    Some(sheet) => match what_if(sheet, (orow, ocol), value, (tr, tc)) {
                        Ok((before, after)) => {
                            format!("what-if {cell}={value}: {target} changes {before} -> {after}")
                        }
                        Err(e) => format!("what-if failed: {e}"),
                    },
                    None => format!("no sheet named {sheet_name}"),
                }
            }
            other => {
                format!("unknown tool {other}; available: search, expand, cells, formulas, what_if")
            }
        }
    }
}

/// `Sheet1!A1:D4` -> (sheet, r0, c0, r1, c1), 0-based inclusive.
pub fn parse_range(a1: &str) -> Option<(String, usize, usize, usize, usize)> {
    let (sheet, rest) = a1.split_once('!')?;
    let (a, b) = rest.split_once(':')?;
    let (_, r0, c0) = parse_cell_ref(&format!("{sheet}!{a}"))?;
    let (_, r1, c1) = parse_cell_ref(&format!("{sheet}!{b}"))?;
    Some((sheet.to_string(), r0, c0, r1, c1))
}

/// `Sheet1!B2` -> (sheet, row, col), 0-based.
pub fn parse_cell_ref(a1: &str) -> Option<(String, usize, usize)> {
    let (sheet, cell) = a1.split_once('!')?;
    let cell = cell.trim_matches('$');
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
        col = col * 26 + (ch as usize - 'A' as usize + 1);
    }
    Some((sheet.to_string(), row - 1, col - 1))
}

/// Run an investigation. The deterministic planner always performs
/// search -> cells -> formulas (plus what_if when the question asks
/// for it); the backend then composes the final cited answer from
/// the gathered context. Every step lands in the trail.
pub fn investigate(
    question: &str,
    tools: &Tools,
    backend: &dyn ChatBackend,
    max_steps: usize,
) -> Investigation {
    let mut trail: Vec<TrailStep> = Vec::new();
    let run = |tool: &str, args: &str, trail: &mut Vec<TrailStep>| -> String {
        let result = tools.execute(tool, args);
        trail.push(TrailStep {
            tool: tool.to_string(),
            args: args.to_string(),
            result: result.clone(),
        });
        result
    };

    let search_result = run("search", &format!("q={question}"), &mut trail);
    let top_range = search_result
        .lines()
        .next()
        .and_then(|l| l.strip_prefix('['))
        .and_then(|l| l.split(']').next())
        .map(|s| s.to_string());

    let mut context = format!("search results:\n{search_result}");
    let mut citations: Vec<String> = Vec::new();
    if let Some(range) = &top_range {
        citations.push(range.clone());
        if trail.len() < max_steps {
            let expanded = run("expand", &format!("range={range}"), &mut trail);
            context.push_str(&format!("\n\ngraph expansion:\n{expanded}"));
        }
        if trail.len() < max_steps {
            let cells = run("cells", &format!("range={range}"), &mut trail);
            context.push_str(&format!("\n\ncells:\n{cells}"));
        }
        if trail.len() < max_steps {
            let formulas = run("formulas", &format!("range={range}"), &mut trail);
            context.push_str(&format!("\n\nformulas:\n{formulas}"));
        }
    }

    // What-if: "what if B2 is 10" / "what if B2 = 10"
    let lower = question.to_lowercase();
    if lower.contains("what if") && trail.len() < max_steps {
        if let Some((cell, value)) = parse_what_if(question) {
            if let Some(range) = &top_range {
                if let Some((sheet, _, _, _, _)) = parse_range(range) {
                    // Target: first formula cell in the top region is
                    // resolved by the tool via an explicit target when
                    // the question names one; otherwise use the last
                    // cell of the region as the aggregate.
                    let target =
                        parse_what_if_target(question).unwrap_or_else(|| format!("{sheet}!A1"));
                    let result = run(
                        "what_if",
                        &format!("cell={cell};value={value};target={target}"),
                        &mut trail,
                    );
                    context.push_str(&format!("\n\nwhat-if:\n{result}"));
                }
            }
        }
    }

    let req = ChatRequest {
        system: "You answer questions about a spreadsheet using only the tool results provided. Always cite the range like Sheet1!A1:D4.".into(),
        user: format!("Question: {question}\n\nTool results:\n{context}"),
        max_tokens: 512,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let mut answer = String::new();
    let _ = backend.stream(&req, &cancel, &mut |t| answer.push_str(t));
    if answer.trim().is_empty() {
        answer = context.clone();
    }

    Investigation {
        answer,
        trail,
        citations,
    }
}

fn parse_what_if(question: &str) -> Option<(String, f64)> {
    // "... B2 is 10 ..." / "... B2 = 10.5 ..."
    let tokens: Vec<&str> = question.split_whitespace().collect();
    for (i, tok) in tokens.iter().enumerate() {
        let clean = tok.trim_matches(|c: char| !c.is_alphanumeric());
        let is_ref = clean.len() >= 2
            && clean.chars().take_while(|c| c.is_alphabetic()).count() >= 1
            && clean.chars().any(|c| c.is_ascii_digit())
            && clean.chars().all(|c| c.is_alphanumeric());
        if is_ref {
            for next in tokens.iter().skip(i + 1).take(2) {
                let n = next.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'));
                if let Ok(v) = n.parse::<f64>() {
                    return Some((clean.to_string(), v));
                }
            }
        }
    }
    None
}

fn parse_what_if_target(question: &str) -> Option<String> {
    // "what happens to D4" / "effect on Sheet1!D4"
    for tok in question.split_whitespace() {
        let clean = tok.trim_matches(|c: char| !(c.is_alphanumeric() || c == '!'));
        if clean.contains('!') && parse_cell_ref(clean).is_some() {
            return Some(clean.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::{build_graph, parse_delimited};
    use gridtalk_index::HybridIndex;
    use gridtalk_infer::EchoBackend;

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
    fn tools_execute_and_investigate_trails() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
        };
        assert!(tools.execute("search", "q=provision").contains("Sheet1!"));
        assert!(tools
            .execute("formulas", "range=Sheet1!A1:D3")
            .contains("B2*C2"));
        let inv = investigate("how is the provision calculated", &tools, &EchoBackend, 8);
        assert!(!inv.trail.is_empty());
        assert_eq!(inv.trail[0].tool, "search");
        assert!(!inv.citations.is_empty());
        assert!(inv.answer.contains("Provision") || inv.answer.contains("provision"));
    }

    #[test]
    fn what_if_tool_and_question_parsing() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
        };
        let out = tools.execute("what_if", "cell=B2;value=10;target=Sheet1!D2");
        assert!(out.contains("6 -> 30"), "got: {out}");
        assert_eq!(
            parse_what_if("what if B2 is 10"),
            Some(("B2".to_string(), 10.0))
        );
    }
}
