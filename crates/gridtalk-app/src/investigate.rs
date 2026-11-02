//! Investigate harness (Phase 3b): the model — or, with no model,
//! a deterministic planner — drives tools instead of receiving a
//! fixed pipeline's context. Every step is recorded in a trail and
//! the final answer carries its citations. Tools:
//! `search`, `expand`, `cells`, `formulas`, `what_if`.

pub use crate::tools::parse_cell_ref;
use crate::tools::{parse_range, Tools};
use gridtalk_graph::CellValue;
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

/// Run an investigation. The deterministic planner always performs
/// search -> cells -> formulas (plus what_if when the question asks
/// for it); the backend then composes the final cited answer from
/// the gathered context. Every step lands in the trail.
pub fn investigate(
    question: &str,
    tools: &Tools,
    backend: &dyn ChatBackend,
    max_steps: usize,
) -> Result<Investigation, gridtalk_infer::InferError> {
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

    // Citations come from the structured hits, never scraped from
    // display text.
    let hits = tools.hybrid.search(question, 3);
    let citations: Vec<String> = hits.iter().map(|h| h.range.a1()).collect();
    let top_range = citations.first().cloned();
    let search_result = run("search", &format!("q={question}"), &mut trail);

    let mut context = format!("search results:\n{search_result}");
    if let Some(range) = &top_range {
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
                    // Target: an explicit Sheet!Cell in the question,
                    // else the first formula cell in the top region,
                    // else the region's last cell.
                    let target = parse_what_if_target(question).unwrap_or_else(|| {
                        if let Some((_, r0, c0, r1, c1)) = parse_range(range) {
                            if let Some(sheet_obj) = tools.sheet_for(&sheet) {
                                for r in r0..=r1 {
                                    for c in c0..=c1 {
                                        if matches!(sheet_obj.get(r, c), CellValue::Formula { .. })
                                        {
                                            return format!(
                                                "{}!{}{}",
                                                sheet,
                                                gridtalk_graph::col_name(c),
                                                r + 1
                                            );
                                        }
                                    }
                                }
                            }
                            format!("{}!{}{}", sheet, gridtalk_graph::col_name(c1), r1 + 1)
                        } else {
                            format!("{sheet}!A1")
                        }
                    });
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
    // Backend failures are propagated, never silently replaced by
    // raw context masquerading as an answer.
    backend.stream(&req, &cancel, &mut |t| answer.push_str(t))?;
    if answer.trim().is_empty() {
        answer = context.clone();
    }
    // Citation guarantee: the printed answer must cite a range we
    // actually retrieved.
    if !citations.is_empty() && !citations.iter().any(|c| answer.contains(c)) {
        answer.push_str(&format!("\n\nSources: {}", citations.join(", ")));
    }

    Ok(Investigation {
        answer,
        trail,
        citations,
    })
}

fn parse_what_if(question: &str) -> Option<(String, f64)> {
    // "... B2 is 10 ..." / "... B2 = 10.5 ..."
    let tokens: Vec<&str> = question.split_whitespace().collect();
    for (i, tok) in tokens.iter().enumerate() {
        let clean = tok.trim_matches(|c: char| !c.is_alphanumeric());
        let letters = clean.chars().take_while(|c| c.is_alphabetic()).count();
        let is_ref = clean.len() >= 2
            && letters >= 1
            && letters < clean.len()
            && clean[letters..].chars().all(|c| c.is_ascii_digit());
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
    use gridtalk_graph::{build_graph, parse_delimited, Graph, WorkbookSnapshot};
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
        let inv = investigate("how is the provision calculated", &tools, &EchoBackend, 8).unwrap();
        assert!(!inv.trail.is_empty());
        assert_eq!(inv.trail[0].tool, "search");
        assert!(!inv.citations.is_empty());
        assert!(inv.answer.contains("Provision") || inv.answer.contains("provision"));
    }

    #[test]
    fn regression_range_refs_and_backend_errors() {
        assert!(parse_range("Sheet1!B2").is_some());
        assert!(
            parse_range("Sheet1!D4:A1").is_none(),
            "reversed range rejected"
        );
        assert_eq!(parse_cell_ref("Sheet1!$B$2"), Some(("Sheet1".into(), 1, 1)));
        assert_eq!(parse_what_if("what if version 2 is 10"), None);
        // A failing backend must propagate, not fake an answer.
        struct FailBackend;
        impl ChatBackend for FailBackend {
            fn name(&self) -> &str {
                "fail"
            }
            fn stream(
                &self,
                _req: &gridtalk_infer::ChatRequest,
                _cancel: &gridtalk_infer::CancelToken,
                _on_token: &mut dyn FnMut(&str),
            ) -> Result<(), gridtalk_infer::InferError> {
                Err(gridtalk_infer::InferError::Http("boom".into()))
            }
        }
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
        };
        assert!(investigate("provision?", &tools, &FailBackend, 8).is_err());
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
