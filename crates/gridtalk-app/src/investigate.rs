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
#[allow(dead_code)] // convenience wrapper over investigate_with
pub fn investigate(
    question: &str,
    tools: &Tools,
    backend: &dyn ChatBackend,
    max_steps: usize,
) -> Result<Investigation, gridtalk_infer::InferError> {
    investigate_with(question, tools, backend, max_steps, None, None)
}

/// Adapter-aware variant: the adapter's rules/skills are appended
/// to the system prompt, its `top_k` sizes the search, and an
/// optional per-workbook sidecar's rules are appended last.
pub fn investigate_with(
    question: &str,
    tools: &Tools,
    backend: &dyn ChatBackend,
    max_steps: usize,
    adapter: Option<&crate::adapter::Adapter>,
    sidecar: Option<&str>,
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

    if max_steps == 0 {
        // The step budget gates every tool call, including search.
        return Ok(Investigation {
            answer: String::new(),
            trail: Vec::new(),
            citations: Vec::new(),
        });
    }

    // Citations come from the structured hits, never scraped from
    // display text. The search step reuses this one search (it is
    // recorded in the trail, not re-executed).
    let top_k = adapter.map(|a| a.top_k).unwrap_or(3);
    let hits = tools.hybrid.search(question, top_k);
    let citations: Vec<String> = hits.iter().map(|h| h.range.a1()).collect();
    let top_range = citations.first().cloned();
    let search_result = if hits.is_empty() {
        format!("no regions matched \"{question}\"")
    } else {
        hits.iter()
            .map(|h| format!("[{}] score {:.3}", h.range.a1(), h.score))
            .collect::<Vec<_>>()
            .join("\n")
    };
    trail.push(TrailStep {
        tool: "search".into(),
        args: format!("q={question}"),
        result: search_result.clone(),
    });

    let mut context = format!("search results:\n{search_result}");
    // Adapter knob: a question naming a cell gets that cell's
    // formulas/cells directly, even when retrieval missed it.
    if adapter.is_some_and(|a| a.ref_context) && trail.len() < max_steps {
        for tok in question.split_whitespace() {
            let clean = tok.trim_matches(|c: char| {
                !(c.is_alphanumeric() || c == '!' || c == '\'' || c == '$')
            });
            if clean.contains('!') && parse_cell_ref(clean).is_some() {
                let formulas = run("formulas", &format!("range={clean}"), &mut trail);
                context.push_str(&format!("\n\ndirect reference {clean}:\n{formulas}"));
                break;
            }
        }
    }
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
                    // Fallback target: the first formula cell in
                    // the region that actually *depends on* the
                    // override — never an unaffected cell.
                    let override_coord = if cell.contains('!') {
                        parse_cell_ref(&cell)
                    } else {
                        parse_cell_ref(&format!("{sheet}!{cell}"))
                    };
                    let dependent_target = || -> Option<String> {
                        let (_, orow, ocol) = override_coord.clone()?;
                        let (_, r0, c0, r1, c1) = parse_range(range)?;
                        let sheet_obj = tools.sheet_for(&sheet)?;
                        for r in r0..=r1 {
                            for c in c0..=c1 {
                                if let CellValue::Formula { expr, .. } = sheet_obj.get(r, c) {
                                    let refs = gridtalk_graph::formula_refs(&expr);
                                    if refs.contains(&(orow, ocol)) {
                                        return Some(format!(
                                            "{}!{}{}",
                                            sheet,
                                            gridtalk_graph::col_name(c),
                                            r + 1
                                        ));
                                    }
                                }
                            }
                        }
                        None
                    };
                    match parse_what_if_target(question).or_else(dependent_target) {
                        Some(target) => {
                            let result = run(
                                "what_if",
                                &format!("cell={cell};value={value};target={target}"),
                                &mut trail,
                            );
                            context.push_str(&format!("\n\nwhat-if:\n{result}"));
                        }
                        None => {
                            context.push_str(
                                "\n\nwhat-if: no formula in the top region depends on the override cell, so no what-if was run.",
                            );
                        }
                    }
                }
            }
        }
    }

    let mut system = "You answer questions about a spreadsheet using only the tool results provided. Always cite the range like Sheet1!A1:D4.".to_string();
    if let Some(a) = adapter {
        system.push_str(&a.prompt_for(question));
    }
    if let Some(sidecar) = sidecar {
        if !sidecar.trim().is_empty() {
            system.push_str("\n\nWorkbook notes (from the .gridtalk.md sidecar):\n");
            system.push_str(sidecar.trim());
        }
    }
    let req = ChatRequest {
        system,
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
    // "... B2 is 10 ..." / "... B2 = 10.5 ..." / "... Sheet1!B2 ..."
    let tokens: Vec<&str> = question.split_whitespace().collect();
    // Sheet-qualified form first (Sheet1!B2 / 'My Sheet'!B2): the
    // token carries '!' and parses as a full cell reference.
    for (i, tok) in tokens.iter().enumerate() {
        let clean =
            tok.trim_matches(|c: char| !(c.is_alphanumeric() || c == '!' || c == '\'' || c == '$'));
        if clean.contains('!') && parse_cell_ref(clean).is_some() {
            for next in tokens.iter().skip(i + 1).take(3) {
                let n = next.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'));
                if let Ok(v) = n.parse::<f64>() {
                    return Some((clean.to_string(), v));
                }
            }
        }
    }
    for (i, tok) in tokens.iter().enumerate() {
        let clean = tok.trim_matches(|c: char| !c.is_alphanumeric());
        // Byte index of the first non-letter, from char_indices —
        // never slice by char count (multi-byte letters would panic).
        let byte_at = |char_idx: usize| {
            clean
                .char_indices()
                .nth(char_idx)
                .map(|(b, _)| b)
                .unwrap_or(clean.len())
        };
        let letters = clean.chars().take_while(|c| c.is_alphabetic()).count();
        let is_ref = clean.len() >= 2
            && clean.chars().take(letters).all(|c| c.is_ascii_alphabetic())
            && (1..=3).contains(&letters)
            && letters < clean.chars().count()
            && clean[byte_at(letters)..]
                .chars()
                .all(|c| c.is_ascii_digit())
            && !clean[byte_at(letters)..].is_empty();
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
        // Multi-byte letters must not panic the byte slice, and a
        // long letter prefix (COVID19) is not a cell reference.
        assert_eq!(parse_what_if("what if \u{e9}2 is 10"), None);
        assert_eq!(parse_what_if("what if COVID19 is 500"), None);
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
        // Sheet-qualified overrides parse too.
        assert_eq!(
            parse_what_if("What if Sheet1!B2 becomes 12? What happens to Sheet1!D2?"),
            Some(("Sheet1!B2".to_string(), 12.0))
        );
    }
}
