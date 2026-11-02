//! Investigate harness (Phase 3b): the model — or, with no model,
//! a deterministic planner — drives tools instead of receiving a
//! fixed pipeline's context. Every step is recorded in a trail and
//! the final answer carries its citations. Tools:
//! `search`, `expand`, `cells`, `formulas`, `what_if`.

pub use crate::tools::parse_cell_ref;
use crate::tools::Tools;
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
    let mut citations: Vec<String> = hits.iter().map(|h| h.range.a1()).collect();
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
        if let Some(clean) = qualified_refs(question).into_iter().next() {
            let formulas = run("formulas", &format!("range={clean}"), &mut trail);
            context.push_str(&format!("\n\ndirect reference {clean}:\n{formulas}"));
            if trail.len() < max_steps {
                let cells = run("cells", &format!("range={clean}"), &mut trail);
                context.push_str(&format!("\n\ndirect reference cells {clean}:\n{cells}"));
            }
            // Directly-read ref_context cells are citations too —
            // but only when the named sheet actually exists (the
            // tool calls above return error text for a bad sheet).
            let sheet_ok = parse_cell_ref(&clean).is_some_and(|(sheet_name, r, c)| {
                tools
                    .sheet_for(&sheet_name)
                    .is_some_and(|sh| r < sh.cells.len() && c < sh.max_col())
            });
            if sheet_ok && !citations.contains(&clean) {
                citations.push(clean.clone());
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
            // With an explicit target the what-if runs even when
            // retrieval missed; without one it needs the top region
            // to find a dependent formula.
            let explicit_target = parse_what_if_target(question, &cell);
            if top_range.is_none() && explicit_target.is_none() {
                context.push_str(
                    "\n\nwhat-if: skipped — retrieval found no region and the question names no target cell.",
                );
            }
            if let Some(range) = top_range.as_ref().or(explicit_target.as_ref()) {
                if let Some((sheet, _, _, _, _)) = crate::tools::parse_range_uncapped(range) {
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
                        let (osheet, orow, ocol) = override_coord.clone()?;
                        let (_, r0, c0, r1, c1) = crate::tools::parse_range_uncapped(range)?;
                        let sheet_obj = tools.sheet_for(&sheet)?;
                        for r in r0..=r1 {
                            for c in c0..=c1 {
                                if let CellValue::Formula { expr, .. } = sheet_obj.get(r, c) {
                                    let refs = gridtalk_graph::formula_refs_qualified(&expr);
                                    // A qualified ref names the
                                    // OVERRIDE's sheet; an unqualified
                                    // one means the formula's own sheet.
                                    let depends = refs.iter().any(|q| {
                                        let ref_sheet = q.sheet.as_deref().unwrap_or(&sheet);
                                        ref_sheet.eq_ignore_ascii_case(&osheet)
                                            && q.row == orow
                                            && q.col == ocol
                                    });
                                    if depends {
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
                    match parse_what_if_target(question, &cell).or_else(dependent_target) {
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
                } else {
                    context.push_str(
                        "\n\nwhat-if: skipped — the top region could not be parsed for target discovery.",
                    );
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
    // Bound the prompt: adapter/skill/sidecar text is user input.
    const MAX_SYSTEM_CHARS: usize = 12_000;
    if system.chars().count() > MAX_SYSTEM_CHARS {
        system = system.chars().take(MAX_SYSTEM_CHARS).collect();
        system.push_str("\n[adapter content truncated]");
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
        // Never substitute raw tool context as if it were an
        // answer; say plainly that the backend produced nothing.
        answer = "(the backend returned an empty answer; see the trail for the tool results)"
            .to_string();
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

/// All sheet-qualified cell refs in a question, in order,
/// honouring quoted sheet names ('My Sheet'!B2) — whitespace
/// tokenising alone would shred those.
pub fn qualified_refs(question: &str) -> Vec<String> {
    let chars: Vec<char> = question.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        if chars[i] == '\'' {
            // Quoted sheet name up to the closing quote (with ''
            // escapes), then '!'.
            let mut j = i + 1;
            let mut closed = false;
            while j < chars.len() {
                if chars[j] == '\'' {
                    if chars.get(j + 1) == Some(&'\'') {
                        j += 2;
                        continue;
                    }
                    closed = true;
                    break;
                }
                j += 1;
            }
            if closed && chars.get(j + 1) == Some(&'!') {
                let mut k = j + 2;
                while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '$') {
                    k += 1;
                }
                let token: String = chars[start..k].iter().collect();
                if parse_cell_ref(&token).is_some() {
                    out.push(token);
                }
                i = k;
                continue;
            }
            i += 1;
        } else if chars[i].is_alphabetic() || chars[i] == '_' {
            let mut j = i;
            while j < chars.len()
                && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '.')
            {
                j += 1;
            }
            if chars.get(j) == Some(&'!') {
                let mut k = j + 1;
                while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '$') {
                    k += 1;
                }
                let token: String = chars[start..k].iter().collect();
                if parse_cell_ref(&token).is_some() {
                    out.push(token);
                }
                i = k;
                continue;
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    out
}

fn parse_what_if(question: &str) -> Option<(String, f64)> {
    // Quoted-sheet form first via the quote-aware parser:
    // split_whitespace would shred 'My Sheet'!B2 into two tokens.
    for qref in qualified_refs(question) {
        if parse_cell_ref(&qref).is_some() {
            if let Some(pos) = question.find(&qref) {
                // Accept a number only as the immediate value or
                // after an introducer (is/becomes/=/to/set): scanning
                // blindly invents overrides from unrelated numbers
                // later in the question.
                let tail: Vec<&str> = question[pos + qref.len()..].split_whitespace().collect();
                for (j, next) in tail.iter().take(12).enumerate() {
                    let n =
                        next.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'));
                    if let Ok(v) = n.parse::<f64>() {
                        let introduced = j == 0
                            || tail[..j].iter().rev().take(3).any(|t| {
                                matches!(
                                    t.trim_matches(|c: char| !c.is_alphanumeric()),
                                    "is" | "becomes" | "equals" | "to" | "set" | "at"
                                ) || t.contains('=')
                            });
                        if introduced {
                            return Some((qref, v));
                        }
                    }
                }
            }
        }
    }
    // "... B2 is 10 ..." / "... B2 = 10.5 ..." / "... Sheet1!B2 ..."
    let tokens: Vec<&str> = question.split_whitespace().collect();
    // Sheet-qualified form first (Sheet1!B2 / 'My Sheet'!B2): the
    // token carries '!' and parses as a full cell reference.
    for (i, tok) in tokens.iter().enumerate() {
        let clean =
            tok.trim_matches(|c: char| !(c.is_alphanumeric() || c == '!' || c == '\'' || c == '$'));
        if clean.contains('!') && parse_cell_ref(clean).is_some() {
            for (j, next) in tokens.iter().skip(i + 1).take(4).enumerate() {
                let n = next.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'));
                if let Ok(v) = n.parse::<f64>() {
                    // Same introducer guard as the quoted path:
                    // immediate value, or after is/becomes/=/to/set.
                    let introduced = j == 0
                        || tokens[..i + 1 + j].iter().rev().take(3).any(|t| {
                            matches!(
                                t.trim_matches(|c: char| !c.is_alphanumeric()),
                                "is" | "becomes" | "equals" | "to" | "set" | "at"
                            ) || t.contains('=')
                        });
                    if introduced {
                        return Some((clean.to_string(), v));
                    }
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

fn parse_what_if_target(question: &str, override_cell: &str) -> Option<String> {
    // "what happens to Sheet1!D4": the first qualified ref that is
    // NOT the override itself. (The bench's own phrasing names the
    // override first, so the naive first-token pick targets the
    // wrong cell.)
    let override_ref = parse_cell_ref(override_cell);
    qualified_refs(question)
        .into_iter()
        .find(|cell| parse_cell_ref(cell) != override_ref)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::parse_range;
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
        // The bench phrasing names the override first; the target
        // must be the OTHER qualified ref, never the override.
        assert_eq!(
            parse_what_if_target(
                "What if Sheet1!B2 becomes 12? What happens to Sheet1!D2?",
                "Sheet1!B2"
            ),
            Some("Sheet1!D2".to_string())
        );
        // Quoted sheet names survive ref extraction.
        assert_eq!(
            qualified_refs("what is 'My Sheet'!B2 worth?"),
            vec!["'My Sheet'!B2".to_string()]
        );
        // Sheet-qualified overrides parse too.
        assert_eq!(
            parse_what_if("What if Sheet1!B2 becomes 12? What happens to Sheet1!D2?"),
            Some(("Sheet1!B2".to_string(), 12.0))
        );
    }
}
