//! Investigate harness (Phase 3b): the model — or, with no model,
//! a deterministic planner — drives tools instead of receiving a
//! fixed pipeline's context. Every step is recorded in a trail and
//! the final answer carries its citations. Tools:
//! `search`, `expand`, `cells`, `formulas`, `what_if`.

pub use crate::tools::parse_cell_ref;
use crate::tools::Tools;
use gridtalk_graph::{CellValue, NodeKind};
use gridtalk_index::{history_prompt, standalone_query, ChatTurn};
use gridtalk_infer::{CancelToken, ChatBackend, ChatRequest};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct TrailStep {
    pub tool: String,
    pub args: String,
    pub result: String,
    /// Whether the call ran and returned a result (a search that
    /// matched nothing, an error, a call that was not run: `false`).
    pub ok: bool,
}

#[derive(Debug)]
pub struct Investigation {
    pub answer: String,
    pub trail: Vec<TrailStep>,
    pub citations: Vec<String>,
    /// Ranges the answer cites that no tool result read or named
    /// (tool loop only): the model has them from somewhere else.
    pub unread_citations: Vec<String>,
    /// Model turns taken: 1 in the fixed pipeline, one per turn of
    /// the tool loop.
    pub model_turns: usize,
    /// Whether `answer` ends in a `Sources:` line this code added
    /// because the model cited nothing that was read. A caller that
    /// shows the citations its own way can leave the line out.
    pub sources_appended: bool,
}

impl Investigation {
    /// The answer without the `Sources:` line added to it, if one
    /// was.
    pub fn answer_body(&self) -> &str {
        match self.answer.rsplit_once("\n\nSources: ") {
            Some((body, _)) if self.sources_appended => body,
            _ => &self.answer,
        }
    }
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
    adapter: Option<&gridtalk_adapter::Adapter>,
    sidecar: Option<&str>,
) -> Result<Investigation, gridtalk_infer::InferError> {
    investigate_with_context(
        question,
        tools,
        backend,
        max_steps,
        adapter,
        sidecar,
        &[],
        None,
    )
}

/// [`investigate_with`] with the conversation so far and the
/// caller's current selection. A follow-up is condensed against
/// `history` into a stand-alone query before searching (see
/// [`standalone_query`]) and the turns ride along in the prompt;
/// a `selection` range is read (`expand`, `get_cells`) and its
/// results lead the prompt, ahead of the search results.
#[allow(clippy::too_many_arguments)]
pub fn investigate_with_context(
    question: &str,
    tools: &Tools,
    backend: &dyn ChatBackend,
    max_steps: usize,
    adapter: Option<&gridtalk_adapter::Adapter>,
    sidecar: Option<&str>,
    history: &[ChatTurn],
    selection: Option<&str>,
) -> Result<Investigation, gridtalk_infer::InferError> {
    investigate_run(
        &Ask {
            question,
            max_steps,
            adapter,
            sidecar,
            history,
            selection,
        },
        tools,
        backend,
        &mut Run {
            cancel: Arc::new(AtomicBool::new(false)),
            on_token: &mut |_| {},
            on_step: &mut |_| {},
            trace: None,
            tool_loop: true,
        },
    )
}

/// What is asked, and what it is asked against.
#[derive(Clone, Copy)]
pub struct Ask<'a> {
    pub question: &'a str,
    /// The tool-call budget. In the fixed pipeline it bounds every
    /// call. In the tool loop it bounds the calls the model chooses;
    /// the planner's opening reads (at most
    /// [`crate::agent::OPENING_STEPS`]) come on top.
    pub max_steps: usize,
    pub adapter: Option<&'a gridtalk_adapter::Adapter>,
    /// The per-workbook sidecar's rules, appended last.
    pub sidecar: Option<&'a str>,
    /// Earlier turns of the conversation, oldest first.
    pub history: &'a [ChatTurn],
    /// The range the caller has selected.
    pub selection: Option<&'a str>,
}

/// How a run reports back while it is under way.
pub struct Run<'a> {
    /// Set to stop the run; the backend polls it.
    pub cancel: CancelToken,
    /// The answer's text as it is produced.
    pub on_token: &'a mut dyn FnMut(&str),
    /// Each tool call once it has run.
    pub on_step: &'a mut dyn FnMut(&TrailStep),
    /// Every model request and reply and every tool call of the tool
    /// loop, one JSON object each (see [`crate::agent`]).
    pub trace: Option<&'a mut dyn FnMut(&serde_json::Value)>,
    /// `false` keeps the fixed pipeline whatever the backend can do.
    pub tool_loop: bool,
}

/// Answer `ask.question`. With a backend that takes tool turns (and
/// no `tool_loop = false` from the caller or the adapter) the model
/// chooses its own tool calls after the planner's opening reads
/// ([`crate::agent::tool_loop`]). Otherwise the planner makes every
/// call and the backend words the answer from what it read.
pub fn investigate_run(
    ask: &Ask,
    tools: &Tools,
    backend: &dyn ChatBackend,
    run: &mut Run,
) -> Result<Investigation, gridtalk_infer::InferError> {
    if ask.max_steps == 0 {
        // The step budget gates every tool call, including search.
        return Ok(Investigation {
            answer: String::new(),
            trail: Vec::new(),
            citations: Vec::new(),
            unread_citations: Vec::new(),
            model_turns: 0,
            sources_appended: false,
        });
    }
    if run.tool_loop && ask.adapter.is_none_or(|a| a.tool_loop) && backend.supports_tools() {
        return crate::agent::tool_loop(ask, tools, backend, run);
    }
    let gathered = gather(ask, tools, ask.max_steps);
    for step in &gathered.trail {
        (run.on_step)(step);
    }
    let Gathered {
        trail,
        mut context,
        citations,
        standalone,
        skipped,
    } = gathered;
    let (question, adapter, sidecar, history, max_steps) = (
        ask.question,
        ask.adapter,
        ask.sidecar,
        ask.history,
        ask.max_steps,
    );
    if !skipped.is_empty() {
        context.push_str(&format!(
            "\n\nnote: the step budget ({max_steps}) was spent before {} could be read.",
            skipped.join(" and ")
        ));
    }

    let mut system = "You answer questions about a spreadsheet using only the tool results provided. Always cite the range like Sheet1!A1:D4.".to_string();
    system.push_str(&gridtalk_adapter::prompt_text(
        adapter,
        &standalone,
        sidecar,
    ));
    let system = gridtalk_adapter::cap_prompt(system);
    let mut user = history_prompt(history);
    user.push_str(&format!("Question: {question}\n"));
    if standalone != question {
        user.push_str(&format!("(searched as: {standalone})\n"));
    }
    user.push_str(&format!("\nTool results:\n{context}"));
    let req = ChatRequest {
        system,
        user,
        max_tokens: 512,
    };
    let mut answer = String::new();
    // Backend failures are propagated, never silently replaced by
    // raw context masquerading as an answer.
    backend.stream(&req, &run.cancel, &mut |t| {
        answer.push_str(t);
        (run.on_token)(t);
    })?;
    if answer.trim().is_empty() {
        // Never substitute raw tool context as if it were an
        // answer; say plainly that the backend produced nothing.
        answer = "(the backend returned an empty answer; see the trail for the tool results)"
            .to_string();
    }
    // Citation guarantee: the printed answer must cite a range we
    // actually retrieved.
    let sources_appended =
        !citations.is_empty() && !citations.iter().any(|c| contains_bounded(&answer, c));
    if sources_appended {
        answer.push_str(&format!("\n\nSources: {}", citations.join(", ")));
    }

    Ok(Investigation {
        answer,
        trail,
        citations,
        unread_citations: Vec::new(),
        model_turns: 1,
        sources_appended,
    })
}

/// What the planner read for a question.
pub struct Gathered {
    pub trail: Vec<TrailStep>,
    /// The results as prompt text, the selection's first.
    pub context: String,
    /// The ranges the reads stand on.
    pub citations: Vec<String>,
    /// The question as it was searched (a follow-up made to stand
    /// alone).
    pub standalone: String,
    /// Planned reads the step budget dropped.
    pub skipped: Vec<&'static str>,
}

/// The deterministic planner: search, then read what the question
/// names (the selection, an asked-for what-if, cells and columns
/// named as references or in words, the workbook's sheet list) and
/// the top region, in that order, in at most `max_steps` tool calls.
pub fn gather(ask: &Ask, tools: &Tools, max_steps: usize) -> Gathered {
    let (question, adapter, history, selection) =
        (ask.question, ask.adapter, ask.history, ask.selection);
    let mut trail: Vec<TrailStep> = Vec::new();
    let run = |tool: &str, args: &str, trail: &mut Vec<TrailStep>| -> String {
        let (result, ok) = tools.execute_step(tool, args);
        trail.push(TrailStep {
            tool: tool.to_string(),
            args: args.to_string(),
            result: result.clone(),
            ok,
        });
        result
    };

    // A follow-up is searched in its stand-alone form: the
    // history's questions supply the words its pronouns stand for.
    let standalone = standalone_query(question, history);
    // Citations come from the structured hits, never scraped from
    // display text. The search step reuses this one search (it is
    // recorded in the trail, not re-executed).
    let top_k = adapter.map(|a| a.top_k).unwrap_or(3);
    let hits = tools.hybrid.search(&standalone, top_k);
    let mut citations: Vec<String> = hits.iter().map(|h| h.range.a1()).collect();
    let top_range = citations.first().cloned();
    let search_result = if hits.is_empty() {
        format!("no regions matched \"{standalone}\"")
    } else {
        hits.iter()
            .map(|h| format!("[{}] score {:.3}", h.range.a1(), h.score))
            .collect::<Vec<_>>()
            .join("\n")
    };
    trail.push(TrailStep {
        tool: "search".into(),
        args: format!("q={standalone}"),
        result: search_result.clone(),
        ok: !hits.is_empty(),
    });

    let mut context = format!("search results:\n{search_result}");
    // Ranges already expanded or read, so no step repeats one.
    let mut expanded_ranges: Vec<String> = Vec::new();
    let mut cells_read: Vec<String> = Vec::new();
    // Planned reads the step budget dropped, named in the prompt
    // at the end instead of silently absent.
    let mut skipped: Vec<&str> = Vec::new();
    // A citation must name a sheet the workbook actually has.
    let range_sheet_exists = |range: &str| -> bool {
        crate::tools::parse_range_uncapped(range)
            .is_some_and(|(sheet, ..)| tools.sheet_for(&sheet).is_some())
    };
    // The selection is read next but leads the prompt (assembled
    // at the end): "what feeds this?" stands on it, not on the
    // search that its three words happen to match.
    let mut selection_section = String::new();
    if let Some(sel) = selection.filter(|s| !s.trim().is_empty()) {
        let canonical = crate::tools::parse_range_uncapped(sel).map(|(sheet, r0, c0, r1, c1)| {
            let sheet = tools
                .sheet_for(&sheet)
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
        let sel = canonical.as_deref().unwrap_or(sel);
        if trail.len() < max_steps {
            let expanded = run("expand", &format!("range={sel}"), &mut trail);
            selection_section.push_str(&format!("selected {sel} (graph):\n{expanded}"));
            expanded_ranges.push(sel.to_string());
        }
        if trail.len() < max_steps {
            // A tall selection (a whole column) is read from its top
            // rows, like a range the question names.
            let (read, clipped) = clip_range(sel);
            let note = if clipped {
                format!(", first rows {read}")
            } else {
                String::new()
            };
            let cells = run("get_cells", &format!("range={read}"), &mut trail);
            if !selection_section.is_empty() {
                selection_section.push_str("\n\n");
            }
            selection_section.push_str(&format!("selected {sel} (cells{note}):\n{cells}"));
            cells_read.push(read);
        }
        // Cite the selection only when its sheet exists and a
        // step actually read it.
        if canonical.is_some()
            && range_sheet_exists(sel)
            && !selection_section.is_empty()
            && !citations.iter().any(|c| c == sel)
        {
            citations.push(sel.to_string());
        }
    }
    // What-if: "what if B2 is 10" / "what if B2 = 10". It runs
    // before the named-reference reads below: the question asked
    // for it explicitly, so the step budget must not starve it.
    let lower = question.to_lowercase();
    if lower.contains("what if") && trail.len() >= max_steps {
        context.push_str("\n\nwhat-if: skipped — the step budget was already spent.");
    }
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
                    // Dependent = a formula whose value actually
                    // CHANGES under the override (trial what-if):
                    // captures transitive dependencies and needs no
                    // range expansion. Bounded candidate count.
                    let dependent_target = || -> Option<String> {
                        let (osheet, orow, ocol) = override_coord.clone()?;
                        let (_, r0, c0, r1, c1) = crate::tools::parse_range_uncapped(range)?;
                        let sheet_obj = tools.sheet_for(&sheet)?;
                        // One copy with the override applied answers
                        // every candidate.
                        let base = tools.calc.get(tools.wb);
                        let changed = base.with_override((&osheet, orow, ocol), value).ok()?;
                        let mut tried = 0usize;
                        for r in r0..=r1 {
                            for c in c0..=c1 {
                                if tried >= 200 {
                                    return None;
                                }
                                if (sheet.as_str(), r, c) == (osheet.as_str(), orow, ocol) {
                                    continue;
                                }
                                if let CellValue::Formula { .. } = sheet_obj.get(r, c) {
                                    tried += 1;
                                    // Any typed change counts, so a text
                                    // or boolean dependent is found too.
                                    if let (Ok(before), Ok(after)) =
                                        (base.value(&sheet, r, c), changed.value(&sheet, r, c))
                                    {
                                        if before.differs_from(&after) {
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

    // A question naming a cell gets that cell's formulas/cells
    // directly, even when retrieval ranked something else first
    // (a title cell, say). This is the default; an adapter that
    // learned otherwise (`ref_context = false`) still turns it off.
    let named_refs: Vec<String> = if adapter.is_none_or(|a| a.ref_context) {
        // The current question's refs first; refs carried in from
        // the history by condensation only fill what is left, so
        // they can never truncate away the cell just asked about.
        let mut refs: Vec<String> = Vec::new();
        let in_words = |q: &str| -> Vec<String> {
            refs_in_words(q, tools)
                .into_iter()
                .filter(|w| !w.column)
                .map(|w| w.range)
                .collect()
        };
        for r in qualified_ranges(question)
            .into_iter()
            .chain(in_words(question))
            .chain(qualified_ranges(&standalone))
            .chain(in_words(&standalone))
        {
            if !refs.iter().any(|done| same_range(done, &r)) {
                refs.push(r);
            }
        }
        refs.truncate(3);
        refs
    } else {
        Vec::new()
    };
    for clean in &named_refs {
        if trail.len() >= max_steps {
            skipped.push("some named references");
            break;
        }
        // A large range is read from its top rows: a whole column's
        // cells would crowd out the rest of the prompt.
        let (read, clipped) = clip_range(clean);
        let note = if clipped {
            format!(" (first rows {read} of {clean})")
        } else {
            String::new()
        };
        let formulas = run("list_formulas", &format!("range={read}"), &mut trail);
        context.push_str(&format!("\n\ndirect reference {clean}{note}:\n{formulas}"));
        if trail.len() < max_steps && !cells_read.iter().any(|done| same_range(done, clean)) {
            let cells = run("get_cells", &format!("range={read}"), &mut trail);
            context.push_str(&format!(
                "\n\ndirect reference cells {clean}{note}:\n{cells}"
            ));
            cells_read.push(clean.clone());
        }
        // Directly-read named cells are citations too — but only
        // when the named sheet actually exists (the tool calls
        // above return error text for a bad sheet).
        let sheet_ok =
            crate::tools::parse_range_uncapped(clean).is_some_and(|(sheet_name, r, c, ..)| {
                tools
                    .sheet_for(&sheet_name)
                    .is_some_and(|sh| r < sh.cells.len() && c < sh.max_col())
            });
        if sheet_ok && !citations.iter().any(|c| same_range(c, clean)) {
            citations.push(clean.clone());
        }
    }

    // Expand the column the question is about, not only the
    // region search ranked first: a named cell's column, and any
    // column whose heading the question uses. `expand` on a
    // column answers "what does this read / what reads this"
    // outright, with headings and ranges.
    let mut expand_targets: Vec<String> = Vec::new();
    let columns_in_words = |q: &str| -> Vec<String> {
        refs_in_words(q, tools)
            .into_iter()
            .filter(|w| w.column)
            .map(|w| w.range)
            .collect()
    };
    for r in named_refs
        .iter()
        .chain(columns_in_words(question).iter())
        .chain(columns_in_words(&standalone).iter())
        .chain(columns_asked_about(question, tools.graph).iter())
        .chain(columns_asked_about(&standalone, tools.graph).iter())
    {
        // A named cell and its own column expand to the same
        // node; one of the two steps is enough.
        let covered = expand_targets
            .iter()
            .chain(expanded_ranges.iter())
            .any(|done| ranges_nested(done, r));
        if !covered {
            expand_targets.push(r.clone());
        }
    }
    for range in expand_targets.into_iter().take(3) {
        if trail.len() >= max_steps {
            skipped.push("some column expansions");
            break;
        }
        let expanded = run("expand", &format!("range={range}"), &mut trail);
        context.push_str(&format!("\n\ncolumn expansion {range}:\n{expanded}"));
        expanded_ranges.push(range.clone());
        if range_sheet_exists(&range) && !citations.iter().any(|c| same_range(c, &range)) {
            citations.push(range);
        }
    }
    // Workbook-scoped questions ("what is this workbook?", "are
    // there hidden sheets?") stand on the sheet/region list, which
    // no single region's cells carry. The current question decides,
    // not words an earlier turn happened to use; the list is capped
    // in the prompt like a search passage.
    if asks_about_the_workbook(question) {
        if trail.len() < max_steps {
            let info = run("workbook_info", "", &mut trail);
            let info = truncate_chars(&info, 4_000);
            context.push_str(&format!("\n\nworkbook:\n{info}"));
        } else {
            skipped.push("the workbook sheet list");
        }
    }
    // "Are there any errors?", "is anything broken or hard-coded?":
    // the issues check reads every formula, which no search hit's
    // cells cover. Capped in the prompt like the sheet list.
    if asks_about_issues(question) {
        if trail.len() < max_steps {
            let found = run("issues", "", &mut trail);
            let found = truncate_chars(&found, 4_000);
            context.push_str(&format!("\n\nworkbook issues:\n{found}"));
        } else {
            skipped.push("the workbook issues check");
        }
    }
    if let Some(range) = &top_range {
        let mut top_done = true;
        if !expanded_ranges.iter().any(|done| same_range(done, range)) {
            if trail.len() < max_steps {
                let expanded = run("expand", &format!("range={range}"), &mut trail);
                context.push_str(&format!("\n\ngraph expansion:\n{expanded}"));
                expanded_ranges.push(range.clone());
            } else {
                top_done = false;
            }
        }
        if !cells_read.iter().any(|done| same_range(done, range)) {
            if trail.len() < max_steps {
                let cells = run("get_cells", &format!("range={range}"), &mut trail);
                context.push_str(&format!("\n\ncells:\n{cells}"));
                cells_read.push(range.clone());
            } else {
                top_done = false;
            }
        }
        if trail.len() < max_steps {
            let formulas = run("list_formulas", &format!("range={range}"), &mut trail);
            context.push_str(&format!("\n\nformulas:\n{formulas}"));
        } else {
            top_done = false;
        }
        if !top_done {
            skipped.push("some reads of the top region");
        }
    }
    if !selection_section.is_empty() {
        context = format!("{selection_section}\n\n{context}");
    }
    Gathered {
        trail,
        context,
        citations,
        standalone,
        skipped,
    }
}

/// Whether either range contains the other on the same sheet, so
/// expanding both would reach the same graph node twice.
fn ranges_nested(a: &str, b: &str) -> bool {
    let (Some((sa, ar0, ac0, ar1, ac1)), Some((sb, br0, bc0, br1, bc1))) = (
        crate::tools::parse_range_uncapped(a),
        crate::tools::parse_range_uncapped(b),
    ) else {
        return a == b;
    };
    if !gridtalk_graph::sheet_name_eq(&sa, &sb) {
        return false;
    }
    let a_holds_b = ar0 <= br0 && ac0 <= bc0 && ar1 >= br1 && ac1 >= bc1;
    let b_holds_a = br0 <= ar0 && bc0 <= ac0 && br1 >= ar1 && bc1 >= ac1;
    if !(a_holds_b || b_holds_a) {
        return false;
    }
    // Containment implies the same node only when the ranges are
    // equal, or the inner range is a single cell (its smallest
    // covering node is its column) and the outer one spans that
    // one column: a region contains its columns, but the Region
    // node and a Column node carry different edges.
    let single = |r0: usize, c0: usize, r1: usize, c1: usize| r0 == r1 && c0 == c1;
    (ar0, ac0, ar1, ac1) == (br0, bc0, br1, bc1)
        || (single(br0, bc0, br1, bc1) && ac0 == ac1)
        || (single(ar0, ac0, ar1, ac1) && bc0 == bc1)
}

/// Whether two ranges name the same cells, however spelled
/// (`Sheet1!D2` and the canonical `Sheet1!D2:D2`, `sheet1!$D$2`).
pub(crate) fn same_range(a: &str, b: &str) -> bool {
    match (
        crate::tools::parse_range_uncapped(a),
        crate::tools::parse_range_uncapped(b),
    ) {
        (Some((sa, ar0, ac0, ar1, ac1)), Some((sb, br0, bc0, br1, bc1))) => {
            gridtalk_graph::sheet_name_eq(&sa, &sb) && (ar0, ac0, ar1, ac1) == (br0, bc0, br1, bc1)
        }
        _ => a == b,
    }
}

/// Whether `needle` occurs in `haystack` not flanked by word
/// characters (alphanumeric or `_`), so "yes" is not found in
/// "yesterday" nor "6" in "36".
pub fn contains_word(haystack: &str, needle: &str) -> bool {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(pos) = haystack[from..].find(needle) {
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = haystack[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !word(c));
        let after_ok = haystack[end..].chars().next().is_none_or(|c| !word(c));
        if before_ok && after_ok {
            return true;
        }
        from = start + needle.chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Column ranges whose heading the question uses ("what reads
/// its **Hrs** column?"), most specific first: a column on a sheet
/// the question also names beats a same-named column elsewhere,
/// and a longer heading beats a shorter one it contains. At most
/// two, so a question naming common words cannot spend the whole
/// step budget on expansions.
fn columns_asked_about(question: &str, graph: &gridtalk_graph::Graph) -> Vec<String> {
    let lower = question.to_lowercase();
    let mut found: Vec<(bool, usize, String)> = Vec::new();
    for node in &graph.nodes {
        if node.kind != NodeKind::Column {
            continue;
        }
        let label = node.label.trim().to_lowercase();
        // One-letter headings and plain function words ("It",
        // "To") match half the dictionary.
        const STOPWORDS: [&str; 18] = [
            "it", "to", "of", "in", "on", "or", "and", "for", "the", "a", "an", "is", "are", "was",
            "has", "had", "per", "via",
        ];
        if label.chars().count() < 2
            || STOPWORDS.contains(&label.as_str())
            || !contains_word(&lower, &label)
        {
            continue;
        }
        let Some(range) = &node.range else { continue };
        let sheet_named = contains_word(&lower, &range.sheet.to_lowercase());
        found.push((sheet_named, label.len(), range.a1()));
    }
    found.sort_by(|a, b| b.cmp(a));
    let mut out: Vec<String> = Vec::new();
    for (_, _, a1) in found {
        if !out.contains(&a1) {
            out.push(a1);
        }
        if out.len() == 2 {
            break;
        }
    }
    out
}

/// The most cells of a named range the planner reads.
const MAX_NAMED_CELLS: usize = 200;

/// `range`, or its top rows when it holds more than
/// [`MAX_NAMED_CELLS`] cells (at least one row), and whether it was
/// clipped.
fn clip_range(range: &str) -> (String, bool) {
    let Some((sheet, r0, c0, r1, c1)) = crate::tools::parse_range_uncapped(range) else {
        return (range.to_string(), false);
    };
    let width = c1 - c0 + 1;
    let rows = (MAX_NAMED_CELLS / width).max(1);
    if r1 - r0 < rows {
        return (range.to_string(), false);
    }
    let clipped = gridtalk_graph::RangeRef {
        sheet,
        r0,
        c0,
        r1: r0 + rows - 1,
        c1,
    };
    (clipped.a1(), true)
}

/// A reference a question names in words.
#[derive(Debug, Clone, PartialEq)]
struct WordRef {
    /// Sheet-qualified, e.g. `Labour!J2:J98`.
    range: String,
    /// Named as a column ("column J on the Labour sheet"): its column
    /// node's range, to expand. Otherwise cells to read.
    column: bool,
}

/// References named in words rather than as `Sheet!A1` (#41):
/// `cells J2:J98 on the Labour sheet`, `I2:I247 in the 'Marks
/// Capture' tab`, `column J on the Labour sheet`, `col J in Labour`.
/// The sheet may be bare or quoted, in any case, and mistyped by one
/// letter; it must be one the workbook has. A column letter resolves
/// to that column's node (the tallest, when regions stack), else the
/// sheet's used rows of it.
fn refs_in_words(question: &str, tools: &Tools) -> Vec<WordRef> {
    let sheets: Vec<&str> = tools.wb.sheets.iter().map(|s| s.name.as_str()).collect();
    let words: Vec<&str> = question.split_whitespace().collect();
    let bare = |w: &str| -> String {
        w.trim_matches(|c: char| !(c.is_alphanumeric() || c == '$'))
            .to_string()
    };
    let mut out: Vec<WordRef> = Vec::new();
    for i in 0..words.len() {
        let w = bare(words[i]).to_lowercase();
        // (the reference's A1 text, whether a column, where the
        // preposition should be)
        let (a1, column, next) = if (w == "column" || w == "col") && i + 1 < words.len() {
            let letters = bare(words[i + 1]);
            let ok = (1..=3).contains(&letters.len())
                && letters.chars().all(|c| c.is_ascii_alphabetic());
            if !ok {
                continue;
            }
            (letters, true, i + 2)
        } else if !words[i].contains('!') && looks_like_a1(&bare(words[i])) {
            (bare(words[i]), false, i + 1)
        } else {
            continue;
        };
        let mut j = next;
        let Some(prep) = words.get(j).map(|p| bare(p).to_lowercase()) else {
            continue;
        };
        if !matches!(prep.as_str(), "on" | "in" | "of" | "from") {
            continue;
        }
        j += 1;
        if words
            .get(j)
            .is_some_and(|t| bare(t).eq_ignore_ascii_case("the"))
        {
            j += 1;
        }
        let rest = words[j.min(words.len())..].join(" ");
        let Some(sheet) = sheet_named_at(&rest, &sheets) else {
            continue;
        };
        let Some(snap) = tools.sheet_for(sheet) else {
            continue;
        };
        let quoted = gridtalk_graph::quote_sheet(&snap.name);
        let range = if column {
            let Some((_, _, col, _, _)) =
                crate::tools::parse_range_uncapped(&format!("{quoted}!{a1}1"))
            else {
                continue;
            };
            if col >= snap.max_col() || snap.cells.is_empty() {
                continue;
            }
            let node = tools
                .graph
                .nodes
                .iter()
                .filter(|n| n.kind == NodeKind::Column)
                .filter_map(|n| n.range.as_ref())
                .filter(|r| {
                    gridtalk_graph::sheet_name_eq(&r.sheet, &snap.name)
                        && r.c0 == col
                        && r.c1 == col
                })
                .max_by_key(|r| r.r1 - r.r0);
            match node {
                Some(r) => r.a1(),
                None => gridtalk_graph::RangeRef {
                    sheet: snap.name.clone(),
                    r0: 0,
                    c0: col,
                    r1: snap.cells.len() - 1,
                    c1: col,
                }
                .a1(),
            }
        } else {
            let range = format!("{quoted}!{}", a1.to_uppercase());
            if crate::tools::parse_range_uncapped(&range).is_none() {
                continue;
            }
            range
        };
        let found = WordRef { range, column };
        if !out.contains(&found) {
            out.push(found);
        }
    }
    out
}

/// Whether a word is an A1 cell or range (`J2`, `$J$2`, `J2:J98`).
fn looks_like_a1(word: &str) -> bool {
    let cell = |t: &str| {
        let t = t.replace('$', "");
        let letters = t.chars().take_while(|c| c.is_ascii_alphabetic()).count();
        (1..=3).contains(&letters)
            && t.len() > letters
            && t[letters..].chars().all(|c| c.is_ascii_digit())
    };
    match word.split_once(':') {
        Some((a, b)) => cell(a) && cell(b),
        None => cell(word),
    }
}

/// The sheet `text` starts with: a sheet name, bare or quoted, in any
/// case, followed by the end or a non-word character; the longest such
/// name wins. Failing that, a sheet of at least four letters whose name
/// is one edit from the same number of words (a typo).
fn sheet_named_at<'a>(text: &str, sheets: &[&'a str]) -> Option<&'a str> {
    let text = text.trim_start_matches([
        '\'', '"', '`', '\u{2018}', '\u{2019}', '\u{201c}', '\u{201d}',
    ]);
    let lower = text.to_lowercase();
    let exact = sheets
        .iter()
        .filter(|name| {
            let name = name.to_lowercase();
            lower.starts_with(&name)
                && lower[name.len()..]
                    .chars()
                    .next()
                    .is_none_or(|c| !(c.is_alphanumeric() || c == '_'))
        })
        .max_by_key(|name| name.len());
    if let Some(name) = exact {
        return Some(name);
    }
    let words: Vec<&str> = lower.split_whitespace().collect();
    let mut best: Option<(usize, &'a str)> = None;
    for name in sheets {
        let name_lower = name.to_lowercase();
        let n = name_lower.split_whitespace().count();
        if name_lower.chars().count() < 4 || n == 0 || words.len() < n {
            continue;
        }
        let candidate = words[..n].join(" ");
        let candidate = candidate
            .strip_suffix("'s")
            .unwrap_or(&candidate)
            .trim_end_matches(|c: char| !(c.is_alphanumeric() || c == '_'));
        let d = crate::tools::edit_distance(candidate, &name_lower);
        if d <= 1 && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, name));
        }
    }
    best.map(|(_, name)| name)
}

/// Whether the question is about the workbook as a whole — its
/// sheets, their count, or hidden tabs — rather than one region.
/// Word-based, so "a hidden cost" or "the table" does not fire it,
/// while "the third tab" and "how many worksheets" do.
fn asks_about_the_workbook(question: &str) -> bool {
    let words: Vec<String> = question
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect();
    let has = |w: &str| words.iter().any(|x| x == w);
    let sheetish = ["sheet", "sheets", "worksheet", "worksheets", "tab", "tabs"]
        .iter()
        .any(|w| has(w));
    has("workbook")
        || (has("hidden") && (sheetish || has("workbook")))
        || (sheetish
            && (has("many")
                || has("which")
                || has("what")
                || has("list")
                || has("tab")
                || has("tabs")))
        || (has("this") && has("file"))
}

/// Whether a question asks what is wrong with the workbook's
/// formulas: errors (by name or code), broken or stale results,
/// hard-coded numbers, or an audit.
fn asks_about_issues(question: &str) -> bool {
    let lower = question.to_lowercase();
    if [
        "#ref",
        "#div",
        "#n/a",
        "#value",
        "#name",
        "#num",
        "hard-coded",
        "hard coded",
    ]
    .iter()
    .any(|code| lower.contains(code))
    {
        return true;
    }
    lower.split(|c: char| !c.is_alphanumeric()).any(|w| {
        matches!(
            w,
            "error"
                | "errors"
                | "broken"
                | "stale"
                | "hardcoded"
                | "audit"
                | "mistake"
                | "mistakes"
                | "issues"
                | "problems"
                | "wrong"
        )
    })
}

/// `text` cut to at most `max` chars on a char boundary.
fn truncate_chars(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((byte, _)) => &text[..byte],
        None => text,
    }
}

/// All sheet-qualified cell refs in a question, in order,
/// honouring quoted sheet names ('My Sheet'!B2) — whitespace
/// tokenising alone would shred those.
pub fn qualified_refs(question: &str) -> Vec<String> {
    scan_qualified(question, false)
}

/// [`qualified_refs`], keeping a range whole (`'Marks Capture'!I2:I247`
/// rather than its first cell), for the reads a question's named
/// references get.
pub fn qualified_ranges(question: &str) -> Vec<String> {
    scan_qualified(question, true)
}

fn scan_qualified(question: &str, ranges: bool) -> Vec<String> {
    let chars: Vec<char> = question.chars().collect();
    // The end of a ref's cell part at `k`, and past `:B9` when ranges
    // are kept.
    let cell_end = |mut k: usize| -> usize {
        let run = |mut k: usize| {
            while k < chars.len() && (chars[k].is_alphanumeric() || chars[k] == '$') {
                k += 1;
            }
            k
        };
        k = run(k);
        if ranges && chars.get(k) == Some(&':') {
            let end = run(k + 1);
            if end > k + 1 {
                k = end;
            }
        }
        k
    };
    let valid = |token: &str| {
        if token.contains(':') {
            crate::tools::parse_range_uncapped(token).is_some()
        } else {
            parse_cell_ref(token).is_some()
        }
    };
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
                let k = cell_end(j + 2);
                let token: String = chars[start..k].iter().collect();
                if valid(&token) {
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
                let k = cell_end(j + 1);
                let token: String = chars[start..k].iter().collect();
                if valid(&token) {
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

/// Shared value scan for all what-if branches: the first
/// parseable number within `window` tokens that is either
/// immediate or introduced by is/becomes/equals/to/set/at/=.
fn value_after(tokens: &[&str], window: usize) -> Option<f64> {
    fn is_cell_ref_like(word: &str) -> bool {
        let cell = word.rsplit('!').next().unwrap_or(word).replace('$', "");
        let letters = cell.chars().take_while(|c| c.is_ascii_alphabetic()).count();
        (1..=3).contains(&letters)
            && letters < cell.len()
            && cell[letters..].chars().all(|c| c.is_ascii_digit())
            && !cell[letters..].is_empty()
    }
    for (j, next) in tokens.iter().take(window).enumerate() {
        // A subsequent cell ref (the target) ends the value scan:
        // its trailing digits are not the override value.
        let word = next
            .trim_matches(|c: char| !(c.is_alphanumeric() || c == '!' || c == '\'' || c == '$'));
        if is_cell_ref_like(word) {
            return None;
        }
        // Letter-starting tokens are never values.
        if next
            .trim_matches(|c: char| !c.is_alphanumeric())
            .chars()
            .next()
            .is_some_and(|c| c.is_alphabetic())
        {
            continue;
        }
        let n = next.trim_matches(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'));
        if let Ok(v) = n.parse::<f64>() {
            let introduced = j == 0
                || tokens[..j].iter().rev().take(3).any(|t| {
                    matches!(
                        t.trim_matches(|c: char| !c.is_alphanumeric())
                            .to_ascii_lowercase()
                            .as_str(),
                        "is" | "becomes" | "equals" | "to" | "set" | "at"
                    ) || t.contains('=')
                });
            if introduced {
                return Some(v);
            }
        }
    }
    None
}

fn parse_what_if(question: &str) -> Option<(String, f64)> {
    // Quoted-sheet form first via the quote-aware parser:
    // split_whitespace would shred 'My Sheet'!B2 into two tokens.
    for qref in qualified_refs(question) {
        if parse_cell_ref(&qref).is_some() {
            if let Some(pos) = question.find(&qref) {
                let tail: Vec<&str> = question[pos + qref.len()..].split_whitespace().collect();
                if let Some(v) = value_after(&tail, 12) {
                    return Some((qref, v));
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
            if let Some(v) = value_after(&tokens[i + 1..], 12) {
                return Some((clean.to_string(), v));
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
            if let Some(v) = value_after(&tokens[i + 1..], 12) {
                return Some((clean.to_string(), v));
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

/// `answer` mentions `needle` as a standalone reference, not as a
/// substring of a longer ref (Sheet1!A1 inside Sheet1!A10:B20).
pub fn contains_bounded(answer: &str, needle: &str) -> bool {
    // ASCII case-insensitive (offsets preserved; matches the
    // case-insensitive sheet comparison in citation_hit).
    let answer = &answer.to_ascii_lowercase();
    let needle = &needle.to_ascii_lowercase();
    let boundary = |c: char| !(c.is_alphanumeric() || matches!(c, '!' | ':' | '.' | '$'));
    let first_char_len = needle.chars().next().map_or(1, |c| c.len_utf8());
    let mut from = 0;
    while let Some(pos) = answer[from..].find(needle.as_str()) {
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = answer[..start].chars().next_back().is_none_or(boundary);
        // Sentence punctuation (`... Sheet1!D2.` / `Sheet1!D2: 6`)
        // ends a reference; a '.'/':' that continues a longer ref
        // (Sheet1!A1:B2) does not.
        let mut tail = answer[end..].chars();
        let after_ok = match tail.next() {
            None => true,
            Some(c) if boundary(c) => true,
            Some('.') | Some(':') => tail.next().is_none_or(|n| n.is_whitespace()),
            Some(_) => false,
        };
        if before_ok && after_ok {
            return true;
        }
        from = start + first_char_len;
        if from >= answer.len() {
            break;
        }
    }
    false
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
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        assert!(tools.execute("search", "q=provision").contains("Sheet1!"));
        assert!(tools
            .execute("list_formulas", "range=Sheet1!A1:D3")
            .contains("B2*C2"));
        let inv = investigate("how is the provision calculated", &tools, &EchoBackend, 8).unwrap();
        assert!(!inv.trail.is_empty());
        assert_eq!(inv.trail[0].tool, "search");
        assert!(!inv.citations.is_empty());
        assert!(inv.answer.contains("Provision") || inv.answer.contains("provision"));
    }

    #[test]
    fn what_if_discovery_finds_a_text_dependent() {
        // The region's only formula depending on B2 returns text; the
        // fallback target (the region's last cell, D2) is not it.
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Size,Note\nApples,2,\"=IF(B2>5,\"\"big\"\",\"\"small\"\")\",fresh\n",
            ',',
        );
        assert!(matches!(wb.sheets[0].get(1, 2), CellValue::Formula { .. }));
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty(), "{v:?}");
        let hybrid = HybridIndex::build(&wb, &g);
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let inv = investigate(
            "what if B2 is 10 for the apples size",
            &tools,
            &EchoBackend,
            8,
        )
        .unwrap();
        let step = inv
            .trail
            .iter()
            .find(|s| s.tool == "what_if")
            .expect("a what-if step");
        assert!(step.args.contains("target=Sheet1!C2"), "{}", step.args);
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
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        assert!(investigate("provision?", &tools, &FailBackend, 8).is_err());
    }

    #[test]
    fn named_cells_are_read_by_default_and_columns_expand() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        // No adapter: the cell the question names is still read.
        let inv = investigate(
            "does the total in Sheet1!D2 agree?",
            &tools,
            &EchoBackend,
            12,
        )
        .unwrap();
        assert!(
            inv.trail
                .iter()
                .any(|s| s.tool == "get_cells" && s.args == "range=Sheet1!D2"),
            "{:?}",
            inv.trail.iter().map(|s| &s.args).collect::<Vec<_>>()
        );
        assert!(inv
            .trail
            .iter()
            .any(|s| s.tool == "list_formulas" && s.args == "range=Sheet1!D2"));
        assert!(inv.citations.contains(&"Sheet1!D2".to_string()));
        // The column a question is about gets expanded, and the
        // expansion names what it reads (Qty and Price).
        let inv =
            investigate("what does the Total column read?", &tools, &EchoBackend, 12).unwrap();
        let col = inv
            .trail
            .iter()
            .find(|s| s.tool == "expand" && s.args == "range=Sheet1!D1:D3")
            .expect("the Total column is expanded");
        assert!(col.result.contains("Qty"), "{}", col.result);
        assert!(col.result.contains("Price"), "{}", col.result);
        // A workbook-scoped question reads the sheet list.
        let inv = investigate("are there any hidden sheets?", &tools, &EchoBackend, 12).unwrap();
        assert!(inv.trail.iter().any(|s| s.tool == "workbook_info"));
        assert!(inv.answer.contains("sheet Sheet1"), "{}", inv.answer);
        // A question about what is wrong runs the issues check.
        let inv = investigate("are there any #REF errors?", &tools, &EchoBackend, 12).unwrap();
        assert!(inv.trail.iter().any(|s| s.tool == "issues"));
        let inv = investigate("what is the total for apples?", &tools, &EchoBackend, 12).unwrap();
        assert!(inv.trail.iter().all(|s| s.tool != "issues"));
    }

    /// A cover sheet whose notes mention "Marks Capture", a 260-row
    /// marks sheet and a labelled block of weights (#41, #42).
    fn marks_workbook() -> (WorkbookSnapshot, Graph, HybridIndex) {
        let mut wb = parse_delimited(
            "marks",
            "Cover",
            "Notes\nSee the Marks Capture sheet for every mark\n",
            ',',
        );
        let mut marks = String::from("Exam,Year Mark,Promotion Mark\n");
        for r in 2..=261 {
            marks.push_str(&format!(
                "{},{},\"=ROUND(B{r}*Lists!$B$1+A{r}*Lists!$B$2,0)\"\n",
                40 + r % 50,
                50 + r % 40
            ));
        }
        for (name, csv) in [
            ("Marks Capture", marks.as_str()),
            (
                "Lists",
                "Semester Mark Weight,0.3\nExamination Weight,0.7\nMinimum to Pass,50\n",
            ),
        ] {
            wb.sheets
                .extend(parse_delimited("marks", name, csv, ',').sheets);
        }
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty(), "{v:?}");
        let hybrid = HybridIndex::build(&wb, &g);
        (wb, g, hybrid)
    }

    #[test]
    fn a_column_or_cells_named_in_words_are_read() {
        let (wb, g, hybrid) = marks_workbook();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let args = |inv: &Investigation, tool: &str| -> Vec<String> {
            inv.trail
                .iter()
                .filter(|s| s.tool == tool)
                .map(|s| s.args.clone())
                .collect()
        };
        // A column letter and its sheet, in lower case and mistyped:
        // the column is expanded, and the expansion names the two
        // weights it reads.
        for q in [
            "What does column C on the Marks Capture sheet pull from?",
            "what does col c in marks capture pull from?",
            "What does column C on the Marks Captur tab pull from?",
        ] {
            let inv = investigate(q, &tools, &EchoBackend, 12).unwrap();
            let step = inv
                .trail
                .iter()
                .find(|s| s.tool == "expand" && s.args == "range='Marks Capture'!C1:C261")
                .unwrap_or_else(|| panic!("{q}: {:?}", args(&inv, "expand")));
            assert!(
                step.result.contains("Semester Mark Weight  Lists!B1"),
                "{}",
                step.result
            );
            assert!(step.result.contains("Examination Weight  Lists!B2"));
            assert!(!step.result.contains("Minimum to Pass"), "{}", step.result);
        }
        // Cells and their sheet in words, and a Sheet!range: the range
        // is read whole (clipped to its top rows), not its first cell.
        for q in [
            "What does C2:C261 on the 'Marks Capture' sheet pull from?",
            "What feeds cells C2:C261 in the Marks Capture worksheet?",
            "What feeds 'Marks Capture'!C2:C261?",
        ] {
            let inv = investigate(q, &tools, &EchoBackend, 12).unwrap();
            let formulas = args(&inv, "list_formulas");
            assert!(
                formulas.contains(&"range='Marks Capture'!C2:C201".to_string()),
                "{q}: {formulas:?}"
            );
            assert!(
                inv.citations
                    .contains(&"'Marks Capture'!C2:C261".to_string()),
                "{q}: {:?}",
                inv.citations
            );
            assert!(
                args(&inv, "expand").contains(&"range='Marks Capture'!C2:C261".to_string()),
                "{q}: {:?}",
                args(&inv, "expand")
            );
        }
        // An unknown sheet, or a column past the sheet's edge, names
        // nothing.
        for q in [
            "what does column C on the Nowhere sheet read?",
            "what does column Z on the Marks Capture sheet read?",
        ] {
            assert!(refs_in_words(q, &tools).is_empty(), "{q}");
        }
    }

    #[test]
    fn sheet_names_and_a1_words_are_recognised() {
        let sheets = ["Labour", "Lab", "Marks Capture"];
        assert_eq!(sheet_named_at("Labour sheet", &sheets), Some("Labour"));
        assert_eq!(
            sheet_named_at("'marks capture' tab", &sheets),
            Some("Marks Capture")
        );
        assert_eq!(sheet_named_at("Labor tab", &sheets), Some("Labour"));
        assert_eq!(sheet_named_at("Lab.", &sheets), Some("Lab"));
        assert_eq!(sheet_named_at("Labs", &sheets), None);
        assert_eq!(sheet_named_at("Staff sheet", &sheets), None);
        for w in ["J2", "$J$2", "J2:J98", "xfd9"] {
            assert!(looks_like_a1(w), "{w}");
        }
        for w in ["J", "2", "Rate", "J2:", "ABCD1", "J2:K"] {
            assert!(!looks_like_a1(w), "{w}");
        }
    }

    #[test]
    fn follow_ups_and_the_selection_reach_the_prompt() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let history = [ChatTurn {
            question: "What does Sheet1 hold?".into(),
            answer: "Items with a Total column [Sheet1!A1:D3].".into(),
        }];
        let inv = investigate_with_context(
            "and what reads its Total column?",
            &tools,
            &EchoBackend,
            14,
            None,
            None,
            &history,
            Some("Sheet1!B1:B3"),
        )
        .unwrap();
        // The follow-up was searched in stand-alone form.
        assert_eq!(inv.trail[0].tool, "search");
        assert!(
            inv.trail[0].args.contains("What does Sheet1 hold?"),
            "{}",
            inv.trail[0].args
        );
        // The history rides in the prompt (the echo backend echoes it).
        assert!(inv.answer.contains("Conversation so far:"));
        assert!(inv.answer.contains("Items with a Total column"));
        // The selection was read and leads the tool results.
        let selected = inv
            .answer
            .find("selected Sheet1!B1:B3")
            .expect("selection read");
        let search = inv.answer.find("search results:").expect("search shown");
        assert!(selected < search, "selection leads the prompt");
        assert!(inv.citations.contains(&"Sheet1!B1:B3".to_string()));
    }

    #[test]
    fn default_budget_still_runs_what_if_and_never_cites_missing_sheets() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        // The production budget (8), a selection, and two named
        // refs: the explicitly asked what-if must still run.
        let inv = investigate_with_context(
            "what if Sheet1!B2 is 10, what happens to Sheet1!D2?",
            &tools,
            &EchoBackend,
            8,
            None,
            None,
            &[],
            Some("Sheet1!B1:B3"),
        )
        .unwrap();
        assert!(
            inv.trail.iter().any(|s| s.tool == "what_if"),
            "{:?}",
            inv.trail.iter().map(|s| &s.tool).collect::<Vec<_>>()
        );
        // A selection on a sheet that does not exist is read as an
        // error, never cited.
        let inv = investigate_with_context(
            "what feeds this?",
            &tools,
            &EchoBackend,
            8,
            None,
            None,
            &[],
            Some("Missing!A1:B2"),
        )
        .unwrap();
        assert!(
            !inv.citations.iter().any(|c| c.contains("Missing")),
            "{:?}",
            inv.citations
        );
        // Nor is a named cell on a missing sheet.
        let inv = investigate("what is Nope!A1 worth?", &tools, &EchoBackend, 8).unwrap();
        assert!(
            !inv.citations.iter().any(|c| c.contains("Nope")),
            "{:?}",
            inv.citations
        );
        // History refs cannot crowd out the current question's ref.
        let history = [ChatTurn {
            question: "What are Sheet1!A1, Sheet1!B2 and Sheet1!C2?".into(),
            answer: "Headings and inputs.".into(),
        }];
        let inv = investigate_with_context(
            "and Sheet1!D2?",
            &tools,
            &EchoBackend,
            12,
            None,
            None,
            &history,
            None,
        )
        .unwrap();
        assert!(
            inv.citations.contains(&"Sheet1!D2".to_string()),
            "{:?}",
            inv.citations
        );
        // Workbook triggers are word-based.
        assert!(!asks_about_the_workbook(
            "is there a hidden cost in Sheet1!D2?"
        ));
        assert!(asks_about_the_workbook("what is on the third tab?"));
        assert!(asks_about_the_workbook("how many worksheets are there?"));
        assert!(asks_about_the_workbook("are there any hidden sheets?"));
    }

    #[test]
    fn nested_ranges_count_only_when_they_expand_to_one_node() {
        // A cell and its own column expand to the column node.
        assert!(ranges_nested("Sheet1!D1:D3", "Sheet1!D2"));
        assert!(ranges_nested("Sheet1!D2", "Sheet1!D1:D3"));
        // A region holds the cell, but the cell expands to its
        // column, not the region.
        assert!(!ranges_nested("Sheet1!A1:D3", "Sheet1!D2"));
        assert!(!ranges_nested("Sheet1!D2", "Sheet1!A1:D3"));
        assert!(ranges_nested("Sheet1!A1:D3", "sheet1!A1:D3"));
        // One cell however spelled.
        assert!(same_range("Sheet1!D2", "Sheet1!D2:D2"));
        assert!(same_range("sheet1!$D$2", "Sheet1!D2"));
        assert!(!same_range("Sheet1!D2", "Sheet1!D3"));
    }

    #[test]
    fn what_if_tool_and_question_parsing() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
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
