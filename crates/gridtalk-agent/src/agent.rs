//! The tool loop: the model chooses its own tool calls.
//!
//! The fixed pipeline spends its budget the same way whatever was
//! asked. Questions differ: following an input to a result needs an
//! `expand` at each step of the chain, and a question about one
//! column needs that column read and nothing else. Here the planner
//! still opens (it reads what the question names, at no model
//! turn), and then the model is offered the tools and decides what
//! to read next, turn by turn, until it answers or the budget is
//! spent.
//!
//! What the loop holds the model to:
//!
//! - its first turn is a tool call (adapter knob `first_call`), so
//!   it cannot answer from the question's own words;
//! - at most `max_steps` calls, then a turn that can only answer;
//! - each result is cut to a share of the context when it is added,
//!   and older results are shortened only once the conversation
//!   outgrows the context (a rewritten history costs the local
//!   backend its cached prefix, so it is the last resort);
//! - a call repeated with the same arguments is not run again;
//! - the answer's citations are checked against what the calls read.
//!
//! Every request, reply and tool call can be written to a trace
//! ([`Run::trace`]), one JSON object per event.

use crate::investigate::{
    contains_bounded, gather, qualified_ranges, same_range, Ask, Gathered, Investigation, Run,
    TrailStep,
};
use crate::tools::{parse_range_uncapped, ToolRequest, Tools};
use gridtalk_graph::sheet_name_eq;
use gridtalk_index::history_prompt;
use gridtalk_infer::{
    ChatBackend, InferError, Message, Role, ToolCall, ToolChoice, ToolSpec, TurnRequest,
};
use std::sync::atomic::Ordering;

/// Most tool calls the planner makes before the model's first turn.
pub const OPENING_STEPS: usize = 4;
/// Tokens a turn may generate: room for the answer, and for the
/// thinking a reasoning model behind an external endpoint does first.
pub const ANSWER_TOKENS: usize = 1_536;
/// The context assumed when the backend does not report one.
pub const DEFAULT_CONTEXT_TOKENS: usize = 16_384;
/// Most characters of one tool result the model is sent.
pub const MAX_RESULT_CHARS: usize = 6_000;
/// What an older result is shortened to when room runs out.
const SHORT_RESULT_CHARS: usize = 400;
/// Most ranges a `Sources:` line names.
const MAX_SOURCES: usize = 6;

const LOOP_PROMPT: &str = "You answer questions about a spreadsheet workbook. You can see the workbook only through tool results.

How to work:
1. Find what the question is about. `search` finds tables by the words in them. `workbook_info` lists the sheets, defined names and tables, and is where a question about the workbook or a whole sheet starts.
2. See how it connects. A column reads the columns its formulas refer to, and feeds the columns whose formulas refer to it. `expand` on a column or table range lists what it reads (-depends_on->) and what reads it (<-depends_on-), each with its range. To follow an input to a result, expand each step of the chain.
3. Read the cells. `list_formulas` gives a column's formula, how far it is filled down and any cell that breaks the pattern. `get_cells` gives values with their addresses.
4. Answer as soon as the results show the answer. Do not repeat a call or read something a result already shows.

Rules:
- Use only what the tool results show. If they do not show the answer, say what is missing.
- Cite each cell or range the answer rests on as Sheet!A1 or Sheet!A1:D4, copied from a tool result.
- Put the answer first and keep it to a few sentences.";

/// The tools the model is offered: every tool whose text result is
/// worth a turn (`graph_overview`'s text is two counts).
pub fn tool_specs() -> Vec<ToolSpec> {
    ToolRequest::schemas()["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .filter(|tool| tool["name"] != "graph_overview")
                .map(|tool| ToolSpec {
                    name: tool["name"].as_str().unwrap_or_default().to_string(),
                    description: tool["description"].as_str().unwrap_or_default().to_string(),
                    parameters: tool["inputSchema"].clone(),
                })
                .collect()
        })
        .unwrap_or_default()
}

/// `text` cut to at most `max` characters at a line end, with a note
/// of how much was cut. The second value says whether it was cut.
fn cut_to(text: &str, max: usize, note: &str) -> (String, bool) {
    if text.chars().count() <= max {
        return (text.to_string(), false);
    }
    let mut end = 0;
    let mut kept_lines = 0;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        let next = end + line.len();
        if text[..next].chars().count() > max {
            if i == 0 {
                // One very long line: cut inside it.
                end = text
                    .char_indices()
                    .nth(max)
                    .map_or(text.len(), |(at, _)| at);
            }
            break;
        }
        end = next;
        kept_lines = i + 1;
    }
    let more = text.lines().count().saturating_sub(kept_lines);
    let mut out = text[..end].trim_end().to_string();
    out.push_str(&format!("\n[{more} more lines not shown; {note}]"));
    (out, true)
}

/// The typed request a call stands for, or what to tell the model
/// about a call that is not one.
fn request_of(call: &ToolCall, specs: &[ToolSpec]) -> Result<ToolRequest, String> {
    let names = || {
        specs
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    if call.name.is_empty() {
        return Err(format!(
            "error: that was not a tool call. Reply with one JSON object naming a tool ({}) and its arguments, or with the answer.",
            names()
        ));
    }
    if !specs.iter().any(|s| s.name == call.name) {
        return Err(format!(
            "error: no tool named {}. The tools are {}.",
            call.name,
            names()
        ));
    }
    let serde_json::Value::Object(arguments) = &call.arguments else {
        return Err(format!(
            "error: the arguments of {} must be a JSON object.",
            call.name
        ));
    };
    let mut request = arguments.clone();
    request.insert("tool".into(), call.name.clone().into());
    serde_json::from_value(serde_json::Value::Object(request))
        .map_err(|e| format!("error: bad arguments for {}: {e}.", call.name))
}

/// A call's arguments as the trail prints them: `range=Sheet1!A1:D4`.
fn args_text(call: &ToolCall) -> String {
    match &call.arguments {
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(key, value)| match value {
                serde_json::Value::String(text) => format!("{key}={text}"),
                other => format!("{key}={other}"),
            })
            .collect::<Vec<_>>()
            .join(";"),
        serde_json::Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

/// Whether two requests would read the same thing.
fn same_call(a: &ToolRequest, b: &ToolRequest) -> bool {
    match (a, b) {
        (ToolRequest::Expand { range: x }, ToolRequest::Expand { range: y })
        | (ToolRequest::GetCells { range: x }, ToolRequest::GetCells { range: y })
        | (ToolRequest::ListFormulas { range: x }, ToolRequest::ListFormulas { range: y }) => {
            same_range(x, y)
        }
        _ => a == b,
    }
}

/// A planner step as a typed request, for the steps that read a
/// range.
fn opening_request(step: &TrailStep) -> Option<ToolRequest> {
    let range = step.args.strip_prefix("range=")?.to_string();
    match step.tool.as_str() {
        "expand" => Some(ToolRequest::Expand { range }),
        "get_cells" => Some(ToolRequest::GetCells { range }),
        "list_formulas" => Some(ToolRequest::ListFormulas { range }),
        _ => None,
    }
}

/// A range in the workbook's own spelling (its sheet name, no `$`),
/// a single cell in its short form. `None` when it does not parse or
/// names a sheet the workbook lacks.
fn canonical_range(range: &str, tools: &Tools) -> Option<String> {
    let (sheet, r0, c0, r1, c1) = parse_range_uncapped(range)?;
    let sheet = tools.sheet_for(&sheet)?.name.clone();
    if (r0, c0) == (r1, c1) {
        // One cell reads as one: `Staff!B2`, as an answer cites it.
        return Some(format!(
            "{}!{}{}",
            gridtalk_graph::quote_sheet(&sheet),
            gridtalk_graph::col_name(c0),
            r0 + 1
        ));
    }
    Some(
        gridtalk_graph::RangeRef {
            sheet,
            r0,
            c0,
            r1,
            c1,
        }
        .a1(),
    )
}

/// The ranges a successful call read, in the workbook's own spelling.
fn ranges_read(request: &ToolRequest, data: &serde_json::Value, tools: &Tools) -> Vec<String> {
    let canonical = |range: &str| canonical_range(range, tools);
    match request {
        ToolRequest::Expand { range }
        | ToolRequest::GetCells { range }
        | ToolRequest::ListFormulas { range } => canonical(range).into_iter().collect(),
        ToolRequest::Search { .. } => data["hits"]
            .as_array()
            .map(|hits| {
                hits.iter()
                    .filter_map(|hit| hit["range"].as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        ToolRequest::WhatIf { cell, target, .. } => {
            let cell = if cell.contains('!') {
                Some(cell.clone())
            } else {
                // A bare cell is on the target's sheet.
                target
                    .rsplit_once('!')
                    .map(|(sheet, _)| format!("{sheet}!{cell}"))
            };
            [Some(target.clone()), cell]
                .into_iter()
                .flatten()
                .filter_map(|r| canonical(&r))
                .collect()
        }
        ToolRequest::WorkbookInfo | ToolRequest::GraphOverview => Vec::new(),
    }
}

/// Whether two ranges share a cell.
fn overlap(a: &str, b: &str) -> bool {
    match (parse_range_uncapped(a), parse_range_uncapped(b)) {
        (Some((sa, ar0, ac0, ar1, ac1)), Some((sb, br0, bc0, br1, bc1))) => {
            sheet_name_eq(&sa, &sb) && ar0 <= br1 && br0 <= ar1 && ac0 <= bc1 && bc0 <= ac1
        }
        _ => false,
    }
}

/// The size of the conversation in characters, as the backend will
/// be sent it.
fn conversation_chars(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            m.content.len()
                + m.tool_calls
                    .iter()
                    .map(|c| c.name.len() + c.arguments.to_string().len() + 16)
                    .sum::<usize>()
                + 8
        })
        .sum()
}

/// Run the loop. Called through
/// [`crate::investigate::investigate_run`], which has checked that
/// the backend takes tool turns.
pub(crate) fn tool_loop(
    ask: &Ask,
    tools: &Tools,
    backend: &dyn ChatBackend,
    run: &mut Run,
) -> Result<Investigation, InferError> {
    let question = ask.question;
    let opening = gather(ask, tools, ask.max_steps.min(OPENING_STEPS));
    for step in &opening.trail {
        (run.on_step)(step);
    }
    let Gathered {
        mut trail,
        context,
        mut citations,
        standalone,
        ..
    } = opening;
    let opening_steps = trail.len();
    let specs = tool_specs();
    let specs_chars: usize = specs
        .iter()
        .map(|s| s.name.len() + s.description.len() + s.parameters.to_string().len())
        .sum();
    let first_call = ask.adapter.is_none_or(|a| a.first_call);

    let mut system = LOOP_PROMPT.to_string();
    system.push_str(&gridtalk_adapter::prompt_text(
        ask.adapter,
        &standalone,
        ask.sidecar,
    ));
    let system = gridtalk_adapter::cap_prompt(system);

    let mut trace = |event: serde_json::Value| {
        if let Some(sink) = run.trace.as_mut() {
            sink(&event);
        }
    };
    trace(serde_json::json!({
        "event": "session", "backend": backend.name(), "question": question,
        "max_steps": ask.max_steps, "first_call": first_call, "tools": specs,
        "opening": trail.iter().map(|s| serde_json::json!({
            "tool": s.tool, "args": s.args, "result": s.result
        })).collect::<Vec<_>>(),
    }));

    let mut answer = String::new();
    let mut model_turns = 0usize;
    // The ranges the model's own calls read, newest last.
    let mut chosen_ranges: Vec<String> = Vec::new();
    // The ranges the results the model was sent name: what a column
    // reads, a formula's references. Citing one of them is citing a
    // tool result, though no call read its cells.
    let mut shown: Vec<String> = Vec::new();
    backend.with_tool_session(&mut |session| {
        let window = session.context_tokens().unwrap_or(DEFAULT_CONTEXT_TOKENS);
        // A turn may generate up to a quarter of a small window; the
        // rest, less a margin for the chat template, is the
        // conversation's.
        let turn_tokens = ANSWER_TOKENS.min(window / 4).max(64);
        let budget = window
            .saturating_sub(turn_tokens + (window / 16).min(256))
            .max(512);
        // Characters per token, measured from the backend's own
        // counts once it reports them. Tables of numbers run well
        // under the four of ordinary prose.
        let mut chars_per_token = 3.0f64;
        let room = |chars_per_token: f64| (budget as f64 * chars_per_token) as usize;
        let result_cap = (room(chars_per_token) / 5).clamp(800, MAX_RESULT_CHARS);
        let (opening_text, opening_cut) = cut_to(
            &context,
            (room(chars_per_token) / 3).clamp(1_500, 2 * MAX_RESULT_CHARS),
            "read the range you need with a tool",
        );
        let mut user = history_prompt(ask.history);
        user.push_str(&format!("Question: {question}\n"));
        if standalone != question {
            user.push_str(&format!("(searched as: {standalone})\n"));
        }
        user.push_str(&format!("\nAlready read for you:\n{opening_text}"));
        let mut messages = vec![Message::system(system.clone()), Message::user(user)];
        shown.extend(qualified_ranges(&opening_text));

        // Requests already answered, with the message that holds the
        // result (none for a planner step: it is in the user message).
        let mut seen: Vec<(ToolRequest, Option<usize>)> = if opening_cut {
            Vec::new()
        } else {
            trail
                .iter()
                .filter_map(opening_request)
                .map(|r| (r, None))
                .collect()
        };
        // Tool messages already shortened.
        let mut shortened: Vec<usize> = Vec::new();
        let mut calls_left = ask.max_steps;
        let mut chosen = 0usize;
        let mut nudged = false;
        loop {
            if run.cancel.load(Ordering::Relaxed) {
                return Err(InferError::Cancelled);
            }
            let choice = if calls_left == 0 {
                ToolChoice::None
            } else if first_call && chosen == 0 && !nudged {
                ToolChoice::Required
            } else {
                ToolChoice::Auto
            };
            model_turns += 1;
            trace(serde_json::json!({
                "event": "request", "turn": model_turns, "choice": choice, "messages": messages,
            }));
            let sent_chars = conversation_chars(&messages) + specs_chars;
            // A turn that must be a call has no answer to pass on.
            let mut silent = |_: &str| {};
            let turn = session.turn(
                &TurnRequest {
                    messages: &messages,
                    tools: &specs,
                    choice,
                    max_tokens: turn_tokens,
                },
                &run.cancel,
                if choice == ToolChoice::Required {
                    &mut silent
                } else {
                    &mut *run.on_token
                },
            )?;
            trace(serde_json::json!({
                "event": "response", "turn": model_turns, "text": turn.text,
                "calls": turn.calls, "usage": turn.usage,
            }));
            if let Some(usage) = turn.usage.filter(|u| u.prompt_tokens > 0) {
                chars_per_token = (sent_chars as f64 / usage.prompt_tokens as f64).clamp(1.5, 5.0);
            }
            if turn.calls.is_empty() {
                if choice == ToolChoice::Required {
                    // The backend could not force a call: say so once.
                    nudged = true;
                    messages.push(Message::assistant(turn.text, Vec::new()));
                    messages.push(Message::user(
                        "Call a tool before you answer: read the cells the answer rests on.",
                    ));
                    continue;
                }
                answer = turn.text;
                return Ok(());
            }
            if choice == ToolChoice::None {
                // A call where only an answer was allowed: take what
                // text there is.
                answer = turn.text;
                return Ok(());
            }
            messages.push(Message::assistant(turn.text.clone(), turn.calls.clone()));
            for call in &turn.calls {
                if run.cancel.load(Ordering::Relaxed) {
                    return Err(InferError::Cancelled);
                }
                chosen += 1;
                let at = messages.len();
                let (text, ok) = if calls_left == 0 {
                    (
                        "not run: the tool budget is spent. Answer from the results above."
                            .to_string(),
                        false,
                    )
                } else {
                    calls_left -= 1;
                    match request_of(call, &specs) {
                        Err(problem) => (problem, false),
                        Ok(request) => {
                            let repeat = seen.iter().position(|(done, message)| {
                                same_call(done, &request)
                                    && message.is_none_or(|m| !shortened.contains(&m))
                            });
                            if repeat.is_some() {
                                (
                                    "already called with these arguments: the result is above. Read something else or answer."
                                        .to_string(),
                                    false,
                                )
                            } else {
                                let result = tools.call(&request);
                                if result.ok {
                                    for range in ranges_read(&request, &result.data, tools) {
                                        if !citations.iter().any(|c| same_range(c, &range)) {
                                            citations.push(range.clone());
                                        }
                                        if !chosen_ranges.iter().any(|c| same_range(c, &range)) {
                                            chosen_ranges.push(range);
                                        }
                                    }
                                    seen.retain(|(done, _)| !same_call(done, &request));
                                    seen.push((request, Some(at)));
                                }
                                let text = if result.ok {
                                    result.text
                                } else {
                                    format!("error: {}", result.text)
                                };
                                (text, result.ok)
                            }
                        }
                    }
                };
                let step = TrailStep {
                    tool: if call.name.is_empty() {
                        "invalid".to_string()
                    } else {
                        call.name.clone()
                    },
                    args: args_text(call),
                    result: text.clone(),
                };
                (run.on_step)(&step);
                trail.push(step);
                let (sent, _) = cut_to(
                    &text,
                    result_cap,
                    "call again with a narrower range for the rest",
                );
                trace(serde_json::json!({
                    "event": "tool", "turn": model_turns, "id": call.id, "name": call.name,
                    "arguments": call.arguments, "ok": ok, "result": sent,
                    "result_chars": text.chars().count(),
                }));
                if ok {
                    shown.extend(qualified_ranges(&sent));
                }
                messages.push(Message::tool(call, sent));
            }
            if calls_left == 0 {
                if let Some(last) = messages.last_mut() {
                    last.content.push_str(
                        "\n\n[That was the last tool call allowed. Answer now from the results.]",
                    );
                }
            }
            // Out of room: shorten the oldest results first, and
            // stop calling tools when that is not enough.
            let over = |messages: &[Message], chars_per_token: f64| {
                conversation_chars(messages) + specs_chars > room(chars_per_token)
            };
            if over(&messages, chars_per_token) {
                let mut cut = 0usize;
                for at in 0..messages.len() {
                    if !over(&messages, chars_per_token) {
                        break;
                    }
                    if messages[at].role != Role::Tool || shortened.contains(&at) {
                        continue;
                    }
                    let (short, was_cut) = cut_to(
                        &messages[at].content,
                        SHORT_RESULT_CHARS,
                        "shortened to make room; call the tool again if you need the rest",
                    );
                    if was_cut {
                        messages[at].content = short;
                        shortened.push(at);
                        cut += 1;
                    }
                }
                trace(serde_json::json!({
                    "event": "shortened", "turn": model_turns, "results": cut,
                }));
                if over(&messages, chars_per_token) && calls_left > 0 {
                    calls_left = 0;
                    if let Some(last) = messages.last_mut() {
                        last.content.push_str(
                            "\n\n[No room is left for more results. Answer now from the results.]",
                        );
                    }
                }
            }
        }
    })?;

    if answer.trim().is_empty() {
        answer = "(the backend returned an empty answer; see the trail for the tool results)"
            .to_string();
    }
    // The answer's own citations, split by whether a call read them.
    let mut unread_citations: Vec<String> = Vec::new();
    let mut supported = false;
    for cited in qualified_ranges(&answer) {
        if citations
            .iter()
            .chain(shown.iter())
            .any(|read| overlap(&cited, read))
        {
            supported = true;
            // A cell inside a range that was read is a citation in
            // its own right, kept as the answer wrote it so that it
            // can be found in the answer again.
            if !citations.iter().any(|c| same_range(c, &cited)) {
                citations.push(cited);
            }
        } else if !unread_citations.iter().any(|c| same_range(c, &cited)) {
            unread_citations.push(cited);
        }
    }
    // Citation guarantee: an answer that cites nothing that was read
    // is given its sources, the model's own reads first.
    if !supported && !citations.iter().any(|c| contains_bounded(&answer, c)) {
        let mut sources: Vec<&String> = chosen_ranges.iter().rev().collect();
        for c in &citations {
            if !sources.iter().any(|s| same_range(s, c)) {
                sources.push(c);
            }
        }
        if !sources.is_empty() {
            let named: Vec<&str> = sources
                .iter()
                .take(MAX_SOURCES)
                .map(|s| s.as_str())
                .collect();
            answer.push_str(&format!("\n\nSources: {}", named.join(", ")));
        }
    }
    trace(serde_json::json!({
        "event": "answer", "answer": answer, "citations": citations,
        "unread_citations": unread_citations, "model_turns": model_turns,
        "tool_calls": trail.len() - opening_steps,
    }));
    Ok(Investigation {
        answer,
        trail,
        citations,
        unread_citations,
        model_turns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::investigate::investigate_run;
    use gridtalk_graph::{build_graph, parse_delimited};
    use gridtalk_index::HybridIndex;
    use gridtalk_infer::{CancelToken, ChatRequest, EchoBackend, ToolSession, Turn};
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, Mutex};

    /// A backend that replays scripted turns and records what it was
    /// asked.
    struct Scripted {
        turns: Mutex<Vec<Turn>>,
        asked: Mutex<Vec<(ToolChoice, Vec<Message>)>>,
        context: Option<usize>,
    }

    impl Scripted {
        fn new(turns: Vec<Turn>) -> Self {
            Scripted {
                turns: Mutex::new(turns.into_iter().rev().collect()),
                asked: Mutex::new(Vec::new()),
                context: None,
            }
        }
        fn asked(&self) -> Vec<(ToolChoice, Vec<Message>)> {
            self.asked.lock().unwrap().clone()
        }
    }

    struct ScriptedSession<'a>(&'a Scripted);

    impl ToolSession for ScriptedSession<'_> {
        fn turn(
            &mut self,
            req: &TurnRequest,
            _cancel: &CancelToken,
            on_token: &mut dyn FnMut(&str),
        ) -> Result<Turn, InferError> {
            self.0
                .asked
                .lock()
                .unwrap()
                .push((req.choice, req.messages.to_vec()));
            // Like a real backend, a turn that may only answer skips
            // the calls the script still holds.
            let mut turns = self.0.turns.lock().unwrap();
            let turn = loop {
                let turn = turns.pop().expect("the script ran out of turns");
                if turn.calls.is_empty() || req.choice != ToolChoice::None {
                    break turn;
                }
            };
            if turn.calls.is_empty() {
                on_token(&turn.text);
            }
            Ok(turn)
        }
        fn context_tokens(&self) -> Option<usize> {
            self.0.context
        }
    }

    impl ChatBackend for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn stream(
            &self,
            _req: &ChatRequest,
            _cancel: &CancelToken,
            on_token: &mut dyn FnMut(&str),
        ) -> Result<(), InferError> {
            on_token("fixed pipeline answer");
            Ok(())
        }
        fn supports_tools(&self) -> bool {
            true
        }
        fn with_tool_session(
            &self,
            run: &mut dyn FnMut(&mut dyn ToolSession) -> Result<(), InferError>,
        ) -> Result<(), InferError> {
            run(&mut ScriptedSession(self))
        }
    }

    fn call(n: usize, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: format!("call_{n}"),
            name: name.into(),
            arguments,
        }
    }
    fn calls(calls: Vec<ToolCall>) -> Turn {
        Turn {
            text: String::new(),
            calls,
            usage: None,
        }
    }
    fn says(text: &str) -> Turn {
        Turn {
            text: text.into(),
            calls: Vec::new(),
            usage: None,
        }
    }

    /// Two sheets: 40 rows of hours and a rate table they look up.
    fn workbook() -> gridtalk_graph::WorkbookSnapshot {
        let mut labour = String::from("Ref,Employee,Hours,Rate,Cost\n");
        for r in 2..=41 {
            labour.push_str(&format!(
                "T{r},E{},{},\"=INDEX(Staff!$B$2:$B$5,MATCH(B{r},Staff!$A$2:$A$5,0))\",=C{r}*D{r}\n",
                r % 4 + 1,
                r % 7 + 1
            ));
        }
        let mut wb = parse_delimited("book", "Labour", &labour, ',');
        let staff = parse_delimited(
            "book",
            "Staff",
            "Code,Rate per Hour\nE1,100\nE2,120\nE3,150\nE4,90\n",
            ',',
        );
        wb.sheets.extend(staff.sheets);
        wb
    }

    struct Fixture {
        wb: gridtalk_graph::WorkbookSnapshot,
        graph: gridtalk_graph::Graph,
        hybrid: HybridIndex,
        calc: gridtalk_graph::eval::EvalCache,
    }

    impl Fixture {
        fn new() -> Self {
            let wb = workbook();
            let (graph, violations) = build_graph(&wb);
            assert!(violations.is_empty(), "{violations:?}");
            let hybrid = HybridIndex::build(&wb, &graph);
            Fixture {
                wb,
                graph,
                hybrid,
                calc: gridtalk_graph::eval::EvalCache::new(),
            }
        }
        fn tools(&self) -> Tools<'_> {
            Tools {
                wb: &self.wb,
                graph: &self.graph,
                hybrid: &self.hybrid,
                calc: &self.calc,
            }
        }
    }

    /// What one run reported while it was under way.
    #[derive(Default)]
    struct Seen {
        tokens: String,
        steps: Vec<String>,
        events: Vec<serde_json::Value>,
    }

    fn run_with(
        fixture: &Fixture,
        backend: &dyn ChatBackend,
        question: &str,
        max_steps: usize,
        adapter: Option<&gridtalk_adapter::Adapter>,
        tool_loop: bool,
    ) -> (Investigation, Seen) {
        let mut seen = Seen::default();
        let (mut tokens, mut steps, mut events) = (String::new(), Vec::new(), Vec::new());
        let inv = investigate_run(
            &Ask {
                question,
                max_steps,
                adapter,
                sidecar: None,
                history: &[],
                selection: None,
            },
            &fixture.tools(),
            backend,
            &mut Run {
                cancel: Arc::new(AtomicBool::new(false)),
                on_token: &mut |t| tokens.push_str(t),
                on_step: &mut |s: &TrailStep| steps.push(s.tool.clone()),
                trace: Some(&mut |e: &serde_json::Value| events.push(e.clone())),
                tool_loop,
            },
        )
        .unwrap();
        seen.tokens = tokens;
        seen.steps = steps;
        seen.events = events;
        (inv, seen)
    }

    fn event_names(seen: &Seen) -> Vec<&str> {
        seen.events
            .iter()
            .map(|e| e["event"].as_str().unwrap())
            .collect()
    }

    #[test]
    fn the_model_chooses_its_calls_after_the_planner_opens() {
        let fixture = Fixture::new();
        let backend = Scripted::new(vec![
            calls(vec![call(
                1,
                "expand",
                serde_json::json!({"range": "Labour!D1:D41"}),
            )]),
            calls(vec![call(
                2,
                "list_formulas",
                serde_json::json!({"range": "labour!d2:d41"}),
            )]),
            says("Cost is Hours times Rate (Labour!E2), and Rate is looked up in Staff!B2:B5."),
        ]);
        let (inv, seen) = run_with(
            &fixture,
            &backend,
            "How is the labour cost worked out?",
            8,
            None,
            true,
        );
        let asked = backend.asked();
        // The first turn must be a call; later ones may answer.
        let choices: Vec<ToolChoice> = asked.iter().map(|(c, _)| *c).collect();
        assert_eq!(
            choices,
            [ToolChoice::Required, ToolChoice::Auto, ToolChoice::Auto]
        );
        // The model starts from what the planner read.
        let first = &asked[0].1;
        assert_eq!(first.len(), 2);
        assert!(first[0]
            .content
            .starts_with("You answer questions about a spreadsheet workbook."));
        assert!(
            first[1]
                .content
                .contains("Question: How is the labour cost worked out?\n")
                && first[1]
                    .content
                    .contains("\nAlready read for you:\nsearch results:\n"),
            "{}",
            first[1].content
        );
        // Each result goes back as a tool message answering its call.
        let last = &asked[2].1;
        assert_eq!(last.len(), 6);
        assert_eq!(last[2].tool_calls[0].name, "expand");
        assert_eq!(last[3].role, Role::Tool);
        assert_eq!(last[3].tool_call_id.as_deref(), Some("call_1"));
        assert!(
            last[3]
                .content
                .contains("-depends_on-> Rate per Hour  Staff!B1:B5"),
            "{}",
            last[3].content
        );
        assert!(
            last[5]
                .content
                .starts_with("column D (Labour!D2:D41): 40 formulas, all like Labour!D2 = INDEX("),
            "{}",
            last[5].content
        );
        // The trail holds the planner's steps, then the model's.
        let opening = inv.trail.len() - 2;
        assert!((1..=OPENING_STEPS).contains(&opening), "{opening}");
        assert_eq!(inv.trail[opening].tool, "expand");
        assert_eq!(inv.trail[opening].args, "range=Labour!D1:D41");
        assert_eq!(inv.trail[opening + 1].tool, "list_formulas");
        assert_eq!(seen.steps.len(), inv.trail.len());
        assert_eq!(inv.model_turns, 3);
        // The answer streamed, and nothing was added to it: it cites
        // cells inside ranges the calls read, which become citations.
        assert_eq!(seen.tokens, inv.answer);
        assert!(!inv.answer.contains("Sources:"), "{}", inv.answer);
        assert!(
            inv.unread_citations.is_empty(),
            "{:?}",
            inv.unread_citations
        );
        for cited in ["Labour!E2", "Labour!D2:D41", "Labour!D1:D41"] {
            assert!(
                inv.citations.iter().any(|c| c == cited),
                "{:?}",
                inv.citations
            );
        }
        // Staff!B2:B5 was not read, but the expansion named the column
        // it is in.
        assert!(
            inv.citations.iter().any(|c| c == "Staff!B2:B5"),
            "{:?}",
            inv.citations
        );
        assert_eq!(
            event_names(&seen),
            [
                "session", "request", "response", "tool", "request", "response", "tool", "request",
                "response", "answer"
            ]
        );
        assert_eq!(seen.events[0]["tools"].as_array().unwrap().len(), 6);
        assert_eq!(seen.events[1]["messages"].as_array().unwrap().len(), 2);
        assert_eq!(seen.events[3]["name"], "expand");
        assert_eq!(seen.events[9]["tool_calls"], 2);
    }

    #[test]
    fn the_budget_ends_in_a_turn_that_can_only_answer() {
        let fixture = Fixture::new();
        // Three calls in the first turn against a budget of two.
        let backend = Scripted::new(vec![
            calls(vec![
                call(1, "get_cells", serde_json::json!({"range": "Staff!A1:B5"})),
                call(2, "get_cells", serde_json::json!({"range": "Labour!A1:E3"})),
                call(3, "get_cells", serde_json::json!({"range": "Labour!A4:E6"})),
            ]),
            says("E3 is paid 150 an hour (Staff!B4)."),
        ]);
        let (inv, _) = run_with(&fixture, &backend, "What is E3 paid?", 2, None, true);
        let asked = backend.asked();
        assert_eq!(asked.len(), 2);
        assert_eq!(asked[1].0, ToolChoice::None);
        let messages = &asked[1].1;
        let results: Vec<&Message> = messages.iter().filter(|m| m.role == Role::Tool).collect();
        assert_eq!(results.len(), 3);
        assert!(
            results[0].content.contains("Staff!B4: 150"),
            "{}",
            results[0].content
        );
        assert!(results[2]
            .content
            .starts_with("not run: the tool budget is spent."));
        assert!(results[2]
            .content
            .ends_with("[That was the last tool call allowed. Answer now from the results.]"));
        assert_eq!(inv.answer, "E3 is paid 150 an hour (Staff!B4).");
        // The planner's opening is bounded by the same budget.
        let opening = inv
            .trail
            .iter()
            .position(|s| s.tool == "get_cells")
            .unwrap();
        assert!(opening <= 2, "{opening}");
    }

    #[test]
    fn an_answer_before_any_call_is_sent_back_once() {
        let fixture = Fixture::new();
        let backend = Scripted::new(vec![
            says("It is probably 100."),
            calls(vec![call(
                1,
                "get_cells",
                serde_json::json!({"range": "Staff!A2:B2"}),
            )]),
            says("E1 is paid 100 an hour (Staff!B2)."),
        ]);
        let (inv, seen) = run_with(&fixture, &backend, "What is E1 paid?", 8, None, true);
        let asked = backend.asked();
        assert_eq!(asked[0].0, ToolChoice::Required);
        assert_eq!(asked[1].0, ToolChoice::Auto);
        let nudge = asked[1].1.last().unwrap();
        assert_eq!(nudge.role, Role::User);
        assert!(nudge.content.starts_with("Call a tool before you answer"));
        assert_eq!(inv.answer, "E1 is paid 100 an hour (Staff!B2).");
        // The refused answer was never shown.
        assert_eq!(seen.tokens, inv.answer);

        // `first_call = false` lets the model answer from the opening.
        let mut adapter = gridtalk_adapter::Adapter::empty("free");
        adapter.first_call = false;
        let backend = Scripted::new(vec![says("Staff!B2 holds 100.")]);
        let (inv, _) = run_with(
            &fixture,
            &backend,
            "What is E1 paid?",
            8,
            Some(&adapter),
            true,
        );
        assert_eq!(backend.asked()[0].0, ToolChoice::Auto);
        assert_eq!(inv.answer, "Staff!B2 holds 100.");
        assert_eq!(inv.model_turns, 1);
    }

    #[test]
    fn a_call_that_cannot_run_is_explained_to_the_model() {
        let fixture = Fixture::new();
        let backend = Scripted::new(vec![
            calls(vec![
                call(1, "read_cells", serde_json::json!({"range": "Staff!A1"})),
                call(2, "get_cells", serde_json::json!({"cells": "Staff!A1"})),
                call(3, "", serde_json::Value::String("{\"tool\":\"get".into())),
                call(4, "get_cells", serde_json::json!("Staff!A1")),
                call(5, "get_cells", serde_json::json!({"range": "Nowhere!A1"})),
                call(6, "graph_overview", serde_json::json!({})),
            ]),
            calls(vec![call(
                7,
                "get_cells",
                serde_json::json!({"range": "Staff!A1:B2"}),
            )]),
            calls(vec![call(
                8,
                "get_cells",
                serde_json::json!({"range": "staff!$A$1:$B$2"}),
            )]),
            says("Staff!B2 is 100."),
        ]);
        let (inv, _) = run_with(&fixture, &backend, "What is E1 paid?", 12, None, true);
        let asked = backend.asked();
        let results: Vec<&str> = asked[3]
            .1
            .iter()
            .filter(|m| m.role == Role::Tool)
            .map(|m| m.content.as_str())
            .collect();
        assert!(results[0].starts_with("error: no tool named read_cells. The tools are workbook_info, search, expand, get_cells, list_formulas, what_if."), "{}", results[0]);
        assert!(
            results[1].starts_with("error: bad arguments for get_cells: unknown field `cells`"),
            "{}",
            results[1]
        );
        assert!(
            results[2].starts_with("error: that was not a tool call."),
            "{}",
            results[2]
        );
        assert_eq!(
            results[3],
            "error: the arguments of get_cells must be a JSON object."
        );
        assert_eq!(results[4], "error: no sheet named Nowhere");
        assert!(
            results[5].starts_with("error: no tool named graph_overview."),
            "{}",
            results[5]
        );
        assert!(results[6].contains("Staff!B2: 100"), "{}", results[6]);
        // The same range in another spelling is not read twice.
        assert!(
            results[7].starts_with("already called with these arguments"),
            "{}",
            results[7]
        );
        // A failed call reads nothing, so it is not a citation.
        assert!(
            !inv.citations.iter().any(|c| c.contains("Nowhere")),
            "{:?}",
            inv.citations
        );
        let invalid = inv.trail.iter().find(|s| s.tool == "invalid").unwrap();
        assert_eq!(invalid.args, "{\"tool\":\"get");
    }

    #[test]
    fn results_are_cut_to_the_context_and_old_ones_shortened() {
        let fixture = Fixture::new();
        let read = |n: usize, range: &str| {
            calls(vec![call(
                n,
                "get_cells",
                serde_json::json!({"range": range}),
            )])
        };
        let mut backend = Scripted::new(vec![
            read(1, "Labour!A1:E40"),
            read(2, "Labour!A2:E41"),
            read(3, "Labour!A3:E41"),
            read(4, "Labour!A4:E41"),
            read(5, "Labour!A5:E41"),
            read(6, "Labour!A6:E41"),
            says("Labour!E2 is the first cost."),
        ]);
        // A small window: 4,000 tokens, a quarter of them kept for the
        // model's turn.
        backend.context = Some(4_000);
        let (inv, seen) = run_with(&fixture, &backend, "What is the first cost?", 8, None, true);
        let asked = backend.asked();
        // The whole table is 40 lines of some 45 characters; the model
        // is sent the top of it and told how to get the rest. (The
        // trace has the result as it was added; a later turn may
        // shorten it further.)
        let first_result = seen
            .events
            .iter()
            .find(|e| e["event"] == "tool")
            .and_then(|e| e["result"].as_str())
            .unwrap();
        assert!(first_result.chars().count() < 2_000, "{first_result}");
        assert!(
            first_result
                .ends_with("more lines not shown; call again with a narrower range for the rest]"),
            "{first_result}"
        );
        // The trail keeps the whole result.
        let full = inv
            .trail
            .iter()
            .find(|s| s.tool == "get_cells" && s.args == "range=Labour!A1:E40");
        assert!(
            full.unwrap().result.contains("\n40 | A: T40"),
            "{}",
            full.unwrap().result
        );
        // Later turns shorten the oldest results instead of dropping
        // them, and the conversation stays inside the window.
        assert!(
            event_names(&seen).contains(&"shortened"),
            "{:?}",
            event_names(&seen)
        );
        let last = &asked.last().unwrap().1;
        assert!(
            last.iter().any(|m| m.role == Role::Tool
                && m.content
                    .contains("shortened to make room; call the tool again")),
            "{last:?}"
        );
        for (_, messages) in &asked {
            assert!(
                conversation_chars(messages) < 4_000 * 3,
                "{}",
                conversation_chars(messages)
            );
        }
        assert_eq!(inv.answer, "Labour!E2 is the first cost.");
    }

    #[test]
    fn citations_are_checked_against_what_was_read() {
        let fixture = Fixture::new();
        // A range no call read is reported, not trusted.
        let backend = Scripted::new(vec![
            calls(vec![call(
                1,
                "get_cells",
                serde_json::json!({"range": "Staff!A1:B5"}),
            )]),
            says("The rate is in Staff!B3, and the budget is in Budget!C9."),
        ]);
        let (inv, _) = run_with(&fixture, &backend, "Where is the rate?", 8, None, true);
        assert_eq!(inv.unread_citations, ["Budget!C9"]);
        assert!(inv.citations.iter().any(|c| c == "Staff!B3"));
        assert!(!inv.answer.contains("Sources:"));
        // An answer that cites nothing is given its sources, the
        // model's own reads first.
        let backend = Scripted::new(vec![
            calls(vec![call(
                1,
                "get_cells",
                serde_json::json!({"range": "staff!b3"}),
            )]),
            says("The rate is 120."),
        ]);
        let (inv, _) = run_with(&fixture, &backend, "Where is the rate?", 8, None, true);
        assert!(
            inv.answer
                .starts_with("The rate is 120.\n\nSources: Staff!B3, "),
            "{}",
            inv.answer
        );
        // An empty reply is said to be one.
        let backend = Scripted::new(vec![
            calls(vec![call(1, "workbook_info", serde_json::json!({}))]),
            says("  "),
        ]);
        let (inv, _) = run_with(&fixture, &backend, "What is this?", 8, None, true);
        assert!(
            inv.answer
                .starts_with("(the backend returned an empty answer"),
            "{}",
            inv.answer
        );
    }

    #[test]
    fn the_fixed_pipeline_stays_for_backends_and_callers_that_want_it() {
        let fixture = Fixture::new();
        // The echo backend takes no tool turns.
        let (inv, seen) = run_with(&fixture, &EchoBackend, "What is E1 paid?", 8, None, true);
        assert!(
            inv.answer.starts_with("Based on the workbook context:"),
            "{}",
            inv.answer
        );
        assert_eq!(inv.model_turns, 1);
        assert!(seen.events.is_empty());
        assert_eq!(seen.steps.len(), inv.trail.len());
        assert!(seen.tokens.starts_with("Based on the workbook context:"));
        // A backend that does is still kept to the pipeline by the
        // caller or by the adapter.
        let backend = Scripted::new(Vec::new());
        let (inv, _) = run_with(&fixture, &backend, "What is E1 paid?", 8, None, false);
        assert!(
            inv.answer.starts_with("fixed pipeline answer"),
            "{}",
            inv.answer
        );
        let mut adapter = gridtalk_adapter::Adapter::empty("fixed");
        adapter.tool_loop = false;
        let (inv, _) = run_with(
            &fixture,
            &backend,
            "What is E1 paid?",
            8,
            Some(&adapter),
            true,
        );
        assert!(
            inv.answer.starts_with("fixed pipeline answer"),
            "{}",
            inv.answer
        );
        assert!(backend.asked().is_empty());
    }
}
