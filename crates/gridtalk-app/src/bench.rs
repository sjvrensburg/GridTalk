//! Phase C (agentic plan): synthetic benchmark generation and
//! scored runs. Tasks are synthesised from a workbook with
//! **deterministic ground truth** — cell values, formula results
//! and what-if outcomes computed by the evaluator, never judged by
//! an LLM. `gridtalk bench run` drives the investigate harness
//! over the tasks and writes a CARD.md score report, for the echo
//! backend (the floor) or the configured local model.

use crate::investigate::{contains_bounded, contains_word, investigate_run, Ask, Run};
use crate::tools::Tools;
use gridtalk_graph::eval::{EvalCache, Value};
use gridtalk_graph::lineage::Lineage;
use gridtalk_graph::{sheet_name_eq, CellValue, WorkbookSnapshot};
use gridtalk_infer::ChatBackend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchTask {
    pub id: String,
    /// cell_value | formula_result | what_if, or a free string for
    /// hand-written tasks (`check_agrees`, `sheet_purpose`, …).
    pub kind: String,
    /// train | dev — dev is never optimised against directly.
    pub split: String,
    pub question: String,
    /// Deterministic expected numeric answer (0 when the answer is
    /// text; see `expected_text`). Task files scored by
    /// `expected_text` or `want` may omit it.
    #[serde(default)]
    pub expected: f64,
    /// Range the answer must cite (e.g. Sheet1!D2). A task with no
    /// cell to cite ("How many sheets are there?") omits it and
    /// stays out of citation precision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citation: Option<String>,
    /// For a text or boolean answer, the value as the cell shows it
    /// (`TRUE`, `big`); it is judged by [`judge_text`] instead of
    /// `expected`. Absent in task files written before typed tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_text: Option<String>,
    /// For an answer that is not a single value ("What does this
    /// column read?"): strings any one of which marks the answer
    /// right, matched case-blind by [`judge_want`]. Takes
    /// precedence over `expected_text`/`expected` when non-empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub want: Vec<String>,
    /// Earlier turns of the conversation, for follow-up questions
    /// ("and what reads its Hrs column?"): the run condenses the
    /// question against them before searching.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<gridtalk_index::ChatTurn>,
    /// The range selected when the question was asked
    /// ("what feeds this?"), read ahead of the search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<String>,
}

/// A task's ground truth: a number, or text as the cell displays it.
#[derive(Debug, Clone, PartialEq)]
enum Expected {
    Number(f64),
    Text(String),
}

impl Expected {
    /// The answer a formula result or what-if outcome gives, if it is
    /// one a task can ask for (a finite number, text or a boolean).
    fn of(value: &Value) -> Option<Expected> {
        match value {
            Value::Number(n) if n.is_finite() => Some(Expected::Number(*n)),
            Value::Number(_) => None,
            Value::Empty => Some(Expected::Number(0.0)),
            // Blank text has no answer to state: judge_text never
            // passes it and hermit's prepare_data rejects it.
            Value::Text(t) if t.trim().is_empty() => None,
            Value::Text(_) | Value::Bool(_) => Some(Expected::Text(value.display())),
        }
    }

    /// How the question asks for it.
    fn ask(&self, what: &str) -> String {
        match self {
            Expected::Number(_) => format!("Answer with just the {what}number."),
            Expected::Text(_) => format!("Answer with just the {what}value."),
        }
    }
}

/// Most precedent cells a what-if task tries per formula, and most
/// cells of its precedent areas scanned looking for them.
const WHAT_IF_ATTEMPTS: usize = 64;
const WHAT_IF_SCAN: usize = 4096;

fn a1(sheet: &str, row: usize, col: usize) -> String {
    format!(
        "{}!{}{}",
        gridtalk_graph::quote_sheet(sheet),
        gridtalk_graph::col_name(col),
        row + 1
    )
}

fn split_for(kind: &str, question: &str) -> &'static str {
    // Stable hash split (not positional): the same task lands in
    // the same split regardless of workbook scan order or --max.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in kind.bytes().chain(question.bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    if h % 3 == 2 {
        "dev"
    } else {
        "train"
    }
}

pub struct TaskGeneration {
    pub tasks: Vec<BenchTask>,
    /// Formula cells skipped because they error or are non-finite
    /// (no deterministic ground truth exists for them).
    pub skipped_formulas: usize,
}

/// Synthesise benchmark tasks from a workbook. Deterministic: the
/// same workbook always yields the same tasks, ids and splits.
#[allow(dead_code)] // convenience wrapper over generate_tasks_ex
pub fn generate_tasks(wb: &WorkbookSnapshot, max_tasks: usize) -> Vec<BenchTask> {
    generate_tasks_ex(wb, max_tasks, &EvalCache::new()).tasks
}

/// [`generate_tasks`] with the skip count. `cache` must belong to `wb`;
/// pass the cache a `Tools` for the same workbook holds, so the bench
/// run reuses the recalculation task generation paid for.
pub fn generate_tasks_ex(
    wb: &WorkbookSnapshot,
    max_tasks: usize,
    cache: &EvalCache,
) -> TaskGeneration {
    let mut skipped_formulas = 0usize;
    // kind, question, expected, citation
    let mut drafts: Vec<(String, String, Expected, String)> = Vec::new();
    // Built on the first formula, like the calc engine.
    let mut lineage: Option<Lineage> = None;
    for (si, sheet) in wb.sheets.iter().enumerate() {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                match cell {
                    CellValue::Number(n) if r > 0 && n.is_finite() => {
                        // Data cells (below the first row, treated
                        // as a header) become value-recall tasks.
                        drafts.push((
                            "cell_value".into(),
                            format!(
                                "What is the value of {}? Answer with just the number.",
                                a1(&sheet.name, r, c)
                            ),
                            Expected::Number(*n),
                            a1(&sheet.name, r, c),
                        ));
                    }
                    CellValue::Formula { expr, .. } => {
                        // Built on the first formula: one
                        // recalculation for every task.
                        let calc = cache.get(wb);
                        // An error has no deterministic answer, and
                        // no before value for a what-if.
                        let Ok(before_value) = calc.value(&sheet.name, r, c) else {
                            skipped_formulas += 1;
                            continue;
                        };
                        // A non-finite number or blank text has no
                        // answer to state, but a what-if can still
                        // change it into one.
                        match Expected::of(&before_value) {
                            Some(expected) => drafts.push((
                                "formula_result".into(),
                                format!(
                                    "What does {} calculate to? {}",
                                    a1(&sheet.name, r, c),
                                    expected.ask("")
                                ),
                                expected,
                                a1(&sheet.name, r, c),
                            )),
                            None => skipped_formulas += 1,
                        }
                        // One what-if task per formula: its precedent
                        // cells as Gridcraft resolves them (names and
                        // 3D references included), deduplicated, until
                        // one changes the result. Any typed change
                        // counts, so text and boolean outcomes too.
                        let lineage = lineage.get_or_insert_with(|| Lineage::new(wb));
                        let mut seen = std::collections::BTreeSet::new();
                        let (mut attempts, mut scanned) = (0usize, 0usize);
                        'areas: for (osi, (r0, c0, r1, c1)) in lineage.areas(wb, si, expr) {
                            let Some(osheet) = wb.sheets.get(osi) else {
                                continue;
                            };
                            // Only cells the sheet has: A:A stays cheap.
                            let rows = osheet.cells.iter().enumerate().take(r1.saturating_add(1));
                            for (qr, row) in rows.skip(r0) {
                                let cols = row.iter().enumerate().take(c1.saturating_add(1));
                                for (qc, cell) in cols.skip(c0) {
                                    scanned += 1;
                                    if attempts >= WHAT_IF_ATTEMPTS || scanned > WHAT_IF_SCAN {
                                        break 'areas;
                                    }
                                    let CellValue::Number(orig) = cell else {
                                        continue;
                                    };
                                    if !seen.insert((osi, qr, qc)) {
                                        continue;
                                    }
                                    attempts += 1;
                                    let new_value = orig + 10.0;
                                    let after = calc
                                        .with_override((&osheet.name, qr, qc), new_value)
                                        .and_then(|e| e.value(&sheet.name, r, c));
                                    let Ok(after) = after else {
                                        continue;
                                    };
                                    // Trivial tasks (outcome unchanged)
                                    // teach nothing.
                                    if !after.differs_from(&before_value) {
                                        continue;
                                    }
                                    let Some(outcome) = Expected::of(&after) else {
                                        continue;
                                    };
                                    drafts.push((
                                        "what_if".into(),
                                        format!(
                                            "What if {} becomes {new_value}? What happens to {}? {}",
                                            a1(&osheet.name, qr, qc),
                                            a1(&sheet.name, r, c),
                                            outcome.ask("resulting ")
                                        ),
                                        outcome,
                                        a1(&sheet.name, r, c),
                                    ));
                                    break 'areas;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // Evenly-spaced sampling (not a row-major prefix) when capped,
    // so the pool covers all sheets instead of the top-left corner.
    if drafts.len() > max_tasks && max_tasks > 0 {
        let total = drafts.len();
        drafts = (0..max_tasks)
            .map(|i| drafts[i * total / max_tasks].clone())
            .collect();
    }
    let tasks = drafts
        .into_iter()
        .enumerate()
        .map(|(i, (kind, question, expected, citation))| {
            let (expected, expected_text) = match expected {
                Expected::Number(n) => (n, None),
                Expected::Text(t) => (0.0, Some(t)),
            };
            BenchTask {
                id: format!("t{:04}", i + 1),
                split: split_for(&kind, &question).into(),
                kind,
                question,
                expected,
                citation: Some(citation),
                expected_text,
                want: Vec::new(),
                history: Vec::new(),
                selection: None,
            }
        })
        .collect();
    TaskGeneration {
        tasks,
        skipped_formulas,
    }
}

/// Numbers mentioned in an answer. Tokens that start with a
/// letter (words, range refs like Sheet1!D2) never contribute, so
/// citation text cannot leak digits into the judge. Percent tokens
/// contribute their scaled value ("50%" -> 0.5); a comma is a
/// thousands separator only when every group after the first has
/// exactly three digits, otherwise it is a decimal comma.
pub fn extract_numbers(text: &str) -> Vec<f64> {
    let mut out = Vec::new();
    for tok in text.split(|ch: char| ch.is_whitespace() || matches!(ch, '|' | ';' | ':')) {
        let t = tok.trim_matches(|ch: char| {
            !(ch.is_alphanumeric() || matches!(ch, '.' | '-' | ',' | '%'))
        });
        let first = t.chars().next();
        if !matches!(first, Some(c) if c.is_ascii_digit() || c == '.' || c == '-') {
            continue;
        }
        let (core, percent) = match t.strip_suffix('%') {
            Some(stripped) => (stripped, true),
            None => (t, false),
        };
        let normalized = if core.contains(',') && core.contains('.') {
            // Both separators: the last one is the decimal point
            // (1.200,50 European or 1,200.50 Anglo).
            if core.rfind(',') > core.rfind('.') {
                core.replace('.', "").replace(',', ".")
            } else {
                core.replace(',', "")
            }
        } else if core.contains(',') {
            // Thousands groups: every group after the first is
            // exactly 3 digits, otherwise a single comma is decimal.
            // Multiple non-thousands commas (1,200,50) are ambiguous
            // and skipped, not mangled into 1.20050.
            let groups: Vec<&str> = core.split(',').collect();
            let thousands = groups.len() > 1
                && groups[1..]
                    .iter()
                    .all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit()));
            if thousands {
                core.replace(',', "")
            } else if groups.len() == 2 {
                format!("{}.{}", groups[0], groups[1])
            } else {
                continue;
            }
        } else {
            core.to_string()
        };
        if let Ok(mut v) = normalized.parse::<f64>() {
            if v.is_finite() {
                if percent {
                    v /= 100.0;
                }
                out.push(v);
            }
        }
    }
    out
}

/// Deterministic judge: the **final** number in the answer must
/// match ground truth. Tolerance is 0.1% relative with a small
/// absolute floor (0.005) so rounded answers near zero pass.
pub fn judge_number(answer: &str, expected: f64) -> bool {
    let tol = (expected.abs() * 0.001).max(0.005);
    extract_numbers(answer)
        .last()
        .is_some_and(|v| (v - expected).abs() <= tol)
}

/// Words that may follow the stated value: they introduce a
/// citation, not another value ("big. Citation: Sheet1!C2").
const CITATION_WORDS: [&str; 7] = [
    "citation",
    "citations",
    "source",
    "sources",
    "see",
    "cited",
    "range",
];

/// Whether the answer's last stated value is the expected text, the
/// way [`judge_number`] wants the final number: the text's last
/// whole-word occurrence (ignoring case) may be followed only by
/// citations, punctuation and citation words. So "It becomes big."
/// passes, "It becomes big, not small" and "not TRUE, it is FALSE"
/// (for TRUE) fail, and a one-letter "A" is not found in "a value".
pub fn judge_text(answer: &str, expected: &str) -> bool {
    let answer = answer.to_lowercase();
    let expected = expected.trim().to_lowercase();
    if expected.is_empty() {
        return false;
    }
    let word = |c: char| c.is_alphanumeric() || c == '_';
    // The last whole-word occurrence.
    let mut last = None;
    let mut from = 0;
    while let Some(pos) = answer[from..].find(&expected) {
        let start = from + pos;
        let end = start + expected.len();
        let before_ok = answer[..start].chars().next_back().is_none_or(|c| !word(c));
        let after_ok = answer[end..].chars().next().is_none_or(|c| !word(c));
        if before_ok && after_ok {
            last = Some(end);
        }
        from = start + answer[start..].chars().next().map_or(1, char::len_utf8);
    }
    let Some(end) = last else {
        return false;
    };
    only_citations_follow(&answer[end..])
}

/// Judge for [`BenchTask::want`]: any one wanted string appears in
/// the answer (case-blind) **and** a tool result carried it, so an
/// answer that only echoes the question cannot pass. The question,
/// its condensed stand-alone form and every history question are
/// stripped from the tool results first, because a
/// "no regions matched" result quotes the stand-alone form back.
/// A match must not be flanked by word characters, so "yes"
/// inside "yesterday" and "6" inside "36" do not count; negation
/// ("does NOT read X") still passes — a deterministic judge
/// cannot read it, and task authors should want the thing named,
/// not the stance. Phrases and ranges ("Rate per Hour",
/// "Staff!C4:C44") match anywhere in a sentence, unlike the
/// single stated values [`judge_text`] and [`judge_number`] expect.
pub fn judge_want(
    answer: &str,
    want: &[String],
    tool_results: &str,
    question: &str,
    history: &[gridtalk_index::ChatTurn],
) -> bool {
    let answer = answer.to_lowercase();
    let carried = carried_text(tool_results, question, history);
    want.iter()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .any(|w| contains_word(&answer, &w) && contains_word(&carried, &w))
}

/// The tool results as the judges read them: lower-cased, with the
/// question, its stand-alone form and the history's questions taken
/// out (a "no regions matched" result quotes the question back).
fn carried_text(
    tool_results: &str,
    question: &str,
    history: &[gridtalk_index::ChatTurn],
) -> String {
    let mut carried = tool_results.to_lowercase();
    let standalone = gridtalk_index::standalone_query(question, history);
    for strip in std::iter::once(question)
        .chain(std::iter::once(standalone.as_str()))
        .chain(history.iter().map(|t| t.question.as_str()))
    {
        let strip = strip.trim().to_lowercase();
        if !strip.is_empty() {
            carried = carried.replace(&strip, " ");
        }
    }
    carried
}

/// Whether the tool results held what the task wants, whatever the
/// answer then made of it: one of its `want` strings, its
/// `expected_text`, or its expected number. A task that failed with
/// this true was read right and worded wrong; one that failed with
/// it false never had the answer in front of the model.
pub fn task_carried(task: &BenchTask, tool_results: &str) -> bool {
    let carried = carried_text(tool_results, &task.question, &task.history);
    let wants: Vec<String> = task
        .want
        .iter()
        .map(|w| w.trim().to_lowercase())
        .filter(|w| !w.is_empty())
        .collect();
    if !wants.is_empty() {
        return wants.iter().any(|w| contains_word(&carried, w));
    }
    match task.expected_text.as_deref().map(str::trim) {
        Some(text) if !text.is_empty() => contains_word(&carried, &text.to_lowercase()),
        _ => extract_numbers(&carried)
            .iter()
            .any(|n| (n - task.expected).abs() <= 1e-6 * task.expected.abs().max(1.0)),
    }
}

/// Whether `rest` holds nothing but citations (any token with a `!`,
/// quoted sheet names included), punctuation and citation words.
fn only_citations_follow(rest: &str) -> bool {
    let mut rest = rest.to_string();
    // Collapse quoted sheet prefixes ('My Sheet'!A1 to !A1) before
    // splitting on whitespace, which would cut them apart.
    while let Some(bang) = rest.find("'!") {
        let start = rest[..bang].rfind('\'').unwrap_or(bang);
        rest.replace_range(start..bang + 2, "!");
    }
    rest.split_whitespace().all(|token| {
        let token = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '!');
        token.is_empty() || token.contains('!') || CITATION_WORDS.contains(&token)
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskResult {
    pub id: String,
    pub kind: String,
    pub split: String,
    pub passed: bool,
    /// `None` when the task has no citation to hit; such tasks
    /// stay out of citation precision.
    pub citation_hit: Option<bool>,
    pub latency_ms: f64,
    pub trail_steps: usize,
    /// Whether the tool results held what the task wants
    /// ([`task_carried`]).
    pub carried: bool,
    /// Model turns taken: 1 in the fixed pipeline, one per turn of
    /// the tool loop.
    pub model_turns: usize,
    /// Backend/investigation error, when the task could not run.
    pub error: Option<String>,
}

/// Citation precision is scored on structured citations only: a
/// cited *range* counts when it contains the expected cell. An
/// answer merely restating the question's citation does not.
fn citation_hit(citations: &[String], citation: &str, answer: &str) -> bool {
    // The cited range must also be *used* in the answer text —
    // retrieval alone does not count as citing.
    let Some((esheet, er, ec)) = crate::tools::parse_cell_ref(citation) else {
        return false;
    };
    // The SAME cited range must both appear in the answer and
    // contain the expected cell (combining two different ranges
    // was a false positive).
    citations.iter().any(|cited| {
        contains_bounded(answer, cited)
            && crate::tools::parse_range_uncapped(cited).is_some_and(|(sheet, r0, c0, r1, c1)| {
                sheet_name_eq(&sheet, &esheet) && er >= r0 && er <= r1 && ec >= c0 && ec <= c1
            })
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchReport {
    pub workbook: String,
    pub backend: String,
    pub results: Vec<TaskResult>,
}

impl BenchReport {
    pub fn accuracy(&self, split: &str) -> Option<f64> {
        let rs: Vec<&TaskResult> = self
            .results
            .iter()
            .filter(|r| split == "all" || r.split == split)
            .collect();
        if rs.is_empty() {
            None
        } else {
            Some(rs.iter().filter(|r| r.passed).count() as f64 / rs.len() as f64)
        }
    }
    pub fn citation_precision(&self, split: &str) -> Option<f64> {
        // Only tasks with a citation to hit count, either way.
        let rs: Vec<&TaskResult> = self
            .results
            .iter()
            .filter(|r| (split == "all" || r.split == split) && r.citation_hit.is_some())
            .collect();
        if rs.is_empty() {
            None
        } else {
            Some(
                rs.iter().filter(|r| r.citation_hit == Some(true)).count() as f64 / rs.len() as f64,
            )
        }
    }
    fn mean_latency(&self, split: &str) -> Option<f64> {
        let rs: Vec<&TaskResult> = self
            .results
            .iter()
            .filter(|r| split == "all" || r.split == split)
            .collect();
        if rs.is_empty() {
            None
        } else {
            Some(rs.iter().map(|r| r.latency_ms).sum::<f64>() / rs.len() as f64)
        }
    }
}

/// Run tasks through the investigate harness and score them.
#[allow(dead_code)] // convenience wrapper over run_bench_with
pub fn run_bench(
    tasks: &[BenchTask],
    tools: &Tools,
    backend: &dyn ChatBackend,
    split: &str,
    max_steps: usize,
) -> Result<BenchReport, gridtalk_infer::InferError> {
    run_bench_with(tasks, tools, backend, split, max_steps, None, true)
}

/// Run and score `tasks`. `tool_loop` is the caller's say on whether
/// the model may choose its own tool calls (the backend and the
/// adapter have theirs, see `investigate_run`); `false` scores the
/// fixed pipeline with the same backend, for comparison.
pub fn run_bench_with(
    tasks: &[BenchTask],
    tools: &Tools,
    backend: &dyn ChatBackend,
    split: &str,
    max_steps: usize,
    adapter: Option<&gridtalk_adapter::Adapter>,
    tool_loop: bool,
) -> Result<BenchReport, gridtalk_infer::InferError> {
    let mut results = Vec::new();
    for task in tasks.iter().filter(|t| split == "all" || t.split == split) {
        let started = std::time::Instant::now();
        // A backend error fails that one task; the run still
        // produces a (partial) CARD for the rest.
        let outcome = investigate_run(
            &Ask {
                question: &task.question,
                max_steps,
                adapter,
                sidecar: None,
                history: &task.history,
                selection: task.selection.as_deref(),
            },
            tools,
            backend,
            &mut Run {
                cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                on_token: &mut |_| {},
                on_step: &mut |_| {},
                trace: None,
                tool_loop,
            },
        );
        let (passed, cited, steps, carried, model_turns, error) = match outcome {
            Ok(inv) => {
                let tool_results = inv
                    .trail
                    .iter()
                    .map(|s| s.result.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                (
                    // A `want` list scores answers that are not a
                    // single value; below it, blank text falls back
                    // to the number, as hermit's score() does.
                    if task.want.iter().any(|w| !w.trim().is_empty()) {
                        judge_want(
                            &inv.answer,
                            &task.want,
                            &tool_results,
                            &task.question,
                            &task.history,
                        )
                    } else {
                        match task.expected_text.as_deref() {
                            Some(text) if !text.trim().is_empty() => judge_text(&inv.answer, text),
                            _ => judge_number(&inv.answer, task.expected),
                        }
                    },
                    task.citation
                        .as_deref()
                        .map(|c| citation_hit(&inv.citations, c, &inv.answer)),
                    inv.trail.len(),
                    task_carried(task, &tool_results),
                    inv.model_turns,
                    None,
                )
            }
            Err(e) => (
                false,
                task.citation.as_ref().map(|_| false),
                0,
                false,
                0,
                Some(e.to_string()),
            ),
        };
        results.push(TaskResult {
            id: task.id.clone(),
            kind: task.kind.clone(),
            split: task.split.clone(),
            passed,
            citation_hit: cited,
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            trail_steps: steps,
            carried,
            model_turns,
            error,
        });
    }
    Ok(BenchReport {
        workbook: tools.wb.name.clone(),
        backend: backend.name().to_string(),
        results,
    })
}

/// Render the CARD.md score report for a bench run.
fn esc(field: &str) -> String {
    field.replace('|', "\\|").replace('\n', " ")
}

#[allow(dead_code)] // convenience wrapper over render_card_with_skipped
pub fn render_card(report: &BenchReport, total_tasks: usize) -> String {
    render_card_with_skipped(report, total_tasks, 0)
}

pub fn render_card_with_skipped(
    report: &BenchReport,
    total_tasks: usize,
    skipped_formulas: usize,
) -> String {
    let mut out = String::new();
    out.push_str("# GridTalk bench CARD\n\n");
    out.push_str(&format!("- Workbook: `{}`\n", esc(&report.workbook)));
    out.push_str(&format!("- Backend: `{}`\n", esc(&report.backend)));
    if skipped_formulas > 0 {
        out.push_str(&format!(
            "- Skipped formulas (erroring/non-finite, no ground truth): {skipped_formulas}\n"
        ));
    }
    if report.results.is_empty() {
        out.push_str("\nNo tasks were scored (empty task set for this split).\n");
        return out;
    }
    out.push_str(&format!(
        "- Tasks generated: {total_tasks}; scored here: {}\n\n",
        report.results.len()
    ));
    out.push_str("Ground truth is deterministic (cell values / evaluator\n");
    out.push_str("results); no LLM judges any task.\n\n");
    if report.backend == "echo-local" {
        out.push_str(
            "Note: the echo backend echoes its context instead of answering, \
             so these scores (want tasks especially) are not a model floor.\n\n",
        );
    }
    out.push_str(
        "| Split | Tasks | Accuracy | Citation precision (answer-cited) | Mean latency (ms) |\n",
    );
    out.push_str("|---|---:|---:|---:|---:|\n");
    for split in ["train", "dev", "all"] {
        let n = report
            .results
            .iter()
            .filter(|r| split == "all" || r.split == split)
            .count();
        if n == 0 {
            continue;
        }
        // No task in the split carries a citation: say so instead
        // of printing a precision of 0%.
        let precision = match report.citation_precision(split) {
            Some(p) => format!("{:.1}%", p * 100.0),
            None => "n/a".into(),
        };
        out.push_str(&format!(
            "| {split} | {n} | {:.1}% | {precision} | {:.0} |\n",
            report.accuracy(split).unwrap_or(0.0) * 100.0,
            report.mean_latency(split).unwrap_or(0.0)
        ));
    }
    // By kind of question, in order of first appearance. "Carried"
    // counts the tasks whose tool results held the answer, so a kind
    // that is carried and not passed was read right and worded
    // wrong, and one that is not carried was never read.
    let mut kinds: Vec<&str> = Vec::new();
    for r in &report.results {
        if !kinds.contains(&r.kind.as_str()) {
            kinds.push(&r.kind);
        }
    }
    out.push_str("\n## By kind of question\n\n");
    out.push_str(
        "Carried: the tool results held what the task wants, whatever the answer made of it.\n\n",
    );
    out.push_str("| Kind | Tasks | Passed | Carried | Mean tool calls | Mean model turns |\n");
    out.push_str("|---|---:|---:|---:|---:|---:|\n");
    for kind in kinds {
        let rs: Vec<&TaskResult> = report.results.iter().filter(|r| r.kind == kind).collect();
        let n = rs.len() as f64;
        out.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {:.1} |\n",
            esc(kind),
            rs.len(),
            rs.iter().filter(|r| r.passed).count(),
            rs.iter().filter(|r| r.carried).count(),
            rs.iter().map(|r| r.trail_steps).sum::<usize>() as f64 / n,
            rs.iter().map(|r| r.model_turns).sum::<usize>() as f64 / n,
        ));
    }
    out.push_str("\n## Per-task results\n\n");
    out.push_str(
        "| Task | Kind | Split | Verdict | Carried | Cited | Trail steps | Model turns |\n",
    );
    out.push_str("|---|---|---|---|---|---|---:|---:|\n");
    for r in &report.results {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            esc(&r.id),
            esc(&r.kind),
            esc(&r.split),
            if r.error.is_some() {
                "ERROR"
            } else if r.passed {
                "PASS"
            } else {
                "FAIL"
            },
            if r.carried { "yes" } else { "no" },
            match r.citation_hit {
                Some(true) => "yes",
                Some(false) => "no",
                None => "n/a",
            },
            r.trail_steps,
            r.model_turns
        ));
    }
    out
}

#[cfg(test)]
mod tests {

    #[test]
    fn bench_numbers_round5() {
        assert_eq!(extract_numbers("total 1.200,50"), vec![1200.5]);
        assert_eq!(extract_numbers("total 1,200.50"), vec![1200.5]);
        assert!(extract_numbers("range 1,200,50 here").is_empty());
        assert_eq!(extract_numbers("sci 1.2e3"), vec![1200.0]);
        // A restated question ref is not a citation hit.
        assert!(!citation_hit(
            &["Sheet1!A1:B2".to_string()],
            "Sheet1!D9",
            "Sheet1!D9 is 5"
        ));
    }
    use super::*;
    use gridtalk_graph::{build_graph, parse_delimited, DefinedName, SheetSnapshot};
    use gridtalk_index::HybridIndex;
    use gridtalk_infer::EchoBackend;

    fn fixture() -> (WorkbookSnapshot, gridtalk_graph::Graph, HybridIndex) {
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\nPears,4,5,=B3*C3\n",
            ',',
        );
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty());
        let hybrid = HybridIndex::build(&wb, &g);
        (wb, g, hybrid)
    }

    #[test]
    fn generation_is_deterministic_with_known_ground_truth() {
        let (wb, _, _) = fixture();
        let a = generate_tasks(&wb, 100);
        let b = generate_tasks(&wb, 100);
        assert_eq!(a, b, "generation must be deterministic");
        assert!(a.iter().any(|t| t.split == "dev"), "dev split exists");
        assert!(a.iter().any(|t| t.split == "train"), "train split exists");
        // D2 = B2*C2 = 6 is a formula-result task.
        let f = a
            .iter()
            .find(|t| t.kind == "formula_result" && t.citation.as_deref() == Some("Sheet1!D2"))
            .expect("formula task for D2");
        assert_eq!(f.expected, 6.0);
        // What-if: B2 2 -> 12 makes D2 = 36.
        let w = a
            .iter()
            .find(|t| t.kind == "what_if" && t.citation.as_deref() == Some("Sheet1!D2"))
            .expect("what-if task for D2");
        assert_eq!(w.expected, 36.0);
        // Value recall: B2 itself is 2.
        let v = a
            .iter()
            .find(|t| t.kind == "cell_value" && t.citation.as_deref() == Some("Sheet1!B2"))
            .expect("value task for B2");
        assert_eq!(v.expected, 2.0);
    }

    #[test]
    fn what_if_tasks_follow_names_3d_refs_and_text_outcomes() {
        let month = |name: &str, n: f64| {
            SheetSnapshot::new(
                name,
                vec![
                    vec![CellValue::Text("Billed".into())],
                    vec![CellValue::Number(n)],
                ],
            )
        };
        let f = |e: &str| CellValue::Formula {
            expr: e.into(),
            cached: None,
        };
        let summary = SheetSnapshot::new(
            "Summary",
            vec![
                vec![
                    CellValue::Text("Q1".into()),
                    CellValue::Text("Tax".into()),
                    CellValue::Text("Size".into()),
                    CellValue::Text("Flag".into()),
                ],
                vec![
                    f("SUM(Jan:Mar!A2)"),
                    f("Tax_Rate*100"),
                    f("IF(Jan!A2>5,\"big\",\"small\")"),
                    f("IF(Jan!A2>5,\"big\",\"\")"),
                ],
            ],
        );
        let rates = SheetSnapshot::new(
            "Rates",
            vec![
                vec![CellValue::Text("Rate".into())],
                vec![CellValue::Number(0.15)],
            ],
        );
        let wb = WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![
                month("Jan", 1.0),
                month("Feb", 2.0),
                month("Mar", 3.0),
                summary,
                rates,
            ],
            names: vec![DefinedName {
                name: "Tax_Rate".into(),
                scope: None,
                formula: "Rates!$A$2".into(),
            }],
            warnings: Vec::new(),
            source: None,
        };
        let tasks = generate_tasks(&wb, 100);
        let what_if = |cell: &str| {
            tasks
                .iter()
                .find(|t| t.kind == "what_if" && t.citation.as_deref() == Some(cell))
                .unwrap_or_else(|| panic!("no what-if task for {cell}: {tasks:#?}"))
        };
        // 3D: Jan!A2 1 -> 11 makes the Q1 total 1+2+3 = 6 into 16.
        let q1 = what_if("Summary!A2");
        assert!(q1.question.contains("Jan!A2"), "{}", q1.question);
        assert_eq!((q1.expected, q1.expected_text.as_deref()), (16.0, None));
        // A defined name: Tax_Rate 0.15 -> 10.15.
        let tax = what_if("Summary!B2");
        assert!(tax.question.contains("Rates!A2"), "{}", tax.question);
        assert!((tax.expected - 1015.0).abs() < 1e-9);
        // Text: Jan!A2 -> 11 turns "small" into "big".
        let size = what_if("Summary!C2");
        assert_eq!(size.expected_text.as_deref(), Some("big"));
        assert!(size
            .question
            .ends_with("Answer with just the resulting value."));
        let result = tasks
            .iter()
            .find(|t| t.kind == "formula_result" && t.citation.as_deref() == Some("Summary!C2"))
            .expect("a text formula-result task");
        assert_eq!(result.expected_text.as_deref(), Some("small"));
        // A blank result has no formula-result task, but a what-if
        // can still turn it into an answer.
        assert_eq!(what_if("Summary!D2").expected_text.as_deref(), Some("big"));
        assert!(!tasks
            .iter()
            .any(|t| t.kind == "formula_result" && t.citation.as_deref() == Some("Summary!D2")));
        // Task files written before typed tasks still load.
        let old: BenchTask = serde_json::from_str(
            r#"{"id":"t0001","kind":"cell_value","split":"dev","question":"q","expected":2.0,"citation":"Sheet1!B2"}"#,
        )
        .unwrap();
        assert_eq!(old.expected_text, None);
        assert!(!serde_json::to_string(&old)
            .unwrap()
            .contains("expected_text"));
    }

    #[test]
    fn blank_text_results_make_no_task() {
        assert_eq!(Expected::of(&Value::Text(String::new())), None);
        assert_eq!(Expected::of(&Value::Text("  ".into())), None);
        assert_eq!(
            Expected::of(&Value::Bool(true)),
            Some(Expected::Text("TRUE".into()))
        );
    }

    #[test]
    fn judge_text_wants_the_last_value_stated() {
        assert!(judge_text("It becomes big.", "big"));
        assert!(judge_text("The result: TRUE", "true"));
        assert!(judge_text("The region is North East", "north east"));
        // Citations, punctuation and citation words may follow.
        assert!(judge_text("big (Sheet1!C2).", "big"));
        assert!(judge_text("big. Citation: Sheet1!C2", "big"));
        assert!(judge_text("big, see 'My Sheet'!A1:B2", "big"));
        // A later value wins, as the final number does.
        assert!(!judge_text("It becomes big, not small", "big"));
        assert!(judge_text("It becomes big, not small", "small"));
        assert!(!judge_text("not TRUE, it is FALSE", "TRUE"));
        assert!(!judge_text("big, because B2 exceeds 5", "big"));
        // Short texts are not found inside ordinary words or ranges.
        assert!(!judge_text("It is a value", "A"));
        assert!(judge_text("The grade is A.", "A"));
        assert!(!judge_text("It becomes bigger.", "big"));
        assert!(!judge_text("untrue", "TRUE"));
        assert!(!judge_text("anything", ""));
    }

    #[test]
    fn want_tasks_score_sentences_and_need_tool_support() {
        let want = vec!["Rate per Hour".to_string(), "Staff!C4:C44".to_string()];
        let carried = "depends_on Employee Code, Rate per Hour (Staff!C4:C44)";
        // Any one wanted string, case-blind, anywhere in a sentence.
        assert!(judge_want(
            "It reads the rate per hour from Staff.",
            &want,
            carried,
            "What does this column read?",
            &[]
        ));
        // In the answer but carried by no tool result: the answer
        // may only be echoing the question.
        assert!(!judge_want(
            "It reads the Rate per Hour.",
            &want,
            "depends_on Employee Code",
            "Does it read Rate per Hour?",
            &[]
        ));
        // A tool result that merely echoes the question back does
        // not count as carrying the string either.
        assert!(!judge_want(
            "It reads the Rate per Hour.",
            &want,
            "no regions matched \"Does it read Rate per Hour?\"",
            "Does it read Rate per Hour?",
            &[]
        ));
        assert!(!judge_want("It reads hours.", &want, carried, "q", &[]));
        // Word-flanked substrings do not count.
        let yes = vec!["yes".to_string(), "agrees".to_string()];
        assert!(!judge_want(
            "No: yesterday it disagrees.",
            &yes,
            "yesterday it disagrees",
            "Does it agree?",
            &[]
        ));
        assert!(!judge_want("36", &["6".to_string()], "6 -> 36", "q", &[]));
        // A want word carried only by a history question, quoted
        // back in a no-hit search result, does not pass.
        let history = [gridtalk_index::ChatTurn {
            question: "What does the Labour tab hold?".into(),
            answer: "Timesheets.".into(),
        }];
        assert!(!judge_want(
            "Labour (searched as: What does the Labour tab hold?. and its Hrs?)",
            &["Labour".to_string()],
            "no regions matched \"What does the Labour tab hold?. and its Hrs?\"",
            "and its Hrs?",
            &history
        ));
        // A task file needs neither `expected` nor `citation` when
        // it carries `want`, and its kind is a free string.
        let task: BenchTask = serde_json::from_str(
            r#"{"id":"t0001","kind":"check_agrees","split":"dev","question":"Does the check agree?","want":["yes","agrees"]}"#,
        )
        .unwrap();
        assert_eq!(task.citation, None);
        assert_eq!(task.want.len(), 2);
        // Neither does an expected_text task need `expected`.
        let task: BenchTask = serde_json::from_str(
            r#"{"id":"t0002","kind":"formula_result","split":"dev","question":"q","expected_text":"big"}"#,
        )
        .unwrap();
        assert_eq!(task.expected_text.as_deref(), Some("big"));
        // Generated tasks still round-trip with their citation.
        let old: BenchTask = serde_json::from_str(
            r#"{"id":"t0001","kind":"cell_value","split":"dev","question":"q","expected":2.0,"citation":"Sheet1!B2"}"#,
        )
        .unwrap();
        assert_eq!(old.citation.as_deref(), Some("Sheet1!B2"));
        assert!(old.want.is_empty());
    }

    #[test]
    fn tasks_without_a_citation_leave_citation_precision() {
        let report = BenchReport {
            workbook: "w".into(),
            backend: "b".into(),
            results: vec![
                TaskResult {
                    id: "t1".into(),
                    kind: "cell_value".into(),
                    split: "train".into(),
                    passed: true,
                    citation_hit: Some(true),
                    latency_ms: 0.0,
                    trail_steps: 1,
                    carried: true,
                    model_turns: 1,
                    error: None,
                },
                TaskResult {
                    id: "t2".into(),
                    kind: "sheet_purpose".into(),
                    split: "train".into(),
                    passed: true,
                    citation_hit: None,
                    latency_ms: 0.0,
                    trail_steps: 1,
                    carried: true,
                    model_turns: 1,
                    error: None,
                },
            ],
        };
        // Only the task with a citation counts towards precision.
        assert_eq!(report.citation_precision("all"), Some(1.0));
        assert_eq!(report.accuracy("all"), Some(1.0));
    }

    #[test]
    fn judge_extracts_numbers_without_llm() {
        assert!(judge_number("The answer is 6.", 6.0));
        assert!(judge_number("Total: 1,200 units", 1200.0));
        assert!(judge_number("Total: 1,200.50", 1200.5));
        assert!(!judge_number("The answer is 7.", 6.0));
        assert!(!judge_number("no numbers here", 6.0));
        assert!(judge_number("changes 6 -> 36", 36.0));
        // A correct value buried mid-answer with a wrong final
        // number must NOT pass.
        assert!(!judge_number("It was 6 earlier; the answer is 7", 6.0));
        assert!(judge_number("It was 6 earlier; the answer is 7", 7.0));
    }

    #[test]
    fn citation_hit_requires_answer_citation() {
        let citations = vec!["Sheet1!A1:D4".to_string()];
        assert!(citation_hit(
            &citations,
            "Sheet1!D2",
            "The total is 6 [Sheet1!A1:D4]"
        ));
        // A citation ending a sentence still counts; a prefix of a
        // longer range does not.
        assert!(contains_bounded("See Sheet1!A1:D4.", "Sheet1!A1:D4"));
        assert!(contains_bounded("Sheet1!D2: 6", "Sheet1!D2"));
        assert!(!contains_bounded("Sheet1!A1:B2", "Sheet1!A1"));
        // Retrieval hit without an answer citation does not count.
        assert!(!citation_hit(&citations, "Sheet1!D2", "The total is 6"));
        // Out-of-range citations do not count either.
        assert!(!citation_hit(&citations, "Sheet1!Z9", "see Sheet1!A1:D4"));
    }

    #[test]
    fn want_task_with_selection_and_history_scores_by_its_want() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        // Asked about the selected Total column, as a follow-up:
        // the answer names what it reads in a sentence, and there
        // is no single cell to cite.
        let task: BenchTask = serde_json::from_str(
            r#"{"id":"t0001","kind":"column_reads","split":"train",
                "question":"and what does it read?",
                "want":["Qty"],
                "selection":"Sheet1!D1:D3",
                "history":[{"question":"What is the Total column?","answer":"A product of two inputs."}]}"#,
        )
        .unwrap();
        let report =
            run_bench(std::slice::from_ref(&task), &tools, &EchoBackend, "all", 14).unwrap();
        assert!(report.results[0].passed, "{:?}", report.results[0]);
        assert_eq!(report.results[0].citation_hit, None);
        assert_eq!(report.citation_precision("all"), None);
        // The setting is load-bearing: without the selection and
        // history forwarding, the same task does not pass.
        let mut bare = task.clone();
        bare.selection = None;
        bare.history = Vec::new();
        let report =
            run_bench(std::slice::from_ref(&bare), &tools, &EchoBackend, "all", 14).unwrap();
        assert!(!report.results[0].passed, "{:?}", report.results[0]);
    }

    #[test]
    fn echo_backend_run_scores_and_writes_card() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
            calc: &gridtalk_graph::eval::EvalCache::new(),
        };
        let tasks = generate_tasks(&wb, 100);
        let report = run_bench(&tasks, &tools, &EchoBackend, "all", 8).unwrap();
        assert_eq!(report.results.len(), tasks.len());
        assert!(report.accuracy("all").is_some());
        // The echo backend echoes context but never answers, so
        // its accuracy floor is 0% under the final-number judge —
        // any real model must beat that, and citation precision
        // (structured citations) is scored separately.
        // Echo may luck into a pass when the expected value is the
        // last number in the echoed context; what matters here is
        // that every task is scored and citations are tracked.
        assert!(report.results.iter().any(|r| r.citation_hit == Some(true)));
        let card = render_card(&report, tasks.len());
        assert!(card.contains("train"));
        assert!(card.contains("dev"));
        assert!(card.contains("echo-local"));
        assert!(card.contains("## Per-task results"));
    }
}
