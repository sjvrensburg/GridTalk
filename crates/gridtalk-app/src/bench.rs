//! Phase C (agentic plan): synthetic benchmark generation and
//! scored runs. Tasks are synthesised from a workbook with
//! **deterministic ground truth** — cell values, formula results
//! and what-if outcomes computed by the evaluator, never judged by
//! an LLM. `gridtalk bench run` drives the investigate harness
//! over the tasks and writes a CARD.md score report, for the echo
//! backend (the floor) or the configured local model.

use crate::investigate::investigate_with;
use crate::tools::Tools;
use gridtalk_graph::eval::{EvalCache, Value};
use gridtalk_graph::lineage::Lineage;
use gridtalk_graph::{sheet_name_eq, CellValue, WorkbookSnapshot};
use gridtalk_infer::ChatBackend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchTask {
    pub id: String,
    /// cell_value | formula_result | what_if
    pub kind: String,
    /// train | dev — dev is never optimised against directly.
    pub split: String,
    pub question: String,
    /// Deterministic expected numeric answer (0 when the answer is
    /// text; see `expected_text`).
    pub expected: f64,
    /// Range the answer must cite (e.g. Sheet1!D2).
    pub citation: String,
    /// For a text or boolean answer, the value as the cell shows it
    /// (`TRUE`, `big`); it is judged by [`judge_text`] instead of
    /// `expected`. Absent in task files written before typed tasks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_text: Option<String>,
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
    let mut drafts: Vec<(String, String, Expected, String)> = Vec::new(); // kind, question, expected, citation
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
                        // An error or a non-finite number has no
                        // deterministic answer.
                        let Ok(before_value) = calc.value(&sheet.name, r, c) else {
                            skipped_formulas += 1;
                            continue;
                        };
                        let Some(expected) = Expected::of(&before_value) else {
                            skipped_formulas += 1;
                            continue;
                        };
                        drafts.push((
                            "formula_result".into(),
                            format!(
                                "What does {} calculate to? {}",
                                a1(&sheet.name, r, c),
                                expected.ask("")
                            ),
                            expected,
                            a1(&sheet.name, r, c),
                        ));
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
                            let last_row = r1.min(osheet.cells.len().saturating_sub(1));
                            for qr in r0..=last_row.max(r0) {
                                let Some(row) = osheet.cells.get(qr) else {
                                    break;
                                };
                                let last_col = c1.min(row.len().saturating_sub(1));
                                for qc in c0..=last_col.max(c0) {
                                    scanned += 1;
                                    if attempts >= WHAT_IF_ATTEMPTS || scanned > WHAT_IF_SCAN {
                                        break 'areas;
                                    }
                                    let Some(CellValue::Number(orig)) = row.get(qc) else {
                                        continue;
                                    };
                                    if (osi, qr, qc) == (si, r, c) || !seen.insert((osi, qr, qc)) {
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
                citation,
                expected_text,
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

/// Whether an answer states the expected text: the text appears in
/// it as a whole word or phrase, ignoring case (`TRUE` in "it becomes
/// true.", not inside "untrue").
pub fn judge_text(answer: &str, expected: &str) -> bool {
    let answer = answer.to_lowercase();
    let expected = expected.trim().to_lowercase();
    if expected.is_empty() {
        return false;
    }
    let word = |c: char| c.is_alphanumeric() || c == '_';
    let mut from = 0;
    while let Some(pos) = answer[from..].find(&expected) {
        let start = from + pos;
        let end = start + expected.len();
        let before_ok = answer[..start].chars().next_back().is_none_or(|c| !word(c));
        let after_ok = answer[end..].chars().next().is_none_or(|c| !word(c));
        if before_ok && after_ok {
            return true;
        }
        from = start + answer[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

#[derive(Debug, Clone, PartialEq)]
pub struct TaskResult {
    pub id: String,
    pub kind: String,
    pub split: String,
    pub passed: bool,
    pub citation_hit: bool,
    pub latency_ms: f64,
    pub trail_steps: usize,
    /// Backend/investigation error, when the task could not run.
    pub error: Option<String>,
}

/// Citation precision is scored on structured citations only: a
/// cited *range* counts when it contains the expected cell. An
/// answer merely restating the question's citation does not.
/// `answer` mentions `needle` as a standalone reference, not as a
/// substring of a longer ref (Sheet1!A1 inside Sheet1!A10:B20).
pub(crate) fn contains_bounded(answer: &str, needle: &str) -> bool {
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
        let rs: Vec<&TaskResult> = self
            .results
            .iter()
            .filter(|r| split == "all" || r.split == split)
            .collect();
        if rs.is_empty() {
            None
        } else {
            Some(rs.iter().filter(|r| r.citation_hit).count() as f64 / rs.len() as f64)
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
    run_bench_with(tasks, tools, backend, split, max_steps, None)
}

pub fn run_bench_with(
    tasks: &[BenchTask],
    tools: &Tools,
    backend: &dyn ChatBackend,
    split: &str,
    max_steps: usize,
    adapter: Option<&gridtalk_adapter::Adapter>,
) -> Result<BenchReport, gridtalk_infer::InferError> {
    let mut results = Vec::new();
    for task in tasks.iter().filter(|t| split == "all" || t.split == split) {
        let started = std::time::Instant::now();
        // A backend error fails that one task; the run still
        // produces a (partial) CARD for the rest.
        let (passed, cited, steps, error) =
            match investigate_with(&task.question, tools, backend, max_steps, adapter, None) {
                Ok(inv) => (
                    match &task.expected_text {
                        Some(text) => judge_text(&inv.answer, text),
                        None => judge_number(&inv.answer, task.expected),
                    },
                    citation_hit(&inv.citations, &task.citation, &inv.answer),
                    inv.trail.len(),
                    None,
                ),
                Err(e) => (false, false, 0, Some(e.to_string())),
            };
        results.push(TaskResult {
            id: task.id.clone(),
            kind: task.kind.clone(),
            split: task.split.clone(),
            passed,
            citation_hit: cited,
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            trail_steps: steps,
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
        out.push_str(&format!(
            "| {split} | {n} | {:.1}% | {:.1}% | {:.0} |\n",
            report.accuracy(split).unwrap_or(0.0) * 100.0,
            report.citation_precision(split).unwrap_or(0.0) * 100.0,
            report.mean_latency(split).unwrap_or(0.0)
        ));
    }
    out.push_str("\n## Per-task results\n\n");
    out.push_str("| Task | Kind | Split | Verdict | Cited | Trail steps |\n");
    out.push_str("|---|---|---|---|---|---:|\n");
    for r in &report.results {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
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
            if r.citation_hit { "yes" } else { "no" },
            r.trail_steps
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
    use gridtalk_graph::{build_graph, parse_delimited};
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
            .find(|t| t.kind == "formula_result" && t.citation == "Sheet1!D2")
            .expect("formula task for D2");
        assert_eq!(f.expected, 6.0);
        // What-if: B2 2 -> 12 makes D2 = 36.
        let w = a
            .iter()
            .find(|t| t.kind == "what_if" && t.citation == "Sheet1!D2")
            .expect("what-if task for D2");
        assert_eq!(w.expected, 36.0);
        // Value recall: B2 itself is 2.
        let v = a
            .iter()
            .find(|t| t.kind == "cell_value" && t.citation == "Sheet1!B2")
            .expect("value task for B2");
        assert_eq!(v.expected, 2.0);
    }

    #[test]
    fn what_if_tasks_follow_names_3d_refs_and_text_outcomes() {
        use gridtalk_graph::{DefinedName, SheetSnapshot};
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
                ],
                vec![
                    f("SUM(Jan:Mar!A2)"),
                    f("Tax_Rate*100"),
                    f("IF(Jan!A2>5,\"big\",\"small\")"),
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
                .find(|t| t.kind == "what_if" && t.citation == cell)
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
            .find(|t| t.kind == "formula_result" && t.citation == "Summary!C2")
            .expect("a text formula-result task");
        assert_eq!(result.expected_text.as_deref(), Some("small"));
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
    fn judge_text_wants_the_whole_word() {
        assert!(judge_text("It becomes big.", "big"));
        assert!(judge_text("The result: TRUE", "true"));
        assert!(judge_text("North East region", "north east"));
        assert!(!judge_text("It becomes bigger.", "big"));
        assert!(!judge_text("untrue", "TRUE"));
        assert!(!judge_text("anything", ""));
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
        assert!(report.results.iter().any(|r| r.citation_hit));
        let card = render_card(&report, tasks.len());
        assert!(card.contains("train"));
        assert!(card.contains("dev"));
        assert!(card.contains("echo-local"));
        assert!(card.contains("## Per-task results"));
    }
}
