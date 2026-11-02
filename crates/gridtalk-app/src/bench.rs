//! Phase C (agentic plan): synthetic benchmark generation and
//! scored runs. Tasks are synthesised from a workbook with
//! **deterministic ground truth** — cell values, formula results
//! and what-if outcomes computed by the evaluator, never judged by
//! an LLM. `gridtalk bench run` drives the investigate harness
//! over the tasks and writes a CARD.md score report, for the echo
//! backend (the floor) or the configured local model.

use crate::investigate::investigate_with;
use crate::tools::Tools;
use gridtalk_graph::eval::{eval_cell_in_workbook, what_if_in_workbook};
use gridtalk_graph::{formula_refs_qualified, CellValue, WorkbookSnapshot};
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
    /// Deterministic expected numeric answer.
    pub expected: f64,
    /// Range the answer must cite (e.g. Sheet1!D2).
    pub citation: String,
}

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
    generate_tasks_ex(wb, max_tasks).tasks
}

pub fn generate_tasks_ex(wb: &WorkbookSnapshot, max_tasks: usize) -> TaskGeneration {
    let mut skipped_formulas = 0usize;
    let mut drafts: Vec<(String, String, f64, String)> = Vec::new(); // kind, question, expected, citation
    for sheet in &wb.sheets {
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
                            *n,
                            a1(&sheet.name, r, c),
                        ));
                    }
                    CellValue::Formula { expr, .. } => {
                        let evaluated = eval_cell_in_workbook(wb, &sheet.name, r, c);
                        if !matches!(evaluated, Ok(v) if v.is_finite()) {
                            skipped_formulas += 1;
                        }
                        if let Ok(v) = evaluated {
                            if v.is_finite() {
                                drafts.push((
                                    "formula_result".into(),
                                    format!(
                                        "What does {} calculate to? Answer with just the number.",
                                        a1(&sheet.name, r, c)
                                    ),
                                    v,
                                    a1(&sheet.name, r, c),
                                ));
                            }
                            // One what-if task per formula: walk
                            // numeric precedents until one produces
                            // a non-trivial, finite outcome.
                            for q in formula_refs_qualified(expr) {
                                let osheet_name =
                                    q.sheet.clone().unwrap_or_else(|| sheet.name.clone());
                                let Some(osheet) = wb
                                    .sheets
                                    .iter()
                                    .find(|s| s.name.eq_ignore_ascii_case(&osheet_name))
                                else {
                                    continue;
                                };
                                if let CellValue::Number(orig) = osheet.get(q.row, q.col) {
                                    let new_value = orig + 10.0;
                                    if let Ok((before, after)) = what_if_in_workbook(
                                        wb,
                                        (&osheet_name, q.row, q.col),
                                        new_value,
                                        (&sheet.name, r, c),
                                    ) {
                                        // Trivial tasks (outcome
                                        // unchanged) teach nothing.
                                        if after.is_finite() && (after - before).abs() > 1e-9 {
                                            drafts.push((
                                                "what_if".into(),
                                                format!(
                                                    "What if {} becomes {new_value}? What happens to {}? Answer with just the resulting number.",
                                                    a1(&osheet_name, q.row, q.col),
                                                    a1(&sheet.name, r, c)
                                                ),
                                                after,
                                                a1(&sheet.name, r, c),
                                            ));
                                            break;
                                        }
                                    }
                                    // Failed/trivial precedent: try
                                    // the next one.
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
        .map(|(i, (kind, question, expected, citation))| BenchTask {
            id: format!("t{:04}", i + 1),
            split: split_for(&kind, &question).into(),
            kind,
            question,
            expected,
            citation,
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
fn contains_bounded(answer: &str, needle: &str) -> bool {
    let boundary = |c: char| !(c.is_alphanumeric() || matches!(c, '!' | ':' | '.' | '$'));
    let mut from = 0;
    while let Some(pos) = answer[from..].find(needle) {
        let start = from + pos;
        let end = start + needle.len();
        let before_ok = answer[..start].chars().next_back().is_none_or(boundary);
        let after_ok = answer[end..].chars().next().is_none_or(boundary);
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
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
                sheet.eq_ignore_ascii_case(&esheet) && er >= r0 && er <= r1 && ec >= c0 && ec <= c1
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
    adapter: Option<&crate::adapter::Adapter>,
) -> Result<BenchReport, gridtalk_infer::InferError> {
    let mut results = Vec::new();
    for task in tasks.iter().filter(|t| split == "all" || t.split == split) {
        let started = std::time::Instant::now();
        // A backend error fails that one task; the run still
        // produces a (partial) CARD for the rest.
        let (passed, cited, steps, error) =
            match investigate_with(&task.question, tools, backend, max_steps, adapter, None) {
                Ok(inv) => (
                    judge_number(&inv.answer, task.expected),
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
