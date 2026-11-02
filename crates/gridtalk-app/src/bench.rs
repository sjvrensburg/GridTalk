//! Phase C (agentic plan): synthetic benchmark generation and
//! scored runs. Tasks are synthesised from a workbook with
//! **deterministic ground truth** — cell values, formula results
//! and what-if outcomes computed by the evaluator, never judged by
//! an LLM. `gridtalk bench run` drives the investigate harness
//! over the tasks and writes a CARD.md score report, for the echo
//! backend (the floor) or the configured local model.

use crate::investigate::{investigate, Investigation};
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
    format!("{sheet}!{}{}", gridtalk_graph::col_name(col), row + 1)
}

/// Synthesise benchmark tasks from a workbook. Deterministic: the
/// same workbook always yields the same tasks, ids and splits.
pub fn generate_tasks(wb: &WorkbookSnapshot, max_tasks: usize) -> Vec<BenchTask> {
    let mut drafts: Vec<(String, String, f64, String)> = Vec::new(); // kind, question, expected, citation
    for sheet in &wb.sheets {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                match cell {
                    CellValue::Number(n) if r > 0 => {
                        // Data cells (below a header row) become
                        // value-recall tasks.
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
                        if let Ok(v) = eval_cell_in_workbook(wb, &sheet.name, r, c) {
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
                            // One what-if task per formula: its first
                            // numeric precedent, shifted by +10.
                            for q in formula_refs_qualified(expr) {
                                let osheet_name =
                                    q.sheet.clone().unwrap_or_else(|| sheet.name.clone());
                                let Some(osheet) = wb.sheets.iter().find(|s| s.name == osheet_name)
                                else {
                                    continue;
                                };
                                if let CellValue::Number(orig) = osheet.get(q.row, q.col) {
                                    let new_value = orig + 10.0;
                                    if let Ok((_, after)) = what_if_in_workbook(
                                        wb,
                                        (&osheet_name, q.row, q.col),
                                        new_value,
                                        (&sheet.name, r, c),
                                    ) {
                                        if after.is_finite() {
                                            drafts.push((
                                                "what_if".into(),
                                                format!(
                                                    "What happens to {} if {} becomes {new_value}? Answer with just the resulting number.",
                                                    a1(&sheet.name, r, c),
                                                    a1(&osheet_name, q.row, q.col)
                                                ),
                                                after,
                                                a1(&sheet.name, r, c),
                                            ));
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    drafts.truncate(max_tasks);
    drafts
        .into_iter()
        .enumerate()
        .map(|(i, (kind, question, expected, citation))| BenchTask {
            id: format!("t{:04}", i + 1),
            // Every third task is a dev task; the rest are train.
            split: if i % 3 == 2 { "dev" } else { "train" }.into(),
            kind,
            question,
            expected,
            citation,
        })
        .collect()
}

/// Numbers mentioned in an answer (commas stripped, trailing
/// punctuation ignored). Range refs like Sheet1!D2 do not parse.
pub fn extract_numbers(text: &str) -> Vec<f64> {
    text.split(|ch: char| ch.is_whitespace() || matches!(ch, '|' | ';' | ':'))
        .filter_map(|tok| {
            let t = tok.trim_matches(|ch: char| {
                !(ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == ',')
            });
            if t.is_empty() || t == "-" || t == "." {
                return None;
            }
            if let Ok(v) = t.parse::<f64>() {
                return v.is_finite().then_some(v);
            }
            // Thousands separators ("1,200") only when the token is
            // otherwise purely numeric — never mine digits out of
            // words or range refs like Sheet1!D2.
            if t.chars()
                .all(|ch| ch.is_ascii_digit() || matches!(ch, ',' | '.' | '%'))
            {
                return t
                    .replace(',', "")
                    .trim_end_matches('%')
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite());
            }
            None
        })
        .collect()
}

/// Deterministic judge: the expected value must appear in the
/// answer (within tolerance). No LLM anywhere in scoring.
pub fn judge_number(answer: &str, expected: f64) -> bool {
    let tol = (expected.abs() * 0.001).max(1e-6);
    extract_numbers(answer)
        .iter()
        .any(|v| (v - expected).abs() <= tol)
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
pub fn run_bench(
    tasks: &[BenchTask],
    tools: &Tools,
    backend: &dyn ChatBackend,
    split: &str,
    max_steps: usize,
) -> Result<BenchReport, gridtalk_infer::InferError> {
    let mut results = Vec::new();
    for task in tasks.iter().filter(|t| split == "all" || t.split == split) {
        let started = std::time::Instant::now();
        let inv: Investigation = investigate(&task.question, tools, backend, max_steps)?;
        let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
        results.push(TaskResult {
            id: task.id.clone(),
            kind: task.kind.clone(),
            split: task.split.clone(),
            passed: judge_number(&inv.answer, task.expected),
            citation_hit: inv.citations.iter().any(|c| c == &task.citation)
                || inv.answer.contains(&task.citation),
            latency_ms,
            trail_steps: inv.trail.len(),
        });
    }
    Ok(BenchReport {
        workbook: tools.wb.name.clone(),
        backend: backend.name().to_string(),
        results,
    })
}

/// Render the CARD.md score report for a bench run.
pub fn render_card(report: &BenchReport, total_tasks: usize) -> String {
    let mut out = String::new();
    out.push_str("# GridTalk bench CARD\n\n");
    out.push_str(&format!("- Workbook: `{}`\n", report.workbook));
    out.push_str(&format!("- Backend: `{}`\n", report.backend));
    out.push_str(&format!(
        "- Tasks generated: {total_tasks}; scored here: {}\n\n",
        report.results.len()
    ));
    out.push_str("Ground truth is deterministic (cell values / evaluator\n");
    out.push_str("results); no LLM judges any task.\n\n");
    out.push_str("| Split | Tasks | Accuracy | Citation precision | Mean latency (ms) |\n");
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
            r.id,
            r.kind,
            r.split,
            if r.passed { "PASS" } else { "FAIL" },
            if r.citation_hit { "yes" } else { "no" },
            r.trail_steps
        ));
    }
    out
}

#[cfg(test)]
mod tests {
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
        assert!(!judge_number("The answer is 7.", 6.0));
        assert!(!judge_number("no numbers here", 6.0));
        assert!(judge_number("changes 6 -> 36", 36.0));
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
        // The echo floor echoes the cited context (including cell
        // values), so the floor accuracy is above zero — that is
        // exactly the baseline a real model must beat.
        assert!(
            report.results.iter().any(|r| r.passed),
            "echo floor must pass some tasks: {:?}",
            report.results
        );
        let card = render_card(&report, tasks.len());
        assert!(card.contains("train"));
        assert!(card.contains("dev"));
        assert!(card.contains("echo-local"));
        assert!(card.contains("PASS"));
    }
}
