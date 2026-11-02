//! Phase D: the native optimise loop — propose → eval →
//! commit/reject, no Python runtime and no LLM judge.
//!
//! The proposer is deterministic and failure-driven: it clusters
//! the failed bench tasks by kind and turns each cluster into one
//! candidate mutation (an ADAPTER.md rule or a knob change). A
//! candidate is committed as the next adapter version only when
//! it improves the **train** split without regressing **dev**
//! (dev is never proposed against). Every round — accepted or
//! rejected — is appended to the adapter's CARD.md with scores.
//! The proposer is deliberately replaceable: Phase E can drive
//! the same evaluate/gate/commit steps from hermit's evolvers.

use crate::adapter::{append_card, commit_version, Adapter};
use crate::bench::{run_bench_with, BenchTask, TaskResult};
use crate::tools::Tools;
use gridtalk_infer::ChatBackend;

#[derive(Debug, Clone)]
pub struct Candidate {
    pub label: String,
    pub adapter: Adapter,
}

fn failing(results: &[TaskResult], kind: &str) -> usize {
    results
        .iter()
        .filter(|r| r.kind == kind && !r.passed)
        .count()
}

/// Failure clusters → candidate mutations, in a fixed order so
/// runs are reproducible against the same failure set.
pub fn propose(current: &Adapter, results: &[TaskResult]) -> Vec<Candidate> {
    let mut out: Vec<Candidate> = Vec::new();
    fn push_rule(out: &mut Vec<Candidate>, current: &Adapter, label: &str, rule: &str, why: &str) {
        if !current.rules.contains(rule) {
            let mut a = current.clone();
            if !a.rules.ends_with('\n') && !a.rules.is_empty() {
                a.rules.push('\n');
            }
            a.rules.push_str(rule);
            a.rules.push('\n');
            out.push(Candidate {
                label: format!("{label} ({why})"),
                adapter: a,
            });
        }
    }
    let formula_fail = failing(results, "formula_result");
    if formula_fail > 0 {
        // Rule + ref_context knob: ref-style questions often miss
        // retrieval entirely, so the planner must include the
        // named cell directly for the rule to have anything to
        // answer from.
        let mut a = current.clone();
        if !a.rules.ends_with('\n') && !a.rules.is_empty() {
            a.rules.push('\n');
        }
        a.rules.push_str("- When asked what a formula cell calculates to, read the formula's computed value from the tool results and answer with just the final number.\n");
        a.ref_context = true;
        out.push(Candidate {
            label: format!(
                "rule:formula-number+ref_context ({formula_fail} formula_result failures)"
            ),
            adapter: a,
        });
    }
    let whatif_fail = failing(results, "what_if");
    if whatif_fail > 0 {
        push_rule(
            &mut out,
            current,
            "rule:what-if",
            "- For what-if questions, use the what_if tool result and answer with just the resulting number.",
            &format!("{whatif_fail} what_if failures"),
        );
    }
    let value_fail = failing(results, "cell_value");
    if value_fail > 0 {
        push_rule(
            &mut out,
            current,
            "rule:cell-value",
            "- When asked for a cell value, answer with just the value and its range citation.",
            &format!("{value_fail} cell_value failures"),
        );
    }
    let citation_misses = results.iter().filter(|r| !r.citation_hit).count();
    if citation_misses > 0 && current.top_k < 8 {
        let mut a = current.clone();
        a.top_k = (a.top_k + 2).min(8);
        out.push(Candidate {
            label: format!("knob:top_k={} ({citation_misses} citation misses)", a.top_k),
            adapter: a,
        });
    }
    let no_trail = results.iter().filter(|r| r.trail_steps <= 1).count();
    if no_trail > 0 && current.max_steps < 16 {
        let mut a = current.clone();
        a.max_steps = 16;
        out.push(Candidate {
            label: format!("knob:max_steps=16 ({no_trail} tasks with no tool trail)"),
            adapter: a,
        });
    }
    out
}

#[allow(dead_code)] // `round`/`tried` are read by tests and CARD writers
pub struct RoundOutcome {
    pub round: usize,
    pub baseline_train: f64,
    pub baseline_dev: f64,
    pub accepted: Option<(String, f64, f64)>, // label, train, dev
    pub tried: Vec<(String, f64, f64, bool)>, // label, train, dev, accepted
}

/// One optimise round: score the current adapter (train+dev),
/// try each candidate on train, gate the best on dev, and commit
/// or record rejection in CARD.md.
pub fn optimise_round(
    root: &std::path::Path,
    workbook_tasks: &[BenchTask],
    tools: &Tools,
    backend: &dyn ChatBackend,
    current: &Adapter,
    round: usize,
) -> Result<RoundOutcome, String> {
    let score =
        |adapter: Option<&Adapter>, split: &str| -> Result<(f64, Vec<TaskResult>), String> {
            let report = run_bench_with(
                workbook_tasks,
                tools,
                backend,
                split,
                adapter.map(|a| a.max_steps).unwrap_or(8),
                adapter,
            )
            .map_err(|e| e.to_string())?;
            Ok((report.accuracy(split).unwrap_or(0.0), report.results))
        };

    let (baseline_train, baseline_results) = score(Some(current), "train")?;
    let (baseline_dev, _) = score(Some(current), "dev")?;
    let candidates = propose(current, &baseline_results);

    let mut tried = Vec::new();
    let mut best: Option<(Candidate, f64)> = None;
    for cand in candidates {
        let (train_acc, _) = score(Some(&cand.adapter), "train")?;
        let better = train_acc > baseline_train + 1e-9;
        if better && best.as_ref().is_none_or(|(_, acc)| train_acc > *acc) {
            best = Some((cand.clone(), train_acc));
        }
        tried.push((cand.label, train_acc, f64::NAN, false));
    }

    let mut accepted = None;
    if let Some((cand, train_acc)) = best {
        let (dev_acc, _) = score(Some(&cand.adapter), "dev")?;
        // Gate: dev must not regress. Dev is a gate, never a
        // proposer input.
        if dev_acc >= baseline_dev - 1e-9 {
            let version = commit_version(root, &cand.adapter)?;
            let mut entry = format!(
                "\n## Round {round}: ACCEPTED {} as v{version}\n\n- train: {:.1}% -> {:.1}%\n- dev: {:.1}% -> {:.1}%\n",
                cand.label,
                baseline_train * 100.0,
                train_acc * 100.0,
                baseline_dev * 100.0,
                dev_acc * 100.0
            );
            for (label, t, _, _) in &tried {
                entry.push_str(&format!("- tried {label}: train {:.1}%\n", t * 100.0));
            }
            append_card(root, &current.name, &entry)?;
            if let Some(row) = tried.iter_mut().find(|(l, _, _, _)| *l == cand.label) {
                row.2 = dev_acc;
                row.3 = true;
            }
            accepted = Some((cand.label.clone(), train_acc, dev_acc));
        } else {
            append_card(
                root,
                &current.name,
                &format!(
                    "\n## Round {round}: REJECTED {} (dev regressed)\n\n- train: {:.1}% -> {:.1}%\n- dev: {:.1}% -> {:.1}% (gate: no regression)\n",
                    cand.label,
                    baseline_train * 100.0,
                    train_acc * 100.0,
                    baseline_dev * 100.0,
                    dev_acc * 100.0
                ),
            )?;
            if let Some(row) = tried.iter_mut().find(|(l, _, _, _)| *l == cand.label) {
                row.2 = dev_acc;
            }
        }
    } else {
        append_card(
            root,
            &current.name,
            &format!(
                "\n## Round {round}: no candidate improved train\n\n- train: {:.1}%\n- dev: {:.1}%\n",
                baseline_train * 100.0,
                baseline_dev * 100.0
            ),
        )?;
    }

    Ok(RoundOutcome {
        round,
        baseline_train,
        baseline_dev,
        accepted,
        tried,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::load_adapter;
    use gridtalk_graph::{build_graph, parse_delimited, WorkbookSnapshot};
    use gridtalk_index::HybridIndex;
    use gridtalk_infer::{ChatRequest, InferError};

    /// A stand-in "small model": it can extract the computed value
    /// from tool results, but only follows the instruction to do
    /// so when the adapter's rule is in its system prompt. This is
    /// the acceptance shape: a known weakness an ADAPTER.md rule
    /// measurably fixes.
    struct RuleFollowingBackend;
    impl ChatBackend for RuleFollowingBackend {
        fn name(&self) -> &str {
            "rule-following-test"
        }
        fn stream(
            &self,
            req: &ChatRequest,
            _cancel: &gridtalk_infer::CancelToken,
            on_token: &mut dyn FnMut(&str),
        ) -> Result<(), InferError> {
            let answer = if req.system.contains("just the final number") {
                // Find the asked cell's formula line
                // ("Sheet1!D2 = B2*C2 -> 6") and answer its value.
                let target = req
                    .user
                    .split_whitespace()
                    .find(|t| t.contains('!') && t.chars().any(|c| c.is_ascii_digit()))
                    .unwrap_or("")
                    .trim_matches(|c: char| !(c.is_alphanumeric() || c == '!'));
                let cell_part = target.rsplit('!').next().unwrap_or(target);
                req.user
                    .lines()
                    .find(|l| {
                        !cell_part.is_empty()
                            && l.contains(&format!("{cell_part} ="))
                            && l.contains("-> ")
                    })
                    .and_then(|l| l.rsplit("-> ").next())
                    .and_then(|tail| {
                        let num: String = tail
                            .trim_start()
                            .chars()
                            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
                            .collect();
                        if num.is_empty() {
                            None
                        } else {
                            Some(num)
                        }
                    })
                    .unwrap_or_else(|| "unknown".into())
            } else {
                "I could not determine that.".to_string()
            };
            on_token(&answer);
            Ok(())
        }
    }

    fn fixture() -> (WorkbookSnapshot, gridtalk_graph::Graph, HybridIndex) {
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\nPears,4,5,=B3*C3\nGrapes,7,2,=B4*C4\nPlums,9,9,=B5*C5\nFigs,3,3,=B6*C6\nKiwi,5,5,=B7*C7\n",
            ',',
        );
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty());
        let hybrid = HybridIndex::build(&wb, &g);
        (wb, g, hybrid)
    }

    #[test]
    fn optimise_cycle_commits_rule_and_improves_dev() {
        let (wb, g, hybrid) = fixture();
        let tools = Tools {
            wb: &wb,
            graph: &g,
            hybrid: &hybrid,
        };
        let tasks = crate::bench::generate_tasks(&wb, 100);
        assert!(tasks
            .iter()
            .any(|t| t.kind == "formula_result" && t.split == "dev"));
        let root = std::env::temp_dir().join(format!("gt-opt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let current = Adapter::empty("demo");
        let outcome =
            optimise_round(&root, &tasks, &tools, &RuleFollowingBackend, &current, 1).unwrap();
        let (label, train, dev) = outcome.accepted.unwrap_or_else(|| {
            panic!(
                "no acceptance: baseline train={} dev={} tried={:?}",
                outcome.baseline_train, outcome.baseline_dev, outcome.tried
            )
        });
        assert!(label.contains("formula-number"), "{label}");
        assert!(train > outcome.baseline_train);
        assert!(dev >= outcome.baseline_dev);
        assert!(
            dev > outcome.baseline_dev || outcome.baseline_dev == 0.0 && dev > 0.0,
            "dev must measurably improve: {} -> {}",
            outcome.baseline_dev,
            dev
        );
        let committed = load_adapter(&root, "demo");
        assert_eq!(committed.version, 1);
        assert!(committed.rules.contains("just the final number"));
        let card = std::fs::read_to_string(root.join("demo/CARD.md")).unwrap();
        assert!(card.contains("ACCEPTED"));
        let _ = std::fs::remove_dir_all(&root);
    }
}
