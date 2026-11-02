# Benchmarks (Phase C)

GridTalk's optimisation loop starts from tasks whose answers are
known **deterministically** — cell values, formula results and
what-if outcomes computed by the evaluator. No LLM judges a task.

## Generate tasks

```bash
gridtalk bench generate <workbook> [--out tasks.json] [--max 100]
```

Task kinds:

| Kind | Question shape | Ground truth |
|---|---|---|
| `cell_value` | "What is the value of Sheet1!B2?" | the stored number |
| `formula_result` | "What does Sheet1!D2 calculate to?" | `eval_cell` result |
| `what_if` | "What happens to Sheet1!D2 if Sheet1!B2 becomes 12?" | `what_if` re-evaluation |

Every third task is labelled `dev`; the rest are `train`. The dev
split is never optimised against directly (Phase D gates adapter
changes on it). Generation is deterministic: the same workbook
always produces the same `tasks.json`.

## Run a bench

```bash
gridtalk bench run <workbook> [--tasks tasks.json] \
    [--split train|dev|all] [--card CARD.md]
```

Each task runs through the investigate harness with the
configured backend (the echo backend when no model is
configured — that score is the floor a real model must beat).
Scoring is a number check against ground truth plus citation
precision (did the answer cite the expected range?).

On the demo workbook the echo floor is 50% accuracy / 100%
citation precision; the failing tasks are the formula-result and
what-if questions, which is the weakness Phase D adapters target.

## CARD.md

The run writes a CARD with per-split accuracy, citation
precision, mean latency, and a per-task PASS/FAIL table — the
record Phase D's optimise loop compares before accepting an
adapter change.
