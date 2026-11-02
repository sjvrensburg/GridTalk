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

Bench runs execute sequentially: task order, trails and CARD
rows are deterministic, and there is no parallelism to tune.

Scoring is strict: the **final** number in the answer must match
ground truth (0.1% relative tolerance, 0.005 absolute floor), and
citation precision counts only structured citations whose range
contains the expected cell — restating the question's citation
does not count. On the demo workbook the echo backend therefore
scores 0% accuracy (it echoes context but never answers), while
its citation precision is high because the harness cites the
right regions. That 0% is the floor a real model must beat, and
the formula-result/what-if trail gaps are what Phase D adapters
target.

Tasks come from a file only when you pass `--tasks`; the split is
assigned by a stable hash of the task (not position), generation
samples evenly across the workbook when `--max` caps the pool,
and formula cells that error have no ground truth — they are
counted as skipped in the CARD, never silently tasked.

## CARD.md

The run writes a CARD with per-split accuracy, citation
precision, mean latency, and a per-task PASS/FAIL table — the
record Phase D's optimise loop compares before accepting an
adapter change.
