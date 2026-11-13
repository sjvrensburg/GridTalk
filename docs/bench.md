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
| `formula_result` | "What does Sheet1!D2 calculate to?" | the computed result |
| `what_if` | "What if Sheet1!B2 becomes 12? What happens to Sheet1!D2?" | the re-evaluated result |

A what-if task tries the formula's precedent cells as Gridcraft
resolves them (defined names and 3D references such as
`Jan:Mar!A2` included), up to 64 numeric precedents among the
first 4096 cells of its precedent areas, until one
changes the result. Any typed change counts. Results that are text
or TRUE/FALSE make tasks too: they carry `expected_text` (the value
as the cell shows it) and ask for "just the value"; numeric tasks
carry `expected` and ask for "just the number". Blank text
results have no answer to state and make no task. Task files written
before `expected_text` load unchanged.

Every third task is labelled `dev`; the rest are `train`. The dev
split is never optimised against directly (Phase D gates adapter
changes on it). Generation is deterministic: the same workbook
always produces the same `tasks.json`.

## Run a bench

```bash
gridtalk bench run <workbook> [--tasks tasks.json] \
    [--split train|dev|all] [--card CARD.md] [--adapter <name>] \
    [--no-tool-loop]
```

Each task runs through the investigate harness with the
configured backend (the echo backend when no model is
configured — that score is the floor a real model must beat).
With a backend that takes tool turns the model chooses its own
tool calls (decision 0006); `--no-tool-loop` scores the fixed
pipeline with the same backend, which is how to tell what the loop
is worth on a set of tasks.
Scoring is a number check against ground truth (or, for a task
with `expected_text`, a check that the text is the last value the
answer states: its last whole-word occurrence, ignoring case, may be
followed only by citations, punctuation and words such as
"Citation:"; so "It becomes big." passes and "big, not small"
fails) plus citation precision (did the answer
cite the expected range?).

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

## Hand-written tasks

A task file passed with `--tasks` may also ask the questions
generation cannot — "what is this sheet for?", "what does this
column read?" — whose right answer names several things in a
sentence. Such a task carries a `want` list instead of a single
value:

```json
{"id": "t0001", "kind": "column_reads", "split": "dev",
 "question": "What does the Total column read?",
 "want": ["Rate per Hour", "Staff!C4:C44"]}
```

Any one `want` string in the answer marks it right (matched
case-blind, and not inside a longer word or number: `yes` does not
match "yesterday", `6` does not match "36"), with one extra rule: a
tool result in the trail must have carried the string too, so an
answer that only echoes the question does not pass. Tool *results*
count, not their arguments; the question, its condensed follow-up
form and the history questions are stripped from them first,
because a no-match search result quotes them back. The judge is
deterministic and cannot read a stance: "it does NOT read Rate per
Hour" still passes `want: ["Rate per Hour"]`, so want the thing a
right answer names, not a yes/no. `kind` is a free string for such
tasks.
A task with no ground truth at all (no non-blank `want` or
`expected_text`, and no `expected` for a hand-written kind) is
judged against `expected`'s default of 0; `bench run` warns about
each such task. Under the echo backend (no model configured) the
answer is the prompt itself, so want tasks pass whenever retrieval
found the string; the CARD says so, and those scores are not a
model floor.
A task scored by `want` or `expected_text` needs no `expected`,
and a task with no cell to cite ("How many sheets are there?")
needs no `citation` — it simply stays out of citation precision
(the CARD shows `n/a` where no task in a split carries one).

A task may also carry the setting it was asked in: `history`, a
list of earlier `{"question", "answer"}` turns a follow-up is
condensed against before searching, and `selection`, the range
selected when asking ("what feeds this?"), which is read ahead
of the search results. `gridtalk investigate` takes the same
setting as `--history-file <turns.json>` and
`--selection <Sheet!A1:B2>`; hermit's `run` passes both through.

Tasks come from a file only when you pass `--tasks`; the split is
assigned by a stable hash of the task (not position), generation
samples evenly across the workbook when `--max` caps the pool,
and formula cells that error or give a non-finite number have no
ground truth — they are
counted as skipped in the CARD, never silently tasked.

## CARD.md

The run writes a CARD with per-split accuracy, citation
precision, mean latency, a table by kind of question and a
per-task PASS/FAIL table — the record Phase D's optimise loop
compares before accepting an adapter change.

The table by kind gives, for each `kind`, how many tasks passed,
how many were *carried*, and the mean tool calls and model turns.
A task is carried when the tool results held what it wants (a
`want` string, its `expected_text`, or its expected number),
whatever the answer made of it. The two counts separate two
failures: a kind that is carried and not passed was read right and
worded wrong, and a kind that is not carried never had the answer
in front of the model, so the calls are what to fix. For a task
scored by a number the count is generous: any number in the tool
text that equals the expected one counts, a row label or a score
included, so read it with care when the answer is a small integer.
