# Hermit interop (Phase E)

GridTalk's supported optimisation path is native (`gridtalk
optimise`, `docs/adapters.md`): no Python runtime, no LLM judge.
This document covers the *interop* surface for driving GridTalk
from [hermit](https://github.com/Infini-AI-Lab/hermit)'s Python
evolvers instead, and what has actually been validated.

## The contract

Hermit's task contract is two callables (`hermit/task.py`):

```python
run(task: dict, adapter: Path | None) -> RunOutput
score(task: dict, output: RunOutput) -> JudgeResult | dict
```

[`scripts/hermit/gridtalk_task.py`](../scripts/hermit/gridtalk_task.py)
implements both over the text CLI (scraping its printed markers):

- `run` shells out to `gridtalk investigate <workbook>
  [--adapter <name>] <question>` (`GRIDTALK_BIN` /
  `GRIDTALK_WORKBOOK` env vars, or per-row `workbook` /
  `gridtalk_adapter` fields).
- `score` mirrors `gridtalk bench`'s deterministic judge: the
  final number in the answer must match the evaluator ground
  truth. No LLM judges anything, on either side of the boundary.

Data prep converts `gridtalk bench generate` output (a JSON
array) into hermit's JSONL + split-id files; GridTalk's `dev`
split maps to hermit's `val` (the gate):

```bash
gridtalk bench generate book.xlsx --out tasks-array.json
.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts/hermit'); \
    import gridtalk_task; gridtalk_task.prepare_data('tasks-array.json', 'scripts/hermit/data')"
hermit check --config scripts/hermit/run.yaml
```

Note that hermit adapters (template trees evolved by its
Trainer) and GridTalk adapters (ADAPTER.md + knobs) are
*different artefacts*. The bridge is the CLI: a hermit evolver
proposes, GridTalk executes and scores. To evolve GridTalk
adapter content with hermit, point its mutation surface at an
adapter directory and gate with `gridtalk bench run --split dev`
— or simply use `gridtalk optimise`, which implements exactly
that loop natively.

## AGENTS.md export

```bash
gridtalk adapter export <name> [--out AGENTS.md]
```

renders an adapter's rules, skills and knobs as an AGENTS.md so
external harnesses working beside the workbook inherit the same
learned behaviour.

## Validated here (2026-11-02)

- `hermit-harness` installed into the project `.venv`
  (`uv pip install hermit-harness`).
- `gridtalk_task.prepare_data` produced hermit's data files from
  the demo workbook's bench tasks; `TaskData` loaded them with
  no split overlaps.
- The pipeline ran through hermit's own loader
  (`load_pipeline` / `load_callable`): three tasks executed
  end-to-end against the echo backend and were scored by the
  deterministic judge (0.0 — the echo backend never answers,
  which is the correct floor).
- `hermit check --config scripts/hermit/run.yaml --no-probe`
  reports **[ok] on every item** (template `continual-harness`,
  evolver `gepa`, task, sandbox, budgets), with two warnings:
  unset `sandbox.env` (the evolver inherits the trainer's
  environment) and validation rows carrying answers (inherent to
  deterministic ground truth; the native loop treats dev as a
  gate, not a proposer input, for the same reason).

## Not validated

A full `hermit train` run: it needs an evolver model backend
configured (`hermit backends`), which is a user-side choice of
model and credentials. Everything up to that boundary — config,
data, pipeline, judge — is verified above. When the evolver runs
against an external model, workbook context leaves the machine
verbatim (decision 0002); keep sensitive workbooks on the
native loop with a local model.
