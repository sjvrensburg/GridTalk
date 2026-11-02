"""Hermit task pipeline for GridTalk (Phase E interop).

Hermit's contract (hermit/task.py):

    run(task: dict, adapter: Path | None) -> RunOutput
    score(task: dict, output: RunOutput) -> JudgeResult | dict

`run` drives the GridTalk JSON CLI (`gridtalk investigate`) for
one bench task; `score` is a deterministic judge mirroring
`gridtalk bench`'s final-number rule — no LLM anywhere.

Task rows come from `gridtalk bench generate`. Convert them to
hermit's JSONL + split-id files with `prepare_data` below, or:

    gridtalk bench generate book.xlsx --out tasks-array.json
    python3 -c "import gridtalk_task; gridtalk_task.prepare_data('tasks-array.json', 'data/')"

Environment:
    GRIDTALK_BIN   path to the gridtalk binary (default: `gridtalk`)
    GRIDTALK_WORKBOOK  workbook path, if task rows don't carry one

The `adapter` argument is hermit's own adapter directory; pass a
GridTalk adapter *name* to the underlying CLI via the task row's
`gridtalk_adapter` field (hermit adapters and GridTalk adapters
are different artefacts — see docs/hermit.md).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from hermit.task import JudgeResult, RunOutput


def _gridtalk_bin() -> str:
    return os.environ.get("GRIDTALK_BIN", "gridtalk")


def run(task: dict, adapter: Path | None = None) -> RunOutput:
    workbook = task.get("workbook") or os.environ.get("GRIDTALK_WORKBOOK")
    if not workbook:
        raise ValueError("task row needs 'workbook' (or set GRIDTALK_WORKBOOK)")
    cmd = [_gridtalk_bin(), "investigate", str(workbook)]
    if task.get("gridtalk_adapter"):
        cmd += ["--adapter", str(task["gridtalk_adapter"])]
    cmd.append(str(task["question"]))
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    wall = time.time() - started
    if proc.returncode != 0:
        return RunOutput(
            output="",
            wall_s=wall,
            extra={"error": proc.stderr.strip()[-500:], "returncode": proc.returncode},
        )
    # The CLI prints the answer after this marker.
    marker = "--- answer"
    text = proc.stdout
    answer = text.split(marker, 1)[1] if marker in text else text
    answer = answer.split(") ---\n", 1)[-1].strip()
    return RunOutput(output=answer, wall_s=wall, extra={"stdout_tail": text[-500:]})


_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")


def _final_number(text: str) -> float | None:
    """Last plain number in the answer (mirrors gridtalk bench)."""
    found = None
    for tok in _NUMBER.findall(text):
        if tok.startswith(("Sheet",)):
            continue
        t = tok.rstrip("%")
        # Thousands groups (1,200 / 1,200.50) vs decimal comma.
        if "," in t:
            groups = t.split(",")
            if all(len(g.split(".")[0]) == 3 for g in groups[1:]):
                t = t.replace(",", "")
            else:
                t = t.replace(",", ".", 1).replace(",", "")
        try:
            value = float(t)
        except ValueError:
            continue
        if tok.endswith("%"):
            value /= 100.0
        found = value
    return found


def score(task: dict, output: RunOutput) -> JudgeResult:
    expected = float(task["expected"])
    got = _final_number(str(output.output))
    tol = max(abs(expected) * 0.001, 0.005)
    passed = got is not None and abs(got - expected) <= tol
    citation = str(task.get("citation", ""))
    cited = bool(citation) and citation.split("!")[0] in str(output.output)
    return JudgeResult(
        score=1.0 if passed else 0.0,
        feedback=(
            f"expected {expected}, final number {got}; "
            f"citation {'hit' if cited else 'miss'}"
        ),
        extra={"passed": passed, "citation_hit": cited, "kind": task.get("kind")},
    )


def prepare_data(tasks_array_json: str, out_dir: str) -> None:
    """Convert `gridtalk bench generate` output (a JSON array) to
    hermit's format: tasks.jsonl plus train/val/test id lists.
    GridTalk's `dev` split maps to hermit's `val` (the gate);
    `test` is left empty unless you hold tasks out yourself."""
    rows = json.loads(Path(tasks_array_json).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tasks.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        splits["val" if row.get("split") == "dev" else "train"].append(row["id"])
    for name, ids in splits.items():
        (out / f"{name}.txt").write_text("".join(f"{i}\n" for i in ids))
