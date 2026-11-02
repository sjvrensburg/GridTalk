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
    if not task.get("question"):
        return RunOutput(output="", extra={"error": f"task {task.get('id', '?')}: missing question"})
    workbook = task.get("workbook") or os.environ.get("GRIDTALK_WORKBOOK")
    if not workbook:
        return RunOutput(output="", extra={"error": f"task {task.get('id', '?')}: needs 'workbook' (or GRIDTALK_WORKBOOK)"})
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
    # Anchor to the CLI's real marker line (main.rs prints
    # "\n--- answer (backend) ---"), not the first "--- answer" anywhere.
    text = proc.stdout
    marker = "\n--- answer ("
    answer = text.rsplit(marker, 1)[-1] if marker in text else text
    answer = answer.split(") ---\n", 1)[-1].strip()
    return RunOutput(output=answer, wall_s=wall, extra={"stdout_tail": text[-500:], "stdout": text})


def _extract_numbers(text: str) -> list[float]:
    """Mirror bench.rs `extract_numbers` (shared with the Rust fix):
    whitespace/|;: tokenised, edge-trimmed, letter-leading tokens
    (citations, words, ranges) skipped, leading-dot and scientific
    notation accepted, thousands vs decimal separators disambiguated,
    ranges and malformed groupings skipped."""
    nums: list[float] = []
    for raw in re.split(r"[\s|;:]+", text or ""):
        token = raw.strip("()[]{}\"'")
        if not token:
            continue
        body = token[1:] if token[0] in "+-" else token
        if not body or not (body[0].isdigit() or body[0] == "."):
            continue  # citations like Sheet1!A1 / A1, words, markers
        if "-" in body:
            continue  # range like 10-20 is ambiguous
        percent = token.endswith("%")
        core = body.rstrip("%")
        try:
            if "." in core and "," in core:
                if core.rfind(",") > core.rfind("."):  # 1.200,50 European
                    core = core.replace(".", "").replace(",", ".")
                else:  # 1,200.50
                    core = core.replace(",", "")
            elif "," in core:
                groups = core.split(",")
                if len(groups) == 2 and groups[0] and groups[1]:
                    if len(groups[1]) == 3 and groups[1].isdigit() and len(groups[0]) <= 3:
                        core = groups[0] + groups[1]  # thousands
                    else:
                        core = groups[0] + "." + groups[1]  # decimal comma
                else:
                    continue  # 1,200,50 etc: ambiguous
            value = float(core)  # handles .5 and 1e5 / 1.2E-3
        except ValueError:
            continue
        if percent:
            value /= 100.0
        nums.append(-value if token.startswith("-") else value)
    return nums


def _final_number(text: str) -> float | None:
    """Last number in the answer (mirrors gridtalk bench)."""
    nums = _extract_numbers(str(text))
    return nums[-1] if nums else None


_CELL = re.compile(r"([A-Za-z]{1,3})([0-9]+)$")


def _parse_cell(ref: str):
    """'Sheet1!B12' or 'B12' -> (sheet|None, row, col)."""
    sheet, _, cell = str(ref).rpartition("!")
    sheet = sheet.strip("'") or None
    m = _CELL.match(cell.strip())
    if not m:
        return None
    col = 0
    for ch in m.group(1).upper():
        col = col * 26 + (ord(ch) - 64)
    return sheet, int(m.group(2)), col


def _citation_present(answer: str, task: dict, extra: dict | None) -> bool:
    """Bench citation semantics: the expected citation CELL must fall
    inside a structured cited range (the CLI's `citations:` line or a
    range used in the answer), and a cited range must appear in the
    answer. A sheet-name substring or a restated question ref is not
    a citation."""
    expected = _parse_cell(str(task.get("citation", "")))
    if expected is None:
        return False
    esheet, erow, ecol = expected
    cited: list[str] = []
    for line in str((extra or {}).get("stdout", "")).splitlines():
        if line.strip().startswith("citations:"):
            cited += [c.strip() for c in line.split(":", 1)[1].split(",")]
    if not any(c and c in (answer or "") for c in cited):
        return False
    for c in cited:
        rng = c.rpartition("!")[2] if "!" in c else c
        sheet = c.rpartition("!")[0].strip("'") or None if "!" in c else None
        lo_s, _, hi_s = rng.partition(":")
        lo, hi = _parse_cell(lo_s), _parse_cell(hi_s or lo_s)
        if lo is None or hi is None:
            continue
        if sheet is not None and esheet is not None and sheet != esheet:
            continue
        if lo[1] <= erow <= hi[1] and min(lo[2], hi[2]) <= ecol <= max(lo[2], hi[2]):
            return True
    return False


def score(task: dict, output: RunOutput) -> JudgeResult:
    raw = task.get("expected")
    try:
        expected = float(raw)
    except (TypeError, ValueError):
        try:
            expected = _final_number(str(raw))
        except Exception:
            expected = None
    if expected is None:
        return JudgeResult(
            score=0.0,
            feedback=f"task {task.get('id', '?')}: unparseable expected value {raw!r}",
            extra={"passed": False, "citation_hit": False, "kind": task.get("kind")},
        )
    got = _final_number(str(output.output))
    tol = max(abs(expected) * 0.001, 0.005)
    passed = got is not None and abs(got - expected) <= tol
    extra = getattr(output, "extra", None) or {}
    cited = _citation_present(str(output.output), task, extra) if task.get("citation") else False
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
    if not isinstance(rows, list):
        raise ValueError("bench output must be a JSON array of task rows")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("id") or "question" not in row or "expected" not in row:
            raise ValueError("each bench row needs id/question/expected")
        if row["id"] in seen:
            raise ValueError(f"duplicate bench task id: {row['id']}")
        seen.add(row["id"])
        if row.get("split", "train") not in ("train", "dev", "test"):
            raise ValueError(f"unknown split {row.get('split')!r} for task {row['id']}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "tasks.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    splits = {"train": [], "val": [], "test": []}
    for row in rows:
        split = row.get("split", "train")
        splits["val" if split == "dev" else "test" if split == "test" else "train"].append(row["id"])
    for name, ids in splits.items():
        (out / f"{name}.txt").write_text("".join(f"{i}\n" for i in ids))
