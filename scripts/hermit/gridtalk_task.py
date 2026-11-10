"""Hermit task pipeline for GridTalk (Phase E interop).

Hermit's contract (hermit/task.py):

    run(task: dict, adapter: Path | None) -> RunOutput
    score(task: dict, output: RunOutput) -> JudgeResult | dict

`run` drives the GridTalk text CLI (`gridtalk investigate`) for
one bench task; `score` is a deterministic judge mirroring
`gridtalk bench`'s final-number rule — no LLM anywhere.

Task rows come from `gridtalk bench generate`. Convert them to
hermit's JSONL + split-id files with `prepare_data` below, or:

    gridtalk bench generate book.xlsx --out tasks-array.json
    .venv/bin/python -c "import sys; sys.path.insert(0, 'scripts/hermit'); import gridtalk_task; gridtalk_task.prepare_data('tasks-array.json', 'data/')"

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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RunOutput(
            output="",
            wall_s=time.time() - started,
            extra={"error": f"could not run gridtalk CLI: {exc}"},
        )
    wall = time.time() - started
    if proc.returncode != 0:
        return RunOutput(
            output="",
            wall_s=wall,
            extra={"error": proc.stderr.strip()[-500:], "returncode": proc.returncode},
        )
    # Anchor to the first answer marker AFTER the CLI citations
    # line: trail text (which quotes the question) can forge an
    # earlier marker, and the model can repeat one later.
    text = proc.stdout
    marker = "\n--- answer ("
    cit_pos = text.find("\ncitations:")
    if cit_pos < 0:
        cit_pos = text.find("citations:")
    marker_pos = text.find(marker, cit_pos if cit_pos >= 0 else 0)
    pre_marker = text[:marker_pos] if marker_pos >= 0 else text
    if marker_pos >= 0:
        answer = text[marker_pos + len(marker):].split(") ---\n", 1)[-1].strip()
    else:
        answer = text
    citations_text = "\n".join(
        line for line in pre_marker.splitlines() if line.strip().startswith("citations:")
    )
    return RunOutput(
        output=answer,
        wall_s=wall,
        extra={
            "stdout_tail": text[-500:],
            "stdout": citations_text,
            "stderr_tail": proc.stderr.strip()[-500:],
        },
    )


def _extract_numbers(text: str) -> list[float]:
    """Exact mirror of bench.rs `extract_numbers`: whitespace/|;:
    tokenised, edges trimmed to alphanumeric/./-/,% characters,
    first char digit/./-, last-separator-is-decimal for mixed
    formats, thousands iff every trailing group is 3 digits, a
    single comma is decimal, other multi-comma forms skipped, and
    only finite Rust-parseable floats count."""
    import math

    nums: list[float] = []
    for raw in re.split(r"[\s|;:]+", text or ""):
        token = raw
        while token and not (token[0].isalnum() or token[0] in ".-,%"):
            token = token[1:]
        while token and not (token[-1].isalnum() or token[-1] in ".-,%"):
            token = token[:-1]
        if not token or not (token[0].isdigit() or token[0] in ".-"):
            continue  # citations, words, markers
        percent = token.endswith("%")
        core = token[:-1] if percent else token
        if "," in core and "." in core:
            if core.rfind(",") > core.rfind("."):  # 1.200,50 European
                core = core.replace(".", "").replace(",", ".")
            else:  # 1,200.50
                core = core.replace(",", "")
        elif "," in core:
            groups = core.split(",")
            if len(groups) > 1 and all(
                len(g) == 3 and g.isdigit() for g in groups[1:]
            ):
                core = core.replace(",", "")  # thousands, any first-group length
            elif len(groups) == 2:
                core = groups[0] + "." + groups[1]  # decimal comma
            else:
                continue  # 1,200,50 etc: ambiguous
        if "_" in core or not core.isascii():
            continue  # Rust rejects underscores and non-ASCII digits
        try:
            value = float(core)  # .5, 1e5, 1.2e-3, -1.2e-3; ranges fail here
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        if percent:
            value /= 100.0
        nums.append(value)
    return nums


def _final_number(text: str) -> float | None:
    """Last number in the answer (mirrors gridtalk bench)."""
    nums = _extract_numbers(str(text))
    return nums[-1] if nums else None


_CELL = re.compile(r"([A-Za-z]{1,3})([0-9]+)$")


def _parse_cell(ref: str):
    """'Sheet1!B12' or 'B12' -> (sheet|None, row, col)."""
    text = str(ref).strip()
    sheet = None
    cell = text
    if text.startswith("'"):
        # Quoted sheet with '' escapes, like _parse_cited_range.
        i = 1
        name = ""
        while i < len(text):
            if text[i] == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    name += "'"
                    i += 2
                    continue
                i += 1
                break
            name += text[i]
            i += 1
        if i < len(text) and text[i] == "!":
            sheet = name
            cell = text[i + 1:]
    elif "!" in text:
        sheet_part, _, cell = text.rpartition("!")
        sheet = sheet_part.strip("'") or None
    m = _CELL.match(cell.strip().replace("$", ""))
    if not m:
        return None
    col = 0
    for ch in m.group(1).upper():
        col = col * 26 + (ord(ch) - 64)
    row = int(m.group(2))
    if row < 1:
        return None
    return sheet, row, col


def _split_citations(text: str) -> list[str]:
    """Split a citations list on commas, respecting single-quoted
    sheet names (with '' escapes) that may contain commas."""
    out, cur, i = [], "", 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            cur += ch
            i += 1
            while i < len(text):
                cur += text[i]
                if text[i] == "'":
                    if i + 1 < len(text) and text[i + 1] == "'":
                        cur += "'"
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if ch == ",":
            out.append(cur)
            cur = ""
            i += 1
            continue
        cur += ch
        i += 1
    out.append(cur)
    return out


def _contains_bounded(answer: str, needle: str) -> bool:
    """Mirror bench.rs contains_bounded: needle must not be flanked
    by alphanumeric / ! / : / . / $ characters."""
    def boundary(ch: str | None) -> bool:
        return ch is None or not (ch.isalnum() or ch in "!.:$")

    start = 0
    while True:
        pos = answer.find(needle, start)
        if pos < 0:
            return False
        end = pos + len(needle)
        before = answer[pos - 1] if pos > 0 else None
        after = answer[end] if end < len(answer) else None
        if boundary(before) and boundary(after):
            return True
        start = pos + 1


def _parse_cited_range(c: str):
    """Parse 'Sheet!A1[:B2]' / "'A:B'!A1:B2" / Sheet!A1:Sheet!B2
    into ((sheet,row,col),(sheet2|None,row,col)) like parse_range."""
    text = c.strip()
    # Split off a quoted sheet prefix first (it may contain ':').
    sheet = None
    rest = text
    if text.startswith("'"):
        i = 1
        name = ""
        while i < len(text):
            if text[i] == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    name += "'"
                    i += 2
                    continue
                i += 1
                break
            name += text[i]
            i += 1
        if i < len(text) and text[i] == "!":
            sheet = name
            rest = text[i + 1:]
            lo_s, _, hi_s = rest.partition(":")
            lo = _parse_cell(lo_s)
            hi = _parse_cell(hi_s or lo_s)
            if lo is None or hi is None:
                return None, None
            if hi[0] is not None and hi[0].casefold() != sheet.casefold():
                return None, None
            return (sheet, lo[1], lo[2]), hi
        return None, None
    lo_full, _, hi_full = text.partition(":")
    lo = _parse_cell(lo_full)
    hi = _parse_cell(hi_full or lo_full)
    if lo is None or hi is None:
        return None, None
    if hi[0] is not None and lo[0] is not None and hi[0].casefold() != lo[0].casefold():
        return None, None
    return lo, hi


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
    if esheet is None:
        return False  # Rust parse_cell_ref requires a sheet
    cited: list[str] = []
    for line in str((extra or {}).get("stdout", "")).splitlines():
        if line.strip().startswith("citations:"):
            cited += [c.strip() for c in _split_citations(line.split(":", 1)[1])]
    for c in cited:
        # The SAME cited range must appear in the answer (as a
        # standalone reference) and contain the expected cell.
        if not c or "!" not in c or not _contains_bounded(answer or "", c):
            continue
        # Mirror tools::parse_range: Sheet!Lo[:End], where the end
        # may repeat the sheet qualifier (Sheet1!A1:Sheet1!B2) and
        # the sheet may be quoted (and contain ':').
        lo, hi = _parse_cited_range(c)
        if lo is None or hi is None:
            continue
        sheet = lo[0]
        if sheet is None or sheet.casefold() != esheet.casefold():
            continue
        # parse_range_uncapped rejects reversed ranges (no size cap).
        if hi[1] < lo[1] or hi[2] < lo[2]:
            continue
        if lo[1] <= erow <= hi[1] and lo[2] <= ecol <= hi[2]:
            return True
    return False


# Mirror of bench.rs CITATION_WORDS.
_CITATION_WORDS = {"citation", "citations", "source", "sources", "see", "cited", "range"}


def _only_citations_follow(rest: str) -> bool:
    """Mirror of bench.rs `only_citations_follow`."""
    while "'!" in rest:
        bang = rest.index("'!")
        start = rest.rfind("'", 0, bang)
        if start == -1:
            start = bang
        rest = rest[:start] + "!" + rest[bang + 2 :]
    for token in rest.split():
        token = token.strip("".join(c for c in token if not c.isalnum() and c != "!"))
        if token and "!" not in token and token not in _CITATION_WORDS:
            return False
    return True


def _text_present(answer: str, expected: str) -> bool:
    """Mirror of bench.rs `judge_text`: the expected text is the last
    value the answer states. Its last whole-word occurrence (ignoring
    case) may be followed only by citations, punctuation and citation
    words."""
    answer = answer.lower()
    expected = expected.strip().lower()
    if not expected:
        return False

    def word(c: str) -> bool:
        return c.isalnum() or c == "_"

    last = None
    start = answer.find(expected)
    while start != -1:
        end = start + len(expected)
        before_ok = start == 0 or not word(answer[start - 1])
        after_ok = end == len(answer) or not word(answer[end])
        if before_ok and after_ok:
            last = end
        start = answer.find(expected, start + 1)
    if last is None:
        return False
    return _only_citations_follow(answer[last:])


def score(task: dict, output: RunOutput) -> JudgeResult:
    text = task.get("expected_text")
    if isinstance(text, str) and text.strip():
        # A text or boolean answer (bench.rs `expected_text`).
        passed = _text_present(str(output.output), text)
        extra = getattr(output, "extra", None) or {}
        cited = _citation_present(str(output.output), task, extra) if task.get("citation") else False
        return JudgeResult(
            score=1.0 if passed else 0.0,
            feedback=f"expected text {text!r} {'found' if passed else 'missing'}; citation {'hit' if cited else 'miss'}",
            extra={"passed": passed, "citation_hit": cited, "kind": task.get("kind")},
        )
    raw = task.get("expected")
    if isinstance(raw, bool):
        return JudgeResult(
            score=0.0,
            feedback=f"task {task.get('id', '?')}: bool expected value is not numeric",
            extra={"passed": False, "citation_hit": False, "kind": task.get("kind")},
        )
    try:
        expected = float(raw)
    except (TypeError, ValueError, OverflowError):
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
    import math as _math

    if not _math.isfinite(expected):
        return JudgeResult(
            score=0.0,
            feedback=f"task {task.get('id', '?')}: non-finite expected value {raw!r}",
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
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("bench task id must be a non-empty string")
        if (
            row["id"] != row["id"].strip()
            or "\n" in row["id"]
            or "\r" in row["id"]
            or row["id"].startswith("#")
        ):
            raise ValueError(f"bench task id {row['id']!r} cannot round-trip hermit's split files")
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"task {row['id']}: question must be a non-empty string")
        if "expected_text" in row and (
            not isinstance(row["expected_text"], str) or not row["expected_text"].strip()
        ):
            raise ValueError(f"task {row['id']}: expected_text must be a non-empty string")
        exp = row["expected"]
        if isinstance(exp, bool) or not isinstance(exp, (int, float, str)):
            raise ValueError(f"task {row['id']}: expected must be a number or numeric string")
        if isinstance(exp, str) and _final_number(exp) is None:
            raise ValueError(f"task {row['id']}: expected {exp!r} is not numeric")
        if isinstance(exp, (int, float)):
            import math

            try:
                finite = math.isfinite(exp)
            except OverflowError:  # huge int like 10**400
                finite = False
            if not finite:
                raise ValueError(f"task {row['id']}: expected must be finite")
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
