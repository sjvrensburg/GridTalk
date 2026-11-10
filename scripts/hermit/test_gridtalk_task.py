"""Mirror-parity tests for gridtalk_task (run: .venv/bin/python -m pytest
scripts/hermit/test_gridtalk_task.py -q). The number/citation mirrors
must match crates/gridtalk-app/src/bench.rs + tools parse functions."""
import importlib.util
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    "gridtalk_task", pathlib.Path(__file__).with_name("gridtalk_task.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_extract_numbers():
    assert mod._extract_numbers("fee-5 due") == []
    assert mod._extract_numbers("pay -5 now") == [-5.0]
    assert mod._extract_numbers("value = -5.5") == [-5.5]
    assert mod._extract_numbers("x -.5 y") == [-0.5]
    assert mod._extract_numbers("$1,234.50 total") == [1234.5]
    assert mod._final_number("a 1 b 2") == 2.0


def test_parse_cell_quoted_escape():
    sheet, row, col = mod._parse_cell("'O''Brien Data'!B2")
    assert (sheet, row, col) == ("O'Brien Data", 2, 2)  # 1-based row/col


def test_citation_present():
    task = {"citation": "Sheet1!B2"}
    extra = {"stdout": "citations: Sheet1!A1:C10"}
    assert mod._citation_present("see Sheet1!A1:C10 for details", task, extra) is True
    out_task = {"citation": "Sheet1!Z9"}
    assert mod._citation_present("see Sheet1!A1:C10", out_task, extra) is False
    # Cited range must appear in the answer, not only the CLI line.
    assert mod._citation_present("no range here", task, extra) is False


def test_prepare_data_roundtrip_and_bad_id(tmp_path):
    tasks = [
        {"id": "t-0001", "question": "q1", "expected": 1.0, "citation": "S!A1", "split": "train"},
        {"id": "t-0002", "question": "q2", "expected": 2.0, "citation": "S!A2", "split": "test"},
    ]
    src = tmp_path / "tasks.json"
    src.write_text(json.dumps(tasks))
    out = tmp_path / "data"
    mod.prepare_data(src, out)
    id_files = [f for f in out.iterdir() if f.suffix == ".txt"]
    all_ids = [line for f in id_files for line in f.read_text().splitlines()]
    assert set(all_ids) == {"t-0001", "t-0002"}, (all_ids, [f.name for f in id_files])
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([dict(tasks[0], id="#comment")]))
    try:
        mod.prepare_data(bad, tmp_path / "badout")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for #-prefixed id")


def test_text_present_mirrors_judge_text():
    assert mod._text_present("It becomes big.", "big")
    assert mod._text_present("The result: TRUE", "true")
    assert mod._text_present("North East region", "north east")
    assert not mod._text_present("It becomes bigger.", "big")
    assert not mod._text_present("untrue", "TRUE")
    assert not mod._text_present("anything", "")


def test_score_judges_text_tasks_by_expected_text():
    task = {"id": "t1", "expected": 0.0, "expected_text": "big", "citation": "Sheet1!C2"}
    hit = mod.score(task, mod.RunOutput(output="It becomes big (Sheet1!C2)."))
    assert hit.extra["passed"] is True
    miss = mod.score(task, mod.RunOutput(output="0"))
    assert miss.extra["passed"] is False
