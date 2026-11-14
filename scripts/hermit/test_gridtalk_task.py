"""Mirror-parity tests for gridtalk_task (run: .venv/bin/python -m pytest
scripts/hermit/test_gridtalk_task.py -q). The number/citation mirrors
must match crates/gridtalk-app/src/bench.rs + tools parse functions."""
import importlib.util
import json
import pathlib
import sys

import pytest

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
    # The same cases as bench.rs judge_text_wants_the_last_value_stated.
    assert mod._text_present("It becomes big.", "big")
    assert mod._text_present("The result: TRUE", "true")
    assert mod._text_present("The region is North East", "north east")
    assert mod._text_present("big (Sheet1!C2).", "big")
    assert mod._text_present("big. Citation: Sheet1!C2", "big")
    assert mod._text_present("big, see 'My Sheet'!A1:B2", "big")
    assert not mod._text_present("It becomes big, not small", "big")
    assert mod._text_present("It becomes big, not small", "small")
    assert not mod._text_present("not TRUE, it is FALSE", "TRUE")
    assert not mod._text_present("big, because B2 exceeds 5", "big")
    assert not mod._text_present("It is a value", "A")
    assert mod._text_present("The grade is A.", "A")
    assert not mod._text_present("It becomes bigger.", "big")
    assert not mod._text_present("untrue", "TRUE")
    assert not mod._text_present("anything", "")


def test_score_judges_want_tasks_and_skips_citation():
    task = {"id": "t1", "kind": "column_reads", "question": "What does this column read?",
            "want": ["Rate per Hour", "Staff!C4:C44"]}
    out = mod.RunOutput(output="It reads the rate per hour from Staff.",
                        extra={"tool_results": "depends_on Rate per Hour (Staff!C4:C44)"})
    hit = mod.score(task, out)
    assert hit.extra["passed"] is True
    assert hit.extra["citation_hit"] is None  # no citation to hit
    # Echoing the question is not enough: no tool result carried it.
    task2 = dict(task, question="Does it read Rate per Hour?")
    echo = mod.RunOutput(output="It reads the Rate per Hour.",
                         extra={"tool_results": 'no regions matched "Does it read Rate per Hour?"'})
    assert mod.score(task2, echo).extra["passed"] is False


def test_prepare_data_accepts_want_without_expected(tmp_path):
    tasks = [{"id": "t-0001", "kind": "check_agrees", "question": "Does it agree?",
              "want": ["yes"], "split": "dev"}]
    src = tmp_path / "tasks.json"
    src.write_text(json.dumps(tasks))
    mod.prepare_data(src, tmp_path / "data")


def test_score_judges_text_tasks_by_expected_text():
    task = {"id": "t1", "expected": 0.0, "expected_text": "big", "citation": "Sheet1!C2"}
    hit = mod.score(task, mod.RunOutput(output="It becomes big (Sheet1!C2)."))
    assert hit.extra["passed"] is True
    miss = mod.score(task, mod.RunOutput(output="0"))
    assert miss.extra["passed"] is False


def test_want_mirrors_rust_word_bounds_and_history_strip():
    # Same cases as bench.rs want_tasks_score_sentences_and_need_tool_support.
    assert not mod._want_present("No: yesterday it disagrees.", ["yes", "agrees"],
                                 "yesterday it disagrees", "Does it agree?")
    assert not mod._want_present("36", ["6"], "6 -> 36", "q")
    history = [{"question": "What does the Labour tab hold?", "answer": "Timesheets."}]
    assert not mod._want_present(
        "Labour (searched as: What does the Labour tab hold?. and its Hrs?)",
        ["Labour"],
        'no regions matched "What does the Labour tab hold?. and its Hrs?"',
        "and its Hrs?",
        history,
    )
    assert mod._want_present("It reads Labour.", ["Labour"], "region Labour!A1:C9", "q", history)


def test_want_all_and_dumps_mirror_bench_rs():
    results = "-depends_on-> Qty Sheet1!B1:B4 -depends_on-> Price Sheet1!C1:C4"
    task = {"id": "t1", "question": "What does it read?", "want_all": ["Qty", "Price"]}

    def judged(answer, task=task, results=results):
        return mod.score(task, mod.RunOutput(output=answer, extra={"tool_results": results})).extra["passed"]

    assert judged("It reads Qty and Price.")
    assert not judged("It reads Qty.")
    assert not judged("It reads Qty and Discount.")
    both = {"id": "t2", "question": "What does it read?", "want": ["Price", "Cost"], "want_all": ["Qty"]}
    assert judged("Qty and Price.", both)
    assert not judged("Qty only.", both)
    # The echo backend's answer, the tool results printed back, fails.
    rows = " ".join(f"Sheet1!A{r}: Item{r} Sheet1!B{r}: {r} Sheet1!C{r}: {r + 1}" for r in range(2, 16))
    echoed = f"Based on the workbook context:\nQuestion: q\n{results} {rows}\n(end of context)"
    assert mod._is_dump(echoed, f"{results} {rows}")
    assert not judged(echoed, {"id": "t3", "question": "What does it read?", "want": ["Qty"]}, f"{results} {rows}")
    # A short answer quoting the results is not a dump.
    assert not mod._is_dump("Total reads Qty (Sheet1!B1:B4) and Price (Sheet1!C1:C4).", results)


def test_prepare_data_accepts_want_all_alone(tmp_path):
    tasks = [{"id": "t-0001", "kind": "k", "question": "Which?", "want_all": ["a", "b"], "split": "dev"}]
    src = tmp_path / "tasks.json"
    src.write_text(json.dumps(tasks))
    mod.prepare_data(src, tmp_path / "data")
    tasks[0]["want_all"] = []
    src.write_text(json.dumps(tasks))
    with pytest.raises(ValueError):
        mod.prepare_data(src, tmp_path / "data2")


def _fake_gridtalk(tmp_path, monkeypatch, stdout):
    """A stand-in `gridtalk` that prints `stdout` and records its argv."""
    script = tmp_path / "gridtalk"
    script.write_text(
        f"#!{sys.executable}\nimport sys, json\n"
        f"open({str(tmp_path / 'argv.json')!r}, 'w').write(json.dumps(sys.argv))\n"
        f"sys.stdout.write({stdout!r})\n"
    )
    script.chmod(0o755)
    monkeypatch.setenv("GRIDTALK_BIN", str(script))


def test_run_counts_only_tool_results_as_carried(tmp_path, monkeypatch):
    stdout = (
        "[tool] search (q=does it read Rate per Hour?) -> no regions matched\n"
        "[tool] get_cells (range=Staff!C4:C44) -> Staff!C4 = 12\n"
        "\ncitations: Staff!C4:C44\n"
        "\n--- answer (echo-local) ---\nIt reads Rate per Hour (Staff!C4:C44).\n"
    )
    _fake_gridtalk(tmp_path, monkeypatch, stdout)
    task = {"id": "t1", "question": "q", "workbook": "book.xlsx",
            "want": ["Rate per Hour", "Staff!C4:C44"]}
    out = mod.run(task)
    assert out.extra["tool_results"] == "no regions matched\nStaff!C4 = 12"
    # Only the args and the citations line carried the wanted strings.
    assert mod.score(task, out).extra["passed"] is False


def test_run_removes_the_history_file_even_when_writing_fails(tmp_path, monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    _fake_gridtalk(tmp_path, monkeypatch, "")
    task = {"id": "t1", "question": "q", "workbook": "book.xlsx",
            "history": [{"question": object(), "answer": "a"}]}
    out = mod.run(task)
    assert "error" in out.extra
    assert not list(tmp_path.glob("*-gridtalk-history.json"))
    # A serialisable history is passed to the CLI, then removed.
    task["history"] = [{"question": "What is Total?", "answer": "Qty times Price."}]
    mod.run(task)
    argv = json.loads((tmp_path / "argv.json").read_text())
    assert "--history-file" in argv
    assert not list(tmp_path.glob("*-gridtalk-history.json"))


def test_unjudgeable_expected_stays_out_of_citation_precision():
    out = mod.RunOutput(output="5")
    assert mod.score({"id": "t1", "expected": "n/a"}, out).extra["citation_hit"] is None
    assert mod.score({"id": "t1", "expected": True}, out).extra["citation_hit"] is None
    cited = {"id": "t1", "expected": "n/a", "citation": "Sheet1!B2"}
    assert mod.score(cited, out).extra["citation_hit"] is False
