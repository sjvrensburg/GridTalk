//! Ingestion limits and empty workbooks, on `.xlsx` files generated
//! here with Gridcraft's writer (no binary fixtures).

use gridcraft_core::{CellRef, Value};
use gridcraft_model::{Cell, Formula, Sheet, Workbook};
use gridtalk_graph::eval::{eval_cell_value_in_workbook, Value as EvalValue};
use gridtalk_graph::{
    build_graph, load_bytes, load_path, read_workbook_bytes, render_passage, to_xlsx_bytes,
    CellValue, IngestError, WorkbookSnapshot, MAX_DELIMITED_BYTES, MAX_PACKAGE_BYTES,
    MAX_SNAPSHOT_CELLS,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// The tests that allocate hundreds of MiB run one at a time.
static HEAVY: Mutex<()> = Mutex::new(());

/// One cell to write: sheet index, row, column (0-based) and content.
type Spec = (usize, u32, u32, Cell);

/// An `.xlsx` package with these sheets and cells.
fn xlsx(sheets: &[&str], cells: Vec<Spec>) -> Vec<u8> {
    let mut wb = Workbook::new();
    wb.sheets = sheets
        .iter()
        .map(|name| Arc::new(Sheet::new(*name)))
        .collect();
    for (sheet, row, col, cell) in cells {
        wb.sheet_mut(sheet)
            .unwrap()
            .set_cell(CellRef::new(row, col), cell);
    }
    gridcraft_xlsx::write_xlsx(&wb).unwrap()
}

fn number(n: f64) -> Cell {
    Cell::value(Value::Number(n))
}

fn text(s: &str) -> Cell {
    Cell::value(Value::Text(s.into()))
}

fn load_xlsx(bytes: &[u8]) -> Result<WorkbookSnapshot, IngestError> {
    load_bytes(Path::new("synthetic.xlsx"), bytes)
}

/// The message of the parse error `result` must be.
fn parse_error<T>(result: Result<T, IngestError>) -> String {
    match result {
        Err(IngestError::Parse(message)) => message,
        Err(other) => panic!("expected a parse error, got {other}"),
        Ok(_) => panic!("expected a parse error, got a value"),
    }
}

const PREFIX: &str = "gt-limits";

/// A fresh scratch directory, removed with everything in it when the
/// guard drops, so a failing assertion leaves nothing behind either.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("{PREFIX}-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}

impl std::ops::Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A file of `len` zero bytes (sparse where the filesystem allows).
fn zeros(path: &Path, len: u64) {
    std::fs::File::create(path).unwrap().set_len(len).unwrap();
}

#[test]
fn a_generated_workbook_round_trips_values_and_formulas() {
    let bytes = xlsx(
        &["Sales", "Notes"],
        vec![
            (0, 0, 0, text("Item")),
            (0, 0, 1, text("Qty")),
            (0, 1, 0, text("Apples")),
            (0, 1, 1, number(2.0)),
            (0, 2, 0, text("Pears")),
            (0, 2, 1, number(3.0)),
            (0, 3, 1, Cell::formula(Formula::new("SUM(B2:B3)"))),
            (1, 0, 0, text("checked")),
        ],
    );
    let wb = load_xlsx(&bytes).unwrap();
    assert_eq!(wb.sheets.len(), 2);
    assert_eq!(wb.sheets[0].get(1, 0), CellValue::Text("Apples".into()));
    assert_eq!(wb.sheets[0].get(2, 1), CellValue::Number(3.0));
    let total = wb.sheets[0].get(3, 1);
    assert!(
        matches!(&total, CellValue::Formula { expr, .. } if expr.contains("SUM(B2:B3)")),
        "{total:?}"
    );
    assert_eq!(
        eval_cell_value_in_workbook(&wb, "Sales", 3, 1).unwrap(),
        EvalValue::Number(5.0)
    );
    let (graph, violations) = build_graph(&wb);
    assert!(violations.is_empty(), "{violations:?}");
    assert!(!graph.regions.is_empty());
}

#[test]
fn the_cell_cap_counts_the_densified_grid() {
    let _heavy = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    // One far cell densifies to the grid up to it: 2000 x 2000 is
    // exactly the cap.
    assert_eq!(MAX_SNAPSHOT_CELLS, 2000 * 2000);
    let at_cap = load_xlsx(&xlsx(&["S"], vec![(0, 1999, 1999, number(1.0))])).unwrap();
    assert_eq!(at_cap.sheets[0].cells.len(), 2000);
    assert_eq!(at_cap.sheets[0].get(1999, 1999), CellValue::Number(1.0));
    drop(at_cap);
    // One row more is over it.
    let over = parse_error(load_xlsx(&xlsx(&["S"], vec![(0, 2000, 1999, number(1.0))])));
    assert!(over.contains("sheet S is too sparse"), "{over}");
    // Excel's last cell (XFD1048576) is refused before any grid is
    // allocated.
    let far = parse_error(load_xlsx(&xlsx(
        &["S"],
        vec![(0, (1 << 20) - 1, (1 << 14) - 1, number(1.0))],
    )));
    assert!(far.contains("sheet S is too sparse"), "{far}");
    // The cap covers the whole workbook: two sheets under it each are
    // over it together.
    let both = parse_error(load_xlsx(&xlsx(
        &["A", "B"],
        vec![(0, 1499, 1499, number(1.0)), (1, 1499, 1499, number(2.0))],
    )));
    assert!(both.contains("across all sheets"), "{both}");
}

#[test]
fn file_size_caps_are_inclusive() {
    let _heavy = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Scratch::new("size");
    for (name, cap) in [
        ("at.csv", MAX_DELIMITED_BYTES),
        ("at.tsv", MAX_DELIMITED_BYTES),
        ("at.xlsx", MAX_PACKAGE_BYTES),
        ("at.ods", MAX_PACKAGE_BYTES),
    ] {
        let path = dir.join(name);
        zeros(&path, cap);
        let read = read_workbook_bytes(&path).unwrap();
        assert_eq!(read.len() as u64, cap, "{name}");
        drop(read);
        zeros(&path, cap + 1);
        let refused = parse_error(read_workbook_bytes(&path));
        assert!(refused.contains("ingestion cap"), "{name}: {refused}");
        let _ = std::fs::remove_file(&path);
    }
    // A package that is not a zip archive is reported by the reader,
    // not taken for an empty workbook.
    let junk = dir.join("junk.xlsx");
    zeros(&junk, 4096);
    assert!(load_path(&junk).is_err());
}

#[test]
fn empty_workbooks_load_and_build_an_empty_graph() {
    let dir = Scratch::new("empty");
    let blank_csv = dir.join("blank.csv");
    std::fs::write(&blank_csv, "").unwrap();
    let cases = [
        (
            "one empty sheet",
            load_xlsx(&xlsx(&["Sheet1"], vec![])).unwrap(),
        ),
        (
            "three empty sheets",
            load_xlsx(&xlsx(&["A", "B", "C"], vec![])).unwrap(),
        ),
        ("empty csv", load_path(&blank_csv).unwrap()),
    ];
    for (label, wb) in &cases {
        assert!(!wb.sheets.is_empty(), "{label}");
        assert!(
            wb.sheets.iter().all(|s| s
                .cells
                .iter()
                .all(|row| row.iter().all(CellValue::is_empty))),
            "{label}"
        );
        let (graph, violations) = build_graph(wb);
        assert!(violations.is_empty(), "{label}: {violations:?}");
        assert!(graph.regions.is_empty(), "{label}: {:?}", graph.regions);
        // Saving an empty workbook gives a package that loads again.
        let again = load_xlsx(&to_xlsx_bytes(wb).unwrap()).unwrap();
        assert_eq!(again.sheets.len(), wb.sheets.len(), "{label}");
    }
    // An empty sheet next to a filled one: only the filled one has a
    // region.
    let mixed = load_xlsx(&xlsx(
        &["Empty", "Data"],
        vec![
            (1, 0, 0, text("Item")),
            (1, 0, 1, text("Qty")),
            (1, 1, 0, text("Apples")),
            (1, 1, 1, number(2.0)),
        ],
    ))
    .unwrap();
    let (graph, violations) = build_graph(&mixed);
    assert!(violations.is_empty(), "{violations:?}");
    assert_eq!(graph.regions.len(), 1);
    assert_eq!(graph.region_sheets, vec![1]);
    let passage = render_passage(&mixed.sheets[1], &graph.regions[0]);
    assert!(passage.contains("Apples"), "{passage}");
}
