//! The issues check on `.xlsx` files written here with Gridcraft's
//! writer, so saved values and defined names go through a real
//! package.

use gridcraft_core::{CellError, CellRef, Value};
use gridcraft_model::{Cell, DefinedName, Formula, Sheet, Workbook};
use gridtalk_graph::eval::Evaluator;
use gridtalk_graph::issues::{find_issues, IssueKind};
use gridtalk_graph::load_bytes;
use std::path::Path;
use std::sync::Arc;

/// A formula cell carrying `saved` as the value the file saved.
fn formula(text: &str, saved: Value) -> Cell {
    let mut cell = Cell::formula(Formula::new(text));
    cell.value = saved;
    cell
}

#[test]
fn saved_values_errors_and_names_survive_a_package() {
    let mut wb = Workbook::new();
    wb.sheets = vec![Arc::new(Sheet::new("Model"))];
    let cells = [
        (0, 0, Cell::value(Value::Number(2.0))),
        // Saved as 999, recalculates to 20.
        (0, 1, formula("A1*10", Value::Number(999.0))),
        // Saved as what it computes.
        (1, 1, formula("A1*10", Value::Number(20.0))),
        // An error saved as that error: an error, not a stale value.
        (2, 1, formula("1/0", Value::Error(CellError::Div0))),
        // A link to another workbook, as a file stores it.
        (3, 1, formula("[1]Prices!A1*2", Value::Number(84.0))),
    ];
    let sheet = wb.sheet_mut(0).unwrap();
    for (row, col, cell) in cells {
        sheet.set_cell(CellRef::new(row, col), cell);
    }
    wb.names.push(DefinedName {
        name: "Lost".into(),
        scope: None,
        formula: "#REF!".into(),
        comment: String::new(),
        hidden: false,
    });
    let bytes = gridcraft_xlsx::write_xlsx(&wb).unwrap();
    let snapshot = load_bytes(Path::new("issues.xlsx"), &bytes).unwrap();

    let found = find_issues(&snapshot, &Evaluator::new(&snapshot), None);
    let summary: Vec<(IssueKind, String)> = found
        .iter()
        .map(|i| {
            (
                i.kind,
                i.range().or_else(|| i.name.clone()).unwrap_or_default(),
            )
        })
        .collect();
    assert_eq!(
        summary,
        vec![
            (IssueKind::FormulaError, "Model!B3".to_string()),
            (IssueKind::BrokenName, "Lost".to_string()),
            (IssueKind::SavedValueDiffers, "Model!B1".to_string()),
            // The 10 written into A1*10.
            (IssueKind::ConstantInFormula, "Model!B1:B2".to_string()),
            (IssueKind::ExternalReference, "Model!B4".to_string()),
        ],
        "{found:?}"
    );
    let stale = &found[2].cells[0];
    assert_eq!(stale.saved.as_deref(), Some("999"));
    assert_eq!(found[4].detail, "[1]Prices");
    assert_eq!(found[4].cells[0].saved.as_deref(), Some("84"));
}
