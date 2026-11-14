//! Workbook formula evaluation through Gridcraft's calc engine
//! (`gridcraft-calc`): its function library (lookups, financial,
//! statistical, text, dynamic arrays), defined names, 3D references
//! and dependency-driven recalculation. The workbook-level entry
//! points in [`crate::eval`] route through here, so what the tools,
//! the UI and what-if report is what Gridcraft computes, not the
//! file's possibly stale cached values.

use crate::eval::{EvalError, Value};
use crate::{sheet_name_eq, to_gridcraft_workbook, WorkbookSnapshot};
use gridcraft_calc::Calc;
use gridcraft_core::{CellError, CellRef, Value as GcValue};
use gridcraft_model::{Cell, Workbook};

use crate::{EXCEL_MAX_COLS as MAX_COLS, EXCEL_MAX_ROWS as MAX_ROWS};

/// A snapshot converted to a Gridcraft workbook and fully
/// recalculated. Cloning it is far cheaper than recalculating, so a
/// what-if runs on a clone.
#[derive(Clone)]
pub(crate) struct Engine {
    wb: Workbook,
    calc: Calc,
}

impl Engine {
    pub(crate) fn new(snapshot: &WorkbookSnapshot) -> Self {
        let mut wb = to_gridcraft_workbook(snapshot);
        settle_unparsed(&mut wb);
        let mut calc = Calc::new();
        calc.recalc_all(&mut wb);
        Engine { wb, calc }
    }

    /// The index of a sheet, matched case-insensitively.
    pub(crate) fn sheet(&self, name: &str) -> Result<usize, EvalError> {
        self.wb
            .sheets
            .iter()
            .position(|s| sheet_name_eq(&s.name, name))
            .ok_or_else(|| EvalError::Ref(format!("unknown sheet {name}")))
    }

    /// The computed value of a cell. A formula Gridcraft cannot parse
    /// is never calculated (it would read as its cached value, often
    /// empty), so it is reported as a parse error instead.
    pub(crate) fn value(&self, sheet: usize, row: usize, col: usize) -> Result<Value, EvalError> {
        let at = addr(row, col)?;
        let Some(s) = self.wb.sheet(sheet) else {
            return Ok(Value::Empty);
        };
        if let Some(f) = s.cell(at).and_then(|c| c.formula.as_ref()) {
            if f.expr.is_none() {
                return Err(EvalError::Parse(format!(
                    "cannot parse formula ={}",
                    f.text
                )));
            }
        }
        convert(s.value(at))
    }

    /// Replace a cell with a number and recalculate its dependents.
    pub(crate) fn set_number(
        &mut self,
        sheet: usize,
        row: usize,
        col: usize,
        value: f64,
    ) -> Result<(), EvalError> {
        if !value.is_finite() {
            return Err(EvalError::Type("what-if value must be finite".into()));
        }
        let at = addr(row, col)?;
        let s = self
            .wb
            .sheet_mut(sheet)
            .ok_or_else(|| EvalError::Ref(format!("no sheet at index {sheet}")))?;
        s.set_cell(at, Cell::value(GcValue::Number(value)));
        self.calc.cells_changed(&mut self.wb, &[(sheet, at)]);
        Ok(())
    }
}

/// Settle every formula Gridcraft cannot parse.
///
/// One that reads another workbook (`=[1]Parameters!$B$2`) cannot be
/// calculated, because GridTalk cannot open that workbook, but the file
/// holds the result it had when it was saved. That result becomes the
/// cell's value, and what reads the cell is calculated from it, so a
/// link does not turn everything downstream into an error. The `issues`
/// tool still reports the link.
///
/// Any other formula that does not parse gets a `#NAME?` value. Such a
/// formula never enters the dependency graph, so recalculation leaves
/// its cached value in place, and its dependents would silently
/// compute with that stale number. As an error it propagates instead.
fn settle_unparsed(wb: &mut Workbook) {
    for i in 0..wb.sheets.len() {
        let unparsed: Vec<(CellRef, Cell)> = wb.sheets[i]
            .cells
            .iter()
            .filter(|(_, c)| c.formula.as_ref().is_some_and(|f| f.expr.is_none()))
            .map(|(at, c)| (at, c.clone()))
            .collect();
        if unparsed.is_empty() {
            continue;
        }
        if let Some(sheet) = wb.sheet_mut(i) {
            for (at, mut cell) in unparsed {
                let linked = cell
                    .formula
                    .as_ref()
                    .is_some_and(|f| !crate::issues::external_sheets(&f.text).is_empty());
                if linked && cell.value != GcValue::Empty {
                    cell.formula = None;
                } else {
                    cell.value = GcValue::Error(CellError::Name);
                }
                sheet.set_cell(at, cell);
            }
        }
    }
}

fn addr(row: usize, col: usize) -> Result<CellRef, EvalError> {
    if row >= MAX_ROWS || col >= MAX_COLS {
        return Err(EvalError::Ref(format!(
            "row {} column {} is outside the sheet grid",
            row + 1,
            col + 1
        )));
    }
    Ok(CellRef::new(row as u32, col as u32))
}

/// Gridcraft's value as GridTalk's. Spreadsheet errors become typed
/// [`EvalError`]s; an array (a dynamic-array anchor) reads as its
/// top-left element, which is what the anchor cell displays.
fn convert(v: GcValue) -> Result<Value, EvalError> {
    match v {
        GcValue::Empty => Ok(Value::Empty),
        GcValue::Number(n) => Ok(Value::Number(n)),
        GcValue::Text(t) => Ok(Value::Text(t.to_string())),
        GcValue::Bool(b) => Ok(Value::Bool(b)),
        GcValue::Error(e) => {
            let msg = format!("{e} ({})", describe(e));
            Err(match e {
                CellError::Circ => EvalError::Cycle(msg),
                CellError::Ref => EvalError::Ref(msg),
                CellError::Name => EvalError::Parse(msg),
                _ => EvalError::Type(msg),
            })
        }
        GcValue::Array(a) => a.data.first().cloned().map_or(Ok(Value::Empty), convert),
    }
}

/// A plain-language gloss for a spreadsheet error code.
fn describe(e: CellError) -> &'static str {
    match e {
        CellError::Null => "ranges do not intersect",
        CellError::Div0 => "division by zero",
        CellError::Value => "wrong type of argument or operand",
        CellError::Ref => "invalid cell reference",
        CellError::Name => "unknown function or name, or a formula that cannot be parsed",
        CellError::Num => "invalid numeric value",
        CellError::NA => "value not available, e.g. a lookup with no match",
        CellError::GettingData => "data not yet available",
        CellError::Spill => "spill range is blocked",
        CellError::Calc => "calculation error",
        CellError::Circ => "circular reference",
        CellError::Python => "a Python cell raised an error",
        CellError::Busy => "a Python cell is waiting to run",
        CellError::Timeout => "a Python cell ran past the time limit",
        CellError::Blocked => "Python cannot run here",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellValue, SheetSnapshot};

    fn formula(expr: &str, cached: Option<f64>) -> CellValue {
        CellValue::Formula {
            expr: expr.into(),
            cached: cached.map(|n| Box::new(CellValue::Number(n))),
        }
    }

    fn workbook(cells: Vec<Vec<CellValue>>) -> WorkbookSnapshot {
        WorkbookSnapshot {
            name: "t".into(),
            sheets: vec![SheetSnapshot::new("Sheet1", cells)],
            names: vec![],
            warnings: Vec::new(),
            source: None,
        }
    }

    #[test]
    fn an_unparseable_precedent_is_an_error_not_its_cached_value() {
        let wb = workbook(vec![
            vec![formula("1+(", Some(42.0)), formula("A1*1", None)],
            vec![CellValue::Number(1.0), formula("SUM(A1:A2)", None)],
        ]);
        let engine = Engine::new(&wb);
        assert!(matches!(engine.value(0, 0, 0), Err(EvalError::Parse(_))));
        assert!(matches!(engine.value(0, 0, 1), Err(EvalError::Parse(_))));
        assert!(matches!(engine.value(0, 1, 1), Err(EvalError::Parse(_))));
    }

    #[test]
    fn a_link_to_another_workbook_keeps_its_saved_value_for_what_reads_it() {
        let wb = workbook(vec![
            vec![
                formula("[1]Parameters!$B$2", Some(1.25)),
                formula("A1*100", Some(0.0)),
                formula("B1+1", None),
            ],
            // A link whose result was not saved has nothing to offer.
            vec![formula("[1]Parameters!$B$3", None), formula("A2*2", None)],
        ]);
        let mut engine = Engine::new(&wb);
        assert_eq!(engine.value(0, 0, 0).unwrap(), Value::Number(1.25));
        assert_eq!(engine.value(0, 0, 1).unwrap(), Value::Number(125.0));
        assert_eq!(engine.value(0, 0, 2).unwrap(), Value::Number(126.0));
        assert!(matches!(engine.value(0, 1, 0), Err(EvalError::Parse(_))));
        assert!(matches!(engine.value(0, 1, 1), Err(EvalError::Parse(_))));
        // What-if can change the linked value as it can any input.
        engine.set_number(0, 0, 0, 2.0).unwrap();
        assert_eq!(engine.value(0, 0, 2).unwrap(), Value::Number(201.0));
    }

    #[test]
    fn parseable_formulas_ignore_stale_cached_values() {
        let wb = workbook(vec![vec![
            CellValue::Number(2.0),
            formula("A1*10", Some(999.0)),
        ]]);
        let engine = Engine::new(&wb);
        assert_eq!(engine.value(0, 0, 1).unwrap(), Value::Number(20.0));
    }

    #[test]
    fn spreadsheet_errors_map_to_typed_eval_errors() {
        let wb = workbook(vec![vec![
            formula("1/0", None),
            formula("NOSUCHFN(1)", None),
            formula("C1", None),
        ]]);
        let engine = Engine::new(&wb);
        assert!(matches!(engine.value(0, 0, 0), Err(EvalError::Type(_))));
        assert!(matches!(engine.value(0, 0, 1), Err(EvalError::Parse(_))));
        assert!(matches!(engine.value(0, 0, 2), Err(EvalError::Cycle(_))));
    }

    #[test]
    fn set_number_recalculates_dependents_and_rejects_non_finite() {
        let wb = workbook(vec![vec![CellValue::Number(1.0), formula("A1+1", None)]]);
        let mut engine = Engine::new(&wb);
        engine.set_number(0, 0, 0, 41.0).unwrap();
        assert_eq!(engine.value(0, 0, 1).unwrap(), Value::Number(42.0));
        assert!(engine.set_number(0, 0, 0, f64::NAN).is_err());
    }

    #[test]
    fn cells_outside_the_grid_are_ref_errors() {
        let engine = Engine::new(&workbook(vec![]));
        assert!(matches!(
            engine.value(0, MAX_ROWS, 0),
            Err(EvalError::Ref(_))
        ));
        assert!(matches!(
            engine.value(0, 0, MAX_COLS),
            Err(EvalError::Ref(_))
        ));
    }
}
