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

/// Excel's grid: 1,048,576 rows by 16,384 columns.
const MAX_ROWS: usize = 1 << 20;
const MAX_COLS: usize = 1 << 14;

/// A snapshot converted to a Gridcraft workbook and fully
/// recalculated.
pub(crate) struct Engine {
    wb: Workbook,
    calc: Calc,
}

impl Engine {
    pub(crate) fn new(snapshot: &WorkbookSnapshot) -> Self {
        let mut wb = to_gridcraft_workbook(snapshot);
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
        CellError::Name => "unknown function or name",
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
