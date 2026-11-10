//! Formula evaluation for what-if analysis, the tools and the UI's
//! computed values. Every entry point evaluates a whole workbook
//! through Gridcraft's calc engine (see `crate::calc`), so results
//! come from one function library and never from a file's cached
//! values.

use crate::calc::Engine;
use crate::{CellValue, WorkbookSnapshot};
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    Parse(String),
    Ref(String),
    Cycle(String),
    Type(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Parse(m) => write!(f, "parse error: {m}"),
            EvalError::Ref(m) => write!(f, "bad reference: {m}"),
            EvalError::Cycle(m) => write!(f, "circular reference: {m}"),
            EvalError::Type(m) => write!(f, "type error: {m}"),
        }
    }
}
impl std::error::Error for EvalError {}

/// The value of a cell or formula.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    Text(String),
    Bool(bool),
    /// A reference to an empty cell. A formula never evaluates to
    /// Empty: like Excel, `=A1` of an empty A1 is 0.
    Empty,
}

impl Value {
    /// Display text, spreadsheet style (`TRUE`, `1800`, `2.5`).
    pub fn display(&self) -> String {
        match self {
            Value::Number(n) => CellValue::Number(*n).display(),
            Value::Text(s) => s.clone(),
            Value::Bool(true) => "TRUE".into(),
            Value::Bool(false) => "FALSE".into(),
            Value::Empty => String::new(),
        }
    }
}

impl Value {
    /// The value as a number, as the numeric API reports it: text is
    /// a type error (it has no number), booleans are 1/0, empty is 0.
    pub fn to_number(&self) -> Result<f64, EvalError> {
        match self {
            Value::Number(n) => Ok(*n),
            Value::Bool(b) => Ok(f64::from(u8::from(*b))),
            Value::Empty => Ok(0.0),
            Value::Text(s) => Err(EvalError::Type(format!(
                "text result {s:?} is not a number"
            ))),
        }
    }
}

/// A numeric result for the numeric API (see [`Value::to_number`]).
fn numeric(result: Result<Value, EvalError>) -> Result<f64, EvalError> {
    result?.to_number()
}

/// Evaluate a cell in a named sheet of a workbook (cross-sheet refs).
/// Numeric results only: a text result is a type error; see
/// [`eval_cell_value_in_workbook`] for any value.
pub fn eval_cell_in_workbook(
    wb: &WorkbookSnapshot,
    sheet_name: &str,
    row: usize,
    col: usize,
) -> Result<f64, EvalError> {
    numeric(eval_cell_value_in_workbook(wb, sheet_name, row, col))
}

/// Like [`eval_cell_in_workbook`], but returns the cell's value as it
/// is — number, text or boolean.
pub fn eval_cell_value_in_workbook(
    wb: &WorkbookSnapshot,
    sheet_name: &str,
    row: usize,
    col: usize,
) -> Result<Value, EvalError> {
    Evaluator::new(wb).value(sheet_name, row, col)
}

/// Evaluate a target cell with one simulated override applied.
pub fn eval_cell_with_override(
    wb: &WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<f64, EvalError> {
    numeric(eval_cell_value_with_override(
        wb,
        override_ref,
        value,
        target_ref,
    ))
}

/// Like [`eval_cell_with_override`], but returns the target's value
/// as it is — number, text or boolean.
pub fn eval_cell_value_with_override(
    wb: &WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<Value, EvalError> {
    Ok(what_if_values_in_workbook(wb, override_ref, value, target_ref)?.1)
}

/// Evaluate several cells of one named sheet (cross-sheet precedents
/// resolve through the workbook). Numeric results only: a text result
/// is a type error; see [`eval_many_values_in_workbook`] for any
/// value.
pub fn eval_many_in_workbook(
    wb: &WorkbookSnapshot,
    sheet_name: &str,
    coords: &[(usize, usize)],
) -> Vec<Result<f64, EvalError>> {
    eval_many_values_in_workbook(wb, sheet_name, coords)
        .into_iter()
        .map(numeric)
        .collect()
}

/// Like [`eval_many_in_workbook`], but returns each cell's value as
/// it is — number, text or boolean (an empty cell is `Empty`). The
/// workbook is recalculated once for the batch.
pub fn eval_many_values_in_workbook(
    wb: &WorkbookSnapshot,
    sheet_name: &str,
    coords: &[(usize, usize)],
) -> Vec<Result<Value, EvalError>> {
    Evaluator::new(wb).values(sheet_name, coords)
}

/// Workbook what-if: the override and target may live on different
/// sheets (formulas may reference across sheets). Returns
/// (before, after) for the target cell; a text result is a type
/// error (see [`what_if_values_in_workbook`] for any value).
pub fn what_if_in_workbook(
    wb: &WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<(f64, f64), EvalError> {
    let (before, after) = what_if_values_in_workbook(wb, override_ref, value, target_ref)?;
    Ok((numeric(Ok(before))?, numeric(Ok(after))?))
}

/// Like [`what_if_in_workbook`], but returns the target's values as
/// they are — number, text or boolean. Gridcraft's engine
/// recalculates the workbook, applies the override, and recalculates
/// its dependents, so the change flows through every function the
/// engine supports (VLOOKUP, INDEX/MATCH, PV, SUMPRODUCT, names, 3D
/// references) instead of stopping at a cached value.
pub fn what_if_values_in_workbook(
    wb: &WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<(Value, Value), EvalError> {
    if !value.is_finite() {
        return Err(EvalError::Type("what-if value must be finite".into()));
    }
    Evaluator::new(wb).what_if(override_ref, value, target_ref)
}

/// A workbook recalculated once, answering any number of evaluations
/// and what-ifs. The free functions above build one per call; a
/// caller that evaluates the same snapshot repeatedly (a tool
/// session, the UI) keeps one in an [`EvalCache`] instead, because
/// building it recalculates the whole workbook.
#[derive(Clone)]
pub struct Evaluator {
    engine: Engine,
}

impl Evaluator {
    /// Convert `wb` to a Gridcraft workbook and recalculate it.
    pub fn new(wb: &WorkbookSnapshot) -> Self {
        Evaluator {
            engine: Engine::new(wb),
        }
    }

    /// The value of a cell in a named sheet (matched
    /// case-insensitively).
    pub fn value(&self, sheet_name: &str, row: usize, col: usize) -> Result<Value, EvalError> {
        self.engine.value(self.engine.sheet(sheet_name)?, row, col)
    }

    /// The values of several cells of one named sheet.
    pub fn values(
        &self,
        sheet_name: &str,
        coords: &[(usize, usize)],
    ) -> Vec<Result<Value, EvalError>> {
        match self.engine.sheet(sheet_name) {
            Ok(idx) => coords
                .iter()
                .map(|(r, c)| self.engine.value(idx, *r, *c))
                .collect(),
            Err(_) => coords
                .iter()
                .map(|_| Err(EvalError::Ref(format!("no sheet named {sheet_name}"))))
                .collect(),
        }
    }

    /// The target's value before and after setting the override cell
    /// to `value`. The override is applied to a copy, recalculating
    /// only its dependents; `self` is unchanged.
    pub fn what_if(
        &self,
        override_ref: (&str, usize, usize),
        value: f64,
        target_ref: (&str, usize, usize),
    ) -> Result<(Value, Value), EvalError> {
        // Reject a bad value before paying for the clone.
        if !value.is_finite() {
            return Err(EvalError::Type("what-if value must be finite".into()));
        }
        let over = self.engine.sheet(override_ref.0)?;
        let target = self.engine.sheet(target_ref.0)?;
        let before = self.engine.value(target, target_ref.1, target_ref.2)?;
        let mut engine = self.engine.clone();
        engine.set_number(over, override_ref.1, override_ref.2, value)?;
        let after = engine.value(target, target_ref.1, target_ref.2)?;
        Ok((before, after))
    }

    /// Like [`what_if`](Self::what_if), with both values as numbers
    /// (see [`Value::to_number`]).
    pub fn what_if_number(
        &self,
        override_ref: (&str, usize, usize),
        value: f64,
        target_ref: (&str, usize, usize),
    ) -> Result<(f64, f64), EvalError> {
        let (before, after) = self.what_if(override_ref, value, target_ref)?;
        Ok((before.to_number()?, after.to_number()?))
    }
}

/// An [`Evaluator`] built on first use and reused after. One cache
/// belongs to one snapshot: when the snapshot changes, replace the
/// cache (or [`clear`](Self::clear) it), or it answers for the old
/// workbook.
#[derive(Default)]
pub struct EvalCache {
    evaluator: OnceLock<Evaluator>,
}

impl EvalCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The evaluator for `wb`, built (with a full recalculation) on
    /// the first call. Later calls must pass the same snapshot.
    pub fn get(&self, wb: &WorkbookSnapshot) -> &Evaluator {
        self.evaluator.get_or_init(|| Evaluator::new(wb))
    }

    /// Forget the evaluator, after the snapshot changed.
    pub fn clear(&mut self) {
        self.evaluator = OnceLock::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{formula_refs_qualified, parse_delimited, SheetSnapshot};

    /// Evaluate a cell of a one-sheet workbook.
    fn eval_cell(sheet: &SheetSnapshot, row: usize, col: usize) -> Result<f64, EvalError> {
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![sheet.clone()],
            names: Vec::new(),
            warnings: Vec::new(),
            source: None,
        };
        eval_cell_in_workbook(&wb, &sheet.name, row, col)
    }

    /// What-if on a one-sheet workbook.
    fn what_if(
        sheet: &SheetSnapshot,
        cell: (usize, usize),
        value: f64,
        target: (usize, usize),
    ) -> Result<(f64, f64), EvalError> {
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![sheet.clone()],
            names: Vec::new(),
            warnings: Vec::new(),
            source: None,
        };
        what_if_in_workbook(
            &wb,
            (&sheet.name, cell.0, cell.1),
            value,
            (&sheet.name, target.0, target.1),
        )
    }

    fn sheet() -> SheetSnapshot {
        parse_delimited(
            "d",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\nPears,4,5,=B3*C3\nAll,0,0,=SUM(D2:D3)\n",
            ',',
        )
        .sheets
        .into_iter()
        .next()
        .unwrap()
    }

    #[test]
    fn evaluates_arithmetic_refs_and_sum() {
        let s = sheet();
        assert_eq!(eval_cell(&s, 1, 3).unwrap(), 6.0);
        assert_eq!(eval_cell(&s, 2, 3).unwrap(), 20.0);
        assert_eq!(eval_cell(&s, 3, 3).unwrap(), 26.0);
    }

    #[test]
    fn what_if_recomputes_dependents() {
        let s = sheet();
        let (before, after) = what_if(&s, (1, 1), 10.0, (3, 3)).unwrap();
        assert_eq!(before, 26.0);
        assert_eq!(after, 50.0); // 10*3 + 4*5
    }

    #[test]
    fn eval_regressions_round2() {
        // A bad reference is an error like Excel's #NAME?, not a panic.
        let wb = parse_delimited("d", "S", "A\n=$1+1\n", ',');
        assert!(matches!(
            eval_cell(&wb.sheets[0], 1, 0),
            Err(EvalError::Parse(_))
        ));
        // Aggregates ignore empty cells: AVERAGE over 5, empty, -5 is 0,
        // COUNT over the same range is 2.
        let wb = parse_delimited(
            "d",
            "S",
            "A,B,C\n5,=AVERAGE(A2:A4),=COUNT(A2:A4)\n\n-5,,\n",
            ',',
        );
        assert_eq!(eval_cell(&wb.sheets[0], 1, 1).unwrap(), 0.0);
        assert_eq!(eval_cell(&wb.sheets[0], 1, 2).unwrap(), 2.0);
        // Formula errors inside a range propagate instead of being swallowed.
        let wb = parse_delimited("d", "S", "A,B\n=1/0,=SUM(A2:A2)\n", ',');
        assert!(eval_cell(&wb.sheets[0], 1, 1).is_err());
        // Percent postfix, exponent, unary plus.
        let wb = parse_delimited("d", "S", "A\n50\n=A2%\n=1E2\n=+A2\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 2, 0).unwrap(), 0.5);
        assert_eq!(eval_cell(&wb.sheets[0], 3, 0).unwrap(), 100.0);
        assert_eq!(eval_cell(&wb.sheets[0], 4, 0).unwrap(), 50.0);
        // Shared precedents are computed once: a depth-30 doubling chain
        // would be 2^30 evaluations otherwise.
        let mut text = String::from("A\n1\n");
        for i in 3..=31 {
            text.push_str(&format!("=A{}+A{}\n", i - 1, i - 1));
        }
        let wb = parse_delimited("d", "S", &text, ',');
        assert_eq!(eval_cell(&wb.sheets[0], 30, 0).unwrap(), 536_870_912.0);
        // LOG10( is a function, not a ref to column LOG.
        assert!(formula_refs_qualified("LOG10(A1)")
            .iter()
            .all(|q| q.col != 8507));
    }

    /// Nesting past the parser's limit is a typed error. The parser
    /// recurses as deep as its limit, which needs more stack than a
    /// test thread has in an unoptimised build.
    #[test]
    fn nesting_past_the_parser_limit_is_a_parse_error() {
        let result = std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let nested = format!("=SUM({}1{})", "SUM(".repeat(300), ")".repeat(300));
                let wb = parse_delimited("d", "S", &format!("A\n{nested}\n"), ',');
                eval_cell(&wb.sheets[0], 1, 0)
            })
            .unwrap()
            .join()
            .unwrap();
        assert!(matches!(result, Err(EvalError::Parse(_))), "{result:?}");
    }

    #[test]
    fn cross_sheet_refs_evaluate() {
        let mut wb = parse_delimited("d", "Sheet1", "A\n=Sheet2!A2*2\n", ',');
        wb.sheets.push(crate::SheetSnapshot::new(
            "Sheet2",
            vec![vec![CellValue::Number(1.0)], vec![CellValue::Number(21.0)]],
        ));
        assert_eq!(eval_cell_in_workbook(&wb, "Sheet1", 1, 0).unwrap(), 42.0);
        let (before, after) =
            what_if_in_workbook(&wb, ("Sheet2", 1, 0), 10.0, ("Sheet1", 1, 0)).unwrap();
        assert_eq!((before, after), (42.0, 20.0));
    }

    #[test]
    fn eval_round5_power_literals_and_cached_errors() {
        // Power is left-associative, as in Excel; TRUE/FALSE literals.
        let wb = parse_delimited("d", "S", "A\n=2^3^2\n=TRUE+FALSE+1\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 0).unwrap(), 64.0);
        assert_eq!(eval_cell(&wb.sheets[0], 2, 0).unwrap(), 2.0);
        // Evaluation errors are not masked by a stale cached value.
        let mut sh = sheet();
        let cached = |expr: &str, value: f64| CellValue::Formula {
            expr: expr.into(),
            cached: Some(Box::new(CellValue::Number(value))),
        };
        *sh.cell_mut(1, 3) = cached("D2+1", 999.0);
        assert!(matches!(eval_cell(&sh, 1, 3), Err(EvalError::Cycle(_))));
        *sh.cell_mut(1, 3) = cached("MEDIAN(1/0)", 42.0);
        assert!(matches!(eval_cell(&sh, 1, 3), Err(EvalError::Type(_))));
        // Neither is an unknown function: callers see the error and
        // can choose to show the cached value (the UI does).
        *sh.cell_mut(1, 3) = cached("NOSUCHFN(1)", 42.0);
        assert!(matches!(eval_cell(&sh, 1, 3), Err(EvalError::Parse(_))));
        // MEDIAN is computed, not read from the cache (B2=2, C2=3).
        *sh.cell_mut(1, 3) = cached("MEDIAN(B2:C2)", 999.0);
        assert_eq!(eval_cell(&sh, 1, 3).unwrap(), 2.5);
        let wb = parse_delimited("d", "S", "A\n=TRUE()+FALSE()\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 0).unwrap(), 1.0);
        // Whole-column ranges and empty MIN.
        let wb = parse_delimited("d", "S", "A,B\n1,=SUM(A:A)\n2,=MIN(C1)\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 1).unwrap(), 3.0);
        assert_eq!(eval_cell(&wb.sheets[0], 2, 1).unwrap(), 0.0);
        // Whole-row ranges clamp to the used rows; columns past the
        // used extents sum to 0 instead of erroring.
        let wb = parse_delimited("d", "S", "=SUM(2:1048576),=SUM(Z:Z)\n1,2\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 0, 0).unwrap(), 3.0);
        assert_eq!(eval_cell(&wb.sheets[0], 0, 1).unwrap(), 0.0);
        // Long dependency chains have no depth limit.
        let mut csv = String::from("A\n1\n");
        for i in 3..=2000 {
            csv.push_str(&format!("=A{}+1\n", i - 1));
        }
        let wb = parse_delimited("d", "S", &csv, ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1999, 0).unwrap(), 1999.0);
    }

    #[test]
    fn eval_regressions_round3() {
        // Tab whitespace, grid bounds, scalar-empty aggregates.
        let wb = parse_delimited("d", "S", "A,B\n1,=A2\t+1\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 1).unwrap(), 2.0);
        let wb = parse_delimited("d", "S", "A\n=AAAA1+1\n", ',');
        assert!(matches!(
            eval_cell(&wb.sheets[0], 1, 0),
            Err(EvalError::Parse(_))
        ));
        let wb = parse_delimited("d", "S", "A,B,C\n,=COUNT(A2),=AVERAGE(A2)\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 1).unwrap(), 0.0);
        assert!(eval_cell(&wb.sheets[0], 1, 2).is_err());
        // Start-only qualified ranges inherit the start sheet,
        // and sheet qualifiers resolve case-insensitively.
        let mut wb2 = parse_delimited("d", "Sheet1", "A\n=SUM(sheet2!A1:A2)\n", ',');
        wb2.sheets.push(crate::SheetSnapshot::new(
            "Sheet2",
            vec![vec![CellValue::Number(3.0)], vec![CellValue::Number(4.0)]],
        ));
        assert_eq!(eval_cell_in_workbook(&wb2, "Sheet1", 1, 0).unwrap(), 7.0);
        // Doubly-qualified range ends are a typed error.
        let mut wb = parse_delimited("d", "Sheet1", "A\n=SUM(Sheet2!A1:Sheet1!B1)\n", ',');
        wb.sheets.push(crate::SheetSnapshot::new(
            "Sheet2",
            vec![vec![CellValue::Number(5.0)]],
        ));
        assert!(eval_cell_in_workbook(&wb, "Sheet1", 1, 0).is_err());
    }

    #[test]
    fn eval_regressions_round4() {
        // An override of an Empty cell inside a SUM range counts.
        let wb = parse_delimited("d", "S", "A,B\n,=SUM(A2:A2)\n", ',');
        let (before, after) = what_if_in_workbook(&wb, ("S", 1, 0), 7.0, ("S", 1, 1)).unwrap();
        assert_eq!((before, after), (0.0, 7.0));
        // A stale cache never overrides the expression (1+2 != 999).
        let sheet = crate::SheetSnapshot::new(
            "S",
            vec![vec![CellValue::Formula {
                expr: "1+2".into(),
                cached: Some(Box::new(CellValue::Number(999.0))),
            }]],
        );
        assert_eq!(eval_cell(&sheet, 0, 0).unwrap(), 3.0);
        // with_override validates finiteness like what_if.
        assert!(eval_cell_with_override(&wb, ("S", 1, 0), f64::NAN, ("S", 1, 1)).is_err());
    }

    #[test]
    fn cycles_and_div_zero_are_typed_errors() {
        // B1 references itself; C1 divides by zero literally.
        let wb = parse_delimited("d", "S", "A,B,C\n1,=B2,=1/0\n", ',');
        let s = &wb.sheets[0];
        assert!(matches!(eval_cell(s, 1, 1), Err(EvalError::Cycle(_))));
        assert!(matches!(eval_cell(s, 1, 2), Err(EvalError::Type(_))));
        // Absolute refs and trailing-token rejection.
        let wb2 = parse_delimited("d", "S", "A,B\n2,=$A$2*10\n", ',');
        assert_eq!(eval_cell(&wb2.sheets[0], 1, 1).unwrap(), 20.0);
        let wb3 = parse_delimited("d", "S", "A,B\n2,=A2%3\n", ',');
        assert!(matches!(
            eval_cell(&wb3.sheets[0], 1, 1),
            Err(EvalError::Parse(_))
        ));
    }

    /// A one-row sheet: numbers, text, or `=formula` strings.
    fn row_sheet(cells: &[&str]) -> SheetSnapshot {
        let cell = |c: &&str| match c.strip_prefix('=') {
            Some(expr) => CellValue::Formula {
                expr: expr.to_string(),
                cached: None,
            },
            None if c.is_empty() => CellValue::Empty,
            None => match c.parse::<f64>() {
                Ok(n) => CellValue::Number(n),
                Err(_) => CellValue::Text(c.to_string()),
            },
        };
        SheetSnapshot::new("S", vec![cells.iter().map(cell).collect()])
    }

    fn values(cells: &[&str]) -> Vec<Result<Value, EvalError>> {
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![row_sheet(cells)],
            names: Vec::new(),
            warnings: Vec::new(),
            source: None,
        };
        let coords: Vec<_> = (0..cells.len()).map(|c| (0, c)).collect();
        eval_many_values_in_workbook(&wb, "S", &coords)
    }

    fn text(s: &str) -> Result<Value, EvalError> {
        Ok(Value::Text(s.into()))
    }

    #[test]
    fn text_and_boolean_results() {
        // A1=Widget, B1=3, C1 empty.
        let v = values(&[
            "Widget",
            "3",
            "",
            "=A1&\" x\"&B1",
            "=IF(B1>2,\"big\",\"small\")",
            "=B1>=3",
            "=A1=\"WIDGET\"",
            "=\"say \"\"hi\"\"\"",
            "=C1",
            "=C1=\"\"",
            "=\"3\"+1",
            "=UPPER(LEFT(A1,3))&MID(A1,4,2)&RIGHT(A1)",
            "=LEN(TRIM(\"  a   b \"))",
            "=CONCAT(A1,\"-\",A1:B1)",
            "=AND(B1>1,TRUE)",
            "=OR(B1>5,FALSE)",
            "=NOT(ISBLANK(C1))",
            "=ROUND(2.5,0)+ROUND(-1.25,1)+ABS(-1)",
            "=ISTEXT(A1)",
            "=A1+1",
            "=IF(B1>2,\"yes\")",
            "=IF(B1>5,\"yes\")",
        ]);
        assert_eq!(v[3], text("Widget x3"));
        assert_eq!(v[4], text("big"));
        assert_eq!(v[5], Ok(Value::Bool(true)));
        assert_eq!(v[6], Ok(Value::Bool(true))); // case-insensitive
        assert_eq!(v[7], text("say \"hi\""));
        assert_eq!(v[8], Ok(Value::Number(0.0))); // =empty is 0
        assert_eq!(v[9], Ok(Value::Bool(true))); // empty = ""
        assert_eq!(v[10], Ok(Value::Number(4.0)));
        assert_eq!(v[11], text("WIDget"));
        assert_eq!(v[12], Ok(Value::Number(3.0)));
        assert_eq!(v[13], text("Widget-Widget3"));
        assert_eq!(v[14], Ok(Value::Bool(true)));
        assert_eq!(v[15], Ok(Value::Bool(false)));
        assert_eq!(v[16], Ok(Value::Bool(false)));
        assert_eq!(v[17], Ok(Value::Number(3.0 - 1.3 + 1.0)));
        assert_eq!(v[18], Ok(Value::Bool(true)));
        assert!(matches!(v[19], Err(EvalError::Type(_))));
        assert_eq!(v[20], text("yes"));
        assert_eq!(v[21], Ok(Value::Bool(false)));
        assert_eq!(Value::Bool(true).display(), "TRUE");
    }

    #[test]
    fn if_and_iferror_evaluate_only_the_taken_branch() {
        // B1 and H1 refer to themselves: cycles, which only matter
        // in a taken branch.
        let v = values(&[
            "0",
            "=IF(TRUE,1,B1)",
            "=IF(A1=0,\"zero\",1/A1)",
            "=IFERROR(1/A1,\"n/a\")",
            "=IFERROR(10/2,1/0)",
            "=IFERROR(Z9+\"x\",-1)",
            "=ISERROR(1/A1)",
            "=IFERROR(H1,0)",
            "=IF(A1,MEDIAN(1),2)",
            "=J1+1",
        ]);
        assert_eq!(v[1], Ok(Value::Number(1.0)));
        assert_eq!(v[2], text("zero"));
        assert_eq!(v[3], text("n/a"));
        assert_eq!(v[4], Ok(Value::Number(5.0)));
        assert_eq!(v[5], Ok(Value::Number(-1.0)));
        assert_eq!(v[6], Ok(Value::Bool(true)));
        // A self-reference inside IFERROR reads as 0, as Excel shows
        // a circular reference; a bare cycle is a typed error.
        assert_eq!(v[7], Ok(Value::Number(0.0)));
        assert!(matches!(v[9], Err(EvalError::Cycle(_))), "{:?}", v[9]);
        // Gridcraft's library covers MEDIAN; the untaken branch is
        // skipped either way.
        assert_eq!(v[8], Ok(Value::Number(2.0)));
    }

    #[test]
    fn aggregates_skip_text_from_references_but_coerce_literals() {
        let v = values(&[
            "x",
            "2",
            "=\"t\"",
            "=SUM(A1:C1)",
            "=SUM(A1,B1,C1)",
            "=SUM(\"3\",TRUE)",
            "=COUNT(A1:C1,\"x\",5)",
            "=SUM(\"x\")",
            "=LEN(A1:B1)",
            "=LEFT(\"abc\",-1)",
            "=NOT()",
        ]);
        assert_eq!(v[3], Ok(Value::Number(2.0)));
        assert_eq!(v[4], Ok(Value::Number(2.0)));
        assert_eq!(v[5], Ok(Value::Number(4.0)));
        assert_eq!(v[6], Ok(Value::Number(2.0)));
        assert!(matches!(v[7], Err(EvalError::Type(_))));
        assert!(matches!(v[8], Err(EvalError::Type(_))));
        assert!(matches!(v[9], Err(EvalError::Type(_))));
        // NOT() has too few arguments: Excel refuses it at entry;
        // evaluated, it is a typed error either way.
        assert!(v[10].is_err());
        // The numeric API reports a text result as a type error and
        // keeps booleans as 1/0.
        let sheet = row_sheet(&["=\"a\"", "=1<2"]);
        assert!(matches!(eval_cell(&sheet, 0, 0), Err(EvalError::Type(_))));
        assert_eq!(eval_cell(&sheet, 0, 1), Ok(1.0));
    }

    #[test]
    fn cached_text_is_the_fallback_for_unsupported_functions() {
        let mut sheet = row_sheet(&["1"]);
        std::sync::Arc::make_mut(&mut sheet.cells[0]).push(CellValue::Formula {
            expr: "TEXT(A1,\"0.00\")".into(),
            cached: Some(Box::new(CellValue::Text("1.00".into()))),
        });
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![sheet],
            names: Vec::new(),
            warnings: Vec::new(),
            source: None,
        };
        assert_eq!(
            eval_many_values_in_workbook(&wb, "S", &[(0, 1)]),
            vec![text("1.00")]
        );
    }

    /// The impairment shape: balances look up a provision rate by
    /// days overdue, a summary totals them, a defined name holds the
    /// tax rate, and a 3D reference sums three monthly sheets.
    fn impairment() -> crate::WorkbookSnapshot {
        let f = |e: &str| CellValue::Formula {
            expr: e.into(),
            cached: Some(Box::new(CellValue::Number(-1.0))),
        };
        let n = CellValue::Number;
        let month = |name: &str, billed: f64| {
            crate::SheetSnapshot::new(
                name,
                vec![vec![CellValue::Text("Billed".into())], vec![n(billed)]],
            )
        };
        crate::WorkbookSnapshot {
            name: "impairment".into(),
            sheets: vec![
                crate::SheetSnapshot::new(
                    "Debtors",
                    vec![
                        vec![
                            CellValue::Text("Balance".into()),
                            CellValue::Text("Days".into()),
                            CellValue::Text("Pct".into()),
                            CellValue::Text("Provision".into()),
                        ],
                        vec![
                            n(1000.0),
                            n(10.0),
                            f("VLOOKUP(B2,Rates!$A$1:$B$3,2)"),
                            f("ROUND(A2*C2,2)"),
                        ],
                        vec![
                            n(5000.0),
                            n(200.0),
                            f("VLOOKUP(B3,Rates!$A$1:$B$3,2)"),
                            f("ROUND(A3*C3,2)"),
                        ],
                    ],
                ),
                crate::SheetSnapshot::new(
                    "Rates",
                    vec![
                        vec![n(0.0), n(0.02)],
                        vec![n(31.0), n(0.1)],
                        vec![n(121.0), n(0.85)],
                        vec![n(0.15)],
                    ],
                ),
                month("Jan", 1.0),
                month("Feb", 2.0),
                month("Mar", 4.0),
                crate::SheetSnapshot::new(
                    "Summary",
                    vec![vec![
                        f("SUM(Debtors!D2:D3)"),
                        f("ROUND(A1*Tax_Rate,2)"),
                        f("SUM(Jan:Mar!A2)"),
                        f("A1+"),
                        f("ROUND(4250.985,2)+ROUND(1.005,2)"),
                    ]],
                ),
            ],
            names: vec![crate::DefinedName {
                name: "Tax_Rate".into(),
                scope: None,
                formula: "Rates!$A$4".into(),
            }],
            warnings: Vec::new(),
            source: None,
        }
    }

    #[test]
    fn workbook_evaluation_uses_gridcraft_lookups_names_and_3d_refs() {
        let wb = impairment();
        let coords: Vec<_> = (0..5).map(|c| (0, c)).collect();
        let v = eval_many_values_in_workbook(&wb, "Summary", &coords);
        // 20 + 4250: VLOOKUP is computed, not read from the cached -1.
        assert_eq!(v[0], Ok(Value::Number(4270.0)));
        assert_eq!(v[1], Ok(Value::Number(640.5)));
        assert_eq!(v[2], Ok(Value::Number(7.0)));
        assert!(matches!(v[3], Err(EvalError::Parse(_))), "{:?}", v[3]);
        // Half-cents round up as in Excel, although 4250.985 and
        // 1.005 are stored just below the half.
        assert_eq!(v[4], Ok(Value::Number(4250.99 + 1.01)));
    }

    #[test]
    fn what_if_flows_through_lookup_tables() {
        let wb = impairment();
        // Raise the 121+ days rate: the 200-day balance's provision
        // must follow through VLOOKUP.
        let (before, after) =
            what_if_in_workbook(&wb, ("Rates", 2, 1), 0.9, ("Summary", 0, 0)).unwrap();
        assert_eq!((before, after), (4270.0, 4520.0));
        // Through a defined name, and across a 3D reference.
        let (b, a) = what_if_in_workbook(&wb, ("Rates", 3, 0), 0.2, ("Summary", 0, 1)).unwrap();
        assert_eq!((b, a), (640.5, 854.0));
        let (b, a) = what_if_in_workbook(&wb, ("Feb", 1, 0), 10.0, ("Summary", 0, 2)).unwrap();
        assert_eq!((b, a), (7.0, 15.0));
    }

    #[test]
    fn defined_names_survive_an_xlsx_round_trip() {
        let wb = impairment();
        let bytes = crate::to_xlsx_bytes(&wb).unwrap();
        let back = crate::load_gridcraft_bytes("impairment", "xlsx", &bytes).unwrap();
        assert_eq!(back.names, wb.names);
        assert_eq!(
            eval_cell_in_workbook(&back, "Summary", 0, 1).unwrap(),
            640.5
        );
    }

    #[test]
    fn an_evaluator_answers_what_ifs_without_changing_itself() {
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![sheet()],
            names: Vec::new(),
            warnings: Vec::new(),
            source: None,
        };
        let cache = EvalCache::new();
        let calc = cache.get(&wb);
        assert!(std::ptr::eq(calc, cache.get(&wb)), "built once");
        let (before, after) = calc
            .what_if(("Sheet1", 1, 1), 10.0, ("Sheet1", 3, 3))
            .unwrap();
        assert_eq!((before, after), (Value::Number(26.0), Value::Number(50.0)));
        assert_eq!(calc.value("sheet1", 3, 3).unwrap(), Value::Number(26.0));
        assert_eq!(
            calc.values("Sheet1", &[(1, 3), (2, 3)]),
            vec![Ok(Value::Number(6.0)), Ok(Value::Number(20.0))]
        );
        assert!(matches!(calc.value("Nope", 0, 0), Err(EvalError::Ref(_))));
    }
}
