//! Workbook health checks: the formulas and defined names that are
//! wrong, or that a reader should not trust without a look. Every
//! check is read-only and deterministic; the `issues` tool reports
//! them. Modelled on OfficeCLI's `view issues` (Apache-2.0).
//!
//! Findings are grouped: cells of one column with the same kind of
//! problem and the same detail (one error code, one set of written-in
//! numbers) are one [`Issue`], so a formula filled down 500 rows is
//! one finding, not 500.

use crate::eval::{EvalError, Evaluator, Value};
use crate::lineage::Lineage;
use crate::{sheet_name_eq, CellValue, RangeRef, WorkbookSnapshot};
use gridcraft_core::CellError;
use gridcraft_formula::{BinOp, Expr, SheetSel, UnOp};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// What is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    /// The formula evaluates to an error, and no formula it reads
    /// does: the error starts here.
    FormulaError,
    /// The formula refers to a sheet the workbook does not have.
    MissingSheet,
    /// The formula holds `#REF!`: a reference whose cells were
    /// deleted.
    DeletedReference,
    /// A defined name that does not parse, holds `#REF!` or refers to
    /// a sheet the workbook does not have.
    BrokenName,
    /// The formula evaluates to an error because a formula it reads
    /// does.
    ErrorFromPrecedent,
    /// The value saved in the file differs from Gridcraft's
    /// recalculation.
    SavedValueDiffers,
    /// The formula (or defined name) reads another workbook, whose
    /// values GridTalk cannot see.
    ExternalReference,
    /// A number written into the formula's arithmetic (`=B5*1.05`),
    /// which cannot be changed or traced from a cell.
    ConstantInFormula,
}

impl IssueKind {
    pub fn as_str(self) -> &'static str {
        match self {
            IssueKind::FormulaError => "formula_error",
            IssueKind::MissingSheet => "missing_sheet",
            IssueKind::DeletedReference => "deleted_reference",
            IssueKind::BrokenName => "broken_name",
            IssueKind::ErrorFromPrecedent => "error_from_precedent",
            IssueKind::SavedValueDiffers => "saved_value_differs",
            IssueKind::ExternalReference => "external_reference",
            IssueKind::ConstantInFormula => "constant_in_formula",
        }
    }

    /// Errors make a result wrong; warnings ask for a look.
    pub fn severity(self) -> Severity {
        match self {
            IssueKind::FormulaError
            | IssueKind::MissingSheet
            | IssueKind::DeletedReference
            | IssueKind::BrokenName => Severity::Error,
            IssueKind::ErrorFromPrecedent
            | IssueKind::SavedValueDiffers
            | IssueKind::ExternalReference
            | IssueKind::ConstantInFormula => Severity::Warning,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
        }
    }
}

/// One formula cell an issue covers.
#[derive(Debug, Clone, PartialEq)]
pub struct IssueCell {
    /// 0-based row.
    pub row: usize,
    pub formula: String,
    /// What Gridcraft computes.
    pub result: Result<Value, EvalError>,
    /// The value saved in the file, as displayed, when there is one.
    pub saved: Option<String>,
}

/// One finding: cells of one column (or one defined name) with the
/// same problem.
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub kind: IssueKind,
    /// The sheet the cells are on; for a broken name, the sheet it is
    /// scoped to (`None`: workbook scope).
    pub sheet: Option<String>,
    /// 0-based column of the cells; `None` for a defined name.
    pub col: Option<usize>,
    /// The defined name, for [`IssueKind::BrokenName`] and a name's
    /// [`IssueKind::ExternalReference`].
    pub name: Option<String>,
    /// What the cells share: the error, the missing sheet, the
    /// written-in numbers.
    pub detail: String,
    /// The cells, top down. Empty for a defined name.
    pub cells: Vec<IssueCell>,
}

impl Issue {
    pub fn severity(&self) -> Severity {
        self.kind.severity()
    }

    /// The range the cells span, `Sheet!D2:D40` (one cell: `Sheet!D2`).
    pub fn range(&self) -> Option<String> {
        let (sheet, col) = (self.sheet.as_ref()?, self.col?);
        let r0 = self.cells.first()?.row;
        let r1 = self.cells.last()?.row;
        Some(cell_range(sheet, r0, col, r1))
    }
}

/// A sheet-qualified range of one column, a single cell in its short
/// form.
pub fn cell_range(sheet: &str, r0: usize, col: usize, r1: usize) -> String {
    if r0 == r1 {
        format!(
            "{}!{}{}",
            crate::quote_sheet(sheet),
            crate::col_name(col),
            r0 + 1
        )
    } else {
        RangeRef {
            sheet: sheet.to_string(),
            r0,
            c0: col,
            r1,
            c1: col,
        }
        .a1()
    }
}

/// Where to look: one sheet (by snapshot index) and a 0-based
/// inclusive rectangle on it, or (`None`) the whole workbook. Defined
/// names are checked only for the whole workbook.
pub type IssueScope = Option<(usize, (usize, usize, usize, usize))>;

/// Functions whose value changes on every recalculation, so a saved
/// value that differs from a fresh one says nothing.
const VOLATILE: &[&str] = &["NOW", "TODAY", "RAND", "RANDBETWEEN", "RANDARRAY"];

/// Every issue in `scope`, errors first, then by sheet, column and
/// row. `eval` must be the evaluator of `wb`.
pub fn find_issues(wb: &WorkbookSnapshot, eval: &Evaluator, scope: IssueScope) -> Vec<Issue> {
    let lineage = Lineage::new(wb);
    // Every formula's computed value, for every sheet: whether an
    // error starts at a cell or comes from one it reads may depend on
    // a cell outside the scope.
    let mut results: Vec<BTreeMap<(usize, usize), Result<Value, EvalError>>> = Vec::new();
    for sheet in &wb.sheets {
        let coords: Vec<(usize, usize)> = sheet
            .cells
            .iter()
            .enumerate()
            .flat_map(|(r, row)| {
                row.iter()
                    .enumerate()
                    .filter(|(_, cell)| matches!(cell, CellValue::Formula { .. }))
                    .map(move |(c, _)| (r, c))
            })
            .collect();
        let values = if coords.is_empty() {
            Vec::new()
        } else {
            eval.values(&sheet.name, &coords)
        };
        results.push(coords.into_iter().zip(values).collect());
    }
    let errors: Vec<BTreeSet<(usize, usize)>> = results
        .iter()
        .map(|sheet| {
            sheet
                .iter()
                .filter(|(_, v)| v.is_err())
                .map(|(at, _)| *at)
                .collect()
        })
        .collect();

    // (kind, sheet, column, detail) -> cells, in a stable order.
    let mut groups: BTreeMap<(IssueKind, usize, usize, String), Vec<IssueCell>> = BTreeMap::new();
    for (s, sheet) in wb.sheets.iter().enumerate() {
        let bounds = match scope {
            Some((only, b)) if only == s => Some(b),
            Some(_) => continue,
            None => None,
        };
        for (&(r, c), result) in &results[s] {
            if let Some((r0, c0, r1, c1)) = bounds {
                if r < r0 || r > r1 || c < c0 || c > c1 {
                    continue;
                }
            }
            let CellValue::Formula { expr, cached } = sheet.get_ref(r, c) else {
                continue;
            };
            let saved = cached.as_deref().map(CellValue::display);
            let cell = |result: &Result<Value, EvalError>| IssueCell {
                row: r,
                formula: expr.clone(),
                result: result.clone(),
                saved: saved.clone(),
            };
            let mut add = |kind: IssueKind, detail: String| {
                groups
                    .entry((kind, s, c, detail))
                    .or_default()
                    .push(cell(result));
            };
            let parsed = gridcraft_formula::parse(expr.strip_prefix('=').unwrap_or(expr)).ok();
            let refs = parsed.as_ref().map(sheet_refs).unwrap_or_default();

            // Static checks on the formula's text.
            let mut deleted = parsed.as_ref().is_some_and(holds_ref_error);
            if parsed.is_none() && expr.to_ascii_uppercase().contains("#REF!") {
                deleted = true;
            }
            if deleted {
                add(IssueKind::DeletedReference, "#REF!".into());
            }
            let (mut external, missing) = classify_sheets(wb, &refs);
            for book in external_sheets(expr) {
                if !external.contains(&book) {
                    external.push(book);
                }
            }
            if !missing.is_empty() {
                add(IssueKind::MissingSheet, missing.join(", "));
            }
            if !external.is_empty() {
                add(IssueKind::ExternalReference, external.join(", "));
            }
            // The static finding already says why such a formula
            // fails (Gridcraft cannot parse a reference to another
            // workbook), so its error is not reported again. Formulas
            // that read it still get `error_from_precedent`.
            let explained = deleted || !missing.is_empty() || !external.is_empty();
            if let Some(e) = &parsed {
                let constants = written_in_numbers(e);
                if !constants.is_empty() {
                    add(IssueKind::ConstantInFormula, constants.join(", "));
                }
            }

            // The computed value.
            match result {
                Err(_) if explained => {}
                Err(e) => {
                    let from_precedent =
                        lineage
                            .areas(wb, s, expr)
                            .into_iter()
                            .any(|(ps, (pr0, pc0, pr1, pc1))| {
                                errors.get(ps).is_some_and(|set| {
                                    set.range((pr0, 0)..=(pr1, usize::MAX)).any(|&(er, ec)| {
                                        ec >= pc0 && ec <= pc1 && (ps, er, ec) != (s, r, c)
                                    })
                                })
                            });
                    let kind = if from_precedent {
                        IssueKind::ErrorFromPrecedent
                    } else {
                        IssueKind::FormulaError
                    };
                    add(kind, error_text(e));
                }
                Ok(value) => {
                    let volatile = parsed.as_ref().is_some_and(calls_volatile);
                    if let Some(saved) = cached.as_deref() {
                        if !volatile && saved_differs(saved, value) {
                            add(IssueKind::SavedValueDiffers, String::new());
                        }
                    }
                }
            }
        }
    }

    let mut issues: Vec<Issue> = groups
        .into_iter()
        .map(|((kind, s, c, detail), cells)| Issue {
            kind,
            sheet: Some(wb.sheets[s].name.clone()),
            col: Some(c),
            name: None,
            detail,
            cells,
        })
        .collect();
    if scope.is_none() {
        issues.extend(name_issues(wb));
    }
    // Errors first; within a severity, by sheet order, column, row.
    let sheet_index = |issue: &Issue| {
        issue
            .sheet
            .as_ref()
            .and_then(|name| wb.sheets.iter().position(|s| &s.name == name))
            .unwrap_or(usize::MAX)
    };
    issues.sort_by_key(|issue| {
        (
            issue.severity(),
            issue.name.is_some(),
            sheet_index(issue),
            issue.col,
            issue.cells.first().map(|cell| cell.row),
            issue.kind,
        )
    });
    issues
}

/// Broken defined names, and names that read another workbook.
/// Excel's own hidden names (`_xlnm.`, `_FilterDatabase`) are skipped:
/// a broken print area or filter range does not change a value.
fn name_issues(wb: &WorkbookSnapshot) -> Vec<Issue> {
    let mut out = Vec::new();
    for name in &wb.names {
        let lower = name.name.to_ascii_lowercase();
        if lower.starts_with("_xlnm.") || lower == "_filterdatabase" {
            continue;
        }
        let issue = |kind: IssueKind, detail: String| Issue {
            kind,
            sheet: name.scope.clone(),
            col: None,
            name: Some(name.name.clone()),
            detail,
            cells: Vec::new(),
        };
        let text = name.formula.strip_prefix('=').unwrap_or(&name.formula);
        let books = external_sheets(text);
        if !books.is_empty() {
            out.push(issue(IssueKind::ExternalReference, books.join(", ")));
            continue;
        }
        let Ok(parsed) = gridcraft_formula::parse(text) else {
            out.push(issue(
                IssueKind::BrokenName,
                format!("={text} does not parse"),
            ));
            continue;
        };
        if holds_ref_error(&parsed) {
            out.push(issue(IssueKind::BrokenName, format!("={text} holds #REF!")));
            continue;
        }
        let (external, missing) = classify_sheets(wb, &sheet_refs(&parsed));
        if !missing.is_empty() {
            out.push(issue(
                IssueKind::BrokenName,
                format!("={text} refers to a missing sheet: {}", missing.join(", ")),
            ));
        } else if !external.is_empty() {
            out.push(issue(IssueKind::ExternalReference, external.join(", ")));
        }
    }
    out
}

/// The sheet names a formula's references carry (both ends of a 3D
/// span).
fn sheet_refs(e: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    e.walk(&mut |x| {
        if let Expr::Ref(r) = x {
            match &r.sheet {
                SheetSel::Current => {}
                SheetSel::Named(n) => out.push(n.clone()),
                SheetSel::Span(a, b) => {
                    out.push(a.clone());
                    out.push(b.clone());
                }
            }
        }
    });
    out
}

/// Split sheet names into those of another workbook (`[1]Sheet1`,
/// `[Book.xlsx]Sheet1`) and those this workbook lacks, each listed
/// once.
fn classify_sheets(wb: &WorkbookSnapshot, names: &[String]) -> (Vec<String>, Vec<String>) {
    let (mut external, mut missing): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for name in names {
        let list = if name.contains('[') && name.contains(']') {
            &mut external
        } else if wb.sheets.iter().any(|s| sheet_name_eq(&s.name, name)) {
            continue;
        } else {
            &mut missing
        };
        if !list.iter().any(|seen| sheet_name_eq(seen, name)) {
            list.push(name.clone());
        }
    }
    (external, missing)
}

/// The sheets of other workbooks a formula's text names, as written:
/// `[1]Prices` (`[1]Prices!A1`, how a file stores the link) or
/// `[Book.xlsx]Prices` (`'C:\dir\[Book.xlsx]Prices'!A1`). Read from
/// the text because Gridcraft's parser does not take them. A
/// structured reference (`Table1[Qty]`, `[@Qty]`) is no link: no
/// sheet and `!` follow its brackets.
pub fn external_sheets(expr: &str) -> Vec<String> {
    let chars: Vec<char> = expr.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut push = |name: String| {
        if !out.contains(&name) {
            out.push(name);
        }
    };
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '"' => {
                // A string literal; `""` inside it is a quote.
                i += 1;
                while i < chars.len() {
                    if chars[i] == '"' {
                        if chars.get(i + 1) == Some(&'"') {
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
            '\'' => {
                // A quoted sheet name; `''` inside it is a quote.
                let mut name = String::new();
                i += 1;
                while i < chars.len() {
                    if chars[i] == '\'' {
                        if chars.get(i + 1) == Some(&'\'') {
                            name.push('\'');
                            i += 2;
                            continue;
                        }
                        break;
                    }
                    name.push(chars[i]);
                    i += 1;
                }
                i += 1;
                if chars.get(i) == Some(&'!') && name.contains('[') && name.contains(']') {
                    push(name.rsplit(['\\', '/']).next().unwrap_or(&name).to_string());
                }
            }
            '[' => {
                let Some(close) = chars[i..].iter().position(|&c| c == ']').map(|p| i + p) else {
                    break;
                };
                if chars[i + 1..close].contains(&'[') {
                    // A nested structured reference: `T[[#This Row],[Qty]]`.
                    i += 1;
                    continue;
                }
                let mut j = close + 1;
                while j < chars.len()
                    && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '.')
                {
                    j += 1;
                }
                let follows_name = i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
                if chars.get(j) == Some(&'!') && !follows_name {
                    push(chars[i..j].iter().collect());
                }
                i = close + 1;
            }
            _ => i += 1,
        }
    }
    out
}

fn holds_ref_error(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| {
        if matches!(x, Expr::Error(CellError::Ref)) {
            found = true;
        }
    });
    found
}

fn calls_volatile(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| {
        if let Expr::Call(name, _) = x {
            if VOLATILE.contains(&name.as_str()) {
                found = true;
            }
        }
    });
    found
}

/// Whether an expression reads a cell or a name.
fn reads_something(e: &Expr) -> bool {
    let mut found = false;
    e.walk(&mut |x| {
        if matches!(x, Expr::Ref(_) | Expr::Name(_) | Expr::Struct(_)) {
            found = true;
        }
    });
    found
}

/// The numbers written as operands of the formula's arithmetic
/// (`+ - * / ^`), as they read, each listed once. Only a formula that
/// reads a cell or name counts (`=5*12` is a typed number in
/// disguise, not an assumption hidden in a calculation). Not counted:
/// 0 and 1 (`=A1+1`, `=1-B2`), exponents (`=A1^2`), and numbers
/// passed to a function (`ROUND(A1, 2)`), whose meaning the function
/// gives.
fn written_in_numbers(e: &Expr) -> Vec<String> {
    if !reads_something(e) {
        return Vec::new();
    }
    let mut out: Vec<String> = Vec::new();
    e.walk(&mut |x| {
        let Expr::Binary(op, a, b) = x else {
            return;
        };
        let operands: &[&Expr] = match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => &[a, b],
            BinOp::Pow => &[a],
            _ => &[],
        };
        for operand in operands {
            if let Some(text) = literal_number(operand) {
                if !out.contains(&text) {
                    out.push(text);
                }
            }
        }
    });
    out
}

/// A number literal as written (`1.05`, `-3`, `5%`), unless it is 0
/// or 1.
fn literal_number(e: &Expr) -> Option<String> {
    match e {
        Expr::Number(n) if *n != 0.0 && *n != 1.0 => Some(CellValue::Number(*n).display()),
        Expr::Paren(inner) | Expr::Unary(UnOp::Plus, inner) => literal_number(inner),
        Expr::Unary(UnOp::Neg, inner) => match inner.as_ref() {
            Expr::Number(n) if *n == 1.0 => None,
            _ => literal_number(inner).map(|t| format!("-{t}")),
        },
        Expr::Unary(UnOp::Percent, inner) => match inner.as_ref() {
            Expr::Number(n) if *n != 0.0 => Some(format!("{}%", CellValue::Number(*n).display())),
            _ => None,
        },
        _ => None,
    }
}

/// An evaluation error without its Rust-side prefix: `#DIV/0!
/// (division by zero)`.
fn error_text(e: &EvalError) -> String {
    match e {
        EvalError::Parse(m) | EvalError::Ref(m) | EvalError::Cycle(m) | EvalError::Type(m) => {
            m.clone()
        }
    }
}

/// Whether the file's saved value disagrees with the recalculated
/// one. Numbers compare to 1e-9 of their size (a saved value is
/// written to 15-17 digits); text, booleans and type changes by
/// equality.
fn saved_differs(saved: &CellValue, value: &Value) -> bool {
    match (saved, value) {
        (CellValue::Number(a), Value::Number(b)) => {
            (a - b).abs() > 1e-9 * a.abs().max(b.abs()).max(1.0)
        }
        (CellValue::Bool(a), Value::Bool(b)) => a != b,
        (CellValue::Text(a), Value::Text(b)) => a != b,
        // An empty text result is often saved as no value at all.
        (CellValue::Text(a), Value::Empty) | (CellValue::Empty, Value::Text(a)) => !a.is_empty(),
        (CellValue::Empty, Value::Empty) => false,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DefinedName, SheetSnapshot};

    fn f(expr: &str, cached: Option<CellValue>) -> CellValue {
        CellValue::Formula {
            expr: expr.into(),
            cached: cached.map(Box::new),
        }
    }

    fn n(v: f64) -> CellValue {
        CellValue::Number(v)
    }

    fn book(sheets: Vec<SheetSnapshot>, names: Vec<DefinedName>) -> WorkbookSnapshot {
        WorkbookSnapshot {
            name: "t".into(),
            sheets,
            names,
            warnings: Vec::new(),
            source: None,
        }
    }

    fn issues(wb: &WorkbookSnapshot) -> Vec<Issue> {
        find_issues(wb, &Evaluator::new(wb), None)
    }

    fn of(issues: &[Issue], kind: IssueKind) -> Vec<&Issue> {
        issues.iter().filter(|i| i.kind == kind).collect()
    }

    #[test]
    fn an_error_starts_where_no_precedent_errs() {
        let wb = book(
            vec![SheetSnapshot::new(
                "S",
                vec![vec![
                    n(0.0),
                    f("1/A1", None),
                    f("B1*2", None),
                    f("A1+1", None),
                ]],
            )],
            vec![],
        );
        let found = issues(&wb);
        let origin = of(&found, IssueKind::FormulaError);
        assert_eq!(origin.len(), 1, "{found:?}");
        assert_eq!(origin[0].range().as_deref(), Some("S!B1"));
        assert!(
            origin[0].detail.starts_with("#DIV/0!"),
            "{}",
            origin[0].detail
        );
        let passed_on = of(&found, IssueKind::ErrorFromPrecedent);
        assert_eq!(passed_on.len(), 1);
        assert_eq!(passed_on[0].range().as_deref(), Some("S!C1"));
        // Errors sort before warnings.
        assert_eq!(found[0].kind, IssueKind::FormulaError);
    }

    #[test]
    fn a_saved_value_that_recalculation_disagrees_with() {
        let wb = book(
            vec![SheetSnapshot::new(
                "S",
                vec![
                    vec![n(2.0), f("A1*10", Some(n(999.0)))],
                    vec![n(2.0), f("A2*10", Some(n(20.0)))],
                    vec![n(2.0), f("NOW()", Some(n(1.0)))],
                    vec![n(2.0), f("A4*10", None)],
                    vec![
                        CellValue::Text("a".into()),
                        f("A5&\"b\"", Some(CellValue::Text("ab".into()))),
                    ],
                ],
            )],
            vec![],
        );
        let found = issues(&wb);
        let stale = of(&found, IssueKind::SavedValueDiffers);
        assert_eq!(stale.len(), 1, "{found:?}");
        assert_eq!(stale[0].range().as_deref(), Some("S!B1"));
        assert_eq!(stale[0].cells[0].saved.as_deref(), Some("999"));
        assert_eq!(stale[0].cells[0].result, Ok(Value::Number(20.0)));
    }

    #[test]
    fn missing_sheets_deleted_references_and_other_workbooks() {
        let wb = book(
            vec![SheetSnapshot::new(
                "S",
                vec![vec![
                    n(1.0),
                    f("Gone!A1+1", None),
                    f("#REF!+A1", None),
                    f("[1]Prices!A1*A1", Some(n(5.0))),
                    f("s!A1+1", None),
                ]],
            )],
            vec![],
        );
        let found = issues(&wb);
        let missing = of(&found, IssueKind::MissingSheet);
        assert_eq!(missing.len(), 1, "{found:?}");
        assert_eq!(missing[0].detail, "Gone");
        assert_eq!(missing[0].range().as_deref(), Some("S!B1"));
        let deleted = of(&found, IssueKind::DeletedReference);
        assert_eq!(deleted.len(), 1, "{found:?}");
        assert_eq!(deleted[0].range().as_deref(), Some("S!C1"));
        let external = of(&found, IssueKind::ExternalReference);
        assert_eq!(external.len(), 1, "{found:?}");
        assert_eq!(external[0].range().as_deref(), Some("S!D1"));
        // The static findings explain these errors; none is reported
        // twice.
        assert!(of(&found, IssueKind::FormulaError).is_empty(), "{found:?}");
        // A sheet named in another case is the same sheet.
        assert!(
            found.iter().all(|i| i.range().as_deref() != Some("S!E1")),
            "{found:?}"
        );
    }

    #[test]
    fn numbers_written_into_arithmetic_grouped_down_a_column() {
        let rows: Vec<Vec<CellValue>> = (1..=3)
            .map(|r| vec![n(10.0), f(&format!("A{r}*1.05"), None)])
            .chain([
                vec![n(10.0), f("ROUND(A4,2)", None)],
                vec![n(10.0), f("A5^2+A5", None)],
                vec![n(10.0), f("1-A6", None)],
                vec![n(10.0), f("5*12", None)],
                vec![n(10.0), f("A8*(1+5%)", None)],
                vec![n(10.0), f("A9*-3", None)],
            ])
            .collect();
        let wb = book(vec![SheetSnapshot::new("S", rows)], vec![]);
        let found = of(&issues(&wb), IssueKind::ConstantInFormula)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let summary: Vec<(String, String)> = found
            .iter()
            .map(|i| (i.range().unwrap(), i.detail.clone()))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("S!B1:B3".to_string(), "1.05".to_string()),
                ("S!B8".to_string(), "5%".to_string()),
                ("S!B9".to_string(), "-3".to_string()),
            ]
        );
        assert_eq!(found[0].cells.len(), 3);
    }

    #[test]
    fn broken_defined_names_skip_excels_own() {
        let name = |name: &str, formula: &str| DefinedName {
            name: name.into(),
            scope: None,
            formula: formula.into(),
        };
        let wb = book(
            vec![SheetSnapshot::new("S", vec![vec![n(1.0)]])],
            vec![
                name("Rate", "S!$A$1"),
                name("Lost", "#REF!"),
                name("Moved", "Gone!$A$1"),
                name("_xlnm.Print_Area", "#REF!"),
                name("Outside", "[2]Rates!$B$2"),
            ],
        );
        let found = issues(&wb);
        let broken: Vec<&str> = of(&found, IssueKind::BrokenName)
            .iter()
            .map(|i| i.name.as_deref().unwrap())
            .collect();
        assert_eq!(broken, vec!["Lost", "Moved"], "{found:?}");
        let external = of(&found, IssueKind::ExternalReference);
        assert_eq!(external.len(), 1);
        assert_eq!(external[0].name.as_deref(), Some("Outside"));
    }

    #[test]
    fn a_scope_limits_the_cells_and_leaves_names_out() {
        let wb = book(
            vec![
                SheetSnapshot::new("S", vec![vec![n(0.0), f("1/A1", None), f("1/A1", None)]]),
                SheetSnapshot::new("T", vec![vec![f("1/0", None)]]),
            ],
            vec![DefinedName {
                name: "Lost".into(),
                scope: None,
                formula: "#REF!".into(),
            }],
        );
        let found = find_issues(&wb, &Evaluator::new(&wb), Some((0, (0, 2, 0, 2))));
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].range().as_deref(), Some("S!C1"));
        assert_eq!(issues(&wb).len(), 4);
    }

    #[test]
    fn links_to_other_workbooks_are_read_from_the_text() {
        assert_eq!(external_sheets("[1]Prices!A1*2"), vec!["[1]Prices"]);
        assert_eq!(
            external_sheets("'C:\\dir\\[Book.xlsx]Rates ''24'!B2+[2]S!A1+[1]Prices!A2"),
            vec!["[Book.xlsx]Rates '24", "[2]S", "[1]Prices"]
        );
        // Structured references, strings and plain quoted sheets are
        // no links.
        assert!(external_sheets("SUM(Table1[Qty])+[@Price]").is_empty());
        assert!(external_sheets("T[[#This Row],[Qty]]*2").is_empty());
        assert!(external_sheets("\"[1]Prices!A1\"&'My Sheet'!A1").is_empty());
    }
}
