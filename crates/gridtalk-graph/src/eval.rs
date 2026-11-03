//! Small formula evaluator for what-if analysis and the UI's computed
//! values: arithmetic, comparisons, `&` concatenation, text and
//! boolean values, cell references, ranges, and a core set of
//! functions (aggregates, IF/IFERROR, logic, text, rounding).
//! Deliberately small — the Gridcraft calc engine remains the
//! authority for full recalculation; this evaluator answers "what
//! does this formula give, and what happens if that cell changes"
//! locally.

use crate::{CellValue, SheetSnapshot};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub enum EvalError {
    Parse(String),
    Ref(String),
    Cycle(String),
    Type(String),
    Depth(String),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalError::Parse(m) => write!(f, "parse error: {m}"),
            EvalError::Ref(m) => write!(f, "bad reference: {m}"),
            EvalError::Cycle(m) => write!(f, "circular reference: {m}"),
            EvalError::Type(m) => write!(f, "type error: {m}"),
            EvalError::Depth(m) => write!(f, "depth error: {m}"),
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

    /// The value of a non-formula cell.
    fn of_cell(cell: &CellValue) -> Option<Value> {
        match cell {
            CellValue::Number(n) => Some(Value::Number(*n)),
            CellValue::Text(s) => Some(Value::Text(s.clone())),
            CellValue::Bool(b) => Some(Value::Bool(*b)),
            CellValue::Empty => Some(Value::Empty),
            CellValue::Formula { .. } => None,
        }
    }
}

/// Coerce a value for arithmetic: booleans are 1/0, empty is 0, and
/// text counts only when it reads as a number (`"3"` + 1 = 4).
fn to_number(v: &Value) -> Result<f64, EvalError> {
    match v {
        Value::Number(n) => Ok(*n),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Empty => Ok(0.0),
        Value::Text(s) => s
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|n| n.is_finite())
            .ok_or_else(|| EvalError::Type(format!("text {s:?} used as a number"))),
    }
}

fn to_text(v: &Value) -> String {
    v.display()
}

/// Truthiness for IF/AND/OR/NOT.
fn to_bool(v: &Value) -> Result<bool, EvalError> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) => Ok(*n != 0.0),
        Value::Empty => Ok(false),
        Value::Text(s) if s.eq_ignore_ascii_case("TRUE") => Ok(true),
        Value::Text(s) if s.eq_ignore_ascii_case("FALSE") => Ok(false),
        Value::Text(s) => Err(EvalError::Type(format!("text {s:?} used as a condition"))),
    }
}

/// Spreadsheet ordering: an empty operand takes the other side's
/// type (0, "" or FALSE); text compares case-insensitively; across
/// types numbers < text < booleans.
fn compare(a: &Value, b: &Value) -> Ordering {
    let blank_like = |other: &Value| match other {
        Value::Text(_) => Value::Text(String::new()),
        Value::Bool(_) => Value::Bool(false),
        _ => Value::Number(0.0),
    };
    let a = if *a == Value::Empty {
        blank_like(b)
    } else {
        a.clone()
    };
    let b = if *b == Value::Empty {
        blank_like(&a)
    } else {
        b.clone()
    };
    let rank = |v: &Value| match v {
        Value::Number(_) | Value::Empty => 0,
        Value::Text(_) => 1,
        Value::Bool(_) => 2,
    };
    match (&a, &b) {
        (Value::Number(x), Value::Number(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (Value::Text(x), Value::Text(y)) => x.to_lowercase().cmp(&y.to_lowercase()),
        (Value::Bool(x), Value::Bool(y)) => x.cmp(y),
        _ => rank(&a).cmp(&rank(&b)),
    }
}

/// A numeric result for the numeric API: text results are a type
/// error (they have no number), booleans are 1/0, empty is 0.
fn numeric(result: Result<Value, EvalError>) -> Result<f64, EvalError> {
    match result? {
        Value::Text(s) => Err(EvalError::Type(format!(
            "text result {s:?} is not a number"
        ))),
        v => to_number(&v),
    }
}

/// Evaluate one cell: numbers pass through, formulas are computed
/// recursively, text/empty evaluate to errors in numeric context.
pub const MAX_EVAL_DEPTH: usize = 256;

pub fn eval_cell(sheet: &SheetSnapshot, row: usize, col: usize) -> Result<f64, EvalError> {
    let sheets: Vec<&SheetSnapshot> = vec![sheet];
    let mut state = EvalState::default();
    numeric(eval_cell_inner(&sheets, 0, row, col, &mut state, 0))
}

/// Evaluate a cell in a named sheet of a workbook (cross-sheet refs).
pub fn eval_cell_in_workbook(
    wb: &crate::WorkbookSnapshot,
    sheet_name: &str,
    row: usize,
    col: usize,
) -> Result<f64, EvalError> {
    numeric(eval_cell_value_in_workbook(wb, sheet_name, row, col))
}

/// Like [`eval_cell_in_workbook`], but returns the cell's value as it
/// is — number, text or boolean.
pub fn eval_cell_value_in_workbook(
    wb: &crate::WorkbookSnapshot,
    sheet_name: &str,
    row: usize,
    col: usize,
) -> Result<Value, EvalError> {
    let sheets: Vec<&SheetSnapshot> = wb.sheets.iter().collect();
    let idx = sheets
        .iter()
        .position(|s| crate::sheet_name_eq(&s.name, sheet_name))
        .ok_or_else(|| EvalError::Ref(format!("unknown sheet {sheet_name}")))?;
    let mut state = EvalState::default();
    eval_cell_inner(&sheets, idx, row, col, &mut state, 0)
}

#[derive(Default)]
struct EvalState {
    visited: BTreeSet<(usize, usize, usize)>,
    memo: std::collections::HashMap<(usize, usize, usize), Value>,
    /// A single simulated cell override (what-if), applied without
    /// cloning the workbook.
    override_cell: Option<(usize, usize, usize, f64)>,
}

fn eval_cell_inner(
    sheets: &[&SheetSnapshot],
    sheet_idx: usize,
    row: usize,
    col: usize,
    state: &mut EvalState,
    depth: usize,
) -> Result<Value, EvalError> {
    if depth > MAX_EVAL_DEPTH {
        return Err(EvalError::Depth(format!(
            "dependency chain deeper than {MAX_EVAL_DEPTH} cells"
        )));
    }
    let key = (sheet_idx, row, col);
    if let Some((os, orow, ocol, value)) = state.override_cell {
        if (os, orow, ocol) == key {
            return Ok(Value::Number(value));
        }
    }
    if let Some(v) = state.memo.get(&key) {
        return Ok(v.clone());
    }
    if !state.visited.insert(key) {
        return Err(EvalError::Cycle(format!(
            "cell {}{}",
            crate::col_name(col),
            row + 1
        )));
    }
    let sheet = sheets[sheet_idx];
    // No early returns between insert and remove: the visited entry is
    // always removed, and successful results are memoised so shared
    // precedents are evaluated once per top-level evaluation.
    let result = eval_cell_value(sheets, sheet_idx, sheet, row, col, state, depth);
    state.visited.remove(&key);
    if let Ok(v) = &result {
        state.memo.insert(key, v.clone());
    }
    result
}

fn eval_cell_value(
    sheets: &[&SheetSnapshot],
    sheet_idx: usize,
    sheet: &SheetSnapshot,
    row: usize,
    col: usize,
    state: &mut EvalState,
    depth: usize,
) -> Result<Value, EvalError> {
    let cell = sheet.get(row, col);
    let CellValue::Formula { expr, cached } = cell else {
        return Ok(Value::of_cell(&cell).unwrap_or(Value::Empty));
    };
    let mut parser = Parser {
        chars: expr.chars().collect(),
        pos: 0,
        sheets,
        sheet_idx,
        state,
        depth,
        nest: 0,
        skip: 0,
    };
    let evaluated = (|| -> Result<Value, EvalError> {
        let v = parser.parse_expr()?;
        parser.skip_ws();
        if parser.pos != parser.chars.len() {
            return Err(EvalError::Parse(format!(
                "trailing tokens at {} in formula",
                parser.pos
            )));
        }
        Ok(v)
    })();
    match evaluated {
        // `=A1` of an empty A1 is 0, as in Excel.
        Ok(Value::Empty) => Ok(Value::Number(0.0)),
        Ok(v) => Ok(v),
        Err(e) => {
            // The expression is the truth; a cached value is only a
            // fallback for formulas we cannot evaluate (unsupported
            // functions). Genuine errors — cycles, type/ref/parse
            // failures in an evaluable formula — propagate, never
            // masked by a possibly stale cached value.
            let unevaluable =
                matches!(&e, EvalError::Parse(m) if m.contains("unsupported function"));
            match cached.as_deref().and_then(Value::of_cell) {
                Some(v) if unevaluable && v != Value::Empty => Ok(v),
                _ => Err(e),
            }
        }
    }
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    sheets: &'a [&'a SheetSnapshot],
    sheet_idx: usize,
    state: &'a mut EvalState,
    depth: usize,
    nest: usize,
    /// Above zero while parsing a branch that is not taken (the
    /// unchosen side of IF, IFERROR's fallback when the value is
    /// fine): syntax is checked, but nothing is evaluated, so the
    /// branch's errors and cycles do not matter.
    skip: usize,
}

/// One function argument.
enum Arg {
    /// A single value; `is_ref` when the argument was a lone cell
    /// reference (aggregates then skip non-numbers, as with ranges).
    Scalar { value: Value, is_ref: bool },
    /// The cells of a range, row by row.
    Range(Vec<Value>),
}

/// Functions the evaluator implements (IF/IFERROR/ISERROR parse
/// their own arguments lazily).
const FUNCTIONS: &[&str] = &[
    "SUM",
    "AVERAGE",
    "AVG",
    "MIN",
    "MAX",
    "COUNT",
    "AND",
    "OR",
    "NOT",
    "CONCAT",
    "CONCATENATE",
    "LEN",
    "UPPER",
    "LOWER",
    "TRIM",
    "LEFT",
    "RIGHT",
    "MID",
    "ROUND",
    "ABS",
    "ISBLANK",
    "ISNUMBER",
    "ISTEXT",
];

const MAX_PARSE_NEST: usize = 64;

#[derive(Clone, Copy)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CmpOp {
    fn holds(self, ord: Ordering) -> bool {
        match self {
            CmpOp::Eq => ord == Ordering::Equal,
            CmpOp::Ne => ord != Ordering::Equal,
            CmpOp::Lt => ord == Ordering::Less,
            CmpOp::Le => ord != Ordering::Greater,
            CmpOp::Gt => ord == Ordering::Greater,
            CmpOp::Ge => ord != Ordering::Less,
        }
    }
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }
    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }
    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }
    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }
    fn expect(&mut self, c: char) -> Result<(), EvalError> {
        self.skip_ws();
        if self.bump() == Some(c) {
            Ok(())
        } else {
            Err(EvalError::Parse(format!("expected '{c}' at {}", self.pos)))
        }
    }
    fn enter(&mut self) -> Result<(), EvalError> {
        self.nest += 1;
        if self.nest > MAX_PARSE_NEST {
            return Err(EvalError::Parse(format!(
                "formula nesting deeper than {MAX_PARSE_NEST}"
            )));
        }
        Ok(())
    }
    fn skipping(&self) -> bool {
        self.skip > 0
    }
    /// Arithmetic coercion; never fails inside a skipped branch.
    fn num(&self, v: &Value) -> Result<f64, EvalError> {
        if self.skipping() {
            return Ok(to_number(v).unwrap_or(0.0));
        }
        to_number(v)
    }

    /// Parse one expression, evaluating it unless `skip` (then it is
    /// only syntax-checked and yields Empty).
    fn parse_lazy(&mut self, skip: bool) -> Result<Value, EvalError> {
        if !skip {
            return self.parse_expr();
        }
        self.skip += 1;
        let result = self.parse_expr();
        self.skip -= 1;
        result.map(|_| Value::Empty)
    }

    /// Full expression: comparisons bind loosest.
    fn parse_expr(&mut self) -> Result<Value, EvalError> {
        let mut v = self.parse_concat()?;
        loop {
            self.skip_ws();
            let (op, len) = match (self.peek(), self.peek_at(1)) {
                (Some('<'), Some('>')) => (CmpOp::Ne, 2),
                (Some('<'), Some('=')) => (CmpOp::Le, 2),
                (Some('>'), Some('=')) => (CmpOp::Ge, 2),
                (Some('<'), _) => (CmpOp::Lt, 1),
                (Some('>'), _) => (CmpOp::Gt, 1),
                (Some('='), _) => (CmpOp::Eq, 1),
                _ => return Ok(v),
            };
            self.pos += len;
            let rhs = self.parse_concat()?;
            v = if self.skipping() {
                Value::Empty
            } else {
                Value::Bool(op.holds(compare(&v, &rhs)))
            };
        }
    }

    /// `&` text concatenation.
    fn parse_concat(&mut self) -> Result<Value, EvalError> {
        let mut v = self.parse_additive()?;
        loop {
            self.skip_ws();
            if self.peek() != Some('&') {
                return Ok(v);
            }
            self.bump();
            let rhs = self.parse_additive()?;
            v = Value::Text(to_text(&v) + &to_text(&rhs));
        }
    }

    fn parse_additive(&mut self) -> Result<Value, EvalError> {
        let mut v = self.parse_term()?;
        loop {
            self.skip_ws();
            let sign = match self.peek() {
                Some('+') => 1.0,
                Some('-') => -1.0,
                _ => return Ok(v),
            };
            self.bump();
            let rhs = self.parse_term()?;
            v = Value::Number(self.num(&v)? + sign * self.num(&rhs)?);
        }
    }

    fn parse_power(&mut self) -> Result<Value, EvalError> {
        let base = self.parse_factor()?;
        self.skip_ws();
        if self.peek() == Some('^') {
            self.bump();
            self.enter()?;
            let exp = self.parse_power()?;
            self.nest -= 1;
            return Ok(Value::Number(self.num(&base)?.powf(self.num(&exp)?)));
        }
        Ok(base)
    }

    fn parse_term(&mut self) -> Result<Value, EvalError> {
        let mut v = self.parse_power()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('*') => {
                    self.bump();
                    let rhs = self.parse_power()?;
                    v = Value::Number(self.num(&v)? * self.num(&rhs)?);
                }
                Some('/') => {
                    self.bump();
                    let rhs = self.parse_power()?;
                    let d = self.num(&rhs)?;
                    if d == 0.0 {
                        if self.skipping() {
                            v = Value::Empty;
                            continue;
                        }
                        return Err(EvalError::Type("division by zero".into()));
                    }
                    v = Value::Number(self.num(&v)? / d);
                }
                _ => return Ok(v),
            }
        }
    }

    /// A `"..."` string literal (`""` is an escaped quote).
    fn parse_string(&mut self) -> Result<Value, EvalError> {
        self.bump(); // opening quote
        let mut s = String::new();
        loop {
            match self.bump() {
                Some('"') if self.peek() == Some('"') => {
                    self.bump();
                    s.push('"');
                }
                Some('"') => return Ok(Value::Text(s)),
                Some(c) => s.push(c),
                None => return Err(EvalError::Parse("unterminated string literal".into())),
            }
        }
    }

    fn parse_factor(&mut self) -> Result<Value, EvalError> {
        self.skip_ws();
        let mut v = match self.peek() {
            Some('-') => {
                self.bump();
                self.enter()?;
                let v = self.parse_factor()?;
                self.nest -= 1;
                Value::Number(-self.num(&v)?)
            }
            Some('+') => {
                self.bump();
                self.enter()?;
                let v = self.parse_factor()?;
                self.nest -= 1;
                v
            }
            Some('(') => {
                self.bump();
                self.enter()?;
                let v = self.parse_expr()?;
                self.expect(')')?;
                self.nest -= 1;
                v
            }
            Some('"') => self.parse_string()?,
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.pos;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.') {
                    self.bump();
                }
                // Scientific notation exponent.
                if matches!(self.peek(), Some('e') | Some('E')) {
                    let save = self.pos;
                    self.bump();
                    if matches!(self.peek(), Some('+') | Some('-')) {
                        self.bump();
                    }
                    if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                            self.bump();
                        }
                    } else {
                        self.pos = save;
                    }
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                Value::Number(
                    s.parse::<f64>()
                        .map_err(|_| EvalError::Parse(format!("bad number {s}")))?,
                )
            }
            Some(c) if c.is_alphabetic() || c == '_' || c == '$' || c == '\'' => {
                // Optional sheet qualifier: Name! or 'My Sheet'!
                let mut target_sheet = self.sheet_idx;
                if c == '\'' || c.is_alphabetic() || c == '_' {
                    if let Some((idx, next)) = self.try_sheet_qualifier() {
                        target_sheet = idx;
                        self.pos = next;
                    }
                }
                if self.peek() == Some('$') {
                    self.bump();
                }
                let start = self.pos;
                while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic()) {
                    self.bump();
                }
                let mut name: String = self.chars[start..self.pos].iter().collect();
                // Digit-bearing function names (LOG10(...)): extend
                // the token over digits when a '(' follows, else it
                // is a cell ref (LOG10 the cell) as before.
                if matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                    let save = self.pos;
                    while matches!(self.peek(), Some(c) if c.is_ascii_alphanumeric()) {
                        self.bump();
                    }
                    let extended: String = self.chars[start..self.pos].iter().collect();
                    self.skip_ws();
                    if self.peek() == Some('(') {
                        name = extended;
                    } else {
                        self.pos = save;
                    }
                }
                self.skip_ws();
                if name.eq_ignore_ascii_case("TRUE") || name.eq_ignore_ascii_case("FALSE") {
                    let v = Value::Bool(name.eq_ignore_ascii_case("TRUE"));
                    if self.peek() == Some('(') {
                        // TRUE() / FALSE() literal form.
                        let save = self.pos;
                        self.bump();
                        self.skip_ws();
                        if self.peek() == Some(')') {
                            self.bump();
                            return Ok(v);
                        }
                        self.pos = save; // not the literal form; fall through
                    } else {
                        return Ok(v);
                    }
                }
                if self.peek() == Some('(') {
                    if target_sheet != self.sheet_idx {
                        return Err(EvalError::Parse(
                            "sheet-qualified function calls are not supported".into(),
                        ));
                    }
                    self.bump();
                    self.enter()?;
                    let v = self.parse_function(&name.to_ascii_uppercase())?;
                    self.nest -= 1;
                    v
                } else {
                    let (_, row, col) = self.finish_ref(&name)?;
                    if self.skipping() {
                        Value::Empty
                    } else {
                        eval_cell_inner(
                            self.sheets,
                            target_sheet,
                            row,
                            col,
                            self.state,
                            self.depth + 1,
                        )?
                    }
                }
            }
            other => {
                return Err(EvalError::Parse(format!(
                    "unexpected character {other:?} at {}",
                    self.pos
                )))
            }
        };
        // Percent postfix: A2% means A2/100.
        loop {
            self.skip_ws();
            if self.peek() == Some('%') {
                self.bump();
                v = Value::Number(self.num(&v)? / 100.0);
            } else {
                break;
            }
        }
        Ok(v)
    }

    /// If a `Name!` / `'Quoted Name'!` qualifier starts at `pos`,
    /// return (sheet index, position after '!').
    #[allow(clippy::needless_late_init)] // assigned in both branches
    fn try_sheet_qualifier(&self) -> Option<(usize, usize)> {
        let mut i = self.pos;
        let name: String;
        if self.chars.get(i) == Some(&'\'') {
            i += 1;
            let mut n = String::new();
            loop {
                match self.chars.get(i) {
                    Some('\'') => {
                        if self.chars.get(i + 1) == Some(&'\'') {
                            n.push('\'');
                            i += 2;
                        } else {
                            i += 1;
                            break;
                        }
                    }
                    Some(c) => {
                        n.push(*c);
                        i += 1;
                    }
                    None => return None,
                }
            }
            name = n;
        } else {
            let start = i;
            while i < self.chars.len()
                && (self.chars[i].is_alphanumeric() || self.chars[i] == '_' || self.chars[i] == '.')
            {
                i += 1;
            }
            if i == start {
                return None;
            }
            name = self.chars[start..i].iter().collect();
        }
        if self.chars.get(i) != Some(&'!') {
            return None;
        }
        let idx = self
            .sheets
            .iter()
            .position(|s| crate::sheet_name_eq(&s.name, &name))?;
        Some((idx, i + 1))
    }

    /// After reading leading letters, read optional $ and digits.
    fn finish_ref(&mut self, letters: &str) -> Result<(String, usize, usize), EvalError> {
        if letters.chars().all(|c| c == '$') || letters.is_empty() {
            return Err(EvalError::Ref(format!("{letters} has no column")));
        }
        if self.peek() == Some('$') {
            self.bump();
        }
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.bump();
        }
        if start == self.pos {
            return Err(EvalError::Ref(format!("{letters} has no row")));
        }
        let digits: String = self.chars[start..self.pos].iter().collect();
        let row1: usize = digits.parse().map_err(|_| EvalError::Ref(digits.clone()))?;
        if row1 == 0 || row1 > 1_048_576 {
            return Err(EvalError::Ref(format!(
                "row {row1} is outside the Excel grid"
            )));
        }
        let mut col = 0usize;
        for c in letters
            .chars()
            .filter(|c| *c != '$')
            .collect::<String>()
            .to_ascii_uppercase()
            .chars()
        {
            col = col
                .checked_mul(26)
                .and_then(|v| v.checked_add(c as usize - 'A' as usize + 1))
                .ok_or_else(|| EvalError::Ref("column too large".into()))?;
        }
        if col == 0 || col > 16_384 {
            return Err(EvalError::Ref(format!(
                "{letters} is outside the Excel grid (max column XFD)"
            )));
        }
        Ok((format!("{letters}{digits}"), row1 - 1, col - 1))
    }

    fn parse_ref_token(&mut self) -> Result<(usize, usize, usize, bool), EvalError> {
        self.skip_ws();
        let mut target = self.sheet_idx;
        let mut qualified = false;
        if matches!(self.peek(), Some('\''))
            || matches!(self.peek(), Some(c) if c.is_alphabetic() || c == '_')
        {
            if let Some((idx, next)) = self.try_sheet_qualifier() {
                target = idx;
                qualified = true;
                self.pos = next;
            }
        }
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic() || c == '$') {
            self.bump();
        }
        let letters: String = self.chars[start..self.pos].iter().collect();
        let (_, row, col) = self.finish_ref(&letters)?;
        Ok((target, row, col, qualified))
    }

    /// One scalar argument. A lone cell reference is flagged so that
    /// aggregates skip it when it is not a number (COUNT of an empty
    /// cell is 0, AVERAGE ignores text) — matching range arguments.
    fn parse_scalar_arg(&mut self) -> Result<Arg, EvalError> {
        let save = self.pos;
        let is_ref = self.parse_ref_token().is_ok() && {
            self.skip_ws();
            matches!(self.peek(), Some(',') | Some(')'))
        };
        self.pos = save;
        let value = self.parse_expr()?;
        Ok(Arg::Scalar { value, is_ref })
    }

    fn collect_range(
        &mut self,
        sheet: usize,
        r0: usize,
        c0: usize,
        r1: usize,
        c1: usize,
    ) -> Result<Vec<Value>, EvalError> {
        let mut values = Vec::new();
        if self.skipping() {
            return Ok(values);
        }
        for r in r0.min(r1)..=r0.max(r1) {
            for c in c0.min(c1)..=c0.max(c1) {
                let cell = self.sheets[sheet].get_ref(r, c);
                let overridden = self
                    .state
                    .override_cell
                    .is_some_and(|(os, orow, ocol, _)| (os, orow, ocol) == (sheet, r, c));
                // Plain cells need no evaluation (or cycle tracking).
                if !overridden {
                    if let Some(v) = Value::of_cell(cell) {
                        values.push(v);
                        continue;
                    }
                }
                values.push(eval_cell_inner(
                    self.sheets,
                    sheet,
                    r,
                    c,
                    self.state,
                    self.depth + 1,
                )?);
            }
        }
        Ok(values)
    }

    /// Whole-column (A:A) / whole-row (1:1) range at the current
    /// position, bounded to the target sheet's used extents.
    /// Consumes the token on success; restores position otherwise.
    /// `Some(None)` is a recognised range that covers no used cell
    /// (empty sheet, or rows/columns past the used extents).
    #[allow(clippy::type_complexity)]
    fn try_whole_range(&mut self) -> Option<Option<(usize, usize, usize, usize, usize)>> {
        let save = self.pos;
        let mut target = self.sheet_idx;
        if matches!(self.peek(), Some('\''))
            || matches!(self.peek(), Some(c) if c.is_alphabetic() || c == '_')
        {
            if let Some((idx, next)) = self.try_sheet_qualifier() {
                target = idx;
                self.pos = next;
            }
        }
        let sheet = self.sheets[target];
        let col_of = |letters: &str| -> Option<usize> {
            let mut col = 0usize;
            for c in letters.to_ascii_uppercase().chars() {
                if !c.is_ascii_alphabetic() {
                    return None;
                }
                col = col
                    .checked_mul(26)?
                    .checked_add(c as usize - 'A' as usize + 1)?;
            }
            if col == 0 || col > 16_384 {
                return None;
            }
            Some(col - 1)
        };
        let take_letters = |p: &mut Self| -> String {
            let st = p.pos;
            if p.peek() == Some('$') {
                p.bump();
            }
            while matches!(p.peek(), Some(c) if c.is_ascii_alphabetic()) {
                p.bump();
            }
            p.chars[st..p.pos]
                .iter()
                .collect::<String>()
                .replace('$', "")
        };
        let take_digits = |p: &mut Self| -> String {
            let st = p.pos;
            if p.peek() == Some('$') {
                p.bump();
            }
            while matches!(p.peek(), Some(c) if c.is_ascii_digit()) {
                p.bump();
            }
            p.chars[st..p.pos]
                .iter()
                .collect::<String>()
                .replace('$', "")
        };
        // Column form: AA:BB
        let after_qual = self.pos;
        let first = take_letters(self);
        if !first.is_empty() && self.peek() == Some(':') {
            self.bump();
            // An end qualifier, if present, must match (skip it
            // syntactically; the start sheet governs).
            if let Some((_, next)) = self.try_sheet_qualifier() {
                self.pos = next;
            }
            let second = take_letters(self);
            if let (Some(c0), Some(c1)) = (col_of(&first), col_of(&second)) {
                let (lo, hi) = (
                    c0.min(c1),
                    c0.max(c1).min(sheet.max_col().saturating_sub(1)),
                );
                if sheet.cells.is_empty() || sheet.max_col() == 0 || lo > hi {
                    return Some(None);
                }
                return Some(Some((target, 0, lo, sheet.cells.len() - 1, hi)));
            }
            self.pos = save;
            return None;
        }
        // Row form: 1:3
        self.pos = after_qual;
        let first = take_digits(self);
        if !first.is_empty() && self.peek() == Some(':') {
            self.bump();
            if let Some((_, next)) = self.try_sheet_qualifier() {
                self.pos = next;
            }
            let second = take_digits(self);
            if let (Ok(r0), Ok(r1)) = (first.parse::<usize>(), second.parse::<usize>()) {
                if r0 >= 1 && r1 >= 1 && r0 <= 1_048_576 && r1 <= 1_048_576 {
                    let lo = r0.min(r1) - 1;
                    let hi = (r0.max(r1) - 1).min(sheet.cells.len().saturating_sub(1));
                    if sheet.cells.is_empty() || sheet.max_col() == 0 || lo > hi {
                        return Some(None);
                    }
                    return Some(Some((target, lo, 0, hi, sheet.max_col() - 1)));
                }
            }
        }
        self.pos = save;
        None
    }

    /// Arguments up to and including the closing ')': ranges (A1:B9,
    /// A:A, 1:1) or scalar expressions.
    fn parse_args(&mut self, name: &str) -> Result<Vec<Arg>, EvalError> {
        let mut args = Vec::new();
        self.skip_ws();
        if self.peek() == Some(')') {
            self.bump();
            return Ok(args);
        }
        loop {
            self.skip_ws();
            let save = self.pos;
            // Whole-column/row ranges (A:A, 1:1) bounded to the
            // sheet's used extents.
            if let Some(whole) = self.try_whole_range() {
                match whole {
                    Some((t, r0, c0, r1, c1)) => {
                        let area = (r1 - r0 + 1).saturating_mul(c1 - c0 + 1);
                        if area > 100_000 {
                            return Err(EvalError::Parse(format!(
                                "range of {area} cells exceeds the 100000-cell evaluation cap"
                            )));
                        }
                        args.push(Arg::Range(self.collect_range(t, r0, c0, r1, c1)?));
                    }
                    None => args.push(Arg::Range(Vec::new())),
                }
            // Try range A1:B9 next.
            } else if let Ok((t0, r0, c0, _)) = self.parse_ref_token() {
                self.skip_ws();
                if self.peek() == Some(':') {
                    self.bump();
                    let (t1, r1, c1, end_qualified) = self.parse_ref_token()?;
                    let area = (r0.max(r1) - r0.min(r1) + 1) * (c0.max(c1) - c0.min(c1) + 1);
                    if area > 100_000 {
                        return Err(EvalError::Parse(format!(
                            "range of {area} cells exceeds the 100000-cell evaluation cap"
                        )));
                    }
                    // An unqualified range end inherits the start
                    // sheet (Sheet2!A1:B2); only an explicitly
                    // different end sheet is an error.
                    if t1 != t0 && end_qualified {
                        return Err(EvalError::Parse(
                            "range ends on a different sheet than it starts; qualify only the start (Sheet2!A1:B2)"
                                .into(),
                        ));
                    }
                    args.push(Arg::Range(self.collect_range(t0, r0, c0, r1, c1)?));
                } else {
                    self.pos = save;
                    args.push(self.parse_scalar_arg()?);
                }
            } else {
                self.pos = save;
                args.push(self.parse_scalar_arg()?);
            }
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(')') => return Ok(args),
                other => {
                    return Err(EvalError::Parse(format!(
                        "expected ',' or ')' in {name}, got {other:?}"
                    )))
                }
            }
        }
    }

    fn parse_function(&mut self, name: &str) -> Result<Value, EvalError> {
        match name {
            "IF" => return self.fn_if(),
            "IFERROR" => return self.fn_iferror(),
            "ISERROR" => return self.fn_iserror(),
            _ => {}
        }
        // Unsupported names fail BEFORE any argument is evaluated,
        // so an argument error cannot mask the cached-value
        // fallback for formulas we cannot evaluate at all.
        if !FUNCTIONS.contains(&name) {
            return Err(EvalError::Parse(format!("unsupported function {name}")));
        }
        let args = self.parse_args(name)?;
        if self.skipping() {
            return Ok(Value::Empty);
        }
        apply_function(name, &args)
    }

    /// IF(condition, then, [else]): only the chosen branch is
    /// evaluated. A missing else gives FALSE.
    fn fn_if(&mut self) -> Result<Value, EvalError> {
        let outer_skip = self.skipping();
        let cond = self.parse_expr()?;
        let take = outer_skip || to_bool(&cond)?;
        self.expect(',')?;
        let then_v = self.parse_lazy(!take)?;
        self.skip_ws();
        let else_v = if self.peek() == Some(',') {
            self.bump();
            Some(self.parse_lazy(take)?)
        } else {
            None
        };
        self.expect(')')?;
        if outer_skip {
            return Ok(Value::Empty);
        }
        Ok(if take {
            then_v
        } else {
            else_v.unwrap_or(Value::Bool(false))
        })
    }

    /// Evaluate one argument, turning a value error (bad reference,
    /// type error, division by zero) into `None`. Syntax errors,
    /// unsupported functions, cycles and depth errors still fail.
    fn parse_catching(&mut self) -> Result<Option<Value>, EvalError> {
        let (pos, nest) = (self.pos, self.nest);
        match self.parse_expr() {
            Ok(v) => Ok(Some(v)),
            Err(EvalError::Ref(_) | EvalError::Type(_)) if !self.skipping() => {
                // Re-read the argument without evaluating it to find
                // where it ends.
                self.pos = pos;
                self.nest = nest;
                self.parse_lazy(true)?;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// IFERROR(value, fallback): the fallback is evaluated only when
    /// the value is an error.
    fn fn_iferror(&mut self) -> Result<Value, EvalError> {
        let value = self.parse_catching()?;
        self.expect(',')?;
        let fallback = self.parse_lazy(value.is_some())?;
        self.expect(')')?;
        if self.skipping() {
            return Ok(Value::Empty);
        }
        Ok(match value {
            Some(Value::Empty) => Value::Number(0.0),
            Some(v) => v,
            None => fallback,
        })
    }

    fn fn_iserror(&mut self) -> Result<Value, EvalError> {
        let value = self.parse_catching()?;
        self.expect(')')?;
        Ok(Value::Bool(value.is_none()))
    }
}

/// Numbers among the arguments for an aggregate: range cells and
/// lone references count only when they hold numbers; other scalar
/// arguments are coerced (SUM("3", TRUE) = 4) unless `lenient`, when
/// an uncoercible one is skipped (COUNT("x") = 0).
fn numbers(args: &[Arg], lenient: bool) -> Result<Vec<f64>, EvalError> {
    let mut out = Vec::new();
    for arg in args {
        match arg {
            Arg::Range(values) => out.extend(values.iter().filter_map(|v| match v {
                Value::Number(n) => Some(*n),
                _ => None,
            })),
            Arg::Scalar {
                value,
                is_ref: true,
            } => {
                if let Value::Number(n) = value {
                    out.push(*n);
                }
            }
            Arg::Scalar { value, .. } => match to_number(value) {
                Ok(n) => out.push(n),
                Err(_) if lenient => {}
                Err(e) => return Err(e),
            },
        }
    }
    Ok(out)
}

/// The single value of argument `i` of `name`.
fn scalar<'v>(name: &str, args: &'v [Arg], i: usize) -> Result<&'v Value, EvalError> {
    match args.get(i) {
        Some(Arg::Scalar { value, .. }) => Ok(value),
        Some(Arg::Range(values)) if values.len() == 1 => Ok(&values[0]),
        Some(Arg::Range(_)) => Err(EvalError::Type(format!(
            "{name} expects a single value, not a range"
        ))),
        None => Err(EvalError::Parse(format!(
            "{name} is missing argument {}",
            i + 1
        ))),
    }
}

fn arity(name: &str, args: &[Arg], min: usize, max: usize) -> Result<(), EvalError> {
    if args.len() < min || args.len() > max {
        let expected = if min == max {
            format!("{min}")
        } else {
            format!("{min} to {max}")
        };
        return Err(EvalError::Parse(format!(
            "{name} takes {expected} arguments, got {}",
            args.len()
        )));
    }
    Ok(())
}

/// A whole-number count argument (LEFT's length, MID's start).
fn count_arg(name: &str, v: &Value) -> Result<usize, EvalError> {
    let n = to_number(v)?.floor();
    if n < 0.0 {
        return Err(EvalError::Type(format!(
            "{name} needs a non-negative count"
        )));
    }
    Ok(n.min(u32::MAX as f64) as usize)
}

fn apply_function(name: &str, args: &[Arg]) -> Result<Value, EvalError> {
    let text = |i: usize| scalar(name, args, i).map(to_text);
    let number = |i: usize| scalar(name, args, i).and_then(to_number);
    let num = Value::Number;
    match name {
        "SUM" => Ok(num(numbers(args, false)?.iter().sum())),
        "AVERAGE" | "AVG" => {
            let values = numbers(args, false)?;
            if values.is_empty() {
                Err(EvalError::Type("AVERAGE of nothing".into()))
            } else {
                Ok(num(values.iter().sum::<f64>() / values.len() as f64))
            }
        }
        // Excel returns 0 for MIN/MAX over no numbers.
        "MIN" => Ok(num(numbers(args, false)?
            .into_iter()
            .reduce(f64::min)
            .unwrap_or(0.0))),
        "MAX" => Ok(num(numbers(args, false)?
            .into_iter()
            .reduce(f64::max)
            .unwrap_or(0.0))),
        "COUNT" => Ok(num(numbers(args, true)?.len() as f64)),
        "AND" | "OR" => {
            let mut seen = false;
            let mut all = true;
            let mut any = false;
            for arg in args {
                let mut take = |b: bool| {
                    seen = true;
                    all &= b;
                    any |= b;
                };
                match arg {
                    // Ranges and references: text and empty cells
                    // are ignored.
                    Arg::Range(values) => {
                        for v in values {
                            if matches!(v, Value::Number(_) | Value::Bool(_)) {
                                take(to_bool(v)?);
                            }
                        }
                    }
                    Arg::Scalar {
                        value,
                        is_ref: true,
                    } => {
                        if matches!(value, Value::Number(_) | Value::Bool(_)) {
                            take(to_bool(value)?);
                        }
                    }
                    Arg::Scalar { value, .. } => take(to_bool(value)?),
                }
            }
            if !seen {
                return Err(EvalError::Type(format!("{name} has no logical values")));
            }
            Ok(Value::Bool(if name == "AND" { all } else { any }))
        }
        "NOT" => {
            arity(name, args, 1, 1)?;
            Ok(Value::Bool(!to_bool(scalar(name, args, 0)?)?))
        }
        "CONCAT" | "CONCATENATE" => {
            let mut out = String::new();
            for arg in args {
                match arg {
                    Arg::Range(values) => values.iter().for_each(|v| out.push_str(&to_text(v))),
                    Arg::Scalar { value, .. } => out.push_str(&to_text(value)),
                }
            }
            Ok(Value::Text(out))
        }
        "LEN" => {
            arity(name, args, 1, 1)?;
            Ok(num(text(0)?.chars().count() as f64))
        }
        "UPPER" => {
            arity(name, args, 1, 1)?;
            Ok(Value::Text(text(0)?.to_uppercase()))
        }
        "LOWER" => {
            arity(name, args, 1, 1)?;
            Ok(Value::Text(text(0)?.to_lowercase()))
        }
        "TRIM" => {
            arity(name, args, 1, 1)?;
            Ok(Value::Text(
                text(0)?.split_whitespace().collect::<Vec<_>>().join(" "),
            ))
        }
        "LEFT" | "RIGHT" => {
            arity(name, args, 1, 2)?;
            let s: Vec<char> = text(0)?.chars().collect();
            let n = match args.len() {
                2 => count_arg(name, scalar(name, args, 1)?)?,
                _ => 1,
            }
            .min(s.len());
            let part = if name == "LEFT" {
                &s[..n]
            } else {
                &s[s.len() - n..]
            };
            Ok(Value::Text(part.iter().collect()))
        }
        "MID" => {
            arity(name, args, 3, 3)?;
            let s: Vec<char> = text(0)?.chars().collect();
            let start = count_arg(name, scalar(name, args, 1)?)?;
            if start == 0 {
                return Err(EvalError::Type("MID start must be at least 1".into()));
            }
            let len = count_arg(name, scalar(name, args, 2)?)?;
            Ok(Value::Text(
                s.iter().skip(start - 1).take(len).collect::<String>(),
            ))
        }
        "ROUND" => {
            arity(name, args, 1, 2)?;
            let x = number(0)?;
            let digits = match args.len() {
                2 => number(1)?.trunc().clamp(-15.0, 15.0),
                _ => 0.0,
            };
            let scale = 10f64.powf(digits);
            // f64::round rounds half away from zero, like Excel.
            Ok(num((x * scale).round() / scale))
        }
        "ABS" => {
            arity(name, args, 1, 1)?;
            Ok(num(number(0)?.abs()))
        }
        "ISBLANK" | "ISNUMBER" | "ISTEXT" => {
            arity(name, args, 1, 1)?;
            let v = scalar(name, args, 0)?;
            Ok(Value::Bool(match name {
                "ISBLANK" => *v == Value::Empty,
                "ISNUMBER" => matches!(v, Value::Number(_)),
                _ => matches!(v, Value::Text(_)),
            }))
        }
        other => Err(EvalError::Parse(format!("unsupported function {other}"))),
    }
}

/// Evaluate a target cell with one simulated override applied,
/// without cloning the workbook.
pub fn eval_cell_with_override(
    wb: &crate::WorkbookSnapshot,
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
    wb: &crate::WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<Value, EvalError> {
    if !value.is_finite() {
        return Err(EvalError::Type("what-if value must be finite".into()));
    }
    for (sheet, r, c) in [override_ref, target_ref] {
        if r > 10_000 || c > 1_000 {
            return Err(EvalError::Ref(format!(
                "coordinates beyond the evaluation grid on {sheet}"
            )));
        }
    }
    let sheets: Vec<&SheetSnapshot> = wb.sheets.iter().collect();
    let find = |name: &str| {
        sheets
            .iter()
            .position(|s| crate::sheet_name_eq(&s.name, name))
            .ok_or_else(|| EvalError::Ref(format!("unknown sheet {name}")))
    };
    let mut state = EvalState {
        override_cell: Some((find(override_ref.0)?, override_ref.1, override_ref.2, value)),
        ..EvalState::default()
    };
    eval_cell_inner(
        &sheets,
        find(target_ref.0)?,
        target_ref.1,
        target_ref.2,
        &mut state,
        0,
    )
}

/// Evaluate several cells of one sheet sharing a single memo, so
/// shared precedent chains are walked once across the batch.
pub fn eval_many(sheet: &SheetSnapshot, coords: &[(usize, usize)]) -> Vec<Result<f64, EvalError>> {
    let sheets: Vec<&SheetSnapshot> = vec![sheet];
    let mut state = EvalState::default();
    coords
        .iter()
        .map(|(r, c)| numeric(eval_cell_inner(&sheets, 0, *r, *c, &mut state, 0)))
        .collect()
}

/// Evaluate several cells of one named sheet with a shared memo
/// (cross-sheet precedents resolve through the workbook). Numeric
/// results only: a text result is a type error; see
/// [`eval_many_values_in_workbook`] for any value.
pub fn eval_many_in_workbook(
    wb: &crate::WorkbookSnapshot,
    sheet_name: &str,
    coords: &[(usize, usize)],
) -> Vec<Result<f64, EvalError>> {
    eval_many_values_in_workbook(wb, sheet_name, coords)
        .into_iter()
        .map(numeric)
        .collect()
}

/// Like [`eval_many_in_workbook`], but returns each cell's value as
/// it is — number, text or boolean (an empty cell is `Empty`).
pub fn eval_many_values_in_workbook(
    wb: &crate::WorkbookSnapshot,
    sheet_name: &str,
    coords: &[(usize, usize)],
) -> Vec<Result<Value, EvalError>> {
    let Some(idx) = wb
        .sheets
        .iter()
        .position(|s| crate::sheet_name_eq(&s.name, sheet_name))
    else {
        return coords
            .iter()
            .map(|_| Err(EvalError::Ref(format!("no sheet named {sheet_name}"))))
            .collect();
    };
    let sheets: Vec<&SheetSnapshot> = wb.sheets.iter().collect();
    let mut state = EvalState::default();
    coords
        .iter()
        .map(|(r, c)| eval_cell_inner(&sheets, idx, *r, *c, &mut state, 0))
        .collect()
}

/// Workbook what-if: the override and target may live on different
/// sheets (formulas may reference across sheets). Returns
/// (before, after) for the target cell; a text result is a type
/// error (see [`what_if_values_in_workbook`] for any value).
pub fn what_if_in_workbook(
    wb: &crate::WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<(f64, f64), EvalError> {
    let (before, after) = what_if_values_in_workbook(wb, override_ref, value, target_ref)?;
    Ok((numeric(Ok(before))?, numeric(Ok(after))?))
}

/// Like [`what_if_in_workbook`], but returns the target's values as
/// they are — number, text or boolean.
pub fn what_if_values_in_workbook(
    wb: &crate::WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<(Value, Value), EvalError> {
    if !value.is_finite() {
        return Err(EvalError::Type("what-if value must be finite".into()));
    }
    for (sheet, r, c) in [override_ref, target_ref] {
        if r > 10_000 || c > 1_000 {
            return Err(EvalError::Ref(format!(
                "what-if coordinates beyond the evaluation grid on {sheet}"
            )));
        }
        if !wb
            .sheets
            .iter()
            .any(|s| crate::sheet_name_eq(&s.name, sheet))
        {
            return Err(EvalError::Ref(format!("unknown sheet {sheet}")));
        }
    }
    let before = eval_cell_value_in_workbook(wb, target_ref.0, target_ref.1, target_ref.2)?;
    let after = eval_cell_value_with_override(wb, override_ref, value, target_ref)?;
    Ok((before, after))
}

/// What-if: clone the sheet, override one cell, re-evaluate a target
/// formula cell. Returns (before, after).
pub fn what_if(
    sheet: &SheetSnapshot,
    override_cell: (usize, usize),
    value: f64,
    target: (usize, usize),
) -> Result<(f64, f64), EvalError> {
    if override_cell.0 > 10_000 || override_cell.1 > 1_000 || target.0 > 10_000 || target.1 > 1_000
    {
        return Err(EvalError::Ref(
            "what-if coordinates beyond the 10000x1000 evaluation grid".into(),
        ));
    }
    if !value.is_finite() {
        return Err(EvalError::Type("what-if value must be finite".into()));
    }
    let before = eval_cell(sheet, target.0, target.1)?;
    let mut cloned = sheet.clone();
    while cloned.cells.len() <= override_cell.0 {
        cloned.cells.push(Vec::new());
    }
    let row = &mut cloned.cells[override_cell.0];
    while row.len() <= override_cell.1 {
        row.push(CellValue::Empty);
    }
    row[override_cell.1] = CellValue::Number(value);
    let after = eval_cell(&cloned, target.0, target.1)?;
    Ok((before, after))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{formula_refs_qualified, parse_delimited};

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
        // $1 (no column) must be a Ref error, not a panic.
        let wb = parse_delimited("d", "S", "A\n=$1+1\n", ',');
        assert!(matches!(
            eval_cell(&wb.sheets[0], 1, 0),
            Err(EvalError::Ref(_))
        ));
        // Deep parser nesting must error, not overflow the stack.
        let nested = format!("=SUM({}1{})", "SUM(".repeat(200), ")".repeat(200));
        let text = format!("A\n{nested}\n");
        let wb = parse_delimited("d", "S", &text, ',');
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
        // Shared precedents are memoised: a depth-30 doubling chain
        // would be 2^30 evaluations without the cache.
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

    #[test]
    fn cross_sheet_refs_evaluate() {
        let mut wb = parse_delimited("d", "Sheet1", "A\n=Sheet2!A2*2\n", ',');
        wb.sheets.push(crate::SheetSnapshot {
            name: "Sheet2".into(),
            cells: vec![vec![CellValue::Number(1.0)], vec![CellValue::Number(21.0)]],
        });
        assert_eq!(eval_cell_in_workbook(&wb, "Sheet1", 1, 0).unwrap(), 42.0);
        let (before, after) =
            what_if_in_workbook(&wb, ("Sheet2", 1, 0), 10.0, ("Sheet1", 1, 0)).unwrap();
        assert_eq!((before, after), (42.0, 20.0));
    }

    #[test]
    fn eval_round5_power_literals_and_cached_errors() {
        // Power operator (right-associative) and TRUE/FALSE literals.
        let wb = parse_delimited("d", "S", "A\n=2^3^2\n=TRUE+FALSE+1\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 0).unwrap(), 512.0);
        assert_eq!(eval_cell(&wb.sheets[0], 2, 0).unwrap(), 2.0);
        // A genuine evaluation error (cycle) is NOT masked by a
        // stale cached value; an unsupported function is.
        let mut sh = sheet();
        sh.cells[1][3] = CellValue::Formula {
            expr: "D2+1".into(),
            cached: Some(Box::new(CellValue::Number(999.0))),
        };
        assert!(matches!(eval_cell(&sh, 1, 3), Err(EvalError::Cycle(_))));
        sh.cells[1][3] = CellValue::Formula {
            expr: "MEDIAN(B2:C2)".into(),
            cached: Some(Box::new(CellValue::Number(999.0))),
        };
        assert_eq!(eval_cell(&sh, 1, 3).unwrap(), 999.0);
        // Unsupported function whose ARGUMENTS would error still
        // falls back to the cached value.
        sh.cells[1][3] = CellValue::Formula {
            expr: "MEDIAN(1/0)".into(),
            cached: Some(Box::new(CellValue::Number(42.0))),
        };
        assert_eq!(eval_cell(&sh, 1, 3).unwrap(), 42.0);
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
        // Long dependency chains: 120 cells deep, no depth error.
        let mut csv = String::from("A\n1\n");
        for i in 3..=121 {
            csv.push_str(&format!("=A{}+1\n", i - 1));
        }
        let wb = parse_delimited("d", "S", &csv, ',');
        assert_eq!(eval_cell(&wb.sheets[0], 120, 0).unwrap(), 120.0);
    }

    #[test]
    fn eval_regressions_round3() {
        // Tab whitespace, grid bounds, scalar-empty aggregates.
        let wb = parse_delimited("d", "S", "A,B\n1,=A2\t+1\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 1).unwrap(), 2.0);
        let wb = parse_delimited("d", "S", "A\n=AAAA1+1\n", ',');
        assert!(matches!(
            eval_cell(&wb.sheets[0], 1, 0),
            Err(EvalError::Ref(_))
        ));
        let wb = parse_delimited("d", "S", "A,B,C\n,=COUNT(A2),=AVERAGE(A2)\n", ',');
        assert_eq!(eval_cell(&wb.sheets[0], 1, 1).unwrap(), 0.0);
        assert!(eval_cell(&wb.sheets[0], 1, 2).is_err());
        // Start-only qualified ranges inherit the start sheet,
        // and sheet qualifiers resolve case-insensitively.
        let mut wb2 = parse_delimited("d", "Sheet1", "A\n=SUM(sheet2!A1:A2)\n", ',');
        wb2.sheets.push(crate::SheetSnapshot {
            name: "Sheet2".into(),
            cells: vec![vec![CellValue::Number(3.0)], vec![CellValue::Number(4.0)]],
        });
        assert_eq!(eval_cell_in_workbook(&wb2, "Sheet1", 1, 0).unwrap(), 7.0);
        // Doubly-qualified range ends are a typed error.
        let mut wb = parse_delimited("d", "Sheet1", "A\n=SUM(Sheet2!A1:Sheet1!B1)\n", ',');
        wb.sheets.push(crate::SheetSnapshot {
            name: "Sheet2".into(),
            cells: vec![vec![CellValue::Number(5.0)]],
        });
        assert!(eval_cell_in_workbook(&wb, "Sheet1", 1, 0).is_err());
    }

    #[test]
    fn eval_regressions_round4() {
        // An override of an Empty cell inside a SUM range counts.
        let wb = parse_delimited("d", "S", "A,B\n,=SUM(A2:A2)\n", ',');
        let (before, after) = what_if_in_workbook(&wb, ("S", 1, 0), 7.0, ("S", 1, 1)).unwrap();
        assert_eq!((before, after), (0.0, 7.0));
        // A stale cache never overrides the expression (1+2 != 999).
        let sheet = crate::SheetSnapshot {
            name: "S".into(),
            cells: vec![vec![CellValue::Formula {
                expr: "1+2".into(),
                cached: Some(Box::new(CellValue::Number(999.0))),
            }]],
        };
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
        SheetSnapshot {
            name: "S".into(),
            cells: vec![cells.iter().map(cell).collect()],
        }
    }

    fn values(cells: &[&str]) -> Vec<Result<Value, EvalError>> {
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![row_sheet(cells)],
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
            "=ROUND(2.5)+ROUND(-1.25,1)+ABS(-1)",
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
        ]);
        assert_eq!(v[1], Ok(Value::Number(1.0)));
        assert_eq!(v[2], text("zero"));
        assert_eq!(v[3], text("n/a"));
        assert_eq!(v[4], Ok(Value::Number(5.0)));
        assert_eq!(v[5], Ok(Value::Number(-1.0)));
        assert_eq!(v[6], Ok(Value::Bool(true)));
        // Cycles are not swallowed by IFERROR.
        assert!(matches!(v[7], Err(EvalError::Cycle(_))));
        // An unsupported function is a parse error even when skipped.
        assert!(matches!(v[8], Err(EvalError::Parse(_))));
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
        assert!(matches!(v[10], Err(EvalError::Parse(_))));
        // The numeric API reports a text result as a type error and
        // keeps booleans as 1/0.
        let sheet = row_sheet(&["=\"a\"", "=1<2"]);
        assert!(matches!(eval_cell(&sheet, 0, 0), Err(EvalError::Type(_))));
        assert_eq!(eval_cell(&sheet, 0, 1), Ok(1.0));
    }

    #[test]
    fn cached_text_is_the_fallback_for_unsupported_functions() {
        let mut sheet = row_sheet(&["1"]);
        sheet.cells[0].push(CellValue::Formula {
            expr: "TEXT(A1,\"0.00\")".into(),
            cached: Some(Box::new(CellValue::Text("1.00".into()))),
        });
        let wb = crate::WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![sheet],
        };
        assert_eq!(
            eval_many_values_in_workbook(&wb, "S", &[(0, 1)]),
            vec![text("1.00")]
        );
    }
}
