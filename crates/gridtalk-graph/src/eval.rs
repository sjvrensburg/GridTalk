//! Minimal formula evaluator for what-if analysis: arithmetic,
//! cell references, ranges, and SUM/AVERAGE/MIN/MAX/COUNT.
//! Deliberately small — the Gridcraft calc engine remains the
//! authority for full recalculation; this evaluator answers
//! "what happens to this formula if that cell changes" locally.

use crate::{CellValue, SheetSnapshot};
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

/// Evaluate one cell: numbers pass through, formulas are computed
/// recursively, text/empty evaluate to errors in numeric context.
pub const MAX_EVAL_DEPTH: usize = 256;

pub fn eval_cell(sheet: &SheetSnapshot, row: usize, col: usize) -> Result<f64, EvalError> {
    let sheets: Vec<&SheetSnapshot> = vec![sheet];
    let mut state = EvalState::default();
    eval_cell_inner(&sheets, 0, row, col, &mut state, 0)
}

/// Evaluate a cell in a named sheet of a workbook (cross-sheet refs).
pub fn eval_cell_in_workbook(
    wb: &crate::WorkbookSnapshot,
    sheet_name: &str,
    row: usize,
    col: usize,
) -> Result<f64, EvalError> {
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
    memo: std::collections::HashMap<(usize, usize, usize), f64>,
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
) -> Result<f64, EvalError> {
    if depth > MAX_EVAL_DEPTH {
        return Err(EvalError::Depth(format!(
            "dependency chain deeper than {MAX_EVAL_DEPTH} cells"
        )));
    }
    let key = (sheet_idx, row, col);
    if let Some((os, orow, ocol, value)) = state.override_cell {
        if (os, orow, ocol) == key {
            return Ok(value);
        }
    }
    if let Some(v) = state.memo.get(&key) {
        return Ok(*v);
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
    if let Ok(v) = result {
        state.memo.insert(key, v);
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
) -> Result<f64, EvalError> {
    match sheet.get(row, col) {
        CellValue::Number(n) => Ok(n),
        CellValue::Bool(b) => Ok(if b { 1.0 } else { 0.0 }),
        CellValue::Empty => Ok(0.0),
        CellValue::Formula { expr, cached } => {
            // The expression is the truth; a stale cached value is
            // only a fallback for formulas we cannot evaluate
            // (unsupported functions), never an override of it.
            let cached_num = match cached.as_deref() {
                Some(crate::CellValue::Number(n)) => Some(*n),
                _ => None,
            };
            let mut parser = Parser {
                chars: expr.chars().collect(),
                pos: 0,
                sheets,
                sheet_idx,
                state,
                depth,
                nest: 0,
            };
            let evaluated = (|| -> Result<f64, EvalError> {
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
                Ok(v) => Ok(v),
                Err(e) => {
                    // Cached fallback ONLY for formulas we cannot
                    // evaluate (unsupported functions). Genuine
                    // errors — cycles, type/ref/parse failures in an
                    // evaluable formula — must propagate, not be
                    // masked by a stale cached value.
                    let unevaluable =
                        matches!(&e, EvalError::Parse(m) if m.contains("unsupported function"));
                    match cached_num {
                        Some(n) if unevaluable => Ok(n),
                        _ => Err(e),
                    }
                }
            }
        }
        other => Err(EvalError::Type(format!(
            "cell contains non-numeric {other:?}"
        ))),
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
}

const MAX_PARSE_NEST: usize = 64;

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
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

    fn parse_expr(&mut self) -> Result<f64, EvalError> {
        let mut v = self.parse_term()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('+') => {
                    self.bump();
                    v += self.parse_term()?;
                }
                Some('-') => {
                    self.bump();
                    v -= self.parse_term()?;
                }
                _ => return Ok(v),
            }
        }
    }

    fn parse_power(&mut self) -> Result<f64, EvalError> {
        let base = self.parse_factor()?;
        self.skip_ws();
        if self.peek() == Some('^') {
            self.bump();
            self.enter()?;
            let exp = self.parse_power()?;
            self.nest -= 1;
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn parse_term(&mut self) -> Result<f64, EvalError> {
        let mut v = self.parse_power()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('*') => {
                    self.bump();
                    v *= self.parse_power()?;
                }
                Some('/') => {
                    self.bump();
                    let d = self.parse_power()?;
                    if d == 0.0 {
                        return Err(EvalError::Type("division by zero".into()));
                    }
                    v /= d;
                }
                _ => return Ok(v),
            }
        }
    }

    fn parse_factor(&mut self) -> Result<f64, EvalError> {
        self.skip_ws();
        let mut v = match self.peek() {
            Some('-') => {
                self.bump();
                self.enter()?;
                let v = -self.parse_factor()?;
                self.nest -= 1;
                v
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
                s.parse::<f64>()
                    .map_err(|_| EvalError::Parse(format!("bad number {s}")))?
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
                    let v = if name.eq_ignore_ascii_case("TRUE") {
                        1.0
                    } else {
                        0.0
                    };
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
                v /= 100.0;
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

    /// Push one scalar argument, except that a lone reference to an
    /// empty cell contributes nothing to an aggregate (COUNT of an
    /// empty cell is 0, AVERAGE ignores it) — matching range args.
    fn push_scalar(&mut self, values: &mut Vec<f64>) -> Result<(), EvalError> {
        let save = self.pos;
        if let Ok((t, r, c, _)) = self.parse_ref_token() {
            self.skip_ws();
            let overridden = self
                .state
                .override_cell
                .is_some_and(|(os, orow, ocol, _)| (os, orow, ocol) == (t, r, c));
            if !overridden
                && matches!(self.peek(), Some(',') | Some(')'))
                && matches!(
                    self.sheets[t].get(r, c),
                    CellValue::Empty | CellValue::Text(_) | CellValue::Bool(_)
                )
            {
                return Ok(());
            }
        }
        self.pos = save;
        values.push(self.parse_expr()?);
        Ok(())
    }

    fn collect_range(
        &mut self,
        values: &mut Vec<f64>,
        sheet: usize,
        r0: usize,
        c0: usize,
        r1: usize,
        c1: usize,
    ) -> Result<(), EvalError> {
        for r in r0.min(r1)..=r0.max(r1) {
            for c in c0.min(c1)..=c0.max(c1) {
                if let Some((os, orow, ocol, value)) = self.state.override_cell {
                    if (os, orow, ocol) == (sheet, r, c) {
                        values.push(value);
                        continue;
                    }
                }
                let cell = self.sheets[sheet].get_ref(r, c);
                if matches!(
                    cell,
                    CellValue::Empty | CellValue::Text(_) | CellValue::Bool(_)
                ) {
                    continue;
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
        Ok(())
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

    fn parse_function(&mut self, name: &str) -> Result<f64, EvalError> {
        // Unsupported names fail BEFORE any argument is evaluated,
        // so an argument error cannot mask the cached-value
        // fallback for formulas we cannot evaluate at all.
        if !matches!(name, "SUM" | "AVERAGE" | "AVG" | "MIN" | "MAX" | "COUNT") {
            return Err(EvalError::Parse(format!("unsupported function {name}")));
        }
        // One range or a comma list of expressions.
        let mut values: Vec<f64> = Vec::new();
        loop {
            self.skip_ws();
            let save = self.pos;
            // Whole-column/row ranges (A:A, 1:1) bounded to the
            // sheet's used extents.
            if let Some(whole) = self.try_whole_range() {
                if let Some((t, r0, c0, r1, c1)) = whole {
                    let area = (r1 - r0 + 1).saturating_mul(c1 - c0 + 1);
                    if area > 100_000 {
                        return Err(EvalError::Parse(format!(
                            "range of {area} cells exceeds the 100000-cell evaluation cap"
                        )));
                    }
                    self.collect_range(&mut values, t, r0, c0, r1, c1)?;
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
                    let range_sheet = t0;
                    self.collect_range(&mut values, range_sheet, r0, c0, r1, c1)?;
                } else {
                    self.pos = save;
                    self.push_scalar(&mut values)?;
                }
            } else {
                self.pos = save;
                self.push_scalar(&mut values)?;
            }
            self.skip_ws();
            match self.bump() {
                Some(',') => continue,
                Some(')') => break,
                other => {
                    return Err(EvalError::Parse(format!(
                        "expected ',' or ')' in {name}, got {other:?}"
                    )))
                }
            }
        }
        match name {
            "SUM" => Ok(values.iter().sum()),
            "AVERAGE" | "AVG" => {
                if values.is_empty() {
                    Err(EvalError::Type("AVERAGE of nothing".into()))
                } else {
                    Ok(values.iter().sum::<f64>() / values.len() as f64)
                }
            }
            // Excel returns 0 for MIN/MAX over no numbers.
            "MIN" => Ok(values
                .iter()
                .copied()
                .fold(None, |a: Option<f64>, v| {
                    Some(a.map_or(v, |x: f64| x.min(v)))
                })
                .unwrap_or(0.0)),
            "MAX" => Ok(values
                .iter()
                .copied()
                .fold(None, |a: Option<f64>, v| {
                    Some(a.map_or(v, |x: f64| x.max(v)))
                })
                .unwrap_or(0.0)),
            "COUNT" => Ok(values.len() as f64),
            other => Err(EvalError::Parse(format!("unsupported function {other}"))),
        }
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
        .map(|(r, c)| eval_cell_inner(&sheets, 0, *r, *c, &mut state, 0))
        .collect()
}

/// Evaluate several cells of one named sheet with a shared memo
/// (cross-sheet precedents resolve through the workbook).
pub fn eval_many_in_workbook(
    wb: &crate::WorkbookSnapshot,
    sheet_name: &str,
    coords: &[(usize, usize)],
) -> Vec<Result<f64, EvalError>> {
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
/// (before, after) for the target cell.
pub fn what_if_in_workbook(
    wb: &crate::WorkbookSnapshot,
    override_ref: (&str, usize, usize),
    value: f64,
    target_ref: (&str, usize, usize),
) -> Result<(f64, f64), EvalError> {
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
    let before = eval_cell_in_workbook(wb, target_ref.0, target_ref.1, target_ref.2)?;
    let after = eval_cell_with_override(wb, override_ref, value, target_ref)?;
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
}
