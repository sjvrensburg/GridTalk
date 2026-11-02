//! Minimal formula evaluator for what-if analysis: arithmetic,
//! cell references, ranges, and SUM/AVERAGE/MIN/MAX/COUNT.
//! Deliberately small — the Gridcraft calc engine remains the
//! authority for full recalculation; this evaluator answers
//! "what happens to this formula if that cell changes" locally.

use crate::{formula_refs, CellValue, SheetSnapshot};
use std::collections::BTreeSet;

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

/// Evaluate one cell: numbers pass through, formulas are computed
/// recursively, text/empty evaluate to errors in numeric context.
pub fn eval_cell(sheet: &SheetSnapshot, row: usize, col: usize) -> Result<f64, EvalError> {
    let mut visited = BTreeSet::new();
    eval_cell_inner(sheet, row, col, &mut visited)
}

fn eval_cell_inner(
    sheet: &SheetSnapshot,
    row: usize,
    col: usize,
    visited: &mut BTreeSet<(usize, usize)>,
) -> Result<f64, EvalError> {
    if !visited.insert((row, col)) {
        return Err(EvalError::Cycle(format!(
            "cell {}{}",
            crate::col_name(col),
            row + 1
        )));
    }
    let result = match sheet.get(row, col) {
        CellValue::Number(n) => Ok(n),
        CellValue::Formula { expr, cached } => {
            if let Some(crate::CellValue::Number(n)) = cached.as_deref() {
                // A cached value is only trusted when the formula has
                // no live precedents we can evaluate ourselves.
                if formula_refs(&expr).is_empty() {
                    return Ok(*n);
                }
            }
            let mut p = Parser {
                chars: expr.chars().collect(),
                pos: 0,
                sheet,
                visited,
            };
            p.parse_expr()
        }
        CellValue::Empty => Ok(0.0),
        other => Err(EvalError::Type(format!(
            "cell contains non-numeric {other:?}"
        ))),
    };
    visited.remove(&(row, col));
    result
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    sheet: &'a SheetSnapshot,
    visited: &'a mut BTreeSet<(usize, usize)>,
}

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
        while self.peek() == Some(' ') {
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

    fn parse_term(&mut self) -> Result<f64, EvalError> {
        let mut v = self.parse_factor()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some('*') => {
                    self.bump();
                    v *= self.parse_factor()?;
                }
                Some('/') => {
                    self.bump();
                    let d = self.parse_factor()?;
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
        match self.peek() {
            Some('-') => {
                self.bump();
                Ok(-self.parse_factor()?)
            }
            Some('(') => {
                self.bump();
                let v = self.parse_expr()?;
                self.expect(')')?;
                Ok(v)
            }
            Some(c) if c.is_ascii_digit() || c == '.' => {
                let start = self.pos;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.') {
                    self.bump();
                }
                let s: String = self.chars[start..self.pos].iter().collect();
                s.parse::<f64>()
                    .map_err(|_| EvalError::Parse(format!("bad number {s}")))
            }
            Some(c) if c.is_ascii_alphabetic() || c == '$' => {
                // Function name or cell reference.
                let start = self.pos;
                while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic()) {
                    self.bump();
                }
                let name: String = self.chars[start..self.pos].iter().collect();
                self.skip_ws();
                if self.peek() == Some('(') {
                    self.bump();
                    self.parse_function(&name.to_ascii_uppercase())
                } else {
                    // Cell ref: letters may include $ already consumed? Re-read raw.
                    let (_, row, col) = self.finish_ref(&name)?;
                    eval_cell_inner(self.sheet, row, col, self.visited)
                }
            }
            other => Err(EvalError::Parse(format!(
                "unexpected character {other:?} at {}",
                self.pos
            ))),
        }
    }

    /// After reading leading letters, read optional $ and digits.
    fn finish_ref(&mut self, letters: &str) -> Result<(String, usize, usize), EvalError> {
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
        if row1 == 0 {
            return Err(EvalError::Ref("row 0".into()));
        }
        let mut col = 0usize;
        for c in letters
            .chars()
            .filter(|c| *c != '$')
            .collect::<String>()
            .to_ascii_uppercase()
            .chars()
        {
            col = col * 26 + (c as usize - 'A' as usize + 1);
        }
        Ok((format!("{letters}{digits}"), row1 - 1, col - 1))
    }

    fn parse_ref_token(&mut self) -> Result<(usize, usize), EvalError> {
        self.skip_ws();
        let start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_alphabetic() || c == '$') {
            self.bump();
        }
        let letters: String = self.chars[start..self.pos].iter().collect();
        let (_, row, col) = self.finish_ref(&letters)?;
        Ok((row, col))
    }

    fn parse_function(&mut self, name: &str) -> Result<f64, EvalError> {
        // One range or a comma list of expressions.
        let mut values: Vec<f64> = Vec::new();
        loop {
            self.skip_ws();
            let save = self.pos;
            // Try range A1:B9 first.
            if let Ok((r0, c0)) = self.parse_ref_token() {
                self.skip_ws();
                if self.peek() == Some(':') {
                    self.bump();
                    let (r1, c1) = self.parse_ref_token()?;
                    for r in r0.min(r1)..=r0.max(r1) {
                        for c in c0.min(c1)..=c0.max(c1) {
                            values.push(eval_cell_inner(self.sheet, r, c, self.visited)?);
                        }
                    }
                } else {
                    self.pos = save;
                    values.push(self.parse_expr()?);
                }
            } else {
                self.pos = save;
                values.push(self.parse_expr()?);
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
            "MIN" => values
                .iter()
                .copied()
                .fold(None, |a: Option<f64>, v| Some(a.map_or(v, |x| x.min(v))))
                .ok_or_else(|| EvalError::Type("MIN of nothing".into())),
            "MAX" => values
                .iter()
                .copied()
                .fold(None, |a: Option<f64>, v| Some(a.map_or(v, |x| x.max(v))))
                .ok_or_else(|| EvalError::Type("MAX of nothing".into())),
            "COUNT" => Ok(values.len() as f64),
            other => Err(EvalError::Parse(format!("unsupported function {other}"))),
        }
    }
}

/// What-if: clone the sheet, override one cell, re-evaluate a target
/// formula cell. Returns (before, after).
pub fn what_if(
    sheet: &SheetSnapshot,
    override_cell: (usize, usize),
    value: f64,
    target: (usize, usize),
) -> Result<(f64, f64), EvalError> {
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
    use crate::parse_delimited;

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
    fn cycles_and_div_zero_are_typed_errors() {
        let wb = parse_delimited("d", "S", "A,B\n=A1,=A1/0\n", ',');
        let s = &wb.sheets[0];
        assert!(matches!(eval_cell(s, 0, 1), Err(EvalError::Type(_))));
    }
}
