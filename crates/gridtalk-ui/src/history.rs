//! Undo and redo for cell edits made in the grid.

use gridtalk_graph::{sheet_name_eq, CellValue};

/// Changes kept for undo at most; the oldest go first.
pub const MAX_HISTORY: usize = 500;

/// One cell's change, as undo and redo replay it.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    pub sheet: String,
    pub row: usize,
    pub col: usize,
    pub before: CellValue,
    pub after: CellValue,
}

/// The changes that undo and redo walk through. Keystrokes typed into
/// one cell without leaving it are one change, so undo restores the
/// cell as it was before the typing, not one character at a time.
#[derive(Debug, Default)]
pub struct History {
    done: Vec<Change>,
    undone: Vec<Change>,
    /// Whether the last change in `done` is still being typed.
    typing: bool,
}

impl History {
    /// A grid edit of `sheet!(row, col)` from `before` to `after`.
    pub fn record(
        &mut self,
        sheet: &str,
        row: usize,
        col: usize,
        before: CellValue,
        after: CellValue,
    ) {
        self.undone.clear();
        if self.typing {
            if let Some(top) = self.done.last_mut() {
                if top.row == row && top.col == col && sheet_name_eq(&top.sheet, sheet) {
                    top.after = after;
                    return;
                }
            }
        }
        self.end_typing();
        if before == after {
            return;
        }
        self.done.push(Change {
            sheet: sheet.to_string(),
            row,
            col,
            before,
            after,
        });
        self.typing = true;
        if self.done.len() > MAX_HISTORY {
            self.done.remove(0);
        }
    }

    /// Whether a cell is being typed into (its change is still open).
    pub fn typing(&self) -> bool {
        self.typing
    }

    /// The cell being typed into lost focus: the next edit, even of
    /// the same cell, is a new change.
    pub fn end_typing(&mut self) {
        if std::mem::take(&mut self.typing) {
            // Typing that ended where it began changed nothing.
            if self.done.last().is_some_and(|c| c.before == c.after) {
                self.done.pop();
            }
        }
    }

    /// The change to undo (apply its `before`), if any.
    pub fn undo(&mut self) -> Option<Change> {
        self.end_typing();
        let change = self.done.pop()?;
        self.undone.push(change.clone());
        Some(change)
    }

    /// The change to redo (apply its `after`), if any.
    pub fn redo(&mut self) -> Option<Change> {
        self.end_typing();
        let change = self.undone.pop()?;
        self.done.push(change.clone());
        Some(change)
    }

    pub fn clear(&mut self) {
        *self = History::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: f64) -> CellValue {
        CellValue::Number(v)
    }

    #[test]
    fn typing_in_one_cell_is_one_change() {
        let mut h = History::default();
        // Typing "12" into A1 (was 1): two keystrokes, one change.
        h.record("S", 0, 0, n(1.0), n(1.0) /* "1" again */);
        h.record("S", 0, 0, n(1.0), n(12.0));
        h.end_typing();
        let undo = h.undo().unwrap();
        assert_eq!((undo.before, undo.after), (n(1.0), n(12.0)));
        assert_eq!(h.undo(), None);
        let redo = h.redo().unwrap();
        assert_eq!(redo.after, n(12.0));
        assert_eq!(h.redo(), None);
    }

    #[test]
    fn changes_undo_in_reverse_and_a_new_edit_clears_redo() {
        let mut h = History::default();
        h.record("S", 0, 0, n(1.0), n(2.0));
        h.record("S", 1, 0, n(3.0), n(4.0));
        assert_eq!(h.undo().unwrap().row, 1);
        // A new edit after an undo drops what could be redone.
        h.record("S", 2, 0, n(5.0), n(6.0));
        assert_eq!(h.redo(), None);
        assert_eq!(h.undo().unwrap().row, 2);
        assert_eq!(h.undo().unwrap().row, 0);
        assert_eq!(h.undo(), None);
    }

    #[test]
    fn the_same_cell_after_leaving_it_is_a_new_change() {
        let mut h = History::default();
        h.record("S", 0, 0, n(1.0), n(2.0));
        h.end_typing();
        h.record("s", 0, 0, n(2.0), n(3.0));
        assert_eq!(h.undo().unwrap().before, n(2.0));
        assert_eq!(h.undo().unwrap().before, n(1.0));
    }

    #[test]
    fn typing_back_to_the_start_leaves_nothing_to_undo() {
        let mut h = History::default();
        h.record("S", 0, 0, n(1.0), n(12.0));
        h.record("S", 0, 0, n(12.0), n(1.0));
        h.end_typing();
        assert_eq!(h.undo(), None);
    }

    #[test]
    fn history_is_capped() {
        let mut h = History::default();
        for i in 0..MAX_HISTORY + 10 {
            h.record("S", i, 0, n(0.0), n(1.0));
        }
        let mut undone = 0;
        while h.undo().is_some() {
            undone += 1;
        }
        assert_eq!(undone, MAX_HISTORY);
    }
}
