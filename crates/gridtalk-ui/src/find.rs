//! Find (Ctrl+F): the cells whose text contains a query.

use gridtalk_graph::{CellValue, WorkbookSnapshot};

/// Matches returned at most, so a one-letter query on a large
/// workbook stays quick.
pub const MAX_MATCHES: usize = 10_000;

/// A matching cell: sheet index, row, column.
pub type Match = (usize, usize, usize);

/// The cells whose shown value, or formula text as typed (`=SUM(..)`),
/// contains `query` (ignoring case), sheet by sheet in reading order.
/// A blank query matches nothing.
pub fn find_cells(wb: &WorkbookSnapshot, query: &str) -> Vec<Match> {
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (s, sheet) in wb.sheets.iter().enumerate() {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                let hit = match cell {
                    CellValue::Empty => false,
                    CellValue::Formula { expr, .. } => {
                        format!("={expr}").to_lowercase().contains(&needle)
                            || cell.display().to_lowercase().contains(&needle)
                    }
                    _ => cell.display().to_lowercase().contains(&needle),
                };
                if hit {
                    out.push((s, r, c));
                    if out.len() >= MAX_MATCHES {
                        return out;
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::{parse_delimited, SheetSnapshot};

    #[test]
    fn finds_values_and_formula_text_across_sheets() {
        let mut wb = parse_delimited("d", "Data", "Item,Qty\nApples,2\npineapple,=B2*2\n", ',');
        wb.sheets.push(SheetSnapshot::new(
            "Notes",
            vec![vec![CellValue::Text("apple pie".into())]],
        ));
        assert_eq!(find_cells(&wb, "APPLE"), [(0, 1, 0), (0, 2, 0), (1, 0, 0)]);
        // Formula text is searched, not only the value.
        assert_eq!(find_cells(&wb, "b2*"), [(0, 2, 1)]);
        // A formula with a saved result is found by its `=` text too.
        *wb.sheets[1].cell_mut(0, 1) = CellValue::Formula {
            expr: "SUM(Data!B2)".into(),
            cached: Some(Box::new(CellValue::Number(2.0))),
        };
        assert_eq!(find_cells(&wb, "=sum("), [(1, 0, 1)]);
        assert!(find_cells(&wb, "   ").is_empty());
        assert!(find_cells(&wb, "zebra").is_empty());
    }
}
