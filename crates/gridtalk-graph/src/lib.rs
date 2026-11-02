//! GridTalk graph layer: workbook snapshots, region inference, and the
//! lifted property graph (workbook -> sheet -> region -> column).
//! Ported in spirit from ExcelGRAG's eg-structure / eg-graph, retargeted
//! to snapshots so downstream code never touches live grid cells.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CellValue {
    Empty,
    Text(String),
    Number(f64),
    Bool(bool),
    Formula {
        expr: String,
        cached: Option<Box<CellValue>>,
    },
}

impl CellValue {
    pub fn display(&self) -> String {
        match self {
            CellValue::Empty => String::new(),
            CellValue::Text(s) => s.clone(),
            CellValue::Number(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{n}")
                }
            }
            CellValue::Bool(b) => b.to_string(),
            CellValue::Formula { expr, cached } => match cached {
                Some(c) => c.display(),
                None => format!("={expr}"),
            },
        }
    }
    pub fn is_empty(&self) -> bool {
        matches!(self, CellValue::Empty)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SheetSnapshot {
    pub name: String,
    /// Row-major cells; rows may be ragged.
    pub cells: Vec<Vec<CellValue>>,
}

impl SheetSnapshot {
    pub fn get(&self, row: usize, col: usize) -> CellValue {
        self.cells
            .get(row)
            .and_then(|r| r.get(col))
            .cloned()
            .unwrap_or(CellValue::Empty)
    }
    pub fn max_col(&self) -> usize {
        self.cells.iter().map(|r| r.len()).max().unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkbookSnapshot {
    pub name: String,
    pub sheets: Vec<SheetSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellRef {
    pub sheet: String,
    pub row: usize, // 0-based
    pub col: usize, // 0-based
}

impl CellRef {
    pub fn a1(&self) -> String {
        format!("{}!{}{}", self.sheet, col_name(self.col), self.row + 1)
    }
}

pub fn col_name(mut col: usize) -> String {
    let mut s = String::new();
    loop {
        s.insert(0, (b'A' + (col % 26) as u8) as char);
        if col < 26 {
            break;
        }
        col = col / 26 - 1;
    }
    s
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RangeRef {
    pub sheet: String,
    pub r0: usize,
    pub c0: usize,
    pub r1: usize, // inclusive
    pub c1: usize, // inclusive
}

impl RangeRef {
    pub fn a1(&self) -> String {
        format!(
            "{}!{}{}:{}{}",
            self.sheet,
            col_name(self.c0),
            self.r0 + 1,
            col_name(self.c1),
            self.r1 + 1
        )
    }
    pub fn contains(&self, row: usize, col: usize) -> bool {
        row >= self.r0 && row <= self.r1 && col >= self.c0 && col <= self.c1
    }
}

/// A detected rectangular data region with an inferred header row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub range: RangeRef,
    pub headers: Vec<String>,
    pub header_row: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Workbook,
    Sheet,
    Region,
    Column,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,
    pub kind: NodeKind,
    pub label: String,
    pub range: Option<RangeRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub from: usize,
    pub to: usize,
    pub kind: String, // "contains" | "depends_on"
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub regions: Vec<Region>,
}

impl Graph {
    pub fn node_for_cell(&self, sheet: &str, row: usize, col: usize) -> Option<&Node> {
        // Most specific (Region/Column) node containing the cell wins.
        self.nodes
            .iter()
            .filter(|n| matches!(n.kind, NodeKind::Region | NodeKind::Column))
            .filter(|n| {
                n.range
                    .as_ref()
                    .map(|r| r.sheet == sheet && r.contains(row, col))
                    .unwrap_or(false)
            })
            .min_by_key(|n| {
                n.range
                    .as_ref()
                    .map(|r| (r.r1 - r.r0 + 1) * (r.c1 - r.c0 + 1))
                    .unwrap_or(usize::MAX)
            })
    }
}

#[derive(Debug)]
pub enum IngestError {
    UnsupportedFormat { ext: String, hint: String },
    Io(std::io::Error),
    Parse(String),
}

impl fmt::Display for IngestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IngestError::UnsupportedFormat { ext, hint } => {
                write!(f, "unsupported format .{ext}: {hint}")
            }
            IngestError::Io(e) => write!(f, "io error: {e}"),
            IngestError::Parse(m) => write!(f, "parse error: {m}"),
        }
    }
}
impl std::error::Error for IngestError {}
impl From<std::io::Error> for IngestError {
    fn from(e: std::io::Error) -> Self {
        IngestError::Io(e)
    }
}

/// Load a workbook snapshot from a path. CSV/TSV are fully supported.
/// XLSX/ODS are routed to the Gridcraft engine once integrated; until
/// then they return a typed error instead of silently mis-parsing.
/// XLSB/XLS are explicitly out of scope (see plan Phase 2).
pub fn load_path(path: &std::path::Path) -> Result<WorkbookSnapshot, IngestError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("workbook")
        .to_string();
    match ext.as_str() {
        "csv" | "tsv" => {
            let text = std::fs::read_to_string(path)?;
            let delim = if ext == "tsv" { '\t' } else { ',' };
            Ok(parse_delimited(&name, "Sheet1", &text, delim))
        }
        "xlsb" | "xls" | "xlsm" => Err(IngestError::UnsupportedFormat {
            ext,
            hint: "GridTalk does not support Excel binary/legacy formats; save as .xlsx or .csv and re-open".into(),
        }),
        "xlsx" | "ods" => {
            let bytes = std::fs::read(path)?;
            load_gridcraft_bytes(&name, &ext, &bytes)
        }
        _ => Err(IngestError::UnsupportedFormat {
            ext,
            hint: "expected .csv, .tsv, .xlsx or .ods".into(),
        }),
    }
}

/// Read XLSX/ODS bytes through the vendored Gridcraft engine and
/// convert its workbook model into a GridTalk snapshot.
#[cfg(feature = "gridcraft")]
pub fn load_gridcraft_bytes(
    name: &str,
    ext: &str,
    bytes: &[u8],
) -> Result<WorkbookSnapshot, IngestError> {
    let (wb, _report) = if ext == "ods" {
        gridcraft_xlsx::read_ods(bytes)
    } else {
        gridcraft_xlsx::read_xlsx(bytes)
    }
    .map_err(|e| IngestError::Parse(format!("gridcraft {ext} reader: {e}")))?;
    Ok(from_gridcraft_workbook(name, &wb))
}

#[cfg(not(feature = "gridcraft"))]
pub fn load_gridcraft_bytes(
    _name: &str,
    ext: &str,
    _bytes: &[u8],
) -> Result<WorkbookSnapshot, IngestError> {
    Err(IngestError::UnsupportedFormat {
        ext: ext.into(),
        hint: "built without the `gridcraft` feature; rebuild with default features for XLSX/ODS"
            .into(),
    })
}

/// Convert a Gridcraft `Workbook` into a snapshot. Formulas keep their
/// canonical text; the cached value (when Gridcraft has one) is kept
/// alongside so passages can show both.
#[cfg(feature = "gridcraft")]
pub fn from_gridcraft_workbook(name: &str, wb: &gridcraft_model::Workbook) -> WorkbookSnapshot {
    use gridcraft_core::Value;
    let mut sheets = Vec::new();
    for sheet in &wb.sheets {
        let (mut max_row, mut max_col) = (0usize, 0usize);
        for (addr, _) in sheet.cells.iter() {
            max_row = max_row.max(addr.row as usize);
            max_col = max_col.max(addr.col as usize);
        }
        let mut cells = vec![vec![CellValue::Empty; max_col + 1]; max_row + 1];
        if sheet.cells.iter().next().is_none() {
            cells = Vec::new();
        }
        for (addr, cell) in sheet.cells.iter() {
            let cached = match &cell.value {
                Value::Empty => None,
                Value::Number(n) => Some(Box::new(CellValue::Number(*n))),
                Value::Text(t) => Some(Box::new(CellValue::Text(t.to_string()))),
                Value::Bool(b) => Some(Box::new(CellValue::Bool(*b))),
                other => Some(Box::new(CellValue::Text(other.display()))),
            };
            let value = match &cell.formula {
                Some(f) => CellValue::Formula {
                    expr: f.text.clone(),
                    cached,
                },
                None => cached.map(|c| *c).unwrap_or(CellValue::Empty),
            };
            cells[addr.row as usize][addr.col as usize] = value;
        }
        sheets.push(SheetSnapshot {
            name: sheet.name.clone(),
            cells,
        });
    }
    WorkbookSnapshot {
        name: name.into(),
        sheets,
    }
}

/// Serialise a snapshot to XLSX bytes via Gridcraft (used for
/// fixtures and Save-As). Formulas are written as formulas.
#[cfg(feature = "gridcraft")]
pub fn to_xlsx_bytes(wb: &WorkbookSnapshot) -> Result<Vec<u8>, IngestError> {
    use gridcraft_core::Value;
    let mut gc = gridcraft_model::Workbook::new();
    gc.sheets.clear();
    for sheet in &wb.sheets {
        let mut gs = gridcraft_model::Sheet::new(sheet.name.clone());
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                let addr = gridcraft_core::CellRef::new(r as u32, c as u32);
                match cell {
                    CellValue::Empty => {}
                    CellValue::Text(t) => gs.set_value(addr, Value::Text(t.as_str().into())),
                    CellValue::Number(n) => gs.set_value(addr, Value::Number(*n)),
                    CellValue::Bool(b) => gs.set_value(addr, Value::Bool(*b)),
                    CellValue::Formula { expr, .. } => gs.set_cell(
                        addr,
                        gridcraft_model::Cell::formula(gridcraft_model::Formula::new(expr)),
                    ),
                }
            }
        }
        gc.sheets.push(std::sync::Arc::new(gs));
    }
    if gc.sheets.is_empty() {
        gc.sheets
            .push(std::sync::Arc::new(gridcraft_model::Sheet::new("Sheet1")));
    }
    gridcraft_xlsx::write_xlsx(&gc)
        .map_err(|e| IngestError::Parse(format!("gridcraft xlsx writer: {e}")))
}

pub fn parse_delimited(name: &str, sheet: &str, text: &str, delim: char) -> WorkbookSnapshot {
    let mut rows: Vec<Vec<CellValue>> = Vec::new();
    for line in text.lines() {
        if line.is_empty() && rows.is_empty() {
            continue;
        }
        let row = split_delimited(line, delim)
            .into_iter()
            .map(|f| infer_value(&f))
            .collect();
        rows.push(row);
    }
    WorkbookSnapshot {
        name: name.into(),
        sheets: vec![SheetSnapshot {
            name: sheet.into(),
            cells: rows,
        }],
    }
}

fn split_delimited(line: &str, delim: char) -> Vec<String> {
    // Minimal RFC4180-style splitter: handles double-quoted fields.
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if in_q {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cur.push('"');
                } else {
                    in_q = false;
                }
            } else {
                cur.push(ch);
            }
        } else if ch == '"' {
            in_q = true;
        } else if ch == delim {
            out.push(cur.trim().to_string());
            cur = String::new();
        } else {
            cur.push(ch);
        }
    }
    out.push(cur.trim().to_string());
    out
}

fn infer_value(f: &str) -> CellValue {
    if f.is_empty() {
        return CellValue::Empty;
    }
    if let Some(expr) = f.strip_prefix('=') {
        return CellValue::Formula {
            expr: expr.to_string(),
            cached: None,
        };
    }
    if let Ok(n) = f.parse::<f64>() {
        return CellValue::Number(n);
    }
    match f.to_ascii_lowercase().as_str() {
        "true" => CellValue::Bool(true),
        "false" => CellValue::Bool(false),
        _ => CellValue::Text(f.to_string()),
    }
}

/// Extract A1-style references (e.g. B2, $A$1:B9 parts split upstream)
/// from a formula expression. Returns (row, col) 0-based pairs.
pub fn formula_refs(expr: &str) -> Vec<(usize, usize)> {
    let bytes = expr.as_bytes();
    let mut refs = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'$' {
            let start = i;
            let mut j = i;
            while j < bytes.len() && (bytes[j].is_ascii_alphabetic() || bytes[j] == b'$') {
                j += 1;
            }
            let letters: String = expr[start..j].chars().filter(|c| *c != '$').collect();
            let mut k = j;
            while k < bytes.len() && bytes[k].is_ascii_digit() {
                k += 1;
            }
            if !letters.is_empty() && k > j && letters.len() <= 3 {
                if let Ok(row1) = expr[j..k].parse::<usize>() {
                    if row1 >= 1 {
                        let mut col = 0usize;
                        for c in letters.to_ascii_uppercase().chars() {
                            col = col * 26 + (c as usize - 'A' as usize + 1);
                        }
                        refs.push((row1 - 1, col - 1));
                        i = k;
                        continue;
                    }
                }
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    refs
}

/// Detect contiguous non-empty rectangular regions per sheet and infer
/// a header row (first row all-text, row below contains a number).
pub fn detect_regions(sheet: &SheetSnapshot) -> Vec<Region> {
    let mut regions = Vec::new();
    let mut visited = vec![vec![false; sheet.max_col()]; sheet.cells.len()];
    for r in 0..sheet.cells.len() {
        for c in 0..sheet.max_col() {
            if visited[r][c] || sheet.get(r, c).is_empty() {
                continue;
            }
            // Grow rectangle right/down while cells are non-empty.
            let mut r1 = r;
            let mut c1 = c;
            while c1 + 1 < sheet.max_col() && !sheet.get(r, c1 + 1).is_empty() {
                c1 += 1;
            }
            while r1 + 1 < sheet.cells.len() && (c..=c1).any(|cc| !sheet.get(r1 + 1, cc).is_empty())
            {
                r1 += 1;
            }
            for rr in r..=r1 {
                for cc in c..=c1 {
                    if rr < visited.len() && cc < visited[rr].len() {
                        visited[rr][cc] = true;
                    }
                }
            }
            let first_all_text = (c..=c1).all(|cc| matches!(sheet.get(r, cc), CellValue::Text(_)));
            let below_has_num =
                r1 > r && (c..=c1).any(|cc| matches!(sheet.get(r + 1, cc), CellValue::Number(_)));
            let header_row = if first_all_text && below_has_num {
                Some(r)
            } else {
                None
            };
            let headers = if header_row.is_some() {
                (c..=c1).map(|cc| sheet.get(r, cc).display()).collect()
            } else {
                (c..=c1).map(col_name).collect()
            };
            regions.push(Region {
                range: RangeRef {
                    sheet: sheet.name.clone(),
                    r0: r,
                    c0: c,
                    r1,
                    c1,
                },
                headers,
                header_row,
            });
        }
    }
    regions
}

/// Build the lifted graph. Formula dependencies are lifted to the
/// region containing the referenced cell, so edges are region->region,
/// never cell->cell. Returns the graph plus invariant violations.
pub fn build_graph(wb: &WorkbookSnapshot) -> (Graph, Vec<String>) {
    let mut g = Graph::default();
    let mut next = 0usize;
    let mut push =
        |g: &mut Graph, kind: NodeKind, label: String, range: Option<RangeRef>| -> usize {
            let id = next;
            next += 1;
            g.nodes.push(Node {
                id,
                kind,
                label,
                range,
            });
            id
        };
    let wb_id = push(&mut g, NodeKind::Workbook, wb.name.clone(), None);
    let mut region_node: BTreeMap<String, usize> = BTreeMap::new();
    for sheet in &wb.sheets {
        let srange = if sheet.cells.is_empty() {
            None
        } else {
            Some(RangeRef {
                sheet: sheet.name.clone(),
                r0: 0,
                c0: 0,
                r1: sheet.cells.len() - 1,
                c1: sheet.max_col().saturating_sub(1),
            })
        };
        let sid = push(&mut g, NodeKind::Sheet, sheet.name.clone(), srange);
        g.edges.push(Edge {
            from: wb_id,
            to: sid,
            kind: "contains".into(),
        });
        for region in detect_regions(sheet) {
            let label = format!("{} region {}", sheet.name, region.range.a1());
            let rid = push(&mut g, NodeKind::Region, label, Some(region.range.clone()));
            region_node.insert(region.range.a1(), rid);
            g.edges.push(Edge {
                from: sid,
                to: rid,
                kind: "contains".into(),
            });
            for (i, h) in region.headers.iter().enumerate() {
                let col = region.range.c0 + i;
                let crange = RangeRef {
                    sheet: sheet.name.clone(),
                    r0: region.range.r0,
                    c0: col,
                    r1: region.range.r1,
                    c1: col,
                };
                let cid = push(&mut g, NodeKind::Column, h.clone(), Some(crange));
                g.edges.push(Edge {
                    from: rid,
                    to: cid,
                    kind: "contains".into(),
                });
            }
            g.regions.push(region);
        }
    }
    // Lifted dependency edges, re-derived from formulas.
    let mut seen = std::collections::BTreeSet::new();
    for sheet in &wb.sheets {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    let src = g.node_for_cell(&sheet.name, r, c).map(|n| n.id);
                    for (rr, cc) in formula_refs(expr) {
                        if let (Some(s), Some(t)) =
                            (src, g.node_for_cell(&sheet.name, rr, cc).map(|n| n.id))
                        {
                            if s != t && seen.insert((s, t)) {
                                g.edges.push(Edge {
                                    from: s,
                                    to: t,
                                    kind: "depends_on".into(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    let violations = check_invariants(wb, &g);
    (g, violations)
}

/// Every lifted depends_on edge must be re-derivable from raw formulas,
/// and vice versa. Mirrors ExcelGRAG's lifted-edge invariant test.
pub fn check_invariants(wb: &WorkbookSnapshot, g: &Graph) -> Vec<String> {
    let mut problems = Vec::new();
    let lifted: std::collections::BTreeSet<(usize, usize)> = g
        .edges
        .iter()
        .filter(|e| e.kind == "depends_on")
        .map(|e| (e.from, e.to))
        .collect();
    let mut expected = std::collections::BTreeSet::new();
    for sheet in &wb.sheets {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    if let Some(s) = g.node_for_cell(&sheet.name, r, c).map(|n| n.id) {
                        for (rr, cc) in formula_refs(expr) {
                            if let Some(t) = g.node_for_cell(&sheet.name, rr, cc).map(|n| n.id) {
                                if s != t {
                                    expected.insert((s, t));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for e in expected.difference(&lifted) {
        problems.push(format!("missing lifted edge {e:?}"));
    }
    for e in lifted.difference(&expected) {
        problems.push(format!("spurious lifted edge {e:?}"));
    }
    problems
}

/// One cell edit coming from the grid / engine command history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CellEdit {
    pub sheet: String,
    pub row: usize,
    pub col: usize,
    pub value: CellValue,
}

/// Incremental re-snapshot: edits are recorded as they happen, but
/// the snapshot (and therefore graph/index rebuild) is only updated
/// once edits have been quiet for `debounce`. Only the edited cells
/// are touched — untouched rows are never re-copied — and the dirty
/// ranges are handed to the graph builder so callers know exactly
/// what changed.
#[derive(Debug, Clone)]
pub struct IncrementalSnapshot {
    snapshot: WorkbookSnapshot,
    pending: Vec<CellEdit>,
    dirty: Vec<RangeRef>,
    last_edit: Option<std::time::Instant>,
    pub debounce: std::time::Duration,
    pub revision: u64,
}

impl IncrementalSnapshot {
    pub fn new(snapshot: WorkbookSnapshot) -> Self {
        IncrementalSnapshot {
            snapshot,
            pending: Vec::new(),
            dirty: Vec::new(),
            last_edit: None,
            debounce: std::time::Duration::from_millis(500),
            revision: 0,
        }
    }

    pub fn snapshot(&self) -> &WorkbookSnapshot {
        &self.snapshot
    }

    pub fn pending_edits(&self) -> usize {
        self.pending.len()
    }

    pub fn record_edit(&mut self, edit: CellEdit, now: std::time::Instant) {
        self.dirty.push(RangeRef {
            sheet: edit.sheet.clone(),
            r0: edit.row,
            c0: edit.col,
            r1: edit.row,
            c1: edit.col,
        });
        self.pending.push(edit);
        self.last_edit = Some(now);
    }

    /// Apply pending edits if the debounce window has elapsed.
    /// Returns true when the snapshot changed (revision bumped).
    pub fn poll(&mut self, now: std::time::Instant) -> bool {
        let due = self
            .last_edit
            .map(|t| now.duration_since(t) >= self.debounce)
            .unwrap_or(false);
        if due && !self.pending.is_empty() {
            self.flush();
            true
        } else {
            false
        }
    }

    /// Apply pending edits immediately, patching only edited cells.
    pub fn flush(&mut self) {
        let edits = std::mem::take(&mut self.pending);
        for edit in edits {
            if let Some(sheet) = self
                .snapshot
                .sheets
                .iter_mut()
                .find(|s| s.name == edit.sheet)
            {
                while sheet.cells.len() <= edit.row {
                    sheet.cells.push(Vec::new());
                }
                let row = &mut sheet.cells[edit.row];
                while row.len() <= edit.col {
                    row.push(CellValue::Empty);
                }
                row[edit.col] = edit.value;
            }
        }
        self.revision += 1;
    }

    /// Dirty ranges since the last flush, then clears the list.
    pub fn take_dirty_ranges(&mut self) -> Vec<RangeRef> {
        std::mem::take(&mut self.dirty)
    }
}

/// Render a region as a cited passage for the LLM context.
pub fn render_passage(sheet: &SheetSnapshot, region: &Region) -> String {
    let mut out = format!("Range {}\n", region.range.a1());
    for r in region.range.r0..=region.range.r1 {
        let cells: Vec<String> = (region.range.c0..=region.range.c1)
            .map(|c| sheet.get(r, c).display())
            .collect();
        out.push_str(&cells.join(" | "));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> WorkbookSnapshot {
        parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\nPears,4,5,=B3*C3\n",
            ',',
        )
    }

    #[test]
    fn parses_and_infers_types() {
        let wb = sample();
        assert_eq!(wb.sheets[0].get(1, 0), CellValue::Text("Apples".into()));
        assert_eq!(wb.sheets[0].get(1, 1), CellValue::Number(2.0));
        assert!(matches!(wb.sheets[0].get(1, 3), CellValue::Formula { .. }));
    }

    #[test]
    fn detects_header_region() {
        let wb = sample();
        let regions = detect_regions(&wb.sheets[0]);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].headers, vec!["Item", "Qty", "Price", "Total"]);
        assert_eq!(regions[0].header_row, Some(0));
    }

    #[test]
    fn graph_invariants_hold_and_refs_parse() {
        assert_eq!(formula_refs("B2*C2+$A$1"), vec![(1, 1), (1, 2), (0, 0)]);
        let wb = sample();
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Region));
        let node = g.node_for_cell("Sheet1", 1, 3).unwrap();
        assert_eq!(node.kind, NodeKind::Column); // most specific node wins
    }

    #[test]
    fn incremental_snapshot_debounces_and_patches_only_dirty_cells() {
        let wb = sample();
        let mut inc = IncrementalSnapshot::new(wb);
        let t0 = std::time::Instant::now();
        inc.record_edit(
            CellEdit {
                sheet: "Sheet1".into(),
                row: 1,
                col: 1,
                value: CellValue::Number(10.0),
            },
            t0,
        );
        // Inside the debounce window: snapshot unchanged.
        assert!(!inc.poll(t0 + std::time::Duration::from_millis(100)));
        assert_eq!(inc.snapshot().sheets[0].get(1, 1), CellValue::Number(2.0));
        // After the window: exactly the edited cell changed.
        assert!(inc.poll(t0 + std::time::Duration::from_millis(600)));
        assert_eq!(inc.snapshot().sheets[0].get(1, 1), CellValue::Number(10.0));
        assert_eq!(
            inc.snapshot().sheets[0].get(1, 0),
            CellValue::Text("Apples".into())
        );
        assert_eq!(inc.revision, 1);
        let dirty = inc.take_dirty_ranges();
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0].a1(), "Sheet1!B2:B2");
        assert!(inc.take_dirty_ranges().is_empty());
    }

    #[cfg(feature = "gridcraft")]
    #[test]
    fn xlsx_round_trip_via_gridcraft_preserves_values_and_formulas() {
        let wb = sample();
        let bytes = to_xlsx_bytes(&wb).unwrap();
        assert_eq!(&bytes[0..2], b"PK"); // zip container
        let back = load_gridcraft_bytes("demo", "xlsx", &bytes).unwrap();
        let sheet = &back.sheets[0];
        assert_eq!(sheet.get(0, 0), CellValue::Text("Item".into()));
        assert_eq!(sheet.get(1, 1), CellValue::Number(2.0));
        match sheet.get(1, 3) {
            CellValue::Formula { ref expr, .. } => {
                assert!(expr.contains("B2"), "formula text kept: {expr}")
            }
            other => panic!("D2 must survive as a formula, got {other:?}"),
        }
        let (g, violations) = build_graph(&back);
        assert!(violations.is_empty(), "{violations:?}");
        assert!(g.nodes.iter().any(|n| n.kind == NodeKind::Region));
    }

    #[cfg(feature = "gridcraft")]
    #[test]
    fn ods_fixture_loads_via_gridcraft() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/demo.ods");
        let wb = load_path(&path).unwrap();
        assert_eq!(wb.sheets[0].get(0, 0), CellValue::Text("Item".into()));
        assert_eq!(wb.sheets[0].get(1, 0), CellValue::Text("Apples".into()));
        assert_eq!(wb.sheets[0].get(1, 1), CellValue::Number(2.0));
    }

    #[test]
    fn unsupported_binary_format_is_typed_error() {
        let err = load_path(std::path::Path::new("book.xlsb")).unwrap_err();
        assert!(matches!(err, IngestError::UnsupportedFormat { .. }));
    }
}
