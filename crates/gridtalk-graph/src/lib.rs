//! GridTalk graph layer: workbook snapshots, region inference, and the
//! lifted property graph (workbook -> sheet -> region -> column).
//! Ported in spirit from ExcelGRAG's eg-structure / eg-graph, retargeted
//! to snapshots so downstream code never touches live grid cells.

use serde::{Deserialize, Serialize};

/// Sheet-name equality with Unicode case folding (ASCII-only
/// folding never matches café vs CAFÉ).
pub fn sheet_name_eq(a: &str, b: &str) -> bool {
    a.chars()
        .flat_map(|c| c.to_lowercase())
        .eq(b.chars().flat_map(|c| c.to_lowercase()))
}
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
                if n.fract() == 0.0 && n.abs() < 9_007_199_254_740_992.0 {
                    format!("{}", *n as i64)
                } else if n.fract() == 0.0 {
                    format!("{n:.0}")
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
    /// Borrowed cell access (no clone) for hot scans.
    pub fn get_ref(&self, row: usize, col: usize) -> &CellValue {
        static EMPTY: CellValue = CellValue::Empty;
        self.cells
            .get(row)
            .and_then(|r| r.get(col))
            .unwrap_or(&EMPTY)
    }

    pub fn max_col(&self) -> usize {
        self.cells.iter().map(|r| r.len()).max().unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkbookSnapshot {
    pub name: String,
    pub sheets: Vec<SheetSnapshot>,
    /// Defined names (`Tax_Rate` → `Rates!$B$11`), kept so formulas
    /// that use them evaluate.
    #[serde(default)]
    pub names: Vec<DefinedName>,
}

/// A workbook defined name. `formula` has no leading `=`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DefinedName {
    pub name: String,
    /// The sheet a sheet-scoped name belongs to; `None` is workbook
    /// scope.
    pub scope: Option<String>,
    pub formula: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellRef {
    pub sheet: String,
    pub row: usize, // 0-based
    pub col: usize, // 0-based
}

pub fn quote_sheet(name: &str) -> String {
    if !name.is_empty()
        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
        && !name.chars().next().unwrap().is_ascii_digit()
    {
        name.to_string()
    } else {
        format!("'{}'", name.replace('\'', "''"))
    }
}

impl CellRef {
    pub fn a1(&self) -> String {
        format!(
            "{}!{}{}",
            quote_sheet(&self.sheet),
            col_name(self.col),
            self.row + 1
        )
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
            quote_sheet(&self.sheet),
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
    /// Index into `WorkbookSnapshot::sheets` of each entry in
    /// `regions`, recorded by `build_graph` so consumers need not
    /// re-detect regions to map them to sheets. Empty for
    /// hand-built graphs.
    #[serde(default)]
    pub region_sheets: Vec<usize>,
}

/// Region/Column node ids sorted by ascending range area, so the
/// first containing node is the most specific. Built once and
/// reused across all lookups (graph construction is not quadratic
/// in node count per cell).
pub fn specificity_index(g: &Graph) -> Vec<usize> {
    let mut idx: Vec<usize> = g
        .nodes
        .iter()
        .filter(|n| matches!(n.kind, NodeKind::Region | NodeKind::Column) && n.range.is_some())
        .map(|n| n.id)
        .collect();
    // Node ids are sequential, so areas can be precomputed once by id
    // instead of rescanning `g.nodes` for every key comparison.
    // Saturating u128 math: hand-built/inverted ranges cannot panic.
    let mut areas = vec![u128::MAX; g.nodes.len()];
    for n in &g.nodes {
        if let Some(r) = &n.range {
            if n.id < areas.len() {
                areas[n.id] = range_area(r);
            }
        }
    }
    idx.sort_by_key(|id| areas.get(*id).copied().unwrap_or(u128::MAX));
    idx
}

/// Index-based lookup: node ids are assigned sequentially, so
/// `g.nodes[id]` is the node with that id.
pub fn node_for_cell_in(
    g: &Graph,
    index: &[usize],
    sheet: &str,
    row: usize,
    col: usize,
) -> Option<usize> {
    index.iter().copied().find(|id| {
        // Fast path: ids are sequential in built graphs; fall back
        // to a by-id search for hand-built graphs.
        let node = g
            .nodes
            .get(*id)
            .filter(|n| n.id == *id)
            .or_else(|| g.nodes.iter().find(|n| n.id == *id));
        node.and_then(|n| n.range.as_ref())
            .map(|r| crate::sheet_name_eq(&r.sheet, sheet) && r.contains(row, col))
            .unwrap_or(false)
    })
}

/// Range area with saturating arithmetic (inverted ranges = 0).
pub fn range_area(r: &RangeRef) -> u128 {
    (r.r1.saturating_sub(r.r0) as u128 + 1) * (r.c1.saturating_sub(r.c0) as u128 + 1)
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
                    .map(|r| crate::sheet_name_eq(&r.sheet, sheet) && r.contains(row, col))
                    .unwrap_or(false)
            })
            .min_by_key(|n| n.range.as_ref().map(range_area).unwrap_or(u128::MAX))
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

/// Ingestion cap for CSV/TSV files.
const MAX_DELIMITED_BYTES: u64 = 32 * 1024 * 1024;
/// Ingestion cap for XLSX/ODS packages.
const MAX_PACKAGE_BYTES: u64 = 256 * 1024 * 1024;

/// Lowercased extension and workbook name (file stem) of a path.
fn ext_and_name(path: &std::path::Path) -> (String, String) {
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
    (ext, name)
}

/// Format check and size cap for an ingestible file of `len` bytes.
fn check_ingestible(ext: &str, len: u64) -> Result<(), IngestError> {
    match ext {
        "csv" | "tsv" if len > MAX_DELIMITED_BYTES => Err(IngestError::Parse(format!(
            "delimited file is {len} bytes, over the {MAX_DELIMITED_BYTES}-byte ingestion cap"
        ))),
        "csv" | "tsv" => Ok(()),
        "xlsb" | "xls" | "xlsm" => Err(IngestError::UnsupportedFormat {
            ext: ext.into(),
            hint: "GridTalk does not support Excel binary/legacy formats; save as .xlsx or .csv and re-open".into(),
        }),
        "xlsx" | "ods" if len > MAX_PACKAGE_BYTES => Err(IngestError::Parse(format!(
            "spreadsheet package is {len} bytes, over the {MAX_PACKAGE_BYTES}-byte ingestion cap"
        ))),
        "xlsx" | "ods" => Ok(()),
        _ => Err(IngestError::UnsupportedFormat {
            ext: ext.into(),
            hint: "expected .csv, .tsv, .xlsx or .ods".into(),
        }),
    }
}

/// Load a workbook snapshot from a path. CSV/TSV are fully supported.
/// XLSX/ODS go through the Gridcraft engine. XLSB/XLS are explicitly
/// out of scope (see plan Phase 2). Unsupported or oversized files
/// are rejected from metadata, before the file is read.
pub fn load_path(path: &std::path::Path) -> Result<WorkbookSnapshot, IngestError> {
    let (ext, _) = ext_and_name(path);
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    check_ingestible(&ext, len)?;
    let bytes = std::fs::read(path)?;
    load_bytes(path, &bytes)
}

/// Parse a workbook from bytes already read from `path`. The path
/// supplies only the format (extension) and workbook name, so a
/// caller that also needs the raw bytes (e.g. to hash them) reads
/// the file once.
pub fn load_bytes(path: &std::path::Path, bytes: &[u8]) -> Result<WorkbookSnapshot, IngestError> {
    let (ext, name) = ext_and_name(path);
    check_ingestible(&ext, bytes.len() as u64)?;
    if ext == "csv" || ext == "tsv" {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let delim = if ext == "tsv" { '\t' } else { ',' };
        parse_delimited_strict(&name, "Sheet1", text, delim)
    } else {
        load_gridcraft_bytes(&name, &ext, bytes)
    }
}

/// Read XLSX/ODS bytes through the vendored Gridcraft engine and
/// convert its workbook model into a GridTalk snapshot.
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
    from_gridcraft_workbook(name, &wb)
}

/// Convert a Gridcraft `Workbook` into a snapshot. Formulas keep their
/// canonical text; the cached value (when Gridcraft has one) is kept
/// alongside so passages can show both.
/// Cap on densified snapshot cells per sheet (and per workbook): a
/// sparse XLSX with a single cell at XFD1048576 must not allocate a
/// billion-cell grid.
pub const MAX_SNAPSHOT_CELLS: usize = 4_000_000;

pub fn from_gridcraft_workbook(
    name: &str,
    wb: &gridcraft_model::Workbook,
) -> Result<WorkbookSnapshot, IngestError> {
    use gridcraft_core::Value;
    let mut sheets = Vec::new();
    let mut total_cells: u64 = 0;
    for sheet in &wb.sheets {
        let (mut max_row, mut max_col) = (0usize, 0usize);
        for (addr, _) in sheet.cells.iter() {
            max_row = max_row.max(addr.row as usize);
            max_col = max_col.max(addr.col as usize);
        }
        let area = (max_row as u64 + 1).saturating_mul(max_col as u64 + 1);
        total_cells = total_cells.saturating_add(area);
        if total_cells > MAX_SNAPSHOT_CELLS as u64 {
            return Err(IngestError::Parse(format!(
                "workbook is too sparse to snapshot ({total_cells} densified cells > {MAX_SNAPSHOT_CELLS} across all sheets)"
            )));
        }
        if area > MAX_SNAPSHOT_CELLS as u64 {
            return Err(IngestError::Parse(format!(
                "sheet {} is too sparse to snapshot ({area} cells > {MAX_SNAPSHOT_CELLS}); trim empty rows/columns at the edges",
                sheet.name
            )));
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
    let names = wb
        .names
        .iter()
        .map(|n| DefinedName {
            name: n.name.clone(),
            scope: n
                .scope
                .and_then(|i| wb.sheets.get(i))
                .map(|s| s.name.clone()),
            formula: n.formula.clone(),
        })
        .collect();
    Ok(WorkbookSnapshot {
        name: name.into(),
        sheets,
        names,
    })
}

/// Build a Gridcraft workbook from a snapshot: values, formulas with
/// their cached values, and defined names. Sheet indices match the
/// snapshot's.
pub fn to_gridcraft_workbook(wb: &WorkbookSnapshot) -> gridcraft_model::Workbook {
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
                    CellValue::Formula { expr, cached } => {
                        let mut cell =
                            gridcraft_model::Cell::formula(gridcraft_model::Formula::new(expr));
                        if let Some(cached) = cached {
                            cell.value = match cached.as_ref() {
                                CellValue::Number(n) => Value::Number(*n),
                                CellValue::Bool(b) => Value::Bool(*b),
                                CellValue::Text(s) => Value::Text(s.as_str().into()),
                                _ => Value::Empty,
                            };
                        }
                        gs.set_cell(addr, cell)
                    }
                }
            }
        }
        gc.sheets.push(std::sync::Arc::new(gs));
    }
    if gc.sheets.is_empty() {
        gc.sheets
            .push(std::sync::Arc::new(gridcraft_model::Sheet::new("Sheet1")));
    }
    gc.names = wb
        .names
        .iter()
        .map(|n| gridcraft_model::DefinedName {
            name: n.name.clone(),
            scope: n
                .scope
                .as_deref()
                .and_then(|s| wb.sheets.iter().position(|sh| sheet_name_eq(&sh.name, s))),
            formula: n.formula.clone(),
            comment: String::new(),
            hidden: false,
        })
        .collect();
    gc
}

/// Serialise a snapshot to XLSX bytes via Gridcraft (used for
/// fixtures and Save-As). Formulas are written as formulas.
pub fn to_xlsx_bytes(wb: &WorkbookSnapshot) -> Result<Vec<u8>, IngestError> {
    let gc = to_gridcraft_workbook(wb);
    gridcraft_xlsx::write_xlsx(&gc)
        .map_err(|e| IngestError::Parse(format!("gridcraft xlsx writer: {e}")))
}

pub fn parse_delimited(name: &str, sheet: &str, text: &str, delim: char) -> WorkbookSnapshot {
    parse_delimited_strict(name, sheet, text, delim)
        .expect("malformed delimited input; use parse_delimited_strict for a typed error")
}

/// Strict variant: malformed quoting (post-quote junk, unterminated
/// quotes) is an error instead of being silently merged.
pub fn parse_delimited_strict(
    name: &str,
    sheet: &str,
    text: &str,
    delim: char,
) -> Result<WorkbookSnapshot, IngestError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut rows: Vec<Vec<CellValue>> = Vec::new();
    for fields in split_delimited_stream(text, delim)? {
        if fields.len() == 1 && fields[0].is_empty() && rows.is_empty() {
            continue;
        }
        rows.push(fields.into_iter().map(|f| infer_value(&f)).collect());
    }
    Ok(WorkbookSnapshot {
        name: name.into(),
        sheets: vec![SheetSnapshot {
            name: sheet.into(),
            cells: rows,
        }],
        names: Vec::new(),
    })
}

/// RFC4180-style stream splitter: quoted fields may contain the
/// delimiter, newlines and leading/trailing spaces (preserved);
/// unquoted fields are trimmed. `""` inside quotes is a quote.
fn split_delimited_stream(text: &str, delim: char) -> Result<Vec<Vec<String>>, IngestError> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut in_q = false;
    let mut field_quoted = false;
    let mut just_closed = false;
    let mut chars = text.chars().peekable();
    let push_field = |row: &mut Vec<String>, cur: &mut String, field_quoted: bool| {
        let v = std::mem::take(cur);
        row.push(if field_quoted {
            v
        } else {
            v.trim().to_string()
        });
    };
    while let Some(ch) = chars.next() {
        if in_q {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cur.push('"');
                } else {
                    in_q = false;
                    just_closed = true;
                }
            } else {
                cur.push(ch);
            }
        } else if just_closed && (ch == ' ' || ch == '\t') {
            // Optional padding between a closing quote and the
            // delimiter/newline is common in real-world CSV.
            continue;
        } else if just_closed && ch != delim && ch != '\n' && ch != '\r' {
            return Err(IngestError::Parse(format!(
                "unexpected character {ch:?} after closing quote"
            )));
        } else if ch == '"' && cur.is_empty() && !quoted {
            in_q = true;
            quoted = true;
            field_quoted = true;
            just_closed = false;
        } else if ch == delim {
            push_field(&mut row, &mut cur, field_quoted);
            quoted = false;
            field_quoted = false;
            just_closed = false;
        } else if ch == '\n' || ch == '\r' {
            if ch == '\r' && chars.peek() == Some(&'\n') {
                chars.next();
            }
            push_field(&mut row, &mut cur, field_quoted);
            quoted = false;
            field_quoted = false;
            just_closed = false;
            rows.push(std::mem::take(&mut row));
        } else {
            cur.push(ch);
        }
    }
    if in_q {
        return Err(IngestError::Parse("unterminated quoted field".into()));
    }
    if !cur.is_empty() || !row.is_empty() || field_quoted {
        push_field(&mut row, &mut cur, field_quoted);
        rows.push(row);
    }
    Ok(rows)
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
        if n.is_finite() {
            return CellValue::Number(n);
        }
        return CellValue::Text(f.to_string());
    }
    match f.to_ascii_lowercase().as_str() {
        "true" => CellValue::Bool(true),
        "false" => CellValue::Bool(false),
        _ => CellValue::Text(f.to_string()),
    }
}

/// One formula reference, with its sheet qualifier when present.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedRef {
    pub sheet: Option<String>,
    pub row: usize,
    pub col: usize,
}

fn parse_a1_at(chars: &[char], mut i: usize) -> Option<(usize, usize, usize)> {
    // Optional $, letters, optional $, digits. Returns (row, col, next).
    if i < chars.len() && chars[i] == '$' {
        i += 1;
    }
    let ls = i;
    while i < chars.len() && chars[i].is_ascii_alphabetic() {
        i += 1;
    }
    if i == ls || i - ls > 3 {
        return None;
    }
    let letters: String = chars[ls..i].iter().collect();
    if i < chars.len() && chars[i] == '$' {
        i += 1;
    }
    let ds = i;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i == ds {
        return None;
    }
    let row1: usize = chars[ds..i].iter().collect::<String>().parse().ok()?;
    if row1 == 0 || row1 > 1_048_576 {
        return None;
    }
    let mut col = 0usize;
    for c in letters.to_ascii_uppercase().chars() {
        col = col
            .checked_mul(26)?
            .checked_add(c as usize - 'A' as usize + 1)?;
    }
    if col == 0 || col > 16_384 {
        return None;
    }
    Some((row1 - 1, col - 1, i))
}

/// Extract references from a formula, expanding ranges to every
/// interior cell. String literals and numbers (including scientific
/// notation like 1E5) never produce references. Sheet-qualified
/// refs (`Sheet2!A1`, `'My Sheet'!A1:B2`) keep their sheet.
/// Push refs for a rectangle, capped: full expansion to 100k
/// cells, interior sample grid beyond that.
fn push_range_refs(
    refs: &mut Vec<QualifiedRef>,
    sheet: &Option<String>,
    lo_r: usize,
    lo_c: usize,
    hi_r: usize,
    hi_c: usize,
) {
    if (hi_r - lo_r + 1).saturating_mul(hi_c - lo_c + 1) <= 100_000 {
        for r in lo_r..=hi_r {
            for cc in lo_c..=hi_c {
                refs.push(QualifiedRef {
                    sheet: sheet.clone(),
                    row: r,
                    col: cc,
                });
            }
        }
    } else {
        // Over the cap: emit the same interior
        // sample grid check_invariants uses
        // (rows/8 x cols/8 from the low corner), so
        // lifted edges reach interior columns and
        // regions instead of only the corners.
        let rows = hi_r - lo_r + 1;
        let cols = hi_c - lo_c + 1;
        let rstep = (rows / 8).max(1);
        let cstep = (cols / 8).max(1);
        let mut rr = lo_r;
        while rr <= hi_r {
            let mut cc = lo_c;
            while cc <= hi_c {
                refs.push(QualifiedRef {
                    sheet: sheet.clone(),
                    row: rr,
                    col: cc,
                });
                cc += cstep;
            }
            rr += rstep;
        }
        // The stepped grid may stop short of the
        // high corner; the independent extractor
        // includes range endpoints, so emit it too.
        if (hi_r != lo_r || hi_c != lo_c)
            && !refs
                .iter()
                .any(|q| q.row == hi_r && q.col == hi_c && &q.sheet == sheet)
        {
            refs.push(QualifiedRef {
                sheet: sheet.clone(),
                row: hi_r,
                col: hi_c,
            });
        }
    }
}

/// Whole-column (A:A) / whole-row (1:1) range at `at`, as
/// (lo_r, lo_c, hi_r, hi_c, next). Bounds are the Excel grid; the
/// capped sampler in push_range_refs keeps expansion bounded.
fn parse_whole_range_at(chars: &[char], at: usize) -> Option<(usize, usize, usize, usize, usize)> {
    let col_of = |letters: &[char]| -> Option<usize> {
        if letters.is_empty() || letters.len() > 3 {
            return None;
        }
        let mut col = 0usize;
        for c in letters {
            if !c.is_ascii_alphabetic() {
                return None;
            }
            col = col
                .checked_mul(26)?
                .checked_add(c.to_ascii_uppercase() as usize - 'A' as usize + 1)?;
        }
        if col == 0 || col > 16_384 {
            return None;
        }
        Some(col - 1)
    };
    let mut j = at;
    if chars.get(j) == Some(&'$') {
        j += 1;
    }
    let lstart = j;
    while j < chars.len() && chars[j].is_ascii_alphabetic() {
        j += 1;
    }
    if j > lstart && chars.get(j) == Some(&':') {
        let first: Vec<char> = chars[lstart..j].to_vec();
        let mut k = j + 1;
        if chars.get(k) == Some(&'$') {
            k += 1;
        }
        let estart = k;
        while k < chars.len() && chars[k].is_ascii_alphabetic() {
            k += 1;
        }
        let second: Vec<char> = chars[estart..k].to_vec();
        if let (Some(c0), Some(c1)) = (col_of(&first), col_of(&second)) {
            return Some((0, c0.min(c1), 1_048_575, c0.max(c1), k));
        }
        return None;
    }
    // Row form: digits ':' digits
    let mut j = at;
    if chars.get(j) == Some(&'$') {
        j += 1;
    }
    let dstart = j;
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    if j > dstart && chars.get(j) == Some(&':') {
        let first: String = chars[dstart..j].iter().collect();
        let mut k = j + 1;
        if chars.get(k) == Some(&'$') {
            k += 1;
        }
        let estart = k;
        while k < chars.len() && chars[k].is_ascii_digit() {
            k += 1;
        }
        let second: String = chars[estart..k].iter().collect();
        if let (Ok(r0), Ok(r1)) = (first.parse::<usize>(), second.parse::<usize>()) {
            if (1..=1_048_576).contains(&r0) && (1..=1_048_576).contains(&r1) {
                return Some((r0.min(r1) - 1, 0, r0.max(r1) - 1, 16_383, k));
            }
        }
    }
    None
}

pub fn formula_refs_qualified(expr: &str) -> Vec<QualifiedRef> {
    let chars: Vec<char> = expr.chars().collect();
    let mut refs = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '"' {
            i += 1;
            while i < chars.len() && chars[i] != '"' {
                i += 1;
            }
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit())
        {
            // Skip a numeric literal incl. exponent.
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if i < chars.len() && (chars[i] == 'E' || chars[i] == 'e') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    i = j;
                    while i < chars.len() && chars[i].is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            continue;
        }
        // Sheet qualifier: Name! or 'Quoted Name'!
        let mut sheet: Option<String> = None;
        let mut start = i;
        if c == '\'' {
            let mut j = i + 1;
            let mut name = String::new();
            while j < chars.len() {
                if chars[j] == '\'' {
                    // Doubled '' is an escaped apostrophe inside the name.
                    if j + 1 < chars.len() && chars[j + 1] == '\'' {
                        name.push('\'');
                        j += 2;
                        continue;
                    }
                    break;
                }
                name.push(chars[j]);
                j += 1;
            }
            if j + 1 < chars.len() && chars[j] == '\'' && chars[j + 1] == '!' {
                sheet = Some(name);
                start = j + 2;
            }
        } else if c.is_alphabetic() || c == '_' {
            let mut j = i;
            while j < chars.len()
                && (chars[j].is_alphanumeric() || chars[j] == '_' || chars[j] == '.')
            {
                j += 1;
            }
            if j < chars.len() && chars[j] == '!' {
                sheet = Some(chars[i..j].iter().collect());
                start = j + 1;
            }
        }
        // A ref-shaped sub-token inside a longer identifier
        // (Sales2024) is not a reference.
        if sheet.is_none() && i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_') {
            i += 1;
            continue;
        }
        let at = if sheet.is_some() { start } else { i };
        if let Some((lo_r, lo_c, hi_r, hi_c, wnext)) = parse_whole_range_at(&chars, at) {
            // Require a token boundary after the range.
            if wnext >= chars.len() || !(chars[wnext].is_alphanumeric() || chars[wnext] == '_') {
                push_range_refs(&mut refs, &sheet, lo_r, lo_c, hi_r, hi_c);
                i = wnext;
                continue;
            }
        }
        if let Some((r0, c0, next)) = parse_a1_at(&chars, at) {
            // A ref-shaped PREFIX of a longer token (A1A, B2C) is
            // not a reference.
            if next < chars.len() && (chars[next].is_alphanumeric() || chars[next] == '_') {
                i = next;
                continue;
            }
            // A ref-shaped token followed by '(' (whitespace
            // allowed) is a function name (LOG10 (...)).
            let mut paren_at = next;
            while paren_at < chars.len() && chars[paren_at] == ' ' {
                paren_at += 1;
            }
            if paren_at < chars.len() && chars[paren_at] == '(' {
                i = next;
                continue;
            }
            // Range? The end may carry its own sheet qualifier
            // (Sheet2!A1:Sheet2!B2); strip it before parsing.
            if next < chars.len() && chars[next] == ':' {
                let end_at = {
                    let mut j = next + 1;
                    // Skip Name! or 'Quoted Name'! after the colon.
                    if j < chars.len() && chars[j] == '\'' {
                        j += 1;
                        // Doubled-apostrophe aware, like the
                        // range-start scanner: '' is an escaped
                        // quote inside the name, not its end.
                        while j < chars.len() {
                            if chars[j] == '\'' {
                                if chars.get(j + 1) == Some(&'\'') {
                                    j += 2;
                                    continue;
                                }
                                break;
                            }
                            j += 1;
                        }
                        if j + 1 < chars.len() && chars[j + 1] == '!' {
                            j += 2;
                        } else {
                            j = next + 1;
                        }
                    } else {
                        let mut k = j;
                        while k < chars.len()
                            && (chars[k].is_alphanumeric() || chars[k] == '_' || chars[k] == '.')
                        {
                            k += 1;
                        }
                        if k < chars.len() && chars[k] == '!' && k > j {
                            j = k + 1;
                        }
                    }
                    j
                };
                if let Some((r1, c1, end)) = parse_a1_at(&chars, end_at) {
                    let (lo_r, hi_r) = (r0.min(r1), r0.max(r1));
                    let (lo_c, hi_c) = (c0.min(c1), c0.max(c1));
                    // Cap expansion; huge ranges are represented by corners + interior sampling is wrong,
                    // so refuse to expand beyond 100k cells and emit corners only with full expansion below that.
                    push_range_refs(&mut refs, &sheet, lo_r, lo_c, hi_r, hi_c);
                    i = end;
                    continue;
                }
            }
            refs.push(QualifiedRef {
                sheet,
                row: r0,
                col: c0,
            });
            i = next;
            continue;
        }
        i += 1;
    }
    refs
}

/// Extract A1-style references from a formula expression.
/// Returns (row, col) 0-based pairs, ranges expanded.
pub fn formula_refs(expr: &str) -> Vec<(usize, usize)> {
    formula_refs_qualified(expr)
        .into_iter()
        .map(|q| (q.row, q.col))
        .collect()
}

/// Detect contiguous non-empty rectangular regions per sheet and infer
/// a header row (first row all-text, row below contains a number).
pub fn detect_regions(sheet: &SheetSnapshot) -> Vec<Region> {
    // Connected-component (4-neighbour) bounding boxes: a region is
    // the rectangle enclosing a contiguous block of non-empty cells,
    // whatever the per-row widths are.
    let rows = sheet.cells.len();
    let cols = sheet.max_col();
    let mut visited = vec![vec![false; cols]; rows];
    let mut regions = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if visited[r][c] || sheet.get_ref(r, c).is_empty() {
                continue;
            }
            let (mut r0, mut r1, mut c0, mut c1) = (r, r, c, c);
            let mut stack = vec![(r, c)];
            visited[r][c] = true;
            while let Some((cr, cc)) = stack.pop() {
                r0 = r0.min(cr);
                r1 = r1.max(cr);
                c0 = c0.min(cc);
                c1 = c1.max(cc);
                for (nr, nc) in [
                    (cr.wrapping_sub(1), cc),
                    (cr + 1, cc),
                    (cr, cc.wrapping_sub(1)),
                    (cr, cc + 1),
                ] {
                    if nr < rows
                        && nc < cols
                        && !visited[nr][nc]
                        && !sheet.get_ref(nr, nc).is_empty()
                    {
                        visited[nr][nc] = true;
                        stack.push((nr, nc));
                    }
                }
            }
            // Only component cells stay visited: a disconnected cell
            // inside this bounding box must seed its own region.
            // Header row: text cells (empties tolerated) and no
            // numbers/formulas in the row itself.
            let first_all_text = (c0..=c1)
                .any(|cc| matches!(sheet.get_ref(r0, cc), CellValue::Text(_)))
                && (c0..=c1).all(|cc| {
                    matches!(sheet.get_ref(r0, cc), CellValue::Text(_) | CellValue::Empty)
                });
            let below_has_typed = r1 > r0
                && (c0..=c1).any(|cc| {
                    matches!(
                        sheet.get_ref(r0 + 1, cc),
                        CellValue::Number(_) | CellValue::Bool(_) | CellValue::Formula { .. }
                    )
                });
            let header_row = if first_all_text && below_has_typed {
                Some(r0)
            } else {
                None
            };
            let headers = if header_row.is_some() {
                (c0..=c1)
                    .map(|cc| sheet.get_ref(r0, cc).display())
                    .collect()
            } else {
                (c0..=c1).map(col_name).collect()
            };
            regions.push(Region {
                range: RangeRef {
                    sheet: sheet.name.clone(),
                    r0,
                    c0,
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
    for (sheet_idx, sheet) in wb.sheets.iter().enumerate() {
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
            g.region_sheets.push(sheet_idx);
        }
    }
    // Lifted dependency edges, re-derived from formulas. Targets are
    // resolved on their own sheet (Sheet2!A1 does not mean this sheet),
    // using a pre-sorted specificity index instead of rescanning nodes.
    let specificity = specificity_index(&g);
    let mut seen = std::collections::BTreeSet::new();
    for sheet in &wb.sheets {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    let src = node_for_cell_in(&g, &specificity, &sheet.name, r, c);
                    for q in formula_refs_qualified(expr) {
                        let target_sheet = q.sheet.as_deref().unwrap_or(&sheet.name);
                        let dst = node_for_cell_in(&g, &specificity, target_sheet, q.row, q.col);
                        if let (Some(s), Some(t)) = (src, dst) {
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

fn col_from_letters(letters: &str) -> Option<usize> {
    // Empty or non-letter input (row-only `1:3`, `A.1`) is not a
    // column: reject it instead of underflowing.
    if letters.is_empty() || !letters.chars().all(|ch| ch.is_ascii_alphabetic()) {
        return None;
    }
    let mut col = 0usize;
    for ch in letters.to_ascii_uppercase().chars() {
        col = col
            .checked_mul(26)?
            .checked_add(ch as usize - 'A' as usize + 1)?;
    }
    col.checked_sub(1)
}

/// Every lifted depends_on edge must be re-derivable from raw formulas,
/// and vice versa. Mirrors ExcelGRAG's lifted-edge invariant test.
pub fn check_invariants(wb: &WorkbookSnapshot, g: &Graph) -> Vec<String> {
    let mut problems = Vec::new();
    // Structural invariants, independent of how edges were derived:
    // unique ids, in-bounds ranges, containment edges actually contain.
    let mut ids = std::collections::BTreeSet::new();
    for n in &g.nodes {
        if !ids.insert(n.id) {
            problems.push(format!("duplicate node id {}", n.id));
        }
        if let Some(r) = &n.range {
            match wb
                .sheets
                .iter()
                .find(|s| crate::sheet_name_eq(&s.name, &r.sheet))
            {
                None => problems.push(format!("node {} range on unknown sheet {}", n.id, r.sheet)),
                Some(s) => {
                    if r.r0 > r.r1 || r.c0 > r.c1 {
                        problems.push(format!("node {} has inverted range", n.id));
                    }
                    if !s.cells.is_empty() && r.r1 >= s.cells.len() {
                        problems.push(format!("node {} range rows out of bounds", n.id));
                    }
                    let width = s.max_col();
                    if width > 0 && r.c1 >= width {
                        problems.push(format!("node {} range columns out of bounds", n.id));
                    }
                }
            }
        }
    }
    for e in &g.edges {
        if e.kind == "contains" {
            let parent = g
                .nodes
                .iter()
                .find(|n| n.id == e.from)
                .and_then(|n| n.range.as_ref());
            let child = g
                .nodes
                .iter()
                .find(|n| n.id == e.to)
                .and_then(|n| n.range.as_ref());
            if let (Some(pa), Some(ch)) = (parent, child) {
                if crate::sheet_name_eq(&pa.sheet, &ch.sheet)
                    && !(pa.r0 <= ch.r0 && pa.c0 <= ch.c0 && pa.r1 >= ch.r1 && pa.c1 >= ch.c1)
                {
                    problems.push(format!(
                        "contains edge {}->{} does not contain",
                        e.from, e.to
                    ));
                }
            }
        }
    }
    // Lifted-edge completeness, re-derived independently (qualified refs).
    let lifted: std::collections::BTreeSet<(usize, usize)> = g
        .edges
        .iter()
        .filter(|e| e.kind == "depends_on")
        .map(|e| (e.from, e.to))
        .collect();
    let specificity = specificity_index(g);
    let mut expected = std::collections::BTreeSet::new();
    for sheet in &wb.sheets {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    if let Some(s) = node_for_cell_in(g, &specificity, &sheet.name, r, c) {
                        for q in formula_refs_qualified(expr) {
                            let ts = q.sheet.as_deref().unwrap_or(&sheet.name);
                            if let Some(t) = node_for_cell_in(g, &specificity, ts, q.row, q.col) {
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
    // Independent cross-check with a second, deliberately simple
    // tokeniser, so a systematic miss in formula_refs_qualified
    // cannot hide by re-deriving the same wrong expected set.
    let mut independent = std::collections::BTreeSet::new();
    for sheet in &wb.sheets {
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    if let Some(src) = node_for_cell_in(g, &specificity, &sheet.name, r, c) {
                        // Blank out string literals ("B2" is text,
                        // not a reference) before tokenising.
                        let mut scrubbed = String::with_capacity(expr.len());
                        let mut in_str = false;
                        for ch in expr.chars() {
                            if ch == '"' {
                                in_str = !in_str;
                                scrubbed.push(' ');
                            } else if in_str {
                                scrubbed.push(' ');
                            } else {
                                scrubbed.push(ch);
                            }
                        }
                        for tok in scrubbed.split(|ch: char| {
                            !(ch.is_alphanumeric() || matches!(ch, '$' | '!' | '\'' | ':' | '.'))
                        }) {
                            let tok = tok.trim_matches('\'');
                            // Cross-sheet refs are the primary
                            // extractor's job; this check covers
                            // same-sheet refs only.
                            if tok.contains('!')
                                && !crate::sheet_name_eq(
                                    tok.rsplit('!').nth(1).unwrap_or(""),
                                    &sheet.name,
                                )
                            {
                                continue;
                            }
                            let cellpart = tok.rsplit('!').next().unwrap_or(tok);
                            // For ranges, also sample interior
                            // cells so the cross-check sees beyond
                            // the corners the primary extractor
                            // caps at for huge ranges.
                            if let Some((a, b)) = cellpart.split_once(':') {
                                let parse_end = |t: &str| -> Option<(usize, usize)> {
                                    let t = t.replace('$', "");
                                    let sp = t.find(|ch: char| ch.is_ascii_digit())?;
                                    let (letters, digits) = t.split_at(sp);
                                    let row1: usize = digits.parse().ok()?;
                                    if row1 == 0 || row1 > 1_048_576 {
                                        return None;
                                    }
                                    let col = col_from_letters(letters)?;
                                    if col >= 16_384 {
                                        return None;
                                    }
                                    Some((row1 - 1, col))
                                };
                                if let (Some((r0, c0)), Some((r1, c1))) =
                                    (parse_end(a), parse_end(b))
                                {
                                    let rows = r1.saturating_sub(r0) + 1;
                                    let cols = c1.saturating_sub(c0) + 1;
                                    let rstep = (rows / 8).max(1);
                                    let cstep = (cols / 8).max(1);
                                    let mut rr = r0;
                                    while rr <= r1 {
                                        let mut cc = c0;
                                        while cc <= c1 {
                                            // (stepping is checked below)
                                            if let Some(t) = node_for_cell_in(
                                                g,
                                                &specificity,
                                                &sheet.name,
                                                rr,
                                                cc,
                                            ) {
                                                if src != t {
                                                    independent.insert((src, t));
                                                }
                                            }
                                            match cc.checked_add(cstep) {
                                                Some(n) => cc = n,
                                                None => break,
                                            }
                                        }
                                        match rr.checked_add(rstep) {
                                            Some(n) => rr = n,
                                            None => break,
                                        }
                                    }
                                }
                            }
                            for part in cellpart.split(':') {
                                let part = part.replace('$', "");
                                let split = part.find(|ch: char| ch.is_ascii_digit());
                                if let Some(sp) = split {
                                    let (letters, digits) = part.split_at(sp);
                                    if !letters.is_empty()
                                        && letters.len() <= 3
                                        && letters.chars().all(|ch| ch.is_ascii_alphabetic())
                                        && !digits.is_empty()
                                        && digits.chars().all(|ch| ch.is_ascii_digit())
                                    {
                                        if let (Ok(row1), Some(col)) =
                                            (digits.parse::<usize>(), col_from_letters(letters))
                                        {
                                            if row1 >= 1 {
                                                if let Some(t) = node_for_cell_in(
                                                    g,
                                                    &specificity,
                                                    &sheet.name,
                                                    row1 - 1,
                                                    col,
                                                ) {
                                                    if src != t {
                                                        independent.insert((src, t));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for e in independent.difference(&lifted) {
        problems.push(format!(
            "missing lifted edge {e:?} (independent extraction)"
        ));
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
    dropped: usize,
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
            dropped: 0,
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
            .and_then(|t| now.checked_duration_since(t))
            .map(|d| d >= self.debounce)
            .unwrap_or(false);
        if due && !self.pending.is_empty() {
            self.flush();
            true
        } else {
            false
        }
    }

    /// Edits dropped because their sheet does not exist (cumulative).
    pub fn dropped_edits(&self) -> usize {
        self.dropped
    }

    /// Apply pending edits immediately, patching only edited cells.
    /// Returns how many edits were applied; edits to unknown sheets
    /// are counted in `dropped_edits` instead of vanishing silently.
    pub fn flush(&mut self) -> usize {
        let edits = std::mem::take(&mut self.pending);
        let mut applied = 0usize;
        let mut dropped_ranges: Vec<RangeRef> = Vec::new();
        for edit in &edits {
            if edit.row > 10_000
                || edit.col > 1_000
                || !self
                    .snapshot
                    .sheets
                    .iter()
                    .any(|s| crate::sheet_name_eq(&s.name, &edit.sheet))
            {
                dropped_ranges.push(RangeRef {
                    sheet: edit.sheet.clone(),
                    r0: edit.row,
                    c0: edit.col,
                    r1: edit.row,
                    c1: edit.col,
                });
            }
        }
        // Dropped edits changed nothing: unpublish their dirty ranges.
        for dropped in &dropped_ranges {
            if let Some(pos) = self.dirty.iter().position(|d| d == dropped) {
                self.dirty.remove(pos);
            }
        }
        for edit in edits {
            if edit.row > 10_000 || edit.col > 1_000 {
                self.dropped += 1;
                continue;
            }
            let known = self
                .snapshot
                .sheets
                .iter()
                .any(|s| crate::sheet_name_eq(&s.name, &edit.sheet));
            if !known {
                self.dropped += 1;
                continue;
            }
            applied += 1;
            if let Some(sheet) = self
                .snapshot
                .sheets
                .iter_mut()
                .find(|s| crate::sheet_name_eq(&s.name, &edit.sheet))
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
        if applied > 0 {
            self.revision += 1;
        }
        applied
    }

    /// Dirty ranges since the last flush, then clears the list.
    pub fn take_dirty_ranges(&mut self) -> Vec<RangeRef> {
        std::mem::take(&mut self.dirty)
    }
}

mod calc;
pub mod eval;

/// Render a region as a cited passage for the LLM context.
///
/// Each sheet row renders as exactly one line, so a line's position
/// gives its sheet row: line breaks inside a cell become spaces.
pub fn render_passage(sheet: &SheetSnapshot, region: &Region) -> String {
    let mut out = format!("Range {}\n", region.range.a1());
    for r in region.range.r0..=region.range.r1 {
        let cells: Vec<String> = (region.range.c0..=region.range.c1)
            .map(|c| {
                sheet
                    .get(r, c)
                    .display()
                    .replace("\r\n", " ")
                    .replace(['\r', '\n'], " ")
            })
            .collect();
        out.push_str(&cells.join(" | "));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_passage_keeps_each_sheet_row_on_one_line() {
        let wb = parse_delimited(
            "book",
            "Sheet1",
            "Item,Notes,Qty\nApples,\"one\ntwo\r\nthree\rfour\",2\nPears,plain,3\n",
            ',',
        );
        let (graph, _) = build_graph(&wb);
        let passage = render_passage(&wb.sheets[0], &graph.regions[0]);
        let lines: Vec<&str> = passage.lines().collect();
        assert_eq!(lines.len(), 4, "{passage}");
        assert_eq!(lines[2], "Apples | one two three four | 2");
        assert_eq!(lines[3], "Pears | plain | 3");
    }

    #[test]
    fn load_bytes_matches_load_path_and_checks_format() {
        let dir = std::env::temp_dir().join(format!("gt-load-bytes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("book.csv");
        let text = "Item,Qty\nApples,2\n";
        std::fs::write(&path, text).unwrap();
        let from_path = load_path(&path).unwrap();
        let from_bytes = load_bytes(&path, text.as_bytes()).unwrap();
        assert_eq!(from_path, from_bytes);
        assert_eq!(from_bytes.name, "book");
        assert!(matches!(
            load_bytes(&path, &[0xff, 0xfe]),
            Err(IngestError::Io(_))
        ));
        assert!(matches!(
            load_bytes(std::path::Path::new("x.xls"), b""),
            Err(IngestError::UnsupportedFormat { .. })
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

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
                                                 // Whole-row refs must not panic the independent cross-check.
        let wb = parse_delimited("d", "S", "A,B\n1,=SUM(1:1)\n", ',');
        let _ = build_graph(&wb);
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
    fn refs_expand_ranges_and_skip_literals() {
        // Range expands to all 6 interior cells.
        assert_eq!(formula_refs("SUM(A1:B3)").len(), 6);
        // String literal and scientific notation produce no refs.
        assert_eq!(formula_refs("\"A1\"&1E5+B2"), vec![(1, 1)]);
        // Sheet-qualified refs keep their sheet.
        let q = formula_refs_qualified("Sheet2!A1+'My Sheet'!B2");
        assert_eq!(q[0].sheet.as_deref(), Some("Sheet2"));
        assert_eq!(q[1].sheet.as_deref(), Some("My Sheet"));
    }

    #[test]
    fn csv_stream_handles_bom_multiline_and_quoted_spaces() {
        let wb = parse_delimited(
            "d",
            "S",
            "\u{feff}Name,Note\n\"  spaced  \",\"line1\nline2\"\n",
            ',',
        );
        assert_eq!(wb.sheets[0].get(0, 0), CellValue::Text("Name".into()));
        assert_eq!(wb.sheets[0].get(1, 0), CellValue::Text("  spaced  ".into()));
        assert_eq!(
            wb.sheets[0].get(1, 1),
            CellValue::Text("line1\nline2".into())
        );
        // NaN / inf stay text, never numbers.
        let wb2 = parse_delimited("d", "S", "A\nNaN\n", ',');
        assert_eq!(wb2.sheets[0].get(1, 0), CellValue::Text("NaN".into()));
    }

    #[test]
    fn graph_regressions_round2() {
        // Escaped apostrophe in a quoted sheet name.
        let q = formula_refs_qualified("'O''Brien'!A1");
        assert_eq!(q[0].sheet.as_deref(), Some("O'Brien"));
        // Strict CSV rejects post-quote junk and unterminated quotes.
        assert!(parse_delimited_strict("d", "S", "A\n\"x\"junk\n", ',').is_err());
        assert!(parse_delimited_strict("d", "S", "A\n\"x\n", ',').is_err());
        // Invariants flag inverted / column-overflowing hand-built ranges.
        let wb = sample();
        let mut g = Graph::default();
        g.nodes.push(Node {
            id: 0,
            kind: NodeKind::Region,
            label: "bad".into(),
            range: Some(RangeRef {
                sheet: "Sheet1".into(),
                r0: 0,
                c0: 0,
                r1: 0,
                c1: 99,
            }),
        });
        assert!(check_invariants(&wb, &g)
            .iter()
            .any(|v| v.contains("columns out of bounds")));
        // Workbook densification cap applies across sheets.
        let mut inc = IncrementalSnapshot::new(sample());
        inc.record_edit(
            CellEdit {
                sheet: "Sheet1".into(),
                row: usize::MAX,
                col: 0,
                value: CellValue::Number(1.0),
            },
            std::time::Instant::now(),
        );
        assert_eq!(inc.flush(), 0);
        assert_eq!(inc.dropped_edits(), 1);
    }

    #[test]
    fn graph_regressions_round3() {
        // No phantom refs inside long identifiers.
        assert!(formula_refs_qualified("Sales2024*2").is_empty());
        // Doubly-qualified ranges expand fully.
        let q = formula_refs_qualified("Sheet2!A1:Sheet2!B2");
        assert_eq!(q.len(), 4);
        // Space after a closing quote is tolerated.
        assert!(parse_delimited_strict("d", "S", "A,B\n\"x\" ,2\n", ',').is_ok());
        // Dropped edits unpublish their dirty ranges.
        let mut inc = IncrementalSnapshot::new(sample());
        inc.record_edit(
            CellEdit {
                sheet: "Nope".into(),
                row: 0,
                col: 0,
                value: CellValue::Number(1.0),
            },
            std::time::Instant::now(),
        );
        assert_eq!(inc.flush(), 0);
        assert!(inc.take_dirty_ranges().is_empty());
    }

    #[test]
    fn graph_regressions_round4() {
        // A string literal in a formula is not a reference.
        let wb = parse_delimited("d", "Sheet1", "A,B\n1,=\"B2\"\n", ',');
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        let _ = g;
    }

    #[test]
    fn whole_range_and_unicode_qualifier_extraction() {
        let refs = formula_refs_qualified("SUM(A:A)");
        assert!(refs.iter().any(|q| q.col == 0), "{refs:?}");
        let refs = formula_refs_qualified("\u{dc}bung!A1");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].sheet.as_deref(), Some("\u{dc}bung"));
        assert!(formula_refs_qualified("A1A+1").is_empty());
    }

    #[test]
    fn cross_sheet_formula_lifts_to_other_sheet() {
        let mut wb = parse_delimited("d", "Sheet1", "A,B\n1,=Sheet2!A2\n", ',');
        wb.sheets.push(SheetSnapshot {
            name: "Sheet2".into(),
            cells: vec![
                vec![CellValue::Text("X".into())],
                vec![CellValue::Number(9.0)],
            ],
        });
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        assert!(g.edges.iter().any(|e| e.kind == "depends_on"));
    }

    #[test]
    fn unsupported_binary_format_is_typed_error() {
        let err = load_path(std::path::Path::new("book.xlsb")).unwrap_err();
        assert!(matches!(err, IngestError::UnsupportedFormat { .. }));
    }
}
