//! GridTalk graph layer: workbook snapshots, region inference, and the
//! lifted property graph (workbook -> sheet -> region -> column).
//! Ported in spirit from ExcelGRAG's eg-structure / eg-graph, retargeted
//! to snapshots so downstream code never touches live grid cells.

use gridcraft_core::{CellRef as GcRef, Value as GcValue};
use serde::{Deserialize, Serialize};
use std::io::Read;

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

/// One row of a sheet. Rows are shared between copies of a snapshot:
/// cloning a snapshot copies one pointer per row, not every cell, and
/// an edit copies only the row it changes, and only while another copy
/// still holds it (`Arc::make_mut`, see [`SheetSnapshot::cell_mut`]).
/// So the UI can hand each revision to its worker threads without
/// copying the workbook on every edit.
pub type Row = std::sync::Arc<Vec<CellValue>>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SheetSnapshot {
    pub name: String,
    /// Row-major cells; rows may be ragged. Read them as usual
    /// (`cells[r][c]`, `cells.get(r)`); write through
    /// [`cell_mut`](Self::cell_mut) or `Arc::make_mut`.
    pub cells: Vec<Row>,
}

impl SheetSnapshot {
    /// A sheet from row-major cells.
    pub fn new(name: impl Into<String>, rows: Vec<Vec<CellValue>>) -> Self {
        SheetSnapshot {
            name: name.into(),
            cells: rows.into_iter().map(std::sync::Arc::new).collect(),
        }
    }

    /// The cell at (row, col) for writing, growing the sheet (with
    /// empty cells) to reach it. Only this row is copied if another
    /// snapshot shares it.
    pub fn cell_mut(&mut self, row: usize, col: usize) -> &mut CellValue {
        if self.cells.len() <= row {
            self.cells.resize(row + 1, Row::default());
        }
        let cells = std::sync::Arc::make_mut(&mut self.cells[row]);
        if cells.len() <= col {
            cells.resize(col + 1, CellValue::Empty);
        }
        &mut cells[col]
    }

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
    /// What the reader skipped or could not import (Gridcraft's
    /// `ReadReport`, e.g. ODS formulas imported as their results).
    /// Shown to users so a partial import is never silent.
    #[serde(default)]
    pub warnings: Vec<String>,
    /// The Gridcraft workbook an XLSX/ODS snapshot was read from
    /// (decision 0005). It carries everything the snapshot does not
    /// model: styles, merges, tables, array formulas, error values,
    /// the date system, charts. [`to_gridcraft_workbook`] starts from
    /// it and overlays only the cells the snapshot changed, so
    /// evaluation and saving keep those features. `None` for CSV/TSV
    /// and hand-built snapshots.
    #[serde(skip)]
    pub source: Option<std::sync::Arc<gridcraft_model::Workbook>>,
}

impl WorkbookSnapshot {
    /// An estimate of the heap this snapshot holds: every cell's slot
    /// plus the text, formula text and cached value it owns. Rows
    /// shared with another snapshot are counted in full. It does not
    /// count the Gridcraft `source` workbook or anything built from
    /// the snapshot (graph, index); callers that budget memory scale
    /// it for those.
    pub fn approx_bytes(&self) -> u64 {
        fn cell(c: &CellValue) -> u64 {
            match c {
                CellValue::Text(t) => t.capacity() as u64,
                CellValue::Formula { expr, cached } => {
                    expr.capacity() as u64
                        + cached
                            .as_deref()
                            .map_or(0, |v| std::mem::size_of::<CellValue>() as u64 + cell(v))
                }
                _ => 0,
            }
        }
        let slot = std::mem::size_of::<CellValue>() as u64;
        self.sheets
            .iter()
            .flat_map(|s| s.cells.iter())
            .map(|row| row.capacity() as u64 * slot + row.iter().map(cell).sum::<u64>())
            .sum()
    }
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

/// Cell → node lookups for a built graph: its region and column
/// nodes grouped by sheet, most specific (smallest area) first. The
/// answers match [`node_for_cell_in`], but each lookup scans one
/// sheet's nodes with integer range checks instead of every node with
/// a case-folded sheet-name comparison, which dominated graph builds
/// and invariant checks on large workbooks. It owns its data, so the
/// graph can keep growing edges while it is in use.
pub struct NodeLocator {
    /// (sheet name, its nodes in specificity order).
    sheets: Vec<(String, Vec<LocatedNode>)>,
}

/// A node id and its range bounds: (id, r0, c0, r1, c1).
type LocatedNode = (usize, usize, usize, usize, usize);

impl NodeLocator {
    pub fn new(g: &Graph) -> Self {
        let mut sheets: Vec<(String, Vec<LocatedNode>)> = Vec::new();
        for id in specificity_index(g) {
            let Some(r) = g
                .nodes
                .get(id)
                .filter(|n| n.id == id)
                .or_else(|| g.nodes.iter().find(|n| n.id == id))
                .and_then(|n| n.range.as_ref())
            else {
                continue;
            };
            let entry = (id, r.r0, r.c0, r.r1, r.c1);
            match sheets
                .iter_mut()
                .find(|(name, _)| sheet_name_eq(name, &r.sheet))
            {
                Some((_, nodes)) => nodes.push(entry),
                None => sheets.push((r.sheet.clone(), vec![entry])),
            }
        }
        NodeLocator { sheets }
    }

    /// The slot of a sheet (matched case-insensitively), for repeated
    /// lookups on it with [`find`](Self::find).
    pub fn sheet(&self, name: &str) -> Option<usize> {
        self.sheets.iter().position(|(n, _)| sheet_name_eq(n, name))
    }

    /// The most specific node holding (row, col) on a sheet slot.
    pub fn find(&self, slot: usize, row: usize, col: usize) -> Option<usize> {
        self.sheets
            .get(slot)?
            .1
            .iter()
            .find_map(|&(id, r0, c0, r1, c1)| {
                (r0 <= row && row <= r1 && c0 <= col && col <= c1).then_some(id)
            })
    }

    /// The most specific node holding a cell of a named sheet.
    pub fn node_for_cell(&self, sheet: &str, row: usize, col: usize) -> Option<usize> {
        self.find(self.sheet(sheet)?, row, col)
    }
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
pub const MAX_DELIMITED_BYTES: u64 = 32 * 1024 * 1024;
/// Ingestion cap for XLSX/ODS packages.
pub const MAX_PACKAGE_BYTES: u64 = 256 * 1024 * 1024;

/// Largest total unpacked size an XLSX/ODS package may declare: room
/// for a sheet at [`MAX_SNAPSHOT_CELLS`] plus its strings and styles,
/// and half of what Gridcraft's reader allows (1 GiB).
pub const MAX_UNPACKED_BYTES: u64 = 512 * 1024 * 1024;
/// Most zip entries a package may have (Gridcraft's own limit).
const MAX_PACKAGE_ENTRIES: usize = 200_000;

/// Reject a package whose zip directory declares more than
/// [`MAX_UNPACKED_BYTES`] unpacked, before Gridcraft decompresses
/// anything. This reads only the directory at the end of the file, so
/// it costs no decompression. A package that understates its sizes
/// passes this check, but Gridcraft's reader counts the bytes it
/// actually unpacks and stops at 512 MiB per part and 1 GiB in all.
/// Bytes that are not a zip archive are left to Gridcraft to report.
fn check_package_unpacked(bytes: &[u8]) -> Result<(), IngestError> {
    let Ok(mut zip) = zip::ZipArchive::new(std::io::Cursor::new(bytes)) else {
        return Ok(());
    };
    if zip.len() > MAX_PACKAGE_ENTRIES {
        return Err(IngestError::Parse(format!(
            "spreadsheet package has {} zip entries, over the {MAX_PACKAGE_ENTRIES}-entry cap",
            zip.len()
        )));
    }
    let mut total: u64 = 0;
    for i in 0..zip.len() {
        let Ok(entry) = zip.by_index_raw(i) else {
            continue;
        };
        total = total.saturating_add(entry.size());
        if total > MAX_UNPACKED_BYTES {
            return Err(IngestError::Parse(format!(
                "spreadsheet package unpacks to more than {MAX_UNPACKED_BYTES} bytes (the cap was passed at {}, which declares {} bytes)",
                entry.name(),
                entry.size()
            )));
        }
    }
    Ok(())
}

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

/// Spreadsheet files read through Gridcraft (everything but CSV/TSV),
/// all under [`MAX_PACKAGE_BYTES`]. The reader is chosen from the
/// content, not the extension (see [`load_gridcraft_bytes`]).
pub const PACKAGE_EXTENSIONS: &[&str] = &["xlsx", "xlsm", "xltx", "xltm", "xls", "xlsb", "ods"];

/// Format check and size cap for an ingestible file of `len` bytes.
fn check_ingestible(ext: &str, len: u64) -> Result<(), IngestError> {
    match ext {
        "csv" | "tsv" if len > MAX_DELIMITED_BYTES => Err(IngestError::Parse(format!(
            "delimited file is {len} bytes, over the {MAX_DELIMITED_BYTES}-byte ingestion cap"
        ))),
        "csv" | "tsv" => Ok(()),
        _ if PACKAGE_EXTENSIONS.contains(&ext) && len > MAX_PACKAGE_BYTES => {
            Err(IngestError::Parse(format!(
                "spreadsheet file is {len} bytes, over the {MAX_PACKAGE_BYTES}-byte ingestion cap"
            )))
        }
        _ if PACKAGE_EXTENSIONS.contains(&ext) => Ok(()),
        _ => Err(IngestError::UnsupportedFormat {
            ext: ext.into(),
            hint: format!(
                "expected .csv, .tsv or one of .{}",
                PACKAGE_EXTENSIONS.join(", .")
            ),
        }),
    }
}

/// Load a workbook snapshot from a path. CSV/TSV are fully supported.
/// XLSX/ODS go through the Gridcraft engine. XLSB/XLS are explicitly
/// out of scope (see plan Phase 2). Unsupported or oversized files
/// are rejected before more than the cap is read
/// ([`read_workbook_bytes`]).
pub fn load_path(path: &std::path::Path) -> Result<WorkbookSnapshot, IngestError> {
    let bytes = read_workbook_bytes(path)?;
    load_bytes(path, &bytes)
}

/// Read a workbook file's bytes, never more than its format's
/// ingestion cap. The format is checked first, the path must be a
/// regular file (a FIFO or device such as `/dev/zero` reports length
/// 0 and would otherwise be read without bound), and at most one byte
/// past the cap is read, so a file that grows after the size check is
/// still rejected.
pub fn read_workbook_bytes(path: &std::path::Path) -> Result<Vec<u8>, IngestError> {
    let (ext, _) = ext_and_name(path);
    check_ingestible(&ext, 0)?;
    let meta = std::fs::metadata(path)?;
    if !meta.is_file() {
        return Err(IngestError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} is not a regular file", path.display()),
        )));
    }
    check_ingestible(&ext, meta.len())?;
    let cap = if ext == "csv" || ext == "tsv" {
        MAX_DELIMITED_BYTES
    } else {
        MAX_PACKAGE_BYTES
    };
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(cap + 1)
        .read_to_end(&mut bytes)?;
    check_ingestible(&ext, bytes.len() as u64)?;
    Ok(bytes)
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

/// What GridTalk adds to Gridcraft's own warning for an XLSB file:
/// what values-only means for the graph and the tools.
pub const XLSB_VALUES_ONLY_WARNING: &str = "Opened as values only: GridTalk cannot read .xlsb formulas yet, so every formula cell holds its last saved result as a constant. There are no formula dependencies in the graph, list_formulas finds none, and what_if changes nothing downstream. For live formulas, save the workbook as .xlsx in Excel and open that.";

/// One of Gridcraft's readers (`read_xlsx`, `read_xls`, ...).
type GridcraftReader =
    fn(
        &[u8],
    )
        -> Result<(gridcraft_model::Workbook, gridcraft_xlsx::ReadReport), gridcraft_xlsx::IoError>;

/// Read a spreadsheet file (anything but CSV/TSV) through the vendored
/// Gridcraft engine and convert its workbook model into a GridTalk
/// snapshot. The reader is chosen from the content, so a file whose
/// extension is wrong (an `.xls` download that is really an XLSX
/// package) still opens, with a warning; `ext` only names the format
/// in messages.
pub fn load_gridcraft_bytes(
    name: &str,
    ext: &str,
    bytes: &[u8],
) -> Result<WorkbookSnapshot, IngestError> {
    use gridcraft_xlsx::Format;
    let format = gridcraft_xlsx::sniff(bytes);
    let (read, label, expected): (GridcraftReader, &str, &[&str]) = match format {
        Format::Xlsx => (
            gridcraft_xlsx::read_xlsx,
            "XLSX",
            &["xlsx", "xlsm", "xltx", "xltm"],
        ),
        Format::Xlsb => (gridcraft_xlsx::read_xlsb, "XLSB", &["xlsb"]),
        Format::Ods => (gridcraft_xlsx::read_ods, "ODS", &["ods"]),
        Format::Xls => (gridcraft_xlsx::read_xls, "XLS", &["xls"]),
        Format::Encrypted => {
            return Err(IngestError::Parse(
                "the workbook is password-protected (encrypted); GridTalk cannot open encrypted workbooks yet. Remove the password in Excel (File > Info > Protect Workbook) and open it again".into(),
            ))
        }
        Format::LegacyBinary => {
            return Err(IngestError::Parse(format!(
                "this .{ext} file is an Office compound file but not an Excel workbook"
            )))
        }
        Format::Csv | Format::Unknown => {
            return Err(IngestError::Parse(format!(
                "this .{ext} file is not a spreadsheet package (no XLSX, XLSB, XLS or ODS content); if it is text, such as a CSV or an HTML export, save it as .csv"
            )))
        }
    };
    // Zip packages only: the XLS reader bounds its own records.
    if format != Format::Xls {
        check_package_unpacked(bytes)?;
    }
    let (wb, report) =
        read(bytes).map_err(|e| IngestError::Parse(format!("gridcraft {label} reader: {e}")))?;
    let mut snapshot = from_gridcraft_workbook(name, &wb)?;
    let mut warnings = Vec::new();
    if !expected.contains(&ext) {
        warnings.push(format!(
            "this .{ext} file holds {label} content and was read as {label}"
        ));
    }
    if format == Format::Xlsb {
        warnings.push(XLSB_VALUES_ONLY_WARNING.to_string());
    }
    warnings.extend(report.warnings);
    snapshot.warnings = warnings;
    snapshot.source = Some(std::sync::Arc::new(wb));
    Ok(snapshot)
}

/// Cap on densified snapshot cells per sheet (and per workbook): a
/// sparse XLSX with a single cell at XFD1048576 must not allocate a
/// billion-cell grid.
pub const MAX_SNAPSHOT_CELLS: usize = 4_000_000;

/// Excel's grid: 1,048,576 rows by 16,384 columns.
pub const EXCEL_MAX_ROWS: usize = 1 << 20;
pub const EXCEL_MAX_COLS: usize = 1 << 14;

/// Convert a Gridcraft `Workbook` into a snapshot. Formulas keep their
/// canonical text; the cached value (when Gridcraft has one) is kept
/// alongside so passages can show both.
pub fn from_gridcraft_workbook(
    name: &str,
    wb: &gridcraft_model::Workbook,
) -> Result<WorkbookSnapshot, IngestError> {
    let mut sheets = Vec::new();
    let mut total_cells: u64 = 0;
    for sheet in &wb.sheets {
        let (mut max_row, mut max_col) = (0usize, 0usize);
        for (addr, _) in sheet.cells.iter() {
            max_row = max_row.max(addr.row as usize);
            max_col = max_col.max(addr.col as usize);
        }
        // An empty sheet densifies to no cells at all.
        let empty = sheet.cells.iter().next().is_none();
        let area = if empty {
            0
        } else {
            (max_row as u64 + 1).saturating_mul(max_col as u64 + 1)
        };
        // The per-sheet check comes first: the running total is never
        // below `area`, so checked second it could never fire.
        if area > MAX_SNAPSHOT_CELLS as u64 {
            return Err(IngestError::Parse(format!(
                "sheet {} is too sparse to snapshot ({area} cells > {MAX_SNAPSHOT_CELLS}); trim empty rows/columns at the edges",
                sheet.name
            )));
        }
        total_cells = total_cells.saturating_add(area);
        if total_cells > MAX_SNAPSHOT_CELLS as u64 {
            return Err(IngestError::Parse(format!(
                "workbook is too sparse to snapshot ({total_cells} densified cells > {MAX_SNAPSHOT_CELLS} across all sheets)"
            )));
        }
        let mut cells = if empty {
            Vec::new()
        } else {
            vec![vec![CellValue::Empty; max_col + 1]; max_row + 1]
        };
        for (addr, cell) in sheet.cells.iter() {
            cells[addr.row as usize][addr.col as usize] = snapshot_cell(cell);
        }
        sheets.push(SheetSnapshot::new(sheet.name.clone(), cells));
    }
    Ok(WorkbookSnapshot {
        name: name.into(),
        sheets,
        names: snapshot_names(wb),
        warnings: Vec::new(),
        source: None,
    })
}

/// A Gridcraft cell as a snapshot cell. Formulas keep their
/// canonical text and cached value; an error value (constant or
/// cached) reads as its display text, e.g. `#N/A`.
fn snapshot_cell(cell: &gridcraft_model::Cell) -> CellValue {
    let cached = match &cell.value {
        GcValue::Empty => None,
        GcValue::Number(n) => Some(Box::new(CellValue::Number(*n))),
        GcValue::Text(t) => Some(Box::new(CellValue::Text(t.to_string()))),
        GcValue::Bool(b) => Some(Box::new(CellValue::Bool(*b))),
        other => Some(Box::new(CellValue::Text(other.display()))),
    };
    match &cell.formula {
        Some(f) => CellValue::Formula {
            expr: f.text.clone(),
            cached,
        },
        None => cached.map(|c| *c).unwrap_or(CellValue::Empty),
    }
}

/// A Gridcraft workbook's defined names as the snapshot keeps them.
fn snapshot_names(wb: &gridcraft_model::Workbook) -> Vec<DefinedName> {
    wb.names
        .iter()
        .map(|n| DefinedName {
            name: n.name.clone(),
            scope: n
                .scope
                .and_then(|i| wb.sheets.get(i))
                .map(|s| s.name.clone()),
            formula: n.formula.clone(),
        })
        .collect()
}

/// A snapshot cell as Gridcraft content (value and formula) on top
/// of `base`, which keeps its style. An empty cell clears the content.
/// A formula whose text is unchanged keeps `base`'s formula, so its
/// legacy-array range and dynamic-array mark survive.
fn gridcraft_cell(cell: &CellValue, mut base: gridcraft_model::Cell) -> gridcraft_model::Cell {
    let old_formula = base.formula.take();
    base.clear_content();
    match cell {
        CellValue::Empty => {}
        CellValue::Text(t) => base.value = GcValue::Text(t.as_str().into()),
        CellValue::Number(n) => base.value = GcValue::Number(*n),
        CellValue::Bool(b) => base.value = GcValue::Bool(*b),
        CellValue::Formula { expr, cached } => {
            base.formula = match old_formula {
                Some(f) if f.text == *expr => Some(f),
                _ => Some(std::sync::Arc::new(gridcraft_model::Formula::new(expr))),
            };
            base.value = match cached.as_deref() {
                Some(CellValue::Number(n)) => GcValue::Number(*n),
                Some(CellValue::Bool(b)) => GcValue::Bool(*b),
                Some(CellValue::Text(s)) => GcValue::Text(s.as_str().into()),
                _ => GcValue::Empty,
            };
        }
    }
    base
}

/// Build a Gridcraft workbook from a snapshot. Sheet indices match
/// the snapshot's.
///
/// When the snapshot keeps its [`WorkbookSnapshot::source`] and still
/// has the same sheets, the result is that workbook with only the
/// changed cells rewritten (each keeping its style), so everything
/// the snapshot does not model survives. Otherwise it is built from
/// the snapshot alone: values, formulas with their cached values, and
/// defined names.
pub fn to_gridcraft_workbook(wb: &WorkbookSnapshot) -> gridcraft_model::Workbook {
    match &wb.source {
        Some(source) if same_sheets(wb, source) => overlay_snapshot(wb, source),
        _ => build_from_snapshot(wb),
    }
}

fn same_sheets(wb: &WorkbookSnapshot, source: &gridcraft_model::Workbook) -> bool {
    wb.sheets.len() == source.sheets.len()
        && wb
            .sheets
            .iter()
            .zip(&source.sheets)
            .all(|(a, b)| a.name == b.name)
}

/// `source` with every cell whose snapshot value differs rewritten.
/// Untouched sheets stay shared with `source` (sheets are `Arc`s).
fn overlay_snapshot(
    wb: &WorkbookSnapshot,
    source: &gridcraft_model::Workbook,
) -> gridcraft_model::Workbook {
    let mut gc = source.clone();
    let mut renames: Vec<ColumnRename> = Vec::new();
    for (i, sheet) in wb.sheets.iter().enumerate() {
        let src = &source.sheets[i];
        let mut changed: Vec<(GcRef, gridcraft_model::Cell)> = Vec::new();
        // Cells the source has: rewrite those the snapshot changed.
        for (at, cell) in src.cells.iter() {
            let now = sheet.get_ref(at.row as usize, at.col as usize);
            if *now != snapshot_cell(cell) {
                changed.push((at, gridcraft_cell(now, cell.clone())));
            }
        }
        // Cells only the snapshot has: edits into empty cells, which
        // take their row or column style as Gridcraft's own entry does.
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if cell.is_empty() {
                    continue;
                }
                let at = GcRef::new(r as u32, c as u32);
                if src.cell(at).is_none() {
                    let base = gridcraft_model::Cell::styled(GcValue::Empty, src.style_id(at));
                    changed.push((at, gridcraft_cell(cell, base)));
                }
            }
        }
        if changed.is_empty() {
            continue;
        }
        if let Some(target) = gc.sheet_mut(i) {
            for (at, cell) in &changed {
                target.set_cell(*at, cell.clone());
            }
        }
        let edited: Vec<GcRef> = changed.iter().map(|(at, _)| *at).collect();
        renames.extend(sync_table_headers(&mut gc, i, &edited));
    }
    rename_structured_refs(&mut gc, &renames);
    if wb.names != snapshot_names(source) {
        gc.names = gridcraft_names(wb);
    }
    gc
}

/// A table column renamed by an edit to its header cell: (sheet,
/// table name, table range, old column name, new column name).
type ColumnRename = (usize, String, gridcraft_core::RangeRef, String, String);

/// Rename the table columns whose header cells were edited, as Excel
/// (and Gridcraft's engine) does: the header text names the column,
/// and an empty or repeated header gets a unique generated name that
/// is written back into the cell. Without this, Gridcraft's writer
/// would put the old column name back into the header on save.
fn sync_table_headers(
    gc: &mut gridcraft_model::Workbook,
    sheet: usize,
    edited: &[GcRef],
) -> Vec<ColumnRename> {
    let mut renames = Vec::new();
    let Some(sh) = gc.sheet_mut(sheet) else {
        return renames;
    };
    for &at in edited {
        let Some(ti) = sh
            .tables
            .iter()
            .position(|t| t.header_row && at.row == t.range.start.row && t.range.contains(at))
        else {
            continue;
        };
        let col = (at.col - sh.tables[ti].range.start.col) as usize;
        let Some(old) = sh.tables[ti].columns.get(col).map(|c| c.name.clone()) else {
            continue;
        };
        let value = sh.value(at);
        let is_text = value.is_text() && sh.cell(at).is_some_and(|c| c.formula.is_none());
        let text = match value.as_text() {
            Some(t) => t.to_string(),
            None => value.display(),
        };
        if text == old && is_text {
            continue;
        }
        let table = &sh.tables[ti];
        let taken: std::collections::HashSet<String> = table
            .columns
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != col)
            .map(|(_, c)| c.name.to_lowercase())
            .collect();
        let base = if text.trim().is_empty() {
            format!("Column{}", col + 1)
        } else {
            text.clone()
        };
        let mut name = base.clone();
        let mut k = 2;
        while taken.contains(&name.to_lowercase()) {
            name = format!("{base}{k}");
            k += 1;
        }
        let (table_name, range) = (table.name.clone(), table.range);
        sh.tables[ti].columns[col].name = name.clone();
        // Headers hold constant text: a number, a formula or a
        // generated name is written back as text.
        if name != text || !is_text {
            let style = sh.style_id(at);
            sh.set_cell(
                at,
                gridcraft_model::Cell::styled(GcValue::Text(name.as_str().into()), style),
            );
        }
        if !old.eq_ignore_ascii_case(&name) {
            renames.push((sheet, table_name, range, old, name));
        }
    }
    renames
}

/// Point structured references at renamed table columns.
fn rename_structured_refs(gc: &mut gridcraft_model::Workbook, renames: &[ColumnRename]) {
    if renames.is_empty() {
        return;
    }
    for i in 0..gc.sheets.len() {
        let formulas: Vec<(GcRef, std::sync::Arc<gridcraft_model::Formula>)> = gc.sheets[i]
            .cells
            .iter()
            .filter_map(|(at, c)| c.formula.clone().map(|f| (at, f)))
            .collect();
        for (at, f) in formulas {
            let Some(expr) = f.expr() else {
                continue;
            };
            let mut hit = false;
            let renamed = expr.map(&mut |x| match x {
                gridcraft_formula::Expr::Struct(mut st) => {
                    for (sheet, table, range, old, name) in renames {
                        // `[@Col]` inside the table has no table name.
                        let this = if st.table.is_empty() {
                            i == *sheet && range.contains(at)
                        } else {
                            st.table.eq_ignore_ascii_case(table)
                        };
                        if this {
                            for end in [&mut st.col_start, &mut st.col_end].into_iter().flatten() {
                                if end.eq_ignore_ascii_case(old) {
                                    *end = name.clone();
                                    hit = true;
                                }
                            }
                        }
                    }
                    gridcraft_formula::Expr::Struct(st)
                }
                other => other,
            });
            if !hit {
                continue;
            }
            if let Some(cell) = gc.sheet_mut(i).and_then(|s| s.cells.get_mut(at)) {
                let mut formula = gridcraft_model::Formula::from_expr(renamed);
                formula.array = f.array;
                formula.dynamic = f.dynamic;
                cell.formula = Some(std::sync::Arc::new(formula));
            }
        }
    }
}

fn build_from_snapshot(wb: &WorkbookSnapshot) -> gridcraft_model::Workbook {
    let mut gc = gridcraft_model::Workbook::new();
    gc.sheets.clear();
    for sheet in &wb.sheets {
        let mut gs = gridcraft_model::Sheet::new(sheet.name.clone());
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if !cell.is_empty() {
                    let addr = gridcraft_core::CellRef::new(r as u32, c as u32);
                    gs.set_cell(addr, gridcraft_cell(cell, gridcraft_model::Cell::default()));
                }
            }
        }
        gc.sheets.push(std::sync::Arc::new(gs));
    }
    if gc.sheets.is_empty() {
        gc.sheets
            .push(std::sync::Arc::new(gridcraft_model::Sheet::new("Sheet1")));
    }
    gc.names = gridcraft_names(wb);
    gc
}

pub(crate) fn gridcraft_names(wb: &WorkbookSnapshot) -> Vec<gridcraft_model::DefinedName> {
    wb.names
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
        .collect()
}

/// Serialise a snapshot to XLSX bytes via Gridcraft (used for
/// fixtures and Save-As). Formulas are written as formulas; a
/// snapshot read from XLSX/ODS keeps what its source workbook has
/// beyond cells (see [`to_gridcraft_workbook`]).
pub fn to_xlsx_bytes(wb: &WorkbookSnapshot) -> Result<Vec<u8>, IngestError> {
    to_package_bytes(wb, "xlsx")
}

/// The extensions [`to_package_bytes`] writes: the `.xlsx` family.
pub const WRITABLE_PACKAGE_EXTENSIONS: &[&str] = &["xlsx", "xlsm", "xltx", "xltm"];

/// Serialise a snapshot as the `.xlsx`-family member `ext` names, so
/// the package's content type matches its file name (Excel refuses an
/// `.xlsx` file declaring macros). Only `.xlsm` and `.xltm` keep a VBA
/// project read from the source (see [`has_macros`]); `.xls`, `.xlsb`
/// and `.ods` are read-only.
pub fn to_package_bytes(wb: &WorkbookSnapshot, ext: &str) -> Result<Vec<u8>, IngestError> {
    use gridcraft_xlsx::XlsxKind;
    let kind = match ext {
        "xlsx" => XlsxKind::Workbook,
        "xlsm" => XlsxKind::MacroWorkbook,
        "xltx" => XlsxKind::Template,
        "xltm" => XlsxKind::MacroTemplate,
        _ => {
            return Err(IngestError::UnsupportedFormat {
                ext: ext.into(),
                hint: format!(
                    "GridTalk writes .{} (and .csv/.tsv)",
                    WRITABLE_PACKAGE_EXTENSIONS.join(", .")
                ),
            })
        }
    };
    let gc = to_gridcraft_workbook(wb);
    gridcraft_xlsx::write_xlsx_as(&gc, kind)
        .map_err(|e| IngestError::Parse(format!("gridcraft .{ext} writer: {e}")))
}

/// Whether the workbook was read with a VBA project (macros), which
/// only an `.xlsm` or `.xltm` save keeps. GridTalk never runs it.
pub fn has_macros(wb: &WorkbookSnapshot) -> bool {
    wb.source
        .as_ref()
        .is_some_and(|source| source.vba_project.is_some())
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
        sheets: vec![SheetSnapshot::new(sheet, rows)],
        names: Vec::new(),
        warnings: Vec::new(),
        source: None,
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
    formula_refs_qualified_on(expr, None)
}

/// [`formula_refs_qualified`] for a formula on sheet `home`: a range
/// whose start is unqualified and whose end names `home`
/// (`A1:Sheet1!B2` on Sheet1) is the rectangle it means. Without a
/// home sheet such an end cannot be checked, so the two cells are
/// kept apart.
pub fn formula_refs_qualified_on(expr: &str, home: Option<&str>) -> Vec<QualifiedRef> {
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
                // The end's own qualifier, if any, and where its cell
                // starts.
                let (end_sheet, end_at) = {
                    let mut j = next + 1;
                    let mut end_sheet: Option<String> = None;
                    // Skip Name! or 'Quoted Name'! after the colon.
                    if j < chars.len() && chars[j] == '\'' {
                        j += 1;
                        let mut name = String::new();
                        // Doubled-apostrophe aware, like the
                        // range-start scanner: '' is an escaped
                        // quote inside the name, not its end.
                        while j < chars.len() {
                            if chars[j] == '\'' {
                                if chars.get(j + 1) == Some(&'\'') {
                                    name.push('\'');
                                    j += 2;
                                    continue;
                                }
                                break;
                            }
                            name.push(chars[j]);
                            j += 1;
                        }
                        if j + 1 < chars.len() && chars[j + 1] == '!' {
                            end_sheet = Some(name);
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
                            end_sheet = Some(chars[j..k].iter().collect());
                            j = k + 1;
                        }
                    }
                    (end_sheet, j)
                };
                if let Some((r1, c1, end)) = parse_a1_at(&chars, end_at) {
                    // An end qualified with a different sheet does not
                    // make a rectangle (Excel rejects it): keep the
                    // two cells, each on its own sheet.
                    let same_sheet = match (&end_sheet, &sheet) {
                        (None, _) => true,
                        (Some(e), Some(s)) => sheet_name_eq(e, s),
                        (Some(e), None) => home.is_some_and(|h| sheet_name_eq(e, h)),
                    };
                    if !same_sheet {
                        refs.push(QualifiedRef {
                            sheet: sheet.clone(),
                            row: r0,
                            col: c0,
                        });
                        refs.push(QualifiedRef {
                            sheet: end_sheet,
                            row: r1,
                            col: c1,
                        });
                        i = end;
                        continue;
                    }
                    let (lo_r, hi_r) = (r0.min(r1), r0.max(r1));
                    let (lo_c, hi_c) = (c0.min(c1), c0.max(c1));
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
    build_graph_cached(wb, &mut GraphCache::default(), None)
}

/// Work kept between graph builds of one workbook, so a rebuild after
/// an edit redoes only what the edit can have changed: each sheet's
/// detected regions and each formula cell's precedent areas. Pass the
/// edited ranges to [`build_graph_cached`]; the cache resets itself
/// when the sheets or defined names change (precedents depend on
/// both).
#[derive(Debug, Default, Clone)]
pub struct GraphCache {
    /// The sheet names and defined names the entries were made for.
    context: Option<(Vec<String>, Vec<DefinedName>)>,
    /// Per sheet index: its detected regions.
    regions: Vec<Option<Vec<Region>>>,
    /// Per formula cell (sheet index, row, col): its precedent areas.
    /// Shared with each build's invariant check, so passing them on
    /// copies a pointer, not the areas.
    areas: std::collections::HashMap<(usize, usize, usize), std::sync::Arc<[lineage::Area]>>,
    /// Where the next audit window starts, counted over the entries a
    /// rebuild reuses (see [`CACHE_AUDIT_SAMPLE`]).
    audit_cursor: usize,
}

/// How many reused precedent entries each cached rebuild re-parses
/// from the cell's current text. The window rotates, so over
/// successive rebuilds every entry is checked. A mismatch means the
/// edit tracking missed a change: the cache is dropped and the graph
/// rebuilt from scratch.
pub const CACHE_AUDIT_SAMPLE: usize = 256;

impl GraphCache {
    /// Drop what `dirty` can have changed, or everything when the
    /// context changed or the edits are unknown (`None`).
    fn invalidate(&mut self, wb: &WorkbookSnapshot, dirty: Option<&[RangeRef]>) {
        let context = (
            wb.sheets.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
            wb.names.clone(),
        );
        let dirty = match dirty {
            Some(d) if self.context.as_ref() == Some(&context) => d,
            _ => {
                *self = GraphCache {
                    context: Some(context),
                    regions: vec![None; wb.sheets.len()],
                    ..GraphCache::default()
                };
                return;
            }
        };
        for (slot, s) in self.regions.iter_mut().zip(&wb.sheets) {
            if dirty.iter().any(|d| sheet_name_eq(&d.sheet, &s.name)) {
                *slot = None;
            }
        }
        if !dirty.is_empty() {
            self.areas.retain(|&(si, r, c), _| {
                !dirty
                    .iter()
                    .any(|d| sheet_name_eq(&d.sheet, &wb.sheets[si].name) && d.contains(r, c))
            });
        }
    }
}

/// [`build_graph`], reusing `cache` for everything outside the edited
/// `dirty` ranges (`None`: no edit information, rebuild everything).
/// The graph and its invariant check still cover the whole workbook;
/// only region detection and formula parsing are skipped where the
/// edit cannot have changed them.
pub fn build_graph_cached(
    wb: &WorkbookSnapshot,
    cache: &mut GraphCache,
    dirty: Option<&[RangeRef]>,
) -> (Graph, Vec<String>) {
    cache.invalidate(wb, dirty);
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
        let regions = match cache.regions.get(sheet_idx).cloned().flatten() {
            Some(regions) => regions,
            None => {
                let regions = detect_regions(sheet);
                if let Some(slot) = cache.regions.get_mut(sheet_idx) {
                    *slot = Some(regions.clone());
                }
                regions
            }
        };
        for region in regions {
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
    // Lifted dependency edges: each formula's precedent areas, as
    // Gridcraft resolves them (sheet qualifiers, 3D spans, names),
    // mapped onto the nodes they touch. A formula reading its own
    // column (a running total) keeps that self-edge.
    let lineage = lineage::Lineage::new(wb);
    let area_nodes = lineage::AreaNodes::new(wb, &g);
    let locator = NodeLocator::new(&g);
    let mut seen = std::collections::BTreeSet::new();
    let mut targets = Vec::new();
    // Each formula's areas, kept for the invariant check so formulas
    // are parsed once per build.
    let mut precedents = FormulaAreas::default();
    // Audit window over the reused entries: CACHE_AUDIT_SAMPLE of them,
    // starting where the last rebuild's window ended.
    // The window runs over the entries the cache holds now (at most
    // that many can be reused).
    let window = cache.areas.len().max(1);
    let audit_start = cache.audit_cursor % window;
    let audited =
        |ordinal: usize| (ordinal % window + window - audit_start) % window < CACHE_AUDIT_SAMPLE;
    let mut reused = 0usize;
    let mut stale: Vec<String> = Vec::new();
    for (si, sheet) in wb.sheets.iter().enumerate() {
        let Some(home) = locator.sheet(&sheet.name) else {
            continue;
        };
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    let Some(src) = locator.find(home, r, c) else {
                        continue;
                    };
                    let key = (si, r, c);
                    if let Some(cached) = cache.areas.get(&key) {
                        if audited(reused) && cached[..] != lineage.areas(wb, si, expr)[..] {
                            stale.push(format!(
                                "{}!{}{}",
                                quote_sheet(&sheet.name),
                                col_name(c),
                                r + 1
                            ));
                        }
                        reused += 1;
                    }
                    let areas = cache
                        .areas
                        .entry(key)
                        .or_insert_with(|| lineage.areas(wb, si, expr).into());
                    precedents.push(src, areas.clone());
                    for area in areas.iter() {
                        targets.clear();
                        area_nodes.nodes(area, &mut targets);
                        for &t in &targets {
                            if seen.insert((src, t)) {
                                g.edges.push(Edge {
                                    from: src,
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
    if !stale.is_empty() {
        // The edit tracking missed these changes, so other entries may
        // be stale too: start over rather than trust the cache.
        eprintln!(
            "gridtalk: graph cache held stale precedents for {}; rebuilding from scratch",
            stale.join(", ")
        );
        *cache = GraphCache::default();
        return build_graph_cached(wb, cache, None);
    }
    cache.audit_cursor = (audit_start + CACHE_AUDIT_SAMPLE) % window;
    let violations = check_invariants_with(wb, &g, &precedents, &area_nodes);
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
    let lineage = lineage::Lineage::new(wb);
    let locator = NodeLocator::new(g);
    let mut precedents = FormulaAreas::default();
    for (si, sheet) in wb.sheets.iter().enumerate() {
        let Some(home) = locator.sheet(&sheet.name) else {
            continue;
        };
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    if let Some(src) = locator.find(home, r, c) {
                        precedents.push(src, lineage.areas(wb, si, expr).into());
                    }
                }
            }
        }
    }
    check_invariants_with(wb, g, &precedents, &lineage::AreaNodes::new(wb, g))
}

/// Every formula's precedent areas, keyed by the node of its cell,
/// stored flat.
#[derive(Default)]
struct FormulaAreas {
    /// (source node, its areas), sharing the graph cache's entries.
    formulas: Vec<(usize, std::sync::Arc<[lineage::Area]>)>,
}

impl FormulaAreas {
    fn push(&mut self, src: usize, areas: std::sync::Arc<[lineage::Area]>) {
        self.formulas.push((src, areas));
    }

    fn iter(&self) -> impl Iterator<Item = (usize, &[lineage::Area])> {
        self.formulas.iter().map(|(src, areas)| (*src, &areas[..]))
    }
}

fn check_invariants_with(
    wb: &WorkbookSnapshot,
    g: &Graph,
    precedents: &FormulaAreas,
    area_nodes: &lineage::AreaNodes,
) -> Vec<String> {
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
    let locator = NodeLocator::new(g);
    // Expected edges: the precedent areas mapped onto nodes by an
    // independent rule (`nodes_by_coverage`).
    let mut expected = std::collections::BTreeSet::new();
    let mut targets = Vec::new();
    for (s, areas) in precedents.iter() {
        for area in areas {
            targets.clear();
            area_nodes.nodes_by_coverage(area, &mut targets);
            expected.extend(targets.iter().map(|&t| (s, t)));
        }
    }
    // Independent cross-check with a second, deliberately simple
    // tokeniser, so a systematic miss in formula_refs_qualified
    // cannot hide by re-deriving the same wrong expected set.
    // A region and a column with the same bounds (a one-column
    // region) are equally specific; the lifted edges name the column,
    // so the locator's region answer maps to it.
    let same_bounds_column: std::collections::HashMap<usize, usize> = {
        let mut columns = std::collections::HashMap::new();
        for n in g.nodes.iter().filter(|n| n.kind == NodeKind::Column) {
            if let Some(r) = &n.range {
                columns.insert((r.sheet.to_lowercase(), r.r0, r.c0, r.r1, r.c1), n.id);
            }
        }
        g.nodes
            .iter()
            .filter(|n| n.kind == NodeKind::Region)
            .filter_map(|n| {
                let r = n.range.as_ref()?;
                let key = (r.sheet.to_lowercase(), r.r0, r.c0, r.r1, r.c1);
                columns.get(&key).map(|&c| (n.id, c))
            })
            .collect()
    };
    let canon = |t: usize| same_bounds_column.get(&t).copied().unwrap_or(t);
    let mut independent = std::collections::BTreeSet::new();
    for sheet in &wb.sheets {
        let Some(home) = locator.sheet(&sheet.name) else {
            continue;
        };
        for (r, row) in sheet.cells.iter().enumerate() {
            for (c, cell) in row.iter().enumerate() {
                if let CellValue::Formula { expr, .. } = cell {
                    if let Some(src) = locator.find(home, r, c) {
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
                                            if let Some(t) = locator.find(home, rr, cc).map(canon) {
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
                                                if let Some(t) =
                                                    locator.find(home, row1 - 1, col).map(canon)
                                                {
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
    last_drop_reason: Option<String>,
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
            last_drop_reason: None,
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

    /// Edits dropped because their sheet does not exist, or because
    /// applying them would grow the sheet past Excel's grid or the
    /// [`MAX_SNAPSHOT_CELLS`] cap (cumulative).
    pub fn dropped_edits(&self) -> usize {
        self.dropped
    }

    /// Why `edit` cannot be applied, or `None` when it can. `extents`
    /// holds each sheet's densified `(rows, cols)`, computed once per
    /// flush. Only an edit that grows its sheet can trip the cell cap,
    /// so overwriting a cell inside a workbook that is already over
    /// the cap (a large CSV has no cell cap at load) stays allowed.
    fn rejection(&self, edit: &CellEdit, extents: &[(u64, u64)]) -> Option<String> {
        let Some(idx) = self
            .snapshot
            .sheets
            .iter()
            .position(|s| crate::sheet_name_eq(&s.name, &edit.sheet))
        else {
            return Some(format!("unknown sheet {}", edit.sheet));
        };
        if edit.row >= EXCEL_MAX_ROWS || edit.col >= EXCEL_MAX_COLS {
            return Some(format!(
                "row {} column {} is outside the sheet grid",
                edit.row.saturating_add(1),
                edit.col.saturating_add(1)
            ));
        }
        // Clearing a cell never grows the sheet.
        if edit.value == CellValue::Empty {
            return None;
        }
        let (old_rows, old_cols) = extents[idx];
        let rows = old_rows.max(edit.row as u64 + 1);
        let cols = old_cols.max(edit.col as u64 + 1);
        let old_area = old_rows.saturating_mul(old_cols);
        let area = rows.saturating_mul(cols);
        if area <= old_area {
            return None;
        }
        let total: u64 = extents
            .iter()
            .map(|(r, c)| r.saturating_mul(*c))
            .fold(0u64, u64::saturating_add);
        if area > MAX_SNAPSHOT_CELLS as u64
            || (total - old_area).saturating_add(area) > MAX_SNAPSHOT_CELLS as u64
        {
            return Some(format!(
                "editing {} would grow the workbook past {MAX_SNAPSHOT_CELLS} cells",
                CellRef {
                    sheet: self.snapshot.sheets[idx].name.clone(),
                    row: edit.row,
                    col: edit.col,
                }
                .a1()
            ));
        }
        None
    }

    /// The reason the most recent dropped edit was rejected.
    pub fn last_drop_reason(&self) -> Option<&str> {
        self.last_drop_reason.as_deref()
    }

    /// Apply pending edits immediately, patching only edited cells.
    /// Returns how many edits were applied; edits that cannot be
    /// applied are counted in `dropped_edits` (with the reason in
    /// `last_drop_reason`) instead of vanishing silently.
    pub fn flush(&mut self) -> usize {
        let edits = std::mem::take(&mut self.pending);
        let mut applied = 0usize;
        let mut extents: Vec<(u64, u64)> = self
            .snapshot
            .sheets
            .iter()
            .map(|s| (s.cells.len() as u64, s.max_col() as u64))
            .collect();
        for edit in edits {
            if let Some(reason) = self.rejection(&edit, &extents) {
                self.dropped += 1;
                self.last_drop_reason = Some(reason);
                // A dropped edit changed nothing: unpublish its dirty range.
                let range = RangeRef {
                    sheet: edit.sheet.clone(),
                    r0: edit.row,
                    c0: edit.col,
                    r1: edit.row,
                    c1: edit.col,
                };
                if let Some(pos) = self.dirty.iter().position(|d| *d == range) {
                    self.dirty.remove(pos);
                }
                continue;
            }
            applied += 1;
            if let Some((idx, sheet)) = self
                .snapshot
                .sheets
                .iter_mut()
                .enumerate()
                .find(|(_, s)| crate::sheet_name_eq(&s.name, &edit.sheet))
            {
                if edit.value == CellValue::Empty {
                    // Clearing a cell past the sheet's edge is a no-op:
                    // padding up to it would grow the sheet unchecked.
                    if !sheet.get_ref(edit.row, edit.col).is_empty() {
                        *sheet.cell_mut(edit.row, edit.col) = CellValue::Empty;
                    }
                    continue;
                }
                let (rows, cols) = &mut extents[idx];
                *rows = (*rows).max(edit.row as u64 + 1);
                *cols = (*cols).max(edit.col as u64 + 1);
                *sheet.cell_mut(edit.row, edit.col) = edit.value;
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
pub mod lineage;

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
    fn workbook_reads_are_capped_and_need_a_regular_file() {
        let dir = std::env::temp_dir().join(format!("gt-read-capped-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // A directory with a workbook extension is not a regular file.
        let not_file = dir.join("folder.csv");
        std::fs::create_dir_all(&not_file).unwrap();
        assert!(matches!(
            read_workbook_bytes(&not_file),
            Err(IngestError::Io(e)) if e.kind() == std::io::ErrorKind::InvalidInput
        ));
        // A sparse file one byte over the cap is rejected unread.
        let big = dir.join("big.csv");
        std::fs::File::create(&big)
            .unwrap()
            .set_len(MAX_DELIMITED_BYTES + 1)
            .unwrap();
        assert!(matches!(
            read_workbook_bytes(&big),
            Err(IngestError::Parse(_))
        ));
        // The format is checked before anything is opened.
        assert!(matches!(
            read_workbook_bytes(&dir.join("missing.numbers")),
            Err(IngestError::UnsupportedFormat { .. })
        ));
        // A FIFO reports length 0; it must be refused, not read forever.
        #[cfg(unix)]
        {
            let fifo = dir.join("pipe.csv");
            let made = std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .is_ok_and(|s| s.success());
            if made {
                assert!(matches!(
                    read_workbook_bytes(&fifo),
                    Err(IngestError::Io(_))
                ));
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A package whose zip directory declares more than
    /// `MAX_UNPACKED_BYTES` is refused before Gridcraft unpacks it.
    /// The package is tiny: its directory states a size it never had.
    #[test]
    fn package_declaring_a_huge_unpacked_size_is_refused() {
        use std::io::Write as _;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("xl/worksheets/sheet1.xml", options).unwrap();
            zip.write_all(b"<worksheet/>").unwrap();
            zip.finish().unwrap();
        }
        let honest = buf.into_inner();
        assert!(check_package_unpacked(&honest).is_ok());
        // Patch the central directory entry's uncompressed size
        // (offset 24 of the 0x02014b50 header) to 1 GiB.
        let mut bomb = honest.clone();
        let at = bomb
            .windows(4)
            .position(|w| w == [0x50, 0x4b, 0x01, 0x02])
            .expect("central directory header");
        bomb[at + 24..at + 28].copy_from_slice(&(1u32 << 30).to_le_bytes());
        let err = load_bytes(std::path::Path::new("bomb.xlsx"), &bomb).unwrap_err();
        assert!(err.to_string().contains("unpacks to more than"), "{err}");
        let err = load_bytes(std::path::Path::new("bomb.ods"), &bomb).unwrap_err();
        assert!(err.to_string().contains("unpacks to more than"), "{err}");
        // Bytes that are not a zip are Gridcraft's to report.
        assert!(check_package_unpacked(b"not a zip").is_ok());
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
            load_bytes(std::path::Path::new("x.numbers"), b""),
            Err(IngestError::UnsupportedFormat { .. })
        ));
        // A supported extension with no spreadsheet inside is a parse
        // error that says so.
        assert!(matches!(
            load_bytes(std::path::Path::new("x.xls"), b""),
            Err(IngestError::Parse(m)) if m.contains("not a spreadsheet package")
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

    /// An XLSX written by Gridcraft with what the snapshot cannot
    /// model: a number format, a merge, a constant error value and
    /// the 1904 date system.
    fn featureful_xlsx() -> Vec<u8> {
        use gridcraft_core::Value as GcValue;
        use gridcraft_core::{CellError, CellRef as GcRef, DateSystem, RangeRef as GcRange};
        use gridcraft_model::{Cell, Formula, NumFmt, Style};
        let mut gc = gridcraft_model::Workbook::new();
        let money = gc.styles.intern(Style {
            num_fmt: NumFmt::new("0.00"),
            ..Style::default()
        });
        gc.date_system = DateSystem::D1904;
        let s = gc.sheet_mut(0).unwrap();
        s.set_cell(GcRef::new(0, 0), Cell::styled(GcValue::Number(2.0), money));
        s.set_cell(GcRef::new(0, 1), Cell::value(GcValue::Error(CellError::NA)));
        s.set_cell(GcRef::new(0, 2), Cell::formula(Formula::new("ISERROR(B1)")));
        s.set_cell(GcRef::new(0, 3), Cell::formula(Formula::new("A1*10")));
        s.set_cell(GcRef::new(0, 4), Cell::styled(GcValue::Number(7.0), money));
        s.merges.push(GcRange::parse("F1:G2").unwrap());
        // A table J1:J3 ("Sales", column "Qty") read by a structured
        // reference in L1.
        s.set_value(GcRef::new(0, 9), GcValue::Text("Qty".into()));
        s.set_value(GcRef::new(1, 9), GcValue::Number(2.0));
        s.set_value(GcRef::new(2, 9), GcValue::Number(3.0));
        s.set_cell(
            GcRef::new(0, 11),
            Cell::formula(Formula::new("SUM(Sales[Qty])")),
        );
        let column = |name: &str| gridcraft_model::TableColumn {
            name: name.into(),
            totals: Default::default(),
            totals_label: None,
            formula: None,
            unique_name: None,
            xml: None,
        };
        s.tables.push(gridcraft_model::Table {
            id: 1,
            name: "Sales".into(),
            range: GcRange::parse("J1:J3").unwrap(),
            header_row: true,
            totals_row: false,
            columns: vec![column("Qty")],
            style: String::new(),
            banded_rows: false,
            banded_cols: false,
            first_col: false,
            last_col: false,
            filter_button: false,
            filter: Vec::new(),
            table_type: None,
            connection_id: None,
        });
        gridcraft_xlsx::write_xlsx(&gc).unwrap()
    }

    #[test]
    fn a_gridcraft_source_keeps_what_the_snapshot_cannot_model() {
        use crate::eval::{eval_cell_value_in_workbook, Value};
        use gridcraft_core::Value as GcValue;
        use gridcraft_core::{CellError, CellRef as GcRef, DateSystem, RangeRef as GcRange};
        let mut wb = load_gridcraft_bytes("t", "xlsx", &featureful_xlsx()).unwrap();
        assert!(wb.source.is_some());
        // The constant #N/A reaches the engine as an error, not as
        // the text the snapshot shows.
        assert_eq!(wb.sheets[0].get(0, 1), CellValue::Text("#N/A".into()));
        assert_eq!(
            eval_cell_value_in_workbook(&wb, "Sheet1", 0, 2).unwrap(),
            Value::Bool(true)
        );
        // The table survives, so its structured reference evaluates.
        assert_eq!(
            eval_cell_value_in_workbook(&wb, "Sheet1", 0, 11).unwrap(),
            Value::Number(5.0)
        );
        // Edits overlay the source: A1 changes, E1 is cleared, H3 is
        // new.
        *wb.sheets[0].cell_mut(0, 0) = CellValue::Number(5.0);
        *wb.sheets[0].cell_mut(0, 4) = CellValue::Empty;
        *wb.sheets[0].cell_mut(2, 7) = CellValue::Formula {
            expr: "D1+1".into(),
            cached: None,
        };
        assert_eq!(
            eval_cell_value_in_workbook(&wb, "Sheet1", 2, 7).unwrap(),
            Value::Number(51.0)
        );
        let (back, _) = gridcraft_xlsx::read_xlsx(&to_xlsx_bytes(&wb).unwrap()).unwrap();
        let s = &back.sheets[0];
        let a1 = s.cell(GcRef::new(0, 0)).unwrap();
        assert_eq!(a1.value, GcValue::Number(5.0));
        assert_eq!(back.styles.get(a1.style).num_fmt.as_str(), "0.00");
        assert_eq!(
            s.cell(GcRef::new(0, 1)).unwrap().value,
            GcValue::Error(CellError::NA)
        );
        // A cleared cell keeps its format but loses its value.
        assert_eq!(s.value(GcRef::new(0, 4)), GcValue::Empty);
        assert_eq!(
            back.styles
                .get(s.style_id(GcRef::new(0, 4)))
                .num_fmt
                .as_str(),
            "0.00"
        );
        assert_eq!(
            s.cell(GcRef::new(2, 7))
                .and_then(|c| c.formula.as_ref())
                .map(|f| f.text.as_str()),
            Some("D1+1")
        );
        assert!(s.merges.contains(&GcRange::parse("F1:G2").unwrap()));
        assert_eq!(back.date_system, DateSystem::D1904);
        assert_eq!(back.sheets[0].tables[0].name, "Sales");
    }

    #[test]
    fn editing_a_table_header_renames_the_column_and_its_references() {
        use crate::eval::{eval_cell_value_in_workbook, Value};
        let mut wb = load_gridcraft_bytes("t", "xlsx", &featureful_xlsx()).unwrap();
        *wb.sheets[0].cell_mut(0, 9) = CellValue::Text("Count".into());
        // The formula still reads SUM(Sales[Qty]) in the snapshot; it
        // follows the rename in Gridcraft, so it keeps evaluating.
        assert_eq!(
            eval_cell_value_in_workbook(&wb, "Sheet1", 0, 11).unwrap(),
            Value::Number(5.0)
        );
        let (back, _) = gridcraft_xlsx::read_xlsx(&to_xlsx_bytes(&wb).unwrap()).unwrap();
        let s = &back.sheets[0];
        assert_eq!(s.tables[0].columns[0].name, "Count");
        assert_eq!(s.value(GcRef::new(0, 9)), GcValue::Text("Count".into()));
        assert_eq!(
            s.cell(GcRef::new(0, 11))
                .and_then(|c| c.formula.as_ref())
                .map(|f| f.text.as_str()),
            Some("SUM(Sales[Count])")
        );
        // An emptied header gets a generated name, written into the cell.
        *wb.sheets[0].cell_mut(0, 9) = CellValue::Empty;
        let gc = to_gridcraft_workbook(&wb);
        assert_eq!(gc.sheets[0].tables[0].columns[0].name, "Column1");
        assert_eq!(
            gc.sheets[0].value(GcRef::new(0, 9)),
            GcValue::Text("Column1".into())
        );
    }

    #[test]
    fn rewriting_a_cell_with_the_same_formula_keeps_its_array_range() {
        let mut formula = gridcraft_model::Formula::new("A1:A2*2");
        formula.array = Some(gridcraft_core::RangeRef::parse("B1:B2").unwrap());
        let mut base = gridcraft_model::Cell::formula(formula);
        base.value = GcValue::Number(4.0);
        let same = CellValue::Formula {
            expr: "A1:A2*2".into(),
            cached: None,
        };
        let kept = gridcraft_cell(&same, base.clone());
        assert!(kept.formula.as_ref().unwrap().array.is_some());
        let other = CellValue::Formula {
            expr: "A1*3".into(),
            cached: None,
        };
        let replaced = gridcraft_cell(&other, base);
        assert!(replaced.formula.as_ref().unwrap().array.is_none());
    }

    #[test]
    fn a_snapshot_whose_sheets_changed_is_converted_on_its_own() {
        let mut wb = load_gridcraft_bytes("t", "xlsx", &featureful_xlsx()).unwrap();
        wb.sheets[0].name = "Renamed".into();
        let gc = to_gridcraft_workbook(&wb);
        assert_eq!(gc.sheets[0].name, "Renamed");
        assert_eq!(gc.date_system, gridcraft_core::DateSystem::D1900);
        assert!(gc.sheets[0].merges.is_empty());
    }

    #[test]
    fn snapshot_copies_share_rows_and_an_edit_copies_only_its_row() {
        let mut inc = IncrementalSnapshot::new(sample());
        let held = inc.snapshot().clone();
        let shared = |a: &WorkbookSnapshot, b: &WorkbookSnapshot, r: usize| {
            std::sync::Arc::ptr_eq(&a.sheets[0].cells[r], &b.sheets[0].cells[r])
        };
        assert!((0..held.sheets[0].cells.len()).all(|r| shared(&held, inc.snapshot(), r)));
        inc.record_edit(
            CellEdit {
                sheet: "Sheet1".into(),
                row: 1,
                col: 1,
                value: CellValue::Number(7.0),
            },
            std::time::Instant::now(),
        );
        inc.flush();
        assert!(
            !shared(&held, inc.snapshot(), 1),
            "the edited row was copied"
        );
        assert!(shared(&held, inc.snapshot(), 0) && shared(&held, inc.snapshot(), 2));
        assert_eq!(
            held.sheets[0].get(1, 1),
            CellValue::Number(2.0),
            "the held copy is unchanged"
        );
        assert_eq!(inc.snapshot().sheets[0].get(1, 1), CellValue::Number(7.0));
        // A write past the edge grows the sheet.
        let mut sheet = SheetSnapshot::new("S", vec![]);
        *sheet.cell_mut(2, 3) = CellValue::Bool(true);
        assert_eq!((sheet.cells.len(), sheet.cells[2].len()), (3, 4));
        assert_eq!(sheet.get(2, 3), CellValue::Bool(true));
    }

    #[test]
    fn the_node_locator_agrees_with_the_linear_lookup() {
        let mut wb = sample();
        let mut other = parse_delimited("d", "Café", "A,B\n1,=Sheet1!D2\n\nX,Y\n2,3\n", ',');
        wb.sheets.append(&mut other.sheets);
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        let spec = specificity_index(&g);
        let locator = NodeLocator::new(&g);
        for sheet in ["Sheet1", "sheet1", "CAFÉ", "Nope"] {
            for r in 0..8 {
                for c in 0..6 {
                    assert_eq!(
                        locator.node_for_cell(sheet, r, c),
                        node_for_cell_in(&g, &spec, sheet, r, c),
                        "{sheet} {r} {c}"
                    );
                }
            }
        }
    }

    #[test]
    fn xlsx_round_trip_via_gridcraft_preserves_values_and_formulas() {
        let wb = sample();
        let bytes = to_xlsx_bytes(&wb).unwrap();
        assert_eq!(&bytes[0..2], b"PK"); // zip container
        let back = load_gridcraft_bytes("demo", "xlsx", &bytes).unwrap();
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
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
        // Gridcraft's ODS import is data-only; that must reach the user.
        assert!(wb.warnings.iter().any(|w| w.contains("data-only")));
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
    fn edits_far_down_a_sheet_are_applied_and_oversized_edits_report_why() {
        let edit = |row, col| CellEdit {
            sheet: "Sheet1".into(),
            row,
            col,
            value: CellValue::Number(1.0),
        };
        let now = std::time::Instant::now();
        // Row 20,000 is well inside the grid and the cell cap.
        let mut inc = IncrementalSnapshot::new(sample());
        inc.record_edit(edit(19_999, 0), now);
        assert_eq!(inc.flush(), 1);
        assert_eq!(inc.revision, 1);
        assert_eq!(
            inc.snapshot().sheets[0].get(19_999, 0),
            CellValue::Number(1.0)
        );
        assert_eq!(inc.dropped_edits(), 0);
        // Outside Excel's grid.
        inc.record_edit(edit(EXCEL_MAX_ROWS, 0), now);
        assert_eq!(inc.flush(), 0);
        assert!(inc
            .last_drop_reason()
            .unwrap()
            .contains("outside the sheet grid"));
        // Inside the grid, but the densified sheet would pass the cap.
        inc.record_edit(edit(500_000, 100), now);
        assert_eq!(inc.flush(), 0);
        assert_eq!(inc.dropped_edits(), 2);
        assert!(inc.last_drop_reason().unwrap().contains("Sheet1!CW500001"));
        assert_eq!(inc.revision, 1);
        // Clearing a far cell never grows the sheet, so it is allowed.
        inc.record_edit(
            CellEdit {
                value: CellValue::Empty,
                ..edit(500_000, 100)
            },
            now,
        );
        assert_eq!(inc.flush(), 1);
        assert_eq!(inc.snapshot().sheets[0].cells.len(), 20_000);
        // A sheet already over the cap (e.g. a ragged CSV) still takes
        // edits that do not grow it.
        let mut wide = sample();
        wide.sheets[0]
            .cells
            .push(Row::new(vec![CellValue::Empty; 5_000]));
        wide.sheets[0].cells.extend(std::iter::repeat_n(
            Row::new(vec![CellValue::Number(0.0)]),
            1_000,
        ));
        let mut inc = IncrementalSnapshot::new(wide);
        inc.record_edit(edit(500, 0), now);
        assert_eq!(inc.flush(), 1);
        assert_eq!(inc.dropped_edits(), 0);
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

    /// The `depends_on` edges as (from label, to label).
    fn depends(g: &Graph) -> Vec<(String, String)> {
        let label = |id: usize| g.nodes.iter().find(|n| n.id == id).unwrap().label.clone();
        let mut out: Vec<(String, String)> = g
            .edges
            .iter()
            .filter(|e| e.kind == "depends_on")
            .map(|e| (label(e.from), label(e.to)))
            .collect();
        out.sort();
        out
    }

    #[test]
    fn lineage_follows_3d_refs_names_and_running_totals() {
        let month = |name: &str, n: f64| {
            SheetSnapshot::new(
                name,
                vec![
                    vec![CellValue::Text("Billed".into())],
                    vec![CellValue::Number(n)],
                ],
            )
        };
        let f = |e: &str| CellValue::Formula {
            expr: e.into(),
            cached: None,
        };
        let summary = SheetSnapshot::new(
            "Summary",
            vec![
                vec![
                    CellValue::Text("Q1".into()),
                    CellValue::Text("Tax".into()),
                    CellValue::Text("Amount".into()),
                    CellValue::Text("Running".into()),
                ],
                vec![
                    f("SUM(Jan:Mar!A2)"),
                    f("Quarter*Tax_Rate"),
                    CellValue::Number(5.0),
                    f("C2"),
                ],
                vec![
                    CellValue::Number(0.0),
                    CellValue::Number(0.0),
                    CellValue::Number(7.0),
                    f("D2+C3"),
                ],
            ],
        );
        let rates = SheetSnapshot::new(
            "Rates",
            vec![
                vec![CellValue::Text("Rate".into())],
                vec![CellValue::Number(0.15)],
            ],
        );
        let wb = WorkbookSnapshot {
            name: "w".into(),
            sheets: vec![
                month("Jan", 1.0),
                month("Feb", 2.0),
                month("Mar", 3.0),
                summary,
                rates,
            ],
            names: vec![
                DefinedName {
                    name: "Tax_Rate".into(),
                    scope: None,
                    formula: "Rates!$A$2".into(),
                },
                DefinedName {
                    name: "Quarter".into(),
                    scope: None,
                    formula: "Summary!$A$2".into(),
                },
            ],
            warnings: Vec::new(),
            source: None,
        };
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        let edges = depends(&g);
        let has = |from: &str, to: &str| edges.iter().any(|(a, b)| a == from && b == to);
        // 3D: Summary!A reads Billed on Jan, Feb and Mar, and is not
        // misread as the columns JAN:MAR of its own sheet.
        assert!(has("Q1", "Billed"), "{edges:?}");
        let billed_sheets: std::collections::BTreeSet<_> = g
            .edges
            .iter()
            .filter(|e| e.kind == "depends_on")
            .filter_map(|e| g.nodes.iter().find(|n| n.id == e.to))
            .filter(|n| n.label == "Billed")
            .map(|n| n.range.as_ref().unwrap().sheet.clone())
            .collect();
        assert_eq!(billed_sheets.len(), 3, "{billed_sheets:?}");
        // Names: B2 reads Quarter (Summary!A2, column Q1) and
        // Tax_Rate (Rates!A2).
        assert!(has("Tax", "Q1") && has("Tax", "Rate"), "{edges:?}");
        // A running total reads its own column.
        assert!(
            has("Running", "Running") && has("Running", "Amount"),
            "{edges:?}"
        );
        // The old reading of Jan:Mar!A2 as columns JAN:MAR plus a bare
        // A2 invented a self-edge on Q1.
        assert!(!has("Q1", "Q1"), "{edges:?}");
    }

    #[test]
    fn cyclic_defined_names_resolve_once() {
        let mut wb = parse_delimited("d", "Sheet1", "A,B\n1,=X+Y\n", ',');
        for (name, formula) in [("X", "Y+Y+Sheet1!$A$2"), ("Y", "X+X")] {
            wb.names.push(DefinedName {
                name: name.into(),
                scope: None,
                formula: formula.into(),
            });
        }
        let areas = lineage::Lineage::new(&wb).areas(&wb, 0, "X+Y");
        assert_eq!(areas, vec![(0, (1, 0, 1, 0))]);
    }

    #[test]
    fn a_cached_rebuild_after_edits_matches_a_fresh_build() {
        let mut wb = parse_delimited(
            "d",
            "Sheet1",
            "Item,Qty,Price,Total\nApples,2,3,=B2*C2\nPears,4,5,=B3*C3\n",
            ',',
        );
        wb.sheets.push(SheetSnapshot::new(
            "Sheet2",
            vec![
                vec![CellValue::Text("Ref".into())],
                vec![CellValue::Formula {
                    expr: "Sheet1!D2".into(),
                    cached: None,
                }],
            ],
        ));
        let mut inc = IncrementalSnapshot::new(wb);
        let mut cache = GraphCache::default();
        let (g, v) = build_graph_cached(inc.snapshot(), &mut cache, None);
        assert!(v.is_empty(), "{v:?}");
        assert_eq!((g.nodes.clone(), g.edges.clone()), {
            let (f, _) = build_graph(inc.snapshot());
            (f.nodes, f.edges)
        });
        let f = |e: &str| CellValue::Formula {
            expr: e.into(),
            cached: None,
        };
        let edits: Vec<(&str, usize, usize, CellValue)> = vec![
            // A value inside a region.
            ("Sheet1", 1, 1, CellValue::Number(9.0)),
            // A formula now reading another column.
            ("Sheet1", 2, 3, f("A3&B3")),
            // A header rename (labels change).
            ("Sheet1", 0, 2, CellValue::Text("Cost".into())),
            // A new column grows Sheet1's region: node ids shift, and
            // Sheet2's cross-sheet edge must follow.
            ("Sheet1", 0, 4, CellValue::Text("Tax".into())),
            ("Sheet1", 1, 4, f("D2*0.1")),
            // A running total on Sheet2.
            ("Sheet2", 2, 0, f("A2+1")),
        ];
        for (sheet, row, col, value) in edits {
            inc.record_edit(
                CellEdit {
                    sheet: sheet.into(),
                    row,
                    col,
                    value,
                },
                std::time::Instant::now(),
            );
            inc.flush();
            let dirty = inc.take_dirty_ranges();
            let (cached, v) = build_graph_cached(inc.snapshot(), &mut cache, Some(&dirty));
            assert!(v.is_empty(), "{v:?}");
            let (fresh, _) = build_graph(inc.snapshot());
            assert_eq!(cached.nodes, fresh.nodes, "after {sheet}!{row},{col}");
            assert_eq!(cached.edges, fresh.edges, "after {sheet}!{row},{col}");
            assert_eq!(cached.regions, fresh.regions);
        }
        // A new defined name changes what formulas can read: the cache
        // starts over rather than keeping stale precedents.
        let mut wb = inc.snapshot().clone();
        wb.names.push(DefinedName {
            name: "Rate".into(),
            scope: None,
            formula: "Sheet1!$B$2".into(),
        });
        *wb.sheets[1].cell_mut(1, 0) = f("Rate");
        let (cached, _) = build_graph_cached(&wb, &mut cache, Some(&[]));
        let (fresh, _) = build_graph(&wb);
        assert_eq!(cached.edges, fresh.edges);
    }

    #[test]
    fn the_cache_audit_catches_a_change_the_edit_tracking_missed() {
        // 600 formulas: more than one audit window.
        let mut csv = String::from("Item,Qty,Price,Total\n");
        for r in 2..=601 {
            csv.push_str(&format!("I{r},{r},2,=B{r}*C{r}\n"));
        }
        let mut wb = parse_delimited("d", "Sheet1", &csv, ',');
        let mut cache = GraphCache::default();
        build_graph_cached(&wb, &mut cache, None);
        let edges_to_item = |g: &Graph| {
            let item = g.nodes.iter().find(|n| n.label == "Item").unwrap().id;
            g.edges
                .iter()
                .any(|e| e.kind == "depends_on" && e.to == item)
        };
        // A formula changes behind the cache's back (no dirty range):
        // D500 now reads Item, which no other formula does.
        *wb.sheets[0].cell_mut(499, 3) = CellValue::Formula {
            expr: "A500&B500".into(),
            cached: None,
        };
        assert!(edges_to_item(&build_graph(&wb).0));
        let windows = 600_usize.div_ceil(CACHE_AUDIT_SAMPLE);
        let mut caught = None;
        for i in 0..windows {
            let (g, v) = build_graph_cached(&wb, &mut cache, Some(&[]));
            if edges_to_item(&g) {
                assert!(v.is_empty(), "{v:?}");
                // Caught: the graph is a fresh build's.
                assert_eq!(g.edges, build_graph(&wb).0.edges);
                caught = Some(i);
                break;
            }
        }
        assert!(caught.is_some(), "not caught within {windows} rebuilds");
        // The first window after a fresh build covers the start of the
        // cache, not all of it.
        let mut cache = GraphCache::default();
        let mut wb2 = parse_delimited("d", "Sheet1", &csv, ',');
        build_graph_cached(&wb2, &mut cache, None);
        *wb2.sheets[0].cell_mut(599, 3) = CellValue::Formula {
            expr: "A600&B600".into(),
            cached: None,
        };
        let (g, _) = build_graph_cached(&wb2, &mut cache, Some(&[]));
        assert!(
            !edges_to_item(&g),
            "one window audits {CACHE_AUDIT_SAMPLE} entries, not every one"
        );
    }

    #[test]
    fn overlapping_regions_lift_to_the_most_specific_node() {
        // The inverted U A1:E5's bounding box holds the lone C3, its
        // own region: G1 reads C3 (and B2:D4) through C3's node only.
        // A lone one-column region (region and column alike) read
        // from elsewhere on its sheet.
        let wb = parse_delimited("d", "Sheet1", "A,,=C3\n1,,\n,,9\n", ',');
        let (_, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        for formula in ["=C3", "=SUM(B2:D4)"] {
            let csv =
                format!("H1,H2,H3,H4,H5,,{formula}\n1,,,,2,,\n3,,9,,4,,\n5,,,,6,,\n7,,,,8,,\n");
            let wb = parse_delimited("d", "Sheet1", &csv, ',');
            let (g, violations) = build_graph(&wb);
            assert!(violations.is_empty(), "{formula}: {violations:?}");
            assert!(g.regions.iter().any(|r| r.range.a1() == "Sheet1!C3:C3"));
        }
    }

    #[test]
    fn a_range_end_naming_the_home_sheet_is_a_rectangle() {
        let cells = |home: Option<&str>| -> Vec<(Option<String>, usize, usize)> {
            formula_refs_qualified_on("SUM(A1:Sheet1!B2)", home)
                .into_iter()
                .map(|q| (q.sheet, q.row, q.col))
                .collect()
        };
        // On Sheet1 the end names the home sheet: all of A1:B2.
        assert_eq!(cells(Some("sheet1")).len(), 4);
        // On another sheet, or with no home, the two cells stay apart,
        // the end on the sheet it names.
        let apart = vec![(None, 0, 0), (Some("Sheet1".to_string()), 1, 1)];
        assert_eq!(cells(Some("Sheet2")), apart);
        assert_eq!(cells(None), apart);
    }

    #[test]
    fn a_range_end_on_another_sheet_is_not_a_rectangle_on_the_first() {
        let mut wb = parse_delimited("d", "Sheet1", "A,B,C\n1,2,=SUM(Sheet2!A1:Sheet1!B2)\n", ',');
        wb.sheets.push(SheetSnapshot::new(
            "Sheet2",
            vec![
                vec![CellValue::Text("X".into()), CellValue::Text("Y".into())],
                vec![CellValue::Number(9.0), CellValue::Number(8.0)],
            ],
        ));
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        let edges = depends(&g);
        // Gridcraft rejects the formula; the lexical fallback keeps
        // Sheet2!A1 (X) and Sheet1!B2 (B), never the invented
        // Sheet2!A1:B2 rectangle (which would reach Sheet2's Y).
        assert_eq!(
            edges,
            vec![
                ("C".to_string(), "B".to_string()),
                ("C".to_string(), "X".to_string())
            ]
        );
    }

    #[test]
    fn cross_sheet_formula_lifts_to_other_sheet() {
        let mut wb = parse_delimited("d", "Sheet1", "A,B\n1,=Sheet2!A2\n", ',');
        wb.sheets.push(SheetSnapshot::new(
            "Sheet2",
            vec![
                vec![CellValue::Text("X".into())],
                vec![CellValue::Number(9.0)],
            ],
        ));
        let (g, violations) = build_graph(&wb);
        assert!(violations.is_empty(), "{violations:?}");
        assert!(g.edges.iter().any(|e| e.kind == "depends_on"));
    }

    #[test]
    fn unsupported_format_is_typed_error() {
        let err = load_path(std::path::Path::new("book.numbers")).unwrap_err();
        assert!(matches!(err, IngestError::UnsupportedFormat { .. }));
    }
}
