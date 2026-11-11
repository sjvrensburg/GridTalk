//! The Excel formats beyond `.xlsx`: `.xlsm`/`.xltx`/`.xltm`, `.xls`
//! (BIFF8), `.xlsb` (values only), files whose extension does not match
//! their content, and encrypted workbooks. Every file is built here:
//! packages with Gridcraft's writer and the zip crate, BIFF8 and BIFF12
//! records by hand from the public MS-XLS and MS-XLSB layouts.

use gridcraft_core::{CellRef, Value};
use gridcraft_model::{Cell, Formula, Workbook};
use gridcraft_xlsx::{testing, write_xlsx_as, XlsxKind};
use gridtalk_graph::eval::{eval_cell_value_in_workbook, Value as EvalValue};
use gridtalk_graph::{
    build_graph, has_macros, load_bytes, to_package_bytes, values_only_warning, CellValue,
    IngestError, WorkbookSnapshot,
};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

fn load(name: &str, bytes: &[u8]) -> Result<WorkbookSnapshot, IngestError> {
    load_bytes(Path::new(name), bytes)
}

fn parse_error(result: Result<WorkbookSnapshot, IngestError>) -> String {
    match result {
        Err(IngestError::Parse(message)) => message,
        Err(other) => panic!("expected a parse error, got {other}"),
        Ok(_) => panic!("expected a parse error, got a workbook"),
    }
}

/// A one-sheet workbook: A1 = 2, A2 = 3, A3 = A1+A2.
fn small_workbook() -> Workbook {
    let mut wb = Workbook::new();
    let sheet = wb.sheet_mut(0).unwrap();
    sheet.set_value(CellRef::new(0, 0), Value::Number(2.0));
    sheet.set_value(CellRef::new(1, 0), Value::Number(3.0));
    sheet.set_cell(CellRef::new(2, 0), Cell::formula(Formula::new("A1+A2")));
    wb
}

fn package(kind: XlsxKind, macros: bool) -> Vec<u8> {
    let mut wb = small_workbook();
    if macros {
        // Any bytes: Gridcraft keeps the VBA project opaque and never runs it.
        wb.vba_project = Some(Arc::new(b"not really VBA".to_vec()));
    }
    write_xlsx_as(&wb, kind).unwrap()
}

fn assert_formula_sum(wb: &WorkbookSnapshot) {
    let a3 = wb.sheets[0].get(2, 0);
    assert!(
        matches!(&a3, CellValue::Formula { expr, .. } if expr == "A1+A2"),
        "{a3:?}"
    );
    let sheet = wb.sheets[0].name.clone();
    assert_eq!(
        eval_cell_value_in_workbook(wb, &sheet, 2, 0).unwrap(),
        EvalValue::Number(5.0)
    );
    let (_, violations) = build_graph(wb);
    assert!(violations.is_empty(), "{violations:?}");
}

// BIFF8 (.xls): a Workbook stream inside a compound file.

fn biff(id: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = id.to_le_bytes().to_vec();
    out.extend((payload.len() as u16).to_le_bytes());
    out.extend(payload);
    out
}

fn u16s(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// BOF of a BIFF8 substream: 0x0005 globals, 0x0010 worksheet.
fn bof(kind: u16) -> Vec<u8> {
    let mut payload = u16s(&[0x0600, kind, 0x0DBB, 0x07CC]);
    payload.extend([0; 8]);
    biff(0x0809, &payload)
}

/// NUMBER: row, column, XF, value.
fn number(row: u16, col: u16, value: f64) -> Vec<u8> {
    let mut payload = u16s(&[row, col, 0]);
    payload.extend(value.to_le_bytes());
    biff(0x0203, &payload)
}

/// An `.xls` workbook with sheet "Data": A1 = 2, A2 = 3, A3 = A1+A2
/// (cached 5).
fn xls() -> Vec<u8> {
    // PtgRef (value class) with relative row and column, then PtgAdd.
    let ptg_ref = |row: u16, col: u16| [vec![0x44], u16s(&[row, col | 0xC000])].concat();
    let rgce = [ptg_ref(0, 0), ptg_ref(1, 0), vec![0x03]].concat();
    let mut formula = u16s(&[2, 0, 0]);
    formula.extend(5.0f64.to_le_bytes());
    formula.extend([0; 6]); // Flags and chn.
    formula.extend((rgce.len() as u16).to_le_bytes());
    formula.extend(&rgce);
    let sheet = [
        bof(0x0010),
        number(0, 0, 2.0),
        number(1, 0, 3.0),
        biff(0x0006, &formula),
        biff(0x000A, &[]),
    ]
    .concat();
    let xf = biff(0x00E0, &[u16s(&[0, 0, 0]), vec![0; 14]].concat());
    let globals = |sheet_at: u32| {
        let mut boundsheet = sheet_at.to_le_bytes().to_vec();
        boundsheet.extend([0, 0, 4, 0]);
        boundsheet.extend(b"Data");
        [
            bof(0x0005),
            xf.clone(),
            biff(0x0085, &boundsheet),
            biff(0x000A, &[]),
        ]
        .concat()
    };
    let at = globals(0).len() as u32;
    let stream = [globals(at), sheet].concat();
    testing::compound_file(&[("Workbook", &stream)])
}

// BIFF12 (.xlsb): binary records in a zip package.

fn record(id: u16, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    if id < 128 {
        out.push(id as u8);
    } else {
        out.extend([(id as u8 & 0x7f) | 0x80, (id >> 7) as u8]);
    }
    let mut len = payload.len();
    loop {
        let byte = (len & 0x7f) as u8;
        len >>= 7;
        out.push(if len == 0 { byte } else { byte | 0x80 });
        if len == 0 {
            break;
        }
    }
    out.extend(payload);
    out
}

fn wide(text: &str) -> Vec<u8> {
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut out = (units.len() as u32).to_le_bytes().to_vec();
    out.extend(units.iter().flat_map(|u| u.to_le_bytes()));
    out
}

/// A cell record: column, style, value, and for a formula its tokens.
fn xlsb_cell(id: u16, col: u32, value: &[u8], rgce: Option<&[u8]>) -> Vec<u8> {
    let mut payload = col.to_le_bytes().to_vec();
    payload.extend([0; 4]);
    payload.extend(value);
    if let Some(rgce) = rgce {
        payload.extend([0; 2]);
        payload.extend((rgce.len() as u32).to_le_bytes());
        payload.extend(rgce);
        payload.extend(0u32.to_le_bytes());
    }
    record(id, &payload)
}

fn zip(parts: &[(&str, &[u8])]) -> Vec<u8> {
    let mut z = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    for (name, data) in parts {
        z.start_file(*name, zip::write::SimpleFileOptions::default())
            .unwrap();
        z.write_all(data).unwrap();
    }
    z.finish().unwrap().into_inner()
}

/// An `.xlsb` workbook with sheet "Data": A1 = 2, A2 = 3, A3 = A1+A2
/// (saved result 5).
fn xlsb() -> Vec<u8> {
    let mut book = record(131, &[]);
    let mut props = 0u32.to_le_bytes().to_vec();
    props.extend(0u32.to_le_bytes());
    props.extend(wide(""));
    book.extend(record(153, &props));
    book.extend(record(143, &[]));
    let mut sheet_entry = 0u32.to_le_bytes().to_vec();
    sheet_entry.extend(1u32.to_le_bytes());
    sheet_entry.extend(wide("rId1"));
    sheet_entry.extend(wide("Data"));
    book.extend(record(156, &sheet_entry));
    book.extend(record(144, &[]));
    book.extend(record(132, &[]));

    let row = |r: u32| record(0, &[r.to_le_bytes().to_vec(), vec![0; 13]].concat());
    // PtgRef (value class): row, then column with both relative flags.
    let ptg_ref = |r: u32, c: u16| {
        [
            vec![0x44],
            r.to_le_bytes().to_vec(),
            (c | 0xC000).to_le_bytes().to_vec(),
        ]
        .concat()
    };
    let rgce = [ptg_ref(0, 0), ptg_ref(1, 0), vec![0x03]].concat();
    let mut sheet = record(129, &[]);
    sheet.extend(record(145, &[]));
    sheet.extend(row(0));
    sheet.extend(xlsb_cell(5, 0, &2.0f64.to_le_bytes(), None));
    sheet.extend(row(1));
    sheet.extend(xlsb_cell(5, 0, &3.0f64.to_le_bytes(), None));
    sheet.extend(row(2));
    sheet.extend(xlsb_cell(9, 0, &5.0f64.to_le_bytes(), Some(&rgce)));
    sheet.extend(record(146, &[]));
    sheet.extend(record(130, &[]));

    let types = r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/xl/workbook.bin" ContentType="application/vnd.ms-excel.sheet.binary.macroEnabled.main"/><Override PartName="/xl/worksheets/sheet1.bin" ContentType="application/vnd.ms-excel.worksheet"/></Types>"#;
    let root = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="book" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.bin"/></Relationships>"#;
    let rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.bin"/></Relationships>"#;
    zip(&[
        ("[Content_Types].xml", types.as_bytes()),
        ("_rels/.rels", root.as_bytes()),
        ("xl/workbook.bin", &book),
        ("xl/_rels/workbook.bin.rels", rels.as_bytes()),
        ("xl/worksheets/sheet1.bin", &sheet),
    ])
}

#[test]
fn xls_formulas_are_read_as_formulas() {
    let wb = load("legacy.xls", &xls()).unwrap();
    assert_eq!(wb.sheets[0].name, "Data");
    assert_eq!(wb.sheets[0].get(0, 0), CellValue::Number(2.0));
    assert_formula_sum(&wb);
    assert!(!wb.warnings.iter().any(|w| w.contains("values only")));
    // The Gridcraft workbook is kept, so saving as .xlsx works.
    let saved = to_package_bytes(&wb, "xlsx").unwrap();
    assert_formula_sum(&load("saved.xlsx", &saved).unwrap());
}

#[test]
fn xlsb_opens_as_values_only_with_a_warning() {
    let wb = load("binary.xlsb", &xlsb()).unwrap();
    assert_eq!(wb.sheets[0].name, "Data");
    assert_eq!(wb.sheets[0].get(1, 0), CellValue::Number(3.0));
    // The formula is its saved result, a constant.
    assert_eq!(wb.sheets[0].get(2, 0), CellValue::Number(5.0));
    assert_eq!(
        wb.warnings[0],
        values_only_warning("xlsb"),
        "{:?}",
        wb.warnings
    );
    // Gridcraft's own warning follows.
    assert!(
        wb.warnings
            .iter()
            .any(|w| w.contains("XLSB data-only import")),
        "{:?}",
        wb.warnings
    );
    let (_, violations) = build_graph(&wb);
    assert!(violations.is_empty(), "{violations:?}");
}

#[test]
fn macro_enabled_and_template_packages_open_and_save_by_extension() {
    for (name, kind) in [
        ("book.xlsm", XlsxKind::MacroWorkbook),
        ("book.xltx", XlsxKind::Template),
        ("book.xltm", XlsxKind::MacroTemplate),
    ] {
        let wb = load(name, &package(kind, kind.is_macro_enabled())).unwrap();
        assert_formula_sum(&wb);
        assert_eq!(has_macros(&wb), kind.is_macro_enabled(), "{name}");
        assert!(
            !wb.warnings
                .iter()
                .any(|w| w.contains("content and was read")),
            "{name}: {:?}",
            wb.warnings
        );
    }
    // Macros survive an .xlsm save and are left out of an .xlsx one,
    // whose content type then matches its name.
    let wb = load("book.xlsm", &package(XlsxKind::MacroWorkbook, true)).unwrap();
    assert!(
        wb.warnings.iter().any(|w| w.contains("macros")),
        "{:?}",
        wb.warnings
    );
    let as_xlsm = load("again.xlsm", &to_package_bytes(&wb, "xlsm").unwrap()).unwrap();
    assert!(has_macros(&as_xlsm));
    let as_xlsx = load("again.xlsx", &to_package_bytes(&wb, "xlsx").unwrap()).unwrap();
    assert!(!has_macros(&as_xlsx));
    assert_formula_sum(&as_xlsx);
    // The read-only formats are refused by the writer.
    for ext in ["xls", "xlsb", "ods", "csv"] {
        assert!(
            matches!(
                to_package_bytes(&wb, ext),
                Err(IngestError::UnsupportedFormat { .. })
            ),
            "{ext}"
        );
    }
}

#[test]
fn the_reader_follows_the_content_not_the_extension() {
    // An .xls download that is really an XLSX package, and the reverse.
    let disguised = load("export.xls", &package(XlsxKind::Workbook, false)).unwrap();
    assert_formula_sum(&disguised);
    assert!(
        disguised.warnings[0].contains("this .xls file holds XLSX content"),
        "{:?}",
        disguised.warnings
    );
    let legacy = load("renamed.xlsx", &xls()).unwrap();
    assert_formula_sum(&legacy);
    assert!(
        legacy.warnings[0].contains("this .xlsx file holds XLS content"),
        "{:?}",
        legacy.warnings
    );
}

#[test]
fn encrypted_and_non_spreadsheet_files_get_a_clear_error() {
    let encrypted = testing::encrypt_workbook(&package(XlsxKind::Workbook, false), "secret");
    let message = parse_error(load("locked.xlsx", &encrypted));
    assert!(message.contains("password-protected"), "{message}");
    let document = testing::compound_file(&[("WordDocument", b"not a workbook")]);
    let message = parse_error(load("letter.xls", &document));
    assert!(message.contains("not an Excel workbook"), "{message}");
    let message = parse_error(load(
        "page.xls",
        b"<html><table><tr><td>1</td></tr></table></html>",
    ));
    assert!(message.contains("not a spreadsheet package"), "{message}");
}
