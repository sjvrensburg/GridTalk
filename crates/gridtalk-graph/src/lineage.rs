//! Formula lineage: which cells each formula reads, as Gridcraft's
//! parser and dependency analysis see them, mapped onto the lifted
//! graph's region and column nodes.
//!
//! Gridcraft resolves sheet qualifiers, 3D spans (`Jan:Mar!A2` reads
//! A2 on every sheet from Jan to Mar), whole rows and columns, and
//! ranges whose ends carry their own sheet. It only flags defined
//! names as dynamic, so names are resolved here, through their own
//! formulas. A formula Gridcraft cannot parse falls back to GridTalk's
//! lexical extractor, so it still contributes the edges it did before.

use crate::{
    formula_refs_qualified, gridcraft_names, sheet_name_eq, Graph, NodeKind, WorkbookSnapshot,
};
use gridcraft_core::RangeRef as GcRange;
use gridcraft_formula::Expr;

/// One area a formula reads: a snapshot sheet index and an inclusive
/// 0-based rectangle (r0, c0, r1, c1).
pub type Area = (usize, Bounds);

/// How deep defined names may refer to other names.
const MAX_NAME_DEPTH: usize = 8;

/// Resolves formula precedents for one snapshot.
pub struct Lineage {
    /// The snapshot's sheets (names only) and defined names: all that
    /// precedent resolution reads.
    wb: gridcraft_model::Workbook,
}

impl Lineage {
    pub fn new(snapshot: &WorkbookSnapshot) -> Self {
        let mut wb = gridcraft_model::Workbook::new();
        wb.sheets = snapshot
            .sheets
            .iter()
            .map(|s| std::sync::Arc::new(gridcraft_model::Sheet::new(s.name.clone())))
            .collect();
        wb.names = gridcraft_names(snapshot);
        Lineage { wb }
    }

    /// The areas a formula on sheet `sheet` reads. `expr` has no
    /// leading `=`.
    pub fn areas(&self, snapshot: &WorkbookSnapshot, sheet: usize, expr: &str) -> Vec<Area> {
        let mut out = Vec::new();
        match gridcraft_formula::parse(expr.strip_prefix('=').unwrap_or(expr)) {
            Ok(e) => self.collect(sheet, &e, 0, &mut out),
            Err(_) => {
                // Unparseable: keep the lexical extractor's cells.
                for q in formula_refs_qualified(expr) {
                    let target = match &q.sheet {
                        None => Some(sheet),
                        Some(name) => snapshot
                            .sheets
                            .iter()
                            .position(|s| sheet_name_eq(&s.name, name)),
                    };
                    if let Some(t) = target {
                        out.push((t, (q.row, q.col, q.row, q.col)));
                    }
                }
            }
        }
        out
    }

    fn collect(&self, sheet: usize, e: &Expr, depth: usize, out: &mut Vec<Area>) {
        let (areas, _) = gridcraft_calc::eval::precedents(&self.wb, sheet, e);
        out.extend(areas.into_iter().map(|a| (a.sheet, bounds(a.range))));
        if depth >= MAX_NAME_DEPTH {
            return;
        }
        let mut names: Vec<&str> = Vec::new();
        e.walk(&mut |x| {
            if let Expr::Name(n) = x {
                names.push(n);
            }
        });
        for name in names {
            if let Some(def) = self.name(sheet, name) {
                if let Ok(target) = gridcraft_formula::parse(def.strip_prefix('=').unwrap_or(def)) {
                    self.collect(sheet, &target, depth + 1, out);
                }
            }
        }
    }

    /// A defined name's formula as seen from `sheet`: a name scoped
    /// to that sheet wins over a workbook-scoped one.
    fn name(&self, sheet: usize, name: &str) -> Option<&str> {
        let matches = |n: &&gridcraft_model::DefinedName| n.name.eq_ignore_ascii_case(name);
        self.wb
            .names
            .iter()
            .filter(matches)
            .find(|n| n.scope == Some(sheet))
            .or_else(|| {
                self.wb
                    .names
                    .iter()
                    .filter(matches)
                    .find(|n| n.scope.is_none())
            })
            .map(|n| n.formula.as_str())
    }
}

fn bounds(r: GcRange) -> Bounds {
    (
        r.start.row as usize,
        r.start.col as usize,
        r.end.row as usize,
        r.end.col as usize,
    )
}

fn intersects(a: Bounds, b: Bounds) -> bool {
    a.0 <= b.2 && b.0 <= a.2 && a.1 <= b.3 && b.1 <= a.3
}

fn contains(outer: Bounds, inner: Bounds) -> bool {
    outer.0 <= inner.0 && outer.1 <= inner.1 && outer.2 >= inner.2 && outer.3 >= inner.3
}

type Bounds = (usize, usize, usize, usize);

/// A region node with the column nodes inside it.
struct RegionNodes {
    id: usize,
    bounds: Bounds,
    columns: Vec<(usize, Bounds)>,
}

/// The region and column nodes of each snapshot sheet, for mapping
/// areas onto the graph.
pub struct AreaNodes {
    /// Per snapshot sheet: its regions, each with its columns.
    sheets: Vec<Vec<RegionNodes>>,
    /// Per snapshot sheet: every region and column node (id, is a
    /// column, bounds), for the invariant check's independent rule.
    flat: Vec<Vec<(usize, bool, Bounds)>>,
}

impl AreaNodes {
    pub fn new(snapshot: &WorkbookSnapshot, g: &Graph) -> Self {
        let mut sheets: Vec<Vec<RegionNodes>> =
            (0..snapshot.sheets.len()).map(|_| Vec::new()).collect();
        let mut flat: Vec<Vec<(usize, bool, Bounds)>> = vec![Vec::new(); snapshot.sheets.len()];
        let mut columns = Vec::new();
        for n in &g.nodes {
            if !matches!(n.kind, NodeKind::Column | NodeKind::Region) {
                continue;
            }
            let Some(r) = &n.range else {
                continue;
            };
            let Some(i) = snapshot
                .sheets
                .iter()
                .position(|s| sheet_name_eq(&s.name, &r.sheet))
            else {
                continue;
            };
            let b = (r.r0, r.c0, r.r1, r.c1);
            flat[i].push((n.id, n.kind == NodeKind::Column, b));
            match n.kind {
                NodeKind::Region => sheets[i].push(RegionNodes {
                    id: n.id,
                    bounds: b,
                    columns: Vec::new(),
                }),
                _ => columns.push((i, n.id, b)),
            }
        }
        for (i, id, b) in columns {
            match sheets[i].iter_mut().find(|reg| contains(reg.bounds, b)) {
                Some(reg) => reg.columns.push((id, b)),
                // A column outside every region stands on its own.
                None => sheets[i].push(RegionNodes {
                    id,
                    bounds: b,
                    columns: Vec::new(),
                }),
            }
        }
        AreaNodes { sheets, flat }
    }

    /// The nodes an area reads: every column node it touches, and a
    /// region it touches through none of its columns. A region's
    /// columns partition it, so a column is the most specific node of
    /// the cells it holds.
    pub fn nodes(&self, area: &Area, out: &mut Vec<usize>) {
        let Some(regions) = self.sheets.get(area.0) else {
            return;
        };
        let rect = area.1;
        for reg in regions {
            if !intersects(reg.bounds, rect) {
                continue;
            }
            let before = out.len();
            out.extend(
                reg.columns
                    .iter()
                    .filter(|(_, b)| intersects(*b, rect))
                    .map(|(id, _)| *id),
            );
            if out.len() == before {
                out.push(reg.id);
            }
        }
    }

    /// The same set by a different rule, for the invariant check: a
    /// node is read when the area touches it and some column of that
    /// overlap is not covered, across the overlap's rows, by a more
    /// specific node (smaller, or a column the size of its region).
    pub fn nodes_by_coverage(&self, area: &Area, out: &mut Vec<usize>) {
        let Some(nodes) = self.flat.get(area.0) else {
            return;
        };
        let rect = area.1;
        let size = |b: Bounds| (b.2 - b.0 + 1) as u128 * (b.3 - b.1 + 1) as u128;
        for &(id, column, b) in nodes {
            if !intersects(b, rect) {
                continue;
            }
            let overlap = (
                b.0.max(rect.0),
                b.1.max(rect.1),
                b.2.min(rect.2),
                b.3.min(rect.3),
            );
            let more_specific = |other_column: bool, ob: Bounds| {
                size(ob) < size(b) || (size(ob) == size(b) && other_column && !column)
            };
            let smaller: Vec<Bounds> = nodes
                .iter()
                .filter(|&&(other, oc, ob)| {
                    other != id && more_specific(oc, ob) && intersects(ob, overlap)
                })
                .map(|&(_, _, ob)| ob)
                .collect();
            let uncovered = (overlap.1..=overlap.3).any(|col| {
                !smaller
                    .iter()
                    .any(|ob| ob.1 <= col && col <= ob.3 && ob.0 <= overlap.0 && ob.2 >= overlap.2)
            });
            if uncovered {
                out.push(id);
            }
        }
    }
}
