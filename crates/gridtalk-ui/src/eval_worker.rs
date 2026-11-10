//! Formula evaluation off the UI thread.
//!
//! The grid asks for the results of the formula cells it shows. Cells
//! with no result for the current revision are sent to a worker
//! thread; until the answer arrives the grid keeps showing the
//! previous result (or "…" if there is none), so an edit never stalls
//! a frame or makes every formula flicker.

use gridtalk_graph::eval::{EvalCache, EvalError, Value};
use gridtalk_graph::{CellValue, WorkbookSnapshot};
use std::collections::HashMap;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

/// `Ok(display text)` or `Err(reason)` for one formula cell.
pub type CellResult = Result<String, String>;

/// Cell key: (sheet index, row, col).
pub type CellKey = (usize, usize, usize);

/// A computed result and the snapshot revision it was computed at.
struct Entry {
    revision: u64,
    value: CellResult,
}

/// What the grid should draw for a formula cell.
pub enum Lookup<'a> {
    /// The result for the current revision.
    Fresh(&'a CellResult),
    /// An older revision's result, shown while the new one computes.
    Stale(&'a CellResult),
    /// Nothing computed yet.
    Missing,
}

struct Job {
    revision: u64,
    sheet_idx: usize,
    coords: Vec<(usize, usize)>,
    rx: Receiver<Vec<CellResult>>,
}

/// Formula results for the open workbook, filled by a worker thread.
#[derive(Default)]
pub struct ValueCache {
    entries: HashMap<CellKey, Entry>,
    job: Option<Job>,
    /// The snapshot shared with workers, cloned once per revision,
    /// with its calc engine: the first job of a revision recalculates
    /// the workbook, later jobs reuse the result.
    snapshot: Option<(u64, Arc<Shared>)>,
}

struct Shared {
    wb: WorkbookSnapshot,
    calc: EvalCache,
}

impl ValueCache {
    /// Forget everything (a different workbook was opened). A job in
    /// flight is abandoned: its results belong to the old workbook.
    pub fn clear(&mut self) {
        *self = ValueCache::default();
    }

    pub fn get(&self, key: CellKey, revision: u64) -> Lookup<'_> {
        match self.entries.get(&key) {
            Some(e) if e.revision == revision => Lookup::Fresh(&e.value),
            Some(e) => Lookup::Stale(&e.value),
            None => Lookup::Missing,
        }
    }

    /// Whether a worker is still computing.
    #[cfg(test)]
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    /// Collect a finished job, if any. Returns true when new results
    /// landed (the caller should repaint).
    pub fn poll(&mut self) -> bool {
        let Some(job) = &self.job else {
            return false;
        };
        let results = match job.rx.try_recv() {
            Ok(results) => results,
            Err(TryRecvError::Empty) => return false,
            // The worker died (a panic in the evaluator): record the
            // failure so the cells are not resubmitted every frame.
            Err(TryRecvError::Disconnected) => {
                vec![Err("formula evaluation failed".to_string()); job.coords.len()]
            }
        };
        let job = self.job.take().expect("job checked above");
        for (&(r, c), value) in job.coords.iter().zip(results) {
            let key = (job.sheet_idx, r, c);
            // Never let an older job overwrite a newer result.
            if self
                .entries
                .get(&key)
                .is_some_and(|e| e.revision > job.revision)
            {
                continue;
            }
            self.entries.insert(
                key,
                Entry {
                    revision: job.revision,
                    value,
                },
            );
        }
        true
    }

    /// Start evaluating `coords` on `sheet_idx` at `revision`, unless
    /// a job is already running (the misses are resubmitted on a
    /// later frame once it finishes).
    pub fn request(
        &mut self,
        snapshot: &WorkbookSnapshot,
        revision: u64,
        sheet_idx: usize,
        coords: Vec<(usize, usize)>,
        ctx: Option<egui::Context>,
    ) {
        if coords.is_empty() || self.job.is_some() {
            return;
        }
        let shared = match &self.snapshot {
            Some((rev, shared)) if *rev == revision => shared.clone(),
            _ => {
                let shared = Arc::new(Shared {
                    wb: snapshot.clone(),
                    calc: EvalCache::new(),
                });
                self.snapshot = Some((revision, shared.clone()));
                shared
            }
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let job_coords = coords.clone();
        std::thread::spawn(move || {
            let results = evaluate(&shared, sheet_idx, &job_coords);
            let _ = tx.send(results);
            // Wake the frame loop so the results are drawn even when
            // the window is otherwise idle.
            if let Some(ctx) = ctx {
                ctx.request_repaint();
            }
        });
        self.job = Some(Job {
            revision,
            sheet_idx,
            coords,
            rx,
        });
    }

    /// Wait for the running job (tests only).
    #[cfg(test)]
    pub fn wait(&mut self) {
        while self.busy() {
            if !self.poll() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }
}

fn evaluate(shared: &Shared, sheet_idx: usize, coords: &[(usize, usize)]) -> Vec<CellResult> {
    let wb = &shared.wb;
    let Some(sheet) = wb.sheets.get(sheet_idx) else {
        return vec![Err("sheet no longer exists".to_string()); coords.len()];
    };
    let results = shared.calc.get(wb).values(&sheet.name, coords);
    coords
        .iter()
        .zip(results)
        .map(|(&(r, c), result)| {
            let cached = match sheet.get(r, c) {
                CellValue::Formula { cached, .. } => cached,
                _ => None,
            };
            formula_display(result, cached.as_deref())
        })
        .collect()
}

/// Display text for a formula cell's result: the computed value
/// (number, text or TRUE/FALSE), else — for formulas the evaluator
/// cannot parse (unsupported syntax) — the value cached in the file
/// (e.g. by Excel), else the evaluation error (shown as `#ERR` with
/// the reason on hover). Genuine errors (type errors such as division
/// by zero, cycles, bad references, depth) are never masked by a
/// possibly stale cached value.
pub fn formula_display(result: Result<Value, EvalError>, cached: Option<&CellValue>) -> CellResult {
    match (result, cached) {
        (Ok(v), _) => Ok(v.display()),
        (Err(EvalError::Parse(_)), Some(c)) if !c.is_empty() => Ok(c.display()),
        (Err(e), _) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::parse_delimited;

    #[test]
    fn formula_display_prefers_result_then_cached_value() {
        let num = |n| Ok(Value::Number(n));
        assert_eq!(formula_display(num(1800.0), None), Ok("1800".to_string()));
        assert_eq!(formula_display(num(2.5), None), Ok("2.5".to_string()));
        assert_eq!(
            formula_display(Ok(Value::Text("Q1".into())), None),
            Ok("Q1".to_string())
        );
        assert_eq!(
            formula_display(Ok(Value::Bool(true)), None),
            Ok("TRUE".to_string())
        );
        // Syntax the evaluator cannot parse falls back to the file's
        // cached value.
        let cached = CellValue::Text("Q1".into());
        assert_eq!(
            formula_display(Err(EvalError::Parse("{1,2}".into())), Some(&cached)),
            Ok("Q1".to_string())
        );
        let err = formula_display(Err(EvalError::Cycle("A1".into())), Some(&CellValue::Empty));
        assert!(err.unwrap_err().contains("circular"));
        // Genuine errors are not masked by a stale cached value.
        let err = formula_display(Err(EvalError::Cycle("A1".into())), Some(&cached));
        assert!(err.unwrap_err().contains("circular"));
        let err = formula_display(
            Err(EvalError::Type("division by zero".into())),
            Some(&cached),
        );
        assert!(err.unwrap_err().contains("division"));
    }

    #[test]
    fn text_formulas_show_their_text() {
        let wb = parse_delimited("d", "Sheet1", "A\tB\nWidget\t=A2&\"!\"\n", '\t');
        let mut cache = ValueCache::default();
        cache.request(&wb, 0, 0, vec![(1, 1)], None);
        cache.wait();
        match cache.get((0, 1, 1), 0) {
            Lookup::Fresh(Ok(v)) => assert_eq!(v, "Widget!"),
            _ => panic!("expected a fresh text result"),
        }
    }

    #[test]
    fn results_arrive_from_the_worker_and_go_stale_on_new_revisions() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n2,=A2*3\n", ',');
        let mut cache = ValueCache::default();
        assert!(matches!(cache.get((0, 1, 1), 0), Lookup::Missing));
        cache.request(&wb, 0, 0, vec![(1, 1)], None);
        assert!(cache.busy());
        // A second request while busy is dropped, not queued.
        cache.request(&wb, 0, 0, vec![(1, 0)], None);
        cache.wait();
        match cache.get((0, 1, 1), 0) {
            Lookup::Fresh(Ok(v)) => assert_eq!(v, "6"),
            _ => panic!("expected a fresh result"),
        }
        // The next revision still shows the old value until recomputed.
        assert!(matches!(cache.get((0, 1, 1), 1), Lookup::Stale(Ok(_))));
        cache.clear();
        assert!(matches!(cache.get((0, 1, 1), 0), Lookup::Missing));
    }

    #[test]
    fn an_older_job_never_overwrites_a_newer_result() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n2,=A2*3\n", ',');
        let mut cache = ValueCache::default();
        cache.request(&wb, 5, 0, vec![(1, 1)], None);
        cache.wait();
        cache.request(&wb, 3, 0, vec![(1, 1)], None);
        cache.wait();
        assert!(matches!(cache.get((0, 1, 1), 5), Lookup::Fresh(_)));
    }
}
