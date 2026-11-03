//! Lexical index over workbook regions. This crate keeps the
//! hybrid-search public shape (index regions, hybrid search, cited
//! hits) with a dependency-light linear-scan index: documents are
//! lifted regions (tens to hundreds), not cells, so a scan is the
//! intentional trade-off until region counts justify posting lists.
//! A tantivy-backed index remains a possible future swap.

pub mod embed;

use embed::{reciprocal_rank_fusion_weighted, EmbeddingModel, NGramEmbedding, VectorIndex};
use gridtalk_graph::{render_passage, Graph, RangeRef, WorkbookSnapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    pub range: RangeRef,
    pub score: f64,
    pub passage: String,
    /// Query terms that matched lexically. Empty for hits that came
    /// only from the vector (morphological) ranker.
    pub matched_terms: Vec<String>,
    /// True when this hit had no lexical match and was rescued by
    /// the embedding ranker alone. Morphological, not semantic.
    #[serde(default)]
    pub vector_only: bool,
    /// True when `passage` is an excerpt of a region too large to
    /// return whole (see [`MAX_PASSAGE_CHARS`]); fetch the rest of
    /// `range` cell by cell.
    #[serde(default)]
    pub truncated: bool,
}

/// Budget for a hit's passage. A region is indexed whole, but a hit
/// returns at most this many characters (about 1K tokens) so a
/// 2,000-row table cannot flood an agent's or a local model's
/// context.
pub const MAX_PASSAGE_CHARS: usize = 4_000;

/// Fit a rendered passage (`Range …` line, header row, data rows)
/// into `max_chars`. A passage that fits is returned unchanged.
/// Otherwise the excerpt keeps the range and header lines, then the
/// data rows that contain a query term (all rows, in order, when
/// none match), each prefixed with its 1-based sheet row so it stays
/// citable, and ends with a note naming how much was left out.
/// Returns the passage and whether it was truncated.
pub fn excerpt(
    passage: &str,
    range: &RangeRef,
    terms: &[String],
    max_chars: usize,
) -> (String, bool) {
    if passage.len() <= max_chars {
        return (passage.to_string(), false);
    }
    let mut lines = passage.lines();
    let mut out = String::new();
    for line in lines.by_ref().take(2) {
        out.push_str(line);
        out.push('\n');
    }
    let rows: Vec<(usize, &str)> = lines
        .enumerate()
        .map(|(i, line)| (range.r0 + 2 + i, line))
        .collect();
    let matches = |line: &str| tokenize(line).iter().any(|t| terms.contains(t));
    let matching: Vec<&(usize, &str)> = rows.iter().filter(|(_, l)| matches(l)).collect();
    let (pool, how): (Vec<&(usize, &str)>, &str) = if matching.is_empty() {
        (rows.iter().collect(), "first rows")
    } else {
        (matching, "rows matching the query")
    };
    // Leave room for the closing note.
    let budget = max_chars.saturating_sub(200);
    let mut shown = 0;
    for (row, line) in &pool {
        let entry = format!("row {row}: {line}\n");
        if out.len() + entry.len() > budget {
            break;
        }
        out.push_str(&entry);
        shown += 1;
    }
    out.push_str(&format!(
        "[truncated: {shown} of {} data rows shown ({how}); read more with get_cells on {}]\n",
        rows.len(),
        range.a1()
    ));
    (out, true)
}

pub type DocEntry = (RangeRef, String, BTreeMap<String, usize>, String, usize);

#[derive(Debug, Default)]
pub struct Index {
    docs: Vec<DocEntry>,
    df: BTreeMap<String, usize>,
    /// Regions skipped because their sheet was absent (diagnostic).
    pub dropped_regions: usize,
}

fn tokenize(s: &str) -> Vec<String> {
    embed::tokenize(s)
}

fn dedup_terms(terms: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    terms
        .into_iter()
        .filter(|t| seen.insert(t.clone()))
        .collect()
}

/// The sheet each graph region renders from, or `None` when the
/// graph does not match the workbook (stale or mismatched graph),
/// so one sheet's cells never render under another sheet's
/// citation range.
///
/// Graphs from `build_graph` record each region's sheet index in
/// `region_sheets`; those are used directly, checked against the
/// sheet's name and extent, without re-scanning cells. Hand-built
/// graphs fall back to re-detecting regions per sheet and matching
/// them by sheet-local ordinal.
fn region_sheet_assignments<'a>(
    wb: &'a WorkbookSnapshot,
    graph: &Graph,
) -> Vec<Option<&'a gridtalk_graph::SheetSnapshot>> {
    if graph.region_sheets.len() == graph.regions.len() {
        return graph
            .regions
            .iter()
            .zip(&graph.region_sheets)
            .map(|(region, &sheet_idx)| {
                wb.sheets.get(sheet_idx).filter(|sheet| {
                    let r = &region.range;
                    gridtalk_graph::sheet_name_eq(&sheet.name, &r.sheet)
                        && r.r0 <= r.r1
                        && r.c0 <= r.c1
                        && r.r1 < sheet.cells.len()
                        && r.c1 < sheet.max_col()
                })
            })
            .collect();
    }
    // Regions arrive in workbook sheet order, contiguous per sheet.
    // Assign them to sheets by IDENTITY (region counts re-derived
    // per sheet), so duplicate sheet names render from their own
    // sheet instead of collapsing by name.
    let derived_regions: Vec<Vec<gridtalk_graph::Region>> = wb
        .sheets
        .iter()
        .map(gridtalk_graph::detect_regions)
        .collect();
    // Prefix sums: region idx -> sheet in O(log sheets).
    let mut prefix: Vec<usize> = Vec::with_capacity(derived_regions.len() + 1);
    prefix.push(0);
    for regions in &derived_regions {
        prefix.push(prefix.last().unwrap() + regions.len());
    }
    graph
        .regions
        .iter()
        .enumerate()
        .map(|(region_idx, region)| {
            // The graph region must equal the re-derived region at
            // its sheet-local ordinal, or a stale graph could
            // cross-render duplicate sheets.
            let sheet_idx = prefix
                .partition_point(|end| *end <= region_idx)
                .saturating_sub(1);
            let local_ordinal = region_idx - prefix[sheet_idx];
            let ordinal_ok = derived_regions
                .get(sheet_idx)
                .and_then(|rs| rs.get(local_ordinal))
                .is_some_and(|derived| {
                    let d = &derived.range;
                    let r = &region.range;
                    (d.r0, d.c0, d.r1, d.c1) == (r.r0, r.c0, r.r1, r.c1)
                });
            if !ordinal_ok {
                return None;
            }
            wb.sheets
                .get(sheet_idx)
                .filter(|sheet| gridtalk_graph::sheet_name_eq(&sheet.name, &region.range.sheet))
        })
        .collect()
}

impl Index {
    pub fn build(wb: &WorkbookSnapshot, graph: &Graph) -> Self {
        let mut idx = Index::default();
        let assignments = region_sheet_assignments(wb, graph);
        for (region, assigned) in graph.regions.iter().zip(assignments) {
            if let Some(sheet) = assigned {
                let passage = render_passage(sheet, region);
                // Index the body only: the "Range Sheet1!A1:B2" header
                // is citation boilerplate, not content.
                let body: String = passage.lines().skip(1).collect::<Vec<_>>().join("\n");
                let mut tf: BTreeMap<String, usize> = BTreeMap::new();
                for t in tokenize(&body) {
                    *tf.entry(t).or_default() += 1;
                }
                for t in tf.keys() {
                    *idx.df.entry(t.clone()).or_default() += 1;
                }
                let doc_len: usize = tf.values().sum();
                idx.docs
                    .push((region.range.clone(), passage, tf, body, doc_len));
            } else {
                idx.dropped_regions += 1;
            }
        }
        idx
    }

    /// TF-IDF ranked lexical search. Returns hits with the rendered
    /// passage so callers can cite the exact range.
    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let terms = dedup_terms(tokenize(query));
        let n = self.docs.len().max(1) as f64;
        let mut hits: Vec<Hit> = self
            .docs
            .iter()
            .filter_map(|(range, passage, tf, _, doc_len)| {
                let score = self.score_doc(tf, *doc_len, &terms, n);
                let matched: Vec<String> = terms
                    .iter()
                    .filter(|t| tf.contains_key(*t))
                    .cloned()
                    .collect();
                if score > 0.0 {
                    let (passage, truncated) = excerpt(passage, range, &matched, MAX_PASSAGE_CHARS);
                    Some(Hit {
                        range: range.clone(),
                        score,
                        passage,
                        matched_terms: matched,
                        vector_only: false,
                        truncated,
                    })
                } else {
                    None
                }
            })
            .collect();
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        hits
    }

    /// Doc indices ranked by lexical score (for hybrid fusion).
    pub fn ranking(&self, query: &str) -> Vec<usize> {
        let terms = dedup_terms(tokenize(query));
        let n = self.docs.len().max(1) as f64;
        let mut scored: Vec<(usize, f64)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, tf, _, doc_len))| {
                let score = self.score_doc(tf, *doc_len, &terms, n);
                if score > 0.0 {
                    Some((i, score))
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Shared TF-IDF scoring (with length normalisation) used by
    /// both `search` and `ranking` so they cannot drift apart.
    fn score_doc(
        &self,
        tf: &BTreeMap<String, usize>,
        doc_len: usize,
        terms: &[String],
        n: f64,
    ) -> f64 {
        let mut score = 0.0;
        for t in terms {
            if let Some(&count) = tf.get(t) {
                let df = *self.df.get(t).unwrap_or(&1) as f64;
                score += (1.0 + (count as f64).ln()) * (1.0 + (n / (1.0 + df)).ln());
            }
        }
        if doc_len > 0 {
            // Stronger than the old 0.1 factor: long regions must
            // not win purely by accumulating more matched terms.
            score /= 1.0 + (doc_len as f64).ln() * 0.35;
        }
        score
    }

    pub fn doc(&self, i: usize) -> Option<(&RangeRef, &str)> {
        self.docs.get(i).map(|(r, p, _, _, _)| (r, p.as_str()))
    }

    /// The indexed body text (passage without the Range header).
    pub fn doc_body(&self, i: usize) -> Option<&str> {
        self.docs.get(i).map(|(_, _, _, b, _)| b.as_str())
    }

    /// Query terms present in doc `i`, from the stored tf map.
    pub fn matched_terms(&self, i: usize, terms: &[String]) -> Vec<String> {
        match self.docs.get(i) {
            Some((_, _, tf, _, _)) => terms
                .iter()
                .filter(|t| tf.contains_key(*t))
                .cloned()
                .collect(),
            None => Vec::new(),
        }
    }

    pub fn doc_count(&self) -> usize {
        self.docs.len()
    }
}

/// Hybrid search: lexical TF-IDF + local embedding cosine, fused
/// with Reciprocal Rank Fusion. This is the retrieval entry point
/// the chat pipeline and the investigate harness use.
pub struct HybridIndex {
    lexical: Index,
    vector: VectorIndex,
    model: NGramEmbedding,
    pub embedding_model_name: String,
}

impl HybridIndex {
    /// Regions dropped while building (stale/mismatched graph).
    pub fn dropped_regions(&self) -> usize {
        self.lexical.dropped_regions
    }
    pub fn build(wb: &WorkbookSnapshot, graph: &Graph) -> Self {
        let lexical = Index::build(wb, graph);
        let model = NGramEmbedding::default();
        let texts: Vec<String> = (0..lexical.doc_count())
            .map(|i| lexical.doc_body(i).unwrap().to_string())
            .collect();
        let vector = VectorIndex::build(&texts, &model);
        HybridIndex {
            lexical,
            vector,
            embedding_model_name: model.name().to_string(),
            model,
        }
    }

    /// Incremental refresh after an edit: lexical docs are rebuilt
    /// (cheap token counts), but unchanged region bodies reuse their
    /// stored embedding vectors.
    pub fn update(&mut self, wb: &WorkbookSnapshot, graph: &Graph) {
        self.lexical = Index::build(wb, graph);
        let texts: Vec<String> = (0..self.lexical.doc_count())
            .map(|i| self.lexical.doc_body(i).unwrap().to_string())
            .collect();
        self.vector.rebuild(&texts, &self.model);
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let lexical_ranking = self.lexical.ranking(query);
        let vector_ranking: Vec<usize> = self
            .vector
            .search(query, &self.model, self.lexical.doc_count().max(1))
            .into_iter()
            .map(|h| h.doc)
            .collect();
        // Relevance floor: lexical ranking only contains docs with a
        // positive TF-IDF score and vector ranking only docs above
        // the cosine threshold, so a fused hit always cleared one.
        let fused = reciprocal_rank_fusion_weighted(
            &[lexical_ranking, vector_ranking],
            &[1.0, 0.5],
            60,
            limit,
        );
        let terms = dedup_terms(tokenize(query));
        fused
            .into_iter()
            .filter_map(|(doc, score)| {
                self.lexical.doc(doc).map(|(range, passage)| {
                    let lexically_matched = self.lexical.matched_terms(doc, &terms);
                    let (passage, truncated) =
                        excerpt(passage, range, &lexically_matched, MAX_PASSAGE_CHARS);
                    Hit {
                        range: range.clone(),
                        score,
                        passage,
                        vector_only: lexically_matched.is_empty(),
                        matched_terms: lexically_matched,
                        truncated,
                    }
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridtalk_graph::{build_graph, parse_delimited};

    /// A 2,000-row ledger; row 1234's territory is unique.
    fn ledger() -> WorkbookSnapshot {
        let mut csv = String::from("Account,Territory,Balance\n");
        for i in 0..2000 {
            let territory = if i == 1232 { "Ironwood" } else { "Northgate" };
            csv.push_str(&format!("RB-{i},{territory},{}.5\n", 1000 + i));
        }
        parse_delimited("ledger", "Debtors", &csv, ',')
    }

    #[test]
    fn large_regions_return_a_bounded_excerpt_of_matching_rows() {
        let wb = ledger();
        let (graph, _) = build_graph(&wb);
        for hit in [
            Index::build(&wb, &graph).search("ironwood", 3),
            HybridIndex::build(&wb, &graph).search("ironwood", 3),
        ]
        .into_iter()
        .flatten()
        {
            assert!(hit.truncated);
            assert!(
                hit.passage.len() <= MAX_PASSAGE_CHARS,
                "{}",
                hit.passage.len()
            );
            // Header kept; the one matching row is shown with its
            // sheet row (header is row 1, RB-1232 is on row 1234).
            assert!(hit.passage.contains("Account | Territory | Balance"));
            assert!(hit
                .passage
                .contains("row 1234: RB-1232 | Ironwood | 2232.5"));
            assert!(!hit.passage.contains("Northgate"));
            assert!(hit
                .passage
                .contains("[truncated: 1 of 2000 data rows shown"));
        }
    }

    #[test]
    fn excerpt_falls_back_to_leading_rows_and_keeps_small_passages_whole() {
        let wb = ledger();
        let (graph, _) = build_graph(&wb);
        let range = &graph.regions[0].range;
        let full = render_passage(&wb.sheets[0], &graph.regions[0]);
        let (cut, truncated) = excerpt(&full, range, &["nomatch".into()], MAX_PASSAGE_CHARS);
        assert!(truncated && cut.len() <= MAX_PASSAGE_CHARS);
        assert!(cut.contains("row 2: RB-0 | Northgate | 1000.5"));
        assert!(cut.contains("(first rows)"));
        let small = "Range S!A1:A2\nA\n1\n";
        assert_eq!(
            excerpt(small, range, &[], MAX_PASSAGE_CHARS),
            (small.into(), false)
        );
    }

    #[test]
    fn recorded_region_sheets_match_redetection_and_reject_mismatches() {
        let mut wb = parse_delimited("demo", "Sheet1", "Item,Qty\nApples,2\n", ',');
        let mut second = parse_delimited("demo", "Sheet1", "Name,Cost\nRent,9\n", ',');
        wb.sheets.append(&mut second.sheets);
        let (g, _) = build_graph(&wb);
        assert_eq!(g.region_sheets, vec![0, 1]);
        // Duplicate sheet names render from their own sheet.
        let fast = Index::build(&wb, &g);
        assert_eq!(fast.dropped_regions, 0);
        assert_eq!(fast.search("rent", 5).len(), 1);
        // The re-detection fallback (hand-built graphs) agrees.
        let mut hand = g.clone();
        hand.region_sheets.clear();
        let slow = Index::build(&wb, &hand);
        assert_eq!(slow.dropped_regions, 0);
        assert_eq!(slow.doc_count(), fast.doc_count());
        // A graph from another workbook never renders here: a
        // renamed sheet or a smaller sheet drops the region.
        let mut renamed = wb.clone();
        renamed.sheets[1].name = "Other".into();
        assert_eq!(Index::build(&renamed, &g).dropped_regions, 1);
        let mut shrunk = wb.clone();
        shrunk.sheets[1].cells.truncate(1);
        assert_eq!(Index::build(&shrunk, &g).dropped_regions, 1);
    }

    #[test]
    fn search_finds_cited_region() {
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price\nApples,2,3\nProvision,9,9\n",
            ',',
        );
        let (g, v) = build_graph(&wb);
        assert!(v.is_empty());
        let idx = Index::build(&wb, &g);
        let hits = idx.search("provision", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].range.a1(), "Sheet1!A1:C3");
        assert!(hits[0].passage.contains("Provision"));
        assert!(idx.search("nonexistentterm", 5).is_empty());
    }

    #[test]
    fn regression_single_char_dedup_and_vector_only_flag() {
        let wb = parse_delimited("d", "Sheet1", "A,B\n5,Provision\n", ',');
        let (g, _) = build_graph(&wb);
        let idx = Index::build(&wb, &g);
        // Single-character query term is kept and matches.
        assert_eq!(idx.search("5", 5).len(), 1);
        // Duplicate query terms do not change the score.
        let one = idx.search("provision", 5);
        let two = idx.search("provision provision", 5);
        assert_eq!(one.len(), two.len());
        assert!((one[0].score - two[0].score).abs() < 1e-9);
        // Hybrid: morphological rescue is flagged, not claimed lexical.
        let hybrid = HybridIndex::build(&wb, &g);
        let hits = hybrid.search("provisions", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].vector_only);
        assert!(hits[0].matched_terms.is_empty());
    }

    #[test]
    fn hybrid_unrelated_query_returns_nothing() {
        let wb = parse_delimited("d", "Sheet1", "Item,Qty\nApples,2\n", ',');
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        assert!(hybrid
            .search("zebra xylophone quantum banana", 5)
            .is_empty());
        // "range" (the passage header word) must not match every doc.
        assert!(hybrid.search("range", 5).is_empty());
    }

    #[test]
    fn hybrid_vector_rescue_survives_long_regions() {
        // A morphological variant inside a long region must still
        // hit via per-line chunk scoring (whole-region dilution
        // used to sink it below the threshold).
        let mut text = String::from("Item,Qty,Price\n");
        for i in 0..30 {
            text.push_str(&format!("Filler{i},1,1\n"));
        }
        text.push_str("Provision,9,9\n");
        let wb = parse_delimited("demo", "Sheet1", &text, ',');
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        let hits = hybrid.search("provisions", 5);
        assert_eq!(hits.len(), 1, "chunk scoring must rescue the variant");
        assert!(hits[0].vector_only);
    }

    #[test]
    fn hybrid_search_fuses_lexical_and_vector() {
        let wb = parse_delimited(
            "demo",
            "Sheet1",
            "Item,Qty,Price\nApples,2,3\nProvision,9,9\n",
            ',',
        );
        let (g, _) = build_graph(&wb);
        let hybrid = HybridIndex::build(&wb, &g);
        assert!(hybrid.embedding_model_name.contains("offline"));
        // Exact term: both rankers must agree.
        let hits = hybrid.search("provision", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].passage.contains("Provision"));
        // Morphological variant with no exact token match still hits
        // via n-gram embeddings.
        let hits = hybrid.search("provisions", 5);
        assert_eq!(hits.len(), 1, "vector ranker must rescue the variant");
    }
}
