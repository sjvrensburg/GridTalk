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

impl Index {
    pub fn build(wb: &WorkbookSnapshot, graph: &Graph) -> Self {
        let mut idx = Index::default();
        // Regions arrive in workbook sheet order, contiguous per
        // sheet. Assign them to sheets by IDENTITY (region counts
        // re-derived per sheet), so duplicate sheet names render
        // from their own sheet instead of collapsing by name.
        let region_counts: Vec<usize> = wb
            .sheets
            .iter()
            .map(|sh| gridtalk_graph::detect_regions(sh).len())
            .collect();
        // Prefix sums: region idx -> sheet in O(log sheets).
        let mut prefix: Vec<usize> = Vec::with_capacity(region_counts.len() + 1);
        prefix.push(0);
        for count in &region_counts {
            prefix.push(prefix.last().unwrap() + count);
        }
        let sheet_of_region = |idx: usize| -> Option<&gridtalk_graph::SheetSnapshot> {
            let sheet_idx = prefix.partition_point(|end| *end <= idx).saturating_sub(1);
            wb.sheets.get(sheet_idx)
        };
        for (region_idx, region) in graph.regions.iter().enumerate() {
            // A stale/mismatched graph must not render one sheet's
            // cells under another sheet's citation range.
            let assigned = sheet_of_region(region_idx)
                .filter(|sheet| sheet.name.eq_ignore_ascii_case(&region.range.sheet));
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
                    Some(Hit {
                        range: range.clone(),
                        score,
                        passage: passage.clone(),
                        matched_terms: matched,
                        vector_only: false,
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
                    Hit {
                        range: range.clone(),
                        score,
                        passage: passage.to_string(),
                        vector_only: lexically_matched.is_empty(),
                        matched_terms: lexically_matched,
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
