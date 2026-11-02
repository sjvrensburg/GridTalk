//! Lexical index over workbook regions. This crate keeps the
//! hybrid-search public shape (index regions, hybrid search, cited
//! hits) with a dependency-light linear-scan index: documents are
//! lifted regions (tens to hundreds), not cells, so a scan is the
//! intentional trade-off until region counts justify posting lists.
//! A tantivy-backed index remains a possible future swap.

pub mod embed;

use embed::{reciprocal_rank_fusion, EmbeddingModel, NGramEmbedding, VectorIndex};
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

#[derive(Debug, Default)]
pub struct Index {
    docs: Vec<(RangeRef, String, BTreeMap<String, usize>, String)>,
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
        let sheets: BTreeMap<&str, &gridtalk_graph::SheetSnapshot> =
            wb.sheets.iter().map(|s| (s.name.as_str(), s)).collect();
        for region in &graph.regions {
            if let Some(sheet) = sheets.get(region.range.sheet.as_str()) {
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
                idx.docs.push((region.range.clone(), passage, tf, body));
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
            .filter_map(|(range, passage, tf, _)| {
                let (score, matched) = self.score_doc(tf, &terms, n);
                let _ = &matched;
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
            .filter_map(|(i, (_, _, tf, _))| {
                let (score, _) = self.score_doc(tf, &terms, n);
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
        terms: &[String],
        n: f64,
    ) -> (f64, Vec<String>) {
        let mut score = 0.0;
        let mut matched = Vec::new();
        for t in terms {
            if let Some(&count) = tf.get(t) {
                let df = *self.df.get(t).unwrap_or(&1) as f64;
                score += (1.0 + (count as f64).ln()) * (1.0 + (n / (1.0 + df)).ln());
                matched.push(t.clone());
            }
        }
        let doc_len: usize = tf.values().sum();
        if doc_len > 0 {
            score /= 1.0 + (doc_len as f64).ln() * 0.1;
        }
        if score > 0.0 {
            (score, matched)
        } else {
            (0.0, Vec::new())
        }
    }

    pub fn doc(&self, i: usize) -> Option<(&RangeRef, &str)> {
        self.docs.get(i).map(|(r, p, _, _)| (r, p.as_str()))
    }

    /// The indexed body text (passage without the Range header).
    pub fn doc_body(&self, i: usize) -> Option<&str> {
        self.docs.get(i).map(|(_, _, _, b)| b.as_str())
    }

    /// Query terms present in doc `i`, from the stored tf map.
    pub fn matched_terms(&self, i: usize, terms: &[String]) -> Vec<String> {
        match self.docs.get(i) {
            Some((_, _, tf, _)) => terms
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
        let fused = reciprocal_rank_fusion(&[lexical_ranking, vector_ranking], 60, limit);
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
