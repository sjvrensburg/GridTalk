//! Lexical index over workbook regions. The production target is
//! tantivy + local embeddings (ExcelGRAG eg-index); this crate keeps
//! the same public shape (index regions, hybrid search, cited hits)
//! with a dependency-light inverted index so the pipeline is testable
//! before the heavier engines are wired in.

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
    pub matched_terms: Vec<String>,
}

#[derive(Debug, Default)]
pub struct Index {
    docs: Vec<(RangeRef, String, BTreeMap<String, usize>)>,
    df: BTreeMap<String, usize>,
}

fn tokenize(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(|t| t.to_string())
        .collect()
}

impl Index {
    pub fn build(wb: &WorkbookSnapshot, graph: &Graph) -> Self {
        let mut idx = Index::default();
        for region in &graph.regions {
            if let Some(sheet) = wb.sheets.iter().find(|s| s.name == region.range.sheet) {
                let passage = render_passage(sheet, region);
                let mut tf: BTreeMap<String, usize> = BTreeMap::new();
                for t in tokenize(&passage) {
                    *tf.entry(t).or_default() += 1;
                }
                for t in tf.keys() {
                    *idx.df.entry(t.clone()).or_default() += 1;
                }
                idx.docs.push((region.range.clone(), passage, tf));
            }
        }
        idx
    }

    /// TF-IDF ranked lexical search. Returns hits with the rendered
    /// passage so callers can cite the exact range.
    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let terms = tokenize(query);
        let n = self.docs.len().max(1) as f64;
        let mut hits: Vec<Hit> = self
            .docs
            .iter()
            .filter_map(|(range, passage, tf)| {
                let mut score = 0.0;
                let mut matched = Vec::new();
                for t in &terms {
                    if let Some(&count) = tf.get(t) {
                        let df = *self.df.get(t).unwrap_or(&1) as f64;
                        score += (1.0 + (count as f64).ln()) * (1.0 + (n / (1.0 + df)).ln());
                        matched.push(t.clone());
                    }
                }
                if score > 0.0 {
                    Some(Hit {
                        range: range.clone(),
                        score,
                        passage: passage.clone(),
                        matched_terms: matched,
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
        let terms = tokenize(query);
        let n = self.docs.len().max(1) as f64;
        let mut scored: Vec<(usize, f64)> = self
            .docs
            .iter()
            .enumerate()
            .filter_map(|(i, (_, _, tf))| {
                let mut score = 0.0;
                for t in &terms {
                    if let Some(&count) = tf.get(t) {
                        let df = *self.df.get(t).unwrap_or(&1) as f64;
                        score += (1.0 + (count as f64).ln()) * (1.0 + (n / (1.0 + df)).ln());
                    }
                }
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

    pub fn doc(&self, i: usize) -> Option<(&RangeRef, &str)> {
        self.docs.get(i).map(|(r, p, _)| (r, p.as_str()))
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
            .map(|i| lexical.doc(i).unwrap().1.to_string())
            .collect();
        let vector = VectorIndex::build(&texts, &model);
        HybridIndex {
            lexical,
            vector,
            embedding_model_name: model.name().to_string(),
            model,
        }
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<Hit> {
        let lexical_ranking = self.lexical.ranking(query);
        let vector_ranking: Vec<usize> = self
            .vector
            .search(query, &self.model, self.lexical.doc_count().max(1))
            .into_iter()
            .map(|h| h.doc)
            .collect();
        let fused = reciprocal_rank_fusion(&[lexical_ranking, vector_ranking], 60, limit);
        let terms = tokenize(query);
        fused
            .into_iter()
            .filter_map(|(doc, score)| {
                self.lexical.doc(doc).map(|(range, passage)| Hit {
                    range: range.clone(),
                    score,
                    passage: passage.to_string(),
                    matched_terms: terms.clone(),
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
