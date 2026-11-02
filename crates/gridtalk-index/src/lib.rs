//! Lexical index over workbook regions. The production target is
//! tantivy + local embeddings (ExcelGRAG eg-index); this crate keeps
//! the same public shape (index regions, hybrid search, cited hits)
//! with a dependency-light inverted index so the pipeline is testable
//! before the heavier engines are wired in.

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
}
