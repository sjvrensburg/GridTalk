//! Local embeddings and vector search.
//!
//! The production embedding model is `bge-small-en-v1.5` run locally
//! (ExcelGRAG used fastembed/ONNX). That model is an *adapter*
//! behind [`EmbeddingModel`]: this module ships a dependency-light,
//! fully offline n-gram embedding that always works (including in
//! tests and first run before any model download), plus cosine
//! vector search over region passages. Swapping in the ONNX model
//! changes only the `embed` implementation, not the hybrid index.

use std::collections::BTreeMap;

/// Anything that can turn text into a fixed-dimension vector.
/// Implementations must be deterministic for a given input.
pub trait EmbeddingModel: Send + Sync {
    fn name(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// Offline n-gram embedding: hashed word + character-trigram
/// features, L2-normalised. Captures morphological similarity
/// (e.g. "provision" / "provisions") with zero downloads and zero
/// runtime dependencies — data never leaves the machine.
pub struct NGramEmbedding {
    pub dim: usize,
}

impl Default for NGramEmbedding {
    fn default() -> Self {
        NGramEmbedding { dim: 256 }
    }
}

fn hash_token(token: &str) -> u64 {
    // FNV-1a; deterministic across runs and platforms.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in token.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl EmbeddingModel for NGramEmbedding {
    fn name(&self) -> &str {
        "ngram-hashing-v1 (offline; bge-small ONNX adapter pending)"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .collect();
        for w in &words {
            // Whole word feature.
            let h = hash_token(w);
            v[(h % self.dim as u64) as usize] += 1.0;
            // Character trigrams for morphology.
            let padded = format!("##{w}##");
            let chars: Vec<char> = padded.chars().collect();
            for tri in chars.windows(3) {
                let s: String = tri.iter().collect();
                let h = hash_token(&s);
                v[(h % self.dim as u64) as usize] += 0.5;
            }
        }
        // Word bigrams capture short phrases.
        for pair in words.windows(2) {
            let h = hash_token(&format!("{} {}", pair[0], pair[1]));
            v[(h % self.dim as u64) as usize] += 0.75;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    pub doc: usize,
    pub score: f32,
}

/// Brute-force cosine index (region counts are small by design —
/// the graph is lifted, so documents are regions, not cells).
pub struct VectorIndex {
    pub model_name: String,
    vectors: Vec<Vec<f32>>,
}

impl VectorIndex {
    pub fn build(texts: &[String], model: &dyn EmbeddingModel) -> Self {
        VectorIndex {
            model_name: model.name().to_string(),
            vectors: texts.iter().map(|t| model.embed(t)).collect(),
        }
    }

    pub fn search(&self, query: &str, model: &dyn EmbeddingModel, limit: usize) -> Vec<VectorHit> {
        let q = model.embed(query);
        let mut hits: Vec<VectorHit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(doc, v)| VectorHit {
                doc,
                score: cosine(&q, v),
            })
            .filter(|h| h.score > 0.05)
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

/// Reciprocal Rank Fusion: combines ranked lists without needing
/// comparable score scales (lexical TF-IDF vs cosine).
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<usize>],
    k: usize,
    limit: usize,
) -> Vec<(usize, f64)> {
    let mut scores: BTreeMap<usize, f64> = BTreeMap::new();
    for ranking in rankings {
        for (rank, doc) in ranking.iter().enumerate() {
            *scores.entry(*doc).or_default() += 1.0 / (k as f64 + rank as f64 + 1.0);
        }
    }
    let mut fused: Vec<(usize, f64)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_are_normalised_and_morphology_aware() {
        let m = NGramEmbedding::default();
        let a = m.embed("how is the provision calculated");
        let b = m.embed("provisions calculation");
        let c = m.embed("banana smoothie recipe");
        assert!((a.iter().map(|x| x * x).sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(
            cosine(&a, &b) > cosine(&a, &c),
            "morphologically related text must score higher: {} vs {}",
            cosine(&a, &b),
            cosine(&a, &c)
        );
        assert_eq!(m.embed("same text"), m.embed("same text"));
    }

    #[test]
    fn rrf_rewards_docs_ranked_by_both_lists() {
        // Doc 2 is ranked by both lists (2nd + 1st); it must beat
        // docs ranked by only one list.
        let fused = reciprocal_rank_fusion(&[vec![0, 2], vec![2, 1]], 60, 3);
        assert_eq!(fused[0].0, 2);
        assert_eq!(fused.len(), 3);
    }
}
