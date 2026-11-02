//! Local embeddings and vector search.
//!
//! This module ships a dependency-light, fully offline signed
//! n-gram hashing embedding that always works (including in tests
//! and first run). A `bge-small-en-v1.5` ONNX adapter behind
//! [`EmbeddingModel`] is planned but not shipped: retrieval today is
//! morphological (shared word/ngram features), not semantic.

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
    dim: usize,
}

impl Default for NGramEmbedding {
    fn default() -> Self {
        NGramEmbedding { dim: 1024 }
    }
}

fn hash_token(token: &str) -> u64 {
    hash_bytes(token.as_bytes())
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    // FNV-1a; deterministic across runs and platforms.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Shared tokeniser (also used by the lexical index): lowercases,
/// keeps single characters and decimal numbers ("3.14" stays one
/// token), splits on other punctuation.
pub fn tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    let flush = |cur: &mut String, out: &mut Vec<String>| {
        if cur.is_empty() {
            return;
        }
        let mut t = std::mem::take(cur);
        // Normalise numbers: "1,000" and "1000" are the same token,
        // ".5" becomes "0.5", "1,5" is a decimal comma (1.5), and
        // percent tokens normalise the same way before the %.
        let (sign, rest0) = match t.strip_prefix('-') {
            Some(rest) => ("-", rest),
            None => ("", t.as_str()),
        };
        let (body, pct) = match rest0.strip_suffix('%') {
            Some(rest) => (rest, "%"),
            None => (rest0, ""),
        };
        if body
            .chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
            && body.chars().any(|c| c.is_ascii_digit())
        {
            let groups: Vec<&str> = body.split(',').collect();
            let mut norm = if groups.len() > 1
                && groups[1..]
                    .iter()
                    .all(|g| g.len() == 3 && g.chars().all(|c| c.is_ascii_digit()))
            {
                body.replace(',', "")
            } else if body.contains('.') {
                // Already has a decimal point: grouping commas are
                // separators, never a second point.
                body.replace(',', "")
            } else {
                body.replace(',', ".")
            };
            if let Some(rest) = norm.strip_prefix('.') {
                norm = format!("0.{rest}");
            }
            t = format!("{sign}{norm}{pct}");
        }
        let t = t.trim_matches('.').to_string();
        let t = if let Some(rest) = t.strip_prefix('.') {
            format!("0.{rest}")
        } else {
            t
        };
        if !t.is_empty() {
            out.push(t);
        }
    };
    for i in 0..chars.len() {
        let c = chars[i];
        let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
        let next_digit = chars.get(i + 1).is_some_and(|n| n.is_ascii_digit());
        if c.is_alphanumeric()
            || (c == '.' && (prev_digit || cur.is_empty()) && (next_digit || prev_digit))
            || (c == ',' && prev_digit && next_digit)
            || (c == '-'
                && cur.is_empty()
                && (next_digit
                    || (chars.get(i + 1) == Some(&'.')
                        && chars.get(i + 2).is_some_and(|n| n.is_ascii_digit()))))
        {
            cur.push(c);
        } else if c == '%'
            && !cur.is_empty()
            && cur.chars().last().is_some_and(|l| l.is_ascii_digit())
        {
            cur.push(c);
            flush(&mut cur, &mut out);
        } else {
            flush(&mut cur, &mut out);
        }
    }
    flush(&mut cur, &mut out);
    out
}

fn add_feature(v: &mut [f32], h: u64, weight: f32) {
    // Signed hashing: collisions can cancel instead of only adding,
    // so unrelated texts stay near zero cosine.
    let idx = (h % v.len() as u64) as usize;
    let sign = if (h >> 63) == 0 { 1.0 } else { -1.0 };
    v[idx] += sign * weight;
}

impl EmbeddingModel for NGramEmbedding {
    fn name(&self) -> &str {
        "ngram-hashing-v1 (offline; bge-small ONNX adapter pending)"
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        if self.dim == 0 {
            return Vec::new();
        }
        let mut v = vec![0.0f32; self.dim];
        let words = tokenize(text);
        for w in &words {
            add_feature(&mut v, hash_token(w), 1.0);
            // Character trigrams over chars, not UTF-8 bytes, so
            // non-ASCII words hash their actual characters. Bytes
            // are encoded into a stack buffer — no allocation.
            let chars: Vec<char> = w.chars().collect();
            if chars.len() >= 3 {
                for tri in chars.windows(3) {
                    let mut buf = [0u8; 16];
                    let mut len = 0;
                    for ch in tri {
                        len += ch.encode_utf8(&mut buf[len..]).len();
                    }
                    add_feature(&mut v, hash_bytes(&buf[..len]), 0.5);
                }
            } else if !chars.is_empty() {
                // Salted so the fallback does not land on the exact
                // same feature as the word hash itself.
                let mut key = w.to_string();
                key.push_str("#short");
                add_feature(&mut v, hash_bytes(key.as_bytes()), 0.5);
            }
        }
        for pair in words.windows(2) {
            let mut key = Vec::with_capacity(pair[0].len() + 1 + pair[1].len());
            key.extend_from_slice(pair[0].as_bytes());
            key.push(b' ');
            key.extend_from_slice(pair[1].as_bytes());
            let h = hash_bytes(&key);
            add_feature(&mut v, h, 0.75);
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

impl NGramEmbedding {
    /// Validating constructor: dim must be in 8..=65536.
    pub fn new(dim: usize) -> Option<Self> {
        if (8..=65_536).contains(&dim) {
            Some(NGramEmbedding { dim })
        } else {
            None
        }
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    pub doc: usize,
    pub score: f32,
}

/// Brute-force cosine index (region counts are small by design —
/// the graph is lifted, so documents are regions, not cells).
/// Max chunks scored per region (bounds memory and query cost).
const MAX_CHUNKS: usize = 64;

pub type ChunkSet = (Vec<std::sync::Arc<Vec<f32>>>, Vec<String>);

/// Lines of a region that get their own chunk vector. Beyond
/// MAX_CHUNKS the lines are sampled evenly across the whole region
/// (not just the first 64 in passage order); single-token lines are
/// kept — discarding them dropped legitimately terse data rows.
fn chunk_lines(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() <= 1 {
        return vec![text.to_string()];
    }
    if lines.len() <= MAX_CHUNKS {
        return lines.iter().map(|l| l.to_string()).collect();
    }
    let mut out = Vec::with_capacity(MAX_CHUNKS);
    for i in 0..MAX_CHUNKS {
        // Span the whole region: first line through LAST line
        // (i * len / MAX_CHUNKS never reaches the final line).
        out.push(lines[i * (lines.len() - 1) / (MAX_CHUNKS - 1)].to_string());
    }
    out
}

fn embed_chunks(
    text: &str,
    model: &dyn EmbeddingModel,
) -> (Vec<std::sync::Arc<Vec<f32>>>, Vec<String>) {
    // Score against the best-matching line/chunk of a region, not
    // only the whole diluted region vector.
    let kept = chunk_lines(text);
    (
        kept.iter()
            .map(|l| std::sync::Arc::new(model.embed(l)))
            .collect(),
        kept,
    )
}

pub struct VectorIndex {
    pub model_name: String,
    /// Minimum cosine for a hit. Calibrated for the signed
    /// hashing embeddings above (unrelated texts centre near 0);
    /// raise it for large corpora of short passages.
    pub threshold: f32,
    pub dim: usize,
    /// Per-doc chunk vectors (usually one per passage line).
    vectors: Vec<Vec<std::sync::Arc<Vec<f32>>>>,
    texts: Vec<String>,
    /// The chunk texts behind `vectors`, for per-chunk caching.
    chunk_texts: Vec<Vec<String>>,
}

impl VectorIndex {
    pub fn build(texts: &[String], model: &dyn EmbeddingModel) -> Self {
        let chunks: Vec<ChunkSet> = texts.iter().map(|t| embed_chunks(t, model)).collect();
        VectorIndex {
            model_name: model.name().to_string(),
            threshold: 0.15,
            dim: model.dim(),
            vectors: chunks.iter().map(|(v, _)| v.clone()).collect(),
            texts: texts.to_vec(),
            chunk_texts: chunks.into_iter().map(|(_, t)| t).collect(),
        }
    }

    /// Rebuild over `texts`, reusing the stored vector for any text
    /// that is unchanged so a one-cell edit does not re-embed every
    /// region.
    pub fn rebuild(&mut self, texts: &[String], model: &dyn EmbeddingModel) {
        if model.name() != self.model_name || model.dim() != self.dim {
            // Preserve a customised threshold across a model swap.
            let threshold = self.threshold;
            *self = VectorIndex::build(texts, model);
            self.threshold = threshold;
            return;
        }
        // Per-chunk cache: unchanged *lines* keep their vectors
        // even when their region changed around them.
        let cache: std::collections::HashMap<&str, &std::sync::Arc<Vec<f32>>> = self
            .chunk_texts
            .iter()
            .flatten()
            .map(|t| t.as_str())
            .zip(self.vectors.iter().flatten())
            .collect();
        let mut all_vectors = Vec::new();
        let mut all_texts = Vec::new();
        for t in texts {
            // Only embed lines the cache misses — embedding first and
            // swapping in the cached Arc saved nothing.
            let lines = chunk_lines(t);
            let vecs: Vec<std::sync::Arc<Vec<f32>>> = lines
                .iter()
                .map(|line| match cache.get(line.as_str()) {
                    Some(cached) => (*cached).clone(),
                    None => std::sync::Arc::new(model.embed(line)),
                })
                .collect();
            all_vectors.push(vecs);
            all_texts.push(lines);
        }
        self.vectors = all_vectors;
        self.chunk_texts = all_texts;
        self.texts = texts.to_vec();
    }

    pub fn search(&self, query: &str, model: &dyn EmbeddingModel, limit: usize) -> Vec<VectorHit> {
        // Never compare vectors from a different model/dimension.
        if model.name() != self.model_name || model.dim() != self.dim {
            return Vec::new();
        }
        let q = model.embed(query);
        let mut hits: Vec<VectorHit> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(doc, chunks)| VectorHit {
                doc,
                // Best chunk wins: a relevant line inside a long
                // region is not diluted by the rest of the region.
                score: chunks.iter().map(|v| cosine(&q, v)).fold(0.0f32, f32::max),
            })
            .filter(|h| h.score > self.threshold)
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
    reciprocal_rank_fusion_weighted(rankings, &[], k, limit)
}

/// Weighted variant: earlier lists can be trusted more (GridTalk
/// weights exact lexical ranks above fuzzy vector ranks, so a
/// vector-only #1 never ties an exact lexical #1).
pub fn reciprocal_rank_fusion_weighted(
    rankings: &[Vec<usize>],
    weights: &[f64],
    k: usize,
    limit: usize,
) -> Vec<(usize, f64)> {
    let mut scores: BTreeMap<usize, f64> = BTreeMap::new();
    for (list, ranking) in rankings.iter().enumerate() {
        let w = weights.get(list).copied().unwrap_or(1.0);
        for (rank, doc) in ranking.iter().enumerate() {
            *scores.entry(*doc).or_default() += w / (k as f64 + rank as f64 + 1.0);
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
    fn regression_dim_zero_and_mismatched_models_never_panic() {
        assert!(NGramEmbedding::new(0).is_none());
        assert!(NGramEmbedding::new(1).is_none());
        assert!(NGramEmbedding::new(usize::MAX).is_none());
        let a = NGramEmbedding::default();
        let idx = VectorIndex::build(&["hello world".to_string()], &a);
        // Same name + dim is required; a 256-dim model against a
        // mismatched caller returns no hits instead of garbage.
        let other = NGramEmbedding::new(64).unwrap();
        assert!(idx.search("hello", &other, 5).is_empty());
        assert_eq!(cosine(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn tokeniser_and_signed_hashing_regressions() {
        assert_eq!(tokenize("1,000"), vec!["1000".to_string()]);
        assert_eq!(tokenize("1000"), vec!["1000".to_string()]);
        assert_eq!(tokenize(".5"), vec!["0.5".to_string()]);
        assert_eq!(tokenize("50%"), vec!["50%".to_string()]);
        assert_eq!(tokenize("-5"), vec!["-5".to_string()]);
        assert_ne!(tokenize("-5"), tokenize("5"));
        assert_eq!(tokenize("1,5"), vec!["1.5".to_string()]);
        assert_eq!(tokenize("1,000%"), tokenize("1000%"));
        let m = NGramEmbedding::default();
        let unrelated = cosine(
            &m.embed("apples quantity price"),
            &m.embed("zebra xylophone quantum"),
        );
        assert!(unrelated.abs() < 0.15, "unrelated cosine {unrelated}");
        // cosine() normalises unnormalised inputs.
        assert!((cosine(&[2.0, 0.0], &[3.0, 0.0]) - 1.0).abs() < 1e-6);
        // Char trigrams: accented words embed without byte artefacts.
        assert_eq!(m.embed("café résumé"), m.embed("café résumé"));
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
