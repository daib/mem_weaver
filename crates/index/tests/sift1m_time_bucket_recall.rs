//! Recall@k of [`index::TimeBucketIndex`] vs exhaustive L2 on a SIFT1M *prefix*,
//! parameterized by the number of time buckets the corpus is split across.
//!
//! Each base vector at row `i` is assigned `Timestamp(i)`; `bucket_duration` is
//! chosen so the corpus splits evenly into approximately `num_buckets` windows.
//! Search runs with `time_range = None` (all buckets); ground truth is brute
//! force over the full corpus, so a high recall here means the time-bucket
//! merge is finding the same neighbors as a single global HNSW would.
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory with `sift_base.fvecs` and `sift_query.fvecs`. If unset the test returns immediately.
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors to load and index (default `10_000`).
//! - `SIFT1M_RECALL_N_QUERIES` — number of queries to evaluate (default `10`).
//! - `SIFT1M_HNSW_EF` — search `ef` at level 0 (default `100`).
//! - `SIFT1M_TIME_BUCKET_COUNTS` — comma-separated bucket counts to try (default `1,4,16`).

use common::benchmark::{sift_recall_stats, try_load_sift_ctx};
use common::{top_k_quickselect, Timestamp};
use index::{BucketedNodeId, TimeBucketIndex, DEFAULT_ALIGNMENT};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use vector::{read_fvecs_vector_at, VectorId};

const K: usize = 10;
const M: usize = 16;
const M_MAX0: usize = 32;

const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;

const DEFAULT_NUM_BASE_VECTORS: usize = 10_000;
const DEFAULT_NUM_QUERIES: usize = 10;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_BUCKET_COUNTS: &[usize] = &[1, 4, 16];

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn parse_bucket_counts() -> Vec<usize> {
    let Ok(raw) = std::env::var("SIFT1M_TIME_BUCKET_COUNTS") else {
        return DEFAULT_BUCKET_COUNTS.to_vec();
    };
    let parsed: Vec<usize> = raw
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n| n > 0)
        .collect();
    if parsed.is_empty() {
        DEFAULT_BUCKET_COUNTS.to_vec()
    } else {
        parsed
    }
}

#[test]
fn sift1m_time_bucket_recall_vs_bruteforce() {
    let t_load = Instant::now();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };
    eprintln!(
        "sift: load context (mmap base + query, validation) {:.3} ms",
        ms(t_load.elapsed())
    );

    let base_data = ctx.base_data();
    let q_data = ctx.q_data();
    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let n_q = ctx.n_q;
    let ef = ctx.search_ef.max(K);

    let t_corpus = Instant::now();
    let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(n_base);
    for i in 0..n_base {
        corpus.push(read_fvecs_vector_at(base_data, dim, i).expect("uniform fvecs"));
    }
    eprintln!(
        "sift: decode corpus into Vec<Vec<f32>> (n_base={n_base}, dim={dim}) {:.3} ms",
        ms(t_corpus.elapsed())
    );

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    let bucket_counts = parse_bucket_counts();

    eprintln!(
        "sift_time_bucket_recall: dim={dim} n_base={n_base} n_q={n_q} k={K} m={M} m_max0={M_MAX0} ef_search={ef} ef_construction={ef_construction} alignment={DEFAULT_ALIGNMENT} rng_seed={RNG_SEED} bucket_counts={bucket_counts:?}"
    );

    for &num_buckets in &bucket_counts {
        // Choose bucket_duration so the row-indexed timestamps split into ~num_buckets windows.
        // `bucket_duration` must be ≥ 1; clamp when num_buckets > n_base.
        let bucket_duration_secs = (n_base as u64 / num_buckets as u64).max(1);
        let bucket_duration = Duration::from_secs(bucket_duration_secs);
        let label = format!("time_bucket(n={num_buckets})");

        let mut index = TimeBucketIndex::new(
            dim,
            M,
            M_MAX0,
            ef_construction,
            bucket_duration,
            top_k_quickselect,
            StdRng::seed_from_u64(RNG_SEED),
        )
        .expect("valid TimeBucketIndex config");

        // BucketedNodeId is (bucket_seq, NodeId) — neither is the corpus row index, so we
        // build the map at insert time the same way sift1m_hnsw_recall does for NodeId.
        let mut id_to_corpus: HashMap<BucketedNodeId, usize> = HashMap::with_capacity(n_base);

        let t_idx = Instant::now();
        let mut batch_start = Instant::now();
        for (i, v) in corpus.iter().enumerate() {
            let bid = index.insert(v.as_slice(), Timestamp(i as u64));
            id_to_corpus.insert(bid, i);
            if (i + 1) % 10_000 == 0 {
                eprintln!(
                    "{label}: inserted [{}, {}) 10_000 vectors in {:.3} ms (cumulative build {:.3} ms)",
                    i + 1 - 10_000,
                    i + 1,
                    ms(batch_start.elapsed()),
                    ms(t_idx.elapsed())
                );
                batch_start = Instant::now();
            } else if i + 1 == n_base && (n_base % 10_000 != 0) {
                let n = n_base % 10_000;
                eprintln!(
                    "{label}: inserted [{}, {}) {n} vectors in {:.3} ms (cumulative build {:.3} ms)",
                    n_base - n,
                    n_base,
                    ms(batch_start.elapsed()),
                    ms(t_idx.elapsed())
                );
            }
        }
        eprintln!(
            "{label}: index build (insert n={n_base}) {:.3} ms — {} buckets created (bucket_duration={}s)",
            ms(t_idx.elapsed()),
            index.bucket_count(),
            bucket_duration_secs
        );
        assert_eq!(
            index.len(),
            n_base,
            "{label}: TimeBucketIndex should hold every base vector"
        );

        let (_stats, _, _) = sift_recall_stats(&label, &corpus, q_data, dim, n_q, ef, |q| {
            index
                .search(q, K, ef, |_, d| d, None, top_k_quickselect)
                .iter()
                .map(|bid| {
                    let corpus_row = id_to_corpus[bid];
                    VectorId(corpus_row as u64)
                })
                .collect()
        });
        assert!(
            _stats.min >= 0.75,
            "{label}: minimum recall@{K} vs brute force expected >= 0.75 (try SIFT1M_HNSW_EF=512 or lower SIFT1M_RECALL_N_BASE)"
        );
    }
}
