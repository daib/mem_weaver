//! Recall@k of HNSW indexes vs exhaustive L2 on a SIFT1M prefix.
//!
//! Tests sequential insertion (naive + arena) and two-phase parallel batch insertion.
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory with `sift_base.fvecs` and `sift_query.fvecs`
//!   (Texmex layout). If unset, tests return immediately.
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors to index.
//! - `SIFT1M_RECALL_N_QUERIES` — number of queries to evaluate.
//! - `SIFT1M_HNSW_EF` — search `ef` at level 0.
//! - `SIFT1M_PARALLEL_THREADS` — thread count for batch insertion (default `4`).

use common::benchmark::{try_load_sift_ctx, RecallStats};
use common::distance::euclidean_distance_sq;
use common::eval::{recall_at_k, validate_recall_score};
use index::{HnswArena, HnswIndex, HnswNaive, DEFAULT_ALIGNMENT};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use vector::{read_fvecs_vector_at, VectorId};

// Serializes all tests in this file so they don't race for CPU and skew timings.
static TEST_MUTEX: Mutex<()> = Mutex::new(());

const K: usize = 10;
const M: usize = 16;
const M_MAX0: usize = 32;
const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_EF_CONSTRUCTION: usize = 100;
const DEFAULT_NUM_BASE_VECTORS: usize = 10_000;
const DEFAULT_NUM_QUERIES: usize = 10;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn load_corpus(base_data: &[u8], dim: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| read_fvecs_vector_at(base_data, dim, i).expect("uniform fvecs"))
        .collect()
}

fn brute_force_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, euclidean_distance_sq(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
    scored.truncate(k);
    scored
}

fn format_ranked(ranked: &[(u64, f32)]) -> String {
    ranked
        .iter()
        .map(|(id, dist)| format!("{id}({dist:.4})"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn compute_recall_stats(recalls: &mut [f32]) -> RecallStats {
    if recalls.is_empty() {
        return RecallStats {
            min: 0.0,
            mean: 0.0,
            p95: 0.0,
        };
    }
    recalls.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min = recalls[0];
    let mean = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let idx = (((recalls.len() as f32) * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(recalls.len() - 1);
    RecallStats {
        min,
        mean,
        p95: recalls[idx],
    }
}

fn run_case(
    label: &str,
    index: &dyn HnswIndex,
    corpus: &[Vec<f32>],
    q_data: &[u8],
    dim: usize,
    n_q: usize,
    ef: usize,
    min_recall: f32,
) {
    assert_eq!(
        index.len(),
        corpus.len(),
        "{label}: len mismatch after insert"
    );

    let mut recalls = Vec::with_capacity(n_q);
    for qi in 0..n_q {
        // if qi != 8500 && qi != 9049 {
        //     continue;
        // }
        let q = read_fvecs_vector_at(q_data, dim, qi).expect("query fvecs");
        let expected = brute_force_topk(&q, corpus, K);
        let output = index.search(&q, K, ef);

        let gt_ids: Vec<VectorId> = expected.iter().map(|(id, _)| VectorId(*id)).collect();
        let retrieved: Vec<VectorId> = output.iter().map(|(id, _)| VectorId(*id)).collect();
        let r = recall_at_k(&retrieved, &gt_ids).expect("valid recall@k");
        validate_recall_score(r).expect("in-range score");
        recalls.push(r);

        eprintln!(
            "{label} query {qi}: recall@{K}={r:.4}\n  expected: {}\n  output:   {}",
            format_ranked(&expected),
            format_ranked(&output),
        );
    }
    let stats = compute_recall_stats(&mut recalls);

    eprintln!(
        "{label}: recall@{K} min={:.3} mean={:.3} p95={:.3}",
        stats.min, stats.mean, stats.p95
    );

    if stats.min < min_recall {
        eprintln!(
            "{label}: recall@{K} min={:.3} < {min_recall:.2} \
             (try raising SIFT1M_HNSW_EF or lowering SIFT1M_RECALL_N_BASE)",
            stats.min
        );
    }
}

fn insert_single_thread(
    label: &str,
    index: &mut dyn HnswIndex,
    corpus: &[Vec<f32>],
    chunk_size: usize,
) {
    let n_base = corpus.len();
    let t_idx = Instant::now();
    let mut inserted = 0;
    for chunk in corpus.chunks(chunk_size) {
        for v in chunk {
            index.insert(v, inserted as u64);
            inserted += 1;
        }
        eprintln!(
            "{label}: inserted {inserted}/{n_base} vectors (cumulative {:.3} ms)",
            ms(t_idx.elapsed()),
        );
    }
    eprintln!("{label}: build total {:.3} ms", ms(t_idx.elapsed()));
}

fn insert_parallel(
    label: &str,
    index: &mut dyn HnswIndex,
    corpus: &[Vec<f32>],
    vector_ids: &[u64],
    num_threads: usize,
    chunk_size: usize,
) {
    let n_base = corpus.len();
    let t_idx = Instant::now();
    let mut inserted = 0;

    for (vec_chunk, id_chunk) in corpus.chunks(chunk_size).zip(vector_ids.chunks(chunk_size)) {
        index.insert_batch_parallel(vec_chunk, id_chunk, num_threads);
        inserted += vec_chunk.len();
        eprintln!(
            "{label}: inserted {inserted}/{n_base} vectors (cumulative {:.3} ms)",
            ms(t_idx.elapsed()),
        );
    }
    eprintln!("{label}: build total {:.3} ms", ms(t_idx.elapsed()));
}

#[test]
fn sift1m_hnsw_recall_vs_bruteforce() {
    let _serial = TEST_MUTEX.lock().unwrap();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };

    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let n_q = ctx.n_q;
    let ef = ctx.search_ef.max(K);
    let corpus = load_corpus(ctx.base_data(), dim, n_base);

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    eprintln!(
        "sift_hnsw_recall: dim={dim} n_base={n_base} n_q={n_q} k={K} m={M} m_max0={M_MAX0} \
         ef_search={ef} ef_construction={ef_construction} alignment={DEFAULT_ALIGNMENT} rng_seed={RNG_SEED}"
    );

    let mut cases: Vec<(&str, Box<dyn HnswIndex>, usize)> = vec![
        (
            "naive",
            Box::new(HnswNaive::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                StdRng::seed_from_u64(RNG_SEED),
            )),
            0, // default single threaded implementation
        ),
        (
            "arena",
            Box::new(HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            )),
            0, // default single threaded implementation
        ),
        (
            "arena/single",
            Box::new(HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            )),
            1, // multi-threaded implementation but only one thread is used
        ),
        (
            "arena/parallel-4",
            Box::new(HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            )),
            4,
        ),
        (
            "arena/parallel-6",
            Box::new(HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            )),
            6,
        ),
    ];

    for (label, index, num_threads) in &mut cases {
        if *num_threads == 0 {
            insert_single_thread(label, index.as_mut(), &corpus, 10_000);
        } else {
            let vector_ids: Vec<u64> = (0..n_base as u64).collect();

            insert_parallel(
                label,
                index.as_mut(),
                &corpus,
                &vector_ids,
                *num_threads,
                10_000,
            );
        }
        run_case(
            label,
            index.as_ref(),
            &corpus,
            ctx.q_data(),
            dim,
            n_q,
            ef,
            0.75,
        );
        index.reset();
    }
}
