//! Recall@k of [`index::ParallelHnsw`] vs exhaustive L2 on a SIFT1M prefix.
//!
//! Tests both single-threaded insertion (baseline parity check) and
//! multi-threaded batch insertion.
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory with `sift_base.fvecs` and `sift_query.fvecs`
//!   (Texmex layout). If unset, the test returns immediately.
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors to index (default `8192`).
//! - `SIFT1M_RECALL_N_QUERIES` — number of queries to evaluate (default `10`).
//! - `SIFT1M_HNSW_EF` — search `ef` at level 0 (default `256`).
//! - `SIFT1M_PARALLEL_THREADS` — thread count for batch insertion (default `8`).

use common::benchmark::{sift_recall_stats, try_load_sift_ctx};
use common::top_k_quickselect;
use index::ParallelHnsw;
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
const EF_CONSTRUCTION: usize = 40;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_NUM_BASE_VECTORS: usize = 1000_000;
const DEFAULT_NUM_QUERIES: usize = 10_000;
const DEFAULT_THREADS: usize = 4;
const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn load_corpus(base_data: &[u8], dim: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| read_fvecs_vector_at(base_data, dim, i).expect("uniform fvecs"))
        .collect()
}

fn run_case(
    label: &str,
    index: &ParallelHnsw,
    corpus: &[Vec<f32>],
    q_data: &[u8],
    dim: usize,
    n_q: usize,
    ef: usize,
) {
    assert_eq!(
        index.len(),
        corpus.len(),
        "{label}: len mismatch after insert"
    );

    let (stats, _, _) = sift_recall_stats(label, corpus, q_data, dim, n_q, ef, |q| {
        index
            .search(q, K, ef)
            .into_iter()
            .map(|(vid, _)| VectorId(vid))
            .collect()
    });

    eprintln!(
        "{label}: recall@{K} min={:.3} mean={:.3} p95={:.3}",
        stats.min, stats.mean, stats.p95
    );

    assert!(
        stats.min >= 0.70,
        "{label}: recall@{K} min={:.3} < 0.70 (try raising SIFT1M_HNSW_EF or lowering SIFT1M_RECALL_N_BASE)",
        stats.min
    );
}

/// Insert `corpus` into `index` one vector at a time, printing progress every 10 000 vectors.
fn insert_single_thread(label: &str, index: &ParallelHnsw, corpus: &[Vec<f32>], rng: &mut StdRng) {
    let n_base = corpus.len();
    let t_idx = Instant::now();
    let mut batch_start = Instant::now();
    for (i, v) in corpus.iter().enumerate() {
        index.insert(v, i as u64, rng);
        if (i + 1) % 10_000 == 0 {
            eprintln!(
                "{label}: inserted [{}, {}) 10_000 vectors in {:.3} ms (cumulative {:.3} ms)",
                i + 1 - 10_000,
                i + 1,
                ms(batch_start.elapsed()),
                ms(t_idx.elapsed()),
            );
            batch_start = Instant::now();
        } else if i + 1 == n_base && n_base % 10_000 != 0 {
            let n = n_base % 10_000;
            eprintln!(
                "{label}: inserted [{}, {}) {n} vectors in {:.3} ms (cumulative {:.3} ms)",
                n_base - n,
                n_base,
                ms(batch_start.elapsed()),
                ms(t_idx.elapsed()),
            );
        }
    }
    eprintln!("{label}: build total {:.3} ms", ms(t_idx.elapsed()));
}

#[test]
fn sift1m_parallel_hnsw_single_thread() {
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

    eprintln!(
        "parallel_hnsw single-thread: dim={dim} n_base={n_base} n_q={n_q} k={K} \
         m={M} m_max0={M_MAX0} ef_construction={EF_CONSTRUCTION} ef_search={ef}"
    );

    let index = ParallelHnsw::new(dim, M, M_MAX0, EF_CONSTRUCTION, top_k_quickselect).unwrap();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    insert_single_thread("parallel_hnsw/single", &index, &corpus, &mut rng);

    run_case(
        "parallel_hnsw/single",
        &index,
        &corpus,
        ctx.q_data(),
        dim,
        n_q,
        ef,
    );
}

/// Insert `corpus` using two-phase batch insertion, printing progress after each batch.
/// The batch size matches `insert_parallel`'s partitioning: `n_base / num_threads`,
/// so one progress line is emitted per thread-sized chunk.
fn insert_two_phase(
    label: &str,
    index: &ParallelHnsw,
    corpus: &[Vec<f32>],
    vector_ids: &[u64],
    rng: &mut StdRng,
    num_threads: usize,
    chunk_size: usize,
) {
    let n_base = corpus.len();
    let num_threads = num_threads.clamp(1, n_base.max(1));
    let t_idx = Instant::now();
    let mut inserted = 0;

    for (vec_chunk, id_chunk) in corpus.chunks(chunk_size).zip(vector_ids.chunks(chunk_size)) {
        index.insert_batch_two_phase(vec_chunk, id_chunk, num_threads, rng);
        inserted += vec_chunk.len();
        eprintln!(
            "{label}: inserted {inserted}/{n_base} vectors (cumulative {:.3} ms)",
            ms(t_idx.elapsed()),
        );
    }
    eprintln!("{label}: build total {:.3} ms", ms(t_idx.elapsed()));
}

#[test]
fn sift1m_parallel_hnsw_two_phase() {
    let _serial = TEST_MUTEX.lock().unwrap();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };

    let num_threads = std::env::var("SIFT1M_PARALLEL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREADS);

    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let n_q = ctx.n_q;
    let ef = ctx.search_ef.max(K);

    let corpus = load_corpus(ctx.base_data(), dim, n_base);
    let vector_ids: Vec<u64> = (0..n_base as u64).collect();

    eprintln!(
        "parallel_hnsw two-phase: dim={dim} n_base={n_base} n_q={n_q} k={K} \
         m={M} m_max0={M_MAX0} ef_construction={EF_CONSTRUCTION} ef_search={ef} threads={num_threads}"
    );

    let index = ParallelHnsw::new(dim, M, M_MAX0, EF_CONSTRUCTION, top_k_quickselect).unwrap();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    insert_two_phase(
        "parallel_hnsw/two-phase",
        &index,
        &corpus,
        &vector_ids,
        &mut rng,
        num_threads,
        10_000,
    );

    run_case(
        "parallel_hnsw/two-phase",
        &index,
        &corpus,
        ctx.q_data(),
        dim,
        n_q,
        ef,
    );
}

/// Compare throughput: single-thread vs two-phase parallel insertion.
///
/// Two-phase runs phase 1 (neighbor search) in parallel via rayon and phase 2b
/// (edge commit) concurrently with region-based serialization.  This test
/// measures whether the parallel phases yield a meaningful wall-time speedup
/// over sequential insertion on the same hardware.
#[test]
fn sift1m_parallel_hnsw_two_phase_throughput() {
    let _serial = TEST_MUTEX.lock().unwrap();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };

    let num_threads = std::env::var("SIFT1M_PARALLEL_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREADS);

    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let ef = ctx.search_ef.max(K);

    let corpus = load_corpus(ctx.base_data(), dim, n_base);
    let vector_ids: Vec<u64> = (0..n_base as u64).collect();

    eprintln!(
        "two-phase throughput: dim={dim} n_base={n_base} k={K} \
         m={M} m_max0={M_MAX0} ef_construction={EF_CONSTRUCTION} ef_search={ef} threads={num_threads}"
    );

    // Single-threaded baseline.
    let idx_single = ParallelHnsw::new(dim, M, M_MAX0, EF_CONSTRUCTION, top_k_quickselect).unwrap();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let t_single = Instant::now();
    insert_single_thread(
        "two-phase-throughput/single",
        &idx_single,
        &corpus,
        &mut rng,
    );
    let single_ms = ms(t_single.elapsed());
    let single_throughput = n_base as f64 / (single_ms / 1e3);

    run_case(
        "two-phase-throughput/single",
        &idx_single,
        &corpus,
        ctx.q_data(),
        dim,
        ctx.n_q,
        ef,
    );
    drop(idx_single);

    // Two-phase parallel.
    let idx_two = ParallelHnsw::new(dim, M, M_MAX0, EF_CONSTRUCTION, top_k_quickselect).unwrap();
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let t_two = Instant::now();
    insert_two_phase(
        "two-phase-throughput/two-phase",
        &idx_two,
        &corpus,
        &vector_ids,
        &mut rng,
        num_threads,
        10_000,
    );
    let two_ms = ms(t_two.elapsed());
    let two_throughput = n_base as f64 / (two_ms / 1e3);

    eprintln!(
        "two-phase throughput: single={single_throughput:.0} vec/s ({single_ms:.1} ms)  \
         two-phase({num_threads}t)={two_throughput:.0} vec/s ({two_ms:.1} ms)  \
         speedup {:.2}x",
        single_ms / two_ms,
    );

    run_case(
        "two-phase-throughput/two-phase",
        &idx_two,
        &corpus,
        ctx.q_data(),
        dim,
        ctx.n_q,
        ef,
    );
}
