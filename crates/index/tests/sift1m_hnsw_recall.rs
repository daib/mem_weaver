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

mod helpers;

use common::benchmark::{compute_recall_stats, load_or_compute_ground_truth, try_load_sift_ctx};
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
const DEFAULT_PARALLEL_THREADS: usize = 6;

const DEFAULT_QPS_NUM_QUERIES: usize = 10_000;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn load_corpus(base_data: &[u8], dim: usize, n: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| read_fvecs_vector_at(base_data, dim, i).expect("uniform fvecs"))
        .collect()
}

fn parallel_thread_counts() -> Vec<usize> {
    std::env::var("SIFT1M_PARALLEL_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&threads| threads > 0)
        .map(|threads| vec![threads])
        .unwrap_or_else(|| vec![DEFAULT_PARALLEL_THREADS, 6])
}

fn run_case(
    label: &str,
    index: &dyn HnswIndex,
    ground_truth: &[Vec<VectorId>],
    q_data: &[u8],
    dim: usize,
    n_q: usize,
    ef: usize,
    min_recall: f32,
) {
    assert_eq!(
        index.len(),
        ground_truth.len().max(index.len()),
        "{label}: len mismatch after insert"
    );

    let mut recalls = Vec::with_capacity(n_q);
    for qi in 0..n_q {
        let q = read_fvecs_vector_at(q_data, dim, qi).expect("query fvecs");
        let output = index.search(&q, K, ef);

        let retrieved: Vec<VectorId> = output.iter().map(|(id, _)| VectorId(*id)).collect();
        let r = recall_at_k(&retrieved, &ground_truth[qi]).expect("valid recall@k");
        validate_recall_score(r).expect("in-range score");
        recalls.push(r);

        eprintln!("{label} query {qi}: recall@{K}={r:.4}");
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
    helpers::sift::insert_in_batches(label, corpus.len(), chunk_size, |start, end| {
        index.insert_batch_parallel(&corpus[start..end], &vector_ids[start..end], num_threads);
    });
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
    let ground_truth =
        load_or_compute_ground_truth(&ctx.base_dir, &corpus, ctx.q_data(), dim, n_q, K);

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);
    let parallel_thread_counts = parallel_thread_counts();

    eprintln!(
        "sift_hnsw_recall: dim={dim} n_base={n_base} n_q={n_q} k={K} m={M} m_max0={M_MAX0} \
         ef_search={ef} ef_construction={ef_construction} parallel_threads={parallel_thread_counts:?} \
         alignment={DEFAULT_ALIGNMENT} rng_seed={RNG_SEED}"
    );

    let mut cases: Vec<(String, Box<dyn HnswIndex>, usize)> = vec![
        (
            "naive".to_string(),
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
            "arena".to_string(),
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
            "arena/single".to_string(),
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
    ];
    cases.extend(parallel_thread_counts.into_iter().map(|num_threads| {
        (
            format!("arena/parallel-{num_threads}"),
            Box::new(HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            )) as Box<dyn HnswIndex>,
            num_threads,
        )
    }));

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
            &ground_truth,
            ctx.q_data(),
            dim,
            n_q,
            ef,
            0.75,
        );
        index.reset();
    }
}

/// Unique temp dir for disk-swap QPS tests; caller responsible for cleanup.
fn unique_swap_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("mem_weaver_hnsw_qps_disk_{tag}_{pid}_{n}"))
}

/// Recursively deletes the directory on drop so the test stays self-cleaning on panic.
struct DirGuard(std::path::PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Same sweep as [`sift1m_hnsw_qps_parallel`], but every arena block is swapped to disk
/// (via [`HnswIndex::swap_out`]) before the queries run, so reads go through the
/// `pread`-based on-disk path (`NodeBlockStorage::OnDisk`) instead of the anonymous mmap.
/// Quantifies the QPS/latency cost of the on-disk read path relative to RAM.
#[test]
fn sift1m_hnsw_qps_disk() {
    let _serial = TEST_MUTEX.lock().unwrap();

    let n_q: usize = std::env::var("SIFT1M_HNSW_QPS_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_QPS_NUM_QUERIES);

    let Some(ctx) = try_load_sift_ctx(DEFAULT_NUM_BASE_VECTORS, n_q, DEFAULT_SEARCH_EF) else {
        return;
    };

    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let ef = ctx.search_ef.max(K);

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    eprintln!(
        "sift1m_hnsw_qps_disk: dim={dim} n_base={n_base} n_q={n_q} n_threads=6 \
         k={K} m={M} m_max0={M_MAX0} ef_search={ef} ef_construction={ef_construction} rng_seed={RNG_SEED}"
    );

    let corpus: Vec<Vec<f32>> = (0..n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();

    let mut index = HnswArena::new(
        dim,
        M,
        M_MAX0,
        ef_construction,
        n_base,
        StdRng::seed_from_u64(RNG_SEED),
    );

    let vector_ids: Vec<u64> = (0..n_base as u64).collect();
    insert_parallel(
        "sift1m_hnsw_qps_disk",
        &mut index,
        &corpus,
        &vector_ids,
        6,
        10_000,
    );

    let dir = unique_swap_dir("qps");
    let _guard = DirGuard(dir.clone());
    let t_swap = Instant::now();
    let moved = index.swap_out(&dir).expect("swap_out to disk");
    eprintln!(
        "sift1m_hnsw_qps_disk: swapped {moved} blocks to disk at {dir:?} in {:.3} ms \
         ({:.3} ms/1k vectors)",
        ms(t_swap.elapsed()),
        ms(t_swap.elapsed()) / (n_base as f64 / 1000.0),
    );

    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|qi| read_fvecs_vector_at(ctx.q_data(), dim, qi % ctx.n_q).expect("query fvecs"))
        .collect();

    // First query after swap_out has to fault in every block it touches straight from
    // the OnDisk (pread) path with nothing warmed — no separate index-level cache to
    // populate, unlike LanceDB's Session/GlobalIndexCache, so this should only reflect
    // per-block pread cost, not a cold-cache penalty.
    let t_first = Instant::now();
    let _ = index.search(&queries[0], K, ef);
    eprintln!(
        "sift1m_hnsw_qps_disk: COLD first query latency = {:.3} ms",
        ms(t_first.elapsed())
    );
    let n_warmup = 20.min(n_q.saturating_sub(1));
    eprint!("sift1m_hnsw_qps_disk: warm-up curve (ms):");
    for q in &queries[1..=n_warmup] {
        let t0 = Instant::now();
        let _ = index.search(q, K, ef);
        eprint!(" {:.3}", ms(t0.elapsed()));
    }
    eprintln!();

    helpers::sift::measure_qps("sift1m_hnsw_qps_disk", &queries, &[1, 2, 4, 6], |query| {
        index.search(query, K, ef)
    });

    // Full disk-to-memory reload: reads every swapped-out block back into a fresh
    // anonymous-mmap Arena (read_exact + CRC32 check per block), the same path used
    // by TimeBucketIndex::add_restored_bucket / swap_bucket_in on service startup —
    // as opposed to the lazy per-node pread the OnDisk path above just measured.
    let t_load = Instant::now();
    let restored = index.swap_in().expect("swap_in from disk");
    eprintln!(
        "sift1m_hnsw_qps_disk: swap_in restored {restored} blocks ({n_base} vectors) to RAM in \
         {:.3} ms ({:.3} ms/1k vectors)",
        ms(t_load.elapsed()),
        ms(t_load.elapsed()) / (n_base as f64 / 1000.0),
    );
}

#[test]
fn sift1m_hnsw_qps_parallel() {
    let _serial = TEST_MUTEX.lock().unwrap();

    let n_q: usize = std::env::var("SIFT1M_HNSW_QPS_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_QPS_NUM_QUERIES);

    let Some(ctx) = try_load_sift_ctx(DEFAULT_NUM_BASE_VECTORS, n_q, DEFAULT_SEARCH_EF) else {
        return;
    };

    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let ef = ctx.search_ef.max(K);

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    eprintln!(
        "sift1m_hnsw_qps_parallel: dim={dim} n_base={n_base} n_q={n_q} \
         k={K} m={M} m_max0={M_MAX0} ef_search={ef} ef_construction={ef_construction} rng_seed={RNG_SEED}"
    );

    let corpus: Vec<Vec<f32>> = (0..n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();

    let mut index = HnswArena::new(
        dim,
        M,
        M_MAX0,
        ef_construction,
        n_base,
        StdRng::seed_from_u64(RNG_SEED),
    );

    let vector_ids: Vec<u64> = (0..n_base as u64).collect();
    insert_parallel(
        "sift1m_hnsw_qps_parallel",
        &mut index,
        &corpus,
        &vector_ids,
        6,
        10_000,
    );

    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|qi| read_fvecs_vector_at(ctx.q_data(), dim, qi % ctx.n_q).expect("query fvecs"))
        .collect();

    helpers::sift::measure_qps(
        "sift1m_hnsw_qps_parallel",
        &queries,
        &[1, 2, 4, 6],
        |query| index.search(query, K, ef),
    );
}
