//! Memory profiling of SIFT1M HNSW insertion without the Criterion harness.
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory containing `sift_base.fvecs` and
//!   `sift_query.fvecs`. The test returns immediately when unset.
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors to insert (default
//!   `1_000_000`).
//! - `SIFT1M_RECALL_N_QUERIES` — query-vector cap passed to the shared SIFT
//!   loader (default `10_000`; queries are not searched by this test).
//! - `SIFT1M_HNSW_EF_CONSTRUCTION` — construction `ef` (default `100`).
//! - `SIFT1M_HNSW_BENCH_VARIANT` — `arena` (default), `naive`, or `both`.
//! - `SIFT1M_HNSW_MEM_INSERT_STEP` — insert interval for RSS logging (default
//!   `10_000`).

use common::benchmark::try_load_sift_ctx;
use common::memory_usage::{peak_rss_kb, rss_kb};
use index::{HnswArena, HnswIndex, HnswNaive, DEFAULT_ALIGNMENT};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;
use std::time::Instant;
use vector::read_fvecs_vector_at;

const M: usize = 16;
const M_MAX0: usize = 32;
const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;
const DEFAULT_NUM_BASE_VECTORS: usize = 10_000;
const DEFAULT_NUM_QUERIES: usize = 10;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_EF_CONSTRUCTION: usize = 100;
const DEFAULT_MEM_INSERT_STEP: usize = 10_000;

#[derive(Clone, Copy)]
enum Variant {
    Arena,
    Naive,
    Both,
}

fn variant_from_env() -> Variant {
    match std::env::var("SIFT1M_HNSW_BENCH_VARIANT")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "" | "arena" => Variant::Arena,
        "naive" => Variant::Naive,
        "both" => Variant::Both,
        value => {
            panic!("invalid SIFT1M_HNSW_BENCH_VARIANT={value:?}; expected arena, naive, or both")
        }
    }
}

fn memory_insert_step() -> usize {
    std::env::var("SIFT1M_HNSW_MEM_INSERT_STEP")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&step| step > 0)
        .unwrap_or(DEFAULT_MEM_INSERT_STEP)
}

fn log_rss_after_insert(inserted: usize, total: usize, step: usize, started: Instant) {
    if inserted % step != 0 && inserted != total {
        return;
    }
    eprintln!(
        "memory [hnsw_insert {inserted}/{total}]: rss_kb={} peak_rss_kb={} insert_phase_ms={:.3}",
        rss_kb(),
        peak_rss_kb(),
        started.elapsed().as_secs_f64() * 1e3,
    );
    let _ = std::io::stderr().flush();
}

fn profile_inserts(label: &str, index: &mut dyn HnswIndex, corpus: &[Vec<f32>], step: usize) {
    let started = Instant::now();
    for (i, vector) in corpus.iter().enumerate() {
        index.insert(vector, i as u64);
        log_rss_after_insert(i + 1, corpus.len(), step, started);
    }
    eprintln!(
        "memory [{label}]: insert_wall_ms={:.3} rss_kb={} peak_rss_kb={}",
        started.elapsed().as_secs_f64() * 1e3,
        rss_kb(),
        peak_rss_kb(),
    );
}

#[test]
fn sift1m_memory() {
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };
    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);
    let step = memory_insert_step();
    let corpus: Vec<Vec<f32>> = (0..n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("uniform fvecs"))
        .collect();

    eprintln!(
        "sift1m_memory: dim={dim} n_base={n_base} m={M} m_max0={M_MAX0} \
         ef_construction={ef_construction} alignment={DEFAULT_ALIGNMENT} mem_insert_step={step} \
         rng_seed={RNG_SEED}"
    );
    eprintln!(
        "memory [after_decode_corpus]: rss_kb={} peak_rss_kb={}",
        rss_kb(),
        peak_rss_kb(),
    );

    match variant_from_env() {
        Variant::Arena => {
            let mut index = HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            );
            profile_inserts("arena", &mut index, &corpus, step);
        }
        Variant::Naive => {
            let mut index = HnswNaive::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                StdRng::seed_from_u64(RNG_SEED),
            );
            profile_inserts("naive", &mut index, &corpus, step);
        }
        Variant::Both => {
            let mut arena = HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                StdRng::seed_from_u64(RNG_SEED),
            );
            profile_inserts("arena", &mut arena, &corpus, step);

            drop(arena);

            let mut naive = HnswNaive::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                StdRng::seed_from_u64(RNG_SEED),
            );
            profile_inserts("naive", &mut naive, &corpus, step);

            drop(naive);
        }
    }
}
