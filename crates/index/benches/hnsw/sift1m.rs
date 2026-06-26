//! SIFT1M-style **HNSW** benchmarks ([`index::HnswArena`]) using the shared loader in
//! [`common::benchmark::try_load_sift_ctx`].
//!
//! # Environment
//!
//! See workspace SIFT docs; highlights:
//!
//! - `SIFT1M_BASE_PATH` — directory with `sift_base.fvecs` and `sift_query.fvecs`.
//! - `SIFT1M_LIMIT` — default corpus cap passed to the loader when **`SIFT1M_RECALL_N_BASE`** is unset
//!   (**`1_000_000`** vectors; clamped by what `sift_base.fvecs` contains).
//! - `SIFT1M_RECALL_N_BASE` / `SIFT1M_RECALL_N_QUERIES` — override base / query counts (see
//!   [`common::benchmark::try_load_sift_ctx`]).
//! - `SIFT1M_HNSW_EF_CONSTRUCTION` — construction list size (default `200`).
//! - `SIFT1M_HNSW_BENCH_QUERIES` — max queries per timed search batch (default `min(n_q, 100)`).
//! - `SIFT1M_HNSW_BENCH_SAMPLE_SIZE` — number of timed repetitions per benchmark section:
//!   - **`1` … `9`** — **smoke mode**: Criterion is **not** used; wall-clock timings print to stderr (no
//!     `sample_size` / Criterion harness).
//!   - **`10`** or higher — **Criterion** (`BenchmarkGroup::sample_size`; Criterion 0.5 requires **`≥ 10`**).
//!   - **unset** — defaults to **`10`** (Criterion).
//! - `SIFT1M_HNSW_BENCH_VARIANT` — **`arena`** (default; mmap arena graph only), **`naive`** (heap graph
//!   only), or **`both`** (arena then naive — **two** full benchmark suites).
//! - `SIFT1M_HNSW_CRITERION_LEGACY_INSERT_ORDER=1` — **Criterion only**: restore the old order (timed
//!   **`hnsw_arena_build`** first, then a **second** full insert before search). Default is **warmup
//!   insert first**, then timed build (independent scratch arenas), then search — **one** insert into the
//!   index used for search.
//!
//! **Total insertion wall time** (the full corpus load into the search index) is printed as
//! **`total_search_index_insert_wall_ms`** (Criterion) or **`total_insertion_wall_ms`** / **`total_warmup_insert_wall_ms`**
//! (smoke) on stderr after that phase completes.
//!
//! ## Memory
//!
//! Prints **`rss_kb`** / **`peak_rss_kb`** (`common::memory_usage`) to stderr after decoding the
//! corpus. During the **warm-up** full insert (the index reused for search), also logs every
//! **`SIFT1M_HNSW_MEM_INSERT_STEP`** inserts (default **10000**) and on the final insert. Each line
//! includes **`insert_phase_ms`**: wall time since that insert batch started, followed by a stderr flush
//! so lines appear under piped or fully buffered stderr (some IDE terminals).
//!
//! Inside the timed **`hnsw_arena_build`** loop, logging is **off** by default for Criterion (it skews
//! timings). Set **`SIFT1M_HNSW_MEM_LOG_BUILD_ITER=1`** to enable it.
//!
//! **Smoke mode** (`SIFT1M_HNSW_BENCH_SAMPLE_SIZE` **1–9**): build-loop logging defaults to **on** so long
//! inserts are not silent; set **`SIFT1M_HNSW_MEM_LOG_BUILD_ITER=0`** to disable.
//!
//! With **`SIFT1M_HNSW_BENCH_SAMPLE_SIZE=1`**, the timed build (**arena** or **naive**) **reuses that graph for
//! **`search_batch`** (no second full insert). With **`sample_size` ≥ 2**, smoke still runs multiple timed cold
//! builds, then **one** warmup insert into a fresh index for search (same as before).

use common::benchmark::{try_load_sift_ctx, SiftCtx};
use common::memory_usage::{peak_rss_kb, rss_kb};
use common::top_k_quickselect;
use criterion::{black_box, Criterion};
use index::{HnswArena, HnswIndex, HnswNaive, DEFAULT_ALIGNMENT};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;
use std::time::{Duration, Instant};
use vector::read_fvecs_vector_at;

const K: usize = 10;
const M: usize = 16;
const M_MAX0: usize = 32;
const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;

const DEFAULT_EF_CONSTRUCTION: usize = 40;
const DEFAULT_SEARCH_EF: usize = 100;
/// Default base-vector cap when `SIFT1M_LIMIT` / `SIFT1M_RECALL_N_BASE` do not override (full SIFT1M).
const DEFAULT_SIFT_NUM_BASE: usize = 1_000_000;
const DEFAULT_BENCH_SAMPLE_SIZE: usize = 10_000;
/// Criterion 0.5 `BenchmarkGroup::sample_size` requires this minimum (`assert!(n >= 10)`).
const CRITERION_MIN_SAMPLE_SIZE: usize = 10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BenchVariant {
    Arena,
    Naive,
    Both,
}

/// `SIFT1M_HNSW_BENCH_VARIANT`: **`arena`** (default), **`naive`**, or **`both`**.
fn bench_variant_from_env() -> BenchVariant {
    let s = std::env::var("SIFT1M_HNSW_BENCH_VARIANT").unwrap_or_default();
    if s.is_empty() || s.eq_ignore_ascii_case("arena") {
        BenchVariant::Arena
    } else if s.eq_ignore_ascii_case("naive") {
        BenchVariant::Naive
    } else if s.eq_ignore_ascii_case("both") {
        BenchVariant::Both
    } else {
        eprintln!(
            "sift1m_hnsw: invalid SIFT1M_HNSW_BENCH_VARIANT={s:?} (expected arena, naive, or both)"
        );
        std::process::exit(1);
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct BenchSetup {
    dim: usize,
    n_base: usize,
    _n_q: usize,
    ef: usize,
    ef_construction: usize,
    max_bench_q: usize,
    mem_insert_step: usize,
    log_mem_inside_build_bench: bool,
    corpus: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
}

fn try_open_sift_ctx() -> Option<SiftCtx> {
    let limit: usize = std::env::var("SIFT1M_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SIFT_NUM_BASE);

    let n_queries_cfg: usize = std::env::var("SIFT1M_RECALL_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    try_load_sift_ctx(limit, n_queries_cfg, DEFAULT_SEARCH_EF)
}

fn prepare_setup(ctx: &SiftCtx) -> BenchSetup {
    let base_data = ctx.base_data();
    let q_data = ctx.q_data();
    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let n_q = ctx.n_q;
    let ef = ctx.search_ef.max(K);

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    let max_bench_q: usize = std::env::var("SIFT1M_HNSW_BENCH_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
        .min(n_q);

    let mem_insert_step: usize = std::env::var("SIFT1M_HNSW_MEM_INSERT_STEP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000)
        .max(1);

    let log_mem_inside_build_bench = parse_mem_log_build_iter_env();

    let corpus: Vec<Vec<f32>> = (0..n_base)
        .map(|i| read_fvecs_vector_at(base_data, dim, i).expect("base fvecs row"))
        .collect();

    let queries: Vec<Vec<f32>> = (0..max_bench_q)
        .map(|qi| read_fvecs_vector_at(q_data, dim, qi).expect("query fvecs"))
        .collect();

    BenchSetup {
        dim,
        n_base,
        _n_q: n_q,
        ef,
        ef_construction,
        max_bench_q,
        mem_insert_step,
        log_mem_inside_build_bench,
        corpus,
        queries,
    }
}

/// `SIFT1M_HNSW_MEM_LOG_BUILD_ITER=1` enables logging inside the timed **`hnsw_arena_build`** loop (Criterion).
fn parse_mem_log_build_iter_env() -> bool {
    std::env::var("SIFT1M_HNSW_MEM_LOG_BUILD_ITER")
        .ok()
        .as_deref()
        == Some("1")
}

fn prepare_setup_smoke(ctx: &SiftCtx) -> BenchSetup {
    let mut setup = prepare_setup(ctx);
    // Smoke spends a long time in the timed arena-build loop before warmup; enable build-step RSS lines
    // unless explicitly turned off (`SIFT1M_HNSW_MEM_LOG_BUILD_ITER=0`).
    setup.log_mem_inside_build_bench =
        match std::env::var("SIFT1M_HNSW_MEM_LOG_BUILD_ITER").as_deref() {
            Ok("0") => false,
            _ => true,
        };
    setup
}

/// Emit RSS / peak RSS when `insert_done` hits every `step` inserts or finishes (`insert_done == n_base`).
/// **`insert_started`** is `Instant::now()` immediately before the first `insert` in this phase.
fn log_rss_after_insert(insert_done: usize, n_base: usize, step: usize, insert_started: Instant) {
    if insert_done == 0 || step == 0 {
        return;
    }
    if insert_done % step != 0 && insert_done != n_base {
        return;
    }
    let insert_phase_ms = insert_started.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "memory [hnsw_insert {insert_done}/{n_base}]: rss_kb={} peak_rss_kb={} insert_phase_ms={insert_phase_ms:.3}",
        rss_kb(),
        peak_rss_kb()
    );
    let _ = std::io::stderr().flush();
}

fn execute_benchmark(c: &mut Criterion, ctx: &SiftCtx, index: &mut dyn HnswIndex) {
    let legacy_insert_order = std::env::var("SIFT1M_HNSW_CRITERION_LEGACY_INSERT_ORDER")
        .map(|v| v == "1")
        .unwrap_or(false);

    let setup = prepare_setup(ctx);

    let raw_sample: Option<usize> = std::env::var("SIFT1M_HNSW_BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok());
    let bench_sample_size = raw_sample
        .unwrap_or(DEFAULT_BENCH_SAMPLE_SIZE)
        .max(CRITERION_MIN_SAMPLE_SIZE);

    let mut group = c.benchmark_group("sift1m_hnsw");
    group.sample_size(bench_sample_size);
    group.measurement_time(Duration::from_secs(30));

    let BenchSetup {
        dim,
        n_base,
        ef,
        ef_construction,
        max_bench_q,
        ref corpus,
        ref queries,
        mem_insert_step,
        log_mem_inside_build_bench,
        ..
    } = &setup;

    eprintln!(
        "sift1m_hnsw: dim={dim} n_base={n_base} n_q={} k={K} m={M} m_max0={M_MAX0} ef_search={ef} ef_construction={ef_construction} search_batch={max_bench_q} criterion_sample_size={bench_sample_size} alignment={DEFAULT_ALIGNMENT} mem_insert_step={mem_insert_step} rng_seed={RNG_SEED} criterion_legacy_insert_order={legacy_insert_order}",
        setup._n_q
    );

    eprintln!(
        "memory [after_decode_corpus]: rss_kb={} peak_rss_kb={}",
        rss_kb(),
        peak_rss_kb()
    );

    let mut insert_search_index = || {
        let insert_started = Instant::now();
        for (i, v) in corpus.iter().enumerate() {
            index.insert(v.as_slice(), i as u64);
            log_rss_after_insert(i + 1, *n_base, *mem_insert_step, insert_started);
        }
        assert_eq!(index.len(), *n_base);
    };

    if !legacy_insert_order {
        eprintln!(
            "sift1m_hnsw: warming search index (full insert) before timed hnsw_arena_build — skipping second insert before search_batch"
        );
        let t_insert = Instant::now();
        insert_search_index();
        eprintln!(
            "sift1m_hnsw: total_search_index_insert_wall_ms={:.3} (one full corpus load into the index used for search_batch)",
            ms(t_insert.elapsed())
        );
        let _ = std::io::stderr().flush();
    }

    group.bench_function("hnsw_arena_build", |b| {
        b.iter(|| {
            let mut arena = HnswArena::new(
                *dim,
                M,
                M_MAX0,
                *ef_construction,
                *n_base,
                StdRng::seed_from_u64(RNG_SEED),
            );
            let insert_started = Instant::now();
            for (i, v) in corpus.iter().enumerate() {
                arena.insert(v.as_slice(), i as u64);
                if *log_mem_inside_build_bench {
                    log_rss_after_insert(i + 1, *n_base, *mem_insert_step, insert_started);
                }
            }
            black_box(arena.len());
        })
    });

    if legacy_insert_order {
        let t_insert = Instant::now();
        insert_search_index();
        eprintln!(
            "sift1m_hnsw: total_search_index_insert_wall_ms={:.3} (full corpus after timed hnsw_arena_build; legacy insert order)",
            ms(t_insert.elapsed())
        );
        let _ = std::io::stderr().flush();
    }

    group.bench_function("hnsw_arena_search_batch", |b| {
        b.iter(|| {
            for q in queries {
                black_box(index.search(q.as_slice(), K, *ef));
            }
        })
    });

    group.finish();
}

#[derive(Clone, Copy)]
enum SmokeFlavor {
    Arena,
    Naive,
}

fn smoke_run_flavor(iterations: usize, setup: &BenchSetup, flavor: SmokeFlavor) {
    let dim = setup.dim;
    let n_base = setup.n_base;
    let ef_construction = setup.ef_construction;
    let ef = setup.ef;
    let tag = match flavor {
        SmokeFlavor::Arena => "bench_hnsw_sift1m",
        SmokeFlavor::Naive => "bench_hnsw_naive_sift1m",
    };
    let (build_bench, search_bench) = match flavor {
        SmokeFlavor::Arena => ("hnsw_arena_build", "hnsw_arena_search_batch"),
        SmokeFlavor::Naive => ("hnsw_naive_build", "hnsw_naive_search_batch"),
    };

    eprintln!(
        "smoke [{tag}] {build_bench}: {n_base} inserts (memory lines every {} vectors to stderr)",
        setup.mem_insert_step
    );
    let _ = std::io::stderr().flush();

    // Single smoke rep: reuse the graph from the timed cold build for search (matches Criterion single-insert story).
    if iterations == 1 {
        let t0 = Instant::now();
        let index: Box<dyn HnswIndex> = match flavor {
            SmokeFlavor::Arena => {
                let mut arena = HnswArena::new(
                    dim,
                    M,
                    M_MAX0,
                    ef_construction,
                    n_base,
                    StdRng::seed_from_u64(RNG_SEED),
                );
                let insert_started = Instant::now();
                for (i, v) in setup.corpus.iter().enumerate() {
                    arena.insert(v.as_slice(), i as u64);
                    if setup.log_mem_inside_build_bench {
                        log_rss_after_insert(i + 1, n_base, setup.mem_insert_step, insert_started);
                    }
                }
                black_box(arena.len());
                Box::new(arena)
            }
            SmokeFlavor::Naive => {
                let mut naive = HnswNaive::new(
                    dim,
                    M,
                    M_MAX0,
                    ef_construction,
                    StdRng::seed_from_u64(RNG_SEED),
                );
                let insert_started = Instant::now();
                for (i, v) in setup.corpus.iter().enumerate() {
                    naive.insert(v.as_slice(), i as u64);
                    if setup.log_mem_inside_build_bench {
                        log_rss_after_insert(i + 1, n_base, setup.mem_insert_step, insert_started);
                    }
                }
                black_box(naive.len());
                Box::new(naive)
            }
        };
        let insert_ms = ms(t0.elapsed());
        eprintln!(
            "smoke [{tag}] {build_bench} rep 0: {:.3} ms (total_insertion_wall_ms={insert_ms:.3})",
            insert_ms,
        );
        eprintln!(
            "smoke [{tag}] using index from build for search_batch (skipping duplicate warmup insert)"
        );
        let _ = std::io::stderr().flush();

        let t_search = Instant::now();
        for q in &setup.queries {
            black_box(index.search(q.as_slice(), K, ef));
        }
        eprintln!(
            "smoke [{tag}] {search_bench} rep 0: {:.3} ms",
            ms(t_search.elapsed()),
        );
        return;
    }

    for rep in 0..iterations {
        let t0 = Instant::now();
        match flavor {
            SmokeFlavor::Arena => {
                let mut arena = HnswArena::new(
                    dim,
                    M,
                    M_MAX0,
                    ef_construction,
                    n_base,
                    StdRng::seed_from_u64(RNG_SEED),
                );
                let insert_started = Instant::now();
                for (i, v) in setup.corpus.iter().enumerate() {
                    arena.insert(v.as_slice(), i as u64);
                    if setup.log_mem_inside_build_bench {
                        log_rss_after_insert(i + 1, n_base, setup.mem_insert_step, insert_started);
                    }
                }
                black_box(arena.len());
            }
            SmokeFlavor::Naive => {
                let mut naive = HnswNaive::new(
                    dim,
                    M,
                    M_MAX0,
                    ef_construction,
                    StdRng::seed_from_u64(RNG_SEED),
                );
                let insert_started = Instant::now();
                for (i, v) in setup.corpus.iter().enumerate() {
                    naive.insert(v.as_slice(), i as u64);
                    if setup.log_mem_inside_build_bench {
                        log_rss_after_insert(i + 1, n_base, setup.mem_insert_step, insert_started);
                    }
                }
                black_box(naive.len());
            }
        }
        let insert_ms = ms(t0.elapsed());
        eprintln!(
            "smoke [{tag}] {build_bench} rep {rep}: {:.3} ms (total_insertion_wall_ms={insert_ms:.3})",
            insert_ms,
        );
    }

    let mut index: Box<dyn HnswIndex> = match flavor {
        SmokeFlavor::Arena => Box::new(HnswArena::new(
            dim,
            M,
            M_MAX0,
            ef_construction,
            n_base,
            StdRng::seed_from_u64(RNG_SEED),
        )),
        SmokeFlavor::Naive => Box::new(HnswNaive::new(
            dim,
            M,
            M_MAX0,
            ef_construction,
            StdRng::seed_from_u64(RNG_SEED),
        )),
    };
    let insert_started = Instant::now();
    let t_warmup_insert = Instant::now();
    for (i, v) in setup.corpus.iter().enumerate() {
        index.insert(v.as_slice(), i as u64);
        log_rss_after_insert(i + 1, n_base, setup.mem_insert_step, insert_started);
    }
    assert_eq!(index.len(), n_base);
    eprintln!(
        "smoke [{tag}] total_warmup_insert_wall_ms={:.3} (full corpus → index used for search_batch)",
        ms(t_warmup_insert.elapsed())
    );
    let _ = std::io::stderr().flush();

    for rep in 0..iterations {
        let t0 = Instant::now();
        for q in &setup.queries {
            black_box(index.search(q.as_slice(), K, ef));
        }
        eprintln!(
            "smoke [{tag}] {search_bench} rep {rep}: {:.3} ms",
            ms(t0.elapsed()),
        );
    }
}

fn run_smoke_benchmarks(iterations: usize) {
    let sift_ctx = try_open_sift_ctx();
    let Some(ctx) = sift_ctx.as_ref() else {
        eprintln!(
            "sift1m_hnsw: try_load_sift_ctx failed; skip (set SIFT1M_BASE_PATH + .fvecs files)"
        );
        return;
    };

    let setup = prepare_setup_smoke(ctx);

    eprintln!(
        "sift1m_hnsw smoke: dim={} n_base={} n_q={} k={K} m={M} m_max0={M_MAX0} ef_search={} ef_construction={} search_batch={} alignment={DEFAULT_ALIGNMENT} mem_insert_step={} rng_seed={RNG_SEED} repetitions={iterations} mem_log_build={}",
        setup.dim,
        setup.n_base,
        setup._n_q,
        setup.ef,
        setup.ef_construction,
        setup.max_bench_q,
        setup.mem_insert_step,
        setup.log_mem_inside_build_bench,
    );

    eprintln!(
        "memory [after_decode_corpus]: rss_kb={} peak_rss_kb={}",
        rss_kb(),
        peak_rss_kb()
    );

    match bench_variant_from_env() {
        BenchVariant::Arena => smoke_run_flavor(iterations, &setup, SmokeFlavor::Arena),
        BenchVariant::Naive => smoke_run_flavor(iterations, &setup, SmokeFlavor::Naive),
        BenchVariant::Both => {
            smoke_run_flavor(iterations, &setup, SmokeFlavor::Arena);
            smoke_run_flavor(iterations, &setup, SmokeFlavor::Naive);
        }
    }
}

fn bench_hnsw_sift1m(c: &mut Criterion) {
    let sift_ctx = try_open_sift_ctx();
    let Some(ctx) = sift_ctx.as_ref() else {
        eprintln!(
            "sift1m_hnsw: try_load_sift_ctx failed; skip (set SIFT1M_BASE_PATH + .fvecs files)"
        );
        let group = c.benchmark_group("sift1m_hnsw");
        group.finish();
        return;
    };

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    let mut arena = HnswArena::new(
        ctx.dim,
        M,
        M_MAX0,
        ef_construction,
        ctx.n_base,
        StdRng::seed_from_u64(RNG_SEED),
    );
    execute_benchmark(c, ctx, &mut arena);
}

fn bench_hnsw_naive_sift1m(c: &mut Criterion) {
    let sift_ctx = try_open_sift_ctx();
    let Some(ctx) = sift_ctx.as_ref() else {
        eprintln!(
            "sift1m_hnsw: try_load_sift_ctx failed; skip (set SIFT1M_BASE_PATH + .fvecs files)"
        );
        let group = c.benchmark_group("sift1m_hnsw");
        group.finish();
        return;
    };

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    let mut naive = HnswNaive::new(
        ctx.dim,
        M,
        M_MAX0,
        ef_construction,
        StdRng::seed_from_u64(RNG_SEED),
    );
    execute_benchmark(c, ctx, &mut naive);
}

fn main() {
    let raw_sample: Option<usize> = std::env::var("SIFT1M_HNSW_BENCH_SAMPLE_SIZE")
        .ok()
        .and_then(|s| s.parse().ok());

    match raw_sample {
        Some(0) => {
            eprintln!("sift1m_hnsw: SIFT1M_HNSW_BENCH_SAMPLE_SIZE must be > 0");
            std::process::exit(1);
        }
        Some(n) if n < CRITERION_MIN_SAMPLE_SIZE => {
            eprintln!(
                "sift1m_hnsw: smoke mode — Criterion disabled (sample size {n} < {CRITERION_MIN_SAMPLE_SIZE})"
            );
            let variant = bench_variant_from_env();
            eprintln!("sift1m_hnsw: SIFT1M_HNSW_BENCH_VARIANT={variant:?} (set to Both for arena + naive)");
            run_smoke_benchmarks(n);
        }
        _ => {
            let variant = bench_variant_from_env();
            eprintln!("sift1m_hnsw: SIFT1M_HNSW_BENCH_VARIANT={variant:?} (set to Both for arena + naive)");
            let mut criterion = Criterion::default().configure_from_args();
            match variant {
                BenchVariant::Arena => bench_hnsw_sift1m(&mut criterion),
                BenchVariant::Naive => bench_hnsw_naive_sift1m(&mut criterion),
                BenchVariant::Both => {
                    bench_hnsw_sift1m(&mut criterion);
                    bench_hnsw_naive_sift1m(&mut criterion);
                }
            }
        }
    }
}
