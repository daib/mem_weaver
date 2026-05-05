//! Import throughput for SIFT-scale `.fvecs` data into [`vector::VectorStore`].
//!
//! # Dataset (128-D SIFT, 1M base vectors)
//!
//! Typical layout: `sift_base.fvecs` from the [Texmex](http://corpus-texmex.irisa.fr/)
//! or [BIGANN](http://corpus-texmex.irisa.fr/) SIFT tarball. Each vector is:
//! `i32` dimension (little-endian) + `dim` `f32` values (little-endian).
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory containing `sift_base.fvecs` and `sift_query.fvecs` (same layout
//!   as other SIFT tests; see [`common::benchmark::try_load_sift_ctx`]). If unset or load fails, only
//!   the synthetic micro-benchmark runs.
//! - `SIFT1M_LIMIT` — default max base vectors to import per iteration (passed as the base-vector
//!   cap to [`common::benchmark::try_load_sift_ctx`]; overridden by `SIFT1M_RECALL_N_BASE` when set).
//!   Use `1000000` for full SIFT1M (needs a large arena and patience).
//!
//! Run HNSW SIFT benchmarks with: `cargo bench -p index --bench hnsw_sift1m` (source:
//! `crates/index/benches/hnsw/sift1m.rs`).
//!
//! ## Memory
//!
//! With `SIFT1M_BASE_PATH` set, prints **`rss_kb`** / **`peak_rss_kb`** after mmap context load and
//! after one cold full [`VectorStore::import_fvecs`] (stderr).

use common::benchmark::try_load_sift_ctx;
use common::memory_usage::{peak_rss_kb, rss_kb};
use common::DEFAULT_ARENA_CAPACITY;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use vector::VectorStore;

fn import_synthetic(dim: usize, n: usize) -> usize {
    let stride = 4 + dim * 4;
    let mut synthetic = vec![0u8; n * stride];
    for i in 0..n {
        let off = i * stride;
        synthetic[off..off + 4].copy_from_slice(&(dim as i32).to_le_bytes());
        for j in 0..dim {
            let f = (i * dim + j) as f32 * 0.001;
            synthetic[off + 4 + j * 4..off + 4 + (j + 1) * 4].copy_from_slice(&f.to_le_bytes());
        }
    }
    let mut store = VectorStore::new(dim, DEFAULT_ARENA_CAPACITY);
    store.import_fvecs(&synthetic, dim, n)
}

fn bench_sift1m_import(c: &mut Criterion) {
    let limit: usize = std::env::var("SIFT1M_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000);

    let mut group = c.benchmark_group("sift1m_import");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(5));

    group.bench_with_input(
        BenchmarkId::new("synthetic_fvecs_push", limit),
        &limit,
        |b, &lim| {
            b.iter(|| {
                let got = import_synthetic(128, lim);
                black_box(got)
            });
        },
    );

    // Keep [`common::benchmark::SiftCtx`] alive until `group.finish()` so mmap-backed slices stay valid.
    let sift_ctx = try_load_sift_ctx(limit, 1, 100);
    if let Some(ctx) = sift_ctx.as_ref() {
        let dim = ctx.dim;
        let n_import = ctx.n_base;
        let base_data = ctx.base_data();
        eprintln!(
            "mmap_fvecs_push: dim={dim}  n_import={n_import}  (see try_load_sift_ctx / SIFT1M_* env)"
        );

        eprintln!(
            "memory [mmap_ctx_loaded]: rss_kb={} peak_rss_kb={}",
            rss_kb(),
            peak_rss_kb()
        );
        {
            let mut store_once = VectorStore::new(dim, DEFAULT_ARENA_CAPACITY);
            let imported = store_once.import_fvecs(base_data, dim, n_import);
            eprintln!(
                "memory [after_mmap_vector_import_once]: rss_kb={} peak_rss_kb={} vectors_imported={}",
                rss_kb(),
                peak_rss_kb(),
                imported
            );
        }

        let bench_input = (dim, n_import);
        group.bench_with_input(
            BenchmarkId::new("mmap_fvecs_push", n_import),
            &bench_input,
            |b, &(dim, n_import)| {
                b.iter(|| {
                    let mut store = VectorStore::new(dim, DEFAULT_ARENA_CAPACITY);
                    black_box(store.import_fvecs(base_data, dim, n_import))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sift1m_import);
criterion_main!(benches);
