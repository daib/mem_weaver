//! Shared batching and progress reporting for SIFT1M insertion tests.

use common::benchmark::latency_percentile;
use std::time::Instant;

fn ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Runs `insert_batch` over contiguous corpus ranges and reports cumulative progress.
pub fn insert_in_batches(
    label: &str,
    total: usize,
    chunk_size: usize,
    mut insert_batch: impl FnMut(usize, usize),
) {
    assert!(chunk_size > 0, "chunk_size must be greater than zero");

    let started = Instant::now();
    for start in (0..total).step_by(chunk_size) {
        let end = (start + chunk_size).min(total);
        insert_batch(start, end);
        eprintln!(
            "{label}: inserted {end}/{total} vectors (cumulative {:.3} ms)",
            ms(started.elapsed()),
        );
    }
    eprintln!("{label}: build total {:.3} ms", ms(started.elapsed()));
}

/// Measures query throughput and latency for each requested thread count.
///
/// This is intentionally measurement-only: callers decide whether and how to
/// validate the search results separately.
pub fn measure_qps<T>(
    label: &str,
    queries: &[Vec<f32>],
    thread_counts: &[usize],
    search: impl Fn(&[f32]) -> T + Sync,
) {
    assert!(!queries.is_empty(), "at least one query is required");
    for &num_threads in thread_counts {
        assert!(num_threads > 0, "thread count must be greater than zero");
        let chunk_size = queries.len().div_ceil(num_threads);
        eprintln!(
            "{label}: running {} queries across {num_threads} threads",
            queries.len()
        );
        let started = Instant::now();
        let search = &search;

        let per_thread_latencies: Vec<Vec<f64>> = std::thread::scope(|scope| {
            queries
                .chunks(chunk_size)
                .map(|chunk| {
                    scope.spawn(move || {
                        let mut latencies = Vec::with_capacity(chunk.len());
                        for query in chunk {
                            let query_started = Instant::now();
                            std::hint::black_box(search(query));
                            latencies.push(ms(query_started.elapsed()));
                        }
                        latencies
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().expect("query worker panicked"))
                .collect()
        });

        let elapsed = started.elapsed().as_secs_f64();
        let qps = queries.len() as f64 / elapsed;
        let mut latencies_ms: Vec<f64> = per_thread_latencies.into_iter().flatten().collect();
        let p50 = latency_percentile(&mut latencies_ms, 50.0);
        let p95 = latency_percentile(&mut latencies_ms, 95.0);
        let p99 = latency_percentile(&mut latencies_ms, 99.0);
        eprintln!(
            "{label}: n_q={} threads={num_threads} total={:.3}ms qps={qps:.1} \
             p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms",
            queries.len(),
            elapsed * 1e3,
        );
    }
}
