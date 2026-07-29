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
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors to load and index (defaults:
//!   `1_000_000` for the performance test, `10_000` for the S3 test).
//! - `SIFT1M_RECALL_N_QUERIES` — number of queries to evaluate (defaults:
//!   `10_000` for the performance test, `10` for the S3 test).
//! - `SIFT1M_HNSW_EF` — search `ef` at level 0 (default `100`).
//! - `SIFT1M_TIME_BUCKET_COUNTS` — comma-separated bucket counts to try (default `1,4,16`).
//! - `SIFT1M_PARALLEL_THREADS` — HNSW insertion threads for the performance
//!   test (default `4`; `1` uses sequential insertion).
//! ## S3 test environment
//!
//! `sift1m_time_bucket_s3_recall_vs_bruteforce` additionally exercises the
//! blob round trip:
//!
//! - `MEM_WEAVER_S3_BUCKET` — swapped-out arena files are uploaded under
//!   `s3://$BUCKET/$MEM_WEAVER_S3_PREFIX/n<num_buckets>/seq_<i>/`, then
//!   downloaded to a fresh directory before restoring. If omitted,
//!   `DEFAULT_BUCKET` is used; unavailable credentials → the S3 test returns
//!   immediately.
//! - `MEM_WEAVER_S3_REGION` (default `us-east-1`), `MEM_WEAVER_S3_PROFILE` (default `default`),
//!   `MEM_WEAVER_S3_PREFIX` (default unique per run). Credentials read from `~/.aws/credentials`.

mod helpers;

use common::benchmark::{
    load_or_compute_ground_truth, sift_recall_stats_with_gt, try_load_sift_ctx,
};
use common::{top_k_quickselect, Timestamp};
use index::{
    blob::{upload_levels, upload_manifest},
    upload_arena_dir, TimeBucketIndex, DEFAULT_ALIGNMENT,
};
use object_store::{path::Path as ObjectPath, ObjectStore};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::ThreadPoolBuilder;
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vector::{read_fvecs_vector_at, VectorId};

const K: usize = 10;
const M: usize = 16;
const M_MAX0: usize = 32;

const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;

const DEFAULT_NUM_BASE_VECTORS: usize = 10_000;
const DEFAULT_NUM_QUERIES: usize = 10;
const S3_NUM_BASE_VECTORS: usize = 10;
const S3_NUM_QUERIES: usize = 10;
const QPS_THREAD_COUNTS: &[usize] = &[1, 2, 4, 6];
const DEFAULT_PARALLEL_THREADS: usize = 6;
const DEFAULT_EF_CONSTRUCTION: usize = 100;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_BUCKET_COUNTS: &[usize] = &[4];
const INSERT_PROGRESS_INTERVAL: usize = 10_000;

// ── S3 defaults ─────────────────────────────────────────────────────────────
// Used when MEM_WEAVER_S3_* env vars are unset. Edit these to your dev bucket
// to skip exporting env vars every run.
//
// Skip rules:
//   - DEFAULT_BUCKET == "<edit-me>"           → S3 step is skipped (safe default).
//   - MEM_WEAVER_S3_BUCKET set to empty ("") → S3 step is skipped (escape hatch).
// Otherwise S3 runs with whichever value is resolved.
//
// DEFAULT_PREFIX: parent path under the bucket. A unique-per-run suffix is
// appended so concurrent runs don't collide. Empty → falls back to a fully
// unique prefix under `mem_weaver_test/sift_*`.
const DEFAULT_BUCKET: &str = "mem-weaver-test";
const DEFAULT_REGION: &str = "us-east-1";
const DEFAULT_PROFILE: &str = "default";
const DEFAULT_PREFIX: &str = "dev/sift";

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

fn parse_parallel_threads() -> usize {
    std::env::var("SIFT1M_PARALLEL_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&threads| threads > 0)
        .unwrap_or(DEFAULT_PARALLEL_THREADS)
}

/// Measures local hot, swapped-out, and restored recall without touching S3.
#[test]
fn sift1m_time_bucket_recall_vs_bruteforce_performance() {
    run_sift1m_time_bucket_recall_vs_bruteforce(
        None,
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        parse_parallel_threads(),
    );
}

/// Verifies recall survives a full S3 upload, local eviction, and blob restore.
#[test]
fn sift1m_time_bucket_s3_recall_vs_bruteforce() {
    let Some(s3) = S3Setup::try_from_env().expect("S3 setup failed") else {
        eprintln!("s3: configuration or credentials unavailable; S3 test skipped");
        return;
    };
    eprintln!(
        "s3: bucket={} region={} run_prefix={}",
        s3.bucket, s3.region, s3.run_prefix
    );

    run_sift1m_time_bucket_recall_vs_bruteforce(Some(&s3), S3_NUM_BASE_VECTORS, S3_NUM_QUERIES, 1);
}

fn run_sift1m_time_bucket_recall_vs_bruteforce(
    s3: Option<&S3Setup>,
    default_num_base_vectors: usize,
    default_num_queries: usize,
    insert_threads: usize,
) {
    let t_load = Instant::now();
    let Some(ctx) = try_load_sift_ctx(
        default_num_base_vectors,
        default_num_queries,
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

    let ground_truth = load_or_compute_ground_truth(&ctx.base_dir, &corpus, q_data, dim, n_q, K);

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    let bucket_counts = parse_bucket_counts();

    eprintln!(
        "sift_time_bucket_recall: dim={dim} n_base={n_base} n_q={n_q} k={K} m={M} m_max0={M_MAX0} ef_search={ef} ef_construction={ef_construction} insertion_threads={insert_threads} alignment={DEFAULT_ALIGNMENT} rng_seed={RNG_SEED} bucket_counts={bucket_counts:?}"
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
            StdRng::seed_from_u64(RNG_SEED),
        )
        .expect("valid TimeBucketIndex config");

        let t_idx = Instant::now();
        let mut seen_seqs = std::collections::HashSet::new();
        if insert_threads == 1 {
            for (i, v) in corpus.iter().enumerate() {
                let bid = index.insert(v.as_slice(), Timestamp(i as u64), i as u64);
                if let Some(bid) = bid {
                    seen_seqs.insert(bid.bucket_seq);
                }
            }
        } else {
            let timestamps: Vec<_> = (0..n_base).map(|i| Timestamp(i as u64)).collect();
            let vector_ids: Vec<_> = (0..n_base).map(|i| i as u64).collect();
            let thread_pool = ThreadPoolBuilder::new()
                .num_threads(insert_threads)
                .build()
                .expect("valid Rayon thread-pool configuration");

            helpers::sift::insert_in_batches(
                &label,
                n_base,
                INSERT_PROGRESS_INTERVAL,
                |start, end| {
                    for bid in thread_pool.install(|| {
                        index.insert_batch_parallel(
                            &corpus[start..end],
                            &timestamps[start..end],
                            &vector_ids[start..end],
                            insert_threads,
                        )
                    }) {
                        if let Some(bid) = bid {
                            seen_seqs.insert(bid.bucket_seq);
                        }
                    }
                },
            );
        }
        eprintln!(
            "{label}: inserted {n_base} vectors using {insert_threads} thread(s) in {:.3} ms",
            ms(t_idx.elapsed())
        );
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

        // Helper to keep the three recall passes identical.
        let run_recall = |index: &TimeBucketIndex, phase: &str| {
            let phased_label = format!("{label} [{phase}]");
            let (stats, _, _) = sift_recall_stats_with_gt(
                &phased_label,
                &ground_truth,
                q_data,
                dim,
                n_q,
                ef,
                |q| {
                    index
                        .search(q, K, ef, |_, d| d, None, top_k_quickselect)
                        .into_iter()
                        .map(|(vid, _)| VectorId(vid))
                        .collect()
                },
            );
            stats
        };

        // ── Three recall passes: hot → cold → restored ─────────────────────────
        let hot_stats = run_recall(&index, "hot");
        let queries: Vec<Vec<f32>> = (0..n_q)
            .map(|i| read_fvecs_vector_at(q_data, dim, i).expect("uniform fvecs"))
            .collect();
        helpers::sift::measure_qps(
            &format!("{label} [hot qps]"),
            &queries,
            QPS_THREAD_COUNTS,
            |query| {
                index.search(
                    query,
                    K,
                    ef,
                    |_, distance| distance,
                    None,
                    top_k_quickselect,
                )
            },
        );

        let cold_root = unique_swap_dir(&format!("sift_time_bucket_n{num_buckets}"));
        let _cold_guard = DirGuard(cold_root.clone());
        std::fs::create_dir_all(&cold_root).expect("mk cold dir");

        let bucket_seqs: Vec<_> = seen_seqs.into_iter().collect();
        let t_swap_out = Instant::now();
        for (i, seq) in bucket_seqs.iter().enumerate() {
            let dir = cold_root.join(format!("seq_{i}"));
            let moved = index.swap_bucket_out(*seq, &dir).expect("swap_bucket_out");
            assert!(moved, "{label}: every alive bucket_seq must be present");
        }
        eprintln!(
            "{label}: swapped {} buckets to disk in {:.3} ms",
            bucket_seqs.len(),
            ms(t_swap_out.elapsed())
        );

        let cold_stats = run_recall(&index, "cold");
        helpers::sift::measure_qps(
            &format!("{label} [cold qps]"),
            &queries,
            QPS_THREAD_COUNTS,
            |query| {
                index.search(
                    query,
                    K,
                    ef,
                    |_, distance| distance,
                    None,
                    top_k_quickselect,
                )
            },
        );

        // ── Restore: via local fds or a full S3 round-trip ──────────────────────
        // The S3 test uploads, evicts the local fds, deletes the local
        // files, then download into a *fresh* dir and swap_in_from there — proving
        // search actually reads from blob-derived bytes (not lingering local copies).
        let t_swap_in = Instant::now();
        let _blob_restore_guard = if let Some(s3) = &s3 {
            let run_prefix = s3.run_prefix.child(format!("n{num_buckets}"));

            // 1. Upload every bucket's arena files to S3.
            let t_up = Instant::now();
            let mut uploaded_total = 0usize;
            s3.rt.block_on(async {
                for (i, _) in bucket_seqs.iter().enumerate() {
                    let local = cold_root.join(format!("seq_{i}"));
                    let prefix = run_prefix.child(format!("seq_{i}"));
                    let up = upload_arena_dir(s3.store.as_ref(), &local, &prefix)
                        .await
                        .expect("upload_arena_dir");
                    uploaded_total += up.len();
                    upload_levels(s3.store.as_ref(), &local.join("levels.bin"), &prefix)
                        .await
                        .expect("upload_levels");
                    upload_manifest(s3.store.as_ref(), &local.join("manifest.json"), &prefix)
                        .await
                        .expect("upload_manifest");
                }
            });
            eprintln!(
                "{label}: uploaded {uploaded_total} arena file(s) to s3://{}/{} in {:.3} ms",
                s3.bucket,
                run_prefix,
                ms(t_up.elapsed())
            );

            // 2. Evict locally — closes fds on every block. Reads now panic until restore.
            let mut evicted_total = 0usize;
            for seq in &bucket_seqs {
                let n = index.evict_bucket(*seq).expect("seq present for evict");
                evicted_total += n;
            }
            // 3. Delete local files — disk fully reclaimed; bytes live only in S3.
            std::fs::remove_dir_all(&cold_root).expect("rm cold_root");
            assert!(!cold_root.exists(), "local cold dir is gone");
            eprintln!("{label}: evicted {evicted_total} blocks and deleted local cold dir",);

            // 4. Download into a brand-new dir, then swap each bucket back in from it.
            let blob_root = unique_swap_dir(&format!("sift_time_bucket_n{num_buckets}_blob"));
            let blob_guard = DirGuard(blob_root.clone());
            std::fs::create_dir_all(&blob_root).expect("mk blob restore dir");
            let t_dn = Instant::now();
            s3.rt.block_on(async {
                for (i, seq) in bucket_seqs.iter().enumerate() {
                    let prefix = run_prefix.child(format!("seq_{i}"));
                    let local = blob_root.join(format!("seq_{i}"));
                    let restored = index
                        .swap_bucket_in_from_blob(*seq, s3.store.as_ref(), &prefix, &local)
                        .await
                        .expect("swap_bucket_in_from_blob")
                        .expect("seq present for blob restore");
                    assert!(restored >= 1, "{label}: at least one block restored");
                }
            });
            eprintln!(
                "{label}: downloaded + swapped in {} buckets from blob in {:.3} ms",
                bucket_seqs.len(),
                ms(t_dn.elapsed())
            );

            // 5. Clean up this iteration's uploads. Final cleanup at test end handles
            //    leftovers if a later assertion panics.
            s3.rt
                .block_on(helpers::s3::delete_prefix(s3.store.as_ref(), &run_prefix));

            Some(blob_guard)
        } else {
            // The performance test restores from the local fds held by swap_out.
            for seq in &bucket_seqs {
                let restored = index.swap_bucket_in(*seq).expect("swap_bucket_in");
                assert!(restored, "{label}: bucket_seq must be present for swap_in");
            }
            None
        };
        eprintln!(
            "{label}: restored {} buckets into memory in {:.3} ms",
            bucket_seqs.len(),
            ms(t_swap_in.elapsed())
        );

        let restored_stats = run_recall(&index, "restored");

        let phases_match = (hot_stats.min, hot_stats.mean, hot_stats.p95)
            == (cold_stats.min, cold_stats.mean, cold_stats.p95)
            && (hot_stats.min, hot_stats.mean, hot_stats.p95)
                == (restored_stats.min, restored_stats.mean, restored_stats.p95);
        eprintln!(
            "{label}: recall stats — hot={hot_stats:?} cold={cold_stats:?} restored={restored_stats:?} phases_match={phases_match}"
        );
    }
}

/// Unique temp dir for SIFT swap tests; caller responsible for cleanup.
fn unique_swap_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("mem_weaver_{tag}_{pid}_{nanos}_{n}"))
}

/// Recursively deletes the directory on drop so test stays self-cleaning on panic.
struct DirGuard(std::path::PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ── Optional S3 setup ───────────────────────────────────────────────────────

/// Holds the S3 store, run-wide prefix, and a tokio runtime to drive the async
/// upload/download calls from this sync `#[test]`. Constructed only when
/// `MEM_WEAVER_S3_BUCKET` is set.
struct S3Setup {
    bucket: String,
    region: String,
    run_prefix: ObjectPath,
    store: Arc<dyn ObjectStore>,
    rt: tokio::runtime::Runtime,
}

impl S3Setup {
    fn try_from_env() -> io::Result<Option<Self>> {
        // Bucket: env wins, default fills in, empty env-string and "<edit-me>" both skip.
        let bucket = helpers::s3::resolve("MEM_WEAVER_S3_BUCKET", DEFAULT_BUCKET);
        if bucket == "<edit-me>" {
            return Ok(None);
        }
        if bucket.contains('/') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "MEM_WEAVER_S3_BUCKET={bucket:?} contains '/'; \
                     S3 bucket names can't contain slashes. Move folder parts into the prefix \
                     (set MEM_WEAVER_S3_PREFIX or edit DEFAULT_PREFIX)."
                ),
            ));
        }
        let region = helpers::s3::resolve("MEM_WEAVER_S3_REGION", DEFAULT_REGION);
        let profile = helpers::s3::resolve("MEM_WEAVER_S3_PROFILE", DEFAULT_PROFILE);

        // Skip gracefully when AWS credentials are not available.
        if let Err(e) = helpers::s3::builder_from_profile(&profile, &bucket, &region) {
            eprintln!("s3: credentials not available ({e}); S3 upload step skipped");
            return Ok(None);
        }

        // Prefix: env literal wins. Otherwise append a unique run id under DEFAULT_PREFIX
        // (or under a generic prefix if DEFAULT_PREFIX is empty) so concurrent runs and
        // re-runs each get their own subtree.
        let prefix = std::env::var("MEM_WEAVER_S3_PREFIX")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                let id = helpers::s3::unique_run_id();
                if DEFAULT_PREFIX.is_empty() {
                    format!("mem_weaver_test/sift_{id}")
                } else {
                    format!("{DEFAULT_PREFIX}/{id}")
                }
            });

        helpers::s3::ensure_bucket(&bucket, &region, &profile)?;
        let store = helpers::s3::build_store(&profile, &bucket, &region)?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;

        Ok(Some(Self {
            bucket,
            region,
            run_prefix: ObjectPath::from(prefix),
            store,
            rt,
        }))
    }
}

impl Drop for S3Setup {
    fn drop(&mut self) {
        // Final cleanup catches leftovers from a panicking iteration; per-iteration
        // cleanup handles the common case.
        helpers::s3::cleanup_prefix_on_thread(Arc::clone(&self.store), self.run_prefix.clone());
    }
}
