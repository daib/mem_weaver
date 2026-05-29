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
//! - `MEM_WEAVER_S3_BUCKET` — when set, the swapped-out arena files are also uploaded to S3
//!   under `s3://$BUCKET/$MEM_WEAVER_S3_PREFIX/n<num_buckets>/seq_<i>/` and the upload is
//!   verified by downloading to a fresh dir and byte-comparing. Unset → S3 step skipped.
//! - `MEM_WEAVER_S3_REGION` (default `us-east-1`), `MEM_WEAVER_S3_PROFILE` (default `default`),
//!   `MEM_WEAVER_S3_PREFIX` (default unique per run). Credentials read from `~/.aws/credentials`.

mod helpers;

use common::benchmark::{sift_recall_stats, try_load_sift_ctx};
use common::{top_k_quickselect, Timestamp};
use index::{
    blob::{upload_levels, upload_manifest},
    upload_arena_dir, TimeBucketIndex, DEFAULT_ALIGNMENT,
};
use object_store::{path::Path as ObjectPath, ObjectStore};
use rand::rngs::StdRng;
use rand::SeedableRng;
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
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_BUCKET_COUNTS: &[usize] = &[4];

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

    // Optional S3 setup: only initialized if MEM_WEAVER_S3_BUCKET is set.
    let s3 = match S3Setup::try_from_env() {
        Ok(Some(s)) => {
            eprintln!(
                "s3: bucket={} region={} run_prefix={}",
                s.bucket, s.region, s.run_prefix
            );
            Some(s)
        }
        Ok(None) => {
            eprintln!("s3: MEM_WEAVER_S3_BUCKET unset; S3 upload step skipped");
            None
        }
        Err(e) => panic!("S3 setup failed: {e}"),
    };

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

        let t_idx = Instant::now();
        let mut batch_start = Instant::now();
        let mut seen_seqs = std::collections::HashSet::new();
        for (i, v) in corpus.iter().enumerate() {
            let bid = index.insert(v.as_slice(), Timestamp(i as u64), i as u64);
            seen_seqs.insert(bid.bucket_seq);
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

        // Helper to keep the three recall passes identical.
        let run_recall = |index: &TimeBucketIndex, phase: &str| {
            let phased_label = format!("{label} [{phase}]");
            let (stats, _, _) =
                sift_recall_stats(&phased_label, &corpus, q_data, dim, n_q, ef, |q| {
                    index
                        .search(q, K, ef, |_, d| d, None, top_k_quickselect)
                        .into_iter()
                        .map(|bid| VectorId(bid.vector_id))
                        .collect()
                });
            stats
        };

        // ── Three recall passes: hot → cold → restored ─────────────────────────
        let hot_stats = run_recall(&index, "hot");

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

        // ── Restore: either via local fds (no S3) or via full S3 round-trip ─────
        // When S3 is configured, we upload, evict the local fds, delete the local
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
            // No S3 — restore from the local fds held by swap_out.
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

        // One assertion: recall is acceptable AND all three phases agree exactly.
        // (Same graph + same algorithm → only the byte source changes.)
        assert!(
            hot_stats.min >= 0.75
                && (hot_stats.min, hot_stats.mean, hot_stats.p95)
                    == (cold_stats.min, cold_stats.mean, cold_stats.p95)
                && (hot_stats.min, hot_stats.mean, hot_stats.p95)
                    == (restored_stats.min, restored_stats.mean, restored_stats.p95),
            "{label}: recall floor and hot/cold/restored equivalence — hot={hot_stats:?} cold={cold_stats:?} restored={restored_stats:?}"
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
