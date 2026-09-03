//! SIFT1M cold-load benchmark for mem_weaver's `HnswArena`, mirroring
//! `sift1m_lancedb_cold.rs` so the two can be compared directly.
//!
//! `sift1m_hnsw_qps_disk` (in `sift1m_hnsw_recall.rs`) swaps out and swaps back in
//! within the *same process*, so its `swap_in()` timing is served entirely from the
//! OS page cache the just-completed `swap_out()` populated — not a real disk read.
//!
//! This benchmark splits build and load into two separate `cargo test` invocations
//! (separate OS processes) against a persistent, non-self-deleting directory, with
//! an optional page-cache drop in between for a genuinely disk-cold number.
//!
//! # Setup
//!
//! ```bash
//! export SIFT1M_BASE_PATH=/path/to/sift
//!
//! # 1. Build the index and swap every block to disk. Persists to
//! #    SIFT1M_HNSW_COLD_DIR (default: a fixed tmp dir) and does NOT delete it.
//! cargo test -p index --test sift1m_hnsw_cold --release -- \
//!   sift1m_hnsw_cold_build --nocapture
//!
//! # 2. Optional: drop the OS page cache for a truly disk-cold run.
//! sudo purge                              # macOS
//! # echo 3 | sudo tee /proc/sys/vm/drop_caches   # Linux
//!
//! # 3. In a fresh process, reconstruct the index from the persisted directory —
//! #    load_blocks_from_dir + load_levels + rebuild_lens + load_manifest — then
//! #    query it: first-query latency, a short warm-up curve, then a concurrency
//! #    sweep for comparison against `sift1m_hnsw_qps_disk` / `sift1m_lancedb_cold_query`.
//! cargo test -p index --test sift1m_hnsw_cold --release -- \
//!   sift1m_hnsw_cold_query --nocapture
//! ```
//!
//! # Environment variables
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `SIFT1M_BASE_PATH` | — | Directory with `sift_base.fvecs` and `sift_query.fvecs`. **Required.** |
//! | `SIFT1M_RECALL_N_BASE` | `1_000_000` | Base vectors to index. |
//! | `SIFT1M_HNSW_EF` | `100` | Search `ef` (HNSW candidate list size). |
//! | `SIFT1M_HNSW_COLD_M` | `16` | HNSW `m` (max degree above level 0). |
//! | `SIFT1M_HNSW_COLD_M_MAX0` | `32` | HNSW `m_max0` (max degree at level 0). |
//! | `SIFT1M_HNSW_COLD_EF_CONSTRUCTION` | `100` | HNSW `ef_construction`. |
//! | `SIFT1M_HNSW_COLD_DIR` | `<tmp>/mw_hnsw_cold_dir` | Persistent on-disk location shared between the build and query tests. |
//! | `SIFT1M_HNSW_COLD_WARMUP_QUERIES` | `20` | Queries to print individually after the cold first query. |
//! | `SIFT1M_HNSW_QPS_N_QUERIES` | `10_000` | Total queries for the closing warm concurrency sweep. |

use common::benchmark::try_load_sift_ctx;
use crc32fast::Hasher as Crc32Hasher;
use index::{HnswArena, HnswIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;
use vector::read_fvecs_vector_at;

mod helpers;

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const K: usize = 10;
const DEFAULT_M: usize = 16;
const DEFAULT_M_MAX0: usize = 32;
const DEFAULT_EF_CONSTRUCTION: usize = 100;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_NUM_BASE_VECTORS: usize = 1_000_000;
const DEFAULT_NUM_QUERIES: usize = 10_000;
const DEFAULT_WARMUP_QUERIES: usize = 20;
const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn cold_dir() -> PathBuf {
    std::env::var("SIFT1M_HNSW_COLD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mw_hnsw_cold_dir"))
}

struct Setup {
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    search_ef: usize,
    n_base: usize,
}

impl Setup {
    fn from_env() -> Self {
        Self {
            m: std::env::var("SIFT1M_HNSW_COLD_M")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_M),
            m_max0: std::env::var("SIFT1M_HNSW_COLD_M_MAX0")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_M_MAX0),
            ef_construction: std::env::var("SIFT1M_HNSW_COLD_EF_CONSTRUCTION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_EF_CONSTRUCTION),
            search_ef: std::env::var("SIFT1M_HNSW_EF")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SEARCH_EF),
            n_base: std::env::var("SIFT1M_RECALL_N_BASE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_NUM_BASE_VECTORS),
        }
    }
}

/// Builds the index, swaps every block to disk, and persists levels/manifest — all
/// under a fixed, non-self-deleting directory — so `sift1m_hnsw_cold_query` can
/// reconstruct it from scratch in a fresh process.
#[test]
fn sift1m_hnsw_cold_build() {
    let setup = Setup::from_env();
    let n_q = 1; // build doesn't need real queries
    let Some(ctx) = try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_hnsw_cold_build: SIFT1M_BASE_PATH not set — skipping");
        return;
    };
    let dim = ctx.dim;
    let n_base = ctx.n_base;

    let dir = cold_dir();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("mk cold dir");

    eprintln!(
        "sift1m_hnsw_cold_build: dim={dim} n_base={n_base} m={} m_max0={} ef_construction={} \
         path={}",
        setup.m,
        setup.m_max0,
        setup.ef_construction,
        dir.display(),
    );

    let corpus: Vec<Vec<f32>> = (0..n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();
    let vector_ids: Vec<u64> = (0..n_base as u64).collect();

    let mut index = HnswArena::new(
        dim,
        setup.m,
        setup.m_max0,
        setup.ef_construction,
        n_base,
        StdRng::seed_from_u64(RNG_SEED),
    );

    helpers::sift::insert_in_batches("sift1m_hnsw_cold_build", n_base, 10_000, |start, end| {
        index.insert_batch_parallel(&corpus[start..end], &vector_ids[start..end], 6);
    });

    let t_persist = Instant::now();
    let moved = index.swap_out(&dir).expect("swap_out to disk");
    index
        .save_levels(&dir.join("levels.bin"))
        .expect("save levels");
    index
        .save_manifest(&dir.join("manifest.json"))
        .expect("save manifest");
    eprintln!(
        "sift1m_hnsw_cold_build: persisted {moved} blocks + levels + manifest in {:.3} ms",
        ms(t_persist.elapsed())
    );

    eprintln!(
        "sift1m_hnsw_cold_build: index persisted at {} — run sift1m_hnsw_cold_query next \
         (ideally in a fresh `cargo test` invocation, optionally after dropping the OS page cache)",
        dir.display()
    );
}

/// Reads every persisted block file into a plain heap `Vec<u8>` via `std::fs::read` —
/// no mmap allocation, no CRC — to isolate raw disk I/O throughput from the other costs
/// baked into `sift1m_hnsw_cold_query`'s reconstruction number. Run this in a fresh
/// process, ideally right after dropping the OS page cache, for a genuine disk-speed
/// number; run it a second time immediately after (still warm) to see the page-cache
/// floor for comparison.
#[test]
fn sift1m_hnsw_cold_raw_io() {
    let dir = cold_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!(
            "sift1m_hnsw_cold_raw_io: no persisted index found at {} — run sift1m_hnsw_cold_build \
             first",
            dir.display()
        );
        return;
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read cold dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("arena"))
        .collect();
    entries.sort();

    let t_read = Instant::now();
    let mut total_bytes = 0usize;
    for path in &entries {
        let bytes = std::fs::read(path).expect("read block file");
        total_bytes += bytes.len();
        std::hint::black_box(&bytes);
    }
    let elapsed = t_read.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "sift1m_hnsw_cold_raw_io: read {} blocks / {:.1} MB via std::fs::read (no mmap, no CRC) \
         in {:.3} ms ({:.1} MB/s)",
        entries.len(),
        mb,
        ms(elapsed),
        mb / (ms(elapsed) / 1000.0),
    );
}

/// Consolidates every persisted block file into one file (writing it once, if not
/// already present) and times reading it back via a *single* `open()` + sequential
/// `read_exact` calls — no per-block open/close. Compare against
/// `sift1m_hnsw_cold_raw_io`'s 308-separate-files number (same total bytes) to isolate
/// whether per-file syscall overhead (as opposed to raw throughput) is contributing to
/// the reconstruction cost. Run in a fresh process, ideally right after dropping the OS
/// page cache.
#[test]
fn sift1m_hnsw_cold_raw_io_consolidated() {
    let dir = cold_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!(
            "sift1m_hnsw_cold_raw_io_consolidated: no persisted index found at {} — run \
             sift1m_hnsw_cold_build first",
            dir.display()
        );
        return;
    }

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read cold dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("arena"))
        .collect();
    entries.sort();

    let consolidated_path = dir.join("consolidated.blob");
    if !consolidated_path.exists() {
        eprintln!(
            "sift1m_hnsw_cold_raw_io_consolidated: writing {} (one-time setup, not timed) — \
             re-run after this to get a cold number",
            consolidated_path.display()
        );
        let mut out = std::fs::File::create(&consolidated_path).expect("create consolidated");
        for path in &entries {
            let bytes = std::fs::read(path).expect("read block file");
            out.write_all(&bytes).expect("write consolidated");
        }
        out.sync_all().expect("sync consolidated");
        return;
    }

    let t_read = Instant::now();
    let mut file = std::fs::File::open(&consolidated_path).expect("open consolidated");
    let total_bytes = file.metadata().expect("stat consolidated").len() as usize;
    let mut buf = vec![0u8; total_bytes];
    file.read_exact(&mut buf).expect("read consolidated");
    std::hint::black_box(&buf);
    let elapsed = t_read.elapsed();
    let mb = total_bytes as f64 / (1024.0 * 1024.0);
    eprintln!(
        "sift1m_hnsw_cold_raw_io_consolidated: read {:.1} MB from 1 file (single open, no CRC) \
         in {:.3} ms ({:.1} MB/s)",
        mb,
        ms(elapsed),
        mb / (ms(elapsed) / 1000.0),
    );
}

/// Reconstructs the index from the directory `sift1m_hnsw_cold_build` persisted, in a
/// fresh process (guaranteed cold in-process state), then reports:
/// 1. Reconstruction time — load_blocks_from_dir + load_levels + rebuild_lens + load_manifest.
/// 2. The very first query's latency — cold cache, cold(-ish) disk.
/// 3. A short warm-up curve as the OS page cache fills in.
/// 4. A standard concurrency sweep once warm, for comparison against
///    `sift1m_hnsw_qps_disk` / `sift1m_lancedb_cold_query`.
#[test]
fn sift1m_hnsw_cold_query() {
    let setup = Setup::from_env();
    let n_q: usize = std::env::var("SIFT1M_HNSW_QPS_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES);
    let n_warmup: usize = std::env::var("SIFT1M_HNSW_COLD_WARMUP_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP_QUERIES);

    let Some(ctx) = try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_hnsw_cold_query: SIFT1M_BASE_PATH not set — skipping");
        return;
    };
    let dim = ctx.dim;
    let ef = setup.search_ef.max(K);

    let dir = cold_dir();
    if !dir.join("manifest.json").exists() {
        eprintln!(
            "sift1m_hnsw_cold_query: no persisted index found at {} — run sift1m_hnsw_cold_build \
             first",
            dir.display()
        );
        return;
    }

    eprintln!(
        "sift1m_hnsw_cold_query: reconstructing from {} (fresh process)",
        dir.display()
    );

    let mut index = HnswArena::new(
        dim,
        setup.m,
        setup.m_max0,
        setup.ef_construction,
        setup.n_base,
        StdRng::seed_from_u64(RNG_SEED),
    );

    let t_load = Instant::now();
    let restored = index.load_blocks_from_dir(&dir).expect("load blocks");
    index
        .load_levels(&dir.join("levels.bin"))
        .expect("load levels");
    index.rebuild_lens();
    index
        .load_manifest(&dir.join("manifest.json"))
        .expect("load manifest");
    eprintln!(
        "sift1m_hnsw_cold_query: reconstructed {restored} blocks ({} vectors) in {:.3} ms \
         ({:.3} ms/1k vectors)",
        index.len(),
        ms(t_load.elapsed()),
        ms(t_load.elapsed()) / (setup.n_base as f64 / 1000.0),
    );

    // Isolate CRC32's contribution to the reconstruction number above: `load_blocks_from_dir`
    // reads each block via `swap_in_from`, which reads and verifies a CRC32 over the block's
    // bytes. Re-read the same files (now warm, so I/O cost here is negligible) and time just
    // the hashing, separately from the read, to see how much of the 389ms-class number above
    // is CRC compute vs disk I/O.
    let block_paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read cold dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("arena"))
        .collect();
    let bufs: Vec<Vec<u8>> = block_paths
        .iter()
        .map(|p| std::fs::read(p).expect("re-read block (warm)"))
        .collect();
    let total_bytes: usize = bufs.iter().map(|b| b.len()).sum();
    let t_crc = Instant::now();
    for buf in &bufs {
        let payload_len = buf.len().saturating_sub(4);
        let mut hasher = Crc32Hasher::new();
        hasher.update(&buf[..payload_len]);
        let _ = hasher.finalize();
    }
    eprintln!(
        "sift1m_hnsw_cold_query: isolated CRC32 compute over {} blocks / {:.1} MB took {:.3} ms \
         (warm re-read, CPU-bound only)",
        bufs.len(),
        total_bytes as f64 / (1024.0 * 1024.0),
        ms(t_crc.elapsed()),
    );
    drop(bufs);

    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|qi| read_fvecs_vector_at(ctx.q_data(), dim, qi % ctx.n_q).expect("query fvecs"))
        .collect();

    let t_first = Instant::now();
    let _ = index.search(&queries[0], K, ef);
    eprintln!(
        "sift1m_hnsw_cold_query: COLD first query latency = {:.3} ms",
        ms(t_first.elapsed())
    );

    let n_warmup = n_warmup.min(n_q.saturating_sub(1));
    eprint!("sift1m_hnsw_cold_query: warm-up curve (ms):");
    for q in &queries[1..=n_warmup] {
        let t0 = Instant::now();
        let _ = index.search(q, K, ef);
        eprint!(" {:.3}", ms(t0.elapsed()));
    }
    eprintln!();

    helpers::sift::measure_qps("sift1m_hnsw_cold_query", &queries, &[1, 2, 4, 6], |query| {
        index.search(query, K, ef)
    });
}
