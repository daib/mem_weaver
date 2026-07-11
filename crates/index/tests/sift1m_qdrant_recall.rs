//! SIFT1M recall and QPS benchmark tests comparing mem-weaver's HNSW against Qdrant.
//!
//! # Setup
//!
//! ## 1. Install Qdrant
//!
//! ```bash
//! brew install qdrant/qdrant/qdrant
//! ```
//!
//! Or set `QDRANT_BIN=/path/to/qdrant` if the binary is not on `$PATH`.
//! Tests spawn the binary automatically; no need to start it manually.
//!
//! ## 2. Download SIFT1M
//!
//! ```bash
//! wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//! tar xf sift.tar.gz          # produces sift/ with sift_base.fvecs, sift_query.fvecs
//! export SIFT1M_BASE_PATH=$PWD/sift
//! ```
//!
//! ## 3. Run the tests
//!
//! ```bash
//! # Recall test (10 queries against 10 k base vectors by default)
//! SIFT1M_BASE_PATH=/path/to/sift \
//!   cargo test -p index --test sift1m_qdrant_recall --release -- \
//!   sift1m_qdrant_recall_vs_bruteforce --nocapture
//!
//! # QPS benchmark (1 000 queries, concurrency sweep 1/2/4/8)
//! SIFT1M_BASE_PATH=/path/to/sift \
//!   cargo test -p index --test sift1m_qdrant_recall --release -- \
//!   sift1m_qdrant_qps --nocapture
//! ```
//!
//! # Environment variables
//!
//! ## Shared
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `SIFT1M_BASE_PATH` | — | Directory with `sift_base.fvecs` and `sift_query.fvecs`. **Required.** |
//! | `QDRANT_BIN` | `qdrant` | Path to local Qdrant binary. |
//! | `QDRANT_URL` | — | If set, connect to this gRPC URL instead of spawning a binary. |
//! | `QDRANT_SEARCH_THREADS` | `6` | `MAX_SEARCH_THREADS` for the spawned process. |
//! | `SIFT1M_RECALL_N_BASE` | `10_000` | Base vectors to index. |
//! | `SIFT1M_HNSW_EF` | `100` | Search `ef`. |
//! | `SIFT1M_QDRANT_M` | `16` | HNSW `m`. |
//! | `SIFT1M_QDRANT_EF_CONSTRUCTION` | `100` | HNSW `ef_construct`. |
//! | `SIFT1M_QDRANT_BATCH_SIZE` | `256` | Upsert batch size. |
//!
//! ## Recall test (`sift1m_qdrant_recall_vs_bruteforce`)
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `SIFT1M_RECALL_N_QUERIES` | `10` | Queries to evaluate. |
//! | `SIFT1M_QDRANT_COLLECTION` | `sift1m_recall_test` | Collection name. |
//!
//! ## QPS test (`sift1m_qdrant_qps`)
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `SIFT1M_QDRANT_QPS_N_QUERIES` | `1_000` | Total queries to run. |
//! | `SIFT1M_QDRANT_QPS_COLLECTION` | `sift1m_qps_test` | Collection name. |

use common::benchmark::{
    compute_recall_stats, latency_percentile, load_or_compute_ground_truth, try_load_sift_ctx,
};
use common::eval::{recall_at_k, validate_recall_score};
use qdrant_client::qdrant::PointStruct;
use qdrant_client::qdrant::{
    CollectionStatus, CreateCollectionBuilder, Distance, HnswConfigDiffBuilder, ScoredPoint,
    SearchParamsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use vector::{read_fvecs_vector_at, VectorId};

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

const DEFAULT_SEARCH_THREADS: usize = 6;
// ── Local Qdrant process ─────────────────────────────────────────────────────

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("mw_qdrant_{}_{}", tag, std::process::id()));
        std::fs::create_dir_all(&p).expect("mk tempdir");
        Self(p)
    }
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct QdrantGuard {
    child: Option<std::process::Child>,
    _data_dir: Option<TempDir>,
    pub url: String,
}

impl QdrantGuard {
    async fn connect_or_spawn() -> Option<Self> {
        if let Ok(url) = std::env::var("QDRANT_URL") {
            return Some(Self {
                child: None,
                _data_dir: None,
                url,
            });
        }

        let bin = std::env::var("QDRANT_BIN").unwrap_or_else(|_| "qdrant".to_string());
        let data_dir = TempDir::new("/tmp/data");

        let mut cmd = std::process::Command::new(&bin);
        cmd.env("QDRANT__STORAGE__PATH", data_dir.path());
        let search_threads = std::env::var("QDRANT_SEARCH_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_SEARCH_THREADS);
        cmd.env(
            "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS",
            search_threads.to_string(),
        );
        cmd.stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("sift1m_qdrant: cannot spawn '{bin}': {e} — skipping");
                return None;
            }
        };

        // Poll gRPC port until ready (up to 10 s).
        for _ in 0..50 {
            tokio::time::sleep(Duration::from_millis(200)).await;
            if tokio::net::TcpStream::connect("127.0.0.1:6334")
                .await
                .is_ok()
            {
                tokio::time::sleep(Duration::from_millis(300)).await;
                return Some(Self {
                    child: Some(child),
                    _data_dir: Some(data_dir),
                    url: "http://localhost:6334".to_string(),
                });
            }
        }
        eprintln!("sift1m_qdrant: qdrant did not start within 10 s — skipping");
        None
    }
}

impl Drop for QdrantGuard {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

const K: usize = 10;
const DEFAULT_M: u64 = 16;
const DEFAULT_EF_CONSTRUCTION: u64 = 100;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_NUM_BASE_VECTORS: usize = 1000_000;
const DEFAULT_BATCH_SIZE: usize = 256;
const DEFAULT_NUM_QUERIES: usize = 10_000;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct IndexSetup {
    url: String,
    m: u64,
    ef_construction: u64,
    search_ef: usize,
    batch_size: usize,
    n_base: usize,
}

impl IndexSetup {
    fn from_env(url: String) -> Self {
        Self {
            url,
            m: std::env::var("SIFT1M_QDRANT_M")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_M),
            ef_construction: std::env::var("SIFT1M_QDRANT_EF_CONSTRUCTION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_EF_CONSTRUCTION),
            search_ef: std::env::var("SIFT1M_HNSW_EF")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SEARCH_EF),
            batch_size: std::env::var("SIFT1M_QDRANT_BATCH_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_BATCH_SIZE),
            n_base: std::env::var("SIFT1M_RECALL_N_BASE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_NUM_BASE_VECTORS),
        }
    }
}

async fn build_collection(
    client: &Qdrant,
    collection: &str,
    corpus: &[Vec<f32>],
    setup: &IndexSetup,
) {
    let dim = corpus[0].len();
    let _ = client.delete_collection(collection).await;
    client
        .create_collection(
            CreateCollectionBuilder::new(collection)
                .vectors_config(VectorParamsBuilder::new(dim as u64, Distance::Euclid))
                .hnsw_config(
                    HnswConfigDiffBuilder::default()
                        .m(setup.m)
                        .ef_construct(setup.ef_construction),
                ),
        )
        .await
        .expect("create collection");

    let n_base = corpus.len();
    eprintln!(
        "sift1m_qdrant [{collection}]: inserting {n_base} vectors in batches of {}",
        setup.batch_size
    );
    let t_insert = Instant::now();
    let mut inserted = 0usize;
    for batch in corpus.chunks(setup.batch_size) {
        let points: Vec<PointStruct> = batch
            .iter()
            .enumerate()
            .map(|(i, v)| PointStruct::new((inserted + i) as u64, v.clone(), Payload::default()))
            .collect();
        client
            .upsert_points(UpsertPointsBuilder::new(collection, points).wait(true))
            .await
            .expect("upsert points");
        inserted += batch.len();
    }
    eprintln!(
        "sift1m_qdrant [{collection}]: upsert calls done in {:.3} ms, waiting for indexing",
        ms(t_insert.elapsed())
    );

    wait_for_indexing(client, collection, n_base, Duration::from_secs(120)).await;

    eprintln!(
        "sift1m_qdrant [{collection}]: insert + indexing done in {:.3} ms",
        ms(t_insert.elapsed())
    );
}

/// Qdrant's upsert `wait(true)` only blocks until points are written, not until
/// HNSW indexing of those points completes in the background. Poll collection
/// status/indexed count so callers can measure true "ready to search" time.
async fn wait_for_indexing(client: &Qdrant, collection: &str, n_points: usize, timeout: Duration) {
    let t_wait = Instant::now();
    loop {
        let info = client
            .collection_info(collection)
            .await
            .expect("collection info")
            .result
            .expect("collection info result");

        let indexed = info.indexed_vectors_count.unwrap_or(0);
        let status_ready = info.status == CollectionStatus::Green as i32;
        if status_ready && indexed >= n_points as u64 {
            break;
        }
        if t_wait.elapsed() > timeout {
            panic!(
                "sift1m_qdrant [{collection}]: indexing did not finish within {:?} \
                 (status={:?} indexed={indexed}/{n_points})",
                timeout, info.status,
            );
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    eprintln!(
        "sift1m_qdrant [{collection}]: indexing settled after extra {:.3} ms",
        ms(t_wait.elapsed())
    );
}

fn scored_to_ids(results: &[ScoredPoint]) -> Vec<VectorId> {
    results
        .iter()
        .filter_map(|p| p.id.as_ref())
        .map(|id| {
            VectorId(match &id.point_id_options {
                Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) => *n,
                _ => u64::MAX,
            })
        })
        .collect()
}

#[tokio::test]
async fn sift1m_qdrant_recall_vs_bruteforce() {
    let _serial = TEST_MUTEX.lock().unwrap();

    let Some(guard) = QdrantGuard::connect_or_spawn().await else {
        return;
    };
    let setup = IndexSetup::from_env(guard.url.clone());

    let n_q: usize = std::env::var("SIFT1M_RECALL_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES);
    let collection = std::env::var("SIFT1M_QDRANT_COLLECTION")
        .unwrap_or_else(|_| "sift1m_recall_test".to_owned());

    let Some(ctx) = try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_qdrant recall: SIFT1M_BASE_PATH not set — skipping");
        return;
    };

    let dim = ctx.dim;
    let corpus: Vec<Vec<f32>> = (0..setup.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();
    let ground_truth =
        load_or_compute_ground_truth(&ctx.base_dir, &corpus, ctx.q_data(), dim, n_q, K);

    eprintln!(
        "sift1m_qdrant recall: url={} collection={collection} dim={dim} n_base={} n_q={n_q} \
         k={K} m={} ef_construction={} ef_search={}",
        setup.url, setup.n_base, setup.m, setup.ef_construction, setup.search_ef,
    );

    let client = Qdrant::from_url(&setup.url).build().expect("qdrant client");
    build_collection(&client, &collection, &corpus, &setup).await;

    let mut recalls = Vec::with_capacity(n_q);
    for qi in 0..n_q {
        let q = read_fvecs_vector_at(ctx.q_data(), dim, qi).expect("query fvecs");
        let result = client
            .search_points(
                SearchPointsBuilder::new(&collection, q, K as u64).params(
                    SearchParamsBuilder::default()
                        .hnsw_ef(setup.search_ef as u64)
                        .exact(false),
                ),
            )
            .await
            .expect("search");
        let retrieved = scored_to_ids(&result.result);
        let r = recall_at_k(&retrieved, &ground_truth[qi]).expect("valid recall@k");
        validate_recall_score(r).expect("in-range score");
        recalls.push(r);
        eprintln!("sift1m_qdrant recall query {qi}: recall@{K}={r:.4}");
    }

    let stats = compute_recall_stats(&mut recalls);
    eprintln!(
        "sift1m_qdrant recall: recall@{K} min={:.3} mean={:.3} p95={:.3}",
        stats.min, stats.mean, stats.p95
    );

    let _ = client.delete_collection(&collection).await;
}

#[tokio::test]
async fn sift1m_qdrant_qps() {
    let _serial = TEST_MUTEX.lock().unwrap();

    let Some(guard) = QdrantGuard::connect_or_spawn().await else {
        return;
    };
    let setup = IndexSetup::from_env(guard.url.clone());

    let n_q: usize = std::env::var("SIFT1M_QDRANT_QPS_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES);
    let collection = std::env::var("SIFT1M_QDRANT_QPS_COLLECTION")
        .unwrap_or_else(|_| "sift1m_qps_test".to_owned());

    let Some(ctx) = try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_qdrant qps: SIFT1M_BASE_PATH not set — skipping");
        return;
    };

    let dim = ctx.dim;
    let corpus: Vec<Vec<f32>> = (0..setup.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();

    eprintln!(
        "sift1m_qdrant qps: url={} collection={collection} dim={dim} n_base={} n_q={n_q} \
         k={K} m={} ef_construction={} ef_search={}",
        setup.url, setup.n_base, setup.m, setup.ef_construction, setup.search_ef,
    );

    let client = Arc::new(Qdrant::from_url(&setup.url).build().expect("qdrant client"));
    build_collection(&client, &collection, &corpus, &setup).await;

    let queries: Arc<Vec<Vec<f32>>> = Arc::new(
        (0..n_q)
            .map(|qi| read_fvecs_vector_at(ctx.q_data(), dim, qi % ctx.n_q).expect("query fvecs"))
            .collect(),
    );

    for n_tasks in [1, 2, 4, 6] {
        let chunk_size = (n_q + n_tasks - 1) / n_tasks;

        eprintln!("sift1m_qdrant qps: running {n_q} queries across {n_tasks} concurrent tasks");
        let t_total = Instant::now();

        let mut handles = Vec::with_capacity(n_tasks);
        for chunk_idx in 0..n_tasks {
            let client = Arc::clone(&client);
            let queries = Arc::clone(&queries);
            let collection = collection.clone();
            let search_ef = setup.search_ef;
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(n_q);
            if start >= end {
                break;
            }
            handles.push(tokio::task::spawn(async move {
                let mut lats = Vec::with_capacity(end - start);
                for qi in start..end {
                    let t0 = Instant::now();
                    client
                        .search_points(
                            SearchPointsBuilder::new(&collection, queries[qi].clone(), K as u64)
                                .params(
                                    SearchParamsBuilder::default()
                                        .hnsw_ef(search_ef as u64)
                                        .exact(false),
                                ),
                        )
                        .await
                        .expect("search");
                    lats.push(ms(t0.elapsed()));
                }
                lats
            }));
        }

        let per_task_latencies: Vec<Vec<f64>> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.expect("task panicked"))
            .collect();

        let elapsed = t_total.elapsed().as_secs_f64();
        let qps = n_q as f64 / elapsed;
        let mut latencies_ms: Vec<f64> = per_task_latencies.into_iter().flatten().collect();
        let p50 = latency_percentile(&mut latencies_ms, 50.0);
        let p95 = latency_percentile(&mut latencies_ms, 95.0);
        let p99 = latency_percentile(&mut latencies_ms, 99.0);

        eprintln!(
            "sift1m_qdrant qps: n_q={n_q} tasks={n_tasks} total={:.3}ms qps={qps:.1} \
             p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms",
            elapsed * 1e3,
        );
    }

    let _ = client.delete_collection(&collection).await;
}
