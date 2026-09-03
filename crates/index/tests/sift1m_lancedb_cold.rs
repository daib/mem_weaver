//! SIFT1M cold-load query benchmark for LanceDB.
//!
//! `sift1m_lancedb_qps` (in `sift1m_lancedb_ondisk.rs`) measures query speed
//! right after building the index in the same `Connection`/`Session` that
//! wrote it — so every query is served by LanceDB's warm 6 GiB in-process
//! `GlobalIndexCache` (see `lance::session::Session::default()`), plus
//! whatever the OS page cache still holds from the just-completed writes.
//! That is a best-case number, not a cold-start one.
//!
//! This benchmark isolates the cold path in two dimensions:
//!
//! 1. **In-process index/metadata cache** — `connect()` builds a fresh
//!    `Session` (fresh `GlobalIndexCache`/`GlobalMetadataCache`) unless one
//!    is explicitly reused, so opening a brand-new `Connection` against an
//!    already-built table is guaranteed cache-cold at the LanceDB level
//!    regardless of process boundaries.
//! 2. **OS page cache** — surviving across process restarts unless
//!    explicitly dropped. This benchmark can't drop it itself (that needs
//!    root: `sudo purge` on macOS, `echo 3 > /proc/sys/vm/drop_caches` on
//!    Linux), so it's split into two tests you run as **separate `cargo
//!    test` invocations** (separate OS processes) against a
//!    non-self-deleting on-disk table, with an optional cache-drop in
//!    between for a genuinely disk-cold number.
//!
//! # Setup
//!
//! ```bash
//! export SIFT1M_BASE_PATH=/path/to/sift
//!
//! # 1. Build the on-disk table + IVF_HNSW_SQ index once. Persists to
//! #    SIFT1M_LANCEDB_COLD_DB_PATH (default: a fixed tmp dir) and does NOT
//! #    delete it afterwards.
//! cargo test -p index --test sift1m_lancedb_cold --release -- \
//!   sift1m_lancedb_cold_build --nocapture
//!
//! # 2. Optional: drop the OS page cache for a truly disk-cold run.
//! sudo purge                              # macOS
//! # echo 3 | sudo tee /proc/sys/vm/drop_caches   # Linux
//!
//! # 3. Open a fresh Connection/Session against the persisted table and
//! #    query it — first-query latency, a short warm-up curve, then the
//! #    same concurrency sweep as sift1m_lancedb_qps for comparison.
//! cargo test -p index --test sift1m_lancedb_cold --release -- \
//!   sift1m_lancedb_cold_query --nocapture
//! ```
//!
//! Running both tests in the same `cargo test` invocation still measures
//! something meaningful (cold in-process cache) but the OS page cache will
//! be warm from the build step, since it's the same machine's kernel.
//!
//! # Environment variables
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `SIFT1M_BASE_PATH` | — | Directory with `sift_base.fvecs` and `sift_query.fvecs`. **Required.** |
//! | `SIFT1M_RECALL_N_BASE` | `1_000_000` | Base vectors to index. |
//! | `SIFT1M_HNSW_EF` | `100` | Search `ef` (HNSW candidate list size). |
//! | `SIFT1M_LANCEDB_M` | `20` | HNSW `m` (`num_edges`). |
//! | `SIFT1M_LANCEDB_EF_CONSTRUCTION` | `300` | HNSW `ef_construction`. |
//! | `SIFT1M_LANCEDB_NUM_PARTITIONS` | auto (`~sqrt(n)`) | IVF partitions wrapping the HNSW graphs. |
//! | `SIFT1M_LANCEDB_BATCH_SIZE` | `100_000` | Rows per `add()` call during ingestion. |
//! | `SIFT1M_LANCEDB_COLD_DB_PATH` | `<tmp>/mw_lancedb_cold_db` | Persistent on-disk table location shared between the build and query tests. |
//! | `SIFT1M_LANCEDB_COLD_WARMUP_QUERIES` | `20` | Queries to print individually after the cold first query, to show the in-process warm-up curve. |
//! | `SIFT1M_LANCEDB_QPS_N_QUERIES` | `10_000` | Total queries for the closing warm concurrency sweep. |

use common::benchmark::latency_percentile;
use vector::read_fvecs_vector_at;

use arrow_array::types::Float32Type;
use arrow_array::{FixedSizeListArray, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::index::vector::IvfHnswSqIndexBuilder;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, DistanceType, Table};

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const K: usize = 10;
const DEFAULT_M: u32 = 20;
const DEFAULT_EF_CONSTRUCTION: u32 = 300;
const DEFAULT_SEARCH_EF: usize = 100;
const DEFAULT_NUM_BASE_VECTORS: usize = 1_000_000;
const DEFAULT_BATCH_SIZE: usize = 100_000;
const DEFAULT_NUM_QUERIES: usize = 10_000;
const DEFAULT_WARMUP_QUERIES: usize = 20;
const TABLE_NAME: &str = "sift1m_cold";
const VECTOR_COLUMN: &str = "vector";
const INDEX_NAME: &str = "vector_idx";

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn db_path() -> PathBuf {
    std::env::var("SIFT1M_LANCEDB_COLD_DB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("mw_lancedb_cold_db"))
}

struct IndexSetup {
    m: u32,
    ef_construction: u32,
    search_ef: usize,
    num_partitions: Option<u32>,
    batch_size: usize,
    n_base: usize,
}

impl IndexSetup {
    fn from_env() -> Self {
        Self {
            m: std::env::var("SIFT1M_LANCEDB_M")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_M),
            ef_construction: std::env::var("SIFT1M_LANCEDB_EF_CONSTRUCTION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_EF_CONSTRUCTION),
            search_ef: std::env::var("SIFT1M_HNSW_EF")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(DEFAULT_SEARCH_EF),
            num_partitions: std::env::var("SIFT1M_LANCEDB_NUM_PARTITIONS")
                .ok()
                .and_then(|s| s.parse().ok()),
            batch_size: std::env::var("SIFT1M_LANCEDB_BATCH_SIZE")
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

fn schema(dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            VECTOR_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

fn make_batch(schema: SchemaRef, dim: usize, base: &[Vec<f32>], start: usize, end: usize) -> RecordBatch {
    let ids: Int64Array = Int64Array::from_iter_values((start..end).map(|i| i as i64));
    let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        base[start..end]
            .iter()
            .map(|v| Some(v.iter().copied().map(Some).collect::<Vec<_>>())),
        dim as i32,
    );
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(vectors)]).expect("record batch")
}

async fn build_table(db: &Connection, corpus: &[Vec<f32>], dim: usize, setup: &IndexSetup) -> Table {
    let n_base = corpus.len();
    let sch = schema(dim);

    eprintln!(
        "sift1m_lancedb_cold [build]: inserting {n_base} vectors in batches of {}",
        setup.batch_size
    );
    let t_insert = Instant::now();

    let first_end = setup.batch_size.min(n_base);
    let first_batch = make_batch(sch.clone(), dim, corpus, 0, first_end);
    let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(vec![Ok(first_batch)], sch.clone()));
    let table = db
        .create_table(TABLE_NAME, reader)
        .execute()
        .await
        .expect("create table");

    let mut offset = first_end;
    while offset < n_base {
        let end = (offset + setup.batch_size).min(n_base);
        let batch = make_batch(sch.clone(), dim, corpus, offset, end);
        let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
            Box::new(RecordBatchIterator::new(vec![Ok(batch)], sch.clone()));
        table.add(reader).execute().await.expect("add batch");
        offset = end;
    }
    eprintln!(
        "sift1m_lancedb_cold [build]: insert done in {:.3} ms",
        ms(t_insert.elapsed())
    );

    let mut builder = IvfHnswSqIndexBuilder::default()
        .distance_type(DistanceType::L2)
        .num_edges(setup.m)
        .ef_construction(setup.ef_construction);
    if let Some(np) = setup.num_partitions {
        builder = builder.num_partitions(np);
    }

    let t_index = Instant::now();
    table
        .create_index(&[VECTOR_COLUMN], Index::IvfHnswSq(builder))
        .execute()
        .await
        .expect("create index");
    table
        .wait_for_index(&[INDEX_NAME], Duration::from_secs(600))
        .await
        .expect("wait for index");
    eprintln!(
        "sift1m_lancedb_cold [build]: index build settled in {:.3} ms",
        ms(t_index.elapsed())
    );

    table
}

async fn search_ids(table: &Table, query: &[f32], ef: usize) -> Vec<i64> {
    let batches: Vec<RecordBatch> = table
        .query()
        .nearest_to(query)
        .expect("nearest_to")
        .column(VECTOR_COLUMN)
        .distance_type(DistanceType::L2)
        .ef(ef)
        .limit(K)
        .execute()
        .await
        .expect("search")
        .try_collect()
        .await
        .expect("collect batches");

    let mut ids = Vec::with_capacity(K);
    for batch in &batches {
        let id_col = batch
            .column_by_name("id")
            .expect("id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("id column is Int64Array");
        ids.extend(id_col.iter().flatten());
    }
    ids
}

/// Builds the on-disk table once and leaves it on disk (does not clean up)
/// so `sift1m_lancedb_cold_query` can open it fresh, optionally from a
/// separate `cargo test` process after dropping the OS page cache.
#[tokio::test]
async fn sift1m_lancedb_cold_build() {
    let setup = IndexSetup::from_env();
    let n_q = 1; // build doesn't need real queries
    let Some(ctx) = common::benchmark::try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_lancedb_cold_build: SIFT1M_BASE_PATH not set — skipping");
        return;
    };
    let dim = ctx.dim;
    let corpus: Vec<Vec<f32>> = (0..setup.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();

    let path = db_path();
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("mk db dir");

    eprintln!(
        "sift1m_lancedb_cold_build: dim={dim} n_base={} m={} ef_construction={} \
         num_partitions={:?} path={}",
        setup.n_base,
        setup.m,
        setup.ef_construction,
        setup.num_partitions,
        path.display(),
    );

    let db = connect(path.to_str().unwrap()).execute().await.expect("connect");
    build_table(&db, &corpus, dim, &setup).await;

    eprintln!(
        "sift1m_lancedb_cold_build: table persisted at {} — run sift1m_lancedb_cold_query next \
         (ideally in a fresh `cargo test` invocation, optionally after dropping the OS page cache)",
        path.display()
    );
}

/// Opens a brand-new `Connection` (fresh `Session`, hence a cold in-process
/// index/metadata cache regardless of process boundaries) against the table
/// built by `sift1m_lancedb_cold_build`, then reports:
/// 1. The very first query's latency — cold cache, cold(-ish) disk.
/// 2. A short warm-up curve as the in-process cache fills in.
/// 3. A standard concurrency sweep once warm, for comparison against
///    `sift1m_lancedb_qps`.
#[tokio::test]
async fn sift1m_lancedb_cold_query() {
    let setup = IndexSetup::from_env();
    let n_q: usize = std::env::var("SIFT1M_LANCEDB_QPS_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES);
    let n_warmup: usize = std::env::var("SIFT1M_LANCEDB_COLD_WARMUP_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_WARMUP_QUERIES);

    let Some(ctx) = common::benchmark::try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_lancedb_cold_query: SIFT1M_BASE_PATH not set — skipping");
        return;
    };
    let dim = ctx.dim;

    let path = db_path();
    if !path.join(format!("{TABLE_NAME}.lance")).exists() {
        eprintln!(
            "sift1m_lancedb_cold_query: no table found at {} — run sift1m_lancedb_cold_build first",
            path.display()
        );
        return;
    }

    eprintln!(
        "sift1m_lancedb_cold_query: opening fresh Connection against {} (cold in-process cache)",
        path.display()
    );
    let t_open = Instant::now();
    let db = connect(path.to_str().unwrap()).execute().await.expect("connect");
    let table = db.open_table(TABLE_NAME).execute().await.expect("open table");
    eprintln!(
        "sift1m_lancedb_cold_query: open+attach took {:.3} ms",
        ms(t_open.elapsed())
    );

    let queries: Vec<Vec<f32>> = (0..n_q)
        .map(|qi| read_fvecs_vector_at(ctx.q_data(), dim, qi % ctx.n_q).expect("query fvecs"))
        .collect();

    let t_first = Instant::now();
    let _ = search_ids(&table, &queries[0], setup.search_ef).await;
    eprintln!(
        "sift1m_lancedb_cold_query: COLD first query latency = {:.3} ms",
        ms(t_first.elapsed())
    );

    let warmup_n = n_warmup.min(n_q.saturating_sub(1));
    eprint!("sift1m_lancedb_cold_query: warm-up curve (ms):");
    for qi in 1..=warmup_n {
        let t0 = Instant::now();
        let _ = search_ids(&table, &queries[qi], setup.search_ef).await;
        eprint!(" {:.3}", ms(t0.elapsed()));
    }
    eprintln!();

    let table = Arc::new(table);
    let queries: Arc<Vec<Vec<f32>>> = Arc::new(queries);

    for n_tasks in [1, 2, 4, 6, 8] {
        let chunk_size = (n_q + n_tasks - 1) / n_tasks;

        eprintln!(
            "sift1m_lancedb_cold_query: running {n_q} queries across {n_tasks} concurrent tasks \
             (warm, for comparison)"
        );
        let t_total = Instant::now();

        let mut handles = Vec::with_capacity(n_tasks);
        for chunk_idx in 0..n_tasks {
            let table = Arc::clone(&table);
            let queries = Arc::clone(&queries);
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
                    let _ = search_ids(&table, &queries[qi], search_ef).await;
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
            "sift1m_lancedb_cold_query: n_q={n_q} tasks={n_tasks} total={:.3}ms qps={qps:.1} \
             p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms",
            elapsed * 1e3,
        );
    }
}
