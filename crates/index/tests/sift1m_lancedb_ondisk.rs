//! SIFT1M on-disk ingestion and query-speed benchmark for LanceDB.
//!
//! Unlike the Qdrant benchmark, LanceDB is an embedded, on-disk library —
//! there is no server to spawn. Data is written straight to a temp
//! directory on local disk and read back through mmap'd Lance files.
//!
//! # Setup
//!
//! ## 1. Download SIFT1M
//!
//! ```bash
//! wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
//! tar xf sift.tar.gz          # produces sift/ with sift_base.fvecs, sift_query.fvecs
//! export SIFT1M_BASE_PATH=$PWD/sift
//! ```
//!
//! ## 2. Run the tests
//!
//! ```bash
//! # Ingestion benchmark: insert + IVF_HNSW_SQ index build time, on-disk size
//! SIFT1M_BASE_PATH=/path/to/sift \
//!   cargo test -p index --test sift1m_lancedb_ondisk --release -- \
//!   sift1m_lancedb_ingest --nocapture
//!
//! # QPS benchmark (concurrency sweep 1/2/4/6/8)
//! SIFT1M_BASE_PATH=/path/to/sift \
//!   cargo test -p index --test sift1m_lancedb_ondisk --release -- \
//!   sift1m_lancedb_qps --nocapture
//! ```
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
//! | `SIFT1M_LANCEDB_QPS_N_QUERIES` | `10_000` | Total queries for the QPS sweep. |

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
const VECTOR_COLUMN: &str = "vector";
const INDEX_NAME: &str = "vector_idx";

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("mw_lancedb_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
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

fn dir_size_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            if meta.is_dir() {
                stack.push(entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
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

/// Loads the SIFT1M base vectors, ingests them into a fresh on-disk LanceDB
/// table in `setup.batch_size`-row chunks, builds an IVF_HNSW_SQ index, and
/// waits for indexing to finish. Returns the table plus phase timings.
async fn build_table(
    db: &Connection,
    table_name: &str,
    corpus: &[Vec<f32>],
    dim: usize,
    setup: &IndexSetup,
) -> (Table, Duration, Duration) {
    let n_base = corpus.len();
    let sch = schema(dim);

    eprintln!(
        "sift1m_lancedb [{table_name}]: inserting {n_base} vectors in batches of {}",
        setup.batch_size
    );
    let t_insert = Instant::now();

    let first_end = setup.batch_size.min(n_base);
    let first_batch = make_batch(sch.clone(), dim, corpus, 0, first_end);
    let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
        Box::new(RecordBatchIterator::new(vec![Ok(first_batch)], sch.clone()));
    let table = db
        .create_table(table_name, reader)
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
    let insert_elapsed = t_insert.elapsed();
    eprintln!(
        "sift1m_lancedb [{table_name}]: insert done in {:.3} ms",
        ms(insert_elapsed)
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
    let index_elapsed = t_index.elapsed();
    eprintln!(
        "sift1m_lancedb [{table_name}]: index build settled in {:.3} ms",
        ms(index_elapsed)
    );

    (table, insert_elapsed, index_elapsed)
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

#[tokio::test]
async fn sift1m_lancedb_ingest() {
    let setup = IndexSetup::from_env();
    let n_q = 1; // ingestion benchmark doesn't need real queries
    let Some(ctx) = common::benchmark::try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_lancedb ingest: SIFT1M_BASE_PATH not set — skipping");
        return;
    };
    let dim = ctx.dim;
    let corpus: Vec<Vec<f32>> = (0..setup.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();

    let data_dir = TempDir::new("ingest");
    let db = connect(data_dir.path().to_str().unwrap())
        .execute()
        .await
        .expect("connect");

    eprintln!(
        "sift1m_lancedb ingest: dim={dim} n_base={} m={} ef_construction={} num_partitions={:?}",
        setup.n_base, setup.m, setup.ef_construction, setup.num_partitions,
    );

    let (_, insert_elapsed, index_elapsed) =
        build_table(&db, "sift1m_ingest", &corpus, dim, &setup).await;

    let total = insert_elapsed + index_elapsed;
    let bytes = dir_size_bytes(data_dir.path());
    let n_base = setup.n_base as f64;

    eprintln!(
        "sift1m_lancedb ingest: n_base={} insert={:.3}ms ({:.3} ms/vec, {:.1} vec/s) \
         index_build={:.3}ms total={:.3}ms on_disk={} bytes ({:.1} bytes/vec)",
        setup.n_base,
        ms(insert_elapsed),
        ms(insert_elapsed) / n_base,
        n_base / insert_elapsed.as_secs_f64(),
        ms(index_elapsed),
        ms(total),
        bytes,
        bytes as f64 / n_base,
    );
}

#[tokio::test]
async fn sift1m_lancedb_qps() {
    let setup = IndexSetup::from_env();
    let n_q: usize = std::env::var("SIFT1M_LANCEDB_QPS_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_QUERIES);

    let Some(ctx) = common::benchmark::try_load_sift_ctx(setup.n_base, n_q, setup.search_ef) else {
        eprintln!("sift1m_lancedb qps: SIFT1M_BASE_PATH not set — skipping");
        return;
    };
    let dim = ctx.dim;
    let corpus: Vec<Vec<f32>> = (0..setup.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), dim, i).expect("base fvecs"))
        .collect();

    let data_dir = TempDir::new("qps");
    let db = connect(data_dir.path().to_str().unwrap())
        .execute()
        .await
        .expect("connect");

    eprintln!(
        "sift1m_lancedb qps: dim={dim} n_base={} n_q={n_q} k={K} m={} ef_construction={} \
         ef_search={}",
        setup.n_base, setup.m, setup.ef_construction, setup.search_ef,
    );

    let (table, _, _) = build_table(&db, "sift1m_qps", &corpus, dim, &setup).await;

    let queries: Arc<Vec<Vec<f32>>> = Arc::new(
        (0..n_q)
            .map(|qi| read_fvecs_vector_at(ctx.q_data(), dim, qi % ctx.n_q).expect("query fvecs"))
            .collect(),
    );

    for n_tasks in [1, 2, 4, 6, 8] {
        let chunk_size = (n_q + n_tasks - 1) / n_tasks;

        eprintln!("sift1m_lancedb qps: running {n_q} queries across {n_tasks} concurrent tasks");
        let t_total = Instant::now();

        let mut handles = Vec::with_capacity(n_tasks);
        for chunk_idx in 0..n_tasks {
            let table = table.clone();
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
            "sift1m_lancedb qps: n_q={n_q} tasks={n_tasks} total={:.3}ms qps={qps:.1} \
             p50={p50:.3}ms p95={p95:.3}ms p99={p99:.3}ms",
            elapsed * 1e3,
        );
    }
}
