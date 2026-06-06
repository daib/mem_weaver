//! Recall@k of the gRPC server vs exhaustive L2 on a SIFT1M prefix.
//!
//! Mirrors `sift1m_hnsw_recall_vs_bruteforce` and
//! `sift1m_time_bucket_recall_vs_bruteforce` from the `index` crate, but
//! exercises the full gRPC stack (server + CreateCollection + BatchInsert + Search).
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory with `sift_base.fvecs` and `sift_query.fvecs`
//!   (Texmex layout). If unset the tests return immediately.
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors (default `8192`).
//! - `SIFT1M_RECALL_N_QUERIES` — number of queries (default `10`).
//! - `SIFT1M_HNSW_EF` — search ef (default `100`).

use std::time::{Duration, Instant};

use common::benchmark::{sift_recall_stats, try_load_sift_ctx};
use grpc::proto::mem_weaver_client::MemWeaverClient;
use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem, SearchRequest};
use grpc::{MemWeaverServer, MemWeaverService};
use tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use vector::{read_fvecs_vector_at, VectorId};

const K: usize = 10;
const M: usize = 16;
const M_MAX0: usize = 32;
const DEFAULT_NUM_BASE_VECTORS: usize = 8_192;
const DEFAULT_NUM_QUERIES: usize = 10;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const DEFAULT_SEARCH_EF: usize = 100;
const BATCH_SIZE: usize = 1_000;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Bind a random local port, spawn the gRPC server in the background, and
/// return a connected client.
async fn start_server() -> MemWeaverClient<tonic::transport::Channel> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let service = MemWeaverService::new();
    tokio::spawn(async move {
        Server::builder()
            .add_service(MemWeaverServer::new(service))
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let endpoint = format!("http://{addr}");
    for _ in 0..20 {
        if let Ok(c) = MemWeaverClient::connect(endpoint.clone()).await {
            return c;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    MemWeaverClient::connect(endpoint).await.unwrap()
}

/// Insert `corpus` into the server via BatchInsert, sending `BATCH_SIZE`
/// vectors per RPC. `timestamp_fn(i)` is called for each base vector index.
async fn insert_corpus(
    client: &mut MemWeaverClient<tonic::transport::Channel>,
    collection: &str,
    corpus: &[Vec<f32>],
    timestamp_fn: impl Fn(usize) -> u64,
) {
    for (chunk_start, chunk) in corpus.chunks(BATCH_SIZE).enumerate() {
        let base = chunk_start * BATCH_SIZE;
        let items = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| InsertItem {
                vector: v.clone(),
                timestamp: timestamp_fn(base + j),
                vector_id: (base + j) as u64,
            })
            .collect();
        client
            .batch_insert(BatchInsertRequest {
                collection: collection.to_string(),
                items,
            })
            .await
            .expect("batch_insert");
    }
}

/// Run all queries against the server and collect results in order.
async fn collect_search_results(
    client: &mut MemWeaverClient<tonic::transport::Channel>,
    collection: &str,
    q_data: &[u8],
    dim: usize,
    n_q: usize,
    ef: usize,
) -> Vec<Vec<VectorId>> {
    let mut results = Vec::with_capacity(n_q);
    for qi in 0..n_q {
        let q = read_fvecs_vector_at(q_data, dim, qi).expect("query fvecs");
        let resp = client
            .search(SearchRequest {
                collection: collection.to_string(),
                query: q,
                k: K as u32,
                ef: ef as u32,
                time_range_start: None,
                time_range_end: None,
            })
            .await
            .expect("search");
        let hits: Vec<VectorId> = resp
            .into_inner()
            .hits
            .into_iter()
            .map(|h| VectorId(h.vector_id))
            .collect();
        results.push(hits);
    }
    results
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sift1m_hnsw_recall_vs_bruteforce() {
    let t_load = Instant::now();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };
    eprintln!(
        "grpc/sift_hnsw: load context {:.3} ms",
        ms(t_load.elapsed())
    );

    let corpus: Vec<Vec<f32>> = (0..ctx.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), ctx.dim, i).expect("fvecs"))
        .collect();
    let ef = ctx.search_ef.max(K);

    let mut client = start_server().await;

    client
        .create_collection(CreateCollectionRequest {
            collection: "test".to_string(),
            dim: ctx.dim as u32,
            m: M as u32,
            m_max0: M_MAX0 as u32,
            ef_construction: DEFAULT_EF_CONSTRUCTION as u32,
            // Single bucket: duration=0 means u64::MAX/2 (see lib.rs)
            bucket_duration_secs: 0,
        })
        .await
        .expect("create_collection");

    let t_insert = Instant::now();
    insert_corpus(&mut client, "test", &corpus, |_| 0).await;
    eprintln!(
        "grpc/sift_hnsw: inserted {} vectors in {:.3} ms",
        ctx.n_base,
        ms(t_insert.elapsed())
    );

    let t_search = Instant::now();
    let search_results =
        collect_search_results(&mut client, "test", ctx.q_data(), ctx.dim, ctx.n_q, ef).await;
    eprintln!(
        "grpc/sift_hnsw: {} queries in {:.3} ms",
        ctx.n_q,
        ms(t_search.elapsed())
    );

    let mut result_iter = search_results.into_iter();
    let (stats, _, _) = sift_recall_stats(
        "grpc/sift_hnsw",
        &corpus,
        ctx.q_data(),
        ctx.dim,
        ctx.n_q,
        ef,
        |_q| result_iter.next().unwrap(),
    );
    assert!(
        stats.min >= 0.75,
        "grpc/sift_hnsw: recall@{K} min={} expected >= 0.75",
        stats.min
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sift1m_time_bucket_recall_vs_bruteforce() {
    let t_load = Instant::now();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };
    eprintln!(
        "grpc/sift_time_bucket: load context {:.3} ms",
        ms(t_load.elapsed())
    );

    let corpus: Vec<Vec<f32>> = (0..ctx.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), ctx.dim, i).expect("fvecs"))
        .collect();
    let ef = ctx.search_ef.max(K);

    // ~4 buckets: each bucket covers n_base/4 seconds.
    let num_buckets = 4usize;
    let bucket_duration_secs = (ctx.n_base as u64 / num_buckets as u64).max(1);

    let mut client = start_server().await;

    client
        .create_collection(CreateCollectionRequest {
            collection: "test".to_string(),
            dim: ctx.dim as u32,
            m: M as u32,
            m_max0: M_MAX0 as u32,
            ef_construction: DEFAULT_EF_CONSTRUCTION as u32,
            bucket_duration_secs,
        })
        .await
        .expect("create_collection");

    let t_insert = Instant::now();
    // Each vector gets timestamp = i so they spread across ~4 buckets.
    insert_corpus(&mut client, "test", &corpus, |i| i as u64).await;
    eprintln!(
        "grpc/sift_time_bucket: inserted {} vectors ({num_buckets} buckets) in {:.3} ms",
        ctx.n_base,
        ms(t_insert.elapsed())
    );

    let t_search = Instant::now();
    let search_results =
        collect_search_results(&mut client, "test", ctx.q_data(), ctx.dim, ctx.n_q, ef).await;
    eprintln!(
        "grpc/sift_time_bucket: {} queries in {:.3} ms",
        ctx.n_q,
        ms(t_search.elapsed())
    );

    let label = format!("grpc/sift_time_bucket(n={num_buckets})");
    let mut result_iter = search_results.into_iter();
    let (stats, _, _) =
        sift_recall_stats(&label, &corpus, ctx.q_data(), ctx.dim, ctx.n_q, ef, |_q| {
            result_iter.next().unwrap()
        });
    assert!(
        stats.min >= 0.75,
        "{label}: recall@{K} min={} expected >= 0.75",
        stats.min
    );
}
