/// SIFT1M recall benchmark against a running mem-weaver gRPC server.
///
/// Environment variables:
///   MEMWEAVER_ADDR      server address (default: http://localhost:50051)
///   SIFT1M_BASE_PATH    directory containing sift_base.fvecs and sift_query.fvecs
///   SIFT1M_N_BASE       number of base vectors to insert (default: 100_000)
///   SIFT1M_N_QUERIES    number of queries to run     (default: 100)
///   SIFT1M_EF           search ef parameter          (default: 100)
use std::fs::File;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use common::{fvecs_vector_count, read_fvecs_dim_le, read_fvecs_vector_at};
use grpc::proto::{
    mem_weaver_client::MemWeaverClient, BatchInsertRequest, CreateCollectionRequest, InsertItem,
    SearchRequest,
};
use memmap2::Mmap;

const SIFT_DIM: usize = 128;
const K: u32 = 10;
const M: u32 = 16;
const M_MAX0: u32 = 32;
const EF_CONSTRUCTION: u32 = 200;
const BATCH_SIZE: usize = 1_000;
const COLLECTION: &str = "sift1m";

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn open_mmap(path: &PathBuf) -> Mmap {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    unsafe { Mmap::map(&file) }.unwrap_or_else(|e| panic!("mmap {}: {e}", path.display()))
}

fn euclidean_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn brute_force_topk(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, euclidean_sq(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("MEMWEAVER_ADDR").unwrap_or_else(|_| "http://localhost:50051".into());
    let base_dir = PathBuf::from(std::env::var("SIFT1M_BASE_PATH").expect(
        "SIFT1M_BASE_PATH must be set to the directory with sift_base.fvecs / sift_query.fvecs",
    ));
    let n_base: usize = std::env::var("SIFT1M_N_BASE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);
    let n_queries: usize = std::env::var("SIFT1M_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let ef: u32 = std::env::var("SIFT1M_EF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    // Load mmaps
    let base_mmap = open_mmap(&base_dir.join("sift_base.fvecs"));
    let query_mmap = open_mmap(&base_dir.join("sift_query.fvecs"));

    let dim = read_fvecs_dim_le(&base_mmap, 0).expect("empty sift_base.fvecs");
    assert_eq!(dim, SIFT_DIM, "unexpected dim {dim}");

    let avail_base = fvecs_vector_count(&base_mmap, dim);
    let n_base = n_base.min(avail_base);
    let n_queries = n_queries.min(fvecs_vector_count(&query_mmap, dim));
    println!("sift1m: dim={dim} base={n_base} queries={n_queries} ef={ef}");

    // Pre-load corpus into memory for brute-force ground truth
    let t = Instant::now();
    let corpus: Vec<Vec<f32>> = (0..n_base)
        .map(|i| read_fvecs_vector_at(&base_mmap, dim, i).expect("base fvecs"))
        .collect();
    println!("loaded {n_base} base vectors in {:.1} ms", ms(t.elapsed()));

    // Connect and create collection
    println!("connecting to {addr}");
    let mut client = MemWeaverClient::connect(addr).await?;

    client
        .create_collection(CreateCollectionRequest {
            collection: COLLECTION.into(),
            dim: dim as u32,
            m: M,
            m_max0: M_MAX0,
            ef_construction: EF_CONSTRUCTION,
            bucket_duration_secs: 0, // single bucket
        })
        .await?;
    println!("collection '{COLLECTION}' created");

    // Insert base vectors
    let t_insert = Instant::now();
    let mut inserted = 0usize;
    for (chunk_idx, chunk) in corpus.chunks(BATCH_SIZE).enumerate() {
        let base = chunk_idx * BATCH_SIZE;
        let items: Vec<InsertItem> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| InsertItem {
                vector: v.clone(),
                timestamp: 0,
                vector_id: (base + j) as u64,
            })
            .collect();
        let count = items.len();
        client
            .batch_insert(BatchInsertRequest {
                collection: COLLECTION.into(),
                items,
            })
            .await?;
        inserted += count;
        if inserted % 10_000 == 0 || inserted == n_base {
            println!("  inserted {inserted}/{n_base}");
        }
    }
    let insert_ms = ms(t_insert.elapsed());
    println!(
        "insert done: {n_base} vectors in {insert_ms:.1} ms ({:.0} vec/s)",
        n_base as f64 / (insert_ms / 1e3)
    );

    // Search and compute recall@K
    let t_search = Instant::now();
    let mut recall_sum = 0.0f32;
    let mut recall_min = 1.0f32;

    for qi in 0..n_queries {
        let query = read_fvecs_vector_at(&query_mmap, dim, qi).expect("query fvecs");

        let gt: Vec<u64> = brute_force_topk(&query, &corpus, K as usize);

        let resp = client
            .search(SearchRequest {
                collection: COLLECTION.into(),
                query: query.clone(),
                k: K,
                ef,
                time_range_start: None,
                time_range_end: None,
            })
            .await?;

        let hits: Vec<u64> = resp
            .into_inner()
            .hits
            .into_iter()
            .map(|h| h.vector_id)
            .collect();
        let overlap = hits.iter().filter(|id| gt.contains(id)).count();
        let recall = overlap as f32 / K as f32;
        recall_sum += recall;
        if recall < recall_min {
            recall_min = recall;
        }
    }

    let search_ms = ms(t_search.elapsed());
    let recall_mean = recall_sum / n_queries as f32;
    println!(
        "search done: {n_queries} queries in {search_ms:.1} ms ({:.2} ms/query)",
        search_ms / n_queries as f64
    );
    println!("recall@{K}: mean={recall_mean:.4} min={recall_min:.4}");

    Ok(())
}
