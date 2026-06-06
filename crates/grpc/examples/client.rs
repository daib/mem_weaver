/// Simple client that creates a collection, inserts random vectors, and searches.
///
/// Usage: MEMWEAVER_ADDR=http://localhost:50051 cargo run --example client
use grpc::proto::{
    mem_weaver_client::MemWeaverClient, BatchInsertRequest, CreateCollectionRequest, InsertItem,
    SearchRequest,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const DIM: usize = 128;
const NUM_VECTORS: usize = 1000;
const COLLECTION: &str = "demo";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("MEMWEAVER_ADDR").unwrap_or_else(|_| "http://localhost:50051".into());

    println!("connecting to {addr}");
    let mut client = MemWeaverClient::connect(addr).await?;

    // Create collection
    client
        .create_collection(CreateCollectionRequest {
            collection: COLLECTION.into(),
            dim: DIM as u32,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            bucket_duration_secs: 3600,
        })
        .await?;
    println!("collection '{COLLECTION}' created (dim={DIM})");

    // Insert random vectors in batches of 100
    let mut rng = StdRng::seed_from_u64(42);
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    let mut inserted = 0usize;
    for batch_start in (0..NUM_VECTORS).step_by(100) {
        let batch_end = (batch_start + 100).min(NUM_VECTORS);
        let items: Vec<InsertItem> = (batch_start..batch_end)
            .map(|i| InsertItem {
                vector: (0..DIM).map(|_| rng.gen::<f32>()).collect(),
                timestamp,
                vector_id: i as u64,
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
        println!("inserted {inserted}/{NUM_VECTORS}");
    }

    // Search with a random query
    let query: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
    let resp = client
        .search(SearchRequest {
            collection: COLLECTION.into(),
            query,
            k: 10,
            ef: 64,
            time_range_start: None,
            time_range_end: None,
        })
        .await?;

    println!("\ntop-10 results:");
    for (rank, hit) in resp.into_inner().hits.iter().enumerate() {
        println!(
            "  #{}: vector_id={} distance={:.4}",
            rank + 1,
            hit.vector_id,
            hit.distance
        );
    }

    Ok(())
}
