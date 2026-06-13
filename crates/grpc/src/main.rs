//! gRPC server binary for mem_weaver.
//!
//! Configuration via environment variables:
//!   LISTEN_ADDR            listen address (default: 0.0.0.0:50051)
//!   DATA_DIR               base directory for on-disk bucket files (default: ./data)
//!
//! Blob storage (all required if any BLOB env var is set):
//!   BLOB_BUCKET            bucket name
//!   BLOB_REGION            region (default: us-east-1)
//!   BLOB_PROFILE           AWS credentials profile (default: default)
//!   BLOB_PREFIX            base key prefix (default: mem-weaver)
//!
//! Periodic snapshots (requires blob storage to be configured):
//!   SNAPSHOT_INTERVAL_SECS snapshot every N seconds; unset disables snapshots
//!
//! Collections are created at runtime via the CreateCollection RPC.

use std::sync::Arc;
use std::time::Duration;

use grpc::{MemWeaverServer, MemWeaverService, SnapshotConfig};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;

    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".into());

    let blob_bucket = std::env::var("BLOB_BUCKET").ok();
    let blob_region = std::env::var("BLOB_REGION").unwrap_or_else(|_| "us-east-1".into());
    let blob_profile = std::env::var("BLOB_PROFILE").unwrap_or_else(|_| "default".into());
    let blob_prefix = std::env::var("BLOB_PREFIX").unwrap_or_else(|_| "mem-weaver".into());

    let snapshot_interval: Option<Duration> = std::env::var("SNAPSHOT_INTERVAL_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .map(Duration::from_secs);

    let store = match blob_bucket {
        Some(bucket) => {
            let s = common::s3::build_store(&blob_profile, &bucket, &blob_region)
                .map_err(|e| format!("failed to build blob storage client: {e}"))?;
            eprintln!("mem-weaver-server: blob storage enabled (bucket={bucket} region={blob_region} prefix={blob_prefix})");
            Some(s as Arc<dyn object_store::ObjectStore>)
        }
        None => None,
    };

    let service = MemWeaverService::with_storage(data_dir.clone(), store, blob_prefix);

    eprintln!("mem-weaver-server: recovering from blob snapshots…");
    if let Err(e) = service.recover_from_snapshots().await {
        eprintln!("mem-weaver-server: recovery failed: {e}");
    }

    if let Some(interval) = snapshot_interval {
        match service.spawn_snapshot_task(SnapshotConfig { interval }) {
            Some(_) => eprintln!(
                "mem-weaver-server: snapshot task started (interval={}s)",
                interval.as_secs()
            ),
            None => eprintln!(
                "mem-weaver-server: SNAPSHOT_INTERVAL_SECS set but blob storage is not configured — snapshots disabled"
            ),
        }
    }

    eprintln!("mem-weaver-server: data_dir={data_dir} listening on {addr}");

    Server::builder()
        .add_service(MemWeaverServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
