//! gRPC server binary for mem_weaver.
//!
//! Configuration via environment variables:
//!   LISTEN_ADDR   listen address (default: 0.0.0.0:50051)
//!   DATA_DIR      base directory for on-disk bucket files (default: ./data)
//!
//! S3 storage (all required if any S3 env var is set):
//!   S3_BUCKET     S3 bucket name
//!   S3_REGION     AWS region (default: us-east-1)
//!   S3_PROFILE    AWS credentials profile (default: default)
//!   S3_PREFIX     base S3 key prefix (default: mem-weaver)
//!
//! Collections are created at runtime via the CreateCollection RPC.

use std::sync::Arc;

use grpc::{MemWeaverServer, MemWeaverService};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;

    let data_dir = std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".into());

    let s3_bucket = std::env::var("S3_BUCKET").ok();
    let s3_region = std::env::var("S3_REGION").unwrap_or_else(|_| "us-east-1".into());
    let s3_profile = std::env::var("S3_PROFILE").unwrap_or_else(|_| "default".into());
    let s3_prefix = std::env::var("S3_PREFIX").unwrap_or_else(|_| "mem-weaver".into());

    let store = match s3_bucket {
        Some(bucket) => {
            let s = common::s3::build_store(&s3_profile, &bucket, &s3_region)
                .map_err(|e| format!("failed to build S3 client: {e}"))?;
            eprintln!("mem-weaver-server: S3 storage enabled (bucket={bucket} region={s3_region} prefix={s3_prefix})");
            Some(s as Arc<dyn object_store::ObjectStore>)
        }
        None => None,
    };

    let service = MemWeaverService::with_storage(data_dir.clone(), store, s3_prefix);

    eprintln!("mem-weaver-server: data_dir={data_dir} listening on {addr}");

    Server::builder()
        .add_service(MemWeaverServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
