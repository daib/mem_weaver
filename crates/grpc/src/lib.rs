//! gRPC service wrapping [`index::TimeBucketIndex`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use common::{top_k_quickselect, Timestamp};
use index::{download_arena_dir, download_levels, download_manifest};
use index::{upload_arena_dir, upload_levels, upload_manifest};
use index::{BucketSeq, TimeBucketIndex};
use object_store::{path::Path as ObjectPath, ObjectStore};
use rand::{rngs::StdRng, SeedableRng};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

pub mod proto {
    tonic::include_proto!("mem_weaver.v1");
}

use proto::mem_weaver_server::MemWeaver;
use proto::{
    BatchInsertRequest, BatchInsertResponse, CreateCollectionRequest, CreateCollectionResponse,
    EvictBucketRequest, EvictBucketResponse, InsertResult, SearchHit, SearchRequest,
    SearchResponse, SwapBucketInFromBlobRequest, SwapBucketInFromBlobResponse, SwapBucketInRequest,
    SwapBucketInResponse, SwapBucketOutRequest, SwapBucketOutResponse, SwapBucketOutToBlobRequest,
    SwapBucketOutToBlobResponse,
};

pub use proto::mem_weaver_server::MemWeaverServer;

type Collection = Arc<RwLock<TimeBucketIndex>>;

pub struct MemWeaverService {
    collections: Arc<RwLock<HashMap<String, Collection>>>,
    /// Base directory for on-disk bucket files. Bucket `seq` of collection `c` is stored
    /// under `<data_dir>/<c>/seq_<seq>/`.
    data_dir: PathBuf,
    /// Optional S3 (or compatible) store for blob operations.
    store: Option<Arc<dyn ObjectStore>>,
    /// Base S3 prefix. Bucket `seq` of collection `c` lives at
    /// `<s3_prefix>/<c>/seq_<seq>/`.
    s3_prefix: ObjectPath,
}

impl MemWeaverService {
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            data_dir: PathBuf::from("./data"),
            store: None,
            s3_prefix: ObjectPath::from("mem-weaver"),
        }
    }

    pub fn with_storage(
        data_dir: impl Into<PathBuf>,
        store: Option<Arc<dyn ObjectStore>>,
        s3_prefix: impl Into<String>,
    ) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            data_dir: data_dir.into(),
            store,
            s3_prefix: ObjectPath::from(s3_prefix.into()),
        }
    }

    fn bucket_local_dir(&self, collection: &str, seq: u32) -> PathBuf {
        self.data_dir.join(collection).join(format!("seq_{seq}"))
    }

    fn bucket_s3_prefix(&self, collection: &str, seq: u32) -> ObjectPath {
        self.s3_prefix.child(collection).child(format!("seq_{seq}"))
    }
}

impl Default for MemWeaverService {
    fn default() -> Self {
        Self::new()
    }
}

fn invalid(msg: impl Into<String>) -> Status {
    Status::invalid_argument(msg.into())
}

fn internal(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
}

async fn get_collection(
    cols: &RwLock<HashMap<String, Collection>>,
    name: &str,
) -> Result<Collection, Status> {
    cols.read()
        .await
        .get(name)
        .cloned()
        .ok_or_else(|| Status::not_found(format!("collection '{name}' not found")))
}

fn validate_name(name: &str) -> Result<(), Status> {
    if name.is_empty() {
        Err(invalid("collection name must not be empty"))
    } else {
        Ok(())
    }
}

#[tonic::async_trait]
impl MemWeaver for MemWeaverService {
    async fn create_collection(
        &self,
        req: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        if r.dim == 0 {
            return Err(invalid("dim must be > 0"));
        }
        let bucket_duration = if r.bucket_duration_secs == 0 {
            Duration::from_secs(u64::MAX / 2)
        } else {
            Duration::from_secs(r.bucket_duration_secs)
        };
        let index = TimeBucketIndex::new(
            r.dim as usize,
            r.m as usize,
            r.m_max0 as usize,
            r.ef_construction as usize,
            bucket_duration,
            top_k_quickselect,
            StdRng::seed_from_u64(0),
        )
        .map_err(|e| invalid(format!("invalid index config: {e}")))?;

        let mut cols = self.collections.write().await;
        if cols.contains_key(&r.collection) {
            return Err(Status::already_exists(format!(
                "collection '{}' already exists",
                r.collection
            )));
        }
        cols.insert(r.collection, Arc::new(RwLock::new(index)));
        Ok(Response::new(CreateCollectionResponse {}))
    }

    async fn batch_insert(
        &self,
        req: Request<BatchInsertRequest>,
    ) -> Result<Response<BatchInsertResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        if r.items.is_empty() {
            return Ok(Response::new(BatchInsertResponse { results: vec![] }));
        }
        for (i, item) in r.items.iter().enumerate() {
            if item.vector.is_empty() {
                return Err(invalid(format!("item[{i}]: vector must not be empty")));
            }
        }
        let col = get_collection(&self.collections, &r.collection).await?;
        let mut idx = col.write().await;
        let results = r
            .items
            .iter()
            .map(|item| {
                let bid = idx.insert(&item.vector, Timestamp(item.timestamp), item.vector_id);
                InsertResult {
                    bucket_seq: bid.bucket_seq.0,
                    vector_id: bid.vector_id,
                }
            })
            .collect();
        Ok(Response::new(BatchInsertResponse { results }))
    }

    async fn search(
        &self,
        req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        if r.query.is_empty() {
            return Err(invalid("query must not be empty"));
        }
        let k = r.k as usize;
        let ef = r.ef as usize;
        if k == 0 {
            return Err(invalid("k must be > 0"));
        }
        let time_range = match (r.time_range_start, r.time_range_end) {
            (Some(start), Some(end)) => Some(Timestamp(start)..Timestamp(end)),
            _ => None,
        };
        let col = get_collection(&self.collections, &r.collection).await?;
        let idx = col.read().await;
        let results = idx.search(&r.query, k, ef, |_, d| d, time_range, top_k_quickselect);
        let hits = results
            .into_iter()
            .map(|(vector_id, distance)| SearchHit {
                vector_id,
                distance,
            })
            .collect();
        Ok(Response::new(SearchResponse { hits }))
    }

    async fn swap_bucket_out(
        &self,
        req: Request<SwapBucketOutRequest>,
    ) -> Result<Response<SwapBucketOutResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let local_dir = self.bucket_local_dir(&r.collection, r.bucket_seq);
        std::fs::create_dir_all(&local_dir).map_err(internal)?;
        let seq = BucketSeq(r.bucket_seq);
        let found = col
            .write()
            .await
            .swap_bucket_out(seq, &local_dir)
            .map_err(internal)?;
        Ok(Response::new(SwapBucketOutResponse { found }))
    }

    async fn swap_bucket_in(
        &self,
        req: Request<SwapBucketInRequest>,
    ) -> Result<Response<SwapBucketInResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let seq = BucketSeq(r.bucket_seq);
        let found = col.write().await.swap_bucket_in(seq).map_err(internal)?;
        Ok(Response::new(SwapBucketInResponse { found }))
    }

    async fn evict_bucket(
        &self,
        req: Request<EvictBucketRequest>,
    ) -> Result<Response<EvictBucketResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let seq = BucketSeq(r.bucket_seq);
        let resp = match col.write().await.evict_bucket(seq) {
            Some(count) => EvictBucketResponse {
                found: true,
                evicted_count: count as u32,
            },
            None => EvictBucketResponse {
                found: false,
                evicted_count: 0,
            },
        };
        Ok(Response::new(resp))
    }

    async fn swap_bucket_out_to_blob(
        &self,
        req: Request<SwapBucketOutToBlobRequest>,
    ) -> Result<Response<SwapBucketOutToBlobResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("S3 not configured"))?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let local_dir = self.bucket_local_dir(&r.collection, r.bucket_seq);
        let prefix = self.bucket_s3_prefix(&r.collection, r.bucket_seq);
        std::fs::create_dir_all(&local_dir).map_err(internal)?;
        let seq = BucketSeq(r.bucket_seq);

        // Swap to disk first (holds write lock only during disk I/O).
        let found = col
            .write()
            .await
            .swap_bucket_out(seq, &local_dir)
            .map_err(internal)?;

        // Upload to S3 without any lock — files are stable on disk.
        if found {
            upload_arena_dir(store.as_ref(), &local_dir, &prefix)
                .await
                .map_err(internal)?;
            upload_levels(store.as_ref(), &local_dir.join("levels.bin"), &prefix)
                .await
                .map_err(internal)?;
            upload_manifest(store.as_ref(), &local_dir.join("manifest.json"), &prefix)
                .await
                .map_err(internal)?;
        }
        Ok(Response::new(SwapBucketOutToBlobResponse { found }))
    }

    async fn swap_bucket_in_from_blob(
        &self,
        req: Request<SwapBucketInFromBlobRequest>,
    ) -> Result<Response<SwapBucketInFromBlobResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("S3 not configured"))?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let local_dir = self.bucket_local_dir(&r.collection, r.bucket_seq);
        let prefix = self.bucket_s3_prefix(&r.collection, r.bucket_seq);
        let seq = BucketSeq(r.bucket_seq);

        // Probe existence before downloading.
        {
            if !col.read().await.has_bucket(seq) {
                return Ok(Response::new(SwapBucketInFromBlobResponse {
                    found: false,
                    restored_count: 0,
                }));
            }
        }

        // Download from S3 without any lock.
        std::fs::create_dir_all(&local_dir).map_err(internal)?;
        download_arena_dir(store.as_ref(), &prefix, &local_dir)
            .await
            .map_err(internal)?;
        download_levels(store.as_ref(), &prefix, &local_dir.join("levels.bin"))
            .await
            .map_err(internal)?;
        download_manifest(store.as_ref(), &prefix, &local_dir.join("manifest.json"))
            .await
            .map_err(internal)?;

        // Restore from local files (holds write lock only during disk I/O).
        let resp = match col
            .write()
            .await
            .restore_bucket_from_local(seq, &local_dir)
            .map_err(internal)?
        {
            Some(count) => SwapBucketInFromBlobResponse {
                found: true,
                restored_count: count as u32,
            },
            None => SwapBucketInFromBlobResponse {
                found: false,
                restored_count: 0,
            },
        };
        Ok(Response::new(resp))
    }
}
