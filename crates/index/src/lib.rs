//! Approximate nearest-neighbor **indexes** for dense vectors.
//!
//! Currently provides [`Hnsw`], [`HnswNaive`], and [`HnswArena`] (HNSW with
//! [`HnswVectorStore`] implementations), and [`TimeBucketIndex`] for streaming
//! workloads that need temporal partitioning and recency-weighted search.

pub mod blob;
mod hnsw;
mod time_bucket;

pub use blob::{
    delete_prefix, download_arena_dir, download_bucket_meta, download_catalog,
    download_collection_meta, download_levels, download_manifest, upload_arena_dir,
    upload_bucket_meta, upload_catalog, upload_collection_meta, upload_levels, upload_manifest,
    BucketMeta, Catalog, CatalogEntry, CollectionMeta, Uploaded,
};
pub use hnsw::{
    ArenaNodeStore, GraphNode, Hnsw, HnswArena, HnswIndex, HnswNaive, HnswNodeStore,
    HnswVectorStore, NaiveNodeStore, NodeBlock, NodeId, DEFAULT_ALIGNMENT,
};
pub use time_bucket::{BucketSeq, BucketedNodeId, ConfigError, TimeBucketIndex};
