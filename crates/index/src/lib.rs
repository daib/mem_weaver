//! Approximate nearest-neighbor **indexes** for dense vectors.
//!
//! Currently provides [`Hnsw`], [`HnswNaive`], and [`HnswArena`] (HNSW with
//! [`HnswVectorStore`] implementations), and [`TimeBucketIndex`] for streaming
//! workloads that need temporal partitioning and recency-weighted search.

pub mod blob;
mod hnsw;
mod time_bucket;

pub use blob::{
    decode_wal_entry, delete_prefix, delete_wal_entries_up_to, download_arena_dir,
    download_bucket_meta, download_catalog, download_collection_meta, download_levels,
    download_manifest, download_wal_entry, encode_wal_entry, list_wal_seqs, upload_arena_dir,
    upload_bucket_meta, upload_catalog, upload_collection_meta, upload_levels, upload_manifest,
    upload_wal_bytes, BucketMeta, Catalog, CatalogEntry, CollectionMeta, Uploaded, WalEntry,
    WalItem,
};
pub use hnsw::{
    ArenaNodeStore, GraphNode, Hnsw, HnswArena, HnswIndex, HnswNaive, HnswNodeStore,
    HnswVectorStore, NaiveNodeStore, NodeBlock, NodeId, ParallelHnsw, DEFAULT_ALIGNMENT,
};
pub use time_bucket::{BucketSeq, BucketedNodeId, ConfigError, TimeBucketIndex};
