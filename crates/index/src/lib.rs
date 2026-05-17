//! Approximate nearest-neighbor **indexes** for dense vectors.
//!
//! Currently provides [`Hnsw`], [`HnswNaive`], and [`HnswArena`] (HNSW with
//! [`HnswVectorStore`] implementations), and [`TimeBucketIndex`] for streaming
//! workloads that need temporal partitioning and recency-weighted search.

mod hnsw;
mod time_bucket;

pub use hnsw::{
    ArenaNodeStore, GraphNode, Hnsw, HnswArena, HnswIndex, HnswNaive, HnswNodeStore,
    HnswVectorStore, NaiveNodeStore, NodeBlock, NodeId, DEFAULT_ALIGNMENT,
};
pub use time_bucket::{BucketedNodeId, ConfigError, TimeBucketIndex};
