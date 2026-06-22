//! HNSW: pluggable vector storage ([`HnswVectorStore`](store::HnswVectorStore)) and graph storage
//! in [`nodes`](crate::hnsw::nodes) ([`HnswNodeStore`], [`NodeBlock`], [`ArenaNodeStore`], [`NaiveNodeStore`]).

mod index;
mod nodes;
mod parallel;
mod store;

pub use index::{Hnsw, HnswIndex};
pub use nodes::{
    ArenaNodeStore, GraphNode, HnswNodeStore, NaiveNodeStore, NodeBlock, NodeId, DEFAULT_ALIGNMENT,
};
pub use parallel::ParallelHnsw;
pub use store::HnswVectorStore;

pub type HnswNaive = Hnsw<NaiveNodeStore>;
pub type HnswArena = Hnsw<ArenaNodeStore>;
