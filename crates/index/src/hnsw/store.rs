//! Pluggable storage for HNSW vertex vectors.

use super::nodes::NodeId;

/// How dense vectors for graph vertices are held (e.g. heap `Vec`s vs arena [`VectorStore`]).
pub trait HnswVectorStore: Default {
    /// Store the vector for internal id `id` (must be `0..=len` of existing vertices before a new
    /// node is appended, i.e. the next slot). Returns `false` if allocation fails.
    fn store_new(&mut self, id: NodeId, data: &[f32]) -> bool;
    /// Slice of stored data for a committed internal id.
    fn vector_at(&self, id: NodeId) -> &[f32];
}

/// `Vec` of per-vertex `Vec<f32>`.
#[derive(Debug, Default, Clone)]
pub struct NaiveVectorStore(Vec<Vec<f32>>);

impl NaiveVectorStore {
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }
}

impl HnswVectorStore for NaiveVectorStore {
    fn store_new(&mut self, id: NodeId, data: &[f32]) -> bool {
        if id != NodeId(self.0.len() as u32) {
            return false;
        }
        self.0.push(data.to_vec());
        true
    }

    fn vector_at(&self, id: NodeId) -> &[f32] {
        &self.0[id.0 as usize]
    }
}
