use crate::hnsw::store::{HnswVectorStore, NaiveVectorStore};
pub use common::types::NodeId;
use common::DEFAULT_ARENA_CAPACITY;
use std::mem::{align_of, size_of};
use vector::Arena;

/// Alignment used for arena-backed node storage (`try_alloc_slice_aligned`).
pub const DEFAULT_ALIGNMENT: usize = 8;
// ── Heap-allocated graph (naive) + trait ───────────────────────────────────

/// One vertex: per-level neighbor lists (same layout as the historical `Vec<Node>` implementation).
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub neighbors: Vec<Vec<NodeId>>,
}

impl GraphNode {
    pub fn new(max_level: usize) -> Self {
        Self {
            neighbors: vec![Vec::new(); max_level + 1],
        }
    }

    /// Highest allocated level (`neighbors.len() - 1`). Level `l` uses `neighbors[l]`.
    #[inline]
    pub fn max_level(&self) -> usize {
        self.neighbors.len().saturating_sub(1)
    }

    pub fn ensure_level(&mut self, level: usize) {
        while self.neighbors.len() <= level {
            self.neighbors.push(Vec::new());
        }
    }

    pub fn neighbors_at(&self, level: usize) -> &[NodeId] {
        self.neighbors
            .get(level)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Heap-backed `Vec<GraphNode>`: fully naive graph storage (no arena).
#[derive(Debug, Clone)]
pub struct NaiveNodeStore {
    m: usize,
    m_max0: usize,
    pub(crate) nodes: Vec<GraphNode>,
    pub(crate) vector_store: NaiveVectorStore,
}

impl NaiveNodeStore {
    pub(crate) fn new(m: usize, m_max0: usize) -> Self {
        Self {
            m,
            m_max0,
            nodes: Vec::new(),
            vector_store: NaiveVectorStore::default(),
        }
    }
}

// ── Arena strided blocks + store (VectorStore-style) ─────────────────────────

const MAX_LEVEL: usize = 32;
const LEVELS: usize = MAX_LEVEL + 1;
pub const INVALID_NODE_ID: NodeId = u32::MAX;

// the layout of the node is:
// - vector: f32[dim]
// - edges: NodeId[edge_count]
pub struct Node;

impl Node {
    /// Byte size of `f32[dim]` plus padding so `max_level: usize` is aligned.
    #[inline]
    fn vector_span(dim: usize) -> usize {
        (dim * size_of::<f32>()).next_multiple_of(align_of::<usize>())
    }

    /// Byte offset from node base to the start of the packed edge array.
    #[inline]
    fn edges_byte_offset(dim: usize) -> usize {
        Self::vector_span(dim)
    }

    #[inline]
    unsafe fn vector<'a>(node_address: *mut u8, dim: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(node_address.cast::<f32>(), dim)
    }

    /// Total `NodeId` slots: level `0` uses `m_max0`, each level `1..=max_level` uses `m`.
    #[inline]
    const fn edge_count(max_level: usize, m: usize, m_max0: usize) -> usize {
        m_max0 + max_level * m
    }

    #[inline]
    fn total_size(dim: usize, max_level: usize, m: usize, m_max0: usize) -> usize {
        Self::edges_byte_offset(dim)
            + Self::edge_count(max_level, m, m_max0) * size_of::<NodeId>()
            + 1
    }

    #[inline]
    unsafe fn edges<'a>(
        node_address: *mut u8,
        dim: usize,
        max_level: usize,
        m: usize,
        m_max0: usize,
    ) -> &'a mut [NodeId] {
        std::slice::from_raw_parts_mut(
            node_address
                .add(Self::edges_byte_offset(dim))
                .cast::<NodeId>(),
            Self::edge_count(max_level, m, m_max0),
        )
    }

    #[inline]
    unsafe fn edges_at_level<'a>(
        node_address: *const u8,
        dim: usize,
        level: usize,
        m: usize,
        m_max0: usize,
    ) -> &'a [NodeId] {
        let edge_offset = if level == 0 {
            0
        } else {
            m_max0 + (level - 1) * m
        };
        let cap = if level == 0 { m_max0 } else { m };
        std::slice::from_raw_parts(
            node_address
                .add(Self::edges_byte_offset(dim))
                .wrapping_add(edge_offset * size_of::<NodeId>())
                .cast::<NodeId>(),
            cap,
        )
    }

    #[inline]
    unsafe fn edges_at_level_mut<'a>(
        node_address: *mut u8,
        dim: usize,
        level: usize,
        m: usize,
        m_max0: usize,
    ) -> &'a mut [NodeId] {
        let edge_offset = if level == 0 {
            0
        } else {
            m_max0 + (level - 1) * m
        };
        let cap = if level == 0 { m_max0 } else { m };
        std::slice::from_raw_parts_mut(
            node_address
                .add(Self::edges_byte_offset(dim))
                .wrapping_add(edge_offset * size_of::<NodeId>())
                .cast::<NodeId>(),
            cap,
        )
    }
}

pub struct NodeBlock {
    arena: Arena,
    block_index: usize,
    len: usize,
    dim: usize,    // vector dimension
    m: usize,      // target max degree on levels > 0
    m_max0: usize, // target max degree on level 0
}

impl NodeBlock {
    pub fn try_new(dim: usize, m: usize, m_max0: usize, block_index: usize) -> Option<Self> {
        Some(Self {
            arena: Arena::try_with_capacity(DEFAULT_ARENA_CAPACITY).unwrap(),
            len: 0,
            block_index,
            dim,
            m,
            m_max0,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn calculate_node_id(&mut self, offset: usize) -> NodeId {
        // assume that we align to 8 bytes
        // TODO: Make this configurable based on the arena size and the number of bits to ignore for alignment
        // At the moment: 2MB arena = 2^21. Ignore 3 last bits for alignment arrive at 18 bits for offset.
        // The remaining is for node_index
        (self.block_index << 18 | ((offset >> 3) & ((1 << 18) - 1))) as NodeId
    }

    #[inline]
    fn derive_block_index(node_id: NodeId) -> usize {
        (node_id >> 18) as usize
    }

    #[inline]
    fn derive_node_offset(node_id: NodeId) -> usize {
        ((node_id & ((1 << 18) - 1)) as usize) << 3
    }

    // return the index of the new node in the block
    pub fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId> {
        let total = Node::total_size(self.dim, max_level, self.m, self.m_max0);
        // we need to align by 8 bytes at the moment
        let node_storage = self
            .arena
            .try_alloc_slice_aligned::<u8>(total, DEFAULT_ALIGNMENT)?;
        let p = node_storage.as_mut_ptr();
        unsafe {
            Node::vector(p, self.dim).copy_from_slice(vector);
            // *Node::max_level_mut(p, self.dim) = max_level;
            Node::edges(p, self.dim, max_level, self.m, self.m_max0).fill(INVALID_NODE_ID);
        }

        // add the node to the block
        self.len += 1;
        Some(self.calculate_node_id(node_storage.as_ptr() as usize - self.arena.as_ptr() as usize))
    }

    #[inline]
    fn calculate_node_address(&self, node_id: NodeId) -> *const u8 {
        let node_offset = NodeBlock::derive_node_offset(node_id);
        self.arena.as_ptr().wrapping_add(node_offset) as *const u8
    }
    // get the neighbors of the node at the given level
    pub fn neighbors_at(&self, node_id: NodeId, level: usize) -> &[NodeId] {
        assert!(level < LEVELS);
        let node_address = self.calculate_node_address(node_id);
        unsafe { Node::edges_at_level(node_address, self.dim, level, self.m, self.m_max0) }
    }

    // get the neighbors of the node at the given level
    pub fn neighbors_at_mut(&mut self, node_id: NodeId, level: usize) -> &mut [NodeId] {
        assert!(level < LEVELS);
        let node_address = self.calculate_node_address(node_id);
        unsafe {
            Node::edges_at_level_mut(
                node_address as *mut u8,
                self.dim,
                level,
                self.m,
                self.m_max0,
            )
        }
    }

    pub fn save_neighbors(&mut self, node_id: NodeId, neighbors: &[NodeId], level: usize) {
        assert!(level < LEVELS);
        let node_address = self.calculate_node_address(node_id);
        let row = unsafe {
            Node::edges_at_level_mut(
                node_address as *mut u8,
                self.dim,
                level,
                self.m,
                self.m_max0,
            )
        };
        assert!(
            neighbors.len() <= row.len(),
            "neighbors len {} exceeds level capacity {}",
            neighbors.len(),
            row.len()
        );
        row[..neighbors.len()].copy_from_slice(neighbors);
    }

    // TODO: How to ensure the level is valid?
    pub fn ensure_level(&mut self, _: NodeId, _: usize) {}
}

pub struct ArenaNodeStore {
    dim: usize,
    m: usize,
    m_max0: usize,
    blocks: Vec<NodeBlock>,
}

impl ArenaNodeStore {
    pub fn try_new(dim: usize, m: usize, m_max0: usize) -> std::io::Result<Self> {
        Ok(Self {
            dim,
            m,
            m_max0,
            blocks: Vec::new(),
        })
    }

    #[inline]
    fn block(&self, block_index: usize) -> &NodeBlock {
        &self.blocks[block_index]
    }

    #[inline]
    fn block_mut(&mut self, block_index: usize) -> &mut NodeBlock {
        &mut self.blocks[block_index]
    }

    /// Removes `target` from `node_id`'s outgoing neighbors at `level` (sentinel layout).
    fn remove_outgoing_to(&mut self, node_id: NodeId, target: NodeId, level: usize) {
        if node_id == INVALID_NODE_ID || target == INVALID_NODE_ID {
            return;
        }
        let bi = NodeBlock::derive_block_index(node_id);
        if bi >= self.blocks.len() {
            return;
        }
        let block = self.block_mut(bi);
        let neighbors = block.neighbors_at_mut(node_id, level);
        let mut target_index = neighbors.len();
        let mut first_empty_index = neighbors.len();
        for i in 0..neighbors.len() {
            if neighbors[i] == target {
                target_index = i;
            }

            if neighbors[i] == INVALID_NODE_ID {
                first_empty_index = i;
                break;
            }
        }

        if target_index == neighbors.len() {
            return;
        }

        neighbors[target_index] = INVALID_NODE_ID;

        // this neighbor list is empty
        if first_empty_index == 0 {
            return;
        }
        // swap the last non empty neighbor to the evicted slot
        let last_non_empty_index = first_empty_index - 1;
        neighbors[target_index] = neighbors[last_non_empty_index];
        neighbors[last_non_empty_index] = INVALID_NODE_ID;
    }
}
/// Graph (node / edge) side of HNSW; see [`NaiveNodeStore`] and [`ArenaNodeStore`].
pub trait HnswNodeStore {
    fn len(&self) -> usize;
    /// Returns new internal id, or `None` if the store is at capacity.
    fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId>;
    fn neighbors_at(&self, id: NodeId, level: usize) -> &[NodeId];
    fn ensure_level(&mut self, id: NodeId, level: usize);
    // save the neighbors of a node to the edges at the given level
    fn save_neighbors(&mut self, id: NodeId, neighbors: &[NodeId], level: usize);
    /// `distance_fn` is a plain [`fn`] pointer so this trait stays **object-safe** (`dyn HnswNodeStore`,
    /// `Box<dyn HnswNodeStore>`). Use e.g. [`vector::distance::euclidean_distance_sq`].
    fn add_directed_edge(
        &mut self,
        src_id: NodeId,
        dst_id: NodeId,
        level: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> bool;
    fn vector_at(&self, id: NodeId) -> &[f32];
}

impl HnswNodeStore for NaiveNodeStore {
    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId> {
        let id = self.nodes.len() as NodeId;
        self.vector_store.store_new(id, vector);
        self.nodes.push(GraphNode::new(max_level));
        Some(id)
    }

    fn neighbors_at(&self, id: NodeId, level: usize) -> &[NodeId] {
        self.nodes[id as usize].neighbors_at(level)
    }

    fn ensure_level(&mut self, id: NodeId, level: usize) {
        self.nodes[id as usize].ensure_level(level);
    }

    fn save_neighbors(&mut self, id: NodeId, neighbors: &[NodeId], level: usize) {
        self.nodes[id as usize].neighbors[level].extend(neighbors);
    }

    // update the edge from the node to the neighbor at the given level
    fn add_directed_edge(
        &mut self,
        src_id: NodeId,
        dst_id: NodeId,
        level: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> bool {
        let cap = if level == 0 { self.m_max0 } else { self.m };
        if self.nodes[src_id as usize].neighbors[level].contains(&dst_id) {
            return true;
        }
        if self.nodes[src_id as usize].neighbors[level].len() < cap {
            self.nodes[src_id as usize].neighbors[level].push(dst_id);
            return true;
        }

        let (farthest_neighbor_index, farthest_distance) = {
            let neighbors = &self.nodes[src_id as usize].neighbors[level];
            let mut idx = 0usize;
            let mut dist = f32::MIN;
            for i in 0..neighbors.len() {
                let d = distance_fn(self.vector_at(src_id), self.vector_at(neighbors[i]));
                if d > dist {
                    idx = i;
                    dist = d;
                }
            }
            (idx, dist)
        };

        let new_distance = distance_fn(self.vector_at(src_id), self.vector_at(dst_id));
        if new_distance > farthest_distance {
            return false;
        }

        let removed_neighbor =
            self.nodes[src_id as usize].neighbors[level][farthest_neighbor_index];
        if removed_neighbor != src_id {
            self.nodes[removed_neighbor as usize].neighbors[level].retain(|&x| x != src_id);
        }
        self.nodes[src_id as usize].neighbors[level][farthest_neighbor_index] = dst_id;
        true
    }

    fn vector_at(&self, id: NodeId) -> &[f32] {
        self.vector_store.vector_at(id)
    }
}

impl HnswNodeStore for ArenaNodeStore {
    fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).sum()
    }

    fn push_node(&mut self, vector: &[f32], max_level: usize) -> Option<NodeId> {
        if self.blocks.is_empty() {
            self.blocks
                .push(NodeBlock::try_new(self.dim, self.m, self.m_max0, 0).unwrap());
        }
        let mut is_new_block = false;
        loop {
            if let Some(block) = self.blocks.last_mut() {
                if let Some(node_id) = block.push_node(vector, max_level) {
                    return Some(node_id);
                }

                if is_new_block {
                    // already tried to allocate a new block. Return None to indicate that the store is at capacity
                    return None;
                }

                // failed to push the node to the last block. Try to allocate a new block
                let new_block =
                    NodeBlock::try_new(self.dim, self.m, self.m_max0, self.blocks.len()).unwrap();
                self.blocks.push(new_block);
                is_new_block = true;
            }
        }
    }

    fn neighbors_at(&self, id: NodeId, level: usize) -> &[NodeId] {
        let block_index = NodeBlock::derive_block_index(id);
        match self.blocks.get(block_index) {
            Some(block) => block.neighbors_at(id, level),
            None => &[],
        }
    }

    fn ensure_level(&mut self, id: NodeId, level: usize) {
        let block_index = NodeBlock::derive_block_index(id);
        if let Some(block) = self.blocks.get_mut(block_index) {
            block.ensure_level(id, level);
        }
    }

    fn save_neighbors(&mut self, id: NodeId, neighbors: &[NodeId], level: usize) {
        let block_index = NodeBlock::derive_block_index(id);
        if let Some(block) = self.blocks.get_mut(block_index) {
            block.save_neighbors(id, neighbors, level);
        }
    }

    fn add_directed_edge(
        &mut self,
        src_id: NodeId,
        dst_id: NodeId,
        level: usize,
        distance_fn: fn(&[f32], &[f32]) -> f32,
    ) -> bool {
        let src_block_index = NodeBlock::derive_block_index(src_id);
        if src_block_index >= self.blocks.len() {
            return false;
        }

        // {
        //     let block = self.block(src_block_index);
        //     let addr = block.calculate_node_address(src_id);
        //     if unsafe { level > Node::max_level(addr, block.dim) } {
        //         return false;
        //     }
        // }

        // Empty slot or duplicate — keep `src` borrow local.
        {
            let block = self.block_mut(src_block_index);
            let neighbors = block.neighbors_at_mut(src_id, level);
            for i in 0..neighbors.len() {
                if neighbors[i] == INVALID_NODE_ID {
                    neighbors[i] = dst_id;
                    return true;
                }
                if neighbors[i] == dst_id {
                    return true;
                }
            }
        }

        // Row full: pick farthest existing neighbor, then maybe swap. Snapshot neighbor ids first:
        // `neighbors_at_mut` borrows `self` mutably; `vector_at` needs `&self`.
        let neighbor_ids: Vec<NodeId> = {
            let block = self.block(src_block_index);
            block.neighbors_at(src_id, level).to_vec()
        };

        let (farthest_idx, farthest_dist, removed_neighbor, found) = {
            let mut farthest_neighbor_index = 0usize;
            let mut farthest_distance = f32::MIN;
            let mut found = false;
            for i in 0..neighbor_ids.len() {
                let nid = neighbor_ids[i];
                if nid == INVALID_NODE_ID {
                    continue;
                }
                let nb_i = NodeBlock::derive_block_index(nid);
                if nb_i >= self.blocks.len() {
                    continue;
                }
                let distance = distance_fn(self.vector_at(src_id), self.vector_at(nid));
                if !found || distance >= farthest_distance {
                    farthest_neighbor_index = i;
                    farthest_distance = distance;
                    found = true;
                }
            }
            let removed = if found {
                neighbor_ids[farthest_neighbor_index]
            } else {
                INVALID_NODE_ID
            };
            (farthest_neighbor_index, farthest_distance, removed, found)
        };

        if !found {
            return false;
        }

        let new_distance = distance_fn(self.vector_at(src_id), self.vector_at(dst_id));
        if new_distance > farthest_dist {
            return false;
        }

        if removed_neighbor != src_id {
            let rbi = NodeBlock::derive_block_index(removed_neighbor);
            if rbi < self.blocks.len() {
                self.remove_outgoing_to(removed_neighbor, src_id, level);
            }
        }

        let block = self.block_mut(src_block_index);
        let neighbors = block.neighbors_at_mut(src_id, level);
        neighbors[farthest_idx] = dst_id;
        true
    }

    fn vector_at(&self, id: NodeId) -> &[f32] {
        let block_index = NodeBlock::derive_block_index(id);
        let block = self.block(block_index);
        let node_address = block.calculate_node_address(id);
        unsafe { Node::vector(node_address as *mut u8, self.dim) }
    }
}
#[cfg(test)]
mod tests {
    //! Unit tests for [`NaiveNodeStore`], [`ArenaNodeStore`], and [`NodeBlock`] layout helpers.

    use super::*;
    use vector::distance::euclidean_distance_sq;

    /// [`NaiveNodeStore::push_node`] assigns contiguous [`NodeId`]s (0, 1, …), stores vectors, and
    /// allocates `max_level + 1` neighbor rows on each [`GraphNode`].
    #[test]
    fn naive_push_yields_sequential_ids_and_vectors() {
        let mut store = NaiveNodeStore::new(4, 8);
        assert_eq!(store.len(), 0);

        let a = store.push_node(&[1.0, 2.0, 3.0, 4.0], 1).expect("push");
        assert_eq!(a, 0);
        let b = store.push_node(&[5.0, 6.0, 7.0, 8.0], 0).expect("push");
        assert_eq!(b, 1);
        assert_eq!(store.len(), 2);

        assert_eq!(store.vector_at(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(store.vector_at(1), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(store.nodes[0].max_level(), 1);
        assert_eq!(store.nodes[1].max_level(), 0);
    }

    /// Naive `save_neighbors` appends to per-level `Vec`s (separate calls for levels 0 and 1).
    #[test]
    fn naive_save_neighbors_extends_level_lists() {
        let mut store = NaiveNodeStore::new(4, 8);
        let id = store.push_node(&[0.0; 4], 2).expect("push");
        let n0 = vec![10_u32, 11];
        let n1 = vec![20_u32];
        store.save_neighbors(id, &n0, 0);
        store.save_neighbors(id, &n1, 1);
        assert_eq!(store.neighbors_at(id, 0), &[10, 11]);
        assert_eq!(store.neighbors_at(id, 1), &[20]);
        assert!(store.neighbors_at(id, 2).is_empty());
    }

    /// [`GraphNode::ensure_level`] grows the neighbor slot table when connecting at a higher level
    /// than the node was created with.
    #[test]
    fn naive_ensure_level_extends_neighbor_structure() {
        let mut store = NaiveNodeStore::new(4, 8);
        let id = store.push_node(&[0.0; 4], 0).expect("push");
        assert_eq!(store.nodes[id as usize].neighbors.len(), 1);
        store.ensure_level(id, 3);
        assert_eq!(store.nodes[id as usize].neighbors.len(), 4);
        assert_eq!(store.nodes[id as usize].max_level(), 3);
    }

    /// Strips [`INVALID_NODE_ID`] sentinels from a fixed-width arena neighbor row for assertions.
    fn nonzero_neighbors(slice: &[NodeId]) -> Vec<NodeId> {
        slice
            .iter()
            .copied()
            .filter(|&x| x != INVALID_NODE_ID)
            .collect()
    }

    /// Shared geometry for eviction / back-edge tests: four 2-D points on the x-axis.
    const COLLINEAR_DIM: usize = 2;
    const COLLINEAR_M: usize = 2;
    const COLLINEAR_M_MAX0: usize = 2;
    /// From origin, distances² are 1, 10_000, 40_000 — third outgoing replaces `[200,0]` with `[1,0]`.
    const COLLINEAR_MAX_LEVEL: usize = 1;

    fn collinear_four_vectors() -> [[f32; 2]; 4] {
        [[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [1.0, 0.0]]
    }

    /// First `add_directed_edge` inserts; a second identical edge is a no-op but still reports success.
    /// Naive uses contiguous ids; arena uses sentinel-filled rows and encoded ids.
    #[test]
    fn add_directed_edge_inserts_once_and_duplicate_ok_naive_and_arena() {
        let test_cases: Vec<Box<dyn HnswNodeStore + 'static>> = vec![
            Box::new(NaiveNodeStore::new(4, 8)),
            Box::new(ArenaNodeStore::try_new(2, 4, 8).expect("new")),
        ];
        for mut store in test_cases {
            let id0 = store.push_node(&[0.0, 0.0], 1).expect("n0");
            let id1 = store.push_node(&[1.0, 0.0], 1).expect("n1");
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert_eq!(nonzero_neighbors(store.neighbors_at(id0, 0)), vec![id1]);
        }
    }

    /// With level-0 capacity two, a third outgoing edge evicts the farthest neighbor (same graph on
    /// both stores; arena uses encoded [`NodeId`]s and sentinels).
    #[test]
    fn add_directed_edge_evicts_farthest_when_level_zero_full_naive_and_arena() {
        let test_cases: Vec<Box<dyn HnswNodeStore + 'static>> = vec![
            Box::new(NaiveNodeStore::new(COLLINEAR_M, COLLINEAR_M_MAX0)),
            Box::new(
                ArenaNodeStore::try_new(COLLINEAR_DIM, COLLINEAR_M, COLLINEAR_M_MAX0).expect("new"),
            ),
        ];
        let v = collinear_four_vectors();
        for mut store in test_cases {
            let id0 = store.push_node(&v[0], COLLINEAR_MAX_LEVEL).expect("n0");
            let id1 = store.push_node(&v[1], COLLINEAR_MAX_LEVEL).expect("n1");
            let id2 = store.push_node(&v[2], COLLINEAR_MAX_LEVEL).expect("n2");
            let id3 = store.push_node(&v[3], COLLINEAR_MAX_LEVEL).expect("n3");
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id2, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id3, 0, euclidean_distance_sq));
            let present = nonzero_neighbors(store.neighbors_at(id0, 0));
            assert_eq!(present.len(), 2);
            assert!(present.contains(&id1));
            assert!(present.contains(&id3));
            assert!(!present.contains(&id2));
        }
    }

    /// Full row eviction removes the reverse edge from the dropped neighbor (naive `Vec` vs arena
    /// sentinels + `remove_outgoing_to`).
    #[test]
    fn add_directed_edge_drops_back_edge_from_evicted_neighbor_naive_and_arena() {
        let test_cases: Vec<Box<dyn HnswNodeStore + 'static>> = vec![
            Box::new(NaiveNodeStore::new(COLLINEAR_M, COLLINEAR_M_MAX0)),
            Box::new(
                ArenaNodeStore::try_new(COLLINEAR_DIM, COLLINEAR_M, COLLINEAR_M_MAX0).expect("new"),
            ),
        ];
        let v = collinear_four_vectors();
        for mut store in test_cases {
            let id0 = store.push_node(&v[0], COLLINEAR_MAX_LEVEL).expect("n0");
            let id1 = store.push_node(&v[1], COLLINEAR_MAX_LEVEL).expect("n1");
            let id2 = store.push_node(&v[2], COLLINEAR_MAX_LEVEL).expect("n2");
            let id3 = store.push_node(&v[3], COLLINEAR_MAX_LEVEL).expect("n3");
            assert!(store.add_directed_edge(id0, id1, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id2, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id1, id0, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id2, id0, 0, euclidean_distance_sq));
            assert!(store.add_directed_edge(id0, id3, 0, euclidean_distance_sq));
            assert!(!nonzero_neighbors(store.neighbors_at(id2, 0)).contains(&id0));
            assert!(nonzero_neighbors(store.neighbors_at(id1, 0)).contains(&id0));
        }
    }

    /// Cannot attach edges at a level higher than the node’s allocated neighbor rows.
    #[test]
    fn arena_add_directed_edge_returns_false_for_level_above_max_level() {
        let mut store = ArenaNodeStore::try_new(2, 2, 2).expect("new");
        let id0 = store.push_node(&[0.0, 0.0], 0).expect("push");
        let id1 = store.push_node(&[1.0, 0.0], 0).expect("push");
        assert!(!store.add_directed_edge(id0, id1, 1, euclidean_distance_sq));
    }

    /// Single [`NodeBlock`]: allocation alignment, vector/`max_level` fields, packed edges,
    /// `save_neighbors`, and raw `Node::*` slice views per level.
    #[test]
    fn node_data_store() {
        const DIM: usize = 128;
        const M: usize = 16;
        const M_MAX0: usize = 32;
        let mut node_block = NodeBlock::try_new(DIM, M, M_MAX0, 0).expect("test alloc");
        for i in 0..10 {
            let fill = 1.0f32 * i as f32;
            let stored = vec![fill; DIM];
            let max_level = ((i + 5) * 11usize).pow(5).min(MAX_LEVEL);
            let node_id = node_block
                .push_node(&stored, max_level)
                .expect("test alloc");
            let node_address = node_block.calculate_node_address(node_id);
            // vector is aligned to 8 bytes
            assert_eq!(node_address as usize % 8, 0);

            assert_eq!(
                unsafe { Node::vector(node_address as *mut u8, DIM) },
                stored.as_slice()
            );

            let edge_slots = Node::edge_count(max_level, M, M_MAX0);
            let expected_edges = vec![INVALID_NODE_ID; edge_slots];
            assert_eq!(
                unsafe { Node::edges(node_address as *mut u8, DIM, max_level, M, M_MAX0) },
                expected_edges.as_slice()
            );

            // save some edges
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let neighbors = vec![(i * (max_level + 10) + l) as NodeId; num_neighbors];
                node_block.save_neighbors(node_id, &neighbors.as_slice(), l);
            }

            // validate the edges
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let expected_neighbors = vec![(i * (max_level + 10) + l) as NodeId; num_neighbors];
                let actual_neighbors = node_block.neighbors_at(node_id, l);
                assert_eq!(actual_neighbors.len(), num_neighbors);
                assert_eq!(actual_neighbors, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id} at level {l} should be {expected_neighbors:?}");

                // test neighbors_at_mut
                let neighbors_at_mut = node_block.neighbors_at_mut(node_id, l);
                assert_eq!(neighbors_at_mut.len(), num_neighbors);
                assert_eq!(neighbors_at_mut, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id} at level {l} should be {expected_neighbors:?}");

                // test using unsafe api
                let unsafe_neighbors =
                    unsafe { Node::edges_at_level(node_address as *mut u8, DIM, l, M, M_MAX0) };
                assert_eq!(unsafe_neighbors.len(), num_neighbors);
                assert_eq!(unsafe_neighbors, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id} at level {l} should be {expected_neighbors:?}");

                // test using unsafe api with mut
                let unsafe_neighbors_mut =
                    unsafe { Node::edges_at_level_mut(node_address as *mut u8, DIM, l, M, M_MAX0) };
                assert_eq!(unsafe_neighbors_mut.len(), num_neighbors);
                assert_eq!(unsafe_neighbors_mut, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id} at level {l} should be {expected_neighbors:?}");

                // test the unsafe edges api
                let unsafe_edges =
                    unsafe { Node::edges(node_address as *mut u8, DIM, max_level, M, M_MAX0) };
                assert_eq!(unsafe_edges.len(), edge_slots);
                // extract the edges for this level and compare with the expected neighbors
                let start_index = if l == 0 { 0 } else { M_MAX0 + (l - 1) * M };
                let end_index = if l == max_level {
                    edge_slots
                } else {
                    M_MAX0 + l * M
                };
                let edges = unsafe_edges[start_index..end_index].to_vec();
                assert_eq!(
                    edges,
                    expected_neighbors.as_slice(),
                    "edges at i {i} node {node_id} at level {l} should be {expected_neighbors:?}"
                );
            }
        }
    }

    /// [`ArenaNodeStore`] spanning multiple [`NodeBlock`]s: push until several blocks exist, then
    /// verify `vector_at`, `max_level`, and filled neighbor rows round-trip per encoded id.
    #[test]
    fn multiple_arena_stores() {
        const DIM: usize = 128;
        const M: usize = 16;
        const M_MAX0: usize = 32;
        let mut store = ArenaNodeStore::try_new(DIM, M, M_MAX0).expect("new");
        let max_num_blocks = 4;
        let mut i = 0;
        let mut node_ids: Vec<NodeId> = Vec::new();
        while store.blocks.len() < max_num_blocks {
            let fill = 1.0f32 * i as f32;
            let stored = vec![fill; DIM];
            let max_level = ((i + 6) * 11usize).pow(2).min(MAX_LEVEL);

            let node_id = store.push_node(&stored, max_level).expect("test alloc");

            node_ids.push(node_id);

            // initialize the neighbors
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let neighbors = vec![(i * (max_level + 10) + l) as NodeId; num_neighbors];
                store.save_neighbors(node_id, &neighbors.as_slice(), l);
            }
            i += 1;
        }

        for i in 0..node_ids.len() {
            let node_id = node_ids[i];
            let block_index = NodeBlock::derive_block_index(node_id);
            let block = store.block(block_index);
            let node_address = block.calculate_node_address(node_id);
            assert_eq!(node_address as usize % 8, 0);

            let fill = 1.0f32 * i as f32;
            let stored = vec![fill; DIM];
            let max_level = ((i + 6) * 11usize).pow(2).min(MAX_LEVEL);

            // check data by arena storage api
            assert_eq!(
                store.vector_at(node_id),
                stored.as_slice(),
                "vector at node {node_id} should be {stored:?}"
            );

            // validate the neighbors
            for l in 0..max_level + 1 {
                let num_neighbors = if l == 0 { M_MAX0 } else { M };
                let expected_neighbors = vec![(i * (max_level + 10) + l) as NodeId; num_neighbors];
                let actual_neighbors = store.neighbors_at(node_id, l);
                assert_eq!(actual_neighbors.len(), num_neighbors);
                assert_eq!(actual_neighbors, expected_neighbors.as_slice(), "neighbors at i {i} node {node_id} at level {l} should be {expected_neighbors:?}");
            }
        }
    }
}
