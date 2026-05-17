//! Shared HNSW graph and search over [`super::store::HnswVectorStore`] and [`super::nodes::HnswNodeStore`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::fmt;

use common::OrdF32;
use rand::rngs::StdRng;
use vector::distance::euclidean_distance_sq;

use super::nodes::{ArenaNodeStore, HnswNodeStore, NaiveNodeStore, NodeId, INVALID_NODE_ID};

/// Façade over [`Hnsw`] (`insert` / `search`).
/// [`HnswNaive`] / [`HnswArena`] can be used as `dyn HnswIndex`.
pub trait HnswIndex {
    fn len(&self) -> usize;
    fn insert(&mut self, vector: &[f32]) -> NodeId;
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)>;
}

/// HNSW index: `N` holds the graph ([`NaiveNodeStore`] or [`ArenaNodeStore`](super::nodes::ArenaNodeStore)), `S` holds vectors.
pub struct Hnsw<N: HnswNodeStore> {
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ml: f64,
    // We dont need to know the max level of each node during search and insertion, we only need to know the max level of the entry point.
    // we only need the max_level to remove the node, and thus the edges of the node. We can store the max level of each node in an auxiliary hash map later if needed
    max_depth: i32,
    pub entry_point: Option<NodeId>,
    graph: N,
    closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
    rng: StdRng,
}

impl<N: HnswNodeStore> fmt::Debug for Hnsw<N>
where
    N: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hnsw")
            .field("dim", &self.dim)
            .field("m", &self.m)
            .field("m_max0", &self.m_max0)
            .field("ef_construction", &self.ef_construction)
            .field("len", &self.len())
            .field("graph_len", &self.graph.len())
            .field("entry_point", &self.entry_point)
            .finish_non_exhaustive()
    }
}

impl Hnsw<NaiveNodeStore> {
    /// `m`: target max degree on levels `> 0`. Level 0 allows up to `m_max0` (often `2 * m`).
    ///
    /// Graph uses heap [`NaiveNodeStore`] (naive vector + naive node storage).
    pub fn new(
        dim: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
        rng: StdRng,
    ) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(m >= 2, "m must be at least 2");
        assert!(m_max0 >= m, "m_max0 must be >= m");
        assert!(ef_construction >= m, "ef_construction should be >= m");

        Self {
            dim,
            m,
            m_max0,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            max_depth: -1,
            entry_point: None,
            graph: NaiveNodeStore::new(m, m_max0),
            closest_m_candidates,
            rng,
        }
    }
}

impl<N: HnswNodeStore> HnswIndex for Hnsw<N> {
    fn len(&self) -> usize {
        Hnsw::len(self)
    }

    fn insert(&mut self, vector: &[f32]) -> NodeId {
        // Call inherent `Hnsw::insert`, not this trait method (same name would recurse).
        Hnsw::insert(self, vector)
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        Hnsw::search(self, query, k, ef)
    }
}

impl Hnsw<ArenaNodeStore> {
    /// Like [`HnswNaive::new`](HnswNaive) (four arguments) but vectors use [`ArenaVectorStore`]
    /// and the graph uses mmap [`NodeBlock`](super::nodes::NodeBlock) chunks via [`ArenaNodeStore`]
    /// (same pattern as [`vector::VectorStore`]: new mmap arenas as chunks fill).
    ///
    /// `node_block_capacity` is the vertex capacity of each chunk; additional chunks are
    /// allocated as needed when the index grows.
    pub fn new(
        dim: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        node_block_capacity: usize,
        closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
        rng: StdRng,
    ) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(m >= 2, "m must be at least 2");
        assert!(m_max0 >= m, "m_max0 must be >= m");
        assert!(ef_construction >= m, "ef_construction should be >= m");
        assert!(
            node_block_capacity > 0,
            "node_block_capacity must be positive"
        );

        let graph = ArenaNodeStore::try_new(dim, m, m_max0)
            .expect("failed to map arena or allocate NodeBlock for HNSW graph");

        Self {
            dim,
            m,
            m_max0,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            max_depth: -1,
            entry_point: None,
            graph,
            closest_m_candidates,
            rng,
        }
    }
}

impl<N: HnswNodeStore> Hnsw<N> {
    /// Lookup stored vector (internal id).
    #[inline]
    pub fn vector_at(&self, id: NodeId) -> &[f32] {
        self.graph.vector_at(id)
    }

    #[inline]
    fn dist_sq(&self, q: &[f32], id: NodeId) -> f32 {
        euclidean_distance_sq(q, self.graph.vector_at(id))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.len() == 0
    }

    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    fn random_level(&mut self) -> usize {
        use rand::Rng;
        let mut level = ((-self.rng.gen::<f64>().ln()) * self.ml).floor() as i32;
        if level < 0 {
            level = 0;
        }
        level.min(32) as usize
    }

    /// Insert a vector; returns internal id (`0 .. self.len()-1` after insert).
    pub fn insert(&mut self, vector: &[f32]) -> NodeId {
        assert_eq!(
            vector.len(),
            self.dim,
            "vector dim {} does not match index dim {}",
            vector.len(),
            self.dim
        );

        // select a random level for the new node
        let level = self.random_level();
        let q_buf = vector;
        let new_id = self
            .graph
            .push_node(vector, level)
            .expect("node graph chunk allocation failed (mmap / OOM)");

        // Store the entry point, e.g., the first point to search
        if self.entry_point.is_none() {
            self.entry_point = Some(new_id);
            self.max_depth = level as i32;
            return new_id;
        }

        let mut ep = self.entry_point.expect("non-empty");

        {
            let q = q_buf;
            let mut lc = self.max_depth;
            // for all the levels above the new node's level, find the closest node greedily
            while lc > level as i32 {
                // find the closest node from the entry point to the current level
                ep = self.greedy_closest(q, ep, lc as usize);
                // move to the next level
                lc -= 1;
            }

            for lc in (0..=level).rev() {
                // find the ef closest candidates for the current level
                let candidates = if lc <= self.max_depth as usize {
                    self.search_level(q, ep, self.ef_construction, lc)
                } else {
                    Vec::new()
                };
                // retain only the m closest candidates
                let cap = if lc == 0 { self.m_max0 } else { self.m };
                let selected = (self.closest_m_candidates)(&candidates, cap);
                // create bidirectional edges between the new node and the selected candidates
                self.update_neighbors(new_id, &selected, lc);

                if let Some(&best) = selected.first() {
                    ep = best;
                }
            }
        }

        if level as i32 > self.max_depth {
            self.max_depth = level as i32;
            self.entry_point = Some(new_id);
        }

        new_id
    }

    /// k-NN search: returns up to `k` pairs `(internal_id, distance_sq)` sorted by distance.
    /// `ef` is the dynamic list size at level 0 (must be ≥ `k`).
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(NodeId, f32)> {
        assert_eq!(query.len(), self.dim);
        assert!(k > 0);
        let ef = ef.max(k);

        if self.is_empty() {
            return Vec::new();
        }

        let mut ep = self.entry_point.expect("non-empty");
        let mut lc = self.max_depth;

        while lc > 0 {
            ep = self.greedy_closest(query, ep, lc as usize);
            lc -= 1;
        }

        let mut cands = self.search_level(query, ep, ef, 0);
        cands.sort_by(|a, b| a.1.total_cmp(&b.1));
        cands.truncate(k);
        cands
    }

    fn greedy_closest(&self, q: &[f32], mut best: NodeId, level: usize) -> NodeId {
        let mut best_d = self.dist_sq(q, best);
        loop {
            let mut improved = false;
            for &nb in self.graph.neighbors_at(best, level) {
                if nb == INVALID_NODE_ID {
                    continue;
                }
                let d = self.dist_sq(q, nb);
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    fn search_level(&self, q: &[f32], ep: NodeId, ef: usize, level: usize) -> Vec<(NodeId, f32)> {
        let mut visited = HashSet::with_capacity(2 * ef);
        visited.insert(ep);

        let d0 = self.dist_sq(q, ep);
        let mut w: BinaryHeap<(OrdF32, NodeId)> = BinaryHeap::new();
        w.push((OrdF32(d0), ep));
        let mut candidates: BinaryHeap<Reverse<(OrdF32, NodeId)>> = BinaryHeap::new();
        candidates.push(Reverse((OrdF32(d0), ep)));

        while let Some(Reverse((dc, c))) = candidates.pop() {
            if w.len() == ef {
                let worst = w.peek().unwrap().0.inner();
                if dc.inner() > worst {
                    break;
                }
            }
            for &e in self.graph.neighbors_at(c, level) {
                if e == INVALID_NODE_ID {
                    continue;
                }
                if visited.insert(e) {
                    let de = self.dist_sq(q, e);
                    if w.len() < ef {
                        w.push((OrdF32(de), e));
                        candidates.push(Reverse((OrdF32(de), e)));
                    } else {
                        let worst = w.peek().unwrap().0.inner();
                        if de < worst {
                            w.pop();
                            w.push((OrdF32(de), e));
                            candidates.push(Reverse((OrdF32(de), e)));
                        }
                    }
                }
            }
        }

        w.into_iter().map(|(d, i)| (i, d.inner())).collect()
    }

    // connect the node to the neighbors at the given level
    fn update_neighbors(&mut self, nid: NodeId, neighbors: &[NodeId], level: usize) {
        let graph = &mut self.graph;
        graph.ensure_level(nid, level);
        let cap = if level == 0 { self.m_max0 } else { self.m };
        assert!(
            neighbors.len() <= cap,
            "neighbors length {} exceeds capacity {}",
            neighbors.len(),
            cap
        );

        let mut good_neighbors: Vec<NodeId> = Vec::with_capacity(neighbors.len());
        // update the edges from the neighbors to the current node at the given level
        for &nb in neighbors {
            graph.ensure_level(nb, level);
            let added = graph.add_directed_edge(nb, nid, level, euclidean_distance_sq);

            if added {
                good_neighbors.push(nb);
            }
        }
        // save the neighbors of the current node at the given level, e.g., the edges from this node to the neighbors
        graph.save_neighbors(nid, good_neighbors.as_slice(), level);
    }
}

#[cfg(test)]
mod tests {
    use common::top_k_quickselect;

    use super::super::{HnswArena, HnswNaive};
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn hnsw_naive_recalls_bruteforce_small() {
        let dim = 8usize;
        let rng = StdRng::seed_from_u64(42);
        let mut index: HnswNaive = HnswNaive::new(dim, 8, 16, 128, top_k_quickselect, rng);

        let n = 200usize;
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.03).sin())
                .collect();
            index.insert(v.as_slice());
        }

        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).cos()).collect();

        let mut brute: Vec<(NodeId, f32)> = (0..n)
            .map(|i| {
                let id = NodeId(i as u32);
                (id, euclidean_distance_sq(&query, index.vector_at(id)))
            })
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt: Vec<NodeId> = brute.iter().take(10).map(|(i, _)| *i).collect();

        let got = index.search(&query, 10, 128);
        let got_ids: Vec<NodeId> = got.iter().map(|(i, _)| *i).collect();

        let hit = gt.iter().filter(|id| got_ids.contains(id)).count();
        let recall = hit as f32 / 10.0;
        assert!(
            recall >= 0.8,
            "recall@10 vs brute expected high, got {recall} gt={gt:?} got={got_ids:?}"
        );
    }

    #[test]
    fn hnsw_naive_first_insert_is_entry() {
        let rng = StdRng::seed_from_u64(1);
        let mut index: HnswNaive = HnswNaive::new(4, 4, 8, 32, top_k_quickselect, rng);
        let id = index.insert(vec![1.0, 0.0, 0.0, 0.0].as_slice());
        assert_eq!(id, NodeId(0));
        assert_eq!(index.entry_point, Some(NodeId(0)));
        let r = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, NodeId(0));
    }

    #[test]
    fn hnsw_arena_recalls_bruteforce_small() {
        let dim = 8usize;
        let rng = StdRng::seed_from_u64(42);
        let mut index: HnswArena = HnswArena::new(dim, 8, 16, 128, 1024, top_k_quickselect, rng);

        let n = 200usize;
        let mut inserted: Vec<NodeId> = Vec::with_capacity(n);
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.03).sin())
                .collect();
            inserted.push(index.insert(v.as_slice()));
        }

        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).cos()).collect();

        let mut brute: Vec<(NodeId, f32)> = inserted
            .iter()
            .map(|&id| (id, euclidean_distance_sq(&query, index.vector_at(id))))
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt: Vec<NodeId> = brute.iter().take(10).map(|(i, _)| *i).collect();

        let got = index.search(&query, 10, 128);
        let got_ids: Vec<NodeId> = got.iter().map(|(i, _)| *i).collect();

        let hit = gt.iter().filter(|id| got_ids.contains(id)).count();
        let recall = hit as f32 / 10.0;
        assert!(
            recall >= 0.8,
            "recall@10 vs brute expected high, got {recall} gt={gt:?} got={got_ids:?}"
        );
    }

    #[test]
    fn hnsw_arena_first_insert_is_entry() {
        let rng = StdRng::seed_from_u64(1);
        let mut index: HnswArena = HnswArena::new(4, 4, 8, 32, 32, top_k_quickselect, rng);
        let id = index.insert(vec![1.0, 0.0, 0.0, 0.0].as_slice());
        assert_eq!(id, NodeId(0));
        assert_eq!(index.entry_point, Some(NodeId(0)));
        let r = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, NodeId(0));
    }

    /// Matches SIFT recall test scale (10k × dim 128, same RNG seed) to catch arena block / id bugs.
    #[test]
    fn hnsw_arena_stress_10k_sift_params() {
        const DIM: usize = 128;
        const M: usize = 16;
        const M_MAX0: usize = 32;
        const N: usize = 10_000;
        const EF: usize = 200;
        let rng = StdRng::seed_from_u64(0x_4853_4E57_5F53_4954);
        let mut index: HnswArena = HnswArena::new(DIM, M, M_MAX0, EF, N, top_k_quickselect, rng);
        for i in 0..N {
            let v: Vec<f32> = (0..DIM)
                .map(|j| ((i * DIM + j) as f32 * 0.03).sin())
                .collect();
            index.insert(v.as_slice());
        }
        let q: Vec<f32> = (0..DIM).map(|j| (j as f32 * 0.1).cos()).collect();
        let _ = index.search(&q, 10, 100);
    }
}
