//! Shared HNSW graph and search over [`super::store::HnswVectorStore`] and [`super::nodes::HnswNodeStore`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher as Crc32Hasher;

use common::OrdF32;
use rand::rngs::StdRng;
use vector::distance::euclidean_distance_sq;

use super::nodes::{ArenaNodeStore, HnswNodeStore, NaiveNodeStore, NodeId, INVALID_NODE_ID};

/// Façade over [`Hnsw`] (`insert` / `search`).
/// [`HnswNaive`] / [`HnswArena`] can be used as `dyn HnswIndex`.
pub trait HnswIndex {
    fn len(&self) -> usize;
    fn insert(&mut self, vector: &[f32], vector_id: u64) -> NodeId;
    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)>;
    /// Move the index's hot data to disk under `dir`. Returns the number of underlying
    /// storage units (e.g. arena node blocks) that transitioned to on-disk state.
    /// No-op for indexes without arena-backed storage (e.g. naive heap).
    fn swap_out(&mut self, dir: &std::path::Path) -> std::io::Result<usize>;
    /// Restore on-disk storage units to memory. Returns the number restored.
    fn swap_in(&mut self) -> std::io::Result<usize>;
    /// Drop the local backing for every storage unit (closes open fds and releases
    /// arena memory). Bytes must be restored via [`Self::swap_in_from`] before the
    /// next read. Returns the number of units transitioned to evicted.
    fn evict(&mut self) -> usize;
    /// Restore every storage unit by reading `dir/block_<i>.arena`. Inverse of
    /// [`Self::swap_out`] but accepts arbitrary paths so the bytes can come from a
    /// fresh download (e.g. blob storage). Returns the number of units restored.
    fn swap_in_from(&mut self, dir: &std::path::Path) -> std::io::Result<usize>;
    /// Serialize `node_ids` and `levels` to `path`. See [`Hnsw::save_levels`].
    fn save_levels(&self, path: &std::path::Path) -> std::io::Result<()>;
    /// Deserialize `node_ids` and `levels` from `path`. See [`Hnsw::load_levels`].
    fn load_levels(&mut self, path: &std::path::Path) -> std::io::Result<()>;
    /// Write a manifest capturing entry point, max layer, and arena file names.
    fn save_manifest(&self, path: &std::path::Path) -> std::io::Result<()>;
    /// Restore entry point and max layer from a manifest written by [`save_manifest`].
    fn load_manifest(&mut self, path: &std::path::Path) -> std::io::Result<()>;
    /// Drop the in-memory `node_ids` and `levels` buffers after they have been persisted.
    fn clear_level_data(&mut self);
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
    pub node_ids: Vec<NodeId>,
    pub vector_ids: Vec<u64>,
    pub levels: Vec<u8>,
    node_to_vector_id: HashMap<u32, u64>,
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
            node_ids: Vec::new(),
            vector_ids: Vec::new(),
            levels: Vec::new(),
            node_to_vector_id: HashMap::new(),
        }
    }
}

impl<N: HnswNodeStore> HnswIndex for Hnsw<N> {
    fn len(&self) -> usize {
        Hnsw::len(self)
    }

    fn insert(&mut self, vector: &[f32], vector_id: u64) -> NodeId {
        // Call inherent `Hnsw::insert`, not this trait method (same name would recurse).
        Hnsw::insert(self, vector, vector_id)
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        Hnsw::search(self, query, k, ef)
    }

    fn swap_out(&mut self, dir: &std::path::Path) -> std::io::Result<usize> {
        self.graph.swap_out(dir)
    }

    fn swap_in(&mut self) -> std::io::Result<usize> {
        self.graph.swap_in()
    }

    fn evict(&mut self) -> usize {
        self.graph.evict()
    }

    fn swap_in_from(&mut self, dir: &std::path::Path) -> std::io::Result<usize> {
        self.graph.swap_in_from(dir)
    }

    fn save_levels(&self, path: &std::path::Path) -> std::io::Result<()> {
        Hnsw::save_levels(self, path)
    }

    fn load_levels(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        Hnsw::load_levels(self, path)
    }

    fn save_manifest(&self, path: &std::path::Path) -> std::io::Result<()> {
        Hnsw::save_manifest(self, path)
    }

    fn load_manifest(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        Hnsw::load_manifest(self, path)
    }

    fn clear_level_data(&mut self) {
        Hnsw::clear_level_data(self)
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
            node_ids: Vec::new(),
            vector_ids: Vec::new(),
            levels: Vec::new(),
            node_to_vector_id: HashMap::new(),
        }
    }
}

impl<N: HnswNodeStore> Hnsw<N> {
    /// Lookup stored vector (internal id). `buf` is scratch for the on-disk path;
    /// in-memory reads ignore it.
    #[inline]
    pub fn vector_at<'a>(&'a self, id: NodeId, buf: &'a mut Vec<u8>) -> &'a [f32] {
        self.graph.vector_at(id, buf)
    }

    #[inline]
    fn dist_sq(&self, q: &[f32], id: NodeId, buf: &mut Vec<u8>) -> f32 {
        euclidean_distance_sq(q, self.graph.vector_at(id, buf))
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

    /// Insert a vector with its external application id; returns the internal [`NodeId`].
    pub fn insert(&mut self, vector: &[f32], vector_id: u64) -> NodeId {
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
        self.node_ids.push(new_id);
        self.vector_ids.push(vector_id);
        self.levels.push(level as u8);
        self.node_to_vector_id.insert(new_id.0, vector_id);

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
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
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
            .into_iter()
            .map(|(node_id, dist)| {
                let vid = self.node_to_vector_id[&node_id.0];
                (vid, dist)
            })
            .collect()
    }

    fn greedy_closest(&self, q: &[f32], mut best: NodeId, level: usize) -> NodeId {
        // Independent scratch buffers: `neighbor_buf` is borrowed for the duration of the
        // for-loop iteration; `vector_buf` is needed inside the body by `dist_sq`. They
        // cannot share storage because the neighbor slice borrow is live during dist_sq.
        let mut neighbor_buf: Vec<u8> = Vec::new();
        let mut vector_buf: Vec<u8> = Vec::new();
        let mut best_d = self.dist_sq(q, best, &mut vector_buf);
        loop {
            let mut improved = false;
            for &nb in self.graph.neighbors_at(best, level, &mut neighbor_buf) {
                if nb == INVALID_NODE_ID {
                    continue;
                }
                let d = self.dist_sq(q, nb, &mut vector_buf);
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

        let mut neighbor_buf: Vec<u8> = Vec::new();
        let mut vector_buf: Vec<u8> = Vec::new();

        let d0 = self.dist_sq(q, ep, &mut vector_buf);
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
            for &e in self.graph.neighbors_at(c, level, &mut neighbor_buf) {
                if e == INVALID_NODE_ID {
                    continue;
                }
                if visited.insert(e) {
                    let de = self.dist_sq(q, e, &mut vector_buf);
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

    /// Write `node_ids` and `levels` to `path`.
    ///
    /// Layout (all little-endian):
    /// ```text
    /// version    : u32
    /// count      : u64
    /// node_ids   : [u32; count]
    /// vector_ids : [u64; count]
    /// levels     : [u8;  count]
    /// crc32      : u32            — CRC32 of all preceding bytes
    /// ```
    pub fn save_levels(&self, path: impl AsRef<Path>) -> io::Result<()> {
        const VERSION: u32 = 2;
        let count = self.node_ids.len() as u64;
        let mut hasher = Crc32Hasher::new();
        let mut w = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?,
        );
        let mut write = |bytes: &[u8]| -> io::Result<()> {
            hasher.update(bytes);
            w.write_all(bytes)
        };
        write(&VERSION.to_le_bytes())?;
        write(&count.to_le_bytes())?;
        for id in &self.node_ids {
            write(&id.0.to_le_bytes())?;
        }
        for vid in &self.vector_ids {
            write(&vid.to_le_bytes())?;
        }
        write(&self.levels)?;
        w.write_all(&hasher.finalize().to_le_bytes())?;
        w.flush()
    }

    /// Read `node_ids`, `levels`, and `vector_ids` from a file written by [`save_levels`].
    /// Replaces the current contents of all three lists.
    pub fn load_levels(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        const EXPECTED_VERSION: u32 = 2;
        let mut f = File::open(path)?;
        let file_len = f.metadata()?.len() as usize;
        if file_len < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "levels file too small",
            ));
        }
        let mut data = vec![0u8; file_len - 4];
        f.read_exact(&mut data)?;
        let mut crc_buf = [0u8; 4];
        f.read_exact(&mut crc_buf)?;
        let stored_crc = u32::from_le_bytes(crc_buf);
        let mut hasher = Crc32Hasher::new();
        hasher.update(&data);
        let computed_crc = hasher.finalize();
        if computed_crc != stored_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "levels CRC32 mismatch: expected {stored_crc:#010x}, got {computed_crc:#010x}"
                ),
            ));
        }
        let mut pos = 0;
        let version = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if version != EXPECTED_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported levels file version {version}"),
            ));
        }
        let count = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        let mut node_ids = Vec::with_capacity(count);
        for _ in 0..count {
            node_ids.push(NodeId(u32::from_le_bytes(
                data[pos..pos + 4].try_into().unwrap(),
            )));
            pos += 4;
        }
        let mut vector_ids = Vec::with_capacity(count);
        for _ in 0..count {
            vector_ids.push(u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()));
            pos += 8;
        }
        let levels = data[pos..pos + count].to_vec();
        pos += count;
        let mut node_to_vector_id = HashMap::with_capacity(count);
        for (nid, &vid) in node_ids.iter().zip(vector_ids.iter()) {
            node_to_vector_id.insert(nid.0, vid);
        }
        self.node_ids = node_ids;
        self.vector_ids = vector_ids;
        self.levels = levels;
        self.node_to_vector_id = node_to_vector_id;
        Ok(())
    }

    /// Drop the in-memory `node_ids` and `levels` buffers, reclaiming their heap.
    /// Called after both have been persisted to disk by [`save_levels`].
    pub fn clear_level_data(&mut self) {
        self.node_ids = Vec::new();
        self.vector_ids = Vec::new();
        self.levels = Vec::new();
        // node_to_vector_id is intentionally kept: needed for search after swap-out
    }

    /// Write a manifest for this index to `path` as JSON.
    ///
    /// ```json
    /// {
    ///   "version": 1,
    ///   "entry_point": 7,
    ///   "max_layer": 2,
    ///   "arena_files": ["block_0.arena", "block_1.arena"]
    /// }
    /// ```
    /// `entry_point` is `null` when the index is empty.
    pub fn save_manifest(&self, path: impl AsRef<Path>) -> io::Result<()> {
        #[derive(serde::Serialize)]
        struct Manifest {
            version: u32,
            entry_point: Option<u32>,
            max_layer: i32,
            arena_files: Vec<String>,
        }
        let m = Manifest {
            version: 1,
            entry_point: self.entry_point.map(|id| id.0),
            max_layer: self.max_depth,
            arena_files: self.graph.arena_file_names(),
        };
        let json = serde_json::to_string_pretty(&m)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut f = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        f.write_all(json.as_bytes())?;
        f.flush()
    }

    /// Restore entry point and max layer from a manifest written by [`save_manifest`].
    pub fn load_manifest(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
        #[derive(serde::Deserialize)]
        struct Manifest {
            version: u32,
            entry_point: Option<u32>,
            max_layer: i32,
            #[allow(dead_code)]
            arena_files: Vec<String>,
        }
        let text = std::fs::read_to_string(path)?;
        let m: Manifest = serde_json::from_str(&text)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if m.version != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported manifest version {}", m.version),
            ));
        }
        self.entry_point = m.entry_point.map(NodeId);
        self.max_depth = m.max_layer;
        Ok(())
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
            index.insert(v.as_slice(), i as u64);
        }

        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).cos()).collect();

        let mut buf: Vec<u8> = Vec::new();
        // Naive NodeId == insertion index, so vector_id == i as u64.
        let mut brute: Vec<(u64, f32)> = (0..n)
            .map(|i| {
                let id = NodeId(i as u32);
                (
                    i as u64,
                    euclidean_distance_sq(&query, index.vector_at(id, &mut buf)),
                )
            })
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt: Vec<u64> = brute.iter().take(10).map(|(i, _)| *i).collect();

        let got = index.search(&query, 10, 128);
        let got_ids: Vec<u64> = got.iter().map(|(i, _)| *i).collect();

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
        let id = index.insert(vec![1.0, 0.0, 0.0, 0.0].as_slice(), 0u64);
        assert_eq!(id, NodeId(0));
        assert_eq!(index.entry_point, Some(NodeId(0)));
        let r = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0u64);
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
            inserted.push(index.insert(v.as_slice(), i as u64));
        }

        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).cos()).collect();

        let mut buf: Vec<u8> = Vec::new();
        // inserted[i] is the NodeId for vector_id i (we insert with i as u64).
        let mut brute: Vec<(u64, f32)> = inserted
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                (
                    i as u64,
                    euclidean_distance_sq(&query, index.vector_at(id, &mut buf)),
                )
            })
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1));
        let gt: Vec<u64> = brute.iter().take(10).map(|(i, _)| *i).collect();

        let got = index.search(&query, 10, 128);
        let got_ids: Vec<u64> = got.iter().map(|(i, _)| *i).collect();

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
        let id = index.insert(vec![1.0, 0.0, 0.0, 0.0].as_slice(), 0u64);
        assert_eq!(id, NodeId(0));
        assert_eq!(index.entry_point, Some(NodeId(0)));
        let r = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 0u64);
    }

    #[test]
    fn save_load_levels_roundtrip() {
        let rng = StdRng::seed_from_u64(7);
        let mut index: HnswNaive = HnswNaive::new(4, 4, 8, 32, top_k_quickselect, rng);
        for i in 0..20u32 {
            index.insert(&[i as f32, 0.0, 0.0, 0.0], i as u64);
        }
        let dir = std::env::temp_dir();
        let path = dir.join("levels_roundtrip_test.bin");
        index.save_levels(&path).unwrap();

        let mut index2: HnswNaive =
            HnswNaive::new(4, 4, 8, 32, top_k_quickselect, StdRng::seed_from_u64(0));
        index2.load_levels(&path).unwrap();
        assert_eq!(index.node_ids, index2.node_ids);
        assert_eq!(index.levels, index2.levels);
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
            index.insert(v.as_slice(), i as u64);
        }
        let q: Vec<f32> = (0..DIM).map(|j| (j as f32 * 0.1).cos()).collect();
        let _ = index.search(&q, 10, 100);
    }
}
