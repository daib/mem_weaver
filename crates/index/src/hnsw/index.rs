//! Shared HNSW graph and search over [`super::store::HnswVectorStore`] and [`super::nodes::HnswNodeStore`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt;

use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

use crc32fast::Hasher as Crc32Hasher;

use common::OrdF32;
use rand::rngs::StdRng;
use vector::distance::euclidean_distance_sq;

use super::nodes::{ArenaNodeStore, HnswNodeStore, NaiveNodeStore, NodeId, INVALID_NODE_ID};

// ── Two-phase batch insertion ─────────────────────────────────────────────────

/// Pre-computed insertion plan from [`Hnsw::plan_insert`].
/// `neighbors_per_level[l]` = `(neighbor_id, dist_sq)` pairs at level `l`.
struct InsertPlan {
    level: usize,
    neighbors_per_level: Vec<Vec<(NodeId, f32)>>,
}

const DEGREE_RATIO_CAP: f32 = 0.8;
const MIN_GREEDY_SEARCH_LEVEL: i32 = 3;
const BEAM_SEARCH_EF: usize = 5;

/// Return `true` if any already-committed batch vector is closer to one of
/// `plan`'s level-0 neighbors than `plan`'s own recorded distance.
fn has_closer_committed(plan: &InsertPlan, committed_l0: &[HashMap<NodeId, f32>]) -> bool {
    let Some(l0) = plan.neighbors_per_level.first() else {
        return false;
    };
    for &(n, v_dist) in l0 {
        for committed in committed_l0 {
            if let Some(&u_dist) = committed.get(&n) {
                if u_dist < v_dist {
                    return true;
                }
            }
        }
    }
    false
}

#[inline]
fn degree_penalized_score(dist: f32, neighbors: &[NodeId]) -> f32 {
    let degree = neighbors
        .iter()
        .rposition(|&nb| nb != INVALID_NODE_ID)
        .map_or(0, |i| i + 1);
    dist * (1.0 + (degree as f32 / neighbors.len() as f32) * DEGREE_RATIO_CAP)
}

// ── Public trait ──────────────────────────────────────────────────────────────

/// Façade over [`Hnsw`] (`insert` / `search`).
/// [`HnswNaive`] / [`HnswArena`] can be used as `dyn HnswIndex`.
pub trait HnswIndex {
    fn len(&self) -> usize;
    /// Clear all nodes, edges, and metadata, returning the index to its empty initial state.
    fn reset(&mut self);
    fn insert(&mut self, vector: &[f32], vector_id: u64) -> NodeId;
    fn insert_batch_parallel(
        &mut self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        num_threads: usize,
    ) -> Vec<NodeId>;

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)>;
    /// Populate the index's arena from `block_*.arena` files in `dir`, replacing any
    /// existing blocks. Used during crash recovery. No-op for non-arena indexes.
    fn load_blocks_from_dir(&mut self, dir: &std::path::Path) -> std::io::Result<usize>;
    /// Recompute per-block node counts from the in-memory `node_ids` loaded by
    /// [`load_levels`]. Must be called after both [`load_blocks_from_dir`] and
    /// [`load_levels`] so that `len()` / `is_empty()` return correct values.
    fn rebuild_lens(&mut self);
    /// Returns `true` if any arena block has been written to since the last successful
    /// snapshot upload. Always `false` for non-arena indexes.
    fn has_dirty_blocks(&self) -> bool;
    /// Returns the write count at the time of the snapshot. Capture under the same
    /// read lock as the snapshot and pass to [`mark_clean_after_snapshot_if_version`].
    fn snapshot_write_count(&self) -> u64;
    /// Number of vectors inserted since the last successful snapshot upload.
    fn dirty_vector_count(&self) -> u64;
    /// Mark all arena blocks as clean after a successful snapshot upload, but only if
    /// no writes have occurred since `version` was captured. This prevents clearing
    /// dirty on blocks that were written between the snapshot and the upload completion.
    fn mark_clean_after_snapshot_if_version(&mut self, version: u64);
    /// Mark all arena blocks as clean unconditionally. Prefer
    /// [`mark_clean_after_snapshot_if_version`] where possible.
    fn mark_clean_after_snapshot(&mut self);
    /// Copy in-memory arena blocks to `dir` without changing storage state. Returns the
    /// number of blocks written. Output format matches [`swap_out`], so the files can be
    /// uploaded to S3 and later restored via [`swap_in_from`]. No-op for non-arena indexes.
    fn snapshot_to_dir(&self, dir: &std::path::Path) -> std::io::Result<usize>;
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
    /// Return the application-level vector IDs of every node in this index.
    /// Populated after [`load_levels`]; empty after [`clear_level_data`].
    fn vector_ids(&self) -> &[u64];
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

// SAFETY: `rng: StdRng` is `!Sync`, but it is only accessed sequentially.
// The rayon phase-1 par_iter is read-only and never touches `rng`. Callers
// must not invoke any mutating method concurrently with phase-1 reads.
unsafe impl<N: HnswNodeStore + Send + Sync> Sync for Hnsw<N> {}

impl Hnsw<NaiveNodeStore> {
    /// `m`: target max degree on levels `> 0`. Level 0 allows up to `m_max0` (often `2 * m`).
    ///
    /// Graph uses heap [`NaiveNodeStore`] (naive vector + naive node storage).
    pub fn new(dim: usize, m: usize, m_max0: usize, ef_construction: usize, rng: StdRng) -> Self {
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
            rng,
            node_ids: Vec::new(),
            vector_ids: Vec::new(),
            levels: Vec::new(),
            node_to_vector_id: HashMap::new(),
        }
    }
}

impl<N: HnswNodeStore + Send + Sync> HnswIndex for Hnsw<N> {
    fn len(&self) -> usize {
        Hnsw::len(self)
    }

    fn reset(&mut self) {
        self.graph.reset();
        self.entry_point = None;
        self.max_depth = -1;
        self.node_ids.clear();
        self.vector_ids.clear();
        self.levels.clear();
        self.node_to_vector_id.clear();
    }

    fn insert(&mut self, vector: &[f32], vector_id: u64) -> NodeId {
        // Call inherent `Hnsw::insert`, not this trait method (same name would recurse).
        Hnsw::insert(self, vector, vector_id)
    }

    fn insert_batch_parallel(
        &mut self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        num_threads: usize,
    ) -> Vec<NodeId> {
        Hnsw::insert_batch_parallel(self, vectors, vector_ids, num_threads)
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        Hnsw::search(self, query, k, ef)
    }

    fn load_blocks_from_dir(&mut self, dir: &std::path::Path) -> std::io::Result<usize> {
        self.graph.load_from_dir(dir)
    }

    fn rebuild_lens(&mut self) {
        self.graph.rebuild_lens_from_node_ids(&self.node_ids);
    }

    fn has_dirty_blocks(&self) -> bool {
        self.graph.has_dirty_blocks()
    }

    fn snapshot_write_count(&self) -> u64 {
        self.graph.write_count()
    }

    fn dirty_vector_count(&self) -> u64 {
        self.graph.dirty_vector_count()
    }

    fn mark_clean_after_snapshot_if_version(&mut self, version: u64) {
        self.graph.mark_clean_if_version(version);
    }

    fn mark_clean_after_snapshot(&mut self) {
        self.graph.mark_all_clean();
    }

    fn snapshot_to_dir(&self, dir: &std::path::Path) -> std::io::Result<usize> {
        self.graph.snapshot_to_dir(dir)
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

    fn vector_ids(&self) -> &[u64] {
        &self.vector_ids
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
        let new_id = self
            .graph
            .push_node(vector, level)
            .expect("node graph chunk allocation failed (mmap / OOM)");
        self.node_ids.push(new_id);
        self.vector_ids.push(vector_id);
        self.levels.push(level as u8);
        self.node_to_vector_id.insert(new_id.0, vector_id);

        let plan = self.plan_insert(vector, level);
        self.commit_plan(new_id, &plan);

        if self.entry_point.is_none() || level as i32 > self.max_depth {
            self.max_depth = level as i32;
            self.entry_point = Some(new_id);
        }

        new_id
    }

    /// k-NN search: returns up to `k` pairs `(vector_id, distance_sq)` sorted by distance.
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

        while lc > MIN_GREEDY_SEARCH_LEVEL {
            ep = self.greedy_closest(query, ep, lc as usize);
            lc -= 1;
        }

        for l in (0..=lc).rev() {
            let this_ef = if l == 0 { ef } else { BEAM_SEARCH_EF };
            // perform beam search at higher levels to avoid local minima
            let mut cands = self.search_level(query, ep, this_ef, l as usize);
            if l > 0 {
                ep = cands.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
            } else {
                if cands.len() > k {
                    cands.select_nth_unstable_by(k, |a, b| a.1.total_cmp(&b.1));
                    cands.truncate(k);
                }
                cands.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                return cands
                    .into_iter()
                    .map(|(node_id, dist)| {
                        let vid = self.node_to_vector_id[&node_id.0];
                        (vid, dist)
                    })
                    .collect();
            }
        }
        return Vec::new();
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

    // Return the closest `ef` neighbors of `ep` at `level` in the graph with node ids and distances.
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

    // ── Two-phase batch insertion ─────────────────────────────────────────────

    /// Insert `vectors` using a two-phase strategy with rayon parallelism.
    ///
    /// Phase 1 (parallel, read-only): compute neighbor candidates for every
    /// vector in each chunk concurrently. Phase 2 (sequential): allocate nodes
    /// and commit pre-computed edges, deferring vectors whose plans are stale.
    ///

    pub fn insert_batch_parallel(
        &mut self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        num_threads: usize,
    ) -> Vec<NodeId>
    where
        N: Send + Sync,
    {
        assert_eq!(vectors.len(), vector_ids.len());
        if vectors.is_empty() {
            return Vec::new();
        }

        let mut all_ids = Vec::with_capacity(vectors.len());
        let mut start = 0;

        // Seed with ef_construction sequential inserts so phase-1 beam search
        // has enough graph structure to work with.
        if self.entry_point.is_none() {
            let seed_n = self.ef_construction.min(vectors.len());
            for i in 0..seed_n {
                all_ids.push(self.insert(&vectors[i], vector_ids[i]));
            }
            start = seed_n;
        }

        for (vec_chunk, id_chunk) in vectors[start..]
            .chunks(num_threads)
            .zip(vector_ids[start..].chunks(num_threads))
        {
            all_ids.extend(self.two_phase_parallel(vec_chunk, id_chunk));
        }

        all_ids
    }

    fn two_phase_parallel(&mut self, vectors: &[Vec<f32>], vector_ids: &[u64]) -> Vec<NodeId>
    where
        N: Send + Sync,
    {
        let levels: Vec<usize> = vectors.iter().map(|_| self.random_level()).collect();

        // Phase 1: parallel neighbor search (read-only).
        // SAFETY: the closure only reads graph data; no mutation happens until
        // phase 2a begins after collect() returns.
        let s: &Self = self;
        let plans: Vec<InsertPlan> = vectors
            .par_iter()
            .zip(levels.par_iter())
            .map(|(vector, &level)| s.plan_insert(vector, level))
            .collect();

        // Phase 2a: sequential node allocation.
        let mut node_ids = Vec::with_capacity(vectors.len());
        for (i, (vector, plan)) in vectors.iter().zip(plans.iter()).enumerate() {
            let vid = vector_ids[i];
            let new_id = self
                .graph
                .push_node(vector, plan.level)
                .expect("HNSW node allocation failed (arena OOM)");
            self.node_ids.push(new_id);
            self.vector_ids.push(vid);
            self.levels.push(plan.level as u8);
            self.node_to_vector_id.insert(new_id.0, vid);
            node_ids.push(new_id);
        }

        // Phase 2b: commit edges with conflict detection.
        let mut committed_l0: Vec<HashMap<NodeId, f32>> = Vec::new();
        let mut deferred: Vec<usize> = Vec::new();

        for i in 0..vectors.len() {
            if has_closer_committed(&plans[i], &committed_l0) {
                deferred.push(i);
                continue;
            }
            self.commit_plan(node_ids[i], &plans[i]);
            if plans[i].level as i32 > self.max_depth {
                self.max_depth = plans[i].level as i32;
                self.entry_point = Some(node_ids[i]);
            }
            committed_l0.push(
                plans[i]
                    .neighbors_per_level
                    .first()
                    .map(|l0| l0.iter().cloned().collect())
                    .unwrap_or_default(),
            );
        }

        // Phase 2c: re-plan deferred vectors against the committed graph.
        for &i in &deferred {
            let fresh = self.plan_insert(&vectors[i], plans[i].level);
            self.commit_plan(node_ids[i], &fresh);
            if fresh.level as i32 > self.max_depth {
                self.max_depth = fresh.level as i32;
                self.entry_point = Some(node_ids[i]);
            }
        }

        node_ids
    }

    fn plan_insert(&self, vector: &[f32], level: usize) -> InsertPlan {
        let mut neighbors_per_level = vec![Vec::new(); level + 1];

        let Some(mut ep) = self.entry_point else {
            return InsertPlan {
                level,
                neighbors_per_level,
            };
        };

        let mut lc = self.max_depth;
        // for all the levels above the new node's level, find the closest node greedily
        while lc > level as i32 {
            // find the closest node from the entry point to the current level
            ep = self.greedy_closest(vector, ep, lc as usize);
            // move to the next level
            lc -= 1;
        }

        // Beam search + neighbor selection at each level.
        let mut node_buf: Vec<u8> = Vec::new();
        for l in (0..=level).rev() {
            let candidates = if (l as i32) <= self.max_depth {
                self.search_level(vector, ep, self.ef_construction, l)
            } else {
                Vec::new()
            };
            let cap = if l == 0 { self.m_max0 } else { self.m };

            // Score each candidate by distance penalised by its current degree:
            //   score = dist * (1 + degree / cap)
            // Candidates that already carry many edges pay a higher effective
            // distance, steering new connections toward less-loaded nodes.
            let mut scored: Vec<(NodeId, f32)> = candidates
                .iter()
                .map(|&(n, dist)| {
                    let neighbors = self.graph.neighbors_at(n, l, &mut node_buf);
                    let score = degree_penalized_score(dist, neighbors);
                    (n, score)
                })
                .collect();

            let selected_len = cap.min(scored.len());
            if selected_len < scored.len() {
                scored.select_nth_unstable_by(selected_len, |a, b| a.1.total_cmp(&b.1));
                scored.truncate(selected_len);
            }

            if let Some((best, _)) = scored.iter().min_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                ep = *best;
            }
            neighbors_per_level[l] = scored;
        }

        InsertPlan {
            level,
            neighbors_per_level,
        }
    }

    fn commit_plan(&mut self, nid: NodeId, plan: &InsertPlan) {
        for (l, neighbors) in plan.neighbors_per_level.iter().enumerate() {
            let neighbor_ids: Vec<NodeId> = neighbors.iter().map(|&(n, _)| n).collect();
            self.update_neighbors(nid, &neighbor_ids, l);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{HnswArena, HnswNaive};
    use super::*;
    use rand::SeedableRng;

    fn make_plan(level: usize, l0_neighbors: Vec<(NodeId, f32)>) -> InsertPlan {
        let mut neighbors_per_level = vec![Vec::new(); level + 1];
        neighbors_per_level[0] = l0_neighbors;
        InsertPlan {
            level,
            neighbors_per_level,
        }
    }

    fn make_committed(entries: &[(NodeId, f32)]) -> HashMap<NodeId, f32> {
        entries.iter().cloned().collect()
    }

    #[test]
    fn degree_penalized_score_counts_valid_neighbors() {
        let invalid = INVALID_NODE_ID;
        let high_degree = [
            NodeId(0),
            NodeId(1),
            NodeId(2),
            NodeId(3),
            NodeId(4),
            NodeId(5),
            NodeId(6),
            NodeId(7),
            NodeId(8),
            NodeId(9),
        ];
        let low_degree = [
            NodeId(0),
            NodeId(1),
            NodeId(2),
            NodeId(3),
            NodeId(4),
            NodeId(5),
            NodeId(6),
            invalid,
            invalid,
            invalid,
        ];

        // 8 valid / cap 10 → ratio 0.8, clamped to 0.8 → 10 * 1.8 = 18
        assert_eq!(
            degree_penalized_score(10.0, &high_degree),
            10.0 * (1.0 + 1.0 * DEGREE_RATIO_CAP)
        );
        // 2 valid / cap 10 → ratio 0.2 → 10 * 1.2 = 12
        assert_eq!(
            degree_penalized_score(10.0, &low_degree),
            10.0 * (1.0 + 0.7 * DEGREE_RATIO_CAP)
        );
        // all invalid → degree 0 → no penalty
        assert_eq!(
            degree_penalized_score(
                10.0,
                &[
                    invalid, invalid, invalid, invalid, invalid, invalid, invalid, invalid,
                    invalid, invalid
                ]
            ),
            10.0 * (1.0 + 0.0 * DEGREE_RATIO_CAP)
        );
    }

    #[test]
    fn test_has_closer_committed() {
        // Empty committed slice.
        let plan = make_plan(0, vec![(NodeId(0), 1.0), (NodeId(1), 2.0)]);
        assert!(!has_closer_committed(&plan, &[]));

        // Empty plan neighbors.
        let plan_empty = make_plan(0, vec![]);
        assert!(!has_closer_committed(
            &plan_empty,
            &[make_committed(&[(NodeId(0), 0.5)])]
        ));

        // Committed shares no neighbors with the plan.
        let plan = make_plan(0, vec![(NodeId(0), 1.0)]);
        assert!(!has_closer_committed(
            &plan,
            &[make_committed(&[(NodeId(1), 0.5)])]
        ));

        // Committed shares a neighbor but is farther.
        assert!(!has_closer_committed(
            &plan,
            &[make_committed(&[(NodeId(0), 2.0)])]
        ));

        // Committed shares a neighbor and is closer.
        assert!(has_closer_committed(
            &plan,
            &[make_committed(&[(NodeId(0), 0.5)])]
        ));

        // Multiple neighbors and committed: no entry closer than the plan.
        let plan = make_plan(0, vec![(NodeId(0), 1.0), (NodeId(1), 2.0)]);
        let far_a = make_committed(&[(NodeId(0), 1.5)]);
        let far_b = make_committed(&[(NodeId(1), 3.0)]);
        assert!(!has_closer_committed(&plan, &[far_a, far_b]));

        // Multiple committed: second is closer to NodeId(1).
        let no_overlap = make_committed(&[(NodeId(0), 1.5)]);
        let closer = make_committed(&[(NodeId(1), 1.5)]);
        assert!(has_closer_committed(&plan, &[no_overlap, closer]));
    }

    #[test]
    fn hnsw_naive_recalls_bruteforce_small() {
        let dim = 8usize;
        let rng = StdRng::seed_from_u64(42);
        let mut index: HnswNaive = HnswNaive::new(dim, 8, 16, 128, rng);

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
        let mut index: HnswNaive = HnswNaive::new(4, 4, 8, 32, rng);
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
        let mut index: HnswArena = HnswArena::new(dim, 8, 16, 128, 1024, rng);

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
        let mut index: HnswArena = HnswArena::new(4, 4, 8, 32, 32, rng);
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
        let mut index: HnswNaive = HnswNaive::new(4, 4, 8, 32, rng);
        for i in 0..20u32 {
            index.insert(&[i as f32, 0.0, 0.0, 0.0], i as u64);
        }
        let dir = std::env::temp_dir();
        let path = dir.join("levels_roundtrip_test.bin");
        index.save_levels(&path).unwrap();

        let mut index2: HnswNaive = HnswNaive::new(4, 4, 8, 32, StdRng::seed_from_u64(0));
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
        let mut index: HnswArena = HnswArena::new(DIM, M, M_MAX0, EF, N, rng);
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
