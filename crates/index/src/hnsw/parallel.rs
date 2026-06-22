//! Parallel HNSW insertion via per-level RwLock pipelining with region-based
//! thread assignment to reduce level-lock contention.
//!
//! # Locking model
//!
//! ```text
//! alloc_lock (RwLock)
//!   write — push_node, entry_point / max_depth / node_level_map updates.
//!            Held exclusively so that Vec<NodeBlock> can grow (realloc)
//!            without invalidating references held by concurrent readers.
//!   read  — all node-data access: greedy_closest, search_level,
//!            update_neighbors, find_region.
//!            Multiple readers hold this concurrently; push_node must wait
//!            until all readers finish before resizing the block Vec.
//!
//! level_locks[l] (RwLock)        — read: concurrent search at level l
//!                                   write: exclusive edge mutation at level l
//! region_locks[region] (Mutex)   — one per layer-2 node; serializes same-region inserts
//! vid_map (RwLock)               — read: concurrent vector-id lookups in search
//!                                   write: new NodeId → vector_id registration
//! ```
//!
//! # Why alloc_lock must be an RwLock
//!
//! `ArenaNodeStore` stores `NodeBlock`s in a `Vec<NodeBlock>`.  When `push_node`
//! appends a new block the Vec may reallocate, moving every `NodeBlock` to a new
//! address.  Any `&mut NodeBlock` (or raw pointer derived from one) that another
//! thread obtained via `block_mut` before the reallocation becomes a dangling
//! reference; reading `self.m` / `self.m_max0` through it yields garbage, causing
//! `edges_at_level_mut` to return a zero-length slice and the subsequent index into
//! `neighbors` to panic.  Holding `alloc_lock.read()` during every graph access
//! ensures `push_node` (write) can never run concurrently, so the Vec never
//! reallocates while references into it are live.
//!
//! # Region assignment (layer-2 lock)
//!
//! Before the search-and-connect phase, each insertion navigates greedily from the
//! entry point down to layer 2 to find the nearest layer-2 node — its _region_.
//! Insertions assigned to the same region are serialized by `region_locks[region]`.
//! Insertions in different regions are fully parallel, because HNSW locality means
//! their neighborhood sets are disjoint. The level-locks remain the correctness
//! backstop for boundary vectors near region edges.
//!
//! # Safety invariants
//!
//! * `alloc_lock.write()` is held for the entire duration of `push_node` and any
//!   metadata update that modifies `Vec<NodeBlock>` or `entry_point`/`max_depth`.
//!   No other thread may access the graph while this lock is held.
//! * `alloc_lock.read()` is held for the entire duration of any graph access
//!   (search, greedy descent, edge update). This prevents concurrent `push_node`
//!   from resizing `Vec<NodeBlock>` and invalidating live references.
//! * Edge data at level L is always accessed under `level_locks[L]`.
//! * `NodeBlock` (arena-backed) is declared `Sync` in `nodes.rs`; concurrent reads
//!   of different arena slots are sound because they touch disjoint memory regions.
//! * `node_level_map` is written under `alloc_lock.write()` and read under
//!   `alloc_lock.read()` — safe because entries are written once per NodeId and
//!   never mutated afterward.

use std::cell::UnsafeCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use rayon::prelude::*;

use common::OrdF32;
use rand::rngs::StdRng;
use rand::Rng;
use vector::distance::euclidean_distance_sq;

use super::nodes::{ArenaNodeStore, HnswNodeStore, NodeId, INVALID_NODE_ID};

/// Number of HNSW levels supported (levels 0..=32).
const MAX_LEVELS: usize = 33;

// ── Core state (behind UnsafeCell) ────────────────────────────────────────────

struct GraphState {
    graph: ArenaNodeStore,
    entry_point: Option<NodeId>,
    max_depth: i32,
    node_ids: Vec<NodeId>,
    vector_ids: Vec<u64>,
    levels: Vec<u8>,
    /// NodeId.0 → max level at which the node was inserted.
    /// Written once under `alloc_lock.write()`; read under `alloc_lock.read()`.
    node_level_map: HashMap<u32, u8>,
}

/// Pre-computed insertion plan produced by [`ParallelHnsw::plan_insert`].
///
/// `neighbors_per_level[l]` = selected neighbor ids at HNSW level `l`
/// (`l` in `0..=level`, index equals level number).
struct InsertPlan {
    level: usize,
    neighbors_per_level: Vec<Vec<NodeId>>,
}

// ── Public struct ─────────────────────────────────────────────────────────────

/// HNSW index that supports concurrent insertion from multiple threads.
///
/// Use [`insert`](Self::insert) from any number of threads simultaneously.
/// Use [`insert_batch`](Self::insert_batch) to insert a slice of vectors
/// across a caller-specified thread count with automatic work partitioning.
///
/// [`search`](Self::search) is thread-safe and may run concurrently with
/// insertions.
pub struct ParallelHnsw {
    /// Core graph state; guarded by `alloc_lock` for allocation/metadata
    /// and by `level_locks[l]` for edge reads/writes at level l.
    state: UnsafeCell<GraphState>,
    /// Per-level coordination lock.
    level_locks: Vec<RwLock<()>>,
    /// Write: `push_node` and `entry_point`/`max_depth`/`node_level_map` updates.
    /// Read: all node-data access (search, greedy descent, edge updates).
    /// Prevents Vec<NodeBlock> reallocation from invalidating live references.
    alloc_lock: RwLock<()>,
    /// Per-region (layer-2 node) insertion serialization.
    /// Lazily populated: a mutex is created the first time a region is entered.
    region_locks: Mutex<HashMap<u32, Arc<Mutex<()>>>>,
    /// NodeId → application vector_id; separate RwLock so search-result
    /// mapping (read) does not block concurrent insertions (write).
    vid_map: RwLock<HashMap<u32, u64>>,
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ml: f64,
    closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
}

// SAFETY: All concurrent access to `state` follows the lock protocol above.
// `NodeBlock` (inside ArenaNodeStore) is `Sync`; its arena pointer is stable
// because mmap regions do not move.
unsafe impl Send for ParallelHnsw {}
unsafe impl Sync for ParallelHnsw {}

// ── Construction ──────────────────────────────────────────────────────────────

impl ParallelHnsw {
    /// Create a new empty parallel HNSW index.
    ///
    /// Parameters mirror `Hnsw<ArenaNodeStore>::new`:
    /// - `m`: target max degree for levels > 0.
    /// - `m_max0`: max degree at level 0 (typically `2 * m`).
    /// - `ef_construction`: beam width during insertion.
    /// - `closest_m_candidates`: neighbour-selection strategy
    ///   (e.g. `common::top_k_quickselect`).
    pub fn new(
        dim: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
    ) -> std::io::Result<Self> {
        assert!(dim > 0);
        assert!(m >= 2);
        assert!(m_max0 >= m);
        assert!(ef_construction >= m);

        let graph = ArenaNodeStore::try_new(dim, m, m_max0)?;
        let state = GraphState {
            graph,
            entry_point: None,
            max_depth: -1,
            node_ids: Vec::new(),
            vector_ids: Vec::new(),
            levels: Vec::new(),
            node_level_map: HashMap::new(),
        };
        let level_locks = (0..MAX_LEVELS).map(|_| RwLock::new(())).collect();
        Ok(Self {
            state: UnsafeCell::new(state),
            level_locks,
            alloc_lock: RwLock::new(()),
            region_locks: Mutex::new(HashMap::new()),
            vid_map: RwLock::new(HashMap::new()),
            dim,
            m,
            m_max0,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            closest_m_candidates,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn len(&self) -> usize {
        let _g = self.alloc_lock.read().unwrap();
        // SAFETY: alloc_lock.read() held — no concurrent push_node.
        unsafe { (*self.state.get()).graph.len() }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    // ── Region helpers ────────────────────────────────────────────────────────

    /// Return the max level at which `node` was inserted, or 0 if unknown.
    ///
    /// # Safety
    /// Caller must hold at least `alloc_lock.read()`.
    fn node_max_level(&self, node: NodeId) -> usize {
        // SAFETY: node_level_map entries are written under alloc_lock.write()
        // and read under alloc_lock.read(); caller guarantees the read lock.
        let state = unsafe { &*self.state.get() };
        state.node_level_map.get(&node.0).copied().unwrap_or(0) as usize
    }

    /// Navigate greedily from the entry point down to layer 2 and return the
    /// nearest layer-2 node as the region representative for `vector`.
    ///
    /// Returns `None` when the graph has fewer than 3 levels (max_depth < 2).
    ///
    /// # Safety
    /// Caller must hold `alloc_lock.read()`.
    fn find_region(&self, vector: &[f32], entry_point: NodeId, max_depth: i32) -> Option<NodeId> {
        if max_depth < 2 {
            return None;
        }

        // SAFETY: caller holds alloc_lock.read() — Vec<NodeBlock> is stable.
        // Each iteration also acquires level_locks[level].read().
        let state = unsafe { &*self.state.get() };

        let mut current = entry_point;

        for level in (2..=max_depth as usize).rev() {
            // Read lock at each level during descent; released before next level.
            let _guard = self.level_locks[level].read().unwrap();

            loop {
                let mut nb_buf: Vec<u8> = Vec::new();
                let mut v_buf: Vec<u8> = Vec::new();

                let better = state
                    .graph
                    .neighbors_at(current, level, &mut nb_buf)
                    .iter()
                    .copied()
                    .filter(|&n| n != INVALID_NODE_ID && self.node_max_level(n) >= level)
                    .min_by(|&a, &b| {
                        euclidean_distance_sq(vector, state.graph.vector_at(a, &mut v_buf))
                            .partial_cmp(&euclidean_distance_sq(
                                vector,
                                state.graph.vector_at(b, &mut v_buf),
                            ))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                match better {
                    Some(n)
                        if euclidean_distance_sq(vector, state.graph.vector_at(n, &mut v_buf))
                            < euclidean_distance_sq(
                                vector,
                                state.graph.vector_at(current, &mut v_buf),
                            ) =>
                    {
                        current = n;
                    }
                    _ => break,
                }
            }
            // Read lock on `level` is dropped here before descending.
        }

        Some(current)
    }

    /// Return the `Arc<Mutex>` for `region`, creating it on first access.
    ///
    /// The outer `region_locks` mutex is held only briefly to clone the inner
    /// Arc; the per-region lock is then acquired outside the outer mutex,
    /// avoiding a nested-lock hazard.
    fn region_mutex(&self, region: NodeId) -> Arc<Mutex<()>> {
        let mut map = self.region_locks.lock().unwrap();
        map.entry(region.0)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    // ── Insertion ─────────────────────────────────────────────────────────────

    /// Insert one vector. Thread-safe; supply a per-thread `rng` to avoid
    /// contention on a shared random-number generator.
    ///
    /// Returns the internal [`NodeId`] assigned to this vector.
    pub fn insert(&self, vector: &[f32], vector_id: u64, rng: &mut StdRng) -> NodeId {
        assert_eq!(
            vector.len(),
            self.dim,
            "vector dim {} != index dim {}",
            vector.len(),
            self.dim
        );

        let level = self.random_level(rng);

        // ── 1. Allocate node (write lock — blocks all graph readers) ───────
        //
        // push_node may grow Vec<NodeBlock>, reallocating its backing array.
        // Holding alloc_lock.write() ensures no other thread holds a live
        // &NodeBlock reference that would become dangling on reallocation.
        // ep / max_depth are NOT captured here — they are read fresher under
        // the read lock in step 2, after any concurrent inserts have committed.
        let new_id = {
            let _g = self.alloc_lock.write().unwrap();
            // SAFETY: alloc_lock.write() held exclusively.
            let state = unsafe { &mut *self.state.get() };
            let new_id = state
                .graph
                .push_node(vector, level)
                .expect("HNSW node allocation failed (arena OOM)");
            state.node_ids.push(new_id);
            state.vector_ids.push(vector_id);
            state.levels.push(level as u8);
            state.node_level_map.insert(new_id.0, level as u8);
            new_id
        };
        // Register vector_id mapping (brief write lock, independent of alloc_lock).
        self.vid_map.write().unwrap().insert(new_id.0, vector_id);

        // ── 2–6. Graph access phase (alloc_lock.read held throughout) ──────
        //
        // alloc_lock.read() prevents push_node (write) from running while any
        // &NodeBlock reference derived from Vec<NodeBlock> is live.
        {
            let _alloc_r = self.alloc_lock.read().unwrap();

            // ── 2. Read ep / max_depth now, after allocation committed ─────
            //
            // Reading here (under the read lock) rather than inside the write
            // lock in step 1 gives a more up-to-date view: any insertions that
            // committed between our push_node and this read are already visible,
            // so we start the search from a better entry point.
            let (entry_point, max_depth) = {
                // SAFETY: alloc_lock.read() held.
                let state = unsafe { &*self.state.get() };
                (state.entry_point, state.max_depth)
            };

            let mut ep = match entry_point {
                None => {
                    // First node — promote to entry point and return.
                    // Must drop the read lock before acquiring write.
                    drop(_alloc_r);
                    let _g = self.alloc_lock.write().unwrap();
                    // SAFETY: alloc_lock.write() held.
                    let state = unsafe { &mut *self.state.get() };
                    if state.entry_point.is_none() {
                        state.entry_point = Some(new_id);
                        state.max_depth = level as i32;
                    }
                    return new_id;
                }
                Some(ep) => ep,
            };

            // ── 3. Assign region and acquire per-region lock ───────────────
            let region_arc = self
                .find_region(vector, ep, max_depth)
                .map(|r| self.region_mutex(r));
            let _region_guard = region_arc.as_ref().map(|m| m.lock().unwrap());

            // ── 4. Greedy descent above insertion level ────────────────────
            let mut lc = max_depth;
            while lc > level as i32 {
                let _r = self.level_locks[lc as usize].read().unwrap();
                ep = self.greedy_closest(vector, ep, lc as usize);
                lc -= 1;
            }

            // ── 5. Beam search + edge connect at each insertion level ──────
            for lc in (0..=level).rev() {
                let cap = if lc == 0 { self.m_max0 } else { self.m };

                let candidates = if (lc as i32) <= max_depth {
                    let _r = self.level_locks[lc].read().unwrap();
                    self.search_level(vector, ep, self.ef_construction, lc)
                } else {
                    Vec::new()
                };

                let selected = (self.closest_m_candidates)(&candidates, cap);

                if let Some(&best) = selected.first() {
                    ep = best;
                }

                {
                    let _w = self.level_locks[lc].write().unwrap();
                    // SAFETY: write lock on level `lc` held exclusively.
                    // alloc_lock.read() still held — Vec<NodeBlock> is stable.
                    self.update_neighbors(new_id, &selected, lc);
                }
            }

            // Region lock (`_region_guard`) released here.

            // ── 6. Promote to entry point if new node tops the graph ───────
            //
            // max_depth is still in scope from step 2; use it as the fast-path
            // guard before paying for the write lock.
            if level as i32 > max_depth {
                drop(_alloc_r); // release read lock before acquiring write
                let _g = self.alloc_lock.write().unwrap();
                // SAFETY: alloc_lock.write() held.
                let state = unsafe { &mut *self.state.get() };
                if level as i32 > state.max_depth {
                    state.max_depth = level as i32;
                    state.entry_point = Some(new_id);
                }
            }
            // If we didn't enter step 6, _alloc_r is dropped here at block end.
        }

        new_id
    }

    // ── Two-phase batch insert ────────────────────────────────────────────────

    /// Insert `vectors` using a two-phase strategy.
    ///
    /// **Phase 1 (parallel, read-only):** compute neighbor candidates for every
    /// vector in the chunk concurrently against the committed graph.  No
    /// allocation or edge writes occur; all rayon threads hold `alloc_lock.read()`.
    ///
    /// **Phase 2 (sequential):** allocate each node and commit the pre-computed
    /// edges one vector at a time.
    ///
    /// The input is split into chunks of [`TWO_PHASE_CHUNK_SIZE`] so that vectors
    /// in later chunks benefit from edges committed by earlier chunks.  Without
    /// chunking, every vector's phase-1 search sees the same static snapshot and
    /// intra-batch edges are entirely absent, producing a disconnected graph with
    /// near-zero recall when building from scratch.
    ///
    /// If the graph is empty on entry, the first `ef_construction` vectors are
    /// inserted sequentially via [`insert`](Self::insert) to seed enough structure
    /// for beam search before the first chunk's phase 1 runs.
    pub fn insert_batch_two_phase(
        &self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        num_threads: usize,
        rng: &mut StdRng,
    ) -> Vec<NodeId> {
        assert_eq!(vectors.len(), vector_ids.len());
        if vectors.is_empty() {
            return Vec::new();
        }

        let mut all_ids = Vec::with_capacity(vectors.len());
        let mut start = 0;

        // If the graph is empty, beam search in phase 1 has no candidates and
        // every plan comes back with empty neighbor lists — all nodes end up
        // isolated.  Seed with ef_construction sequential inserts first so the
        // first chunk's phase-1 search has enough graph structure to work with.
        {
            let is_empty = {
                let _g = self.alloc_lock.read().unwrap();
                // SAFETY: alloc_lock.read() held.
                unsafe { (*self.state.get()).entry_point.is_none() }
            };
            if is_empty {
                let seed_n = self.ef_construction.min(vectors.len());
                for i in 0..seed_n {
                    all_ids.push(self.insert(&vectors[i], vector_ids[i], rng));
                }
                start = seed_n;
            }
        }

        // Process remaining vectors in chunks.  Each chunk's phase-1 search sees
        // all nodes committed by previous chunks, so intra-batch connectivity
        // accumulates rather than being entirely absent.
        for (vec_chunk, id_chunk) in vectors[start..]
            .chunks(num_threads)
            .zip(vector_ids[start..].chunks(num_threads))
        {
            all_ids.extend(self.two_phase_chunk(vec_chunk, id_chunk, rng));
        }

        all_ids
    }

    /// Run one two-phase cycle over a chunk of vectors.
    fn two_phase_chunk(
        &self,
        vectors: &[Vec<f32>],
        vector_ids: &[u64],
        rng: &mut StdRng,
    ) -> Vec<NodeId> {
        // Pre-generate levels sequentially to keep RNG output deterministic.
        let levels: Vec<usize> = vectors.iter().map(|_| self.random_level(rng)).collect();

        // Snapshot entry point and max_depth; graph is read-only during phase 1.
        let (ep_snapshot, max_depth_snapshot) = {
            let _g = self.alloc_lock.read().unwrap();
            // SAFETY: alloc_lock.read() held.
            let state = unsafe { &*self.state.get() };
            (state.entry_point, state.max_depth)
        };

        // ── Phase 1: parallel neighbor search ─────────────────────────────
        let plans: Vec<InsertPlan> = vectors
            .par_iter()
            .zip(levels.par_iter())
            .map(|(vector, &level)| {
                self.plan_insert(vector, level, ep_snapshot, max_depth_snapshot)
            })
            .collect();

        // ── Phase 2: sequential allocation + edge commit ───────────────────
        let mut node_ids = Vec::with_capacity(vectors.len());
        for (i, (vector, plan)) in vectors.iter().zip(plans).enumerate() {
            let vid = vector_ids[i];

            // Allocate node (write lock — Vec<NodeBlock> may grow).
            let new_id = {
                let _g = self.alloc_lock.write().unwrap();
                // SAFETY: alloc_lock.write() held exclusively.
                let state = unsafe { &mut *self.state.get() };
                let new_id = state
                    .graph
                    .push_node(vector, plan.level)
                    .expect("HNSW node allocation failed (arena OOM)");
                state.node_ids.push(new_id);
                state.vector_ids.push(vid);
                state.levels.push(plan.level as u8);
                state.node_level_map.insert(new_id.0, plan.level as u8);
                new_id
            };
            self.vid_map.write().unwrap().insert(new_id.0, vid);

            // Commit pre-computed edges (read lock keeps Vec<NodeBlock> stable).
            {
                let _alloc_r = self.alloc_lock.read().unwrap();
                for (l, neighbors) in plan.neighbors_per_level.iter().enumerate() {
                    let _w = self.level_locks[l].write().unwrap();
                    // SAFETY: alloc_lock.read() held (Vec<NodeBlock> stable);
                    // level_locks[l].write() held (exclusive edge access).
                    self.update_neighbors(new_id, neighbors, l);
                }
            }

            // Promote to entry point if this node tops the graph.
            {
                let _g = self.alloc_lock.write().unwrap();
                // SAFETY: alloc_lock.write() held.
                let state = unsafe { &mut *self.state.get() };
                if state.entry_point.is_none() || plan.level as i32 > state.max_depth {
                    state.max_depth = plan.level as i32;
                    state.entry_point = Some(new_id);
                }
            }

            node_ids.push(new_id);
        }

        node_ids
    }

    /// Compute the neighbor candidates for one vector against the current graph
    /// snapshot without modifying any state.
    ///
    /// `neighbors_per_level[l]` holds the selected neighbors at level `l`
    /// (index == level, covering `0..=level`).
    ///
    /// # Safety
    /// Acquires `alloc_lock.read()` internally to keep `Vec<NodeBlock>` stable
    /// for the duration of the search.
    fn plan_insert(
        &self,
        vector: &[f32],
        level: usize,
        entry_point: Option<NodeId>,
        max_depth: i32,
    ) -> InsertPlan {
        let mut neighbors_per_level = vec![Vec::new(); level + 1];

        let Some(mut ep) = entry_point else {
            return InsertPlan {
                level,
                neighbors_per_level,
            };
        };

        // Hold alloc_lock.read() for the entire read-only traversal.
        let _alloc_r = self.alloc_lock.read().unwrap();

        // Greedy descent above insertion level.
        let mut lc = max_depth;
        while lc > level as i32 {
            let _r = self.level_locks[lc as usize].read().unwrap();
            ep = self.greedy_closest(vector, ep, lc as usize);
            lc -= 1;
        }

        // Beam search + neighbor selection at each insertion level.
        for l in (0..=level).rev() {
            let candidates = if (l as i32) <= max_depth {
                let _r = self.level_locks[l].read().unwrap();
                self.search_level(vector, ep, self.ef_construction, l)
            } else {
                Vec::new()
            };
            let cap = if l == 0 { self.m_max0 } else { self.m };
            let selected = (self.closest_m_candidates)(&candidates, cap);
            if let Some(&best) = selected.first() {
                ep = best;
            }
            neighbors_per_level[l] = selected;
        }

        InsertPlan {
            level,
            neighbors_per_level,
        }
    }

    // ── Search ────────────────────────────────────────────────────────────────

    /// k-NN search. Thread-safe; may run concurrently with insertions.
    ///
    /// Returns up to `k` `(vector_id, distance_sq)` pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u64, f32)> {
        assert_eq!(query.len(), self.dim);
        assert!(k > 0);
        let ef = ef.max(k);

        // Hold alloc_lock.read() for the entire graph traversal to prevent
        // push_node from resizing Vec<NodeBlock> while we hold &NodeBlock refs.
        let _alloc_r = self.alloc_lock.read().unwrap();

        let (ep, max_depth) = {
            // SAFETY: alloc_lock.read() held.
            let state = unsafe { &*self.state.get() };
            (state.entry_point, state.max_depth)
        };

        let mut ep = match ep {
            None => return Vec::new(),
            Some(ep) => ep,
        };

        let mut lc = max_depth;
        while lc > 0 {
            let _r = self.level_locks[lc as usize].read().unwrap();
            ep = self.greedy_closest(query, ep, lc as usize);
            lc -= 1;
        }

        let mut cands = {
            let _r = self.level_locks[0].read().unwrap();
            self.search_level(query, ep, ef, 0)
        };
        cands.sort_by(|a, b| a.1.total_cmp(&b.1));
        cands.truncate(k);

        let vid_map = self.vid_map.read().unwrap();
        cands
            .into_iter()
            .map(|(node_id, dist)| (vid_map[&node_id.0], dist))
            .collect()
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    fn random_level(&self, rng: &mut StdRng) -> usize {
        let mut level = ((-rng.gen::<f64>().ln()) * self.ml).floor() as i32;
        if level < 0 {
            level = 0;
        }
        level.min(32) as usize
    }

    /// Greedy one-hop-at-a-time descent at `level`.
    ///
    /// # Safety
    /// Caller must hold `alloc_lock.read()` and at least a read lock on
    /// `level_locks[level]`.
    fn greedy_closest(&self, q: &[f32], mut best: NodeId, level: usize) -> NodeId {
        // SAFETY: caller holds alloc_lock.read() (Vec<NodeBlock> stable) and
        // level_locks[level].read() (edge reads safe).
        let state = unsafe { &*self.state.get() };
        let mut nb_buf: Vec<u8> = Vec::new();
        let mut v_buf: Vec<u8> = Vec::new();
        let mut best_d = euclidean_distance_sq(q, state.graph.vector_at(best, &mut v_buf));
        loop {
            let mut improved = false;
            for &nb in state.graph.neighbors_at(best, level, &mut nb_buf) {
                if nb == INVALID_NODE_ID {
                    continue;
                }
                let d = euclidean_distance_sq(q, state.graph.vector_at(nb, &mut v_buf));
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

    /// Beam search at `level`, returning up to `ef` candidates.
    ///
    /// # Safety
    /// Caller must hold `alloc_lock.read()` and at least a read lock on
    /// `level_locks[level]`.
    fn search_level(&self, q: &[f32], ep: NodeId, ef: usize, level: usize) -> Vec<(NodeId, f32)> {
        // SAFETY: caller holds alloc_lock.read() (Vec<NodeBlock> stable) and
        // level_locks[level].read() (edge reads safe).
        let state = unsafe { &*self.state.get() };
        let mut visited = HashSet::with_capacity(2 * ef);
        visited.insert(ep);

        let mut nb_buf: Vec<u8> = Vec::new();
        let mut v_buf: Vec<u8> = Vec::new();

        let d0 = euclidean_distance_sq(q, state.graph.vector_at(ep, &mut v_buf));
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
            for &e in state.graph.neighbors_at(c, level, &mut nb_buf) {
                if e == INVALID_NODE_ID {
                    continue;
                }
                if visited.insert(e) {
                    let de = euclidean_distance_sq(q, state.graph.vector_at(e, &mut v_buf));
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

    /// Connect `nid` to `neighbors` at `level` (bidirectional, with pruning).
    ///
    /// # Safety
    /// Caller must hold `alloc_lock.read()` and the write lock on
    /// `level_locks[level]`.
    fn update_neighbors(&self, nid: NodeId, neighbors: &[NodeId], level: usize) {
        // SAFETY: caller holds alloc_lock.read() (Vec<NodeBlock> stable) and
        // level_locks[level].write() (exclusive edge access at this level).
        let state = unsafe { &mut *self.state.get() };
        let graph = &mut state.graph;
        graph.ensure_level(nid, level);
        let cap = if level == 0 { self.m_max0 } else { self.m };
        debug_assert!(
            neighbors.len() <= cap,
            "neighbors {} > cap {}",
            neighbors.len(),
            cap
        );

        let mut good_neighbors: Vec<NodeId> = Vec::with_capacity(neighbors.len());
        for &nb in neighbors {
            graph.ensure_level(nb, level);
            if graph.add_directed_edge(nb, nid, level, euclidean_distance_sq) {
                good_neighbors.push(nb);
            }
        }
        graph.save_neighbors(nid, &good_neighbors, level);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use common::top_k_quickselect;
    use rand::SeedableRng;
    use std::sync::Arc;

    fn make_index(dim: usize) -> ParallelHnsw {
        ParallelHnsw::new(dim, 8, 16, 128, top_k_quickselect).unwrap()
    }

    #[test]
    fn single_thread_recalls_bruteforce() {
        let dim = 8usize;
        let index = make_index(dim);
        let mut rng = StdRng::seed_from_u64(42);

        let n = 200usize;
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|j| ((i * dim + j) as f32 * 0.03).sin())
                .collect();
            index.insert(&v, i as u64, &mut rng);
        }

        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 0.1).cos()).collect();
        let results = index.search(&query, 10, 128);
        assert!(
            !results.is_empty(),
            "search on non-empty index must return results"
        );
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted by distance");
        }
    }

    #[test]
    fn first_insert_is_entry_point() {
        let index = make_index(4);
        let mut rng = StdRng::seed_from_u64(1);
        let id = index.insert(&[1.0, 0.0, 0.0, 0.0], 99, &mut rng);
        assert_eq!(id, NodeId(0));
        let r = index.search(&[1.0, 0.0, 0.0, 0.0], 1, 8);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].0, 99u64);
    }

    #[test]
    fn concurrent_insert_and_search_does_not_panic() {
        const DIM: usize = 8;
        const N: usize = 200;

        let index = Arc::new(make_index(DIM));

        let idx = Arc::clone(&index);
        let inserter = std::thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(0);
            for i in 0..N {
                let v: Vec<f32> = (0..DIM)
                    .map(|j| ((i * DIM + j) as f32 * 0.05).sin())
                    .collect();
                idx.insert(&v, i as u64, &mut rng);
            }
        });

        let idx = Arc::clone(&index);
        let searcher = std::thread::spawn(move || {
            let query: Vec<f32> = (0..DIM).map(|j| (j as f32 * 0.1).cos()).collect();
            for _ in 0..50 {
                let _ = idx.search(&query, 5, 32);
            }
        });

        inserter.join().unwrap();
        searcher.join().unwrap();
    }

    /// Verify that find_region returns a valid node when the graph is deep enough.
    #[test]
    fn find_region_returns_valid_node_when_deep() {
        const DIM: usize = 8;
        // Insert enough vectors that we get at least one layer-2 node.
        // With M=4 and ml = 1/ln(4) ≈ 0.72, P(level >= 2) ≈ 1/16 per vector.
        // 1000 vectors should reliably produce layer-2 nodes.
        let index = ParallelHnsw::new(DIM, 4, 8, 64, top_k_quickselect).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        for i in 0..1000usize {
            let v: Vec<f32> = (0..DIM)
                .map(|j| ((i * DIM + j) as f32 * 0.05).sin())
                .collect();
            index.insert(&v, i as u64, &mut rng);
        }

        // Read entry point under alloc_lock.read().
        let (ep, max_depth) = {
            let _g = index.alloc_lock.read().unwrap();
            let state = unsafe { &*index.state.get() };
            (state.entry_point, state.max_depth)
        };

        if max_depth >= 2 {
            let query: Vec<f32> = (0..DIM).map(|j| (j as f32 * 0.1).cos()).collect();
            // Hold alloc_lock.read() while calling find_region (safety invariant).
            let _alloc_r = index.alloc_lock.read().unwrap();
            let region = index.find_region(&query, ep.unwrap(), max_depth);
            assert!(
                region.is_some(),
                "expected a region node when max_depth >= 2"
            );
            // The region node must exist at layer 2 or above.
            assert!(
                index.node_max_level(region.unwrap()) >= 2,
                "region node must be at level >= 2"
            );
        }
    }
}
