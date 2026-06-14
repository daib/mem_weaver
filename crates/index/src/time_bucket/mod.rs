//! Time-bucketed HNSW index for streaming workloads.
//!
//! Vectors are partitioned into independent HNSW indexes called buckets. Each
//! bucket covers a fixed time window of `bucket_duration` units. Insertions
//! land in the current (newest) bucket; a new bucket is created automatically
//! when the insertion timestamp advances past the current window, or on an
//! explicit [`TimeBucketIndex::rotate_bucket`] call.
//!
//! Searching fans out across all buckets and merges results with optional
//! recency weighting: distances from older buckets are scaled up so that
//! semantically equivalent results from newer buckets rank higher.
//!
//! Eviction drops an entire bucket's arena in one operation — no per-object
//! cleanup, no fragmentation.

use std::path::PathBuf;
use std::{collections::VecDeque, time::Duration};

use common::Timestamp;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::hnsw::{HnswArena, HnswIndex, NodeId};

/// Default node-block capacity hint passed to [`HnswArena`] on bucket creation.
/// The arena auto-extends, so this is only a first-block sizing hint.
const NODE_BLOCK_CAPACITY: usize = 1024;

/// Errors returned by [`TimeBucketIndex::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    ZeroDim,
    /// `m` must be ≥ 2.
    MTooSmall,
    /// `m_max0` must be ≥ `m`.
    MMax0TooSmall,
    /// `ef_construction` must be ≥ `m`.
    EfConstructionTooSmall,
    ZeroBucketDuration,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDim => f.write_str("dim must be > 0"),
            Self::MTooSmall => f.write_str("m must be >= 2"),
            Self::MMax0TooSmall => f.write_str("m_max0 must be >= m"),
            Self::EfConstructionTooSmall => f.write_str("ef_construction must be >= m"),
            Self::ZeroBucketDuration => f.write_str("bucket_duration must be > 0"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// Monotonically increasing identifier assigned to each bucket on creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BucketSeq(pub u32);

/// Identifies a result uniquely across all time buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BucketedNodeId {
    /// Monotonically increasing sequence number assigned to the bucket on creation.
    pub bucket_seq: BucketSeq,
    /// Application-provided vector id passed to [`TimeBucketIndex::insert`].
    pub vector_id: u64,
}

struct Bucket {
    seq: BucketSeq,
    created_at: Timestamp,
    index: Box<dyn HnswIndex + Send + Sync>,
    /// Set by `swap_bucket_out`; consumed by `swap_bucket_in` to reload levels.
    swap_dir: Option<PathBuf>,
}

/// Multi-HNSW index partitioned into time buckets.
///
/// # Memory model
/// Each bucket owns a `Box<dyn HnswIndex>` (currently backed by [`HnswArena`]
/// and its mmap arenas). Dropping a bucket (via
/// [`evict_oldest`](TimeBucketIndex::evict_oldest) or
/// [`evict_before`](TimeBucketIndex::evict_before)) frees all of its memory
/// atomically without any per-node cleanup.
///
/// # Ordering
/// `buckets` is a deque where `front` = newest and `back` = oldest. This lets
/// push/search hit the hot front cheaply and eviction pop cheaply from the back.
pub struct TimeBucketIndex {
    dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    /// Length of each time window in the same units as the `timestamp` passed to
    /// [`insert`](TimeBucketIndex::insert).
    bucket_duration: Duration,
    closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
    buckets: VecDeque<Bucket>,
    next_seq: BucketSeq,
    rng: StdRng,
    /// Set of all vector IDs currently stored across all buckets. Used to deduplicate
    /// inserts — `insert` is a no-op for IDs already present, returning `None`.
    known_vector_ids: std::collections::HashSet<u64>,
}

impl TimeBucketIndex {
    /// Snap `timestamp` down to the nearest grid boundary.
    ///
    /// All bucket `created_at` values are set via this function, so the deque
    /// position of any timestamp can be derived with pure arithmetic:
    /// `deque_idx_from_back = (bucket_start(t) - back.created_at) / bucket_duration`.
    #[inline]
    fn bucket_start(&self, timestamp: Timestamp) -> Timestamp {
        let d = self.bucket_duration.as_secs();
        Timestamp((timestamp.0 / d) * d)
    }

    /// Create a new empty index.
    ///
    /// `bucket_duration`: length of each time window in the same units as the
    /// `timestamp` argument to [`insert`](TimeBucketIndex::insert). A new bucket
    /// is started when an insertion timestamp falls into a later grid slot than
    /// the current bucket.
    pub fn new(
        dim: usize,
        m: usize,
        m_max0: usize,
        ef_construction: usize,
        bucket_duration: Duration,
        closest_m_candidates: fn(&[(NodeId, f32)], usize) -> Vec<NodeId>,
        rng: StdRng,
    ) -> Result<Self, ConfigError> {
        if dim == 0 {
            return Err(ConfigError::ZeroDim);
        }
        if m < 2 {
            return Err(ConfigError::MTooSmall);
        }
        if m_max0 < m {
            return Err(ConfigError::MMax0TooSmall);
        }
        if ef_construction < m {
            return Err(ConfigError::EfConstructionTooSmall);
        }
        if bucket_duration.is_zero() {
            return Err(ConfigError::ZeroBucketDuration);
        }
        Ok(Self {
            dim,
            m,
            m_max0,
            ef_construction,
            bucket_duration,
            closest_m_candidates,
            buckets: VecDeque::new(),
            next_seq: BucketSeq(0),
            rng,
            known_vector_ids: std::collections::HashSet::new(),
        })
    }

    /// Returns `(dim, m, m_max0, ef_construction, bucket_duration)` — the configuration
    /// needed to recreate this index after a crash.
    pub fn config(&self) -> (usize, usize, usize, usize, Duration) {
        (
            self.dim,
            self.m,
            self.m_max0,
            self.ef_construction,
            self.bucket_duration,
        )
    }

    /// Returns the sequence numbers of all active buckets, newest first.
    pub fn bucket_seqs(&self) -> Vec<BucketSeq> {
        self.buckets.iter().map(|b| b.seq).collect()
    }

    /// Returns `(seq, created_at)` for every active bucket, newest first.
    pub fn bucket_metas(&self) -> Vec<(BucketSeq, Timestamp)> {
        self.buckets.iter().map(|b| (b.seq, b.created_at)).collect()
    }

    /// Total number of vectors inserted across all buckets since their last snapshot.
    /// Used to decide whether the collection has accumulated enough dirty vectors to
    /// justify a snapshot cycle.
    pub fn total_dirty_vector_count(&self) -> u64 {
        self.buckets
            .iter()
            .map(|b| b.index.dirty_vector_count())
            .sum()
    }

    /// Returns `true` if the bucket identified by `seq` has any arena blocks that
    /// have been written to since the last successful snapshot upload. Returns `false`
    /// if the bucket is not found, has no in-memory blocks, or all blocks are clean.
    pub fn is_bucket_dirty(&self, seq: BucketSeq) -> bool {
        self.buckets
            .iter()
            .find(|b| b.seq == seq)
            .map(|b| b.index.has_dirty_blocks())
            .unwrap_or(false)
    }

    /// Returns the write count for bucket `seq` at the current instant.
    /// Capture this under the same read lock used to take the snapshot, then pass
    /// to [`mark_bucket_clean_if_version`] after a successful upload.
    pub fn bucket_write_count(&self, seq: BucketSeq) -> u64 {
        self.buckets
            .iter()
            .find(|b| b.seq == seq)
            .map(|b| b.index.snapshot_write_count())
            .unwrap_or(0)
    }

    /// Mark all arena blocks in bucket `seq` as clean, but only if the write count
    /// matches `version`. If new vectors were inserted between the snapshot and the
    /// upload completion, the count will have advanced and the flag is left dirty so
    /// the next cycle re-uploads the updated arena.
    pub fn mark_bucket_clean_if_version(&mut self, seq: BucketSeq, version: u64) {
        if let Some(bucket) = self.buckets.iter_mut().find(|b| b.seq == seq) {
            bucket.index.mark_clean_after_snapshot_if_version(version);
        }
    }

    /// Mark all arena blocks in the bucket identified by `seq` as clean.
    /// Called after a successful snapshot upload so the next cycle can skip
    /// buckets with no new inserts.
    pub fn mark_bucket_clean(&mut self, seq: BucketSeq) {
        if let Some(bucket) = self.buckets.iter_mut().find(|b| b.seq == seq) {
            bucket.index.mark_clean_after_snapshot();
        }
    }

    /// Returns the grid-aligned creation timestamp of the bucket with `seq`, or
    /// `None` if no such bucket exists.
    pub fn bucket_created_at(&self, seq: BucketSeq) -> Option<Timestamp> {
        self.buckets
            .iter()
            .find(|b| b.seq == seq)
            .map(|b| b.created_at)
    }

    /// Returns `true` if a bucket with `seq` is tracked by this index.
    pub fn has_bucket(&self, seq: BucketSeq) -> bool {
        self.buckets.iter().any(|b| b.seq == seq)
    }

    /// Total number of vectors across all buckets.
    pub fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.index.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of active time buckets.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    fn alloc_bucket(&mut self, start: Timestamp) {
        let seq = self.next_seq;
        self.next_seq.0 += 1;
        let bucket_rng = StdRng::seed_from_u64(self.rng.gen());
        let index: Box<dyn HnswIndex + Send + Sync> = Box::new(HnswArena::new(
            self.dim,
            self.m,
            self.m_max0,
            self.ef_construction,
            NODE_BLOCK_CAPACITY,
            self.closest_m_candidates,
            bucket_rng,
        ));
        self.buckets.push_front(Bucket {
            seq,
            created_at: start,
            index,
            swap_dir: None,
        });
    }

    /// Insert `vector` into the bucket whose window covers `timestamp`.
    ///
    /// Returns `Some(BucketedNodeId)` on success, or `None` if `vector_id` is already
    /// present in this index (duplicate skipped). This makes inserts idempotent: a
    /// retry or WAL replay for a vector that was already captured in a snapshot will
    /// silently no-op rather than creating a second node with the same application ID.
    pub fn insert(
        &mut self,
        vector: &[f32],
        timestamp: Timestamp,
        vector_id: u64,
    ) -> Option<BucketedNodeId> {
        if !self.known_vector_ids.insert(vector_id) {
            return None; // already present
        }

        let start = self.bucket_start(timestamp);
        let needs_new = self.buckets.front().map_or(true, |b| start > b.created_at);
        if needs_new {
            self.alloc_bucket(start);
        }

        let bucket = self.buckets.front_mut().expect("just allocated");
        bucket.index.insert(vector, vector_id);
        Some(BucketedNodeId {
            bucket_seq: bucket.seq,
            vector_id,
        })
    }

    /// Force-start a new bucket at the grid slot for `timestamp`.
    ///
    /// If the current front bucket is empty its `created_at` is updated in place
    /// rather than leaving a zero-size bucket in the deque.
    pub fn rotate_bucket(&mut self, timestamp: Timestamp) {
        let start = self.bucket_start(timestamp);
        if self.buckets.front().map_or(false, |b| b.index.len() == 0) {
            self.buckets.front_mut().unwrap().created_at = start;
            return;
        }
        self.alloc_bucket(start);
    }

    /// Search buckets and return up to `k` results selected by `top_k_fn`.
    ///
    /// `time_range`: when `Some(start..end)`, only buckets whose window overlaps
    /// `[start, end)` are searched. The matching deque slice is derived from the
    /// grid-aligned bucket starts using pure arithmetic — no per-bucket scan.
    ///
    /// `adjust_distance_fn(created_at, dist)`: called for every candidate with the bucket's
    /// grid-aligned start time and the raw squared-euclidean distance. The returned
    /// value is used as-is in the global top-k merge, giving the caller full control
    /// over how temporal position influences ranking. Pass `|_, d| d` for a pure
    /// semantic search with no temporal bias.
    ///
    /// `top_k_fn`: selects the `k` results with smallest adjusted distance from the
    /// per-bucket candidate pool. Pass [`common::top_k_quickselect`] for the standard
    /// introselect-based selection instead of a full sort.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        adjust_distance_fn: impl Fn(Timestamp, f32) -> f32,
        time_range: Option<std::ops::Range<Timestamp>>,
        top_k_fn: fn(&[(BucketedNodeId, f32)], usize) -> Vec<BucketedNodeId>,
    ) -> Vec<(u64, f32)> {
        let n = self.buckets.len();
        if n == 0 {
            return Vec::new();
        }

        // Determine which contiguous slice of the deque to search.
        // Deque layout: index 0 = newest, index n-1 = oldest.
        let (deque_start, deque_end) = match time_range {
            None => (0, n),
            Some(range) => {
                if range.is_empty() {
                    return Vec::new();
                }
                let oldest_start = self.buckets.back().unwrap().created_at;
                let newest_start = self.buckets.front().unwrap().created_at;
                let d = self.bucket_duration.as_secs();

                // Grid-aligned bounds, clamped to the deque's extent.
                // range.end.0 >= 1 because range.is_empty() was checked above.
                let first_bucket = self.bucket_start(range.start).max(oldest_start);
                let last_bucket = self
                    .bucket_start(Timestamp(range.end.0 - 1))
                    .min(newest_start);

                if first_bucket > last_bucket {
                    return Vec::new();
                }

                // Convert grid positions to deque indices (newer = smaller index).
                let start = ((newest_start.0 - last_bucket.0) / d) as usize;
                let end = ((newest_start.0 - first_bucket.0) / d) as usize + 1;
                (start, end.min(n))
            }
        };

        let mut all: Vec<(BucketedNodeId, f32)> = Vec::with_capacity(k * (deque_end - deque_start));

        for i in deque_start..deque_end {
            let bucket = &self.buckets[i];
            for (vector_id, dist) in bucket.index.search(query, k, ef) {
                all.push((
                    BucketedNodeId {
                        bucket_seq: bucket.seq,
                        vector_id,
                    },
                    adjust_distance_fn(bucket.created_at, dist),
                ));
            }
        }

        let bids = top_k_fn(&all, k);
        let score_map: std::collections::HashMap<BucketedNodeId, f32> = all.into_iter().collect();
        bids.into_iter()
            .map(|bid| (bid.vector_id, score_map[&bid]))
            .collect()
    }

    /// Drop all buckets whose window starts strictly before `timestamp`.
    ///
    /// Because all `created_at` values are grid-aligned multiples of
    /// `bucket_duration`, the number of buckets to drop is derived with pure
    /// arithmetic — no per-bucket comparison loop. Each dropped bucket's arena
    /// is freed atomically.
    ///
    /// Returns the number of evicted buckets.
    pub fn evict_before(&mut self, timestamp: Timestamp) -> usize {
        let oldest_start = match self.buckets.back() {
            Some(b) => b.created_at,
            None => return 0,
        };
        if timestamp <= oldest_start {
            return 0;
        }
        let d = self.bucket_duration.as_secs();
        // Ceiling division: count of grid slots in [oldest_start, timestamp).
        let count =
            ((timestamp.0 - oldest_start.0 + d - 1) / d).min(self.buckets.len() as u64) as usize;
        for _ in 0..count {
            self.buckets.pop_back();
        }
        count
    }

    /// Drop the single oldest bucket and return its sequence number.
    /// Returns `None` if the index is empty.
    pub fn evict_oldest(&mut self) -> Option<BucketSeq> {
        self.buckets.pop_back().map(|b| b.seq)
    }

    /// Write a point-in-time snapshot of the bucket identified by `seq` to `dir` without
    /// changing any storage state. The output is identical in format to what
    /// [`swap_bucket_out`](TimeBucketIndex::swap_bucket_out) produces, so the directory
    /// can be uploaded to S3 and later restored.
    ///
    /// Two cases are handled:
    /// - **In-memory bucket** (never swapped): arena blocks are copied via
    ///   [`HnswIndex::snapshot_to_dir`], then `levels.bin` and `manifest.json` are written
    ///   from the live in-memory index state.
    /// - **On-disk bucket** (previously swapped via [`swap_bucket_out`]): all files are
    ///   copied directly from the existing swap directory. This is necessary because
    ///   `levels.bin`/`manifest.json` are already on disk from the swap, and
    ///   `clear_level_data` will have cleared the in-memory graph metadata.
    ///
    /// Evicted buckets (no local copy at all) are treated as on-disk if a swap directory
    /// is recorded; otherwise only metadata is written (arena files will be absent).
    ///
    /// Returns `Ok(true)` if the bucket was found, `Ok(false)` if no such bucket exists.
    pub fn snapshot_bucket(&self, seq: BucketSeq, dir: &std::path::Path) -> std::io::Result<bool> {
        let Some(bucket) = self.buckets.iter().find(|b| b.seq == seq) else {
            return Ok(false);
        };
        std::fs::create_dir_all(dir)?;

        if let Some(swap_dir) = &bucket.swap_dir {
            // Bucket was (at least partially) swapped to disk. The authoritative files —
            // block_*.arena, levels.bin, manifest.json — are already there. Copy them
            // all so the snapshot dir is a complete, self-contained copy.
            //
            // If the swap_dir no longer exists the local files were deleted after a
            // successful S3 upload (`SwapBucketOutToBlob` + manual remove_dir_all).
            // The bucket already lives in S3 — return `Ok(false)` to tell the caller
            // there is nothing left to snapshot locally.
            match std::fs::read_dir(swap_dir) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
                Err(e) => return Err(e),
                Ok(entries) => {
                    for entry in entries {
                        let entry = entry?;
                        let path = entry.path();
                        if path.is_file() {
                            let filename = path.file_name().expect("file in swap dir has a name");
                            std::fs::copy(&path, dir.join(filename))?;
                        }
                    }
                }
            }
        } else {
            // Bucket is fully in memory. Snapshot arena bytes and write metadata now.
            bucket.index.snapshot_to_dir(dir)?;
            bucket.index.save_levels(&dir.join("levels.bin"))?;
            bucket.index.save_manifest(&dir.join("manifest.json"))?;
        }

        Ok(true)
    }

    /// Restore a bucket from files previously written by
    /// [`snapshot_bucket`](TimeBucketIndex::snapshot_bucket) (or
    /// [`swap_bucket_out`](TimeBucketIndex::swap_bucket_out)) and add it to this index.
    ///
    /// `local_dir` must contain `block_*.arena`, `levels.bin`, and `manifest.json`.
    /// The bucket is inserted in deque order: call this method with buckets sorted
    /// oldest-first (ascending `created_at`) to maintain the front=newest invariant.
    ///
    /// `next_seq` is advanced past `seq` so future allocations never reuse the same id.
    ///
    /// Returns the number of arena blocks restored.
    pub fn add_restored_bucket(
        &mut self,
        seq: BucketSeq,
        created_at: Timestamp,
        local_dir: &std::path::Path,
    ) -> std::io::Result<usize> {
        let bucket_rng = StdRng::seed_from_u64(seq.0 as u64);
        let mut index: Box<dyn HnswIndex + Send + Sync> = Box::new(HnswArena::new(
            self.dim,
            self.m,
            self.m_max0,
            self.ef_construction,
            NODE_BLOCK_CAPACITY,
            self.closest_m_candidates,
            bucket_rng,
        ));
        let restored = index.load_blocks_from_dir(local_dir)?;
        index.load_levels(&local_dir.join("levels.bin"))?;
        index.rebuild_lens(); // fix block.len so len()/is_empty() are correct
        index.load_manifest(&local_dir.join("manifest.json"))?;

        // The restored content is already committed to S3 — mark the bucket clean so
        // the snapshot task doesn't re-upload it unnecessarily. WAL replay that follows
        // will call push_node, advancing write_count past clean_version and re-dirtying
        // the bucket only if new vectors are actually inserted.
        index.mark_clean_after_snapshot();

        // Register all restored vector IDs in the dedup set so that WAL replay
        // and future inserts won't create duplicate nodes for already-present vectors.
        self.known_vector_ids
            .extend(index.vector_ids().iter().copied());

        // push_front so that the last call (newest bucket) ends up at the front.
        self.buckets.push_front(Bucket {
            seq,
            created_at,
            index,
            swap_dir: None,
        });
        if seq.0 >= self.next_seq.0 {
            self.next_seq = BucketSeq(seq.0 + 1);
        }
        Ok(restored)
    }

    /// Move the bucket identified by `seq` from RAM to disk under `dir`. The bucket
    /// stays in the index (search via `time_range` still finds it) — only its
    /// underlying memory mapping is released. Calls [`HnswIndex::swap_out`] on the
    /// inner index, which fans out across its arena blocks.
    ///
    /// Returns `Ok(true)` if a bucket with that `seq` was found and swapped, `Ok(false)`
    /// if no such bucket exists. Forwards any I/O error from the underlying swap.
    pub fn swap_bucket_out(
        &mut self,
        seq: BucketSeq,
        dir: &std::path::Path,
    ) -> std::io::Result<bool> {
        let Some(bucket) = self.buckets.iter_mut().find(|b| b.seq == seq) else {
            return Ok(false);
        };
        bucket.index.swap_out(dir)?;
        bucket.index.save_levels(&dir.join("levels.bin"))?;
        bucket.index.save_manifest(&dir.join("manifest.json"))?;
        bucket.index.clear_level_data();
        bucket.swap_dir = Some(dir.to_path_buf());
        Ok(true)
    }

    /// Restore the bucket identified by `seq` from disk to RAM. Mirror of
    /// [`TimeBucketIndex::swap_bucket_out`].
    pub fn swap_bucket_in(&mut self, seq: BucketSeq) -> std::io::Result<bool> {
        let Some(bucket) = self.buckets.iter_mut().find(|b| b.seq == seq) else {
            return Ok(false);
        };
        bucket.index.swap_in()?;
        if let Some(dir) = bucket.swap_dir.take() {
            bucket.index.load_levels(&dir.join("levels.bin"))?;
        }
        Ok(true)
    }

    /// Drop the local backing (arena bytes or open fds) of the bucket identified by
    /// `seq`. After this returns, the bucket has no in-memory or on-disk presence —
    /// only the metadata (block count, dim, etc.) is preserved. Reads via
    /// [`TimeBucketIndex::search`] across this bucket will panic until the bucket is
    /// restored via [`TimeBucketIndex::swap_bucket_in_from`].
    ///
    /// Typical use: after [`TimeBucketIndex::swap_bucket_out_to_blob`], call this and
    /// then `std::fs::remove_dir_all(local_dir)` to fully reclaim disk; the bucket now
    /// lives only in remote storage and can be restored on demand.
    ///
    /// Returns `Some(num_storage_units_evicted)` if the bucket was found, `None` otherwise.
    pub fn evict_bucket(&mut self, seq: BucketSeq) -> Option<usize> {
        let bucket = self.buckets.iter_mut().find(|b| b.seq == seq)?;
        Some(bucket.index.evict())
    }

    /// Restore the bucket identified by `seq` by reading each block file from `dir`.
    /// Expects `dir/block_<i>.arena` to exist for every block (same layout
    /// [`TimeBucketIndex::swap_bucket_out`] produced and [`crate::download_arena_dir`]
    /// recreates on download).
    ///
    /// Returns `Ok(Some(num_restored))` on success, `Ok(None)` if the bucket isn't found.
    pub fn swap_bucket_in_from(
        &mut self,
        seq: BucketSeq,
        dir: &std::path::Path,
    ) -> std::io::Result<Option<usize>> {
        let Some(bucket) = self.buckets.iter_mut().find(|b| b.seq == seq) else {
            return Ok(None);
        };
        Ok(Some(bucket.index.swap_in_from(dir)?))
    }

    /// Restore the bucket identified by `seq` from files already present in `local_dir`.
    /// Expects `block_*.arena`, `levels.bin`, and `manifest.json` to be present
    /// (same layout that [`TimeBucketIndex::swap_bucket_out`] produces).
    ///
    /// Returns `Ok(Some(num_restored))` on success, `Ok(None)` if the bucket isn't found.
    pub fn restore_bucket_from_local(
        &mut self,
        seq: BucketSeq,
        local_dir: &std::path::Path,
    ) -> std::io::Result<Option<usize>> {
        let result = self.swap_bucket_in_from(seq, local_dir)?;
        if let Some(bucket) = self.buckets.iter_mut().find(|b| b.seq == seq) {
            bucket.index.load_levels(&local_dir.join("levels.bin"))?;
            bucket
                .index
                .load_manifest(&local_dir.join("manifest.json"))?;
            bucket.swap_dir = None;
        }
        Ok(result)
    }

    /// Download `<prefix>/block_*.arena` from `store` into `local_dir`, then restore the
    /// bucket identified by `seq` by reading those files. Convenience wrapper around
    /// [`crate::download_arena_dir`] + [`TimeBucketIndex::restore_bucket_from_local`].
    ///
    /// `local_dir` is created if missing. Returns `Ok(Some(num_restored))` if the bucket
    /// was found, `Ok(None)` otherwise.
    pub async fn swap_bucket_in_from_blob(
        &mut self,
        seq: BucketSeq,
        store: &dyn object_store::ObjectStore,
        prefix: &object_store::path::Path,
        local_dir: &std::path::Path,
    ) -> std::io::Result<Option<usize>> {
        // Probe first: avoids an unnecessary download if the seq is unknown.
        if !self.buckets.iter().any(|b| b.seq == seq) {
            return Ok(None);
        }
        std::fs::create_dir_all(local_dir)?;
        crate::blob::download_arena_dir(store, prefix, local_dir).await?;
        crate::blob::download_levels(store, prefix, &local_dir.join("levels.bin")).await?;
        crate::blob::download_manifest(store, prefix, &local_dir.join("manifest.json")).await?;
        self.restore_bucket_from_local(seq, local_dir)
    }

    /// Swap the bucket identified by `seq` to disk under `local_dir` and immediately
    /// upload every produced `block_*.arena` file to `store` under `prefix`.
    ///
    /// Combines [`TimeBucketIndex::swap_bucket_out`] with [`crate::blob::upload_arena_dir`].
    /// After this returns:
    /// - The bucket stays in the index; search keeps working through the on-disk read path.
    /// - Local files at `local_dir` remain on disk — the index's open file descriptors
    ///   reference them, so `swap_bucket_in` can still restore the bucket. Delete them
    ///   only after a successful `swap_bucket_in` or full eviction.
    /// - Blobs live at `<prefix>/block_*.arena` in `store`.
    ///
    /// Returns `Ok(true)` if the bucket was found and processed, `Ok(false)` if no bucket
    /// with `seq` exists. Errors from either swap_out or the upload are forwarded.
    pub async fn swap_bucket_out_to_blob(
        &mut self,
        seq: BucketSeq,
        local_dir: &std::path::Path,
        store: &dyn object_store::ObjectStore,
        prefix: &object_store::path::Path,
    ) -> std::io::Result<bool> {
        if !self.swap_bucket_out(seq, local_dir)? {
            return Ok(false);
        }
        crate::blob::upload_arena_dir(store, local_dir, prefix).await?;
        crate::blob::upload_levels(store, &local_dir.join("levels.bin"), prefix).await?;
        crate::blob::upload_manifest(store, &local_dir.join("manifest.json"), prefix).await?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use common::{top_k_quickselect, Timestamp};
    use vector::distance::euclidean_distance_sq;

    use super::*;

    fn ts(v: u64) -> Timestamp {
        Timestamp(v)
    }

    fn make_index(bucket_duration: u64) -> TimeBucketIndex {
        TimeBucketIndex::new(
            4,
            4,
            8,
            32,
            Duration::from_secs(bucket_duration),
            top_k_quickselect,
            StdRng::seed_from_u64(42),
        )
        .unwrap()
    }

    #[test]
    fn empty_search_returns_empty() {
        let idx = make_index(10);
        assert!(idx
            .search(&[0.0; 4], 10, 32, |_, d| d, None, top_k_quickselect)
            .is_empty());
    }

    #[test]
    fn single_insert_and_exact_recall() {
        let mut idx = make_index(10);
        let v = [1.0f32, 0.0, 0.0, 0.0];
        let bid = idx.insert(&v, ts(0), 0u64).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.bucket_count(), 1);

        let results = idx.search(&v, 1, 8, |_, d| d, None, top_k_quickselect);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, bid.vector_id);
    }

    #[test]
    fn same_window_stays_in_one_bucket() {
        // duration=10: t=0,5,9 all fall within [0, 10) → no rotation.
        let mut idx = make_index(10);
        idx.insert(&[0.0; 4], ts(0), 0u64);
        idx.insert(&[1.0; 4], ts(5), 1u64);
        idx.insert(&[2.0; 4], ts(9), 2u64);
        assert_eq!(idx.bucket_count(), 1);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn auto_rotate_on_time() {
        // duration=10: t=0 and t=9 share a bucket; t=10 starts a new one.
        let mut idx = make_index(10);
        idx.insert(&[0.0; 4], ts(0), 0u64);
        idx.insert(&[1.0; 4], ts(9), 1u64);
        assert_eq!(idx.bucket_count(), 1);
        idx.insert(&[2.0; 4], ts(10), 2u64);
        assert_eq!(idx.bucket_count(), 2);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn explicit_rotate_bucket() {
        let mut idx = make_index(100);
        idx.insert(&[0.0; 4], ts(0), 0u64);
        idx.rotate_bucket(ts(10));
        assert_eq!(idx.bucket_count(), 2);
        idx.insert(&[1.0; 4], ts(10), 1u64);
        assert_eq!(idx.bucket_count(), 2);
    }

    #[test]
    fn rotate_bucket_reuses_empty_front() {
        let mut idx = make_index(100);
        idx.rotate_bucket(ts(10));
        assert_eq!(idx.bucket_count(), 1);
        // Front is still empty; rotating updates created_at in place.
        idx.rotate_bucket(ts(20));
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn evict_oldest_removes_single_bucket() {
        // duration=1: each new second starts a new bucket.
        let mut idx = make_index(1);
        idx.insert(&[0.0; 4], ts(0), 0u64);
        idx.insert(&[1.0; 4], ts(1), 1u64);
        assert_eq!(idx.bucket_count(), 2);
        let seq = idx.evict_oldest().expect("non-empty");
        assert_eq!(seq, BucketSeq(0)); // seq=0 is the oldest
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn evict_before_drops_old_buckets() {
        // duration=1: t=0,1,2 each in their own bucket.
        let mut idx = make_index(1);
        idx.insert(&[0.0; 4], ts(0), 0u64);
        idx.insert(&[1.0; 4], ts(1), 1u64);
        idx.insert(&[2.0; 4], ts(2), 2u64);
        assert_eq!(idx.bucket_count(), 3);

        let evicted = idx.evict_before(ts(2)); // drop created_at < ts(2)
        assert_eq!(evicted, 2);
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn evict_before_no_match_is_noop() {
        let mut idx = make_index(10);
        idx.insert(&[0.0; 4], ts(10), 0u64);
        let evicted = idx.evict_before(ts(5));
        assert_eq!(evicted, 0);
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn recency_weight_biases_toward_newer_bucket() {
        // Older bucket holds the geometrically nearest point; newer holds a farther one.
        // A high penalty on the older bucket must make the newer result win.
        let mut idx = make_index(1);
        let bid_near = idx.insert(&[0.1f32, 0.0, 0.0, 0.0], ts(0), 0u64).unwrap();
        idx.rotate_bucket(ts(1));
        let bid_far = idx.insert(&[1.0f32, 0.0, 0.0, 0.0], ts(1), 1u64).unwrap();

        let query = [0.0f32; 4];

        let unweighted = idx.search(&query, 1, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(unweighted[0].0, bid_near.vector_id);

        let weighted = idx.search(
            &query,
            1,
            16,
            |created_at, d| if created_at == ts(0) { d * 1000.0 } else { d },
            None,
            top_k_quickselect,
        );
        assert_eq!(weighted[0].0, bid_far.vector_id);
    }

    #[test]
    fn multi_bucket_search_returns_k_best() {
        const DIM: usize = 4;
        const K: usize = 3;
        // duration=1: i/5 → t=0 (i=0..4), t=1 (i=5..9), t=2 (i=10..14) → 3 buckets.
        let mut idx = TimeBucketIndex::new(
            DIM,
            4,
            8,
            32,
            Duration::from_secs(1),
            top_k_quickselect,
            StdRng::seed_from_u64(42),
        )
        .unwrap();

        for i in 0..15usize {
            let v: [f32; DIM] = std::array::from_fn(|j| (i * DIM + j) as f32);
            idx.insert(&v, ts(i as u64 / 5), i as u64);
        }
        assert_eq!(idx.bucket_count(), 3);

        let query = [0.0f32; DIM];
        let results = idx.search(&query, K, 32, |_, d| d, None, top_k_quickselect);
        assert_eq!(results.len(), K);

        let mut brute: Vec<(u64, f32)> = (0..15usize)
            .map(|i| {
                let v: Vec<f32> = (0..DIM).map(|j| (i * DIM + j) as f32).collect();
                (i as u64, euclidean_distance_sq(&query, &v))
            })
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert!(results.iter().any(|(vid, _)| *vid == brute[0].0));
    }

    #[test]
    fn time_range_restricts_to_matching_buckets() {
        // Three buckets: t=0 (duration=10 → window [0,10)), t=10 ([10,20)), t=20 ([20,30)).
        let mut idx = make_index(10);
        let bid0 = idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 0u64).unwrap(); // bucket [0,10)
        idx.insert(&[9.0, 0.0, 0.0, 0.0], ts(5), 1u64); // also bucket [0,10)
        let bid1 = idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(10), 2u64).unwrap(); // bucket [10,20)
        idx.insert(&[9.0, 0.0, 0.0, 0.0], ts(15), 3u64); // also bucket [10,20)
        let bid2 = idx.insert(&[3.0, 0.0, 0.0, 0.0], ts(20), 4u64).unwrap(); // bucket [20,30)
        assert_eq!(idx.bucket_count(), 3);

        let query = [0.0f32; 4];

        // Only bucket [0,10) — vector_ids 0 and 1; must not include 2, 3, or 4.
        let r = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(0)..ts(10)),
            top_k_quickselect,
        );
        assert!(r.iter().all(|(v, _)| *v == 0 || *v == 1));
        assert!(!r
            .iter()
            .any(|(v, _)| *v == bid1.vector_id || *v == bid2.vector_id));

        // Only bucket [10,20) — vector_ids 2 and 3.
        let r = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(10)..ts(20)),
            top_k_quickselect,
        );
        assert!(r.iter().all(|(v, _)| *v == 2 || *v == 3));

        // Span [5,25) overlaps [0,10), [10,20), [20,30) — vectors from all three buckets.
        let r = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(5)..ts(25)),
            top_k_quickselect,
        );
        let vids: std::collections::HashSet<u64> = r.iter().map(|(v, _)| *v).collect();
        assert!(vids.contains(&bid0.vector_id) || vids.contains(&1));
        assert!(vids.contains(&bid1.vector_id) || vids.contains(&3));
        assert!(vids.contains(&bid2.vector_id));

        // Empty range returns nothing.
        assert!(idx
            .search(
                &query,
                10,
                32,
                |_, d| d,
                Some(ts(5)..ts(5)),
                top_k_quickselect
            )
            .is_empty());

        // Range entirely outside deque returns nothing.
        assert!(idx
            .search(
                &query,
                10,
                32,
                |_, d| d,
                Some(ts(100)..ts(200)),
                top_k_quickselect
            )
            .is_empty());
    }

    // ── swap_bucket_out / swap_bucket_in ─────────────────────────────────────

    /// Unique temp dir for time-bucket swap tests; caller responsible for cleanup.
    fn unique_swap_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id();
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("mem_weaver_time_bucket_{tag}_{pid}_{nanos}_{n}"))
    }

    /// Recursively deletes the directory on drop so tests stay self-cleaning on panic.
    struct DirGuard(std::path::PathBuf);
    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn swap_bucket_out_unknown_seq_returns_false() {
        let root = unique_swap_dir("unknown_seq");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64); // creates one bucket (seq=0)

        let dir = root.join("seq_999");
        let moved = idx
            .swap_bucket_out(BucketSeq(999), &dir)
            .expect("no IO when seq not found");
        assert!(!moved);
        assert!(!dir.exists(), "no files written when bucket absent");

        let restored = idx.swap_bucket_in(BucketSeq(999)).expect("no IO");
        assert!(!restored);
    }

    #[test]
    fn swap_bucket_out_then_in_preserves_search_results() {
        // Three buckets, distinct contents per bucket. Swap the middle one out, then back
        // in, and verify search returns the same result list as before the swap.
        //
        // Note: we explicitly do NOT call `search` while a bucket is on disk. Although
        // `neighbors_at` supports the on-disk read path, `vector_at` does not yet (it would
        // need the same buf-plumbing); search inside a cold bucket would dereference a null
        // pointer. The round-trip test still validates that swap_out → swap_in is lossless.
        let root = unique_swap_dir("rt_search");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(1);
        idx.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64); // bucket 0 (seq=0)
        idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(1), 1u64); // bucket 1 (seq=1)
        idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(2), 2u64); // bucket 2 (seq=2)
        assert_eq!(idx.bucket_count(), 3);

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(before.len(), 3, "all three inserts retrievable before swap");

        let dir_seq1 = root.join("seq_1");
        let moved = idx
            .swap_bucket_out(BucketSeq(1), &dir_seq1)
            .expect("swap_out");
        assert!(moved, "bucket seq=1 must be found and swapped");
        assert!(dir_seq1.exists(), "swap_out wrote files");

        let restored = idx.swap_bucket_in(BucketSeq(1)).expect("swap_in");
        assert!(restored);

        let after = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(after, before, "search results identical after swap_in");
    }

    /// `vector_at` and `neighbors_at` are both buf-plumbed for the on-disk path, so search
    /// across a cold bucket works end-to-end via the trait object.
    #[test]
    fn search_across_cold_bucket_works_after_vector_at_buf_plumbing() {
        let root = unique_swap_dir("mixed_range");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        let bid0 = idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 0u64).unwrap();
        let bid1 = idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(10), 1u64).unwrap();
        let bid2 = idx.insert(&[3.0, 0.0, 0.0, 0.0], ts(20), 2u64).unwrap();

        idx.swap_bucket_out(bid1.bucket_seq, &root.join("seq_1"))
            .expect("swap_out");

        let query = [0.0; 4];
        let all = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(0)..ts(30)),
            top_k_quickselect,
        );
        let vids: std::collections::HashSet<u64> = all.iter().map(|(v, _)| *v).collect();
        assert!(vids.contains(&bid0.vector_id));
        assert!(vids.contains(&bid1.vector_id), "cold bucket missing");
        assert!(vids.contains(&bid2.vector_id));
    }

    #[test]
    fn double_swap_out_on_same_bucket_is_idempotent() {
        // ArenaNodeStore::swap_out skips already-on-disk blocks, so calling
        // swap_bucket_out twice returns Ok(true) both times without error. After the
        // round-trip the bucket can still be swapped back in.
        let root = unique_swap_dir("double_out");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);
        let seq = BucketSeq(0);

        assert!(idx
            .swap_bucket_out(seq, &root.join("seq_0"))
            .expect("first out"));
        assert!(idx
            .swap_bucket_out(seq, &root.join("seq_0"))
            .expect("second out (idempotent, no-op)"));
        assert!(idx.swap_bucket_in(seq).expect("swap_in still works"));
    }

    // ── swap_bucket_out_to_blob ──────────────────────────────────────────────

    #[tokio::test]
    async fn swap_bucket_out_to_blob_uploads_all_block_files() {
        use object_store::{memory::InMemory, path::Path as ObjectPath, ObjectStore};
        use std::sync::Arc;

        let root = unique_swap_dir("blob_upload");
        let _guard = DirGuard(root.clone());

        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);
        idx.insert(&[2.0; 4], ts(0), 1u64);
        let seq = BucketSeq(0);

        let local = root.join("seq_0");
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("dev/mem-weaver-test/buckets/seq_0");

        let moved = idx
            .swap_bucket_out_to_blob(seq, &local, store.as_ref(), &prefix)
            .await
            .expect("swap_bucket_out_to_blob");
        assert!(moved, "bucket must be found and uploaded");

        // Every local block_*.arena file must exist as <prefix>/<filename> in the store.
        let local_files: Vec<_> = std::fs::read_dir(&local)
            .expect("read_dir")
            .filter_map(|e| {
                let p = e.ok()?.path();
                (p.extension().and_then(|s| s.to_str()) == Some("arena")).then_some(p)
            })
            .collect();
        assert!(
            !local_files.is_empty(),
            "swap_out produced at least one arena file"
        );

        for f in &local_files {
            let name = f.file_name().unwrap().to_str().unwrap();
            let got = store
                .get(&prefix.child(name))
                .await
                .expect("get")
                .bytes()
                .await
                .unwrap();
            let want = std::fs::read(f).expect("local read");
            assert_eq!(got.as_ref(), want.as_slice(), "{name} differs in blob");
        }

        // Bucket still indexed; swap back in to confirm we didn't leave it in a broken state.
        assert!(idx.swap_bucket_in(seq).expect("swap_in"));
    }

    #[tokio::test]
    async fn evict_then_swap_in_from_blob_restores_search_results() {
        use object_store::{memory::InMemory, path::Path as ObjectPath, ObjectStore};
        use std::sync::Arc;

        let root = unique_swap_dir("evict_blob_rt");
        let _guard = DirGuard(root.clone());

        let mut idx = make_index(10);
        idx.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64);
        idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 1u64);
        idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(0), 2u64);
        let seq = BucketSeq(0);

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(before.len(), 3, "baseline search returns all three");

        // 1. Swap to local + upload to blob.
        let local = root.join("seq_0");
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("buckets/seq_0");
        assert!(idx
            .swap_bucket_out_to_blob(seq, &local, store.as_ref(), &prefix)
            .await
            .expect("swap_bucket_out_to_blob"));

        // 2. Evict locally — fds closed, then delete the local dir to fully reclaim disk.
        let evicted = idx.evict_bucket(seq).expect("bucket present");
        assert!(evicted >= 1, "at least one block evicted");
        std::fs::remove_dir_all(&local).expect("rm local");
        assert!(!local.exists(), "local copy is gone");

        // 3. Restore from blob into a *fresh* directory (proves the bytes really came
        //    back from S3, not from any lingering local file).
        let restored_dir = root.join("seq_0_restored");
        let restored = idx
            .swap_bucket_in_from_blob(seq, store.as_ref(), &prefix, &restored_dir)
            .await
            .expect("swap_bucket_in_from_blob")
            .expect("bucket present");
        assert_eq!(restored, evicted, "every evicted block restored");
        assert!(
            restored_dir.exists(),
            "files materialized locally for swap_in"
        );

        // 4. Search must return identical results — search has been reading entirely
        //    from blob-derived bytes since step 3.
        let after = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(
            after, before,
            "search must match baseline after full S3 round-trip"
        );
    }

    #[tokio::test]
    async fn evict_bucket_unknown_seq_returns_none() {
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);
        assert!(idx.evict_bucket(BucketSeq(999)).is_none());
    }

    #[tokio::test]
    async fn swap_bucket_in_from_blob_unknown_seq_returns_none_and_downloads_nothing() {
        use object_store::{memory::InMemory, path::Path as ObjectPath, ObjectStore};
        use std::sync::Arc;

        let root = unique_swap_dir("in_from_blob_unknown");
        let _guard = DirGuard(root.clone());

        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);

        let local = root.join("seq_999");
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("buckets/seq_999");

        let res = idx
            .swap_bucket_in_from_blob(BucketSeq(999), store.as_ref(), &prefix, &local)
            .await
            .expect("no IO when seq not found");
        assert!(res.is_none());
        assert!(!local.exists(), "no local dir created when bucket absent");
    }

    #[tokio::test]
    async fn swap_bucket_out_to_blob_unknown_seq_returns_false_and_uploads_nothing() {
        use object_store::{memory::InMemory, path::Path as ObjectPath, ObjectStore};
        use std::sync::Arc;

        let root = unique_swap_dir("blob_unknown");
        let _guard = DirGuard(root.clone());

        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);

        let local = root.join("seq_999");
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("dev/mem-weaver-test/buckets/seq_999");

        let moved = idx
            .swap_bucket_out_to_blob(BucketSeq(999), &local, store.as_ref(), &prefix)
            .await
            .expect("no IO when seq not found");
        assert!(!moved);
        assert!(!local.exists(), "no local files written when bucket absent");

        // Store must be empty under prefix.
        let mut list = store.list(Some(&prefix));
        use futures::StreamExt;
        assert!(list.next().await.is_none(), "no objects uploaded");
    }

    // ── snapshot_bucket ───────────────────────────────────────────────────────

    #[test]
    fn snapshot_bucket_unknown_seq_returns_false() {
        let root = unique_swap_dir("snap_unknown");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);

        let snap_dir = root.join("snap_999");
        let found = idx
            .snapshot_bucket(BucketSeq(999), &snap_dir)
            .expect("no IO for missing seq");
        assert!(!found);
        assert!(!snap_dir.exists(), "no directory created for unknown seq");
    }

    #[test]
    fn snapshot_bucket_writes_arena_levels_and_manifest() {
        let root = unique_swap_dir("snap_files");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);
        idx.insert(&[2.0; 4], ts(0), 1u64);
        let seq = BucketSeq(0);

        let snap_dir = root.join("snap_0");
        let found = idx.snapshot_bucket(seq, &snap_dir).expect("snapshot");
        assert!(found, "bucket must be found");

        // At least one arena file, plus levels.bin and manifest.json.
        let arena_files: Vec<_> = std::fs::read_dir(&snap_dir)
            .expect("read snap_dir")
            .filter_map(|e| {
                let p = e.ok()?.path();
                let is_arena = p.extension().and_then(|s| s.to_str()) == Some("arena");
                is_arena.then_some(p)
            })
            .collect();
        assert!(
            !arena_files.is_empty(),
            "snapshot must produce at least one .arena file"
        );
        assert!(snap_dir.join("levels.bin").exists(), "levels.bin missing");
        assert!(
            snap_dir.join("manifest.json").exists(),
            "manifest.json missing"
        );
    }

    #[test]
    fn snapshot_bucket_does_not_change_storage_state() {
        // After snapshot_bucket the index must still be fully searchable (no swap happened).
        let root = unique_swap_dir("snap_state");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        let bid0 = idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 0u64).unwrap();
        let bid1 = idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(0), 1u64).unwrap();
        let seq = BucketSeq(0);

        let before = idx.search(&[0.0; 4], 2, 16, |_, d| d, None, top_k_quickselect);

        idx.snapshot_bucket(seq, &root.join("snap_0"))
            .expect("snapshot");

        // Search results must be identical — no storage state was changed.
        let after = idx.search(&[0.0; 4], 2, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(
            after, before,
            "search results must not change after snapshot"
        );
        assert_eq!(idx.bucket_count(), 1, "bucket count must not change");
        assert_eq!(idx.len(), 2, "vector count must not change");
        let _ = (bid0, bid1);
    }

    #[test]
    fn snapshot_bucket_in_memory_files_survive_crc32_and_restore_search() {
        // Snapshot an in-memory bucket, then evict and restore from the snapshot.
        let root = unique_swap_dir("snap_restore_mem");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64);
        idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 1u64);
        idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(0), 2u64);
        let seq = BucketSeq(0);

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(before.len(), 3);

        // Snapshot while bucket is in memory.
        let snap_dir = root.join("snap_0");
        assert!(idx.snapshot_bucket(seq, &snap_dir).expect("snapshot"));

        // Evict via an intermediate swap so we can restore from the snapshot only.
        let swap_dir = root.join("swap_0");
        assert!(idx.swap_bucket_out(seq, &swap_dir).expect("swap_out"));
        idx.evict_bucket(seq).expect("evict");

        // Restore from snapshot — CRC32 must pass.
        let restored = idx
            .restore_bucket_from_local(seq, &snap_dir)
            .expect("restore_bucket_from_local from snapshot")
            .expect("bucket must be present");
        assert!(restored >= 1, "at least one block restored");

        let after = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(
            after, before,
            "search results must match baseline after restore from snapshot"
        );
    }

    #[test]
    fn snapshot_bucket_on_disk_copies_swap_dir_files() {
        // After swap_bucket_out, snapshot_bucket must copy the on-disk arena files
        // (not try to re-snapshot in-memory blocks, which no longer exist).
        let root = unique_swap_dir("snap_ondisk");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64);
        idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 1u64);
        idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(0), 2u64);
        let seq = BucketSeq(0);

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);

        // Swap the bucket out to disk.
        let swap_dir = root.join("swap_0");
        assert!(idx.swap_bucket_out(seq, &swap_dir).expect("swap_out"));

        // Now snapshot the already-on-disk bucket.
        let snap_dir = root.join("snap_0");
        assert!(idx
            .snapshot_bucket(seq, &snap_dir)
            .expect("snapshot on-disk bucket"));

        // The snapshot dir must contain at least one .arena file and both metadata files.
        let arena_count = std::fs::read_dir(&snap_dir)
            .expect("read snap_dir")
            .filter(|e| {
                e.as_ref()
                    .ok()
                    .and_then(|e| {
                        e.path()
                            .extension()
                            .and_then(|s| s.to_str())
                            .map(|ext| ext == "arena")
                    })
                    .unwrap_or(false)
            })
            .count();
        assert!(
            arena_count >= 1,
            "snapshot of on-disk bucket must include arena files"
        );
        assert!(
            snap_dir.join("levels.bin").exists(),
            "levels.bin must be present in snapshot"
        );
        assert!(
            snap_dir.join("manifest.json").exists(),
            "manifest.json must be present in snapshot"
        );

        // The snapshot files must be byte-identical to the swap_dir files.
        for filename in &["levels.bin", "manifest.json"] {
            assert_eq!(
                std::fs::read(snap_dir.join(filename)).unwrap(),
                std::fs::read(swap_dir.join(filename)).unwrap(),
                "{filename}: snapshot and swap_dir bytes must match"
            );
        }

        // Evict and restore from the snapshot to prove it is self-contained.
        idx.evict_bucket(seq).expect("evict");
        let restored = idx
            .restore_bucket_from_local(seq, &snap_dir)
            .expect("restore from snapshot")
            .expect("bucket present");
        assert!(restored >= 1);

        let after = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(
            after, before,
            "search must match baseline after on-disk snapshot round-trip"
        );
    }

    #[test]
    fn snapshot_bucket_multiple_buckets_each_snapshotted_independently() {
        // Three buckets; snapshot only the middle one. The other two must be untouched.
        let root = unique_swap_dir("snap_multi");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(1);
        idx.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64); // seq=0
        idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(1), 1u64); // seq=1
        idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(2), 2u64); // seq=2
        assert_eq!(idx.bucket_count(), 3);

        let seqs: Vec<BucketSeq> = idx.bucket_seqs();
        assert_eq!(seqs.len(), 3);

        // Snapshot only the middle bucket (seq=1).
        let snap_dir = root.join("snap_seq1");
        assert!(idx
            .snapshot_bucket(BucketSeq(1), &snap_dir)
            .expect("snapshot seq=1"));
        assert!(!root.join("snap_seq0").exists(), "seq=0 not snapshotted");
        assert!(!root.join("snap_seq2").exists(), "seq=2 not snapshotted");

        // All three buckets remain searchable.
        let results = idx.search(&[0.0; 4], 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(results.len(), 3, "all three vectors still retrievable");
    }

    // ── add_restored_bucket ───────────────────────────────────────────────────

    #[test]
    fn add_restored_bucket_recreates_searchable_index() {
        let root = unique_swap_dir("restore_bucket");
        let _guard = DirGuard(root.clone());

        // Build an original index with two buckets.
        let mut src = make_index(10);
        let bid0 = src.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 10u64).unwrap();
        let bid1 = src.insert(&[2.0, 0.0, 0.0, 0.0], ts(0), 11u64).unwrap();

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = src.search(&query, 2, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(before.len(), 2);

        // Snapshot bucket seq=0.
        let snap = root.join("snap_0");
        let (seq, created_at) = src.bucket_metas()[0];
        assert!(src.snapshot_bucket(seq, &snap).expect("snapshot"));

        // Restore into a fresh index.
        let mut dst = make_index(10);
        let n = dst
            .add_restored_bucket(seq, created_at, &snap)
            .expect("add_restored_bucket");
        assert!(n >= 1, "at least one block restored");
        assert_eq!(dst.bucket_count(), 1);

        // Search results must match.
        let after = dst.search(&query, 2, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(
            after, before,
            "restored index must return identical results"
        );
        let _ = (bid0, bid1);
    }

    #[test]
    fn add_restored_bucket_multiple_buckets_sorted_oldest_first() {
        let root = unique_swap_dir("restore_multi");
        let _guard = DirGuard(root.clone());

        let mut src = TimeBucketIndex::new(
            4,
            4,
            8,
            32,
            std::time::Duration::from_secs(1),
            top_k_quickselect,
            rand::rngs::StdRng::seed_from_u64(1),
        )
        .unwrap();
        src.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64); // seq=0 oldest
        src.insert(&[1.0, 0.0, 0.0, 0.0], ts(1), 1u64); // seq=1
        src.insert(&[2.0, 0.0, 0.0, 0.0], ts(2), 2u64); // seq=2 newest
        assert_eq!(src.bucket_count(), 3);

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = src.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);

        // Snapshot all buckets.
        let metas = src.bucket_metas();
        for &(seq, _) in &metas {
            let snap = root.join(format!("snap_{}", seq.0));
            assert!(src.snapshot_bucket(seq, &snap).expect("snapshot"));
        }

        // Restore oldest-first into a fresh index.
        let mut dst = TimeBucketIndex::new(
            4,
            4,
            8,
            32,
            std::time::Duration::from_secs(1),
            top_k_quickselect,
            rand::rngs::StdRng::seed_from_u64(0),
        )
        .unwrap();
        // metas is newest-first; reverse to restore oldest-first.
        let mut sorted = metas.clone();
        sorted.sort_by_key(|&(seq, _)| seq);
        for (seq, created_at) in sorted {
            let snap = root.join(format!("snap_{}", seq.0));
            dst.add_restored_bucket(seq, created_at, &snap)
                .expect("add_restored_bucket");
        }

        assert_eq!(dst.bucket_count(), 3);
        // next_seq must be past the highest restored seq.
        dst.insert(&[3.0, 0.0, 0.0, 0.0], ts(3), 99u64);
        assert_eq!(
            dst.bucket_count(),
            4,
            "new insert must open a new bucket, not reuse seq=2"
        );

        let mut after = dst.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        let mut before_sorted = before.clone();
        // quickselect does not guarantee order; sort by vector_id for a stable comparison.
        after.sort_by_key(|&(vid, _)| vid);
        before_sorted.sort_by_key(|&(vid, _)| vid);
        assert_eq!(
            after, before_sorted,
            "restored multi-bucket index must match original search"
        );
    }

    #[test]
    fn snapshot_bucket_evicted_with_deleted_local_files_returns_false() {
        // After swap_out + upload to S3 + remove_dir_all, the swap_dir no longer exists
        // locally. snapshot_bucket must return Ok(false) — the bucket is already in S3
        // and there is nothing to snapshot locally.
        let root = unique_swap_dir("snap_evicted_nogap");
        let _guard = DirGuard(root.clone());
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64);
        let seq = BucketSeq(0);

        // Simulate the swap-out → upload → remove_dir_all lifecycle.
        let swap_dir = root.join("swap_0");
        assert!(idx.swap_bucket_out(seq, &swap_dir).expect("swap_out"));
        idx.evict_bucket(seq).expect("evict");
        std::fs::remove_dir_all(&swap_dir).expect("remove swap_dir");
        assert!(!swap_dir.exists(), "local files gone");

        let snap_dir = root.join("snap_0");
        let found = idx
            .snapshot_bucket(seq, &snap_dir)
            .expect("no error for evicted bucket with deleted local files");
        assert!(
            !found,
            "Ok(false) signals bucket already in S3 — nothing to snapshot locally"
        );
        // No files written to the snap_dir.
        assert!(
            !snap_dir.exists()
                || std::fs::read_dir(&snap_dir)
                    .map(|mut d| d.next().is_none())
                    .unwrap_or(true),
            "snap_dir must be empty or absent when bucket is already in S3"
        );
    }

    #[tokio::test]
    async fn snapshot_bucket_files_uploadable_and_restorable_via_blob() {
        // Full round-trip: snapshot → upload to InMemory store → evict → download → restore.
        use object_store::{memory::InMemory, path::Path as ObjectPath, ObjectStore};
        use std::sync::Arc;

        let root = unique_swap_dir("snap_blob_rt");
        let _guard = DirGuard(root.clone());

        let mut idx = make_index(10);
        idx.insert(&[0.1, 0.0, 0.0, 0.0], ts(0), 0u64);
        idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0), 1u64);
        idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(0), 2u64);
        let seq = BucketSeq(0);

        let query = [0.5, 0.0, 0.0, 0.0];
        let before = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(before.len(), 3);

        // 1. Snapshot to local dir (read-only, in-memory copy preserved).
        let snap_dir = root.join("snap_0");
        assert!(idx.snapshot_bucket(seq, &snap_dir).expect("snapshot"));

        // 2. Upload snapshot to blob store.
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("collections/test/seq_0");
        crate::blob::upload_arena_dir(store.as_ref(), &snap_dir, &prefix)
            .await
            .expect("upload arenas");
        crate::blob::upload_levels(store.as_ref(), &snap_dir.join("levels.bin"), &prefix)
            .await
            .expect("upload levels");
        crate::blob::upload_manifest(store.as_ref(), &snap_dir.join("manifest.json"), &prefix)
            .await
            .expect("upload manifest");

        // 3. Evict the in-memory bucket entirely (also wipe local snap dir).
        let evicted = idx.evict_bucket(seq).expect("evict");
        assert!(evicted >= 1);
        std::fs::remove_dir_all(&snap_dir).expect("remove snap_dir");

        // 4. Restore from blob.
        let restore_dir = root.join("restored_0");
        let restored = idx
            .swap_bucket_in_from_blob(seq, store.as_ref(), &prefix, &restore_dir)
            .await
            .expect("swap_bucket_in_from_blob")
            .expect("bucket present");
        assert_eq!(restored, evicted);

        // 5. Search must match baseline.
        let after = idx.search(&query, 3, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(
            after, before,
            "search must match baseline after snapshot → S3 → restore round-trip"
        );
    }

    // ── dirty-flag race condition fix ─────────────────────────────────────────

    /// Simulates the race: snapshot captured at version N, new insert arrives
    /// during the upload (version advances to N+1), then mark_clean_if_version(N)
    /// is called. The bucket must stay dirty so the next cycle re-uploads.
    #[test]
    fn mark_clean_if_version_leaves_dirty_when_write_happened_after_snapshot() {
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64).unwrap();
        let seq = BucketSeq(0);

        assert!(idx.is_bucket_dirty(seq), "new bucket must start dirty");

        // ── Simulate: snapshot taken, version captured ────────────────────────
        let version_at_snapshot = idx.bucket_write_count(seq);

        // ── Simulate: new insert arrives during the upload (no lock held) ─────
        idx.insert(&[2.0; 4], ts(0), 1u64).unwrap();

        let version_after_insert = idx.bucket_write_count(seq);
        assert!(
            version_after_insert > version_at_snapshot,
            "insert must advance write_count"
        );

        // ── Simulate: upload completes, stale version passed to mark_clean ────
        idx.mark_bucket_clean_if_version(seq, version_at_snapshot);

        assert!(
            idx.is_bucket_dirty(seq),
            "bucket must stay dirty — write happened after snapshot was taken"
        );
    }

    /// Verifies the happy path: when no writes occur between snapshot and upload,
    /// mark_clean_if_version correctly clears the dirty flag.
    #[test]
    fn mark_clean_if_version_clears_flag_when_no_write_happened_after_snapshot() {
        let mut idx = make_index(10);
        idx.insert(&[1.0; 4], ts(0), 0u64).unwrap();
        let seq = BucketSeq(0);

        // Capture version and immediately mark clean (no intervening writes).
        let version = idx.bucket_write_count(seq);
        idx.mark_bucket_clean_if_version(seq, version);

        assert!(
            !idx.is_bucket_dirty(seq),
            "bucket must be clean when no write happened after snapshot"
        );

        // A subsequent insert re-dirties the bucket for the next cycle.
        idx.insert(&[2.0; 4], ts(0), 1u64).unwrap();
        assert!(
            idx.is_bucket_dirty(seq),
            "insert after mark_clean must re-dirty the bucket"
        );
    }
}
