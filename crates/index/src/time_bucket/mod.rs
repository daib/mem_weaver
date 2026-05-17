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

/// Identifies a node uniquely across all time buckets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BucketedNodeId {
    /// Monotonically increasing sequence number assigned to the bucket on creation.
    pub bucket_seq: BucketSeq,
    /// Node id within that bucket's HNSW index.
    pub node_id: NodeId,
}

struct Bucket {
    seq: BucketSeq,
    created_at: Timestamp,
    index: Box<dyn HnswIndex + Send>,
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
        })
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
        let index: Box<dyn HnswIndex + Send> = Box::new(HnswArena::new(
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
        });
    }

    /// Insert `vector` into the bucket whose window covers `timestamp`.
    ///
    /// The target window start is `bucket_start(timestamp)`. If it is ahead of
    /// the current newest bucket a new bucket is created at that grid position.
    /// `timestamp` drives the grid-aligned `created_at` of any new bucket.
    pub fn insert(&mut self, vector: &[f32], timestamp: Timestamp) -> BucketedNodeId {
        let start = self.bucket_start(timestamp);
        let needs_new = self.buckets.front().map_or(true, |b| start > b.created_at);
        if needs_new {
            self.alloc_bucket(start);
        }

        let bucket = self.buckets.front_mut().expect("just allocated");
        let node_id = bucket.index.insert(vector);
        BucketedNodeId {
            bucket_seq: bucket.seq,
            node_id,
        }
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
    ) -> Vec<BucketedNodeId> {
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
            for (node_id, dist) in bucket.index.search(query, k, ef) {
                all.push((
                    BucketedNodeId {
                        bucket_seq: bucket.seq,
                        node_id,
                    },
                    adjust_distance_fn(bucket.created_at, dist),
                ));
            }
        }

        top_k_fn(&all, k)
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
        let bid = idx.insert(&v, ts(0));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.bucket_count(), 1);

        let results = idx.search(&v, 1, 8, |_, d| d, None, top_k_quickselect);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], bid);
    }

    #[test]
    fn same_window_stays_in_one_bucket() {
        // duration=10: t=0,5,9 all fall within [0, 10) → no rotation.
        let mut idx = make_index(10);
        idx.insert(&[0.0; 4], ts(0));
        idx.insert(&[1.0; 4], ts(5));
        idx.insert(&[2.0; 4], ts(9));
        assert_eq!(idx.bucket_count(), 1);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn auto_rotate_on_time() {
        // duration=10: t=0 and t=9 share a bucket; t=10 starts a new one.
        let mut idx = make_index(10);
        idx.insert(&[0.0; 4], ts(0));
        idx.insert(&[1.0; 4], ts(9));
        assert_eq!(idx.bucket_count(), 1);
        idx.insert(&[2.0; 4], ts(10));
        assert_eq!(idx.bucket_count(), 2);
        assert_eq!(idx.len(), 3);
    }

    #[test]
    fn explicit_rotate_bucket() {
        let mut idx = make_index(100);
        idx.insert(&[0.0; 4], ts(0));
        idx.rotate_bucket(ts(10));
        assert_eq!(idx.bucket_count(), 2);
        idx.insert(&[1.0; 4], ts(10));
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
        idx.insert(&[0.0; 4], ts(0));
        idx.insert(&[1.0; 4], ts(1));
        assert_eq!(idx.bucket_count(), 2);
        let seq = idx.evict_oldest().expect("non-empty");
        assert_eq!(seq, BucketSeq(0)); // seq=0 is the oldest
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn evict_before_drops_old_buckets() {
        // duration=1: t=0,1,2 each in their own bucket.
        let mut idx = make_index(1);
        idx.insert(&[0.0; 4], ts(0));
        idx.insert(&[1.0; 4], ts(1));
        idx.insert(&[2.0; 4], ts(2));
        assert_eq!(idx.bucket_count(), 3);

        let evicted = idx.evict_before(ts(2)); // drop created_at < ts(2)
        assert_eq!(evicted, 2);
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn evict_before_no_match_is_noop() {
        let mut idx = make_index(10);
        idx.insert(&[0.0; 4], ts(10));
        let evicted = idx.evict_before(ts(5));
        assert_eq!(evicted, 0);
        assert_eq!(idx.bucket_count(), 1);
    }

    #[test]
    fn recency_weight_biases_toward_newer_bucket() {
        // Older bucket holds the geometrically nearest point; newer holds a farther one.
        // A high penalty on the older bucket must make the newer result win.
        let mut idx = make_index(1);
        let bid_near = idx.insert(&[0.1f32, 0.0, 0.0, 0.0], ts(0));
        idx.rotate_bucket(ts(1));
        let bid_far = idx.insert(&[1.0f32, 0.0, 0.0, 0.0], ts(1));

        let query = [0.0f32; 4];

        let unweighted = idx.search(&query, 1, 16, |_, d| d, None, top_k_quickselect);
        assert_eq!(unweighted[0], bid_near);

        let weighted = idx.search(
            &query,
            1,
            16,
            |created_at, d| if created_at == ts(0) { d * 1000.0 } else { d },
            None,
            top_k_quickselect,
        );
        assert_eq!(weighted[0], bid_far);
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

        let mut all_bids = Vec::new();
        for i in 0..15usize {
            let v: [f32; DIM] = std::array::from_fn(|j| (i * DIM + j) as f32);
            all_bids.push(idx.insert(&v, ts(i as u64 / 5)));
        }
        assert_eq!(idx.bucket_count(), 3);

        let query = [0.0f32; DIM];
        let results = idx.search(&query, K, 32, |_, d| d, None, top_k_quickselect);
        assert_eq!(results.len(), K);

        let query_v = [0.0f32; DIM];
        let mut brute: Vec<_> = all_bids
            .iter()
            .enumerate()
            .map(|(i, &bid)| {
                let v: Vec<f32> = (0..DIM).map(|j| (i * DIM + j) as f32).collect();
                (bid, euclidean_distance_sq(&query_v, &v))
            })
            .collect();
        brute.sort_by(|a, b| a.1.total_cmp(&b.1));
        assert!(results.contains(&brute[0].0));
    }

    #[test]
    fn time_range_restricts_to_matching_buckets() {
        // Three buckets: t=0 (duration=10 → window [0,10)), t=10 ([10,20)), t=20 ([20,30)).
        let mut idx = make_index(10);
        let bid0 = idx.insert(&[1.0, 0.0, 0.0, 0.0], ts(0)); // bucket [0,10)
        idx.insert(&[9.0, 0.0, 0.0, 0.0], ts(5)); // also bucket [0,10)
        let bid1 = idx.insert(&[2.0, 0.0, 0.0, 0.0], ts(10)); // bucket [10,20)
        idx.insert(&[9.0, 0.0, 0.0, 0.0], ts(15)); // also bucket [10,20)
        let bid2 = idx.insert(&[3.0, 0.0, 0.0, 0.0], ts(20)); // bucket [20,30)
        assert_eq!(idx.bucket_count(), 3);

        let query = [0.0f32; 4];

        // Only bucket [0,10) — must contain bid0, not bid1/bid2.
        let r = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(0)..ts(10)),
            top_k_quickselect,
        );
        let ids: Vec<_> = r.iter().map(|b| b.bucket_seq).collect();
        assert!(ids.iter().all(|&s| s == bid0.bucket_seq));
        assert!(!ids.contains(&bid1.bucket_seq));

        // Only bucket [10,20) — must contain bid1.
        let r = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(10)..ts(20)),
            top_k_quickselect,
        );
        let ids: Vec<_> = r.iter().map(|b| b.bucket_seq).collect();
        assert!(ids.iter().all(|&s| s == bid1.bucket_seq));

        // Span [5,25) overlaps [0,10), [10,20), [20,30) — all three buckets searched.
        let r = idx.search(
            &query,
            10,
            32,
            |_, d| d,
            Some(ts(5)..ts(25)),
            top_k_quickselect,
        );
        let seqs: std::collections::HashSet<_> = r.iter().map(|b| b.bucket_seq).collect();
        assert!(seqs.contains(&bid0.bucket_seq));
        assert!(seqs.contains(&bid1.bucket_seq));
        assert!(seqs.contains(&bid2.bucket_seq));

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
}
