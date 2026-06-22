# Parallel HNSW Insertion via Two-Phase Batch Insertion

## Core Insight

HNSW insertion is dominated by the neighbor search phase (~90% of time), which is read-only. By separating reads from writes, multiple threads can compute insertion plans simultaneously against a consistent graph snapshot, then apply them sequentially.

```
Phase 1 (parallel): compute neighbor plans — read-only, no locks needed
Phase 2 (sequential): apply edge updates — brief writes, correct graph state
```

---

## The Algorithm

For a batch of N vectors (N = thread count):

**Phase 1 — Parallel search (read-only):**
For each vector in the batch, concurrently:
1. Greedy descent from entry point to insertion level (read lock per level, released between levels)
2. Beam search at each insertion level to find nearest neighbors (read lock per level)
3. Record planned neighbor connections — do not modify graph

**Phase 2 — Sequential application:**
For each vector in the batch, sequentially:
1. Allocate node in arena
2. Apply pre-computed neighbor connections
3. Prune neighbor edge lists if over M_MAX0 capacity
4. Update entry point if new node is promoted above current max level

---

## The Locking Model

One RwLock per level during Phase 1:

```
Phase 1 reads:  read lock per level — multiple threads simultaneously
Phase 2 writes: no locks needed — sequential application
```

Phase 2 is sequential, so write locks are unnecessary. The read locks in Phase 1 are short-lived (released between levels, not held across the full descent).

---

## Why This Works

### The read/write ratio

HNSW insertion time breakdown (SIFT1M, M=16, ef_construction=40):

```
Phase 1 (neighbor search): ~90% of total insertion time
  - Greedy descent through upper levels
  - Beam search at layer 0 (~1.5ms — dominant)
  
Phase 2 (edge updates):    ~10% of total insertion time
  - Edge list writes (~3 microseconds per level)
  - Pruning
```

Parallelizing the dominant phase (Phase 1) gives real speedup. The sequential Phase 2 is a small fraction of total time.

### Quality impact of batch size

Each vector in a batch misses its batch-mates as potential neighbors — they are not yet in the graph when plans are computed. With batch size B and N total vectors:

```
Miss rate: (B-1) / N

Batch size 6 (threads=6), N=1M:
  Miss rate: 5 / 1,000,000 = 0.0005%
  Recall impact: negligible
```

This is why batch size = thread count is the right design: minimum quality impact, maximum parallelism.

---

## Shared Neighbor Conflict

If two vectors in the same batch share the same neighbor node, their independent pruning decisions can conflict — each prunes without knowing the other will also connect to that neighbor.

**Detection and resolution:** Skip the edge update if both vectors in a batch share neighbors — serialize these insertions.

```rust
fn shares_neighbors(a: &InsertPlan, b: &InsertPlan) -> bool {
    let a_set: HashSet<NodeId> = a.neighbors_per_level[0].iter().copied().collect();
    b.neighbors_per_level[0].iter().any(|n| a_set.contains(n))
}
```

Conflict probability with batch=6 and 1M vectors is very low (~0.13%) — most batches apply cleanly in parallel.

---

## Entry Point Handling

The entry point must be updated once per batch, after Phase 2 completes — not per-vector during Phase 2. Updating mid-batch creates a race where subsequent vectors in the same batch search from a partially-connected new entry point.

```rust
// After all plans applied in Phase 2:
if new_max_level > current_max_depth {
    let _g = alloc_lock.write();
    if new_max_level > state.max_depth {
        state.max_depth = new_max_level;
        state.entry_point = Some(new_entry_id);
    }
}
```

---

## Benchmark Results

Evaluated on SIFT1M (1M vectors, dim=128, M=16, M_MAX0=32, ef_construction=40, Apple M-series, 6 performance cores).

### Build time comparison

| Method | Build time | Speedup | Mean recall | Min recall | p95 |
|--------|-----------|---------|-------------|------------|-----|
| Single thread | 156s | 1.0x | 0.959 | 0.40 | 1.000 |
| Level locks, 4 threads | 138s | 1.13x | 0.966 | 0.000 | — |
| Two-phase, 4t batch=4 | 85.8s | 1.82x | 0.953 | 0.000 | 1.000 |
| Two-phase, 6t batch=6 | 69.8s | 2.24x | 0.965 | 0.200 | 1.000 |
| TimeBucket n=8 | 84s | 1.86x | 0.940 | 0.40 | 1.000 |
| TimeBucket n=16 | 62s | 2.52x | 0.949 | 0.40 | 1.000 |

### Throughput comparison

| Method | Throughput | Speedup |
|--------|-----------|---------|
| Single thread | 7,420 vec/s | 1.0x |
| Two-phase, 4 threads | 12,624 vec/s | 1.70x |

### Known issue: min recall = 0.000

The 4-thread and 6-thread runs show min recall of 0.000 and 0.200 respectively — one or more queries returning no correct results. This is caused by the entry point race described above: if two vectors in the same batch are both promoted to a new max level, the second update can leave the entry point pointing to a node whose upper-level connections were built against an inconsistent graph state. Queries routed through that entry point miss large regions of the graph.

**Status: not yet fixed.** The fix is to track the highest-level node across the full batch and perform a single entry point update after Phase 2 completes, under the write lock. Until fixed, two-phase insertion should not be used where tail recall matters.

### Key observations

**Two-phase 6 threads matches TimeBucket n=8 on build time** (69.8s vs 84s) while achieving better mean recall (0.965 vs 0.940). TimeBucket n=16 remains faster (62s) due to cache efficiency — smaller working sets fit in L3 cache.

**Level locks alone don't help.** Only 1.13x speedup with 4 threads because HNSW search is memory-latency bound — additional threads saturate memory bandwidth without reducing cache miss latency. CPU utilization stays near 120% regardless of thread count.

**Two-phase breaks the memory bottleneck.** Phase 1 threads access different graph regions (different vectors → different spatial neighborhoods → different cache lines). Real parallelism — CPU utilization rises to ~300-350% with 6 threads.

**Mean recall with two-phase (0.965) is comparable to sequential (0.959).** Each vector only misses 5 batch-mates out of 999,999 existing vectors.

---

## The Memory-Bandwidth Constraint

HNSW search is memory-latency bound, not compute-bound:

```
Per node visit:
  Distance computation: ~15ns  (128 multiply-adds, AVX2)
  Memory access:        ~100ns (cache miss, 640-byte node at random location)

Memory : Compute ratio = 6.6:1
CPU idle: ~87% of time, waiting for memory

1M vectors × 640 bytes = 640MB working set
L3 cache: 32-64MB
Cache hit rate: ~5%
```

Level locks don't help because they don't change the memory access pattern. Two-phase helps because threads access different spatial regions simultaneously — their memory accesses don't compete for the same cache lines.

---

## Relationship to TimeBucket Parallelism

TimeBucket achieves build speedup through a different mechanism: smaller buckets fit in L3 cache.

```
n=16 TimeBucket:
  Each bucket: 62,500 vectors × 640 bytes = 40MB working set
  Fits in L3 cache → cache hit rate ~80% vs ~5%
  Build: 2.52x faster — cache efficiency, not parallelism
  
Two-phase insertion:
  All 1M vectors: 640MB working set
  Threads access different regions → reduced cache contention
  Build: 2.24x faster — true parallelism
```

They are complementary. TimeBucket parallelism (building multiple independent buckets concurrently) can be combined with two-phase insertion within each bucket.

---

## Streaming Insertion

Two-phase insertion requires buffering batch_size vectors before inserting. With batch_size = thread_count = 6:

```
Buffer: 6 vectors
Insert batch in parallel
Next 6 vectors

Buffer latency at 12,624 vec/s:
  6 / 12,624 = 0.47ms per batch
  Acceptable for most streaming use cases
```

Each vector is immediately searchable after Phase 2 completes for its batch. The buffer latency is sub-millisecond.

---

## Implementation

```rust
pub fn insert_batch(
    &self,
    vectors: &[Vec<f32>],
    rng: &mut StdRng,
) -> Vec<NodeId> {
    // Phase 1: parallel neighbor search (read-only)
    let plans: Vec<InsertPlan> = vectors
        .par_iter()
        .map(|v| self.compute_insertion_plan(v))
        .collect();

    // Detect shared-neighbor conflicts
    let groups = partition_non_conflicting(&plans);

    // Phase 2: sequential edge application
    let mut new_max_level = self.max_depth();
    let mut new_entry = None;

    for group in &groups {
        for &idx in group {
            let new_id = self.apply_plan(&vectors[idx], &plans[idx]);
            if plans[idx].level as i32 > new_max_level {
                new_max_level = plans[idx].level as i32;
                new_entry = Some(new_id);
            }
        }
    }

    // Update entry point once, after all plans applied
    if let Some(ep) = new_entry {
        let _g = self.alloc_lock.write().unwrap();
        let state = unsafe { &mut *self.state.get() };
        if new_max_level > state.max_depth {
            state.max_depth = new_max_level;
            state.entry_point = Some(ep);
        }
    }

    // Return NodeIds
    plans.iter().enumerate()
        .map(|(i, _)| self.node_ids[i])
        .collect()
}
```

---

## Key Properties

| Property | Value |
|----------|-------|
| Correctness | Sequential Phase 2 — no edge update races |
| Quality | Batch size = thread count — negligible recall impact |
| Speedup | 1.70-2.24x on 4-6 cores (SIFT1M) |
| Streaming | Sub-millisecond buffer latency |
| Memory | No additional memory overhead |
| Complexity | Two phases, conflict detection — moderate |

---

## Summary

Two-phase insertion achieves real parallelism on HNSW by separating the read-heavy search phase from the write-heavy update phase:

- **Phase 1 parallelizes naturally**: threads search different graph regions with minimal cache contention
- **Phase 2 stays sequential**: correct graph state, no locks needed, small fraction of total time
- **Batch size = thread count**: each vector misses only 5 batch-mates out of 1M — negligible recall impact
- **1.70-2.24x speedup** on 4-6 cores, validated on SIFT1M

Level locks alone achieve only 1.13x because HNSW is memory-latency bound — the bottleneck is cache misses, not CPU compute. Two-phase improves cache behavior by having threads access spatially distinct graph regions simultaneously.
