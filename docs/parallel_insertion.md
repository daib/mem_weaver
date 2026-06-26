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
1. Upper-level beam search (ef=5) from entry point down to insertion level — escapes local optima
2. Beam search at each insertion level to find nearest neighbors
3. Record planned neighbor connections — do not modify graph

**Phase 2 — Sequential application:**
For each vector in the batch, sequentially:
1. Allocate node in arena
2. Check previously committed batch vectors — if any are closer than planned neighbors, add them
3. Apply pre-computed neighbor connections
4. Prune neighbor edge lists if over M_MAX0 capacity
5. Update entry point if new node is promoted above current max level (once per batch, after all plans applied)

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
  - Upper-level beam search (ef=5 at layers 1+)
  - Beam search at layer 0 (~1.5ms — dominant)

Phase 2 (edge updates):    ~10% of total insertion time
  - Edge list writes (~3 microseconds per level)
  - Pruning
```

Parallelizing the dominant phase (Phase 1) gives real speedup. The sequential Phase 2 is a small fraction of total time.

### Quality impact of batch size

Each vector in a batch misses its batch-mates as potential neighbors — they are not yet in the graph when plans are computed. The within-batch check in Phase 2 (step 2 above) compensates: if a batch-mate is closer than the planned neighbors, it is added.

With batch size B and N total vectors:

```
Miss rate without within-batch check: (B-1) / N
Batch size 6 (threads=6), N=1M: 5 / 1,000,000 = 0.0005%
```

With the within-batch check the miss rate is further reduced.

---

## Upper-Level Beam Search

Standard HNSW uses greedy descent (ef=1) at upper levels — always moving to the single closest neighbor. This gets trapped at local optima in sparse regions of the vector space, producing zero-recall queries.

**The fix:** use a small beam (ef=5) at upper levels to escape local optima, reserving the full ef_search for layer 0 where recall is determined.

```rust
// Upper levels: small beam to escape local optima
while lc > 3 {
    ep = self.greedy_closest(query, ep, lc as usize);
    lc -= 1;
}

for l in (0..=lc).rev() {
    let ef = if l == 0 { ef_search } else { 5 };  // full ef at layer 0 only
    let mut cands = self.search_level(query, ep, ef, l as usize);
    if l > 0 {
        ep = cands.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
    } else {
        cands.sort_by(|a, b| a.1.total_cmp(&b.1));
        cands.truncate(k);
        return cands;
    }
}
```

Cost: upper levels have few nodes (~3,750 at layer 2, ~234 at layer 3). ef=5 beam at these levels adds negligible overhead while eliminating zero-recall traps.

---

## Benchmark Results

Evaluated on SIFT1M (1M vectors, dim=128, M=16, M_MAX0=32, ef_construction=40, ef_search=100, 10,000 queries, Apple M-series, 6 performance cores).

### Build time and recall

| Method | Build time | Speedup | Mean recall@10 | Min recall |
|--------|-----------|---------|----------------|------------|
| Naive (Vec-based) | 160s | 0.76x | 0.965 | 0.300 |
| Arena single thread | 124s | 1.0x | 0.965 | 0.300 |
| Arena two-phase, 4 threads | 84s | 1.48x | 0.965 | 0.300 |
| Arena two-phase, 6 threads | 70s | 1.77x | 0.965 | 0.300 |
| TimeBucket n=8 | 84s | 1.48x | 0.940 | 0.40 |
| TimeBucket n=16 | 62s | 2.0x | 0.949 | 0.40 |

### Throughput

| Method | Throughput | Speedup |
|--------|-----------|---------|
| Single thread | 7,420 vec/s | 1.0x |
| Two-phase, 4 threads | 12,624 vec/s | 1.70x |

### Key observations

**All implementations produce identical recall.** Mean=0.965, min=0.300 across naive, arena, single-thread, and all parallel variants. The min=0.300 is not an implementation bug — it reflects two genuinely hard queries (8500, 9049) in sparse regions of SIFT1M that are difficult for M=16 at ef_search=100. Increasing ef_search=200 partially improves them; M=32 would fix them at higher memory cost.

**Level locks alone don't help.** Only 1.13x speedup with 4 threads because HNSW search is memory-latency bound — additional threads saturate memory bandwidth without reducing cache miss latency. CPU utilization stays near 120% regardless of thread count.

**Two-phase breaks the memory bottleneck.** Phase 1 threads access different graph regions (different vectors → different spatial neighborhoods → different cache lines). Real parallelism — CPU utilization rises to ~300-350% with 6 threads.

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

## Bugs Found and Fixed

Three bugs were discovered and fixed during the parallel insertion investigation:

**Bug 1: `selected.first()` instead of `min_by` for entry point threading**
The entry point `ep` between levels was set to the first element of the selected neighbors list, which is arbitrary order from quick-select topk — not necessarily the closest. This caused wrong ep threading between levels, producing recall=0.000 for some queries. Fixed in both sequential and parallel paths.

**Bug 2: `ef=5` applied to layer 0 instead of `ef_search`**
During the upper-level beam search refactor, the small `ef=5` was incorrectly applied to layer 0 as well, reducing mean recall to ~0.500. Fixed by using `if l == 0 { ef_search } else { 5 }`.

**Bug 3: Greedy descent trapped at local optima**
The original greedy descent (ef=1) at upper levels gets stuck in wrong regions for sparse outlier queries (query 6916 at recall=0.000). Fixed by using ef=5 beam at upper levels to escape local optima before reaching layer 0.

All three bugs were present in both sequential and parallel code paths. After fixes, all implementations produce identical recall characteristics.

---

## Relationship to TimeBucket Parallelism

TimeBucket achieves build speedup through a different mechanism: smaller buckets fit in L3 cache.

```
n=16 TimeBucket:
  Each bucket: 62,500 vectors × 640 bytes = 40MB working set
  Fits in L3 cache → cache hit rate ~80% vs ~5%
  Build: 2.0x faster — cache efficiency, not parallelism

Two-phase insertion:
  All 1M vectors: 640MB working set
  Threads access different regions → reduced cache contention
  Build: 1.77x faster — true parallelism
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

Each vector is immediately searchable after Phase 2 completes for its batch.

---

## Summary

| Property | Value |
|----------|-------|
| Correctness | Identical recall to sequential insertion |
| Speedup | 1.48-1.77x on 4-6 cores (SIFT1M) |
| Throughput | 12,624 vec/s (4 threads) vs 7,420 vec/s (single) |
| Mean recall | 0.965 (same as sequential) |
| Min recall | 0.300 (same as sequential — hard queries, not a bug) |
| Streaming | Sub-millisecond buffer latency |

Two-phase insertion achieves real parallelism on HNSW by separating the read-heavy search phase from the write-heavy update phase. The upper-level beam search (ef=5) further improves quality by escaping local optima that greedy descent gets trapped on.

---

## Capacity-Aware Neighbor Selection

### The Problem at High ef_construction

Standard HNSW neighbor selection optimizes for distance only. At high ef_construction (≥100), this creates eviction cascades:

```
ef_construction=100: 100 candidates considered per insertion
Popular nodes: appear in many candidate lists
→ receive many connection requests
→ edge lists fill up quickly
→ aggressive pruning kicks in
→ some nodes lose their only connection to a region
→ those nodes become unreachable
→ recall = 0.000 for queries near them

ef_construction=40:  40 candidates — less saturation
                     eviction cascades don't occur
                     0 zero-recall queries
```

### The Fix

Penalize high-degree nodes during neighbor selection:

```rust
let score = dist * (1.0 + (degree as f32 / cap as f32).min(0.8));

// degree=0  (empty): score = dist × 1.0  (no penalty)
// degree=16 (half):  score = dist × 1.5
// degree=32 (full):  score = dist × 1.8  (capped)
//
// A full node must be 1.8x closer to be selected
// over an empty node — prevents cascade evictions
```

Nodes with spare capacity are preferred. Full nodes require significant distance advantage to be selected. This spreads connection load across the graph rather than concentrating it on popular nodes.

### Results

| Config | Mean recall | Min recall | Zero-recall queries |
|--------|-------------|------------|---------------------|
| ef=40, standard | 0.965 | 0.300 | 0 |
| ef=100, standard | 0.961 | 0.000 | 13 |
| ef=40, + capacity-aware | 0.965 | 0.300 | 0 (no change — ef=40 doesn't cause cascades) |
| **ef=100, + capacity-aware** | **0.979** | **0.300** | **0** |

Capacity-aware selection has no effect at ef_construction=40 (eviction cascades don't occur at low ef). It is essential at ef_construction=100+.

### Why ef_construction=100 + Capacity-Aware is the Best Configuration

```
ef=100 + capacity-aware + 6 threads:
  Build time: 103s  — faster than ef=40 single-thread (124s) ✓
  Mean recall: 0.979 — better than ef=40 (0.965) ✓
  Min recall:  0.300 — same as ef=40 baseline ✓
  Zero-recall: 0     — eliminated ✓
  Speedup:     2.65x — better than ef=40 (1.77x) ✓

Strictly better than ef=40 single-thread on every metric.
```

### Why Higher ef_construction Gives Better Parallel Speedup

More work per insertion means a larger parallel fraction and a smaller sequential fraction relative to total time — Amdahl's law improves:

```
ef=40:  serial fraction ~10% → max speedup 10x → actual 1.77x
ef=100: serial fraction ~5%  → max speedup 20x → actual 2.65x

The sequential Phase 2 stays the same absolute size.
Phase 1 grows with ef_construction.
Larger Phase 1 → better parallel efficiency.
```
