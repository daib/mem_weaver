# MemWeaver

A vector database with predictable memory usage and temporal partitioning. Memory cost is a function of vectors, dimensions, and M — known before deployment, not discovered in production.

```
memory ≈ n × node_size
```

For dim=128, M=16: `node_size = 640 bytes`, so 1M vectors requires approximately **640 MB** — calculable before any data is inserted.

---

## Why MemWeaver

Every production vector database shares the same operational problem: you cannot know your memory cost before you deploy. You provision generously, monitor closely, and scale reactively.

MemWeaver is designed around the opposite principle: **memory is a first-class design constraint, not a runtime surprise**.

The hot/cold tiering model enforces this explicitly:

```
memory = hot_bucket_count × (n_vectors × node_size)
```

Cold buckets cost **zero RAM**. You control exactly what stays hot. No OS page cache surprises, no implicit promotion, no "it depends on query patterns" footnote.

See [DESIGN.md](DESIGN.md) for the full technical motivation and comparison with Weaviate, Qdrant, and Milvus.

---

## Benchmarks

### Arena vs Naive: Build and Search Performance

Evaluated on AWS m5d.4xlarge (Intel Xeon, AVX-512, 16 vCPUs, 64 GB RAM) using the [SIFT1M](http://corpus-texmex.irisa.fr/) dataset (1M vectors, dim=128).

**Configuration:** M=16, M_MAX0=32, ef_construction=40, ef_search=100, k=10, 50 search queries. Compiled with `target-cpu=native`.

| Implementation | Build time | Search (50 queries) | Speedup |
|---|---|---|---|
| Arena (MemWeaver) | 187s | 20.4ms | — |
| Naive (Vec-based) | 322s | 34.9ms | 1.72x slower |

Arena allocation is **1.72x faster** to build and **1.71x faster** to query due to cache locality — collocating each vector with its HNSW edges means graph traversal accesses contiguous memory.

### Memory Predictability

The arena implementation maintains a **constant ~8 MB peak overhead** throughout 1M vector insertion. The naive Vec-based implementation shows reallocation spikes of 21.6 MB at 270k vectors and 35.7 MB at 530k vectors as the underlying buffer doubles capacity.

The arena implementation also uses **25% less total RSS** at 1M vectors (608 MB growth vs 759 MB), demonstrating that arena allocation eliminates the fragmentation that accumulates in general-purpose allocators.

![Memory comparison](docs/memory_comparison.png)

---

### TimeBucket: Temporal Partitioning

TimeBucket partitions the HNSW index across time windows. Each bucket maintains an independent HNSW graph. Queries search across all relevant buckets and merge results with optional recency weighting.

**Configuration:** M=16, M_MAX0=32, ef_construction=40, ef_search=100, k=10, 1000 queries, SIFT1M (1M vectors, dim=128), Apple M-series, StdRng.

#### Build and Query Tradeoffs

| Buckets | Vectors/bucket | Build time | HNSW search (1000q) | Mean recall@10 | P95 recall | Build speedup |
|---------|---------------|-----------|---------------------|----------------|------------|---------------|
| n=1 (baseline) | 1,000,000 | 156s | 523ms | 0.959 | 1.000 | 1.0x |
| n=4 | 250,000 | 110s | 1,102ms | 0.905 | 1.000 | 1.42x |
| n=8 | 125,000 | 84s | 2,028ms | 0.940 | 1.000 | 1.86x |
| n=16 | 62,500 | 62s | 3,477ms | 0.949 | 1.000 | 2.52x |

**Key findings:**

- **Build time scales sublinearly** — 2.52x faster at n=16 with comparable recall quality
- **P95 recall = 1.0 across all configurations** — at least 95% of queries return perfect recall regardless of bucket count
- **Mean recall differences between n=4, n=8, n=16 are within noise** over 1000 queries — bucket count does not meaningfully degrade recall quality
- **Query time scales linearly** with bucket count — the expected tradeoff for searching N independent graphs

#### Hot/Cold Tier Performance

MemWeaver supports explicit hot/cold tiering. Buckets are swapped to disk atomically — memory freed immediately on swap-out, fully restored on swap-in. No OS page cache involvement.

**n=8 buckets, 1M vectors, 1000 queries, Apple M-series SSD:**

| State | HNSW search (1000q) | Mean recall | Min recall | Operation time |
|-------|---------------------|-------------|------------|----------------|
| Hot (in memory) | 2,028ms | 0.9396 | 0.40 | — |
| Cold (disk reads) | 10,696ms | 0.9396 | 0.40 | swap out: 130ms |
| Restored (hot again) | 2,017ms | 0.9396 | 0.40 | swap in: 120ms |

**Key findings:**

- **Recall is identical across all three states** — hot, cold, and restored produce exactly the same results to four decimal places. Tiering is a pure performance/memory tradeoff with zero correctness cost.
- **Cold queries are 5.3x slower** on Mac SSD. On NVMe in production this gap is significantly smaller — NVMe random read latency (~20-100μs) versus Mac SSD.
- **Swap operations are fast** — 130ms to swap 8 buckets out, 120ms to restore. Suitable for background eviction without impacting query availability.
- **Memory accounting is exact** — `hot_buckets × bucket_size`. Cold buckets consume zero RAM. No OS mercy required.

#### RNG Matters at Scale

Using `SmallRng` instead of `StdRng` at 1M vectors produces severely degraded results: mean recall drops from 0.959 to 0.799 and multiple queries return zero recall. `SmallRng`'s weaker statistical properties cause poor level assignment during HNSW construction at this scale. **Always use `StdRng`** for production workloads.

| RNG | Mean recall | Min recall |
|-----|-------------|------------|
| StdRng | 0.959 | 0.40 |
| SmallRng | 0.799 | 0.00 |

**When to use TimeBucket:**

- **Write-heavy workloads** — faster build time means lower insertion latency as the index grows
- **Time-range queries** — search only recent buckets, skipping cold data entirely
- **Memory tiering** — age out cold buckets to disk while keeping recent data hot
- **Tune bucket count for your workload** — fewer buckets for query-heavy, more buckets for write-heavy or temporal filtering

---

## Hot/Cold Tier Design

MemWeaver's tiering is explicit and predictable by design. Unlike mmap-based approaches where the OS decides what stays resident, MemWeaver gives you full control:

```rust
// Swap a bucket to disk — memory freed immediately
index.swap_out(bucket_seq)?;

// Restore a bucket — identical search quality guaranteed
index.swap_in(bucket_seq)?;

// Query cold buckets directly without swapping in
// Uses temporary mmap for query duration only, released after search
index.search_including_cold(query, k, ef)?;
```

The memory guarantee holds unconditionally:

```
memory = hot_bucket_count × (n_vectors × node_size)
       + active_cold_queries × ~640KB  // pages touched during traversal
```

This is different from every vector database that uses persistent mmap and calls it "efficient memory usage." Those systems depend on OS page cache behavior. MemWeaver's memory cost is calculable before deployment.

---

## Arena Design

### The problem with naive HNSW storage

A naive implementation stores vectors and edges in separate growable arrays:

```rust
struct HnswNaive {
    nodes: Vec<Vec<Vec<NodeId>>>,  // layers × neighbors per node
    vectors: Vec<Vec<f32>>,
}
```

This causes repeated vector reallocation as data grows. Each reallocation temporarily doubles memory usage, creates fragmentation, and separates vector data from edge data — causing cache misses during graph traversal.

### Collocated arena layout

MemWeaver stores each vector adjacent to its HNSW edges in a single arena block:

```
[ vector_1 | edges_1 | vector_2 | edges_2 | ... | vector_n | edges_n ]
```

Each `(vector, edges)` pair is a **Node**. Nodes within one arena form a **NodeBlock**. The entire HNSW graph is:

```rust
struct HnswArena {
    blocks: Vec<NodeBlock>,
}
```

### NodeId encoding

To locate a node, we need its block index and its offset within the arena. Rather than storing these separately, we encode both directly into the 32-bit NodeId:

```
NodeId (u32): [ block_index: 14 bits | offset >> 3: 18 bits ]
```

- Arena size: 2MB (one huge page). Requires 21 bits to address all offsets.
- 8-byte alignment: bottom 3 bits of offset are always 0, saving 3 bits.
- Result: 18 bits for offset, 14 bits for block index → **32 GB addressable**.

```rust
node_id = (block_index << 18) | ((offset >> 3) & ((1 << 18) - 1))
```

For 64-byte alignment (256 GB addressable):

```rust
node_id = (block_index << 15) | ((offset >> 6) & ((1 << 15) - 1))
```

This encoding reduces node lookup to **two pointer hops** — block array lookup, then offset arithmetic — compared to three hops in a naive approach.

### Alignment

In practice, MemWeaver uses **8-byte alignment**. For typical configurations the node data is naturally a multiple of 64 bytes:

```
dim=128, M=16, max_layer=0:
  node_size = 512 (vector) + 128 (edges) = 640 bytes = 64 × 10 ✓
```

Since node sizes are naturally cache-line aligned, 8-byte and 64-byte arena alignment produce identical performance. This was validated empirically: benchmarks showed no measurable difference between the two configurations.

### Edge layout

Given a node's base address, edge locations are computed arithmetically:

- Layer 0: starts at `base + vector_size`, capacity `M_MAX0`
- Layer l > 0: starts at `base + vector_size + M_MAX0 × 4 + (l-1) × M × 4`, capacity `M`

No per-node metadata is needed to locate edges — only the global constants `dim`, `M`, and `M_MAX0`.

### max_layer is not collocated

The max layer of each node is not stored alongside the vector and edge data. It is not on the critical path during HNSW traversal — only the entry point's level is needed. Storing max_layer separately keeps each node's size a clean multiple of 64 bytes.

### Predictable memory formula

```
memory ≈ n × node_size

node_size = dim × 4  +  M_MAX0 × 4  +  higher_layer_nodes × M × 4
```

For dim=128, M=16, M_MAX0=32 (approximately 6% of nodes reach layer 1+):

```
node_size ≈ 640 bytes
1M vectors ≈ 640 MB
10M vectors ≈ 6.4 GB
100M vectors ≈ 64 GB
```

---

## Running the Benchmarks

```bash
# Download SIFT1M dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz

# Set dataset path
export SIFT1M_BASE_PATH=/path/to/sift

# Run benchmarks (release mode required)
RUSTFLAGS="-C target-cpu=native" cargo test --release

# Arena only
SIFT1M_HNSW_BENCH_VARIANT=arena SIFT1M_HNSW_BENCH_SAMPLE_SIZE=1 cargo bench -p index --bench hnsw_sift1m 2> arena.txt

# Naive only
SIFT1M_HNSW_BENCH_VARIANT=naive SIFT1M_HNSW_BENCH_SAMPLE_SIZE=1 cargo bench -p index --bench hnsw_sift1m 2> naive.txt

# TimeBucket recall + hot/cold/restore benchmark
cargo test --release -p index --test sift1m_time_bucket_recall

# Generate memory comparison chart
python3 scripts/plot_memory.py arena.txt naive.txt
```

---

## Current Status

- [x] HNSW index with arena allocator
- [x] Collocated node layout (vector + edges)
- [x] NodeId encoding for O(1) node lookup
- [x] SIMD distance computation (AVX2/AVX-512 via `wide` crate)
- [x] Benchmarked on SIFT1M
- [x] Time-bucketed multi-HNSW for temporal relevance
- [x] Configurable aging policy
- [x] Recency-weighted search with pluggable distance adjustment
- [x] Explicit hot/cold tiering — swap_out / swap_in
- [x] Bit-perfect restore — identical recall after swap cycle
- [x] Cold bucket search via temporary mmap — zero persistent memory
- [ ] Object storage cold tier (S3/GCS/Azure via `object_store`)
- [ ] Memory controller — automatic eviction policy
- [ ] pgvector comparison benchmarks
- [ ] Parallel index construction
- [ ] Edge compression (delta encoding)

---

## License

MIT
