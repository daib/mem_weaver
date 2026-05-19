# MemWeaver

A vector database built for workloads where time is a first-class search dimension. Memory cost is known before deployment, not discovered in production.

```
memory = hot_bucket_count × (n_vectors × node_size)
```

For dim=128, M=16: `node_size = 640 bytes`. 1M vectors in 4 hot buckets = **2.56 GB**, calculable before any data is inserted.

---

## When to Use MemWeaver

MemWeaver is designed for content that ages — where recent data is more relevant than old data, and memory cost must be predictable at scale.

**Good fit:**

**News and content feeds** — Find similar articles weighted by recency. Recent articles matter more than archived ones. Hot tier: today's content. Cold tier: archives on disk or S3.

**Document search** — Find relevant docs that are actually current. Old architecture docs and recent RFCs should rank differently. Confluence search is terrible partly because it treats a 5-year-old doc the same as one updated last week.

**Code search** — Find recent examples of patterns in your codebase. New implementations ranked above deprecated ones. "How do we handle auth now" should return this year's code, not 2019's.

**Financial and research signals** — Time-window queries: find events similar to this one in the 24 hours before that incident. Temporal precision matters.

**Not a good fit:**

- **AI conversation memory** — collections are small (hundreds to thousands of vectors per user). Use pgvector with a timestamp filter and flat search.
- **High-volume log embedding** — embedding every log line at production ingestion rates is cost-prohibitive. Use structured search for logs.
- **Static collections** — no temporal relevance, no hot/cold tiering needed. pgvector is simpler and sufficient.

---

## Why MemWeaver

Every production vector database has the same operational problem: you cannot know your memory cost before you deploy. You provision generously, monitor closely, and scale reactively.

MemWeaver is designed around the opposite principle: **memory is a first-class design constraint, not a runtime surprise**.

The hot/cold tiering model enforces this explicitly. Cold buckets cost zero RAM. You control exactly what stays hot. No OS page cache surprises, no implicit promotion, no "it depends on query patterns" footnote.

**pgvector doesn't know what year it is. MemWeaver does.**

See [DESIGN.md](DESIGN.md) for the full technical motivation.

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

The arena implementation also uses **25% less total RSS** at 1M vectors (608 MB growth vs 759 MB).

![Memory comparison](docs/memory_comparison.png)

---

### TimeBucket: Temporal Partitioning

TimeBucket partitions the HNSW index across time windows. Each bucket maintains an independent HNSW graph. Queries search across relevant buckets and merge results with optional recency weighting.

**Configuration:** M=16, M_MAX0=32, ef_construction=40, ef_search=100, k=10, 1000 queries, SIFT1M (1M vectors, dim=128), Apple M-series, StdRng.

#### Build and Query Tradeoffs

| Buckets | Vectors/bucket | Build time | HNSW search (1000q) | Mean recall@10 | P95 recall | Build speedup |
|---------|---------------|-----------|---------------------|----------------|------------|---------------|
| n=1 (baseline) | 1,000,000 | 156s | 523ms | 0.959 | 1.000 | 1.0x |
| n=4 | 250,000 | 110s | 1,102ms | 0.905 | 1.000 | 1.42x |
| n=8 | 125,000 | 84s | 2,028ms | 0.940 | 1.000 | 1.86x |
| n=16 | 62,500 | 62s | 3,477ms | 0.949 | 1.000 | 2.52x |

- **Build time scales sublinearly** — 2.52x faster at n=16 with comparable recall
- **P95 recall = 1.0 across all configurations** — 95% of queries return perfect recall
- **Mean recall differences are within noise** — bucket count does not meaningfully degrade recall
- **Query time scales linearly** — expected tradeoff for searching N independent graphs

#### Hot/Cold Tier Performance

MemWeaver supports explicit hot/cold tiering. Buckets are swapped to disk atomically — memory freed immediately on swap-out, fully restored on swap-in. No OS page cache involvement.

**n=8 buckets, 1M vectors, 1000 queries, Apple M-series SSD:**

| State | HNSW search (1000q) | Mean recall | Min recall | Operation time |
|-------|---------------------|-------------|------------|----------------|
| Hot (in memory) | 2,028ms | 0.9396 | 0.40 | — |
| Cold (disk reads) | 10,696ms | 0.9396 | 0.40 | swap out: 130ms |
| Restored (hot again) | 2,017ms | 0.9396 | 0.40 | swap in: 120ms |

- **Recall is identical across all three states** — tiering is a pure performance/memory tradeoff with zero correctness cost
- **Cold queries are 5.3x slower** on Mac SSD — significantly better on NVMe in production
- **Swap operations are fast** — 130ms to swap 8 buckets out, 120ms to restore
- **Memory accounting is exact** — `hot_buckets × bucket_size`, cold buckets cost zero RAM

#### RNG Matters at Scale

Using `SmallRng` instead of `StdRng` at 1M vectors produces severely degraded results. **Always use `StdRng`** for production workloads.

| RNG | Mean recall | Min recall |
|-----|-------------|------------|
| StdRng | 0.959 | 0.40 |
| SmallRng | 0.799 | 0.00 |

---

## Hot/Cold Tier Design

MemWeaver's tiering is explicit and predictable by design. Unlike mmap-based approaches where the OS decides what stays resident, MemWeaver gives you full control:

```rust
// Swap a bucket to disk — memory freed immediately
index.swap_out(bucket_seq)?;

// Restore a bucket — identical search quality guaranteed
index.swap_in(bucket_seq)?;

// Query cold buckets directly without swapping in
// Temporary file read for query duration only
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

This causes repeated reallocation as data grows — temporarily doubling memory usage, creating fragmentation, and separating vector data from edge data causing cache misses during graph traversal.

### Collocated arena layout

MemWeaver stores each vector adjacent to its HNSW edges in a single arena block:

```
[ vector_1 | edges_1 | vector_2 | edges_2 | ... | vector_n | edges_n ]
```

Each `(vector, edges)` pair is a **Node**. Nodes within one arena form a **NodeBlock**:

```rust
struct HnswArena {
    blocks: Vec<NodeBlock>,
}
```

### NodeId encoding

Both block index and offset are encoded directly into the 32-bit NodeId:

```
NodeId (u32): [ block_index: 14 bits | offset >> 3: 18 bits ]
```

- Arena size: 2MB. Requires 21 bits to address all offsets.
- 8-byte alignment: bottom 3 bits always 0, saving 3 bits.
- Result: 18 bits offset, 14 bits block index → **32 GB addressable**.

This encoding reduces node lookup to **two pointer hops** versus three in a naive approach.

### Predictable memory formula

```
memory ≈ n × node_size

node_size = dim × 4  +  M_MAX0 × 4  +  higher_layer_nodes × M × 4
```

For dim=128, M=16, M_MAX0=32:

```
node_size ≈ 640 bytes
1M vectors  ≈ 640 MB
10M vectors ≈ 6.4 GB
100M vectors ≈ 64 GB
```

---

## Durability

MemWeaver is a search index, not a primary store. Vectors should live in durable upstream storage — Kafka, Cassandra, S3, or similar.

On crash recovery, MemWeaver replays from the last committed upstream offset. Cold buckets on disk are recovered from the manifest. Hot buckets are rebuilt from the upstream source.

For small deployments without a durable upstream, a built-in WAL provides single-node crash recovery.

---

## Running the Benchmarks

```bash
# Download SIFT1M dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz

# Set dataset path
export SIFT1M_BASE_PATH=/path/to/sift

# Arena vs naive
SIFT1M_HNSW_BENCH_VARIANT=arena SIFT1M_HNSW_BENCH_SAMPLE_SIZE=1 \
  cargo bench -p index --bench hnsw_sift1m 2> arena.txt

SIFT1M_HNSW_BENCH_VARIANT=naive SIFT1M_HNSW_BENCH_SAMPLE_SIZE=1 \
  cargo bench -p index --bench hnsw_sift1m 2> naive.txt

# TimeBucket recall + hot/cold/restore benchmark
cargo test --release -p index --test sift1m_time_bucket_recall

# Memory comparison chart
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
- [x] Cold bucket search via temporary file read
- [ ] S3/GCS/Azure cold tier via `object_store`
- [ ] Memory controller — automatic eviction policy
- [ ] pgvector comparison benchmarks
- [ ] Parallel index construction (level-lock pipelining)
- [ ] Per-cluster HNSW for reader node cold tier
- [ ] Stateless reader nodes for horizontal read scaling

---

## License

MIT
