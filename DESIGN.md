# MemWeaver: A Vector Database That Knows Its Memory Cost Before You Deploy

I used to be an LLM skeptic. The idea that predicting the next token could produce anything resembling intelligence seemed absurd to me. Then I became a Claude Pro user — and not just for work. I started using it to navigate complex personal situations, and I was floored. A natural next step for an engineer is to go spelunking in the underlying technology. That's how I ended up deep in the source code of vector databases, and why I ended up designing MemWeaver.

---

## The Problem With Every Vector Database Today

Every production vector database I've evaluated — Weaviate, Qdrant, Milvus — shares the same fundamental flaw: **you cannot know your memory cost before you deploy**. You provision generously, monitor closely, and scale reactively. For memory-intensive systems, this is a dangerous game.

When I downloaded and read through [Weaviate](https://github.com/weaviate/weaviate)'s source code, I noticed it's written in Go and uses Go slices to store vectors. My experience with in-memory search and analytics systems immediately set off alarm bells. Weaviate probes Go heap memory every 500ms and blocks new allocations when usage gets too high. This reactive approach has two problems: 500ms is an eternity in a high-throughput system, and the GC overhead means you're effectively paying for 3x your average memory usage just to stay safe from OOMs.

I learned this lesson the hard way working with a [C++ Kafka client](https://github.com/confluentinc/librdkafka) in production. Average memory usage sat comfortably under 2GB, yet pods allocated 8GB would still occasionally OOM. The spiky, unpredictable nature of garbage-collected memory management is not a minor inconvenience — it's a fundamental architectural liability.

[Qdrant](https://github.com/qdrant/qdrant), written in Rust, fares better. It uses [jemalloc](https://github.com/jemalloc/jemalloc), which reduces fragmentation by allocating memory in size classes. But it's still reactive. Jemalloc can still scatter allocations across multiple arenas, undermining the TLB locality benefits you'd otherwise get from huge pages.

[Milvus](https://github.com/milvus-io/milvus) sidesteps Go's limitations for vector math by using CGO to call [FAISS](https://github.com/facebookresearch/faiss). That works, but it's an awkward seam in the architecture.

None of these systems can tell you, before deployment: *here is exactly how much memory this will use*.

MemWeaver can. The memory cost is a function of the number of vectors, their dimensionality, and the HNSW construction parameter M — knowable upfront, not discovered in production.

---

## Memory as a First-Class Citizen

The core insight behind MemWeaver is that **memory should be designed in, not managed around**.

This isn't a new idea in high-performance systems. [PagedAttention](https://arxiv.org/abs/2309.06180) demonstrated the power of explicit memory management for LLM inference. Arena allocators are standard practice in game engines, databases, and trading systems. The question is why vector databases haven't adopted these patterns — and the answer, I think, is that most were built by teams optimizing for feature velocity rather than operational predictability.

Arena-based memory management offers four concrete benefits for a vector database:

**Predictable usage.** When your application controls allocation rather than delegating to a GC or generic allocator, memory cost becomes a function of your data model — not runtime behavior. You can calculate it. You can guarantee it.

**No fragmentation.** Related data is packed together and released together. There are no small holes left behind after eviction. Over time, fragmentation in GC-based systems compounds; in arena-based systems, it simply doesn't form.

**TLB efficiency.** When related data lives in contiguous huge pages, the CPU doesn't need to walk the page table multiple times to access it. Fewer TLB misses mean faster traversal — critical for HNSW graph walks.

**SIMD throughput.** SIMD instructions operate on blocks of data. When vectors are packed together in arenas, similarity computations can be vectorized efficiently. Go-based systems like Weaviate don't leverage SIMD at all for similarity scoring. MemWeaver does.

---

## Timeseries as a Natural Consequence

Once you commit to arena-based memory management, something interesting falls out of the design almost for free: **timeseries bucketing**.

Each arena maps cleanly onto a time window. When a bucket ages out, you release the entire arena — no object-by-object cleanup, no GC pressure, no fragmentation. Eviction becomes a single operation.

This turns out to matter a great deal for real-world use cases like news feed recommendation or social media search, where recent content is almost always more relevant than older content. Current vector databases treat time as just another metadata filter — you query the entire dataset and filter afterward. This is both expensive and semantically wrong. Time should influence *how* data is indexed, not just *which* results are returned.

In MemWeaver, time buckets are first-class structural elements:

**Hot tier (recent buckets):** Data is indexed with HNSW — expensive to build, but extremely fast to query. Since recent data receives the most queries, paying the build cost upfront is the right tradeoff. All of this lives in arenas in memory, potentially on huge pages.

**Cold tier (older buckets):** As buckets age, they're flushed to disk and blob storage. Multiple buckets are consolidated and re-indexed using IVF with product quantization — slower to query, but far more storage-efficient. SIMD instructions accelerate both the compression and the similarity computation.

---

## The Query Path

Query results from different time buckets are merged with recency weighting applied before final top-k selection. Recent buckets score higher than older ones, so the final ranking reflects both semantic similarity and temporal relevance.

This sounds straightforward but is actually where the hardest engineering lives. Score distributions between HNSW and IVF aren't directly comparable, and adding a recency weight on top introduces another tuning dimension. Getting this wrong produces results that are either uselessly recency-biased or no better than a plain similarity search. The right balance is use-case dependent and needs to be tunable — this is where the most work remains.

---

## Distributed Architecture

A single machine can't handle the volume of a large social network's content stream. MemWeaver is designed to partition across query nodes, with queries broadcast to all nodes and results merged with top-k extraction at the coordinator.

Because the system produces approximate results by design, it doesn't require strict consistency — eventual consistency is sufficient. This makes partitioning and rebalancing significantly simpler. When a node fills up, in-memory time bucket data is flushed to disk and then blob storage, following patterns established by Bigtable and Iceberg. A background service consolidates fragments over time.

---

## Why Not Just Fix Qdrant?

A fair question. Qdrant is already in Rust, already uses jemalloc, and already has a reasonable architecture. Why not layer time buckets and explicit arenas on top?

The honest answer is: you might be able to. Incremental improvement on an existing system is often the right call. But the memory model in MemWeaver isn't a feature you add — it's a constraint that shapes the entire data layout, index structure, and eviction strategy from the ground up. Retrofitting arena allocation and time-native indexing into an existing codebase would likely mean rewriting the storage layer anyway.

The Kafka vs. Redpanda story is instructive here, but not in the direction you might expect. [Redpanda](https://redpanda.com/) is a beautifully engineered C++ reimplementation of Kafka — and it hasn't displaced Kafka, because Kafka is disk-bound, not memory-bound. A more efficient memory model doesn't help a disk-bound system. Vector databases are the opposite: they are deeply memory-bound. This is exactly the category where explicit memory management pays off.

---

## What MemWeaver Is, Precisely

**Existing systems:** provision memory empirically, monitor, scale reactively.

**MemWeaver:** `memory cost = f(vectors, dimensions, M)` — known before deployment.

That formula is the product. Everything else — the arenas, the time buckets, the HNSW/IVF tiering, the SIMD throughput — is in service of making that formula true and useful.

I've spent several years building memory-intensive in-memory search and analytics systems serving thousands of backends in production. The patterns described here aren't theoretical. They are things I have implemented, debugged under load, and learned from when they broke. MemWeaver is the system I wished existed when I needed it.

The query path merging problem is unsolved. The distributed coordination details need more rigor. But the memory model is sound, and that's the part that matters most.
