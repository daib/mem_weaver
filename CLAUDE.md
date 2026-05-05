# MemWeaver

Vector database with predictable memory costs.

## Architecture
- Arena-based memory management (no GC)
- HNSW graph for hot tier ANN search
- Time-bucketed design for streaming workloads
- Lance format for cold tier (future)

## Key invariants
- All vector allocations must be 64-byte aligned (AVX-512)
- VectorBlocks dropped before Arenas always
- Arena eviction removes entire time bucket atomically

## Conventions
- No raw pointers in public API
- unsafe blocks must have safety comments
- All public functions documented

## Current state
- VectorStore: complete, SIFT1M validated
- HNSW: in progress

## Memory Model
- VectorBlock has no lifetime parameter (PhantomData removed)
- Safety invariant: arena_blocks dropped before arenas
- Field declaration order in VectorStore enforces drop order

## Key Types
- VectorId(u64) — external application ID, stable
- LocationIndex — (arena_idx, block_idx, slot)
- Arena uses mmap, 4MB pages, 64-byte aligned

## HNSW Parameters
- M = 16 (connections per node)
- ef_construction = 200
- Layer assignment: floor(-ln(uniform) * 1/ln(M))

## Testing
- SIFT1M base vectors loaded and validated
- Recall@10 = 1.0 for brute force confirmed
- Run: cargo test --release for performance-sensitive tests

## What NOT to do
- Don't add PhantomData with lifetime to VectorBlock
- Don't use Vec instead of VecDeque for arenas
- Don't store references to arenas — use indices
