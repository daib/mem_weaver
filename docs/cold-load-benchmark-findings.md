# Cold-load benchmarking: mem_weaver vs. LanceDB

Investigation into disk-cold loading/query performance for mem_weaver's HNSW index vs.
LanceDB, and follow-on optimization work on mem_weaver's block-restore path. All numbers
are from SIFT1M (1M base vectors, dim=128) on the same machine.

## 1. Why "cold" is hard to measure correctly

The first attempt (`swap_out()` then `swap_in()` in the same process) produced a "238ms
disk load" number that turned out to be **page-cache-speed, not real disk I/O** —
build/swap-out and load/swap-in happened seconds apart in the same process, so every page
was still resident in the OS page cache. ~600MB in ~123ms implies ~4.9 GB/s, well above
sustained NVMe throughput and consistent with a page-cache memcpy.

A genuine cold-disk measurement requires either:
- Separate OS processes, with the page cache actually dropped (`sudo purge` on macOS,
  `echo 3 > /proc/sys/vm/drop_caches` on Linux) between persist and load, or
- Data the current process never touched.

This is why the benchmarks below are split into separate `cargo test` invocations
(separate processes) with `sudo purge` run manually between build and query.

## 2. mem_weaver vs. LanceDB: architectural shapes differ

- **mem_weaver**: eager, whole-index bulk load. `swap_in()` / `load_blocks_from_dir()`
  reads every block into RAM up front; once loaded, every query is a pure in-memory HNSW
  walk with no further I/O.
- **LanceDB**: lazy, mmap-backed. `open_table()` is near-instant metadata attach; the
  actual cost is paid lazily per-query as pages fault in during search.

There's no way to make these directly equivalent — the fairest comparison is
**time-to-first-answer** (attach/load + first query) vs. **steady-state throughput**
once each is warm.

## 3. Cold time-to-first-answer

| | mem_weaver (HNSW) | LanceDB (auto ~1000 partitions) | LanceDB (forced 1 partition) |
|---|---|---|---|
| Attach/open | 389–427 ms (full eager reconstruct, CRC-verified) | 1.37 ms | 4.40 ms |
| First query | 0.45–0.46 ms | 210.63 ms | 259.31 ms |
| **Total** | **~389–427 ms** | **~212 ms** | **~264 ms** |

**LanceDB wins on cold time-to-first-answer.** Its lazy/mmap model only pages in the
graph nodes touched by one query's traversal, not the whole 1M-vector index. mem_weaver's
eager model pays for the entire index regardless of how many queries follow.

Forcing LanceDB to use a single IVF partition (i.e., one HNSW graph over all 1M vectors,
no clustering) only made the cold first query ~23% slower (259ms vs 210ms), not
multiple-x slower. This means the dominant reason LanceDB's cold query is fast is **not**
IVF partitioning narrowing the search to a small cluster — it's that **HNSW search
itself only touches O(log N) nodes** during greedy descent from the entry point,
regardless of total graph size. IVF partitioning contributes a smaller, secondary
speedup (~50ms here) from a smaller partition's coarse-quantizer/metadata footprint.

## 4. Warm steady-state throughput — mem_weaver wins

| | mem_weaver | LanceDB |
|---|---|---|
| p50 latency | 0.25–0.34 ms | 1.4–5.2 ms (rises with concurrency) |
| Peak QPS (single run) | ~14,000–15,400 (4–6 threads) | ~1,220–1,320 (8 concurrent tasks) |

Once mem_weaver's index is resident, every query is a pure in-RAM HNSW walk — no
per-query disk/IPC overhead. LanceDB's per-query cost stays in the low single-digit ms
range even warm, likely reflecting per-query connection/IPC overhead in its async layer
rather than disk I/O.

**Bottom line:** one-shot/bursty workloads favor LanceDB's cheap lazy attach; long-lived,
high-QPS workloads favor mem_weaver's amortized eager load by ~10–50x on latency/throughput.

## 5. Breaking down mem_weaver's ~389–427ms cold reconstruction

Directly measured (not extrapolated) components, isolated via standalone probes reading
the same persisted 308 blocks / 616MB:

| Component | Cost | Method |
|---|---|---|
| Raw disk I/O, 308 separate files (`std::fs::read` each) | ~201 ms (3.07 GB/s) | `sift1m_hnsw_cold_raw_io` |
| Raw disk I/O, 1 consolidated file (single open) | **~126 ms (4.88 GB/s)** | `sift1m_hnsw_cold_raw_io_consolidated` |
| CRC32 compute (isolated, CPU-only) | ~79–128 ms (noisy across runs) | inline probe in `sift1m_hnsw_cold_query` |
| Total observed reconstruction | ~389–427 ms | `sift1m_hnsw_cold_query` |

### 5a. CRC32 verification is a real, removable cost on the local-restore path

`swap_in_with()` (the warm, same-process restore path) already skips CRC32 verification
— a prior optimization in this same investigation took it from 321ms → 238ms (600MB) by
dropping the read+verify pass, on the reasoning that a full pass over the block's bytes
just to verify a checksum dominates wall-clock time, and corruption is assumed handled at
a layer below.

`swap_in_from()` / `load_blocks_from_dir()` (the fresh-process / blob-restore path used
by this benchmark and by `TimeBucketIndex::add_restored_bucket`) still verifies CRC32,
deliberately — it's also the path used for restoring from blob/S3 storage, where
corruption risk from network/storage transfer is real. Measured cost: **~79–128ms out of
the ~389–427ms total** (roughly a fifth to a third of the total).

### 5b. Batching the per-block mmap allocation did NOT help

Hypothesis: 308 separate anonymous `mmap()` calls (one per `NodeBlock`) cost more than
one batched `mmap()` sized for all blocks, mirroring the optimization already used by
`ArenaNodeStore::swap_in()` on the warm path (`Arena::try_with_capacity_batch`).

**Implemented and tested; no measurable improvement** (427ms after vs. 389–420ms before —
within run-to-run noise). Root cause: anonymous `mmap()` is lazy/zero-fill-on-demand, so
the actual cost is *page-fault-in* time paid when `read_exact` first writes into those
pages — that total page-fault work is identical whether it comes from one shared mapping
or 308 separate ones. Batching only removes syscall entry/exit overhead, which turns out
to be small next to the page-faulting cost. (This matches an earlier, independent finding
in this project that batching `Arena::try_with_capacity_batch` in `crates/mem/src/arena.rs`
showed no measurable benefit over one-mmap-per-block in a different context.)

The code change was kept (`NodeBlock::swap_in_from_with`, taking a caller-supplied arena,
mirroring the existing `swap_in_with`) since it's correct and all 66 `index` crate unit
tests still pass — but it provides no performance benefit on its own.

### 5c. Per-file syscall overhead (308 files) IS a real, removable cost

Hypothesis: 308 separate `open()`/`stat()`/`read()`/`close()` round-trips (one block =
one file) cost meaningfully more than the same total bytes read from one file.

**Confirmed — this is the biggest lever found so far.** Reading the identical 616MB as
one consolidated file instead of 308 separate files cut raw I/O time by **37%** (200.6ms
→ 126.2ms; 3.07 GB/s → 4.88 GB/s), all cold, all measured directly.

## 6. File consolidation: implemented and measured

File consolidation (§7 item 1 below) has been implemented: blocks are now packed
`BLOCKS_PER_FILE` (32) at a time into `chunk_<n>.arena` files, addressed by a fixed byte
stride (`arena.mapped_bytes() + 4` for the trailing CRC32), with no manifest needed —
block count per chunk is derived from `file_len / stride`. `NodeBlockStorage::OnDisk` grew
a byte `offset` field so `vector_at`/`neighbors_at`/`swap_in_with`/`swap_out` all read/write
through the same offset-aware code path regardless of whether a block owns its whole file
(offset 0, the historical single-block layout) or shares a chunk file (nonzero offset).

Real-world SIFT1M cold measurement (1M vectors / 308 blocks / 616MB, 10 chunk files,
fresh process, CRC32 verification still on, page cache dropped via `sudo purge`):

| Step | Cost |
|---|---|
| Reconstruction (was 389–427 ms) | **247.7 ms** |
| First query (cold) | 0.611 ms |
| **Total cold time-to-first-answer** | **~248 ms** |

This lands right in the ~220–270ms range projected from the isolated raw-I/O measurement,
confirming the projection. mem_weaver's cold time-to-first-answer (~248ms) is now roughly
on par with LanceDB's (~212–264ms), while keeping its ~10–50x warm-throughput advantage
(unaffected: p50 ~0.34–0.35ms, up to ~14.8k QPS at 6 threads, matching pre-change numbers).

Note this result still includes CRC32 verification (~104ms of the 247.7ms) — §7 item 2
below (skip CRC32 on local-restore) remains unimplemented and would cut the total further.

### 6a. Trade-off: consolidation costs the direct on-disk query path

File consolidation only helps the *bulk* restore path (many blocks read together, one
`open()` per chunk instead of per block). It does not help — and measurably hurts — the
*other* on-disk use case: `sift1m_hnsw_qps_disk` (`crates/index/tests/sift1m_hnsw_recall.rs`)
leaves blocks on disk permanently and serves queries via random-access `pread`s straight
into `NodeBlockStorage::OnDisk`, one node at a time, per graph-traversal step. That path
does no bulk restore at all, so it gets none of the "fewer `open()` calls" benefit and
instead pays a cost: 308 blocks now share only 10 file descriptors instead of 308
independent ones, and concurrent threads doing `pread` against the same few files see more
contention (full SIFT1M, 1M vectors / 308 blocks / 10 chunk files, `sudo purge`'d):

| Threads | Before (1 file/block) | After (chunked) | Δ QPS |
|---|---|---|---|
| 1 | 508 QPS / p50 1.851ms | 469.9 QPS / p50 2.018ms | -7% |
| 2 | 748 QPS / p50 2.421ms | 675.7 QPS / p50 2.787ms | -10% |
| 4 (prior optimum) | 1,000 QPS / p50 3.631ms | 807.8 QPS / p50 4.404ms | -19% |
| 6 | 588 QPS / p50 10.689ms | 620.7 QPS / p50 9.342ms | +6% |

Recall was unchanged (0.979) — this is purely a throughput/latency effect, not a
correctness issue. Net picture: consolidation is a clear win for eager cold-load workloads
(mem_weaver's primary design point, §2), but a real regression — up to ~19% at the
previous 4-thread optimum — for workloads that query directly against on-disk blocks
without ever loading them into RAM. `BLOCKS_PER_FILE` (currently 32) is the knob that
trades one cost against the other: lower it toward 1 to recover the old per-query
performance at the expense of bulk-restore `open()` savings, or leave it high for
cold-load-dominated workloads. Not yet re-tuned or made configurable.

## 7. Open items / not yet implemented

1. ~~**File consolidation**~~ — Implemented and measured; see §6.
2. **Skip CRC32 on `swap_in_from`/`load_blocks_from_dir`**: would need to be scoped to
   trusted local-disk restores only (e.g., a `verify_crc: bool` parameter), since
   `swap_in_from` is shared with the blob/S3-restore path where corruption risk is real
   and the check should stay.
3. Whether to keep the `swap_in_from_with` batched-arena code path added in §5b, given it
   has no measured performance benefit — it's correct and tested, but adds a caller-supplied-
   arena code path that doesn't currently pay for itself.
