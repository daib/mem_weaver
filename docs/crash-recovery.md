# Crash Recovery

mem-weaver survives a process crash with two complementary mechanisms: periodic **arena snapshots** for bulk state and a **Write-Ahead Log (WAL)** for individual inserts. Together they bound data loss to at most one WAL upload interval.

## Overview

```
Insert flow (per BatchInsert call)
───────────────────────────────────────────────────────────────────────
  1. Apply vectors to in-memory HNSW index (immediately searchable)
  2. Write WAL entry to local disk  <data_dir>/<col>/wal/<seq>.wal
  3. Wait for WAL uploader to confirm upload to blob storage
  4. Return success to client  ←  only after blob storage confirms

Background tasks
───────────────────────────────────────────────────────────────────────
  WAL uploader   (WAL_UPLOAD_INTERVAL_MS, default 200 ms)
    Scans local WAL dir, uploads pending entries to blob storage,
    signals waiting inserts, removes local files.

  Snapshot task  (SNAPSHOT_INTERVAL_SECS, SNAPSHOT_MIN_DIRTY_VECTORS)
    Each arena store tracks a monotonic write_count and clean_version.
    A bucket is dirty when write_count > clean_version (new vectors since
    last upload). Clean buckets are skipped entirely each cycle.

    If the total dirty vectors across all buckets is below
    SNAPSHOT_MIN_DIRTY_VECTORS, the entire collection is skipped that cycle.
    WAL replay is cheap (milliseconds for thousands of vectors) while S3
    uploads are expensive — a threshold around 10 000 avoids frequent
    small uploads without meaningfully increasing recovery time.

    Only after ALL buckets either succeed or are confirmed clean:
    writes wal_high_seq to collection.json and prunes WAL entries.

Recovery (on startup, before gRPC server opens)
───────────────────────────────────────────────────────────────────────
  1. Read catalog.json  →  list of live collections + configs
  2. Per collection: download and restore each bucket snapshot
  3. Download WAL entries with seq > wal_high_seq, replay in order
     (vectors already in snapshot are deduplicated and skipped)
  4. Start WAL uploader and snapshot tasks
  5. Open gRPC server
```

## Durability guarantee

An insert is durable in blob storage before the client receives a success response. If the process crashes after the client receives its response, the WAL entry is in blob storage and will be replayed on recovery. The only window of potential data loss is inserts that have been written to local disk but not yet uploaded — at most one `WAL_UPLOAD_INTERVAL_MS` worth of inserts.

## Duplicate detection

`TimeBucketIndex` maintains a `HashSet<vector_id>` across all buckets. `insert` is idempotent: if a `vector_id` is already present the insert is silently skipped and `None` is returned instead of a `BucketedNodeId`.

`BatchInsertResponse.results` contains only the vectors that were **accepted** (newly inserted). A `vector_id` absent from the response was a duplicate and was skipped. Clients can use this to detect retries: send the same batch again after a crash and compare the returned list against the original request.

During WAL replay on recovery, the same deduplication applies automatically — vectors already present in the restored snapshot are skipped without any special-casing in the recovery code.

## Blob storage layout

```
<BLOB_PREFIX>/
  catalog.json                     ← live collection registry (written on CreateCollection)
  <collection>/
    collection.json                ← index config + wal_high_seq (written each snapshot)
    wal/
      00000000000000000001.wal     ← WAL entry seq=1
      00000000000000000002.wal
      …
    seq_<N>/                       ← one per time bucket
      bucket_meta.json             ← commit pointer → snap_<T>/
      snap_<T>/
        block_0.arena
        levels.bin
        manifest.json
```

WAL entry filenames are zero-padded to 20 digits so lexicographic listing equals numeric order.

## WAL entry format

Binary, little-endian. One file per `BatchInsert` call.

```
seq        : u64          8 bytes
item_count : u32          4 bytes
per item:
  vector_len : u32        4 bytes
  vector     : f32 × N    4 × N bytes
  timestamp  : u64        8 bytes
  vector_id  : u64        8 bytes
crc32      : u32          4 bytes   ← CRC32 of all preceding bytes
```

Example: a single insert with a 4-dimensional vector is `8 + 4 + (4 + 16 + 8 + 8) + 4 = 52 bytes`, compared to roughly 120 bytes for the equivalent JSON.

## Collection metadata (`collection.json`)

```json
{
  "version": 1,
  "dim": 128,
  "m": 16,
  "m_max0": 32,
  "ef_construction": 200,
  "bucket_duration_secs": 3600,
  "snapshot_at_secs": 1718236800,
  "wal_high_seq": 999
}
```

`wal_high_seq` is the highest WAL sequence number whose insert is captured in **every** bucket's snapshot for this collection. It is only written after all buckets are successfully snapshotted in the same cycle. This guarantees cross-bucket consistency: a `wal_high_seq` of N means every time bucket has data through seq N, so recovery can safely skip entries up to N without losing vectors from any bucket. If any bucket fails to snapshot, `wal_high_seq` is not advanced and WAL entries are kept for the next cycle.

## Dirty tracking and snapshot threshold

Each arena store maintains two counters:

- **`write_count`** — incremented on every `push_node`. Starts at 1 so new buckets are immediately dirty.
- **`clean_version`** — set to `write_count` after a confirmed S3 upload.

A bucket is dirty when `write_count > clean_version`. The snapshot task skips clean buckets entirely, so cold buckets (no new inserts) generate zero S3 uploads.

**Race safety:** the snapshot task captures `write_count` under the same read lock used to take the snapshot. After a successful upload it calls `mark_clean_if_version(captured_count)`. If new vectors arrived during the upload (advancing `write_count`), the version check fails and the bucket remains dirty for re-upload on the next cycle.

**Clean after recovery:** `add_restored_bucket` calls `mark_clean_after_snapshot()` immediately after loading blocks from local files. This sets `clean_version = write_count` so the bucket starts clean at the point of restoration. WAL replay that follows will call `push_node`, advancing `write_count` past `clean_version` and re-dirtying only buckets that actually receive new vectors. Buckets whose snapshot was already current (no WAL entries to replay) remain clean and are skipped by the first post-startup snapshot cycle — avoiding a redundant re-upload of data already in S3.

**Threshold (`SNAPSHOT_MIN_DIRTY_VECTORS`):** if the total dirty vectors across all buckets in a collection is below this value, the entire snapshot cycle is skipped for that collection — no uploads, no `wal_high_seq` advancement, no WAL pruning.

The tradeoff: WAL replay is cheap — replaying 10 000 vectors takes milliseconds. S3 uploads are expensive — each 2 MB arena block is a separate PUT. A threshold around **10 000** is a sensible default for most workloads:

- Below the threshold: skip snapshot, let WAL accumulate, rely on fast replay on restart.
- Above the threshold: snapshot fires, WAL is pruned, recovery is bounded by the snapshot age rather than WAL length.

Set `SNAPSHOT_MIN_DIRTY_VECTORS=0` to disable the threshold and snapshot on every cycle whenever any bucket is dirty.

## Recovery procedure

1. **Read catalog** — `catalog.json` lists every live collection with its configuration. If the catalog is absent (pre-catalog deployment) the server falls back to listing all prefixes under `BLOB_PREFIX`.

2. **Restore buckets** — for each collection, list all `seq_<N>/` prefixes. For each, read `bucket_meta.json` to find the committed `snap_<T>/` directory, download its arena files, `levels.bin`, and `manifest.json`, then call `add_restored_bucket`.

3. **Replay WAL** — list all WAL entries at `<collection>/wal/`. Filter to `seq > wal_high_seq` (from step 2). Sort ascending. For each entry, re-insert all vectors into the restored index. Vectors already present in the restored snapshot are silently skipped by the duplicate-detection set (`known_vector_ids`).

4. **Initialize WAL state** — set the per-collection sequence counter to `max_replayed_seq + 1` so new inserts continue without reusing old sequence numbers.

5. **Start background tasks** — WAL uploader and snapshot task begin running. The gRPC server opens for traffic only after recovery completes.

## Crash scenarios

| When crash happens | Effect |
|---|---|
| Before WAL local write | Insert was never applied; client never received a response. No loss. |
| After local write, before blob upload | Local file survives. Uploaded on next startup before gRPC opens. |
| After blob upload, before client receives response | Entry is in blob storage. Replayed on recovery. If the client retries, the duplicate insert is silently skipped — `vector_id` will be absent from the retry's response, signalling it was already present. |
| Below snapshot threshold | No snapshot taken. WAL accumulates. Recovery replays all pending WAL entries — fast for small counts. |
| During snapshot, before all buckets committed | `wal_high_seq` is not advanced (requires all buckets to succeed). WAL entries are kept. Recovery restores each bucket from its last complete snapshot and replays the full WAL tail. |
| During snapshot, after all buckets committed | `collection.json` is updated with the new `wal_high_seq`. WAL entries up to that seq are pruned on the same cycle. |

## WAL pruning

After a snapshot cycle in which every bucket succeeds (or is confirmed clean), WAL entries with `seq <= wal_high_seq` are deleted from blob storage. If any bucket fails, or the collection is below `SNAPSHOT_MIN_DIRTY_VECTORS`, `wal_high_seq` is not advanced and no WAL entries are pruned that cycle. Local WAL files are removed immediately after each successful upload to blob storage.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BLOB_BUCKET` | — | Blob storage bucket name (required to enable WAL and snapshots) |
| `BLOB_PREFIX` | `mem-weaver` | Key prefix inside the bucket |
| `WAL_UPLOAD_INTERVAL_MS` | `200` | How often the WAL uploader scans for pending entries |
| `SNAPSHOT_INTERVAL_SECS` | — | How often to check for dirty buckets; unset disables snapshots |
| `SNAPSHOT_MIN_DIRTY_VECTORS` | `0` | Minimum new vectors per collection before a snapshot fires. WAL replay is fast for small counts; `10000` is a good starting point to avoid frequent small uploads. `0` = always snapshot dirty buckets |
