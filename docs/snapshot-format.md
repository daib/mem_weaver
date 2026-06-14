# Snapshot Format

Snapshots are written periodically by the snapshot task and read during crash recovery. All files live under a configurable blob prefix (set via `BLOB_PREFIX`).

## Layout

```
<BLOB_PREFIX>/
  catalog.json                          ← live collection registry
  <collection>/
    collection.json                     ← index config (written each snapshot cycle)
    seq_<N>/
      bucket_meta.json                  ← commit pointer: seq, created_at, snap_dir
      snap_<T>/                         ← versioned snapshot (T = Unix seconds)
        block_0.arena                   ← raw arena bytes + CRC32
        block_1.arena
        …
        levels.bin                      ← HNSW node IDs, vector IDs, and level assignments
        manifest.json                   ← HNSW entry point and max layer
```

## Example: two collections, multiple time buckets

**There is at most one committed snapshot per bucket.** Each snapshot is staged in a versioned subdirectory `snap_<T>/` (where T is the Unix timestamp when the upload started). Only after all files in `snap_<T>/` are uploaded does `bucket_meta.json` at the `seq_<N>/` level get updated to point to the new `snap_<T>/`. The previous `snap_<T-1>/` is not touched until the new commit is confirmed, then it is deleted.

```
mem-weaver/
  catalog.json                ← one file, overwritten on every CreateCollection

  embeddings/
    collection.json           ← one file, overwritten every snapshot cycle
    seq_0/                    ← oldest bucket (08:00–09:00)
      bucket_meta.json        ← points to snap_1718229600/
      snap_1718229600/        ← the one complete snapshot for this bucket
        block_0.arena
        block_1.arena
        levels.bin
        manifest.json
    seq_1/                    ← middle bucket (09:00–10:00)
      bucket_meta.json        ← points to snap_1718233200/
      snap_1718233200/
        block_0.arena
        levels.bin
        manifest.json
    seq_2/                    ← newest bucket (10:00–), still receiving inserts
      bucket_meta.json        ← points to snap_1718236800/
      snap_1718236800/
        block_0.arena
        levels.bin
        manifest.json

  events/
    collection.json
    seq_0/
      bucket_meta.json
      snap_1718236800/
        block_0.arena
        levels.bin
        manifest.json
    seq_1/
      bucket_meta.json
      snap_1718236800/
        block_0.arena
        levels.bin
        manifest.json
```

During a snapshot cycle a second `snap_<T+1>/` directory exists briefly alongside the current `snap_<T>/` while the new files are uploading. Once all uploads succeed and `bucket_meta.json` is updated, `snap_<T>/` is deleted. If the process crashes mid-upload, `snap_<T+1>/` is an incomplete orphan — `bucket_meta.json` still points to the old `snap_<T>/`, which is untouched and fully consistent.

When `evict_oldest` is called on `embeddings`, `seq_0/` (including its `snap_*/` subdirectory) is removed from blob storage on the next successful snapshot cycle.

## Files

### `catalog.json`

Written by `CreateCollection` (and updated on `DeleteCollection`). Read first during crash recovery to determine which collections were active at crash time and what configuration to use. Recovery skips any `seq_*/` prefix not associated with a catalogued collection, preventing stale snapshots from deleted collections from being restored.

```json
{
  "version": 1,
  "collections": [
    {
      "name": "embeddings",
      "dim": 128,
      "m": 16,
      "m_max0": 32,
      "ef_construction": 200,
      "bucket_duration_secs": 3600,
      "snapshot_at_secs": 1718236800
    }
  ]
}
```

| Field | Description |
|---|---|
| `version` | Format version, currently `1` |
| `collections` | Sorted list of live collections with their full index configuration |
| `snapshot_at_secs` | Unix timestamp (seconds) when this entry was last written |

### `<collection>/collection.json`

Written once per snapshot cycle. Carries the same index configuration as the catalog entry. Used only as a fallback during recovery when no `catalog.json` exists (nodes that predate the catalog).

```json
{
  "version": 1,
  "dim": 128,
  "m": 16,
  "m_max0": 32,
  "ef_construction": 200,
  "bucket_duration_secs": 3600,
  "snapshot_at_secs": 1718236800
}
```

| Field | Description |
|---|---|
| `version` | Format version, currently `1` |
| `dim` | Vector dimension |
| `m` | HNSW max connections per node on levels > 0 |
| `m_max0` | HNSW max connections per node on level 0 |
| `ef_construction` | HNSW construction beam width |
| `bucket_duration_secs` | Length of each time bucket in seconds (`0` means a single infinite bucket) |
| `snapshot_at_secs` | Unix timestamp (seconds) when this snapshot was written |

### `<collection>/seq_<N>/bucket_meta.json`

The **commit pointer** for a bucket. Written last, after all arena files in `snap_<T>/` have been uploaded. Recovery only uses a bucket if this file is present. Its `snap_dir` field names the versioned subdirectory that holds the complete, consistent snapshot.

```json
{
  "version": 1,
  "seq": 3,
  "created_at_secs": 1718236800,
  "snap_dir": "snap_1718236800"
}
```

| Field | Description |
|---|---|
| `version` | Format version, currently `1` |
| `seq` | Monotonically increasing bucket sequence number |
| `created_at_secs` | Grid-aligned start of the bucket's time window (Unix seconds) |
| `snap_dir` | Name of the versioned subdirectory containing this snapshot's arena files |

### `<collection>/seq_<N>/block_<I>.arena`

Raw arena bytes for one `NodeBlock`, followed by a 4-byte little-endian CRC32 checksum. Each file is approximately 2 MB (one `DEFAULT_ARENA_CAPACITY` arena). Multiple files appear when the HNSW graph has grown beyond a single block.

### `<collection>/seq_<N>/levels.bin`

Serialized HNSW graph metadata. Layout (all little-endian):

```
version    : u32
count      : u64
node_ids   : [u32; count]
vector_ids : [u64; count]
levels     : [u8;  count]
crc32      : u32
```

### `<collection>/seq_<N>/manifest.json`

HNSW index entry point and depth, required to resume search after recovery.

```json
{
  "version": 1,
  "entry_point": 42,
  "max_layer": 3,
  "arena_files": ["block_0.arena", "block_1.arena"]
}
```

| Field | Description |
|---|---|
| `entry_point` | Internal node ID of the HNSW entry point (`null` for an empty index) |
| `max_layer` | Highest allocated HNSW layer |
| `arena_files` | Arena file names in this bucket's prefix |

## Written by

Two paths produce snapshots in this format:

- **Snapshot task** (`SNAPSHOT_INTERVAL_SECS`) — periodic background task. Only buckets with **dirty** arena blocks (written to since the last successful upload) are re-snapshotted. Clean buckets are skipped and their previous snapshot remains valid. After a successful upload, all blocks in that bucket are marked clean.
- **`SwapBucketOutToBlob` RPC** — explicit operator-driven eviction that writes the bucket to local disk first, then uploads using the same versioned `snap_T/` + `bucket_meta.json` layout. Files written by this RPC are interchangeable with snapshot-task output and are visible to crash recovery.

`SwapBucketInFromBlob` is the inverse: it reads `bucket_meta.json` to locate the versioned `snap_T/` directory and downloads from there.

## Recovery procedure

1. Download `catalog.json` to get the list of live collections and their configurations. If the catalog is absent, fall back to listing all `<collection>/` prefixes and reading each `collection.json`.
2. For each collection, create an empty `TimeBucketIndex` using the configuration from the catalog entry.
3. List all `seq_*/` prefixes under the collection prefix and download each `bucket_meta.json`.
4. Sort buckets by `created_at_secs` ascending (oldest first).
5. For each bucket, download `block_*.arena`, `levels.bin`, and `manifest.json` to a local temp directory, then call `add_restored_bucket`.
6. Clean up local temp files.

## Snapshot lifecycle

- **Dirty tracking** — each arena block carries a `dirty` flag and the store maintains a monotonic `write_count`. New blocks start dirty. `push_node` marks the block dirty and increments `write_count`. Loading a block from disk (`swap_in`, `swap_in_from`) marks it clean. A bucket is skipped entirely if none of its blocks are dirty — the previous snapshot is still valid. Cold buckets (no new inserts) generate zero uploads per cycle.
- **Race-safe mark-clean** — the snapshot task captures `write_count` under the same read lock used to take the snapshot. After a successful upload it calls `mark_clean_if_version(captured_count)`: if new vectors were inserted during the upload (advancing `write_count`), the version check fails and the dirty flag is left set, ensuring the next cycle re-uploads the updated arena. This closes the race between snapshot completion and the mark-clean call.
- **Versioned staging** — new snapshot files are uploaded to `seq_<N>/snap_<T>/` before the commit pointer (`bucket_meta.json`) is updated. The previous complete snapshot in `snap_<T-1>/` is untouched during the upload and only deleted after the new commit succeeds.
- **Written**: once per `SNAPSHOT_INTERVAL_SECS`, but **only for dirty buckets**. The hot bucket (receiving new inserts) is re-uploaded every cycle; cold buckets are skipped.
- **Crash safety**: each file PUT is atomic. If the process crashes mid-upload, `bucket_meta.json` still points to the old `snap_<T-1>/`, which is fully consistent. The incomplete `snap_<T>/` is an orphan cleaned up on the next cycle — even if the bucket is clean (no new upload needed), the orphan cleanup still runs when checking the bucket's current pointer.
- **Deleted**: after a successful cycle, the previous `snap_*/` directory for each bucket is removed. Any `seq_<N>/` prefix whose seq is no longer tracked by the index is deleted entirely. Deletion only runs when all live buckets were either successfully re-snapshotted or confirmed clean in the same cycle.
