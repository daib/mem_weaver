# Crash Recovery Test Scenarios

Integration tests live in `crates/grpc/tests/crash_recovery.rs`. Each test constructs a specific blob storage state, drops the service to simulate a crash, creates a fresh service, calls `recover_from_snapshots`, and asserts on search results.

All tests use an in-memory `InMemory` object store so nothing is written to real S3.

---

## Basic recovery

### `recover_from_snapshots_restores_collections_and_search`

Full end-to-end crash recovery with two time buckets and nine distinct queries.

- Inserts 10 vectors across two time buckets (`bucket_duration=100`).
- Captures pre-crash search results for axis-direction queries, diagonal queries, time-range-restricted queries, and top-1 queries.
- Snapshots both buckets, then drops the service.
- Recovers and re-runs every query.
- Asserts vector IDs and distances match exactly (within 1e-5).

### `recover_does_not_overwrite_existing_collection`

A collection already in memory is not replaced during recovery even if a snapshot exists in blob storage for it.

### `recover_with_no_s3_configured_is_noop`

`recover_from_snapshots` returns `Ok(())` immediately when no blob store is configured. No collections are created.

### `recover_with_empty_bucket_is_noop`

A collection whose `collection.json` exists in blob storage but has no `seq_*/` bucket prefixes is restored with zero buckets.

---

## Catalog

### `create_collection_writes_catalog_and_recovery_uses_it`

`CreateCollection` writes `catalog.json`. A stale `collection.json` exists in blob storage for a deleted collection ("ghost"). Recovery reads the catalog and only restores the catalogued collection — the ghost is ignored.

---

## Stale snapshot cleanup

### `stale_bucket_prefixes_are_deleted_after_eviction`

After `evict_oldest` permanently removes a bucket from the index, the next successful snapshot cycle deletes the corresponding `seq_*/` prefix from blob storage. Recovery on a fresh service sees only the live bucket.

### `eviction_during_snapshot_cycle_does_not_delete_snapshot_until_next_cycle`

`evict_oldest` is called while the snapshot task is running. The task reads `bucket_metas` at the start of each cycle. If eviction happens mid-cycle, the evicted bucket is still in that cycle's `live_seqs` and its prefix is not deleted until the following cycle completes cleanly.

---

## Crash safety during snapshot upload

### `mid_upload_crash_leaves_previous_snapshot_intact`

The snapshot task stages new files in `snap_T2/` before committing the pointer. If the process crashes during the upload, `bucket_meta.json` still points to the previous complete `snap_T1/`. The orphaned `snap_T2/` is cleaned up on the next successful cycle.

- Constructs the orphan by copying `snap_T1/` files into `snap_99999/` without updating `bucket_meta.json`.
- Verifies recovery uses `snap_T1/`.
- Verifies `snap_99999/` is removed on the next cycle.

---

## WAL pruning

### `wal_entries_pruned_after_successful_snapshot_cycle`

After a snapshot cycle that commits all live buckets:

- WAL entries with `seq ≤ wal_high_seq` are deleted from blob storage.
- `collection.json` records the correct `wal_high_seq`.
- WAL entries created after the snapshot are preserved for the next cycle.

---

## WAL replay across buckets

### `wal_replay_restores_vectors_to_all_buckets_after_partial_snapshot_crash`

A batch insert spans a bucket boundary: `vid=11 (t=50)` routes to `seq_0` (front at insert time) and `vid=21 (t=100)` creates `seq_1`. Only `seq_0` is snapshotted before the crash; `seq_1` has no snapshot.

Recovery replays the WAL and routes each item to the correct bucket:
- `vid=11 (t=50)`: `seq_0` is front; `bucket_start=0 ≤ seq_0.created_at=0` → stays in `seq_0`.
- `vid=21 (t=100)`: `bucket_start=100 > seq_0.created_at=0` → creates `seq_1` during replay.

Re-recovering a second time returns exactly 3 distinct vectors — no duplicates.

---

## Cross-bucket deduplication

### `partial_snapshot_cycle_crash_deduplicates_already_snapshotted_bucket`

Constructs the partial-cycle crash state deterministically without background tasks.

Setup (no snapshot task involved):

1. Insert `vid=10` → `seq_0`. WAL seq=1 uploaded.
2. `SwapBucketOutToBlob` seq=0 → baseline snapshot `snap_T1(vid=10)`.
3. `SwapBucketIn` seq=0 → `load_levels` repopulates `node_ids`.
4. Write `collection.json` + `catalog.json` with `wal_high_seq=1`.
5. Insert `vid=11 (t=50→seq_0)` and `vid=21 (t=100→seq_1)`. WAL seq=2 uploaded.
6. `SwapBucketOutToBlob` seq=0 → partial-cycle snapshot `snap_T2(vid=10, vid=11)`. `seq_1` and `collection.json` untouched.

Recovery (`wal_high_seq=1`, so WAL seq=2 is replayed):
- seq=0 restored from `snap_T2` → `known_vector_ids = {10, 11}`
- seq=1 not in S3
- WAL seq=2: `vid=11` → **SKIP** (dedup, already in snapshot); `vid=21` → **INSERT** (creates seq=1)

Result: `seq_0 = {10, 11}`, `seq_1 = {21}`.

### `wal_spanning_both_buckets_after_partial_snapshot`

The WAL genuinely contains vectors for two different buckets because the second batch is inserted while `seq_0` is still the only bucket.

Setup:

1. Insert `vid=10 (t=0)` → `seq_0`. WAL seq=1.
2. Baseline snapshot of seq=0 (`snap_T1`: vid=10). `wal_high_seq=1`.
3. Insert `[vid=11 (t=50), vid=21 (t=150)]` while seq=0 is still the only bucket:
   - `vid=11 (t=50)`: `bucket_start=0 ≤ seq_0.created_at=0` → **seq_0**
   - `vid=21 (t=150)`: `bucket_start=100 > seq_0.created_at=0` → **creates seq_1**
   WAL seq=2 contains vectors for **both** seq_0 and seq_1.
4. Partial snapshot: `SwapBucketOutToBlob` seq=0 → `snap_T2(vid=10, vid=11)`. seq=1 not snapshotted. `collection.json` stays at `wal_high_seq=1`.

Recovery (WAL seq=2 replayed):
- seq=0 from `snap_T2` → `known = {10, 11}`
- seq=1 not in S3
- WAL seq=2: `vid=11` → **SKIP** (dedup); `vid=21` → **INSERT** (creates seq_1)

Result: `seq_0 = {10, 11}`, `seq_1 = {21}`.

---

## Dirty-block tracking

### `dirty_bucket_uploaded_clean_bucket_skipped`

Integration test for dirty-block tracking against a real (in-memory) blob store.

1. Insert two vectors → bucket dirty → snapshot cycle uploads → bucket clean.
2. Multiple further snapshot cycles run with no new inserts → `snap_dir` in `bucket_meta.json` is **unchanged** (clean bucket skipped, no redundant upload).
3. Insert a third vector → bucket dirty again → next snapshot cycle uploads a new `snap_T` → `snap_dir` advances.

### `dirty_flag_race_condition_stale_version_leaves_bucket_dirty_for_reupload`

Integration test for the write-count race condition fix, with full crash recovery.

The race: a new vector arrives between the snapshot being taken and `mark_clean_if_version` being called. If the version is stale the dirty flag must not be cleared.

1. Insert vid=1 → bucket dirty.
2. Capture `write_count_before` (simulates snapshot task reading the version under read lock).
3. Insert vid=2 (simulates a concurrent write during the upload — no lock held).
4. Call `mark_bucket_clean_if_version(version=before)` with the stale version → asserts bucket **stays dirty**.
5. Snapshot cycle runs → sees dirty bucket → re-uploads with both vid=1 and vid=2 → bucket marked clean (version now matches).
6. Further cycles with no inserts → `snap_dir` unchanged (clean).
7. Crash + recovery → both vid=1 and vid=2 are found.

Without the fix (`mark_all_clean` instead of version-aware), step 4 would clear dirty, step 5 would skip the bucket, and step 7 would fail to find vid=2.

---

## Snapshot threshold (`min_dirty_vectors`)

### `snapshot_skipped_when_dirty_vectors_below_threshold`

2 vectors inserted, `min_dirty_vectors=100`. The snapshot task runs many cycles but the total dirty vector count (2) is below the threshold, so:

- No `bucket_meta.json` is written to blob storage.
- WAL entries are preserved (wal_high_seq not advanced).
- The bucket remains dirty.

### `snapshot_fires_when_dirty_vectors_reach_threshold`

3 vectors inserted, `min_dirty_vectors=3`. Total dirty vectors equals the threshold exactly, so the snapshot fires:

- `bucket_meta.json` is written and points to a versioned `snap_T/`.
- WAL entries are pruned.
- Bucket is marked clean.

### `threshold_suppresses_snapshot_until_enough_vectors_accumulate`

Inserts arrive in two batches with `min_dirty_vectors=5`:

- After batch 1 (2 vectors): threshold not reached → no snapshot, WAL preserved.
- After batch 2 (+3 vectors, total=5): threshold reached → snapshot fires.
- Crash + recovery: all 5 vectors found, whether they came from the snapshot or from WAL replay.

---

## Dirty state after recovery

### `recovered_bucket_is_clean_without_wal_replay_dirty_with_wal_replay`

Directly tests the `mark_clean_after_snapshot()` call in `add_restored_bucket`.

Two collections — `stable` (snapshot current, no new inserts before crash) and `active` (snapshot taken, then one more insert before crash):

1. Both collections snapshotted → WAL pruned → both buckets clean.
2. New vector inserted into `active` only → WAL entry uploaded, `active` dirty.
3. Crash.
4. Recovery replays the WAL entry into `active` but has nothing to replay for `stable`.
5. Assert `stable` bucket is **not dirty** — snapshot was current, no WAL replay ran, no re-upload needed.
6. Assert `active` bucket **is dirty** — WAL replay added vid=99 which is not yet in the S3 snapshot.
7. Search confirms both collections return the correct vectors.

Without the fix (`mark_clean_after_snapshot` missing from `add_restored_bucket`), both buckets would start dirty after recovery and the first snapshot cycle would re-upload `stable` unnecessarily.

---

## Key invariants tested

| Property | Test(s) |
|---|---|
| Recovery restores search results exactly | `recover_from_snapshots_restores_collections_and_search` |
| Catalog prevents ghost collection restoration | `create_collection_writes_catalog_and_recovery_uses_it` |
| Stale bucket prefixes cleaned up after eviction | `stale_bucket_prefixes_are_deleted_after_eviction` |
| Eviction mid-cycle deferred to next cycle | `eviction_during_snapshot_cycle_does_not_delete_snapshot_until_next_cycle` |
| Mid-upload crash leaves previous snapshot intact | `mid_upload_crash_leaves_previous_snapshot_intact` |
| WAL entries pruned after successful full cycle | `wal_entries_pruned_after_successful_snapshot_cycle` |
| WAL replays correctly across multiple buckets | `wal_replay_restores_vectors_to_all_buckets_after_partial_snapshot_crash` |
| Partial cycle: already-snapshotted vectors deduped | `partial_snapshot_cycle_crash_deduplicates_already_snapshotted_bucket` |
| WAL spanning two buckets replays each to correct bucket | `wal_spanning_both_buckets_after_partial_snapshot` |
| Dirty bucket uploaded, clean bucket skipped | `dirty_bucket_uploaded_clean_bucket_skipped` |
| Stale write-count prevents premature mark-clean | `dirty_flag_race_condition_stale_version_leaves_bucket_dirty_for_reupload` |
| Threshold suppresses snapshot below min vectors | `snapshot_skipped_when_dirty_vectors_below_threshold` |
| Threshold fires snapshot at min vectors | `snapshot_fires_when_dirty_vectors_reach_threshold` |
| Snapshot accumulates vectors until threshold met | `threshold_suppresses_snapshot_until_enough_vectors_accumulate` |
| Bucket clean after recovery when snapshot was current | `recovered_bucket_is_clean_without_wal_replay_dirty_with_wal_replay` |
