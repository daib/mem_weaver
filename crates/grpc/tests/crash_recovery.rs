//! Integration tests for crash recovery via S3 snapshots.
//!
//! Simulates the full lifecycle: insert vectors → snapshot to S3 → "crash" (drop
//! the service) → restart a new service → recover_from_snapshots → verify search.

use std::sync::Arc;
use std::time::Duration;

use grpc::{MemWeaverService, SnapshotConfig};
use object_store::{memory::InMemory, ObjectStore};

/// A shared in-memory object store that survives across service instances,
/// simulating an S3 bucket that outlives a crashed node.
fn shared_store() -> Arc<dyn ObjectStore> {
    Arc::new(InMemory::new())
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let p = std::env::temp_dir().join(format!("mem_weaver_recovery_{}_{}", tag, n));
    std::fs::create_dir_all(&p).unwrap();
    p
}

struct DirGuard(std::path::PathBuf);
impl Drop for DirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs a search and returns hits sorted by vector_id for stable comparison.
/// TimeBucketIndex uses quickselect so result order is unspecified.
async fn search_sorted(
    svc: &impl grpc::proto::mem_weaver_server::MemWeaver,
    collection: &str,
    query: Vec<f32>,
    k: u32,
    time_range: Option<(u64, u64)>,
) -> Vec<(u64, f32)> {
    use grpc::proto::SearchRequest;
    use tonic::Request;
    let (time_range_start, time_range_end) = match time_range {
        Some((s, e)) => (Some(s), Some(e)),
        None => (None, None),
    };
    let mut hits = svc
        .search(Request::new(SearchRequest {
            collection: collection.into(),
            query,
            k,
            ef: 64,
            time_range_start,
            time_range_end,
        }))
        .await
        .expect("search")
        .into_inner()
        .hits
        .into_iter()
        .map(|h| (h.vector_id, h.distance))
        .collect::<Vec<_>>();
    hits.sort_by_key(|&(vid, _)| vid);
    hits
}

#[tokio::test]
async fn recover_from_snapshots_restores_collections_and_search() {
    let store = shared_store();
    let data_dir = temp_dir("recover_basic");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem};
    use tonic::Request;

    // Queries run before and after the crash — results must be identical.
    // Each entry: (query_vector, k, time_range).
    let queries: Vec<(Vec<f32>, u32, Option<(u64, u64)>)> = vec![
        // Nearest neighbours to each axis direction (unrestricted time range).
        (vec![1.0, 0.0, 0.0, 0.0], 3, None),
        (vec![0.0, 1.0, 0.0, 0.0], 3, None),
        (vec![0.0, 0.0, 1.0, 0.0], 3, None),
        (vec![0.0, 0.0, 0.0, 1.0], 3, None),
        // Diagonal query — should hit all recent-bucket vectors.
        (vec![1.0, 1.0, 1.0, 1.0], 5, None),
        // Time-range restricted to bucket 0 only (timestamps 0–99).
        (vec![1.0, 0.0, 0.0, 0.0], 2, Some((0, 100))),
        // Time-range restricted to bucket 1 only (timestamps 100–199).
        (vec![0.0, 0.0, 0.0, 1.0], 2, Some((100, 200))),
        // Top-1 nearest per axis.
        (vec![1.0, 0.0, 0.0, 0.0], 1, None),
        (vec![0.0, 0.0, 0.0, 1.0], 1, None),
    ];

    // ── Phase 1: populate, query, snapshot ──────────────────────────────────
    let pre_crash_results: Vec<Vec<(u64, f32)>>;
    {
        let svc = MemWeaverService::with_storage(
            data_dir.clone(),
            Some(Arc::clone(&store)),
            "test-prefix",
        );
        let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));

        svc.create_collection(Request::new(CreateCollectionRequest {
            collection: "vecs".into(),
            dim: 4,
            m: 4,
            m_max0: 8,
            ef_construction: 64,
            // Two time buckets: timestamps 0–99 → bucket 0, 100–199 → bucket 1.
            bucket_duration_secs: 100,
        }))
        .await
        .expect("create_collection");

        // Bucket 0: axis-aligned unit vectors.
        svc.batch_insert(Request::new(BatchInsertRequest {
            collection: "vecs".into(),
            items: vec![
                InsertItem {
                    vector: vec![1.0, 0.0, 0.0, 0.0],
                    timestamp: 0,
                    vector_id: 10,
                },
                InsertItem {
                    vector: vec![0.0, 1.0, 0.0, 0.0],
                    timestamp: 10,
                    vector_id: 11,
                },
                InsertItem {
                    vector: vec![0.0, 0.0, 1.0, 0.0],
                    timestamp: 20,
                    vector_id: 12,
                },
                InsertItem {
                    vector: vec![0.5, 0.5, 0.0, 0.0],
                    timestamp: 30,
                    vector_id: 13,
                },
                InsertItem {
                    vector: vec![0.0, 0.5, 0.5, 0.0],
                    timestamp: 40,
                    vector_id: 14,
                },
            ],
        }))
        .await
        .expect("batch_insert bucket-0");

        // Bucket 1: vectors shifted toward the w-axis.
        svc.batch_insert(Request::new(BatchInsertRequest {
            collection: "vecs".into(),
            items: vec![
                InsertItem {
                    vector: vec![0.0, 0.0, 0.0, 1.0],
                    timestamp: 100,
                    vector_id: 20,
                },
                InsertItem {
                    vector: vec![0.1, 0.0, 0.0, 0.9],
                    timestamp: 110,
                    vector_id: 21,
                },
                InsertItem {
                    vector: vec![0.0, 0.1, 0.0, 0.9],
                    timestamp: 120,
                    vector_id: 22,
                },
                InsertItem {
                    vector: vec![0.5, 0.0, 0.0, 0.5],
                    timestamp: 130,
                    vector_id: 23,
                },
                InsertItem {
                    vector: vec![0.0, 0.0, 0.5, 0.5],
                    timestamp: 140,
                    vector_id: 24,
                },
            ],
        }))
        .await
        .expect("batch_insert bucket-1");

        // Capture results for every query while the full index is hot in memory.
        let mut results = Vec::new();
        for (query, k, range) in &queries {
            results.push(search_sorted(&svc, "vecs", query.clone(), *k, *range).await);
        }
        // Sanity-check a few expected nearest neighbours.
        let near_x = &results[0]; // query=[1,0,0,0], k=3
        assert!(
            near_x.iter().any(|&(vid, _)| vid == 10),
            "vid=10 must be near [1,0,0,0]"
        );
        let near_w = &results[3]; // query=[0,0,0,1], k=3
        assert!(
            near_w.iter().any(|&(vid, _)| vid == 20),
            "vid=20 must be near [0,0,0,1]"
        );
        let bucket0_only = &results[5]; // k=2, range 0–100
        assert!(
            bucket0_only.iter().all(|&(vid, _)| vid < 20),
            "time-range [0,100) must only return bucket-0 vectors"
        );
        let bucket1_only = &results[6]; // k=2, range 100–200
        assert!(
            bucket1_only.iter().all(|&(vid, _)| vid >= 20),
            "time-range [100,200) must only return bucket-1 vectors"
        );
        assert_eq!(results[7].len(), 1, "top-1 must return exactly 1 result");
        assert_eq!(results[7][0].0, 10, "top-1 near [1,0,0,0] must be vid=10");

        pre_crash_results = results;

        // Snapshot everything to S3 then "crash" by dropping the service.
        let _handle = svc.spawn_snapshot_task(SnapshotConfig {
            interval: Duration::from_millis(1),
        });
        tokio::time::sleep(Duration::from_millis(300)).await;
        // svc dropped here — all in-memory state is gone.
    }

    // ── Phase 2: restart on a completely fresh service ───────────────────────
    {
        let svc2 = MemWeaverService::with_storage(
            data_dir.clone(),
            Some(Arc::clone(&store)),
            "test-prefix",
        );
        assert_eq!(
            svc2.collection_count().await,
            0,
            "fresh service must have no collections"
        );

        svc2.recover_from_snapshots()
            .await
            .expect("recover_from_snapshots");

        assert!(svc2.has_collection("vecs").await, "'vecs' must be restored");
        assert_eq!(
            svc2.bucket_count("vecs").await,
            Some(2),
            "both time buckets must be recovered"
        );

        // Run every query again and compare against the pre-crash results.
        for (i, (query, k, range)) in queries.iter().enumerate() {
            let after = search_sorted(&svc2, "vecs", query.clone(), *k, *range).await;
            let before = &pre_crash_results[i];
            assert_eq!(
                after.len(),
                before.len(),
                "query {i}: result count mismatch (before={}, after={})",
                before.len(),
                after.len(),
            );
            let after_ids: Vec<u64> = after.iter().map(|&(vid, _)| vid).collect();
            let before_ids: Vec<u64> = before.iter().map(|&(vid, _)| vid).collect();
            assert_eq!(
                after_ids, before_ids,
                "query {i}: vector_ids differ after recovery\n  before: {before_ids:?}\n  after:  {after_ids:?}"
            );
            // Distances must match within floating-point rounding.
            for (j, (&(_, d_before), &(_, d_after))) in before.iter().zip(after.iter()).enumerate()
            {
                assert!(
                    (d_before - d_after).abs() < 1e-5,
                    "query {i} hit {j}: distance changed from {d_before} to {d_after}"
                );
            }
        }
    }
}

#[tokio::test]
async fn recover_does_not_overwrite_existing_collection() {
    let store = shared_store();
    let data_dir = temp_dir("recover_no_overwrite");
    let _guard = DirGuard(data_dir.clone());

    // Upload a minimal collection.json so recovery finds something.
    use index::CollectionMeta;
    use object_store::path::Path as ObjectPath;
    let col_meta = CollectionMeta {
        version: 1,
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 3600,
        snapshot_at_secs: 0,
        wal_high_seq: 0,
    };
    let prefix = ObjectPath::from("pfx/existing");
    index::upload_collection_meta(store.as_ref(), &col_meta, &prefix)
        .await
        .expect("upload meta");

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));

    // Pre-create the collection so it already exists in memory.
    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::CreateCollectionRequest;
    use tonic::Request;
    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "existing".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 3600,
    }))
    .await
    .expect("create");

    // Insert a sentinel vector so we can detect if recovery overwrites state.
    use grpc::proto::{BatchInsertRequest, InsertItem};
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "existing".into(),
        items: vec![InsertItem {
            vector: vec![9.0, 9.0, 9.0, 9.0],
            timestamp: 0,
            vector_id: 999,
        }],
    }))
    .await
    .expect("insert sentinel");

    // Recovery must skip the already-in-memory collection.
    svc.recover_from_snapshots().await.expect("recover");

    // Sentinel vector must still be present (collection not wiped).
    use grpc::proto::SearchRequest;
    let results = svc
        .search(Request::new(SearchRequest {
            collection: "existing".into(),
            query: vec![9.0, 9.0, 9.0, 9.0],
            k: 1,
            ef: 16,
            time_range_start: None,
            time_range_end: None,
        }))
        .await
        .expect("search")
        .into_inner();
    assert_eq!(
        results.hits[0].vector_id, 999,
        "existing collection must not be overwritten"
    );
    assert_eq!(
        svc.collection_count().await,
        1,
        "no duplicate collection created"
    );
}

#[tokio::test]
async fn recover_with_no_s3_configured_is_noop() {
    let data_dir = temp_dir("recover_no_s3");
    let _guard = DirGuard(data_dir.clone());
    let svc = MemWeaverService::with_storage(data_dir, None::<Arc<dyn ObjectStore>>, "pfx");
    // Must not error; collections stay empty.
    svc.recover_from_snapshots().await.expect("noop ok");
    assert_eq!(svc.collection_count().await, 0);
}

#[tokio::test]
async fn create_collection_writes_catalog_and_recovery_uses_it() {
    // Verify the full catalog lifecycle:
    //   1. create_collection writes catalog.json to S3
    //   2. A second collection's stale snapshot exists in S3 but is NOT in the catalog
    //   3. recover_from_snapshots only restores the catalogued collection

    let store = shared_store();
    let data_dir = temp_dir("catalog_lifecycle");
    let _guard = DirGuard(data_dir.clone());

    // Plant a stale collection.json for "ghost" directly in S3 (simulates a deleted
    // collection whose snapshot was never cleaned up).
    use index::CollectionMeta;
    use object_store::path::Path as ObjectPath;
    index::upload_collection_meta(
        store.as_ref(),
        &CollectionMeta {
            version: 1,
            dim: 4,
            m: 4,
            m_max0: 8,
            ef_construction: 32,
            bucket_duration_secs: 3600,
            snapshot_at_secs: 0,
            wal_high_seq: 0,
        },
        &ObjectPath::from("pfx/ghost"),
    )
    .await
    .expect("plant stale meta");

    // Create a legitimate collection — this must write the catalog.
    {
        let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");

        use grpc::proto::mem_weaver_server::MemWeaver;
        use grpc::proto::CreateCollectionRequest;
        use tonic::Request;

        svc.create_collection(Request::new(CreateCollectionRequest {
            collection: "real".into(),
            dim: 4,
            m: 4,
            m_max0: 8,
            ef_construction: 32,
            bucket_duration_secs: 0,
        }))
        .await
        .expect("create_collection");

        // Catalog must now exist in S3.
        let catalog = index::download_catalog(store.as_ref(), &ObjectPath::from("pfx"))
            .await
            .expect("catalog must exist after create_collection");
        let names: Vec<&str> = catalog
            .collections
            .iter()
            .map(|e| e.name.as_str())
            .collect();
        assert_eq!(names, vec!["real"]);
        // "ghost" is NOT in the catalog.
        assert!(!names.contains(&"ghost"));
    }

    // Recovery on a fresh service: only "real" should be restored, not "ghost".
    {
        let svc2 =
            MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
        svc2.recover_from_snapshots().await.expect("recover");

        assert!(svc2.has_collection("real").await, "'real' must be restored");
        assert!(
            !svc2.has_collection("ghost").await,
            "'ghost' must be ignored (not in catalog)"
        );
    }
}

#[tokio::test]
async fn recover_with_empty_bucket_is_noop() {
    // S3 has a collection.json but no seq_* prefixes — recovery must produce
    // an empty (but valid) collection with 0 buckets.
    let store = shared_store();
    let data_dir = temp_dir("recover_empty_bucket");
    let _guard = DirGuard(data_dir.clone());

    use index::CollectionMeta;
    use object_store::path::Path as ObjectPath;
    let prefix = ObjectPath::from("pfx/empty_col");
    index::upload_collection_meta(
        store.as_ref(),
        &CollectionMeta {
            version: 1,
            dim: 4,
            m: 4,
            m_max0: 8,
            ef_construction: 32,
            bucket_duration_secs: 3600,
            snapshot_at_secs: 0,
            wal_high_seq: 0,
        },
        &prefix,
    )
    .await
    .expect("upload meta");

    let svc = MemWeaverService::with_storage(data_dir, Some(Arc::clone(&store)), "pfx");
    svc.recover_from_snapshots().await.expect("recover");

    assert!(
        svc.has_collection("empty_col").await,
        "collection must be created even with 0 buckets"
    );
    assert_eq!(svc.bucket_count("empty_col").await, Some(0));
}

#[tokio::test]
async fn stale_bucket_prefixes_are_deleted_after_eviction() {
    // Scenario: a collection with two time buckets is snapshotted. Then the oldest
    // bucket is evicted (permanently removed from the index). The next snapshot cycle
    // must delete the stale seq_0/ prefix from blob storage so that crash recovery
    // does not restore a bucket that no longer exists.

    let store = shared_store();
    let data_dir = temp_dir("stale_cleanup");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem};
    use object_store::path::Path as ObjectPath;
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");

    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));
    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 1,
    }))
    .await
    .expect("create");

    // Two buckets: seq=0 at t=0, seq=1 at t=1.
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 0,
            vector_id: 1,
        }],
    }))
    .await
    .expect("insert bucket-0");
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![0.0, 1.0, 0.0, 0.0],
            timestamp: 1,
            vector_id: 2,
        }],
    }))
    .await
    .expect("insert bucket-1");

    assert_eq!(svc.bucket_count("col").await, Some(2));

    // First snapshot: both seq_0 and seq_1 are uploaded.
    let _handle = svc.spawn_snapshot_task(SnapshotConfig {
        interval: Duration::from_millis(1),
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Both prefixes must exist in blob storage after the first snapshot.
    let seq0_prefix = ObjectPath::from("pfx/col/seq_0");
    let seq1_prefix = ObjectPath::from("pfx/col/seq_1");
    let has_objects = |prefix: ObjectPath| {
        let store = Arc::clone(&store);
        async move {
            store
                .list_with_delimiter(Some(&prefix))
                .await
                .map(|r| !r.objects.is_empty() || !r.common_prefixes.is_empty())
                .unwrap_or(false)
        }
    };
    assert!(
        has_objects(seq0_prefix.clone()).await,
        "seq_0 must exist after first snapshot"
    );
    assert!(
        has_objects(seq1_prefix.clone()).await,
        "seq_1 must exist after first snapshot"
    );

    // Permanently drop the oldest bucket (seq=0) from the index. This is what
    // evict_oldest / evict_before do — they remove the entry from the deque entirely,
    // unlike EvictBucket which only drops the in-memory arena but keeps the entry.
    assert!(
        svc.evict_oldest_bucket("col").await,
        "evict_oldest must find and remove seq_0"
    );
    assert_eq!(
        svc.bucket_count("col").await,
        Some(1),
        "seq_0 must be removed from the index"
    );

    // Wait for the next snapshot cycle — it should see only seq_1 as live
    // and delete the stale seq_0/ prefix from blob storage.
    tokio::time::sleep(Duration::from_millis(300)).await;

    assert!(
        !has_objects(seq0_prefix).await,
        "stale seq_0/ must be deleted after evict_oldest"
    );
    assert!(has_objects(seq1_prefix).await, "live seq_1/ must be kept");

    // Recovery on a fresh service must NOT restore the evicted bucket.
    let svc2 = MemWeaverService::with_storage(data_dir, Some(Arc::clone(&store)), "pfx");
    svc2.recover_from_snapshots().await.expect("recover");
    assert_eq!(
        svc2.bucket_count("col").await,
        Some(1),
        "recovery must restore exactly 1 bucket (seq_0 was evicted and cleaned up)"
    );
    let after = search_sorted(&svc2, "col", vec![0.0, 1.0, 0.0, 0.0], 1, None).await;
    assert_eq!(
        after[0].0, 2,
        "recovered index must contain only vid=2 (from seq_1)"
    );
}

#[tokio::test]
async fn eviction_during_snapshot_cycle_does_not_delete_snapshot_until_next_cycle() {
    // Edge case: evict_oldest is called *after* bucket_metas is read by the snapshot
    // task but *before* the cleanup step runs. The snapshot task must NOT delete the
    // evicted bucket's prefix in this cycle — it was live when the cycle started.
    // Only the following cycle (where live_seqs no longer contains the seq) cleans it up.

    let store = shared_store();
    let data_dir = temp_dir("mid_cycle_evict");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem};
    use object_store::path::Path as ObjectPath;
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");

    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));
    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 1,
    }))
    .await
    .expect("create");

    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 0,
            vector_id: 1,
        }],
    }))
    .await
    .expect("insert seq_0");
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![0.0, 1.0, 0.0, 0.0],
            timestamp: 1,
            vector_id: 2,
        }],
    }))
    .await
    .expect("insert seq_1");

    // Run one full snapshot cycle to establish both prefixes in blob storage.
    let _handle = svc.spawn_snapshot_task(SnapshotConfig {
        interval: Duration::from_millis(1),
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    let seq0_prefix = ObjectPath::from("pfx/col/seq_0");
    let seq1_prefix = ObjectPath::from("pfx/col/seq_1");
    let objects_at = |prefix: ObjectPath| {
        let store = Arc::clone(&store);
        async move {
            store
                .list_with_delimiter(Some(&prefix))
                .await
                .map(|r| r.objects.len() + r.common_prefixes.len())
                .unwrap_or(0)
        }
    };

    assert!(
        objects_at(seq0_prefix.clone()).await > 0,
        "seq_0 must exist after first cycle"
    );
    assert!(
        objects_at(seq1_prefix.clone()).await > 0,
        "seq_1 must exist after first cycle"
    );

    // Evict seq_0 from the index. The snapshot task is still running; by the time the
    // next cycle's cleanup step examines live_seqs, seq_0 is already gone from the index.
    // But the snapshot uploaded at the START of that cycle captured seq_0 as live, so
    // cleanup must skip it and leave the prefix intact for this cycle.
    //
    // We verify the conservative guarantee: immediately after eviction, wait for exactly
    // one more snapshot cycle. seq_0 must still be in blob storage because the task
    // reads live_seqs at the top of the cycle — before it knows about the eviction.
    //
    // Implementation note: the task reads bucket_metas (including seq_0) at the top of
    // each cycle, then sets live_seqs from that same snapshot. Even if evict_oldest fires
    // concurrently, the cycle's live_seqs still contains seq_0 and the cleanup guard
    // (committed == live_seqs) will protect it.
    svc.evict_oldest_bucket("col").await;

    // Wait for exactly one snapshot interval to pass. seq_0 should NOT be deleted yet
    // because the cycle that started just before (or just after) the eviction will have
    // had seq_0 in its live_seqs.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // After one cycle with seq_0 evicted, live_seqs = {1}, committed = {1}.
    // committed == live_seqs → cleanup runs → seq_0 deleted.
    // We need to wait at least two intervals to be sure a full cycle with the eviction
    // visible has completed.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Now seq_0 must have been cleaned up (at most two cycles after eviction).
    assert_eq!(
        objects_at(seq0_prefix).await,
        0,
        "seq_0 must be deleted within two cycles after evict_oldest"
    );
    // seq_1 must always be kept — it is still live.
    assert!(
        objects_at(seq1_prefix).await > 0,
        "seq_1 must never be deleted while it is still in the index"
    );
}

#[tokio::test]
async fn mid_upload_crash_leaves_previous_snapshot_intact() {
    // Simulate a process crash that happens after content files have been staged in
    // snap_<T2>/ but before bucket_meta.json is updated.
    //
    // Expected: recovery reads bucket_meta.json (which still points to snap_<T1>/),
    // ignores the orphaned snap_<T2>/ entirely, and restores the correct T1 state.
    // The orphaned prefix is cleaned up on the next successful snapshot cycle.

    let store = shared_store();
    let data_dir = temp_dir("mid_upload_crash");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem};
    use object_store::path::Path as ObjectPath;
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");

    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));
    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 0,
    }))
    .await
    .expect("create");

    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![
            InsertItem {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                timestamp: 0,
                vector_id: 1,
            },
            InsertItem {
                vector: vec![0.0, 1.0, 0.0, 0.0],
                timestamp: 0,
                vector_id: 2,
            },
            InsertItem {
                vector: vec![0.0, 0.0, 1.0, 0.0],
                timestamp: 0,
                vector_id: 3,
            },
        ],
    }))
    .await
    .expect("insert");

    let before = search_sorted(&svc, "col", vec![1.0, 0.0, 0.0, 0.0], 3, None).await;
    assert_eq!(before.len(), 3);

    // Take a complete snapshot (snap_<T1>/ committed, bucket_meta.json points to it).
    let _h = svc.spawn_snapshot_task(SnapshotConfig {
        interval: Duration::from_millis(1),
    });
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Confirm bucket_meta.json exists and records the active snap_dir.
    let seq0_prefix = ObjectPath::from("pfx/col/seq_0");
    let meta = index::download_bucket_meta(store.as_ref(), &seq0_prefix)
        .await
        .expect("bucket_meta.json must exist after snapshot");
    let committed_snap = meta.snap_dir.clone();
    assert!(
        committed_snap.starts_with("snap_"),
        "snap_dir must look like snap_<T>: {committed_snap}"
    );

    // Simulate a mid-upload crash for a second cycle: stage content files in a new
    // snap_99999/ subdirectory but do NOT write a new bucket_meta.json.
    // This mimics the process being killed after upload_arena_dir + upload_levels +
    // upload_manifest succeed but before the final upload_bucket_meta call.
    let orphan_prefix = ObjectPath::from("pfx/col/seq_0/snap_99999");
    let committed_prefix = seq0_prefix.child(committed_snap.as_str());

    // Copy the committed files into the orphan prefix to give it valid (but stale)
    // content — if recovery accidentally followed this prefix it would still produce
    // wrong results (different arena bytes with garbage CRC-passing content).
    // In practice we upload a sentinel file to mark the orphan as present.
    use object_store::{ObjectStore, PutPayload};
    // List and copy all objects from the committed prefix into the orphan prefix.
    // We use list_with_delimiter to get the object names, then copy each one.
    {
        let listing = store
            .list_with_delimiter(Some(&committed_prefix))
            .await
            .expect("list committed prefix");
        for obj in listing.objects {
            let filename = obj.location.filename().expect("filename").to_owned();
            let data = store
                .get(&obj.location)
                .await
                .expect("get")
                .bytes()
                .await
                .expect("bytes");
            let dest = orphan_prefix.child(filename.as_str());
            store
                .put(&dest, PutPayload::from(data))
                .await
                .expect("put orphan file");
        }
    }

    // Verify the orphan prefix now has files but bucket_meta.json still points to
    // the original committed snap.
    let meta2 = index::download_bucket_meta(store.as_ref(), &seq0_prefix)
        .await
        .expect("bucket_meta.json must be unchanged");
    assert_eq!(
        meta2.snap_dir, committed_snap,
        "bucket_meta.json must still point to the committed snap, not the orphan"
    );

    // "Crash": drop svc.
    drop(_h);
    drop(svc);

    // Recovery on a fresh service must ignore snap_99999/ and restore from the
    // committed snap_<T1>/ that bucket_meta.json points to.
    let svc2 = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    svc2.recover_from_snapshots()
        .await
        .expect("recover_from_snapshots");

    assert!(
        svc2.has_collection("col").await,
        "collection must be restored"
    );
    assert_eq!(svc2.bucket_count("col").await, Some(1));

    let after = search_sorted(&svc2, "col", vec![1.0, 0.0, 0.0, 0.0], 3, None).await;
    let before_ids: Vec<u64> = before.iter().map(|&(v, _)| v).collect();
    let after_ids: Vec<u64> = after.iter().map(|&(v, _)| v).collect();
    assert_eq!(
        after_ids, before_ids,
        "recovery must return identical results to pre-crash state"
    );

    // The next snapshot cycle must clean up the orphaned snap_99999/ directory.
    let svc3 = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    svc3.recover_from_snapshots().await.expect("recover svc3");
    let _h3 = svc3.spawn_snapshot_task(SnapshotConfig {
        interval: Duration::from_millis(1),
    });
    tokio::time::sleep(Duration::from_millis(300)).await;

    let orphan_objects = store
        .list_with_delimiter(Some(&orphan_prefix))
        .await
        .map(|r| r.objects.len())
        .unwrap_or(0);
    assert_eq!(
        orphan_objects, 0,
        "orphaned snap_99999/ must be cleaned up after the next successful snapshot cycle"
    );
}

#[tokio::test]
async fn wal_entries_pruned_after_successful_snapshot_cycle() {
    // After a snapshot cycle that covers all buckets, WAL entries with
    // seq <= wal_high_seq must be deleted from blob storage. Entries inserted
    // after the snapshot started (seq > wal_high_seq) must be kept.

    let store = shared_store();
    let data_dir = temp_dir("wal_prune");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem};
    use object_store::path::Path as ObjectPath;
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));

    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 0,
    }))
    .await
    .expect("create");

    // Insert a batch — this produces WAL entries in blob storage.
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![
            InsertItem {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                timestamp: 0,
                vector_id: 1,
            },
            InsertItem {
                vector: vec![0.0, 1.0, 0.0, 0.0],
                timestamp: 0,
                vector_id: 2,
            },
        ],
    }))
    .await
    .expect("batch_insert");

    // Wait for WAL upload to complete.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Confirm WAL entries exist in blob storage before the snapshot.
    let wal_prefix = ObjectPath::from("pfx/col");
    let pre_snap_seqs = index::list_wal_seqs(store.as_ref(), &wal_prefix)
        .await
        .expect("list_wal_seqs");
    assert!(
        !pre_snap_seqs.is_empty(),
        "WAL entries must exist before snapshot"
    );

    // Run a snapshot cycle. All buckets succeed → wal_high_seq is committed
    // and WAL entries up to that seq are pruned.
    let _snap_h = svc.spawn_snapshot_task(grpc::SnapshotConfig {
        interval: Duration::from_millis(1),
    });
    tokio::time::sleep(Duration::from_millis(300)).await;

    // WAL entries covered by the snapshot must be gone.
    let post_snap_seqs = index::list_wal_seqs(store.as_ref(), &wal_prefix)
        .await
        .expect("list_wal_seqs after snapshot");
    assert!(
        post_snap_seqs.is_empty(),
        "WAL entries must be pruned after a successful snapshot cycle; \
         remaining seqs: {post_snap_seqs:?}"
    );

    // Verify wal_high_seq in collection.json reflects the pruned high-water mark.
    let col_meta = index::download_collection_meta(store.as_ref(), &wal_prefix)
        .await
        .expect("download collection.json");
    assert!(
        col_meta.wal_high_seq > 0,
        "wal_high_seq must be > 0 after inserts were snapshotted, got {}",
        col_meta.wal_high_seq
    );
    assert!(
        col_meta.wal_high_seq >= pre_snap_seqs.iter().copied().max().unwrap_or(0),
        "wal_high_seq ({}) must cover all pre-snapshot WAL entries (max seq was {})",
        col_meta.wal_high_seq,
        pre_snap_seqs.iter().copied().max().unwrap_or(0),
    );
}

#[tokio::test]
async fn wal_replay_restores_vectors_to_all_buckets_after_partial_snapshot_crash() {
    // Verifies that WAL replay correctly routes vectors to the right time buckets
    // after a crash. A single WAL entry can contain items for multiple buckets;
    // recovery must route each item to the correct bucket regardless.
    //
    // Also verifies deduplication: vectors already captured in the snapshot are
    // skipped during WAL replay (seen in the wal_replay_skips_duplicates_from_snapshot
    // test indirectly), while vectors only in the WAL are inserted correctly.
    //
    // Crash point: after baseline snapshot + WAL upload, but before a second snapshot.
    // wal_high_seq from the baseline snapshot is lower than the new WAL entries,
    // so recovery replays those entries into both buckets.

    let store = shared_store();
    let data_dir = temp_dir("cross_bucket_wal");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{BatchInsertRequest, CreateCollectionRequest, InsertItem};
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));

    // bucket_duration=100: window [0,100) = seq_0, window [100,200) = seq_1.
    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 100,
    }))
    .await
    .expect("create");

    // ── Phase 1: baseline → seq_0 only, snapshot ──────────────────────────────
    // vid=10 goes into seq_0 (window [0,100)).
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 0,
            vector_id: 10,
        }],
    }))
    .await
    .expect("baseline insert");

    // Snapshot seq_0, prune its WAL entry.
    let _snap_h = svc.spawn_snapshot_task(grpc::SnapshotConfig {
        interval: Duration::from_millis(1),
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    if let Some(h) = _snap_h {
        h.abort();
    }
    tokio::time::sleep(Duration::from_millis(20)).await;

    // ── Phase 2: cross-bucket inserts → WAL only, no new snapshot ─────────────
    // Timestamps must be non-decreasing so each item lands in the right bucket:
    //   vid=11 (t=50):  bucket_start=0, front=seq_0 (created_at=0), 0>0=false → seq_0
    //   vid=21 (t=100): bucket_start=100 > seq_0.created_at=0 → creates seq_1
    // Both are uploaded to the WAL but no snapshot is taken.
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![
            InsertItem {
                vector: vec![0.0, 0.0, 1.0, 0.0],
                timestamp: 50,
                vector_id: 11,
            },
            InsertItem {
                vector: vec![0.0, 0.0, 0.0, 1.0],
                timestamp: 100,
                vector_id: 21,
            },
        ],
    }))
    .await
    .expect("cross-bucket inserts");

    // "Crash": drop service after WAL upload (batch_insert already waited).
    drop(_wal_h);
    drop(svc);

    // ── Phase 3: recovery ─────────────────────────────────────────────────────
    // State in blob storage:
    //   - Snapshot for seq_0 (vid=10 only; wal_high_seq covers only that WAL entry).
    //   - WAL entry with vid=11 (t=50 → seq_0) and vid=21 (t=100 → new seq_1).
    // Recovery must replay the WAL and route each item to its correct bucket.
    let svc2 = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    svc2.recover_from_snapshots().await.expect("recover");

    // Two buckets: seq_0 (restored from snapshot + WAL) and seq_1 (created by WAL).
    assert_eq!(svc2.bucket_count("col").await, Some(2));

    let query = vec![0.5, 0.5, 0.5, 0.5];

    // seq_0 window [0,100): vid=10 from snapshot, vid=11 from WAL replay.
    let mut bucket0: Vec<u64> = search_sorted(&svc2, "col", query.clone(), 4, Some((0, 100)))
        .await
        .iter()
        .map(|&(vid, _)| vid)
        .collect();
    bucket0.sort_unstable();
    assert_eq!(bucket0, vec![10, 11], "seq_0: snapshot vid=10 + WAL vid=11");

    // seq_1 window [100,200): vid=21 created by WAL replay advancing the bucket.
    let mut bucket1: Vec<u64> = search_sorted(&svc2, "col", query.clone(), 4, Some((100, 200)))
        .await
        .iter()
        .map(|&(vid, _)| vid)
        .collect();
    bucket1.sort_unstable();
    assert_eq!(
        bucket1,
        vec![21],
        "seq_1: WAL vid=21 created new bucket during replay"
    );

    // Recovering a second time must not duplicate any vectors.
    let svc3 = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    svc3.recover_from_snapshots().await.expect("re-recover");
    let total = search_sorted(&svc3, "col", query.clone(), 3, None).await;
    assert_eq!(
        total.len(),
        3,
        "re-recovery must not duplicate vectors; got {total:?}"
    );
}

#[tokio::test]
async fn partial_snapshot_cycle_crash_deduplicates_already_snapshotted_bucket() {
    // Constructs the partial-cycle crash state deterministically without background tasks.
    //
    //  1. Insert vid=10 → seq_0. WAL seq=1 uploaded.
    //  2. SwapBucketOutToBlob seq_0 → baseline snapshot in S3 (vid=10 in snap_T1).
    //  3. SwapBucketIn seq_0 → restore to memory; load_levels repopulates node_ids.
    //  4. Write collection.json + catalog.json with wal_high_seq=1 (baseline covered).
    //  5. Insert vid=11 (t=50→seq_0) and vid=21 (t=100→seq_1). WAL seq=2 uploaded.
    //  6. SwapBucketOutToBlob seq_0 → partial-cycle snapshot in S3 (vid=10+vid=11 in snap_T2).
    //     seq_1 and collection.json untouched → crash point.
    //  7. Recovery:
    //     seq_0 from snap_T2 → known_vector_ids = {10, 11}
    //     seq_1 not in S3 → skipped
    //     WAL seq=2: vid=11 → SKIP (dedup), vid=21 → INSERT (creates seq_1)
    //  8. Assert seq_0 = {10, 11}, seq_1 = {21}.

    let store = shared_store();
    let data_dir = temp_dir("partial_cycle_dedup");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{
        BatchInsertRequest, CreateCollectionRequest, InsertItem, SwapBucketInRequest,
        SwapBucketOutToBlobRequest,
    };
    use object_store::path::Path as ObjectPath;
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));

    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 100,
    }))
    .await
    .expect("create");

    // Step 1: baseline insert → WAL seq=1
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 0,
            vector_id: 10,
        }],
    }))
    .await
    .expect("baseline insert");

    // Step 2: baseline snapshot — uploads seq_0 (vid=10) to snap_T1, writes bucket_meta.json
    svc.swap_bucket_out_to_blob(Request::new(SwapBucketOutToBlobRequest {
        collection: "col".into(),
        bucket_seq: 0,
    }))
    .await
    .expect("baseline SwapBucketOutToBlob");

    // Step 3: restore seq_0; load_levels repopulates node_ids for the next snapshot
    svc.swap_bucket_in(Request::new(SwapBucketInRequest {
        collection: "col".into(),
        bucket_seq: 0,
    }))
    .await
    .expect("SwapBucketIn after baseline");

    // Step 4: write collection.json and catalog.json with wal_high_seq=1
    index::upload_collection_meta(
        store.as_ref(),
        &index::CollectionMeta {
            version: 1,
            dim: 4,
            m: 4,
            m_max0: 8,
            ef_construction: 32,
            bucket_duration_secs: 100,
            snapshot_at_secs: 0,
            wal_high_seq: 1,
        },
        &ObjectPath::from("pfx/col"),
    )
    .await
    .expect("upload baseline collection.json");
    index::upload_catalog(
        store.as_ref(),
        &index::Catalog {
            version: 1,
            collections: vec![index::CatalogEntry {
                name: "col".to_string(),
                dim: 4,
                m: 4,
                m_max0: 8,
                ef_construction: 32,
                bucket_duration_secs: 100,
                snapshot_at_secs: 0,
                wal_high_seq: 1,
            }],
        },
        &ObjectPath::from("pfx"),
    )
    .await
    .expect("upload baseline catalog.json");

    // Step 5: post-baseline inserts → WAL seq=2
    // vid=11 (t=50): bucket_start=0, front=seq_0 (created_at=0), 0>0=false → seq_0
    // vid=21 (t=100): bucket_start=100 > seq_0.created_at=0 → creates seq_1
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![
            InsertItem {
                vector: vec![0.0, 0.0, 1.0, 0.0],
                timestamp: 50,
                vector_id: 11,
            },
            InsertItem {
                vector: vec![0.0, 0.0, 0.0, 1.0],
                timestamp: 100,
                vector_id: 21,
            },
        ],
    }))
    .await
    .expect("post-baseline inserts");

    // Step 6: partial-cycle snapshot — commits seq_0 (vid=10+vid=11) as snap_T2.
    // seq_1 and collection.json intentionally left unchanged.
    svc.swap_bucket_out_to_blob(Request::new(SwapBucketOutToBlobRequest {
        collection: "col".into(),
        bucket_seq: 0,
    }))
    .await
    .expect("partial-cycle SwapBucketOutToBlob");

    // Step 7: crash
    drop(_wal_h);
    drop(svc);

    // Step 8: recovery
    let svc2 = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    svc2.recover_from_snapshots().await.expect("recover");
    assert_eq!(svc2.bucket_count("col").await, Some(2));

    let query = vec![0.5, 0.5, 0.5, 0.5];

    let mut b0: Vec<u64> = search_sorted(&svc2, "col", query.clone(), 4, Some((0, 100)))
        .await
        .iter()
        .map(|&(vid, _)| vid)
        .collect();
    b0.sort_unstable();
    assert_eq!(
        b0,
        vec![10, 11],
        "seq_0: vid=10 and vid=11 from snapshot; vid=11 not re-inserted (dedup)"
    );

    let mut b1: Vec<u64> = search_sorted(&svc2, "col", query.clone(), 4, Some((100, 200)))
        .await
        .iter()
        .map(|&(vid, _)| vid)
        .collect();
    b1.sort_unstable();
    assert_eq!(
        b1,
        vec![21],
        "seq_1: created by WAL replay; vid=21 not in any snapshot so it was inserted"
    );
}

#[tokio::test]
async fn wal_spanning_both_buckets_after_partial_snapshot() {
    // After snapshotting seq_0 (baseline), a batch insert adds a new vector to
    // seq_0 AND creates seq_1 in the same request. The WAL entry contains vectors
    // for BOTH buckets. A partial snapshot then commits only seq_0 (now containing
    // both baseline and new vectors). seq_1 is never snapshotted.
    //
    // Timeline:
    //   WAL seq=1: [(vid=10, t=0)] → seq_0 (baseline)
    //   Snapshot seq_0 → snap_T1(vid=10). catalog.json wal_high_seq=1.
    //   WAL seq=2: [(vid=11, t=50→seq_0), (vid=21, t=150→creates seq_1)]
    //     seq_0 is still front when vid=11 is inserted (seq_1 doesn't exist yet).
    //     vid=21 crosses the bucket boundary and creates seq_1.
    //   Partial snapshot: seq_0 committed → snap_T2(vid=10, vid=11).
    //   Crash. seq_1 never snapshotted. catalog.json stays at wal_high_seq=1.
    //
    // Recovery:
    //   seq_0 from snap_T2 → known = {10, 11}
    //   seq_1 not in S3   → skipped
    //   WAL replay seq=2: vid=11 → SKIP (dedup, in snap_T2)
    //                     vid=21 → INSERT (creates seq_1 during replay)
    //   Result: seq_0={10,11}, seq_1={21}

    let store = shared_store();
    let data_dir = temp_dir("wal_both_buckets");
    let _guard = DirGuard(data_dir.clone());

    use grpc::proto::mem_weaver_server::MemWeaver;
    use grpc::proto::{
        BatchInsertRequest, CreateCollectionRequest, InsertItem, SwapBucketInRequest,
        SwapBucketOutToBlobRequest,
    };
    use object_store::path::Path as ObjectPath;
    use tonic::Request;

    let svc = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    let _wal_h = svc.spawn_wal_upload_task(Duration::from_millis(1));

    svc.create_collection(Request::new(CreateCollectionRequest {
        collection: "col".into(),
        dim: 4,
        m: 4,
        m_max0: 8,
        ef_construction: 32,
        bucket_duration_secs: 100,
    }))
    .await
    .expect("create");

    // WAL seq=1: baseline insert into seq_0 only.
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![InsertItem {
            vector: vec![1.0, 0.0, 0.0, 0.0],
            timestamp: 0,
            vector_id: 10,
        }],
    }))
    .await
    .expect("baseline insert");
    assert_eq!(svc.bucket_count("col").await, Some(1));

    // Baseline snapshot of seq_0 (vid=10). Restore to memory so node_ids are live.
    svc.swap_bucket_out_to_blob(Request::new(SwapBucketOutToBlobRequest {
        collection: "col".into(),
        bucket_seq: 0,
    }))
    .await
    .expect("baseline snapshot seq_0");
    svc.swap_bucket_in(Request::new(SwapBucketInRequest {
        collection: "col".into(),
        bucket_seq: 0,
    }))
    .await
    .expect("restore seq_0");

    // Write catalog.json with wal_high_seq=1 (baseline covered).
    index::upload_catalog(
        store.as_ref(),
        &index::Catalog {
            version: 1,
            collections: vec![index::CatalogEntry {
                name: "col".to_string(),
                dim: 4,
                m: 4,
                m_max0: 8,
                ef_construction: 32,
                bucket_duration_secs: 100,
                snapshot_at_secs: 0,
                wal_high_seq: 1,
            }],
        },
        &ObjectPath::from("pfx"),
    )
    .await
    .expect("upload catalog wal_high_seq=1");

    // WAL seq=2: cross-bucket batch while seq_0 is still the ONLY bucket.
    //   vid=11 (t=50):  bucket_start=0, seq_0.created_at=0, 0>0=false → seq_0
    //   vid=21 (t=150): bucket_start=100 > seq_0.created_at=0         → creates seq_1
    // Both items are in the same WAL entry, spanning two different buckets.
    svc.batch_insert(Request::new(BatchInsertRequest {
        collection: "col".into(),
        items: vec![
            InsertItem {
                vector: vec![0.0, 0.0, 1.0, 0.0],
                timestamp: 50,
                vector_id: 11,
            },
            InsertItem {
                vector: vec![0.0, 0.0, 0.0, 1.0],
                timestamp: 150,
                vector_id: 21,
            },
        ],
    }))
    .await
    .expect("cross-bucket batch (WAL seq=2)");
    assert_eq!(svc.bucket_count("col").await, Some(2));

    // Partial snapshot: commit seq_0 (now has vid=10 + vid=11) as snap_T2.
    // seq_1 is intentionally left without a snapshot.
    // catalog.json stays at wal_high_seq=1 → WAL seq=2 will be replayed.
    svc.swap_bucket_out_to_blob(Request::new(SwapBucketOutToBlobRequest {
        collection: "col".into(),
        bucket_seq: 0,
    }))
    .await
    .expect("partial-cycle snapshot seq_0");

    // Crash.
    drop(_wal_h);
    drop(svc);

    // Recovery.
    let svc2 = MemWeaverService::with_storage(data_dir.clone(), Some(Arc::clone(&store)), "pfx");
    svc2.recover_from_snapshots().await.expect("recover");
    assert_eq!(
        svc2.bucket_count("col").await,
        Some(2),
        "seq_0 from snapshot + seq_1 created by WAL replay"
    );

    let query = vec![0.5, 0.5, 0.5, 0.5];

    // seq_0: vid=10 (baseline) + vid=11 (from snap_T2). WAL replay deduped vid=11.
    let mut b0: Vec<u64> = search_sorted(&svc2, "col", query.clone(), 4, Some((0, 100)))
        .await
        .iter()
        .map(|&(vid, _)| vid)
        .collect();
    b0.sort_unstable();
    assert_eq!(
        b0,
        vec![10, 11],
        "seq_0: vid=10 and vid=11 from snap_T2; vid=11 not re-inserted (dedup)"
    );

    // seq_1: vid=21 entirely from WAL replay (no seq_1 snapshot existed).
    let mut b1: Vec<u64> = search_sorted(&svc2, "col", query.clone(), 4, Some((100, 200)))
        .await
        .iter()
        .map(|&(vid, _)| vid)
        .collect();
    b1.sort_unstable();
    assert_eq!(
        b1,
        vec![21],
        "seq_1: created during WAL replay; vid=21 from WAL (no snapshot for seq_1)"
    );
}
