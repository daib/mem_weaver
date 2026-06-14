//! gRPC service wrapping [`index::TimeBucketIndex`].

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use common::{top_k_quickselect, Timestamp};
use index::{
    decode_wal_entry, delete_prefix, delete_wal_entries_up_to, download_wal_entry,
    encode_wal_entry, list_wal_seqs, upload_wal_bytes, BucketMeta, BucketSeq, Catalog,
    CatalogEntry, CollectionMeta, TimeBucketIndex, WalEntry, WalItem,
};
use index::{
    download_arena_dir, download_bucket_meta, download_catalog, download_collection_meta,
    download_levels, download_manifest,
};
use index::{
    upload_arena_dir, upload_bucket_meta, upload_catalog, upload_collection_meta, upload_levels,
    upload_manifest,
};
use object_store::{path::Path as ObjectPath, ObjectStore};
use rand::{rngs::StdRng, SeedableRng};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

/// Per-collection WAL state. Tracks the monotonic sequence counter and provides
/// a watch channel so `batch_insert` can block until its entry is uploaded to S3.
struct WalState {
    next_seq: AtomicU64,
    /// Sender updated by the WAL uploader each time a seq is confirmed uploaded.
    uploaded_tx: tokio::sync::watch::Sender<u64>,
    uploaded_rx: tokio::sync::watch::Receiver<u64>,
}

impl WalState {
    fn new(start_seq: u64) -> Arc<Self> {
        let (tx, rx) = tokio::sync::watch::channel(start_seq.saturating_sub(1));
        Arc::new(Self {
            next_seq: AtomicU64::new(start_seq),
            uploaded_tx: tx,
            uploaded_rx: rx,
        })
    }

    fn alloc_seq(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::Relaxed)
    }

    fn current_seq(&self) -> u64 {
        self.next_seq.load(Ordering::Relaxed).saturating_sub(1)
    }

    fn mark_uploaded(&self, seq: u64) {
        let _ = self.uploaded_tx.send(seq);
    }

    async fn wait_for_upload(&self, seq: u64) {
        let mut rx = self.uploaded_rx.clone();
        loop {
            if *rx.borrow() >= seq {
                return;
            }
            if rx.changed().await.is_err() {
                return; // sender dropped (service shutting down)
            }
        }
    }
}

/// Configuration for the periodic arena snapshot task.
pub struct SnapshotConfig {
    /// How often to take a snapshot of every in-memory bucket across all collections.
    pub interval: Duration,
}

pub mod proto {
    tonic::include_proto!("mem_weaver.v1");
}

use proto::mem_weaver_server::MemWeaver;
use proto::{
    BatchInsertRequest, BatchInsertResponse, CreateCollectionRequest, CreateCollectionResponse,
    EvictBucketRequest, EvictBucketResponse, InsertResult, SearchHit, SearchRequest,
    SearchResponse, SwapBucketInFromBlobRequest, SwapBucketInFromBlobResponse, SwapBucketInRequest,
    SwapBucketInResponse, SwapBucketOutRequest, SwapBucketOutResponse, SwapBucketOutToBlobRequest,
    SwapBucketOutToBlobResponse,
};

pub use proto::mem_weaver_server::MemWeaverServer;

type Collection = Arc<RwLock<TimeBucketIndex>>;

pub struct MemWeaverService {
    collections: Arc<RwLock<HashMap<String, Collection>>>,
    /// Per-collection WAL state (seq counter + upload notification channel).
    wal_states: Arc<std::sync::Mutex<HashMap<String, Arc<WalState>>>>,
    /// Base directory for on-disk bucket files. Bucket `seq` of collection `c` is stored
    /// under `<data_dir>/<c>/seq_<seq>/`.
    data_dir: PathBuf,
    /// Optional blob store (S3-compatible).
    store: Option<Arc<dyn ObjectStore>>,
    /// Base blob prefix. Bucket `seq` of collection `c` lives at
    /// `<blob_prefix>/<c>/seq_<seq>/`.
    blob_prefix: ObjectPath,
}

impl MemWeaverService {
    pub fn new() -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            wal_states: Arc::new(std::sync::Mutex::new(HashMap::new())),
            data_dir: PathBuf::from("./data"),
            store: None,
            blob_prefix: ObjectPath::from("mem-weaver"),
        }
    }

    /// Returns the number of in-memory collections. Useful for testing recovery.
    pub async fn collection_count(&self) -> usize {
        self.collections.read().await.len()
    }

    /// Returns `true` if a collection with `name` exists in memory.
    pub async fn has_collection(&self, name: &str) -> bool {
        self.collections.read().await.contains_key(name)
    }

    /// Returns the number of buckets in collection `name`, or `None` if the collection
    /// does not exist.
    pub async fn bucket_count(&self, name: &str) -> Option<usize> {
        let col = self.collections.read().await.get(name).cloned()?;
        let n = col.read().await.bucket_count();
        Some(n)
    }

    /// Returns whether the specified bucket has dirty arena blocks. Test helper.
    pub async fn is_bucket_dirty_test(&self, collection: &str, bucket_seq: u32) -> bool {
        match self.collections.read().await.get(collection).cloned() {
            None => false,
            Some(col) => col.read().await.is_bucket_dirty(BucketSeq(bucket_seq)),
        }
    }

    /// Returns the current write count for the specified bucket. Intended for testing
    /// the dirty-flag race condition fix: capture this value before a simulated
    /// concurrent insert, then pass it to [`mark_bucket_clean_if_version`].
    pub async fn bucket_write_count(&self, collection: &str, bucket_seq: u32) -> u64 {
        match self.collections.read().await.get(collection).cloned() {
            None => 0,
            Some(col) => col.read().await.bucket_write_count(BucketSeq(bucket_seq)),
        }
    }

    /// Call `mark_bucket_clean_if_version` on the specified bucket. Used in tests to
    /// simulate the snapshot task's mark-clean step with a captured version number,
    /// exercising the race condition fix without actually running the full task.
    pub async fn mark_bucket_clean_if_version(
        &self,
        collection: &str,
        bucket_seq: u32,
        version: u64,
    ) -> bool {
        match self.collections.read().await.get(collection).cloned() {
            None => false,
            Some(col) => {
                col.write()
                    .await
                    .mark_bucket_clean_if_version(BucketSeq(bucket_seq), version);
                true
            }
        }
    }

    /// Permanently drop the oldest bucket from `name`'s index (delegates to
    /// [`TimeBucketIndex::evict_oldest`]). Returns `false` if the collection does
    /// not exist or has no buckets. Intended for TTL-style bucket management and tests.
    pub async fn evict_oldest_bucket(&self, name: &str) -> bool {
        let col = match self.collections.read().await.get(name).cloned() {
            Some(c) => c,
            None => return false,
        };
        let found = col.write().await.evict_oldest().is_some();
        found
    }

    pub fn with_storage(
        data_dir: impl Into<PathBuf>,
        store: Option<Arc<dyn ObjectStore>>,
        blob_prefix: impl Into<String>,
    ) -> Self {
        Self {
            collections: Arc::new(RwLock::new(HashMap::new())),
            wal_states: Arc::new(std::sync::Mutex::new(HashMap::new())),
            data_dir: data_dir.into(),
            store,
            blob_prefix: ObjectPath::from(blob_prefix.into()),
        }
    }

    fn wal_local_dir(&self, collection: &str) -> PathBuf {
        self.data_dir.join(collection).join("wal")
    }

    fn wal_blob_prefix(&self, collection: &str) -> ObjectPath {
        self.blob_prefix.child(collection)
    }

    fn get_or_create_wal(&self, collection: &str, start_seq: u64) -> Arc<WalState> {
        let mut map = self.wal_states.lock().expect("wal_states lock");
        map.entry(collection.to_owned())
            .or_insert_with(|| WalState::new(start_seq))
            .clone()
    }

    fn bucket_local_dir(&self, collection: &str, seq: u32) -> PathBuf {
        self.data_dir.join(collection).join(format!("seq_{seq}"))
    }

    fn bucket_blob_prefix(&self, collection: &str, seq: u32) -> ObjectPath {
        self.blob_prefix
            .child(collection)
            .child(format!("seq_{seq}"))
    }
}

impl Default for MemWeaverService {
    fn default() -> Self {
        Self::new()
    }
}

impl MemWeaverService {
    /// Spawn a background task that periodically snapshots every in-memory bucket to S3.
    ///
    /// Each snapshot cycle: for every collection and every bucket, arena blocks are copied
    /// to a temporary local directory (under `<data_dir>/<collection>/snap_<seq>/`), then
    /// uploaded to `<blob_prefix>/<collection>/seq_<seq>/` in the configured object store.
    /// The local copy is removed after a successful upload. Storage state is never changed —
    /// in-memory arenas stay in memory and search keeps working throughout.
    ///
    /// Returns `None` if no object store is configured (S3 must be set up via
    /// [`MemWeaverService::with_storage`]).
    pub fn spawn_snapshot_task(
        &self,
        config: SnapshotConfig,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let store = self.store.clone()?;
        let collections = Arc::clone(&self.collections);
        let wal_states = Arc::clone(&self.wal_states);
        let data_dir = self.data_dir.clone();
        let blob_prefix = self.blob_prefix.clone();

        Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(config.interval).await;

                let now_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Snapshot each collection under a read lock so inserts are never blocked.
                let col_map: Vec<(String, Arc<RwLock<TimeBucketIndex>>)> = collections
                    .read()
                    .await
                    .iter()
                    .map(|(name, col)| (name.clone(), Arc::clone(col)))
                    .collect();

                for (col_name, col) in col_map {
                    let col_prefix = blob_prefix.child(col_name.as_str());

                    // Read config and bucket list under a single read lock.
                    let (col_config, bucket_metas) = {
                        let idx = col.read().await;
                        (idx.config(), idx.bucket_metas())
                    };

                    let (dim, m, m_max0, ef_construction, bucket_duration) = col_config;
                    // Snapshot the WAL high-water mark now. collection.json is only
                    // written after ALL buckets succeed, so this value is never
                    // committed unless the full cycle completes.
                    let wal_high_seq = {
                        let map = wal_states.lock().expect("wal_states lock");
                        map.get(&col_name).map_or(0, |w| w.current_seq())
                    };

                    // Track seqs that were successfully committed this cycle so we
                    // can decide whether to advance wal_high_seq and delete stale prefixes.
                    let mut committed: std::collections::HashSet<u32> =
                        std::collections::HashSet::new();

                    for (seq, created_at) in &bucket_metas {
                        let seq = *seq;
                        let created_at = *created_at;
                        let snap_dir = data_dir.join(&col_name).join(format!("snap_{}", seq.0));
                        let bucket_prefix = col_prefix.child(format!("seq_{}", seq.0));

                        // Skip buckets with no dirty blocks — their previous snapshot
                        // is still valid so no upload is needed.
                        let is_dirty = col.read().await.is_bucket_dirty(seq);
                        if !is_dirty {
                            committed.insert(seq.0);
                            // Still clean up any orphaned staging dirs for this bucket
                            // (e.g. from a crash mid-upload in a previous cycle).
                            if let Ok(meta) =
                                download_bucket_meta(store.as_ref(), &bucket_prefix).await
                            {
                                cleanup_old_snaps(store.as_ref(), &bucket_prefix, &meta.snap_dir)
                                    .await;
                            }
                            continue;
                        }

                        // Capture write_count and snapshot atomically under the same
                        // read lock. Used after upload to avoid clearing dirty on blocks
                        // written between the snapshot and the upload completion.
                        let (snapped, write_count) = {
                            let idx = col.read().await;
                            let wc = idx.bucket_write_count(seq);
                            (idx.snapshot_bucket(seq, &snap_dir), wc)
                        };

                        match snapped {
                            // Evicted with local files gone: already in blob storage, still live.
                            Ok(false) => {
                                committed.insert(seq.0);
                                continue;
                            }
                            Err(e) => {
                                eprintln!(
                                    "snapshot: {col_name}/seq_{}: disk write failed: {e}",
                                    seq.0
                                );
                                let _ = std::fs::remove_dir_all(&snap_dir);
                                continue;
                            }
                            Ok(true) => {}
                        }

                        // Stage files in a versioned subdirectory so the current
                        // complete snapshot is never partially overwritten.
                        // Layout: seq_<N>/snap_<T>/{block_*.arena, levels.bin, manifest.json}
                        // Commit: seq_<N>/bucket_meta.json points to snap_<T>.
                        let snap_subdir = format!(
                            "snap_{}",
                            std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_millis()
                        );
                        let staged_prefix = bucket_prefix.child(snap_subdir.as_str());

                        let bucket_meta = BucketMeta {
                            version: 1,
                            seq: seq.0,
                            created_at_secs: created_at.0,
                            snap_dir: snap_subdir.clone(),
                        };
                        let upload = async {
                            // Upload all content files to the staged prefix.
                            upload_arena_dir(store.as_ref(), &snap_dir, &staged_prefix).await?;
                            upload_levels(
                                store.as_ref(),
                                &snap_dir.join("levels.bin"),
                                &staged_prefix,
                            )
                            .await?;
                            upload_manifest(
                                store.as_ref(),
                                &snap_dir.join("manifest.json"),
                                &staged_prefix,
                            )
                            .await?;
                            // Atomic commit: write bucket_meta.json at seq_<N>/ pointing
                            // to the new staged prefix. The old snapshot is still intact
                            // until this PUT succeeds.
                            upload_bucket_meta(store.as_ref(), &bucket_meta, &bucket_prefix).await
                        };
                        match upload.await {
                            Ok(()) => {
                                committed.insert(seq.0);
                                // Only mark clean if no writes arrived after the snapshot
                                // was taken (write_count captured under the same read lock).
                                col.write()
                                    .await
                                    .mark_bucket_clean_if_version(seq, write_count);
                                // Clean up stale snap_* dirs for this bucket.
                                cleanup_old_snaps(store.as_ref(), &bucket_prefix, &snap_subdir)
                                    .await;
                            }
                            Err(e) => eprintln!(
                                "snapshot: {col_name}/seq_{}: blob upload failed: {e}",
                                seq.0
                            ),
                        }

                        let _ = std::fs::remove_dir_all(&snap_dir);
                    }

                    // Delete any blob-prefix that belongs to a bucket no longer in the
                    // index (evicted via evict_oldest / evict_before). We only do this
                    // when the full cycle completed without skipping any live buckets, to
                    // avoid deleting data that failed to re-upload this cycle.
                    let live_seqs: std::collections::HashSet<u32> =
                        bucket_metas.iter().map(|(s, _)| s.0).collect();
                    if committed == live_seqs {
                        // Every live bucket was successfully snapshotted this cycle.
                        // Only now is it safe to advance wal_high_seq: all buckets
                        // have data up to this point, so recovery can safely skip
                        // WAL entries up to wal_high_seq without losing vectors from
                        // any bucket.
                        let col_meta = CollectionMeta {
                            version: 1,
                            dim,
                            m,
                            m_max0,
                            ef_construction,
                            bucket_duration_secs: bucket_duration.as_secs(),
                            snapshot_at_secs: now_secs,
                            wal_high_seq,
                        };
                        match upload_collection_meta(store.as_ref(), &col_meta, &col_prefix).await {
                            Err(e) => eprintln!(
                                "snapshot: {col_name}: collection.json upload failed: {e}"
                            ),
                            Ok(()) => {
                                if let Err(e) = delete_wal_entries_up_to(
                                    store.as_ref(),
                                    &col_prefix,
                                    wal_high_seq,
                                )
                                .await
                                {
                                    eprintln!("snapshot: {col_name}: WAL prune failed: {e}");
                                }
                            }
                        }

                        // List all seq_*/ prefixes in blob storage for this collection.
                        let listed = store.list_with_delimiter(Some(&col_prefix)).await;
                        match listed {
                            Err(e) => eprintln!(
                                "snapshot: {col_name}: failed to list blob prefixes for cleanup: {e}"
                            ),
                            Ok(result) => {
                                for stale_prefix in result.common_prefixes {
                                    // Extract seq number from "seq_N" path component.
                                    let last = stale_prefix
                                        .parts()
                                        .last()
                                        .map(|p| p.as_ref().to_owned());
                                    let seq_num = last
                                        .as_deref()
                                        .and_then(|s| s.strip_prefix("seq_"))
                                        .and_then(|s| s.parse::<u32>().ok());
                                    let Some(seq_num) = seq_num else { continue };
                                    if !live_seqs.contains(&seq_num) {
                                        match delete_prefix(store.as_ref(), &stale_prefix).await {
                                            Ok(n) => eprintln!(
                                                "snapshot: {col_name}/seq_{seq_num}: deleted {n} stale object(s)"
                                            ),
                                            Err(e) => eprintln!(
                                                "snapshot: {col_name}/seq_{seq_num}: stale prefix delete failed: {e}"
                                            ),
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }))
    }

    /// Spawn a background task that uploads pending WAL entries to blob storage and
    /// signals waiting `batch_insert` callers once their entry is confirmed uploaded.
    ///
    /// Spawn a background task that uploads pending WAL entries to blob storage and
    /// tick, uploads any file not yet in blob storage, then calls `mark_uploaded` so
    /// `batch_insert` can return. Local WAL files are removed after a successful upload.
    ///
    /// Returns `None` if no object store is configured.
    pub fn spawn_wal_upload_task(&self, interval: Duration) -> Option<tokio::task::JoinHandle<()>> {
        let store = self.store.clone()?;
        let wal_states = Arc::clone(&self.wal_states);
        let data_dir = self.data_dir.clone();
        let blob_prefix = self.blob_prefix.clone();

        Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;

                // Collect (collection_name, wal_dir) pairs for all live collections.
                let entries: Vec<(String, PathBuf, ObjectPath)> = {
                    let map = wal_states.lock().expect("wal_states lock");
                    map.keys()
                        .map(|name| {
                            let wal_dir = data_dir.join(name).join("wal");
                            let prefix = blob_prefix.child(name.as_str());
                            (name.clone(), wal_dir, prefix)
                        })
                        .collect()
                };

                for (col_name, wal_dir, col_prefix) in entries {
                    // List local WAL files sorted ascending.
                    let local_files: Vec<(u64, PathBuf)> = match std::fs::read_dir(&wal_dir) {
                        Err(_) => continue, // no WAL dir yet
                        Ok(dir) => {
                            let mut v: Vec<(u64, PathBuf)> = dir
                                .filter_map(|e| {
                                    let p = e.ok()?.path();
                                    if p.extension().and_then(|s| s.to_str()) != Some("wal") {
                                        return None;
                                    }
                                    let seq = p
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .and_then(|s| s.parse::<u64>().ok())?;
                                    Some((seq, p))
                                })
                                .collect();
                            v.sort_by_key(|(seq, _)| *seq);
                            v
                        }
                    };

                    let wal = {
                        let map = wal_states.lock().expect("wal_states lock");
                        map.get(&col_name).cloned()
                    };
                    let Some(wal) = wal else { continue };

                    for (seq, path) in local_files {
                        // Read raw bytes — validate CRC32 before uploading.
                        let bytes = match std::fs::read(&path) {
                            Ok(b) => b,
                            Err(e) => {
                                eprintln!("wal: {col_name}/seq_{seq}: read failed: {e}");
                                continue;
                            }
                        };
                        if decode_wal_entry(&bytes).is_err() {
                            eprintln!("wal: {col_name}/seq_{seq}: corrupt entry, skipping");
                            let _ = std::fs::remove_file(&path);
                            continue;
                        }

                        // Upload raw bytes directly — no re-serialisation needed.
                        let blob_bytes = bytes::Bytes::from(bytes);
                        if let Err(e) =
                            upload_wal_bytes(store.as_ref(), seq, blob_bytes, &col_prefix).await
                        {
                            eprintln!("wal: {col_name}/seq_{seq}: upload failed: {e}");
                            continue;
                        }

                        // Remove local file and signal waiting inserts.
                        let _ = std::fs::remove_file(&path);
                        wal.mark_uploaded(seq);
                    }
                }
            }
        }))
    }

    /// Restore all collections from the most recent snapshots in S3.
    ///
    /// Called once at startup (before the gRPC server begins accepting requests) to
    /// recover from a crash. For each collection found under `blob_prefix`:
    /// 1. `collection.json` is read to obtain index configuration.
    /// 2. Every `seq_<N>/` prefix is scanned for `bucket_meta.json`.
    /// 3. Arena blocks, `levels.bin`, and `manifest.json` are downloaded for each bucket.
    /// 4. The collection is reconstructed in memory with buckets sorted oldest-first.
    ///
    /// Buckets that fail to restore (corrupt data, missing files) are skipped with a
    /// warning. Returns an error only if listing blob storage fails at the top level. Collections
    /// that already exist in memory are not overwritten.
    ///
    /// Returns `Ok(())` immediately if no object store is configured.
    pub async fn recover_from_snapshots(&self) -> std::io::Result<()> {
        let store = match &self.store {
            Some(s) => s.clone(),
            None => return Ok(()),
        };

        // Use the catalog to discover which collections to restore. It carries both the
        // collection names and their index configs — no per-collection blob read needed.
        //
        // If no catalog exists yet (first boot, or a node that pre-dates the catalog)
        // fall back to listing all prefixes and downloading each collection.json.
        let collection_entries: Vec<CatalogEntry> =
            match download_catalog(store.as_ref(), &self.blob_prefix).await {
                Ok(catalog) => {
                    eprintln!(
                        "recovery: catalog found ({} collection(s))",
                        catalog.collections.len()
                    );
                    catalog.collections
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!("recovery: no catalog found, scanning all blob prefixes");
                    let list = store
                        .list_with_delimiter(Some(&self.blob_prefix))
                        .await
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                    let mut entries = Vec::new();
                    for col_prefix in list.common_prefixes {
                        let col_name =
                            match col_prefix.parts().last().map(|s| s.as_ref().to_owned()) {
                                Some(n) if !n.is_empty() => n,
                                _ => continue,
                            };
                        match download_collection_meta(store.as_ref(), &col_prefix).await {
                            Ok(m) => entries.push(CatalogEntry {
                                name: col_name,
                                dim: m.dim,
                                m: m.m,
                                m_max0: m.m_max0,
                                ef_construction: m.ef_construction,
                                bucket_duration_secs: m.bucket_duration_secs,
                                snapshot_at_secs: m.snapshot_at_secs,
                                wal_high_seq: m.wal_high_seq,
                            }),
                            Err(e) => eprintln!("recovery: {col_name}: collection.json error: {e}"),
                        }
                    }
                    entries
                }
                Err(e) => return Err(e),
            };

        for entry in collection_entries {
            let col_name = entry.name.clone();
            let col_prefix = self.blob_prefix.child(col_name.as_str());

            // Skip if this collection already exists in memory.
            if self.collections.read().await.contains_key(&col_name) {
                continue;
            }

            // Config comes directly from the catalog entry — no per-collection blob read needed.
            let bucket_duration = Duration::from_secs(entry.bucket_duration_secs);
            let mut index = match TimeBucketIndex::new(
                entry.dim,
                entry.m,
                entry.m_max0,
                entry.ef_construction,
                bucket_duration,
                top_k_quickselect,
                rand::rngs::StdRng::seed_from_u64(0),
            ) {
                Ok(idx) => idx,
                Err(e) => {
                    eprintln!("recovery: {col_name}: invalid config in collection.json: {e}");
                    continue;
                }
            };

            // List seq_*/ prefixes to find bucket snapshots.
            let bucket_list = match store
                .list_with_delimiter(Some(&col_prefix))
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
            {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("recovery: {col_name}: failed to list buckets: {e}");
                    continue;
                }
            };

            // Gather (created_at, seq, versioned_prefix) for all buckets, then sort oldest-first.
            // bucket_meta.json lives at seq_<N>/ and points to the versioned snap_<T>/ subdir
            // where the actual arena files reside.
            let mut bucket_infos: Vec<(u64, u32, ObjectPath)> = Vec::new();
            for bucket_prefix in bucket_list.common_prefixes {
                let meta = match download_bucket_meta(store.as_ref(), &bucket_prefix).await {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!(
                            "recovery: {col_name}/{}: bucket_meta.json missing: {e}",
                            bucket_prefix
                        );
                        continue;
                    }
                };
                // Resolve the versioned subdirectory where the arena files live.
                let versioned_prefix = bucket_prefix.child(meta.snap_dir.as_str());
                bucket_infos.push((meta.created_at_secs, meta.seq, versioned_prefix));
            }
            bucket_infos.sort_unstable_by_key(|&(created_at, seq, _)| (created_at, seq));

            // Download and restore each bucket oldest-first.
            for (created_at_secs, seq, versioned_prefix) in bucket_infos {
                let local_dir = self.data_dir.join(&col_name).join(format!("recover_{seq}"));

                if let Err(e) = std::fs::create_dir_all(&local_dir) {
                    eprintln!("recovery: {col_name}/seq_{seq}: mkdir failed: {e}");
                    continue;
                }

                let download = async {
                    download_arena_dir(store.as_ref(), &versioned_prefix, &local_dir).await?;
                    download_levels(
                        store.as_ref(),
                        &versioned_prefix,
                        &local_dir.join("levels.bin"),
                    )
                    .await?;
                    download_manifest(
                        store.as_ref(),
                        &versioned_prefix,
                        &local_dir.join("manifest.json"),
                    )
                    .await
                };
                if let Err(e) = download.await {
                    eprintln!("recovery: {col_name}/seq_{seq}: blob download failed: {e}");
                    let _ = std::fs::remove_dir_all(&local_dir);
                    continue;
                }

                match index.add_restored_bucket(
                    BucketSeq(seq),
                    Timestamp(created_at_secs),
                    &local_dir,
                ) {
                    Ok(n) => eprintln!("recovery: {col_name}/seq_{seq}: restored {n} block(s)"),
                    Err(e) => eprintln!("recovery: {col_name}/seq_{seq}: restore failed: {e}"),
                }

                let _ = std::fs::remove_dir_all(&local_dir);
            }

            eprintln!(
                "recovery: {col_name}: {} bucket(s) restored",
                index.bucket_count()
            );

            // Replay WAL entries that arrived after the snapshot was taken.
            // Duplicate detection is handled by TimeBucketIndex::insert, which tracks
            // known_vector_ids internally — no dedup logic needed here.
            let wal_high_seq = entry.wal_high_seq;
            let col_prefix = self.blob_prefix.child(col_name.as_str());
            let wal_seqs = list_wal_seqs(store.as_ref(), &col_prefix)
                .await
                .unwrap_or_default();
            let mut max_replayed_seq = wal_high_seq;
            for seq in wal_seqs.into_iter().filter(|&s| s > wal_high_seq) {
                match download_wal_entry(store.as_ref(), &col_prefix, seq).await {
                    Ok(wal_entry) => {
                        let mut inserted = 0usize;
                        let mut skipped = 0usize;
                        for item in &wal_entry.items {
                            match index.insert(
                                &item.vector,
                                Timestamp(item.timestamp),
                                item.vector_id,
                            ) {
                                Some(_) => inserted += 1,
                                None => skipped += 1,
                            }
                        }
                        max_replayed_seq = max_replayed_seq.max(seq);
                        eprintln!(
                            "recovery: {col_name}: replayed WAL seq {seq} \
                             ({inserted} inserted, {skipped} duplicate(s) skipped)"
                        );
                    }
                    Err(e) => {
                        eprintln!("recovery: {col_name}: WAL seq {seq} download failed: {e}");
                    }
                }
            }

            // Initialise WalState so new inserts continue from where WAL left off.
            let start_seq = max_replayed_seq + 1;
            self.get_or_create_wal(&col_name, start_seq);

            self.collections
                .write()
                .await
                .insert(col_name, Arc::new(RwLock::new(index)));
        }

        Ok(())
    }
}

/// Delete all `snap_*/` subdirectories under `bucket_prefix` in `store` except for
/// `active_snap`. Called after a successful upload (dirty bucket) and after confirming
/// a clean bucket's current pointer, so orphaned staging dirs from crashed cycles are
/// always cleaned up regardless of whether a new upload happened this cycle.
async fn cleanup_old_snaps(store: &dyn ObjectStore, bucket_prefix: &ObjectPath, active_snap: &str) {
    if let Ok(listing) = store.list_with_delimiter(Some(bucket_prefix)).await {
        for old in listing.common_prefixes {
            let name = old.parts().last().map(|p| p.as_ref().to_owned());
            if let Some(name) = name {
                if name != active_snap && name.starts_with("snap_") {
                    let _ = delete_prefix(store, &old).await;
                }
            }
        }
    }
}

fn invalid(msg: impl Into<String>) -> Status {
    Status::invalid_argument(msg.into())
}

fn internal(e: impl std::fmt::Display) -> Status {
    Status::internal(e.to_string())
}

async fn get_collection(
    cols: &RwLock<HashMap<String, Collection>>,
    name: &str,
) -> Result<Collection, Status> {
    cols.read()
        .await
        .get(name)
        .cloned()
        .ok_or_else(|| Status::not_found(format!("collection '{name}' not found")))
}

fn validate_name(name: &str) -> Result<(), Status> {
    if name.is_empty() {
        Err(invalid("collection name must not be empty"))
    } else {
        Ok(())
    }
}

#[tonic::async_trait]
impl MemWeaver for MemWeaverService {
    async fn create_collection(
        &self,
        req: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        if r.dim == 0 {
            return Err(invalid("dim must be > 0"));
        }
        let bucket_duration = if r.bucket_duration_secs == 0 {
            Duration::from_secs(u64::MAX / 2)
        } else {
            Duration::from_secs(r.bucket_duration_secs)
        };
        let index = TimeBucketIndex::new(
            r.dim as usize,
            r.m as usize,
            r.m_max0 as usize,
            r.ef_construction as usize,
            bucket_duration,
            top_k_quickselect,
            StdRng::seed_from_u64(0),
        )
        .map_err(|e| invalid(format!("invalid index config: {e}")))?;

        // Insert under the write lock, then release it before any async S3 work.
        {
            let mut cols = self.collections.write().await;
            if cols.contains_key(&r.collection) {
                return Err(Status::already_exists(format!(
                    "collection '{}' already exists",
                    r.collection
                )));
            }
            cols.insert(r.collection.clone(), Arc::new(RwLock::new(index)));
        }

        // Create WAL state for this collection starting at seq 1.
        self.get_or_create_wal(&r.collection, 1);

        // Best-effort: write the updated catalog to blob storage so recovery knows this
        // collection exists (with its full config). Done after releasing the write
        // lock to avoid holding it across async I/O. Failure is logged, not fatal.
        if let Some(store) = &self.store {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // Read configs from all collections under a shared read lock.
            let mut entries: Vec<CatalogEntry> = {
                let cols = self.collections.read().await;
                let mut v = Vec::with_capacity(cols.len());
                for (name, col) in cols.iter() {
                    let (dim, m, m_max0, ef_construction, bd) = col.read().await.config();
                    let wal_high = {
                        let map = self.wal_states.lock().expect("wal_states");
                        map.get(name).map_or(0, |w| w.current_seq())
                    };
                    v.push(CatalogEntry {
                        name: name.clone(),
                        dim,
                        m,
                        m_max0,
                        ef_construction,
                        bucket_duration_secs: bd.as_secs(),
                        snapshot_at_secs: now_secs,
                        wal_high_seq: wal_high,
                    });
                }
                v
            };
            entries.sort_unstable_by(|a, b| a.name.cmp(&b.name));
            let catalog = Catalog {
                version: 1,
                collections: entries,
            };
            if let Err(e) = upload_catalog(store.as_ref(), &catalog, &self.blob_prefix).await {
                eprintln!("create_collection: catalog upload failed: {e}");
            }
        }

        Ok(Response::new(CreateCollectionResponse {}))
    }

    async fn batch_insert(
        &self,
        req: Request<BatchInsertRequest>,
    ) -> Result<Response<BatchInsertResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        if r.items.is_empty() {
            return Ok(Response::new(BatchInsertResponse { results: vec![] }));
        }
        for (i, item) in r.items.iter().enumerate() {
            if item.vector.is_empty() {
                return Err(invalid(format!("item[{i}]: vector must not be empty")));
            }
        }
        let col = get_collection(&self.collections, &r.collection).await?;

        // Alloc a WAL seq, apply to the in-memory index under the write lock, then
        // release the lock before the (potentially slow) WAL write + blob upload.
        let wal = self.get_or_create_wal(&r.collection, 1);
        let seq = wal.alloc_seq();
        let results: Vec<InsertResult> = {
            let mut idx = col.write().await;
            r.items
                .iter()
                .filter_map(|item| {
                    idx.insert(&item.vector, Timestamp(item.timestamp), item.vector_id)
                        .map(|bid| InsertResult {
                            bucket_seq: bid.bucket_seq.0,
                            vector_id: bid.vector_id,
                        })
                })
                .collect()
        };

        // Write WAL entry to local disk so it survives a crash before S3 upload.
        let wal_dir = self.wal_local_dir(&r.collection);
        if let Err(e) = std::fs::create_dir_all(&wal_dir) {
            return Err(internal(format!("WAL dir create failed: {e}")));
        }
        let wal_path = wal_dir.join(format!("{seq:020}.wal"));
        let wal_entry = WalEntry {
            seq,
            items: r
                .items
                .iter()
                .map(|item| WalItem {
                    vector: item.vector.clone(),
                    timestamp: item.timestamp,
                    vector_id: item.vector_id,
                })
                .collect(),
        };
        let encoded = encode_wal_entry(&wal_entry);
        std::fs::write(&wal_path, &encoded)
            .map_err(|e| internal(format!("WAL disk write failed: {e}")))?;

        // Wait until the background WAL uploader confirms this entry is in blob storage.
        // If no store is configured the uploader never runs; skip the wait so inserts
        // are acked immediately (same behaviour as before WAL was added).
        if self.store.is_some() {
            wal.wait_for_upload(seq).await;
        }

        Ok(Response::new(BatchInsertResponse { results }))
    }

    async fn search(
        &self,
        req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        if r.query.is_empty() {
            return Err(invalid("query must not be empty"));
        }
        let k = r.k as usize;
        let ef = r.ef as usize;
        if k == 0 {
            return Err(invalid("k must be > 0"));
        }
        let time_range = match (r.time_range_start, r.time_range_end) {
            (Some(start), Some(end)) => Some(Timestamp(start)..Timestamp(end)),
            _ => None,
        };
        let col = get_collection(&self.collections, &r.collection).await?;
        let idx = col.read().await;
        let results = idx.search(&r.query, k, ef, |_, d| d, time_range, top_k_quickselect);
        let hits = results
            .into_iter()
            .map(|(vector_id, distance)| SearchHit {
                vector_id,
                distance,
            })
            .collect();
        Ok(Response::new(SearchResponse { hits }))
    }

    async fn swap_bucket_out(
        &self,
        req: Request<SwapBucketOutRequest>,
    ) -> Result<Response<SwapBucketOutResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let local_dir = self.bucket_local_dir(&r.collection, r.bucket_seq);
        std::fs::create_dir_all(&local_dir).map_err(internal)?;
        let seq = BucketSeq(r.bucket_seq);
        let found = col
            .write()
            .await
            .swap_bucket_out(seq, &local_dir)
            .map_err(internal)?;
        Ok(Response::new(SwapBucketOutResponse { found }))
    }

    async fn swap_bucket_in(
        &self,
        req: Request<SwapBucketInRequest>,
    ) -> Result<Response<SwapBucketInResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let seq = BucketSeq(r.bucket_seq);
        let found = col.write().await.swap_bucket_in(seq).map_err(internal)?;
        Ok(Response::new(SwapBucketInResponse { found }))
    }

    async fn evict_bucket(
        &self,
        req: Request<EvictBucketRequest>,
    ) -> Result<Response<EvictBucketResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let seq = BucketSeq(r.bucket_seq);
        let resp = match col.write().await.evict_bucket(seq) {
            Some(count) => EvictBucketResponse {
                found: true,
                evicted_count: count as u32,
            },
            None => EvictBucketResponse {
                found: false,
                evicted_count: 0,
            },
        };
        Ok(Response::new(resp))
    }

    async fn swap_bucket_out_to_blob(
        &self,
        req: Request<SwapBucketOutToBlobRequest>,
    ) -> Result<Response<SwapBucketOutToBlobResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("blob storage not configured"))?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let local_dir = self.bucket_local_dir(&r.collection, r.bucket_seq);
        let prefix = self.bucket_blob_prefix(&r.collection, r.bucket_seq);
        std::fs::create_dir_all(&local_dir).map_err(internal)?;
        let seq = BucketSeq(r.bucket_seq);

        // Swap to disk first (holds write lock only during disk I/O).
        // Capture write_count so we can use the version-aware mark_clean after upload.
        let (found, created_at_secs, write_count) = {
            let mut idx = col.write().await;
            let created_at = idx.bucket_created_at(seq).map(|t| t.0).unwrap_or(0);
            let wc = idx.bucket_write_count(seq);
            let found = idx.swap_bucket_out(seq, &local_dir).map_err(internal)?;
            (found, created_at, wc)
        };

        // Upload to blob storage without any lock — files are stable on disk.
        // Uses the same versioned staging format as the snapshot task so that
        // recovery can find the snapshot via bucket_meta.json.
        if found {
            let snap_subdir = format!(
                "snap_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis()
            );
            let staged = prefix.child(snap_subdir.as_str());
            upload_arena_dir(store.as_ref(), &local_dir, &staged)
                .await
                .map_err(internal)?;
            upload_levels(store.as_ref(), &local_dir.join("levels.bin"), &staged)
                .await
                .map_err(internal)?;
            upload_manifest(store.as_ref(), &local_dir.join("manifest.json"), &staged)
                .await
                .map_err(internal)?;
            upload_bucket_meta(
                store.as_ref(),
                &BucketMeta {
                    version: 1,
                    seq: r.bucket_seq,
                    created_at_secs,
                    snap_dir: snap_subdir,
                },
                &prefix,
            )
            .await
            .map_err(internal)?;

            // Clear dirty flags only if no new inserts arrived since we captured
            // write_count — same race-free semantics as the snapshot task.
            col.write()
                .await
                .mark_bucket_clean_if_version(seq, write_count);
        }
        Ok(Response::new(SwapBucketOutToBlobResponse { found }))
    }

    async fn swap_bucket_in_from_blob(
        &self,
        req: Request<SwapBucketInFromBlobRequest>,
    ) -> Result<Response<SwapBucketInFromBlobResponse>, Status> {
        let r = req.into_inner();
        validate_name(&r.collection)?;
        let store = self
            .store
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("blob storage not configured"))?;
        let col = get_collection(&self.collections, &r.collection).await?;
        let local_dir = self.bucket_local_dir(&r.collection, r.bucket_seq);
        let prefix = self.bucket_blob_prefix(&r.collection, r.bucket_seq);
        let seq = BucketSeq(r.bucket_seq);

        // Probe existence before downloading.
        {
            if !col.read().await.has_bucket(seq) {
                return Ok(Response::new(SwapBucketInFromBlobResponse {
                    found: false,
                    restored_count: 0,
                }));
            }
        }

        // Read bucket_meta.json to find the versioned snap_T/ subdirectory,
        // then download from there — consistent with how the snapshot task and
        // recovery code write and read blobs.
        let bucket_meta = download_bucket_meta(store.as_ref(), &prefix)
            .await
            .map_err(internal)?;
        let versioned = prefix.child(bucket_meta.snap_dir.as_str());

        std::fs::create_dir_all(&local_dir).map_err(internal)?;
        download_arena_dir(store.as_ref(), &versioned, &local_dir)
            .await
            .map_err(internal)?;
        download_levels(store.as_ref(), &versioned, &local_dir.join("levels.bin"))
            .await
            .map_err(internal)?;
        download_manifest(store.as_ref(), &versioned, &local_dir.join("manifest.json"))
            .await
            .map_err(internal)?;

        // Restore from local files (holds write lock only during disk I/O).
        let resp = match col
            .write()
            .await
            .restore_bucket_from_local(seq, &local_dir)
            .map_err(internal)?
        {
            Some(count) => SwapBucketInFromBlobResponse {
                found: true,
                restored_count: count as u32,
            },
            None => SwapBucketInFromBlobResponse {
                found: false,
                restored_count: 0,
            },
        };
        Ok(Response::new(resp))
    }
}
