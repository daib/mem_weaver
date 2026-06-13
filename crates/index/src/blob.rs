//! Upload arena files written by [`crate::ArenaNodeStore::swap_out`] (and the
//! per-bucket variant in [`crate::TimeBucketIndex::swap_bucket_out`]) to any
//! [`object_store::ObjectStore`] backend (S3, GCS, Azure, local file, in-memory).
//!
//! `swap_out` produces files named `block_<idx>.arena` in a directory. The helpers
//! here walk that directory and stream each file to `<prefix>/block_<idx>.arena`
//! in the destination store. Blocks are ~2 MiB (one [`common::DEFAULT_ARENA_CAPACITY`]
//! arena rounded to a page) so each upload fits in a single PUT — no multipart.

use std::io;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use object_store::{path::Path as ObjectPath, ObjectStore, PutPayload};

const ARENA_EXT: &str = "arena";
const LEVELS_FILE: &str = "levels.bin";
const MANIFEST_FILE: &str = "manifest.json";
const COLLECTION_META_FILE: &str = "collection.json";
const BUCKET_META_FILE: &str = "bucket_meta.json";
const CATALOG_FILE: &str = "catalog.json";

/// Configuration snapshot for a collection, stored at `<prefix>/collection.json`.
/// Written once per snapshot cycle and read during crash recovery to recreate the
/// [`crate::TimeBucketIndex`] with the correct parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollectionMeta {
    pub version: u32,
    pub dim: usize,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub bucket_duration_secs: u64,
    /// Unix timestamp (seconds) when this snapshot was written.
    pub snapshot_at_secs: u64,
}

/// Per-bucket metadata stored at `<prefix>/seq_<N>/bucket_meta.json`.
///
/// Acts as the atomic commit pointer for a bucket snapshot. The actual arena
/// files live in `<prefix>/seq_<N>/<snap_dir>/` rather than directly in
/// `seq_<N>/`, so that a new upload can be staged without overwriting the
/// current complete snapshot. Only after all files are uploaded is
/// `bucket_meta.json` updated to point to the new `snap_dir`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BucketMeta {
    pub version: u32,
    pub seq: u32,
    pub created_at_secs: u64,
    /// Subdirectory under `seq_<N>/` that holds this snapshot's arena files,
    /// `levels.bin`, and `manifest.json`. Format: `"snap_<unix_secs>"`.
    pub snap_dir: String,
}

/// One entry in the [`Catalog`], carrying both the collection name and the index
/// configuration required to recreate a [`crate::TimeBucketIndex`] after a crash.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub dim: usize,
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub bucket_duration_secs: u64,
    /// Unix timestamp (seconds) when this catalog entry was last written.
    pub snapshot_at_secs: u64,
}

/// Registry of live collections, stored at `<prefix>/catalog.json`.
///
/// Written atomically on every `CreateCollection` (and `DeleteCollection`) so that
/// crash recovery can determine which collections were active at crash time without
/// blindly re-importing every S3 prefix, and can recreate each index with the
/// correct configuration without needing a prior snapshot to be present.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Catalog {
    pub version: u32,
    /// Sorted list of active collections with their index configurations.
    pub collections: Vec<CatalogEntry>,
}

/// Upload a [`Catalog`] as JSON to `<prefix>/catalog.json` in `store`.
/// Overwrites any existing catalog atomically (single PUT).
pub async fn upload_catalog(
    store: &dyn ObjectStore,
    catalog: &Catalog,
    prefix: &ObjectPath,
) -> io::Result<()> {
    let json =
        serde_json::to_vec(catalog).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let remote = prefix.child(CATALOG_FILE);
    store
        .put(&remote, PutPayload::from(Bytes::from(json)))
        .await
        .map_err(to_io)
        .map(|_| ())
}

/// Download `<prefix>/catalog.json` from `store` and deserialize it.
/// Returns `Err` with `ErrorKind::NotFound` if the catalog does not exist yet,
/// allowing callers to distinguish a missing catalog from other S3 errors.
pub async fn download_catalog(store: &dyn ObjectStore, prefix: &ObjectPath) -> io::Result<Catalog> {
    let remote = prefix.child(CATALOG_FILE);
    let get = store.get(&remote).await.map_err(|e| match e {
        object_store::Error::NotFound { .. } => io::Error::new(io::ErrorKind::NotFound, e),
        other => to_io(other),
    })?;
    let bytes = get.bytes().await.map_err(to_io)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Upload a [`CollectionMeta`] as JSON to `<prefix>/collection.json` in `store`.
pub async fn upload_collection_meta(
    store: &dyn ObjectStore,
    meta: &CollectionMeta,
    prefix: &ObjectPath,
) -> io::Result<()> {
    let json =
        serde_json::to_vec(meta).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let remote = prefix.child(COLLECTION_META_FILE);
    store
        .put(&remote, PutPayload::from(Bytes::from(json)))
        .await
        .map_err(to_io)
        .map(|_| ())
}

/// Download `<prefix>/collection.json` from `store` and deserialize it.
pub async fn download_collection_meta(
    store: &dyn ObjectStore,
    prefix: &ObjectPath,
) -> io::Result<CollectionMeta> {
    let remote = prefix.child(COLLECTION_META_FILE);
    let bytes = store
        .get(&remote)
        .await
        .map_err(to_io)?
        .bytes()
        .await
        .map_err(to_io)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Upload a [`BucketMeta`] as JSON to `<prefix>/bucket_meta.json` in `store`.
pub async fn upload_bucket_meta(
    store: &dyn ObjectStore,
    meta: &BucketMeta,
    prefix: &ObjectPath,
) -> io::Result<()> {
    let json =
        serde_json::to_vec(meta).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let remote = prefix.child(BUCKET_META_FILE);
    store
        .put(&remote, PutPayload::from(Bytes::from(json)))
        .await
        .map_err(to_io)
        .map(|_| ())
}

/// Download `<prefix>/bucket_meta.json` from `store` and deserialize it.
pub async fn download_bucket_meta(
    store: &dyn ObjectStore,
    prefix: &ObjectPath,
) -> io::Result<BucketMeta> {
    let remote = prefix.child(BUCKET_META_FILE);
    let bytes = store
        .get(&remote)
        .await
        .map_err(to_io)?
        .bytes()
        .await
        .map_err(to_io)?;
    serde_json::from_slice(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Delete every object whose key starts with `prefix` in `store`.
///
/// Used to remove stale bucket snapshots after they are superseded. Lists all
/// objects under `prefix`, then issues individual deletes. Returns the number of
/// objects deleted. Partial failures are returned as the first error; already
/// deleted objects are not retried.
pub async fn delete_prefix(store: &dyn ObjectStore, prefix: &ObjectPath) -> io::Result<usize> {
    let mut list = store.list(Some(prefix));
    let mut paths = Vec::new();
    while let Some(meta) = list.next().await {
        paths.push(meta.map_err(to_io)?.location);
    }
    let count = paths.len();
    for path in paths {
        store.delete(&path).await.map_err(to_io)?;
    }
    Ok(count)
}

/// Result of a single block upload: the source path and the destination object path.
#[derive(Debug, Clone)]
pub struct Uploaded {
    pub local: PathBuf,
    pub remote: ObjectPath,
}

/// Upload every `block_*.arena` file in `local_dir` to `<prefix>/<filename>` in `store`.
///
/// Files are uploaded concurrently. The returned vec lists every successfully uploaded
/// block (order is unspecified). Returns the first error encountered if any upload fails;
/// already-uploaded blocks are not rolled back.
pub async fn upload_arena_dir(
    store: &dyn ObjectStore,
    local_dir: &Path,
    prefix: &ObjectPath,
) -> io::Result<Vec<Uploaded>> {
    let entries = collect_arena_files(local_dir)?;

    let mut futs = FuturesUnordered::new();
    for local in entries {
        let file_name = local
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("non-utf8 arena file name in {}", local_dir.display()),
                )
            })?
            .to_owned();
        let remote = prefix.child(file_name);
        futs.push(upload_one(store, local, remote));
    }

    let mut out = Vec::with_capacity(futs.len());
    while let Some(res) = futs.next().await {
        out.push(res?);
    }
    Ok(out)
}

/// Download every `block_*.arena` object under `prefix` in `store` into `local_dir`,
/// preserving file names. Inverse of [`upload_arena_dir`]; useful as the source side
/// of a swap-in flow that pulls cold blocks back from blob storage before
/// [`crate::ArenaNodeStore::swap_in`].
pub async fn download_arena_dir(
    store: &dyn ObjectStore,
    prefix: &ObjectPath,
    local_dir: &Path,
) -> io::Result<Vec<PathBuf>> {
    std::fs::create_dir_all(local_dir)?;

    let mut list = store.list(Some(prefix));
    let mut targets = Vec::new();
    while let Some(meta) = list.next().await {
        let meta = meta.map_err(to_io)?;
        let name = match meta.location.filename() {
            Some(n) if n.ends_with(&format!(".{ARENA_EXT}")) => n.to_owned(),
            _ => continue,
        };
        targets.push((meta.location.clone(), local_dir.join(name)));
    }

    let mut futs = FuturesUnordered::new();
    for (remote, local) in targets {
        futs.push(download_one(store, remote, local));
    }

    let mut out = Vec::with_capacity(futs.len());
    while let Some(res) = futs.next().await {
        out.push(res?);
    }
    Ok(out)
}

/// Upload a local `levels.bin` file to `<prefix>/levels.bin` in `store`.
pub async fn upload_levels(
    store: &dyn ObjectStore,
    local: &Path,
    prefix: &ObjectPath,
) -> io::Result<Uploaded> {
    let remote = prefix.child(LEVELS_FILE);
    upload_one(store, local.to_owned(), remote).await
}

/// Download `<prefix>/levels.bin` from `store` to `local`.
pub async fn download_levels(
    store: &dyn ObjectStore,
    prefix: &ObjectPath,
    local: &Path,
) -> io::Result<()> {
    let remote = prefix.child(LEVELS_FILE);
    download_one(store, remote, local.to_owned())
        .await
        .map(|_| ())
}

/// Upload a local `manifest.json` file to `<prefix>/manifest.json` in `store`.
pub async fn upload_manifest(
    store: &dyn ObjectStore,
    local: &Path,
    prefix: &ObjectPath,
) -> io::Result<Uploaded> {
    let remote = prefix.child(MANIFEST_FILE);
    upload_one(store, local.to_owned(), remote).await
}

/// Download `<prefix>/manifest.json` from `store` to `local`.
pub async fn download_manifest(
    store: &dyn ObjectStore,
    prefix: &ObjectPath,
    local: &Path,
) -> io::Result<()> {
    let remote = prefix.child(MANIFEST_FILE);
    download_one(store, remote, local.to_owned())
        .await
        .map(|_| ())
}

async fn upload_one(
    store: &dyn ObjectStore,
    local: PathBuf,
    remote: ObjectPath,
) -> io::Result<Uploaded> {
    let bytes = tokio::fs::read(&local).await?;
    let payload = PutPayload::from(Bytes::from(bytes));
    store.put(&remote, payload).await.map_err(to_io)?;
    Ok(Uploaded { local, remote })
}

async fn download_one(
    store: &dyn ObjectStore,
    remote: ObjectPath,
    local: PathBuf,
) -> io::Result<PathBuf> {
    let get = store.get(&remote).await.map_err(to_io)?;
    let bytes = get.bytes().await.map_err(to_io)?;
    tokio::fs::write(&local, &bytes).await?;
    Ok(local)
}

fn collect_arena_files(local_dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(local_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some(ARENA_EXT) {
            continue;
        }
        if !entry.file_type()?.is_file() {
            continue;
        }
        out.push(path);
    }
    out.sort();
    Ok(out)
}

fn to_io(err: object_store::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use std::sync::Arc;

    fn temp_dir(tag: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "mem_weaver_blob_{}_{}_{}",
            tag,
            std::process::id(),
            n
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    struct DirGuard(PathBuf);
    impl Drop for DirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
    }

    #[tokio::test]
    async fn upload_then_download_roundtrips_block_files() {
        let src = temp_dir("src");
        let _g_src = DirGuard(src.clone());
        let dst = temp_dir("dst");
        let _g_dst = DirGuard(dst.clone());

        write(&src.join("block_0.arena"), b"alpha-payload");
        write(&src.join("block_1.arena"), b"beta-payload");
        // non-arena file must be ignored
        write(&src.join("README"), b"skip me");

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("buckets/seq_1");

        let uploaded = upload_arena_dir(store.as_ref(), &src, &prefix)
            .await
            .expect("upload");
        assert_eq!(
            uploaded.len(),
            2,
            "two arena files expected, README skipped"
        );

        let restored = download_arena_dir(store.as_ref(), &prefix, &dst)
            .await
            .expect("download");
        assert_eq!(restored.len(), 2);

        assert_eq!(
            std::fs::read(dst.join("block_0.arena")).unwrap(),
            b"alpha-payload"
        );
        assert_eq!(
            std::fs::read(dst.join("block_1.arena")).unwrap(),
            b"beta-payload"
        );
    }

    #[tokio::test]
    async fn upload_empty_dir_succeeds_with_zero_uploads() {
        let src = temp_dir("empty");
        let _g = DirGuard(src.clone());
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("buckets/empty");
        let uploaded = upload_arena_dir(store.as_ref(), &src, &prefix)
            .await
            .expect("upload");
        assert!(uploaded.is_empty());
    }

    #[tokio::test]
    async fn upload_download_levels_roundtrip() {
        use crate::HnswNaive;
        use common::top_k_quickselect;
        use rand::{rngs::StdRng, SeedableRng};

        let local = temp_dir("levels");
        let _g = DirGuard(local.clone());

        let mut idx = HnswNaive::new(4, 4, 8, 32, top_k_quickselect, StdRng::seed_from_u64(3));
        for i in 0..10 {
            idx.insert(&[i as f32, 0.0, 0.0, 0.0], i as u64);
        }
        let levels_path = local.join("levels.bin");
        idx.save_levels(&levels_path).expect("save_levels");

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("hnsw/run-1");

        upload_levels(store.as_ref(), &levels_path, &prefix)
            .await
            .expect("upload");

        let dst = temp_dir("levels_dst");
        let _g2 = DirGuard(dst.clone());
        let restored = dst.join("levels.bin");
        download_levels(store.as_ref(), &prefix, &restored)
            .await
            .expect("download");

        let mut idx2 = HnswNaive::new(4, 4, 8, 32, top_k_quickselect, StdRng::seed_from_u64(0));
        idx2.load_levels(&restored).expect("load_levels");
        assert_eq!(idx.node_ids, idx2.node_ids);
        assert_eq!(idx.levels, idx2.levels);
    }

    #[tokio::test]
    async fn upload_after_swap_out_pushes_real_arena_files() {
        use crate::{HnswArena, HnswIndex};
        use common::top_k_quickselect;
        use rand::{rngs::StdRng, SeedableRng};

        let local = temp_dir("hnsw_swap");
        let _g = DirGuard(local.clone());

        let mut idx = HnswArena::new(4, 4, 8, 32, 32, top_k_quickselect, StdRng::seed_from_u64(7));
        for i in 0..16 {
            let v = [i as f32, 0.0, 0.0, 0.0];
            idx.insert(&v, i as u64);
        }
        let moved = idx.swap_out(&local).expect("swap_out");
        assert!(moved >= 1, "at least one block must have been written");

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let prefix = ObjectPath::from("hnsw/run-1");
        let uploaded = upload_arena_dir(store.as_ref(), &local, &prefix)
            .await
            .expect("upload");
        assert_eq!(
            uploaded.len(),
            moved,
            "uploaded count must match swap_out block count"
        );

        for u in &uploaded {
            // remote path is <prefix>/<filename>
            let got = store
                .get(&u.remote)
                .await
                .expect("get")
                .bytes()
                .await
                .unwrap();
            let want = std::fs::read(&u.local).expect("local read");
            assert_eq!(got.as_ref(), want.as_slice(), "round-trip bytes mismatch");
        }
    }
}
