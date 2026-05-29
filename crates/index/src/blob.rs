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
