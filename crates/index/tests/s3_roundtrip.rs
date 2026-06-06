//! End-to-end test: build an HNSW index, swap arena blocks to disk, push them
//! to S3 via [`index::upload_arena_dir`], pull them back with
//! [`index::download_arena_dir`], and verify that querying the swapped-in index
//! returns the same results as before.
//!
//! Skips automatically when AWS credentials are not available. To run against a
//! real bucket:
//!
//! ```bash
//! MEM_WEAVER_S3_BUCKET=my-bucket \
//! MEM_WEAVER_S3_REGION=us-east-1 \
//! MEM_WEAVER_S3_PROFILE=default \
//! cargo test -p index --test s3_roundtrip -- --nocapture
//! ```
//!
//! Credentials are read from `~/.aws/credentials` for the named profile (default `"default"`).
//! `MEM_WEAVER_S3_PREFIX` is optional; a unique per-run prefix is generated when unset.
//! The test cleans up every object it uploads on success.

mod helpers;

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::top_k_quickselect;
use futures::stream::StreamExt;
use index::{download_arena_dir, upload_arena_dir, HnswArena, HnswIndex};
use object_store::{path::Path as ObjectPath, ObjectStore};
use rand::{rngs::StdRng, Rng, SeedableRng};

const DIM: usize = 32;
const N: usize = 256;
const K: usize = 10;
const EF: usize = 64;

// ── Default test config ─────────────────────────────────────────────────────
// Edit these to your dev bucket so you can run the test without exporting
// env vars. Env vars (MEM_WEAVER_S3_*) still override the defaults.
//
// While `DEFAULT_BUCKET` equals the placeholder below, the test skips with a
// "configure me" message — that keeps unedited checkouts from trying to talk
// to a bucket that doesn't exist.
const DEFAULT_BUCKET: &str = "mem-weaver-test";
const DEFAULT_REGION: &str = "us-east-1";
const DEFAULT_PROFILE: &str = "default";
/// Optional explicit prefix; when empty a unique per-run prefix is generated.
const DEFAULT_PREFIX: &str = "dev";

#[tokio::test]
async fn upload_arena_to_s3_then_download_and_query() {
    let cfg = match TestConfig::from_env() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("skipping: {msg}");
            return;
        }
    };

    helpers::s3::ensure_bucket(&cfg.bucket, &cfg.region, &cfg.profile)
        .expect("ensure bucket exists");

    let store: Arc<dyn ObjectStore> =
        helpers::s3::build_store(&cfg.profile, &cfg.bucket, &cfg.region)
            .expect("build S3 client from ~/.aws/credentials");

    // 1. Build an in-memory HNSW with deterministic data and capture a query result.
    let mut idx: HnswArena = HnswArena::new(
        DIM,
        16,
        32,
        EF,
        N, // one block per arena page is plenty for N=256
        top_k_quickselect,
        StdRng::seed_from_u64(0xABCD_1234),
    );
    let mut rng = StdRng::seed_from_u64(0x7777);
    for i in 0..N {
        let v: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
        idx.insert(&v, i as u64);
    }
    let query: Vec<f32> = (0..DIM).map(|_| rng.gen::<f32>()).collect();
    let before: Vec<(u64, f32)> = idx.search(&query, K, EF);
    assert_eq!(before.len(), K, "baseline query must return K hits");

    // 2. Swap arena blocks to a local dir.
    let local_out = TempDir::new("s3_roundtrip_out");
    let moved = idx.swap_out(local_out.path()).expect("swap_out");
    assert!(moved >= 1, "swap_out must produce at least one arena file");
    let local_files = list_arena_files(local_out.path());
    assert_eq!(local_files.len(), moved, "file count matches block count");
    eprintln!(
        "swap_out produced {moved} arena file(s) in {:?}",
        local_out.path()
    );

    // 3. Upload to S3 under a unique prefix.
    let uploaded = upload_arena_dir(store.as_ref(), local_out.path(), &cfg.prefix)
        .await
        .expect("upload_arena_dir");
    assert_eq!(uploaded.len(), moved, "every block uploaded");
    eprintln!(
        "uploaded {} block(s) to s3://{}/{}",
        uploaded.len(),
        cfg.bucket,
        cfg.prefix
    );

    // Cleanup uploaded objects on every exit path beyond this point.
    let cleanup = Cleanup {
        store: Arc::clone(&store),
        prefix: cfg.prefix.clone(),
    };

    // 4. Verify the list under the prefix matches.
    let listed = list_prefix(store.as_ref(), &cfg.prefix)
        .await
        .expect("list");
    assert_eq!(
        listed.len(),
        uploaded.len(),
        "S3 list count must match uploaded count: {listed:?}"
    );

    // 5. Download to a fresh dir, byte-compare to originals.
    let local_in = TempDir::new("s3_roundtrip_in");
    let restored = download_arena_dir(store.as_ref(), &cfg.prefix, local_in.path())
        .await
        .expect("download_arena_dir");
    assert_eq!(restored.len(), moved, "every block downloaded");

    for src in &local_files {
        let name = src.file_name().expect("name");
        let dst = local_in.path().join(name);
        let a = std::fs::read(src).expect("read src");
        let b = std::fs::read(&dst).expect("read dst");
        assert_eq!(a, b, "downloaded {:?} differs from original", name);
    }

    // 6. Restore the index from disk (uses the original fds from swap_out — still
    //    live) and run the same query. Results must match the in-memory baseline.
    let restored_count = idx.swap_in().expect("swap_in");
    assert_eq!(restored_count, moved, "swap_in restores every block");
    let after = idx.search(&query, K, EF);
    assert_eq!(
        after, before,
        "query results must be identical after swap_out → upload → download → swap_in"
    );

    drop(cleanup);
}

// ── helpers ─────────────────────────────────────────────────────────────────

struct TestConfig {
    bucket: String,
    region: String,
    profile: String,
    prefix: ObjectPath,
}

impl TestConfig {
    fn from_env() -> Result<Self, String> {
        let bucket =
            std::env::var("MEM_WEAVER_S3_BUCKET").unwrap_or_else(|_| DEFAULT_BUCKET.into());
        if bucket == "<edit-me>" {
            return Err(
                "set MEM_WEAVER_S3_BUCKET or edit DEFAULT_BUCKET in tests/s3_roundtrip.rs".into(),
            );
        }
        if let Some((head, tail)) = bucket.split_once('/') {
            return Err(format!(
                "bucket name {bucket:?} contains '/'. S3 bucket names cannot contain slashes — \
                 the slash you wrote is the object-key delimiter, not a bucket separator. \
                 Set DEFAULT_BUCKET={head:?} (or {tail:?}) and DEFAULT_PREFIX={:?} instead.",
                if head == "dev" || head == "stage" || head == "prod" {
                    head
                } else {
                    tail
                }
            ));
        }
        let region =
            std::env::var("MEM_WEAVER_S3_REGION").unwrap_or_else(|_| DEFAULT_REGION.into());
        let profile =
            std::env::var("MEM_WEAVER_S3_PROFILE").unwrap_or_else(|_| DEFAULT_PROFILE.into());

        // Skip gracefully when AWS credentials are not available.
        if let Err(e) = helpers::s3::builder_from_profile(&profile, &bucket, &region) {
            return Err(format!("credentials not available ({e}); skipping S3 test"));
        }

        let prefix = std::env::var("MEM_WEAVER_S3_PREFIX")
            .ok()
            .or_else(|| (!DEFAULT_PREFIX.is_empty()).then(|| DEFAULT_PREFIX.to_string()))
            .unwrap_or_else(unique_prefix);
        Ok(Self {
            bucket,
            region,
            profile,
            prefix: ObjectPath::from(prefix),
        })
    }
}

fn unique_prefix() -> String {
    format!("mem_weaver_test/{}", helpers::s3::unique_run_id())
}

fn list_arena_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read_dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension().and_then(|s| s.to_str()) == Some("arena")).then_some(p)
        })
        .collect();
    out.sort();
    out
}

async fn list_prefix(store: &dyn ObjectStore, prefix: &ObjectPath) -> io::Result<Vec<ObjectPath>> {
    let mut s = store.list(Some(prefix));
    let mut out = Vec::new();
    while let Some(m) = s.next().await {
        out.push(
            m.map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
                .location,
        );
    }
    Ok(out)
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let p = std::env::temp_dir().join(format!(
            "mw_s3_{}_{}_{}_{}",
            tag,
            std::process::id(),
            nanos,
            n
        ));
        std::fs::create_dir_all(&p).expect("mk tempdir");
        Self(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Best-effort cleanup of uploaded objects. Runs on test success and on panic.
struct Cleanup {
    store: Arc<dyn ObjectStore>,
    prefix: ObjectPath,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        // Can't reuse the test's runtime — `Drop` may fire mid-poll on panic, and
        // `Handle::block_on` from inside an existing runtime panics. The helper spawns a
        // dedicated thread + fresh runtime so cleanup is independent of caller state.
        helpers::s3::cleanup_prefix_on_thread(Arc::clone(&self.store), self.prefix.clone());
    }
}
