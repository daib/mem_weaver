//! Shared helpers for integration tests. Both `tests/s3_roundtrip.rs` and
//! `tests/sift1m_time_bucket_recall.rs` reach the S3 plumbing through here via
//! `mod helpers;`.
//!
//! Production-grade S3 functions live in [`common::s3`]; this module re-exports
//! them and adds test-only helpers on top (env resolution with empty-string-as-unset,
//! Drop-safe cleanup that spawns its own runtime, unique run ids for prefixing).

#![allow(dead_code, unused_imports)] // each test binary uses a different subset

pub mod s3 {
    use object_store::{path::Path as ObjectPath, ObjectStore};
    use std::sync::Arc;

    // Re-export the production-grade primitives from `common::s3` so callers don't
    // need to know they live in two places.
    pub use common::s3::{build_store, builder_from_profile, delete_prefix, ensure_bucket};

    /// Resolve a string setting: `env_key` value if set *and non-empty*, else `default`.
    /// Empty env-string is treated as unset so users have an escape hatch
    /// (`MEM_WEAVER_S3_BUCKET="" cargo test ...`).
    pub fn resolve(env_key: &str, default: &str) -> String {
        std::env::var(env_key)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| default.to_string())
    }

    /// `<pid>_<unix_nanos>` — unique enough for concurrent test runs to avoid colliding
    /// under a shared S3 prefix.
    pub fn unique_run_id() -> String {
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{pid}_{nanos}")
    }

    /// Run [`delete_prefix`] to completion on a fresh OS thread with its own tokio runtime.
    /// Safe to call from inside an existing runtime (e.g. `Drop` impls on `#[tokio::test]`s)
    /// where `Handle::block_on` would panic with "cannot start a runtime from within a runtime".
    pub fn cleanup_prefix_on_thread(store: Arc<dyn ObjectStore>, prefix: ObjectPath) {
        let _ = std::thread::spawn(move || {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            rt.block_on(delete_prefix(store.as_ref(), &prefix));
        })
        .join();
    }
}
