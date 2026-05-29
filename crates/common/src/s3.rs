//! S3 helpers built on top of [`object_store`]. Gated behind the `s3` feature so
//! non-S3 crates (e.g. `vector`, `mem`) don't pay for the AWS HTTP stack.
//!
//! Two roles:
//! 1. **Build a client** — parse `~/.aws/credentials` for a named profile and produce
//!    an [`AmazonS3Builder`] or an `Arc<dyn ObjectStore>` ready to use.
//! 2. **Bucket / prefix admin** — ensure a bucket exists (via the `aws` CLI), and
//!    delete every object under a prefix.

use futures::StreamExt;
use object_store::{aws::AmazonS3Builder, path::Path as ObjectPath, ObjectStore};
use std::io;
use std::sync::Arc;

/// Parse `~/.aws/credentials` for `profile` and return an [`AmazonS3Builder`] populated
/// with the bucket, region, access key, secret key, and (if present) session token.
///
/// The parser is intentionally minimal: section header is `[profile]`; each line in the
/// section is `key = value`. Unknown keys are ignored. Errors:
/// - `NotFound` if `$HOME` is unset.
/// - I/O error if `~/.aws/credentials` cannot be read.
/// - `InvalidData` if the profile is missing `aws_access_key_id` or `aws_secret_access_key`.
pub fn builder_from_profile(
    profile: &str,
    bucket: &str,
    region: &str,
) -> io::Result<AmazonS3Builder> {
    let home =
        std::env::var("HOME").map_err(|_| io::Error::new(io::ErrorKind::NotFound, "HOME unset"))?;
    let creds_path = format!("{home}/.aws/credentials");
    let body = std::fs::read_to_string(&creds_path)?;

    let mut in_section = false;
    let mut key: Option<String> = None;
    let mut secret: Option<String> = None;
    let mut token: Option<String> = None;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_section = &trimmed[1..trimmed.len() - 1] == profile;
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((k, v)) = trimmed.split_once('=') else {
            continue;
        };
        match k.trim() {
            "aws_access_key_id" => key = Some(v.trim().to_owned()),
            "aws_secret_access_key" => secret = Some(v.trim().to_owned()),
            "aws_session_token" => token = Some(v.trim().to_owned()),
            _ => {}
        }
    }
    let key = key.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no aws_access_key_id for profile [{profile}] in {creds_path}"),
        )
    })?;
    let secret = secret.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("no aws_secret_access_key for profile [{profile}] in {creds_path}"),
        )
    })?;
    let mut b = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region(region)
        .with_access_key_id(key)
        .with_secret_access_key(secret);
    if let Some(t) = token {
        b = b.with_token(t);
    }
    Ok(b)
}

/// One-shot: [`builder_from_profile`] + `.build()` + `Arc::new`.
pub fn build_store(profile: &str, bucket: &str, region: &str) -> io::Result<Arc<dyn ObjectStore>> {
    let b = builder_from_profile(profile, bucket, region)?;
    Ok(Arc::new(
        b.build()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?,
    ))
}

/// Ensure `bucket` exists in `region` for `profile`. Probes via `aws s3api head-bucket`;
/// if missing, creates with `aws s3api create-bucket`. Requires the AWS CLI on `PATH`.
///
/// Treats `BucketAlreadyOwnedByYou` from `create-bucket` as success (race-safe).
pub fn ensure_bucket(bucket: &str, region: &str, profile: &str) -> io::Result<()> {
    let head = std::process::Command::new("aws")
        .args([
            "s3api",
            "head-bucket",
            "--bucket",
            bucket,
            "--profile",
            profile,
        ])
        .output()
        .map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    "aws CLI not found on PATH; install awscli to auto-create the bucket",
                )
            } else {
                e
            }
        })?;
    if head.status.success() {
        return Ok(());
    }

    eprintln!("bucket {bucket:?} not found; creating in region {region}");
    let mut cmd = std::process::Command::new("aws");
    cmd.args([
        "s3api",
        "create-bucket",
        "--bucket",
        bucket,
        "--profile",
        profile,
        "--region",
        region,
    ]);
    // us-east-1 rejects --create-bucket-configuration; every other region requires it.
    if region != "us-east-1" {
        cmd.args([
            "--create-bucket-configuration",
            &format!("LocationConstraint={region}"),
        ]);
    }
    let out = cmd.output()?;
    if out.status.success() {
        eprintln!("created bucket {bucket:?} in {region}");
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    if stderr.contains("BucketAlreadyOwnedByYou") {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::Other,
        format!("aws s3api create-bucket failed: {stderr}"),
    ))
}

/// List every object under `prefix` in `store` and delete it. Errors on individual
/// deletes are swallowed — callers use this for best-effort cleanup where partial
/// failure is acceptable.
pub async fn delete_prefix(store: &dyn ObjectStore, prefix: &ObjectPath) {
    let mut s = store.list(Some(prefix));
    while let Some(meta) = s.next().await {
        if let Ok(meta) = meta {
            let _ = store.delete(&meta.location).await;
        }
    }
}
