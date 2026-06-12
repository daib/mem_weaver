//! K8s + Helm integration test for the mem-weaver gRPC server.
//!
//! Deploys the chart via the `helm` CLI, waits for the deployment to become
//! ready, port-forwards, and exercises CreateCollection / BatchInsert / Search
//! including error-case assertions.
//!
//! Skip guard: set `MEMWEAVER_K8S_TEST=1` to opt in.
//!
//! Optional env vars:
//!   MEMWEAVER_IMAGE      image repository (default: mem-weaver)
//!   MEMWEAVER_IMAGE_TAG  image tag        (default: latest)
//!
//! Run:
//!   MEMWEAVER_K8S_TEST=1 cargo test --test k8s_deploy -- --nocapture

use std::collections::{BTreeMap, HashSet};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::benchmark::try_load_sift_ctx;
use common::read_fvecs_vector_at;
use grpc::proto::{
    mem_weaver_client::MemWeaverClient, BatchInsertRequest, CreateCollectionRequest,
    EvictBucketRequest, InsertItem, SearchRequest,
};
use tonic::transport::Channel;

const RELEASE_NAME: &str = "mw-itest";
const NAMESPACE: &str = "default";
const SERVER_PORT: u16 = 50051;
const DIM: usize = 128;
const K: u32 = 10;
const BATCH_SIZE: usize = 1_000;
const COLLECTION: &str = "integration-test";
const DEFAULT_N_BASE: usize = 8_192;
const DEFAULT_N_QUERIES: usize = 10;
const DEFAULT_EF: usize = 100;
const MIN_RECALL: f32 = 0.75;
// Maximum vectors kept in memory. When exceeded, the oldest bucket is evicted.
// Override with MEMWEAVER_MEM_LIMIT.
const DEFAULT_MEM_LIMIT: usize = 4_096;

fn chart_dir() -> String {
    format!("{}/../../k8s/mem-weaver", env!("CARGO_MANIFEST_DIR"))
}

fn workspace_root() -> String {
    format!("{}/../../", env!("CARGO_MANIFEST_DIR"))
}

// ── system helpers ─────────────────────────────────────────────────────────────

fn ensure_image(repo: &str, tag: &str) {
    let image_ref = format!("{repo}:{tag}");
    let present = Command::new("docker")
        .args(["image", "inspect", "--format", "{{.Id}}", &image_ref])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .expect("docker not in PATH");

    if present.status.success() && !present.stdout.trim_ascii().is_empty() {
        eprintln!("image {image_ref} already present");
        return;
    }

    eprintln!("building image {image_ref} from {} ...", workspace_root());
    let status = Command::new("docker")
        .args(["build", "-t", &image_ref, &workspace_root()])
        .status()
        .expect("docker build failed to start");
    assert!(status.success(), "docker build exited with {status}");
    eprintln!("image {image_ref} built");
}

struct HelmRelease;

impl HelmRelease {
    fn install(image_repo: &str, image_tag: &str) -> Self {
        let status = Command::new("helm")
            .args([
                "install",
                RELEASE_NAME,
                &chart_dir(),
                "--namespace",
                NAMESPACE,
                "--create-namespace",
                "--set",
                &format!("image.repository={image_repo}"),
                "--set",
                &format!("image.tag={image_tag}"),
                "--set",
                "image.pullPolicy=IfNotPresent",
                "--set",
                "persistence.enabled=false",
            ])
            .status()
            .expect("helm not in PATH");
        assert!(status.success(), "helm install failed with {status}");
        eprintln!("helm install: release {RELEASE_NAME} deployed");
        Self
    }
}

impl Drop for HelmRelease {
    fn drop(&mut self) {
        let status = Command::new("helm")
            .args(["uninstall", RELEASE_NAME, "--namespace", NAMESPACE])
            .status();
        match status {
            Ok(s) if s.success() => eprintln!("helm uninstall: {RELEASE_NAME} removed"),
            Ok(s) => eprintln!("helm uninstall exited with {s}"),
            Err(e) => eprintln!("helm uninstall error: {e}"),
        }
    }
}

fn wait_for_deployment(name: &str, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        let out = Command::new("kubectl")
            .args([
                "rollout",
                "status",
                &format!("deployment/{name}"),
                "--namespace",
                NAMESPACE,
                "--timeout=10s",
            ])
            .output()
            .expect("kubectl not in PATH");

        if out.status.success() {
            eprintln!("deployment {name} ready");
            return;
        }
        assert!(
            Instant::now() < deadline,
            "deployment {name} not ready after {timeout:?}"
        );
        std::thread::sleep(Duration::from_secs(2));
    }
}

struct PortForward {
    _child: Child,
    pub local_port: u16,
}

impl PortForward {
    fn start(deployment: &str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let local_port = listener.local_addr().unwrap().port();
        drop(listener);

        let child = Command::new("kubectl")
            .args([
                "port-forward",
                &format!("deployment/{deployment}"),
                &format!("{local_port}:{SERVER_PORT}"),
                "--namespace",
                NAMESPACE,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("kubectl port-forward failed to start");

        let pf = Self {
            _child: child,
            local_port,
        };

        // Wait until the local port accepts connections.
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if std::net::TcpStream::connect(("127.0.0.1", local_port)).is_ok() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "port {local_port} not reachable after 30s"
            );
            std::thread::sleep(Duration::from_millis(200));
        }
        eprintln!("port-forward ready: localhost:{local_port} -> pod:{SERVER_PORT}");
        pf
    }
}

impl Drop for PortForward {
    fn drop(&mut self) {
        // Child is killed on drop automatically via std::process::Child.
    }
}

// ── gRPC test logic ────────────────────────────────────────────────────────────

async fn run_error_tests(client: &mut MemWeaverClient<Channel>) {
    // Search on an unknown collection must fail.
    let err = client
        .search(SearchRequest {
            collection: "does-not-exist".into(),
            query: vec![0.0; DIM],
            k: K,
            ef: 50,
            ..Default::default()
        })
        .await;
    assert!(
        err.is_err(),
        "search unknown collection: expected error, got ok"
    );

    // Insert into an unknown collection must fail.
    let err = client
        .batch_insert(BatchInsertRequest {
            collection: "does-not-exist".into(),
            items: vec![InsertItem {
                vector: vec![0.0; DIM],
                timestamp: 1,
                vector_id: 0,
            }],
        })
        .await;
    assert!(
        err.is_err(),
        "insert unknown collection: expected error, got ok"
    );

    // Creating the same collection twice must fail.
    let dup_req = || CreateCollectionRequest {
        collection: "dup-test".into(),
        dim: DIM as u32,
        m: 16,
        m_max0: 32,
        ef_construction: 200,
        bucket_duration_secs: 0,
    };
    client
        .create_collection(dup_req())
        .await
        .expect("first CreateCollection");
    let err = client.create_collection(dup_req()).await;
    assert!(err.is_err(), "duplicate collection: expected error, got ok");

    // Inserting a vector with the wrong dimension must fail.
    client
        .create_collection(CreateCollectionRequest {
            collection: "wrong-dim-test".into(),
            dim: DIM as u32,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            bucket_duration_secs: 0,
        })
        .await
        .expect("CreateCollection for wrong-dim test");
    let err = client
        .batch_insert(BatchInsertRequest {
            collection: "wrong-dim-test".into(),
            items: vec![InsertItem {
                vector: vec![1.0], // 1 component instead of DIM
                timestamp: 1,
                vector_id: 0,
            }],
        })
        .await;
    assert!(err.is_err(), "wrong dim insert: expected error, got ok");

    eprintln!("error cases: all passed");
}

async fn run_sift1m(client: &mut MemWeaverClient<Channel>) {
    let Some(ctx) = try_load_sift_ctx(DEFAULT_N_BASE, DEFAULT_N_QUERIES, DEFAULT_EF) else {
        eprintln!("SIFT1M_BASE_PATH not set — skipping sift1m happy-path");
        return;
    };

    let mem_limit: usize = std::env::var("MEMWEAVER_MEM_LIMIT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MEM_LIMIT)
        .min(ctx.n_base);

    // Size buckets so roughly 4 buckets form before the limit is first hit.
    let vectors_per_bucket = ((mem_limit / 4) as u64).max(256);
    let ef = ctx.search_ef.max(K as usize);

    client
        .create_collection(CreateCollectionRequest {
            collection: COLLECTION.into(),
            dim: ctx.dim as u32,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            // timestamp == vector_id, so this partitions the corpus evenly.
            bucket_duration_secs: vectors_per_bucket,
        })
        .await
        .expect("CreateCollection");

    // Load full corpus into memory for ground-truth computation.
    let corpus: Vec<Vec<f32>> = (0..ctx.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), ctx.dim, i).expect("base fvecs"))
        .collect();

    // bucket_seq -> corpus indices of vectors in that bucket (for ground-truth filtering).
    let mut bucket_map: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    // corpus indices that have been evicted.
    let mut evicted_ids: HashSet<u64> = HashSet::new();
    let mut in_memory: usize = 0;
    let mut total_evicted: usize = 0;

    let t_insert = Instant::now();
    for (chunk_idx, chunk) in corpus.chunks(BATCH_SIZE).enumerate() {
        let base = chunk_idx * BATCH_SIZE;
        let items: Vec<InsertItem> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| InsertItem {
                vector: v.clone(),
                // Use vector_id as timestamp so vectors spread across time buckets.
                timestamp: (base + j) as u64,
                vector_id: (base + j) as u64,
            })
            .collect();

        let resp = client
            .batch_insert(BatchInsertRequest { collection: COLLECTION.into(), items })
            .await
            .expect("BatchInsert");

        for r in resp.into_inner().results {
            bucket_map.entry(r.bucket_seq).or_default().push(r.vector_id as usize);
            in_memory += 1;
        }

        // Evict oldest buckets while over the memory limit.
        while in_memory > mem_limit {
            let Some((&oldest_seq, _)) = bucket_map.first_key_value() else { break };
            let evict_resp = client
                .evict_bucket(EvictBucketRequest {
                    collection: COLLECTION.into(),
                    bucket_seq: oldest_seq,
                })
                .await
                .expect("EvictBucket");

            let server_count = evict_resp.into_inner().evicted_count as usize;
            let indices = bucket_map.remove(&oldest_seq).unwrap_or_default();
            for &idx in &indices {
                evicted_ids.insert(idx as u64);
            }
            in_memory = in_memory.saturating_sub(indices.len());
            total_evicted += indices.len();
            eprintln!(
                "evicted bucket seq={oldest_seq}: server={server_count} tracked={}  \
                 in_memory={in_memory}",
                indices.len()
            );
        }
    }

    eprintln!(
        "sift1m: inserted {} vectors in {:.1} ms  \
         in_memory={in_memory} evicted={total_evicted}",
        ctx.n_base,
        t_insert.elapsed().as_secs_f64() * 1e3,
    );

    // Eviction must have fired if n_base > mem_limit.
    if ctx.n_base > mem_limit {
        assert!(total_evicted > 0, "expected evictions for n_base={} mem_limit={mem_limit}", ctx.n_base);
    }

    // Build surviving corpus (only in-memory vectors, preserving original IDs).
    let surviving: Vec<(u64, &Vec<f32>)> = corpus
        .iter()
        .enumerate()
        .filter(|(i, _)| !evicted_ids.contains(&(*i as u64)))
        .map(|(i, v)| (i as u64, v))
        .collect();

    // Search all queries and verify:
    //   1. No evicted vector ID appears in results.
    //   2. Recall@K vs brute-force over surviving corpus meets MIN_RECALL.
    let mut recalls: Vec<f32> = Vec::with_capacity(ctx.n_q);
    for qi in 0..ctx.n_q {
        let query = read_fvecs_vector_at(ctx.q_data(), ctx.dim, qi).expect("query fvecs");

        // Brute-force ground truth restricted to surviving vectors.
        let mut scored: Vec<(u64, f32)> = surviving
            .iter()
            .map(|(id, v)| {
                let d: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                (*id, d)
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let gt: HashSet<u64> = scored.iter().take(K as usize).map(|(id, _)| *id).collect();

        let resp = client
            .search(SearchRequest {
                collection: COLLECTION.into(),
                query,
                k: K,
                ef: ef as u32,
                ..Default::default()
            })
            .await
            .expect("Search");

        let hits = resp.into_inner().hits;
        for h in &hits {
            assert!(
                !evicted_ids.contains(&h.vector_id),
                "evicted vector_id={} appeared in search results",
                h.vector_id
            );
        }

        let overlap = hits.iter().filter(|h| gt.contains(&h.vector_id)).count();
        recalls.push(overlap as f32 / K as f32);
    }

    let min_recall = recalls.iter().cloned().fold(f32::INFINITY, f32::min);
    let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    eprintln!("sift1m: recall@{K} min={min_recall:.4} mean={mean_recall:.4} — passed");

    assert!(
        min_recall >= MIN_RECALL,
        "k8s/sift1m: recall@{K} min={min_recall:.4} expected >= {MIN_RECALL}"
    );
}

// ── entry point ────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn test_helm_deploy_integration() {
    if std::env::var("MEMWEAVER_K8S_TEST").unwrap_or_default() != "1" {
        eprintln!("MEMWEAVER_K8S_TEST=1 not set — skipping k8s integration test");
        return;
    }

    let image_repo = std::env::var("MEMWEAVER_IMAGE").unwrap_or_else(|_| "mem-weaver".into());
    let image_tag = std::env::var("MEMWEAVER_IMAGE_TAG").unwrap_or_else(|_| "latest".into());

    ensure_image(&image_repo, &image_tag);

    let _release = HelmRelease::install(&image_repo, &image_tag);

    let deploy_name = format!("{RELEASE_NAME}-mem-weaver");
    wait_for_deployment(&deploy_name, Duration::from_secs(180));

    let pf = PortForward::start(&deploy_name);

    let mut client = MemWeaverClient::connect(format!("http://127.0.0.1:{}", pf.local_port))
        .await
        .expect("gRPC connect");

    run_error_tests(&mut client).await;
    run_sift1m(&mut client).await;

    eprintln!("all integration tests passed");
}
