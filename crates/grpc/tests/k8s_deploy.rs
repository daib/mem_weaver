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

use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use common::benchmark::{sift_recall_stats, try_load_sift_ctx};
use common::read_fvecs_vector_at;
use grpc::proto::{
    mem_weaver_client::MemWeaverClient, BatchInsertRequest, CreateCollectionRequest, InsertItem,
    SearchRequest,
};
use tonic::transport::Channel;
use vector::VectorId;

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

    let ef = ctx.search_ef.max(K as usize);

    client
        .create_collection(CreateCollectionRequest {
            collection: COLLECTION.into(),
            dim: ctx.dim as u32,
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            bucket_duration_secs: 0,
        })
        .await
        .expect("CreateCollection");

    // Load corpus into memory (needed for brute-force ground truth later).
    let corpus: Vec<Vec<f32>> = (0..ctx.n_base)
        .map(|i| read_fvecs_vector_at(ctx.base_data(), ctx.dim, i).expect("base fvecs"))
        .collect();

    // Insert in batches.
    let t_insert = Instant::now();
    for (chunk_idx, chunk) in corpus.chunks(BATCH_SIZE).enumerate() {
        let base = chunk_idx * BATCH_SIZE;
        let items: Vec<InsertItem> = chunk
            .iter()
            .enumerate()
            .map(|(j, v)| InsertItem {
                vector: v.clone(),
                timestamp: 0,
                vector_id: (base + j) as u64,
            })
            .collect();
        client
            .batch_insert(BatchInsertRequest {
                collection: COLLECTION.into(),
                items,
            })
            .await
            .expect("BatchInsert");
    }
    eprintln!(
        "sift1m: inserted {} vectors in {:.1} ms",
        ctx.n_base,
        t_insert.elapsed().as_secs_f64() * 1e3
    );

    // Search all queries, collect results for recall computation.
    let mut search_results: Vec<Vec<VectorId>> = Vec::with_capacity(ctx.n_q);
    for qi in 0..ctx.n_q {
        let query = read_fvecs_vector_at(ctx.q_data(), ctx.dim, qi).expect("query fvecs");
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
        let hits: Vec<VectorId> = resp
            .into_inner()
            .hits
            .into_iter()
            .map(|h| VectorId(h.vector_id))
            .collect();
        search_results.push(hits);
    }

    // Compute recall@K vs brute-force ground truth.
    let mut result_iter = search_results.into_iter();
    let (stats, _, _) = sift_recall_stats(
        "k8s/sift1m",
        &corpus,
        ctx.q_data(),
        ctx.dim,
        ctx.n_q,
        ef,
        |_q| result_iter.next().unwrap(),
    );

    assert!(
        stats.min >= MIN_RECALL,
        "k8s/sift1m: recall@{K} min={:.4} expected >= {MIN_RECALL}",
        stats.min
    );
    eprintln!(
        "sift1m: recall@{K} min={:.4} mean={:.4} p95={:.4} — passed",
        stats.min, stats.mean, stats.p95
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
