use common::benchmark::{sift_recall_stats, try_load_sift_ctx};
use common::distance::euclidean_distance_sq;
use common::{
    import_fvecs, read_fvecs_vector_at, top_k_quickselect, VectorId, DEFAULT_ARENA_CAPACITY,
};
use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use vector::{recall_at_k, validate_recall_score, VectorStore};

fn read_i32_le(data: &[u8], off: usize) -> Option<i32> {
    let b = data.get(off..off + 4)?;
    Some(i32::from_le_bytes(b.try_into().ok()?))
}

/// One record from a Texmex `.ivecs` file: leading `i32` count, then that many `i32` values.
fn read_ivecs_record(data: &[u8], rec_index: usize) -> Option<Vec<i32>> {
    let mut off = 0usize;
    for _ in 0..rec_index {
        let d = read_i32_le(data, off)? as usize;
        off = off.checked_add(4)?.checked_add(4usize.checked_mul(d)?)?;
        if off > data.len() {
            return None;
        }
    }
    let d = read_i32_le(data, off)? as usize;
    off += 4;
    let end = off.checked_add(4usize.checked_mul(d)?)?;
    if end > data.len() {
        return None;
    }
    let mut v = Vec::with_capacity(d);
    for j in 0..d {
        v.push(read_i32_le(data, off + 4 * j)?);
    }
    Some(v)
}

/// Brute-force top-`k` by L2² — same selection rule as [`common::benchmark::sift_min_recall`] / HNSW tests (`top_k_quickselect`).
fn exhaustive_l2_topk<ID: Ord + Copy, V: AsRef<[f32]>>(
    query: &[f32],
    k: usize,
    corpus: impl IntoIterator<Item = (ID, V)>,
) -> Vec<ID> {
    let scored: Vec<(ID, f32)> = corpus
        .into_iter()
        .map(|(id, v)| (id, euclidean_distance_sq(query, v.as_ref())))
        .collect();
    top_k_quickselect(&scored, k)
}

fn open_mmap(path: &PathBuf) -> std::io::Result<Mmap> {
    let file = File::open(path)?;
    unsafe { Mmap::map(&file) }
}

#[test]
fn sift1m_recall_exhaustive_nn() {
    const K: usize = 10;
    const DEFAULT_N_BASE: usize = 100_000;
    const DEFAULT_N_QUERIES: usize = 50;

    let Some(ctx) = try_load_sift_ctx(DEFAULT_N_BASE, DEFAULT_N_QUERIES, 100) else {
        return;
    };

    let base_data = ctx.base_data();
    let q_data = ctx.q_data();
    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let n_q = ctx.n_q;
    let ef_log = ctx.search_ef.max(K);

    let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(n_base);
    for i in 0..n_base {
        corpus.push(read_fvecs_vector_at(base_data, dim, i).expect("uniform fvecs"));
    }

    let mut store = VectorStore::new(dim, DEFAULT_ARENA_CAPACITY);
    let loaded = import_fvecs(base_data, dim, n_base, |i, buf| {
        store.insert(VectorId(i), buf).is_some()
    });
    assert_eq!(
        loaded, n_base,
        "import into VectorStore should load the full SIFT base prefix"
    );
    assert_eq!(store.num_vectors(), n_base);

    let (stats, _, _) =
        sift_recall_stats("exhaustive_nn", &corpus, q_data, dim, n_q, ef_log, |q| {
            exhaustive_l2_topk(q, K, store.iter())
        });
    assert_eq!(
        stats.min, 1.0,
        "exhaustive top-k over VectorStore must match brute-force GT on corpus rows"
    );

    for qi in 0..n_q {
        let q = read_fvecs_vector_at(q_data, dim, qi).expect("uniform query fvecs");
        let ground_truth = exhaustive_l2_topk(
            &q,
            K,
            corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (VectorId(i as u64), v)),
        );

        // Intersection: half the GT ids wrong → recall@10 = 0.5
        let mut half_wrong: Vec<VectorId> = ground_truth
            .iter()
            .take(5)
            .copied()
            .chain((0u64..5).map(|i| VectorId(900_000 + i + (qi * 5) as u64)))
            .collect();
        half_wrong.truncate(10);
        let r2 = recall_at_k(&half_wrong, &ground_truth).expect("valid recall");
        assert!(
            (r2 - 0.5).abs() < 1e-4,
            "query {qi}: expected recall 0.5, got {r2}"
        );
        validate_recall_score(r2).unwrap();
    }
}

#[test]
fn sift1m_ground_truth_ivecs_parsing_and_recall() {
    let base_dir = match std::env::var("SIFT1M_BASE_PATH") {
        Ok(s) => PathBuf::from(s),
        Err(_) => {
            eprintln!("SIFT1M_BASE_PATH not set; skipping ivecs test");
            return;
        }
    };
    let path = base_dir.join("sift_groundtruth.ivecs");
    if !path.is_file() {
        eprintln!("{} missing; skip ivecs test", path.display());
        return;
    }

    let gt_mmap = match open_mmap(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("open groundtruth: {e}");
            return;
        }
    };
    let data: &[u8] = &gt_mmap[..];
    // First query, first 10 neighbor ids in the 1M benchmark
    let row0 = read_ivecs_record(data, 0).expect("at least one ivecs record");
    assert!(
        row0.len() >= 10,
        "expected ≥10 ground truth neighbors, got {}",
        row0.len()
    );
    let as_ids: Vec<VectorId> = row0.iter().map(|&x| VectorId(x as u64)).collect();
    let prefix: Vec<VectorId> = as_ids.iter().take(10).copied().collect();
    let r = recall_at_k(&prefix, &prefix).expect("ok");
    assert_eq!(r, 1.0, "self-recall on ground-truth slice must be 1.0");
    validate_recall_score(r).unwrap();

    for &x in row0.iter().take(10) {
        assert!(
            x >= 0 && (x as u64) < 1_000_000,
            "SIFT1M neighbor id {x} out of expected base range"
        );
    }
}
