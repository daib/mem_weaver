use crate::distance::euclidean_distance_sq;
use crate::eval::{recall_at_k, validate_recall_score};
use crate::{
    fvecs_vector_count, read_fvecs_dim_le, read_fvecs_vector_at, top_k_quickselect, VectorId,
};
use memmap2::Mmap;
use std::fs::File;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// SIFT base vectors are 128-D in standard Texmex SIFT1M.
const SIFT_DIM: usize = 128;
/// Default recall@k and minimum corpus size for the SIFT HNSW smoke test.
const K: usize = 10;

#[derive(Default)]
pub struct QueryPhaseTimings {
    pub brute_force_gt: Duration,
    pub hnsw_search: Duration,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Brute-force top-`k` by L2² for ground truth on the in-memory prefix.
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

pub struct SiftCtx {
    base_mmap: Mmap,
    q_mmap: Mmap,
    pub dim: usize,
    pub n_base: usize,
    pub n_q: usize,
    pub search_ef: usize,
}

impl SiftCtx {
    pub fn base_data(&self) -> &[u8] {
        &self.base_mmap[..]
    }

    pub fn q_data(&self) -> &[u8] {
        &self.q_mmap[..]
    }
}

pub fn try_load_sift_ctx(
    num_base_vectors: usize,
    num_queries: usize,
    default_search_ef: usize,
) -> Option<SiftCtx> {
    let base_dir = match std::env::var("SIFT1M_BASE_PATH") {
        Ok(s) => PathBuf::from(s),
        Err(_) => {
            eprintln!("SIFT1M_BASE_PATH not set; skipping HNSW SIFT recall test");
            return None;
        }
    };
    let base_fvecs = base_dir.join("sift_base.fvecs");
    let query_fvecs = base_dir.join("sift_query.fvecs");
    if !base_fvecs.is_file() || !query_fvecs.is_file() {
        eprintln!(
            "sift_base.fvecs and/or sift_query.fvecs missing under {:?}; skip",
            base_dir
        );
        return None;
    }

    let n_base_cfg: usize = std::env::var("SIFT1M_RECALL_N_BASE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_base_vectors);
    let n_queries: usize = std::env::var("SIFT1M_RECALL_N_QUERIES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(num_queries);
    let search_ef: usize = std::env::var("SIFT1M_HNSW_EF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_search_ef);

    let base_mmap = match open_mmap(&base_fvecs) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("open {}: {e}", base_fvecs.display());
            return None;
        }
    };
    let q_mmap = match open_mmap(&query_fvecs) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("open {}: {e}", query_fvecs.display());
            return None;
        }
    };

    let base_data: &[u8] = &base_mmap[..];

    let Some(dim) = read_fvecs_dim_le(base_data, 0) else {
        eprintln!("empty sift_base.fvecs");
        return None;
    };
    if dim != SIFT_DIM {
        eprintln!("unexpected base dim {dim}, expected {SIFT_DIM}");
        return None;
    }
    let avail = fvecs_vector_count(base_data, dim);
    if avail == 0 {
        eprintln!("no base vectors");
        return None;
    }
    let n_base = n_base_cfg.min(avail);
    if n_base < K {
        eprintln!("n_base {n_base} < k {K}; skip");
        return None;
    }

    let n_q_avail = fvecs_vector_count(&q_mmap[..], dim);
    let n_q = n_queries.min(n_q_avail);
    if n_q == 0 {
        eprintln!("no query vectors");
        return None;
    }

    Some(SiftCtx {
        base_mmap,
        q_mmap,
        dim,
        n_base,
        n_q,
        search_ef,
    })
}

pub fn sift_min_recall(
    label: &str,
    corpus: &[Vec<f32>],
    q_data: &[u8],
    dim: usize,
    n_q: usize,
    ef: usize,
    mut search: impl FnMut(&[f32]) -> Vec<VectorId>,
) -> (f32, Duration, QueryPhaseTimings) {
    let wall = Instant::now();
    let mut timings = QueryPhaseTimings::default();
    let mut min_recall = 1.0f32;
    for qi in 0..n_q {
        let q = read_fvecs_vector_at(q_data, dim, qi).expect("query fvecs");

        let t0 = Instant::now();
        let gt: Vec<VectorId> = exhaustive_l2_topk(
            &q,
            K,
            corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (VectorId(i as u64), v)),
        );
        timings.brute_force_gt += t0.elapsed();

        let t1 = Instant::now();
        let retrieved = search(&q);
        timings.hnsw_search += t1.elapsed();

        let r = recall_at_k(&retrieved, &gt).expect("valid recall@k");
        validate_recall_score(r).expect("in-range score");
        min_recall = min_recall.min(r);
        eprintln!("{label} query {qi}: recall@{K} = {r:.4} (ef={ef})");
    }
    let query_wall = wall.elapsed();
    eprintln!(
        "{label}: query phase wall {:.3} ms total | brute-force gt {:.3} ms | HNSW search {:.3} ms | {} queries | {:.3} ms/query avg (wall)",
        ms(query_wall),
        ms(timings.brute_force_gt),
        ms(timings.hnsw_search),
        n_q,
        ms(query_wall) / n_q.max(1) as f64,
    );
    (min_recall, query_wall, timings)
}
