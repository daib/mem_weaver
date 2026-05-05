//! Recall@k of [`index::HnswNaive`] and [`index::HnswArena`] vs exhaustive L2 on a SIFT1M *prefix*
//! when the dataset is on disk.
//!
//! # Environment
//!
//! - `SIFT1M_BASE_PATH` — directory with `sift_base.fvecs` and `sift_query.fvecs` (Texmex layout). If
//!   unset, the test returns immediately.
//! - `SIFT1M_RECALL_N_BASE` — number of base vectors to load and index (default `8192`).
//! - `SIFT1M_RECALL_N_QUERIES` — number of queries to evaluate (default `10`).
//! - `SIFT1M_HNSW_EF` — search `ef` at level 0 (default `256`; raise if recall is low on large N).

use common::benchmark::{sift_min_recall, try_load_sift_ctx};
use common::top_k_quickselect;
use index::{HnswArena, HnswIndex, HnswNaive, NodeId, DEFAULT_ALIGNMENT};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use vector::{read_fvecs_vector_at, VectorId};

const K: usize = 10;
const M: usize = 16;
const M_MAX0: usize = 32;

const RNG_SEED: u64 = 0x_4853_4E57_5F53_4954;

const DEFAULT_NUM_BASE_VECTORS: usize = 10_000;
const DEFAULT_NUM_QUERIES: usize = 10;
const DEFAULT_EF_CONSTRUCTION: usize = 200;
const DEFAULT_SEARCH_EF: usize = 100;

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
fn sift1m_hnsw_recall_vs_bruteforce() {
    let t_load = Instant::now();
    let Some(ctx) = try_load_sift_ctx(
        DEFAULT_NUM_BASE_VECTORS,
        DEFAULT_NUM_QUERIES,
        DEFAULT_SEARCH_EF,
    ) else {
        return;
    };
    eprintln!(
        "sift: load context (mmap base + query, validation) {:.3} ms",
        ms(t_load.elapsed())
    );

    let base_data = ctx.base_data();
    let q_data = ctx.q_data();
    let dim = ctx.dim;
    let n_base = ctx.n_base;
    let n_q = ctx.n_q;
    let ef = ctx.search_ef.max(K);

    let t_corpus = Instant::now();
    let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(n_base);
    for i in 0..n_base {
        corpus.push(read_fvecs_vector_at(base_data, dim, i).expect("uniform fvecs"));
    }
    eprintln!(
        "sift: decode corpus into Vec<Vec<f32>> (n_base={n_base}, dim={dim}) {:.3} ms",
        ms(t_corpus.elapsed())
    );

    let ef_construction = std::env::var("SIFT1M_HNSW_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EF_CONSTRUCTION);

    eprintln!(
        "sift_hnsw_recall: dim={dim} n_base={n_base} n_q={n_q} k={K} m={M} m_max0={M_MAX0} ef_search={ef} ef_construction={ef_construction} alignment={DEFAULT_ALIGNMENT} rng_seed={RNG_SEED}"
    );

    /// Pair of [`HnswIndex`] with a logging label (cannot use anonymous struct syntax in `Vec<...>`).
    struct LabeledIndex {
        label: &'static str,
        index: Box<dyn HnswIndex>,
        skip: bool,
    }

    let test_cases: Vec<LabeledIndex> = vec![
        LabeledIndex {
            label: "naive",
            index: Box::new(HnswNaive::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                top_k_quickselect,
            )),
            skip: false,
        },
        LabeledIndex {
            label: "arena",
            index: Box::new(HnswArena::new(
                dim,
                M,
                M_MAX0,
                ef_construction,
                n_base,
                top_k_quickselect,
            )),
            skip: false,
        },
    ];

    for test_case in test_cases {
        if test_case.skip {
            continue;
        }
        let label = test_case.label;
        let mut index = test_case.index;
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        // Ground truth uses corpus row indices as VectorId. Arena [`NodeId`]s are block-encoded,
        // not 0..n-1 — map each insert's returned id → corpus index for recall comparison.
        let mut graph_id_to_corpus: HashMap<NodeId, usize> = HashMap::with_capacity(n_base);

        let t_idx = Instant::now();
        let mut batch_start = Instant::now();
        for (i, v) in corpus.iter().enumerate() {
            let nid = index.insert(v.as_slice(), &mut rng);
            graph_id_to_corpus.insert(nid, i);
            if (i + 1) % 10_000 == 0 {
                eprintln!(
                    "{label}: inserted [{}, {}) 10_000 vectors in {:.3} ms (cumulative build {:.3} ms)",
                    i + 1 - 10_000,
                    i + 1,
                    ms(batch_start.elapsed()),
                    ms(t_idx.elapsed())
                );
                batch_start = Instant::now();
            } else if i + 1 == n_base && (n_base % 10_000 != 0) {
                let n = n_base % 10_000;
                eprintln!(
                    "{label}: inserted [{}, {}) {n} vectors in {:.3} ms (cumulative build {:.3} ms)",
                    n_base - n,
                    n_base,
                    ms(batch_start.elapsed()),
                    ms(t_idx.elapsed())
                );
            }
        }
        eprintln!(
            "{label}: index build (insert n={n_base}) {:.3} ms",
            ms(t_idx.elapsed())
        );
        eprintln!("{label}: done with indexing");
        assert_eq!(
            index.len(),
            n_base,
            "{label}: HNSW should index every base vector"
        );
        let (min_recall, _, _) = sift_min_recall(label, &corpus, q_data, dim, n_q, ef, |q| {
            index
                .search(q, K, ef)
                .iter()
                .map(|(nid, _)| {
                    let corpus_row = graph_id_to_corpus[nid];
                    VectorId(corpus_row as u64)
                })
                .collect()
        });
        assert!(
            min_recall >= 0.75,
            "{label}: minimum recall@{K} vs brute force expected >= 0.75, got {min_recall} (try SIFT1M_HNSW_EF=512 or lower SIFT1M_RECALL_N_BASE)"
        );
    }
}
