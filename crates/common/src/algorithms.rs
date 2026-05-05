use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Total-order wrapper for `f32` using [`f32::total_cmp`] (e.g. binary heaps of distances).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrdF32(pub f32);

impl Eq for OrdF32 {}

impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.total_cmp(&other.0))
    }
}

impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl OrdF32 {
    #[must_use]
    #[inline]
    pub fn inner(self) -> f32 {
        self.0
    }
}

#[inline]
fn cmp_pair<T: Ord>(a: &(T, f32), b: &(T, f32)) -> Ordering {
    OrdF32(a.1).cmp(&OrdF32(b.1)).then_with(|| a.0.cmp(&b.0))
}

/// Retains the `k` pairs with smallest distance (`f32`), breaking ties by [`Ord`] on `T`.
pub fn top_k_heap<T: Ord + Copy>(candidates: &[(T, f32)], k: usize) -> Vec<T> {
    let mut heap = BinaryHeap::with_capacity(k + 1);
    for &(id, dist) in candidates {
        heap.push((OrdF32(dist), id));
        if heap.len() > k {
            heap.pop(); // remove worst (largest distance)
        }
    }
    heap.into_vec().into_iter().map(|(_, id)| id).collect()
}

pub fn top_k_sort<T: Ord + Copy>(candidates: &[(T, f32)], k: usize) -> Vec<T> {
    let mut pairs: Vec<(T, f32)> = candidates.to_vec();
    pairs.sort_by(|a, b| cmp_pair(a, b));
    pairs.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Partitions `v[lo..=hi]` so elements `<=` pivot (by [`cmp_pair`]) land left of the returned index.
fn partition<T: Ord + Copy>(v: &mut [(T, f32)], lo: usize, hi: usize) -> usize {
    let pivot = v[hi];
    let mut i = lo;
    for j in lo..hi {
        if cmp_pair(&v[j], &pivot) != Ordering::Greater {
            v.swap(i, j);
            i += 1;
        }
    }
    v.swap(i, hi);
    i
}

/// Rearranges `v` so the first `k` elements are exactly the `k` smallest by [`cmp_pair`] (unordered among themselves).
fn quickselect_smallest_k<T: Ord + Copy>(v: &mut [(T, f32)], k: usize) {
    let n = v.len();
    if k == 0 || k >= n {
        return;
    }
    let target = k - 1;
    let mut lo = 0usize;
    let mut hi = n - 1;
    loop {
        if lo >= hi {
            break;
        }
        let p = partition(v, lo, hi);
        match p.cmp(&target) {
            Ordering::Equal => break,
            Ordering::Greater => hi = p.saturating_sub(1),
            Ordering::Less => lo = p + 1,
        }
    }
}

/// Same semantics as [`top_k_sort`], using introselect-style partitioning instead of full sort.
pub fn top_k_quickselect<T: Ord + Copy>(candidates: &[(T, f32)], k: usize) -> Vec<T> {
    let n = candidates.len();
    if k == 0 || n == 0 {
        return Vec::new();
    }
    let take = k.min(n);
    let mut v = candidates.to_vec();
    quickselect_smallest_k(&mut v, take);
    v.into_iter().take(take).map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    fn sorted_multiset<T: Ord + Copy>(xs: &[T]) -> Vec<T> {
        let mut v = xs.to_vec();
        v.sort();
        v
    }

    fn brute_top_k<T: Ord + Copy>(candidates: &[(T, f32)], k: usize) -> Vec<T> {
        let mut pairs: Vec<(T, f32)> = candidates.to_vec();
        pairs.sort_by(|a, b| cmp_pair(a, b));
        pairs.into_iter().take(k).map(|(id, _)| id).collect()
    }

    #[test]
    fn top_k_empty_and_zero_k() {
        assert!(top_k_heap::<u32>(&[], 5).is_empty());
        assert!(top_k_sort::<u32>(&[], 5).is_empty());
        assert!(top_k_quickselect::<u32>(&[], 5).is_empty());

        let c = [(1u32, 1.0), (2, 2.0)];
        assert!(top_k_heap(&c, 0).is_empty());
        assert!(top_k_sort(&c, 0).is_empty());
        assert!(top_k_quickselect(&c, 0).is_empty());
    }

    #[test]
    fn top_k_distinct_distances_match_brute() {
        let c = [(10u32, 5.0), (20, 1.0), (30, 3.0), (40, 4.0)];
        let k = 2;
        let want = brute_top_k(&c, k);
        assert_eq!(sorted_multiset(&top_k_heap(&c, k)), sorted_multiset(&want));
        assert_eq!(sorted_multiset(&top_k_sort(&c, k)), sorted_multiset(&want));
        assert_eq!(
            sorted_multiset(&top_k_quickselect(&c, k)),
            sorted_multiset(&want)
        );
    }

    #[test]
    fn top_k_ties_break_by_id() {
        let c = [(5u32, 1.0), (2, 1.0), (9, 0.5), (7, 2.0)];
        let k = 3;
        let want = brute_top_k(&c, k);
        assert_eq!(sorted_multiset(&top_k_heap(&c, k)), sorted_multiset(&want));
        assert_eq!(sorted_multiset(&top_k_sort(&c, k)), sorted_multiset(&want));
        assert_eq!(
            sorted_multiset(&top_k_quickselect(&c, k)),
            sorted_multiset(&want)
        );
    }

    #[test]
    fn top_k_k_ge_len_returns_all() {
        let c = [(3u32, 3.0), (1, 1.0), (2, 2.0)];
        let k = 10;
        let want = brute_top_k(&c, k);
        assert_eq!(sorted_multiset(&top_k_heap(&c, k)), sorted_multiset(&want));
        assert_eq!(sorted_multiset(&top_k_sort(&c, k)), sorted_multiset(&want));
        assert_eq!(
            sorted_multiset(&top_k_quickselect(&c, k)),
            sorted_multiset(&want)
        );
    }

    #[test]
    fn top_k_single_candidate() {
        let c = [(42u32, 1.5)];
        assert_eq!(top_k_heap(&c, 1), vec![42]);
        assert_eq!(top_k_sort(&c, 1), vec![42]);
        assert_eq!(top_k_quickselect(&c, 1), vec![42]);
    }

    #[test]
    fn top_k_heap_quickselect_agree_randomized() {
        const N: usize = 200;
        const ITERATIONS: usize = 10;
        for iter in 0..ITERATIONS {
            let mut rng = StdRng::seed_from_u64(42 + iter as u64);
            let candidates: Vec<(usize, f32)> =
                (0..N).map(|i| (i, rng.gen_range(0.0..1.0))).collect();
            for k in 10..=candidates.len() {
                let h = sorted_multiset(&top_k_heap(&candidates, k));
                let q = sorted_multiset(&top_k_quickselect(&candidates, k));
                let s = sorted_multiset(&brute_top_k(&candidates, k));
                assert_eq!(h, s, "k={k}");
                assert_eq!(q, s, "k={k}");
            }
        }
    }
}
