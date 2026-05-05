use crate::types::DistanceMetric;
use wide::f32x8;

/// Dispatch to the correct distance function.
pub fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    match metric {
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::DotProduct => dot_product(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
    }
}

/// Cosine distance in [0, 2] — lower is more similar.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

/// Dot product using AVX2-width SIMD chunks (8 × f32 at a time).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut acc = f32x8::ZERO;

    for i in 0..chunks {
        let base = i * 8;
        let va = f32x8::new(a[base..base + 8].try_into().unwrap());
        let vb = f32x8::new(b[base..base + 8].try_into().unwrap());
        acc += va * vb;
    }

    // Horizontal sum of SIMD accumulator
    let mut result: f32 = acc.reduce_add();

    // Scalar tail for dimensions not divisible by 8
    for i in (chunks * 8)..len {
        result += a[i] * b[i];
    }

    result
}

/// Squared Euclidean distance (L2²) — avoids a sqrt for comparisons.
pub fn euclidean_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    for i in 0..chunks {
        let base = i * 8;
        let va = f32x8::new(a[base..base + 8].try_into().unwrap());
        let vb = f32x8::new(b[base..base + 8].try_into().unwrap());
        let diff = va - vb;
        acc += diff * diff;
    }

    let mut result: f32 = acc.reduce_add();
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        result += diff * diff;
    }
    result
}

/// Euclidean (L2) distance.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_sq(a, b).sqrt()
}

/// Normalize a vector in-place to unit length.
pub fn normalize(v: &mut [f32]) {
    let norm = dot_product(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = vec![1.0f32; 8];
        let b = vec![2.0f32; 8];
        assert!((dot_product(&a, &b) - 16.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_identical_vectors() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(cosine_distance(&a, &a).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_basic() {
        let a = vec![0.0f32; 8];
        let b = vec![1.0f32; 8];
        let expected = (8.0f32).sqrt();
        assert!((euclidean_distance(&a, &b) - expected).abs() < 1e-5);
    }

    #[test]
    fn normalize_gives_unit_vector() {
        let mut v = vec![3.0f32, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        normalize(&mut v);
        let norm = dot_product(&v, &v).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
