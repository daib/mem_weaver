use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vector::distance::{cosine_distance, dot_product, euclidean_distance};

fn random_vec(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * 0.1).sin()).collect()
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    for dim in [64, 128, 256, 512, 1024, 1536] {
        let a = random_vec(dim);
        let b = random_vec(dim);

        group.bench_with_input(BenchmarkId::new("dot_product", dim), &dim, |bench, _| {
            bench.iter(|| dot_product(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| cosine_distance(black_box(&a), black_box(&b)))
        });

        group.bench_with_input(BenchmarkId::new("euclidean", dim), &dim, |bench, _| {
            bench.iter(|| euclidean_distance(black_box(&a), black_box(&b)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_distance);
criterion_main!(benches);
