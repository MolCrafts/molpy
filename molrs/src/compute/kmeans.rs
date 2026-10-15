//! k-means clustering with k-means++ initialization.

use molrs::op::F;
use ndarray::{Array2, ArrayView1};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::compute::{Compute, ComputeError, FeatureMatrix, Partition};

/// k-means clustering of the rows of a [`FeatureMatrix`] (Lloyd's algorithm
/// with k-means++ initialization).
///
/// Configuration: `k`, the iteration cap and the RNG seed. Data: any
/// [`FeatureMatrix`] — a [`PcaResult`](crate::compute::PcaResult), a
/// descriptor result or an `Array2` — clustered in its own dimensionality
/// (`n_cols`). Deterministic for a fixed seed. The result is a [`Partition`]
/// of the rows.
///
/// Defaults: [`DEFAULT_MAX_ITER`](Self::DEFAULT_MAX_ITER) Lloyd iterations
/// and seed [`DEFAULT_SEED`](Self::DEFAULT_SEED). Lloyd stops early once no
/// centroid moves by more than `1e-8` in total (squared move `< 1e-16`).
#[derive(Debug, Clone, Copy)]
pub struct Kmeans {
    k: usize,
    max_iter: usize,
    seed: u64,
}

/// The data a [`Kmeans`] clusters.
#[derive(Clone, Copy)]
pub struct KmeansInput<'a> {
    /// The samples, one row each.
    pub features: &'a (dyn FeatureMatrix + Sync),
}

/// One cluster label per row, and the centroids.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct KmeansResult {
    /// Row → cluster label in `0..k`.
    pub labels: Vec<u32>,
    /// The final centroids, `k × n_cols`.
    pub centroids: Array2<F>,
}

impl Partition for KmeansResult {
    fn n_items(&self) -> usize {
        self.labels.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        Some(u64::from(self.labels[item]))
    }
}

/// Squared total centroid move below which Lloyd's iteration has converged.
const CENTROID_MOVE_SQ: F = 1e-16;

impl Kmeans {
    /// The Lloyd iteration cap unless [`with_max_iter`](Self::with_max_iter)
    /// sets one.
    pub const DEFAULT_MAX_ITER: usize = 100;

    /// The k-means++ RNG seed unless [`with_seed`](Self::with_seed) sets one.
    pub const DEFAULT_SEED: u64 = 0;

    /// `k` clusters.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `k = 0`.
    pub fn new(k: usize) -> Result<Self, ComputeError> {
        if k == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Kmeans::k",
                value: k.to_string(),
            });
        }
        Ok(Self {
            k,
            max_iter: Self::DEFAULT_MAX_ITER,
            seed: Self::DEFAULT_SEED,
        })
    }

    /// At most `max_iter` Lloyd iterations.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for 0.
    pub fn with_max_iter(mut self, max_iter: usize) -> Result<Self, ComputeError> {
        if max_iter == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Kmeans::max_iter",
                value: max_iter.to_string(),
            });
        }
        self.max_iter = max_iter;
        Ok(self)
    }

    /// The seed of the k-means++ initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The number of clusters.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The Lloyd iteration cap.
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// The RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// k-means++: the first centroid uniformly, each next one with
    /// probability proportional to its squared distance from the nearest
    /// chosen centroid.
    fn initial_centroids(&self, x: &Array2<F>, rng: &mut StdRng) -> Array2<F> {
        let n_rows = x.nrows();
        let mut centroids = Array2::<F>::zeros((self.k, x.ncols()));
        let first = rng.random_range(0..n_rows);
        centroids.row_mut(0).assign(&x.row(first));

        let mut min_sq = vec![F::INFINITY; n_rows];
        for c in 0..self.k {
            if c > 0 {
                let total: F = min_sq.iter().sum();
                let next = if total > 0.0 {
                    let mut target: F = rng.random::<F>() * total;
                    let mut chosen = n_rows - 1;
                    for (i, &d) in min_sq.iter().enumerate() {
                        target -= d;
                        if target <= 0.0 {
                            chosen = i;
                            break;
                        }
                    }
                    chosen
                } else {
                    rng.random_range(0..n_rows)
                };
                centroids.row_mut(c).assign(&x.row(next));
            }
            for (i, row) in x.rows().into_iter().enumerate() {
                min_sq[i] = min_sq[i].min(Self::sq_dist(row, centroids.row(c)));
            }
        }
        centroids
    }

    /// The nearest centroid of every row.
    fn assign(x: &Array2<F>, centroids: &Array2<F>, labels: &mut [u32]) {
        for (label, row) in labels.iter_mut().zip(x.rows()) {
            let mut best = 0;
            let mut best_d2 = F::INFINITY;
            for (c, centroid) in centroids.rows().into_iter().enumerate() {
                let d2 = Self::sq_dist(row, centroid);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = c;
                }
            }
            *label = best as u32;
        }
    }

    /// The mean of each cluster's rows; an emptied cluster keeps its previous
    /// centroid.
    fn update(x: &Array2<F>, labels: &[u32], previous: &Array2<F>) -> Array2<F> {
        let mut sums = Array2::<F>::zeros(previous.dim());
        let mut counts = vec![0usize; previous.nrows()];
        for (&label, row) in labels.iter().zip(x.rows()) {
            let c = label as usize;
            counts[c] += 1;
            let mut sum = sums.row_mut(c);
            sum += &row;
        }
        for (c, &count) in counts.iter().enumerate() {
            if count == 0 {
                sums.row_mut(c).assign(&previous.row(c));
            } else {
                let inv = 1.0 / count as F;
                sums.row_mut(c).mapv_inplace(|v| v * inv);
            }
        }
        sums
    }

    fn sq_dist(a: ArrayView1<'_, F>, b: ArrayView1<'_, F>) -> F {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }
}

impl Compute for Kmeans {
    type Input<'a> = KmeansInput<'a>;
    type Output = KmeansResult;

    /// # Errors
    /// [`ComputeError::OutOfRange`] when there are no feature columns or
    /// fewer rows than `k`; [`ComputeError::NonFinite`] for a non-finite
    /// feature.
    fn compute(&self, input: KmeansInput<'_>) -> Result<KmeansResult, ComputeError> {
        let features = input.features;
        let (n_rows, n_dims) = (features.n_rows(), features.n_cols());
        if n_dims == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Kmeans::n_dims",
                value: n_dims.to_string(),
            });
        }
        if self.k > n_rows {
            return Err(ComputeError::OutOfRange {
                field: "Kmeans::k",
                value: format!("k={} exceeds n_rows={}", self.k, n_rows),
            });
        }
        let x = Array2::from_shape_fn((n_rows, n_dims), |(i, j)| features.row(i)[j]);
        if let Some(index) = x.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "Kmeans features",
                index,
            });
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut centroids = self.initial_centroids(&x, &mut rng);
        let mut labels = vec![0u32; n_rows];
        for _ in 0..self.max_iter {
            Self::assign(&x, &centroids, &mut labels);
            let next = Self::update(&x, &labels, &centroids);
            let moved: F = (&next - &centroids).mapv(|d| d * d).sum();
            centroids = next;
            if moved < CENTROID_MOVE_SQ {
                break;
            }
        }
        Self::assign(&x, &centroids, &mut labels);

        Ok(KmeansResult { labels, centroids })
    }
}

crate::compute::spec::compute_spec! {
    Kmeans as "Kmeans";
    params |s| {
        k: Int;
        max_iter: Int = Kmeans::DEFAULT_MAX_ITER => s.max_iter;
        seed: Int = Kmeans::DEFAULT_SEED => s.seed;
    }
    inputs {
        features: FeatureMatrix;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::{random_points, self_neighbors};
    use crate::compute::{Pca, PcaInput, Steinhardt, SteinhardtInput};
    use crate::op::standard_normal;
    use molrs::core::SimBox;
    use ndarray::array;
    use std::collections::HashSet;

    fn three_blobs(n_per_cluster: usize, seed: u64) -> Array2<F> {
        let centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let sigma: F = 0.5;
        let mut rng = StdRng::seed_from_u64(seed);
        let total = 3 * n_per_cluster;
        let mut coords = Vec::with_capacity(total * 2);
        for (cx, cy) in centers.iter().copied() {
            for _ in 0..n_per_cluster {
                coords.push(cx + sigma * standard_normal(&mut rng));
                coords.push(cy + sigma * standard_normal(&mut rng));
            }
        }
        Array2::from_shape_vec((total, 2), coords).unwrap()
    }

    fn kmeans(k: usize, max_iter: usize, seed: u64) -> Kmeans {
        Kmeans::new(k)
            .unwrap()
            .with_max_iter(max_iter)
            .unwrap()
            .with_seed(seed)
    }

    #[test]
    fn new_rejects_zero_k() {
        assert!(Kmeans::new(0).is_err());
    }

    #[test]
    fn new_rejects_zero_max_iter() {
        assert!(Kmeans::new(3).unwrap().with_max_iter(0).is_err());
    }

    #[test]
    fn defaults_live_in_core() {
        let km = Kmeans::new(3).unwrap();
        assert_eq!(km.max_iter(), Kmeans::DEFAULT_MAX_ITER);
        assert_eq!(km.seed(), Kmeans::DEFAULT_SEED);
        assert_eq!((Kmeans::DEFAULT_MAX_ITER, Kmeans::DEFAULT_SEED), (100, 0));
    }

    #[test]
    fn three_blobs_produce_three_clusters() {
        let blobs = three_blobs(20, 7);
        let labels = kmeans(3, 100, 42)
            .compute(KmeansInput { features: &blobs })
            .unwrap();

        assert_eq!(labels.labels.len(), blobs.nrows());
        assert_eq!(labels.centroids.dim(), (3, 2));
        let unique: HashSet<u32> = labels.labels.iter().copied().collect();
        assert_eq!(unique.len(), 3);

        let expected = blobs.nrows() as i32 / 3;
        for c in 0..3u32 {
            let count = labels.labels.iter().filter(|&&l| l == c).count() as i32;
            assert!(
                (count - expected).abs() <= 5,
                "cluster {c} size {count} not within ±5 of {expected}"
            );
        }
    }

    #[test]
    fn same_seed_identical_labels() {
        let blobs = three_blobs(20, 7);
        let km = kmeans(3, 100, 42);
        let a = km.compute(KmeansInput { features: &blobs }).unwrap();
        let b = km.compute(KmeansInput { features: &blobs }).unwrap();
        assert_eq!(a.labels, b.labels);
    }

    #[test]
    fn err_when_k_exceeds_rows() {
        let rows = array![[0.0, 0.0], [1.0, 1.0]];
        let err = kmeans(5, 10, 42)
            .compute(KmeansInput { features: &rows })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn err_on_nan_input() {
        let rows = array![[0.0, F::NAN], [1.0, 1.0], [2.0, 2.0]];
        let err = kmeans(2, 10, 42)
            .compute(KmeansInput { features: &rows })
            .unwrap_err();
        assert!(matches!(err, ComputeError::NonFinite { .. }));
    }

    /// The dimensionality is the matrix's (it was hardcoded to 2): three
    /// blobs separated only along z are found in 3-D.
    #[test]
    fn clusters_in_the_dimensionality_of_the_matrix() {
        let rows = Array2::from_shape_fn((30, 3), |(i, d)| {
            let blob = (i / 10) as F;
            let jitter = 0.01 * ((i * 3 + d) as F).sin();
            if d == 2 { 10.0 * blob + jitter } else { jitter }
        });
        let result = kmeans(3, 100, 0)
            .compute(KmeansInput { features: &rows })
            .unwrap();
        assert_eq!(result.centroids.ncols(), 3);
        for blob in 0..3 {
            let members = &result.labels[blob * 10..(blob + 1) * 10];
            assert!(members.iter().all(|&l| l == members[0]));
        }
        let distinct: HashSet<u32> = result.labels.iter().copied().collect();
        assert_eq!(distinct.len(), 3);
        assert_eq!(
            result.sizes().into_values().collect::<Vec<_>>(),
            vec![10, 10, 10]
        );
    }

    /// Acceptance ac-010: Steinhardt → Pca(3) → Kmeans, with no glue between
    /// steps — each result is the next step's `FeatureMatrix`, and k-means
    /// clusters in 3-D.
    #[test]
    fn steinhardt_pca_kmeans_composes() {
        let box_len = 6.0;
        let simbox = SimBox::cube(box_len, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
        let points = random_points(200, box_len, 11);
        let neighbors = self_neighbors(points.view(), &simbox, 1.6);

        let descriptors = Steinhardt::new(&[4, 6, 8])
            .unwrap()
            .with_wl(true)
            .compute(SteinhardtInput {
                neighbors: &neighbors,
            })
            .unwrap();
        let projection = Pca::new(3)
            .unwrap()
            .compute(PcaInput {
                features: &descriptors,
            })
            .unwrap();
        let clusters = Kmeans::new(2)
            .unwrap()
            .compute(KmeansInput {
                features: &projection,
            })
            .unwrap();

        assert_eq!(projection.coords.dim(), (200, 3));
        assert_eq!(clusters.centroids.dim(), (2, 3));
        assert_eq!(clusters.n_items(), 200);
        assert!(clusters.labels.iter().all(|&l| l < 2));
        assert_eq!(clusters.sizes().len(), 2);
    }
}
