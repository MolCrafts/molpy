//! Radial distribution function g(r) computation.

mod accumulator;
mod result;

pub use accumulator::RdfAccumulator;
pub use result::{RdfMode, RdfResult};

use molrs::core::FrameAccess;
use molrs::core::SimBox;
use molrs::core::neighbors::Backend;
use molrs::core::{LinkCell, NeighborList, Neighbors};
use molrs::op::{F, Fnx3View};
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::compute::require_dist_sq;

/// Radial distribution function g(r) calculator.
///
/// Stateless parameter container: bin count, radial cutoffs, and precomputed
/// bin edges/centers. Actual histograms are built inside each
/// [`compute`](LegacyCompute::compute) call.
///
/// Accumulates pair distances from neighbor lists (one per frame) into a
/// histogram between `r_min` and `r_max`, then normalizes by the ideal-gas
/// shell volume at the system number density during
/// [`ComputeResult::finalize`](crate::compute::ComputeResult::finalize).
///
/// Each frame contributes its own `SimBox` volume; non-periodic frames error
/// out — the compute never fabricates a bounding box.
#[derive(Debug, Clone)]
pub struct Rdf {
    n_bins: usize,
    r_min: F,
    r_max: F,
    r_min_sq: F,
    r_max_sq: F,
    bin_width: F,
    bin_edges: Array1<F>,
    bin_centers: Array1<F>,
    dimensionality: u8,
}

impl Rdf {
    /// Create an RDF analysis binning pair distances in `[r_min, r_max]`
    /// (angstrom) into `n_bins` bins.
    ///
    /// # Arguments
    ///
    /// **Note the argument order — `r_max` before `r_min`** (matches freud's
    /// convention, kept for cross-check compatibility):
    ///
    /// * `n_bins` — histogram bin count. Must be ≥ 1.
    /// * `r_max` — upper edge of the last bin, Å. Must be > `r_min`.
    /// * `r_min` — lower edge of bin 0, Å. Must be ≥ 0. Often 0.0.
    ///
    /// # References
    ///
    /// Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §2.6.
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<Self, ComputeError> {
        if n_bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::n_bins",
                value: n_bins.to_string(),
            });
        }
        if r_min.is_nan() || r_min < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::r_min",
                value: r_min.to_string(),
            });
        }
        if r_max.is_nan() || r_max <= r_min {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::r_max",
                value: format!("r_max={r_max}, r_min={r_min}"),
            });
        }
        let bin_width = (r_max - r_min) / n_bins as F;
        let bin_edges = Array1::from_iter((0..=n_bins).map(|i| r_min + i as F * bin_width));
        let bin_centers =
            Array1::from_iter((0..n_bins).map(|i| r_min + (i as F + 0.5) * bin_width));
        Ok(Self {
            n_bins,
            r_min,
            r_max,
            r_min_sq: r_min * r_min,
            r_max_sq: r_max * r_max,
            bin_width,
            bin_edges,
            bin_centers,
            dimensionality: 3,
        })
    }

    /// Switch the shell-volume normalization to 2-D (`2π r dr`) instead of
    /// the default 3-D (`(4/3) π (r_o³ − r_i³)`).
    ///
    /// For a 2-D system the `SimBox` should be set up so that `volume()`
    /// returns the in-plane area (e.g. `Lz = 1`). Matches the convention
    /// of `freud.density.Rdf` when `freud.box.Box.is2D` is true.
    pub fn with_dimensionality(mut self, dim: u8) -> Self {
        assert!(dim == 2 || dim == 3, "Rdf dimensionality must be 2 or 3");
        self.dimensionality = dim;
        self
    }

    pub fn dimensionality(&self) -> u8 {
        self.dimensionality
    }

    pub fn bin_width(&self) -> F {
        self.bin_width
    }
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }
    pub fn r_min(&self) -> F {
        self.r_min
    }
    pub fn r_max(&self) -> F {
        self.r_max
    }

    /// Bin one squared distance into `n_r` (self/cross agnostic).
    #[inline(always)]
    fn accumulate_d2(&self, d2: F, n_r: &mut Array1<F>) {
        if d2 <= 0.0 {
            return;
        }
        if d2 < self.r_min_sq || d2 >= self.r_max_sq {
            return;
        }
        let d = d2.sqrt();
        let bin = ((d - self.r_min) / self.bin_width) as usize;
        if bin < self.n_bins {
            n_r[bin] += 1.0;
        }
    }

    /// Bin every pair distance of one table into the running histogram.
    ///
    /// A table without the `dist_sq` column carries no distances to bin, so it
    /// is rejected as [`ComputeError::BadShape`] — the same answer every other
    /// distance-consuming kernel gives, rather than an all-zero g(r) for an
    /// indices-only list. Callers with
    /// no materialized distances want [`compute_self`](Self::compute_self) /
    /// [`compute_frame`](Self::compute_frame), which stream them.
    fn accumulate_into(&self, nlist: &Neighbors, n_r: &mut Array1<F>) -> Result<(), ComputeError> {
        for &d2 in require_dist_sq(nlist)? {
            self.accumulate_d2(d2, n_r);
        }
        Ok(())
    }

    /// Self-query RDF via cell-list **index + visit** (no `Neighbors` heap).
    ///
    /// This is the production path for large \(N\): memory is \(O(N)\) for the
    /// cell index plus \(O(n_{\mathrm{bins}})\) for the histogram, not
    /// \(O(P)\) for materialized pairs.
    pub fn compute_self(
        &self,
        points: Fnx3View<'_>,
        bx: &SimBox,
    ) -> Result<RdfResult, ComputeError> {
        let vol = bx.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::volume",
                value: vol.to_string(),
            });
        }
        let n = points.nrows();
        if n < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let mut nl = NeighborList::new(self.r_max);
        nl.build(points, bx);
        let mut n_r = Array1::zeros(self.n_bins);
        nl.for_each_pair(|pair| {
            self.accumulate_d2(pair.dist_sq, &mut n_r);
        });
        self.finalize_histogram(n_r, n, n, RdfMode::SelfQuery, vol)
    }

    /// Cross-query RDF (group A vs group B) via cell-list index + per-query
    /// neighbor visits — still no full pair materialization.
    ///
    /// `ref_points` become the spatial index (B); `query_points` are A.
    pub fn compute_cross(
        &self,
        ref_points: Fnx3View<'_>,
        query_points: Fnx3View<'_>,
        bx: &SimBox,
    ) -> Result<RdfResult, ComputeError> {
        let vol = bx.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::volume",
                value: vol.to_string(),
            });
        }
        let n_ref = ref_points.nrows();
        let n_query = query_points.nrows();
        if n_ref < 1 || n_query < 1 {
            return Err(ComputeError::EmptyInput);
        }
        let mut n_r = Array1::zeros(self.n_bins);
        let cutoff_sq = self.r_max * self.r_max;
        // Stays on the cell-list backend rather than the engine: this asks for
        // the neighbors of one arbitrary query point at a time, and the engine
        // has no per-point streaming — its pair stream is the whole self list.
        let mut lc = LinkCell::new().cutoff(self.r_max);
        lc.build_index(ref_points, bx);
        for qi in 0..n_query {
            let qp = [
                query_points[[qi, 0]],
                query_points[[qi, 1]],
                query_points[[qi, 2]],
            ];
            lc.visit_neighbors_of_pt(qp, bx, |_, d2, _| {
                if d2 <= cutoff_sq {
                    self.accumulate_d2(d2, &mut n_r);
                }
            });
        }
        self.finalize_histogram(n_r, n_ref, n_query, RdfMode::CrossQuery, vol)
    }

    /// Self-query RDF from a [`FrameAccess`] (reads `atoms.x/y/z` + simbox).
    pub fn compute_frame<FA: FrameAccess + Sync>(
        &self,
        frame: &FA,
    ) -> Result<RdfResult, ComputeError> {
        let bx = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (xs, ys, zs) = get_positions_ref(frame)?;
        let xs = xs.slice();
        let ys = ys.slice();
        let zs = zs.slice();
        let vol = bx.volume();
        if !(vol.is_finite() && vol > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::volume",
                value: vol.to_string(),
            });
        }
        let n = xs.len();
        if n < 2 {
            return Err(ComputeError::EmptyInput);
        }
        let mut nl = NeighborList::new(self.r_max);
        nl.build_columns(xs, ys, zs, bx);
        let mut n_r = Array1::zeros(self.n_bins);
        nl.for_each_pair(|pair| {
            self.accumulate_d2(pair.dist_sq, &mut n_r);
        });
        self.finalize_histogram(n_r, n, n, RdfMode::SelfQuery, vol)
    }

    fn finalize_histogram(
        &self,
        n_r: Array1<F>,
        n_points: usize,
        n_query_points: usize,
        mode: RdfMode,
        volume: F,
    ) -> Result<RdfResult, ComputeError> {
        let mut result = RdfResult {
            bin_edges: self.bin_edges.clone(),
            bin_centers: self.bin_centers.clone(),
            rdf: Array1::zeros(self.n_bins),
            n_r,
            n_points,
            n_query_points,
            mode,
            volume,
            r_min: self.r_min,
            n_frames: 1,
            dimensionality: self.dimensionality,
            finalized: false,
        };
        use crate::compute::ComputeResult;
        result.finalize();
        Ok(result)
    }
}

impl LegacyCompute for Rdf {
    type Args<'a> = &'a [Neighbors];
    type Output = RdfResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        neighbors: &'a [Neighbors],
    ) -> Result<RdfResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if neighbors.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: neighbors.len(),
                what: "neighbor-list count",
            });
        }

        // The batch path is the streaming accumulator driven over the frame
        // slice — one source of truth for the accumulation + normalization.
        let mut acc = RdfAccumulator::new(self.clone());
        for (frame, nlist) in frames.iter().zip(neighbors.iter()) {
            acc.accumulate(*frame, nlist)?;
        }
        // `finalize` normalizes eagerly so direct callers (outside Graph) can
        // read `rdf` without having to call `finalize` themselves.
        acc.finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::super::positions::get_positions_ref;
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};
    use rand::RngExt;

    fn random_frame(n: usize, box_len: F, seed: u64) -> Frame {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let mut block = Block::new();
        let x = A1::from_iter((0..n).map(|_| rng.random::<F>() * box_len));
        let y = A1::from_iter((0..n).map(|_| rng.random::<F>() * box_len));
        let z = A1::from_iter((0..n).map(|_| rng.random::<F>() * box_len));
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox = Some(
            SimBox::cube(
                box_len,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [true, true, true],
            )
            .unwrap(),
        );
        frame
    }

    #[test]
    fn ideal_gas_rdf_approaches_one() {
        let n = 500;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 40;

        let frame = random_frame(n, box_len, 42);
        let nlist = nlist_from_frame(&frame, r_max);

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap();
        let result = rdf.compute(&[&frame], &[nlist]).unwrap();

        for i in 5..n_bins {
            assert!(
                (result.rdf[i] - 1.0).abs() < 0.5,
                "g(r={:.2}) = {:.3}, expected ~1.0",
                result.bin_centers[i],
                result.rdf[i]
            );
        }
    }

    #[test]
    fn multi_frame_reduces_variance() {
        // Batch compute over 10 frames; variance of g(r) - 1 should be smaller
        // than any single-frame variance at the same density.
        let n = 200;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 20;

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap();

        // Single-frame baseline.
        let frame0 = random_frame(n, box_len, 100);
        let nlist0 = nlist_from_frame(&frame0, r_max);
        let single = rdf.compute(&[&frame0], &[nlist0]).unwrap();
        let var_single: F = single
            .rdf
            .iter()
            .skip(3)
            .map(|g| (g - 1.0).powi(2))
            .sum::<F>()
            / (n_bins - 3) as F;

        // Multi-frame batch.
        let frames_owned: Vec<Frame> = (100..110u64).map(|s| random_frame(n, box_len, s)).collect();
        let nlists: Vec<Neighbors> = frames_owned
            .iter()
            .map(|f| nlist_from_frame(f, r_max))
            .collect();
        let frame_refs: Vec<&Frame> = frames_owned.iter().collect();
        let multi = rdf.compute(&frame_refs, &nlists).unwrap();
        let var_multi: F = multi
            .rdf
            .iter()
            .skip(3)
            .map(|g| (g - 1.0).powi(2))
            .sum::<F>()
            / (n_bins - 3) as F;

        assert!(
            var_multi < var_single,
            "multi-frame variance ({var_multi:.6}) should be less than single-frame ({var_single:.6})"
        );
    }

    #[test]
    fn streaming_accumulator_matches_batch_bitwise() {
        // The batch compute is implemented on the accumulator, but assert the
        // public streaming contract explicitly: frame-by-frame accumulate +
        // finalize == one-shot compute, bit-for-bit, including under per-frame
        // box changes (NPT-style).
        let n = 200;
        let r_max: F = 4.0;
        let n_bins = 20;

        let frames_owned: Vec<Frame> = (0..8u64)
            .map(|s| random_frame(n, 10.0 + 0.05 * s as F, 200 + s))
            .collect();
        let nlists: Vec<Neighbors> = frames_owned
            .iter()
            .map(|f| nlist_from_frame(f, r_max))
            .collect();

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap();

        let frame_refs: Vec<&Frame> = frames_owned.iter().collect();
        let batch = rdf.compute(&frame_refs, &nlists).unwrap();

        let mut acc = RdfAccumulator::new(rdf);
        for (f, nl) in frames_owned.iter().zip(nlists.iter()) {
            acc.accumulate(f, nl).unwrap();
        }
        assert_eq!(acc.n_frames(), frames_owned.len());
        let streamed = acc.finalize().unwrap();

        assert_eq!(streamed.n_r, batch.n_r);
        assert_eq!(streamed.rdf, batch.rdf);
        assert_eq!(streamed.n_points, batch.n_points);
        assert_eq!(streamed.volume, batch.volume);
        assert_eq!(streamed.n_frames, batch.n_frames);
    }

    #[test]
    fn accumulator_finalize_before_any_frame_is_error() {
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        let acc = RdfAccumulator::new(rdf);
        assert!(matches!(
            acc.finalize().unwrap_err(),
            ComputeError::EmptyInput
        ));
    }

    #[test]
    fn empty_frames_is_error() {
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        let frames: Vec<&Frame> = Vec::new();
        let nlists: Vec<Neighbors> = Vec::new();
        let err = rdf.compute(&frames, &nlists).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_nlist_count_is_error() {
        let frame = random_frame(50, 10.0, 1);
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        let err = rdf
            .compute(&[&frame], &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn missing_simbox_is_error() {
        let mut frame = random_frame(50, 10.0, 1);
        frame.simbox = None;
        let nlist = {
            use molrs::core::NeighborQuery;
            let (xs, ys, zs) = get_positions_ref(&frame).unwrap();
            NeighborQuery::unbounded_columns(xs.slice(), ys.slice(), zs.slice(), 4.0).query_self()
        };
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        let err = rdf.compute(&[&frame], &[nlist]).unwrap_err();
        assert!(matches!(err, ComputeError::MissingSimBox));
    }

    #[test]
    fn r_min_shifts_bins_and_filters_pairs() {
        let box_len: F = 10.0;
        let frame = random_frame(200, box_len, 99);
        let nlist = nlist_from_frame(&frame, 4.0);

        let r_min: F = 1.5;
        let r_max: F = 4.0;
        let n_bins = 25;
        let rdf = Rdf::new(n_bins, r_max, r_min).unwrap();
        let result = rdf.compute(&[&frame], &[nlist]).unwrap();

        assert!((result.bin_edges[0] - r_min).abs() < 1e-12);
        assert!((result.bin_edges[n_bins] - r_max).abs() < 1e-12);
        assert!((result.r_min - r_min).abs() < 1e-12);

        let dr = (r_max - r_min) / n_bins as F;
        for i in 0..n_bins {
            let expected = r_min + (i as F + 0.5) * dr;
            assert!((result.bin_centers[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn zero_distance_pairs_are_skipped() {
        use molrs::core::Block;

        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        let simbox = SimBox::cube(10.0, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
        frame.simbox = Some(simbox);

        let nlist = nlist_from_frame(&frame, 2.0);

        let rdf = Rdf::new(10, 2.0, 0.0).unwrap();
        let result = rdf.compute(&[&frame], &[nlist]).unwrap();

        for (i, &c) in result.n_r.iter().enumerate() {
            assert_eq!(c, 0.0, "bin {i} should be empty, got {c}");
        }
    }

    /// ac-004 (spec neighborlist-03-compute): the self-query normalization
    /// multiplies the histogram by **2** because a self list is *half-shell* —
    /// each unordered pair is counted once, where the ideal-gas reference
    /// counts both orderings. A cross-query list is directed and already
    /// carries both orderings, so it must **not** be multiplied.
    ///
    /// The two paths are made to describe the same physical system, so the
    /// factor 2 is exactly what makes their `g(r)` agree.
    ///
    /// Fixture (hard-coded, no search, no RNG): two particles 1.5 Å apart in a
    /// cubic box `L = 10 Å` (`V = 1000 Å³`), 4 bins over `[0, 4) Å` so
    /// `Δr = 1 Å` and the pair lands in bin 1, whose shell spans `[1, 2) Å`.
    ///
    /// ```text
    /// V_shell = (4/3)π(2³ − 1³) = 28π/3   = 29.321531433504735 Å³
    ///
    /// self  (half list, n_r = 1):
    ///   ρ = N/V = 2/1000
    ///   ideal = ρ · V_shell · N = 4·(28π/3)/1000 = 0.11728612573401893
    ///   g = 2 · 1 / ideal = 6000/(112π)          = 17.052315331274503
    ///
    /// cross (directed list, n_r = 2):
    ///   ideal = N_a · N_b · V_shell / V = 4·(28π/3)/1000 (same)
    ///   g = 2 / ideal                             = 17.052315331274503
    /// ```
    ///
    /// Drop the factor 2 from the self branch and the self value halves to
    /// 8.526157665637252 — both asserts below fail.
    #[test]
    fn rdf_self_half_list_factor_two_vs_cross() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        /// g(r) in bin 1 for the two-particle fixture, factor 2 included.
        const G_BIN1: F = 17.052315331274503;

        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(vec![0.0 as F, 1.5]).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(vec![0.0 as F, 0.0]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(10.0, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());

        let d_sq: F = 1.5 * 1.5;
        // Half-shell self list: the unordered pair {0, 1} exactly once.
        let self_list = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 1,
                dist_sq: d_sq,
                disp: [1.5, 0.0, 0.0],
            }],
            NeighborColumns::DIST_SQ,
            QueryMode::SelfQuery { n_points: 2 },
        );
        // The same two particles as a directed cross list of the set against
        // itself with the two zero-distance self-pairs removed: both orderings
        // of {0, 1} are present.
        let cross_list = Neighbors::from_pairs(
            [
                NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: d_sq,
                    disp: [1.5, 0.0, 0.0],
                },
                NeighborPair {
                    i: 1,
                    j: 0,
                    dist_sq: d_sq,
                    disp: [-1.5, 0.0, 0.0],
                },
            ],
            NeighborColumns::DIST_SQ,
            QueryMode::CrossQuery {
                n_query_points: 2,
                n_points: 2,
            },
        );

        let rdf = Rdf::new(4, 4.0, 0.0).unwrap();
        let self_result = rdf.compute(&[&frame], &[self_list]).unwrap();
        let cross_result = rdf.compute(&[&frame], &[cross_list]).unwrap();

        // Raw counts: the half list holds one pair, the directed list two.
        assert_eq!(self_result.n_r.to_vec(), vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cross_result.n_r.to_vec(), vec![0.0, 2.0, 0.0, 0.0]);

        assert!(
            (self_result.rdf[1] - G_BIN1).abs() < 1e-10,
            "self-query g(r) in bin 1 = {} (expected {G_BIN1}); \
             a missing half-list factor 2 halves it",
            self_result.rdf[1]
        );
        assert!(
            (cross_result.rdf[1] - G_BIN1).abs() < 1e-10,
            "cross-query g(r) in bin 1 = {} (expected {G_BIN1}); \
             a spurious factor 2 doubles it",
            cross_result.rdf[1]
        );
        assert!(
            (self_result.rdf[1] - cross_result.rdf[1]).abs() < 1e-10,
            "half list × 2 must reproduce the directed list: {} vs {}",
            self_result.rdf[1],
            cross_result.rdf[1]
        );
        for bin in [0usize, 2, 3] {
            assert_eq!(self_result.rdf[bin], 0.0, "self bin {bin} must be empty");
            assert_eq!(cross_result.rdf[bin], 0.0, "cross bin {bin} must be empty");
        }
    }

    #[test]
    fn finalize_is_idempotent() {
        use crate::compute::ComputeResult;

        let frame = random_frame(200, 10.0, 42);
        let nlist = nlist_from_frame(&frame, 4.0);
        let rdf = Rdf::new(20, 4.0, 0.0).unwrap();
        let mut result = rdf.compute(&[&frame], &[nlist]).unwrap();
        let first = result.rdf.clone();
        result.finalize();
        result.finalize();
        assert_eq!(result.rdf, first);
    }

    #[test]
    fn new_validates_inputs() {
        assert!(Rdf::new(0, 1.0, 0.0).is_err());
        assert!(Rdf::new(10, 1.0, -0.1).is_err());
        assert!(Rdf::new(10, 1.0, 1.0).is_err());
        assert!(Rdf::new(10, 0.5, 1.0).is_err());
        assert!(Rdf::new(10, 1.0, 0.0).is_ok());
    }

    /// Sanity check that 2D normalization (`2 π r dr` shells) reproduces the
    /// expected `g(r) → 1` plateau for a random 2-D ideal gas. Mirrors
    /// freud's `freud.density.Rdf` behaviour when `box.is2D == True`.
    #[test]
    fn rdf_2d_orthorhombic_box_plateaus_to_one() {
        // Pack 600 points uniformly in a 10×10 plane (z fixed). Use Lz=1
        // so simbox.volume() returns the 2-D area (Lx · Ly).
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let n = 600;
        let lx: F = 10.0;
        let ly: F = 10.0;
        let lz: F = 1.0;
        let r_max: F = 3.5;
        let n_bins = 35;

        let mut rng = StdRng::seed_from_u64(123);
        let mut block = Block::new();
        let xs = A1::from_iter((0..n).map(|_| rng.random::<F>() * lx));
        let ys = A1::from_iter((0..n).map(|_| rng.random::<F>() * ly));
        let zs = A1::from_iter((0..n).map(|_| 0.0_f64));
        block.insert("x", xs.into_dyn()).unwrap();
        block.insert("y", ys.into_dyn()).unwrap();
        block.insert("z", zs.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        let simbox = SimBox::ortho(
            array![lx, ly, lz],
            array![0.0 as F, 0.0, 0.0],
            [true, true, false],
        )
        .unwrap();
        frame.simbox = Some(simbox);

        // Build the neighbor list via the native SoA path (z is identical, so a
        // single-cell column).
        let nlist = nlist_from_frame(&frame, r_max);

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap().with_dimensionality(2);
        let result = rdf.compute(&[&frame], &[nlist]).unwrap();

        // Plateau check: bins outside the first-shell artifact should average
        // to ~1.0. Use a generous tolerance since this is a finite ideal gas.
        let plateau: F = result.rdf.iter().skip(5).copied().sum::<F>() / (n_bins - 5) as F;
        assert!(
            (plateau - 1.0).abs() < 0.15,
            "2-D ideal-gas RDF plateau = {plateau:.3}, expected ≈ 1.0"
        );
        assert_eq!(result.dimensionality, 2);
    }
}
