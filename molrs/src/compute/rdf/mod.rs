//! Radial distribution function g(r) computation.

mod accumulator;
mod result;

pub use accumulator::RdfAccumulator;
pub use result::RdfResult;

use molrs::op::F;

use crate::compute::{Axis, ComputeError, Stream};

/// Radial distribution function g(r).
///
/// Configuration: the radial bins (`[r_min, r_max)` in Å, half-open) and the
/// dimensionality of the shell-volume normalization. Data: one
/// [`PairSource`](molrs::core::PairSource) per frame — a materialized
/// [`Neighbors`](molrs::core::Neighbors) table (self- or cross-query) or a
/// built [`NeighborList`](molrs::core::NeighborList), which streams its pairs
/// out of the cell list without materializing them (the bounded-memory path
/// for large `N`). Each source carries the box it was searched in; its volume
/// normalizes the frame, and a source with no box is an error — the compute
/// never invents one.
///
/// Whether the histogram is doubled (a half-shell self-query) or not (a
/// directed cross-query) is read from the source's
/// [`QueryMode`](molrs::core::QueryMode); see [`RdfResult`]. Every frame must
/// have the same pairing.
///
/// `Rdf` is a [`Stream`] only: [`accumulator`](Stream::accumulator) hands
/// out an [`RdfAccumulator`]; push one pair source per frame and
/// [`finish`](crate::compute::Accumulate::finish) normalizes the running histogram. State is
/// `O(n_bins)` whatever the trajectory length.
///
/// # References
///
/// Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §2.6.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rdf {
    axis: Axis,
    dimensionality: u8,
}

impl Rdf {
    /// The dimensionality of the shell normalisation when not set: 3.
    pub const DEFAULT_DIMENSIONALITY: u8 = 3;

    /// Bin pair distances in `[r_min, r_max)` (Å) into `n_bins` bins, with
    /// the 3-D shell normalization.
    ///
    /// **Note the argument order — `r_max` before `r_min`** (freud's
    /// convention).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `r_min < 0` (or NaN), `n_bins == 0`
    /// or `r_max ≤ r_min`.
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<Self, ComputeError> {
        if r_min.is_nan() || r_min < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::r_min",
                value: r_min.to_string(),
            });
        }
        Ok(Self {
            axis: Axis::new(r_min, r_max, n_bins)?,
            dimensionality: Self::DEFAULT_DIMENSIONALITY,
        })
    }

    /// Use the 2-D shell normalization (`π (r_o² − r_i²)`) instead of the
    /// 3-D one (`(4/3) π (r_o³ − r_i³)`).
    ///
    /// For a 2-D system the box should be set up so that its volume is the
    /// in-plane area (e.g. `Lz = 1`), as `freud.density.Rdf` does for a 2-D
    /// box.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a dimensionality other than 2 or 3.
    pub fn with_dimensionality(mut self, dim: u8) -> Result<Self, ComputeError> {
        if !(dim == 2 || dim == 3) {
            return Err(ComputeError::OutOfRange {
                field: "Rdf::dimensionality (2 or 3)",
                value: dim.to_string(),
            });
        }
        self.dimensionality = dim;
        Ok(self)
    }

    /// The radial bins.
    pub fn axis(&self) -> Axis {
        self.axis
    }

    /// The shell-volume dimensionality (2 or 3).
    pub fn dimensionality(&self) -> u8 {
        self.dimensionality
    }

    /// Bin width, Å.
    pub fn bin_width(&self) -> F {
        self.axis.width()
    }

    /// Number of bins.
    pub fn n_bins(&self) -> usize {
        self.axis.n_bins()
    }

    /// Lower edge of bin 0, Å.
    pub fn r_min(&self) -> F {
        self.axis.min()
    }

    /// Upper edge of the last bin, Å.
    pub fn r_max(&self) -> F {
        self.axis.max()
    }
}

impl Stream for Rdf {
    type Accumulator = RdfAccumulator;

    fn accumulator(&self) -> RdfAccumulator {
        RdfAccumulator::new(*self)
    }
}

crate::compute::spec::compute_spec! {
    Rdf as "Rdf", Frames;
    params |s| {
        n_bins: Int;
        r_max: Float;
        r_min: Float;
        dimensionality: Int = Rdf::DEFAULT_DIMENSIONALITY => s.dimensionality;
    }
    inputs {
        pairs: PairSource;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Accumulate;
    use crate::compute::fixtures::{random_points, self_neighbors};
    use crate::core::Cell;
    use molrs::core::{
        NeighborColumns, NeighborList, NeighborPair, Neighbors, PairSource, QueryMode, SimBox,
    };
    use ndarray::Array2;

    fn cube(box_len: F) -> SimBox {
        SimBox::new(
            Cell::Lengths([box_len; 3]),
            [0.0 as F, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap()
    }

    /// One random frame's self-query table.
    fn random_table(n: usize, box_len: F, seed: u64, cutoff: F) -> Neighbors {
        self_neighbors(
            random_points(n, box_len, seed).view(),
            &cube(box_len),
            cutoff,
        )
    }

    /// Push every source into a fresh accumulator, then finish.
    fn fold(rdf: &Rdf, sources: &[&(dyn PairSource + Sync)]) -> Result<RdfResult, ComputeError> {
        rdf.accumulator().fold(sources.iter().copied())
    }

    fn rdf_of(rdf: &Rdf, tables: &[Neighbors]) -> Result<RdfResult, ComputeError> {
        let sources: Vec<&(dyn PairSource + Sync)> = tables
            .iter()
            .map(|t| t as &(dyn PairSource + Sync))
            .collect();
        fold(rdf, &sources)
    }

    #[test]
    fn ideal_gas_rdf_approaches_one() {
        let n = 500;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 40;

        let nlist = random_table(n, box_len, 42, r_max);

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap();
        let result = rdf_of(&rdf, &[nlist]).unwrap();

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
        // Fold 10 frames; variance of g(r) - 1 should be smaller
        // than any single-frame variance at the same density.
        let n = 200;
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let n_bins = 20;

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap();

        let single = rdf_of(&rdf, &[random_table(n, box_len, 100, r_max)]).unwrap();
        let var_single: F = single
            .rdf
            .iter()
            .skip(3)
            .map(|g| (g - 1.0).powi(2))
            .sum::<F>()
            / (n_bins - 3) as F;

        let tables: Vec<Neighbors> = (100..110u64)
            .map(|s| random_table(n, box_len, s, r_max))
            .collect();
        let multi = rdf_of(&rdf, &tables).unwrap();
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

    /// ac-007: frame-by-frame push + finish == one fold over the frames, bit
    /// for bit, in every bin, including under per-frame box changes
    /// (NPT-style).
    #[test]
    fn push_then_finish_matches_a_fold_bitwise() {
        let n = 200;
        let r_max: F = 4.0;
        let n_bins = 20;

        let tables: Vec<Neighbors> = (0..8u64)
            .map(|s| random_table(n, 10.0 + 0.05 * s as F, 200 + s, r_max))
            .collect();

        let rdf = Rdf::new(n_bins, r_max, 0.0).unwrap();
        let batch = rdf_of(&rdf, &tables).unwrap();

        let mut acc = rdf.accumulator();
        for nl in &tables {
            acc.push(nl).unwrap();
        }
        assert_eq!(acc.n_frames(), tables.len());
        let streamed = acc.finish().unwrap();

        assert_eq!(streamed, batch);
    }

    /// The zero-materialisation path: a built `NeighborList` handed over as
    /// the pair source counts exactly the pairs of its materialized table.
    #[test]
    fn built_neighbor_list_matches_its_table() {
        let box_len: F = 10.0;
        let r_max: F = 4.0;
        let points = random_points(300, box_len, 7);
        let mut nl = NeighborList::new(r_max);
        nl.build(points.view(), &cube(box_len));
        let table = nl.neighbors(NeighborColumns::DIST_SQ);

        let rdf = Rdf::new(16, r_max, 0.0).unwrap();
        let streamed = fold(&rdf, &[&nl]).unwrap();
        let materialized = fold(&rdf, &[&table]).unwrap();

        assert_eq!(streamed.n_r, materialized.n_r);
        assert_eq!(streamed.rdf, materialized.rdf);
        assert!(streamed.self_query);
    }

    #[test]
    fn finish_before_any_frame_is_error() {
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        assert!(matches!(
            rdf.accumulator().finish().unwrap_err(),
            ComputeError::EmptyInput
        ));
    }

    /// A hand-built table records no box; g(r) needs the volume and refuses
    /// to invent one.
    #[test]
    fn source_without_a_box_is_error() {
        let table = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            NeighborColumns::DIST_SQ,
            QueryMode::SelfQuery { n_points: 2 },
        );
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        assert!(matches!(
            fold(&rdf, &[&table]).unwrap_err(),
            ComputeError::BadShape { .. }
        ));
        assert!(fold(&rdf, &[&table.with_simbox(cube(10.0))]).is_ok());
    }

    #[test]
    fn table_without_distances_is_error() {
        let table = random_table(50, 10.0, 3, 4.0).repack(NeighborColumns::INDICES_ONLY);
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        let err = fold(&rdf, &[&table]).unwrap_err();
        let ComputeError::BadShape { expected, .. } = err else {
            panic!("expected BadShape, got {err:?}");
        };
        assert!(expected.contains("dist_sq"));
    }

    /// A NaN distance used to land in bin 0; it is an error, and the frame
    /// leaves the running state untouched.
    #[test]
    fn nan_distance_is_non_finite_and_rejects_the_frame() {
        let pair = |d2: F| NeighborPair {
            i: 0,
            j: 1,
            dist_sq: d2,
            disp: [0.0; 3],
        };
        let good = Neighbors::from_pairs(
            [pair(1.0)],
            NeighborColumns::DIST_SQ,
            QueryMode::SelfQuery { n_points: 2 },
        )
        .with_simbox(cube(10.0));
        let bad = Neighbors::from_pairs(
            [pair(2.25), pair(F::NAN)],
            NeighborColumns::DIST_SQ,
            QueryMode::SelfQuery { n_points: 2 },
        )
        .with_simbox(cube(10.0));

        let rdf = Rdf::new(4, 4.0, 0.0).unwrap();
        let mut acc = rdf.accumulator();
        acc.push(&good).unwrap();
        let before = acc.finish().unwrap();
        assert!(matches!(
            acc.push(&bad),
            Err(ComputeError::NonFinite { index: 1, .. })
        ));
        assert_eq!(acc.finish().unwrap(), before);
    }

    #[test]
    fn mixed_pairing_across_frames_is_error() {
        let self_table = random_table(20, 10.0, 1, 4.0);
        let cross = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 0,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            NeighborColumns::DIST_SQ,
            QueryMode::CrossQuery {
                n_query_points: 1,
                n_points: 1,
            },
        )
        .with_simbox(cube(10.0));
        let rdf = Rdf::new(10, 4.0, 0.0).unwrap();
        assert!(matches!(
            fold(&rdf, &[&self_table, &cross]).unwrap_err(),
            ComputeError::BadShape { .. }
        ));
    }

    #[test]
    fn r_min_shifts_bins_and_filters_pairs() {
        let nlist = random_table(200, 10.0, 99, 4.0);

        let r_min: F = 1.5;
        let r_max: F = 4.0;
        let n_bins = 25;
        let rdf = Rdf::new(n_bins, r_max, r_min).unwrap();
        let result = rdf_of(&rdf, &[nlist]).unwrap();

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
        let points = Array2::<F>::zeros((2, 3));
        let nlist = self_neighbors(points.view(), &cube(10.0), 2.0);

        let rdf = Rdf::new(10, 2.0, 0.0).unwrap();
        let result = rdf_of(&rdf, &[nlist]).unwrap();

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
        /// g(r) in bin 1 for the two-particle fixture, factor 2 included.
        const G_BIN1: F = 17.052315331274503;

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
        )
        .with_simbox(cube(10.0));
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
        )
        .with_simbox(cube(10.0));

        let rdf = Rdf::new(4, 4.0, 0.0).unwrap();
        let self_result = fold(&rdf, &[&self_list]).unwrap();
        let cross_result = fold(&rdf, &[&cross_list]).unwrap();

        // Raw counts: the half list holds one pair, the directed list two.
        assert_eq!(self_result.n_r.to_vec(), vec![0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cross_result.n_r.to_vec(), vec![0.0, 2.0, 0.0, 0.0]);
        assert!(self_result.self_query);
        assert!(!cross_result.self_query);

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

    /// `finish` peeks: calling it twice gives the same normalized result.
    #[test]
    fn finish_is_repeatable() {
        let rdf = Rdf::new(20, 4.0, 0.0).unwrap();
        let mut acc = rdf.accumulator();
        acc.push(&random_table(200, 10.0, 42, 4.0)).unwrap();
        assert_eq!(acc.finish().unwrap(), acc.finish().unwrap());
    }

    #[test]
    fn new_validates_inputs() {
        assert!(Rdf::new(0, 1.0, 0.0).is_err());
        assert!(Rdf::new(10, 1.0, -0.1).is_err());
        assert!(Rdf::new(10, 1.0, 1.0).is_err());
        assert!(Rdf::new(10, 0.5, 1.0).is_err());
        assert!(Rdf::new(10, 1.0, 0.0).is_ok());
    }

    /// `with_dimensionality` used to `assert!`; an unsupported value is now
    /// an error.
    #[test]
    fn invalid_dimensionality_is_out_of_range() {
        let rdf = Rdf::new(10, 1.0, 0.0).unwrap();
        for dim in [0u8, 1, 4] {
            assert!(matches!(
                rdf.with_dimensionality(dim),
                Err(ComputeError::OutOfRange { .. })
            ));
        }
        assert_eq!(rdf.with_dimensionality(2).unwrap().dimensionality(), 2);
    }

    /// Sanity check that 2D normalization (`π (r_o² − r_i²)` shells) reproduces
    /// the expected `g(r) → 1` plateau for a random 2-D ideal gas. Mirrors
    /// freud's `freud.density.Rdf` behaviour when `box.is2D == True`.
    #[test]
    fn rdf_2d_orthorhombic_box_plateaus_to_one() {
        // Pack 600 points uniformly in a 10×10 plane (z fixed). Use Lz=1
        // so simbox.volume() returns the 2-D area (Lx · Ly).
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let n = 600;
        let lx: F = 10.0;
        let ly: F = 10.0;
        let lz: F = 1.0;
        let r_max: F = 3.5;
        let n_bins = 35;

        let mut rng = StdRng::seed_from_u64(123);
        let mut points = Array2::<F>::zeros((n, 3));
        for i in 0..n {
            points[[i, 0]] = rng.random::<F>() * lx;
        }
        for i in 0..n {
            points[[i, 1]] = rng.random::<F>() * ly;
        }
        let simbox = SimBox::new(
            Cell::Lengths([lx, ly, lz]),
            [0.0 as F, 0.0, 0.0],
            [true, true, false],
        )
        .unwrap();
        let nlist = self_neighbors(points.view(), &simbox, r_max);

        let rdf = Rdf::new(n_bins, r_max, 0.0)
            .unwrap()
            .with_dimensionality(2)
            .unwrap();
        let result = rdf_of(&rdf, &[nlist]).unwrap();

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
