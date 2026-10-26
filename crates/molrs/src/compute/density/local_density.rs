//! Per-particle local number density in a sphere of radius `r_max`.

use molrs::core::FOUR_THIRDS_PI;
use molrs::core::{PairSource, QueryMode};
use molrs::op::F;

use crate::compute::{Compute, ComputeError, require_dist_sq};

/// Local-density calculator.
///
/// Mirrors `freud.density.LocalDensity`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/LocalDensity.cc)).
///
/// Configuration: the cutoff `r_max` and the smoothing `diameter`. Data: one
/// snapshot's pair source. For each query point `i` the analyzer counts
/// neighbors within `r_max` and reports `density_i = count_i / (4/3 π
/// r_max³)` (number per unit volume). A half-shell self-query credits each
/// pair to both ends; a cross-query credits the query point.
///
/// freud also supports a `diameter` correction that subtracts the "hard-
/// sphere fraction" near the edge of the cutoff sphere — for two unit
/// spheres at distance `r` the overlap on the boundary linearly interpolates
/// a partial count between 0 and 1. We replicate that smoothing: each
/// neighbor `j` contributes
///
/// ```text
///   weight_j = clamp((r_max + diameter/2 − r_ij) / diameter, 0, 1)
/// ```
///
/// which collapses to the standard `1.0` count when `diameter = 0`. The
/// identical formula appears in `LocalDensity::compute` in freud. The source
/// must reach `r_max + diameter/2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalDensity {
    r_max: F,
    diameter: F,
}

/// The data a [`LocalDensity`] analyses.
#[derive(Clone, Copy)]
pub struct LocalDensityInput<'a> {
    /// The snapshot's pairs; must carry the `dist_sq` column.
    pub pairs: &'a (dyn PairSource + Sync),
}

impl LocalDensity {
    /// The hard-sphere smoothing diameter when not set: 0, a sharp cutoff.
    pub const DEFAULT_DIAMETER: F = 0.0;

    /// New analyzer with a cutoff `r_max` and no smoothing (`diameter = 0`).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or NaN `r_max`.
    pub fn new(r_max: F) -> Result<Self, ComputeError> {
        if r_max.is_nan() || r_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "LocalDensity::r_max",
                value: r_max.to_string(),
            });
        }
        Ok(Self {
            r_max,
            diameter: Self::DEFAULT_DIAMETER,
        })
    }

    /// Apply a "hard-sphere diameter" smoothing at the cutoff edge.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a negative or non-finite diameter.
    pub fn with_diameter(mut self, diameter: F) -> Result<Self, ComputeError> {
        if !(diameter.is_finite() && diameter >= 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "LocalDensity::diameter",
                value: diameter.to_string(),
            });
        }
        self.diameter = diameter;
        Ok(self)
    }

    /// The cutoff, Å.
    pub fn r_max(&self) -> F {
        self.r_max
    }

    /// The smoothing diameter, Å.
    pub fn diameter(&self) -> F {
        self.diameter
    }
}

impl Compute for LocalDensity {
    type Input<'a> = LocalDensityInput<'a>;
    type Output = LocalDensityResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a source without `dist_sq`;
    /// [`ComputeError::NonFinite`] for a non-finite distance.
    fn compute(&self, input: LocalDensityInput<'_>) -> Result<LocalDensityResult, ComputeError> {
        let pairs = input.pairs;
        let (n_query, symmetric) = match pairs.mode() {
            QueryMode::SelfQuery { n_points } => (n_points, true),
            QueryMode::CrossQuery { n_query_points, .. } => (n_query_points, false),
        };
        require_dist_sq(pairs)?;

        let mut num = vec![0.0_f64; n_query];
        let half_diam = self.diameter * 0.5;
        let inv_diam = if self.diameter > 0.0 {
            1.0 / self.diameter
        } else {
            0.0
        };
        let mut failure: Option<ComputeError> = None;
        let mut k = 0usize;
        pairs.visit_pairs(&mut |i, j, dist_sq, _| {
            let index = k;
            k += 1;
            if failure.is_some() {
                return;
            }
            let Some(d2) = dist_sq else {
                return;
            };
            if !d2.is_finite() {
                failure = Some(ComputeError::NonFinite {
                    where_: "pair dist_sq",
                    index,
                });
                return;
            }
            let r = d2.sqrt();
            if r > self.r_max + half_diam {
                return;
            }
            let weight = if self.diameter == 0.0 {
                if r <= self.r_max { 1.0 } else { 0.0 }
            } else {
                ((self.r_max + half_diam - r) * inv_diam).clamp(0.0, 1.0)
            };
            let (i, j) = (i as usize, j as usize);
            if i >= n_query || (symmetric && j >= n_query) {
                failure = Some(ComputeError::OutOfRange {
                    field: "pair index",
                    value: format!("({i}, {j}) of pair {index} past the source's point counts"),
                });
                return;
            }
            num[i] += weight;
            if symmetric {
                num[j] += weight;
            }
        });
        if let Some(err) = failure {
            return Err(err);
        }

        let inv_vol = 1.0 / (FOUR_THIRDS_PI * self.r_max.powi(3));
        let density: Vec<F> = num.iter().map(|&c| c * inv_vol).collect();
        Ok(LocalDensityResult {
            n_neighbors: num,
            density,
        })
    }
}

/// One snapshot's local densities.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct LocalDensityResult {
    /// Fractional (or integer when `diameter = 0`) neighbor count per query
    /// point.
    pub n_neighbors: Vec<F>,
    /// Number density per query point: `n_neighbors / (4/3 π r_max³)`.
    pub density: Vec<F>,
}

crate::compute::spec::compute_spec! {
    LocalDensity as "LocalDensity";
    params |s| {
        r_max: Float;
        diameter: Float = LocalDensity::DEFAULT_DIAMETER => s.diameter;
    }
    inputs {
        pairs: PairSource;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::self_neighbors;
    use crate::core::Cell;
    use molrs::core::{Neighbors, SimBox};
    use ndarray::Array2;

    fn table(positions: &[[F; 3]], box_len: F, cutoff: F) -> Neighbors {
        let points = Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]);
        let simbox = SimBox::new(
            Cell::Lengths([box_len; 3]),
            [0.0 as F, 0.0 as F, 0.0 as F],
            [false; 3],
        )
        .unwrap();
        self_neighbors(points.view(), &simbox, cutoff)
    }

    fn density(ld: &LocalDensity, nl: &Neighbors) -> LocalDensityResult {
        ld.compute(LocalDensityInput { pairs: nl }).unwrap()
    }

    #[test]
    fn isolated_particle_has_zero_density() {
        let nl = table(&[[5.0, 5.0, 5.0]], 20.0, 2.0);
        let r = density(&LocalDensity::new(2.0).unwrap(), &nl);
        assert_eq!(r.n_neighbors[0], 0.0);
        assert_eq!(r.density[0], 0.0);
    }

    #[test]
    fn pair_counts_symmetric_in_self_query() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0, 2.0);
        let r = density(&LocalDensity::new(2.0).unwrap(), &nl);
        // Each of the two particles has exactly 1 neighbor within 2.0.
        assert!((r.n_neighbors[0] - 1.0).abs() < 1e-12);
        assert!((r.n_neighbors[1] - 1.0).abs() < 1e-12);
        let v = FOUR_THIRDS_PI * 2.0_f64.powi(3);
        assert!((r.density[0] - 1.0 / v).abs() < 1e-12);
    }

    #[test]
    fn diameter_smoothing_clamps_at_edge() {
        // Two particles at exactly r = r_max + diameter/2 → weight ≈ 0.
        let r_max = 3.0;
        let diameter = 1.0;
        let r = r_max + diameter / 2.0 - 1e-9;
        let nl = table(&[[0.0, 0.0, 0.0], [r, 0.0, 0.0]], 20.0, r + 1.0);
        let ld = LocalDensity::new(r_max)
            .unwrap()
            .with_diameter(diameter)
            .unwrap();
        let res = density(&ld, &nl);
        assert!(
            res.n_neighbors[0] < 1e-8,
            "weight at edge should be ≈ 0, got {}",
            res.n_neighbors[0]
        );

        // At r = r_max − diameter/2 → weight = 1 (fully inside).
        let r_in = r_max - diameter / 2.0;
        let nl2 = table(&[[0.0, 0.0, 0.0], [r_in, 0.0, 0.0]], 20.0, r + 1.0);
        let res2 = density(&ld, &nl2);
        assert!((res2.n_neighbors[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_r_max_errors() {
        assert!(LocalDensity::new(0.0).is_err());
        assert!(LocalDensity::new(-1.0).is_err());
    }

    /// A negative diameter used to pass a `debug_assert` in release builds;
    /// it is an error.
    #[test]
    fn invalid_diameter_is_out_of_range() {
        let ld = LocalDensity::new(2.0).unwrap();
        for bad in [-1.0, F::NAN, F::INFINITY] {
            assert!(matches!(
                ld.with_diameter(bad),
                Err(ComputeError::OutOfRange { .. })
            ));
        }
    }
}
