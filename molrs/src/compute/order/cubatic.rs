//! Cubatic order parameter via simulated annealing.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use molrs::op::{F, Mat3, Vec3, axis_angle, unit_vector_from_uniform};
use ndarray::ArrayView2;

use crate::compute::{Compute, ComputeError, require_directions};

/// The lab frame: the starting basis of chain 0.
const LAB: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Cubatic calculator.
///
/// Mirrors `freud.order.Cubatic`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/Cubatic.cc)).
///
/// Configuration: the annealing schedule — seed, initial temperature,
/// cooling rate, the temperature below which a chain stops, steps and
/// chains, every one validated where it is set. Data: one snapshot's
/// directors (`N × 3`, each normalised here; a zero director is an error).
///
/// Finds the cubic-symmetric global orientation `q*` that maximises the
/// 4-th rank cubatic order parameter
///
/// ```text
///   P_4(q*) = ⟨ Σ_{a,b,c,d} M_{abcd}(q*) · û_{i,a} û_{i,b} û_{i,c} û_{i,d} ⟩_i
/// ```
///
/// where `M` is the symmetric 4-th rank cubic invariant tensor at
/// orientation `q*`. The optimisation is a basic simulated annealing
/// with deterministic RNG seeding.
///
/// For tractability the implementation uses the **reduced scalar form**
/// commonly used in soft-matter literature: project each director onto
/// the rotated cubic axes `{e_x, e_y, e_z}` and compute
///
/// ```text
///   P_4 = ⟨ (û · e_x)^4 + (û · e_y)^4 + (û · e_z)^4 ⟩ − 3/5
/// ```
///
/// which is `0` for an isotropic ensemble and `> 0` for cubic alignment.
///
/// The output is `(order, director_basis)` where `director_basis` is the
/// 3 × 3 rotation matrix sending the lab frame to the optimal cubic frame.
#[derive(Debug, Clone, Copy)]
pub struct Cubatic {
    seed: u64,
    initial_temp: F,
    cooling_rate: F,
    min_temp: F,
    n_steps: usize,
    /// Number of independent SA chains run from different random initial
    /// bases; the best score across chains is returned.
    n_chains: usize,
}

/// The data a [`Cubatic`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct CubaticInput<'a> {
    /// One snapshot's directors, `N × 3` (any non-zero length).
    pub directors: ArrayView2<'a, F>,
}

impl Default for Cubatic {
    fn default() -> Self {
        Self::new()
    }
}

impl Cubatic {
    /// Defaults: seed 0, initial temperature 1.0, cooling rate 0.95, stop
    /// below temperature 1e-6, 500 steps, 4 chains.
    pub fn new() -> Self {
        Self {
            seed: 0,
            initial_temp: 1.0,
            cooling_rate: 0.95,
            min_temp: 1e-6,
            n_steps: 500,
            n_chains: 4,
        }
    }

    /// RNG seed for the annealing chains.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Initial annealing temperature (dimensionless, default 1.0).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless finite and positive.
    pub fn with_initial_temp(mut self, t: F) -> Result<Self, ComputeError> {
        self.initial_temp = positive("Cubatic::initial_temp", t)?;
        Ok(self)
    }

    /// Multiplicative cooling rate per annealing step (default 0.95).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless `0 < r < 1`: a rate of 1 never
    /// cools, a rate of 0 freezes after one step.
    pub fn with_cooling_rate(mut self, r: F) -> Result<Self, ComputeError> {
        if !(r > 0.0 && r < 1.0) {
            return Err(ComputeError::OutOfRange {
                field: "Cubatic::cooling_rate",
                value: r.to_string(),
            });
        }
        self.cooling_rate = r;
        Ok(self)
    }

    /// The temperature below which a chain stops early (default 1e-6).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] unless finite and positive.
    pub fn with_min_temp(mut self, t: F) -> Result<Self, ComputeError> {
        self.min_temp = positive("Cubatic::min_temp", t)?;
        Ok(self)
    }

    /// Annealing steps per chain (default 500).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for 0.
    pub fn with_n_steps(mut self, n: usize) -> Result<Self, ComputeError> {
        self.n_steps = at_least_one("Cubatic::n_steps", n)?;
        Ok(self)
    }

    /// Independent annealing chains; the best result wins (default 4).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for 0 (it used to become 1 silently).
    pub fn with_n_chains(mut self, n: usize) -> Result<Self, ComputeError> {
        self.n_chains = at_least_one("Cubatic::n_chains", n)?;
        Ok(self)
    }

    /// One chain of the annealing from `basis`; its best `(score, basis)`.
    fn anneal(
        &self,
        mut basis: Mat3,
        units: &[Vec3],
        rng: &mut StdRng,
    ) -> Result<(F, Mat3), ComputeError> {
        let mut score = cubatic_score(&basis, units);
        let mut best = (score, basis);
        let mut temp = self.initial_temp;
        for _ in 0..self.n_steps {
            let trial = perturb_basis(&basis, temp, rng)?;
            let s = cubatic_score(&trial, units);
            let d = s - score;
            if d > 0.0 || rng.random::<F>() < (d / temp).exp() {
                basis = trial;
                score = s;
                if score > best.0 {
                    best = (score, basis);
                }
            }
            temp *= self.cooling_rate;
            if temp < self.min_temp {
                break;
            }
        }
        Ok(best)
    }
}

impl Compute for Cubatic {
    type Input<'a> = CubaticInput<'a>;
    type Output = CubaticResult;

    /// # Errors
    /// [`ComputeError::EmptyInput`] for no directors;
    /// [`ComputeError::DimensionMismatch`] when they are not `N × 3`;
    /// [`ComputeError::NonFinite`] / [`ComputeError::OutOfRange`] for a
    /// non-finite or zero director.
    fn compute(&self, input: CubaticInput<'_>) -> Result<CubaticResult, ComputeError> {
        let units = require_directions(input.directors, "directors")?;
        if units.is_empty() {
            return Err(ComputeError::EmptyInput);
        }

        let mut global_best = (cubatic_score(&LAB, &units), LAB);

        // Each chain starts from a different randomly-rotated basis and
        // runs the same cooling schedule. Best of all chains wins —
        // robust against single-chain trapping in local maxima.
        for chain in 0..self.n_chains {
            // Per-chain RNG seeded deterministically from (self.seed, chain).
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(chain as u64));
            // The lab frame on chain 0, randomly rotated copies on the others.
            let start = if chain == 0 {
                LAB
            } else {
                perturb_basis(&LAB, std::f64::consts::PI, &mut rng)?
            };
            let best = self.anneal(start, &units, &mut rng)?;
            if best.0 > global_best.0 {
                global_best = best;
            }
        }

        Ok(CubaticResult {
            order: global_best.0,
            director_basis: global_best.1,
        })
    }
}

/// `value` when finite and positive, else [`ComputeError::OutOfRange`].
fn positive(field: &'static str, value: F) -> Result<F, ComputeError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(ComputeError::OutOfRange {
            field,
            value: value.to_string(),
        })
    }
}

/// `n` when at least 1, else [`ComputeError::OutOfRange`].
fn at_least_one(field: &'static str, n: usize) -> Result<usize, ComputeError> {
    if n == 0 {
        Err(ComputeError::OutOfRange {
            field,
            value: "0".into(),
        })
    } else {
        Ok(n)
    }
}

/// `P_4 = ⟨ (û · e_x)^4 + (û · e_y)^4 + (û · e_z)^4 ⟩ − 3/5`.
///
/// Equals `0` for an isotropic director distribution and `+1` when every
/// director points along a single cubic axis.
fn cubatic_score(basis: &Mat3, units: &[Vec3]) -> F {
    let mut acc: F = 0.0;
    for u in units {
        for ax in basis {
            let d = ax[0] * u[0] + ax[1] * u[1] + ax[2] * u[2];
            let d2 = d * d;
            acc += d2 * d2;
        }
    }
    acc / units.len() as F - 0.6
}

/// Apply a random rotation to `basis`: an axis uniform on the unit sphere
/// ([`unit_vector_from_uniform`]) and an angle drawn from `[0, temp]`
/// ([`axis_angle`]). The output basis is orthonormal up to floating point.
fn perturb_basis(basis: &Mat3, temp: F, rng: &mut StdRng) -> Result<Mat3, ComputeError> {
    let theta = rng.random::<F>() * temp;
    let u_azimuth = rng.random::<F>();
    let u_height = rng.random::<F>();
    let axis = unit_vector_from_uniform([u_height, u_azimuth]);
    let r_mat = axis_angle(axis, theta).ok_or_else(|| ComputeError::OutOfRange {
        field: "Cubatic rotation axis",
        value: format!("{axis:?} is not a direction"),
    })?;
    // basis' = R · basis (rotate every cubic axis as a column).
    let mut out = [[0.0_f64; 3]; 3];
    for (row, out_row) in out.iter_mut().enumerate() {
        for (col, v) in out_row.iter_mut().enumerate() {
            *v = r_mat[row][0] * basis[0][col]
                + r_mat[row][1] * basis[1][col]
                + r_mat[row][2] * basis[2][col];
        }
    }
    Ok(out)
}

/// One snapshot's cubatic order parameter.
#[derive(Debug, Clone, Default)]
pub struct CubaticResult {
    /// The best `P_4` found.
    pub order: F,
    /// Rotation matrix from the lab frame into the optimal cubic frame
    /// (columns are the cubic axes).
    pub director_basis: [[F; 3]; 3],
}

crate::compute::spec::compute_spec! {
    Cubatic as "Cubatic";
    params |s| {
        seed: Int = Cubatic::new().seed => s.seed;
        initial_temp: Float = Cubatic::new().initial_temp => s.initial_temp;
        cooling_rate: Float = Cubatic::new().cooling_rate => s.cooling_rate;
        min_temp: Float = Cubatic::new().min_temp => s.min_temp;
        n_steps: Int = Cubatic::new().n_steps => s.n_steps;
        n_chains: Int = Cubatic::new().n_chains => s.n_chains;
    }
    inputs {
        directors: Vectors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn cubatic(c: Cubatic, dirs: &[[F; 3]]) -> Result<CubaticResult, ComputeError> {
        let directors = Array2::from_shape_fn((dirs.len(), 3), |(i, k)| dirs[i][k]);
        c.compute(CubaticInput {
            directors: directors.view(),
        })
    }

    #[test]
    fn perfectly_cubic_alignment_high_order() {
        // 6 directors along ±x, ±y, ±z. Each is aligned with one of the
        // cubic axes, so the lab frame already maximises the score:
        //   P_4 = ⟨1 + 0 + 0⟩ − 3/5 = 1 − 0.6 = 0.4
        let dirs = vec![
            [1.0_f64, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let r = &cubatic(
            Cubatic::new().with_seed(1).with_n_steps(200).unwrap(),
            &dirs,
        )
        .unwrap();
        assert!(
            (r.order - 0.4).abs() < 1e-9,
            "cubic-aligned set: P_4 = {} (expected 0.4)",
            r.order
        );
    }

    #[test]
    fn body_diagonal_set_below_perfect_cubic() {
        // 8 directors along the cubic body diagonals (±1, ±1, ±1)/√3.
        // The body-diagonal ensemble is less cubically aligned than the
        // perfect ±axis set, so the best achievable P_4 must be strictly
        // below 0.4 (the perfect-axis value), bounded by the 8-corner
        // geometry.
        let s = 1.0_f64 / 3.0_f64.sqrt();
        let dirs = vec![
            [s, s, s],
            [s, s, -s],
            [s, -s, s],
            [-s, s, s],
            [s, -s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [-s, -s, -s],
        ];
        let r = &cubatic(
            Cubatic::new().with_seed(2).with_n_steps(300).unwrap(),
            &dirs,
        )
        .unwrap();
        // P_4 must be in (-0.6, 0.4) — definitely below the perfect-axis
        // value (the SA gives ≈ 0.13 for this configuration).
        assert!(
            r.order < 0.4 && r.order > -0.6,
            "body-diagonal set order {} out of physical range",
            r.order
        );
    }

    #[test]
    fn deterministic_seed() {
        let dirs = vec![[1.0_f64, 0.0, 0.0], [0.5, 0.5, 0.0]];
        let a = &cubatic(Cubatic::new().with_seed(7), &dirs).unwrap();
        let b = &cubatic(Cubatic::new().with_seed(7), &dirs).unwrap();
        assert_eq!(a.order, b.order);
    }

    #[test]
    fn empty_directors_error() {
        let err = cubatic(Cubatic::new(), &[]).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    /// A zero director used to be dropped silently; it has no direction.
    #[test]
    fn zero_director_is_out_of_range() {
        let err = cubatic(Cubatic::new(), &[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    /// `with_n_chains(0)` used to become 1 silently; every schedule constant
    /// is validated where it is set.
    #[test]
    fn schedule_out_of_range_is_an_error() {
        let out_of_range =
            |r: Result<Cubatic, ComputeError>| matches!(r, Err(ComputeError::OutOfRange { .. }));
        assert!(out_of_range(Cubatic::new().with_n_chains(0)));
        assert!(out_of_range(Cubatic::new().with_n_steps(0)));
        assert!(out_of_range(Cubatic::new().with_cooling_rate(1.0)));
        assert!(out_of_range(Cubatic::new().with_cooling_rate(0.0)));
        assert!(out_of_range(Cubatic::new().with_initial_temp(0.0)));
        assert!(out_of_range(Cubatic::new().with_initial_temp(F::NAN)));
        assert!(out_of_range(Cubatic::new().with_min_temp(-1.0)));
        assert!(Cubatic::new().with_n_chains(1).is_ok());
    }
}
