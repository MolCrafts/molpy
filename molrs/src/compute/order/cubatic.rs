//! Cubatic order parameter via simulated annealing.

use crate::compute::ComputeResult;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;

/// Cubatic calculator. Stateless: SA seed, schedule, and chain count live
/// on the struct.
///
/// Mirrors `freud.order.Cubatic`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/Cubatic.cc)).
///
/// For a set of unit director quaternions, finds the cubic-symmetric
/// global orientation `q*` that maximises the 4-th rank cubatic order
/// parameter
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
    n_steps: usize,
    /// Number of independent SA chains run from different random initial
    /// bases; the best score across chains is returned. Defaults to 4
    /// — robust against local maxima.
    n_chains: usize,
}

impl Default for Cubatic {
    fn default() -> Self {
        Self::new()
    }
}

impl Cubatic {
    /// Defaults: seed 0, initial temp 1.0, cooling rate 0.95, 500 steps, 4 chains.
    pub fn new() -> Self {
        Self {
            seed: 0,
            initial_temp: 1.0,
            cooling_rate: 0.95,
            n_steps: 500,
            n_chains: 4,
        }
    }

    /// RNG seed for the annealing chains.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    /// Initial annealing temperature (dimensionless).
    pub fn with_initial_temp(mut self, t: F) -> Self {
        self.initial_temp = t;
        self
    }
    /// Multiplicative cooling rate per annealing step (0 < r < 1).
    pub fn with_cooling_rate(mut self, r: F) -> Self {
        self.cooling_rate = r;
        self
    }
    /// Annealing steps per chain.
    pub fn with_n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }
    /// Independent annealing chains (min 1); the best result wins.
    pub fn with_n_chains(mut self, n: usize) -> Self {
        self.n_chains = n.max(1);
        self
    }

    fn one_frame(&self, directors: &[[F; 3]]) -> Result<CubaticResult, ComputeError> {
        if directors.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        // Pre-normalise directors so the inner loop is dot products only.
        let units: Vec<[F; 3]> = directors
            .iter()
            .filter_map(|u| {
                let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                if r == 0.0 {
                    None
                } else {
                    Some([u[0] / r, u[1] / r, u[2] / r])
                }
            })
            .collect();
        if units.is_empty() {
            return Err(ComputeError::EmptyInput);
        }

        let mut global_best_basis = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut global_best_score = cubatic_score(&global_best_basis, &units);

        // Each chain starts from a different randomly-rotated basis and
        // runs the same cooling schedule. Best of all chains wins —
        // robust against single-chain trapping in local maxima.
        for chain in 0..self.n_chains {
            // Per-chain RNG seeded deterministically from (self.seed, chain).
            let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(chain as u64));

            // Random initial basis: the lab frame on chain 0, perturbed
            // copies on subsequent chains.
            let mut basis = if chain == 0 {
                [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            } else {
                perturb_basis(
                    &[[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    std::f64::consts::PI,
                    &mut rng,
                )
            };
            let mut score = cubatic_score(&basis, &units);
            let mut best_basis = basis;
            let mut best_score = score;

            let mut temp = self.initial_temp;
            for _ in 0..self.n_steps {
                let trial = perturb_basis(&basis, temp, &mut rng);
                let s = cubatic_score(&trial, &units);
                let d = s - score;
                if d > 0.0 || rng.random::<F>() < (d / temp).exp() {
                    basis = trial;
                    score = s;
                    if score > best_score {
                        best_score = score;
                        best_basis = basis;
                    }
                }
                temp *= self.cooling_rate;
                if temp < 1e-6 {
                    break;
                }
            }

            if best_score > global_best_score {
                global_best_score = best_score;
                global_best_basis = best_basis;
            }
        }

        Ok(CubaticResult {
            order: global_best_score,
            director_basis: global_best_basis,
        })
    }
}

/// `P_4 = ⟨ (û · e_x)^4 + (û · e_y)^4 + (û · e_z)^4 ⟩ − 3/5`.
///
/// Equals `0` for an isotropic director distribution and `+1` when every
/// director points along a single cubic axis.
fn cubatic_score(basis: &[[F; 3]; 3], units: &[[F; 3]]) -> F {
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

/// Apply a small random rotation to `basis` (Rodrigues form with axis on
/// the unit sphere and angle drawn from `[0, temp]`). Output basis is
/// orthonormal up to floating-point.
fn perturb_basis(basis: &[[F; 3]; 3], temp: F, rng: &mut StdRng) -> [[F; 3]; 3] {
    let theta = rng.random::<F>() * temp;
    // Sample uniform axis on the unit sphere.
    let phi = rng.random::<F>() * 2.0 * std::f64::consts::PI;
    let z = rng.random::<F>() * 2.0 - 1.0;
    let r = (1.0_f64 - z * z).max(0.0).sqrt();
    let ax = [r * phi.cos(), r * phi.sin(), z];

    let c = theta.cos();
    let s = theta.sin();
    let omc = 1.0 - c;
    // Rodrigues rotation matrix.
    let r_mat = [
        [
            c + ax[0] * ax[0] * omc,
            ax[0] * ax[1] * omc - ax[2] * s,
            ax[0] * ax[2] * omc + ax[1] * s,
        ],
        [
            ax[1] * ax[0] * omc + ax[2] * s,
            c + ax[1] * ax[1] * omc,
            ax[1] * ax[2] * omc - ax[0] * s,
        ],
        [
            ax[2] * ax[0] * omc - ax[1] * s,
            ax[2] * ax[1] * omc + ax[0] * s,
            c + ax[2] * ax[2] * omc,
        ],
    ];
    // basis' = R · basis (rotate every cubic axis as a column).
    let mut out = [[0.0_f64; 3]; 3];
    for col in 0..3 {
        for row in 0..3 {
            out[row][col] = r_mat[row][0] * basis[0][col]
                + r_mat[row][1] * basis[1][col]
                + r_mat[row][2] * basis[2][col];
        }
    }
    out
}

impl LegacyCompute for Cubatic {
    type Args<'a> = &'a [[F; 3]];
    type Output = Vec<CubaticResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        directors: &'a [[F; 3]],
    ) -> Result<Vec<CubaticResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let mut out = Vec::with_capacity(frames.len());
        for _ in frames {
            out.push(self.one_frame(directors)?);
        }
        Ok(out)
    }
}

/// Per-frame cubatic order parameter result.
#[derive(Debug, Clone, Default)]
pub struct CubaticResult {
    pub order: F,
    /// Rotation matrix from the lab frame into the optimal cubic frame
    /// (columns are the cubic axes).
    pub director_basis: [[F; 3]; 3],
}

impl ComputeResult for CubaticResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;

    fn frame() -> Frame {
        Frame::new()
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
        let r = &Cubatic::new()
            .with_seed(1)
            .with_n_steps(200)
            .compute(&[&frame()], &dirs)
            .unwrap()[0];
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
        let r = &Cubatic::new()
            .with_seed(2)
            .with_n_steps(300)
            .compute(&[&frame()], &dirs)
            .unwrap()[0];
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
        let a = &Cubatic::new()
            .with_seed(7)
            .compute(&[&frame()], &dirs)
            .unwrap()[0];
        let b = &Cubatic::new()
            .with_seed(7)
            .compute(&[&frame()], &dirs)
            .unwrap()[0];
        assert_eq!(a.order, b.order);
    }

    #[test]
    fn empty_directors_error() {
        let err = Cubatic::new().compute(&[&frame()], &[]).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
