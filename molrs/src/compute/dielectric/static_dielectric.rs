//! [`StaticDielectric`] — the static permittivity from dipole fluctuations.

use molrs::core::UnitPreset;
use molrs::op::F;
use ndarray::{Array1, ArrayView2};

use super::{DielectricPrefactor, EwaldBoundary};
use crate::compute::{Compute, ComputeError};

/// Static dielectric constant from the Neumann fluctuation formula,
///
/// ```text
/// ε(0) = ε_∞ + 4π κ ⟨|M − ⟨M⟩|²⟩ / (3 V k_B T)
/// ```
///
/// isotropically and per Cartesian axis (`ε_d = ε_∞ + 4π κ ⟨δM_d²⟩ /
/// (V k_B T)`, the MDAnalysis `DielectricConstant` layout). `κ = 1/(4πε₀)`
/// and `k_B` come from the configured [`UnitPreset`]; in the `real` preset
/// the dipole series is in e·Å, the volume in Å³ and the temperature in K.
/// The formula holds under conducting (tin-foil) Ewald boundaries; the
/// boundary is data of the trajectory and a vacuum boundary is refused.
///
/// The variance is the two-pass centred one, stable when `|M| ≫ |δM|`.
///
/// Reference: Neumann, *Mol. Phys.* **50**, 841 (1983).
#[derive(Debug, Clone)]
pub struct StaticDielectric {
    units: UnitPreset,
    epsilon_inf: F,
}

/// The dipole trajectory and state a [`StaticDielectric`] reads.
#[derive(Debug, Clone, Copy)]
pub struct StaticDielectricInput<'a> {
    /// Total dipole moment per frame, `T × 3`, from unwrapped coordinates.
    pub dipoles: ArrayView2<'a, F>,
    /// Volume of the cell.
    pub volume: F,
    /// Temperature of the state.
    pub temperature: F,
    /// Ewald boundary the trajectory was sampled under.
    pub boundary: EwaldBoundary,
}

/// Static permittivity, isotropic and per axis, with the dipole statistics.
#[derive(Debug, Clone)]
pub struct StaticDielectricResult {
    /// Isotropic static permittivity ε(0), dimensionless.
    pub epsilon: F,
    /// Per-axis permittivity ε_d (d = x, y, z); their mean is `epsilon`.
    pub epsilon_axes: Array1<F>,
    /// Per-axis mean dipole ⟨M_d⟩.
    pub dipole_mean: Array1<F>,
    /// Per-axis mean squared dipole ⟨M_d²⟩.
    pub dipole_sq_mean: Array1<F>,
    /// Per-axis fluctuation ⟨M_d²⟩ − ⟨M_d⟩².
    pub fluctuation: Array1<F>,
    /// The high-frequency permittivity ε_∞ used.
    pub epsilon_inf: F,
    /// Number of frames analysed.
    pub n_frames: usize,
}

impl StaticDielectric {
    /// The high-frequency (electronic) permittivity ε_∞ when none is given:
    /// 1, a non-polarizable model. The ε(ω) routes share it.
    pub const DEFAULT_EPSILON_INF: F = 1.0;

    /// The static permittivity in `units`, with `ε_∞ = 1` (a non-polarizable
    /// force field).
    pub fn new(units: UnitPreset) -> Self {
        Self {
            units,
            epsilon_inf: Self::DEFAULT_EPSILON_INF,
        }
    }

    /// The high-frequency (electronic) permittivity ε_∞ the fluctuation term
    /// is added to.
    pub fn with_epsilon_inf(mut self, epsilon_inf: F) -> Self {
        self.epsilon_inf = epsilon_inf;
        self
    }

    /// The unit system.
    pub fn units(&self) -> &UnitPreset {
        &self.units
    }

    /// ε_∞.
    pub fn epsilon_inf(&self) -> F {
        self.epsilon_inf
    }
}

impl Compute for StaticDielectric {
    type Input<'a> = StaticDielectricInput<'a>;
    type Output = StaticDielectricResult;

    /// # Errors
    /// * [`ComputeError::DimensionMismatch`] unless the dipoles are `T × 3`.
    /// * [`ComputeError::EmptyInput`] for fewer than two frames.
    /// * [`ComputeError::NonFinite`] for a non-finite dipole or ε_∞.
    /// * [`ComputeError::OutOfRange`] for a non-positive volume or
    ///   temperature, or a vacuum boundary.
    fn compute(
        &self,
        input: StaticDielectricInput<'_>,
    ) -> Result<StaticDielectricResult, ComputeError> {
        let dm = input.dipoles;
        if dm.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: dm.ncols(),
                what: "dipoles (expected (n_frames, 3))",
            });
        }
        let n_frames = dm.nrows();
        if n_frames < 2 {
            return Err(ComputeError::EmptyInput);
        }
        if let Some(index) = dm.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "dipoles",
                index,
            });
        }
        if !self.epsilon_inf.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "epsilon_inf",
                index: 0,
            });
        }
        input.boundary.require_supported()?;
        let prefactor = DielectricPrefactor::new(&self.units, input.volume, input.temperature)?;

        let n = n_frames as F;
        let mut mean_m = Array1::<F>::zeros(3);
        for t in 0..n_frames {
            for d in 0..3 {
                mean_m[d] += dm[[t, d]];
            }
        }
        mean_m /= n;

        let mut variance = 0.0;
        let mut fluctuation = Array1::<F>::zeros(3);
        let mut mean_sq = Array1::<F>::zeros(3);
        for t in 0..n_frames {
            for d in 0..3 {
                let m = dm[[t, d]];
                let dev = m - mean_m[d];
                variance += dev * dev;
                fluctuation[d] += dev * dev;
                mean_sq[d] += m * m;
            }
        }
        variance /= n;
        fluctuation /= n;
        mean_sq /= n;

        let a = prefactor.fluctuation();
        // A diagonal tensor component carries the whole fluctuation of one
        // axis, three times the isotropic per-axis share.
        let per_axis = 3.0 * a;
        Ok(StaticDielectricResult {
            epsilon: self.epsilon_inf + a * variance,
            epsilon_axes: fluctuation.mapv(|f| self.epsilon_inf + per_axis * f),
            dipole_mean: mean_m,
            dipole_sq_mean: mean_sq,
            fluctuation,
            epsilon_inf: self.epsilon_inf,
            n_frames,
        })
    }
}

crate::compute::spec::compute_spec! {
    StaticDielectric as "StaticDielectric";
    params |s| {
        units: Units;
        epsilon_inf: Float = StaticDielectric::DEFAULT_EPSILON_INF => s.epsilon_inf;
    }
    inputs {
        dipoles: Series;
        volume: Scalar;
        temperature: Scalar;
        boundary: Choice(EwaldBoundary::NAMES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::FOUR_THIRDS_PI;
    use molrs::core::constants::COULOMB_REAL as KAPPA;
    use ndarray::Array2;

    fn static_eps(
        dm: &Array2<F>,
        volume: F,
        temperature: F,
        eps_inf: F,
    ) -> Result<StaticDielectricResult, ComputeError> {
        StaticDielectric::new(UnitPreset::real())
            .with_epsilon_inf(eps_inf)
            .compute(StaticDielectricInput {
                dipoles: dm.view(),
                volume,
                temperature,
                boundary: EwaldBoundary::TinFoil,
            })
    }

    #[test]
    fn test_static_dielectric_zero_fluctuation() {
        let dm = Array2::from_elem((10, 3), 0.0);
        let eps = static_eps(&dm, 1000.0, 300.0, 1.0).unwrap().epsilon;
        assert!((eps - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_known_fluctuation() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let eps = static_eps(&dm, 1000.0, 300.0, 1.0).unwrap().epsilon;
        // ⟨M⟩ = 0, ⟨M²⟩ = (1²+(-1)²)/2 = 1.0
        // ε(0) = 1.0 + (4π/3)*332.0637*1.0/(1000*1.9872e-3*300)
        let expected =
            1.0 + FOUR_THIRDS_PI * KAPPA * 1.0 / (1000.0 * UnitPreset::real().boltzmann() * 300.0);
        assert!((eps - expected).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_single_frame_rejected() {
        let dm = Array2::zeros((1, 3));
        assert!(matches!(
            static_eps(&dm, 1000.0, 300.0, 1.0),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn test_static_dielectric_components() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        let result = static_eps(&dm, 1000.0, 300.0, 1.0).unwrap();
        assert_eq!(result.dipole_mean.len(), 3);
        assert_eq!(result.epsilon_axes.len(), 3);
        assert_eq!(result.n_frames, 2);
        // M fluctuates in x only → ε_x > ε_y, ε_z
        assert!(result.epsilon_axes[0] > result.epsilon_axes[1]);
        assert!((result.epsilon_axes[1] - 1.0).abs() < 1e-10);
        assert!((result.epsilon_axes[2] - 1.0).abs() < 1e-10);
        // The per-axis mean matches the isotropic value.
        assert!((result.epsilon_axes.mean().unwrap() - result.epsilon).abs() < 1e-10);
    }

    #[test]
    fn test_static_dielectric_components_isotropic() {
        // Isotropic dipole (equal in all directions) → ε_x = ε_y = ε_z = ε_mean.
        let dm = ndarray::arr2(&[[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]);
        let result = static_eps(&dm, 1000.0, 300.0, 1.0).unwrap();
        let eps_x = result.epsilon_axes[0];
        assert!((result.epsilon_axes[1] - eps_x).abs() < 1e-12);
        assert!((result.epsilon_axes[2] - eps_x).abs() < 1e-12);
        assert!((result.epsilon - eps_x).abs() < 1e-12);
    }

    #[test]
    fn test_static_dielectric_components_large_mean_two_pass() {
        // M = M0 + small noise: one-pass variance fails; two-pass keeps Var≈1.
        let m0 = 1.0e8;
        let n = 1000;
        let mut dm = Array2::<f64>::zeros((n, 3));
        for t in 0..n {
            // unit-variance square wave on x only
            let noise = if t % 2 == 0 { 1.0 } else { -1.0 };
            dm[[t, 0]] = m0 + noise;
            dm[[t, 1]] = m0;
            dm[[t, 2]] = m0;
        }
        let result = static_eps(&dm, 1000.0, 300.0, 1.0).unwrap();
        // fluctuation_x should be ~1.0 (exact for ±1 about mean)
        assert!(
            (result.fluctuation[0] - 1.0).abs() < 1e-9,
            "two-pass Var_x = {}",
            result.fluctuation[0]
        );
        assert!(result.fluctuation[1].abs() < 1e-12);
        assert!(result.fluctuation[2].abs() < 1e-12);
        assert!((result.epsilon_axes.mean().unwrap() - result.epsilon).abs() < 1e-8);
    }

    #[test]
    fn state_and_boundary_are_checked() {
        let dm = ndarray::arr2(&[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]);
        assert!(matches!(
            static_eps(&dm, 0.0, 300.0, 1.0),
            Err(ComputeError::OutOfRange {
                field: "volume",
                ..
            })
        ));
        assert!(matches!(
            static_eps(&dm, 1000.0, -1.0, 1.0),
            Err(ComputeError::OutOfRange {
                field: "temperature",
                ..
            })
        ));
        assert!(matches!(
            StaticDielectric::new(UnitPreset::real()).compute(StaticDielectricInput {
                dipoles: dm.view(),
                volume: 1000.0,
                temperature: 300.0,
                boundary: EwaldBoundary::Vacuum,
            }),
            Err(ComputeError::OutOfRange {
                field: "boundary",
                ..
            })
        ));
        assert!(matches!(
            static_eps(&Array2::zeros((4, 2)), 1000.0, 300.0, 1.0),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
