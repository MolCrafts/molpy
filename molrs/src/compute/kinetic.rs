//! Kinetic readings of one configuration: [`KineticEnergy`] and its
//! equipartition temperature. (The centre-of-mass velocity is
//! [`CenterOfMass`](crate::compute::CenterOfMass) on the velocities.)

use molrs::core::{UnitPreset, UnitRegistry};
use ndarray::{ArrayView1, ArrayView2, Zip};

use crate::compute::{Compute, ComputeError};
use crate::op::F;

fn check_rows(mass: ArrayView1<'_, F>, vel: ArrayView2<'_, F>) -> Result<(), ComputeError> {
    if vel.ncols() != 3 {
        return Err(ComputeError::BadShape {
            expected: "velocities of shape (n_atoms, 3)".into(),
            got: format!("{:?}", vel.shape()),
        });
    }
    if mass.len() != vel.nrows() {
        return Err(ComputeError::DimensionMismatch {
            expected: vel.nrows(),
            got: mass.len(),
            what: "mass length against the velocity rows",
        });
    }
    Ok(())
}

/// Kinetic energy `K = ½ Σᵢ mᵢ |vᵢ|² · mvv2e` of one configuration.
///
/// Two constants configure it: `mvv2e`, the `mass · velocity²` → `energy`
/// factor (LAMMPS' name; 2390.06 in `real`, exactly 1 in `si`), and the
/// Boltzmann constant `k_B` in that energy unit, which gives the
/// equipartition temperature through [`KineticEnergyResult::temperature`].
/// [`new`](Self::new) reads both from a [`UnitPreset`];
/// [`with_constants`](Self::with_constants) takes them as given, for a caller
/// that works in its own consistent units (an MD driver passes `mvv2e = 1`
/// and its own `k_B`).
///
/// ```
/// use molrs::compute::{Compute, KineticEnergy, KineticEnergyInput};
/// use molrs::core::UnitPreset;
/// use ndarray::array;
///
/// let masses = array![2.0, 1.0];                               // kg
/// let velocities = array![[1.0, 0.0, 0.0], [0.0, 2.0, 2.0]];   // m/s
/// let k = KineticEnergy::new(UnitPreset::si())?
///     .compute(KineticEnergyInput { masses: masses.view(), velocities: velocities.view() })?;
/// assert_eq!(k.energy, 5.0);                                   // J
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone)]
pub struct KineticEnergy {
    mvv2e: F,
    boltzmann: F,
}

/// The configuration a [`KineticEnergy`] reads.
#[derive(Debug, Clone, Copy)]
pub struct KineticEnergyInput<'a> {
    /// Per-atom masses, length `N`.
    pub masses: ArrayView1<'a, F>,
    /// Velocities, `N × 3`.
    pub velocities: ArrayView2<'a, F>,
}

/// A kinetic energy, with the Boltzmann constant of its unit system.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct KineticEnergyResult {
    /// `K` in the preset's energy unit.
    pub energy: F,
    /// Boltzmann's constant in the preset's energy per temperature unit.
    pub boltzmann: F,
}

impl KineticEnergyResult {
    /// Kinetic temperature `T = 2K / (n_dof k_B)` by equipartition: `n_dof`
    /// is the number of unconstrained degrees of freedom (`3N` less any
    /// removed by constraints or a fixed centre of mass).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when `n_dof` is zero or `k_B` is not
    /// positive.
    pub fn temperature(&self, n_dof: usize) -> Result<F, ComputeError> {
        if n_dof == 0 {
            return Err(ComputeError::OutOfRange {
                field: "n_dof",
                value: n_dof.to_string(),
            });
        }
        if !(self.boltzmann.is_finite() && self.boltzmann > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "kb",
                value: self.boltzmann.to_string(),
            });
        }
        Ok(2.0 * self.energy / (n_dof as F * self.boltzmann))
    }
}

impl KineticEnergy {
    /// The kinetic energy in `units`: masses and velocities in the preset's
    /// mass and velocity units, the energy in its energy unit.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when the unit registry cannot convert the
    /// preset's `mass · velocity²` to its energy unit.
    pub fn new(units: UnitPreset) -> Result<Self, ComputeError> {
        let from = format!("({})*({})**2", units.mass(), units.velocity());
        let mvv2e = UnitRegistry::global()
            .factor(&from, units.energy())
            .map_err(|e| ComputeError::OutOfRange {
                field: "units",
                value: format!("{}: {from} -> {}: {e}", units.name(), units.energy()),
            })?;
        Self::with_constants(mvv2e, units.boltzmann())
    }

    /// The kinetic energy with an explicit `mvv2e` (the `mass · velocity²` →
    /// `energy` factor; 1 when the units are already consistent) and
    /// Boltzmann constant `boltzmann` in the same energy unit.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when either constant is not positive and
    /// finite.
    pub fn with_constants(mvv2e: F, boltzmann: F) -> Result<Self, ComputeError> {
        for (field, value) in [("mvv2e", mvv2e), ("boltzmann", boltzmann)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(ComputeError::OutOfRange {
                    field,
                    value: value.to_string(),
                });
            }
        }
        Ok(Self { mvv2e, boltzmann })
    }
}

impl Compute for KineticEnergy {
    type Input<'a> = KineticEnergyInput<'a>;
    type Output = KineticEnergyResult;

    /// # Errors
    /// [`ComputeError::BadShape`] when the velocities are not `N × 3`;
    /// [`ComputeError::DimensionMismatch`] when the masses do not have one
    /// entry per atom.
    fn compute(&self, input: KineticEnergyInput<'_>) -> Result<KineticEnergyResult, ComputeError> {
        check_rows(input.masses, input.velocities)?;
        let mut twice = 0.0;
        Zip::from(input.masses)
            .and(input.velocities.rows())
            .for_each(|&m, v| {
                twice += m * v.dot(&v);
            });
        Ok(KineticEnergyResult {
            energy: 0.5 * twice * self.mvv2e,
            boltzmann: self.boltzmann,
        })
    }
}

crate::compute::spec::compute_spec! {
    KineticEnergy as "KineticEnergy";
    params |s| {
        units: Units = crate::compute::ParamDefault::RequiredIn("new");
        mvv2e: Float = crate::compute::ParamDefault::RequiredIn("with_constants");
        boltzmann: Float = crate::compute::ParamDefault::RequiredIn("with_constants");
    }
    inputs {
        masses: Scalars;
        velocities: Vectors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn kinetic(
        units: UnitPreset,
        mass: ArrayView1<'_, F>,
        vel: ArrayView2<'_, F>,
    ) -> Result<KineticEnergyResult, ComputeError> {
        KineticEnergy::new(units)?.compute(KineticEnergyInput {
            masses: mass,
            velocities: vel,
        })
    }

    #[test]
    fn kinetic_energy_is_half_m_v_squared() {
        let mass = array![2.0, 1.0];
        let vel = array![[1.0, 0.0, 0.0], [0.0, 2.0, 2.0]];
        // ½(2·1 + 1·8) = 5
        assert_eq!(
            kinetic(UnitPreset::si(), mass.view(), vel.view())
                .unwrap()
                .energy,
            5.0
        );
        assert!(kinetic(UnitPreset::si(), array![1.0].view(), vel.view()).is_err());
    }

    /// In `real` units g/mol · (Å/fs)² is 2390.057… kcal/mol (LAMMPS
    /// `mvv2e` = 48.88821291²).
    #[test]
    fn real_units_convert_to_kcal_per_mol() {
        let k = kinetic(
            UnitPreset::real(),
            array![1.0].view(),
            array![[1.0, 0.0, 0.0]].view(),
        )
        .unwrap();
        assert!((k.energy / 0.5 - 48.888_212_91_f64.powi(2)).abs() / 2390.0 < 1e-6);
        assert_eq!(k.boltzmann, UnitPreset::real().boltzmann());
    }

    /// Explicit constants scale `½ Σ m v²` by `mvv2e` and carry `k_B` into
    /// the result; `new(units)` is the same with the preset's constants.
    #[test]
    fn explicit_constants_match_the_preset() {
        let mass = array![2.0, 1.0];
        let vel = array![[1.0, 0.0, 0.0], [0.0, 2.0, 2.0]];
        let input = KineticEnergyInput {
            masses: mass.view(),
            velocities: vel.view(),
        };
        let k = KineticEnergy::with_constants(2.0, 0.5)
            .unwrap()
            .compute(input)
            .unwrap();
        assert_eq!((k.energy, k.boltzmann), (10.0, 0.5));
        assert_eq!(k.temperature(6).unwrap(), 2.0 * 10.0 / (6.0 * 0.5));

        let si = UnitPreset::si();
        let from_preset = KineticEnergy::new(si.clone())
            .unwrap()
            .compute(input)
            .unwrap();
        let explicit = KineticEnergy::with_constants(1.0, si.boltzmann())
            .unwrap()
            .compute(input)
            .unwrap();
        assert_eq!(from_preset, explicit);
    }

    #[test]
    fn constants_must_be_positive_and_finite() {
        for (mvv2e, kb, field) in [
            (0.0, 1.0, "mvv2e"),
            (F::NAN, 1.0, "mvv2e"),
            (1.0, -1.0, "boltzmann"),
            (1.0, F::INFINITY, "boltzmann"),
        ] {
            let err = KineticEnergy::with_constants(mvv2e, kb).unwrap_err();
            assert!(
                matches!(err, ComputeError::OutOfRange { field: f, .. } if f == field),
                "{err:?}"
            );
        }
    }

    #[test]
    fn temperature_is_equipartition() {
        // 2K/(n_dof kb) with K = 3, n_dof = 6, kb = 0.5 → 2
        let k = KineticEnergyResult {
            energy: 3.0,
            boltzmann: 0.5,
        };
        assert_eq!(k.temperature(6).unwrap(), 2.0);
        assert!(k.temperature(0).is_err());
        let no_kb = KineticEnergyResult {
            energy: 3.0,
            boltzmann: 0.0,
        };
        assert!(no_kb.temperature(6).is_err());
    }
}
