//! Kinetic readings of one configuration: [`KineticEnergy`] and its
//! equipartition temperature, and the centre-of-mass velocity.

use molrs::core::{UnitPreset, UnitRegistry};
use ndarray::{Array1, ArrayView1, ArrayView2, Zip};

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

/// Kinetic energy `K = ½ Σᵢ mᵢ |vᵢ|²` of one configuration, in the energy
/// unit of the configured [`UnitPreset`].
///
/// Masses and velocities are in the preset's mass and velocity units; the
/// preset's `mass · velocity²` → `energy` factor (LAMMPS' `mvv2e`, 2390.06 in
/// `real`, exactly 1 in `si`) is applied, and its Boltzmann constant gives
/// the equipartition temperature through [`KineticEnergyResult::temperature`].
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
    units: UnitPreset,
    mvv2e: F,
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
    /// The kinetic energy in `units`.
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
        Ok(Self { units, mvv2e })
    }

    /// The unit system.
    pub fn units(&self) -> &UnitPreset {
        &self.units
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
            boltzmann: self.units.boltzmann(),
        })
    }
}

/// The mass-weighted centre-of-mass velocity `Σᵢ mᵢ vᵢ / Σᵢ mᵢ` of the
/// `(n_atoms, 3)` velocities `vel`.
///
/// Kept as a function until link 07 of the compute redesign, whose
/// `CenterOfMass` (on velocities, with no box) replaces it.
///
/// # Errors
///
/// As [`KineticEnergy`]; [`ComputeError::OutOfRange`] when the total mass is
/// zero (the centre of mass is undefined).
pub fn center_of_mass_velocity(
    mass: ArrayView1<'_, F>,
    vel: ArrayView2<'_, F>,
) -> Result<Array1<F>, ComputeError> {
    check_rows(mass, vel)?;
    let mut p = [0.0; 3];
    let mut total = 0.0;
    Zip::from(vel.rows()).and(mass).for_each(|v, &m| {
        total += m;
        p[0] += m * v[0];
        p[1] += m * v[1];
        p[2] += m * v[2];
    });
    if total == 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "mass (total)",
            value: "0: the centre of mass of a massless set is undefined".into(),
        });
    }
    Ok(Array1::from_vec(vec![
        p[0] / total,
        p[1] / total,
        p[2] / total,
    ]))
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

    #[test]
    fn center_of_mass_velocity_is_mass_weighted() {
        let mass = array![3.0, 1.0];
        let vel = array![[1.0, 0.0, 0.0], [-3.0, 4.0, 0.0]];
        let v = center_of_mass_velocity(mass.view(), vel.view()).unwrap();
        assert_eq!(v.to_vec(), vec![0.0, 1.0, 0.0]);
    }

    /// Spec §5: a zero total mass used to return a zero velocity.
    #[test]
    fn a_massless_set_has_no_centre_of_mass_velocity() {
        let vel = array![[1.0, 0.0, 0.0], [-3.0, 4.0, 0.0]];
        assert!(matches!(
            center_of_mass_velocity(array![0.0, 0.0].view(), vel.view()),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
