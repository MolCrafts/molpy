//! Maxwell-Boltzmann velocity distribution. Not a hook — draw only.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::compute::{CenterOfMass, CenterOfMassInput, Compute};
use crate::op::standard_normal;
use molrs::op::{F, Fnx3};

use super::error::MdError;

/// Maxwell-Boltzmann velocity distribution.
///
/// `kbt` is k_B·T in the same energy units as the masses. MD has no unit
/// knowledge — pass `UnitPreset::real().boltzmann() * T` for LAMMPS real.
///
/// ```ignore
/// let mb = MaxwellBoltzmann::new(molrs::core::UnitPreset::real().boltzmann() * 300.0, 0)?;
/// let vel = mb.velocities(pos.view(), mass.view())?;
/// ```
pub struct MaxwellBoltzmann {
    kbt: F,
    seed: u64,
    remove_com: bool,
}

impl MaxwellBoltzmann {
    /// Draw with scale `sqrt(kbt / m)`; the COM velocity is removed after
    /// the draw ([`keep_com`](Self::keep_com) opts out).
    pub fn new(kbt: F, seed: u64) -> Result<Self, MdError> {
        if !kbt.is_finite() || kbt <= 0.0 {
            return Err(MdError::Invalid(
                "MaxwellBoltzmann kbt must be strictly positive".into(),
            ));
        }
        Ok(Self {
            kbt,
            seed,
            remove_com: true,
        })
    }

    /// Leave the centre-of-mass velocity in the draw (the default removes it).
    #[must_use]
    pub fn keep_com(mut self) -> Self {
        self.remove_com = false;
        self
    }

    /// k_B·T in the caller's energy units.
    pub fn kbt(&self) -> F {
        self.kbt
    }

    /// RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether the centre-of-mass velocity is subtracted after the draw.
    pub fn remove_com(&self) -> bool {
        self.remove_com
    }

    /// Draw `(N, 3)` velocities at [`Self::kbt`].
    pub fn velocities(
        &self,
        pos: ArrayView2<'_, F>,
        mass: ArrayView1<'_, F>,
    ) -> Result<Fnx3, MdError> {
        let n = pos.nrows();
        if pos.ncols() != 3 {
            return Err(MdError::Invalid(format!(
                "pos must have shape (N, 3), got {:?}",
                pos.shape()
            )));
        }
        if mass.len() != n {
            return Err(MdError::Invalid(format!(
                "atoms.mass length {} disagrees with n_atoms={n}",
                mass.len()
            )));
        }
        if mass.iter().any(|&m| !m.is_finite() || m <= 0.0) {
            return Err(MdError::Invalid("mass must be strictly positive".into()));
        }
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut vel = Array2::<F>::zeros((n, 3));
        Zip::from(vel.rows_mut()).and(mass).for_each(|mut row, &m| {
            let scale = (self.kbt / m).sqrt();
            for x in row.iter_mut() {
                *x = standard_normal(&mut rng) * scale;
            }
        });
        if self.remove_com {
            let com = Self::com_velocity(vel.view(), mass)?;
            Zip::from(vel.rows_mut()).for_each(|mut v| v -= &com);
        }
        Ok(vel)
    }

    /// The centre-of-mass velocity: [`CenterOfMass`] on the velocities, no
    /// box, every atom in one group.
    fn com_velocity(vel: ArrayView2<'_, F>, mass: ArrayView1<'_, F>) -> Result<Array1<F>, MdError> {
        let one_group: Vec<u8> = vec![0; vel.nrows()];
        let com = CenterOfMass::new()
            .compute(CenterOfMassInput {
                positions: vel,
                simbox: None,
                partition: &one_group,
                masses: mass,
            })
            .map_err(|e| MdError::Invalid(format!("centre-of-mass velocity: {e}")))?;
        Ok(com.centers.row(0).to_owned())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array2};

    use super::*;

    #[test]
    fn same_seed_is_reproducible() {
        let pos = Array2::zeros((8, 3));
        let mass = Array1::from_elem(8, 1.0);
        let a = MaxwellBoltzmann::new(300.0, 7)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        let b = MaxwellBoltzmann::new(300.0, 7)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn remove_com_leaves_zero_com() {
        let pos = Array2::zeros((6, 3));
        let mass = Array1::from_elem(6, 2.0);
        let vel = MaxwellBoltzmann::new(200.0, 1)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap();
        let com = MaxwellBoltzmann::com_velocity(vel.view(), mass.view()).unwrap();
        assert!(com.iter().all(|c| c.abs() < 1e-12));
    }

    #[test]
    fn keep_com_leaves_the_drawn_com() {
        let pos = Array2::zeros((6, 3));
        let mass = Array1::from_elem(6, 2.0);
        let mb = MaxwellBoltzmann::new(200.0, 1).unwrap().keep_com();
        assert!(!mb.remove_com());
        let vel = mb.velocities(pos.view(), mass.view()).unwrap();
        let com = MaxwellBoltzmann::com_velocity(vel.view(), mass.view()).unwrap();
        assert!(com.iter().any(|c| c.abs() > 1e-8));
    }

    #[test]
    fn rejects_nonpositive_kbt() {
        assert!(MaxwellBoltzmann::new(0.0, 0).is_err());
    }

    #[test]
    fn kbt_from_boltzmann_real_is_the_caller_scale() {
        let kb = molrs::core::UnitPreset::real().boltzmann();
        let mb = MaxwellBoltzmann::new(kb * 300.0, 0).unwrap();
        assert_eq!(mb.kbt(), kb * 300.0);
    }

    #[test]
    fn mass_length_must_match() {
        let pos = Array2::zeros((4, 3));
        let mass = Array1::from_elem(3, 1.0);
        let err = MaxwellBoltzmann::new(100.0, 0)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap_err();
        assert!(format!("{err}").contains("n_atoms"));
    }
}
