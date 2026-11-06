//! Maxwell-Boltzmann velocities at an exact temperature, LAMMPS
//! `velocity all create T seed`. Not a hook — draw only.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::compute::{CenterOfMass, CenterOfMassInput, Compute};
use crate::op::standard_normal;
use molrs::op::{F, Fnx3};

use super::error::MdError;
use molrs::core::UnitStyle;

/// Velocities at a temperature, in a LAMMPS units style, as
/// `velocity all create T seed` sets them.
///
/// Each component is drawn from a normal of standard deviation
/// `sqrt(boltz T / (m · mvv2e))` (velocities in the style's velocity unit for
/// masses in its mass unit), the centre-of-mass velocity is removed
/// (LAMMPS `mom yes`, the default; [`keep_com`](Self::keep_com) is
/// `mom no`), and the velocities are then scaled so the temperature is `T`
/// exactly, as `Velocity::create` rescales them. The temperature is LAMMPS
/// `compute temp`'s: `Σ m v² · mvv2e / (dof · boltz)` with `dof = 3N − 3`
/// (`extra_dof` = the dimension, removed whether or not the momentum is).
/// LAMMPS draws uniform components by default (`dist uniform`); the
/// distribution here is LAMMPS's `dist gaussian`.
///
/// ```ignore
/// let vel = MaxwellBoltzmann::new(300.0, molrs::core::UnitStyle::Real, 1)?
///     .velocities(pos.view(), mass.view())?;
/// ```
pub struct MaxwellBoltzmann {
    temperature: F,
    units: UnitStyle,
    seed: u64,
    remove_com: bool,
}

impl MaxwellBoltzmann {
    /// Draw at `temperature` (the style's temperature unit); the COM
    /// velocity is removed after the draw ([`keep_com`](Self::keep_com) opts
    /// out).
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] for a temperature that is not strictly positive.
    pub fn new(temperature: F, units: UnitStyle, seed: u64) -> Result<Self, MdError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(MdError::Invalid(
                "MaxwellBoltzmann temperature must be strictly positive".into(),
            ));
        }
        Ok(Self {
            temperature,
            units,
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

    /// The temperature drawn at.
    pub fn temperature(&self) -> F {
        self.temperature
    }

    /// RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Whether the centre-of-mass velocity is subtracted after the draw.
    pub fn remove_com(&self) -> bool {
        self.remove_com
    }

    /// The units style the velocities are drawn in.
    pub fn units(&self) -> UnitStyle {
        self.units
    }

    /// Draw `(N, 3)` velocities at exactly [`Self::temperature`].
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] for positions that are not `(N, 3)`, masses of
    /// another length or not strictly positive, or fewer than two atoms
    /// (`3N − 3` degrees of freedom leave a single atom no temperature).
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
        if n < 2 {
            return Err(MdError::Invalid(format!(
                "MaxwellBoltzmann needs at least two atoms: the temperature counts 3N - 3 \
                 degrees of freedom, none for {n}"
            )));
        }
        let c = self.units.constants();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut vel = Array2::<F>::zeros((n, 3));
        Zip::from(vel.rows_mut()).and(mass).for_each(|mut row, &m| {
            let scale = (c.boltz * self.temperature / (m * c.mvv2e)).sqrt();
            for x in row.iter_mut() {
                *x = standard_normal(&mut rng) * scale;
            }
        });
        if self.remove_com {
            let com = Self::com_velocity(vel.view(), mass)?;
            Zip::from(vel.rows_mut()).for_each(|mut v| v -= &com);
        }
        // LAMMPS `Velocity::rescale(t, t_desired)`: v *= sqrt(t_desired / t),
        // with `t` from `ComputeTemp::compute_scalar`.
        let dof = (3 * n - 3) as F;
        let mvv: F = Zip::from(vel.rows())
            .and(mass)
            .fold(0.0, |acc, v, &m| acc + m * v.dot(&v));
        let t = mvv * c.mvv2e / (dof * c.boltz);
        if t == 0.0 {
            return Err(MdError::Invalid(
                "MaxwellBoltzmann drew a zero temperature and cannot rescale it".into(),
            ));
        }
        vel *= (self.temperature / t).sqrt();
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

    const REAL: UnitStyle = UnitStyle::Real;

    /// LAMMPS `compute temp` of `vel`: `Σ m v² · mvv2e / ((3N − 3) boltz)`.
    fn temperature(vel: &Fnx3, mass: &Array1<F>, units: UnitStyle) -> F {
        let c = units.constants();
        let mvv: F = vel
            .rows()
            .into_iter()
            .zip(mass)
            .map(|(v, m)| m * v.dot(&v))
            .sum();
        mvv * c.mvv2e / ((3 * mass.len() - 3) as F * c.boltz)
    }

    #[test]
    fn same_seed_is_reproducible() {
        let pos = Array2::zeros((8, 3));
        let mass = Array1::from_elem(8, 1.0);
        let draw = || {
            MaxwellBoltzmann::new(300.0, REAL, 7)
                .unwrap()
                .velocities(pos.view(), mass.view())
                .unwrap()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn remove_com_leaves_zero_com() {
        let pos = Array2::zeros((6, 3));
        let mass = Array1::from_elem(6, 2.0);
        let vel = MaxwellBoltzmann::new(200.0, REAL, 1)
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
        let mb = MaxwellBoltzmann::new(200.0, REAL, 1).unwrap().keep_com();
        assert!(!mb.remove_com());
        let vel = mb.velocities(pos.view(), mass.view()).unwrap();
        let com = MaxwellBoltzmann::com_velocity(vel.view(), mass.view()).unwrap();
        assert!(com.iter().any(|c| c.abs() > 1e-8));
    }

    #[test]
    fn rejects_a_nonpositive_temperature() {
        assert!(MaxwellBoltzmann::new(0.0, REAL, 0).is_err());
    }

    /// The draw is rescaled to the temperature exactly, in every style, with
    /// the COM removed or kept: LAMMPS `velocity create` at `mom yes` and
    /// `mom no`.
    #[test]
    fn the_temperature_is_exact_in_every_style() {
        let pos = Array2::zeros((5, 3));
        let mass = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        for units in UnitStyle::ALL {
            for keep in [false, true] {
                let mut mb = MaxwellBoltzmann::new(250.0, units, 4).unwrap();
                if keep {
                    mb = mb.keep_com();
                }
                let vel = mb.velocities(pos.view(), mass.view()).unwrap();
                let t = temperature(&vel, &mass, units);
                assert!((t / 250.0 - 1.0).abs() < 1e-14, "{units} keep={keep}: {t}");
            }
        }
    }

    #[test]
    fn a_single_atom_has_no_temperature_to_set() {
        let err = MaxwellBoltzmann::new(300.0, REAL, 1)
            .unwrap()
            .velocities(
                Array2::zeros((1, 3)).view(),
                Array1::from_elem(1, 1.0).view(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("3N - 3"), "{err}");
    }

    /// Carbon at 300 K in `real`: σ = sqrt(R·300 K / (12 g/mol)) = 455.9 m/s
    /// = 4.559e-3 Å/fs per component. Over 3·20 000 draws the sample σ is
    /// within 1 %.
    #[test]
    fn the_spread_is_thermal_in_the_run_velocity_unit() {
        let n = 20_000;
        let pos = Array2::zeros((n, 3));
        let mass = Array1::from_elem(n, 12.0);
        let vel = MaxwellBoltzmann::new(300.0, REAL, 3)
            .unwrap()
            .keep_com()
            .velocities(pos.view(), mass.view())
            .unwrap();
        let sigma = (vel.iter().map(|v| v * v).sum::<F>() / (3 * n) as F).sqrt();
        let expected = (8.314_462_618 * 300.0 / 12e-3_f64).sqrt() * 1e-5;
        assert!(
            (sigma / expected - 1.0).abs() < 0.01,
            "{sigma} vs {expected}"
        );
    }

    #[test]
    fn mass_length_must_match() {
        let pos = Array2::zeros((4, 3));
        let mass = Array1::from_elem(3, 1.0);
        let err = MaxwellBoltzmann::new(100.0, REAL, 0)
            .unwrap()
            .velocities(pos.view(), mass.view())
            .unwrap_err();
        assert!(format!("{err}").contains("n_atoms"));
    }
}
