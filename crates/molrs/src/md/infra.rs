//! Components [`MD`](crate::md::MD) binds itself: clear the force before
//! `force`, and write the kinetic energy after the second half-step.

use molrs::compute::Compute;
use molrs::core::Frame;

use super::columns::{self, read_mass, read_vel};
use super::component::Component;
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// Zeros `fx`/`fy`/`fz`, `meta["energy"]` and `meta["virial"]` so evaluators
/// add into a known baseline.
#[derive(Clone, Copy, Debug)]
pub struct ZeroForces;

impl Component for ZeroForces {
    fn name(&self) -> &str {
        "zero_forces"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::PreForce, -1000)]
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        _scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        if phase != Phase::PreForce {
            return Ok(());
        }
        let pos = columns::read_pos(frame)?;
        let zeros = molrs::op::Fnx3::zeros(pos.dim());
        columns::write_force(frame, &zeros)?;
        columns::set_meta_f64(frame, "energy", 0.0);
        frame
            .meta
            .insert(columns::VIRIAL, molrs::core::MetaValue::F64x6([0.0; 6]));
        Ok(())
    }
}

/// `½ Σ m v²` into `meta["kinetic"]`, after the second half-step — the
/// [`compute::KineticEnergy`](molrs::compute::KineticEnergy) sum, with
/// `mvv2e = 1`: the integrator applies `a = f / m` with no conversion factor,
/// so a run's quantities are in one consistent unit system and so is this.
#[derive(Debug, Default)]
pub struct KineticEnergy {
    /// The sum, built once in `build`.
    sum: Option<molrs::compute::KineticEnergy>,
}

impl Component for KineticEnergy {
    fn name(&self) -> &str {
        "kinetic_energy"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::EndOfStep, -100)]
    }

    fn build(&mut self, _frame: &mut Frame) -> Result<(), MdError> {
        let sum = molrs::compute::KineticEnergy::new(molrs::core::UnitPreset::real())
            .and_then(|k| k.with_mvv2e(1.0))
            .map_err(|e| MdError::Invalid(e.to_string()))?;
        self.sum = Some(sum);
        Ok(())
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        _scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        if phase != Phase::EndOfStep {
            return Ok(());
        }
        let vel = read_vel(frame)?;
        let mass = read_mass(frame)?;
        let sum = self
            .sum
            .as_ref()
            .ok_or_else(|| MdError::Invalid("kinetic energy was launched before build".into()))?;
        let kinetic = sum
            .compute(molrs::compute::KineticEnergyInput {
                masses: mass.view(),
                velocities: vel.view(),
            })
            .map_err(|e| MdError::Invalid(e.to_string()))?
            .energy;
        columns::set_meta_f64(frame, "kinetic", kinetic);
        Ok(())
    }
}
