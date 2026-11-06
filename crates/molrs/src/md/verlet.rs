//! Velocity Verlet, bound to `initial_integrate` and `final_integrate`.
//!
//! The half-steps are not a public operation. A fix that needs to run between
//! them binds the phase that already sits there (`post_force` for a Langevin
//! force, `post_integrate` for a hook after the drift).

use molrs::core::{Frame, UnitStyle};
use molrs::op::F;

use super::columns::{self, read_force, read_images, read_mass, read_pos, read_vel};
use super::component::{Component, Integrator};
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// NVE velocity Verlet, LAMMPS `fix nve`.
///
/// `initial_integrate`: `v += ½ Δt f/m · ftm2v`, `x += Δt v`, then wrap into
/// the cell when the frame has a box.
/// `final_integrate`: `v += ½ Δt f/m · ftm2v`.
///
/// `Δt` is the run's timestep ([`MD::set_timestep`](crate::md::MD::set_timestep))
/// and `ftm2v` its units style's ([`UnitConstants::ftm2v`](molrs::core::UnitConstants)),
/// both read at `build`. Mass, positions and velocities are frame columns.
#[derive(Clone, Copy, Debug, Default)]
pub struct VelocityVerlet {
    /// `(Δt, ftm2v)` of the run, set at `build`.
    step: Option<(F, F)>,
}

impl VelocityVerlet {
    /// `v += dtfm · f` with `dtfm = dtf / m`, `dtf = ½ Δt · ftm2v`, in the
    /// order LAMMPS `FixNVE::initial_integrate` rounds it.
    fn kick(frame: &mut Frame, dtf: F) -> Result<(), MdError> {
        let force = read_force(frame)?;
        let mass = read_mass(frame)?;
        let mut vel = read_vel(frame)?;
        if force.nrows() != vel.nrows() || mass.len() != vel.nrows() {
            return Err(MdError::Invalid(format!(
                "force rows {}, velocity rows {}, mass {}",
                force.nrows(),
                vel.nrows(),
                mass.len()
            )));
        }
        for i in 0..vel.nrows() {
            let dtfm = dtf / mass[i];
            for axis in 0..3 {
                vel[[i, axis]] += dtfm * force[[i, axis]];
            }
        }
        columns::write_vel(frame, &vel)
    }

    fn drift_and_wrap(frame: &mut Frame, dt: F) -> Result<(), MdError> {
        let vel = read_vel(frame)?;
        let mut pos = read_pos(frame)?;
        if pos.nrows() != vel.nrows() {
            return Err(MdError::Invalid(format!(
                "position rows {} disagree with velocity rows {}",
                pos.nrows(),
                vel.nrows()
            )));
        }
        for i in 0..pos.nrows() {
            for axis in 0..3 {
                pos[[i, axis]] += dt * vel[[i, axis]];
            }
        }
        if let Some(simbox) = frame.simbox.clone() {
            let (wrapped, shift) = simbox.wrap(pos.view());
            let mut images = read_images(frame)?;
            images += &shift;
            columns::write_pos(frame, &wrapped)?;
            columns::write_images(frame, &images)?;
        } else {
            columns::write_pos(frame, &pos)?;
        }
        Ok(())
    }
}

impl Component for VelocityVerlet {
    fn name(&self) -> &str {
        "velocity_verlet"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::InitialIntegrate, 0), (Phase::FinalIntegrate, 0)]
    }

    fn build(&mut self, _frame: &mut Frame, units: UnitStyle, dt: F) -> Result<(), MdError> {
        self.step = Some((dt, units.constants().ftm2v));
        Ok(())
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        _scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        let (dt, ftm2v) = self
            .step
            .ok_or_else(|| MdError::Invalid("velocity_verlet was launched before build".into()))?;
        let half = 0.5 * dt * ftm2v;
        match phase {
            Phase::InitialIntegrate => {
                Self::kick(frame, half)?;
                Self::drift_and_wrap(frame, dt)?;
            }
            Phase::FinalIntegrate => Self::kick(frame, half)?,
            _ => {}
        }
        Ok(())
    }
}

impl Integrator for VelocityVerlet {}
