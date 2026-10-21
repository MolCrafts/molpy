//! Velocity Verlet, bound to `initial_integrate` and `final_integrate`.
//!
//! The half-steps are not a public operation. A fix that needs to run between
//! them binds the phase that already sits there (`post_force` for a Langevin
//! force, `post_integrate` for a hook after the drift).

use molrs::core::Frame;
use molrs::op::F;

use super::columns::{self, read_force, read_images, read_mass, read_pos, read_vel};
use super::component::{Component, Integrator, TimestepProvider};
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// NVE velocity Verlet.
///
/// `initial_integrate`: `v += ½ Δt f/m`, `x += Δt v`, then wrap into the cell
/// when the frame has a box.
/// `final_integrate`: `v += ½ Δt f/m`.
#[derive(Clone, Copy, Debug)]
pub struct VelocityVerlet {
    dt: F,
}

impl VelocityVerlet {
    /// `dt` in the caller's time unit. Mass, positions, and velocities are
    /// read from the frame at launch, not stored here.
    pub fn new(dt: F) -> Result<Self, MdError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(MdError::Invalid(format!("dt must be > 0, got {dt}")));
        }
        Ok(Self { dt })
    }

    fn kick(frame: &mut Frame, half_dt: F) -> Result<(), MdError> {
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
            let inv_m = 1.0 / mass[i];
            for axis in 0..3 {
                vel[[i, axis]] += half_dt * force[[i, axis]] * inv_m;
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

impl TimestepProvider for VelocityVerlet {
    fn dt(&self) -> F {
        self.dt
    }
}

impl Component for VelocityVerlet {
    fn name(&self) -> &str {
        "velocity_verlet"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::InitialIntegrate, 0), (Phase::FinalIntegrate, 0)]
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        _scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        let half = 0.5 * self.dt;
        match phase {
            Phase::InitialIntegrate => {
                Self::kick(frame, half)?;
                Self::drift_and_wrap(frame, self.dt)?;
            }
            Phase::FinalIntegrate => Self::kick(frame, half)?,
            _ => {}
        }
        Ok(())
    }
}

impl Integrator for VelocityVerlet {}
