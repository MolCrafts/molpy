//! Components [`MD`](crate::md::MD) binds itself: clear the force before
//! `force`, and write the kinetic energy after the second half-step.

use molrs::core::Frame;
use molrs::op::F;

use super::columns::{self, read_mass, read_vel};
use super::component::Component;
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// Zeros `fx`/`fy`/`fz` and `meta["energy"]` so evaluators add into a known
/// baseline.
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
        Ok(())
    }
}

/// `½ Σ m v²` into `meta["kinetic"]`, after the second half-step.
#[derive(Clone, Copy, Debug)]
pub struct KineticEnergy;

impl Component for KineticEnergy {
    fn name(&self) -> &str {
        "kinetic_energy"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::EndOfStep, -100)]
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
        let mut kinetic = 0.0 as F;
        for i in 0..vel.nrows() {
            let speed2 = vel[[i, 0]].powi(2) + vel[[i, 1]].powi(2) + vel[[i, 2]].powi(2);
            kinetic += 0.5 * mass[i] * speed2;
        }
        columns::set_meta_f64(frame, "kinetic", kinetic);
        Ok(())
    }
}
