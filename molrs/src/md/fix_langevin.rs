//! LAMMPS `fix langevin`, bound at `post_force` order +500.
//!
//! The drag and the random kick are forces, added after every physical
//! evaluator and before `final_integrate` consumes `f`:
//!
//! ```text
//! F_drag = -m / damp · v
//! F_rand = sqrt(2 m kb T / (damp · dt)) · ξ
//! ```
//!
//! `kb` is the caller's Boltzmann constant in energy per temperature, because
//! this module does not pick a unit system. `T` is in that temperature.
//! `damp` and `dt` are in the frame's time unit.
//!
//! The noise is a counter stream keyed by `(seed, step, atom id)`, drawn on
//! the host. A GPU run uses the molix component of the same name, which draws
//! on the device.

use molrs::core::Frame;
use molrs::op::F;

use super::columns::{self, read_force, read_mass, read_tags, read_vel};
use super::component::{Component, Thermostat, TimestepProvider};
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// Langevin thermostat as an additive post-force kick.
#[derive(Clone, Copy, Debug)]
pub struct FixLangevin {
    t_start: F,
    t_stop: F,
    damp: F,
    dt: F,
    seed: u64,
    kb: F,
}

impl FixLangevin {
    /// Constant temperature when `t_stop < 0` or `t_stop == t_start`.
    ///
    /// `kb` converts `t_start` into the energy unit of `atoms.mass` and the
    /// velocities. A missing or non-positive `kb` is refused: a Kelvin number
    /// times a force is not a force.
    pub fn new(t_start: F, t_stop: F, damp: F, dt: F, seed: u64, kb: F) -> Result<Self, MdError> {
        if !t_start.is_finite() || t_start < 0.0 {
            return Err(MdError::Invalid(format!(
                "T_start must be >= 0, got {t_start}"
            )));
        }
        if !damp.is_finite() || damp <= 0.0 {
            return Err(MdError::Invalid(format!("damp must be > 0, got {damp}")));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(MdError::Invalid(format!("dt must be > 0, got {dt}")));
        }
        if !kb.is_finite() || kb <= 0.0 {
            return Err(MdError::Invalid(format!("kb must be > 0, got {kb}")));
        }
        Ok(Self {
            t_start,
            t_stop,
            damp,
            dt,
            seed,
            kb,
        })
    }

    fn temperature(&self) -> F {
        // A negative `t_stop` is the LAMMPS sentinel for "hold T_start". A
        // ramp between T_start and T_stop needs a step window this constructor
        // does not take yet, so a non-negative T_stop that differs from
        // T_start still holds T_start rather than inventing the window.
        let _hold = self.t_stop;
        self.t_start
    }
}

impl TimestepProvider for FixLangevin {
    fn dt(&self) -> F {
        self.dt
    }
}

impl Component for FixLangevin {
    fn name(&self) -> &str {
        "fix_langevin"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::PostForce, 500)]
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        if phase != Phase::PostForce {
            return Ok(());
        }
        let vel = read_vel(frame)?;
        let mass = read_mass(frame)?;
        let tags = read_tags(frame)?;
        let mut force = read_force(frame)?;
        let n = vel.nrows();
        if mass.len() != n || force.nrows() != n || tags.len() != n {
            return Err(MdError::Invalid(
                "langevin requires mass, velocity, force, and id to share the atom count".into(),
            ));
        }
        let temperature = self.temperature();
        let scale = (2.0 * self.kb * temperature / (self.damp * self.dt))
            .max(0.0)
            .sqrt();
        for i in 0..n {
            let inv_damp = 1.0 / self.damp;
            let sigma = scale * mass[i].sqrt();
            for axis in 0..3 {
                let xi = standard_normal(self.seed, scratch.step as u64, tags[i], axis as u64);
                force[[i, axis]] += -mass[i] * vel[[i, axis]] * inv_damp + sigma * xi;
            }
        }
        columns::write_force(frame, &force)
    }
}

impl Thermostat for FixLangevin {}

/// One standard normal from a splitmix64 stream.
///
/// `channel` selects the Cartesian component. The same `(seed, step, tag,
/// channel)` always yields the same sample, independent of thread order.
fn standard_normal(seed: u64, step: u64, tag: u64, channel: u64) -> F {
    let u1 = uniform(mix(seed, step, tag, channel * 2));
    let u2 = uniform(mix(seed, step, tag, channel * 2 + 1));
    let u1 = u1.clamp(F::MIN_POSITIVE, 1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn uniform(mut x: u64) -> F {
    x = splitmix64(x);
    ((x >> 11) as F) * (1.0 / ((1u64 << 53) as F))
}

fn mix(seed: u64, step: u64, tag: u64, channel: u64) -> u64 {
    splitmix64(seed ^ splitmix64(step.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ tag ^ channel))
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
