//! LAMMPS `fix langevin`, bound at `post_force` order +500.
//!
//! The drag and the random kick are forces, added after every physical
//! evaluator and before `final_integrate` consumes `f`. In the run's units
//! style, as LAMMPS `FixLangevin::init` and `post_force_templated` form them
//! (`gfactor1`, `gfactor2`):
//!
//! ```text
//! F_drag = −m / damp / ftm2v · v
//! F_rand = sqrt(m) · sqrt(2 boltz / damp / dt / mvv2e) / ftm2v · sqrt(T) · ξ
//! ```
//!
//! `T` is in the style's temperature unit, `damp` (LAMMPS `damp`) and the
//! run's `dt` in its time unit; `boltz`, `ftm2v` and `mvv2e` are the style's
//! LAMMPS constants. LAMMPS writes the kick as `sqrt(24 boltz / …)·(u − ½)`
//! with `u` uniform on [0, 1), which has the variance of `sqrt(2 boltz / …)·ξ`;
//! here `ξ` is a standard normal, so the kick has the same mean and variance.
//! The noise is a counter stream keyed by `(seed, step, atom id)`, drawn on
//! the host; `step` is the run's absolute step, so chained runs draw new noise.

use molrs::core::Frame;
use molrs::op::F;

use super::columns::{self, read_force, read_mass, read_tags, read_vel};
use super::component::{Component, Thermostat};
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;
use molrs::core::UnitStyle;

/// Langevin thermostat as an additive post-force kick at one temperature.
#[derive(Clone, Copy, Debug)]
pub struct FixLangevin {
    temperature: F,
    damp: F,
    seed: u64,
    /// `(ftm2v, sqrt(2 boltz / damp / dt / mvv2e) / ftm2v)` of the run, set
    /// at `build`.
    factors: Option<(F, F)>,
}

impl FixLangevin {
    /// A bath at `temperature` (the run style's temperature unit) with
    /// relaxation time `damp` (its time unit), LAMMPS
    /// `fix langevin Tstart Tstop damp seed` at `Tstart = Tstop`.
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] for a negative temperature or a non-positive
    /// `damp`.
    pub fn new(temperature: F, damp: F, seed: u64) -> Result<Self, MdError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(MdError::Invalid(format!(
                "temperature must be >= 0, got {temperature}"
            )));
        }
        if !damp.is_finite() || damp <= 0.0 {
            return Err(MdError::Invalid(format!("damp must be > 0, got {damp}")));
        }
        Ok(Self {
            temperature,
            damp,
            seed,
            factors: None,
        })
    }

    /// The bath temperature.
    pub fn temperature(&self) -> F {
        self.temperature
    }
}

impl Component for FixLangevin {
    fn name(&self) -> &str {
        "fix_langevin"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::PostForce, 500)]
    }

    fn build(&mut self, _frame: &mut Frame, units: UnitStyle, dt: F) -> Result<(), MdError> {
        let c = units.constants();
        let noise = (2.0 * c.boltz / self.damp / dt / c.mvv2e).sqrt() / c.ftm2v;
        self.factors = Some((c.ftm2v, noise));
        Ok(())
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
        let (ftm2v, noise) = self
            .factors
            .ok_or_else(|| MdError::Invalid("fix_langevin was launched before build".into()))?;
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
        let tsqrt = self.temperature.sqrt();
        for i in 0..n {
            let gamma1 = -mass[i] / self.damp / ftm2v;
            let gamma2 = mass[i].sqrt() * noise * tsqrt;
            for axis in 0..3 {
                let xi = standard_normal(self.seed, scratch.step(), tags[i], axis as u64);
                force[[i, axis]] += gamma1 * vel[[i, axis]] + gamma2 * xi;
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

/// One key per `(seed, step, tag, channel)`. Each field is hashed in its own
/// stage: XOR-ing two small integers together first would let atom 0's y
/// channels coincide with atom 2's x channels.
fn mix(seed: u64, step: u64, tag: u64, channel: u64) -> u64 {
    splitmix64(splitmix64(splitmix64(seed ^ splitmix64(step)) ^ tag) ^ channel)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use ndarray::Array1;

    /// At T = 0 only the drag acts: `−m / damp / ftm2v · v`. In `real`,
    /// m = 12 g/mol, v = 0.01 Å/fs, damp = 100 fs gives
    /// −0.12 / (100 · 4.184e-4) = −2.868 069 kcal/mol/Å.
    #[test]
    fn the_drag_is_in_the_run_force_unit() {
        let mut atoms = Block::new();
        for (key, v) in [
            ("vx", 0.01),
            ("vy", 0.0),
            ("vz", 0.0),
            ("fx", 0.0),
            ("fy", 0.0),
            ("fz", 0.0),
        ] {
            atoms
                .insert(key, Array1::from_elem(1, v).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert("mass", Array1::from_elem(1, 12.0).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let units = UnitStyle::Real;
        let mut fix = FixLangevin::new(0.0, 100.0, 0).unwrap();
        fix.build(&mut frame, units, 1.0).unwrap();
        fix.launch(Phase::PostForce, &mut frame, &mut Scratch::new())
            .unwrap();
        let fx = frame["atoms"].get("fx").unwrap().get::<F>().unwrap()[0];
        let expected = -12.0 / 100.0 / units.constants().ftm2v * 0.01;
        assert_eq!(fx, expected);
        assert!((fx + 2.868_069).abs() < 1e-5, "{fx}");
    }

    /// Every atom and channel draws its own number: no two of 64 atoms × 6
    /// channels share a key at one step.
    #[test]
    fn atoms_and_channels_draw_distinct_noise() {
        let mut keys: Vec<u64> = (0..64)
            .flat_map(|tag| (0..6).map(move |channel| mix(7, 3, tag, channel)))
            .collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n);
    }

    #[test]
    fn a_negative_temperature_is_refused() {
        assert!(FixLangevin::new(-1.0, 100.0, 0).is_err());
    }
}
