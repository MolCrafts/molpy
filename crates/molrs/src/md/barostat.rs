//! Berendsen pressure coupling, bound at `end_of_step` after the kinetic sum.
//!
//! Each step the cell and the positions are scaled isotropically by
//!
//! ```text
//! μ³ = 1 − (Δt / τ_p) · (P₀ − P) / B
//! ```
//!
//! (Berendsen et al., J. Chem. Phys. 81, 3684 (1984); LAMMPS
//! `fix press/berendsen iso`, which takes the bulk modulus `B = 1/β`). `P` is
//! the instantaneous pressure `(2K + tr W) / (3V) · nktv2p` from the step's
//! synchronous velocities `v(t+Δt)` and virial `W(x(t+Δt))`, as LAMMPS
//! `compute pressure` forms it (`dof · boltz · T` is `2K`). The cell is scaled
//! about its centre `c`, as `FixPressBerendsen::remap` scales `boxlo` and
//! `boxhi`: `H' = μH` and `x' = c + μ(x − c)`, so fractional coordinates and
//! image flags are unchanged. Velocities are not scaled.
//!
//! The rescale does not conserve any energy; it relaxes the volume toward
//! `P₀`. It is the equilibration coupling, not an NPT ensemble.

use molrs::core::{Cell, Frame, MetaValue, SimBox, UnitStyle};
use molrs::op::F;

use super::columns::{self, VIRIAL, read_pos};
use super::component::{Barostat, Component};
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// `meta` key of the step's pressure, in the run's pressure unit.
pub(crate) const PRESSURE: &str = "pressure";

/// Isotropic Berendsen barostat.
#[derive(Clone, Copy, Debug)]
pub struct BerendsenBarostat {
    pressure: F,
    tau: F,
    bulk_modulus: F,
    /// `(Δt, nktv2p)` of the run, set at `build`.
    step: Option<(F, F)>,
}

impl BerendsenBarostat {
    /// Couple to `pressure` (the run style's pressure unit) with relaxation
    /// time `tau` (its time unit, LAMMPS `Pdamp`) and the system's
    /// `bulk_modulus` (pressure unit, LAMMPS `modulus`), at the run's
    /// timestep.
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] for a non-finite `pressure` or a non-positive
    /// `tau` or `bulk_modulus`.
    pub fn new(pressure: F, tau: F, bulk_modulus: F) -> Result<Self, MdError> {
        if !pressure.is_finite() {
            return Err(MdError::Invalid(format!(
                "pressure must be finite, got {pressure}"
            )));
        }
        for (name, value) in [("tau", tau), ("bulk_modulus", bulk_modulus)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(MdError::Invalid(format!("{name} must be > 0, got {value}")));
            }
        }
        Ok(Self {
            pressure,
            tau,
            bulk_modulus,
            step: None,
        })
    }

    /// The target pressure.
    pub fn pressure(&self) -> F {
        self.pressure
    }
}

impl Component for BerendsenBarostat {
    fn name(&self) -> &str {
        "press_berendsen"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::EndOfStep, 0)]
    }

    fn build(&mut self, frame: &mut Frame, units: UnitStyle, dt: F) -> Result<(), MdError> {
        let Some(simbox) = frame.simbox.as_ref().filter(|b| b.is_cell_defined()) else {
            return Err(MdError::Invalid(
                "press_berendsen scales the cell, and the frame has none".into(),
            ));
        };
        if simbox.pbc() != [true; 3] {
            return Err(MdError::Invalid(format!(
                "press_berendsen couples a pressure that is defined only for a cell \
                 periodic on every axis; this one is periodic on {:?}",
                simbox.pbc()
            )));
        }
        self.step = Some((dt, units.constants().nktv2p));
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
        let (dt, nktv2p) = self
            .step
            .ok_or_else(|| MdError::Invalid("press_berendsen was launched before build".into()))?;
        let Some(MetaValue::F64x6(w)) = frame.meta.get(VIRIAL).cloned() else {
            return Err(MdError::Invalid(
                "press_berendsen needs the step's virial and this step has none: every \
                 force provider must give one (a Python potential returns a virial buffer \
                 from bind; an external field has none)"
                    .into(),
            ));
        };
        let kinetic = frame
            .meta
            .get("kinetic")
            .and_then(MetaValue::as_f64)
            .ok_or_else(|| MdError::Invalid("press_berendsen ran before the kinetic sum".into()))?;
        let simbox = frame.simbox.as_ref().ok_or_else(|| {
            MdError::Invalid("press_berendsen scales the cell, and the frame has none".into())
        })?;
        let volume = simbox.volume();
        let pressure = (2.0 * kinetic + w[0] + w[1] + w[2]) / (3.0 * volume) * nktv2p;
        let mu3 = 1.0 - dt / self.tau * (self.pressure - pressure) / self.bulk_modulus;
        if !(mu3.is_finite() && mu3 > 0.0) {
            return Err(MdError::Invalid(format!(
                "press_berendsen: the volume factor μ³ = {mu3} is not positive at pressure \
                 {pressure}; the coupling is too stiff for this step (raise tau or \
                 bulk_modulus)"
            )));
        }
        // LAMMPS: pow(…, 1.0 / 3.0).
        let mu = mu3.powf(1.0 / 3.0);
        let origin = simbox.origin();
        let h = simbox.matrix();
        let centre: [F; 3] =
            std::array::from_fn(|k| origin[k] + 0.5 * (h[k][0] + h[k][1] + h[k][2]));
        let scaled = SimBox::new(
            Cell::Matrix(h.map(|row| row.map(|x| x * mu))),
            std::array::from_fn(|k| (origin[k] - centre[k]) * mu + centre[k]),
            simbox.pbc(),
        )
        .map_err(|e| MdError::Invalid(format!("press_berendsen: {e}")))?;
        let mut pos = read_pos(frame)?;
        for mut row in pos.rows_mut() {
            for axis in 0..3 {
                row[axis] = centre[axis] + mu * (row[axis] - centre[axis]);
            }
        }
        columns::write_pos(frame, &pos)?;
        frame.simbox = Some(scaled);
        columns::set_meta_f64(frame, PRESSURE, pressure);
        Ok(())
    }
}

impl Barostat for BerendsenBarostat {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, Cell, SimBox};
    use ndarray::Array1;

    /// A cube of edge 10 at origin (−5, −5, −5), with the step's kinetic
    /// energy and virial on the frame, in SI (coherent: `nktv2p = 1`).
    fn state(xs: &[F], kinetic: F, w_xx: F) -> Frame {
        let mut atoms = Block::new();
        let n = xs.len();
        atoms
            .insert("x", Array1::from(xs.to_vec()).into_dyn(), None)
            .unwrap();
        for key in ["y", "z"] {
            atoms
                .insert(key, Array1::<F>::zeros(n).into_dyn(), None)
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.simbox = Some(SimBox::new(Cell::Lengths([10.0; 3]), [-5.0; 3], [true; 3]).unwrap());
        columns::set_meta_f64(&mut frame, "kinetic", kinetic);
        frame
            .meta
            .insert(VIRIAL, MetaValue::F64x6([w_xx, 0.0, 0.0, 0.0, 0.0, 0.0]));
        frame
    }

    fn couple(frame: &mut Frame, barostat: &mut BerendsenBarostat) -> Result<(), MdError> {
        barostat.build(frame, UnitStyle::Si, 1.0)?;
        barostat.launch(Phase::EndOfStep, frame, &mut Scratch::new())
    }

    /// Ideal gas: K = 300, W = 0, V = 1000, so P = 0.2. With P₀ = 0,
    /// B = 2 (β = 0.5) and Δt/τ = 0.1, μ³ = 1 + 0.1·0.2/2 = 1.01:
    /// V' = 1010, L' = 10.033223. The cube's centre (0, 0, 0) stays put, so
    /// the origin moves to −5μ = −5.016612 and x = 3 maps to 3μ = 3.009967.
    #[test]
    fn an_ideal_gas_expands_by_the_hand_factor() {
        let mut frame = state(&[3.0], 300.0, 0.0);
        let mut barostat = BerendsenBarostat::new(0.0, 10.0, 2.0).unwrap();
        couple(&mut frame, &mut barostat).unwrap();
        let simbox = frame.simbox.as_ref().unwrap();
        assert!(
            (simbox.volume() - 1010.0).abs() < 1e-9,
            "{}",
            simbox.volume()
        );
        assert!((simbox.lengths()[0] - 10.033_223).abs() < 1e-6);
        for o in simbox.origin() {
            assert!((o + 5.016_612).abs() < 1e-6, "{o}");
        }
        let x = read_pos(&frame).unwrap()[[0, 0]];
        assert!((x - 3.009_967).abs() < 1e-6, "{x}");
        assert!((columns::meta_f64(&frame, PRESSURE) - 0.2).abs() < 1e-12);
    }

    /// The compressed bond of `evaluator.rs` at rest: W_xx = 48, P = 0.016.
    /// With B = 1 and Δt/τ = 0.5, μ³ = 1.008 and V' = 1008.
    #[test]
    fn a_compressed_bond_pushes_the_cell_out() {
        let mut frame = state(&[0.2, 0.8], 0.0, 48.0);
        let mut barostat = BerendsenBarostat::new(0.0, 2.0, 1.0).unwrap();
        couple(&mut frame, &mut barostat).unwrap();
        let volume = frame.simbox.as_ref().unwrap().volume();
        assert!((volume - 1008.0).abs() < 1e-9, "{volume}");
    }

    #[test]
    fn a_cell_with_a_free_axis_is_refused() {
        let mut frame = state(&[0.0], 300.0, 0.0);
        frame.simbox =
            Some(SimBox::new(Cell::Lengths([10.0; 3]), [-5.0; 3], [true, true, false]).unwrap());
        let err = couple(
            &mut frame,
            &mut BerendsenBarostat::new(0.0, 10.0, 2.0).unwrap(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("periodic"), "{err}");
    }

    #[test]
    fn a_step_without_a_virial_is_refused() {
        let mut frame = state(&[0.0], 300.0, 0.0);
        frame.meta.remove(VIRIAL);
        let err = couple(
            &mut frame,
            &mut BerendsenBarostat::new(0.0, 10.0, 2.0).unwrap(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("virial"), "{err}");
    }

    /// P = 0.2 below a target of 30 with B = 1 and Δt/τ = 0.1: μ³ = −1.98.
    #[test]
    fn a_non_positive_volume_factor_is_refused() {
        let mut frame = state(&[0.0], 300.0, 0.0);
        let err = couple(
            &mut frame,
            &mut BerendsenBarostat::new(30.0, 10.0, 1.0).unwrap(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("μ³"), "{err}");
    }
}
