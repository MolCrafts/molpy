//! A force provider of its own, bound to the `force` phase: [`Calculator`].
//!
//! A calculator is what ASE calls one: something that turns positions into
//! energy and forces by itself — a neural potential on a GPU, an external
//! program. The driver hands it the positions, the cell when it changed, and
//! whether its neighbour search must run again; it hands back energy, forces,
//! and a virial when it has one. It reads no pair table the driver built and
//! is bound to no force-field term.
//!
//! The driver owns the rebuild decision. A calculator that searches on its
//! own device builds at `cutoff + skin`, and the driver's [`VerletCadence`]
//! says when that list is stale, so the policy exists once.

use molrs::core::{Frame, NeighborPolicy, PresetDim, SimBox, UnitPreset, VerletCadence, Virial};
use molrs::op::F;
use ndarray::ArrayView2;

use super::columns::{self, gather_pos};
use super::component::Component;
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// The error a calculator raises; the driver carries it unchanged in
/// [`MdError::Calculator`].
pub type CalculatorError = Box<dyn std::error::Error + Send + Sync>;

/// What the driver tells a calculator at one force evaluation.
#[derive(Clone, Copy, Debug)]
pub struct CalculatorStep<'a> {
    /// The run's step index ([`MD::step`](super::MD::step)); the evaluation
    /// before the first step is step 0.
    pub step: u64,
    /// The neighbour list the calculator built at `cutoff + skin` is stale:
    /// search again before evaluating. Always `true` at the first evaluation
    /// after a bind.
    pub rebuild: bool,
    /// The cell, when it differs from the one last passed — and at the first
    /// evaluation after a bind. `None` otherwise, and for a frame with no box.
    pub cell: Option<&'a SimBox>,
}

/// What a calculator returns from one evaluation, in its own
/// [`units`](Calculator::units).
#[derive(Clone, Copy, Debug)]
pub struct CalculatorResult {
    /// Potential energy.
    pub energy: F,
    /// `W = Σ_pairs f ⊗ d` (symmetric), equivalently `−∂E/∂ε` at zero strain:
    /// positive under compression, in the calculator's energy unit and in
    /// `[xx, yy, zz, xy, xz, yz]` order; `None` when the calculator has none.
    /// ASE's stress is `−W/V` in `[xx, yy, zz, yz, xz, xy]` order, so
    /// `stress · V` is `−W` and must be negated and reordered.
    pub virial: Option<Virial>,
}

/// A force provider that turns positions into energy and forces by itself.
pub trait Calculator: Send {
    /// Timing and error name.
    fn name(&self) -> &str;

    /// The unit system of the numbers it returns. The driver converts its
    /// energies, forces and virial into the run's units once per step.
    fn units(&self) -> &UnitPreset;

    /// The interaction cutoff of its own neighbour search, in the run's
    /// length unit; `None` for a calculator that searches for no neighbours
    /// (and is then never told to rebuild after the first evaluation).
    fn cutoff(&self) -> Option<F>;

    /// Bind to the frame the run will step, before the first evaluation and
    /// again after any change to the run. `skin` is the margin to build the
    /// neighbour list at beyond [`cutoff`](Self::cutoff).
    fn bind(&mut self, frame: &Frame, skin: F) -> Result<(), CalculatorError>;

    /// Evaluate at `positions` (flat `[x0, y0, z0, …]`, the run's length
    /// unit), writing every component of `forces` (same layout).
    fn compute(
        &mut self,
        step: &CalculatorStep<'_>,
        positions: &[F],
        forces: &mut [F],
    ) -> Result<CalculatorResult, CalculatorError>;
}

/// A [`Calculator`] mounted on the `force` phase, with the rebuild cadence
/// and the unit conversion the driver owns.
pub(crate) struct Mounted {
    calculator: Box<dyn Calculator>,
    skin: F,
    name: String,
    /// Built at `build`; `None` for a calculator with no cutoff.
    cadence: Option<VerletCadence>,
    /// The cell last passed to the calculator.
    sent_cell: Option<SimBox>,
    /// No evaluation since the bind.
    fresh: bool,
    energy_scale: F,
    force_scale: F,
    positions: Vec<F>,
    forces: Vec<F>,
}

impl Mounted {
    pub(crate) fn new(calculator: Box<dyn Calculator>, skin: F) -> Result<Self, MdError> {
        if !(skin >= 0.0 && skin.is_finite()) {
            return Err(MdError::Invalid(format!(
                "skin must be a finite length >= 0, got {skin}"
            )));
        }
        let name = calculator.name().to_owned();
        Ok(Self {
            calculator,
            skin,
            name,
            cadence: None,
            sent_cell: None,
            fresh: true,
            energy_scale: 1.0,
            force_scale: 1.0,
            positions: Vec::new(),
            forces: Vec::new(),
        })
    }

    fn failed(&self, step: u64, source: CalculatorError) -> MdError {
        MdError::Calculator {
            name: self.name.clone(),
            step,
            source,
        }
    }

    /// Whether the calculator's list is stale at the gathered positions.
    fn rebuild_due(&mut self, simbox: Option<&SimBox>) -> Result<bool, MdError> {
        if self.fresh {
            return Ok(true);
        }
        let Some(cadence) = self.cadence.as_mut() else {
            return Ok(false);
        };
        let n = self.positions.len() / 3;
        let positions = ArrayView2::from_shape((n, 3), &self.positions)
            .expect("gathered positions are 3N long");
        let due = cadence.due(positions, simbox)?;
        if due {
            cadence.rebuilt(positions, simbox);
        }
        Ok(due)
    }
}

impl Component for Mounted {
    fn name(&self) -> &str {
        &self.name
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::Force, 0)]
    }

    fn build(&mut self, frame: &mut Frame, units: &UnitPreset) -> Result<(), MdError> {
        let declared = self.calculator.units().clone();
        if declared.length() != units.length() {
            return Err(MdError::Invalid(format!(
                "calculator {} returns lengths in {} and the run is in {}; positions \
                 are not converted, so the two must agree",
                self.name,
                declared.length(),
                units.length()
            )));
        }
        self.energy_scale = units.factor_from(&declared, PresetDim::Energy)?;
        self.force_scale = units.factor_from(&declared, PresetDim::Force)?;
        self.calculator
            .bind(frame, self.skin)
            .map_err(|e| self.failed(0, e))?;
        gather_pos(frame, &mut self.positions)?;
        self.forces.resize(self.positions.len(), 0.0);
        let n = self.positions.len() / 3;
        let positions = ArrayView2::from_shape((n, 3), &self.positions)
            .expect("gathered positions are 3N long");
        self.cadence = self
            .calculator
            .cutoff()
            .map(|cutoff| {
                let policy = NeighborPolicy {
                    skin: self.skin,
                    ..NeighborPolicy::default()
                };
                VerletCadence::new(policy, cutoff + self.skin, positions, frame.simbox.as_ref())
            })
            .transpose()?;
        self.sent_cell = None;
        self.fresh = true;
        Ok(())
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        if phase != Phase::Force {
            return Ok(());
        }
        gather_pos(frame, &mut self.positions)?;
        if self.positions.len() != self.forces.len() {
            return Err(MdError::Invalid(format!(
                "calculator {} was bound to {} atoms and the frame has {}",
                self.name,
                self.forces.len() / 3,
                self.positions.len() / 3
            )));
        }
        let simbox = frame.simbox.as_ref();
        let rebuild = self.rebuild_due(simbox)?;
        let cell_changed = self.fresh
            || match (simbox, self.sent_cell.as_ref()) {
                (Some(now), Some(sent)) => !now.approx_eq(sent, 0.0),
                (None, None) => false,
                _ => true,
            };
        if cell_changed {
            self.sent_cell = simbox.cloned();
        }
        let step = CalculatorStep {
            step: scratch.step(),
            rebuild,
            cell: if cell_changed { simbox } else { None },
        };
        let result = self
            .calculator
            .compute(&step, &self.positions, &mut self.forces)
            .map_err(|e| self.failed(scratch.step(), e))?;
        if !result.energy.is_finite() || self.forces.iter().any(|f| !f.is_finite()) {
            return Err(self.failed(
                scratch.step(),
                format!("non-finite energy ({}) or forces", result.energy).into(),
            ));
        }
        self.fresh = false;
        for f in &mut self.forces {
            *f *= self.force_scale;
        }
        let virial = result.virial.map(|w| Virial {
            components: w.components.map(|c| c * self.energy_scale),
        });
        columns::add_force_terms(
            frame,
            &self.forces,
            result.energy * self.energy_scale,
            virial,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::md::{MD, VelocityVerlet};
    use molrs::core::{Block, Cell, MetaValue};
    use ndarray::Array1;
    use std::sync::{Arc, Mutex};

    /// What the harmonic test calculator saw, per evaluation.
    #[derive(Default)]
    struct Seen {
        steps: Vec<u64>,
        rebuilds: Vec<bool>,
        cells: Vec<bool>,
        binds: usize,
    }

    /// `E = k (r − r0)²` between atoms 0 and 1, a virial, and a failure on
    /// request — a hand-written calculator in `preset` units.
    struct Bond {
        k: F,
        r0: F,
        preset: UnitPreset,
        cutoff: Option<F>,
        fail_at: Option<u64>,
        seen: Arc<Mutex<Seen>>,
    }

    impl Bond {
        fn new(preset: UnitPreset) -> Self {
            Self {
                k: 100.0,
                r0: 1.0,
                preset,
                cutoff: Some(3.0),
                fail_at: None,
                seen: Arc::default(),
            }
        }
    }

    impl Calculator for Bond {
        fn name(&self) -> &str {
            "bond"
        }
        fn units(&self) -> &UnitPreset {
            &self.preset
        }
        fn cutoff(&self) -> Option<F> {
            self.cutoff
        }
        fn bind(&mut self, _frame: &Frame, _skin: F) -> Result<(), CalculatorError> {
            self.seen.lock().unwrap().binds += 1;
            Ok(())
        }
        fn compute(
            &mut self,
            step: &CalculatorStep<'_>,
            x: &[F],
            forces: &mut [F],
        ) -> Result<CalculatorResult, CalculatorError> {
            {
                let mut seen = self.seen.lock().unwrap();
                seen.steps.push(step.step);
                seen.rebuilds.push(step.rebuild);
                seen.cells.push(step.cell.is_some());
            }
            if self.fail_at == Some(step.step) {
                return Err("the model diverged".into());
            }
            let d = [x[3] - x[0], x[4] - x[1], x[5] - x[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let g = 2.0 * self.k * (r - self.r0) / r;
            forces.fill(0.0);
            for a in 0..3 {
                forces[a] = g * d[a];
                forces[3 + a] = -g * d[a];
            }
            let mut w = Virial::ZERO;
            w.add_outer([-g * d[0], -g * d[1], -g * d[2]], d);
            Ok(CalculatorResult {
                energy: self.k * (r - self.r0) * (r - self.r0),
                virial: Some(w),
            })
        }
    }

    fn two_atoms(x1: F) -> Frame {
        let mut atoms = Block::new();
        for (key, values) in [("x", [0.2, x1]), ("y", [1.0, 1.0]), ("z", [1.0, 1.0])] {
            atoms
                .insert(key, Array1::from(values.to_vec()).into_dyn(), None)
                .unwrap();
        }
        for key in ["vx", "vy", "vz"] {
            atoms
                .insert(key, Array1::<F>::zeros(2).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert("mass", Array1::from_elem(2, 12.0).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.simbox = Some(SimBox::new(Cell::Lengths([10.0; 3]), [0.0; 3], [true; 3]).unwrap());
        frame
    }

    fn md(calculator: Bond) -> MD {
        let mut md = MD::new(UnitPreset::real()).unwrap();
        md.set_integrator(VelocityVerlet::new(0.5).unwrap())
            .unwrap();
        md.set_calculator(calculator, 0.5).unwrap();
        md
    }

    fn fx(frame: &Frame) -> F {
        frame["atoms"].get("fx").unwrap().get::<F>().unwrap()[0]
    }

    /// The bond of `evaluator.rs`'s face test, 0.6 apart and compressed by
    /// 0.4: E = 100·0.16 = 16, atom 0 is pushed away along −x with
    /// f₀ = 2k(r − r0)·d/r = −80, and W_xx = f₁·d = 80·0.6 = 48.
    #[test]
    fn the_calculator_prices_the_bond_by_hand() {
        let mut frame = two_atoms(0.8);
        let mut md = md(Bond::new(UnitPreset::real()));
        md.run(&mut frame, 0).unwrap();
        assert!((columns::meta_f64(&frame, "energy") - 16.0).abs() < 1e-9);
        assert!((fx(&frame) + 80.0).abs() < 1e-9, "{}", fx(&frame));
        let Some(MetaValue::F64x6(w)) = frame.meta.get(columns::VIRIAL) else {
            panic!("a virial");
        };
        assert!((w[0] - 48.0).abs() < 1e-9, "{w:?}");
    }

    /// A calculator in `metal` driving a `real` run is read through the exact
    /// eV → kcal/mol factor.
    #[test]
    fn a_metal_calculator_is_converted_into_a_real_run() {
        let mut frame = two_atoms(0.8);
        let mut md = md(Bond::new(UnitPreset::metal()));
        md.run(&mut frame, 0).unwrap();
        let ev = md
            .units()
            .factor_from(&UnitPreset::metal(), PresetDim::Energy)
            .unwrap();
        assert!((columns::meta_f64(&frame, "energy") - 16.0 * ev).abs() < 1e-9);
        assert!((fx(&frame) + 80.0 * ev).abs() < 1e-9);
        let Some(MetaValue::F64x6(w)) = frame.meta.get(columns::VIRIAL) else {
            panic!("a virial");
        };
        assert!((w[0] - 48.0 * ev).abs() < 1e-9, "{w:?}");
    }

    /// A failure at step 3 stops the run there, with the step and the
    /// calculator's own error.
    #[test]
    fn a_failure_stops_the_run_at_its_step() {
        let mut calc = Bond::new(UnitPreset::real());
        calc.fail_at = Some(3);
        let seen = Arc::clone(&calc.seen);
        let mut md = md(calc);
        let err = md.run(&mut two_atoms(1.1), 10).unwrap_err();
        let MdError::Calculator { step, source, .. } = &err else {
            panic!("{err}");
        };
        assert_eq!(*step, 3);
        assert_eq!(source.to_string(), "the model diverged");
        assert_eq!(seen.lock().unwrap().steps, [0, 1, 2, 3]);
    }

    /// The first evaluation rebuilds and sees the cell; a still pair does
    /// neither again. Two runs bind once.
    #[test]
    fn rebuild_and_cell_are_reported_once_until_they_change() {
        let calc = Bond::new(UnitPreset::real());
        let seen = Arc::clone(&calc.seen);
        let mut md = md(calc);
        let mut frame = two_atoms(1.2);
        md.run(&mut frame, 2).unwrap();
        md.run(&mut frame, 1).unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen.binds, 1);
        assert_eq!(seen.rebuilds, [true, false, false, false]);
        assert_eq!(seen.cells, [true, false, false, false]);
    }

    /// A changed cell is passed once, at the evaluation after the change.
    #[test]
    fn a_changed_cell_is_passed_at_the_next_evaluation() {
        let calc = Bond::new(UnitPreset::real());
        let seen = Arc::clone(&calc.seen);
        let mut md = md(calc);
        let mut frame = two_atoms(1.2);
        md.run(&mut frame, 1).unwrap();
        frame.simbox = Some(SimBox::new(Cell::Lengths([10.01; 3]), [0.0; 3], [true; 3]).unwrap());
        md.run(&mut frame, 2).unwrap();
        assert_eq!(seen.lock().unwrap().cells, [true, false, true, false]);
    }

    #[test]
    fn a_calculator_in_another_length_unit_is_refused() {
        let mut md = md(Bond::new(UnitPreset::si()));
        let err = md.run(&mut two_atoms(1.2), 1).unwrap_err();
        assert!(err.to_string().contains("lengths"), "{err}");
    }

    /// Under a barostat the rescaled cell reaches the calculator at the next
    /// evaluation. Step 1's force precedes its own rescale (at `end_of_step`),
    /// so the cell is new at the setup evaluation and from step 2 on.
    #[test]
    fn a_barostat_cell_reaches_the_calculator_each_step() {
        let calc = Bond::new(UnitPreset::real());
        let seen = Arc::clone(&calc.seen);
        let mut md = md(calc);
        md.set_barostat(crate::md::BerendsenBarostat::new(1.0, 500.0, 2.2e4, 0.5).unwrap())
            .unwrap();
        let mut frame = two_atoms(0.8);
        let before = frame.simbox.as_ref().unwrap().volume();
        md.run(&mut frame, 3).unwrap();
        assert!(frame.simbox.as_ref().unwrap().volume() > before);
        assert_eq!(seen.lock().unwrap().cells, [true, false, true, true]);
        assert!(columns::meta_f64(&frame, "pressure") > 0.0);
    }
}
