//! [`MD`] walks [`Phase::ALL`] once per step and launches the components
//! bound to each phase in ascending order.

use molrs::core::{Frame, NeighborList, UnitPreset};
use molrs::op::F;

use super::calculator::{Calculator, Mounted};
use super::component::{Component, Flags, Integrator, Kind, Thermostat};
use super::error::MdError;
use super::evaluator::FfEvaluator;
use super::infra::{KineticEnergy, ZeroForces};
use super::neighbor_list::Neighbor;
use super::phase::Phase;
use super::scratch::Scratch;

struct Installed {
    bindings: Vec<(Phase, i32)>,
    flags: Flags,
    component: Box<dyn Component>,
}

/// In-process molecular dynamics. One driver, the LAMMPS phase list, and
/// components inserted onto it.
///
/// `run` writes the frame it is given. It does not return a second state.
pub struct MD {
    installed: Vec<Installed>,
    /// Per phase, the installed indices in launch order; built with the run.
    schedule: [Vec<usize>; Phase::COUNT],
    scratch: Scratch,
    units: UnitPreset,
    /// The step index of the last force evaluation; it counts across runs.
    step: u64,
    built: bool,
}

impl MD {
    /// A driver in `units`, with the force-clear and kinetic-energy
    /// infrastructure bound. The clear runs at `pre_force` order -1000. The
    /// kinetic energy runs at `end_of_step` order -100 and writes
    /// `frame.meta["kinetic"]`.
    ///
    /// `units` is the run's unit system: `dt` is in its time unit, energies
    /// and forces in its energy and force units, masses in its mass unit.
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] when the registry cannot derive the preset's
    /// step constants ([`UnitPreset::ftm2v`], [`UnitPreset::mvv2e`]).
    pub fn new(units: UnitPreset) -> Result<Self, MdError> {
        units.ftm2v()?;
        units.mvv2e()?;
        let mut md = Self {
            installed: Vec::new(),
            schedule: Default::default(),
            scratch: Scratch::new(),
            units,
            step: 0,
            built: false,
        };
        md.install(ZeroForces, ZeroForces.default_bindings(), Flags::default());
        let kinetic = KineticEnergy::default();
        let bindings = kinetic.default_bindings();
        md.install(kinetic, bindings, Flags::default());
        Ok(md)
    }

    /// The run's unit system.
    pub fn units(&self) -> &UnitPreset {
        &self.units
    }

    /// The step index of the last force evaluation: 0 after the evaluation
    /// that precedes the first step, `k` after step `k`. It counts across
    /// [`run`](Self::run) calls, so two runs of 5 steps are one run of 10.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Replace any integrator already bound, then bind `integrator` at its
    /// `default_bindings`.
    pub fn set_integrator<T: Integrator + 'static>(
        &mut self,
        integrator: T,
    ) -> Result<(), MdError> {
        let dt = require_positive_dt(integrator.dt())?;
        self.remove_kind(Kind::Integrator);
        let bindings = integrator.default_bindings();
        self.install(
            integrator,
            bindings,
            Flags {
                kind: Kind::Integrator,
                dt: Some(dt),
                ..Flags::default()
            },
        );
        Ok(())
    }

    /// Replace any evaluator already bound through this setter.
    ///
    /// A second force term is `on(Phase::Force).add`, which does not go
    /// through this slot and is not removed here.
    pub fn set_evaluator(&mut self, evaluator: FfEvaluator) -> Result<(), MdError> {
        self.remove_kind(Kind::Evaluator);
        let flags = evaluator.flags();
        let bindings = evaluator.default_bindings();
        self.install(evaluator, bindings, flags);
        Ok(())
    }

    /// Append a thermostat. Its `dt` must equal the integrator's `dt`; that
    /// check runs at [`build`](Self::build), so the integrator may be set after.
    pub fn set_thermostat<T: Thermostat + 'static>(
        &mut self,
        thermostat: T,
    ) -> Result<(), MdError> {
        let dt = require_positive_dt(thermostat.dt())?;
        let bindings = thermostat.default_bindings();
        self.install(
            thermostat,
            bindings,
            Flags {
                kind: Kind::Thermostat,
                dt: Some(dt),
                ..Flags::default()
            },
        );
        Ok(())
    }

    /// Mount `list` as the run's neighbour list, replacing any before it.
    ///
    /// The list searches at its own cutoff and knows nothing else; the frame
    /// box supplies the periodicity. Pairs within `cutoff − skin` reach the
    /// pair kernels, and the list is searched again once the motion and the
    /// cell strain since the last search could have brought a pair across the
    /// `skin` ([`VerletCadence`](molrs::core::VerletCadence); `0` searches
    /// every step).
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] for a `skin` that is negative, not finite, or not
    /// below the list's cutoff.
    pub fn set_neighbor(&mut self, list: NeighborList, skin: F) -> Result<(), MdError> {
        let neighbor = Neighbor::new(list, skin)?;
        self.remove_kind(Kind::Neighbor);
        let bindings = neighbor.default_bindings();
        self.install(
            neighbor,
            bindings,
            Flags {
                kind: Kind::Neighbor,
                ..Flags::default()
            },
        );
        Ok(())
    }

    /// Mount `calculator` as the run's force provider of its own, replacing
    /// any before it. It may sit beside an evaluator; their forces sum.
    ///
    /// `skin` is the margin the calculator's own neighbour search builds at
    /// beyond its [`cutoff`](Calculator::cutoff). The driver decides when that
    /// search must run again and says so through
    /// [`CalculatorStep::rebuild`](super::CalculatorStep).
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] for a negative or non-finite `skin`.
    pub fn set_calculator<C: Calculator + 'static>(
        &mut self,
        calculator: C,
        skin: F,
    ) -> Result<(), MdError> {
        let mounted = Mounted::new(Box::new(calculator), skin)?;
        self.remove_kind(Kind::Calculator);
        let bindings = mounted.default_bindings();
        self.install(
            mounted,
            bindings,
            Flags {
                kind: Kind::Calculator,
                ..Flags::default()
            },
        );
        Ok(())
    }

    /// Bind `component` to `phase` at `order`, without a categorical kind.
    ///
    /// This is the insertion point for a user fix. Categorical setters are
    /// thin wrappers over the same list.
    pub fn on<T: Component + 'static>(&mut self, phase: Phase, component: T, order: i32) {
        self.install(component, vec![(phase, order)], Flags::default());
    }

    /// Check the mounted set and `build` every component.
    ///
    /// [`run`](Self::run) calls this on the first entry. A setter after that
    /// invalidates it, and the next `run` builds again.
    pub fn build(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        if !self
            .installed
            .iter()
            .any(|c| c.flags.kind == Kind::Integrator)
        {
            return Err(MdError::Invalid(
                "MD requires an integrator; call set_integrator".into(),
            ));
        }
        if !self.has_force_phase_component() {
            return Err(MdError::Invalid(
                "MD requires an evaluator; call set_evaluator".into(),
            ));
        }
        let has_neighbor = self
            .installed
            .iter()
            .any(|c| c.flags.kind == Kind::Neighbor);
        if self.installed.iter().any(|c| c.flags.consumes_neighbors) && !has_neighbor {
            let name = self
                .installed
                .iter()
                .find(|c| c.flags.consumes_neighbors)
                .map_or("evaluator", |c| c.component.name())
                .to_owned();
            return Err(MdError::Invalid(format!(
                "{name} reads a neighbour list and none is mounted; call set_neighbor"
            )));
        }
        if self.installed.iter().any(|c| c.flags.frozen_pairs) && has_neighbor {
            return Err(MdError::Invalid(
                "an evaluator resolved its parameters against a fixed pair list, so it \
                 cannot run beside a neighbour list — it would ignore the list and answer \
                 for the pairs it was built from"
                    .into(),
            ));
        }
        let integ_dt = self
            .installed
            .iter()
            .find(|c| c.flags.kind == Kind::Integrator)
            .and_then(|c| c.flags.dt);
        for installed in &self.installed {
            if installed.flags.kind == Kind::Thermostat && installed.flags.dt != integ_dt {
                return Err(MdError::Invalid(
                    "thermostat dt differs from the integrator dt".into(),
                ));
            }
        }
        for installed in &mut self.installed {
            installed.component.build(frame, &self.units)?;
        }
        for phase in Phase::ALL {
            let mut order: Vec<(i32, usize)> = self
                .installed
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.order_on(phase).map(|ord| (ord, i)))
                .collect();
            order.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
            self.schedule[phase as usize] = order.into_iter().map(|(_, i)| i).collect();
        }
        self.built = true;
        Ok(())
    }

    /// Advance `frame` by `n_steps`. Builds on the first call and after any
    /// setter.
    pub fn run(&mut self, frame: &mut Frame, n_steps: usize) -> Result<(), MdError> {
        if !self.built {
            self.build(frame)?;
            // `initial_integrate` reads the force already on the frame. LAMMPS
            // computes that force during setup, before the first step; without
            // it the opening half-kick treats every force as zero.
            self.setup_forces(frame)?;
        }
        for _ in 0..n_steps {
            self.step += 1;
            self.scratch.begin(self.step);
            self.step_once(frame)?;
        }
        Ok(())
    }

    fn has_force_phase_component(&self) -> bool {
        self.installed.iter().any(|c| {
            c.bindings.iter().any(|(phase, _)| *phase == Phase::Force)
                && c.component.name() != "zero_forces"
        })
    }

    fn setup_forces(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        self.scratch.begin(self.step);
        for phase in [
            Phase::PreNeighbor,
            Phase::Neighbor,
            Phase::PreForce,
            Phase::Force,
            Phase::PostForce,
        ] {
            self.launch_phase(phase, frame)?;
        }
        Ok(())
    }

    fn step_once(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        for phase in Phase::ALL {
            self.launch_phase(phase, frame)?;
        }
        Ok(())
    }

    fn launch_phase(&mut self, phase: Phase, frame: &mut Frame) -> Result<(), MdError> {
        for k in 0..self.schedule[phase as usize].len() {
            let index = self.schedule[phase as usize][k];
            self.installed[index]
                .component
                .launch(phase, frame, &mut self.scratch)?;
        }
        Ok(())
    }

    fn remove_kind(&mut self, kind: Kind) {
        self.installed.retain(|c| c.flags.kind != kind);
        self.built = false;
    }

    fn install<T: Component + 'static>(
        &mut self,
        component: T,
        bindings: Vec<(Phase, i32)>,
        flags: Flags,
    ) {
        self.built = false;
        self.installed.push(Installed {
            bindings,
            flags,
            component: Box::new(component),
        });
    }
}

impl Installed {
    fn order_on(&self, phase: Phase) -> Option<i32> {
        self.bindings
            .iter()
            .find(|(bound, _)| *bound == phase)
            .map(|(_, order)| *order)
    }
}

fn require_positive_dt(dt: F) -> Result<F, MdError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(MdError::Invalid(format!("dt must be > 0, got {dt}")));
    }
    Ok(dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::md::component::Component;
    use crate::md::evaluator::FfEvaluator;
    use crate::md::fix_langevin::FixLangevin;
    use crate::md::verlet::VelocityVerlet;
    use molrs::core::{Block, Frame};
    use molrs::ff::potential::{Accumulator, Potential};
    use molrs::op::F;

    struct Push {
        axis_force: F,
    }

    impl Potential for Push {
        fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
            let n = coords.len() / 3;
            let out = acc.forces_without_virial();
            for i in 0..n {
                out[i * 3] += self.axis_force;
            }
        }
    }

    struct Counter {
        hits: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Component for Counter {
        fn name(&self) -> &str {
            "counter"
        }
        fn default_bindings(&self) -> Vec<(Phase, i32)> {
            vec![]
        }
        fn launch(
            &mut self,
            _phase: Phase,
            _frame: &mut Frame,
            _scratch: &mut Scratch,
        ) -> Result<(), MdError> {
            self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    fn frame_at_rest(n: usize, mass: F) -> Frame {
        let mut atoms = Block::new();
        let z = ndarray::Array1::from_elem(n, 0.0);
        atoms.insert("x", z.clone().into_dyn(), None).unwrap();
        atoms.insert("y", z.clone().into_dyn(), None).unwrap();
        atoms.insert("z", z.clone().into_dyn(), None).unwrap();
        atoms.insert("vx", z.clone().into_dyn(), None).unwrap();
        atoms.insert("vy", z.clone().into_dyn(), None).unwrap();
        atoms.insert("vz", z.into_dyn(), None).unwrap();
        atoms
            .insert("mass", ndarray::Array1::from_elem(n, mass).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    #[test]
    fn an_empty_phase_is_still_walked() {
        let hits = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut md = MD::new(UnitPreset::si()).unwrap();
        md.set_integrator(VelocityVerlet::new(0.1).unwrap())
            .unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 0.0 }))
            .unwrap();
        md.on(Phase::PostIntegrate, Counter { hits: hits.clone() }, 0);
        md.on(Phase::PreExchange, Counter { hits: hits.clone() }, 0);
        let mut frame = frame_at_rest(1, 1.0);
        md.run(&mut frame, 3).unwrap();
        assert_eq!(hits.load(std::sync::atomic::Ordering::Relaxed), 6);
    }

    #[test]
    fn run_refuses_a_missing_integrator() {
        let mut md = MD::new(UnitPreset::si()).unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 0.0 }))
            .unwrap();
        let err = md.run(&mut frame_at_rest(1, 1.0), 1).unwrap_err();
        assert!(err.to_string().contains("set_integrator"), "{err}");
    }

    #[test]
    fn constant_force_matches_the_velocity_verlet_step() {
        let dt = 0.1;
        let mut md = MD::new(UnitPreset::si()).unwrap();
        md.set_integrator(VelocityVerlet::new(dt).unwrap()).unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 1.0 }))
            .unwrap();
        let mut frame = frame_at_rest(1, 1.0);
        md.run(&mut frame, 1).unwrap();
        let x = frame["atoms"].get("x").unwrap().get::<F>().unwrap()[0];
        let vx = frame["atoms"].get("vx").unwrap().get::<F>().unwrap()[0];
        // v = dt * f / m, x = dt * (dt * f / m) / 2
        assert!((vx - dt).abs() < 1e-12, "{vx}");
        assert!((x - 0.5 * dt * dt).abs() < 1e-12, "{x}");
    }

    #[test]
    fn a_second_integrator_replaces_the_first() {
        let mut md = MD::new(UnitPreset::si()).unwrap();
        md.set_integrator(VelocityVerlet::new(0.1).unwrap())
            .unwrap();
        md.set_integrator(VelocityVerlet::new(0.2).unwrap())
            .unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 0.0 }))
            .unwrap();
        md.run(&mut frame_at_rest(1, 1.0), 0).unwrap();
    }

    /// `E = ½ k x²` on every coordinate.
    struct Spring {
        k: F,
    }

    impl Potential for Spring {
        fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
            let mut e = 0.0;
            let out = acc.forces_without_virial();
            for (f, &x) in out.iter_mut().zip(coords) {
                *f -= self.k * x;
                e += 0.5 * self.k * x * x;
            }
            acc.add_energy(e);
        }
    }

    fn column(frame: &Frame, key: &str) -> Vec<F> {
        frame["atoms"]
            .get(key)
            .unwrap()
            .get::<F>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    /// One kick in `real`: m = 12 g/mol, f = 1 kcal/mol/Å, dt = 1 fs gives
    /// Δv = dt·f/m·ftm2v = 4.184e-4/12 Å/fs (both half-kicks, the force
    /// constant).
    #[test]
    fn a_real_step_applies_ftm2v() {
        let mut md = MD::new(UnitPreset::real()).unwrap();
        md.set_integrator(VelocityVerlet::new(1.0).unwrap())
            .unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 1.0 }))
            .unwrap();
        let mut frame = frame_at_rest(1, 12.0);
        md.run(&mut frame, 1).unwrap();
        let vx = column(&frame, "vx")[0];
        assert!((vx - 4.184e-4 / 12.0).abs() < 1e-12, "{vx}");
    }

    /// The same step in `metal`: dt = 1 ps, f = 1 eV/Å, so
    /// Δv = 1/(12 · 1.036 426 9e-4) Å/ps.
    #[test]
    fn a_metal_step_applies_ftm2v() {
        let mut md = MD::new(UnitPreset::metal()).unwrap();
        md.set_integrator(VelocityVerlet::new(1.0).unwrap())
            .unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 1.0 }))
            .unwrap();
        let mut frame = frame_at_rest(1, 12.0);
        md.run(&mut frame, 1).unwrap();
        let vx = column(&frame, "vx")[0];
        let expected = 1.0 / (12.0 * 1.036_426_9e-4);
        assert!((vx / expected - 1.0).abs() < 1e-6, "{vx}");
    }

    /// Velocity Verlet on m = k = 1, h = 0.1 from q = 1, p = 0: one step is
    /// q = 1 − h²/2 = 0.995 and p = −(h/2)(q₀ + q₁) = −0.09975, and the
    /// shadow energy `½p²/(1 − h²/4) + ½q²` stays at 0.5 for 10⁴ steps.
    #[test]
    fn the_harmonic_step_and_its_shadow_energy() {
        let h = 0.1;
        let mut md = MD::new(UnitPreset::si()).unwrap();
        md.set_integrator(VelocityVerlet::new(h).unwrap()).unwrap();
        md.set_evaluator(FfEvaluator::plain(Spring { k: 1.0 }))
            .unwrap();
        let mut frame = frame_at_rest(1, 1.0);
        frame
            .get_mut("atoms")
            .unwrap()
            .insert("x", ndarray::Array1::from_elem(1, 1.0).into_dyn(), None)
            .unwrap();
        md.run(&mut frame, 1).unwrap();
        let (q, p) = (column(&frame, "x")[0], column(&frame, "vx")[0]);
        assert!((q - 0.995).abs() < 1e-15, "{q}");
        assert!((p + 0.099_75).abs() < 1e-15, "{p}");
        md.run(&mut frame, 9_999).unwrap();
        let (q, p) = (column(&frame, "x")[0], column(&frame, "vx")[0]);
        let shadow = 0.5 * p * p / (1.0 - h * h / 4.0) + 0.5 * q * q;
        assert!((shadow - 0.5).abs() < 1e-12, "{shadow}");
        assert_eq!(md.step(), 10_000);
    }

    /// Two runs of 5 steps are one run of 10: the step counter, and with it
    /// the Langevin noise, carries across `run`.
    #[test]
    fn chained_runs_are_one_run() {
        let langevin = || {
            let mut md = MD::new(UnitPreset::real()).unwrap();
            md.set_integrator(VelocityVerlet::new(1.0).unwrap())
                .unwrap();
            md.set_evaluator(FfEvaluator::plain(Spring { k: 0.5 }))
                .unwrap();
            md.set_thermostat(FixLangevin::new(300.0, 100.0, 1.0, 11).unwrap())
                .unwrap();
            md
        };
        let mut chained = frame_at_rest(3, 12.0);
        let mut md = langevin();
        md.run(&mut chained, 5).unwrap();
        md.run(&mut chained, 5).unwrap();
        let mut single = frame_at_rest(3, 12.0);
        langevin().run(&mut single, 10).unwrap();
        for key in ["x", "y", "z", "vx", "vy", "vz", "fx"] {
            assert_eq!(column(&chained, key), column(&single, key), "{key}");
        }
        assert_ne!(column(&single, "vx"), vec![0.0; 3]);
    }
}
