//! [`MD`] walks [`Phase::ALL`] once per step and launches the components
//! bound to each phase in ascending order.

use molrs::core::Frame;
use molrs::op::F;

use super::component::{Component, Flags, Integrator, Kind, TimestepProvider};
use super::error::MdError;
use super::evaluator::FfEvaluator;
use super::fix_langevin::FixLangevin;
use super::infra::{KineticEnergy, ZeroForces};
use super::neighbor_list::NeighborList;
use super::phase::Phase;
use super::scratch::Scratch;

struct Installed {
    bindings: Vec<(Phase, i32)>,
    flags: Flags,
    component: Option<Box<dyn Component>>,
}

/// In-process molecular dynamics. One driver, the LAMMPS phase list, and
/// components inserted onto it.
///
/// `run` writes the frame it is given. It does not return a second state.
pub struct MD {
    installed: Vec<Installed>,
    scratch: Scratch,
    built: bool,
}

impl Default for MD {
    fn default() -> Self {
        Self::new()
    }
}

impl MD {
    /// A driver with the force-clear and kinetic-energy infrastructure bound.
    /// The clear runs at `pre_force` order -1000. The kinetic energy runs at
    /// `end_of_step` order -100 and writes `frame.meta["kinetic"]`.
    pub fn new() -> Self {
        let mut md = Self {
            installed: Vec::new(),
            scratch: Scratch::new(),
            built: false,
        };
        md.install(ZeroForces, ZeroForces.default_bindings(), Flags::default());
        md.install(
            KineticEnergy,
            KineticEnergy.default_bindings(),
            Flags::default(),
        );
        md
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
    pub fn set_thermostat(&mut self, thermostat: FixLangevin) -> Result<(), MdError> {
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

    /// Replace any neighbour list already bound through this setter.
    pub fn set_neighbor(&mut self, neighbor: NeighborList) -> Result<(), MdError> {
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
                .and_then(|c| c.component.as_ref().map(|p| p.name().to_owned()))
                .unwrap_or_else(|| "evaluator".to_owned());
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
            if let Some(component) = installed.component.as_mut() {
                component.build(frame)?;
            }
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
        for step in 0..n_steps {
            self.scratch.step = step;
            self.scratch.pairs = None;
            self.step_once(frame)?;
        }
        Ok(())
    }

    fn has_force_phase_component(&self) -> bool {
        self.installed.iter().any(|c| {
            c.bindings.iter().any(|(phase, _)| *phase == Phase::Force)
                && c.component
                    .as_ref()
                    .is_some_and(|p| p.name() != "zero_forces")
        })
    }

    fn setup_forces(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        self.scratch.step = 0;
        self.scratch.pairs = None;
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
        let mut order: Vec<(i32, usize)> = self
            .installed
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.order_on(phase).map(|ord| (ord, i)))
            .collect();
        order.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        for (_, index) in order {
            let mut component = self.installed[index]
                .component
                .take()
                .expect("component is present between launches");
            let result = component.launch(phase, frame, &mut self.scratch);
            self.installed[index].component = Some(component);
            result?;
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
            component: Some(Box::new(component)),
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
        let mut md = MD::new();
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
        let mut md = MD::new();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 0.0 }))
            .unwrap();
        let err = md.run(&mut frame_at_rest(1, 1.0), 1).unwrap_err();
        assert!(err.to_string().contains("set_integrator"), "{err}");
    }

    #[test]
    fn constant_force_matches_the_velocity_verlet_step() {
        let dt = 0.1;
        let mut md = MD::new();
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
        let mut md = MD::new();
        md.set_integrator(VelocityVerlet::new(0.1).unwrap())
            .unwrap();
        md.set_integrator(VelocityVerlet::new(0.2).unwrap())
            .unwrap();
        md.set_evaluator(FfEvaluator::plain(Push { axis_force: 0.0 }))
            .unwrap();
        md.run(&mut frame_at_rest(1, 1.0), 0).unwrap();
    }
}
