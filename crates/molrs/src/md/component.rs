//! A component is one unit of work bound to one or more phases.
//!
//! Kinds (`Integrator`, `Thermostat`, …) are marker traits the driver records
//! when the component is registered. `Component` itself has no kind method:
//! a probe that every implementor would have to answer belongs on the kind
//! that is actually scanned.

use molrs::core::Frame;
use molrs::op::F;

use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// One scheduled body. `launch` runs once per bound phase per step and does
/// not allocate its buffers; those are sized in `build`.
pub trait Component: Send {
    /// Timing and error name.
    fn name(&self) -> &str;

    /// Where this component runs when a categorical setter installs it.
    /// `on(phase).add` ignores this and uses the caller's phase and order.
    fn default_bindings(&self) -> Vec<(Phase, i32)>;

    /// One-time setup against the frame the run will step. Default: nothing.
    fn build(&mut self, frame: &mut Frame) -> Result<(), MdError> {
        let _ = frame;
        Ok(())
    }

    /// The work of `phase`. `scratch.pairs` is whatever the neighbour phase
    /// published earlier in this step.
    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        scratch: &mut Scratch,
    ) -> Result<(), MdError>;
}

/// Provides the timestep other components must agree with.
pub trait TimestepProvider {
    /// Δt in the caller's time unit (fs when the rest of the frame is).
    fn dt(&self) -> F;
}

/// Advances positions and velocities. Binds `initial_integrate` and
/// `final_integrate`. Does not evaluate a potential.
pub trait Integrator: Component + TimestepProvider {}

/// Modifies the force or the velocities to control temperature.
pub trait Thermostat: Component + TimestepProvider {}

/// What the driver recorded at registration, so a later scan does not need
/// the concrete type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Flags {
    pub kind: Kind,
    pub consumes_neighbors: bool,
    pub frozen_pairs: bool,
    pub dt: Option<F>,
}

impl Default for Flags {
    fn default() -> Self {
        Self {
            kind: Kind::Other,
            consumes_neighbors: false,
            frozen_pairs: false,
            dt: None,
        }
    }
}

/// Why a component was registered. Categorical setters use this for the
/// singular-replace / plural-append rule. `on(phase).add` records [`Kind::Other`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Integrator,
    Evaluator,
    Thermostat,
    Neighbor,
    /// Infrastructure and user `on(phase).add` components.
    Other,
}
