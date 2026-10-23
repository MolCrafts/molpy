//! Per-step scratch the driver owns and hands to every `launch`.
//!
//! The neighbour list publishes its pairs here during `neighbor`; the force
//! component reads them during `force`.

use molrs::core::Neighbors;

/// Data a phase publishes for a later phase in the same step.
pub struct Scratch {
    /// The pairs of the `neighbor` phase. `None` until that phase publishes.
    pub pairs: Option<Neighbors>,
    /// Step index, starting at 0. Thermostats with a temperature ramp read it.
    pub step: usize,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self {
            pairs: None,
            step: 0,
        }
    }
}
