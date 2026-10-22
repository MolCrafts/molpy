//! Per-step scratch the driver owns and hands to every `launch`.
//!
//! The neighbour component publishes the pair table here during `neighbor`.
//! The force component reads it during `force`. The table is not a second
//! copy of the coordinates: it is the pair list for this step.

use molrs::core::Neighbors;

/// Data a phase publishes for a later phase in the same step.
pub struct Scratch {
    /// Pairs from the `neighbor` phase. `None` until that phase publishes.
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
