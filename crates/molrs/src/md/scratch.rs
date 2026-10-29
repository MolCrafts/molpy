//! Per-step scratch the driver owns and hands to every `launch`.
//!
//! The neighbour list publishes its pairs here during `neighbor`; the force
//! component reads them during `force`.

use molrs::core::{NeighborColumns, Neighbors, QueryMode};

/// Data a phase publishes for a later phase in the same step.
pub struct Scratch {
    /// The pair table, kept across steps so a publish reuses its storage.
    pairs: Neighbors,
    /// Whether `neighbor` published `pairs` in this step.
    published: bool,
    step: u64,
}

impl Scratch {
    pub(crate) fn new() -> Self {
        Self {
            pairs: Neighbors::empty(QueryMode::SelfQuery { n_points: 0 }, NeighborColumns::FULL),
            published: false,
            step: 0,
        }
    }

    /// Start step `step`: nothing is published yet.
    pub(crate) fn begin(&mut self, step: u64) {
        self.step = step;
        self.published = false;
    }

    /// The run's step index. It counts across [`MD::run`](crate::md::MD::run)
    /// calls; the force evaluation before the first step is step 0.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// The pairs the `neighbor` phase published this step, or `None`.
    pub fn pairs(&self) -> Option<&Neighbors> {
        self.published.then_some(&self.pairs)
    }

    /// The table to publish into this step. Its rows are the publisher's to
    /// overwrite.
    pub(crate) fn publish_pairs(&mut self) -> &mut Neighbors {
        self.published = true;
        &mut self.pairs
    }
}
