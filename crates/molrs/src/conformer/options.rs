//! Public options for 3D generation.

/// Options for [`Conformer`](super::Conformer): pass them to
/// [`Conformer::new`](super::Conformer::new), which
/// [`generate`](super::Conformer::generate) then runs against.
///
/// The pipeline is fixed: ETKDG embedding, then an MMFF94 cleanup
/// minimization (skipped, with a warning, for a molecule MMFF94 does not
/// cover).
#[derive(Debug, Clone)]
pub struct ConformerOptions {
    /// Add explicit hydrogens before generation.
    pub add_hydrogens: bool,
    /// The ETKDG embedding-attempt budget (RDKit's `maxIterations`); `0`
    /// means RDKit's `10 × n_atoms`.
    pub max_steps: usize,
    /// Optional deterministic RNG seed.
    pub rng_seed: Option<u64>,
}

impl Default for ConformerOptions {
    fn default() -> Self {
        Self {
            add_hydrogens: true,
            max_steps: 0,
            rng_seed: None,
        }
    }
}
