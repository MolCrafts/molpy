//! Public options for 3D generation.

/// Force-field backend selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceFieldKind {
    /// Merck Molecular Force Field 94.
    Mmff94,
    /// Universal Force Field.
    Uff,
    /// Prefer MMFF94, fall back to UFF.
    Auto,
}

/// Preset quality/speed profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformerSpeed {
    /// Favor throughput.
    Fast,
    /// Balanced defaults.
    Medium,
    /// Favor embedding quality.
    Better,
}

/// Options for [`Conformer`](super::Conformer): pass them to
/// [`Conformer::new`](super::Conformer::new), which
/// [`generate`](super::Conformer::generate) then runs against.
#[derive(Debug, Clone)]
pub struct ConformerOptions {
    /// Target force-field family (or auto selection).
    pub forcefield: ForceFieldKind,
    /// Throughput/quality preset.
    pub speed: ConformerSpeed,
    /// Add explicit hydrogens before generation.
    pub add_hydrogens: bool,
    /// Total optimization budget. `0` means "use speed preset default".
    pub max_steps: usize,
    /// Optional deterministic RNG seed.
    pub rng_seed: Option<u64>,
}

impl Default for ConformerOptions {
    fn default() -> Self {
        Self {
            forcefield: ForceFieldKind::Auto,
            speed: ConformerSpeed::Medium,
            add_hydrogens: true,
            max_steps: 0,
            rng_seed: None,
        }
    }
}

impl ConformerOptions {
    // --- ETKDG-internal knobs ------------------------------------------------
    //
    // The ETKDG pipeline supplies defaults for knobs not exposed publicly.
    // `max_steps` doubles as the explicit `maxIterations` override (0 =
    // RDKit's `10×n_atoms` heuristic).

    /// ETKDG `maxIterations` override. `max_steps == 0` means "use the RDKit
    /// `10 × n_atoms` heuristic" (resolved in `etkdg::retry`).
    pub(crate) fn max_iterations_internal(&self) -> usize {
        self.max_steps
    }

    /// Whether the `useRandomCoords` fallback embedding is allowed. Always on
    /// internally (matches RDKit's default retry behavior on hard cases).
    pub(crate) fn use_random_coords_fallback_internal(&self) -> bool {
        true
    }

    /// Whether the second-stage MMFF94 cleanup minimization runs. Always on
    /// internally (the spec's stage-3 contract).
    pub(crate) fn mmff_cleanup_internal(&self) -> bool {
        true
    }
}
