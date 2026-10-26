//! Stage-level reporting for 3D generation.

/// Pipeline stage identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageKind {
    /// Pre-processing (hydrogen handling and molecular-feature perception).
    Preprocess,
    /// Initial coordinate construction.
    BuildInitial,
    /// First minimization pass.
    CoarseOptimize,
    /// Final minimization pass.
    FinalOptimize,
    /// Stereo sanity checks.
    StereoCheck,
}

/// Single-stage run metrics.
#[derive(Debug, Clone)]
pub struct ConformerStageReport {
    /// Stage identifier.
    pub stage: StageKind,
    /// Optional input energy for the stage.
    pub energy_before: Option<f64>,
    /// Optional output energy for the stage.
    pub energy_after: Option<f64>,
    /// Iterations / attempts performed.
    pub steps: usize,
    /// Convergence flag for optimization-like stages.
    pub converged: bool,
}

/// End-to-end generation report.
#[derive(Debug, Clone)]
pub struct ConformerReport {
    /// Stage-by-stage execution data.
    pub stages: Vec<ConformerStageReport>,
    /// Non-fatal issues and behavior notes.
    pub warnings: Vec<String>,
    /// The MMFF94 energy after the cleanup minimization; `None` when the
    /// cleanup did not run.
    pub final_energy: Option<f64>,
}

impl ConformerReport {
    pub(crate) fn new() -> Self {
        Self {
            stages: Vec::new(),
            warnings: Vec::new(),
            final_energy: None,
        }
    }
}
