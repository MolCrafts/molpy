//! Geometry optimization of a [`Frame`].
//!
//! The optimizer contract lives here and is always compiled: the
//! [`Optimizer`] trait, its [`OptimizationReport`], the [`LbfgsSettings`]
//! defaults and the `atoms.free` mask
//! ([`set_free_mask`]). A purely geometric optimizer (e.g. a packer's
//! torsion Monte Carlo) implements [`Optimizer`] without enabling `ff`. The
//! force-field-agnostic L-BFGS engine behind `Lbfgs` is crate-internal; the
//! ETKDG conformer stages drive it on their own distance-geometry energies.
//!
//! The optimizer that minimizes a force-field potential
//! (`ff::potential::Potential`), `Lbfgs` — over an owned or a borrowed
//! potential; a `Frame`, one flat coordinate buffer or a batch of them — is
//! gated on `ff`. It is the one front door for every potential — a potential
//! that rebuilds its pairs as the atoms move
//! (`ff::potential::soft::SoftPotential`) does so itself.
//! The dependency points one way: `optimize` consumes `ff`, never the
//! reverse.

#[cfg(feature = "ff")]
mod lbfgs;
#[cfg(feature = "ff")]
mod lbfgs_driver;

#[cfg(feature = "ff")]
pub use lbfgs_driver::Lbfgs;

use crate::core::keys::FREE;
use crate::core::schema::block_names::ATOMS;
use crate::core::{BlockError, Frame, MolRsError};
use crate::op::F;
use ndarray::Array1;

#[cfg(feature = "conformer")]
pub(crate) use lbfgs::minimize_lbfgs_rms;

/// Outcome of one minimization — the one report every minimizer in molrs
/// returns, the geometry optimizer and the conformer stages alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationReport {
    /// Whether a tolerance stopped the run (`fmax`, `etol` or `ftol`), not a
    /// budget (`max_steps`, `max_evals`) or a stuck line search.
    pub converged: bool,
    /// Which criterion stopped the run.
    pub stop_reason: StopReason,
    /// Number of outer iterations performed.
    pub n_steps: usize,
    /// Force/energy evaluations, line-search trials included.
    pub n_evals: usize,
    /// Potential energy at the returned point, in the potential's energy unit
    /// (kcal/mol for the force fields molrs ships).
    pub final_energy: F,
    /// Maximum per-atom force magnitude at the returned point (energy / Å).
    pub final_fmax: F,
    /// Root-mean-square gradient component at the returned point (energy / Å).
    pub final_grad_rms: F,
}

impl OptimizationReport {
    /// The report of a minimization that stopped at `energy` with gradient
    /// `grad` (= −forces, flat `3·n_atoms`).
    #[cfg(feature = "ff")]
    pub(crate) fn from_gradient(
        stop_reason: StopReason,
        n_steps: usize,
        n_evals: usize,
        energy: F,
        grad: &[F],
    ) -> Self {
        Self {
            converged: stop_reason.converged(),
            stop_reason,
            n_steps,
            n_evals,
            final_energy: energy,
            final_fmax: lbfgs::fmax_from_grad(grad),
            final_grad_rms: lbfgs::grad_rms(grad),
        }
    }

    /// The report of an empty system: nothing to move, converged at zero.
    #[cfg(any(feature = "ff", test))]
    pub(crate) const EMPTY: Self = Self {
        converged: true,
        stop_reason: StopReason::Empty,
        n_steps: 0,
        n_evals: 0,
        final_energy: 0.0,
        final_fmax: 0.0,
        final_grad_rms: 0.0,
    };
}

/// Which criterion ended a minimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The largest per-atom force (or the RMS gradient) fell below its tolerance.
    Gradient,
    /// The 2-norm of the global force vector fell to `ftol` (LAMMPS `ftol`).
    ForceNorm,
    /// The relative energy change fell below `etol` (LAMMPS `etol`).
    EnergyChange,
    /// `max_steps` outer iterations ran (LAMMPS `maxiter`).
    MaxIterations,
    /// `max_evals` force/energy evaluations ran (LAMMPS `maxeval`).
    MaxEvaluations,
    /// No step lowered the energy: the line search is stuck.
    LineSearch,
    /// Nothing to move.
    Empty,
}

impl StopReason {
    /// A tolerance was met (or there was nothing to do).
    pub const fn converged(self) -> bool {
        matches!(
            self,
            Self::Gradient | Self::ForceNorm | Self::EnergyChange | Self::Empty
        )
    }

    /// The lower-snake name the bindings report.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gradient => "gradient",
            Self::ForceNorm => "force_norm",
            Self::EnergyChange => "energy_change",
            Self::MaxIterations => "max_iterations",
            Self::MaxEvaluations => "max_evaluations",
            Self::LineSearch => "line_search",
            Self::Empty => "empty",
        }
    }
}

/// The knobs of an L-BFGS geometry minimization, and their one set of
/// defaults — every binding reads [`LbfgsSettings::DEFAULT`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsSettings {
    /// Converged when the largest per-atom force falls below this (energy / Å).
    pub fmax: F,
    /// Most outer iterations before giving up.
    pub max_steps: usize,
    /// Largest displacement of any coordinate in one step (Å): the trust
    /// region.
    pub max_step: F,
    /// Correction pairs kept in the L-BFGS history.
    pub memory: usize,
    /// Stop when the relative energy change between successive iterations
    /// falls below it (the test LAMMPS's `etol` makes). `0` disables it.
    pub etol: F,
    /// Stop when the 2-norm of the global force vector falls to it
    /// (energy / Å). `0` disables it.
    pub ftol: F,
    /// Most force/energy evaluations.
    pub max_evals: usize,
}

impl LbfgsSettings {
    /// `fmax = 0.05`, `max_steps = 500`, `max_step = 0.2`, `memory = 8`; the
    /// energy-change and force-norm criteria off (`etol = ftol = 0`),
    /// unbounded `max_evals`.
    pub const DEFAULT: Self = Self {
        fmax: 0.05,
        max_steps: 500,
        max_step: 0.2,
        memory: 8,
        etol: 0.0,
        ftol: 0.0,
        max_evals: usize::MAX,
    };

    #[cfg(feature = "ff")]
    pub(crate) fn limits(&self) -> lbfgs::Limits {
        lbfgs::Limits {
            etol: self.etol,
            ftol: self.ftol,
            max_evals: self.max_evals,
        }
    }
}

impl Default for LbfgsSettings {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why a minimization could not run.
#[derive(Debug)]
pub enum OptimizeError {
    /// A flat coordinate buffer is not `3·n_atoms` long.
    CoordsLength {
        /// The buffer's length.
        len: usize,
    },
    /// A batch buffer is not `n_structs · n_atoms · 3` long, or names no
    /// atoms per structure for a non-empty batch.
    BatchShape {
        /// The buffer's length.
        len: usize,
        /// Structures in the batch.
        n_structs: usize,
        /// Atoms per structure.
        n_atoms: usize,
    },
    /// The free mask does not have one entry per atom.
    FreeMask {
        /// Entries in the mask.
        len: usize,
        /// Atoms it was given for.
        n_atoms: usize,
    },
    /// The frame has no `atoms` block to read or write.
    NoAtoms,
    /// The frame's coordinates could not be read or written back.
    Frame(MolRsError),
    /// An optimizer outside molrs refused, in its own terms.
    Implementor(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoordsLength { len } => write!(
                f,
                "coords length {len} is not a multiple of 3 (expected 3·n_atoms)"
            ),
            Self::BatchShape {
                len,
                n_structs,
                n_atoms,
            } => write!(
                f,
                "coords length {len} is not n_structs ({n_structs}) · n_atoms ({n_atoms}) · 3 \
                 with n_atoms > 0"
            ),
            Self::FreeMask { len, n_atoms } => {
                write!(f, "free mask has {len} entries for {n_atoms} atoms")
            }
            Self::NoAtoms => write!(f, "the frame has no atoms block"),
            Self::Frame(err) => write!(f, "frame coordinates: {err}"),
            Self::Implementor(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for OptimizeError {}

impl From<MolRsError> for OptimizeError {
    fn from(err: MolRsError) -> Self {
        Self::Frame(err)
    }
}

impl From<BlockError> for OptimizeError {
    fn from(err: BlockError) -> Self {
        Self::Frame(err.into())
    }
}

/// Geometry optimizer: minimize a [`Frame`] in place.
///
/// Optional free mask: bool column `atoms.free` (missing ⇒ every atom free).
/// Fixed atoms stay in the potential evaluation but are not optimizable DOFs.
pub trait Optimizer: Send + Sync {
    /// Relax `frame` in place. Coordinates are `atoms.{x,y,z}`.
    ///
    /// # Errors
    ///
    /// The [`OptimizeError`] naming why the frame cannot be minimized; an
    /// optimizer outside molrs reports its own failure as
    /// [`OptimizeError::Implementor`].
    fn minimize(&mut self, frame: &mut Frame) -> Result<OptimizationReport, OptimizeError>;
}

/// Write `free` as the `atoms.free` mask a minimization reads.
///
/// # Errors
///
/// [`OptimizeError::NoAtoms`] for a frame without an `atoms` block and
/// [`OptimizeError::FreeMask`] for a mask that is not one entry per atom.
pub fn set_free_mask(frame: &mut Frame, free: &[bool]) -> Result<(), OptimizeError> {
    let atoms = frame.get_mut(ATOMS).ok_or(OptimizeError::NoAtoms)?;
    let n_atoms = atoms.n_rows().unwrap_or(0);
    if free.len() != n_atoms {
        return Err(OptimizeError::FreeMask {
            len: free.len(),
            n_atoms,
        });
    }
    atoms.insert(FREE, Array1::from_vec(free.to_vec()).into_dyn(), None)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A purely geometric optimizer: translates the atoms so their centroid
    /// sits at the origin. Needs no force field, so it builds without `ff`.
    struct Center;

    impl Optimizer for Center {
        fn minimize(&mut self, frame: &mut Frame) -> Result<OptimizationReport, OptimizeError> {
            let mut xyz = frame.coords()?;
            let centroid = xyz.mean_axis(ndarray::Axis(0)).unwrap_or_default();
            xyz -= &centroid;
            frame.set_coords(xyz.view())?;
            Ok(OptimizationReport {
                n_steps: 1,
                ..OptimizationReport::EMPTY
            })
        }
    }

    #[test]
    fn geometric_optimizer_runs_on_a_frame_without_ff() {
        let mut frame = Frame::new();
        frame
            .set_coords(array![[1.0 as F, 0.0, 0.0], [3.0, 2.0, 0.0]].view())
            .unwrap();
        let report = Center.minimize(&mut frame).unwrap();
        assert!(report.converged);
        assert_eq!(
            frame.coords().unwrap(),
            array![[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]
        );
    }

    #[test]
    fn set_free_mask_rejects_a_length_mismatch() {
        let mut frame = Frame::new();
        frame
            .set_coords(array![[0.0 as F, 0.0, 0.0], [1.0, 0.0, 0.0]].view())
            .unwrap();
        assert!(set_free_mask(&mut frame, &[true]).is_err());
        set_free_mask(&mut frame, &[false, true]).unwrap();
        let free = frame
            .get(ATOMS)
            .unwrap()
            .get(FREE)
            .and_then(|c| c.get::<bool>())
            .unwrap();
        assert_eq!(free.iter().copied().collect::<Vec<_>>(), vec![false, true]);
    }
}
