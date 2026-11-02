//! Python wrapper for 3D conformer generation from molecular graphs.
//!
//! The public surface is the [`PyConformer`] class (`molpy.conformer.Conformer`): its
//! constructor declares the generation parameters and [`PyConformer::generate`]
//! runs the pipeline, returning `(mol_3d, report)`.
//!
//! The pipeline takes a [`PyAtomistic`] molecular graph (ports and dummy
//! atoms included) and embeds it in 3D with ETKDGv3 — RDKit's `EmbedMolecule`,
//! and like it no force field:
//!
//! 1. **Preprocess** -- add hydrogens and perceive molecular features.
//! 2. **Build initial** -- distance-geometry embedding (or random coordinates).
//! 3. **Coarse optimize** -- distance and chirality minimization, then the
//!    experimental-torsion / basic-knowledge refinement.
//! 4. **Stereo check** -- verify stereochemistry is preserved.
//!
//! An MMFF94 relaxation is a separate step the caller composes
//! (`ff.typifier.Mmff94Typifier` → `ff.PotentialCompiler` → `optimize.Lbfgs`).
//!
//! # References
//!
//! - Riniker, S.; Landrum, G.A. (2015). J. Chem. Inf. Model. 55, 2562-2574.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use molrs::conformer::{Conformer, ConformerReport, StageKind};

use crate::core::molgraph::PyAtomistic;
use crate::error::molrs_error_to_pyerr;

/// Map a `StageKind` enum to a human-readable name.
fn stage_kind_name(kind: StageKind) -> &'static str {
    match kind {
        StageKind::Preprocess => "preprocess",
        StageKind::BuildInitial => "build_initial",
        StageKind::CoarseOptimize => "coarse_optimize",
        StageKind::StereoCheck => "stereo_check",
    }
}

/// Project a core [`ConformerReport`] onto its Python shape.
///
/// Both `generate` arms reach this, which is why it is a function rather than
/// an inline block: the per-stage mapping is one fact and is written once.
fn report_to_py(report: ConformerReport) -> PyConformerReport {
    let stages = report
        .stages
        .iter()
        .map(|s| PyConformerStageReport {
            stage: stage_kind_name(s.stage).to_string(),
            energy_before: s.energy_before,
            energy_after: s.energy_after,
            steps: s.steps,
            converged: s.converged,
        })
        .collect();
    PyConformerReport {
        warnings: report.warnings,
        stages_inner: stages,
    }
}

/// Report for a single stage of conformer generation.
///
/// Exposed to Python as `molpy.conformer.ConformerStageReport`.
///
/// Attributes
/// ----------
/// stage : str
///     Stage name (e.g. ``"build_initial"``, ``"coarse_optimize"``).
/// energy_before : float | None
///     Energy in kcal/mol before this stage, or ``None`` if not applicable.
/// energy_after : float | None
///     Energy in kcal/mol after this stage, or ``None`` if not applicable.
/// steps : int
///     Number of optimizer steps executed.
/// converged : bool
///     Whether the stage converged.
#[pyclass(module = "molpy.conformer", name = "ConformerStageReport")]
pub struct PyConformerStageReport {
    #[pyo3(get)]
    pub stage: String,
    #[pyo3(get)]
    pub energy_before: Option<f64>,
    #[pyo3(get)]
    pub energy_after: Option<f64>,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyConformerStageReport {
    fn __repr__(&self) -> String {
        format!(
            "ConformerStageReport(stage='{}', steps={}, converged={})",
            self.stage, self.steps, self.converged
        )
    }
}

/// Aggregate report for a conformer generation run.
///
/// Exposed to Python as `molpy.conformer.ConformerReport`.
///
/// Attributes
/// ----------
/// warnings : list[str]
///     Diagnostic warnings generated during the run.
/// stages : list[ConformerStageReport]
///     Per-stage execution reports.
///
/// Examples
/// --------
/// >>> mol_3d, report = Conformer(seed=42).generate(mol)
/// >>> for s in report.stages:
/// ...     print(s.stage, s.converged)
#[pyclass(module = "molpy.conformer", name = "ConformerReport")]
pub struct PyConformerReport {
    #[pyo3(get)]
    pub warnings: Vec<String>,
    pub(crate) stages_inner: Vec<PyConformerStageReport>,
}

#[pymethods]
impl PyConformerReport {
    /// Per-stage execution reports.
    ///
    /// Returns
    /// -------
    /// list[ConformerStageReport]
    #[getter]
    fn stages(&self) -> Vec<PyConformerStageReport> {
        self.stages_inner
            .iter()
            .map(|s| PyConformerStageReport {
                stage: s.stage.clone(),
                energy_before: s.energy_before,
                energy_after: s.energy_after,
                steps: s.steps,
                converged: s.converged,
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConformerReport(stages={}, warnings={})",
            self.stages_inner.len(),
            self.warnings.len()
        )
    }
}

/// ETKDGv3 3D embedding of a molecular graph — RDKit's ``EmbedMolecule``.
///
/// Exposed to Python as `molpy.conformer.Conformer`. Construct it with the
/// embedding parameters (RDKit ``EmbedParameters`` names), then call
/// :meth:`generate`. It runs no force field: relax the result with MMFF94 as a
/// separate step when that is wanted.
///
/// Parameters
/// ----------
/// add_hydrogens : bool, optional
///     Add implicit hydrogens as explicit atoms first (default ``True``).
/// max_iterations : int, optional
///     Embedding attempts before giving up; ``0`` (default) is RDKit's
///     ``10 × n_atoms``.
/// seed : int | None, optional
///     Random seed; with one the result is reproducible. ``None`` draws one.
/// use_random_coords : bool, optional
///     Start from random coordinates instead of the eigenvector embedding of a
///     random distance matrix (default ``False``).
/// use_exp_torsion_angle_prefs : bool, optional
///     The ETKDG experimental-torsion terms, "ET" (default ``True``).
/// use_basic_knowledge : bool, optional
///     Flat aromatic rings and planar sp2 centres, "K" (default ``True``).
///     With both torsion switches off the embedding is plain distance
///     geometry.
///
/// Examples
/// --------
/// >>> gen = Conformer(seed=42)
/// >>> mol_3d, report = gen.generate(mol)
#[pyclass(
    module = "molpy.conformer",
    name = "Conformer",
    subclass,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyConformer {
    pub(crate) inner: Conformer,
}

#[pymethods]
impl PyConformer {
    /// Create an embedder; the parameters are the class's.
    #[new]
    #[pyo3(signature = (
        add_hydrogens=true,
        max_iterations=0,
        seed=None,
        use_random_coords=false,
        use_exp_torsion_angle_prefs=true,
        use_basic_knowledge=true,
    ))]
    fn new(
        add_hydrogens: bool,
        max_iterations: usize,
        seed: Option<u64>,
        use_random_coords: bool,
        use_exp_torsion_angle_prefs: bool,
        use_basic_knowledge: bool,
    ) -> Self {
        Self {
            inner: Conformer::new(
                add_hydrogens,
                max_iterations,
                seed,
                use_random_coords,
                use_exp_torsion_angle_prefs,
                use_basic_knowledge,
            ),
        }
    }

    /// Generate 3D coordinates for a molecular graph.
    ///
    /// Runs the ETKDGv3 embedding. The input molecule is not modified; the result is an :class:`~molpy.core.Atomistic`
    /// that keeps the input's ports and its ``frag_id`` labels.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Input molecular graph (heavy atoms and bonds); its ``ports`` are
    ///     carried across unchanged.
    ///
    /// Returns
    /// -------
    /// tuple[Atomistic, ConformerReport]
    ///     The molecule with generated 3D coordinates — positions in ångström
    ///     (Å) — and a per-stage report.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If `mol` is not an ``Atomistic``.
    /// ValueError
    ///     If the molecular graph is invalid (e.g. an atom without a valid
    ///     ``atomic_number``).
    ///
    /// Examples
    /// --------
    /// >>> mol = molpy.io.smiles.read_smiles("CCO").to_atomistic()
    /// >>> mol_3d, report = Conformer(seed=42).generate(mol)
    /// >>> mol_3d.n_atoms   # includes added hydrogens
    /// 9
    ///
    /// Hydrogens this pipeline adds to a ported unit carry no ``frag_id``;
    /// the caller relabels them, which is one call:
    ///
    /// >>> unit = molpy.io.smiles.read_smiles_fragment("[$]CO").to_template()
    /// >>> unit_3d, _ = Conformer(seed=42).generate(unit)
    /// >>> _ = unit_3d.inherit_frag_ids()
    fn generate(
        &self,
        py: Python<'_>,
        mol: &Bound<'_, PyAny>,
    ) -> PyResult<(Py<PyAny>, PyConformerReport)> {
        // Leaf-first, exactly as the geometry systems dispatch: a leaf must
        // resolve to its own core value, never to the empty base it carries.
        if let Ok(leaf) = mol.cast::<PyAtomistic>() {
            let (out, report) = self
                .inner
                .generate(leaf.borrow().core())
                .map_err(molrs_error_to_pyerr)?;
            return Ok((
                leaf.borrow().derive(py, out)?.into_any(),
                report_to_py(report),
            ));
        }
        Err(PyTypeError::new_err(format!(
            "Conformer.generate expects an Atomistic, got {}",
            mol.get_type().name()?
        )))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

/// Register `molpy.conformer`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConformer>()?;
    m.add_class::<PyConformerReport>()?;
    m.add_class::<PyConformerStageReport>()?;
    Ok(())
}
