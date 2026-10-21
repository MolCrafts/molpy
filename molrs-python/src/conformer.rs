//! Python wrapper for 3D conformer generation from molecular graphs.
//!
//! The public surface is the [`PyConformer`] class (`molrs.conformer.Conformer`): its
//! constructor declares the generation parameters and [`PyConformer::generate`]
//! runs the pipeline, returning `(mol_3d, report)`.
//!
//! The pipeline takes a [`PyAtomistic`] molecular graph (ports included) and
//! produces realistic 3D coordinates through a multi-stage ETKDGv3 process:
//!
//! 1. **Preprocess** -- add hydrogens and perceive molecular features.
//! 2. **Build initial** -- ETKDGv3 distance-geometry embedding.
//! 3. **Coarse optimize** -- distance, chirality, and torsion refinement.
//! 4. **Final optimize** -- MMFF94 cleanup.
//! 5. **Stereo check** -- verify stereochemistry is preserved.
//!
//! # References
//!
//! - Riniker, S.; Landrum, G.A. (2015). J. Chem. Inf. Model. 55, 2562-2574.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use molrs::conformer::{Conformer, ConformerOptions, ConformerReport, ConformerSpeed, StageKind};

use crate::core::molgraph::PyAtomistic;
use crate::error::molrs_error_to_pyerr;

/// Map a `StageKind` enum to a human-readable name.
fn stage_kind_name(kind: StageKind) -> &'static str {
    match kind {
        StageKind::Preprocess => "preprocess",
        StageKind::BuildInitial => "build_initial",
        StageKind::CoarseOptimize => "coarse_optimize",
        StageKind::FinalOptimize => "final_optimize",
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
            elapsed_ms: s.elapsed_ms,
        })
        .collect();
    PyConformerReport {
        final_energy: report.final_energy,
        warnings: report.warnings,
        stages_inner: stages,
    }
}

/// Report for a single stage of conformer generation.
///
/// Exposed to Python as `molrs.conformer.ConformerStageReport`.
///
/// Attributes
/// ----------
/// stage : str
///     Stage name (e.g. ``"build_initial"``, ``"final_optimize"``).
/// energy_before : float | None
///     Energy in kcal/mol before this stage, or ``None`` if not applicable.
/// energy_after : float | None
///     Energy in kcal/mol after this stage, or ``None`` if not applicable.
/// steps : int
///     Number of optimizer steps executed.
/// converged : bool
///     Whether the stage converged.
/// elapsed_ms : int
///     Wall-clock time for this stage in milliseconds.
#[pyclass(module = "molrs.conformer", name = "ConformerStageReport")]
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
    #[pyo3(get)]
    pub elapsed_ms: u64,
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
/// Exposed to Python as `molrs.conformer.ConformerReport`.
///
/// Attributes
/// ----------
/// final_energy : float | None
///     Total energy in kcal/mol after all stages, or ``None`` on failure.
/// warnings : list[str]
///     Diagnostic warnings generated during the run.
/// stages : list[ConformerStageReport]
///     Per-stage execution reports.
///
/// Examples
/// --------
/// >>> mol_3d, report = Conformer().generate(mol)
/// >>> report.final_energy
/// -12.34
/// >>> for s in report.stages:
/// ...     print(s.stage, s.converged)
#[pyclass(module = "molrs.conformer", name = "ConformerReport")]
pub struct PyConformerReport {
    #[pyo3(get)]
    pub final_energy: Option<f64>,
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
                elapsed_ms: s.elapsed_ms,
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConformerReport(final_energy={:?}, stages={}, warnings={})",
            self.final_energy,
            self.stages_inner.len(),
            self.warnings.len()
        )
    }
}

/// 3D conformer generator for molecular graphs.
///
/// Exposed to Python as `molrs.conformer.Conformer`. Construct it with the desired
/// generation parameters, then call :meth:`generate` to produce coordinates.
/// The class is subclassable so downstream wrappers (e.g. ``molpy.conformer``)
/// can inherit it and refine the marshalling.
///
/// Parameters
/// ----------
/// speed : str, optional
///     Quality preset: ``"fast"`` (fewest iterations), ``"medium"`` (default),
///     or ``"better"`` (most thorough search).
/// add_hydrogens : bool, optional
///     If ``True`` (default), add implicit hydrogens before generation.
/// seed : int | None, optional
///     Random seed for reproducibility. ``None`` uses an arbitrary seed.
///
/// Examples
/// --------
/// >>> gen = Conformer(speed="fast", add_hydrogens=True, seed=42)
/// >>> mol_3d, report = gen.generate(mol)
#[pyclass(
    module = "molrs.conformer",
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
    /// Create a conformer generator.
    ///
    /// Parameters
    /// ----------
    /// speed : str
    ///     ``"fast"``, ``"medium"`` (default), or ``"better"``.
    /// add_hydrogens : bool
    ///     Add implicit hydrogens before generation. Default ``True``.
    /// seed : int | None
    ///     RNG seed for reproducibility.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``speed`` is not one of the recognized presets.
    #[new]
    #[pyo3(signature = (speed="medium", add_hydrogens=true, seed=None))]
    fn new(speed: &str, add_hydrogens: bool, seed: Option<u64>) -> PyResult<Self> {
        let sp = match speed {
            "fast" => ConformerSpeed::Fast,
            "medium" => ConformerSpeed::Medium,
            "better" => ConformerSpeed::Better,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown speed '{}', expected 'fast', 'medium', or 'better'",
                    other
                )));
            }
        };
        let opts = ConformerOptions {
            speed: sp,
            add_hydrogens,
            rng_seed: seed,
            ..ConformerOptions::default()
        };
        Ok(Self {
            inner: Conformer::new(opts),
        })
    }

    /// Generate 3D coordinates for a molecular graph.
    ///
    /// Runs the full distance-geometry + optimization pipeline. The input
    /// molecule is not modified; the result is an :class:`~molrs.core.Atomistic`
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
    /// >>> mol = molrs.io.smiles.read_smiles("CCO").to_atomistic()
    /// >>> mol_3d, report = Conformer(speed="fast", seed=42).generate(mol)
    /// >>> mol_3d.n_atoms   # includes added hydrogens
    /// 9
    ///
    /// Hydrogens this pipeline adds to a ported unit carry no ``frag_id``;
    /// the caller relabels them, which is one call:
    ///
    /// >>> unit = molrs.io.smiles.read_smiles_fragment("[$]CO").to_template()
    /// >>> unit_3d, _ = Conformer(speed="fast", seed=42).generate(unit)
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
        let opts = self.inner.options();
        format!(
            "Conformer(speed={:?}, add_hydrogens={}, seed={:?})",
            opts.speed, opts.add_hydrogens, opts.rng_seed
        )
    }
}

/// Register `molrs.conformer`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyConformer>()?;
    m.add_class::<PyConformerReport>()?;
    m.add_class::<PyConformerStageReport>()?;
    Ok(())
}
