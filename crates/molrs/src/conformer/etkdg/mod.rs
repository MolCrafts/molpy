//! ETKDGv3 conformer-embedding pipeline.
//!
//! This is a port of RDKit's `EmbedMolecule` orchestration
//! (`$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`, BSD-3, Copyright (C)
//! 2004-2025 Greg Landrum / Sereina Riniker and other RDKit contributors)
//! wired onto MolCrafts' own constraint generator (`crate::conformer::distgeom`,
//! ETKDGv3 bounds + experimental torsions + chiral sets) and the MMFF94
//! force field (`molrs::ff::typifier::mmff`) for the second-stage cleanup.
//!
//! ## Stages (mapped onto the public `StageKind` variants)
//! 1. `Preprocess`    — optional hydrogen addition.
//! 2. `BuildInitial`  — `build_constraints` → metrization sample → 4D
//!    eigenvalue embedding (`embed4d`).
//! 3. `CoarseOptimize`— first-stage 4D distance/chiral/fourth-dim minimization
//!    then 3D experimental-torsion refinement (`torsion_refinement`).
//! 4. `FinalOptimize` — second-stage MMFF94 energy minimization (`molrs::ff`).
//! 5. `StereoCheck`   — chiral-volume sign verification (no inversion).
//!
//! The maxIterations retry loop + `useRandomCoords` fallback live in `retry`.

mod embed4d;
mod retry;
mod torsion_refinement;

use rand::{SeedableRng, random, rngs::StdRng};

use crate::conformer::ConformerOptions;
use crate::conformer::distgeom::{self, ChiralSign, DgConstraints};
use crate::conformer::{ConformerReport, ConformerStageReport, StageKind};
use molrs::core::Atomistic;
use molrs::core::MolRsError;
use molrs::ff::compile::PotentialCompiler;
use molrs::ff::potential::{Accumulator, Potential, intramolecular_pairs};
use molrs::ff::typifier::mmff::Mmff94Typifier;
use molrs::ff::typifier::{Typing, TypingError};
use molrs::perceive::add_hydrogens;

/// Embedding dimension for the first stage (RDKit ETKDG uses 4D).
const EMBED_DIM: usize = 4;
/// Fraction below which a chiral volume is treated as inverted (RDKit
/// `checkChiralCenters` 0.8 threshold).
const CHIRAL_RATIO_TOL: f64 = 0.8;

/// Per-atom energy threshold above which the first minimization is rejected
/// (RDKit `MAX_MINIMIZED_E_PER_ATOM`).
const MAX_MEAN_ATOM_ENERGY: f64 = 0.05;

/// Run the ETKDGv3 embedding pipeline and return the molecule with 3D
/// coordinates plus a stage report.
///
/// `mol` is treated as a connectivity graph; any pre-existing 3D coordinates
/// are used only to seed chiral-volume signs (so a stereochemically-defined
/// input keeps its handedness) and are otherwise overwritten.
pub(crate) fn generate_3d_impl(
    mol: &Atomistic,
    opts: &ConformerOptions,
) -> Result<(Atomistic, ConformerReport), MolRsError> {
    if mol.n_atoms() == 0 {
        return Err(MolRsError::validation(
            "cannot generate 3D structure for empty molecule",
        ));
    }

    let mut report = ConformerReport::new();

    let seed = opts.rng_seed.unwrap_or_else(random::<u64>);
    if opts.rng_seed.is_none() {
        report.warnings.push(format!(
            "rng_seed not provided; auto-generated seed={seed} for this run"
        ));
    }

    let work = run_preprocess(mol, opts, &mut report)?;
    let n = work.n_atoms();

    // Single atom: place at origin (no geometry to solve). RDKit also returns
    // a valid (trivial) conformer.
    if n == 1 {
        return embed_single_atom(work, report);
    }

    // --- Build ETKDGv3 constraints ---------------------------------------
    // Experimental torsions are assigned through the full CrystalFF
    // three-table set (v2 ++ small-rings ++ macrocycles) matched by the core
    // SMARTS engine (`molrs::perceive::smarts`), reproducing RDKit
    // `getExperimentalTorsions`. See `distgeom::torsion_prefs`.
    let constraints = distgeom::DgConstraints::from_graph(&work)?;

    let mut embedding = run_embedding(&constraints, n, seed, opts, &mut report);
    let mut coords3d = match embedding.best.take() {
        Some(c) => c,
        None => {
            return Err(MolRsError::validation(
                "ETKDG embedding failed: no consistent conformer after retries",
            ));
        }
    };
    push_embedding_stages(&mut report, &embedding);

    let final_energy = run_mmff_cleanup_stage(&work, &mut coords3d, &mut report)?;
    run_stereo_check(&constraints, &coords3d, &mut report);

    // --- Write coordinates back ------------------------------------------
    let mut out = work;
    apply_coords(&mut out, &coords3d)?;
    // The MMFF94 energy, when the cleanup ran; the distance-geometry
    // objective is no force-field energy, so it never stands in for one.
    report.final_energy = final_energy;
    Ok((out, report))
}

/// `Preprocess` stage: optional hydrogen repletion, reported as the atom
/// count the molecule gained.
fn run_preprocess(
    mol: &Atomistic,
    opts: &ConformerOptions,
    report: &mut ConformerReport,
) -> Result<Atomistic, MolRsError> {
    let work = if opts.add_hydrogens {
        add_hydrogens(mol)?
    } else {
        mol.clone()
    };
    let preprocess_steps = work.n_atoms().saturating_sub(mol.n_atoms());
    report.stages.push(ConformerStageReport {
        stage: StageKind::Preprocess,
        energy_before: None,
        energy_after: None,
        steps: preprocess_steps,
        converged: true,
    });
    Ok(work)
}

/// `BuildInitial` stage for the degenerate one-atom case: place the atom at
/// the origin and close the report at zero energy.
fn embed_single_atom(
    work: Atomistic,
    mut report: ConformerReport,
) -> Result<(Atomistic, ConformerReport), MolRsError> {
    let mut out = work;
    place_single_atom(&mut out)?;
    report.stages.push(ConformerStageReport {
        stage: StageKind::BuildInitial,
        energy_before: None,
        energy_after: None,
        steps: 1,
        converged: true,
    });
    report.final_energy = Some(0.0);
    Ok((out, report))
}

/// Best candidate of the embedding stage plus the telemetry the
/// `BuildInitial` / `CoarseOptimize` stage reports are built from.
struct EmbedOutcome {
    /// Best 3D coordinate buffer found, `None` if every attempt was degenerate.
    best: Option<Vec<f64>>,
    /// 4D embedding steps of the last attempt.
    embed4d_steps: usize,
    /// First-stage energy of the last attempt.
    coarse_energy: f64,
    /// First-stage minimizer steps of the last attempt.
    coarse_steps: usize,
    /// Whether the last attempt's first-stage minimization converged.
    coarse_converged: bool,
    /// Whether `best` passed the chiral-volume check.
    chiral_ok: bool,
}

/// Embedding stage: the `maxIterations` retry loop followed by RDKit's
/// `useRandomCoords` fallback attempt.
fn run_embedding(
    constraints: &DgConstraints,
    n: usize,
    seed: u64,
    opts: &ConformerOptions,
    report: &mut ConformerReport,
) -> EmbedOutcome {
    let max_iters = retry::effective_max_iterations(opts.max_steps, n);
    let mut outcome = EmbedOutcome {
        best: None,
        embed4d_steps: 0,
        coarse_energy: f64::NAN,
        coarse_steps: 0,
        coarse_converged: false,
        chiral_ok: false,
    };

    for attempt in 0..max_iters {
        let aseed = retry::attempt_seed(seed, attempt);
        let mut rng = StdRng::seed_from_u64(aseed);
        let (coords3d, e4d_steps, coarse_e, coarse_steps, coarse_conv, chiral_pass) =
            try_embed(constraints, n, &mut rng, false);
        outcome.embed4d_steps = e4d_steps;
        outcome.coarse_energy = coarse_e;
        outcome.coarse_steps = coarse_steps;
        outcome.coarse_converged = coarse_conv;
        if let Some(c) = coords3d {
            if chiral_pass {
                outcome.best = Some(c);
                outcome.chiral_ok = true;
                break;
            }
            // Keep a non-chiral-clean candidate as a fallback.
            if outcome.best.is_none() {
                outcome.best = Some(c);
            }
        }
    }

    // useRandomCoords fallback.
    if outcome.best.is_none() || !outcome.chiral_ok {
        let aseed = retry::attempt_seed(seed, max_iters + 1);
        let mut rng = StdRng::seed_from_u64(aseed);
        let (coords3d, e4d_steps, coarse_e, coarse_steps, coarse_conv, chiral_pass) =
            try_embed(constraints, n, &mut rng, true);
        outcome.embed4d_steps = e4d_steps;
        outcome.coarse_energy = coarse_e;
        outcome.coarse_steps = coarse_steps;
        outcome.coarse_converged = coarse_conv;
        if let Some(c) = coords3d {
            if chiral_pass || outcome.best.is_none() {
                outcome.best = Some(c);
            }
            report
                .warnings
                .push("used random-coordinate fallback embedding".to_string());
        }
    }

    outcome
}

/// Report the `BuildInitial` and `CoarseOptimize` stages of the accepted
/// embedding attempt.
fn push_embedding_stages(report: &mut ConformerReport, outcome: &EmbedOutcome) {
    report.stages.push(ConformerStageReport {
        stage: StageKind::BuildInitial,
        energy_before: None,
        energy_after: None,
        steps: outcome.embed4d_steps,
        converged: true,
    });
    report.stages.push(ConformerStageReport {
        stage: StageKind::CoarseOptimize,
        energy_before: None,
        energy_after: Some(outcome.coarse_energy),
        steps: outcome.coarse_steps,
        converged: outcome.coarse_converged,
    });
}

/// `FinalOptimize` stage: second-stage MMFF94 cleanup minimization, skipped
/// (with a warning) when MMFF94 has no types or terms for the molecule.
///
/// # Errors
///
/// Every other failure of the cleanup (a malformed graph, a force field that
/// does not compile): a defect, never a skipped stage.
fn run_mmff_cleanup_stage(
    work: &Atomistic,
    coords3d: &mut [f64],
    report: &mut ConformerReport,
) -> Result<Option<f64>, MolRsError> {
    let (energy, steps, converged) = match mmff_cleanup(work, coords3d)? {
        Cleanup::Minimized {
            energy,
            steps,
            converged,
        } => (Some(energy), steps, converged),
        Cleanup::NotCovered(reason) => {
            report
                .warnings
                .push(format!("MMFF94 cleanup skipped: {reason}"));
            (None, 0, false)
        }
    };
    report.stages.push(ConformerStageReport {
        stage: StageKind::FinalOptimize,
        energy_before: None,
        energy_after: energy,
        steps,
        converged,
    });
    Ok(energy)
}

/// `StereoCheck` stage: warn for every chiral center whose signed volume
/// inverted or shrank past [`CHIRAL_RATIO_TOL`] relative to its target.
fn run_stereo_check(constraints: &DgConstraints, coords3d: &[f64], report: &mut ConformerReport) {
    let mut stereo_warnings = Vec::new();
    for c in &constraints.chiral {
        if c.sign == ChiralSign::Unknown {
            continue;
        }
        let vol = distgeom::chiral_volume(coords3d, c.neighbors, 3);
        let target_positive = matches!(c.sign, ChiralSign::Positive);
        let got_positive = vol > 0.0;
        if target_positive != got_positive
            || (target_positive && c.volume_lower > 0.0 && vol < c.volume_lower * CHIRAL_RATIO_TOL)
            || (!target_positive && c.volume_upper < 0.0 && vol > c.volume_upper * CHIRAL_RATIO_TOL)
        {
            stereo_warnings.push(format!(
                "tetrahedral-inversion at atom {}: expected {:?} signed volume, got {:.3}",
                c.center, c.sign, vol
            ));
        }
    }
    let stereo_steps = stereo_warnings.len();
    report.warnings.extend(stereo_warnings);
    report.stages.push(ConformerStageReport {
        stage: StageKind::StereoCheck,
        energy_before: None,
        energy_after: None,
        steps: stereo_steps,
        converged: true,
    });
}

/// One embedding attempt: 4D embed → first-stage minimization → 3D
/// experimental-torsion refinement → chiral check.
///
/// Returns `(coords3d, embed4d_steps, coarse_energy, coarse_steps,
/// coarse_converged, chiral_pass)`. `coords3d` is `None` if the 4D embedding
/// was degenerate.
#[allow(clippy::type_complexity)]
fn try_embed<R: rand::Rng + ?Sized>(
    constraints: &DgConstraints,
    n: usize,
    rng: &mut R,
    use_random_coords: bool,
) -> (Option<Vec<f64>>, usize, f64, usize, bool, bool) {
    let bounds = &constraints.bounds;

    let mut coords4d = if use_random_coords {
        embed4d::compute_random_coords(n, EMBED_DIM, 10.0, rng)
    } else {
        let dist = embed4d::pick_random_dist_mat(bounds, rng);
        match embed4d::compute_initial_coords(&dist, n, EMBED_DIM, rng, true, 1) {
            Some(c) => c,
            None => return (None, 0, f64::NAN, 0, false, false),
        }
    };

    // First minimization: distance + chiral + 4th-dimension (RDKit
    // firstMinimization, weightChiral=1.0, weightFourthDim=0.1).
    let field1 = distgeom::ViolationEnergy::new(bounds, &constraints.chiral, EMBED_DIM, 1.0, 0.1);
    let stage1 = minimize(&mut coords4d, 400, |p, g| field1.energy_grad(p, g));
    let (e1, s1) = (stage1.final_energy, stage1.n_steps);
    // Reject obviously-bad first minimizations (RDKit github #971,
    // `MAX_MINIMIZED_E_PER_ATOM`). Random-coords fallback skips this gate.
    if !use_random_coords && e1 / (n as f64) >= MAX_MEAN_ATOM_ENERGY {
        return (None, s1, e1, 0, false, false);
    }

    // Fourth-dimension squeeze (RDKit minimizeFourthDimension, weightChiral=0.2,
    // weightFourthDim=1.0) to collapse 4D → 3D.
    let field1b = distgeom::ViolationEnergy::new(bounds, &constraints.chiral, EMBED_DIM, 0.2, 1.0);
    let _ = minimize(&mut coords4d, 200, |p, g| field1b.energy_grad(p, g));

    // Project to 3D (drop the 4th component).
    let mut coords3d = vec![0.0; n * 3];
    for i in 0..n {
        for k in 0..3 {
            coords3d[i * 3 + k] = coords4d[i * EMBED_DIM + k];
        }
    }

    // Second stage: 3D experimental-torsion refinement (RDKit
    // minimizeWithExpTorsions / construct3DForceField).
    let field2 = torsion_refinement::TorsionRefinement::build(
        bounds,
        constraints
            .experimental_torsions
            .iter()
            .chain(&constraints.flat_ring_torsions),
        &constraints.improper,
    );
    let stage2 = minimize(&mut coords3d, 300, |p, g| field2.energy_grad(p, g));
    let (e2, s2, c2) = (stage2.final_energy, stage2.n_steps, stage2.converged);

    // Chiral check.
    let mut chiral_pass = true;
    for c in &constraints.chiral {
        let vol = distgeom::chiral_volume(&coords3d, c.neighbors, 3);
        let lb = c.volume_lower;
        let ub = c.volume_upper;
        if (lb > 0.0 && vol < lb && (vol / lb < CHIRAL_RATIO_TOL || have_opposite_sign(vol, lb)))
            || (ub < 0.0
                && vol > ub
                && (vol / ub < CHIRAL_RATIO_TOL || have_opposite_sign(vol, ub)))
        {
            chiral_pass = false;
            break;
        }
    }

    (Some(coords3d), s1, e2, s2, c2, chiral_pass)
}

/// Minimize one distance-geometry objective with the crate's L-BFGS
/// ([`crate::optimize::minimize_lbfgs_rms`]) to RDKit's embedding force
/// tolerance (1e-3 RMS gradient). `objective` returns the energy and fills
/// the gradient; L-BFGS takes forces, so the gradient is negated here.
fn minimize(
    coords: &mut [f64],
    max_iters: usize,
    objective: impl Fn(&[f64], &mut [f64]) -> f64,
) -> crate::optimize::OptimizationReport {
    crate::optimize::minimize_lbfgs_rms(coords, max_iters, 1e-3, |p| {
        let mut grad = vec![0.0; p.len()];
        let energy = objective(p, &mut grad);
        grad.iter_mut().for_each(|g| *g = -*g);
        (energy, grad)
    })
}

/// RDKit `haveOppositeSign`.
fn have_opposite_sign(a: f64, b: f64) -> bool {
    a.is_sign_negative() ^ b.is_sign_negative()
}

/// What the MMFF94 cleanup did.
enum Cleanup {
    /// Minimized to `energy` (kcal/mol) in `steps` steps.
    Minimized {
        energy: f64,
        steps: usize,
        converged: bool,
    },
    /// MMFF94 has no atom types or terms for the molecule.
    NotCovered(TypingError),
}

/// MMFF94 second-stage cleanup minimization.
///
/// Runs the standard route — typify → `Frame` → `PotentialCompiler::compile` — the
/// same one every other force field in molrs goes through, so the seven MMFF
/// terms come from `ff::potential::*::mmff` and nowhere else.
///
/// # Errors
///
/// A typing failure other than MMFF94 not covering the molecule, or a frame /
/// compile failure: these are defects to report, not a stage to skip.
fn mmff_cleanup(mol: &Atomistic, coords3d: &mut [f64]) -> Result<Cleanup, MolRsError> {
    // Write current coords so MMFF setup that consults geometry sees them.
    let mut staged = mol.clone();
    apply_coords(&mut staged, coords3d)?;

    // A fresh typing per call: its output holds exactly this molecule's types.
    // `Mmff94Typifier::new` shares the process-wide memoised MMFF94 library, so
    // this costs no parameter assembly.
    let mut typing = Typing::new(Mmff94Typifier::new());
    let typed = match typing.typify(&staged) {
        Ok(typed) => typed,
        Err(e @ (TypingError::MissingAtomType { .. } | TypingError::MissingTerms { .. })) => {
            return Ok(Cleanup::NotCovered(e));
        }
        Err(e) => return Err(MolRsError::validation(format!("MMFF94 cleanup: {e}"))),
    };
    let mut frame = typed.to_frame()?;
    // The neighbour list is the consumer's to build — and here the consumer is the
    // minimizer. Bonded terms and the 1-2/1-3 exclusions are topological, so this
    // list stays valid across the relaxation. Which close neighbours belong in
    // it is MMFF's call, so the list is built from MMFF's own weights rather
    // than from an assumption about them.
    let ff = typing.forcefield();
    let pairs = ff
        .required_special_bonds()
        .and_then(|sb| intramolecular_pairs(&frame, sb, None))
        .map_err(|e| MolRsError::validation(format!("MMFF94 cleanup: {e}")))?;
    frame.insert("pairs", pairs);
    let potentials = PotentialCompiler::new(ff)
        .compile(&frame)
        .map_err(|e| MolRsError::validation(format!("MMFF94 cleanup: {e}")))?;

    // RDKit's MMFFOptimizeMolecule runs a full BFGS minimization to a
    // gradient-norm tolerance. Mirror that with L-BFGS to an RMS-gradient
    // convergence of 1e-3 kcal/mol/Å (matching RDKit's default
    // `MMFFOptimizeMolecule` grad tol) under a generous iteration cap, so the
    // freshly-embedded geometry is relaxed all the way to the MMFF minimum.
    let report = crate::optimize::minimize_lbfgs_rms(coords3d, 1000, 1e-3, |p| {
        let mut acc = Accumulator::new(p.len());
        potentials.evaluate(p, &mut acc);
        (acc.energy(), acc.into_forces())
    });
    Ok(Cleanup::Minimized {
        energy: report.final_energy,
        steps: report.n_steps,
        converged: report.converged,
    })
}

/// Place a single-atom molecule at the origin.
fn place_single_atom(mol: &mut Atomistic) -> Result<(), MolRsError> {
    let id = mol.atoms().map(|(id, _)| id).next();
    if let Some(id) = id {
        mol.set_atom(id, "x", 0.0)?;
        mol.set_atom(id, "y", 0.0)?;
        mol.set_atom(id, "z", 0.0)?;
    }
    Ok(())
}

/// Write a flat `n*3` coordinate buffer back into the molecule (atom-iteration
/// order matches `distgeom`/`MMFF` indexing).
fn apply_coords(mol: &mut Atomistic, coords: &[f64]) -> Result<(), MolRsError> {
    let ids: Vec<_> = mol.atoms().map(|(id, _)| id).collect();
    for (i, id) in ids.into_iter().enumerate() {
        mol.set_atom(id, "x", coords[i * 3])?;
        mol.set_atom(id, "y", coords[i * 3 + 1])?;
        mol.set_atom(id, "z", coords[i * 3 + 2])?;
    }
    Ok(())
}
