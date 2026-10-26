//! Limited-memory BFGS (L-BFGS) core, force-field agnostic.
//!
//! This is the shared minimization engine consumed both by the public
//! geometry optimizer ([`Lbfgs`](super::Lbfgs)) and by
//! the ETKDG conformer pipeline (via the crate-internal [`minimize_lbfgs_rms`]). It operates on a
//! flat `3·n_atoms` coordinate buffer and any `(energy, forces = -grad)`
//! evaluator, exactly the contract that `crate::ff::potential::Potential`
//! exposes — the force field itself is untouched.
//!
//! The two-loop recursion + backtracking line search is an acknowledged port
//! of RDKit's `BFGSOpt::minimize` (BSD-3, Copyright (C) Greg Landrum / RDKit),
//! satisfying the Armijo sufficient-decrease and weak-Wolfe curvature
//! conditions.
//!
//! Reference:
//!   Nocedal & Wright, *Numerical Optimization* (2nd ed.), Algorithm 7.4
//!   (L-BFGS two-loop recursion) + Algorithm 3.1 (backtracking line search).

use super::OptimizationReport;

/// Default number of correction pairs retained by the L-BFGS history.
///
/// Nocedal & Wright (§7.2) report that `m` between 3 and 20 suffices in
/// practice; 8 sits in that range and is what the ETKDG stages are pinned at,
/// so the conformer reference numbers depend on it.
pub(crate) const HISTORY: usize = 8;
/// Armijo sufficient-decrease parameter (`c1`).
const ARMIJO_C1: f64 = 1e-4;
/// Weak-Wolfe curvature parameter (`c2`).
const WOLFE_C2: f64 = 0.9;
/// Line-search step contraction factor on Armijo failure.
const BACKTRACK: f64 = 0.5;
/// Line-search step expansion factor while curvature is still unmet.
const EXPAND: f64 = 2.1;
/// Maximum line-search trial evaluations per outer iteration.
const MAX_LS_TRIALS: usize = 40;

/// Convergence criterion for the core minimizer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Converge {
    /// Stop when the RMS gradient `‖g‖ / √n` drops below the tolerance
    /// (kcal/mol/Å). Used by the ETKDG cleanup to mirror RDKit's BFGS exit.
    GradRms(f64),
    /// Stop when the maximum per-atom force magnitude `max_i ‖F_i‖` drops below
    /// the tolerance (kcal/mol/Å). The ASE / molpy convention used by the
    /// public geometry optimizer.
    Fmax(f64),
}

/// Maximum per-atom force magnitude from a flat `3N` gradient (= -forces).
///
/// `‖F_i‖ = ‖g_i‖`, so this is computed directly from the gradient. A buffer
/// whose length is not a multiple of three ignores the trailing remainder
/// (never the case for atomic coordinates).
#[inline]
pub(crate) fn fmax_from_grad(grad: &[f64]) -> f64 {
    grad.as_chunks::<3>()
        .0
        .iter()
        .map(|g| (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt())
        .fold(0.0_f64, f64::max)
}

/// L-BFGS core. Minimizes `eval` (returning `(energy, forces = -grad)`) in
/// place over `coords`.
///
/// Returns `(energy, final_grad, steps, converged)` where `final_grad` is the
/// gradient (= -forces) at the returned point — callers derive `fmax` or
/// `grad_rms` from it as needed.
///
/// * `max_iters` — outer-iteration cap.
/// * `converge`  — convergence criterion (`GradRms` or `Fmax`).
/// * `max_step`  — per-step trust region in Å. `f64::INFINITY` disables it
///   (the search direction is taken as-is and the line search may expand);
///   any finite value clamps the search direction so no accepted step displaces
///   a coordinate component by more than `max_step`.
/// * `memory`    — L-BFGS correction-pair history size.
pub(crate) fn minimize_core<F>(
    coords: &mut [f64],
    max_iters: usize,
    converge: Converge,
    max_step: f64,
    memory: usize,
    mut eval: F,
) -> (f64, Vec<f64>, usize, bool)
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = coords.len();
    if n == 0 {
        return (0.0, Vec::new(), 0, true);
    }
    let memory = memory.max(1);
    let trust = max_step.is_finite();

    let inv_sqrt_n = 1.0 / (n as f64).sqrt();
    let (mut energy, forces) = eval(coords);
    // grad = -forces.
    let mut grad: Vec<f64> = forces.iter().map(|f| -f).collect();

    // L-BFGS correction history: s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k.
    let mut s_hist: Vec<Vec<f64>> = Vec::with_capacity(memory);
    let mut y_hist: Vec<Vec<f64>> = Vec::with_capacity(memory);
    let mut rho_hist: Vec<f64> = Vec::with_capacity(memory);

    let mut converged = false;
    let mut iters = 0;

    for it in 0..max_iters {
        iters = it + 1;

        let gnorm2 = dot(&grad, &grad);
        let metric_converged = match converge {
            Converge::GradRms(tol) => gnorm2.sqrt() * inv_sqrt_n < tol,
            Converge::Fmax(tol) => fmax_from_grad(&grad) < tol,
        };
        if metric_converged {
            converged = true;
            break;
        }

        // --- L-BFGS two-loop recursion: direction = -H·grad ---------------
        let mut q = grad.clone();
        let m = s_hist.len();
        let mut alpha = vec![0.0; m];
        // Newest correction first.
        for i in (0..m).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        // Initial Hessian scaling H0 = (s·y)/(y·y)·I (Nocedal & Wright 7.20).
        let gamma = if m > 0 {
            let last = m - 1;
            let sy = dot(&s_hist[last], &y_hist[last]);
            let yy = dot(&y_hist[last], &y_hist[last]);
            if yy > 1e-30 { sy / yy } else { 1.0 }
        } else {
            // First step: scale by 1/‖g‖ so the initial displacement is ~1 Å.
            let gn = gnorm2.sqrt();
            if gn > 1e-12 { (1.0 / gn).min(0.1) } else { 1.0 }
        };
        scale(&mut q, gamma);
        for i in 0..m {
            let beta = rho_hist[i] * dot(&y_hist[i], &q);
            axpy(&mut q, alpha[i] - beta, &s_hist[i]);
        }
        // Search direction p = -H·grad.
        let mut dir = q;
        scale(&mut dir, -1.0);

        // Guard against a non-descent direction (numerical breakdown): fall
        // back to steepest descent.
        if dot(&grad, &dir) >= 0.0 {
            dir = grad.clone();
            scale(&mut dir, -1.0);
        }

        // Trust region: clamp the search direction so the unit (step = 1.0)
        // trial displaces no component by more than `max_step`. Combined with
        // the no-expand rule below, every accepted step is bounded by it.
        if trust {
            let dmax = dir.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            if dmax > max_step {
                scale(&mut dir, max_step / dmax);
            }
        }
        let g_dot_dir = dot(&grad, &dir);

        // --- Backtracking + curvature line search -------------------------
        let x0 = coords.to_vec();
        let e0 = energy;
        let mut step = 1.0;
        let mut lo = 0.0;
        let mut hi = f64::INFINITY;
        let mut accepted = false;
        let mut new_grad = grad.clone();

        for _ in 0..MAX_LS_TRIALS {
            let mut trial = x0.clone();
            axpy(&mut trial, step, &dir);
            let (e_trial, f_trial) = eval(&trial);

            if !e_trial.is_finite() {
                hi = step;
                step = 0.5 * (lo + hi);
                continue;
            }

            // Armijo sufficient decrease.
            if e_trial > e0 + ARMIJO_C1 * step * g_dot_dir {
                hi = step;
                step = 0.5 * (lo + hi);
                continue;
            }

            // Weak-Wolfe curvature on the new gradient along dir.
            let g_new: Vec<f64> = f_trial.iter().map(|f| -f).collect();
            let g_new_dot_dir = dot(&g_new, &dir);
            if g_new_dot_dir < WOLFE_C2 * g_dot_dir {
                // Curvature unmet: step too short.
                if trust {
                    // Trust region active — do not expand past the clamped
                    // step; let the Armijo-only fallback take a bounded point.
                    break;
                }
                lo = step;
                if hi.is_finite() {
                    step = 0.5 * (lo + hi);
                } else {
                    step *= EXPAND;
                }
                continue;
            }

            // Both conditions satisfied — accept.
            coords.copy_from_slice(&trial);
            energy = e_trial;
            new_grad = g_new;
            accepted = true;
            break;
        }

        if !accepted {
            // Armijo-only fallback: take the best sufficient-decrease point we
            // can find by pure backtracking, so a too-strict curvature test
            // never blocks progress. `step` never exceeded 1.0 under a trust
            // region, so the accepted displacement stays bounded by max_step.
            let mut s = step.max(BACKTRACK);
            let mut made = false;
            for _ in 0..MAX_LS_TRIALS {
                let mut trial = x0.clone();
                axpy(&mut trial, s, &dir);
                let (e_trial, f_trial) = eval(&trial);
                if e_trial.is_finite() && e_trial < e0 + ARMIJO_C1 * s * g_dot_dir {
                    coords.copy_from_slice(&trial);
                    energy = e_trial;
                    new_grad = f_trial.iter().map(|f| -f).collect();
                    made = true;
                    break;
                }
                s *= BACKTRACK;
            }
            if !made {
                // No step lowers the energy, yet the gradient test at the top
                // of this iteration failed: the line search is stuck (a
                // gradient inconsistent with its energy, or a singular term),
                // not at a minimum. Stop without claiming convergence; a
                // stationary point is caught by that gradient test instead.
                break;
            }
        }

        // --- Update L-BFGS history ---------------------------------------
        let mut s_k = coords.to_vec();
        axpy(&mut s_k, -1.0, &x0);
        let mut y_k = new_grad.clone();
        axpy(&mut y_k, -1.0, &grad);
        let sy = dot(&s_k, &y_k);
        if sy > 1e-12 {
            if s_hist.len() == memory {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            rho_hist.push(1.0 / sy);
            s_hist.push(s_k);
            y_hist.push(y_k);
        }
        grad = new_grad;
    }

    (energy, grad, iters, converged)
}

/// RMS-gradient-tolerance L-BFGS entry point for the ETKDG stages: both
/// distance-geometry minimizations and the MMFF cleanup.
///
/// Converges on the RMS gradient, with no trust region and history size
/// `HISTORY`. `coords` is updated in place.
#[cfg_attr(not(feature = "conformer"), allow(dead_code))]
pub(crate) fn minimize_lbfgs_rms<F>(
    coords: &mut [f64],
    max_iters: usize,
    grad_rms_tol: f64,
    eval: F,
) -> OptimizationReport
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let (energy, grad, iters, converged) = minimize_core(
        coords,
        max_iters,
        Converge::GradRms(grad_rms_tol),
        f64::INFINITY,
        HISTORY,
        eval,
    );
    OptimizationReport::from_gradient(converged, iters, energy, &grad)
}

/// RMS of a flat gradient, `|g| / sqrt(len)`; zero for an empty one.
pub(crate) fn grad_rms(grad: &[f64]) -> f64 {
    if grad.is_empty() {
        0.0
    } else {
        dot(grad, grad).sqrt() / (grad.len() as f64).sqrt()
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `a += alpha · b`.
#[inline]
fn axpy(a: &mut [f64], alpha: f64, b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += alpha * y;
    }
}

#[inline]
fn scale(a: &mut [f64], factor: f64) {
    for x in a.iter_mut() {
        *x *= factor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple anisotropic quadratic bowl: minimum at the origin, stiff in
    /// one direction (condition number 100) — a stress test that plain
    /// steepest descent handles poorly but L-BFGS solves in a few steps.
    #[test]
    fn lbfgs_solves_stiff_quadratic() {
        let k = [1.0, 10.0, 100.0, 1.0, 10.0, 100.0];
        let eval = |x: &[f64]| -> (f64, Vec<f64>) {
            let mut e = 0.0;
            let mut forces = vec![0.0; x.len()];
            for i in 0..x.len() {
                e += 0.5 * k[i] * x[i] * x[i];
                forces[i] = -k[i] * x[i]; // force = -grad
            }
            (e, forces)
        };
        let mut x = vec![1.0, 1.0, 1.0, -1.0, -1.0, -1.0];
        let report = minimize_lbfgs_rms(&mut x, 200, 1e-6, eval);
        let (e, grms, steps, conv) = (
            report.final_energy,
            report.final_grad_rms,
            report.n_steps,
            report.converged,
        );
        assert!(conv, "should converge");
        assert!(e < 1e-8, "energy should reach ~0, got {e}");
        assert!(grms < 1e-6, "grad RMS should be tiny, got {grms}");
        assert!(steps < 100, "should converge quickly, took {steps}");
        for xi in &x {
            assert!(xi.abs() < 1e-4, "coord should reach 0, got {xi}");
        }
    }

    /// A line search that finds no lower energy is not convergence: the
    /// force here is 5 everywhere, but the energy (a flat 0) never drops, so
    /// no step passes Armijo. The run must stop and say it did not converge.
    #[test]
    fn a_failed_line_search_is_not_convergence() {
        let eval = |x: &[f64]| -> (f64, Vec<f64>) { (0.0, vec![5.0; x.len()]) };
        let mut x = vec![0.0; 6];
        let report = minimize_lbfgs_rms(&mut x, 200, 1e-6, eval);
        assert!(!report.converged, "{report:?}");
        assert!(report.final_grad_rms > 1.0, "{report:?}");
    }
}
