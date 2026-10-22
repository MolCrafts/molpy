//! Wigner D-matrix elements for integer ℓ.

use ndarray::Array2;

use super::factorial::ln_factorial;
use crate::core::Complex;
use crate::op::F;

/// Small Wigner d-matrix element `d^ℓ_{m',m}(β)` (real), evaluated via the
/// Wigner sum form (Edmonds (4.1.15)):
///
/// ```text
///   d^ℓ_{m',m}(β) =
///       Σ_s (-1)^{m'-m+s}
///           · √[(ℓ+m)!(ℓ-m)!(ℓ+m')!(ℓ-m')!]
///           / [(ℓ+m-s)! s! (m'-m+s)! (ℓ-m'-s)!]
///           · cos(β/2)^{2ℓ+m-m'-2s}
///           · sin(β/2)^{m'-m+2s}
/// ```
///
/// Factorials are computed via `lgamma` to stay stable for ℓ ≲ 30. The
/// sum range `s ∈ [max(0, m-m'), min(ℓ+m, ℓ-m')]` keeps every factorial
/// argument non-negative.
pub fn wigner_small_d(l: u32, m_prime: i32, m: i32, beta: F) -> F {
    let li = l as i32;
    if m.unsigned_abs() > l || m_prime.unsigned_abs() > l {
        return 0.0;
    }

    let cos_half = (beta * 0.5).cos();
    let sin_half = (beta * 0.5).sin();

    let s_min = 0i64.max((m - m_prime) as i64);
    let s_max = ((li + m) as i64).min((li - m_prime) as i64);
    if s_min > s_max {
        return 0.0;
    }

    // ½ log[(ℓ+m)!(ℓ-m)!(ℓ+m')!(ℓ-m')!]
    let log_prefactor = 0.5
        * (ln_factorial((li + m) as i64)
            + ln_factorial((li - m) as i64)
            + ln_factorial((li + m_prime) as i64)
            + ln_factorial((li - m_prime) as i64));

    let mut sum: F = 0.0;
    for s in s_min..=s_max {
        let si = s as i32;
        let p_cos = 2 * li + m - m_prime - 2 * si;
        let p_sin = m_prime - m + 2 * si;

        let log_denom = ln_factorial((li + m - si) as i64)
            + ln_factorial(s)
            + ln_factorial((m_prime - m + si) as i64)
            + ln_factorial((li - m_prime - si) as i64);

        // cos^a * sin^b — handle exact-zero base safely (0^0 = 1).
        let cos_term = if p_cos == 0 {
            1.0
        } else {
            cos_half.powi(p_cos)
        };
        let sin_term = if p_sin == 0 {
            1.0
        } else {
            sin_half.powi(p_sin)
        };

        let sign = if (m_prime - m + si).rem_euclid(2) == 0 {
            1.0
        } else {
            -1.0
        };
        sum += sign * (log_prefactor - log_denom).exp() * cos_term * sin_term;
    }
    sum
}

/// Wigner D-matrix element `D^ℓ_{m',m}(α, β, γ)` (complex):
///
/// `D^ℓ_{m',m}(α, β, γ) = e^{-i m' α} · d^ℓ_{m',m}(β) · e^{-i m γ}`, with the
/// real small-d matrix from [`wigner_small_d`].
pub fn wigner_d_element(l: u32, m_prime: i32, m: i32, alpha: F, beta: F, gamma: F) -> Complex {
    let d = wigner_small_d(l, m_prime, m, beta);
    let phase = Complex::from_polar(1.0, -(m_prime as F) * alpha - (m as F) * gamma);
    phase.scale(d)
}

/// Full `(2ℓ+1) × (2ℓ+1)` Wigner D-matrix.
///
/// Row index `r = m' + ℓ`, column index `c = m + ℓ`.
pub fn wigner_d_matrix(l: u32, alpha: F, beta: F, gamma: F) -> Array2<Complex> {
    let n = (2 * l + 1) as usize;
    let mut out = Array2::<Complex>::default((n, n));
    for m_prime in -(l as i32)..=(l as i32) {
        for m in -(l as i32)..=(l as i32) {
            let r = (m_prime + l as i32) as usize;
            let c = (m + l as i32) as usize;
            out[[r, c]] = wigner_d_element(l, m_prime, m, alpha, beta, gamma);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const TOL: F = 1e-12;
    const COARSE_TOL: F = 1e-9;

    fn approx_eq(a: F, b: F, tol: F) {
        assert!((a - b).abs() < tol, "expected {b}, got {a} (Δ={})", a - b);
    }

    fn approx_eq_c(a: Complex, b: Complex, tol: F) {
        assert!(
            (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol,
            "expected ({}, {}), got ({}, {})",
            b.re,
            b.im,
            a.re,
            a.im,
        );
    }

    // d^0_{0,0}(β) = 1
    #[test]
    fn d_zero_is_one() {
        for &b in &[0.0, 1.0, PI, 0.4] {
            approx_eq(wigner_small_d(0, 0, 0, b), 1.0, TOL);
        }
    }

    // d^1_{0,0}(β) = cos β
    // d^1_{1,1}(β) = (1 + cos β) / 2
    // d^1_{1,-1}(β) = (1 - cos β) / 2
    // d^1_{1,0}(β) = -sin β / √2
    #[test]
    fn d_one_closed_form() {
        let b = 0.7;
        approx_eq(wigner_small_d(1, 0, 0, b), b.cos(), TOL);
        approx_eq(wigner_small_d(1, 1, 1, b), 0.5 * (1.0 + b.cos()), TOL);
        approx_eq(wigner_small_d(1, 1, -1, b), 0.5 * (1.0 - b.cos()), TOL);
        approx_eq(wigner_small_d(1, 1, 0, b), -b.sin() / 2.0_f64.sqrt(), TOL);
        // d^1_{-1,1}(β) = (1 - cos β) / 2  (same as d^1_{1,-1} by symmetry)
        approx_eq(wigner_small_d(1, -1, 1, b), 0.5 * (1.0 - b.cos()), TOL);
    }

    // d^2_{0,0}(β) = (3 cos²β - 1) / 2
    #[test]
    fn d_two_zero_zero() {
        let b: F = 1.1;
        let expected = 0.5 * (3.0 * b.cos().powi(2) - 1.0);
        approx_eq(wigner_small_d(2, 0, 0, b), expected, TOL);
    }

    // d^j(β = 0) = identity
    #[test]
    fn beta_zero_is_identity() {
        for l in 0..=4 {
            for m_prime in -(l as i32)..=(l as i32) {
                for m in -(l as i32)..=(l as i32) {
                    let expected = if m == m_prime { 1.0 } else { 0.0 };
                    approx_eq(wigner_small_d(l, m_prime, m, 0.0), expected, COARSE_TOL);
                }
            }
        }
    }

    // Symmetry: d^j_{m', m}(-β) = d^j_{m, m'}(β) = (-1)^{m'-m} d^j_{m', m}(β)
    // (The second identity comes from the d-matrix being a real orthogonal rep.)
    #[test]
    fn d_negation_symmetry() {
        let l = 3;
        let b = 0.6;
        for m_prime in -(l as i32)..=(l as i32) {
            for m in -(l as i32)..=(l as i32) {
                let lhs = wigner_small_d(l, m_prime, m, -b);
                let rhs = wigner_small_d(l, m, m_prime, b);
                approx_eq(lhs, rhs, TOL);
            }
        }
    }

    // Unitarity: Σ_m d^j_{m',m}(β) * d^j_{m'',m}(β) = δ_{m',m''}
    #[test]
    fn small_d_row_orthogonal() {
        let l = 3;
        let b = 0.85;
        for m1 in -(l as i32)..=(l as i32) {
            for m2 in -(l as i32)..=(l as i32) {
                let mut acc: F = 0.0;
                for m in -(l as i32)..=(l as i32) {
                    acc += wigner_small_d(l, m1, m, b) * wigner_small_d(l, m2, m, b);
                }
                let expected = if m1 == m2 { 1.0 } else { 0.0 };
                approx_eq(acc, expected, COARSE_TOL);
            }
        }
    }

    // Full D-matrix at α=γ=0, β=0 must be the identity.
    #[test]
    fn d_matrix_identity_at_origin() {
        let l = 2u32;
        let d = wigner_d_matrix(l, 0.0, 0.0, 0.0);
        let n = (2 * l + 1) as usize;
        for r in 0..n {
            for c in 0..n {
                let expected = if r == c { Complex::ONE } else { Complex::ZERO };
                approx_eq_c(d[[r, c]], expected, COARSE_TOL);
            }
        }
    }

    // D^j(α=0, β=π/2, γ=0) phase consistency check.
    // For ℓ=1, m'=m=0: D = d(π/2) = cos(π/2) = 0.
    #[test]
    fn d_element_simple_phase() {
        let alpha = 0.4;
        let gamma = 0.6;
        let beta = FRAC_PI_2;
        let l = 1;
        let m_prime = 1;
        let m = -1;
        let d_small = wigner_small_d(l, m_prime, m, beta); // (1 - cos(π/2))/2 = 0.5
        let elem = wigner_d_element(l, m_prime, m, alpha, beta, gamma);
        let phase = Complex::from_polar(1.0, -(m_prime as F) * alpha - (m as F) * gamma);
        approx_eq_c(elem, phase.scale(d_small), TOL);
    }
}
