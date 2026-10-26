//! Complex and real spherical harmonics with the physics / Condon–Shortley phase convention.

use libm::lgamma;

use super::FOUR_PI;
use crate::core::Complex;
use crate::op::F;

/// Associated Legendre polynomial `P_ℓ^m(x)` for `|x| ≤ 1`, `0 ≤ m ≤ ℓ`.
///
/// Uses the stable upward recursion in ℓ. The Condon–Shortley `(−1)^m`
/// phase is included.
pub fn legendre_plm(l: u32, m: u32, x: F) -> F {
    debug_assert!(m <= l, "legendre_plm: m={m} must be ≤ ℓ={l}");
    debug_assert!(x.abs() <= 1.0 + 1e-12, "legendre_plm: |x|={} > 1", x.abs());

    // P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
    let mut pmm: F = 1.0;
    if m > 0 {
        let somx2 = ((1.0 - x) * (1.0 + x)).sqrt();
        let mut fact: F = 1.0;
        for _ in 0..m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }

    // P_{m+1}^m(x) = x (2m+1) P_m^m(x)
    let mut pmmp1 = x * (2.0 * m as F + 1.0) * pmm;
    if l == m + 1 {
        return pmmp1;
    }

    // Upward in ℓ:
    // (ℓ - m) P_ℓ^m = (2ℓ-1) x P_{ℓ-1}^m - (ℓ + m - 1) P_{ℓ-2}^m
    let mut pll: F = 0.0;
    for ll in (m + 2)..=l {
        pll = ((2.0 * ll as F - 1.0) * x * pmmp1 - (ll as F + m as F - 1.0) * pmm)
            / (ll as F - m as F);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Normalization constant `√((2ℓ+1)/(4π) · (ℓ−m)!/(ℓ+m)!)`.
///
/// Uses `lgamma` to stay well-conditioned for ℓ ≳ 12.
#[inline]
pub fn ylm_normalization(l: u32, m: u32) -> F {
    let lf = l as F;
    let mf = m as F;
    let log_ratio = lgamma(lf - mf + 1.0) - lgamma(lf + mf + 1.0);
    ((2.0 * lf + 1.0) / FOUR_PI * log_ratio.exp()).sqrt()
}

/// Complex spherical harmonic `Y_ℓ^m(θ, φ)`.
///
/// `θ ∈ [0, π]` is the polar angle, `φ ∈ [0, 2π)` the azimuthal angle.
///
/// `m` may be negative; `|m| ≤ ℓ` is required.
///
/// # Convention
///
/// `Y_ℓ^m(θ, φ) = N · P_ℓ^m(cos θ) · e^{i m φ}` with
/// `N = √((2ℓ+1)/(4π) · (ℓ−m)!/(ℓ+m)!)` ([`ylm_normalization`]) and the
/// Condon-Shortley `(−1)^m` phase baked into the associated Legendre polynomial
/// ([`legendre_plm`]) via the upward recursion. Matches
/// `scipy.special.sph_harm` and freud's `Wigner3j.cc` / `Steinhardt.cc`
/// conventions. Negative-m values use the symmetry
/// `Y_ℓ^{−m} = (−1)^m · conj(Y_ℓ^m)`.
///
/// # References
///
/// - Press et al., *Numerical Recipes in C*, §6.8 (recursion form).
/// - Condon & Shortley, *The Theory of Atomic Spectra*, Ch. III.
pub fn ylm_complex(l: u32, m: i32, theta: F, phi: F) -> Complex {
    let abs_m = m.unsigned_abs();
    debug_assert!(abs_m <= l, "ylm_complex: |m|={abs_m} must be ≤ ℓ={l}");

    let n = ylm_normalization(l, abs_m);
    let plm = legendre_plm(l, abs_m, theta.cos());
    let phase = Complex::from_polar(1.0, abs_m as F * phi);
    let positive_m = phase.scale(n * plm);

    if m >= 0 {
        positive_m
    } else {
        // Y_ℓ^{-m} = (-1)^m * conj(Y_ℓ^m)
        let sign = if abs_m & 1 == 0 { 1.0 } else { -1.0 };
        positive_m.conj().scale(sign)
    }
}

/// Real spherical harmonic basis (Wigner-D friendly), as used by some freud
/// modules (e.g. `LocalDescriptors` in real mode).
///
/// `Y_ℓm = { √2·Re(Y_ℓ^|m|)  if m > 0,
///           Y_ℓ^0                if m = 0,
///           √2·Im(Y_ℓ^|m|)  if m < 0 }`
pub fn ylm_real(l: u32, m: i32, theta: F, phi: F) -> F {
    use std::f64::consts::SQRT_2;
    let y = ylm_complex(l, m.abs(), theta, phi);
    match m.cmp(&0) {
        std::cmp::Ordering::Greater => SQRT_2 * y.re,
        std::cmp::Ordering::Equal => y.re,
        std::cmp::Ordering::Less => SQRT_2 * y.im,
    }
}

/// Fill `out[0..=2ℓ]` with `Y_ℓ^m` for `m = -ℓ, …, ℓ` (index `m + ℓ`).
///
/// More efficient than repeated `ylm_complex` calls when all `m` for one ℓ
/// are needed (e.g. Steinhardt qℓm accumulation).
pub fn ylm_all(l: u32, theta: F, phi: F, out: &mut [Complex]) {
    debug_assert_eq!(
        out.len(),
        (2 * l + 1) as usize,
        "ylm_all: out.len() must equal 2ℓ+1"
    );
    let cos_t = theta.cos();
    for m in 0..=l {
        let n = ylm_normalization(l, m);
        let plm = legendre_plm(l, m, cos_t);
        let phase = Complex::from_polar(1.0, m as F * phi);
        let pos = phase.scale(n * plm);
        out[(l + m) as usize] = pos;
        if m > 0 {
            let sign = if m & 1 == 0 { 1.0 } else { -1.0 };
            out[(l - m) as usize] = pos.conj().scale(sign);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, SQRT_2};

    const TOL: F = 1e-12;
    const COARSE_TOL: F = 1e-6;

    // --- Closed-form values from Wolfram / Wikipedia (Condon-Shortley phase) ---

    fn approx_eq(a: F, b: F, tol: F) {
        assert!((a - b).abs() < tol, "expected {b}, got {a} (Δ={})", a - b);
    }

    fn approx_eq_c(a: Complex, b: Complex, tol: F) {
        assert!(
            (a.re - b.re).abs() < tol && (a.im - b.im).abs() < tol,
            "expected ({},{}), got ({},{})",
            b.re,
            b.im,
            a.re,
            a.im,
        );
    }

    #[test]
    fn ylm_00_is_constant() {
        // Y_0^0 = 1 / (2 √π)
        let expected = 1.0 / (2.0 * PI.sqrt());
        for &(t, p) in &[(0.3, 0.0), (1.0, 2.1), (PI / 2.0, 3.0)] {
            let y = ylm_complex(0, 0, t, p);
            approx_eq(y.re, expected, TOL);
            approx_eq(y.im, 0.0, TOL);
        }
    }

    #[test]
    fn ylm_10_axis() {
        // Y_1^0 = ½ √(3/π) cos θ
        let theta: F = 0.7;
        let phi: F = 1.2;
        let expected = 0.5 * (3.0 / PI).sqrt() * theta.cos();
        let y = ylm_complex(1, 0, theta, phi);
        approx_eq(y.re, expected, TOL);
        approx_eq(y.im, 0.0, TOL);
    }

    #[test]
    fn ylm_1pm1_phase_and_amplitude() {
        // Y_1^{ 1} = - ½ √(3/(2π)) sin θ e^{ i φ}
        // Y_1^{-1} =   ½ √(3/(2π)) sin θ e^{-i φ}   (using -m symmetry)
        let theta: F = 0.6;
        let phi: F = 0.4;
        let amp = 0.5 * (3.0 / (2.0 * PI)).sqrt() * theta.sin();

        let yp = ylm_complex(1, 1, theta, phi);
        approx_eq(yp.re, -amp * phi.cos(), TOL);
        approx_eq(yp.im, -amp * phi.sin(), TOL);

        let yn = ylm_complex(1, -1, theta, phi);
        approx_eq(yn.re, amp * phi.cos(), TOL);
        approx_eq(yn.im, -amp * phi.sin(), TOL);
    }

    #[test]
    fn ylm_20_axis() {
        // Y_2^0 = ¼ √(5/π) (3 cos²θ − 1)
        let theta: F = 1.3;
        let expected = 0.25 * (5.0 / PI).sqrt() * (3.0 * theta.cos().powi(2) - 1.0);
        let y = ylm_complex(2, 0, theta, 0.5);
        approx_eq(y.re, expected, TOL);
        approx_eq(y.im, 0.0, TOL);
    }

    #[test]
    fn ylm_22() {
        // Y_2^{2} = ¼ √(15/(2π)) sin²θ e^{2 i φ}
        let theta: F = 0.9;
        let phi: F = 0.7;
        let amp = 0.25 * (15.0 / (2.0 * PI)).sqrt() * theta.sin().powi(2);
        let y = ylm_complex(2, 2, theta, phi);
        approx_eq(y.re, amp * (2.0 * phi).cos(), TOL);
        approx_eq(y.im, amp * (2.0 * phi).sin(), TOL);
    }

    #[test]
    fn symmetry_negative_m() {
        // Y_ℓ^{-m} = (-1)^m conj(Y_ℓ^m)
        let theta = 0.85;
        let phi = 2.3;
        for l in 1..=6u32 {
            for m in 1..=(l as i32) {
                let yp = ylm_complex(l, m, theta, phi);
                let yn = ylm_complex(l, -m, theta, phi);
                let sign = if (m as u32) & 1 == 0 { 1.0 } else { -1.0 };
                let expected = yp.conj().scale(sign);
                approx_eq_c(yn, expected, TOL);
            }
        }
    }

    #[test]
    fn ylm_all_matches_scalar() {
        let theta = 1.1;
        let phi = 2.2;
        for l in 0..=6u32 {
            let mut buf = vec![Complex::ZERO; (2 * l + 1) as usize];
            ylm_all(l, theta, phi, &mut buf);
            for m in -(l as i32)..=(l as i32) {
                let from_scalar = ylm_complex(l, m, theta, phi);
                let from_bulk = buf[(m + l as i32) as usize];
                approx_eq_c(from_bulk, from_scalar, TOL);
            }
        }
    }

    #[test]
    fn orthonormality_gauss_legendre() {
        // ∫ Y_ℓ^m * conj(Y_ℓ'^m') dΩ = δ_{ℓℓ'} δ_{mm'}
        //
        // Cheap product rule: equispaced φ + Gauss-Legendre in cos θ (n=16
        // suffices for ℓ ≤ 5 with full accuracy at FP64).
        let (xs, ws) = gauss_legendre_16();
        let nphi = 32;
        let dphi = 2.0 * PI / nphi as F;

        for l1 in 0..=4u32 {
            for m1 in -(l1 as i32)..=(l1 as i32) {
                for l2 in 0..=4u32 {
                    for m2 in -(l2 as i32)..=(l2 as i32) {
                        let mut acc = Complex::ZERO;
                        for (i, &x) in xs.iter().enumerate() {
                            let theta = x.acos();
                            let w = ws[i];
                            for k in 0..nphi {
                                let phi = k as F * dphi;
                                let y1 = ylm_complex(l1, m1, theta, phi);
                                let y2 = ylm_complex(l2, m2, theta, phi);
                                acc += y1 * y2.conj() * (w * dphi);
                            }
                        }
                        let expected = if l1 == l2 && m1 == m2 { 1.0 } else { 0.0 };
                        approx_eq(acc.re, expected, COARSE_TOL);
                        approx_eq(acc.im, 0.0, COARSE_TOL);
                    }
                }
            }
        }
    }

    #[test]
    fn ylm_real_basis() {
        // For m = 0, real basis == complex Y_ℓ^0
        let theta = 0.4;
        let phi = 1.5;
        for l in 0..=3 {
            let yc = ylm_complex(l, 0, theta, phi);
            let yr = ylm_real(l, 0, theta, phi);
            approx_eq(yr, yc.re, TOL);
        }
        // For m > 0: yr = √2 * Re(Y_ℓ^|m|)
        let l = 2;
        let m = 1;
        let yc = ylm_complex(l, m, theta, phi);
        let yr = ylm_real(l, m, theta, phi);
        approx_eq(yr, SQRT_2 * yc.re, TOL);
        // For m < 0: yr = √2 * Im(Y_ℓ^|m|)
        let yr_neg = ylm_real(l, -m, theta, phi);
        approx_eq(yr_neg, SQRT_2 * yc.im, TOL);
    }

    /// 16-point Gauss–Legendre nodes and weights on (-1, 1).
    /// Standard tabulated values (Abramowitz & Stegun 25.4).
    fn gauss_legendre_16() -> ([F; 16], [F; 16]) {
        let xs: [F; 16] = [
            -0.989_400_934_991_649_9,
            -0.944_575_023_073_232_6,
            -0.865_631_202_387_831_7,
            -0.755_404_408_355_003,
            -0.617_876_244_402_643_7,
            -0.458_016_777_657_227_4,
            -0.281_603_550_779_258_9,
            -0.095_012_509_837_637_44,
            0.095_012_509_837_637_44,
            0.281_603_550_779_258_9,
            0.458_016_777_657_227_4,
            0.617_876_244_402_643_7,
            0.755_404_408_355_003,
            0.865_631_202_387_831_7,
            0.944_575_023_073_232_6,
            0.989_400_934_991_649_9,
        ];
        let ws: [F; 16] = [
            0.027_152_459_411_754_1,
            0.062_253_523_938_647_9,
            0.095_158_511_682_492_8,
            0.124_628_971_255_534,
            0.149_595_988_816_576_7,
            0.169_156_519_395_002_5,
            0.182_603_415_044_923_6,
            0.189_450_610_455_068_5,
            0.189_450_610_455_068_5,
            0.182_603_415_044_923_6,
            0.169_156_519_395_002_5,
            0.149_595_988_816_576_7,
            0.124_628_971_255_534,
            0.095_158_511_682_492_8,
            0.062_253_523_938_647_9,
            0.027_152_459_411_754_1,
        ];
        (xs, ws)
    }
}
