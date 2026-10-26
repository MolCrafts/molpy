//! Wigner 3-j symbols for integer angular momenta: [`wigner_3j`].

use super::factorial::ln_factorial;
use crate::op::F;

/// Integer Wigner 3-j symbol
///
/// ```text
///     ⎛ j1  j2  j3 ⎞
///     ⎝ m1  m2  m3 ⎠
/// ```
///
/// via the Racah single-sum form, evaluated through `lgamma` to stay
/// well-conditioned for ℓ up to ~30. Matches `freud/order/Wigner3j.cc`
/// conventions exactly.
///
/// Returns 0 for any input violating the selection rules (all must hold):
///
/// - `m1 + m2 + m3 = 0`
/// - `|m_i| ≤ j_i`
/// - `|j1 − j2| ≤ j3 ≤ j1 + j2`
///
/// # References
///
/// - Edmonds, *Angular Momentum in Quantum Mechanics*, eq. (3.7.3).
/// - Racah, *Phys. Rev.* 62, 438 (1942).
pub fn wigner_3j(j1: u32, j2: u32, j3: u32, m1: i32, m2: i32, m3: i32) -> F {
    if m1 + m2 + m3 != 0 {
        return 0.0;
    }
    if m1.unsigned_abs() > j1 || m2.unsigned_abs() > j2 || m3.unsigned_abs() > j3 {
        return 0.0;
    }
    let j1i = j1 as i32;
    let j2i = j2 as i32;
    let j3i = j3 as i32;
    if j3i < (j1i - j2i).abs() || j3i > j1i + j2i {
        return 0.0;
    }

    // Δ(j1,j2,j3) = (j1+j2-j3)!(j1-j2+j3)!(-j1+j2+j3)! / (j1+j2+j3+1)!
    let log_delta = ln_factorial((j1i + j2i - j3i) as i64)
        + ln_factorial((j1i - j2i + j3i) as i64)
        + ln_factorial((-j1i + j2i + j3i) as i64)
        - ln_factorial((j1i + j2i + j3i + 1) as i64);

    let log_factorials = ln_factorial((j1i - m1) as i64)
        + ln_factorial((j1i + m1) as i64)
        + ln_factorial((j2i - m2) as i64)
        + ln_factorial((j2i + m2) as i64)
        + ln_factorial((j3i - m3) as i64)
        + ln_factorial((j3i + m3) as i64);

    let log_prefactor = 0.5 * (log_delta + log_factorials);

    let t_min = 0i64
        .max((j2i - j3i - m1) as i64)
        .max((j1i - j3i + m2) as i64);
    let t_max = ((j1i + j2i - j3i) as i64)
        .min((j1i - m1) as i64)
        .min((j2i + m2) as i64);

    if t_min > t_max {
        return 0.0;
    }

    let mut sum: F = 0.0;
    for t in t_min..=t_max {
        let ti = t as i32;
        let log_term = ln_factorial(t)
            + ln_factorial((j3i - j2i + ti + m1) as i64)
            + ln_factorial((j3i - j1i + ti - m2) as i64)
            + ln_factorial((j1i + j2i - j3i - ti) as i64)
            + ln_factorial((j1i - ti - m1) as i64)
            + ln_factorial((j2i - ti + m2) as i64);
        let sign = if t & 1 == 0 { 1.0 } else { -1.0 };
        sum += sign * (log_prefactor - log_term).exp();
    }

    let outer_sign = if (j1i - j2i - m3).rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    };
    outer_sign * sum
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: F = 1e-12;

    fn approx_eq(a: F, b: F, tol: F) {
        assert!((a - b).abs() < tol, "expected {b}, got {a} (Δ={})", a - b);
    }

    // --- Selection-rule zeros ---

    #[test]
    fn m_sum_nonzero_is_zero() {
        assert_eq!(wigner_3j(2, 2, 2, 1, 1, 1), 0.0);
        assert_eq!(wigner_3j(3, 3, 4, 2, -1, 0), 0.0);
    }

    #[test]
    fn triangle_violation_is_zero() {
        assert_eq!(wigner_3j(1, 1, 5, 0, 0, 0), 0.0);
        assert_eq!(wigner_3j(2, 3, 6, 0, 0, 0), 0.0);
    }

    #[test]
    fn abs_m_exceeds_j_is_zero() {
        assert_eq!(wigner_3j(2, 2, 2, 3, -3, 0), 0.0);
    }

    // --- Closed-form reference values (Edmonds Table 2 / Mathematica ThreeJSymbol) ---

    #[test]
    fn all_zero_simple() {
        // (0 0 0; 0 0 0) = 1
        approx_eq(wigner_3j(0, 0, 0, 0, 0, 0), 1.0, TOL);
    }

    #[test]
    fn one_one_zero() {
        // (1 1 0; 0 0 0) = -1/√3
        approx_eq(wigner_3j(1, 1, 0, 0, 0, 0), -1.0 / 3.0_f64.sqrt(), TOL);
    }

    #[test]
    fn two_two_zero() {
        // (2 2 0; 0 0 0) = 1/√5
        approx_eq(wigner_3j(2, 2, 0, 0, 0, 0), 1.0 / 5.0_f64.sqrt(), TOL);
    }

    #[test]
    fn two_two_two() {
        // (2 2 2; 0 0 0) = -√(2/35)
        approx_eq(wigner_3j(2, 2, 2, 0, 0, 0), -(2.0_f64 / 35.0).sqrt(), TOL);
    }

    #[test]
    fn four_four_four_all_zero() {
        // (4 4 4; 0 0 0) = 3 · √(2/1001)   (Edmonds 3.7.17 → 90/√450450)
        // ≈ 0.134180...
        let expected = 3.0_f64 * (2.0_f64 / 1001.0).sqrt();
        approx_eq(wigner_3j(4, 4, 4, 0, 0, 0), expected, TOL);
    }

    // --- Symmetry properties ---

    #[test]
    fn column_permutation_symmetry() {
        // Even permutation of columns leaves the symbol invariant.
        for j1 in 0..=4 {
            for j2 in 0..=4 {
                let lo = (j1 as i32 - j2 as i32).unsigned_abs();
                let hi = j1 + j2;
                for j3 in lo..=hi {
                    for m1 in -(j1 as i32)..=(j1 as i32) {
                        for m2 in -(j2 as i32)..=(j2 as i32) {
                            let m3 = -m1 - m2;
                            if m3.unsigned_abs() > j3 {
                                continue;
                            }
                            let w_123 = wigner_3j(j1, j2, j3, m1, m2, m3);
                            // Cyclic permutation (j2, j3, j1) — even → invariant
                            let w_231 = wigner_3j(j2, j3, j1, m2, m3, m1);
                            approx_eq(w_123, w_231, TOL);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn m_sign_flip_symmetry() {
        // Flipping all m signs multiplies by (-1)^(j1+j2+j3).
        let j1 = 3;
        let j2 = 4;
        let j3 = 5;
        let m1 = 2;
        let m2 = -1;
        let m3 = -m1 - m2;
        let w = wigner_3j(j1, j2, j3, m1, m2, m3);
        let w_neg = wigner_3j(j1, j2, j3, -m1, -m2, -m3);
        let sign = if (j1 + j2 + j3) & 1 == 0 { 1.0 } else { -1.0 };
        approx_eq(w_neg, sign * w, TOL);
    }

    #[test]
    fn orthogonality_sum_over_m1m2() {
        // Edmonds (3.7.8): ∑_{m1,m2} (j1 j2 j3; m1 m2 m3)² = 1 / (2j3+1)
        // when m3 is held fixed and the (j1,j2,j3) triangle inequality holds.
        let j1 = 2u32;
        let j2 = 3u32;
        let j = 4u32;
        for m_fixed in -(j as i32)..=(j as i32) {
            let mut acc: F = 0.0;
            for m1 in -(j1 as i32)..=(j1 as i32) {
                let m2 = -m_fixed - m1;
                if m2.unsigned_abs() > j2 {
                    continue;
                }
                let w = wigner_3j(j1, j2, j, m1, m2, m_fixed);
                acc += w * w;
            }
            approx_eq(acc, 1.0 / (2.0 * j as F + 1.0), 1e-10);
        }
    }
}
