// Jacobi rotations naturally express updates by paired (row, col) index;
// rewriting as iterators hurts readability without speeding the loop up.
#![allow(clippy::needless_range_loop)]

//! Small dense linear algebra: 3×3 determinant and inverse, and the one
//! symmetric eigensolver (cyclic Jacobi), `n × n` and fixed 3×3 / 4×4.

use crate::op::{F, Mat3, Vec3};

/// Maximum Jacobi sweeps; 3×3 and 4×4 typically converge in ≤ 8, and the
/// quadratic convergence of cyclic Jacobi keeps an `n ≈ 100` metric matrix
/// well inside this too.
const MAX_SWEEPS: usize = 50;

/// Jacobi tolerance relative to the Frobenius norm of the input.
const JACOBI_REL_TOL: F = 1e-15;

/// Singularity threshold of [`inv3`], relative to `‖A‖_F³`.
const SINGULAR_REL_TOL: F = 1e-12;

/// Determinant of a 3×3 matrix (cofactor expansion along the first row).
pub fn det3(m: &Mat3) -> F {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Inverse of a 3×3 matrix by the adjugate, or `None` when it is singular.
///
/// The singularity test is scale-invariant: `None` iff `det = 0` or
/// `|det| ≤ 1e-12·‖A‖_F³` (both sides scale as the cube of the entries), and
/// also when an entry is non-finite. `1e-4·I` is therefore invertible, and a
/// rank-2 matrix is not.
pub fn inv3(m: &Mat3) -> Option<Mat3> {
    let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let c01 = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
    let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];

    let c10 = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    let c11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let c12 = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

    let c20 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c21 = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    let c22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];

    let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
    let fro = frobenius(m.as_flattened());
    let threshold = SINGULAR_REL_TOL * fro * fro * fro;
    // Refuse det = 0, |det| at or below the threshold, and any NaN operand.
    if det.is_nan() || threshold.is_nan() || det.abs() <= threshold {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

/// Full eigen-decomposition `A = V · diag(λ) · Vᵀ` of a symmetric 3×3 matrix.
///
/// Returns `(λ, V)` with `λ` sorted descending and column `V[·][i]` the unit
/// eigenvector of `λ[i]`. Only the upper triangle of `a` is read. A zero matrix
/// returns zero eigenvalues and the identity basis.
///
/// Cyclic Jacobi rotations (Press et al., *Numerical Recipes*, §11.1): each
/// step applies a plane rotation that zeroes one off-diagonal pair, and
/// sweeping over all pairs repeatedly drives the matrix to diagonal form, the
/// eigenvalues on the diagonal and the accumulated rotations as the
/// eigenvectors. For these sizes the closed-form (cubic / quartic root)
/// solutions are faster but numerically delicate near degenerate eigenvalues
/// (a liquid-crystal order tensor, the gyration tensor of a sphere, the Horn
/// superposition matrix of a symmetric point set); Jacobi converges in a
/// handful of sweeps and is robust there.
///
/// `‖A‖_F = √(Σᵢⱼ Aᵢⱼ²)` is the Frobenius norm. The Jacobi tolerance is
/// `tol = 1e-15·‖A‖_F`, used for the sweep stop, for
/// skipping an already-small off-diagonal pair and for the equal-diagonal
/// branch. (An absolute tolerance stops early on a small-scale matrix: at
/// 1e-8 scale an absolute 1e-14 returns a basis rotated by 0.93 rad.)
pub fn eigh_sym_3x3(a: &Mat3) -> (Vec3, Mat3) {
    eigh_sym_fixed(a)
}

/// Full eigen-decomposition `A = V · diag(λ) · Vᵀ` of a symmetric 4×4 matrix.
///
/// Same contract as [`eigh_sym_3x3`]: all four eigenvalues sorted descending,
/// unit eigenvectors as the columns of `V`, upper triangle read. This is the
/// solver behind Horn's quaternion superposition
/// ([`superpose`](crate::op::superpose)), which needs the top two
/// eigenpairs to tell a unique best-fit rotation from a family of equally good
/// rotations about one axis (a "free spin").
pub fn eigh_sym_4x4(a: &[[F; 4]; 4]) -> ([F; 4], [[F; 4]; 4]) {
    eigh_sym_fixed(a)
}

/// Full eigen-decomposition `A = V · diag(λ) · Vᵀ` of a symmetric `n × n`
/// matrix `a`, stored row-major (`a[p * n + q]`, upper triangle read).
///
/// The one cyclic-Jacobi solver; [`eigh_sym_3x3`] and [`eigh_sym_4x4`] are
/// this, fixed-size, and share its contract: `λ` sorted descending (a stable
/// sort, so equal eigenvalues keep their index order), and `V` row-major with
/// column `k` (`V[r * n + k]`) the unit eigenvector of `λ[k]`. A zero (or
/// empty) matrix returns zero eigenvalues and the identity basis.
pub(crate) fn eigh_sym(a: &[F], n: usize) -> (Vec<F>, Vec<F>) {
    debug_assert_eq!(a.len(), n * n, "eigh_sym: {n}x{n} needs {} entries", n * n);
    let mut m = vec![0.0; n * n];
    for p in 0..n {
        for q in p..n {
            m[p * n + q] = a[p * n + q];
            m[q * n + p] = a[p * n + q];
        }
    }
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    let fro = frobenius(&m);
    if fro == 0.0 {
        return (vec![0.0; n], v);
    }
    let tol = JACOBI_REL_TOL * fro;

    for _ in 0..MAX_SWEEPS {
        let mut off: F = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[p * n + q].abs();
            }
        }
        if off <= tol {
            break;
        }

        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() <= tol {
                    continue;
                }
                rotated = true;
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                // tan(2θ) = 2 apq / (app − aqq)
                let theta = if (app - aqq).abs() <= tol {
                    std::f64::consts::FRAC_PI_4 * apq.signum()
                } else {
                    0.5 * (2.0 * apq).atan2(app - aqq)
                };
                let c = theta.cos();
                let s = theta.sin();

                // J = [[c, −s], [s, c]] on columns p, q; A' = Jᵀ A J:
                //   A'[p,p] = c²·app + 2cs·apq + s²·aqq
                //   A'[q,q] = s²·app − 2cs·apq + c²·aqq
                //   A'[r,p] = c·A[r,p] + s·A[r,q]   (r ≠ p, q)
                //   A'[r,q] = −s·A[r,p] + c·A[r,q]
                m[p * n + p] = c * c * app + 2.0 * s * c * apq + s * s * aqq;
                m[q * n + q] = s * s * app - 2.0 * s * c * apq + c * c * aqq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let arp = m[r * n + p];
                        let arq = m[r * n + q];
                        let new_arp = c * arp + s * arq;
                        let new_arq = -s * arp + c * arq;
                        m[r * n + p] = new_arp;
                        m[p * n + r] = new_arp;
                        m[r * n + q] = new_arq;
                        m[q * n + r] = new_arq;
                    }
                }
                // V ← V · J
                for r in 0..n {
                    let vrp = v[r * n + p];
                    let vrq = v[r * n + q];
                    v[r * n + p] = c * vrp + s * vrq;
                    v[r * n + q] = -s * vrp + c * vrq;
                }
            }
        }
        // Every remaining pair is below tol: nothing more to rotate.
        if !rotated {
            break;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    // Stable sort: equal eigenvalues keep their index order.
    order.sort_by(|&i, &j| m[j * n + j].total_cmp(&m[i * n + i]));

    let mut vals = vec![0.0; n];
    let mut vecs = vec![0.0; n * n];
    for (k, &i) in order.iter().enumerate() {
        vals[k] = m[i * n + i];
        for r in 0..n {
            vecs[r * n + k] = v[r * n + i];
        }
    }
    (vals, vecs)
}

/// [`eigh_sym`] on a fixed-size `N × N` matrix.
fn eigh_sym_fixed<const N: usize>(a: &[[F; N]; N]) -> ([F; N], [[F; N]; N]) {
    let (vals, vecs) = eigh_sym(a.as_flattened(), N);
    let mut out_vals = [0.0; N];
    out_vals.copy_from_slice(&vals);
    let mut out_vecs = [[0.0; N]; N];
    for r in 0..N {
        out_vecs[r].copy_from_slice(&vecs[r * N..(r + 1) * N]);
    }
    (out_vals, out_vecs)
}

/// Frobenius norm `‖A‖_F` of a matrix given as its entries, row-major.
fn frobenius(entries: &[F]) -> F {
    entries.iter().map(|x| x * x).sum::<F>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{F, Mat3};

    const TOL: F = 1e-10;

    fn approx_eq(a: F, b: F, tol: F) {
        assert!((a - b).abs() < tol, "expected {b}, got {a} (Δ={})", a - b);
    }

    fn scaled(a: &Mat3, s: F) -> Mat3 {
        let mut out = *a;
        for row in out.iter_mut() {
            for x in row.iter_mut() {
                *x *= s;
            }
        }
        out
    }

    // ---------- det3 ----------

    #[test]
    fn det3_of_identity_fixture_and_singular_matrix() {
        let eye: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        approx_eq(det3(&eye), 1.0, 1e-12);

        // | 1  2  3 |
        // | 0  1  4 |
        // | 5  6  0 |
        // det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
        let m: Mat3 = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        approx_eq(det3(&m), 1.0, 1e-12);

        // Row 3 = row 1 + row 2.
        let singular: Mat3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        approx_eq(det3(&singular), 0.0, 1e-12);
    }

    // ---------- inv3 ----------

    #[test]
    fn inv3_of_identity_is_identity() {
        let eye: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = inv3(&eye).expect("identity is invertible");
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                approx_eq(inv[r][c], expected, 1e-12);
            }
        }
    }

    #[test]
    fn inv3_times_matrix_is_identity() {
        let a: Mat3 = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let a_inv = inv3(&a).expect("det = 1, invertible");
        for r in 0..3 {
            for c in 0..3 {
                let product: F = (0..3).map(|k| a[r][k] * a_inv[k][c]).sum();
                let expected = if r == c { 1.0 } else { 0.0 };
                approx_eq(product, expected, 1e-12);
            }
        }
    }

    #[test]
    fn inv3_of_rank_two_matrix_is_none() {
        let singular: Mat3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        assert!(inv3(&singular).is_none());
    }

    #[test]
    fn inv3_singularity_test_is_scale_invariant() {
        // det(1e-4 I) = 1e-12, below an absolute 1e-8 threshold, but the
        // matrix is perfectly conditioned: |det| / ‖A‖_F³ = 1 / 3^{3/2}.
        let small: Mat3 = [[1e-4, 0.0, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]];
        let inv = inv3(&small).expect("1e-4·I is invertible");
        for r in 0..3 {
            for c in 0..3 {
                if r == c {
                    let rel = (inv[r][c] - 1e4).abs() / 1e4;
                    assert!(rel < 1e-6, "inv[{r}][{c}] = {}", inv[r][c]);
                } else {
                    assert_eq!(inv[r][c], 0.0, "inv[{r}][{c}]");
                }
            }
        }
    }

    // ---------- eigh_sym_3x3 ----------

    #[test]
    fn eigh_sym_3x3_diagonal_matrix_sorts_descending() {
        let a: Mat3 = [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let (vals, _) = eigh_sym_3x3(&a);
        approx_eq(vals[0], 3.0, TOL);
        approx_eq(vals[1], 2.0, TOL);
        approx_eq(vals[2], 1.0, TOL);
    }

    #[test]
    fn eigh_sym_3x3_identity_gives_unit_eigenvalues_and_unit_columns() {
        let a: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (vals, vecs) = eigh_sym_3x3(&a);
        for v in vals.iter() {
            approx_eq(*v, 1.0, TOL);
        }
        for c in 0..3 {
            let norm: F = (0..3).map(|r| vecs[r][c].powi(2)).sum();
            approx_eq(norm, 1.0, TOL);
        }
    }

    #[test]
    fn eigh_sym_3x3_embedded_two_by_two_block() {
        // [[2 1 0] [1 2 0] [0 0 5]] → 5, 3, 1.
        let a: Mat3 = [[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]];
        let (vals, _) = eigh_sym_3x3(&a);
        approx_eq(vals[0], 5.0, TOL);
        approx_eq(vals[1], 3.0, TOL);
        approx_eq(vals[2], 1.0, TOL);
    }

    #[test]
    fn eigh_sym_3x3_columns_satisfy_eigen_equation_and_trace() {
        let a: Mat3 = [[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]];
        let (vals, vecs) = eigh_sym_3x3(&a);
        for i in 0..3 {
            for r in 0..3 {
                let av_r: F = (0..3).map(|c| a[r][c] * vecs[c][i]).sum();
                approx_eq(av_r, vals[i] * vecs[r][i], 1e-8);
            }
        }
        let tr_a: F = (0..3).map(|i| a[i][i]).sum();
        approx_eq(tr_a, vals.iter().sum(), TOL);
    }

    #[test]
    fn eigh_sym_3x3_eigenvectors_are_orthonormal() {
        let a: Mat3 = [[7.0, 2.0, -1.0], [2.0, 5.0, 3.0], [-1.0, 3.0, 6.0]];
        let (_, vecs) = eigh_sym_3x3(&a);
        for i in 0..3 {
            for j in 0..3 {
                let dot: F = (0..3).map(|r| vecs[r][i] * vecs[r][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                approx_eq(dot, expected, 1e-10);
            }
        }
    }

    #[test]
    fn eigh_sym_3x3_degenerate_eigenvalues() {
        let a: Mat3 = [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let (vals, _) = eigh_sym_3x3(&a);
        approx_eq(vals[0], 3.0, TOL);
        approx_eq(vals[1], 2.0, TOL);
        approx_eq(vals[2], 2.0, TOL);
    }

    #[test]
    fn eigh_sym_3x3_is_scale_invariant() {
        // Distinct eigenvalues, so each eigenvector is unique up to sign.
        // With an absolute 1e-14 off-diagonal tolerance the 1e-8-scaled copy
        // would stop early and return a rotated basis.
        let a: Mat3 = [[4.0, 1.0, 2.0], [1.0, 3.0, -1.0], [2.0, -1.0, 5.0]];
        let (vals, vecs) = eigh_sym_3x3(&a);
        let (vals_small, vecs_small) = eigh_sym_3x3(&scaled(&a, 1e-8));
        for c in 0..3 {
            let rel = (vals_small[c] * 1e8 - vals[c]).abs() / vals[0].abs();
            assert!(
                rel < 1e-12,
                "eigenvalue {c}: {} vs {}",
                vals_small[c],
                vals[c]
            );
            let dot: F = (0..3).map(|r| vecs[r][c] * vecs_small[r][c]).sum();
            let sign = dot.signum();
            for r in 0..3 {
                assert!(
                    (vecs_small[r][c] * sign - vecs[r][c]).abs() < 1e-12,
                    "column {c} row {r}: {} vs {}",
                    vecs_small[r][c] * sign,
                    vecs[r][c]
                );
            }
        }
    }

    // ---------- eigh_sym_4x4 ----------

    #[test]
    fn eigh_sym_4x4_returns_full_descending_spectrum() {
        // A = H diag(4, 2, -1, -3) Hᵀ with the orthogonal Hadamard
        // H = ½[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]].
        // Entry (i,j) = ¼ Σ_k h_ik h_jk d_k with h = ±1, hand-evaluated:
        let a: [[F; 4]; 4] = [
            [0.5, 1.0, 2.5, 0.0],
            [1.0, 0.5, 0.0, 2.5],
            [2.5, 0.0, 0.5, 1.0],
            [0.0, 2.5, 1.0, 0.5],
        ];
        let (vals, vecs) = eigh_sym_4x4(&a);
        let expected = [4.0, 2.0, -1.0, -3.0];
        for k in 0..4 {
            approx_eq(vals[k], expected[k], 1e-12);
        }
        // Columns are orthonormal eigenvectors.
        for i in 0..4 {
            for r in 0..4 {
                let av_r: F = (0..4).map(|c| a[r][c] * vecs[c][i]).sum();
                approx_eq(av_r, vals[i] * vecs[r][i], 1e-12);
            }
            for j in 0..4 {
                let dot: F = (0..4).map(|r| vecs[r][i] * vecs[r][j]).sum();
                approx_eq(dot, if i == j { 1.0 } else { 0.0 }, 1e-12);
            }
        }
    }

    #[test]
    fn eigh_sym_4x4_diagonal_top_pair_is_the_largest_axis() {
        let m = [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ];
        let (vals, vecs) = eigh_sym_4x4(&m);
        approx_eq(vals[0], 7.0, TOL);
        approx_eq(vecs[0][0].abs(), 0.0, TOL);
        approx_eq(vecs[1][0].abs(), 0.0, TOL);
        approx_eq(vecs[2][0].abs(), 1.0, TOL);
        approx_eq(vecs[3][0].abs(), 0.0, TOL);
    }

    #[test]
    fn eigh_sym_4x4_rank_one_outer_product() {
        // A = u uᵀ, u = (1, 2, 3, 1): eigenvalues (|u|², 0, 0, 0) = (15, 0, 0, 0).
        let u = [1.0, 2.0, 3.0, 1.0];
        let mut m = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                m[i][j] = u[i] * u[j];
            }
        }
        let (vals, vecs) = eigh_sym_4x4(&m);
        approx_eq(vals[0], 15.0, 1e-9);
        for k in 1..4 {
            approx_eq(vals[k], 0.0, 1e-9);
        }
        let dot: F = (0..4).map(|i| vecs[i][0] * u[i]).sum::<F>().abs();
        approx_eq(dot, 15.0_f64.sqrt(), 1e-9);
    }
}
