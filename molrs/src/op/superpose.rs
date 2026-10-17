//! Weighted least-squares superposition of matched point sets (Horn's
//! quaternion method), and the weighted centroid.

use crate::op::vec3::{dot, norm, sub};
use crate::op::{F, Mat3, Vec3};
use crate::op::{Rigid, quat_conj, quat_mul, quat_to_matrix, transform_point};
use crate::op::{eigh_sym_3x3, eigh_sym_4x4};

/// Default threshold on the scale-free eigen-gap `ρ = (λ₁ − λ₂)/(2σ₁)` below
/// which [`superpose`] reports the rotation as under-determined.
///
/// The orientation uncertainty of a fit is `δθ ≲ 2ε_S/ρ`, where `ε_S` is the
/// relative error of `S`: ~`k·1.1e-16` for fp64-exact input of `k` points and
/// ~1e-6 for 6-significant-digit input. At `ρ = τ = 1e-4` the uncertainty is
/// therefore ≤ 0.02 rad (≈1.1°) for 6-digit input and ≤ ~`2e-12·k` rad for
/// exact input. Below `τ` the fit is under-determined and reported as
/// [`Freedom::Spin`], left to the caller to complete.
pub const DEFAULT_GAP_TOL: F = 1e-4;

/// `σ₁` at or below this fraction of `√(Σw|p|²·Σw|x|²)` counts as zero.
const FREE_REL_TOL: F = 1e-15;

/// How far the data determine the rotation of a [`Superposition`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Freedom {
    /// The optimal rotation is unique (`ρ ≥ gap_tol`).
    Unique,
    /// Every rotation `Rot(axis, α)·R` about `axis` through [`Superposition::center`] is
    /// (near-)optimal; `axis` is a unit vector.
    Spin {
        /// Unit spin axis `R₁v`, through [`Superposition::center`].
        axis: Vec3,
    },
    /// No rotation is determined (`σ₁ = 0`); the fit is a translation and
    /// `R = I`.
    Free,
}

/// The result of [`superpose`]: the motion mapping reference onto target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Superposition {
    /// Best-fit proper motion, `yᵢ ≈ R rᵢ + t`.
    pub rigid: Rigid,
    /// Weighted root-mean-square deviation
    /// `√(Σ w |R rᵢ + t − yᵢ|² / Σ w)`, in the coordinates' length unit (Å in
    /// molrs); points of weight 0 do not enter.
    pub rmsd: F,
    /// Scale-free eigen-gap `ρ = (λ₁ − λ₂)/(2σ₁)` (dimensionless); 0 when
    /// [`Freedom::Free`].
    pub rho: F,
    /// Weighted target centroid `c_y`, in the coordinates' length unit: the
    /// point a [`Freedom::Spin`] axis passes through.
    pub center: Vec3,
    /// How far the data determine the rotation.
    pub freedom: Freedom,
}

/// Why [`superpose`] refused its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperpositionError {
    /// `reference`, `target` and `weights` do not have matching lengths.
    LengthMismatch {
        /// Number of reference points.
        reference: usize,
        /// Number of target points.
        target: usize,
        /// Number of weights.
        weights: usize,
    },
    /// No point has a positive weight.
    NoPoints,
    /// The weight at `index` is negative or non-finite.
    BadWeight {
        /// Point index of the offending weight.
        index: usize,
    },
    /// A reference or target coordinate at point `index` is non-finite.
    NonFinite {
        /// Point index of the offending coordinate.
        index: usize,
    },
}

impl std::fmt::Display for SuperpositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LengthMismatch {
                reference,
                target,
                weights,
            } => write!(
                f,
                "superpose length mismatch: {reference} reference points, {target} target points, {weights} weights"
            ),
            Self::NoPoints => write!(f, "superpose needs at least one positive weight"),
            Self::BadWeight { index } => {
                write!(f, "superpose weight {index} is negative or non-finite")
            }
            Self::NonFinite { index } => {
                write!(f, "superpose coordinate of point {index} is non-finite")
            }
        }
    }
}

impl std::error::Error for SuperpositionError {}

/// Weighted centroid `Σ wᵢ pᵢ / Σ wᵢ` (with masses as weights, the centre of
/// mass), in the length unit of `points`.
///
/// Weights need not be positive individually; `None` when the lengths differ
/// or the total weight is not positive and finite (e.g. all zero).
pub fn centroid(points: &[Vec3], weights: &[F]) -> Option<Vec3> {
    if points.len() != weights.len() {
        return None;
    }
    let total: F = weights.iter().sum();
    if !(total > 0.0 && total.is_finite()) {
        return None;
    }
    let mut c = [0.0; 3];
    for (p, &w) in points.iter().zip(weights) {
        for d in 0..3 {
            c[d] += w * p[d];
        }
    }
    Some([c[0] / total, c[1] / total, c[2] / total])
}

/// Best-fit proper rigid motion mapping `reference` onto `target`, weighted by
/// `weights`, with the rotation's [`Freedom`] judged against `gap_tol` on `ρ`
/// (use [`DEFAULT_GAP_TOL`] unless a caller has its own error budget).
///
/// **Superposition** answers: given two copies of the same set of points,
/// `reference` rows `rᵢ` and `target` rows `yᵢ` (row `i` of one matched to row
/// `i` of the other), which rotation `R` and translation `t` lay the reference
/// onto the target as closely as possible? "Proper" means `R` is a true
/// rotation (determinant +1), never a mirror image. With weights `wᵢ ≥ 0`,
/// [`superpose`] returns the proper rigid motion `p' = R p + t` minimising
/// `Σ wᵢ |R rᵢ + t − yᵢ|²`, its weighted RMSD (root-mean-square deviation,
/// `√(Σ wᵢ |R rᵢ + t − yᵢ|² / Σ wᵢ)`, in the coordinates' length unit, Å in
/// molrs), and how well the data fix the rotation ([`Freedom`]).
///
/// # Method
///
/// Horn, *J. Opt. Soc. Am. A* **4**, 629 (1987), doi:10.1364/JOSAA.4.000629,
/// §2.C and App. A2/A3; Coutsias, Seok & Dill, *J. Comput. Chem.* **25**, 1849
/// (2004), doi:10.1002/jcc.20110; Kabsch, *Acta Cryst. A* **32**, 922 (1976),
/// doi:10.1107/S0567739476001873.
///
/// - Points of weight 0 are dropped (their coordinates are not read
///   further). With `c_r = Σ w r / Σ w` and `c_y` alike,
///   `p = r − c_r` and `x = y − c_y`.
/// - `S = Σ w p xᵀ`. The weight enters once (Coutsias's "multiply by `w_k`"
///   means `√w` on each factor).
/// - Horn's symmetric 4×4 key matrix `N(S)` has the optimal rotation quaternion
///   `q₁` as its top eigenvector ([`eigh_sym_4x4`]). A unit quaternion always
///   encodes a proper rotation, so there is no `det = −1` branch: the
///   reflection check Kabsch's singular-value method needs (to reject a
///   mirror-image `det = −1` solution) is automatic here.
/// - `t = c_y − R c_r`.
/// - `RMSD_w² = Σ w |R p − x|² / Σ w`, evaluated from the residuals. The
///   algebraically equal `(Σw|p|² + Σw|x|² − 2λ₁)/Σw` cancels catastrophically
///   near a perfect fit (a 1e-16 error in `λ₁` is a 1e-8 RMSD).
///
/// # Uniqueness
///
/// The eigenvalues of `N` are `{σ₁+σ₂+χσ₃, σ₁−σ₂−χσ₃, −σ₁+σ₂−χσ₃,
/// −σ₁−σ₂+χσ₃}` with `σ` the singular values of `S` and `χ = sgn det S`, so
/// `λ₁ − λ₂ = 2(σ₂ + χσ₃)` and the fit is unique iff `λ₁ > λ₂`. The robust
/// test is on the scale-free gap `ρ = (λ₁ − λ₂)/(2σ₁)` with
/// `σ₁ = √λ_max(SᵀS)`:
///
/// - `σ₁ ≈ 0` (one point, or a target collapsed to a point): no rotation is
///   determined, [`Freedom::Free`], and `R = I`.
/// - `ρ ≥ gap_tol`: [`Freedom::Unique`].
/// - otherwise (two points, collinear points, correspondence-rank loss, mirror
///   data with `det S < 0` and `σ₂ = σ₃`): [`Freedom::Spin`]. With
///   `λ₁ = λ₂` every `q(φ) = q₁ cos φ + q₂ sin φ` is optimal, and
///   `q₁* ⊗ q₂ = (0, v)` with `|v| = 1`, so `R(φ) = Rot(R₁v, 2φ)·R₁`, where
///   `Rot(a, α)` is the rotation by `α` about the unit axis `a`: a free
///   spin about the axis `R₁v` through `c_y`. The returned `rigid` is the
///   `φ = 0` member; choosing another is the caller's job.
///
/// Under-determination is **reported, not refused**.
///
/// # Errors
///
/// - [`SuperpositionError::LengthMismatch`] if the three lengths differ.
/// - [`SuperpositionError::BadWeight`] for a negative or non-finite weight.
/// - [`SuperpositionError::NonFinite`] for a non-finite coordinate of a weighted
///   point, in either set.
/// - [`SuperpositionError::NoPoints`] if no weight is positive.
pub fn superpose(
    reference: &[Vec3],
    target: &[Vec3],
    weights: &[F],
    gap_tol: F,
) -> Result<Superposition, SuperpositionError> {
    if reference.len() != target.len() || reference.len() != weights.len() {
        return Err(SuperpositionError::LengthMismatch {
            reference: reference.len(),
            target: target.len(),
            weights: weights.len(),
        });
    }

    let mut r_kept = Vec::with_capacity(reference.len());
    let mut y_kept = Vec::with_capacity(reference.len());
    let mut w_kept = Vec::with_capacity(reference.len());
    for (index, ((r, y), &w)) in reference.iter().zip(target).zip(weights).enumerate() {
        if !(w.is_finite() && w >= 0.0) {
            return Err(SuperpositionError::BadWeight { index });
        }
        if w == 0.0 {
            continue;
        }
        if !(r.iter().chain(y).all(|x| x.is_finite())) {
            return Err(SuperpositionError::NonFinite { index });
        }
        r_kept.push(*r);
        y_kept.push(*y);
        w_kept.push(w);
    }
    if w_kept.is_empty() {
        return Err(SuperpositionError::NoPoints);
    }

    // A positive finite total is guaranteed unless the sum overflows.
    let c_r = centroid(&r_kept, &w_kept).ok_or(SuperpositionError::NoPoints)?;
    let c_y = centroid(&y_kept, &w_kept).ok_or(SuperpositionError::NoPoints)?;
    let total: F = w_kept.iter().sum();

    let p: Vec<Vec3> = r_kept.iter().map(|&r| sub(r, c_r)).collect();
    let x: Vec<Vec3> = y_kept.iter().map(|&y| sub(y, c_y)).collect();

    let mut s: Mat3 = [[0.0; 3]; 3];
    let mut g_p: F = 0.0;
    let mut g_x: F = 0.0;
    for ((pi, xi), &w) in p.iter().zip(&x).zip(&w_kept) {
        for a in 0..3 {
            for b in 0..3 {
                s[a][b] += w * pi[a] * xi[b];
            }
        }
        g_p += w * dot(*pi, *pi);
        g_x += w * dot(*xi, *xi);
    }

    // σ₁ = √λ_max(SᵀS).
    let mut sts: Mat3 = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            sts[a][b] = s[0][a] * s[0][b] + s[1][a] * s[1][b] + s[2][a] * s[2][b];
        }
    }
    let sigma1 = eigh_sym_3x3(&sts).0[0].max(0.0).sqrt();

    let (rotation, rho, freedom) = if g_p == 0.0 || sigma1 <= FREE_REL_TOL * (g_p * g_x).sqrt() {
        (Rigid::IDENTITY.rotation, 0.0, Freedom::Free)
    } else {
        let (lambda, vecs) = eigh_sym_4x4(&horn_matrix(&s));
        let q1 = [vecs[0][0], vecs[1][0], vecs[2][0], vecs[3][0]];
        let rotation = quat_to_matrix(q1);
        let rho = (lambda[0] - lambda[1]) / (2.0 * sigma1);
        let freedom = if rho >= gap_tol {
            Freedom::Unique
        } else {
            let q2 = [vecs[0][1], vecs[1][1], vecs[2][1], vecs[3][1]];
            // q₁ ⟂ q₂ are unit, so q₁* ⊗ q₂ = (0, v) with |v| = 1 up to
            // rounding; renormalise v.
            let d = quat_mul(quat_conj(q1), q2);
            let v = [d[1], d[2], d[3]];
            let n = norm(v);
            let v = [v[0] / n, v[1] / n, v[2] / n];
            let axis = transform_point(
                &Rigid {
                    rotation,
                    translation: [0.0; 3],
                },
                v,
            );
            Freedom::Spin { axis }
        };
        (rotation, rho, freedom)
    };

    let linear = Rigid {
        rotation,
        translation: [0.0; 3],
    };
    let mut residual: F = 0.0;
    for ((pi, xi), &w) in p.iter().zip(&x).zip(&w_kept) {
        let e = sub(transform_point(&linear, *pi), *xi);
        residual += w * dot(e, e);
    }
    let rc = transform_point(&linear, c_r);

    Ok(Superposition {
        rigid: Rigid {
            rotation,
            translation: sub(c_y, rc),
        },
        rmsd: (residual / total).sqrt(),
        rho,
        center: c_y,
        freedom,
    })
}

/// Horn's symmetric key matrix `N(S)` for `S = Σ w p xᵀ` (Horn 1987, §4).
fn horn_matrix(s: &Mat3) -> [[F; 4]; 4] {
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    [
        [sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
        [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
        [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
        [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{F, Mat3, Vec3};
    use crate::op::{Rigid, transform_point};
    use crate::op::{det3, eigh_sym_4x4};

    const TOL: F = 1e-12;

    const EYE: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// Quarter turn about +z: x̂ → ŷ, ŷ → −x̂.
    const RZ90: Mat3 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_close(got: F, want: F, tol: F, what: &str) {
        assert!(
            (got - want).abs() < tol,
            "{what}: expected {want}, got {got}"
        );
    }

    fn assert_vec_close(got: Vec3, want: Vec3, tol: F, what: &str) {
        for d in 0..3 {
            assert!(
                (got[d] - want[d]).abs() < tol,
                "{what}[{d}]: expected {want:?}, got {got:?}"
            );
        }
    }

    fn assert_mat_close(got: &Mat3, want: &Mat3, tol: F, what: &str) {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got[r][c] - want[r][c]).abs() < tol,
                    "{what}[{r}][{c}]: expected {want:?}, got {got:?}"
                );
            }
        }
    }

    // ---------- centroid ----------

    #[test]
    fn centroid_is_weighted_mean() {
        let c = centroid(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], &[1.0, 3.0]);
        let c = c.expect("positive total weight");
        assert_vec_close(c, [3.0, 0.0, 0.0], TOL, "centroid");
    }

    #[test]
    fn centroid_of_all_zero_weights_is_none() {
        assert_eq!(
            centroid(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], &[0.0, 0.0]),
            None
        );
    }

    // ---------- goldens K1..K7 (spec Domain basis) ----------

    #[test]
    fn k1_single_point_is_a_free_translation() {
        let fit = superpose(
            &[[1.0, 2.0, 3.0]],
            &[[4.0, 4.0, 4.0]],
            &[1.0],
            DEFAULT_GAP_TOL,
        )
        .expect("one weighted point is enough for a translation");
        assert_eq!(fit.freedom, Freedom::Free);
        assert_mat_close(&fit.rigid.rotation, &EYE, TOL, "R");
        assert_vec_close(fit.rigid.translation, [3.0, 2.0, 1.0], TOL, "t");
        assert_vec_close(fit.center, [4.0, 4.0, 4.0], TOL, "center");
        assert_close(fit.rmsd, 0.0, TOL, "rmsd");
    }

    #[test]
    fn k2_two_points_spin_about_the_target_line() {
        let reference = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let target = [[1.0, 1.0, 1.0], [1.0, 3.0, 1.0]];
        let fit = superpose(&reference, &target, &[1.0, 1.0], DEFAULT_GAP_TOL)
            .expect("a two-point fit is reported, not refused");
        // S = 2 x̂ŷᵀ, λ = {2, 2, −2, −2}: every optimal R maps x̂ → ŷ, so the
        // free spin is about ŷ through c_y.
        match fit.freedom {
            Freedom::Spin { axis } => assert_close(axis[1].abs(), 1.0, TOL, "|axis·ŷ|"),
            other => panic!("expected Spin, got {other:?}"),
        }
        let mapped: Vec<Vec3> = reference
            .iter()
            .map(|&p| transform_point(&fit.rigid, p))
            .collect();
        for (i, (m, y)) in mapped.iter().zip(target.iter()).enumerate() {
            assert_vec_close(*m, *y, TOL, &format!("mapped point {i}"));
        }
    }

    #[test]
    fn k3_quarter_turn_is_recovered_exactly() {
        let reference = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let target = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]];
        let fit = superpose(&reference, &target, &[1.0; 3], DEFAULT_GAP_TOL).unwrap();
        assert_eq!(fit.freedom, Freedom::Unique);
        assert_mat_close(&fit.rigid.rotation, &RZ90, TOL, "R");
        assert_vec_close(fit.rigid.translation, [0.0, 0.0, 0.0], TOL, "t");
        assert_close(fit.rmsd, 0.0, TOL, "rmsd");
    }

    #[test]
    fn k3_scaled_by_1e_minus_9_gives_the_same_rotation() {
        let s = 1e-9;
        let reference = [[0.0, 0.0, 0.0], [s, 0.0, 0.0], [0.0, s, 0.0]];
        let target = [[0.0, 0.0, 0.0], [0.0, s, 0.0], [-s, 0.0, 0.0]];
        let fit = superpose(&reference, &target, &[1.0; 3], DEFAULT_GAP_TOL).unwrap();
        assert_eq!(fit.freedom, Freedom::Unique);
        assert_mat_close(&fit.rigid.rotation, &RZ90, TOL, "R at 1e-9 scale");
        assert_vec_close(fit.rigid.translation, [0.0, 0.0, 0.0], TOL, "t");
    }

    // K4 point order: (1,0,0), (0,1,0), (−1,0,0), (0,−1,0) with target
    // z-offsets (+δ, −δ, +δ, −δ), δ = 0.3. This order makes Σ w r = 0 and
    // S diagonal for both weightings below.
    const K4_REFERENCE: [Vec3; 4] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    const K4_TARGET: [Vec3; 4] = [
        [1.0, 0.0, 0.3],
        [0.0, 1.0, -0.3],
        [-1.0, 0.0, 0.3],
        [0.0, -1.0, -0.3],
    ];

    #[test]
    fn k4_unit_weights_fit_identity_with_rmsd_delta() {
        // S = diag(2, 2, 0); RMSD² = (4 + 4.36 − 2·4) / 4 = 0.09.
        let fit = superpose(&K4_REFERENCE, &K4_TARGET, &[1.0; 4], DEFAULT_GAP_TOL).unwrap();
        assert_eq!(fit.freedom, Freedom::Unique);
        assert_mat_close(&fit.rigid.rotation, &EYE, TOL, "R");
        assert_vec_close(fit.rigid.translation, [0.0, 0.0, 0.0], TOL, "t");
        assert_close(fit.rmsd, 0.3, TOL, "rmsd");
    }

    /// Regression example (spec assembly-01-op, K4 weighted). Hard-coded golden,
    /// hand-derived: c_y = (0, 0, 0.1); S = diag(4, 2, 0); λ(N) = {6, 2, −2, −6};
    /// RMSD_w² = (6 + 6.48 − 12) / 6 = 0.08.
    #[test]
    fn k4_weighted_fit_matches_hand_derivation() {
        let weights = [2.0, 1.0, 2.0, 1.0];
        let s: Mat3 = [[4.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]];
        let (lambda, _) = eigh_sym_4x4(&horn_matrix(&s));
        let expected = [6.0, 2.0, -2.0, -6.0];
        for k in 0..4 {
            assert_close(lambda[k], expected[k], TOL, &format!("λ{}", k + 1));
        }

        let fit = superpose(&K4_REFERENCE, &K4_TARGET, &weights, DEFAULT_GAP_TOL).unwrap();
        assert_eq!(fit.freedom, Freedom::Unique);
        assert_mat_close(&fit.rigid.rotation, &EYE, TOL, "R");
        assert_vec_close(fit.rigid.translation, [0.0, 0.0, 0.1], TOL, "t");
        assert_vec_close(fit.center, [0.0, 0.0, 0.1], TOL, "center");
        assert_close(fit.rmsd, 0.28284271247461906, TOL, "rmsd_w");
        // ρ = (λ1 − λ2) / (2σ1) = 4 / 8.
        assert_close(fit.rho, 0.5, TOL, "rho");
    }

    #[test]
    fn k5_mirror_tetrahedron_best_proper_fit() {
        // Moving set = target tetrahedron with z negated. S = D C with
        // D = diag(1,1,−1), C = I − J/4: σ = {1, 1, 1/4}, χ = −1, so
        // λ = {7/4, 1/4, 1/4, −9/4} and RMSD² = (9/4 + 9/4 − 7/2) / 4 = 1/4.
        let target = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        let weights = [1.0; 4];
        // S = D C written out: p = D x, so S = D Σ x xᵀ = D (I − J/4).
        let s: Mat3 = [
            [0.75, -0.25, -0.25],
            [-0.25, 0.75, -0.25],
            [0.25, 0.25, -0.75],
        ];
        let (lambda, _) = eigh_sym_4x4(&horn_matrix(&s));
        assert_close(lambda[0], 7.0 / 4.0, TOL, "λ1");
        assert_close(lambda[1], 1.0 / 4.0, TOL, "λ2");

        let fit = superpose(&reference, &target, &weights, DEFAULT_GAP_TOL).unwrap();
        assert_eq!(fit.freedom, Freedom::Unique);
        assert_close(det3(&fit.rigid.rotation), 1.0, TOL, "det R");
        assert_close(fit.rmsd, 0.5, TOL, "rmsd");
        // ρ = (7/4 − 1/4) / (2·1).
        assert_close(fit.rho, 0.75, TOL, "rho");
    }

    #[test]
    fn k6_mirrored_octahedron_is_a_spin_that_flips_x() {
        // S = diag(−8, 2, 2): σ = {8, 2, 2}, χ = −1, λ = {8, 8, −4, −12}.
        let reference = [
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let target: Vec<Vec3> = reference.iter().map(|p| [-p[0], p[1], p[2]]).collect();
        let fit = superpose(&reference, &target, &[1.0; 6], DEFAULT_GAP_TOL).unwrap();
        // Every optimal R is a half turn about an axis in the yz-plane, so the
        // spin axis is x̂.
        match fit.freedom {
            Freedom::Spin { axis } => assert_close(axis[0].abs(), 1.0, TOL, "|axis·x̂|"),
            other => panic!("expected Spin, got {other:?}"),
        }
        assert_close(fit.rmsd, 1.1547005383792515, TOL, "rmsd");
        assert_close(fit.rho, 0.0, TOL, "rho");
        assert_vec_close(
            transform_point(
                &Rigid {
                    rotation: fit.rigid.rotation,
                    translation: [0.0; 3],
                },
                [1.0, 0.0, 0.0],
            ),
            [-1.0, 0.0, 0.0],
            TOL,
            "R x̂",
        );
    }

    #[test]
    fn k7_correspondence_rank_loss_is_not_unique() {
        // S = 2 x̂x̂ᵀ, λ = {2, 2, −2, −2}.
        let reference = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        let target = [
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let fit = superpose(&reference, &target, &[1.0; 4], DEFAULT_GAP_TOL).unwrap();
        assert_ne!(fit.freedom, Freedom::Unique);
        assert!(
            matches!(fit.freedom, Freedom::Spin { .. }),
            "σ1 = 2 > 0 and ρ = 0: expected Spin, got {:?}",
            fit.freedom
        );
    }

    // ---------- known motion and degenerate input ----------

    // superpose maps reference onto target, so R = R_true and t = t_true.
    #[test]
    fn recovers_known_rotation_and_translation() {
        let reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let (c, s) = (0.7_f64.cos(), 0.7_f64.sin());
        let r_true: Mat3 = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let t_true = [3.0, -2.0, 5.0];
        let target: Vec<Vec3> = reference
            .iter()
            .map(|p| {
                let rp = transform_point(
                    &Rigid {
                        rotation: r_true,
                        translation: [0.0; 3],
                    },
                    *p,
                );
                [rp[0] + t_true[0], rp[1] + t_true[1], rp[2] + t_true[2]]
            })
            .collect();
        let fit = superpose(&reference, &target, &[1.0; 4], DEFAULT_GAP_TOL).unwrap();
        assert_eq!(fit.freedom, Freedom::Unique);
        assert!(fit.rmsd < 1e-9, "rmsd = {}", fit.rmsd);
        assert_close(det3(&fit.rigid.rotation), 1.0, 1e-9, "det R");
        assert_mat_close(&fit.rigid.rotation, &r_true, 1e-9, "R");
        assert_vec_close(fit.rigid.translation, t_true, 1e-9, "t");
    }

    // A line is reported as a free spin about itself, not refused.
    #[test]
    fn collinear_reference_is_reported_as_spin_about_the_line() {
        let line = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let fit = superpose(&line, &line, &[1.0; 3], DEFAULT_GAP_TOL).unwrap();
        match fit.freedom {
            Freedom::Spin { axis } => assert_close(axis[0].abs(), 1.0, TOL, "|axis·x̂|"),
            other => panic!("expected Spin, got {other:?}"),
        }
        let mapped: Vec<Vec3> = line
            .iter()
            .map(|&p| transform_point(&fit.rigid, p))
            .collect();
        for (i, (m, y)) in mapped.iter().zip(line.iter()).enumerate() {
            assert_vec_close(*m, *y, TOL, &format!("mapped point {i}"));
        }
    }

    // ---------- weights ----------

    #[test]
    fn zero_weight_point_does_not_move_the_fit() {
        // K3 plus an outlier at weight 0.
        let reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [100.0, 100.0, 100.0],
        ];
        let target = [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-5.0, 3.0, 2.0],
        ];
        let fit = superpose(&reference, &target, &[1.0, 1.0, 1.0, 0.0], DEFAULT_GAP_TOL).unwrap();
        assert_mat_close(&fit.rigid.rotation, &RZ90, TOL, "R");
        assert_vec_close(fit.rigid.translation, [0.0, 0.0, 0.0], TOL, "t");
    }

    // ---------- refusals ----------

    #[test]
    fn target_length_mismatch_is_refused() {
        let err = superpose(
            &[[0.0; 3], [1.0, 0.0, 0.0]],
            &[[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            &[1.0, 1.0],
            DEFAULT_GAP_TOL,
        )
        .unwrap_err();
        assert_eq!(
            err,
            SuperpositionError::LengthMismatch {
                reference: 2,
                target: 3,
                weights: 2
            }
        );
    }

    #[test]
    fn weights_length_mismatch_is_refused() {
        let err = superpose(
            &[[0.0; 3], [1.0, 0.0, 0.0]],
            &[[0.0; 3], [1.0, 0.0, 0.0]],
            &[1.0],
            DEFAULT_GAP_TOL,
        )
        .unwrap_err();
        assert_eq!(
            err,
            SuperpositionError::LengthMismatch {
                reference: 2,
                target: 2,
                weights: 1
            }
        );
    }

    #[test]
    fn negative_weight_is_refused() {
        let pts = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let err = superpose(&pts, &pts, &[1.0, -1.0, 1.0], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::BadWeight { index: 1 });
    }

    #[test]
    fn nan_weight_is_refused() {
        let pts = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let err = superpose(&pts, &pts, &[F::NAN, 1.0, 1.0], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::BadWeight { index: 0 });
    }

    #[test]
    fn infinite_weight_is_refused() {
        let pts = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let err = superpose(&pts, &pts, &[1.0, 1.0, F::INFINITY], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::BadWeight { index: 2 });
    }

    #[test]
    fn all_zero_weights_are_refused() {
        let pts = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let err = superpose(&pts, &pts, &[0.0; 3], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::NoPoints);
    }

    #[test]
    fn empty_input_is_refused() {
        let err = superpose(&[], &[], &[], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::NoPoints);
    }

    #[test]
    fn nan_reference_coordinate_is_refused() {
        let good = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut bad = good;
        bad[1][0] = F::NAN;
        let err = superpose(&bad, &good, &[1.0; 3], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::NonFinite { index: 1 });
    }

    #[test]
    fn infinite_target_coordinate_is_refused() {
        let good = [[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut bad = good;
        bad[2][1] = F::INFINITY;
        let err = superpose(&good, &bad, &[1.0; 3], DEFAULT_GAP_TOL).unwrap_err();
        assert_eq!(err, SuperpositionError::NonFinite { index: 2 });
    }

    #[test]
    fn coincident_single_point_fits_the_identity() {
        // Free (Σw|p|² = 0) → R = I; t = c_y − c_r = 0.
        let fit = superpose(
            &[[1.0, 2.0, 3.0]],
            &[[1.0, 2.0, 3.0]],
            &[1.0],
            DEFAULT_GAP_TOL,
        )
        .unwrap();
        assert_eq!(fit.rigid, Rigid::IDENTITY);
    }
}
