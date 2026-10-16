//! Rigid motions `p' = R p + t` and the quaternion kernels.

use crate::op::vec3::{add, cross, dot, norm, normalize, perpendicular, scale, sub, unit_or_zero};
use crate::op::{F, Mat3, Quat, Vec3};

/// A rigid motion `p' = R p + t`: rotate by `rotation`, then translate by
/// `translation`.
///
/// [`Rigid`] is a plain value; the operations on it are free functions
/// (the functional style scoped to `molrs::op`). Rotations are proper
/// orthogonal [`Mat3`]s: `Rᵀ R = I` and `det R = +1`, so lengths and angles
/// are kept and no mirror image is made.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigid {
    /// Proper rotation `R` (orthogonal, `det = +1`), row-major.
    pub rotation: Mat3,
    /// Translation `t`, applied after the rotation, in the length unit of the
    /// points it moves (Å throughout molrs).
    pub translation: Vec3,
}

impl Rigid {
    /// The motion that moves nothing: `R = I`, `t = 0`.
    pub const IDENTITY: Rigid = Rigid {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0, 0.0, 0.0],
    };
}

/// `m · v`.
#[inline]
fn mat_vec(m: &Mat3, v: Vec3) -> Vec3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// The image `R p + t` of one point.
#[inline]
pub fn transform_point(rigid: &Rigid, point: Vec3) -> Vec3 {
    add(mat_vec(&rigid.rotation, point), rigid.translation)
}

/// The rotation by `angle` radians (right-handed) about `axis`, by Rodrigues'
/// formula `R = cos θ I + sin θ [k]ₓ + (1 − cos θ) k kᵀ` with `θ = angle`,
/// `k` the unit axis and `[k]ₓ` the cross-product matrix of `k`
/// (`[k]ₓ v = k × v`).
///
/// Only the direction of `axis` matters, never its length, so any finite
/// nonzero axis is accepted however short it is: it is divided by its own
/// norm, and [`MIN_DIRECTION_LENGTH`](crate::op::vec3::MIN_DIRECTION_LENGTH)
/// (a bound on *displacements*) does not apply. `None` only when `axis` is
/// zero, has a non-finite component, or its squared length overflows or
/// underflows to zero.
pub fn axis_angle(axis: Vec3, angle: F) -> Option<Mat3> {
    let norm_sq = dot(axis, axis);
    // NaN fails the comparison, so a NaN component is rejected too.
    if !(norm_sq.is_finite() && norm_sq > 0.0) {
        return None;
    }
    let len = norm_sq.sqrt();
    let k = [axis[0] / len, axis[1] / len, axis[2] / len];
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    Some([
        [
            c + t * k[0] * k[0],
            t * k[0] * k[1] - s * k[2],
            t * k[0] * k[2] + s * k[1],
        ],
        [
            t * k[1] * k[0] + s * k[2],
            c + t * k[1] * k[1],
            t * k[1] * k[2] - s * k[0],
        ],
        [
            t * k[2] * k[0] - s * k[1],
            t * k[2] * k[1] + s * k[0],
            c + t * k[2] * k[2],
        ],
    ])
}

/// The motion that rotates by `rotation` about the fixed point `center`:
/// `t = c − R c`, so `center` maps to itself.
pub fn rotation_about(rotation: Mat3, center: Vec3) -> Rigid {
    let rc = mat_vec(&rotation, center);
    Rigid {
        rotation,
        translation: [center[0] - rc[0], center[1] - rc[1], center[2] - rc[2]],
    }
}

/// The rotation, as a unit axis and an angle in radians, that turns `from`
/// onto `to`.
///
/// The angle is in `[0, π]`. Antiparallel directions (cross product shorter
/// than 1e-8 after normalisation) give a half turn about a perpendicular
/// ([`perpendicular`]). `None` when the two already point the same way (cross
/// product at most 1e-15) or either one is not a direction ([`normalize`]).
pub fn alignment_axis_angle(from: Vec3, to: Vec3) -> Option<(Vec3, F)> {
    let (a, b) = (normalize(from)?, normalize(to)?);
    let axis = cross(a, b);
    let cross_norm = norm(axis);
    let cos = dot(a, b).clamp(-1.0, 1.0);
    // Near-antiparallel: `axis` is too short to carry a reliable direction, so
    // turn half a revolution about any perpendicular instead.
    if cos < 0.0 && cross_norm < 1e-8 {
        return Some((perpendicular(a)?, std::f64::consts::PI));
    }
    (cross_norm > 1e-15).then(|| {
        (
            [
                axis[0] / cross_norm,
                axis[1] / cross_norm,
                axis[2] / cross_norm,
            ],
            cross_norm.atan2(cos),
        )
    })
}

/// The right-handed orthonormal frame whose first axis is `primary` and
/// whose second lies in the plane of `primary` and `secondary`, as the
/// columns `[e₁ e₂ e₃]` of a proper rotation: `e₁ = p̂`,
/// `e₂ = normalize(s − (s·e₁) e₁)` (Gram–Schmidt), `e₃ = e₁ × e₂`.
///
/// `R = orthonormal_frame(a, b) · orthonormal_frame(a′, b′)ᵀ` is the rotation that takes the
/// direction `a′` onto `a` and the plane of `(a′, b′)` onto that of
/// `(a, b)`. `None` when `primary` is not a direction ([`normalize`]) or
/// `secondary` has no component across it.
pub fn orthonormal_frame(primary: Vec3, secondary: Vec3) -> Option<Mat3> {
    let e1 = normalize(primary)?;
    let e2 = normalize(sub(secondary, scale(e1, dot(secondary, e1))))?;
    let e3 = cross(e1, e2);
    Some([
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]],
    ])
}

/// The motion `outer ∘ inner`: apply `inner`, then `outer`.
///
/// `R = R_o R_i`, `t = R_o t_i + t_o`, so `transform_point(compose_rigid(o, i), p)` equals
/// `transform_point(o, transform_point(i, p))` to rounding.
pub fn compose_rigid(outer: &Rigid, inner: &Rigid) -> Rigid {
    let mut rotation = [[0.0; 3]; 3];
    for (row, out) in rotation.iter_mut().enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = (0..3)
                .map(|k| outer.rotation[row][k] * inner.rotation[k][col])
                .sum();
        }
    }
    Rigid {
        rotation,
        translation: transform_point(outer, inner.translation),
    }
}

/// Natural-extension reference frame (NeRF): the point `d` that sits at
/// distance `bond` from `c`, makes the angle `angle` at `c` with `b`
/// (`∠b–c–d`), and the dihedral `torsion` about `b → c` with `a`
/// (`a–b–c–d`). Angles in radians, `bond` in the length unit of the points.
///
/// The exact inverse of [`angle`](crate::op::vec3::angle) and
/// [`dihedral`](crate::op::vec3::dihedral): for the returned `d`,
/// `angle(b, c, d) == angle` and `dihedral(a, b, c, d) == torsion` to
/// rounding. With `b̂c` the unit `b → c`, `n̂` the unit normal of the plane
/// `(a, b, c)` and `m = n̂ × b̂c`, `d = c + b̂c·(−r cos θ) + m·(r sin θ cos φ) +
/// n̂·(r sin θ sin φ)`. Collinear `a, b, c` leave the plane undefined: `n̂`
/// then comes back as zero (see
/// [`unit_or_zero`](crate::op::vec3)), and so does every off-axis component.
///
/// Parsons et al., *J. Comput. Chem.* **26** (2005) 1063.
pub fn place_from_internal_coords(
    a: Vec3,
    b: Vec3,
    c: Vec3,
    bond: F,
    angle: F,
    torsion: F,
) -> Vec3 {
    let bc = unit_or_zero(sub(c, b));
    let n = unit_or_zero(cross(sub(b, a), bc));
    let m = cross(n, bc);
    let (st, ct) = angle.sin_cos();
    let (sp, cp) = torsion.sin_cos();
    let d2 = [-bond * ct, bond * st * cp, bond * st * sp];
    [
        c[0] + bc[0] * d2[0] + m[0] * d2[1] + n[0] * d2[2],
        c[1] + bc[1] * d2[0] + m[1] * d2[1] + n[1] * d2[2],
        c[2] + bc[2] * d2[0] + m[2] * d2[1] + n[2] * d2[2],
    ]
}

/// Quaternion conjugate `q* = (w, −x, −y, −z)`.
#[inline]
pub fn quat_conj(q: Quat) -> Quat {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product `a ⊗ b` (`i ⊗ j = k`, not commutative).
#[inline]
pub fn quat_mul(a: Quat, b: Quat) -> Quat {
    let (aw, ax, ay, az) = (a[0], a[1], a[2], a[3]);
    let (bw, bx, by, bz) = (b[0], b[1], b[2], b[3]);
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

/// Euclidean norm `|q|`.
#[inline]
pub fn quat_norm(q: Quat) -> F {
    quat_dot(q, q).sqrt()
}

/// Four-dimensional dot product `a · b`.
#[inline]
pub fn quat_dot(a: Quat, b: Quat) -> F {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

/// Rotate `v` by the unit quaternion `q`: `q v q*`.
///
/// Evaluated as `v + w t + q⃗ × t` with `t = 2 q⃗ × v`, which needs no
/// quaternion product. `q` is assumed unit; it is not normalised.
#[inline]
pub fn rotate_by_quat(q: Quat, v: Vec3) -> Vec3 {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let tx = 2.0 * (y * v[2] - z * v[1]);
    let ty = 2.0 * (z * v[0] - x * v[2]);
    let tz = 2.0 * (x * v[1] - y * v[0]);
    [
        v[0] + w * tx + (y * tz - z * ty),
        v[1] + w * ty + (z * tx - x * tz),
        v[2] + w * tz + (x * ty - y * tx),
    ]
}

/// The proper rotation matrix of `q`, which is normalised first (so any
/// non-zero multiple of a unit quaternion gives the same matrix; a zero `q`
/// gives NaN entries).
pub fn quat_to_matrix(q: Quat) -> Mat3 {
    let n = quat_norm(q);
    let (w, x, y, z) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{F, Mat3, Quat, Vec3};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const TOL: F = 1e-12;

    /// Quarter turn about +z: x̂ → ŷ, ŷ → −x̂.
    const RZ90: Mat3 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

    fn assert_vec_close(got: Vec3, want: Vec3) {
        for d in 0..3 {
            assert!(
                (got[d] - want[d]).abs() < TOL,
                "component {d}: expected {want:?}, got {got:?}"
            );
        }
    }

    fn assert_quat_close(got: Quat, want: Quat) {
        for d in 0..4 {
            assert!(
                (got[d] - want[d]).abs() < TOL,
                "component {d}: expected {want:?}, got {got:?}"
            );
        }
    }

    fn assert_mat_close(got: Mat3, want: Mat3) {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got[r][c] - want[r][c]).abs() < TOL,
                    "entry [{r}][{c}]: expected {want:?}, got {got:?}"
                );
            }
        }
    }

    // ---------- Rigid / apply ----------

    #[test]
    fn identity_leaves_points_unchanged() {
        assert_eq!(
            transform_point(&Rigid::IDENTITY, [1.5, -2.0, 3.0]),
            [1.5, -2.0, 3.0]
        );
    }

    #[test]
    fn apply_rotates_then_translates() {
        let rigid = Rigid {
            rotation: RZ90,
            translation: [1.0, 2.0, 3.0],
        };
        // R x̂ + t = ŷ + (1,2,3)
        assert_vec_close(transform_point(&rigid, [1.0, 0.0, 0.0]), [1.0, 3.0, 3.0]);
    }

    // ---------- axis_angle ----------

    #[test]
    fn axis_angle_quarter_turn_about_z_maps_x_to_y() {
        let r = axis_angle([0.0, 0.0, 1.0], FRAC_PI_2).expect("ẑ is a direction");
        assert_vec_close(mat_vec(&r, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_mat_close(r, RZ90);
    }

    #[test]
    fn axis_angle_normalizes_a_non_unit_axis() {
        let r = axis_angle([0.0, 0.0, 5.0], FRAC_PI_2).expect("5ẑ is a direction");
        assert_mat_close(r, RZ90);
    }

    #[test]
    fn axis_angle_refuses_zero_axis() {
        assert_eq!(axis_angle([0.0, 0.0, 0.0], FRAC_PI_2), None);
    }

    // ---------- about ----------

    #[test]
    fn about_keeps_its_center_fixed() {
        let c = [1.0, 2.0, 3.0];
        let rigid = rotation_about(RZ90, c);
        assert_vec_close(transform_point(&rigid, c), c);
        // A point one unit along x̂ from the centre ends one unit along ŷ.
        assert_vec_close(transform_point(&rigid, [2.0, 2.0, 3.0]), [1.0, 3.0, 3.0]);
    }

    // ---------- quaternion kernels ----------

    // ---------- alignment ----------

    #[test]
    fn alignment_of_perpendicular_directions_is_a_quarter_turn() {
        let (axis, angle) =
            alignment_axis_angle([1.0, 0.0, 0.0], [0.0, 2.0, 0.0]).expect("distinct");
        assert_vec_close(axis, [0.0, 0.0, 1.0]);
        assert!((angle - FRAC_PI_2).abs() < TOL, "angle = {angle}");
    }

    #[test]
    fn alignment_of_antiparallel_directions_is_a_half_turn_about_a_perpendicular() {
        let (axis, angle) =
            alignment_axis_angle([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]).expect("antiparallel");
        assert!((angle - PI).abs() < TOL, "angle = {angle}");
        assert!(axis[0].abs() < TOL, "axis {axis:?} not ⟂ x̂");
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        assert!((len - 1.0).abs() < TOL, "|axis| = {len}");
    }

    #[test]
    fn alignment_of_parallel_directions_is_none() {
        assert_eq!(alignment_axis_angle([0.0, 1.0, 0.0], [0.0, 3.0, 0.0]), None);
    }

    #[test]
    fn alignment_refuses_a_zero_direction() {
        assert_eq!(alignment_axis_angle([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]), None);
    }

    // ---------- frame ----------

    #[test]
    fn frame_takes_the_primary_axis_and_the_secondary_plane() {
        // primary +y, secondary (1, 1, 0): e1 = ŷ, e2 = x̂, e3 = ŷ × x̂ = −ẑ.
        let f = orthonormal_frame([0.0, 2.0, 0.0], [1.0, 1.0, 0.0]).expect("a frame");
        assert_eq!(f, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]);
    }

    #[test]
    fn frame_refuses_a_secondary_along_the_primary() {
        assert_eq!(orthonormal_frame([1.0, 0.0, 0.0], [-3.0, 0.0, 0.0]), None);
        assert_eq!(orthonormal_frame([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), None);
    }

    // ---------- compose ----------

    #[test]
    fn compose_applies_inner_then_outer() {
        // inner: quarter turn about z then +x; outer: shift +z.
        let inner = Rigid {
            rotation: RZ90,
            translation: [1.0, 0.0, 0.0],
        };
        let outer = Rigid {
            rotation: Rigid::IDENTITY.rotation,
            translation: [0.0, 0.0, 2.0],
        };
        // (1, 0, 0) -> RZ90 -> (0, 1, 0) -> +x -> (1, 1, 0) -> +z -> (1, 1, 2).
        let p = [1.0, 0.0, 0.0];
        assert_vec_close(
            transform_point(&compose_rigid(&outer, &inner), p),
            [1.0, 1.0, 2.0],
        );
        assert_vec_close(
            transform_point(&compose_rigid(&outer, &inner), p),
            transform_point(&outer, transform_point(&inner, p)),
        );
    }

    #[test]
    fn quat_conj_negates_the_vector_part() {
        assert_eq!(quat_conj([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, -3.0, -4.0]);
    }

    #[test]
    fn quat_mul_i_times_j_is_k() {
        let i = [0.0, 1.0, 0.0, 0.0];
        let j = [0.0, 0.0, 1.0, 0.0];
        assert_quat_close(quat_mul(i, j), [0.0, 0.0, 0.0, 1.0]);
        // Hamilton, not commutative: j ⊗ i = −k.
        assert_quat_close(quat_mul(j, i), [0.0, 0.0, 0.0, -1.0]);
    }

    #[test]
    fn quat_mul_conjugate_times_q_is_squared_norm() {
        // |q|² = 1 + 4 + 9 + 16 = 30.
        let q = [1.0, 2.0, 3.0, 4.0];
        assert_quat_close(quat_mul(quat_conj(q), q), [30.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn quat_norm_of_known_quaternion() {
        assert!((quat_norm([1.0, 2.0, 3.0, 4.0]) - 30.0_f64.sqrt()).abs() < TOL);
    }

    #[test]
    fn quat_dot_of_known_quaternions() {
        // 5 + 12 + 21 + 32 = 70
        assert_eq!(quat_dot([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]), 70.0);
    }

    #[test]
    fn rotate_by_quat_quarter_turn_about_z_maps_x_to_y() {
        let q = [FRAC_PI_4.cos(), 0.0, 0.0, FRAC_PI_4.sin()];
        assert_vec_close(rotate_by_quat(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn rotate_by_conjugate_quat_inverts_the_rotation() {
        let q = [FRAC_PI_4.cos(), 0.0, 0.0, FRAC_PI_4.sin()];
        let v = [0.3, -1.2, 2.5];
        let back = rotate_by_quat(quat_conj(q), rotate_by_quat(q, v));
        assert_vec_close(back, v);
    }

    #[test]
    fn quat_to_matrix_half_turn_about_y() {
        let r = quat_to_matrix([0.0, 0.0, 1.0, 0.0]);
        assert_mat_close(r, [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]);
    }

    #[test]
    fn quat_to_matrix_normalizes_its_input() {
        // (2, 0, 0, 0) is the identity rotation once normalised.
        let r = quat_to_matrix([2.0, 0.0, 0.0, 0.0]);
        assert_mat_close(r, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    #[test]
    fn nerf_inverts_angle_and_dihedral() {
        use crate::op::vec3::{angle, dihedral};
        let (a, b, c) = ([0.3, -1.1, 0.2], [0.0, 0.0, 0.0], [1.5, 0.1, -0.2]);
        for &(r, theta, phi) in &[(1.09, 1.91, 0.4), (1.53, 2.1, -2.9), (0.96, 1.2, PI)] {
            let d = place_from_internal_coords(a, b, c, r, theta, phi);
            assert!((norm(sub(d, c)) - r).abs() < 1e-12);
            assert!((angle(b, c, d) - theta).abs() < 1e-12);
            let got = dihedral(a, b, c, d);
            let wrapped = (got - phi + PI).rem_euclid(2.0 * PI) - PI;
            assert!(wrapped.abs() < 1e-12, "torsion {phi} came back {got}");
        }
    }
}
