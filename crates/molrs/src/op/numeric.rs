//! The crate's scalar and array aliases — their one owner.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Primary floating-point scalar type — always `f64`.
///
/// Scientific algorithms (potentials, optimizers, coordinate transforms) require
/// double precision.  Lower precision is only used in accelerator hot-paths
/// (GPU kernels) or estimation algorithms, and those are handled locally, not
/// through this project-wide alias.
pub type F = f64;

// ---- Owned arrays ----

/// An owned float vector: a 3-element position, velocity, force or
/// displacement, or any N-element column.
pub type F3 = Array1<F>;

/// An owned float matrix: N×3 (a collection of 3D vectors) or 3×3 (a box
/// matrix, rotation or stress tensor).
pub type Fnx3 = Array2<F>;

// ---- Views ----

/// Borrowed view of a 3-element vector.
pub type F3View<'a> = ArrayView1<'a, F>;

/// Borrowed N×3 view.
pub type Fnx3View<'a> = ArrayView2<'a, F>;

// ---- Non-float ----

/// Primary signed integer scalar type — always `i32`.
pub type I = i32;

/// An index into a block, or a stable entity identifier.
///
/// Named for what it *means*, not for what it *is*: the width a column
/// stores at and the type a domain value happens to be are separate jobs, and
/// a formal charge wants to be small where an identifier wants to be wide.
///
/// Every column this appears in is identity: `id`, `mol_id`, `type_id`,
/// `res_id`, and the `atomi`/`atomj`/`atomk`/`atoml` relation endpoints.
/// Sixty-four bits because an identifier that wraps is not an identifier.
///
/// Rename this through the compiler -- it points only at type positions --
/// never through a regex: a one-letter type name also matches element
/// symbols and force-field type rows.
pub type Idx = u64;

/// Per-axis periodic boundary condition flags.
pub type Pbc3 = [bool; 3];

// ---- Stack types of the op kernels ----

/// A 3-vector on the stack: a point, a displacement or a direction.
pub type Vec3 = [F; 3];

/// A 3×3 matrix on the stack, row-major: `m[row][col]`.
pub type Mat3 = [[F; 3]; 3];

/// A quaternion `(w, x, y, z)`: scalar part first, then the vector part.
///
/// A quaternion is a four-component number `w + x i + y j + z k` whose
/// imaginary units multiply as `i² = j² = k² = ijk = −1`; unit-length
/// quaternions encode 3D rotations without the gimbal problems of angles.
///
/// This is the crate's one quaternion type. Products are Hamilton products
/// (`i ⊗ j = k`); a unit quaternion `q` rotates the vector `v` as `q v q*`,
/// where `v` is written as the quaternion `(0, v)` and
/// `q* = (w, −x, −y, −z)` is the conjugate. The rotation angle `θ` about the
/// unit axis `n` is encoded as `q = (cos θ/2, sin θ/2 · n)`.
pub type Quat = [F; 4];

/// Copy a 3-element ndarray view onto the stack.
///
/// # Panics
///
/// If `v` does not have exactly 3 elements.
pub fn to_vec3(v: ArrayView1<'_, F>) -> Vec3 {
    assert_eq!(v.len(), 3, "to_vec3 expects 3 elements, got {}", v.len());
    [v[0], v[1], v[2]]
}

/// Copy a 3×3 ndarray view onto the stack, row-major.
///
/// # Panics
///
/// If `m` is not 3×3.
pub fn to_mat3(m: ArrayView2<'_, F>) -> Mat3 {
    assert_eq!(
        m.dim(),
        (3, 3),
        "to_mat3 expects a 3×3 matrix, got {:?}",
        m.dim()
    );
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            *x = m[[r, c]];
        }
    }
    out
}
