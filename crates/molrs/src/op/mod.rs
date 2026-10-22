//! Pure numeric primitives beneath `core`: stack vectors, small linear algebra, rigid motions, superposition and uniform direction sampling.
//!
//! Everything here computes on plain fixed-size arrays (`[f64; 3]` points,
//! row-major 3×3 matrices, `(w, x, y, z)` quaternions) and names no other
//! molrs module. Coordinates are in whatever length unit the caller uses; in
//! molrs that is Å, because every coordinate reader normalises to Å.
//!
//! The namespace is flat — `molrs::op::<name>`, the same path as Python's
//! `molrs.op.<name>` — except [`vec3`], which stays a namespace because its
//! names (`add`, `sub`, `scale`, `dot`, …) only read right qualified.
//!
//! | Kind | What it provides |
//! |---|---|
//! | numeric aliases | the scalar [`F`] `= f64`, the ndarray aliases, and the stack types [`Vec3`], [`Mat3`], [`Quat`] |
//! | [`vec3`] | vector arithmetic: sum, difference, dot and cross products, normalisation; the internal coordinates (bond angle, dihedral) |
//! | linear algebra | 3×3 determinant and inverse; eigenvalues and eigenvectors of symmetric 3×3 / 4×4 matrices |
//! | rigid motions | [`Rigid`] — a rotation followed by a translation, which moves a body without deforming it — the quaternion kernels, and NeRF placement of a point from internal coordinates ([`place_from_internal_coords`]) |
//! | superposition | [`superpose`] → [`Superposition`]: the rigid motion that best lays one set of matched points onto another (least squares), and the weighted [`centroid`] |
//! | sampling | uniform directions on S² ([`unit_vector_from_uniform`]) and the standard normal ([`standard_normal`]) |
//!
//! # Numeric aliases
//!
//! Three families, each alias with exactly one public path, `molrs::op::<Alias>`:
//!
//! - The **F-prefix family** of ndarray-backed aliases over the scalar [`F`]
//!   (always `f64`): [`F3`] (any `Array1<F>`), [`Fnx3`] (any `Array2<F>`) and
//!   their views — the API types of the crate's column stores. One name per
//!   type: a 3×3 box matrix is an `Fnx3`, an N-vector an `F3`.
//! - The **non-float aliases** [`I`] (signed integer), [`Idx`] (an index or
//!   stable identifier) and [`Pbc3`] (per-axis periodic flags).
//! - The **stack aliases** [`Vec3`], [`Mat3`] and [`Quat`] that the `op`
//!   kernels compute on. [`to_vec3`] and [`to_mat3`] are where ndarray meets
//!   them.
//!
//! # Quaternions
//!
//! A **quaternion** is a four-component number `q = w + x i + y j + z k`,
//! stored as [`Quat`] `(w, x, y, z)`, multiplied with the Hamilton rules
//! `i² = j² = k² = ijk = −1` (so `i j = k` but `j i = −k`). Its conjugate is
//! `q* = w − x i − y j − z k`. A *unit* quaternion (`|q| = 1`) encodes a
//! rotation: writing a vector `v` as the pure quaternion `0 + vₓ i + v_y j +
//! v_z k`, the rotated vector is `q v q*`; the rotation by angle `θ` about the
//! unit axis `k̂` is `q = (cos(θ/2), sin(θ/2) k̂)`.
//!
//! # Linear algebra tolerances
//!
//! Every tolerance of [`inv3`], [`eigh_sym_3x3`] and [`eigh_sym_4x4`] is
//! **relative** to the matrix norm, so each result is
//! invariant under a uniform rescaling of the input: a structure in nm and the
//! same structure in Å give the same eigenvectors, and a well-conditioned
//! matrix of small entries is never declared singular.
pub(crate) mod linalg;
mod numeric;
mod random;
mod rigid;
mod so3;
mod superpose;
pub mod vec3;

pub use linalg::{det3, eigh_sym_3x3, eigh_sym_4x4, inv3};
pub use numeric::{
    F, F3, F3View, Fnx3, Fnx3View, I, Idx, Mat3, Pbc3, Quat, Vec3, to_mat3, to_vec3,
};
pub use random::standard_normal;
pub use rigid::{
    Rigid, alignment_axis_angle, axis_angle, compose_rigid, orthonormal_frame,
    place_from_internal_coords, quat_conj, quat_dot, quat_mul, quat_norm, quat_to_matrix,
    rotate_by_quat, rotation_about, transform_point,
};
pub use so3::unit_vector_from_uniform;
pub use superpose::{
    DEFAULT_GAP_TOL, Freedom, Superposition, SuperpositionError, centroid, superpose,
};
