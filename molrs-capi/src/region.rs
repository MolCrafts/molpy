//! `extern "C"` functions for geometric regions.
//!
//! A **region** is a solid that answers a signed distance: negative inside,
//! positive outside, zero on the boundary. `contains` is that sign and
//! `bounds` is the axis-aligned box the solid fits in — the same three
//! questions the Python and WASM binders ask.
//!
//! Regions compose: [`molrs_region_and`], [`molrs_region_or`] and
//! [`molrs_region_not`] build a new region from existing ones, so the outside
//! of a shape is its complement and a shell is `outer AND NOT inner`.
//!
//! # Handles, not pointers
//!
//! A region is internally `Arc<dyn Region>`, a trait object. That never
//! crosses this boundary: the handle registry keeps a `molrs_ffi::RegionRef` — the same
//! handle type the Python capsule carries — and C receives the usual two-word
//! [`MolrsRegionHandle`]. A composed region is an ordinary handle, so
//! compositions nest without a special case.
//!
//! Every handle must be released with [`molrs_region_drop`]. Dropping a
//! composed region does not disturb its operands: they are shared, and each
//! handle owns its own reference.
//!
//! All lengths are in Angstrom.

use std::sync::Arc;

use molrs::core::TriMesh;
use molrs::core::{
    AndRegion, Cuboid, Cylinder, Ellipsoid, HalfSpace, NotRegion, OrRegion, Parallelepiped,
    Polyhedron, Region, Sphere, SphereUnion,
};
use molrs::op::{F3, Fnx3};
use molrs_ffi::RegionRef;
use ndarray::{Array2, ArrayView2};

use crate::error::{self, MolrsStatus};
use crate::handle::{
    MolrsBoxHandle, MolrsRegionHandle, handle_to_box_key, handle_to_region_key,
    region_key_to_handle,
};
use crate::handle_registry::lock_registry;
use crate::simbox::{MolrsCellKind, cell_from_c};
use crate::{ffi_try, null_check};
use molrs::op::F;

/// Insert a freshly built region and hand back its handle.
fn publish(region: Arc<dyn Region + Send + Sync>, out: *mut MolrsRegionHandle) -> MolrsStatus {
    let mut registry = lock_registry();
    let key = registry.regions.insert(RegionRef::new(region));
    unsafe { *out = region_key_to_handle(key) };
    MolrsStatus::Ok
}

/// Run `f` against a live region, or report a stale handle.
fn with_region<R>(
    handle: MolrsRegionHandle,
    f: impl FnOnce(&dyn Region) -> R,
) -> Result<R, MolrsStatus> {
    let registry = lock_registry();
    match registry.regions.get(handle_to_region_key(handle)) {
        Some(r) => Ok(r.with_region(f)),
        None => {
            error::set_last_error("stale or unknown region handle");
            Err(MolrsStatus::InvalidRegionHandle)
        }
    }
}

fn region_arc(handle: MolrsRegionHandle) -> Result<Arc<dyn Region + Send + Sync>, MolrsStatus> {
    let registry = lock_registry();
    match registry.regions.get(handle_to_region_key(handle)) {
        Some(r) => Ok(r.region()),
        None => {
            error::set_last_error("stale or unknown region handle");
            Err(MolrsStatus::InvalidRegionHandle)
        }
    }
}

fn triple(ptr: *const F) -> [F; 3] {
    let s = unsafe { std::slice::from_raw_parts(ptr, 3) };
    [s[0], s[1], s[2]]
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

/// Ball of `radius` about `center3`.
///
/// ```c
/// MolrsStatus molrs_region_sphere(const F center3[3],
///                                 F radius,
///                                 MolrsRegionHandle* out);
/// ```
///
/// # Safety
/// `center3` must point to 3 readable floats; `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_sphere(
    center3: *const F,
    radius: F,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(center3);
        null_check!(out);
        let c = triple(center3);
        publish(Arc::new(Sphere::new(F3::from_vec(c.to_vec()), radius)), out)
    })
}

/// Axis-aligned box with a corner at `origin3` and edge `lengths3`.
///
/// # Safety
/// `origin3` and `lengths3` must each point to 3 readable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_cuboid(
    origin3: *const F,
    lengths3: *const F,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(origin3);
        null_check!(lengths3);
        null_check!(out);
        let o = triple(origin3);
        let l = triple(lengths3);
        publish(
            Arc::new(Cuboid::new(
                F3::from_vec(o.to_vec()),
                F3::from_vec(l.to_vec()),
            )),
            out,
        )
    })
}

/// The parallelepiped a cell spans from the corner `origin3` — the C face
/// of molrs's `Parallelepiped::new(cell, origin)`. The cell is stated as
/// [`molrs_box_new`](crate::molrs_box_new) takes one: a [`MolrsCellKind`]
/// and the values it reads (the edges are the columns of a `MATRIX` cell).
///
/// # Errors
/// `InvalidArgument` for `MOLRS_CELL_KIND_UNDEFINED` (no geometric cell), a
/// non-positive length, an impossible angle, or a singular or non-finite
/// cell matrix; `NullPointer` when a kind that reads values gets a null
/// `cell`.
///
/// # Safety
/// `cell` must point to the number of readable `F` values `kind` names,
/// `origin3` to 3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_parallelepiped(
    kind: MolrsCellKind,
    cell: *const F,
    origin3: *const F,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(origin3);
        null_check!(out);
        let cell = match unsafe { cell_from_c(kind, cell) } {
            Ok(c) => c,
            Err(status) => return status,
        };
        let o = triple(origin3);
        match Parallelepiped::new(cell, F3::from_vec(o.to_vec())) {
            Ok(r) => publish(Arc::new(r), out),
            Err(e) => {
                error::set_last_error(e);
                MolrsStatus::InvalidArgument
            }
        }
    })
}

/// Everything on the `normal3` side of the plane through `point3`.
///
/// # Safety
/// `normal3` and `point3` must each point to 3 readable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_half_space(
    normal3: *const F,
    point3: *const F,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(normal3);
        null_check!(point3);
        null_check!(out);
        match HalfSpace::new(triple(normal3), triple(point3)) {
            Ok(r) => publish(Arc::new(r), out),
            Err(e) => {
                error::set_last_error(e);
                MolrsStatus::InvalidArgument
            }
        }
    })
}

/// Finite cylinder of `radius` and `length` from `base3` along `axis3`.
///
/// # Safety
/// `base3` and `axis3` must each point to 3 readable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_cylinder(
    base3: *const F,
    axis3: *const F,
    radius: F,
    length: F,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(base3);
        null_check!(axis3);
        null_check!(out);
        match Cylinder::new(triple(base3), triple(axis3), radius, length) {
            Ok(r) => publish(Arc::new(r), out),
            Err(e) => {
                error::set_last_error(e);
                MolrsStatus::InvalidArgument
            }
        }
    })
}

/// Ellipsoid about `center3` with `semi_axes3`.
///
/// # Safety
/// `center3` and `semi_axes3` must each point to 3 readable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_ellipsoid(
    center3: *const F,
    semi_axes3: *const F,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(center3);
        null_check!(semi_axes3);
        null_check!(out);
        match Ellipsoid::new(triple(center3), triple(semi_axes3)) {
            Ok(r) => publish(Arc::new(r), out),
            Err(e) => {
                error::set_last_error(e);
                MolrsStatus::InvalidArgument
            }
        }
    })
}

/// Solid bounded by a watertight triangle mesh.
///
/// `vertices` is `n_vertices * 3` floats and `faces` is `n_faces * 3` vertex
/// indices, the same flat layout `molrs_region_polyhedron` callers already use
/// for STL data.
///
/// # Safety
/// `vertices` must hold `n_vertices * 3` floats and `faces` `n_faces * 3`
/// `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_polyhedron(
    vertices: *const F,
    n_vertices: usize,
    faces: *const u32,
    n_faces: usize,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(vertices);
        null_check!(faces);
        null_check!(out);
        let v = unsafe { std::slice::from_raw_parts(vertices, n_vertices * 3) };
        let f = unsafe { std::slice::from_raw_parts(faces, n_faces * 3) };
        let verts: Vec<[F; 3]> = v.as_chunks::<3>().0.to_vec();
        let tris: Vec<[u32; 3]> = f.as_chunks::<3>().0.to_vec();
        let mesh = match TriMesh::from_indexed(verts, tris) {
            Ok(m) => m,
            Err(face) => {
                error::set_last_error(format!("face {face} indexes a vertex that is not there"));
                return MolrsStatus::InvalidArgument;
            }
        };
        match Polyhedron::new(mesh) {
            Ok(r) => publish(Arc::new(r), out),
            Err(e) => {
                error::set_last_error(e.to_string());
                MolrsStatus::InvalidArgument
            }
        }
    })
}

/// Union of one sphere per site, under the minimum image of `bx`.
///
/// # Safety
/// `centers` must hold `n * 3` floats and `radii` `n` floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_sphere_union(
    centers: *const F,
    radii: *const F,
    n: usize,
    bx: MolrsBoxHandle,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(centers);
        null_check!(radii);
        null_check!(out);
        let c = unsafe { std::slice::from_raw_parts(centers, n * 3) };
        let r = unsafe { std::slice::from_raw_parts(radii, n) };
        let centers_arr = match Array2::from_shape_vec((n, 3), c.to_vec()) {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(format!("centers: {e}"));
                return MolrsStatus::InvalidArgument;
            }
        };
        let registry = lock_registry();
        let Some(simbox) = registry.simboxes.get(handle_to_box_key(bx)) else {
            error::set_last_error("stale or unknown box handle");
            return MolrsStatus::InvalidBoxHandle;
        };
        let built = SphereUnion::new(centers_arr.view(), r, simbox);
        drop(registry);
        match built {
            Ok(u) => publish(Arc::new(u), out),
            Err(e) => {
                error::set_last_error(e.to_string());
                MolrsStatus::InvalidArgument
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

type DynRegion = Arc<dyn Region + Send + Sync>;

/// Publish `compose(a, b)` into `out`. The two `extern "C"` doors below are
/// spelled out as functions, not generated by a macro, so cbindgen declares
/// them in `molrs.h`.
fn binary_op(
    a: MolrsRegionHandle,
    b: MolrsRegionHandle,
    out: *mut MolrsRegionHandle,
    compose: fn(DynRegion, DynRegion) -> DynRegion,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let (ra, rb) = match (region_arc(a), region_arc(b)) {
            (Ok(ra), Ok(rb)) => (ra, rb),
            (Err(s), _) | (_, Err(s)) => return s,
        };
        publish(compose(ra, rb), out)
    })
}

/// Region that is the intersection of `a` and `b`.
///
/// The operands keep their own handles and stay usable.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_and(
    a: MolrsRegionHandle,
    b: MolrsRegionHandle,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    binary_op(a, b, out, |ra, rb| Arc::new(AndRegion::new(ra, rb)))
}

/// Region that is the union of `a` and `b`.
///
/// The operands keep their own handles and stay usable.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_or(
    a: MolrsRegionHandle,
    b: MolrsRegionHandle,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    binary_op(a, b, out, |ra, rb| Arc::new(OrRegion::new(ra, rb)))
}

/// Region that is everything `a` is not.
///
/// # Safety
/// `out` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_not(
    a: MolrsRegionHandle,
    out: *mut MolrsRegionHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let ra = match region_arc(a) {
            Ok(r) => r,
            Err(s) => return s,
        };
        publish(Arc::new(NotRegion::new(ra)), out)
    })
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Signed distance of each of `n` points to the boundary.
///
/// `points` is flat `[x, y, z, …]`; `out` receives `n` floats.
///
/// # Safety
/// `points` must hold `n * 3` readable floats and `out` `n` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_distance(
    region: MolrsRegionHandle,
    points: *const F,
    n: usize,
    out: *mut F,
) -> MolrsStatus {
    ffi_try!({
        null_check!(points);
        null_check!(out);
        let p = unsafe { std::slice::from_raw_parts(points, n * 3) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out, n) };
        match with_region(region, |r| {
            for (i, slot) in dst.iter_mut().enumerate() {
                *slot = r.distance(&[p[3 * i], p[3 * i + 1], p[3 * i + 2]]);
            }
        }) {
            Ok(()) => MolrsStatus::Ok,
            Err(s) => s,
        }
    })
}

/// `true` for each of `n` points inside the solid.
///
/// # Safety
/// `points` must hold `n * 3` readable floats and `out` `n` writable bools.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_contains(
    region: MolrsRegionHandle,
    points: *const F,
    n: usize,
    out: *mut bool,
) -> MolrsStatus {
    ffi_try!({
        null_check!(points);
        null_check!(out);
        let p = unsafe { std::slice::from_raw_parts(points, n * 3) };
        let dst = unsafe { std::slice::from_raw_parts_mut(out, n) };
        let pts: Fnx3 = match Array2::from_shape_vec((n, 3), p.to_vec()) {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(format!("points: {e}"));
                return MolrsStatus::InvalidArgument;
            }
        };
        let view: ArrayView2<'_, F> = pts.view();
        match with_region(region, |r| {
            let hits = r.contains(&view.to_owned());
            for (slot, &hit) in dst.iter_mut().zip(hits.iter()) {
                *slot = hit;
            }
        }) {
            Ok(()) => MolrsStatus::Ok,
            Err(s) => s,
        }
    })
}

/// Axis-aligned bounds as `[xmin, xmax, ymin, ymax, zmin, zmax]`.
///
/// # Safety
/// `out6` must point to 6 writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_bounds(
    region: MolrsRegionHandle,
    out6: *mut F,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out6);
        let dst = unsafe { std::slice::from_raw_parts_mut(out6, 6) };
        match with_region(region, |r| {
            for (slot, v) in dst.iter_mut().zip(r.bounds().iter().copied()) {
                *slot = v;
            }
        }) {
            Ok(()) => MolrsStatus::Ok,
            Err(s) => s,
        }
    })
}

/// Release a region handle. Operands of a composition are unaffected.
///
/// # Safety
/// The handle must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_region_drop(region: MolrsRegionHandle) -> MolrsStatus {
    ffi_try!({
        let mut registry = lock_registry();
        match registry.regions.remove(handle_to_region_key(region)) {
            Some(_) => MolrsStatus::Ok,
            None => {
                error::set_last_error("stale or unknown region handle");
                MolrsStatus::InvalidRegionHandle
            }
        }
    })
}
