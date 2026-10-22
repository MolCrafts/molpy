//! Shared geometry helpers for potential kernels.
//!
//! Provides the flat-index [`sub3`] adapter, the flat-array angle/dihedral
//! entry points, and Cartesian force projection routines used by multiple
//! kernel families. The vector arithmetic and the internal coordinates
//! themselves are [`crate::op::vec3`].

use crate::op::vec3::{self, cross, dot, norm};
use molrs::op::F;

// ---------------------------------------------------------------------------
// Flat-index adapter
// ---------------------------------------------------------------------------

/// Compute the vector from atom `bi` to atom `ai`: `a[ai] - b[bi]`.
#[inline]
pub fn sub3(a: &[F], ai: usize, b: &[F], bi: usize) -> [F; 3] {
    [
        a[ai * 3] - b[bi * 3],
        a[ai * 3 + 1] - b[bi * 3 + 1],
        a[ai * 3 + 2] - b[bi * 3 + 2],
    ]
}

/// Atom `i`'s position from a flat `[x0, y0, z0, x1, …]` array.
#[inline]
fn point(coords: &[F], i: usize) -> [F; 3] {
    [coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]]
}

/// Validate that `coords` length is a multiple of 3 and return atom count.
#[inline]
pub fn validate_coords(coords: &[F]) -> usize {
    assert!(
        coords.len().is_multiple_of(3),
        "coords length must be multiple of 3, got {}",
        coords.len()
    );
    coords.len() / 3
}

// ---------------------------------------------------------------------------
// Angle computation
// ---------------------------------------------------------------------------

/// Compute angle i-j-k in radians: [`op::vec3::angle`](crate::op::vec3::angle)
/// over the flat coordinate array.
pub fn compute_angle(coords: &[F], i: usize, j: usize, k: usize) -> F {
    vec3::angle(point(coords, i), point(coords, j), point(coords, k))
}

/// Project dE/dθ into Cartesian forces for angle i-j-k.
pub fn accumulate_angle_forces(
    coords: &[F],
    i: usize,
    j: usize,
    k: usize,
    de_dth: F,
    forces: &mut [F],
) {
    let rji = sub3(coords, i, coords, j);
    let rjk = sub3(coords, k, coords, j);
    let d_ji = norm(rji);
    let d_jk = norm(rjk);
    if d_ji < 1e-12 as F || d_jk < 1e-12 as F {
        return;
    }
    let cos_theta = (dot(rji, rjk) / (d_ji * d_jk)).clamp(-1.0, 1.0);
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-12 as F);
    let prefactor = de_dth / sin_theta;

    for dim in 0..3 {
        let dc_di = rjk[dim] / (d_ji * d_jk) - cos_theta * rji[dim] / (d_ji * d_ji);
        let dc_dk = rji[dim] / (d_ji * d_jk) - cos_theta * rjk[dim] / (d_jk * d_jk);
        let dc_dj = -dc_di - dc_dk;
        forces[i * 3 + dim] += prefactor * dc_di;
        forces[k * 3 + dim] += prefactor * dc_dk;
        forces[j * 3 + dim] += prefactor * dc_dj;
    }
}

// ---------------------------------------------------------------------------
// Dihedral computation
// ---------------------------------------------------------------------------

/// Compute dihedral angle i-j-k-l in radians, in `(−π, π]`:
/// [`op::vec3::dihedral`](crate::op::vec3::dihedral) over the flat
/// coordinate array.
pub fn compute_dihedral(coords: &[F], i: usize, j: usize, k: usize, l: usize) -> F {
    vec3::dihedral(
        point(coords, i),
        point(coords, j),
        point(coords, k),
        point(coords, l),
    )
}

/// Project dE/dφ into Cartesian forces for dihedral i-j-k-l (Blondel-Karplus method).
pub fn accumulate_dihedral_forces(
    coords: &[F],
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    de_dphi: F,
    forces: &mut [F],
) {
    let b1 = sub3(coords, j, coords, i);
    let b2 = sub3(coords, k, coords, j);
    let b3 = sub3(coords, l, coords, k);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let n1_sq = dot(n1, n1);
    let n2_sq = dot(n2, n2);
    let b2_mag = norm(b2);

    if n1_sq < 1e-24 as F || n2_sq < 1e-24 as F || b2_mag < 1e-12 as F {
        return;
    }

    // Force on the end atoms for the standard signed-dihedral convention used
    // by `compute_dihedral` (φ = atan2(|b2|·(b1·n2), n1·n2)):
    //   F_i = +dE/dφ·|b2|/|n1|²·n1,  F_l = −dE/dφ·|b2|/|n2|²·n2.
    let fi = [
        de_dphi * b2_mag / n1_sq * n1[0],
        de_dphi * b2_mag / n1_sq * n1[1],
        de_dphi * b2_mag / n1_sq * n1[2],
    ];
    let fl = [
        -de_dphi * b2_mag / n2_sq * n2[0],
        -de_dphi * b2_mag / n2_sq * n2[1],
        -de_dphi * b2_mag / n2_sq * n2[2],
    ];

    let p_ij = dot(b1, b2) / (b2_mag * b2_mag);
    let p_kl = dot(b3, b2) / (b2_mag * b2_mag);

    for dim in 0..3 {
        // Blondel-Karplus / GROMACS `do_dih_fup` middle-atom redistribution.
        // With b1 = r_j−r_i, b2 = r_k−r_j, b3 = r_l−r_k:
        //   F_j = −F_i − p_ij·F_i + p_kl·F_l,  F_k = −F_l + p_ij·F_i − p_kl·F_l.
        let fj = -fi[dim] - p_ij * fi[dim] + p_kl * fl[dim];
        let fk = -fl[dim] + p_ij * fi[dim] - p_kl * fl[dim];
        forces[i * 3 + dim] += fi[dim];
        forces[j * 3 + dim] += fj;
        forces[k * 3 + dim] += fk;
        forces[l * 3 + dim] += fl[dim];
    }
}

// ---------------------------------------------------------------------------
// Term index tables
// ---------------------------------------------------------------------------

/// Stack a kernel's per-term index columns into the `(n_terms, arity)` table
/// [`IndexedTerms::terms`](crate::ff::potential::IndexedTerms::terms) returns.
///
/// Every bonded kernel keeps its indices as one `Vec<usize>` per position —
/// `atom_i`, `atom_j`, … — because that is the layout its inner loop wants.
/// The table is the layout a *caller* wants: one row per term, so a row can be
/// rewritten to name a periodic copy without the caller knowing which column
/// belongs to which position.
pub fn term_table(columns: &[&[usize]]) -> ndarray::Array2<u32> {
    let arity = columns.len();
    let n_terms = columns.first().map_or(0, |c| c.len());
    debug_assert!(
        columns.iter().all(|c| c.len() == n_terms),
        "a term's index columns must all have one entry per term"
    );
    let mut out = ndarray::Array2::<u32>::zeros((n_terms, arity));
    for (a, col) in columns.iter().enumerate() {
        for (t, &v) in col.iter().enumerate() {
            out[[t, a]] = v as u32;
        }
    }
    out
}
