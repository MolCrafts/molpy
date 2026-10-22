//! Triangle-inequality bounds smoothing.
//!
//! Direct port of RDKit's `DistGeom::triangleSmoothBounds`
//! (`$RDBASE/Code/DistGeom/TriangleSmooth.cpp`, BSD-3, Copyright (C)
//! 2004-2025 Greg Landrum and other RDKit contributors), itself the O(N³)
//! algorithm from Crippen & Havel, *Distance Geometry and Molecular
//! Conformation* (1988), pp. 252-253.
//!
//! RDKit does **not** apply tetrangle (four-point) smoothing in the default
//! `GetMoleculeBoundsMatrix` path — only the triangle pass — so to stay
//! byte-comparable we port the triangle pass exactly and document that
//! tetrangle is intentionally omitted (it is not part of the reference
//! matrix this port is validated against).

use molrs::core::MolRsError;

use super::bounds_matrix::BoundsMatrix;

/// Triangle-smooth `bounds` in place (RDKit `triangleSmoothBounds`, `tol`).
///
/// Returns `Err` when the bounds are inconsistent (a lower bound exceeds the
/// corresponding upper bound by more than `tol`), exactly like RDKit's
/// boolean return mapped onto our error type.
pub fn smooth_bounds_tol(bounds: &mut BoundsMatrix, tol: f64) -> Result<(), MolRsError> {
    let npt = bounds.len();
    for k in 0..npt {
        for i in 0..npt.saturating_sub(1) {
            if i == k {
                continue;
            }
            let (ii, ik) = if i > k { (k, i) } else { (i, k) };
            let u_ik = bounds.raw(ii, ik); // upper(i,k)
            let l_ik = bounds.raw(ik, ii); // lower(i,k)
            for j in (i + 1)..npt {
                if j == k {
                    continue;
                }
                let (jj, jk) = if j > k { (k, j) } else { (j, k) };
                let u_kj = bounds.raw(jj, jk); // upper(j,k)
                let sum_u = u_ik + u_kj;
                if bounds.raw(i, j) > sum_u {
                    bounds.set_raw(i, j, sum_u);
                }

                let diff_lik_ukj = l_ik - u_kj;
                let diff_ljk_uik = bounds.raw(jk, jj) - u_ik; // lower(j,k) - upper(i,k)
                if bounds.raw(j, i) < diff_lik_ukj {
                    bounds.set_raw(j, i, diff_lik_ukj);
                } else if bounds.raw(j, i) < diff_ljk_uik {
                    bounds.set_raw(j, i, diff_ljk_uik);
                }

                let l_bound = bounds.raw(j, i); // lower(i,j)
                let u_bound = bounds.raw(i, j); // upper(i,j)
                if tol > 0.0
                    && (l_bound - u_bound) / l_bound > 0.0
                    && (l_bound - u_bound) / l_bound < tol
                {
                    bounds.set_raw(i, j, l_bound);
                } else if l_bound - u_bound > 0.0 {
                    return Err(MolRsError::validation(
                        "distance-geometry bounds are inconsistent (lower > upper) during triangle smoothing",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Triangle-smooth `bounds` with `tol = 0` (the `GetMoleculeBoundsMatrix`
/// default).
pub fn smooth_bounds(bounds: &mut BoundsMatrix) -> Result<(), MolRsError> {
    smooth_bounds_tol(bounds, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three points: (0,1) and (1,2) tightly bounded, (0,2) wide open.
    fn open_triangle() -> BoundsMatrix {
        let mut b = BoundsMatrix::new(3, 0.0);
        b.set_lower(0, 1, 1.0);
        b.set_upper(0, 1, 1.0);
        b.set_lower(1, 2, 2.0);
        b.set_upper(1, 2, 2.0);
        b.set_lower(0, 2, 0.0);
        b.set_upper(0, 2, 100.0);
        b
    }

    #[test]
    fn the_triangle_inequality_closes_the_open_pair() {
        let mut b = open_triangle();
        smooth_bounds(&mut b).unwrap();
        // upper(0,2) ≤ upper(0,1) + upper(1,2); lower(0,2) ≥ lower(1,2) − upper(0,1).
        assert_eq!(b.upper(0, 2), 3.0);
        assert_eq!(b.lower(0, 2), 1.0);
        // The tight pairs are untouched.
        assert_eq!((b.lower(0, 1), b.upper(0, 1)), (1.0, 1.0));
        assert_eq!((b.lower(1, 2), b.upper(1, 2)), (2.0, 2.0));
    }

    #[test]
    fn inconsistent_bounds_are_an_error() {
        let mut b = open_triangle();
        // (0,2) must be at least 5 apart, but the path 0-1-2 spans at most 3.
        b.set_lower(0, 2, 5.0);
        assert!(smooth_bounds(&mut b).is_err());
    }

    #[test]
    fn a_small_inversion_inside_the_tolerance_is_snapped_not_refused() {
        let mut b = open_triangle();
        b.set_lower(0, 2, 3.01);
        assert!(smooth_bounds_tol(&mut b, 0.01).is_ok());
        assert_eq!(b.upper(0, 2), b.lower(0, 2));
    }
}
