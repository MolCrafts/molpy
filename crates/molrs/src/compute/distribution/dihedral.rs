//! Dihedral observable: the IUPAC-signed torsion φ ∈ (−π, π] of atom quadruples.

use molrs::op::F;

use crate::compute::ComputeError;

use super::observable::{Observable, check_tuples, displacement};
use crate::op::vec3::{cross, dot, norm};
use molrs::core::Mic;
use ndarray::ArrayView2;

/// Signed dihedral φ ∈ (−π, π] (radians) over each quadruple i–j–k–l (arity 4).
///
/// Ported from the reference implementation `Dihedral(vec1, vec2, norm, absolute)` in
/// `src/xdvector3.cpp`: reference implementation projects the two outer bond vectors onto the
/// plane perpendicular to the central bond and takes their angle, flipping the
/// sign by the half-plane test `|angle(p1, t2)| > 90°`. The algebraically
/// equivalent `atan2` form used here reproduces the same signed value while
/// staying branch-free and NaN-safe:
///
/// `φ = atan2((b1 × b2)·(b2/|b2|), (b1 × b2)·(b2 × b3))` with
/// `b1 = r_j − r_i`, `b2 = r_k − r_j`, `b3 = r_l − r_k`.
///
/// This is the Blondel–Karplus convention; φ = 0 for a cis/eclipsed (planar)
/// arrangement and ±π for trans, matching the reference implementation's signed DDF output.
#[derive(Debug, Clone, Default)]
pub struct DihedralObservable;

impl Observable for DihedralObservable {
    fn arity(&self) -> usize {
        4
    }

    fn is_angular(&self) -> bool {
        // Signed torsion spans (−π, π]; the sin θ solid-angle correction is an
        // ADF convention and does not apply to a signed dihedral.
        false
    }

    fn natural_range(&self) -> Option<(F, F)> {
        Some((-std::f64::consts::PI, std::f64::consts::PI))
    }

    fn sample_into(
        &self,
        positions: ArrayView2<'_, F>,
        mic: &Mic,
        tuples: ArrayView2<'_, usize>,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError> {
        check_tuples(self.arity(), positions, tuples)?;
        out.clear();
        out.reserve(tuples.nrows());
        for (g, t) in tuples.outer_iter().enumerate() {
            let (i, j, k, l) = (t[0], t[1], t[2], t[3]);
            let b1 = displacement(mic, positions, i, j); // j - i
            let b2 = displacement(mic, positions, j, k); // k - j
            let b3 = displacement(mic, positions, k, l); // l - k
            let n2 = norm(b2);
            if n2 == 0.0 {
                return Err(ComputeError::NonFinite {
                    where_: "DihedralObservable: zero-length central bond",
                    index: g,
                });
            }
            let c12 = cross(b1, b2);
            let c23 = cross(b2, b3);
            // atan2(|b2| * b1 · (b2×b3), (b1×b2) · (b2×b3))
            let x = dot(c12, c23);
            let y = dot([b1[0] * n2, b1[1] * n2, b1[2] * n2], c23);
            if x == 0.0 && y == 0.0 {
                // Degenerate (collinear) — define φ = 0 rather than NaN.
                out.push(0.0);
            } else {
                out.push(y.atan2(x));
            }
        }
        Ok(())
    }
}
