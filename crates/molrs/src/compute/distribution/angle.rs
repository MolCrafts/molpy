//! Angle observable: the angle θ ∈ [0, π] at the vertex of each atom triple.

use molrs::op::F;

use crate::compute::ComputeError;

use super::observable::{Observable, check_tuples, displacement};
use crate::op::vec3::{dot, norm};
use molrs::core::Mic;
use ndarray::ArrayView2;

/// Angle θ ∈ [0, π] (radians) at atom `j` of each triple i–j–k (arity 3).
///
/// Ported from the reference implementation's angular distribution (`src/df.cpp` ADF mode; the angle
/// itself is `Angle`/`Angle_Deg` in `src/xdvector3.cpp`, here kept in radians):
/// for a triple i–j–k with `j` the vertex, `θ = arccos((r_ij · r_kj) /
/// (|r_ij| |r_kj|))`. The dot-product argument is clamped to `[-1, 1]` so a
/// collinear triple yields exactly 0 or π instead of a rounding NaN.
#[derive(Debug, Clone, Default)]
pub struct AngleObservable;

impl Observable for AngleObservable {
    fn arity(&self) -> usize {
        3
    }

    fn is_angular(&self) -> bool {
        true
    }

    fn natural_range(&self) -> Option<(F, F)> {
        Some((0.0, std::f64::consts::PI))
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
            let (i, j, k) = (t[0], t[1], t[2]);
            // r_ij = i - j, r_kj = k - j (j is the vertex).
            let r_ij = displacement(mic, positions, j, i);
            let r_kj = displacement(mic, positions, j, k);
            let n_ij = norm(r_ij);
            let n_kj = norm(r_kj);
            if n_ij == 0.0 || n_kj == 0.0 {
                return Err(ComputeError::NonFinite {
                    where_: "AngleObservable: zero-length bond vector",
                    index: g,
                });
            }
            let cos = (dot(r_ij, r_kj) / (n_ij * n_kj)).clamp(-1.0, 1.0);
            out.push(cos.acos());
        }
        Ok(())
    }
}
