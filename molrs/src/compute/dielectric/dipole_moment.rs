//! [`DipoleMoment`] — the collective (or per-group) dipole of a trajectory.

use molrs::op::F;
use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, Axis};

use crate::compute::{Compute, ComputeError, Partition};

/// The charge-weighted sum `M_g(t) = Σ_{i ∈ g} q_i r_i(t)` of every frame,
/// for the whole system or for each group of a [`Partition`].
///
/// With positions it is the dipole moment, the series the dielectric and
/// Einstein–Helfand analyses read; the positions must be **unwrapped**
/// ([`Unwrap`](crate::compute::Unwrap)), otherwise `M` jumps by `q L` whenever
/// a charge crosses a box face. With velocities in place of positions the same
/// sum is the charge current `J(t) = Σ q_i v_i`. Its time derivative
/// ([`Derivative`](crate::compute::Derivative)) is the current `Ṁ`; divided by
/// the volume, the current density of the Green–Kubo route.
///
/// A partition — a `mol_id` column, a [`Cluster`](crate::compute::Cluster)
/// result, a water / ion mask as labels — splits the sum into one series per
/// group (groups in ascending key order); items the partition leaves in no
/// group are excluded.
///
/// ```
/// use molrs::compute::{Compute, DipoleMoment, DipoleMomentInput};
/// use ndarray::{array, Array3};
///
/// // One frame, a +1 / −1 pair 2 Å apart along x.
/// let positions = Array3::from_shape_vec((1, 2, 3), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
/// let charges = array![1.0, -1.0];
/// let m = DipoleMoment::new().compute(DipoleMomentInput {
///     positions: positions.view(),
///     charges: charges.view(),
///     partition: None,
/// })?;
/// assert_eq!(m.series(0).row(0).to_vec(), vec![2.0, 0.0, 0.0]);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DipoleMoment;

/// The data a [`DipoleMoment`] sums.
#[derive(Clone, Copy)]
pub struct DipoleMomentInput<'a> {
    /// Unwrapped positions (or velocities), `T × N × 3`.
    pub positions: ArrayView3<'a, F>,
    /// One charge per item, length `N`.
    pub charges: ArrayView1<'a, F>,
    /// Groups to sum separately; `None` sums the whole system.
    pub partition: Option<&'a (dyn Partition + Sync)>,
}

/// One dipole series per group.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DipoleMomentResult {
    /// The group key of each series, ascending; empty when no partition was
    /// given (the single series is the whole system).
    pub groups: Vec<u64>,
    /// The dipoles, `T × G × 3` — directly the `T × N × D` series a
    /// [`Correlator`](crate::compute::Correlator) or [`Msd`](crate::compute::Msd)
    /// takes, with the groups as items.
    pub moments: Array3<F>,
}

impl DipoleMomentResult {
    /// The `T × 3` series of one group (column `0` for the whole system) —
    /// what [`DipoleAcf`](crate::compute::DipoleAcf) and
    /// [`StaticDielectric`](crate::compute::StaticDielectric) take.
    ///
    /// # Panics
    /// If `column` is not below the number of series.
    pub fn series(&self, column: usize) -> ArrayView2<'_, F> {
        self.moments.index_axis(Axis(1), column)
    }
}

impl DipoleMoment {
    /// The dipole sum (it has no configuration).
    pub fn new() -> Self {
        Self
    }
}

impl Compute for DipoleMoment {
    type Input<'a> = DipoleMomentInput<'a>;
    type Output = DipoleMomentResult;

    /// # Errors
    /// * [`ComputeError::DimensionMismatch`] when the positions are not
    ///   `T × N × 3`, or `charges` / the partition do not cover the `N` items.
    /// * [`ComputeError::EmptyInput`] for no frames, or a partition that puts
    ///   no item in any group.
    /// * [`ComputeError::NonFinite`] for a non-finite charge or coordinate.
    fn compute(&self, input: DipoleMomentInput<'_>) -> Result<DipoleMomentResult, ComputeError> {
        let (n_frames, n_items, n_dim) = input.positions.dim();
        if n_dim != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: n_dim,
                what: "components per position (expected T × N × 3)",
            });
        }
        if n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if input.charges.len() != n_items {
            return Err(ComputeError::DimensionMismatch {
                expected: n_items,
                got: input.charges.len(),
                what: "charges against the items of positions",
            });
        }
        if let Some(index) = input.charges.iter().position(|q| !q.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "charges",
                index,
            });
        }
        if let Some(index) = input.positions.iter().position(|x| !x.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "positions",
                index,
            });
        }

        // The output column of each item; `None` = in no group.
        let (groups, column): (Vec<u64>, Vec<Option<usize>>) = match input.partition {
            None => (Vec::new(), vec![Some(0); n_items]),
            Some(partition) => {
                if partition.n_items() != n_items {
                    return Err(ComputeError::DimensionMismatch {
                        expected: n_items,
                        got: partition.n_items(),
                        what: "partition items against the items of positions",
                    });
                }
                let keys: Vec<Option<u64>> = (0..n_items).map(|i| partition.group_of(i)).collect();
                let mut groups: Vec<u64> = keys.iter().flatten().copied().collect();
                groups.sort_unstable();
                groups.dedup();
                if groups.is_empty() {
                    return Err(ComputeError::EmptyInput);
                }
                let column = keys
                    .iter()
                    .map(|key| key.map(|k| groups.binary_search(&k).expect("key was collected")))
                    .collect();
                (groups, column)
            }
        };

        // Without a partition there is one series, the whole system (a given
        // partition with no group was refused above).
        let n_series = if groups.is_empty() { 1 } else { groups.len() };
        let mut moments = Array3::<F>::zeros((n_frames, n_series, 3));
        for t in 0..n_frames {
            for (i, col) in column.iter().enumerate() {
                let Some(g) = *col else { continue };
                let q = input.charges[i];
                for d in 0..3 {
                    moments[[t, g, d]] += q * input.positions[[t, i, d]];
                }
            }
        }
        Ok(DipoleMomentResult { groups, moments })
    }
}

crate::compute::spec::compute_spec! {
    DipoleMoment as "DipoleMoment";
    params |s| {}
    inputs {
        positions: Trajectory;
        charges: Scalars;
        partition: Partition, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{
        Derivative, DerivativeInput, LinearFit, LinearFitInput, Msd, MsdInput, Origins, Stencil,
    };
    use ndarray::{Array2, Array3, arr1, array};

    fn dipoles(
        positions: &Array3<F>,
        charges: &[F],
        partition: Option<&(dyn Partition + Sync)>,
    ) -> Result<DipoleMomentResult, ComputeError> {
        let charges = arr1(charges);
        DipoleMoment::new().compute(DipoleMomentInput {
            positions: positions.view(),
            charges: charges.view(),
            partition,
        })
    }

    fn one_frame(rows: &[[F; 3]]) -> Array3<F> {
        Array3::from_shape_fn((1, rows.len(), 3), |(_, i, d)| rows[i][d])
    }

    #[test]
    fn test_dipole_moment_two_charges() {
        let m = dipoles(
            &one_frame(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            &[1.0, -1.0],
            None,
        )
        .unwrap();
        let m = m.series(0).row(0).to_owned();
        assert!((m[0] - 2.0).abs() < 1e-10);
        assert!((m[1] - 0.0).abs() < 1e-10);
        assert!((m[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dipole_moment_zero_charge() {
        let m = dipoles(&Array3::zeros((1, 3, 3)), &[0.0, 0.0, 0.0], None).unwrap();
        assert!(m.moments.iter().map(|v| v.abs()).sum::<F>() < 1e-10);
        assert!(m.groups.is_empty());
    }

    #[test]
    fn test_dipole_moment_wrong_shape() {
        assert!(matches!(
            dipoles(&Array3::zeros((1, 3, 3)), &[1.0, 2.0], None),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            dipoles(&Array3::zeros((1, 2, 2)), &[1.0, 2.0], None),
            Err(ComputeError::DimensionMismatch { expected: 3, .. })
        ));
        assert!(matches!(
            dipoles(&Array3::zeros((0, 2, 3)), &[1.0, 2.0], None),
            Err(ComputeError::EmptyInput)
        ));
        assert!(matches!(
            dipoles(&Array3::zeros((1, 2, 3)), &[1.0, F::NAN], None),
            Err(ComputeError::NonFinite {
                where_: "charges",
                index: 1
            })
        ));
    }

    #[test]
    fn test_immutability_dipole_moment() {
        let positions = one_frame(&[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]);
        let pos_copy = positions.clone();
        dipoles(&positions, &[1.0, -1.0], None).unwrap();
        assert_eq!(positions, pos_copy);
    }

    /// The former `decompose_current`: the group sums partition the total,
    /// in ascending key order, whatever the label order.
    #[test]
    fn groups_partition_the_total() {
        let (n_frames, n_items) = (5, 4);
        let velocities = Array3::from_shape_fn((n_frames, n_items, 3), |(t, p, d)| match d {
            0 => p as F + t as F,
            1 => p as F * 2.0,
            _ => t as F * 0.5,
        });
        let charges = [1.0, -0.5, 2.0, 0.25];
        let labels: Vec<u32> = vec![7, 7, 3, 3];
        let split = dipoles(&velocities, &charges, Some(&labels)).unwrap();
        assert_eq!(split.groups, vec![3, 7]);
        let total = dipoles(&velocities, &charges, None).unwrap();
        for t in 0..n_frames {
            for d in 0..3 {
                let sum = split.moments[[t, 0, d]] + split.moments[[t, 1, d]];
                assert!((sum - total.moments[[t, 0, d]]).abs() < 1e-12);
            }
        }
        // Group 3 holds items 2 and 3 only.
        let expected_x0 = 2.0 * (2.0 + 0.0) + 0.25 * (3.0 + 0.0);
        assert!((split.moments[[0, 0, 0]] - expected_x0).abs() < 1e-12);
    }

    #[test]
    fn test_decompose_current_mask_mismatch() {
        let labels: Vec<u32> = vec![1, 0, 1];
        assert!(matches!(
            dipoles(&Array3::zeros((3, 2, 3)), &[1.0, 1.0], Some(&labels)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    /// The former `current_density`: the backward-difference dipole rate over
    /// the volume, now without the NaN first row — the stencil drops it.
    #[test]
    fn current_density_is_the_backward_derivative_over_the_volume() {
        let dm: Array2<F> = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let rate = Derivative::new(Stencil::Backward)
            .compute(DerivativeInput {
                series: dm.view(),
                dt: 1.0,
            })
            .unwrap();
        let volume = 1.0;
        let j = rate / volume;
        assert_eq!(j.dim(), (2, 3));
        assert!(j.iter().all(|v| v.is_finite()));
        assert!((j[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((j[[1, 0]] - 1.0).abs() < 1e-10);

        let constant = Array2::from_elem((3, 3), 1.0);
        let j = Derivative::new(Stencil::Backward)
            .compute(DerivativeInput {
                series: constant.view(),
                dt: 1.0,
            })
            .unwrap();
        assert!(j.iter().all(|v| v.abs() < 1e-10));
    }

    /// Acceptance ac-010: `DipoleMoment` → `Msd` → `LinearFit`, with no glue
    /// between steps — the dipole series is `Msd`'s positions as returned.
    /// One charge `q = 2` at `x(t) = √t` gives `M(t) = 2√t`, so the
    /// single-origin mean squared displacement of `M` is `4 t` (t in frames):
    /// with `dt = 0.5` the Einstein–Helfand curve is the line of slope 8.
    #[test]
    fn dipole_series_feeds_msd_then_linear_fit() {
        let positions =
            Array3::from_shape_fn(
                (11, 1, 3),
                |(t, _, d)| {
                    if d == 0 { (t as F).sqrt() } else { 0.0 }
                },
            );
        let series = dipoles(&positions, &[2.0], None).unwrap();
        let curve = Msd::new(10)
            .with_origins(Origins::First)
            .compute(MsdInput {
                positions: series.moments.view(),
                dt: 0.5,
                boxes: None,
            })
            .unwrap();
        let fit = LinearFit::new(0.0, 1.0)
            .unwrap()
            .compute(LinearFitInput {
                correlation: &curve,
                channel: 0,
            })
            .unwrap();
        assert!((fit.slope - 8.0).abs() < 1e-10, "slope {}", fit.slope);
        assert!(fit.intercept.abs() < 1e-10);
        assert!((fit.r2 - 1.0).abs() < 1e-12);
    }
}
