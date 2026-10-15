//! The one group reduction: per-group anchor + minimum-image unwrap, weighted
//! centroid and weighted second moment.
//!
//! Every per-group shape quantity — [`CenterOfMass`](crate::compute::CenterOfMass),
//! [`GyrationTensor`](crate::compute::GyrationTensor),
//! [`InertiaTensor`](crate::compute::InertiaTensor) — is one of these two
//! moments, so the unwrap and the accumulation exist once, here.

use molrs::core::{Mic, SimBox};
use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::compute::{ComputeError, Partition};
use crate::op::vec3::sub;

/// The items of one snapshot sorted into the groups of a [`Partition`], each
/// item held as its minimum-image offset from its group's anchor.
///
/// A group's **anchor** is its first member (lowest item index). Each member
/// is stored as `mic(r − anchor)`, the core minimum image of
/// [`SimBox::mic`], so a group that straddles a periodic boundary is
/// reassembled in one piece before any moment is taken — provided every member
/// lies within half a box of the anchor. Without a box no axis wraps.
///
/// Groups are ordered by ascending key ([`Partition::sizes`]); items in no
/// group are skipped.
#[derive(Debug, Clone)]
pub(crate) struct GroupReduce {
    /// Group keys, ascending.
    keys: Vec<u64>,
    /// Per item: the index of its group in `keys`, `None` when ungrouped.
    group: Vec<Option<usize>>,
    /// Per group: the anchor position.
    anchors: Vec<[F; 3]>,
    /// Per item: minimum-image offset from its group's anchor (zero when
    /// ungrouped).
    offsets: Vec<[F; 3]>,
}

/// Weighted centroids of every group of a [`GroupReduce`].
#[derive(Debug, Clone)]
pub(crate) struct Centroids {
    /// `G × 3`: `anchor + Σ wᵢ dᵢ / Σ wᵢ` per group, not wrapped back into
    /// the box.
    pub(crate) centers: Array2<F>,
    /// Total weight `Σ wᵢ` per group.
    pub(crate) totals: Array1<F>,
    /// Per group: `centre − anchor`, the offset every member's second moment
    /// is taken about.
    shifts: Vec<[F; 3]>,
}

impl GroupReduce {
    /// Sort the `N × 3` `positions` into the groups of `partition`, unwrapping
    /// each group about its anchor under `simbox`'s minimum image.
    ///
    /// # Errors
    /// [`ComputeError::DimensionMismatch`] when `positions` is not `N × 3` or
    /// the partition does not cover exactly its `N` items;
    /// [`ComputeError::NonFinite`] for a non-finite position.
    pub(crate) fn new(
        positions: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
        partition: &dyn Partition,
    ) -> Result<Self, ComputeError> {
        if positions.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: positions.ncols(),
                what: "components per position (expected N × 3)",
            });
        }
        let n = positions.nrows();
        if partition.n_items() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: partition.n_items(),
                what: "partition items against the rows of positions",
            });
        }
        if let Some(index) = positions.iter().position(|x| !x.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "positions",
                index,
            });
        }

        let keys: Vec<u64> = partition.sizes().into_keys().collect();
        let mic = simbox.map_or(Mic::Free, SimBox::mic);
        let mut anchors: Vec<Option<[F; 3]>> = vec![None; keys.len()];
        let mut group = Vec::with_capacity(n);
        let mut offsets = Vec::with_capacity(n);
        for (item, row) in positions.rows().into_iter().enumerate() {
            let slot = partition
                .group_of(item)
                .map(|key| keys.binary_search(&key).expect("sizes() lists every key"));
            let offset = match slot {
                None => [0.0; 3],
                Some(g) => {
                    let r = [row[0], row[1], row[2]];
                    let anchor = *anchors[g].get_or_insert(r);
                    mic.apply(sub(r, anchor))
                }
            };
            group.push(slot);
            offsets.push(offset);
        }
        let anchors = anchors
            .into_iter()
            .map(|a| a.expect("every listed key has a member"))
            .collect();
        Ok(Self {
            keys,
            group,
            anchors,
            offsets,
        })
    }

    /// The group keys, ascending.
    pub(crate) fn keys(&self) -> &[u64] {
        &self.keys
    }

    /// The row of `item`'s group in [`keys`](Self::keys); `None` when the item
    /// is in no group.
    #[cfg(feature = "voronoi")]
    pub(crate) fn group_of(&self, item: usize) -> Option<usize> {
        self.group[item]
    }

    /// The weighted centroid of every group. `name` labels the weights in an
    /// error (`"masses"`, `"weights"`).
    ///
    /// # Errors
    /// [`ComputeError::DimensionMismatch`] unless there is one weight per
    /// item; [`ComputeError::NonFinite`] / [`ComputeError::OutOfRange`] for a
    /// non-finite or negative weight; [`ComputeError::OutOfRange`] for a group
    /// whose total weight is zero (its centroid is undefined).
    pub(crate) fn centroids(
        &self,
        weights: ArrayView1<'_, F>,
        name: &'static str,
    ) -> Result<Centroids, ComputeError> {
        if weights.len() != self.group.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: self.group.len(),
                got: weights.len(),
                what: "weights against the rows of positions",
            });
        }
        if let Some(index) = weights.iter().position(|w| !w.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: name,
                index,
            });
        }
        if let Some(index) = weights.iter().position(|&w| w < 0.0) {
            return Err(ComputeError::OutOfRange {
                field: name,
                value: format!("item {index}: {} is negative", weights[index]),
            });
        }

        let n_groups = self.keys.len();
        let mut sums = vec![[0.0; 3]; n_groups];
        let mut totals = Array1::<F>::zeros(n_groups);
        for ((slot, d), &w) in self.group.iter().zip(&self.offsets).zip(weights) {
            let Some(g) = *slot else { continue };
            for k in 0..3 {
                sums[g][k] += w * d[k];
            }
            totals[g] += w;
        }

        let mut centers = Array2::<F>::zeros((n_groups, 3));
        let mut shifts = Vec::with_capacity(n_groups);
        for (g, sum) in sums.iter().enumerate() {
            let total = totals[g];
            if total == 0.0 {
                return Err(ComputeError::OutOfRange {
                    field: name,
                    value: format!(
                        "group {}: total 0 — the centroid of a weightless group is undefined",
                        self.keys[g]
                    ),
                });
            }
            let shift = sum.map(|s| s / total);
            for k in 0..3 {
                centers[[g, k]] = self.anchors[g][k] + shift[k];
            }
            shifts.push(shift);
        }
        Ok(Centroids {
            centers,
            totals,
            shifts,
        })
    }

    /// The weighted second moment `S_g = Σᵢ wᵢ sᵢ sᵢᵀ` of every group about
    /// its centroid, `sᵢ = dᵢ − (centre − anchor)` (unwrapped, so no second
    /// minimum image), unnormalised, row-major `3 × 3`. `weights` must be the
    /// ones `centroids` was computed with (validated there).
    pub(crate) fn second_moments(
        &self,
        weights: ArrayView1<'_, F>,
        centroids: &Centroids,
    ) -> Vec<[[F; 3]; 3]> {
        let mut moments = vec![[[0.0; 3]; 3]; self.keys.len()];
        for ((slot, d), &w) in self.group.iter().zip(&self.offsets).zip(weights) {
            let Some(g) = *slot else { continue };
            let shift = centroids.shifts[g];
            let s = [d[0] - shift[0], d[1] - shift[1], d[2] - shift[2]];
            for (a, row) in moments[g].iter_mut().enumerate() {
                for (b, value) in row.iter_mut().enumerate() {
                    *value += w * s[a] * s[b];
                }
            }
        }
        moments
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::{Array1, arr1, arr2};

    fn cube(len: F, pbc: [bool; 3]) -> SimBox {
        SimBox::new(Cell::Lengths([len; 3]), [0.0, 0.0, 0.0], pbc).unwrap()
    }

    /// Spec testing strategy (ac-008): a group straddling the periodic
    /// boundary is reassembled about its anchor. Members at x = 9.8, 0.1, 0.3
    /// of a 10 Å box are at offsets 0, 0.3, 0.5 from the anchor, so the
    /// uniform centroid is 9.8 + 0.8/3 and the x second moment is
    /// Σ (dᵢ − 0.8/3)² — not the 9.7 Å spread of the wrapped coordinates.
    #[test]
    fn a_group_straddling_the_boundary_is_unwrapped_about_its_anchor() {
        let positions = arr2(&[[9.8, 5.0, 5.0], [0.1, 5.0, 5.0], [0.3, 5.0, 5.0]]);
        let labels: Vec<u32> = vec![0, 0, 0];
        let simbox = cube(10.0, [true; 3]);
        let reduce = GroupReduce::new(positions.view(), Some(&simbox), &labels).unwrap();
        let weights = Array1::<F>::ones(3);

        let c = reduce.centroids(weights.view(), "weights").unwrap();
        let mean = 0.8 / 3.0;
        assert!((c.centers[[0, 0]] - (9.8 + mean)).abs() < 1e-12);
        assert!((c.centers[[0, 1]] - 5.0).abs() < 1e-12);
        assert_eq!(c.totals[0], 3.0);

        let s = reduce.second_moments(weights.view(), &c);
        let expected: F = [0.0, 0.3, 0.5].iter().map(|d: &F| (d - mean).powi(2)).sum();
        assert!((s[0][0][0] - expected).abs() < 1e-12);
        assert!(s[0][1][1].abs() < 1e-12);

        // Without the box the same coordinates are three far-apart points.
        let open = GroupReduce::new(positions.view(), None, &labels).unwrap();
        let wide = open.second_moments(
            weights.view(),
            &open.centroids(weights.view(), "weights").unwrap(),
        );
        assert!(wide[0][0][0] > 50.0);
    }

    /// Keys need not be contiguous; groups come out by ascending key, and an
    /// item in no group (a filtered cluster member) is skipped.
    #[test]
    fn groups_follow_ascending_keys_and_skip_ungrouped_items() {
        struct Labels(Vec<Option<u64>>);
        impl Partition for Labels {
            fn n_items(&self) -> usize {
                self.0.len()
            }
            fn group_of(&self, item: usize) -> Option<u64> {
                self.0[item]
            }
        }
        let positions = arr2(&[
            [1.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        ]);
        let partition = Labels(vec![Some(42), None, Some(42), Some(7)]);
        let reduce = GroupReduce::new(positions.view(), None, &partition).unwrap();
        assert_eq!(reduce.keys(), &[7, 42]);

        let c = reduce
            .centroids(arr1(&[1.0, 1.0, 1.0, 1.0]).view(), "weights")
            .unwrap();
        assert_eq!(c.centers[[0, 0]], 10.0);
        assert_eq!(c.centers[[1, 0]], 2.0);
        assert_eq!(c.totals.to_vec(), vec![1.0, 2.0]);
    }

    /// Spec §5: no implied unit mass any more — weights are required data,
    /// validated, and a weightless group is an error, not a zero centre.
    #[test]
    fn weights_are_validated() {
        let positions = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let labels: Vec<u32> = vec![0, 0];
        let reduce = GroupReduce::new(positions.view(), None, &labels).unwrap();

        assert!(matches!(
            reduce.centroids(arr1(&[1.0]).view(), "masses"),
            Err(ComputeError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            })
        ));
        assert!(matches!(
            reduce.centroids(arr1(&[1.0, F::NAN]).view(), "masses"),
            Err(ComputeError::NonFinite {
                where_: "masses",
                index: 1
            })
        ));
        assert!(matches!(
            reduce.centroids(arr1(&[1.0, -1.0]).view(), "masses"),
            Err(ComputeError::OutOfRange {
                field: "masses",
                ..
            })
        ));
        assert!(matches!(
            reduce.centroids(arr1(&[0.0, 0.0]).view(), "masses"),
            Err(ComputeError::OutOfRange {
                field: "masses",
                ..
            })
        ));
    }

    #[test]
    fn positions_and_partition_are_validated() {
        let labels: Vec<u32> = vec![0, 0];
        assert!(matches!(
            GroupReduce::new(arr2(&[[0.0, 0.0], [1.0, 0.0]]).view(), None, &labels),
            Err(ComputeError::DimensionMismatch { expected: 3, .. })
        ));
        assert!(matches!(
            GroupReduce::new(arr2(&[[0.0, 0.0, 0.0]]).view(), None, &labels),
            Err(ComputeError::DimensionMismatch {
                expected: 1,
                got: 2,
                ..
            })
        ));
        assert!(matches!(
            GroupReduce::new(
                arr2(&[[0.0, 0.0, 0.0], [F::INFINITY, 0.0, 0.0]]).view(),
                None,
                &labels
            ),
            Err(ComputeError::NonFinite {
                where_: "positions",
                index: 3
            })
        ));
    }
}
