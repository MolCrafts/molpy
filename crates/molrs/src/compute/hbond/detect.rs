//! Per-frame geometric hydrogen-bond detection, and what a detected bond set
//! answers: its presence series and its network components.

use std::collections::{BTreeSet, HashSet};

use molrs::core::{Neighbors, QueryMode, Topology};
use molrs::op::F;
use ndarray::{Array2, ArrayView2, ArrayView3, Axis};

use super::criterion::{HBondCriterion, HBondDistanceKind};
use crate::compute::{Compute, ComputeError, Presence, require_dist_sq};
use crate::op::vec3::{dot, norm, sub};

/// A single detected D–H···A hydrogen bond (atom indices into the frame).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct HBond {
    /// Donor heavy-atom index.
    pub donor: u32,
    /// Bridging hydrogen index.
    pub hydrogen: u32,
    /// Acceptor heavy-atom index.
    pub acceptor: u32,
    /// Donor–acceptor distance r(D···A), ångström (minimum image).
    pub distance: F,
    /// D–H···A angle at the hydrogen, degrees.
    pub angle: F,
}

/// Geometric hydrogen-bond detection, frame by frame.
///
/// Configuration: the [`HBondCriterion`]. The data — the trajectory, the
/// donor `(D, H)` and acceptor selections, and one candidate table per frame
/// — goes to [`compute`](Compute::compute), which returns the bonds of every
/// frame.
///
/// The candidates come from the caller's core neighbour search: each frame's
/// table is a cross-query whose **query points are the donor sites** — the
/// donor heavy atoms for [`HBondDistanceKind::DonorAcceptor`], the hydrogens
/// for [`HBondDistanceKind::HydrogenAcceptor`], in `donors` order — and whose
/// **reference points are the acceptors**, in `acceptors` order, searched
/// with a cutoff of at least `dist_cutoff`:
///
/// ```
/// use molrs::compute::{Compute, HBondCriterion, HBonds, HBondsInput};
/// use molrs::core::{Cell, NeighborQuery, SimBox};
/// use ndarray::{Array3, array};
///
/// // One frame: O–H donor (atoms 0, 1) and an acceptor O (atom 2) in line.
/// let x = Array3::from_shape_vec(
///     (1, 3, 3),
///     vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.8, 0.0, 0.0],
/// ).unwrap();
/// let donors = [(0, 1)];
/// let acceptors = [2];
/// let simbox = SimBox::new(Cell::Lengths([20.0; 3]), [-10.0, -10.0, -10.0], [true, true, true]).unwrap();
/// let table = NeighborQuery::new(&simbox, array![[2.8, 0.0, 0.0]].view(), 3.5)
///     .unwrap()
///     .query(array![[0.0, 0.0, 0.0]].view());
///
/// let bonds = HBonds::new(HBondCriterion::default())?.compute(HBondsInput {
///     positions: x.view(),
///     donors: &donors,
///     acceptors: &acceptors,
///     pairs: &[&table],
/// })?;
/// assert_eq!(bonds.counts, vec![1]);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
///
/// Each candidate is gated by the [`HBondCriterion`] under the minimum image
/// of the box the table was searched in (so any box shape works). Ported
/// from the reference implementation `CHBond::AnalyzeStep` / the bond test in
/// `src/hbond.cpp` (lines ~900–965), with the documented angle-at-H deviation
/// of [`HBondCriterion`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HBonds {
    criterion: HBondCriterion,
}

/// The data [`HBonds`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct HBondsInput<'a> {
    /// Positions, `T × N × 3`, Å.
    pub positions: ArrayView3<'a, F>,
    /// Donor `(heavy, hydrogen)` atom-index pairs.
    pub donors: &'a [(u32, u32)],
    /// Acceptor heavy-atom indices.
    pub acceptors: &'a [u32],
    /// One cross-query per frame: donor sites (query) against acceptors
    /// (reference), with the `dist_sq` column and the box it was searched in.
    pub pairs: &'a [&'a Neighbors],
}

/// Result of [`HBonds`]: the satisfied bonds per frame plus per-frame counts.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct HBondsResult {
    /// `per_frame[t]` is the list of bonds detected in frame `t`.
    pub per_frame: Vec<Vec<HBond>>,
    /// `counts[t] == per_frame[t].len()`.
    pub counts: Vec<usize>,
    /// Number of atoms per frame.
    pub n_atoms: usize,
}

impl HBonds {
    /// A detector gating candidates by `criterion`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a criterion field outside its
    /// physical range.
    pub fn new(criterion: HBondCriterion) -> Result<Self, ComputeError> {
        criterion.validate()?;
        Ok(Self { criterion })
    }

    /// The geometric criterion.
    pub fn criterion(&self) -> HBondCriterion {
        self.criterion
    }

    /// Validate everything but the per-frame tables; returns `(T, N)`.
    fn validate(&self, input: &HBondsInput<'_>) -> Result<(usize, usize), ComputeError> {
        if input.donors.is_empty() || input.acceptors.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let (n_frames, n_atoms, n_dim) = input.positions.dim();
        if n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if n_dim != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: n_dim,
                what: "HBonds positions (T × N × 3)",
            });
        }
        if input.pairs.len() != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: input.pairs.len(),
                what: "HBonds pairs (one table per frame)",
            });
        }
        for &(d, h) in input.donors {
            if d as usize >= n_atoms || h as usize >= n_atoms {
                return Err(ComputeError::OutOfRange {
                    field: "HBonds donors atom index",
                    value: format!("{d}/{h} >= {n_atoms}"),
                });
            }
        }
        for &a in input.acceptors {
            if a as usize >= n_atoms {
                return Err(ComputeError::OutOfRange {
                    field: "HBonds acceptors atom index",
                    value: format!("{a} >= {n_atoms}"),
                });
            }
        }
        if let Some(index) = input.positions.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "HBonds positions",
                index,
            });
        }
        Ok((n_frames, n_atoms))
    }

    /// The bonds of one frame from its candidate table.
    fn detect_frame(
        &self,
        positions: ArrayView2<'_, F>,
        donors: &[(u32, u32)],
        acceptors: &[u32],
        table: &Neighbors,
    ) -> Result<Vec<HBond>, ComputeError> {
        let expected_mode = QueryMode::CrossQuery {
            n_query_points: donors.len(),
            n_points: acceptors.len(),
        };
        if table.mode() != expected_mode {
            return Err(ComputeError::BadShape {
                expected: format!(
                    "{expected_mode:?}: donor sites (query) against acceptors (reference)"
                ),
                got: format!("{:?}", table.mode()),
            });
        }
        let Some(simbox) = table.simbox() else {
            return Err(ComputeError::BadShape {
                expected: "a pair table that records the box it was searched in".to_string(),
                got: "a table with no box".to_string(),
            });
        };
        require_dist_sq(table)?;
        let dist_sq = table.dist_sq().expect("require_dist_sq vouched for it");
        let mic = simbox.mic();
        let pos = |i: u32| -> [F; 3] {
            let r = positions.row(i as usize);
            [r[0], r[1], r[2]]
        };
        let cutoff = self.criterion.dist_cutoff;
        let cutoff_sq = cutoff * cutoff;

        let qi = table.query_point_indices();
        let pj = table.point_indices();
        let mut bonds = Vec::new();
        for k in 0..table.n_pairs() {
            // The gated distance (D···A or H···A, per `dist_kind`) is the
            // table's own: the caller's cutoff may exceed the criterion's.
            if dist_sq[k].is_nan() {
                return Err(ComputeError::NonFinite {
                    where_: "pair dist_sq",
                    index: k,
                });
            }
            if dist_sq[k] > cutoff_sq {
                continue;
            }
            let (donor, hydrogen) = donors[qi[k] as usize];
            let acceptor = acceptors[pj[k] as usize];
            // Exclude self-bonds (acceptor coincides with this donor's atoms).
            if acceptor == donor || acceptor == hydrogen {
                continue;
            }
            let dpos = pos(donor);
            let hpos = pos(hydrogen);
            let apos = pos(acceptor);

            // r(D···A) under MIC.
            let r_da = norm(mic.apply(sub(apos, dpos)));
            let dist_ok = match self.criterion.dist_kind {
                HBondDistanceKind::DonorAcceptor => true,
                HBondDistanceKind::HydrogenAcceptor => {
                    r_da <= cutoff + self.criterion.donor_acceptor_margin
                }
            };

            // D–H···A angle at the hydrogen: angle between H→D and H→A.
            let v_hd = mic.apply(sub(dpos, hpos));
            let v_ha = mic.apply(sub(apos, hpos));
            let n_hd = norm(v_hd);
            let n_ha = norm(v_ha);
            if n_hd == 0.0 || n_ha == 0.0 {
                // Degenerate geometry (coincident atoms) — not a bond, no NaN.
                continue;
            }
            let cos = (dot(v_hd, v_ha) / (n_hd * n_ha)).clamp(-1.0, 1.0);
            let angle = cos.acos().to_degrees();

            if dist_ok && angle >= self.criterion.angle_cutoff {
                bonds.push(HBond {
                    donor,
                    hydrogen,
                    acceptor,
                    distance: r_da,
                    angle,
                });
            }
        }
        Ok(bonds)
    }
}

impl Default for HBonds {
    /// The Luzar–Chandler criterion ([`HBondCriterion::default`]).
    fn default() -> Self {
        Self {
            criterion: HBondCriterion::default(),
        }
    }
}

impl Compute for HBonds {
    type Input<'a> = HBondsInput<'a>;
    type Output = HBondsResult;

    /// # Errors
    /// * [`ComputeError::EmptyInput`] for no frames, or an empty donor or
    ///   acceptor selection.
    /// * [`ComputeError::DimensionMismatch`] for positions that are not
    ///   `T × N × 3` or a table count other than `T`.
    /// * [`ComputeError::OutOfRange`] for a selected atom index `≥ N`.
    /// * [`ComputeError::NonFinite`] for a non-finite position or distance.
    /// * [`ComputeError::BadShape`] for a table that is not the donor-site ×
    ///   acceptor cross-query, has no box, or lacks `dist_sq`.
    fn compute(&self, input: HBondsInput<'_>) -> Result<HBondsResult, ComputeError> {
        let (n_frames, n_atoms) = self.validate(&input)?;
        let frame = |t: usize| {
            self.detect_frame(
                input.positions.index_axis(Axis(0), t),
                input.donors,
                input.acceptors,
                input.pairs[t],
            )
        };
        // Frames are independent; collecting a `Result` keeps frame order
        // and reports the earliest failing frame.
        #[cfg(feature = "rayon")]
        let per_frame: Vec<Vec<HBond>> = {
            use rayon::prelude::*;
            (0..n_frames)
                .into_par_iter()
                .map(frame)
                .collect::<Result<_, _>>()?
        };
        #[cfg(not(feature = "rayon"))]
        let per_frame: Vec<Vec<HBond>> = (0..n_frames).map(frame).collect::<Result<_, _>>()?;

        let counts = per_frame.iter().map(Vec::len).collect();
        Ok(HBondsResult {
            per_frame,
            counts,
            n_atoms,
        })
    }
}

impl HBondsResult {
    /// The presence series of every `(donor, acceptor)` pair that is bonded
    /// in at least one frame, for [`Survival`](crate::compute::Survival):
    ///
    /// ```text
    /// h_b(t) = 1 if pair b is hydrogen-bonded in frame t, else 0
    /// ```
    ///
    /// The continuous `S_HB(t)` and intermittent `C_HB(t)` lifetime
    /// correlations (Luzar & Chandler, *Phys. Rev. Lett.* **76**, 928
    /// (1996)) are `Survival` of this series; divided by their lag-0 value
    /// `⟨h⟩` they start at 1, and [`CumulativeTrapezoid`] integrates them
    /// into the lifetimes.
    ///
    /// `None` when no bond formed in any frame: there is then no lifetime,
    /// not a lifetime of zero.
    ///
    /// [`CumulativeTrapezoid`]: crate::compute::CumulativeTrapezoid
    pub fn presence(&self) -> Option<Presence> {
        let key = |b: &HBond| (b.donor, b.acceptor);
        let pairs: Vec<(u32, u32)> = self
            .per_frame
            .iter()
            .flatten()
            .map(key)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if pairs.is_empty() {
            return None;
        }
        let bonded: Vec<HashSet<(u32, u32)>> = self
            .per_frame
            .iter()
            .map(|bonds| bonds.iter().map(key).collect())
            .collect();
        let values = Array2::from_shape_fn((self.per_frame.len(), pairs.len()), |(t, k)| {
            if bonded[t].contains(&pairs[k]) {
                1.0
            } else {
                0.0
            }
        });
        Some(Presence { pairs, values })
    }

    /// The connected components of frame `frame`'s hydrogen-bond network: a
    /// component label per atom, which is a
    /// [`Partition`](crate::compute::Partition).
    ///
    /// Nodes are the atoms; each bond joins its donor, hydrogen and acceptor.
    /// An atom in no bond is a component of its own. Labels number the
    /// components in order of their lowest atom index, from 0. Ported from
    /// the reference implementation's aggregation topology
    /// (`src/aggrtopo.cpp`), on [`Topology::connected_components`].
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a frame beyond the result.
    pub fn components(&self, frame: usize) -> Result<Vec<u32>, ComputeError> {
        let Some(bonds) = self.per_frame.get(frame) else {
            return Err(ComputeError::OutOfRange {
                field: "HBondsResult::components frame",
                value: format!("{frame} (the result has {})", self.per_frame.len()),
            });
        };
        let edges: Vec<[usize; 2]> = bonds
            .iter()
            .flat_map(|bond| {
                let (d, h, a) = (
                    bond.donor as usize,
                    bond.hydrogen as usize,
                    bond.acceptor as usize,
                );
                [[d, h], [h, a]]
            })
            .collect();
        let mut topology = Topology::with_atoms(self.n_atoms);
        topology.add_bonds(&edges);
        Ok(topology
            .connected_components()
            .into_iter()
            .map(|label| label as u32)
            .collect())
    }
}

crate::compute::spec::compute_spec! {
    HBonds as "HBonds";
    params |s| {
        dist_cutoff: Float = HBonds::default().criterion.dist_cutoff => s.criterion.dist_cutoff;
        dist_kind: Choice(HBondDistanceKind::NAMES)
            = HBonds::default().criterion.dist_kind
            => s.criterion.dist_kind;
        angle_cutoff: Float = HBonds::default().criterion.angle_cutoff => s.criterion.angle_cutoff;
        donor_acceptor_margin: Float
            = HBonds::default().criterion.donor_acceptor_margin
            => s.criterion.donor_acceptor_margin;
    }
    inputs {
        positions: Trajectory;
        donors: IndexPairs;
        acceptors: Indices;
        pairs: NeighborFrames;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{
        CumulativeTrapezoid, CumulativeTrapezoidInput, Partition, Survival, SurvivalInput,
        SurvivalKind,
    };
    use crate::core::Cell;
    use molrs::core::{NeighborQuery, SimBox};
    use ndarray::{Array2, Array3};

    fn simbox() -> SimBox {
        SimBox::new(
            Cell::Lengths([20.0; 3]),
            [-10.0, -10.0, -10.0],
            [true, true, true],
        )
        .unwrap()
    }

    /// Per frame: donor O (0), its H (1), acceptor O (2) at `x_a` on the
    /// line D→H, and a second acceptor O (3) at (0, 2.8, 0), off-axis.
    fn trajectory(acceptor_x: &[F]) -> Array3<F> {
        Array3::from_shape_fn((acceptor_x.len(), 4, 3), |(t, a, d)| {
            let p = match a {
                0 => [0.0, 0.0, 0.0],
                1 => [1.0, 0.0, 0.0],
                2 => [acceptor_x[t], 0.0, 0.0],
                _ => [0.0, 2.8, 0.0],
            };
            p[d]
        })
    }

    /// The caller's candidate search: donor sites against acceptors.
    fn candidates(
        x: &Array3<F>,
        donors: &[(u32, u32)],
        acceptors: &[u32],
        kind: HBondDistanceKind,
        cutoff: F,
    ) -> Vec<Neighbors> {
        let gather = |frame: ArrayView2<'_, F>, atoms: &[u32]| {
            Array2::from_shape_fn((atoms.len(), 3), |(k, d)| frame[[atoms[k] as usize, d]])
        };
        let sites: Vec<u32> = donors
            .iter()
            .map(|&(d, h)| match kind {
                HBondDistanceKind::DonorAcceptor => d,
                HBondDistanceKind::HydrogenAcceptor => h,
            })
            .collect();
        x.outer_iter()
            .map(|frame| {
                NeighborQuery::new(&simbox(), gather(frame, acceptors).view(), cutoff)
                    .unwrap()
                    .query(gather(frame, &sites).view())
            })
            .collect()
    }

    fn detect(hbonds: HBonds, x: &Array3<F>) -> Result<HBondsResult, ComputeError> {
        let donors = [(0, 1)];
        let acceptors = [2, 3];
        let c = hbonds.criterion();
        let tables = candidates(x, &donors, &acceptors, c.dist_kind, c.dist_cutoff);
        let refs: Vec<&Neighbors> = tables.iter().collect();
        hbonds.compute(HBondsInput {
            positions: x.view(),
            donors: &donors,
            acceptors: &acceptors,
            pairs: &refs,
        })
    }

    #[test]
    fn a_linear_bond_is_detected_and_a_bent_one_is_not() {
        let res = detect(HBonds::default(), &trajectory(&[2.8, 5.0])).unwrap();
        assert_eq!(res.counts, vec![1, 0]);
        let bond = res.per_frame[0][0];
        assert_eq!((bond.donor, bond.hydrogen, bond.acceptor), (0, 1, 2));
        assert!((bond.distance - 2.8).abs() < 1e-12);
        assert!((bond.angle - 180.0).abs() < 1e-9);
    }

    /// Gating on H···A: the acceptor must also lie within
    /// `dist_cutoff + donor_acceptor_margin` of the donor.
    #[test]
    fn hydrogen_acceptor_gate_uses_the_margin() {
        // H···A = 2.4 ≤ 2.5; D···A = 3.4.
        let x = trajectory(&[3.4]);
        let criterion = HBondCriterion::new(2.5, HBondDistanceKind::HydrogenAcceptor, 150.0);
        assert_eq!(criterion.donor_acceptor_margin, 1.05);
        assert_eq!(
            detect(HBonds::new(criterion).unwrap(), &x).unwrap().counts,
            vec![1]
        );
        // 3.4 > 2.5 + 0.5.
        let tight = criterion.with_donor_acceptor_margin(0.5);
        assert_eq!(
            detect(HBonds::new(tight).unwrap(), &x).unwrap().counts,
            vec![0]
        );
    }

    /// The caller's search may reach further than the criterion; the kernel
    /// gates on its own cutoff.
    #[test]
    fn a_wider_search_does_not_widen_the_criterion() {
        let x = trajectory(&[4.0]);
        let donors = [(0, 1)];
        let acceptors = [2, 3];
        let tables = candidates(
            &x,
            &donors,
            &acceptors,
            HBondDistanceKind::DonorAcceptor,
            6.0,
        );
        let res = HBonds::default()
            .compute(HBondsInput {
                positions: x.view(),
                donors: &donors,
                acceptors: &acceptors,
                pairs: &[&tables[0]],
            })
            .unwrap();
        assert_eq!(res.counts, vec![0]);
    }

    #[test]
    fn an_empty_selection_is_empty_input() {
        let x = trajectory(&[2.8]);
        let table = &candidates(&x, &[(0, 1)], &[2], HBondDistanceKind::DonorAcceptor, 3.5)[0];
        for (donors, acceptors) in [(&[][..], &[2u32][..]), (&[(0u32, 1u32)][..], &[][..])] {
            let err = HBonds::default()
                .compute(HBondsInput {
                    positions: x.view(),
                    donors,
                    acceptors,
                    pairs: &[table],
                })
                .unwrap_err();
            assert!(matches!(err, ComputeError::EmptyInput));
        }
    }

    #[test]
    fn rejects_bad_configuration_and_tables() {
        let bad = HBondCriterion {
            dist_cutoff: 0.0,
            ..HBondCriterion::default()
        };
        assert!(matches!(
            HBonds::new(bad),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert!(matches!(
            HBonds::new(HBondCriterion::default().with_donor_acceptor_margin(-1.0)),
            Err(ComputeError::OutOfRange { .. })
        ));

        // A table searched over the wrong selections.
        let x = trajectory(&[2.8]);
        let wrong = &candidates(&x, &[(0, 1)], &[2], HBondDistanceKind::DonorAcceptor, 3.5)[0];
        let err = HBonds::default()
            .compute(HBondsInput {
                positions: x.view(),
                donors: &[(0, 1)],
                acceptors: &[2, 3],
                pairs: &[wrong],
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));

        // One table per frame.
        let err = HBonds::default()
            .compute(HBondsInput {
                positions: x.view(),
                donors: &[(0, 1)],
                acceptors: &[2],
                pairs: &[],
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));

        let err = HBonds::default()
            .compute(HBondsInput {
                positions: x.view(),
                donors: &[(0, 9)],
                acceptors: &[2],
                pairs: &[wrong],
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    /// ac-010: `HBonds` → `presence()` → `Survival` (+ `CumulativeTrapezoid`
    /// for the lifetimes), with no glue. One bond present at frames 0, 1, 3,
    /// 4, 5 (absent at 2), dt = 0.5 — the `hbond_lifetimes` golden.
    #[test]
    fn hbonds_presence_survival_lifetimes() {
        let x = trajectory(&[2.8, 2.8, 5.0, 2.8, 2.8, 2.8]);
        let presence = detect(HBonds::default(), &x).unwrap().presence().unwrap();
        assert_eq!(presence.pairs, vec![(0, 2)]);
        assert_eq!(
            presence.values.column(0).to_vec(),
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        );

        let curve = |kind| {
            Survival::new(kind, 3)
                .compute(SurvivalInput {
                    presence: presence.values.view(),
                    dt: 0.5,
                    inner: None,
                    n_reference: None,
                })
                .unwrap()
        };
        let s = curve(SurvivalKind::Continuous);
        let c = curve(SurvivalKind::Intermittent);
        assert_eq!(s.lag_times.to_vec(), vec![0.0, 0.5, 1.0, 1.5]);

        // ⟨h⟩ = 5/6. Continuous: 3/5 and 1/4 unbroken, then none; gapped:
        // 3/5, 2/4, 2/3 — each divided by ⟨h⟩.
        let h = 5.0 / 6.0;
        let continuous = [1.0, 0.6 / h, 0.25 / h, 0.0];
        let intermittent = [1.0, 0.6 / h, 0.5 / h, (2.0 / 3.0) / h];
        for lag in 0..=3 {
            let s_norm = s.values[[lag, 0]] / s.values[[0, 0]];
            let c_norm = c.values[[lag, 0]] / c.values[[0, 0]];
            assert!((s_norm - continuous[lag]).abs() < 1e-12, "S({lag})");
            assert!((c_norm - intermittent[lag]).abs() < 1e-12, "C({lag})");
        }

        let lifetime = |curve: &crate::compute::TimeCorrelation| {
            let running = CumulativeTrapezoid::new()
                .compute(CumulativeTrapezoidInput {
                    correlation: curve,
                    channel: 0,
                })
                .unwrap();
            running.values[[3, 0]] / curve.values[[0, 0]]
        };
        let trapz = |y: &[F]| y.windows(2).map(|w| 0.25 * (w[0] + w[1])).sum::<F>();
        assert!((lifetime(&s) - trapz(&continuous)).abs() < 1e-12);
        assert!((lifetime(&c) - trapz(&intermittent)).abs() < 1e-12);
    }

    /// No bond in any frame: no presence series, so no lifetime — not a
    /// lifetime of zero (`hbond_lifetimes` returned all-zero curves).
    #[test]
    fn no_bond_ever_is_no_presence() {
        let res = detect(HBonds::default(), &trajectory(&[5.0, 5.0, 5.0])).unwrap();
        assert_eq!(res.counts, vec![0, 0, 0]);
        assert!(res.presence().is_none());
    }

    #[test]
    fn components_label_the_bond_network() {
        let res = detect(HBonds::default(), &trajectory(&[2.8, 5.0])).unwrap();
        // Frame 0: D(0)–H(1)···A(2) is one component; A(3) is alone.
        let labels = res.components(0).unwrap();
        assert_eq!(labels, vec![0, 0, 0, 1]);
        let partition: &dyn Partition = &labels;
        assert_eq!(partition.n_items(), 4);
        assert_eq!(partition.group_of(2), Some(0));
        // Frame 1: no bond, every atom on its own.
        assert_eq!(res.components(1).unwrap(), vec![0, 1, 2, 3]);
        assert!(matches!(
            res.components(2),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
