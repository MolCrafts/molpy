//! The soft (penetrable) packing potential: harmonic 1-2 / 1-3 springs plus
//! a soft core between every other pair.
//!
//! [`SoftSpec`] is the parameterization — the topology a frame's `bonds` /
//! `angles` blocks give, the exclusions they imply, and the form's
//! constants. [`SoftSpec::potential`] binds it to a box and hands back a
//! [`SoftPotential`], an ordinary [`Potential`] that resolves its own pairs:
//!
//! * the **bonded** springs are fixed at the first configuration it is
//!   evaluated on — their rest lengths and periodic images are that
//!   configuration's, and do not follow the atoms as they relax;
//! * the **non-bonded** pairs come from molrs's neighbour list, rebuilt
//!   whenever an atom has moved more than half a skin since the last build
//!   (or the atom count changed), so the list is complete inside the
//!   cut-off at every evaluation.
//!
//! Minimizing it is minimizing any potential:
//! `Lbfgs::new(Arc::new(SoftSpec::from_frame(&frame).potential(frame.simbox.as_ref())), …)`.
//!
//! Per pair at distance `r`: `a_rep (σ − r)²` for `r < σ`, and, with an
//! attraction `b > 0`, `−b (r − σ)(r_cut − r)` for `σ ≤ r < r_cut`. A spring
//! between `i` and `j` with rest length `t` is `k (r − t)²`.

use std::collections::HashSet;
use std::sync::Mutex;

use ndarray::ArrayView2;

use crate::ff::potential::flat_coords::sub3;
use crate::ff::potential::{Accumulator, Potential, end_pairs};
use molrs::core::Frame;
use molrs::core::NeighborQuery;
use molrs::core::keys::{ATOMI, ATOMJ, ATOMK};
use molrs::core::schema::block_names::{ANGLES, BONDS};
use molrs::core::{Mic, SimBox};
use molrs::op::F;
use molrs::op::vec3::norm;

/// How far past the cut-off the neighbour list reaches (Å), so that it stays
/// complete until some atom has moved half of it.
const SKIN: F = 1.0;

/// The soft potential's topology and constants.
#[derive(Debug, Clone)]
pub struct SoftSpec {
    bonds: Vec<(usize, usize)>,
    angles: Vec<(usize, usize)>,
    excluded: HashSet<(usize, usize)>,
    sigma: F,
    a_rep: F,
    b_attract: F,
    rcut: F,
    k_bond: F,
    k_ang: F,
}

impl SoftSpec {
    /// 1-2 pairs from the frame's `bonds` block (`atomi`, `atomj`) and 1-3
    /// pairs from its `angles` block's ends (`atomi`, `atomk`); both are
    /// springs, and neither is a non-bonded pair.
    pub fn from_frame(frame: &Frame) -> Self {
        let bonds = end_pairs(frame, BONDS, ATOMI, ATOMJ);
        let angles = end_pairs(frame, ANGLES, ATOMI, ATOMK);
        let excluded = bonds.iter().chain(&angles).copied().collect();
        Self {
            bonds,
            angles,
            excluded,
            sigma: 2.6,
            a_rep: 8.0,
            b_attract: 0.0,
            rcut: 5.0,
            k_bond: 50.0,
            k_ang: 8.0,
        }
    }

    /// Add an attraction of strength `b` between `σ` and `r_cut`.
    pub fn with_attraction(mut self, b: F) -> Self {
        self.b_attract = b;
        self
    }

    /// The potential of this spec in `simbox` (`None`: free space).
    pub fn potential(&self, simbox: Option<&SimBox>) -> SoftPotential {
        SoftPotential {
            spec: self.clone(),
            simbox: simbox.cloned(),
            mic: simbox.map_or(Mic::free(), SimBox::mic),
            pairs: Mutex::new(None),
        }
    }
}

/// A harmonic spring: atoms `(i, j)`, rest length, periodic image `shift`
/// (so the spring's vector is `x_i − x_j − shift`).
type Spring = (usize, usize, F, [F; 3]);
/// A non-bonded pair with its periodic image `shift`.
type Pair = (usize, usize, [F; 3]);

/// What a [`SoftPotential`] has resolved so far.
#[derive(Debug, Clone)]
struct Resolved {
    bonds: Vec<Spring>,
    angles: Vec<Spring>,
    /// The coordinates the non-bonded list was built at.
    anchor: Vec<F>,
    nb: Vec<Pair>,
}

/// [`SoftSpec`] bound to a box: a [`Potential`] that resolves its springs on
/// first use and rebuilds its non-bonded pairs as the atoms move (module
/// docs).
#[derive(Debug)]
pub struct SoftPotential {
    spec: SoftSpec,
    simbox: Option<SimBox>,
    mic: Mic,
    pairs: Mutex<Option<Resolved>>,
}

impl SoftPotential {
    /// Springs from the configuration `coords`: rest length and image are its
    /// minimum-image separation.
    fn springs(&self, coords: &[F], ends: &[(usize, usize)]) -> Vec<Spring> {
        ends.iter()
            .map(|&(i, j)| {
                let raw = sub3(coords, i, coords, j);
                let d = self.mic.apply(raw);
                let shift = [raw[0] - d[0], raw[1] - d[1], raw[2] - d[2]];
                (i, j, norm(d), shift)
            })
            .collect()
    }

    /// The non-bonded pairs within `max(r_cut, σ) + SKIN` of each other.
    fn neighbours(&self, coords: &[F]) -> Vec<Pair> {
        let n = coords.len() / 3;
        let view = ArrayView2::from_shape((n, 3), coords).expect("3·n coordinates");
        let cutoff = self.spec.rcut.max(self.spec.sigma) + SKIN;
        // `rcut` and `sigma` are the spec's fixed positive constants, so the
        // search radius is a positive length.
        const RADIUS: &str = "the fixed rcut/sigma plus the skin is a valid cutoff";
        let table = match &self.simbox {
            Some(b) => NeighborQuery::new(b, view, cutoff)
                .expect(RADIUS)
                .query_self(),
            None => NeighborQuery::unbounded(view, cutoff)
                .expect(RADIUS)
                .query_self(),
        };
        let (qi, qj) = (table.query_point_indices(), table.point_indices());
        (0..table.n_pairs())
            .filter_map(|k| {
                let (i, j) = (qi[k] as usize, qj[k] as usize);
                if i == j || self.spec.excluded.contains(&(i.min(j), i.max(j))) {
                    return None;
                }
                let raw = sub3(coords, i, coords, j);
                let d = self.mic.apply(raw);
                Some((i, j, [raw[0] - d[0], raw[1] - d[1], raw[2] - d[2]]))
            })
            .collect()
    }

    /// Bring the resolved pairs up to date for `coords`.
    fn refresh(&self, coords: &[F], resolved: &mut Option<Resolved>) {
        let stale = match resolved {
            Some(r) if r.anchor.len() == coords.len() => {
                let limit = (0.5 * SKIN) * (0.5 * SKIN);
                coords
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .zip(r.anchor.as_chunks::<3>().0)
                    .any(|(x, a)| {
                        let d = self.mic.apply([x[0] - a[0], x[1] - a[1], x[2] - a[2]]);
                        d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > limit
                    })
            }
            _ => {
                // First use, or another system: its springs are this one's.
                *resolved = Some(Resolved {
                    bonds: self.springs(coords, &self.spec.bonds),
                    angles: self.springs(coords, &self.spec.angles),
                    anchor: Vec::new(),
                    nb: Vec::new(),
                });
                true
            }
        };
        if stale {
            let r = resolved.as_mut().expect("resolved above");
            r.nb = self.neighbours(coords);
            r.anchor = coords.to_vec();
        }
    }
}

impl Potential for SoftPotential {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let mut guard = self.pairs.lock().unwrap_or_else(|e| e.into_inner());
        self.refresh(coords, &mut guard);
        let r = guard.as_ref().expect("refreshed");
        let s = &self.spec;
        let forces = acc.forces_without_virial();
        let mut e: F = 0.0;
        for &(i, j, t, shift) in &r.bonds {
            e += spring(coords, forces, i, j, t, s.k_bond, shift);
        }
        for &(i, j, t, shift) in &r.angles {
            e += spring(coords, forces, i, j, t, s.k_ang, shift);
        }
        for &(i, j, shift) in &r.nb {
            let d = disp(coords, i, j, shift);
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < 1e-18 {
                continue;
            }
            let dist = r2.sqrt();
            let dedr = if dist < s.sigma {
                e += s.a_rep * (s.sigma - dist) * (s.sigma - dist);
                -2.0 * s.a_rep * (s.sigma - dist)
            } else if s.b_attract > 0.0 && dist < s.rcut {
                e += -s.b_attract * (dist - s.sigma) * (s.rcut - dist);
                -s.b_attract * (s.rcut + s.sigma - 2.0 * dist)
            } else {
                continue;
            };
            push_pair(forces, i, j, -dedr / dist, d);
        }
        acc.add_energy(e);
    }
}

/// `x_i − x_j − shift`.
#[inline]
fn disp(coords: &[F], i: usize, j: usize, shift: [F; 3]) -> [F; 3] {
    [
        coords[3 * i] - coords[3 * j] - shift[0],
        coords[3 * i + 1] - coords[3 * j + 1] - shift[1],
        coords[3 * i + 2] - coords[3 * j + 2] - shift[2],
    ]
}

fn push_pair(forces: &mut [F], i: usize, j: usize, c: F, d: [F; 3]) {
    for ax in 0..3 {
        forces[3 * i + ax] += c * d[ax];
        forces[3 * j + ax] -= c * d[ax];
    }
}

fn spring(coords: &[F], forces: &mut [F], i: usize, j: usize, t: F, k: F, shift: [F; 3]) -> F {
    let d = disp(coords, i, j, shift);
    let r = norm(d);
    if r < 1e-9 {
        return 0.0;
    }
    push_pair(forces, i, j, -2.0 * k * (r - t) / r, d);
    k * (r - t) * (r - t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

    /// A frame carrying the bonds/angles blocks of a linear n-atom chain
    /// (1-2 = (i, i+1); 1-3 angle ends = (i, i+2)).
    fn chain_frame(n: usize) -> Frame {
        let mut frame = Frame::new();
        let (bi, bj): (Vec<Idx>, Vec<Idx>) = (0..n - 1).map(|i| (i as Idx, (i + 1) as Idx)).unzip();
        let mut bonds = Block::new();
        bonds
            .insert(ATOMI, Array1::from_vec(bi).into_dyn(), None)
            .unwrap();
        bonds
            .insert(ATOMJ, Array1::from_vec(bj).into_dyn(), None)
            .unwrap();
        frame.insert(BONDS, bonds);
        let (ai, ak): (Vec<Idx>, Vec<Idx>) = (0..n - 2).map(|i| (i as Idx, (i + 2) as Idx)).unzip();
        let mut angles = Block::new();
        angles
            .insert(ATOMI, Array1::from_vec(ai).into_dyn(), None)
            .unwrap();
        angles
            .insert(ATOMK, Array1::from_vec(ak).into_dyn(), None)
            .unwrap();
        frame.insert(ANGLES, angles);
        frame
    }

    const CHAIN: [[F; 3]; 5] = [
        [0.0, 0.0, 0.0],
        [1.4, 0.3, 0.0],
        [2.9, -0.2, 0.4],
        [4.0, 0.5, 0.1],
        [5.2, 0.0, -0.3],
    ];

    #[test]
    fn spec_reads_bonds_and_angle_ends() {
        let s = SoftSpec::from_frame(&chain_frame(5));
        assert_eq!(s.bonds.len(), 4);
        assert_eq!(s.angles.len(), 3);
        assert_eq!(s.excluded.len(), 7);
    }

    /// The forces are the gradient of the energy, both branches of the
    /// non-bonded term included, free and in a periodic box.
    fn fd_check(simbox: Option<SimBox>) {
        let pot = SoftSpec::from_frame(&chain_frame(5))
            .with_attraction(1.0)
            .potential(simbox.as_ref());
        // The springs rest at a configuration other than the one evaluated.
        let rest: Vec<F> = CHAIN.iter().flatten().map(|x| 1.05 * x).collect();
        pot.energy_forces_at(&rest);
        let flat: Vec<F> = CHAIN.iter().flatten().copied().collect();
        let (_e, f) = pot.energy_forces_at(&flat);
        let h = 1e-6;
        for k in 0..flat.len() {
            let mut xp = flat.clone();
            let mut xm = flat.clone();
            xp[k] += h;
            xm[k] -= h;
            let num = (pot.energy_at(&xp) - pot.energy_at(&xm)) / (2.0 * h);
            assert!(
                (f[k] + num).abs() < 1e-3,
                "force[{k}]={} -dE/dx={}",
                f[k],
                -num
            );
        }
    }

    #[test]
    fn forces_match_finite_difference_free() {
        fd_check(None);
    }

    #[test]
    fn forces_match_finite_difference_periodic() {
        fd_check(Some(
            SimBox::new(
                Cell::Lengths([50.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        ));
    }

    /// A pair that comes into range after the first evaluation is priced: the
    /// list is rebuilt once an atom has moved past half the skin.
    #[test]
    fn the_pair_list_follows_the_atoms() {
        let pot = SoftSpec::from_frame(&Frame::new()).potential(None);
        let far = [0.0, 0.0, 0.0, 20.0, 0.0, 0.0];
        assert_eq!(pot.energy_at(&far), 0.0);
        let near = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let want = 8.0 * (2.6 - 2.0) * (2.6 - 2.0);
        assert!((pot.energy_at(&near) - want).abs() < 1e-12);
    }

    /// Across a periodic face the nearest image is the pair.
    #[test]
    fn a_periodic_pair_is_the_nearest_image() {
        let b = SimBox::new(
            Cell::Lengths([14.0; 3]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap();
        let pot = SoftSpec::from_frame(&Frame::new()).potential(Some(&b));
        let x = [0.5, 0.0, 0.0, 13.5, 0.0, 0.0]; // 1 Å apart through the face
        let want = 8.0 * (2.6 - 1.0) * (2.6 - 1.0);
        assert!((pot.energy_at(&x) - want).abs() < 1e-12);
    }
}
