//! A force-field kernel bound to the `force` phase.
//!
//! Pair kernels read the [`Scratch::pairs`] the neighbour list published
//! earlier in the step. Bonded kernels read their own tables, each term
//! placed around one atom across any periodic face ([`BondedImages`]). Plain
//! kernels read the coordinates alone.

use molrs::core::{Frame, MetaValue, Virial};
use molrs::ff::potential::{
    Accumulator, ForceTerm, IndexedTerms, PairDriven, Potential, SpecialWeights,
};
use molrs::op::F;

use super::columns::{self, VIRIAL, read_pos};
use super::component::{Component, Flags, Kind};
use super::error::MdError;
use super::images::BondedImages;
use super::phase::Phase;
use super::scratch::Scratch;

/// One or more potential terms mounted as a single `force` component.
///
/// Each term adds into the forces and `meta["energy"]` that the pre-force
/// clear left at zero.
pub struct FfEvaluator {
    members: Vec<(ForceTerm, SpecialWeights)>,
    factors: Vec<F>,
    buf: Vec<F>,
    /// The bonded tables placed across periodic faces, this step.
    images: BondedImages,
}

impl FfEvaluator {
    /// A kernel summed over the neighbour table.
    pub fn pair(potential: impl PairDriven + 'static) -> Self {
        Self::from_term(ForceTerm::pair(potential))
    }

    /// A bonded kernel, or any kernel that reads coordinates and its own index table.
    pub fn indexed(potential: impl IndexedTerms + 'static) -> Self {
        Self::from_term(ForceTerm::indexed(potential))
    }

    /// A kernel that reads coordinates and nothing else.
    pub fn plain(potential: impl Potential + 'static) -> Self {
        Self::from_term(ForceTerm::plain(potential))
    }

    pub(crate) fn from_term(term: ForceTerm) -> Self {
        Self::from_members(vec![(term, SpecialWeights::default())])
    }

    /// Several terms, each with the weights on its close neighbours.
    pub fn from_members(members: Vec<(ForceTerm, SpecialWeights)>) -> Self {
        Self {
            members,
            factors: Vec::new(),
            buf: Vec::new(),
            images: BondedImages::default(),
        }
    }

    pub(crate) fn flags(&self) -> Flags {
        let consumes_neighbors = self
            .members
            .iter()
            .any(|(term, _)| matches!(term, ForceTerm::Pair(_)) && !term.binds_a_fixed_pair_list());
        let frozen_pairs = self
            .members
            .iter()
            .any(|(term, _)| term.binds_a_fixed_pair_list());
        Flags {
            kind: Kind::Evaluator,
            consumes_neighbors,
            frozen_pairs,
            dt: None,
        }
    }

    /// Add this step's forces, energy and virial onto what the pre-force
    /// clear left. The virial is `meta["virial"]` (`[xx, yy, zz, xy, xz, yz]`);
    /// a term without one removes it, since a virial missing a term is a
    /// different quantity, not a smaller one.
    fn add_onto_frame(
        &self,
        frame: &mut Frame,
        energy: F,
        virial: Option<Virial>,
    ) -> Result<(), MdError> {
        let n = self.buf.len() / 3;
        let mut force = columns::read_force(frame)?;
        if force.nrows() != n {
            return Err(MdError::Invalid(format!(
                "evaluator produced {n} atoms, the frame has {}",
                force.nrows()
            )));
        }
        for i in 0..n {
            for axis in 0..3 {
                force[[i, axis]] += self.buf[i * 3 + axis];
            }
        }
        columns::write_force(frame, &force)?;
        let total = columns::meta_f64(frame, "energy") + energy;
        columns::set_meta_f64(frame, "energy", total);
        match (virial, frame.meta.get(VIRIAL)) {
            (Some(w), Some(MetaValue::F64x6(held))) => {
                let sum: [F; 6] = std::array::from_fn(|k| held[k] + w.components[k]);
                frame.meta.insert(VIRIAL, MetaValue::F64x6(sum));
            }
            _ => {
                frame.meta.remove(VIRIAL);
            }
        }
        Ok(())
    }
}

impl Component for FfEvaluator {
    fn name(&self) -> &str {
        "evaluator"
    }

    fn default_bindings(&self) -> Vec<(Phase, i32)> {
        vec![(Phase::Force, 0)]
    }

    fn launch(
        &mut self,
        phase: Phase,
        frame: &mut Frame,
        scratch: &mut Scratch,
    ) -> Result<(), MdError> {
        if phase != Phase::Force {
            return Ok(());
        }
        let pos = read_pos(frame)?;
        let n_owned = pos.nrows();
        let owned: Vec<F> = pos.iter().copied().collect();

        // The rows the bonded members read: the atoms, then the copies their
        // terms were placed onto across periodic faces.
        let sources: Vec<_> = self.members.iter().map(|(term, _)| term.terms()).collect();
        self.images
            .resolve(&sources, &owned, frame.simbox.as_ref())?;
        let mut rows = owned.clone();
        rows.extend_from_slice(self.images.copies());
        let n_rows = rows.len() / 3;

        // `explicit`: bonded forces on every row, whose virial is `Σ f ⊗ x`
        // before the copies fold. `buf`: pair forces (their kernels tally the
        // virial) and plain terms (which have none) on the atoms.
        let mut explicit = vec![0.0; n_rows * 3];
        self.buf.clear();
        self.buf.resize(n_owned * 3, 0.0);
        let mut energy = 0.0;
        let mut virial = Some(Virial::ZERO);
        let mut explicit_used = false;
        for (m, (term, special)) in self.members.iter_mut().enumerate() {
            match term {
                ForceTerm::Pair(pot) => {
                    let Some(pairs) = scratch.pairs.as_ref() else {
                        return Err(MdError::Invalid(
                            "pair evaluator ran with no neighbour list published this \
                             step; mount one with set_neighbor"
                                .into(),
                        ));
                    };
                    let factors = special.factors_for(pairs, &mut self.factors);
                    let mut acc = Accumulator::adding_into(&mut self.buf).with_virial();
                    pot.evaluate_pairs(&owned, pairs, factors, &mut acc);
                    energy += acc.energy();
                    virial = match (virial, acc.virial()) {
                        (Some(mut total), Some(w)) => {
                            total += w;
                            Some(total)
                        }
                        _ => None,
                    };
                }
                ForceTerm::Indexed(pot) => {
                    let terms = self.images.current(m).ok_or_else(|| {
                        MdError::Invalid(format!("member {m} holds indices but has no table"))
                    })?;
                    let mut acc = Accumulator::adding_into(&mut explicit);
                    pot.evaluate_terms(&rows, terms.view(), &mut acc);
                    energy += acc.energy();
                    explicit_used = true;
                }
                // An external field, a restraint: its forces do not sum to
                // zero, so `Σ f ⊗ x` moves with the cell's origin and is no
                // virial. The step then has none — the honest answer.
                ForceTerm::Plain(pot) => {
                    let mut acc = Accumulator::adding_into(&mut self.buf);
                    pot.evaluate(&owned, &mut acc);
                    energy += acc.energy();
                    virial = None;
                }
            }
        }

        if explicit_used {
            if let Some(total) = virial.as_mut() {
                for a in 0..n_rows {
                    total.add_outer(
                        [explicit[3 * a], explicit[3 * a + 1], explicit[3 * a + 2]],
                        [rows[3 * a], rows[3 * a + 1], rows[3 * a + 2]],
                    );
                }
            }
            self.images.fold(&mut explicit, n_owned);
            for (dst, f) in self.buf.iter_mut().zip(&explicit[..n_owned * 3]) {
                *dst += f;
            }
        }
        self.add_onto_frame(frame, energy, virial)
    }
}

#[cfg(test)]
mod tests {
    use super::super::infra::ZeroForces;
    use super::super::neighbor_list::Neighbor;
    use super::*;
    use molrs::core::NeighborList;
    use molrs::core::{Block, Cell, SimBox};
    use molrs::ff::compile::ExplicitTerms;
    use molrs::ff::ir::Params;
    use molrs::ff::potential::pair::PairLjCut;
    use ndarray::{Array1, array};

    /// A frame of `pos` (rows) in `simbox`.
    fn frame(pos: &ndarray::Array2<F>, simbox: SimBox) -> Frame {
        let mut atoms = Block::new();
        for (k, key) in ["x", "y", "z"].into_iter().enumerate() {
            atoms
                .insert(key, pos.column(k).to_owned().into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert("mass", Array1::from_elem(pos.nrows(), 1.0).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.simbox = Some(simbox);
        frame
    }

    fn cubic(l: F, pbc: bool) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0; 3], [pbc; 3]).unwrap()
    }

    /// The neighbour component over a list searching at `cutoff`.
    fn neighbor(cutoff: F, skin: F) -> Neighbor {
        Neighbor::new(NeighborList::new(cutoff).unwrap(), skin).unwrap()
    }

    /// One `force` phase: the pre-force clear, `neighbor` (when given), then
    /// the evaluator. Returns the energy, the forces and the virial.
    fn force(
        frame: &mut Frame,
        neighbor: Option<&mut dyn Component>,
        evaluator: &mut FfEvaluator,
    ) -> (F, ndarray::Array2<F>, Option<[F; 6]>) {
        let mut scratch = Scratch::new();
        if let Some(neighbor) = neighbor {
            neighbor.build(frame).unwrap();
            neighbor
                .launch(Phase::Neighbor, frame, &mut scratch)
                .unwrap();
        }
        ZeroForces
            .launch(Phase::PreForce, frame, &mut scratch)
            .unwrap();
        evaluator.launch(Phase::Force, frame, &mut scratch).unwrap();
        let virial = match frame.meta.get(columns::VIRIAL) {
            Some(MetaValue::F64x6(w)) => Some(*w),
            _ => None,
        };
        (
            columns::meta_f64(frame, "energy"),
            columns::read_force(frame).unwrap(),
            virial,
        )
    }

    /// A bond through a periodic face is measured the short way round: at
    /// rest it carries no energy, and stretched it pulls through the face.
    /// Read off the wrapped coordinates it would span the box.
    #[test]
    fn a_bond_across_a_face_is_measured_the_short_way() {
        // `bond harmonic`, k = 100, r0 = 1: E = k (r − r0)².
        let bond = || {
            let mut members = ExplicitTerms::new("bond", "harmonic")
                .term(&[0, 1], Params::from_iter([("k", 100.0), ("r0", 1.0)]))
                .compile()
                .unwrap()
                .into_members();
            FfEvaluator::from_term(members.pop().unwrap())
        };
        let mut at_rest = frame(&array![[0.5, 1.0, 1.0], [9.5, 1.0, 1.0]], cubic(10.0, true));
        let (e, f, _) = force(&mut at_rest, None, &mut bond());
        assert!(e.abs() < 1e-12, "{e}");
        assert!(f.iter().all(|v| v.abs() < 1e-9), "{f:?}");

        // 0.6 apart through the face: compressed by 0.4, pushed apart.
        let mut close = frame(&array![[0.2, 1.0, 1.0], [9.6, 1.0, 1.0]], cubic(10.0, true));
        let (e, f, w) = force(&mut close, None, &mut bond());
        assert!((e - 100.0 * 0.16).abs() < 1e-9, "{e}");
        assert!((f[[0, 0]] - 80.0).abs() < 1e-9, "{f:?}");
        assert!((f[[1, 0]] + 80.0).abs() < 1e-9, "{f:?}");
        // Σ f ⊗ r over the bond: 80 · 0.6 in xx.
        let w = w.expect("a bonded virial");
        assert!((w[0] - 48.0).abs() < 1e-9, "{w:?}");
    }

    /// The list knows only its cutoff; the box supplies the periodicity. A
    /// pair 2.5 apart through a face prices as the same pair 2.5 apart inside
    /// the box.
    #[test]
    fn a_pair_through_a_face_prices_as_inside_the_box() {
        let lj = || FfEvaluator::pair(PairLjCut::new(0.3, 3.4, 3.0, 12, 6, false, false).unwrap());
        let mut through = frame(&array![[1.0, 5.0, 5.0], [8.5, 5.0, 5.0]], cubic(10.0, true));
        let (e_through, f_through, _) =
            force(&mut through, Some(&mut neighbor(3.5, 0.5)), &mut lj());
        let mut inside = frame(&array![[4.0, 5.0, 5.0], [6.5, 5.0, 5.0]], cubic(10.0, true));
        let (e_inside, f_inside, _) = force(&mut inside, Some(&mut neighbor(3.5, 0.5)), &mut lj());
        assert!(e_inside.abs() > 1e-6, "a non-trivial reference: {e_inside}");
        assert!((e_through - e_inside).abs() <= 1e-12 * e_inside.abs());
        // Through the face atom 0 is on the right of atom 1: the force flips.
        assert!((f_through[[0, 0]] + f_inside[[0, 0]]).abs() <= 1e-10);
    }

    /// One pair along x: `Σ f ⊗ r` is `f₀ₓ · (x₀ − x₁)` in xx and nothing
    /// else, whichever route found the pair.
    #[test]
    fn a_single_pair_gives_the_hand_computed_virial() {
        let pos = array![[10.0, 10.0, 10.0], [14.0, 10.0, 10.0]];
        let lj = || FfEvaluator::pair(PairLjCut::new(0.3, 3.4, 5.0, 12, 6, false, false).unwrap());
        for periodic in [false, true] {
            let mut f = frame(&pos, cubic(30.0, periodic));
            let (_, forces, w) = force(&mut f, Some(&mut neighbor(5.0, 0.0)), &mut lj());
            let w = w.expect("a pair virial");
            assert!(
                (w[0] - forces[[0, 0]] * (10.0 - 14.0)).abs() < 1e-12,
                "{w:?}"
            );
            assert!(w[1..].iter().all(|c| c.abs() < 1e-12), "{w:?}");
        }
    }

    /// An external force has no virial; the step then reports none rather
    /// than a sum missing a term.
    #[test]
    fn a_plain_term_leaves_no_virial() {
        struct Push;
        impl molrs::ff::potential::Potential for Push {
            fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
                let f = acc.forces_without_virial();
                for i in 0..coords.len() / 3 {
                    f[3 * i] += 1.0;
                }
            }
        }
        let mut f = frame(&array![[1.0, 1.0, 1.0]], cubic(10.0, true));
        let (_, forces, w) = force(&mut f, None, &mut FfEvaluator::plain(Push));
        assert_eq!(forces[[0, 0]], 1.0);
        assert!(w.is_none());
    }
}
