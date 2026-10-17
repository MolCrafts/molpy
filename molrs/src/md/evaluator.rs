//! A force-field kernel bound to the `force` phase.
//!
//! Pair kernels that are not tied to a compiled pair list read
//! [`Scratch::pairs`], which the neighbour component published earlier in
//! the step. Bonded kernels and compiled pair lists call
//! [`Potential::evaluate`] on the coordinates alone.

use molrs::core::Frame;
use molrs::ff::potential::{
    Accumulator, ForceTerm, IndexedTerms, PairDriven, Potential, SpecialWeights,
};
use molrs::op::F;

use super::columns::{self, read_pos};
use super::component::{Component, Evaluator, Flags, Kind};
use super::error::MdError;
use super::phase::Phase;
use super::scratch::Scratch;

/// One potential term. Several of these may be mounted; each adds into the
/// forces and `meta["energy"]` that [`ZeroForces`](crate::md::infra::ZeroForces)
/// cleared.
pub struct FfEvaluator {
    term: ForceTerm,
    special: SpecialWeights,
    factors: Vec<F>,
    buf: Vec<F>,
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
        Self {
            term,
            special: SpecialWeights::default(),
            factors: Vec::new(),
            buf: Vec::new(),
        }
    }

    pub(crate) fn flags(&self) -> Flags {
        let pair = matches!(self.term, ForceTerm::Pair(_));
        let frozen = self.term.binds_a_fixed_pair_list();
        Flags {
            kind: Kind::Evaluator,
            consumes_neighbors: pair && !frozen,
            frozen_pairs: frozen,
            dt: None,
        }
    }

    fn add_onto_frame(&self, frame: &mut Frame, energy: F) -> Result<(), MdError> {
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
        let n = pos.nrows();
        self.buf.clear();
        self.buf.resize(n * 3, 0.0);
        let flat: Vec<F> = pos.iter().copied().collect();
        let energy = match &mut self.term {
            ForceTerm::Pair(pot) => {
                let Some(pairs) = scratch.pairs.as_ref() else {
                    return Err(MdError::Invalid(
                        "pair evaluator ran with no neighbour list published this step; \
                         mount one with set_neighbor"
                            .into(),
                    ));
                };
                let factors = self.special.factors_for(pairs, n, &[], &mut self.factors);
                let mut acc = Accumulator::adding_into(&mut self.buf);
                pot.evaluate_pairs(&flat, pairs, factors, &mut acc);
                acc.energy()
            }
            ForceTerm::Indexed(pot) => {
                let mut acc = Accumulator::adding_into(&mut self.buf);
                pot.evaluate(&flat, &mut acc);
                acc.energy()
            }
            ForceTerm::Plain(pot) => {
                let mut acc = Accumulator::adding_into(&mut self.buf);
                pot.evaluate(&flat, &mut acc);
                acc.energy()
            }
        };
        self.add_onto_frame(frame, energy)
    }
}

impl Evaluator for FfEvaluator {}
