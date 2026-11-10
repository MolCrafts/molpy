//! A parsed string as the polymer model of [`crate::builder`].

use super::{BigSmilesIr, FragmentSpec, MoleculeSpec, ObjectSpec, PartSpec, TerminalSpec};
use crate::builder::{
    ChainSize, Fragment, Measure, Molecule, Part, PolymerError, SiteWeight, StochasticObject,
    System, Terminal,
};
use crate::core::BondNumber;
use crate::io::cgsmiles::templates::port_kind;
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind, Span};

impl BigSmilesIr {
    /// Every molecule as a [`Molecule`], in written order.
    ///
    /// Fragments are named by position, so a sample's `bead_type`s say where
    /// each fragment was written: `"{m}.{p}"` for the SMILES piece `p` of
    /// molecule `m`, `"{m}.{p}.r{k}"` and `"{m}.{p}.e{k}"` for repeat unit
    /// and end group `k` of the object `p`. Size distributions count the
    /// object's heavy-atom mass.
    ///
    /// # Errors
    ///
    /// What building a fragment template raises, unchanged;
    /// [`SmilesErrorKind::BigBuild`] for a model the string spells but the
    /// builder refuses (pieces that do not join, an end group with two
    /// ports, two different fragments under one name, …).
    pub fn to_molecules(&self) -> Result<Vec<Molecule>, SmilesError> {
        self.molecules
            .iter()
            .enumerate()
            .map(|(m, spec)| self.molecule(m, spec))
            .collect()
    }

    /// The whole string as a [`System`]: each molecule in its written
    /// amount, amounts counting heavy-atom mass.
    ///
    /// # Errors
    ///
    /// What [`to_molecules`](Self::to_molecules) raises;
    /// [`SmilesErrorKind::BigBuild`] for a molecule without an amount
    /// `.|…|`, or amounts the system refuses (none absolute, percentages
    /// summing to 100 or more).
    pub fn to_system(&self) -> Result<System, SmilesError> {
        let molecules = self.to_molecules()?;
        let components = molecules
            .into_iter()
            .zip(&self.molecules)
            .enumerate()
            .map(|(m, (molecule, spec))| {
                spec.amount.map(|a| (molecule, a)).ok_or_else(|| {
                    self.build_error(
                        format!("molecule {m} has no system amount — write '.|N|' after it"),
                        spec.span,
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        System::new(components, Measure::HeavyAtomMass)
            .map_err(|e| self.build_error(e.to_string(), self.span))
    }

    fn molecule(&self, m: usize, spec: &MoleculeSpec) -> Result<Molecule, SmilesError> {
        let parts = spec
            .parts
            .iter()
            .enumerate()
            .map(|(p, part)| match part {
                PartSpec::Smiles(body) => Ok(Part::Fixed(self.fragment(format!("{m}.{p}"), body)?)),
                PartSpec::Stochastic(object) => {
                    Ok(Part::Stochastic(self.object(&format!("{m}.{p}"), object)?))
                }
            })
            .collect::<Result<Vec<_>, SmilesError>>()?;
        Molecule::new(parts).map_err(|e| self.build_error(e.to_string(), spec.span))
    }

    fn object(&self, name: &str, spec: &ObjectSpec) -> Result<StochasticObject, SmilesError> {
        let refused = |e: PolymerError| self.build_error(e.to_string(), spec.span);
        let fragments = |tag: char, bodies: &[FragmentSpec]| {
            bodies
                .iter()
                .enumerate()
                .map(|(k, body)| self.fragment(format!("{name}.{tag}{k}"), body))
                .collect::<Result<Vec<_>, SmilesError>>()
        };
        let terminal = |spec: &Option<TerminalSpec>| {
            spec.as_ref()
                .map(|t| {
                    Terminal::new(
                        port_kind(t.kind),
                        t.label.clone(),
                        t.order.map_or(BondNumber::Single, |k| k.bond_number()),
                        // A single weight on a terminal has nothing to weigh.
                        t.weights.clone().filter(|w| w.len() > 1),
                    )
                })
                .transpose()
        };
        let size = spec
            .distribution
            .map(|d| ChainSize::new(d, Measure::HeavyAtomMass))
            .transpose()
            .map_err(refused)?;
        StochasticObject::new(
            fragments('r', &spec.repeat_units)?,
            fragments('e', &spec.end_groups)?,
            terminal(&spec.left).map_err(refused)?,
            terminal(&spec.right).map_err(refused)?,
            size,
        )
        .map_err(refused)
    }

    fn fragment(&self, name: String, body: &FragmentSpec) -> Result<Fragment, SmilesError> {
        // The body parsed; a conversion refusal is still about its text.
        let template = body
            .ir
            .to_template()
            .map_err(|e| SmilesError::new(e.kind, body.span, &self.input, Notation::BigSmiles))?;
        let weights = body
            .weights
            .iter()
            .map(|w| match w.as_deref() {
                None => SiteWeight::default(),
                Some([single]) => SiteWeight::Single(*single),
                Some(list) => SiteWeight::List(list.to_vec()),
            })
            .collect();
        Fragment::new(name, template, weights)
            .map_err(|e| self.build_error(e.to_string(), body.span))
    }

    /// A refusal of the polymer model, spanned at the text that spells the
    /// refused piece.
    fn build_error(&self, reason: String, span: Span) -> SmilesError {
        SmilesError::new(
            SmilesErrorKind::BigBuild(reason),
            span,
            &self.input,
            Notation::BigSmiles,
        )
    }
}
