//! A parsed string as the polymer model of [`crate::builder`].

use super::{BigSmilesIr, FragmentSpec, MoleculeSpec, ObjectSpec, PartSpec, TerminalSpec};
use crate::builder::{
    ChainSize, Fragment, Measure, Molecule, Part, PolymerError, SiteWeight, StochasticObject,
    System, Terminal,
};
use crate::core::{Atomistic, BondNumber, PortKind};
use crate::io::cgsmiles::templates::port_kind;
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind, SmilesIr, Span};

impl BigSmilesIr {
    /// Every molecule as a [`Molecule`], in written order.
    ///
    /// Fragments are named by position, so a sample's `bead_type`s say where
    /// each fragment was written: `"{m}.{p}"` for the SMILES piece `p` of
    /// molecule `m`, `"{m}.{p}.r{k}"` and `"{m}.{p}.e{k}"` for repeat unit
    /// and end group `k` of the object `p`. An object written without end
    /// groups ends in hydrogen: it gets one implicit hydrogen end group per
    /// kind of single-bond repeat-unit port, `"{m}.{p}.h{k}"`. Size distributions count the
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
                PartSpec::Smiles(body) => {
                    Ok(Part::Fixed(self.fragment(format!("{m}.{p}"), body, 0)?))
                }
                PartSpec::Stochastic(object) => {
                    Ok(Part::Stochastic(self.object(&format!("{m}.{p}"), object)?))
                }
            })
            .collect::<Result<Vec<_>, SmilesError>>()?;
        Molecule::new(parts).map_err(|e| self.build_error(e.to_string(), spec.span))
    }

    fn object(&self, name: &str, spec: &ObjectSpec) -> Result<StochasticObject, SmilesError> {
        let refused = |e: PolymerError| self.build_error(e.to_string(), spec.span);
        // An object written without end groups ends in hydrogen: one
        // implicit cap per kind of repeat-unit port. The caps come after
        // every written port, so written list weights keep their positions
        // and are padded with zeros — a list never chooses an implicit cap.
        let caps = if spec.end_groups.is_empty() {
            self.hydrogen_caps(name, spec)?
        } else {
            Vec::new()
        };
        let fragments = |tag: char, bodies: &[FragmentSpec]| {
            bodies
                .iter()
                .enumerate()
                .map(|(k, body)| self.fragment(format!("{name}.{tag}{k}"), body, caps.len()))
                .collect::<Result<Vec<_>, SmilesError>>()
        };
        let pad = |weights: Vec<f64>| {
            let mut weights = weights;
            weights.resize(weights.len() + caps.len(), 0.0);
            weights
        };
        let terminal = |spec: &Option<TerminalSpec>| {
            spec.as_ref()
                .map(|t| {
                    Terminal::new(
                        port_kind(t.kind),
                        t.label.clone(),
                        t.order.map_or(BondNumber::Single, |k| k.bond_number()),
                        // A single weight on a terminal has nothing to weigh.
                        t.weights.clone().filter(|w| w.len() > 1).map(pad),
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
            if caps.is_empty() {
                fragments('e', &spec.end_groups)?
            } else {
                caps.clone()
            },
            terminal(&spec.left).map_err(refused)?,
            terminal(&spec.right).map_err(refused)?,
            size,
        )
        .map_err(refused)
    }

    /// The implicit end groups of an object written without any: a hydrogen
    /// for each distinct single-bond port its repeat units carry, named
    /// `"{name}.h{k}"`. A port of higher order gets none — hydrogen forms
    /// one bond — and stays an open port growth must close another way.
    fn hydrogen_caps(&self, name: &str, spec: &ObjectSpec) -> Result<Vec<Fragment>, SmilesError> {
        let mut seen: Vec<(PortKind, String)> = Vec::new();
        for body in &spec.repeat_units {
            let template = self.template(body)?;
            for id in template.port_ids() {
                let port = template
                    .get_port(id)
                    .map_err(|e| self.build_error(e.to_string(), body.span))?;
                let key = (port.kind.complement(), port.label);
                if port.order == BondNumber::Single && !seen.contains(&key) {
                    seen.push(key);
                }
            }
        }
        seen.iter()
            .enumerate()
            .map(|(k, (kind, label))| {
                let refused = |e: SmilesError| {
                    SmilesError::new(e.kind, spec.span, &self.input, Notation::BigSmiles)
                };
                let template = SmilesIr::from_fragment(&format!("[H][{}{label}]", kind.as_str()))
                    .and_then(|ir| ir.to_template())
                    .map_err(refused)?;
                Fragment::new(
                    format!("{name}.h{k}"),
                    template,
                    vec![SiteWeight::default()],
                )
                .map_err(|e| self.build_error(e.to_string(), spec.span))
            })
            .collect()
    }

    /// A body's ported template; a conversion refusal is still about its
    /// text.
    fn template(&self, body: &FragmentSpec) -> Result<Atomistic, SmilesError> {
        body.ir
            .to_template()
            .map_err(|e| SmilesError::new(e.kind, body.span, &self.input, Notation::BigSmiles))
    }

    /// A written fragment; its list weights padded with `pad` zeros for the
    /// implicit caps that follow the written ports.
    fn fragment(
        &self,
        name: String,
        body: &FragmentSpec,
        pad: usize,
    ) -> Result<Fragment, SmilesError> {
        let template = self.template(body)?;
        let weights = body
            .weights
            .iter()
            .map(|w| match w.as_deref() {
                None => SiteWeight::default(),
                Some([single]) => SiteWeight::Single(*single),
                Some(list) => {
                    let mut list = list.to_vec();
                    list.resize(list.len() + pad, 0.0);
                    SiteWeight::List(list)
                }
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
