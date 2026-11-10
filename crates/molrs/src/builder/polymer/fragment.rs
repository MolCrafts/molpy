//! A fragment: one ported template the generator places, with the generation
//! weight of each of its ports.

use std::sync::Arc;

use super::PolymerError;
use super::size::Measure;
use crate::core::{Atomistic, Element, Port};
use crate::op::F;
use crate::perceive::add_hydrogens;

/// The generation weight a port carries — generative BigSMILES `[<|w|]` or
/// `[<|w₁ w₂ …|]`.
///
/// A single weight is the port's reactivity: how likely it is picked among
/// the open ports, and how likely it is picked as a partner among the
/// repeat-unit ports. A list gives, position by position, the weight of
/// pairing with each port of the stochastic object (repeat units, then end
/// groups, in written order); its sum is the port's reactivity.
#[derive(Debug, Clone, PartialEq)]
pub enum SiteWeight {
    /// One reactivity weight, `≥ 0`.
    Single(F),
    /// One pairing weight per port of the stochastic object, each `≥ 0`.
    List(Vec<F>),
}

impl Default for SiteWeight {
    /// The weight of a port the notation gives none: `|1.0|`.
    fn default() -> Self {
        SiteWeight::Single(1.0)
    }
}

impl SiteWeight {
    /// How likely this port is picked: the single weight, or the sum of the
    /// list.
    pub fn reactivity(&self) -> F {
        match self {
            SiteWeight::Single(w) => *w,
            SiteWeight::List(ws) => ws.iter().sum(),
        }
    }

    fn validate(&self) -> Result<(), String> {
        let ok = |w: &F| w.is_finite() && *w >= 0.0;
        match self {
            SiteWeight::Single(w) if ok(w) => Ok(()),
            SiteWeight::List(ws) if !ws.is_empty() && ws.iter().all(ok) => Ok(()),
            other => Err(format!("weight {other:?} is not finite and non-negative")),
        }
    }
}

/// One ported template — a repeat unit, an end group, or a fixed piece
/// between stochastic objects — under the name the site graph refers to it
/// by.
///
/// Cloning is cheap and keeps the identity: two clones are the same fragment
/// ([`Fragment::is_same`]), which is what lets one end group cap several
/// objects of a molecule under one name. Two fragments built separately are
/// different fragments even from equal inputs — a template has no equality
/// to compare them by — so a fragment used in several places is built once
/// and cloned.
#[derive(Debug, Clone)]
pub struct Fragment(Arc<FragmentData>);

#[derive(Debug)]
struct FragmentData {
    name: String,
    template: Atomistic,
    ports: Vec<Port>,
    weights: Vec<SiteWeight>,
    heavy_atom_mass: F,
    mass: F,
}

impl Fragment {
    /// A fragment named `name` over `template`, its ports weighted by `weights`
    /// in port order (`template.port_ids()` order, the order a fragment body
    /// writes its descriptors in).
    ///
    /// The template is a capped fragment as
    /// [`FragmentIr::to_template`](crate::io::smiles::FragmentIr::to_template)
    /// builds it: heavy atoms plus one leaving hydrogen per port, implicit
    /// hydrogens not yet added. Both masses are fixed here — the heavy-atom
    /// mass sums every non-hydrogen atom; the mass adds the hydrogens the
    /// fragment carries once each port is bonded (each port's handle leaves).
    ///
    /// # Errors
    ///
    /// [`PolymerError::InvalidFragment`] for an empty name, a weight count other
    /// than the port count, a negative or non-finite weight, an atom without
    /// an element, or a template whose hydrogens cannot be counted.
    pub fn new(
        name: impl Into<String>,
        template: Atomistic,
        weights: Vec<SiteWeight>,
    ) -> Result<Self, PolymerError> {
        let name = name.into();
        let invalid = |reason: String| PolymerError::InvalidFragment {
            fragment: name.clone(),
            reason,
        };
        if name.is_empty() {
            return Err(invalid("a fragment needs a name".to_owned()));
        }
        let ports = template
            .port_ids()
            .map(|id| template.get_port(id))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| invalid(e.to_string()))?;
        if weights.len() != ports.len() {
            return Err(invalid(format!(
                "{} weights for {} ports",
                weights.len(),
                ports.len()
            )));
        }
        for weight in &weights {
            weight.validate().map_err(invalid)?;
        }
        let heavy_atom_mass = summed_mass(&template, |z| z > 1).map_err(invalid)?;
        let mass = bonded_mass(&template, &ports).map_err(invalid)?;
        Ok(Self(Arc::new(FragmentData {
            name,
            template,
            ports,
            weights,
            heavy_atom_mass,
            mass,
        })))
    }

    /// The name sites refer to this fragment by.
    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// The ported template.
    pub fn template(&self) -> &Atomistic {
        &self.0.template
    }

    /// The ports, in template port order.
    pub fn ports(&self) -> &[Port] {
        &self.0.ports
    }

    /// The generation weight of each port, in port order.
    pub fn weights(&self) -> &[SiteWeight] {
        &self.0.weights
    }

    /// The summed mass of the non-hydrogen atoms, g/mol.
    pub fn heavy_atom_mass(&self) -> F {
        self.0.heavy_atom_mass
    }

    /// The mass the fragment contributes to a molecule in which every port is
    /// bonded: heavy atoms plus the hydrogens they keep, g/mol.
    pub fn mass(&self) -> F {
        self.0.mass
    }

    /// What one copy adds to a size counted in `measure`; `repeat_unit`
    /// says the copy is placed as a repeat unit (an end group or a fixed
    /// fragment counts no repeat unit).
    pub(super) fn size_in(&self, measure: Measure, repeat_unit: bool) -> F {
        match measure {
            Measure::RepeatUnits => F::from(u8::from(repeat_unit)),
            Measure::HeavyAtomMass => self.heavy_atom_mass(),
            Measure::Mass => self.mass(),
        }
    }

    /// Whether `other` is this fragment (a clone of it), not merely a fragment of
    /// the same name.
    pub fn is_same(&self, other: &Fragment) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// The summed atomic mass of the atoms whose atomic number passes `keep`.
fn summed_mass(mol: &Atomistic, keep: impl Fn(u8) -> bool) -> Result<F, String> {
    let mut total = 0.0;
    for id in mol.atom_ids() {
        let atom = mol.get_atom(id).map_err(|e| e.to_string())?;
        let z = atom
            .atomic_number()
            .ok_or_else(|| format!("atom {id:?} has no atomic number"))?;
        if !keep(z) {
            continue;
        }
        let element =
            Element::by_number(z).ok_or_else(|| format!("atom {id:?} names no element"))?;
        total += F::from(element.atomic_mass());
    }
    Ok(total)
}

/// The mass with the implicit hydrogens filled in, each port counted as its
/// future bond, less the handles that leave when the ports bond.
fn bonded_mass(template: &Atomistic, ports: &[Port]) -> Result<F, String> {
    let filled = add_hydrogens(template).map_err(|e| e.to_string())?;
    let all = summed_mass(&filled, |_| true)?;
    let mut leaving = 0.0;
    for port in ports {
        for id in template.leaving_group(port).map_err(|e| e.to_string())? {
            let z = template
                .get_atom(id)
                .map_err(|e| e.to_string())?
                .atomic_number()
                .ok_or_else(|| format!("atom {id:?} has no atomic number"))?;
            let element =
                Element::by_number(z).ok_or_else(|| format!("atom {id:?} names no element"))?;
            leaving += F::from(element.atomic_mass());
        }
    }
    Ok(all - leaving)
}

#[cfg(all(test, feature = "smiles"))]
mod tests {
    use super::*;
    use crate::io::smiles::SmilesIr;

    fn template(body: &str) -> Atomistic {
        SmilesIr::from_fragment(body)
            .unwrap()
            .to_template()
            .unwrap()
    }

    #[test]
    fn ethylene_unit_masses_count_heavy_atoms_and_kept_hydrogens() {
        let fragment =
            Fragment::new("E", template("[$]CC[$]"), vec![SiteWeight::default(); 2]).unwrap();
        let c = F::from(Element::C.atomic_mass());
        let h = F::from(Element::H.atomic_mass());
        assert!((fragment.heavy_atom_mass() - 2.0 * c).abs() < 1e-9);
        // –CH2–CH2–: each carbon keeps two hydrogens once both ports bond.
        assert!((fragment.mass() - (2.0 * c + 4.0 * h)).abs() < 1e-9);
    }

    #[test]
    fn double_bond_port_reserves_its_valence() {
        // =CH–CH= keeps one hydrogen per carbon. A leading descriptor's order
        // is written after its bracket, a trailing one's before it.
        let fragment =
            Fragment::new("V", template("[$]=CC=[$]"), vec![SiteWeight::default(); 2]).unwrap();
        let c = F::from(Element::C.atomic_mass());
        let h = F::from(Element::H.atomic_mass());
        assert!((fragment.mass() - (2.0 * c + 2.0 * h)).abs() < 1e-9);
    }

    #[test]
    fn weight_count_must_match_port_count() {
        let err =
            Fragment::new("E", template("[$]CC[$]"), vec![SiteWeight::default()]).unwrap_err();
        assert!(matches!(err, PolymerError::InvalidFragment { .. }));
    }

    #[test]
    fn negative_weight_is_refused() {
        let err = Fragment::new(
            "E",
            template("[$]CC[$]"),
            vec![SiteWeight::Single(-1.0), SiteWeight::default()],
        )
        .unwrap_err();
        assert!(matches!(err, PolymerError::InvalidFragment { .. }));
    }

    #[test]
    fn clones_are_the_same_unit_and_rebuilds_are_not() {
        let a = Fragment::new("E", template("[$]CC[$]"), vec![SiteWeight::default(); 2]).unwrap();
        let b = Fragment::new("E", template("[$]CC[$]"), vec![SiteWeight::default(); 2]).unwrap();
        assert!(a.is_same(&a.clone()));
        assert!(!a.is_same(&b));
    }

    #[test]
    fn list_weight_reactivity_is_its_sum() {
        assert_eq!(SiteWeight::List(vec![0.25, 0.5, 0.25]).reactivity(), 1.0);
        assert_eq!(SiteWeight::default().reactivity(), 1.0);
    }
}
