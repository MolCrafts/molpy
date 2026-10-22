//! The currency of the interpolation seam: one bonded term, named by its atoms.

use molrs::core::TypeName;

/// One bonded term awaiting parameters: its arity-tagged endpoint atom types.
///
/// Handed to a [`ParameterInterpolator`](super::ParameterInterpolator) when no
/// force-field table covers the term. Kept small and owned so an interpolator
/// needs no access to the molecular graph.
///
/// Proper terms ([`Bond`](Self::Bond), [`Angle`](Self::Angle),
/// [`Dihedral`](Self::Dihedral)) are **reversal-symmetric** — `i-j-k-l` and
/// `l-k-j-i` are the same term — and their slot order is the chain along the
/// bonds. An [`Improper`](Self::Improper) is not: see its own note.
///
/// [`BondedTerm`] is not an OPLS thing: it is the query
/// type of the generic [`ParameterInterpolator`](super::ParameterInterpolator)
/// seam, and GAFF speaks it too. A term is its **atom-type names**, in the order
/// the force field writes them; which force field named them is not this type's
/// business.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BondedTerm {
    /// A bond: the two endpoint atom types.
    Bond([String; 2]),
    /// An angle: the three atom types, **vertex in the middle**.
    Angle([String; 3]),
    /// A dihedral: the four atom types along the chain, inner pair in the middle.
    Dihedral([String; 4]),
    /// An improper: the four atom types with the **centre third** (`i-j-k-l`,
    /// `k` central), which is AMBER's slot order and the order
    /// [`ImproperPeriodic`](crate::ff::potential::improper::ImproperPeriodic)
    /// reads.
    ///
    /// The three peripherals are an unordered **set** — an improper is a
    /// planarity constraint on a centre, not a walk along bonds — so a matcher
    /// must try them against a row's peripheral slots in any order, and must not
    /// simply reverse the quartet the way it would a proper term.
    Improper([String; 4]),
}

impl BondedTerm {
    /// The centre atom type of an [`Improper`](Self::Improper), or `None` for a
    /// proper term (which has no distinguished centre).
    pub fn improper_centre(&self) -> Option<&str> {
        match self {
            Self::Improper(types) => Some(&types[2]),
            _ => None,
        }
    }

    /// The three peripheral atom types of an [`Improper`](Self::Improper), in the
    /// term's own slot order, or `None` for a proper term.
    pub fn improper_peripherals(&self) -> Option<[&str; 3]> {
        match self {
            Self::Improper(t) => Some([t[0].as_str(), t[1].as_str(), t[3].as_str()]),
            _ => None,
        }
    }

    /// The endpoint atom types of this term's force-field type, in the
    /// orientation the type is stored in.
    ///
    /// A proper term is reversal-symmetric, so it is stored in
    /// [`TypeName::orient`]'s spelling — the smaller of the forward and the
    /// reversed tuple, slot by slot: both spellings of one bond, angle or
    /// torsion give one tuple, and so one type. An improper keeps its slot
    /// order (its centre third).
    pub fn endpoints(&self) -> Vec<&str> {
        let slots: Vec<&str> = match self {
            Self::Bond(t) => t.iter().map(String::as_str).collect(),
            Self::Angle(t) => t.iter().map(String::as_str).collect(),
            Self::Dihedral(t) => t.iter().map(String::as_str).collect(),
            Self::Improper(t) => return t.iter().map(String::as_str).collect(),
        };
        TypeName::orient(&slots)
    }

    /// The force-field type name of this term: [`TypeName::join`] of its
    /// [`endpoints`](Self::endpoints).
    ///
    /// # Errors
    ///
    /// An atom type containing `@`, which [`TypeName::join`] refuses.
    pub fn type_name(&self) -> Result<TypeName, String> {
        TypeName::join(&self.endpoints())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_improper_names_its_centre_third() {
        let improper = BondedTerm::Improper([
            "ca".to_owned(),
            "ca".to_owned(),
            "ca".to_owned(),
            "ha".to_owned(),
        ]);
        assert_eq!(improper.improper_centre(), Some("ca"));
        assert_eq!(improper.improper_peripherals(), Some(["ca", "ca", "ha"]));
    }

    /// Both spellings of one torsion give one endpoint tuple — the smaller
    /// one, slot by slot — and one name.
    #[test]
    fn a_dihedral_takes_the_smaller_of_its_two_orientations() {
        let quartet = |t: [&str; 4]| BondedTerm::Dihedral(t.map(str::to_owned));
        let forward = quartet(["oh", "c3", "c3", "hc"]);
        let backward = quartet(["hc", "c3", "c3", "oh"]);
        assert_eq!(forward.endpoints(), vec!["hc", "c3", "c3", "oh"]);
        assert_eq!(backward.endpoints(), vec!["hc", "c3", "c3", "oh"]);
        assert_eq!(forward.type_name().unwrap().as_str(), "hc-c3-c3-oh");
    }

    /// A bond and an angle are reversal-symmetric too: one spelling, one type,
    /// whichever atom order the graph stores the term in.
    #[test]
    fn a_bond_and_an_angle_take_the_smaller_orientation() {
        let bond = BondedTerm::Bond(["oh".to_owned(), "c3".to_owned()]);
        assert_eq!(bond.endpoints(), vec!["c3", "oh"]);
        assert_eq!(bond.type_name().unwrap().as_str(), "c3-oh");
        let angle = BondedTerm::Angle(["oh", "c3", "hc"].map(str::to_owned));
        assert_eq!(angle.endpoints(), vec!["hc", "c3", "oh"]);
    }

    /// An improper is a planarity constraint on a centre, not a chain: its
    /// slot order is kept.
    #[test]
    fn an_improper_keeps_its_slot_order() {
        let improper = BondedTerm::Improper(["ha", "ca", "ca", "ca"].map(str::to_owned));
        assert_eq!(improper.endpoints(), vec!["ha", "ca", "ca", "ca"]);
    }

    #[test]
    fn a_proper_term_has_no_centre() {
        let bond = BondedTerm::Bond(["c3".to_owned(), "hc".to_owned()]);
        assert_eq!(bond.improper_centre(), None);
        assert_eq!(bond.improper_peripherals(), None);
    }
}
