//! Bond class and localized bond number — the two orthogonal facts about a bond: [`BondOrder`] and [`BondNumber`].

use crate::core::MolRsError;
use crate::core::keys;
use crate::core::{KindId, MolGraph, PropValue, RelationId};

/// The chemical class of a bond.
///
/// Stored under [`keys::BOND_TYPE`] as its
/// [`code`](BondOrder::code).
///
/// A bond carries **two** independent properties, and conflating them is the
/// defect this pair of types exists to prevent:
///
/// * `BondOrder` — what *kind* of bond it is. Aromatic is one of the kinds,
///   peer to single / double / triple. A class, never a number.
/// * [`BondNumber`] — the integer bond number a localized Lewis / Kekulé
///   structure gives it. Always an integer.
///
/// An aromatic bond is therefore `BondOrder::Aromatic` **and** a `BondNumber` of
/// `Single` or `Double`: benzene's ring is six `Aromatic` types over an
/// alternating `1,2,1,2,1,2` of numbers. There is no fractional bond number —
/// `1.5` said "aromatic" and "one-and-a-half bonds" at once, so a consumer
/// reading the number could not tell which was meant and drew six double bonds.
///
/// Fractional bond orders from electronic structure (Wiberg, Mayer, resonance
/// averages) are real quantities, but they are *computed properties* with their
/// own keys — never these two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum BondOrder {
    /// The input did not say, and perception has not run.
    #[default]
    Unknown,
    Single,
    Double,
    Triple,
    /// Part of an aromatic system. The code `4` is this protocol's aromatic
    /// marker — not a quadruple bond, which is a [`BondNumber`], not a type.
    Aromatic,
}

/// The integer bond number of a localized Lewis / Kekulé structure.
///
/// Stored under [`keys::BOND_NUMBER`] as its
/// [`code`](BondNumber::code). Orthogonal to the bond's class, [`BondOrder`]:
/// an aromatic bond is `BondOrder::Aromatic` with a number of `Single` or
/// `Double`, never a fractional one (see [`BondOrder`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub enum BondNumber {
    /// Not yet assigned — an aromatic bond before kekulization, or an input
    /// that never stated one. Not a legal state for a standardized molecule.
    #[default]
    Unknown,
    Single,
    Double,
    Triple,
    Quadruple,
}

impl BondOrder {
    /// The stored code: 0 unknown, 1 single, 2 double, 3 triple, 4 aromatic.
    pub fn code(self) -> u32 {
        match self {
            BondOrder::Unknown => 0,
            BondOrder::Single => 1,
            BondOrder::Double => 2,
            BondOrder::Triple => 3,
            BondOrder::Aromatic => 4,
        }
    }

    /// Read a stored code. Anything outside `0..=4` is [`BondOrder::Unknown`] —
    /// an unreadable class is "we do not know", never a guess.
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => BondOrder::Single,
            2 => BondOrder::Double,
            3 => BondOrder::Triple,
            4 => BondOrder::Aromatic,
            _ => BondOrder::Unknown,
        }
    }

    /// Read the class from a stored prop; missing or non-numeric is `Unknown`.
    pub fn from_prop(prop: Option<&PropValue>) -> Self {
        match prop.and_then(PropValue::as_f64) {
            Some(v) if v >= 0.0 => BondOrder::from_code(v.round() as u32),
            _ => BondOrder::Unknown,
        }
    }

    /// Is this an aromatic bond? The question a renderer must ask *before* it
    /// asks about the number.
    pub fn is_aromatic(self) -> bool {
        matches!(self, BondOrder::Aromatic)
    }

    /// The bond number a non-aromatic class implies.
    ///
    /// Single is one bond, double is two — the two facts agree by construction
    /// for everything but aromatic, whose number only a Kekulé assignment can
    /// decide. `None` for `Aromatic` and `Unknown`, because neither implies one.
    pub fn implied_number(self) -> Option<BondNumber> {
        match self {
            BondOrder::Single => Some(BondNumber::Single),
            BondOrder::Double => Some(BondNumber::Double),
            BondOrder::Triple => Some(BondNumber::Triple),
            BondOrder::Aromatic | BondOrder::Unknown => None,
        }
    }
}

impl BondNumber {
    /// The stored code: 0 unknown, 1 single, 2 double, 3 triple, 4 quadruple.
    pub fn code(self) -> u32 {
        match self {
            BondNumber::Unknown => 0,
            BondNumber::Single => 1,
            BondNumber::Double => 2,
            BondNumber::Triple => 3,
            BondNumber::Quadruple => 4,
        }
    }

    /// Read a stored code. Anything outside `0..=4` is [`BondNumber::Unknown`].
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => BondNumber::Single,
            2 => BondNumber::Double,
            3 => BondNumber::Triple,
            4 => BondNumber::Quadruple,
            _ => BondNumber::Unknown,
        }
    }

    /// Read the number from a stored prop; missing or non-numeric is `Unknown`.
    pub fn from_prop(prop: Option<&PropValue>) -> Self {
        match prop.and_then(PropValue::as_f64) {
            Some(v) if v >= 0.0 => BondNumber::from_code(v.round() as u32),
            _ => BondNumber::Unknown,
        }
    }

    /// The bond class a localized number implies: the inverse of
    /// [`BondOrder::implied_number`], and the one number → class map.
    ///
    /// `Quadruple` is classed [`BondOrder::Double`]: no quadruple class exists
    /// (the class code `4` is aromatic), so it takes the highest multiple-bond
    /// class below it, as the SMILES reader classes `$`. `Unknown` implies
    /// [`BondOrder::Unknown`].
    pub fn implied_type(self) -> BondOrder {
        match self {
            BondNumber::Single => BondOrder::Single,
            BondNumber::Double | BondNumber::Quadruple => BondOrder::Double,
            BondNumber::Triple => BondOrder::Triple,
            BondNumber::Unknown => BondOrder::Unknown,
        }
    }

    /// The number as a count, for valence sums. `Unknown` counts as zero; a
    /// caller that cannot tolerate that must check for it.
    pub fn count(self) -> u32 {
        match self {
            BondNumber::Unknown => 0,
            _ => self.code(),
        }
    }
}

// Graph props are `Int`; the schema declares the *frame* columns `UInt`, and
// `to_frame` re-types on the way out. Writing the code in exactly one place is
// what keeps the two from drifting.
impl From<BondOrder> for PropValue {
    fn from(v: BondOrder) -> Self {
        PropValue::Int(v.code() as i32)
    }
}

impl From<BondNumber> for PropValue {
    fn from(v: BondNumber) -> Self {
        PropValue::Int(v.code() as i32)
    }
}

/// Stamp both facts about a bond — its class and its localized number — onto
/// relation `id` of `kind` in `graph`.
///
/// The two keys are written together because they are only meaningful together:
/// a class without a number leaves the bond un-standardized, and a number
/// without a class leaves a renderer no way to tell aromatic from double. It
/// lives here, beside the vocabulary, because two leaves write it —
/// [`Atomistic::set_bond_class`](crate::core::Atomistic::set_bond_class)
/// and the port join's new bonds (`MolGraph::link`).
///
/// # Errors
///
/// Returns [`MolRsError::NotFound`] when `kind` is unregistered or `id` names
/// no live relation of it, and [`MolRsError::Validation`] when the store
/// refuses either value.
pub(crate) fn stamp_bond_class(
    graph: &mut MolGraph,
    kind: KindId,
    id: RelationId,
    bond_type: BondOrder,
    bond_number: BondNumber,
) -> Result<(), MolRsError> {
    graph.set_relation_prop(kind, id, keys::BOND_TYPE, bond_type)?;
    graph.set_relation_prop(kind, id, keys::BOND_NUMBER, bond_number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MolGraph;
    use crate::core::keys;
    use crate::props;

    /// The two-key write is one function because a class without a number
    /// leaves the bond un-standardized: both keys land, or neither does.
    #[test]
    fn stamp_bond_class_sets_both_keys() {
        let mut graph = MolGraph::new();
        let kind = graph.register_kind("bonds", 2);
        let a = graph.add_node(props! {}).unwrap();
        let b = graph.add_node(props! {}).unwrap();
        let bid = graph.add_relation(kind, &[a, b], props! {}).unwrap();

        stamp_bond_class(
            &mut graph,
            kind,
            bid,
            BondOrder::Aromatic,
            BondNumber::Double,
        )
        .expect("both props are writable on a live relation");

        let rel = graph.get_relation(kind, bid).unwrap();
        assert_eq!(
            BondOrder::from_prop(rel.props.get(keys::BOND_TYPE)),
            BondOrder::Aromatic
        );
        assert_eq!(
            BondNumber::from_prop(rel.props.get(keys::BOND_NUMBER)),
            BondNumber::Double
        );
    }

    #[test]
    fn aromatic_is_a_type_not_a_number() {
        // The invariant this module exists for: `4` means aromatic as a type
        // and quadruple as a number, and neither of them is `1.5`.
        assert_eq!(BondOrder::Aromatic.code(), 4);
        assert_eq!(BondNumber::Quadruple.code(), 4);
        assert!(BondOrder::Aromatic.is_aromatic());
        assert_eq!(BondOrder::Aromatic.implied_number(), None);
    }

    #[test]
    fn a_plain_type_implies_its_own_number() {
        for (t, n) in [
            (BondOrder::Single, BondNumber::Single),
            (BondOrder::Double, BondNumber::Double),
            (BondOrder::Triple, BondNumber::Triple),
        ] {
            assert_eq!(t.implied_number(), Some(n));
            assert_eq!(t.code(), n.code());
        }
    }

    /// The one number → class map, the inverse of
    /// `BondOrder::implied_number`. Quadruple has no class of its own and is
    /// classed `Double`, as the SMILES reader classes `$`.
    #[test]
    fn bond_number_implied_type_maps_each_order() {
        for (n, t) in [
            (BondNumber::Single, BondOrder::Single),
            (BondNumber::Double, BondOrder::Double),
            (BondNumber::Triple, BondOrder::Triple),
            (BondNumber::Quadruple, BondOrder::Double),
        ] {
            assert_eq!(n.implied_type(), t, "{n:?}");
        }
    }

    #[test]
    fn an_unreadable_code_is_unknown_never_a_guess() {
        assert_eq!(BondOrder::from_code(9), BondOrder::Unknown);
        assert_eq!(BondNumber::from_code(9), BondNumber::Unknown);
        assert_eq!(BondOrder::from_prop(None), BondOrder::Unknown);
        assert_eq!(
            BondOrder::from_prop(Some(&PropValue::Str("ar".into()))),
            BondOrder::Unknown
        );
        assert_eq!(BondNumber::Unknown.count(), 0);
    }

    #[test]
    fn codes_round_trip() {
        for code in 0..=4u32 {
            assert_eq!(BondOrder::from_code(code).code(), code);
            assert_eq!(BondNumber::from_code(code).code(), code);
        }
    }

    #[test]
    fn a_fractional_prop_can_never_read_back_as_aromatic() {
        // A fractional order is no alias for aromaticity: 1.5 rounds to 2,
        // which is Double, not Aromatic. Nothing turns 1.5 into aromaticity.
        assert_eq!(
            BondOrder::from_prop(Some(&PropValue::F64(1.5))),
            BondOrder::Double
        );
    }
}
