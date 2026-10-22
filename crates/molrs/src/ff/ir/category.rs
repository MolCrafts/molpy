//! What a category is: how many endpoints its rows name, which Frame block
//! its terms live in, and which coordinate its energy is a function of.

use std::borrow::Cow;

use molrs::core::schema::block_names::{
    ANGLES, ATOMS, BONDS, CMAPS, CONSTRAINTS, DIHEDRALS, DRUDES, IMPROPERS, VIRTUAL_SITES,
};

/// How many endpoints a type row of the category names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arity {
    /// Exactly this many (0 ..= 5): `itom`, `jtom`, … in order.
    Exact(u8),
    /// A non-bonded pair: a self row `{a, a}` per atom type, and cross rows
    /// `{a, b}` where the force field gives a pair its own parameters. Its
    /// terms are whatever pairs a neighbour search turns up.
    SelfOrPair,
}

impl Arity {
    /// How many endpoint columns a type row carries (2 for a pair).
    pub fn endpoints(self) -> usize {
        match self {
            Arity::Exact(n) => n as usize,
            Arity::SelfOrPair => 2,
        }
    }
}

/// The coordinate a category's energy is a function of.
///
/// A [`ScalarForm`](crate::ff::potential::form_kernel::ScalarForm) is priced in
/// it, so it is also the name a Lepton expression reads (`r`, `theta`, `phi`,
/// `chi`), always in radians for an angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Coordinate {
    /// The category prices no energy (atom types, constraints, virtual
    /// sites); compiling it builds nothing.
    None,
    /// `r`, the distance between the two atoms.
    Distance,
    /// `theta`, the angle at the middle atom, in radians.
    Angle,
    /// `phi`, the signed dihedral of the four atoms in the order listed
    /// (IUPAC sign), in radians.
    Dihedral,
    /// The signed dihedral φ(I, J, K, L) of the four atoms in the order
    /// listed, in radians: a form's `q`, an expression's `phi`, and its
    /// `chi = abs(phi)` (LAMMPS `improper harmonic` prices χ = |φ|).
    Improper,
    /// The atoms' positions: an N-body term priced by a
    /// [`CompoundForm`](crate::ff::potential::form_kernel::CompoundForm).
    Compound,
}

impl Coordinate {
    /// The variables an expression reads the coordinate as.
    pub fn variables(self) -> &'static [&'static str] {
        match self {
            Coordinate::Distance => &["r"],
            Coordinate::Angle => &["theta"],
            Coordinate::Dihedral => &["phi"],
            Coordinate::Improper => &["phi", "chi"],
            Coordinate::None | Coordinate::Compound => &[],
        }
    }

    /// How many atoms define one value of the coordinate, for the scalar
    /// coordinates.
    pub fn atoms(self) -> Option<usize> {
        match self {
            Coordinate::Distance => Some(2),
            Coordinate::Angle => Some(3),
            Coordinate::Dihedral | Coordinate::Improper => Some(4),
            Coordinate::None | Coordinate::Compound => None,
        }
    }

    /// Whether this is one number per term (`r`, `theta`, `phi`, `chi`).
    pub fn is_scalar(self) -> bool {
        self.atoms().is_some()
    }
}

/// Which orders of a type row's endpoints name the same row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EndpointOrder {
    /// In order or reversed (a bond, an angle, a proper dihedral).
    Reversible,
    /// In order only (an improper, whose centre is a position).
    Ordered,
    /// Any order (a non-bonded pair).
    Unordered,
}

/// A category of the force-field IR.
///
/// The seven categories molrs prices (`atom`, `bond`, `angle`, `dihedral`,
/// `improper`, `pair`, `cmap`) and molrec's `constraint`, `drude` and
/// `virtual_site` are [`CategorySpec`]s like any other — a category added at
/// run time is registered through the same struct and gated by the same
/// fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CategorySpec {
    /// `^[a-z][a-z0-9_]*$`.
    pub name: Cow<'static, str>,
    pub arity: Arity,
    /// The Frame block whose rows the category prices, with columns
    /// `atomi`, … (its coordinate's atoms) and `type`. A style of the
    /// category contributes nothing when the block is absent or empty. A
    /// pair category's terms are the pairs of its block's atoms.
    pub block: Cow<'static, str>,
    pub coordinate: Coordinate,
    /// How a typifier matches a type row's endpoints to a term's atoms.
    pub order: EndpointOrder,
    /// Whether the category's terms exclude their atoms from the pair
    /// styles. Always `false`: no category excludes today.
    pub excludes: bool,
}

impl CategorySpec {
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        arity: Arity,
        block: impl Into<Cow<'static, str>>,
        coordinate: Coordinate,
        order: EndpointOrder,
    ) -> Self {
        Self {
            name: name.into(),
            arity,
            block: block.into(),
            coordinate,
            order,
            excludes: false,
        }
    }

    /// A category added at run time: `arity` atoms per term, its block
    /// `<name>s` (molrec's rule, so a record alone locates it). Its
    /// coordinate is [`Coordinate::Compound`] unless it names the geometric
    /// variable its arity has (`Distance` 2, `Angle` 3, `Dihedral` /
    /// `Improper` 4); [`register_category`](crate::ff::style_registry::register_category)
    /// refuses any other combination.
    pub fn custom(
        name: impl Into<Cow<'static, str>>,
        arity: u8,
        coordinate: Coordinate,
        order: EndpointOrder,
    ) -> Self {
        let name = name.into();
        let block = format!("{name}s");
        Self::new(name, Arity::Exact(arity), block, coordinate, order)
    }

    /// Whether the category's terms are the pairs a neighbour search finds
    /// (a pair style), rather than the rows of its block.
    pub fn is_pair_driven(&self) -> bool {
        self.arity == Arity::SelfOrPair
    }

    /// Whether compiling a style of this category builds a kernel.
    pub fn prices_energy(&self) -> bool {
        self.coordinate != Coordinate::None
    }
}

/// The categories molrs registers: its seven, and molrec's `constraint`,
/// `drude` and `virtual_site` (molrec `docs/spec/forcefield.md`,
/// Categories).
pub fn builtin_categories() -> Vec<CategorySpec> {
    use Arity::{Exact, SelfOrPair};
    use Coordinate as C;
    use EndpointOrder::{Ordered, Reversible, Unordered};
    vec![
        CategorySpec::new("atom", Exact(0), ATOMS, C::None, Ordered),
        CategorySpec::new("bond", Exact(2), BONDS, C::Distance, Reversible),
        CategorySpec::new("angle", Exact(3), ANGLES, C::Angle, Reversible),
        CategorySpec::new("dihedral", Exact(4), DIHEDRALS, C::Dihedral, Reversible),
        CategorySpec::new("improper", Exact(4), IMPROPERS, C::Improper, Ordered),
        // A pair's terms are the pairs of atoms a neighbour search (or a
        // compiled `pairs` list) turns up, keyed on `atoms.type`.
        CategorySpec::new("pair", SelfOrPair, ATOMS, C::Distance, Unordered),
        CategorySpec::new("cmap", Exact(5), CMAPS, C::Compound, Ordered),
        CategorySpec::new("constraint", Exact(2), CONSTRAINTS, C::None, Reversible),
        // `atomi` is the core, `atomj` its Drude particle.
        CategorySpec::new("drude", Exact(2), DRUDES, C::Distance, Ordered),
        // A construction, not an interaction: its type rows name no
        // endpoints, its `virtual_sites` rows name the site and its frame.
        CategorySpec::new("virtual_site", Exact(0), VIRTUAL_SITES, C::None, Ordered),
    ]
}

/// How many endpoints a row of the built-in `category` names; `None` for a
/// category molrs does not register (its arity is its endpoint prefix).
pub fn category_arity(category: &str) -> Option<usize> {
    builtin_categories()
        .into_iter()
        .find(|c| c.name == category)
        .map(|c| c.arity.endpoints())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// molrec's arity table (`docs/spec/forcefield.md`, Categories).
    #[test]
    fn builtin_arity_is_the_chapters() {
        for (category, arity) in [
            ("atom", 0),
            ("virtual_site", 0),
            ("bond", 2),
            ("pair", 2),
            ("constraint", 2),
            ("drude", 2),
            ("angle", 3),
            ("dihedral", 4),
            ("improper", 4),
            ("cmap", 5),
        ] {
            assert_eq!(category_arity(category), Some(arity), "{category}");
        }
        assert_eq!(category_arity("pair14"), None);
    }

    #[test]
    fn a_scalar_coordinate_names_as_many_atoms_as_its_category() {
        for c in builtin_categories() {
            if let Some(n) = c.coordinate.atoms() {
                assert_eq!(n, c.arity.endpoints(), "{}", c.name);
            }
        }
    }
}
