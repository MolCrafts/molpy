//! Which arity-2 relations a molecular graph walks from one node.
//!
//! [`NeighborKinds::BONDS`] is the chemical graph: the other atom and the
//! bond handle. [`NeighborKinds::with_ports`](NeighborKinds::with_ports) and
//! [`NeighborKinds::with_terms`](NeighborKinds::with_terms) widen that walk,
//! and the item becomes an [`Adjacent`] because a bond id, a port id and a
//! term id live in different tables.

use std::marker::PhantomData;

use super::{NodeId, RelationId};

/// Selects the bonds of one node. The yielded item is `(other, bond)`.
#[derive(Clone, Copy, Debug)]
pub enum BondsOnly {}

/// Selects bonds plus ports and/or other arity-2 terms. The yielded item is
/// an [`Adjacent`].
#[derive(Clone, Copy, Debug)]
pub enum Wide {}

/// One incident arity-2 relation, tagged by what it is.
///
/// `block` on [`Adjacent::Term`](Self::Term) is the relation-block name
/// (`"constraints"`, `"pairs"`, `"exclusions"`, `"drudes"`,
/// `"virtual_sites"`, or another arity-2 kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Adjacent<'a> {
    /// A chemical bond.
    Bond { other: NodeId, id: RelationId },
    /// A port on a bond.
    Port { other: NodeId, id: RelationId },
    /// Any other arity-2 term.
    Term {
        block: &'a str,
        other: NodeId,
        id: RelationId,
    },
}

/// Which arity-2 kinds [`crate::core::Atomistic::neighbors`] and
/// [`crate::core::CoarseGrain::neighbors`] yield.
///
/// Angles, dihedrals and impropers are not neighbors: they have more than
/// two endpoints, so no setting of this value returns them.
#[derive(Clone, Copy, Debug)]
pub struct NeighborKinds<K> {
    bonds: bool,
    ports: bool,
    terms: bool,
    kind: PhantomData<fn() -> K>,
}

impl NeighborKinds<BondsOnly> {
    /// Chemical bonds only. Each item is `(other_atom, bond_id)`.
    pub const BONDS: Self = Self {
        bonds: true,
        ports: false,
        terms: false,
        kind: PhantomData,
    };

    /// Also yield ports. The item type becomes [`Adjacent`].
    pub const fn with_ports(self) -> NeighborKinds<Wide> {
        NeighborKinds {
            bonds: self.bonds,
            ports: true,
            terms: self.terms,
            kind: PhantomData,
        }
    }

    /// Also yield every other arity-2 term. The item type becomes [`Adjacent`].
    pub const fn with_terms(self) -> NeighborKinds<Wide> {
        NeighborKinds {
            bonds: self.bonds,
            ports: self.ports,
            terms: true,
            kind: PhantomData,
        }
    }
}

impl NeighborKinds<Wide> {
    /// Also yield ports.
    pub const fn with_ports(self) -> Self {
        Self {
            ports: true,
            ..self
        }
    }

    /// Also yield every other arity-2 term.
    pub const fn with_terms(self) -> Self {
        Self {
            terms: true,
            ..self
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::BondsOnly {}
    impl Sealed for super::Wide {}
}

/// Marker that selects the item [`NeighborKinds`] yields.
///
/// Implemented only for [`BondsOnly`] and [`Wide`].
pub trait AdjacencyKind: sealed::Sealed + Sized {
    type Item<'a>;

    fn include(which: &NeighborKinds<Self>, is_bond: bool, is_port: bool) -> bool;

    fn item<'a>(
        block: &'a str,
        is_bond: bool,
        is_port: bool,
        other: NodeId,
        id: RelationId,
    ) -> Self::Item<'a>;
}

impl AdjacencyKind for BondsOnly {
    type Item<'a> = (NodeId, RelationId);

    fn include(which: &NeighborKinds<Self>, is_bond: bool, is_port: bool) -> bool {
        let _ = (which, is_port);
        is_bond
    }

    fn item<'a>(
        block: &'a str,
        is_bond: bool,
        is_port: bool,
        other: NodeId,
        id: RelationId,
    ) -> Self::Item<'a> {
        let _ = (block, is_bond, is_port);
        (other, id)
    }
}

impl AdjacencyKind for Wide {
    type Item<'a> = Adjacent<'a>;

    fn include(which: &NeighborKinds<Self>, is_bond: bool, is_port: bool) -> bool {
        if is_bond {
            which.bonds
        } else if is_port {
            which.ports
        } else {
            which.terms
        }
    }

    fn item<'a>(
        block: &'a str,
        is_bond: bool,
        is_port: bool,
        other: NodeId,
        id: RelationId,
    ) -> Self::Item<'a> {
        if is_bond {
            Adjacent::Bond { other, id }
        } else if is_port {
            Adjacent::Port { other, id }
        } else {
            Adjacent::Term { block, other, id }
        }
    }
}

impl<K: AdjacencyKind> NeighborKinds<K> {
    pub(crate) fn include(&self, is_bond: bool, is_port: bool) -> bool {
        K::include(self, is_bond, is_port)
    }

    pub(crate) fn item<'a>(
        &self,
        block: &'a str,
        is_bond: bool,
        is_port: bool,
        other: NodeId,
        id: RelationId,
    ) -> K::Item<'a> {
        let _ = self;
        K::item(block, is_bond, is_port, other, id)
    }
}
