//! The **one** MMFF typifier engine.
//!
//! MMFF94 and MMFF94s are the same typing pipeline over two parameter sets, so
//! there is exactly one implementation here and the public surface
//! ([`Mmff94Typifier`](super::Mmff94Typifier) /
//! [`Mmff94sTypifier`](super::Mmff94sTypifier)) is two newtypes over it. The
//! [`MmffVariant`] is a **private field** of this engine: users pick a variant by
//! picking a front door, never by passing a flag.
//!
//! The variant reaches the parameters by two independent paths, and both must be
//! fed or MMFF94s is only half-applied:
//!
//! 1. **Frame annotation** — `assignment::annotate_mmff` resolves the
//!    per-instance numbers the typing base stamps onto the typed graph: `koop` on
//!    impropers and `(v1, v2, v3)` on dihedrals. These are exactly the columns the
//!    `mmff_oop` / `mmff_torsion` kernels read, so this is where the 94/94s
//!    numerical difference physically enters an energy.
//! 2. **The [`ForceField`] tree** — assembled from the compiled table under the
//!    front door's own name ([`shipped_forcefield`](super::shipped_forcefield)); the typing output is
//!    seeded from it and compiled by
//!    [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile).
//!    It carries the force-field name and the style skeleton.
//!
//! Feeding only path 1 leaves a tree that still calls itself `MMFF94`; feeding
//! only path 2 leaves a Frame whose baked `koop` is still MMFF94's — the potentials
//! would then be bit-identical to MMFF94 while claiming to be MMFF94s.
//!
//! # The engine compiles nothing
//!
//! It matches a graph and it owns a library. Turning the typed graph and the
//! typing output into [`Potentials`](crate::ff::potential::Potentials) is
//! `PotentialCompiler::new(typing.forcefield()).compile(&frame)` — the same call
//! every other force field in molrs goes through. A typifier does not compile;
//! its contract is `assign`.

use std::sync::Arc;

use super::properties::MmffVariant;
use crate::ff::forcefield::ForceField;
use crate::ff::typifier::TypeAssignment;
use molrs::core::Atomistic;

use super::assignment;
use super::atom_properties::MmffAtomProperties;

/// Typing metadata + potential parameters: what an MMFF typifier matches
/// against. The shipped sets are built once per variant and shared
/// ([`shipped_forcefield::library`](super::shipped_forcefield::library)).
pub(super) struct MmffLibrary {
    pub(super) params: MmffAtomProperties,
    pub(super) ff: ForceField,
}

/// The library plus the variant it was read for.
///
/// Crate-private by construction: it is the implementation the two named front
/// doors share, not an API. Nothing outside this module may name it, so nothing
/// outside this module can construct an MMFF typifier with an arbitrary variant.
pub(super) struct MmffEngine {
    variant: MmffVariant,
    library: Arc<MmffLibrary>,
}

impl MmffEngine {
    /// A caller's library: typing metadata and the force field it prices with.
    ///
    /// `variant` is supplied by the front door, never by a user.
    pub(super) fn from_parts(
        variant: MmffVariant,
        params: MmffAtomProperties,
        ff: ForceField,
    ) -> Self {
        Self {
            variant,
            library: Arc::new(MmffLibrary { params, ff }),
        }
    }

    /// One of the **shipped** parameter sets, shared from the compiled table.
    ///
    /// Infallible and cheap: the library is built once per variant and
    /// memoised. `variant` is supplied by the front door and is the only thing
    /// the two doors disagree about — both read the same
    /// [`ff::params::mmff`](crate::ff::params::mmff) rows.
    pub(super) fn shipped(variant: MmffVariant) -> Self {
        Self {
            variant,
            library: super::shipped_forcefield::library(variant),
        }
    }

    pub(super) fn params(&self) -> &MmffAtomProperties {
        &self.library.params
    }

    pub(super) fn source_forcefield(&self) -> &ForceField {
        &self.library.ff
    }

    /// Path 1: match the graph and resolve this variant's per-instance
    /// parameters.
    pub(super) fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
        assignment::annotate_mmff(graph, &self.library.params, &self.library.ff, self.variant)
    }
}
