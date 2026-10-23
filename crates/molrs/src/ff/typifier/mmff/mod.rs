//! MMFF atom/bond/angle/torsion/improper typifiers — two named front doors.
//!
//! Matches an [`Atomistic`] to MMFF type labels and partial charges. That is
//! the typifier's contract, and all of it: MMFF is a parameter set plus a topology
//! labeler, and it computes energies the way every other force field in molrs does
//! — through [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile).
//! **MMFF is not a special case.**
//!
//! # Which door?
//!
//! | Type | Parameter set | Delocalised trivalent N |
//! |---|---|---|
//! | [`Mmff94Typifier`] | `MMFF94` | pyramidal (dynamic / time-averaged picture) |
//! | [`Mmff94sTypifier`] | `MMFF94s` | **planar** (static, Halgren 1999) |
//!
//! There is **one** engine behind both; the variant is its private field. Users
//! choose a parameter set by choosing a type, never by passing a flag.
//!
//! The two parameter sets share all 95 atom types and every bond / angle /
//! stretch-bend / vdW / charge parameter. They differ in **11 out-of-plane rows**
//! and **42 torsion rows**, every one of them centred on MMFF numeric type 10
//! (`NC=O`, amide N) or 40 (`NC=C`, enamine-type N). MMFF94s ("s" = *static*)
//! raises the out-of-plane force constant `koop` on those centres to a flat
//! `+0.015` (type 10) / `+0.030` (type 40) md·Å·rad⁻², which — see
//! [`Mmff94sTypifier`] — makes the planar nitrogen an energy *minimum*.
//!
//! # Example — the one route
//!
//! ```no_run
//! use molrs::core::Atomistic;
//! use molrs::ff::compile::PotentialCompiler;
//! use molrs::ff::potential::{Accumulator, Potential, intramolecular_pairs};
//! use molrs::ff::typifier::Typing;
//! use molrs::ff::typifier::mmff::Mmff94Typifier;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mol = Atomistic::new();                             // build or load your molecule
//! let mut typing = Typing::new(Mmff94Typifier::new());
//!
//! let mut frame = typing.typify(&mol)?.to_frame().map_err(|e| e.to_string())?;
//! let ff = typing.forcefield();                           // exactly the types assigned
//! let special = ff.special_bonds().ok_or("the force field states no special_bonds")?;
//! frame.insert("pairs", intramolecular_pairs(&frame, special, None)?);
//! let potentials = PotentialCompiler::new(ff).compile(&frame)?; // the standard compile path
//!
//! let coords: Vec<f64> = Vec::new();                      // flat [x,y,z, ...]
//! let mut acc = Accumulator::new(coords.len());
//! potentials.evaluate(&coords, &mut acc);                 // energy + forces
//! println!("MMFF94 energy = {} kcal/mol", acc.energy());
//! # Ok(())
//! # }
//! ```
//!
//! The `pairs` block is the caller's because the neighbour list is the caller's:
//! a minimizer that moves atoms decides when to rebuild it. See `docs/interop.md`.

#![allow(clippy::type_complexity)]

use crate::ff::forcefield::ForceField;
use crate::ff::typifier::{TypeAssignment, Typifier};
use molrs::core::Atomistic;
use properties::MmffVariant;

use engine::MmffEngine;

mod assignment;
mod atomtype;
mod charges;
mod engine;
mod properties;
mod resolve;
mod shipped_forcefield;

#[cfg(test)]
mod tests;

// Re-exports

/// Declare a front door: a newtype over the one [`MmffEngine`], with its variant
/// and its force-field name pinned by the type itself.
///
/// The forwarding is written once here rather than twice by hand, so the two doors
/// cannot drift apart — but each door is still a distinct concrete type with a
/// distinct parameter set, which is the whole public contract.
macro_rules! mmff_front_door {
    (
        $(#[$doc:meta])*
        $name:ident, $variant:expr, $set:literal
    ) => {
        $(#[$doc])*
        pub struct $name(MmffEngine);

        impl $name {
            #[doc = concat!("Create a typifier over the shipped `", $set, "` parameter set.")]
            ///
            /// Infallible and cheap: the parameters are compiled-in typed Rust
            /// ([`ff::params::mmff`](crate::ff::params::mmff)), assembled once
            /// per process and shared by every typifier of this door.
            pub fn new() -> Self {
                Self(MmffEngine::shipped($variant))
            }
        }

        impl Typifier for $name {
            #[doc = concat!("Type an all-atom graph against `", $set, "`.")]
            ///
            /// Atoms get their MMFF numeric `type` and partial `charge`; bonds,
            /// angles, dihedrals and impropers get their type labels **and** the
            /// per-instance numbers the kernels read — including `koop` on every
            /// improper and `(v1, v2, v3)` on every dihedral, resolved from *this*
            /// door's parameter set. Each distinct parameter set is one type,
            /// named by its label; the vdW rows of the atom types used are pairs.
            ///
            /// To evaluate an energy, type through
            /// [`Typing`](crate::ff::typifier::Typing), materialize the result
            /// ([`Atomistic::to_frame`]), add the neighbour list
            /// ([`intramolecular_pairs`](crate::ff::potential::intramolecular_pairs)),
            /// and compile it with
            /// `PotentialCompiler::new(typing.forcefield()).compile(&frame)` — see
            /// the module example.
            fn assign(
                &self,
                graph: &mut Atomistic,
            ) -> Result<TypeAssignment, crate::ff::typifier::TypingError> {
                self.0.assign(graph)
            }

            #[doc = concat!("The `", $set, "` force field this door matches against.")]
            fn source_forcefield(&self) -> &ForceField {
                self.0.source_forcefield()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

mmff_front_door! {
    /// MMFF94 typifier (Halgren 1996) — the standard parameterization.
    ///
    /// Owns the MMFF94 typing metadata and force-field parameters, both read from
    /// the compiled table [`crate::ff::params::mmff`].
    ///
    /// ```no_run
    /// use molrs::ff::typifier::mmff::Mmff94Typifier;
    /// use molrs::ff::typifier::{Typifier, Typing};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mol = molrs::core::Atomistic::new();
    /// let typifier = Mmff94Typifier::new();
    /// assert_eq!(typifier.source_forcefield().name, "MMFF94");
    /// let typed = Typing::new(typifier).typify(&mol)?;
    /// # let _ = typed;
    /// # Ok(())
    /// # }
    /// ```
    Mmff94Typifier, MmffVariant::Mmff94, "MMFF94"
}

mmff_front_door! {
    /// MMFF94s typifier (Halgren 1999) — the **static** variant, for energy
    /// minimization.
    ///
    /// Identical to [`Mmff94Typifier`] except on delocalised trivalent nitrogen
    /// (MMFF numeric types 10 `NC=O` and 40 `NC=C`), where it re-parameterises 11
    /// out-of-plane rows and 42 torsion rows so that the nitrogen minimises to a
    /// **planar** geometry — the picture seen in crystal structures, rather than
    /// MMFF94's dynamic / time-averaged pyramidal one.
    ///
    /// The mechanism is the sign and size of the out-of-plane force constant. The
    /// form (`ff::potential::improper::Mmff`) evaluates
    ///
    /// ```text
    /// E_oop = 0.5 · 143.9325 · koop · χ²
    /// ```
    ///
    /// with χ the Wilson out-of-plane angle in **radians** and `koop` in
    /// md·Å·rad⁻² (143.9325 converts md·Å → kcal·mol⁻¹). So `koop > 0` makes the
    /// planar geometry (χ = 0) an energy **minimum** and `koop < 0` makes it a
    /// **maximum**. MMFF94s sets `koop` on every such centre to a flat `+0.015`
    /// (type 10) / `+0.030` (type 40); under MMFF94 those rows range over
    /// `−0.033 … +0.004`.
    ///
    /// Parameters come from the same compiled table as [`Mmff94Typifier`]
    /// ([`crate::ff::params::mmff`]) — the two front doors differ by this
    /// force field's **name** and by the variant, which selects
    /// [`MMFF_OOP_S`](crate::ff::params::mmff::MMFF_OOP_S) /
    /// [`MMFF_TOR_S`](crate::ff::params::mmff::MMFF_TOR_S) and drives the
    /// per-instance `koop` / `(v1, v2, v3)` baked onto the typed graph.
    ///
    /// ```no_run
    /// use molrs::ff::typifier::mmff::Mmff94sTypifier;
    /// use molrs::ff::typifier::{Typifier, Typing};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mol = molrs::core::Atomistic::new();
    /// let typifier = Mmff94sTypifier::new();
    /// assert_eq!(typifier.source_forcefield().name, "MMFF94s");
    /// let typed = Typing::new(typifier).typify(&mol)?;
    /// # let _ = typed;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # References
    ///
    /// - T. A. Halgren, *MMFF VI. MMFF94s option for energy minimization studies*,
    ///   J. Comput. Chem. **20**, 720–729 (1999).
    Mmff94sTypifier, MmffVariant::Mmff94s, "MMFF94s"
}
