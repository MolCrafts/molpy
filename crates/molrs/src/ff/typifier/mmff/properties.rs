//! The per-atom half of MMFF typing: numeric atom types and partial charges.
//!
//! Ported from RDKit `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`
//! (BSD-3, Paolo Tosco / RDKit contributors). The pipeline mirrors
//! `MMFFMolProperties`'s constructor:
//! [`set_mmff_aromaticity`](crate::perceive::mmff_aromaticity::set_mmff_aromaticity) →
//! [`assign_atom_types`](super::atomtype::assign_atom_types) →
//! [`compute_partial_charges`](super::charges::compute_partial_charges).

use molrs::core::Atomistic;
use molrs::core::MolRsError;

use super::{atomtype, charges};
use crate::perceive::mmff_aromaticity::MmffTopology;

/// MMFF parameterization variant.
///
/// `Mmff94s` is the "static" variant (Halgren 1999). Atom typing and partial
/// charges are **identical** for both variants — MMFF94 and MMFF94s share all 95
/// atom types and every bond / angle / stretch-bend / vdW / charge parameter. The
/// two differ only in the **out-of-plane and torsion** tables, and only on
/// delocalized trivalent nitrogen (MMFF numeric types 10 `NC=O` and 40 `NC=C`):
/// 11 Oop rows and 42 Torsion rows.
///
/// Both `_S` tables are shipped in [`crate::ff::params::mmff`] —
/// [`MMFF_OOP_S`](crate::ff::params::mmff::MMFF_OOP_S) (117 rows) and
/// [`MMFF_TOR_S`](crate::ff::params::mmff::MMFF_TOR_S) (926 rows) — and the
/// parameter resolver (`resolve`) dispatches on this variant to read them,
/// falling back to the base table for keys the `_S` table does not
/// re-parameterise.
///
/// The physics of the difference is the out-of-plane force constant `koop`
/// (md·Å·rad⁻²), which the improper kernel evaluates as
/// `E_oop = 0.5 · 143.9325 · koop · χ²` with χ the Wilson out-of-plane angle in
/// **radians**. `koop > 0` makes the planar centre (χ = 0) an energy *minimum*;
/// `koop < 0` makes it a *maximum*. MMFF94s raises `koop` on those nitrogens to a
/// flat `+0.015` (type 10) / `+0.030` (type 40) — i.e. it **flattens** them, which
/// is what "static" means.
///
/// A variant is picked by picking a typifier —
/// [`Mmff94Typifier`](super::Mmff94Typifier) or
/// [`Mmff94sTypifier`](super::Mmff94sTypifier) — never by passing this enum: it
/// is their private field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MmffVariant {
    Mmff94,
    Mmff94s,
}

/// Per-atom MMFF properties (numeric atom types + partial charges) for a
/// molecule, computed once. Both variants type and charge alike, so this takes
/// no variant.
#[derive(Debug, Clone)]
pub(crate) struct MmffMolProperties {
    atom_types: Vec<u8>,
    partial_charges: Vec<f64>,
}

impl MmffMolProperties {
    /// Run the full MMFF setup (aromaticity → typing → charges).
    ///
    /// Returns `Err` if any atom could not be assigned an MMFF type
    /// (e.g. an unsupported element / transition metal with no MMFF type).
    pub(crate) fn compute(mol: &Atomistic) -> Result<Self, MolRsError> {
        let base = MmffTopology::build(mol).map_err(|sym| {
            MolRsError::validation(format!("MMFF: unsupported element symbol '{sym}'"))
        })?;
        let topo = crate::perceive::mmff_aromaticity::set_mmff_aromaticity(&base);
        let atom_types = atomtype::assign_atom_types(&topo);

        // Locate the first untyped atom for a useful error message.
        if let Some(bad) = atom_types.iter().position(|&t| t == 0) {
            let z = topo.atno[bad];
            return Err(MolRsError::validation(format!(
                "MMFF: could not assign an atom type to atom index {bad} (Z={z})"
            )));
        }

        let partial_charges = charges::compute_partial_charges(&topo, &atom_types);

        Ok(Self {
            atom_types,
            partial_charges,
        })
    }

    /// MMFF numeric atom type (1..=99) for atom index `i`
    /// (the index is the molecule's atom iteration order).
    pub(crate) fn atom_type(&self, i: usize) -> u8 {
        self.atom_types[i]
    }

    /// MMFF partial charge for atom index `i`.
    pub(crate) fn partial_charge(&self, i: usize) -> f64 {
        self.partial_charges[i]
    }
}
