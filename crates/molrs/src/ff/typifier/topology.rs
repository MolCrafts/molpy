//! Shared topology enumeration for graph typifiers.
//!
//! A typifier should write labels and parameters, not make each force field
//! carry its own angle/dihedral enumeration policy. Bonds come from the input
//! graph; derived topology is generated here so OPLS-AA and MMFF stay aligned.

use molrs::core::Atomistic;

/// Ensure angle and dihedral relations are present for a typed graph.
///
/// Existing generated angle/dihedral relations are cleared and rebuilt from the
/// bond graph. Existing bonds are preserved. Force-field-specific improper
/// enumeration is intentionally left to each typifier because MMFF uses Wilson
/// out-of-plane permutations, while OPLS-style impropers are force-field-table
/// driven.
pub(crate) fn typify_bonded_topology(mol: &mut Atomistic) -> Result<(), String> {
    mol.generate_topology(true, true, false, true)
        .map(|_| ())
        .map_err(|e| e.to_string())
}
