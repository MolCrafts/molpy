//! MMFF94 typing data: the per-type atom-property table (`MMFFPROP.PAR`).

use std::collections::HashMap;

use crate::ff::params::mmff::MmffProp;

/// The MMFF atom-property rows a typifier types against, by atom type.
///
/// Separate from [`ForceField`](crate::ff::forcefield::ForceField) because these
/// are typing metadata, not potential parameters: used only during topology
/// classification (e.g. the `linh` flag picks the linear-bend form, `sbmb`
/// drives bond-type assignment), never during energy evaluation. A row is the
/// shipped table's own [`MmffProp`], whichever source it came from.
#[derive(Debug, Clone)]
pub struct MmffAtomProperties {
    /// Rows by atom type.
    pub(crate) rows: HashMap<u8, MmffProp>,
}

impl MmffAtomProperties {
    /// The table of `rows`, each keyed by its own `atom_type`.
    pub fn new(rows: impl IntoIterator<Item = MmffProp>) -> Self {
        Self {
            rows: rows.into_iter().map(|p| (p.atom_type, p)).collect(),
        }
    }

    /// The row of atom type `atom_type`.
    pub fn get(&self, atom_type: u8) -> Option<&MmffProp> {
        self.rows.get(&atom_type)
    }
}
