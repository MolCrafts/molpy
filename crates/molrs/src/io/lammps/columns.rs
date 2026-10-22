//! The Frame columns a LAMMPS data / dump reader builds: optional columns,
//! and the dump attribute names they map to (the inserts are
//! `io::frame_columns`'s).
//!
//! Column-name aliases follow the LAMMPS `dump custom` / `compute property/atom`
//! attribute list: <https://docs.lammps.org/dump.html>,
//! <https://docs.lammps.org/compute_property_atom.html>.

use molrs::core::keys;

#[derive(Debug, Clone)]
pub(crate) struct OptionalColumn<T> {
    pub values: Vec<T>,
    pub present: bool,
}

impl<T: Copy + Default> OptionalColumn<T> {
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            values: Vec::with_capacity(n),
            present: false,
        }
    }

    pub(crate) fn push(&mut self, v: T) {
        self.values.push(v);
        self.present = true;
    }
}

/// LAMMPS-native attribute → canonical Frame column.
///
/// Only entries that **rename**. Attributes already identical to our keys
/// (`x`, `vx`, `mux`, `diameter`, `quatw`, `mass`, …) are left unchanged.
///
/// Sources:
/// - dump custom attributes: https://docs.lammps.org/dump.html
/// - compute property/atom: https://docs.lammps.org/compute_property_atom.html
const DUMP_COLUMN_ALIASES: &[(&str, &str)] = &[
    // Core renames used across the stack (molpy / keys)
    ("q", keys::CHARGE),
    ("mol", keys::MOL_ID),
    // A dump's `type` column holds LAMMPS' numeric type ordinal, not the
    // force-field label — the vocabulary keeps those apart as `type_id` and
    // `type`, so the rename happens here at the format boundary. A `type`
    // field holding type *labels* is the exception the dump reader and writer
    // handle themselves: it is the canonical string `type`.
    ("type", keys::TYPE_ID),
    // Forces (already short names; kept as-is) — listed only if renamed
    // EFF package: spin was renamed to espin in LAMMPS (15Sep2022)
    ("spin", "espin"),
    // SPH package energy attribute `e` is ambiguous; leave as "e"
];

/// Reader exit: rename a LAMMPS-native dump column to its canonical field name.
pub(crate) fn canonical_dump_column(name: &str) -> String {
    DUMP_COLUMN_ALIASES
        .iter()
        .find(|(native, _)| *native == name)
        .map_or_else(|| name.to_string(), |(_, c)| (*c).to_string())
}

/// Writer entry: inverse of [`canonical_dump_column`].
pub(crate) fn native_dump_column(name: &str) -> &str {
    DUMP_COLUMN_ALIASES
        .iter()
        .find(|(_, canonical)| *canonical == name)
        .map_or(name, |(native, _)| *native)
}
