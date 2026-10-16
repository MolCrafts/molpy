//! Canonical molecular field-name constants.
//!
//! Column keys are a re-export of the schema's column constants (`core::schema`), whose one public path is here. The schema
//! is the source of truth: a key is declared once, in the column table, and
//! the constant is emitted from that same declaration.
//!
//! A key's dtype is [`crate::core::schema::column()`]'s answer, never a second
//! table here.
//!
//! Frame meta keys name `frame.meta` entries, not block columns, so they are
//! declared once below and are not part of the column schema.
//!
//! # Examples
//!
//! ```
//! use molrs::core::keys;
//!
//! assert_eq!(keys::X, "x");
//! assert_eq!(keys::COORDS, [keys::X, keys::Y, keys::Z]);
//! assert_eq!(keys::UNITS, "units");
//! ```

pub use crate::core::schema::consts::*;

use crate::core::schema::{self, NamedConst};
use crate::core::schema::{KeysDocument, NamedGroup, NamedValue};

macro_rules! meta_keys {
    ($( $(#[$meta:meta])* pub const $name:ident: &str = $value:literal; )*) => {
        $( $(#[$meta])* pub const $name: &str = $value; )*

        /// Frame-meta keys. Not columns — [`SCHEMA_COLUMNS`](crate::core::schema::SCHEMA_COLUMNS)
        /// does not list them, and the bindings project this slice rather than
        /// spelling the strings again.
        pub static META_KEYS: &[NamedConst] = &[
            $(
                NamedConst { const_name: stringify!($name), value: $name },
            )*
        ];
    };
}

meta_keys! {
    /// Frame meta key: the atom-type inventory, packed as `"1:C,2:H"`.
    ///
    /// Declares types no row needs to use; read by
    /// [`TypeLabels`](crate::core::TypeLabels).
    pub const ATOM_TYPE_LABELS: &str = "atom_type_labels";
    /// Frame meta key: the bond-type inventory, packed as `"1:c3-h1,2:c3-c3"`.
    pub const BOND_TYPE_LABELS: &str = "bond_type_labels";
    /// Frame meta key: the angle-type inventory, packed as `"id:label,…"`.
    pub const ANGLE_TYPE_LABELS: &str = "angle_type_labels";
    /// Frame meta key: the dihedral-type inventory, packed as `"id:label,…"`.
    pub const DIHEDRAL_TYPE_LABELS: &str = "dihedral_type_labels";
    /// Frame meta key: the improper-type inventory, packed as `"id:label,…"`.
    pub const IMPROPER_TYPE_LABELS: &str = "improper_type_labels";
    /// Frame meta key: the CMAP-crossterm-type inventory, packed as
    /// `"id:label,…"`.
    pub const CMAP_TYPE_LABELS: &str = "cmap_type_labels";
    /// Frame meta key: the unit system the frame's numbers are in, as the
    /// force-field `units` object — `{"preset": "real"}`, or quantities such
    /// as `{"length": "nm", "energy": "kJ/mol"}` (molrec `conventions.md`,
    /// "Units on a frame"); [`units_preset`] reads its preset. Absent means
    /// the file or caller stated none.
    ///
    /// The LAMMPS molecule-JSON reader writes it and its writer emits it back
    /// (`io::lammps::molecule`); molrs converts no frame between presets.
    pub const UNITS: &str = "units";
}

macro_rules! named_keys {
    ($table:ident, $table_doc:literal; $( $(#[$meta:meta])* pub const $name:ident: &str = $value:literal; )*) => {
        $( $(#[$meta])* pub const $name: &str = $value; )*

        #[doc = $table_doc]
        pub static $table: &[NamedConst] = &[
            $(
                NamedConst { const_name: stringify!($name), value: $name },
            )*
        ];
    };
}

named_keys! {
    GRAPH_KEYS,
    "Molecular-graph keys: relation kinds and node / relation props a graph \
     carries that are not frame columns of the schema. The bindings project \
     this slice.";
    /// The relation kind ports are stored under.
    pub const PORTS: &str = "ports";
    /// Node prop: the fragment instance a node belongs to, an `i32` that
    /// `MolGraph::replicate` stamps on every copy and `set_atom` / `set_node`
    /// with this key writes per atom.
    pub const FRAG_ID: &str = "frag_id";
    /// Atom prop: the 0-based id of the atom's charge-equivalence class,
    /// written by `perceive::assign_equivalence_classes`. Class ids
    /// are assigned in order of first appearance in the atom order, so the
    /// atom antechamber picks as a class's representative (its lowest-indexed
    /// member) names it.
    pub const EQUIV_CLASS: &str = "equiv_class";
    /// Bond prop: the antechamber bond type (`perceive::assign_bcc_bond_types`), read by
    /// the `ATOMTYPE_*.DEF` rule engine and the `BCCPARM.DAT` corrections.
    /// Deliberately not [`TYPE`]: that key is the caller's, the force-field
    /// type name.
    pub const BCC_BOND_TYPE: &str = "bcc_bond_type";
    /// Atom column pairing pre- and post-reaction LAMMPS `fix bond/react`
    /// template atoms.
    pub const REACT_ID: &str = "react_id";
    /// Node field marking an atom as a virtual site and naming its kind.
    pub const VSITE: &str = "vsite";
    /// The field under which a coarse-grained bead answers with its member
    /// atoms.
    pub const BEAD_ATOMS: &str = "atoms";
}

named_keys! {
    LAMMPS_META_KEYS,
    "Frame-meta keys the LAMMPS data reader writes. The bindings project this \
     slice.";
    /// Frame meta key: every `* Coeffs` section of a LAMMPS data file,
    /// verbatim (`PairIJ` and the class2 cross terms included), which
    /// `io::lammps::forcefield_reader::LammpsForcefieldReader::read_data_coeffs`
    /// reads with the type labels the `* Type Labels` sections declared.
    pub const LAMMPS_COEFFS_TEXT: &str = "lammps_coeffs_text";
    /// Frame meta key: the unit style a LAMMPS `write_data` title line stated
    /// (`units = real`); absent when the file states none.
    pub const LAMMPS_UNITS: &str = "lammps_units";
    /// Frame meta key: the mass of each atom type, as a JSON object from type
    /// label to mass (`{"OW": 15.9994, "HW": 1.008}`; an unlabelled type's label
    /// is its id, `"1"`). The LAMMPS data reader writes it from the Masses
    /// section; the writer takes a type's mass from it when no atom of that type
    /// carries a `mass` — a type the labels declare but no atom uses, or a frame
    /// without a `mass` column.
    pub const LAMMPS_TYPE_MASSES: &str = "lammps_type_masses";
}

/// The preset a frame's [`UNITS`] meta value names: the `preset` of a units
/// object. `None` for a value that names no preset.
pub fn units_preset(value: &crate::core::MetaValue) -> Option<&str> {
    use crate::core::MetaValue;
    match value {
        MetaValue::Json(serde_json::Value::Object(object)) => {
            object.get("preset").and_then(serde_json::Value::as_str)
        }
        _ => None,
    }
}

/// Column keys, groups, block names, and frame-meta keys, from the tables.
pub fn keys_document() -> KeysDocument {
    KeysDocument {
        columns: schema::SCHEMA_COLUMNS
            .iter()
            .map(|c| NamedValue {
                const_name: c.const_name.to_string(),
                value: c.key.to_string(),
            })
            .collect(),
        groups: schema::KEY_GROUPS
            .iter()
            .map(|g| NamedGroup {
                const_name: g.const_name.to_string(),
                values: g.keys.iter().map(|k| (*k).to_string()).collect(),
            })
            .collect(),
        blocks: schema::BLOCK_NAMES
            .iter()
            .map(|spec| NamedValue {
                const_name: spec.const_name.to_string(),
                value: spec.value.to_string(),
            })
            .collect(),
        block_groups: schema::BLOCK_GROUPS
            .iter()
            .map(|g| NamedGroup {
                const_name: g.const_name.to_string(),
                values: g.keys.iter().map(|k| (*k).to_string()).collect(),
            })
            .collect(),
        meta: META_KEYS
            .iter()
            .map(|spec| NamedValue {
                const_name: spec.const_name.to_string(),
                value: spec.value.to_string(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_document_covers_every_declared_name() {
        let doc = keys_document();
        assert_eq!(doc.columns.len(), schema::SCHEMA_COLUMNS.len());
        for (got, spec) in doc.columns.iter().zip(schema::SCHEMA_COLUMNS) {
            assert_eq!(got.const_name, spec.const_name);
            assert_eq!(got.value, spec.key);
        }
        assert_eq!(doc.groups.len(), schema::KEY_GROUPS.len());
        for (got, group) in doc.groups.iter().zip(schema::KEY_GROUPS) {
            assert_eq!(got.const_name, group.const_name);
            assert_eq!(
                got.values.iter().map(String::as_str).collect::<Vec<_>>(),
                group.keys
            );
        }
        assert_eq!(doc.blocks.len(), schema::BLOCK_NAMES.len());
        for (got, spec) in doc.blocks.iter().zip(schema::BLOCK_NAMES) {
            assert_eq!(got.const_name, spec.const_name);
            assert_eq!(got.value, spec.value);
        }
        assert_eq!(doc.block_groups.len(), schema::BLOCK_GROUPS.len());
        for (got, group) in doc.block_groups.iter().zip(schema::BLOCK_GROUPS) {
            assert_eq!(got.const_name, group.const_name);
            assert_eq!(
                got.values.iter().map(String::as_str).collect::<Vec<_>>(),
                group.keys
            );
        }
        assert_eq!(doc.meta.len(), META_KEYS.len());
        for (got, spec) in doc.meta.iter().zip(META_KEYS) {
            assert_eq!(got.const_name, spec.const_name);
            assert_eq!(got.value, spec.value);
            assert_eq!(got.value, spec.const_name.to_ascii_lowercase());
        }
        let columns: std::collections::HashSet<_> =
            doc.columns.iter().map(|c| c.value.as_str()).collect();
        for spec in &doc.meta {
            assert!(
                !columns.contains(spec.value.as_str()),
                "{} is a column, not frame meta",
                spec.value
            );
        }
    }
}
