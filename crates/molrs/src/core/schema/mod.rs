//! The Frame schema — a committed, machine-checked vocabulary for block and
//! column names, and the validator that judges a Frame against it.
//!
//! # A canonical key names a quantity, not a slot
//!
//! This is the axiom the whole design rests on, and it is true only because it
//! is kept true: **if two things need different dtypes under one name, they are
//! two quantities and get two keys.** `type` (String, a force-field label) and
//! `type_id` (Uint, a LAMMPS ordinal) are not an exception to the one-dtype
//! rule — splitting them is the operation that *makes* the rule hold.
//!
//! Every future "but this key needs two dtypes" has one correct answer: then it
//! is two keys. A `DTypeSet` would quietly undo the entire design, because a
//! column's dtype is fixed by its first write and molrs refuses to coerce — so
//! the second writer's value silently fails to land.
//!
//! # Closed vocabulary, open block set
//!
//! A [`ColumnSpec`] binds a key *wherever it appears*, in any block. That is
//! what lets [`Block::insert`](crate::core::Block::insert) enforce dtype
//! without knowing which block it is about to live in — dissolving the problem
//! that a standalone `Block` only learns its role when inserted into a `Frame`,
//! by which time a wrong-dtype column would already exist.
//!
//! The *block* set stays open: `MolGraph::to_frame` mints one block per
//! registered relation kind, so a block name with no [`BlockSpec`] is legal and
//! its columns are still checked individually. A column key with no spec is
//! unconstrained — the extension point for perceived facts, per-instance
//! force-field parameters, and format-local columns.
//!
//! # Two layers of enforcement
//!
//! | concern | scope | enforced at |
//! |---|---|---|
//! | dtype, shape | key-global | `Block::insert` / `rename_column` |
//! | required columns, arity | block-scoped | [`BlockSpec`] via [`Validator`] |
//! | endpoint range, row counts | frame-scoped | [`Validator`] |

mod block;
mod column;
mod document;
mod validator;
mod violation;

pub use block::{BlockSpec, EndpointSpec, EndpointTarget, RowKind};
pub use column::{ColShape, ColumnDimension, ColumnSpec};
pub use document::{
    BlockDoc, ColumnDoc, KeysDocument, NamedGroup, NamedValue, SchemaDocument, document,
};
pub use validator::Validator;
pub use violation::{
    InstancePath, MAX_CELL_VIOLATIONS_PER_COLUMN, SchemaReport, Violation, ViolationKind,
};

use crate::core::DType;
use crate::core::PresetDim;

use ColShape::Scalar;
use ColumnDimension::{Dimensionless, NotAQuantity, Of, Product};
use PresetDim::{Charge, Force, Length, Mass, Velocity};
// Identifiers are unsigned and physical quantities are float. `Int` is here for
// the one kind of value that is neither: a periodic image flag, which counts
// cell crossings and must be able to count them backwards.
use DType::{Float, I64, Int, String as Str, Uint};

/// A declared name that is not a column: a block name or a frame-meta key.
#[derive(Debug, Clone, Copy)]
pub struct NamedConst {
    /// Rust constant name (`"ATOMS"`, `"ATOM_TYPE_LABELS"`).
    pub const_name: &'static str,
    /// The string that constant holds.
    pub value: &'static str,
}

/// An ordered group of keys (`COORDS`, `ENDPOINTS`, `TOPOLOGY`).
///
/// The members are constants declared with the columns or the blocks. This
/// slice is what the bindings loop, so a group cannot be added in only one
/// language.
#[derive(Debug, Clone, Copy)]
pub struct KeyGroup {
    /// Rust/Python constant name (`"COORDS"`).
    pub const_name: &'static str,
    /// Member keys, in group order.
    pub keys: &'static [&'static str],
}

macro_rules! columns {
    ($($name:ident: $key:literal, $dtype:expr, $shape:expr, $dimension:expr, $doc:literal;)*) => {
        /// Every canonical column, sorted by key.
        ///
        /// Sortedness and uniqueness are asserted by the vocabulary gate, and
        /// the lookup binary-searches this table. `consts::$name` is emitted
        /// from the same `$key` literal — the string is written once, here.
        pub static SCHEMA_COLUMNS: &[ColumnSpec] = &[
            $(
                ColumnSpec {
                    key: $key,
                    const_name: stringify!($name),
                    dtype: $dtype,
                    shape: $shape,
                    dimension: $dimension,
                    doc: $doc,
                },
            )*
        ];

        /// Canonical string constants for the keys of [`SCHEMA_COLUMNS`].
        ///
        /// Emitted by the `columns!` table macro from the same tokens as the table.
        /// [`crate::core::keys`] re-exports this module. Groups
        /// name these constants; they do not spell the strings again.
        pub(crate) mod consts {
            $(
                #[doc = $doc]
                pub const $name: &str = $key;
            )*

            /// The three Cartesian coordinate keys, in axis order.
            pub const COORDS: [&str; 3] = [X, Y, Z];
            /// The three image-flag keys, in lattice-vector order.
            ///
            /// They travel with [`COORDS`]: a wrapped coordinate without its flags has
            /// lost the atom's history, and a reader that finds one without the other
            /// cannot reconstruct a continuous trajectory.
            pub const IMAGES: [&str; 3] = [IX, IY, IZ];
            /// The three Cartesian velocity keys, in axis order.
            pub const VELOCITIES: [&str; 3] = [VX, VY, VZ];
            /// The three Cartesian force keys, in axis order.
            pub const FORCES: [&str; 3] = [FX, FY, FZ];
            /// The four orientation-quaternion keys, in `(w, i, j, k)` order.
            pub const QUAT: [&str; 4] = [QUATW, QUATI, QUATJ, QUATK];
            /// The three dipole-moment keys, in axis order.
            pub const DIPOLE: [&str; 3] = [MUX, MUY, MUZ];
            /// The three site-axis keys, in axis order.
            pub const AXIS: [&str; 3] = [AXIS_X, AXIS_Y, AXIS_Z];
            /// Relation endpoint keys in position order.
            pub const ENDPOINTS: [&str; 5] = [ATOMI, ATOMJ, ATOMK, ATOML, ATOMM];
        }

        /// Column groups the bindings export next to the scalar key constants.
        pub static KEY_GROUPS: &[KeyGroup] = &[
            KeyGroup { const_name: "COORDS", keys: &consts::COORDS },
            KeyGroup { const_name: "IMAGES", keys: &consts::IMAGES },
            KeyGroup { const_name: "VELOCITIES", keys: &consts::VELOCITIES },
            KeyGroup { const_name: "FORCES", keys: &consts::FORCES },
            KeyGroup { const_name: "QUAT", keys: &consts::QUAT },
            KeyGroup { const_name: "DIPOLE", keys: &consts::DIPOLE },
            KeyGroup { const_name: "AXIS", keys: &consts::AXIS },
            KeyGroup { const_name: "ENDPOINTS", keys: &consts::ENDPOINTS },
        ];
    };
}

columns! {
    ALTLOC: "altloc", Str, Scalar, NotAQuantity, "Alternate-location indicator of a crystallographic site (PDB altLoc, mmCIF label_alt_id); \"\" for none.";
    ATOM_MAP: "atom_map", Uint, Scalar, NotAQuantity, "Atom-map number of a mapped SMILES; 0 for an unmapped atom.";
    ATOMI: "atomi", Uint, Scalar, NotAQuantity, "First endpoint of a relation, 0-indexed into the target node block.";
    ATOMIC_NUMBER: "atomic_number", Uint, Scalar, NotAQuantity, "Atomic number Z.";
    ATOMJ: "atomj", Uint, Scalar, NotAQuantity, "Second endpoint of a relation, 0-indexed.";
    ATOMK: "atomk", Uint, Scalar, NotAQuantity, "Third endpoint of a relation (angle terminus / dihedral), 0-indexed; the angle vertex is `atomj`.";
    ATOML: "atoml", Uint, Scalar, NotAQuantity, "Fourth endpoint of a relation (dihedral / improper), 0-indexed.";
    ATOMM: "atomm", Uint, Scalar, NotAQuantity, "Fifth endpoint of a relation (a CMAP's two consecutive dihedrals `atomi..atoml` and `atomj..atomm`), 0-indexed.";
    AXIS_X: "axis_x", Float, Scalar, Of(Length), "x-component of a coarse-grained site's axis: from the first member of its group to the site.";
    AXIS_Y: "axis_y", Float, Scalar, Of(Length), "y-component of a coarse-grained site's axis: from the first member of its group to the site.";
    AXIS_Z: "axis_z", Float, Scalar, Of(Length), "z-component of a coarse-grained site's axis: from the first member of its group to the site.";
    B_FACTOR: "b_factor", Float, Scalar, Product(Length, Length), "Isotropic crystallographic displacement parameter B (PDB tempFactor, mmCIF B_iso_or_equiv).";
    BEAD_TYPE: "bead_type", Str, Scalar, NotAQuantity, "Coarse-grained bead type label.";
    BOND_NUMBER: "bond_number", Uint, Scalar, NotAQuantity, "Integer bond number of the localized Lewis/Kekule structure: 0 unknown, 1 single, 2 double, 3 triple, 4 quadruple. Never fractional - aromaticity is a bond type, not a number.";
    BOND_TYPE: "bond_type", Uint, Scalar, NotAQuantity, "Chemical bond class: 0 unknown, 1 single, 2 double, 3 triple, 4 aromatic. Orthogonal to `bond_number`: an aromatic bond is `bond_type = 4` carrying a `bond_number` of 1 or 2.";
    CHAIN: "chain", Str, Scalar, NotAQuantity, "Chain label (PDB chain identifier, mmCIF label_asym_id). A label, not an identifier: every `*_id` key is a u64.";
    CHARGE: "charge", Float, Scalar, Of(Charge), "Partial charge.";
    ELEMENT: "element", Str, Scalar, NotAQuantity, "IUPAC element symbol (e.g. \"C\").";
    EXCLUDE_14: "exclude_14", DType::Bool, Scalar, NotAQuantity, "Whether this torsion's 1-4 non-bonded term is suppressed (AMBER negative 3rd pointer)";
    FORMAL_CHARGE: "formal_charge", I64, Scalar, NotAQuantity, "Integer formal charge, in units of the elementary charge.";
    FREE: "free", DType::Bool, Scalar, NotAQuantity, "Whether an atom may move when the coordinates are optimized: `false` pins the atom where it is. A frame without the column has every atom free.";
    FX: "fx", Float, Scalar, Of(Force), "x-component of the force on an atom.";
    FY: "fy", Float, Scalar, Of(Force), "y-component of the force on an atom.";
    FZ: "fz", Float, Scalar, Of(Force), "z-component of the force on an atom.";
    H_COUNT: "h_count", Uint, Scalar, NotAQuantity, "Hydrogens an atom carries that the graph does not draw — the exact count a line-notation bracket atom states (SMILES `[CH3]`); drawing them (`add_hydrogens`) leaves 0.";
    HETERO: "hetero", DType::Bool, Scalar, NotAQuantity, "Whether a site belongs to a hetero group: a ligand, ion or solvent (PDB HETATM, mmCIF group_PDB).";
    IBEAD: "ibead", Uint, Scalar, NotAQuantity, "`members`: the bead's row in `atoms`, 0-indexed.";
    ICODE: "icode", Str, Scalar, NotAQuantity, "Residue insertion code (PDB iCode, mmCIF pdbx_PDB_ins_code); \"\" for none.";
    ID: "id", Uint, Scalar, NotAQuantity, "Identifier carried by the source file. Never an index — endpoints are 0-based row indices and a reader that must map labels to rows does so locally.";
    IS_14: "is_14", DType::Bool, Scalar, NotAQuantity, "Whether a non-bonded pair is a 1-4 (third-neighbour) pair.";
    IS_AROMATIC: "is_aromatic", DType::Bool, Scalar, NotAQuantity, "Whether an atom is aromatic, as a line notation declared it or aromaticity perception decided.";
    ISOTOPE: "isotope", Uint, Scalar, NotAQuantity, "Mass number a line-notation bracket atom states (SMILES `[13C]`).";
    IX: "ix", Int, Scalar, NotAQuantity, "Periodic image flag along the first lattice vector: how many cells this atom has crossed. The continuous position is `xyz + H·(ix, iy, iz)`; the stored coordinate itself stays wrapped. Signed, because an atom can cross back.";
    IY: "iy", Int, Scalar, NotAQuantity, "Periodic image flag along the second lattice vector. See `ix`.";
    IZ: "iz", Int, Scalar, NotAQuantity, "Periodic image flag along the third lattice vector. See `ix`.";
    MASS: "mass", Float, Scalar, Of(Mass), "Atomic mass.";
    MOL_ID: "mol_id", Uint, Scalar, NotAQuantity, "Molecule identifier grouping atoms into molecules.";
    MUX: "mux", Float, Scalar, Product(Charge, Length), "x-component of a per-atom electric dipole moment.";
    MUY: "muy", Float, Scalar, Product(Charge, Length), "y-component of a per-atom electric dipole moment.";
    MUZ: "muz", Float, Scalar, Product(Charge, Length), "z-component of a per-atom electric dipole moment.";
    NAME: "name", Str, Scalar, NotAQuantity, "Human-readable atom name (e.g. \"CA\").";
    OCCUPANCY: "occupancy", Float, Scalar, Dimensionless, "Crystallographic occupancy of a site, a fraction.";
    QUATI: "quati", Float, Scalar, Dimensionless, "First imaginary component of a per-atom orientation quaternion.";
    QUATJ: "quatj", Float, Scalar, Dimensionless, "Second imaginary component of a per-atom orientation quaternion.";
    QUATK: "quatk", Float, Scalar, Dimensionless, "Third imaginary component of a per-atom orientation quaternion.";
    QUATW: "quatw", Float, Scalar, Dimensionless, "Real part of a per-atom orientation quaternion.";
    REACT_ID: "react_id", Uint, Scalar, NotAQuantity, "Identifier pairing an atom of a LAMMPS `fix bond/react` pre-reaction template with the same atom of the post-reaction one.";
    RES_ID: "res_id", Uint, Scalar, NotAQuantity, "Residue identifier. Unsigned like every other id in the vocabulary; a file with negative residue numbers is renumbered at the reader boundary, not accommodated by the schema.";
    RES_NAME: "res_name", Str, Scalar, NotAQuantity, "Residue name (e.g. \"ALA\").";
    STEREO: "stereo", Str, Scalar, NotAQuantity, "Stereo marker. On an atom: tetrahedral chirality `\"CW\"` / `\"CCW\"` (`\"unspecified\"` from perception). On a bond: a line notation's direction `\"up\"` / `\"down\"`, or perceived double-bond geometry `\"E\"` / `\"Z\"` / `\"either\"` / `\"none\"`.";
    STYLE: "style", Str, Scalar, NotAQuantity, "Force-field style of a relation row, picking among styles of one category that hold the row's `type` (hybrid styles).";
    TYPE: "type", Str, Scalar, NotAQuantity, "Force-field type label. Always a String: a label is what survives a round trip through a force field. Numeric ordinals live in `type_id`.";
    TYPE_ID: "type_id", Uint, Scalar, NotAQuantity, "Numeric type ordinal as used by formats that number their types (LAMMPS). Format-local; the force field reads `type`.";
    VX: "vx", Float, Scalar, Of(Velocity), "x-velocity. Unit follows the force field's `units` setting; molrs stores raw numbers.";
    VY: "vy", Float, Scalar, Of(Velocity), "y-velocity.";
    VZ: "vz", Float, Scalar, Of(Velocity), "z-velocity.";
    X: "x", Float, Scalar, Of(Length), "Cartesian x-coordinate. Unit follows the force field / file format; molrs stores raw numbers.";
    Y: "y", Float, Scalar, Of(Length), "Cartesian y-coordinate.";
    Z: "z", Float, Scalar, Of(Length), "Cartesian z-coordinate.";
}

macro_rules! blocks {
    ($(
        $name:ident = $value:literal,
        $kind:expr,
        $endpoints:expr,
        $required:expr,
        $optional:expr,
        $doc:literal;
    )*) => {
        /// Canonical block names — the `name` of every [`SCHEMA_BLOCKS`] entry.
        ///
        /// Emitted by the `blocks!` table macro from the same literal as the table, so
        /// `block_names::BONDS` is the block the schema declares.
        pub mod block_names {
            $(
                #[doc = $doc]
                pub const $name: &str = $value;
            )*

            /// The bonded-topology relation blocks, in increasing arity.
            pub const TOPOLOGY: [&str; 4] = [BONDS, ANGLES, DIHEDRALS, IMPROPERS];
        }

        /// Every canonical block, sorted by name.
        pub static SCHEMA_BLOCKS: &[BlockSpec] = &[
            $(
                BlockSpec {
                    name: block_names::$name,
                    row_kind: $kind,
                    endpoints: $endpoints,
                    required: $required,
                    optional: $optional,
                    open: true,
                    doc: $doc,
                },
            )*
        ];

        /// `(const name, value)` for every [`block_names`] scalar.
        ///
        /// The bindings loop this instead of naming `ATOMS`, `BONDS`, … again.
        pub static BLOCK_NAMES: &[NamedConst] = &[
            $(
                NamedConst { const_name: stringify!($name), value: block_names::$name },
            )*
        ];

        /// Block groups exported beside the scalar block names. `TOPOLOGY` is
        /// not itself a block.
        pub static BLOCK_GROUPS: &[KeyGroup] = &[
            KeyGroup { const_name: "TOPOLOGY", keys: &block_names::TOPOLOGY },
        ];
    };
}

blocks! {
    ANGLES = "angles",
    RowKind::Relation { arity: 3 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMK, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ, consts::ATOMK],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE],
    "Three-body angle terms; `atomj` is the vertex.";
    ATOMS = "atoms",
    RowKind::Node,
    None,
    &[],
    &[consts::X, consts::Y, consts::Z, consts::VX, consts::VY, consts::VZ, consts::FX, consts::FY, consts::FZ, consts::IX, consts::IY, consts::IZ, consts::ID, consts::ATOMIC_NUMBER, consts::ELEMENT, consts::TYPE, consts::TYPE_ID, consts::NAME, consts::CHARGE, consts::FORMAL_CHARGE, consts::ATOM_MAP, consts::IS_AROMATIC, consts::H_COUNT, consts::ISOTOPE, consts::STEREO, consts::MASS, consts::MOL_ID, consts::REACT_ID, consts::RES_ID, consts::RES_NAME, consts::CHAIN, consts::ICODE, consts::ALTLOC, consts::HETERO, consts::OCCUPANCY, consts::B_FACTOR, consts::BEAD_TYPE, consts::FREE, consts::QUATW, consts::QUATI, consts::QUATJ, consts::QUATK, consts::MUX, consts::MUY, consts::MUZ, consts::AXIS_X, consts::AXIS_Y, consts::AXIS_Z],
    "Per-atom properties. The node table relation blocks index into.";
    BONDS = "bonds",
    RowKind::Relation { arity: 2 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE, consts::BOND_TYPE, consts::BOND_NUMBER, consts::STEREO],
    "Two-body bond terms.";
    CMAPS = "cmaps",
    RowKind::Relation { arity: 5 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMK, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOML, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMM, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ, consts::ATOMK, consts::ATOML, consts::ATOMM],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE],
    "Five-body CMAP cross terms: the dihedrals `atomi-atomj-atomk-atoml` and `atomj-atomk-atoml-atomm` (CHARMM phi/psi).";
    CONSTRAINTS = "constraints",
    RowKind::Relation { arity: 2 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE],
    "Holonomic distance constraints between two atoms; a per-instance length rides as a further column (`r0`).";
    DIHEDRALS = "dihedrals",
    RowKind::Relation { arity: 4 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMK, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOML, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ, consts::ATOMK, consts::ATOML],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE, consts::EXCLUDE_14],
    "Four-body proper torsion terms.";
    DRUDES = "drudes",
    RowKind::Relation { arity: 2 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE],
    "Drude oscillators: `atomi` is the core atom, `atomj` its Drude particle; both are `atoms` rows.";
    EXCLUSIONS = "exclusions",
    RowKind::Relation { arity: 2 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ],
    &[],
    "Pairs excluded from non-bonded interaction (PME real-space correction).";
    IMPROPERS = "impropers",
    RowKind::Relation { arity: 4 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMK, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOML, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ, consts::ATOMK, consts::ATOML],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE, consts::EXCLUDE_14],
    "Four-body improper terms enforcing planarity or chirality.";
    MEMBERS = "members",
    RowKind::Relation { arity: 2 },
    Some(EndpointSpec { columns: &[(consts::IBEAD, EndpointTarget::Block(block_names::ATOMS)), (MEMBER_ATOM, EndpointTarget::Declared)] }),
    &[consts::IBEAD, MEMBER_ATOM],
    &[],
    "Coarse-grained membership, one row per (bead, atom): `ibead` is the bead's `atoms` row; `atom` indexes the all-atom block its `targets` names (`/frame/atoms`, …) and is an opaque handle without one.";
    PAIRS = "pairs",
    RowKind::Relation { arity: 2 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE, consts::IS_14],
    "Intramolecular non-bonded pair list. Usually consumer-built (`ff::potential::intramolecular_pairs`); GROMACS `.top` also carries one as its `[ pairs ]` section, which is by definition the 1-4 list and is read in with `is_14` set on every row.";
    VIRTUAL_SITES = "virtual_sites",
    RowKind::Relation { arity: 4 },
    Some(EndpointSpec { columns: &[(consts::ATOMI, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMJ, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOMK, EndpointTarget::Block(block_names::ATOMS)), (consts::ATOML, EndpointTarget::Block(block_names::ATOMS))] }),
    &[consts::ATOMI, consts::ATOMJ, consts::ATOMK, consts::ATOML],
    &[consts::TYPE, consts::TYPE_ID, consts::STYLE],
    "Virtual sites: `atomi` is the site (an `atoms` row, usually massless), built from `atomj`, `atomk`, `atoml`; a site built from fewer atoms leaves the trailing endpoints null.";
}

/// The `members` column naming the grouped atom. Not a canonical key: its
/// target is declared per block (`targets`), so it has no one meaning to fix
/// in the vocabulary.
pub const MEMBER_ATOM: &str = "atom";

/// The per-pair override columns a `pairs` row may carry: what LAMMPS cannot
/// express per pair (a GROMACS `[ pairs ]` row with parameters, an OpenMM
/// exception, an AMBER dihedral's own SCEE / SCNB). `epsilon` / `sigma` (the
/// pair's LJ parameters), `charge_product` (qᵢqⱼ, e²), `lj_scale` /
/// `coul_scale` (its weights in place of `special_bonds`). Float, a null cell
/// (validity mask) taking the force field's value. Not canonical keys —
/// `epsilon` names other things elsewhere — so they live here, as the one
/// list the force-field compiler reads and the LAMMPS writers refuse.
pub const PAIR_OVERRIDE_COLUMNS: [&str; 5] = [
    "epsilon",
    "sigma",
    "charge_product",
    "lj_scale",
    "coul_scale",
];

/// Canonical spec for a column key, or `None` if the key is unconstrained.
pub fn column(key: &str) -> Option<&'static ColumnSpec> {
    SCHEMA_COLUMNS
        .binary_search_by(|c| c.key.cmp(key))
        .ok()
        .map(|i| &SCHEMA_COLUMNS[i])
}

/// Canonical spec for a block name, or `None` if the block is not in the
/// vocabulary (which is legal — the block set is open).
pub fn block(name: &str) -> Option<&'static BlockSpec> {
    SCHEMA_BLOCKS
        .binary_search_by(|b| b.name.cmp(name))
        .ok()
        .map(|i| &SCHEMA_BLOCKS[i])
}

/// One row reference of a block: a column whose values are 0-based rows of
/// `target`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowReference {
    /// The referencing column.
    pub column: String,
    /// The block it indexes: `<block>` of the same frame, or
    /// `/<section>/<block>` of a frame-shaped section of the same record.
    pub target: String,
}

impl RowReference {
    /// Whether the target is a block of the same frame (or, on a trajectory,
    /// of the same resolved frame) rather than an absolute
    /// `/<section>/<block>`.
    pub fn is_local(&self) -> bool {
        !self.target.starts_with('/')
    }
}

/// Check the spelling of a row-reference target: `<block>`, or
/// `/<section>/<block>` naming a frame-shaped section. A trajectory block is
/// never a target (its row count is not fixed), so `/trajectory/…` is
/// refused.
///
/// # Errors
///
/// A message naming the target.
pub fn check_target(target: &str) -> Result<(), String> {
    let well_formed = match target.strip_prefix('/') {
        Some(rest) => matches!(
            rest.split('/').collect::<Vec<_>>().as_slice(),
            [section, block] if !section.is_empty() && !block.is_empty()
        ),
        None => !target.is_empty() && !target.contains('/'),
    };
    if !well_formed {
        return Err(format!(
            "row-reference target {target:?} is neither `<block>` nor `/<section>/<block>`"
        ));
    }
    if target.starts_with("/trajectory/") {
        return Err(format!(
            "row-reference target {target:?} names a trajectory block, whose row count is not \
             fixed; a trajectory block is never a target"
        ));
    }
    Ok(())
}

/// The row-reference rule: which columns of block `name` index rows of which
/// block, honouring the block's declared `targets` over the defaults.
///
/// - A canonical [`RowKind::Relation`] block lists its spec's endpoint
///   columns without consulting `has_column` (a subset or replicate that
///   finds one missing refuses rather than copying stale indices): an
///   [`EndpointTarget::Block`] endpoint references that block unless
///   `declared` names another target for it; an
///   [`EndpointTarget::Declared`] endpoint (`members.atom`) references
///   something only when `declared` says what.
/// - Any other block reads the present subset of [`consts::ENDPOINTS`], in
///   position order, as references into `atoms` (or into what `declared`
///   names) — `MolGraph` mints one block per relation kind, and those carry
///   no spec.
/// - Every other column `declared` names is a reference too, appended in
///   declaration order.
///
/// Empty when the block references nothing. The one rule
/// [`Validator`] (range checks),
/// [`Frame::subset`](crate::core::Frame::subset) (renumbering) and
/// [`Frame::replicate`](crate::core::Frame::replicate) (offsetting)
/// follow; a downstream crate that rewrites row indices asks this function
/// rather than keeping its own list.
///
/// # Examples
///
/// ```
/// use molrs::core::schema::{block_names, relation_endpoints};
///
/// let refs = |name, has: &dyn Fn(&str) -> bool, declared: &[(&str, &str)]| {
///     relation_endpoints(name, has, declared)
///         .into_iter()
///         .map(|r| (r.column, r.target))
///         .collect::<Vec<_>>()
/// };
/// let pair = |c: &str, t: &str| (c.to_string(), t.to_string());
/// // A canonical relation answers from its spec.
/// assert_eq!(
///     refs(block_names::BONDS, &|_| false, &[]),
///     [pair("atomi", "atoms"), pair("atomj", "atoms")],
/// );
/// // `members.atom` references only what its block declares.
/// assert_eq!(refs("members", &|_| true, &[]), [pair("ibead", "atoms")]);
/// assert_eq!(
///     refs("members", &|_| true, &[("atom", "/frame/atoms")]),
///     [pair("ibead", "atoms"), pair("atom", "/frame/atoms")],
/// );
/// // An unspecified block is a relation iff it carries endpoint columns.
/// assert_eq!(refs("ports", &|k| k == "atomi", &[]), [pair("atomi", "atoms")]);
/// assert!(refs("cell", &|_| false, &[]).is_empty());
/// ```
pub fn relation_endpoints(
    name: &str,
    has_column: impl Fn(&str) -> bool,
    declared: &[(&str, &str)],
) -> Vec<RowReference> {
    let declared_for = |column: &str| {
        declared
            .iter()
            .find(|(c, _)| *c == column)
            .map(|(_, target)| *target)
    };
    let mut refs: Vec<RowReference> = Vec::new();
    let mut push = |column: &str, target: &str| {
        refs.push(RowReference {
            column: column.to_string(),
            target: target.to_string(),
        });
    };
    match block(name).and_then(|spec| spec.endpoints) {
        Some(endpoints) => {
            for &(column, target) in endpoints.columns {
                let target = declared_for(column).or(match target {
                    EndpointTarget::Block(block) => Some(block),
                    EndpointTarget::Declared => None,
                });
                if let Some(target) = target {
                    push(column, target);
                }
            }
        }
        None => {
            for column in consts::ENDPOINTS {
                if has_column(column) {
                    push(column, declared_for(column).unwrap_or(block_names::ATOMS));
                }
            }
        }
    }
    for &(column, target) in declared {
        if !refs.iter().any(|r| r.column == column) {
            refs.push(RowReference {
                column: column.to_string(),
                target: target.to_string(),
            });
        }
    }
    refs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn columns_sorted_unique_and_documented() {
        let mut prev = "";
        for c in SCHEMA_COLUMNS {
            assert!(c.key > prev, "SCHEMA_COLUMNS not sorted at {:?}", c.key);
            prev = c.key;
            assert!(!c.doc.is_empty(), "{} has an empty doc", c.key);
            assert!(!c.const_name.is_empty(), "{} has no const_name", c.key);
        }
        let names: HashSet<_> = SCHEMA_COLUMNS.iter().map(|c| c.const_name).collect();
        assert_eq!(names.len(), SCHEMA_COLUMNS.len(), "const_name collision");
    }

    #[test]
    fn blocks_sorted_unique_and_documented() {
        let mut prev = "";
        for b in SCHEMA_BLOCKS {
            assert!(b.name > prev, "SCHEMA_BLOCKS not sorted at {:?}", b.name);
            prev = b.name;
            assert!(!b.doc.is_empty(), "{} has an empty doc", b.name);
        }
    }

    #[test]
    fn block_column_references_resolve() {
        for b in SCHEMA_BLOCKS {
            // A declared endpoint (`members.atom`) has no one meaning, so it
            // is deliberately not a canonical key.
            let declared: Vec<&str> = b
                .endpoints
                .map(|e| {
                    e.columns
                        .iter()
                        .filter(|(_, t)| *t == EndpointTarget::Declared)
                        .map(|(c, _)| *c)
                        .collect()
                })
                .unwrap_or_default();
            for key in b
                .required
                .iter()
                .copied()
                .chain(b.optional.iter().copied())
                .chain(b.endpoint_columns())
                .filter(|k| !declared.contains(k))
            {
                assert!(
                    column(key).is_some(),
                    "block '{}' references unknown column '{key}'",
                    b.name
                );
            }
        }
    }

    #[test]
    fn relation_arity_matches_endpoint_count() {
        for b in SCHEMA_BLOCKS {
            match b.row_kind {
                RowKind::Relation { arity } => {
                    let e = b.endpoints.expect("relation block needs endpoints");
                    assert_eq!(e.columns.len(), arity, "{} arity mismatch", b.name);
                }
                _ => assert!(
                    b.endpoints.is_none(),
                    "{} is not a relation but declares endpoints",
                    b.name
                ),
            }
        }
    }

    #[test]
    fn endpoint_targets_are_node_blocks() {
        for b in SCHEMA_BLOCKS {
            for (_, target) in b.endpoints.map(|e| e.columns).unwrap_or(&[]) {
                if let EndpointTarget::Block(target) = target {
                    let target = block(target).expect("endpoint target must be a known block");
                    assert_eq!(
                        target.row_kind,
                        RowKind::Node,
                        "{} target is not a node table",
                        b.name
                    );
                }
            }
        }
    }

    #[test]
    fn lookup_finds_every_key() {
        for c in SCHEMA_COLUMNS {
            assert_eq!(column(c.key).map(|s| s.key), Some(c.key));
        }
        assert!(column("definitely_not_a_canonical_key").is_none());
    }

    #[test]
    fn const_names_are_the_lowercase_keys() {
        for c in SCHEMA_COLUMNS {
            assert_eq!(
                c.key,
                c.const_name.to_ascii_lowercase(),
                "key and const_name disagree for {}",
                c.const_name
            );
        }
        for spec in BLOCK_NAMES {
            assert_eq!(
                spec.value,
                spec.const_name.to_ascii_lowercase(),
                "block name and const disagree for {}",
                spec.const_name
            );
        }
    }

    #[test]
    fn key_groups_name_columns() {
        let mut seen = HashSet::new();
        for g in KEY_GROUPS {
            assert!(
                seen.insert(g.const_name),
                "duplicate group {}",
                g.const_name
            );
            assert!(!g.keys.is_empty(), "{} is empty", g.const_name);
            for k in g.keys {
                assert!(
                    column(k).is_some(),
                    "group {} contains unknown key {k}",
                    g.const_name
                );
            }
        }
        for name in [
            "COORDS",
            "IMAGES",
            "VELOCITIES",
            "FORCES",
            "QUAT",
            "DIPOLE",
            "AXIS",
            "ENDPOINTS",
        ] {
            assert!(seen.contains(name), "missing group {name}");
        }
    }

    #[test]
    fn exclude_14_registered_bool_scalar() {
        // amber-prmtop-complete-01-structure ac-001: the AMBER negative-3rd-
        // pointer flag is the sibling of `is_14`, so it carries the same
        // dtype convention — registered, not left to a reader's discipline.
        let spec = column("exclude_14").expect("exclude_14 must be in SCHEMA_COLUMNS");
        assert_eq!(spec.const_name, "EXCLUDE_14");
        assert_eq!(spec.dtype, DType::Bool);
        assert_eq!(spec.shape, ColShape::Scalar);
        assert!(matches!(spec.dimension, ColumnDimension::NotAQuantity));
        for name in ["dihedrals", "impropers"] {
            let b = block(name).expect("block must be in the vocabulary");
            assert!(
                b.optional.contains(&"exclude_14"),
                "block '{name}' does not list exclude_14 as optional"
            );
        }
    }

    #[test]
    fn type_and_type_id_are_different_quantities() {
        // The axiom in this module's doc, as a test: one name, one dtype.
        assert_eq!(column("type").unwrap().dtype, DType::String);
        assert_eq!(column("type_id").unwrap().dtype, DType::Uint);
    }

    #[test]
    fn every_identifier_key_shares_one_dtype() {
        // Scans the table rather than asserting a hand-written list, so a new
        // `*_id` key cannot be added at a different dtype without this failing.
        let ids: Vec<&ColumnSpec> = SCHEMA_COLUMNS
            .iter()
            .filter(|c| {
                c.key == "id"
                    || c.key.ends_with("_id")
                    || matches!(c.key, "atomi" | "atomj" | "atomk" | "atoml" | "atomm")
            })
            .collect();
        assert!(ids.len() >= 8, "identifier scan found only {}", ids.len());
        for c in &ids {
            assert_eq!(
                c.dtype,
                DType::Uint,
                "identifier '{}' is {} — every identifier is Uint",
                c.key,
                c.dtype
            );
        }
    }

    #[test]
    fn block_names_agree_with_the_table() {
        let mut table: Vec<&str> = SCHEMA_BLOCKS.iter().map(|b| b.name).collect();
        let mut named: Vec<&str> = BLOCK_NAMES.iter().map(|spec| spec.value).collect();
        table.sort_unstable();
        named.sort_unstable();
        assert_eq!(table, named, "every BlockSpec has exactly one constant");
        for name in block_names::TOPOLOGY {
            assert!(matches!(
                block(name).map(|b| b.row_kind),
                Some(RowKind::Relation { .. })
            ));
        }
        assert_eq!(
            BLOCK_GROUPS
                .iter()
                .map(|g| g.const_name)
                .collect::<Vec<_>>(),
            vec!["TOPOLOGY"]
        );
        assert_eq!(BLOCK_GROUPS[0].keys, block_names::TOPOLOGY.as_slice());
    }

    #[test]
    fn free_is_a_bool_atom_column() {
        let spec = column(consts::FREE).expect("free is canonical");
        assert_eq!(spec.dtype, DType::Bool);
        assert_eq!(spec.shape, ColShape::Scalar);
        assert!(
            block(block_names::ATOMS)
                .unwrap()
                .optional
                .contains(&"free")
        );
    }

    // ---- relation_endpoints ----

    fn refs(
        name: &str,
        has: impl Fn(&str) -> bool,
        declared: &[(&str, &str)],
    ) -> Vec<(String, String)> {
        relation_endpoints(name, has, declared)
            .into_iter()
            .map(|r| (r.column, r.target))
            .collect()
    }

    fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
        list.iter()
            .map(|(c, t)| (c.to_string(), t.to_string()))
            .collect()
    }

    #[test]
    fn relation_endpoints_reads_a_spec_relation_without_consulting_columns() {
        assert_eq!(
            refs("bonds", |_| false, &[]),
            pairs(&[("atomi", "atoms"), ("atomj", "atoms")])
        );
        assert_eq!(
            refs("angles", |_| false, &[]),
            pairs(&[("atomi", "atoms"), ("atomj", "atoms"), ("atomk", "atoms")])
        );
    }

    #[test]
    fn cmaps_is_a_five_endpoint_relation_on_atomi_through_atomm() {
        assert_eq!(column(consts::ATOMM).map(|c| c.dtype), Some(DType::Uint));
        let spec = block(block_names::CMAPS).expect("cmaps is canonical");
        assert_eq!(spec.row_kind, RowKind::Relation { arity: 5 });
        assert_eq!(spec.required, consts::ENDPOINTS.as_slice());
        for key in [consts::TYPE, consts::TYPE_ID, consts::STYLE] {
            assert!(spec.optional.contains(&key), "{key}");
        }
        let all = [
            ("atomi", "atoms"),
            ("atomj", "atoms"),
            ("atomk", "atoms"),
            ("atoml", "atoms"),
            ("atomm", "atoms"),
        ];
        assert_eq!(refs("cmaps", |_| false, &[]), pairs(&all));
        // An unspecified block reads all five endpoint keys too.
        assert_eq!(refs("cross_terms", |_| true, &[]), pairs(&all));
    }

    #[test]
    fn relation_endpoints_infers_atoms_endpoints_for_an_unspecified_block() {
        let present = |k: &str| matches!(k, "atomi" | "atomj" | "port_kind");
        assert_eq!(
            refs("ports", present, &[]),
            pairs(&[("atomi", "atoms"), ("atomj", "atoms")])
        );
        assert!(refs("cell", |_| false, &[]).is_empty());
    }

    #[test]
    fn declared_targets_override_defaults_and_add_references() {
        assert_eq!(refs("members", |_| true, &[]), pairs(&[("ibead", "atoms")]));
        assert_eq!(
            refs("members", |_| true, &[("atom", "/frame/atoms")]),
            pairs(&[("ibead", "atoms"), ("atom", "/frame/atoms")])
        );
        assert_eq!(
            refs("bonds", |_| true, &[("atomj", "sites"), ("site", "sites")]),
            pairs(&[("atomi", "atoms"), ("atomj", "sites"), ("site", "sites")])
        );
        assert_eq!(
            refs("links", |k| k == "atomi", &[("other", "beads")]),
            pairs(&[("atomi", "atoms"), ("other", "beads")])
        );
    }

    #[test]
    fn targets_are_a_block_or_a_section_block_and_never_a_trajectory_block() {
        for ok in ["atoms", "/frame/atoms", "/system/atoms"] {
            assert!(check_target(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "/",
            "a/b",
            "/frame",
            "/frame/",
            "/a/b/c",
            "/trajectory/atoms",
        ] {
            assert!(check_target(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_new_topology_keys_and_blocks_are_canonical() {
        for (key, dtype) in [
            ("fx", DType::Float),
            ("formal_charge", DType::I64),
            ("atom_map", DType::Uint),
            ("chain", DType::String),
            ("icode", DType::String),
            ("altloc", DType::String),
            ("occupancy", DType::Float),
            ("b_factor", DType::Float),
            ("ibead", DType::Uint),
            ("style", DType::String),
        ] {
            assert_eq!(column(key).map(|c| c.dtype), Some(dtype), "{key}");
        }
        for name in ["constraints", "drudes", "members", "virtual_sites"] {
            assert!(matches!(
                block(name).map(|b| b.row_kind),
                Some(RowKind::Relation { .. })
            ));
        }
        for key in ["chain_id", "res_seq", "b_iso"] {
            assert!(column(key).is_none(), "{key} is not canonical");
        }
    }
}
