//! Domain-agnostic dynamic graph for editing-oriented CRUD operations: [`MolGraph`].

use indexmap::IndexMap;
use std::collections::{HashMap, HashSet};
use std::ops::{Index, IndexMut};

use ndarray::ArrayD;
use slotmap::{Key, KeyData, SecondaryMap, new_key_type};
use smallvec::SmallVec;

use crate::core::Block;
use crate::core::Frame;
use crate::core::MolRsError;
use crate::core::keys;
use crate::core::{EntityCell, EntityTable, Validity};
use crate::op::{F, I, Idx};

use crate::core::keys::FRAG_ID;

// ---------------------------------------------------------------------------
// PropValue
// ---------------------------------------------------------------------------

/// Heterogeneous property value stored in an [`Atom`].
#[derive(Debug, Clone, PartialEq)]
pub enum PropValue {
    F64(f64),
    Str(String),
    Int(I),
    Bool(bool),
}

impl PropValue {
    /// Numeric value as `f64`, accepting both `F64` and `Int` variants.
    ///
    /// Use this for quantities that are conceptually numeric but may be stored
    /// as either type depending on the producer — e.g. a bond `"order"` written
    /// as `2` (Int) vs `2.0` (F64). Returns `None` for non-numeric (`Str`,
    /// `Bool`).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropValue::F64(v) => Some(*v),
            PropValue::Int(v) => Some(*v as f64),
            PropValue::Str(_) | PropValue::Bool(_) => None,
        }
    }
}

impl From<f64> for PropValue {
    fn from(v: f64) -> Self {
        PropValue::F64(v)
    }
}
impl From<I> for PropValue {
    fn from(v: I) -> Self {
        PropValue::Int(v)
    }
}
impl From<&str> for PropValue {
    fn from(v: &str) -> Self {
        PropValue::Str(v.to_owned())
    }
}
impl From<bool> for PropValue {
    fn from(v: bool) -> Self {
        PropValue::Bool(v)
    }
}
impl From<String> for PropValue {
    fn from(v: String) -> Self {
        PropValue::Str(v)
    }
}

/// Admit a value into a node or relation column only if it can be stored at
/// the dtype the Frame schema declares for `key`, widening where width is not
/// semantics and refusing otherwise.
///
/// A key the vocabulary does not declare (`tag`, `frag_id`, a perceived fact,
/// a per-instance force-field parameter) carries whatever the caller stores:
/// no frame column type contradicts it.
///
/// For a declared key the refusal belongs *here*, at the write, because the
/// declared dtype is what every later stage already assumes: [`MolGraph::to_frame`]
/// must hand [`Block::insert`] a column of exactly that dtype, the infallible
/// leaf constructors ([`Atomistic::add_atom_xyz`](crate::core::Atomistic::add_atom_xyz)) write their components into
/// a column whose element type a stray write has already fixed, and the Python
/// `Block` boundary re-states the same rule. Every one of those stages is past
/// the point where the caller still holds the value — a `set_node` that
/// accepted a string `"x"` would hand back a graph whose only remaining moves
/// are to lose the property or to fail on the way out. [`set_node`](MolGraph::set_node)
/// has a `Result`, so it is the last door where the caller can still act.
///
/// Width is not semantics: an `Int` under a float key (`x`, `charge` written
/// as `1`) is widened to `F64`. A *sign* is semantics, so a negative under an
/// unsigned key (`id`, `mol_id`, `type_id`, …) keeps its own refusal naming
/// the sign rather than a dtype mismatch. A declared dtype the graph store has
/// no element type for (a narrow width, a complex pair) can hold no
/// [`PropValue`] at all, so every value is refused there rather than stored at
/// a width `to_frame` could not emit.
///
/// The match over the declared dtype is exhaustive by construction: a dtype
/// added to the vocabulary is a compile error here, not a value that slips
/// through a catch-all and detonates downstream.
///
/// # Errors
///
/// [`MolRsError::Validation`] when the value's element type cannot be stored
/// at the declared dtype, naming the key and both dtypes
/// (`'x' is declared float by the Frame schema; got string`), or when a
/// negative is offered under an unsigned key.
fn coerce_canonical(key: &str, pv: PropValue) -> Result<PropValue, MolRsError> {
    use crate::core::DType;

    let Some(declared) = crate::core::schema::column(key).map(|spec| spec.dtype) else {
        return Ok(pv);
    };
    let offered = match &pv {
        PropValue::F64(_) => DType::Float,
        PropValue::Int(_) => DType::Int,
        PropValue::Str(_) => DType::String,
        PropValue::Bool(_) => DType::Bool,
    };
    let refuse = || {
        Err(MolRsError::validation(format!(
            "'{key}' is declared {} by the Frame schema; got {}",
            declared.name(),
            offered.name()
        )))
    };

    match declared {
        DType::Float => match pv {
            PropValue::Int(v) => Ok(PropValue::F64(v as f64)),
            PropValue::F64(_) => Ok(pv),
            PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::Int => match pv {
            PropValue::Int(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::Uint => match pv {
            PropValue::Int(v) if v < 0 => Err(MolRsError::validation(format!(
                "'{key}' is declared unsigned by the Frame schema; got {v}"
            ))),
            PropValue::Int(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::String => match pv {
            PropValue::Str(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Int(_) | PropValue::Bool(_) => refuse(),
        },
        DType::Bool => match pv {
            PropValue::Bool(_) => Ok(pv),
            PropValue::F64(_) | PropValue::Int(_) | PropValue::Str(_) => refuse(),
        },
        // A wide signed key (`formal_charge`) holds any integer the graph's
        // `Int` holds, and an integral float (SMILES stores charges as f64);
        // `to_frame` widens either to the declared `i64`.
        DType::I64 => match pv {
            PropValue::Int(_) => Ok(pv),
            PropValue::F64(v) if v.is_finite() && v.fract() == 0.0 => Ok(pv),
            PropValue::F64(v) => Err(MolRsError::validation(format!(
                "'{key}' is declared i64 by the Frame schema; got {v}, not an integer"
            ))),
            PropValue::Str(_) | PropValue::Bool(_) => refuse(),
        },
        DType::I8 | DType::I16 | DType::U8 | DType::U16 | DType::U32 | DType::C64 | DType::C128 => {
            refuse()
        }
    }
}

/// Materialize `table`'s column `key` into `block` at the dtype the Frame
/// schema declares for that key, carrying the column's validity mask.
///
/// [`EntityTable`] has no unsigned column, so a canonical unsigned field
/// (`id`, `mol_id`, `type_id`, …) lives in the store as [`I`] and has to be
/// re-typed on the way out: [`Block::insert`] validates the key against the
/// schema vocabulary, so handing it the signed column is *rejected* — and the
/// column would vanish from the frame instead of reaching the caller.
///
/// A component is set per entity, so a column is in general partial: the rows
/// no entity ever wrote hold the element type's default. Emitting them as
/// values would state a `frag_id` of instance zero, a charge of zero, an empty
/// name — so the mask travels with the column through
/// [`Block::insert_nullable`], which normalises the fully-populated case back
/// to a plain column.
///
/// Returns `false` when `key` names no column of `table`.
fn emit_column<K: Key>(
    block: &mut Block,
    table: &EntityTable<K>,
    key: &str,
) -> Result<bool, MolRsError> {
    use crate::core::DType;
    use ndarray::Array1;

    let declared = crate::core::schema::column(key).map(|spec| spec.dtype);
    if declared == Some(DType::I64) {
        // A wide signed key: the table holds it as `Int` or as an integral
        // `F64` (see `coerce_canonical`); the frame column is `i64`.
        let (wide, valid): (Vec<i64>, Vec<bool>) = if let Ok((data, valid)) = table.column_f64(key)
        {
            (data.iter().map(|&v| v as i64).collect(), mask(valid))
        } else if let Ok((data, valid)) = table.column_i32(key) {
            (data.iter().map(|&v| i64::from(v)).collect(), mask(valid))
        } else {
            return Ok(false);
        };
        block
            .insert_nullable(key, Array1::from_vec(wide).into_dyn(), valid)
            .map_err(|e| MolRsError::validation(e.to_string()))?;
        return Ok(true);
    }
    let inserted = if let Ok((data, valid)) = table.column_f64(key) {
        block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
    } else if let Ok((data, valid)) = table.column_i32(key) {
        if crate::core::schema::column(key).is_some_and(|spec| spec.dtype == DType::Uint) {
            let unsigned: Vec<Idx> = data
                .iter()
                .map(|&v| {
                    Idx::try_from(v).map_err(|_| {
                        MolRsError::validation(format!(
                            "'{key}' is declared unsigned by the Frame schema; got {v}"
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            block.insert_nullable(key, Array1::from_vec(unsigned).into_dyn(), mask(valid))
        } else {
            block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
        }
    } else if let Ok((data, valid)) = table.column_str(key) {
        block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
    } else if let Ok((data, valid)) = table.column_bool(key) {
        block.insert_nullable(key, Array1::from_vec(data.to_vec()).into_dyn(), mask(valid))
    } else {
        return Ok(false);
    };
    inserted.map_err(|e| MolRsError::validation(e.to_string()))?;
    Ok(true)
}

/// The block-side form of an [`EntityTable`] validity mask.
fn mask(valid: &Validity) -> Vec<bool> {
    valid.as_slice().to_vec()
}

/// A [`Block`]'s columns in block order, each typed by element and paired with
/// its validity mask — the reading counterpart of [`emit_column`].
///
/// Both halves of [`MolGraph::extend_from_frame`] need the same thing: walk a block's
/// columns once, then ask each row for the properties it actually carries. A
/// masked-off cell holds the element type's default, which is a value like any
/// other to the block, so the mask is what keeps an unset `frag_id` from
/// arriving as instance zero. Block order is kept so the graph's components
/// are first written — and later re-emitted by [`MolGraph::to_frame`] — in
/// the order the frame carried them.
struct MaskedColumns<'a> {
    cols: Vec<(&'a str, TypedColumn<'a>, Option<&'a [bool]>)>,
}

/// One block column borrowed at the element type [`MaskedColumns`] reads.
///
/// Unsigned columns are kept apart from signed ones because the canonical
/// `id` / `mol_id` / `type_id` fields are `uint` in the Frame schema and the
/// graph stores them signed: they need narrowing, not a cast.
enum TypedColumn<'a> {
    Float(&'a ArrayD<F>),
    Int(&'a ArrayD<I>),
    I64(&'a ArrayD<i64>),
    Uint(&'a ArrayD<Idx>),
    Str(&'a ArrayD<String>),
    Bool(&'a ArrayD<bool>),
}

impl<'a> MaskedColumns<'a> {
    /// Type `block`'s columns, skipping the keys in `skip` (a relation
    /// block's endpoint columns, which are structure rather than properties).
    /// Columns of any other element type are not graph properties and are
    /// left out.
    fn of(block: &'a Block, skip: &[String]) -> Self {
        let mut cols = Vec::with_capacity(block.len());
        for key in block.keys() {
            if skip.iter().any(|s| s == key) {
                continue;
            }
            let typed = if let Some(arr) = block.get(key).and_then(|c| c.as_float()) {
                TypedColumn::Float(arr)
            } else if let Some(arr) = block.get(key).and_then(|c| c.as_int()) {
                TypedColumn::Int(arr)
            } else if let Some(arr) = block.get(key).and_then(|c| c.as_i64()) {
                TypedColumn::I64(arr)
            } else if let Some(arr) = block.get(key).and_then(|c| c.as_uint()) {
                TypedColumn::Uint(arr)
            } else if let Some(arr) = block.get(key).and_then(|c| c.as_string()) {
                TypedColumn::Str(arr)
            } else if let Some(arr) = block.get(key).and_then(|c| c.as_bool()) {
                TypedColumn::Bool(arr)
            } else {
                continue;
            };
            cols.push((key, typed, block.validity(key)));
        }
        MaskedColumns { cols }
    }

    /// The properties row `row` carries: one entry per column whose mask marks
    /// the row as holding a value, in block column order.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when an unsigned value exceeds the signed
    /// range the graph stores it in.
    fn cells(&self, row: usize) -> Result<Vec<(&'a str, PropValue)>, MolRsError> {
        let mut out: Vec<(&'a str, PropValue)> = Vec::with_capacity(self.cols.len());
        for &(key, ref typed, mask) in &self.cols {
            if !is_set(mask, row) {
                continue;
            }
            let value = match typed {
                #[allow(clippy::unnecessary_cast)]
                TypedColumn::Float(arr) => PropValue::F64(arr[[row]] as f64),
                TypedColumn::Int(arr) => PropValue::Int(arr[[row]]),
                TypedColumn::I64(arr) => PropValue::Int(narrow_i64(key, arr[[row]])?),
                TypedColumn::Uint(arr) => PropValue::Int(narrow_uint(key, arr[[row]])?),
                TypedColumn::Str(arr) => PropValue::Str(arr[[row]].clone()),
                TypedColumn::Bool(arr) => PropValue::Bool(arr[[row]]),
            };
            out.push((key, value));
        }
        Ok(out)
    }
}

/// Whether row `row` of a column with mask `mask` holds a value. A column with
/// no mask is fully valid — see [`Block::validity`].
fn is_set(mask: Option<&[bool]>, row: usize) -> bool {
    mask.is_none_or(|mask| mask[row])
}

/// The node handles a relation row addresses, or `None` when one of its
/// endpoint columns points past the rows the `"atoms"` block had.
fn endpoints_at(
    cols: &[&ArrayD<Idx>],
    row: usize,
    node_ids: &[NodeId],
) -> Option<SmallVec<[NodeId; 4]>> {
    let mut nodes: SmallVec<[NodeId; 4]> = SmallVec::new();
    for col in cols {
        nodes.push(*node_ids.get(col[[row]] as usize)?);
    }
    Some(nodes)
}

/// Narrow one value of a frame's `i64` column to the [`I`] the entity table
/// stores, refusing a value it cannot hold.
fn narrow_i64(key: &str, v: i64) -> Result<I, MolRsError> {
    I::try_from(v).map_err(|_| {
        MolRsError::validation(format!(
            "'{key}' value {v} exceeds the range the graph stores"
        ))
    })
}

/// Narrow one value of a frame's unsigned column to the signed [`I`] the entity
/// table stores, refusing the values that would wrap into a negative rather
/// than storing an identifier the next [`MolGraph::to_frame`] cannot re-widen.
fn narrow_uint(key: &str, v: Idx) -> Result<I, MolRsError> {
    I::try_from(v).map_err(|_| {
        MolRsError::validation(format!(
            "'{key}' value {v} exceeds the signed range the graph stores"
        ))
    })
}

// ---------------------------------------------------------------------------
// Atom  (dynamic node prop bag — the payload of every node, atom or bead)
// ---------------------------------------------------------------------------

/// A dynamic property bag representing a graph node (an atom or a bead).
///
/// All data — including coordinates (`"x"`, `"y"`, `"z"`), element symbol,
/// mass, charge, etc. — is stored as key-value pairs. `MolGraph` treats it
/// purely as an opaque node payload.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Atom {
    props: IndexMap<String, PropValue>,
}

impl Atom {
    /// Create an empty atom.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convenience: create an atom with symbol + xyz (via the [`crate::core::keys`]
    /// field convention — no literal field names).
    pub fn xyz(symbol: &str, x: f64, y: f64, z: f64) -> Self {
        let mut a = Self::new();
        a.set(keys::ELEMENT, symbol);
        a.set(keys::X, x);
        a.set(keys::Y, y);
        a.set(keys::Z, z);
        a
    }

    // ---- dict-like API ----

    /// Insert or update a property.
    pub fn set(&mut self, key: &str, val: impl Into<PropValue>) {
        self.props.insert(key.to_owned(), val.into());
    }

    /// Get a reference to a property value.
    pub fn get(&self, key: &str) -> Option<&PropValue> {
        self.props.get(key)
    }

    /// Get a mutable reference to a property value.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut PropValue> {
        self.props.get_mut(key)
    }

    /// Try to read a property as `f64`.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.props.get(key)? {
            PropValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// Try to read a property as `&str`.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.props.get(key)? {
            PropValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to read a property as `I`.
    pub fn get_int(&self, key: &str) -> Option<I> {
        match self.props.get(key)? {
            PropValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// The node's position `[x, y, z]`, when all three of
    /// [`keys::COORDS`] are `f64` props. Finiteness is not checked.
    pub fn position(&self) -> Option<[f64; 3]> {
        let [x, y, z] = keys::COORDS.map(|k| self.get_f64(k));
        Some([x?, y?, z?])
    }

    /// The atom's formal charge, in units of the elementary charge: its
    /// [`keys::FORMAL_CHARGE`] prop, rounded to the nearest integer, or `0`
    /// when the prop is absent or not a number.
    ///
    /// The prop is an `I64` column in a frame and an integral `f64` when the
    /// SMILES reader writes it; both read the same here. The partial charge
    /// ([`keys::CHARGE`]) is a different quantity and is never consulted.
    pub fn formal_charge(&self) -> i32 {
        self.get(keys::FORMAL_CHARGE)
            .and_then(PropValue::as_f64)
            .map_or(0, |q| q.round() as i32)
    }

    /// Check whether a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.props.contains_key(key)
    }

    /// Remove a property, returning its value if present.
    pub fn remove(&mut self, key: &str) -> Option<PropValue> {
        self.props.shift_remove(key)
    }

    /// Iterate over all property keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.props.keys().map(|k| k.as_str())
    }

    /// Iterate over all `(key, value)` property pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PropValue)> {
        self.props.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of properties.
    pub fn len(&self) -> usize {
        self.props.len()
    }

    /// Whether there are no properties.
    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }
}

impl Index<&str> for Atom {
    type Output = PropValue;
    fn index(&self, key: &str) -> &Self::Output {
        self.props
            .get(key)
            .unwrap_or_else(|| panic!("Atom does not contain key '{}'", key))
    }
}

impl IndexMut<&str> for Atom {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.props
            .get_mut(key)
            .unwrap_or_else(|| panic!("Atom does not contain key '{}'", key))
    }
}

// ---------------------------------------------------------------------------
// Key types
// ---------------------------------------------------------------------------

new_key_type! {
    /// Stable handle to a node in a [`MolGraph`]: an atom of an
    /// [`Atomistic`](crate::core::Atomistic) and a bead of a
    /// [`CoarseGrain`](crate::core::CoarseGrain) alike. It is
    /// the one name for that handle; there are no per-graph aliases.
    pub struct NodeId;
    /// Stable handle to a relation (any kind) in a [`MolGraph`]: a bond,
    /// angle, dihedral, improper or port. The relation's kind says which; it
    /// is the one name for that handle, as [`Relation`] is the one name for
    /// the relation itself.
    pub struct RelationId;
}

/// Convert a [`NodeId`] to/from a stable opaque `u64` handle (the generational
/// slotmap key's FFI form). For exposing stable entity handles across an FFI /
/// language boundary without leaking the `slotmap` types.
pub fn node_to_u64(id: NodeId) -> u64 {
    id.data().as_ffi()
}
/// See [`node_to_u64`].
pub fn node_from_u64(h: u64) -> NodeId {
    NodeId::from(KeyData::from_ffi(h))
}
/// Convert a [`RelationId`] to a stable opaque `u64` handle. See [`node_to_u64`].
pub fn relation_to_u64(id: RelationId) -> u64 {
    id.data().as_ffi()
}
/// See [`relation_to_u64`].
pub fn relation_from_u64(h: u64) -> RelationId {
    RelationId::from(KeyData::from_ffi(h))
}

/// Dense index identifying a registered relation kind. Resolved once at
/// registration; all hot-path relation access goes through this array index,
/// never a per-access string hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KindId(pub u16);

// ---------------------------------------------------------------------------
// Relation
// ---------------------------------------------------------------------------

/// A kind-tagged, fixed-arity relation over [`MolGraph`] nodes.
///
/// `nodes.len()` equals the registered arity of the relation's kind. Endpoints
/// are stored inline for arity ≤ 4 (the common case).
#[derive(Debug, Clone)]
pub struct Relation {
    /// The participating node handles, in order (length == kind arity).
    pub nodes: SmallVec<[NodeId; 4]>,
    /// Per-relation property bag (domain meaning, e.g. a bond `"order"`).
    pub props: IndexMap<String, PropValue>,
}

/// Storage for one relation kind: properties live in the aligned column table
/// (the same [`EntityTable`] machinery as nodes — relations are entities with
/// components), while the fixed-arity endpoints are kept structurally in a
/// [`SecondaryMap`] keyed by the same [`RelationId`] the property table mints.
#[derive(Debug, Clone)]
struct RelationKind {
    /// Property columns, keyed by `RelationId` (the kind's relation arena).
    props: EntityTable<RelationId>,
    /// Endpoint node handles per relation (length == kind arity).
    endpoints: SecondaryMap<RelationId, SmallVec<[NodeId; 4]>>,
}

impl RelationKind {
    fn new() -> Self {
        Self {
            props: EntityTable::new(),
            endpoints: SecondaryMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// MolGraph
// ---------------------------------------------------------------------------

/// A dynamic, domain-agnostic graph: nodes plus kind-tagged, fixed-arity
/// relations. Knows nothing of "atoms" or "bonds" — those live in leaf types.
///
/// `MolGraph` is a **pure graph** with no chemistry vocabulary. A *kind* is
/// registered once by an arbitrary name + arity (`register_kind("bond", 2)`) and
/// addressed thereafter by a dense [`KindId`] (an array index — never a
/// per-access string hash). Relations of the same arity but different meaning
/// (e.g. a 4-ary "dihedral" vs a 4-ary "improper") are distinguished by their
/// [`KindId`], not by arity. The domain concepts live in the leaf types
/// ([`Atomistic`](crate::core::Atomistic) /
/// [`CoarseGrain`](crate::core::CoarseGrain)) that register their kinds
/// and expose the named convenience API.
///
/// Storage uses generational arenas ([`slotmap::SlotMap`]) for O(1) insert /
/// remove / lookup with stable handles, and a [`SmallVec`] for each relation's
/// endpoints so the common arities (≤4) stay inline / heap-allocation-free.
///
/// Every node is a property bag ([`Atom`]): coordinates live as `"x"`, `"y"`,
/// `"z"` keys, matching the Python `Entity(UserDict)` convention.
///
/// # Relations vs. containment
///
/// The `kinds` / relation store is for **fixed-arity peer topology only**.
/// Hierarchical *containment* (a residue owning atoms, a chain owning residues,
/// a coarse-grained bead owning its atoms) is variable-size, nested, directed
/// ownership — **not** a fixed-arity peer relation — and is therefore **not**
/// modeled as a relation kind (doing so would put group handles into the node
/// arena and contaminate every consumer that iterates [`MolGraph::nodes`]). When
/// containment lands it is a separate axis beside nodes and relations.
///
/// # Examples
///
/// ```
/// use molrs::core::{Atom, MolGraph};
///
/// let mut g = MolGraph::new();
/// let bond = g.register_kind("bond", 2);
///
/// let o = g.add_node_with(Atom::xyz("O", 0.0, 0.0, 0.0)).expect("add O");
/// let h1 = g.add_node_with(Atom::xyz("H", 0.96, 0.0, 0.0)).expect("add H");
/// g.add_relation(bond, &[o, h1]).expect("add bond");
///
/// assert_eq!(g.n_nodes(), 2);
/// assert_eq!(g.n_relations(bond), 1);
///
/// g.translate([1.0, 0.0, 0.0]);
/// assert!((g.get_node(o).expect("get node").get_f64("x").unwrap() - 1.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct MolGraph {
    /// Node entities + their components, stored as an aligned column table.
    nodes: EntityTable<NodeId>,
    /// Relation kinds (props columns + endpoints), indexed by `KindId.0`.
    kinds: Vec<RelationKind>,
    /// Arity of each kind, indexed by `KindId.0`.
    kind_arity: Vec<usize>,
    /// Registered name of each kind, indexed by `KindId.0` (used as the
    /// [`Frame`] block name in `to_frame` / `extend_from_frame`).
    kind_name: Vec<String>,
    /// Reverse lookup: name → KindId (resolved once, not on the hot path).
    name_to_kind: HashMap<String, KindId>,
    /// Adjacency over arity-2 relations: node → list of `(kind, relation)`.
    adjacency: HashMap<NodeId, Vec<(KindId, RelationId)>>,
}

impl Default for MolGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl MolGraph {
    /// Create an empty graph with **no** kinds registered.
    pub fn new() -> Self {
        Self {
            nodes: EntityTable::new(),
            kinds: Vec::new(),
            kind_arity: Vec::new(),
            kind_name: Vec::new(),
            name_to_kind: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }

    // =====================================================================
    // Kind registry (generic, domain-neutral)
    // =====================================================================

    /// Register a relation kind by name + fixed arity, returning its dense
    /// [`KindId`]. Idempotent: re-registering the same name with the same arity
    /// returns the existing id. Re-registering with a conflicting arity panics
    /// (a programming error — leaf constructors register fixed kinds).
    pub fn register_kind(&mut self, name: &str, arity: usize) -> KindId {
        if let Some(&kid) = self.name_to_kind.get(name) {
            assert_eq!(
                self.kind_arity[kid.0 as usize], arity,
                "kind '{name}' already registered with a different arity"
            );
            return kid;
        }
        let kid = KindId(self.kinds.len() as u16);
        self.kinds.push(RelationKind::new());
        self.kind_arity.push(arity);
        self.kind_name.push(name.to_owned());
        self.name_to_kind.insert(name.to_owned(), kid);
        kid
    }

    /// Register a relation kind the way a `Result`-returning constructor must:
    /// idempotent on a matching name + arity, an error — never a panic — on a
    /// conflicting one.
    ///
    /// [`register_kind`](Self::register_kind) asserts on an arity conflict
    /// because a leaf constructor registers fixed kinds, which is a programming
    /// error. A promotion such as
    /// [`Atomistic::try_from_molgraph`](crate::core::Atomistic::try_from_molgraph)
    /// registers its standard kinds onto a **caller-supplied** graph, where a
    /// conflicting arity is a data condition and must come back as an `Err`.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when `name` is already registered at
    /// an arity other than `arity`, naming the kind and both arities.
    pub(crate) fn try_register_kind(
        &mut self,
        name: &str,
        arity: usize,
    ) -> Result<KindId, MolRsError> {
        if let Some(kid) = self.kind_id(name) {
            let found = self.arity(kid);
            if found != arity {
                return Err(MolRsError::validation(format!(
                    "kind '{name}' is registered with arity {found}, but {arity} is required"
                )));
            }
            return Ok(kid);
        }
        Ok(self.register_kind(name, arity))
    }

    /// Resolve a kind name to its registered [`KindId`], if any.
    pub fn kind_id(&self, name: &str) -> Option<KindId> {
        self.name_to_kind.get(name).copied()
    }

    /// Registered name of a kind.
    pub fn kind_name(&self, kind: KindId) -> &str {
        &self.kind_name[kind.0 as usize]
    }

    /// Arity of a registered kind.
    pub fn arity(&self, kind: KindId) -> usize {
        self.kind_arity[kind.0 as usize]
    }

    /// Iterate over all registered [`KindId`]s in registration order.
    pub fn kind_ids(&self) -> impl Iterator<Item = KindId> + '_ {
        (0..self.kinds.len() as u16).map(KindId)
    }

    // =====================================================================
    // Node CRUD (generic)
    // =====================================================================

    /// Insert a field-less node, returning its stable handle. The node has no
    /// components set (no `element` / `bead_type` is forced).
    pub fn add_node(&mut self) -> NodeId {
        let id = self.nodes.spawn();
        self.adjacency.insert(id, Vec::new());
        id
    }

    /// Insert a node carrying a property bag, returning its stable handle.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a value of `payload` contradicts the
    /// element type an existing node column holds for that key (a string
    /// `tag` into an `i32` `tag` column); the message names the conflicting
    /// key. Likewise when a value cannot be stored at the dtype the Frame
    /// schema declares for a canonical key (a string under `"x"`), which is
    /// refused here rather than on the way out to a `Frame`. The bag is
    /// not always the caller's own — [`merge`](Self::merge)
    /// and the leaves' `from_frame` hand over property bags read out of
    /// another graph or a file, where a disagreeing key is data and not a
    /// defect — so the conflict is returned rather than asserted.
    ///
    /// The node is **not** rolled back: it keeps the prefix of `payload` that
    /// was written before the conflict, as `self` does for any other partial
    /// write (see [`merge`](Self::merge)).
    pub fn add_node_with(&mut self, payload: Atom) -> Result<NodeId, MolRsError> {
        let id = self.add_node();
        self.set_node_props(id, &payload)?;
        Ok(id)
    }

    /// Remove a node and every relation that references it, across **all**
    /// registered kinds (registry-driven cascade). Returns the node's property
    /// bag (materialized).
    pub fn remove_node(&mut self, id: NodeId) -> Result<Atom, MolRsError> {
        Ok(self
            .remove_nodes(&[id])?
            .pop()
            .expect("one validated node produces one payload"))
    }

    /// Remove several nodes in one registry scan.
    ///
    /// This is the batch primitive reaction compilation needs: deleting one
    /// leaving group at a time scans every relation table once per edit, while
    /// deleting the compiled set scans each table exactly once.
    pub fn remove_nodes(&mut self, ids: &[NodeId]) -> Result<Vec<Atom>, MolRsError> {
        let doomed_nodes: HashSet<NodeId> = ids.iter().copied().collect();
        if doomed_nodes.len() != ids.len() {
            return Err(MolRsError::validation(
                "remove_nodes received a duplicate node handle",
            ));
        }
        for &id in ids {
            if !self.nodes.contains(id) {
                return Err(MolRsError::not_found("node", format!("NodeId {:?}", id)));
            }
        }
        let payloads = ids.iter().map(|&id| self.materialize_atom(id)).collect();

        for kid in 0..self.kinds.len() {
            let doomed: Vec<RelationId> = self.kinds[kid]
                .endpoints
                .iter()
                .filter(|(_, eps)| eps.iter().any(|id| doomed_nodes.contains(id)))
                .map(|(rid, _)| rid)
                .collect();
            for rid in doomed {
                self.detach_relation_from_adjacency(KindId(kid as u16), rid, None);
                let k = &mut self.kinds[kid];
                k.props.despawn(rid);
                k.endpoints.remove(rid);
            }
        }
        for &id in ids {
            self.adjacency.remove(&id);
            self.nodes.despawn(id);
        }
        Ok(payloads)
    }

    /// Materialize a node's property bag (owned copy of its set components).
    pub fn get_node(&self, id: NodeId) -> Result<Atom, MolRsError> {
        if !self.nodes.contains(id) {
            return Err(MolRsError::not_found(
                "node",
                format!("NodeId {}", id.data().as_ffi()),
            ));
        }
        Ok(self.materialize_atom(id))
    }

    /// Set a single component on a node.
    pub fn set_node(
        &mut self,
        id: NodeId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        match coerce_canonical(key, val.into())? {
            PropValue::F64(v) => self.nodes.set_f64(id, key, v),
            PropValue::Int(v) => self.nodes.set_i32(id, key, v),
            PropValue::Str(s) => self.nodes.set_str(id, key, &s),
            PropValue::Bool(v) => self.nodes.set_bool(id, key, v),
        }
    }

    /// Clear a single component on a node (no-op if absent).
    pub fn clear_node(&mut self, id: NodeId, key: &str) -> Result<(), MolRsError> {
        self.nodes.clear(id, key)
    }

    /// Iterate over all `(NodeId, Atom)` pairs (each property bag materialized).
    pub fn nodes(&self) -> impl Iterator<Item = (NodeId, Atom)> + '_ {
        self.nodes
            .handles()
            .map(move |id| (id, self.materialize_atom(id)))
    }

    /// Live node handles in row order.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.handles()
    }

    /// Borrow the underlying node column table (for zero-copy column access).
    pub fn node_table(&self) -> &EntityTable<NodeId> {
        &self.nodes
    }

    /// Mutable access to the underlying node column table.
    ///
    /// Crate-internal: a raw column write bypasses the canonical coercion of
    /// [`set_node`](Self::set_node) and the adjacency index, so only code that
    /// maintains both itself may reach it.
    pub(crate) fn node_table_mut(&mut self) -> &mut EntityTable<NodeId> {
        &mut self.nodes
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Write an [`Atom`]'s properties into node `id`'s columns.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a value's element type contradicts the
    /// component the table already holds for that key, or when the value does
    /// not fit the dtype the Frame schema declares for a canonical key. The
    /// property is *not* written in that case, which is precisely why the
    /// error is returned rather than swallowed: a dropped property is a
    /// molecule that quietly lost a label.
    fn set_node_props(&mut self, id: NodeId, atom: &Atom) -> Result<(), MolRsError> {
        for (key, val) in atom.iter() {
            coerce_canonical(key, val.clone()).and_then(|pv| match pv {
                PropValue::F64(v) => self.nodes.set_f64(id, key, v),
                PropValue::Int(v) => self.nodes.set_i32(id, key, v),
                PropValue::Str(s) => self.nodes.set_str(id, key, &s),
                PropValue::Bool(v) => self.nodes.set_bool(id, key, v),
            })?;
        }
        Ok(())
    }

    /// Materialize node `id`'s set components into an [`Atom`].
    pub(crate) fn materialize_atom(&self, id: NodeId) -> Atom {
        let mut atom = Atom::new();
        for (key, cell) in self.nodes.row_cells(id) {
            match cell {
                EntityCell::F64(v) => atom.set(key, v),
                EntityCell::I32(v) => atom.set(key, v),
                EntityCell::Str(s) => atom.set(key, s),
                EntityCell::Bool(v) => atom.set(key, v),
            }
        }
        atom
    }

    /// Iterate over neighbor node IDs of a given node (via arity-2 relations).
    pub fn neighbors(&self, id: NodeId) -> impl Iterator<Item = NodeId> + '_ {
        self.neighbor_relations(id).map(|(_, _, other)| other)
    }

    /// Iterate over `(kind, relation, other_node)` for each arity-2 relation
    /// incident to a node. Domain leaves build typed neighbor queries on this.
    pub fn neighbor_relations(
        &self,
        id: NodeId,
    ) -> impl Iterator<Item = (KindId, RelationId, NodeId)> + '_ {
        self.adjacency
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(move |&(kind, rid)| {
                let eps = self.kinds[kind.0 as usize].endpoints.get(rid)?;
                let other = if eps[0] == id { eps[1] } else { eps[0] };
                Some((kind, rid, other))
            })
    }

    // =====================================================================
    // Relation CRUD (generic, kind-tagged)
    // =====================================================================

    /// Add a relation of the given kind over the given nodes. Validates the
    /// kind is registered, the node count matches its arity, and every node
    /// exists. Maintains the adjacency index for arity-2 relations.
    pub fn add_relation(
        &mut self,
        kind: KindId,
        nodes: &[NodeId],
    ) -> Result<RelationId, MolRsError> {
        let kidx = kind.0 as usize;
        if kidx >= self.kinds.len() {
            return Err(MolRsError::not_found("kind", format!("KindId {:?}", kind)));
        }
        let arity = self.kind_arity[kidx];
        if nodes.len() != arity {
            return Err(MolRsError::validation(format!(
                "kind '{}' expects arity {}, got {} nodes",
                self.kind_name[kidx],
                arity,
                nodes.len()
            )));
        }
        for &n in nodes {
            if !self.nodes.contains(n) {
                return Err(MolRsError::not_found("node", format!("NodeId {:?}", n)));
            }
        }
        let k = &mut self.kinds[kidx];
        let rid = k.props.spawn();
        k.endpoints.insert(rid, SmallVec::from_slice(nodes));
        if arity == 2 {
            self.adjacency
                .entry(nodes[0])
                .or_default()
                .push((kind, rid));
            self.adjacency
                .entry(nodes[1])
                .or_default()
                .push((kind, rid));
        }
        Ok(rid)
    }

    /// Materialize a relation (endpoints + properties) by kind + handle.
    pub fn get_relation(&self, kind: KindId, id: RelationId) -> Result<Relation, MolRsError> {
        let k = self
            .kinds
            .get(kind.0 as usize)
            .ok_or_else(|| MolRsError::not_found("kind", format!("KindId {:?}", kind)))?;
        if !k.props.contains(id) {
            return Err(MolRsError::not_found(
                "relation",
                format!("RelationId {}", id.data().as_ffi()),
            ));
        }
        Ok(self.materialize_relation(kind, id))
    }

    /// Endpoint node handles of a relation.
    pub fn relation_nodes(
        &self,
        kind: KindId,
        id: RelationId,
    ) -> Result<SmallVec<[NodeId; 4]>, MolRsError> {
        self.kinds
            .get(kind.0 as usize)
            .and_then(|k| k.endpoints.get(id).cloned())
            .ok_or_else(|| MolRsError::not_found("relation", format!("RelationId {:?}", id)))
    }

    /// Set a single property on a relation.
    pub fn set_relation_prop(
        &mut self,
        kind: KindId,
        id: RelationId,
        key: &str,
        val: impl Into<PropValue>,
    ) -> Result<(), MolRsError> {
        let k = self
            .kinds
            .get_mut(kind.0 as usize)
            .ok_or_else(|| MolRsError::not_found("kind", format!("KindId {:?}", kind)))?;
        if !k.props.contains(id) {
            return Err(MolRsError::not_found(
                "relation",
                format!("RelationId {:?}", id),
            ));
        }
        match coerce_canonical(key, val.into())? {
            PropValue::F64(v) => k.props.set_f64(id, key, v),
            PropValue::Int(v) => k.props.set_i32(id, key, v),
            PropValue::Str(s) => k.props.set_str(id, key, &s),
            PropValue::Bool(v) => k.props.set_bool(id, key, v),
        }
    }

    /// Clear a single property on a relation (no-op if absent).
    pub fn clear_relation_prop(
        &mut self,
        kind: KindId,
        id: RelationId,
        key: &str,
    ) -> Result<(), MolRsError> {
        let k = self
            .kinds
            .get_mut(kind.0 as usize)
            .ok_or_else(|| MolRsError::not_found("kind", format!("KindId {:?}", kind)))?;
        k.props.clear(id, key)
    }

    /// Remove a relation by kind + handle, updating adjacency. Returns the
    /// materialized relation.
    pub fn remove_relation(
        &mut self,
        kind: KindId,
        id: RelationId,
    ) -> Result<Relation, MolRsError> {
        let exists = self
            .kinds
            .get(kind.0 as usize)
            .is_some_and(|k| k.props.contains(id));
        if !exists {
            return Err(MolRsError::not_found(
                "relation",
                format!("RelationId {:?}", id),
            ));
        }
        let rel = self.materialize_relation(kind, id);
        self.detach_relation_from_adjacency(kind, id, None);
        let k = &mut self.kinds[kind.0 as usize];
        k.props.despawn(id);
        k.endpoints.remove(id);
        Ok(rel)
    }

    /// Iterate over `(RelationId, Relation)` for one kind (each materialized).
    pub fn relations(&self, kind: KindId) -> impl Iterator<Item = (RelationId, Relation)> + '_ {
        let ids: Vec<RelationId> = self.kinds[kind.0 as usize].props.handles().collect();
        ids.into_iter()
            .map(move |rid| (rid, self.materialize_relation(kind, rid)))
    }

    /// Relation handles of a kind, in row order.
    pub fn relation_ids(&self, kind: KindId) -> impl Iterator<Item = RelationId> + '_ {
        self.kinds[kind.0 as usize].props.handles()
    }

    /// The row of relation `id` in `kind`'s table — its position in
    /// [`relation_ids`](Self::relation_ids) — or `None` if it is not live.
    /// O(1).
    pub fn relation_row(&self, kind: KindId, id: RelationId) -> Option<usize> {
        self.kinds[kind.0 as usize].props.row(id)
    }

    /// Number of relations of a kind.
    pub fn n_relations(&self, kind: KindId) -> usize {
        self.kinds[kind.0 as usize].props.len()
    }

    /// Materialize a relation's endpoints + properties.
    pub(crate) fn materialize_relation(&self, kind: KindId, id: RelationId) -> Relation {
        let k = &self.kinds[kind.0 as usize];
        let nodes = k.endpoints.get(id).cloned().unwrap_or_default();
        let mut props = IndexMap::new();
        for (key, cell) in k.props.row_cells(id) {
            let pv = match cell {
                EntityCell::F64(v) => PropValue::F64(v),
                EntityCell::I32(v) => PropValue::Int(v),
                EntityCell::Str(s) => PropValue::Str(s.to_owned()),
                EntityCell::Bool(v) => PropValue::Bool(v),
            };
            props.insert(key.to_owned(), pv);
        }
        Relation { nodes, props }
    }

    /// Write a property bag into a relation's columns.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a value's element type contradicts the
    /// component the kind already holds for that key, or when the value does
    /// not fit the dtype the Frame schema declares for a canonical key. As on
    /// the node side, the property is not written, so the error is the only
    /// thing standing between the caller and a silently dropped bond label.
    pub(crate) fn set_relation_props(
        &mut self,
        kind: KindId,
        id: RelationId,
        props: &IndexMap<String, PropValue>,
    ) -> Result<(), MolRsError> {
        let k = &mut self.kinds[kind.0 as usize];
        for (key, val) in props {
            coerce_canonical(key, val.clone()).and_then(|pv| match pv {
                PropValue::F64(v) => k.props.set_f64(id, key, v),
                PropValue::Int(v) => k.props.set_i32(id, key, v),
                PropValue::Str(s) => k.props.set_str(id, key, &s),
                PropValue::Bool(v) => k.props.set_bool(id, key, v),
            })?;
        }
        Ok(())
    }

    /// Remove an arity-2 relation from the adjacency lists of its endpoints.
    /// `skip` lets `remove_node` avoid touching the node currently being dropped.
    fn detach_relation_from_adjacency(
        &mut self,
        kind: KindId,
        id: RelationId,
        skip: Option<NodeId>,
    ) {
        if self.kind_arity[kind.0 as usize] != 2 {
            return;
        }
        let endpoints: Option<[NodeId; 2]> = self.kinds[kind.0 as usize]
            .endpoints
            .get(id)
            .map(|eps| [eps[0], eps[1]]);
        if let Some(eps) = endpoints {
            for ep in eps {
                if Some(ep) == skip {
                    continue;
                }
                if let Some(adj) = self.adjacency.get_mut(&ep) {
                    // Relation handles are per-kind slotmap keys, so a handle
                    // of another kind can be equal: match the kind too.
                    adj.retain(|&(k, rid)| !(k == kind && rid == id));
                }
            }
        }
    }

    // Spatial transforms are free-function *systems* — see [`crate::core::geometry`]
    // (`translate`, `rotate`). The data structure carries no geometry methods.

    // =====================================================================
    // Composition
    // =====================================================================

    /// Merge another `MolGraph` into `self`, consuming `other`.
    ///
    /// Registry-driven: every relation of every kind in `other` is transferred
    /// (kinds matched by name, registered on `self` if missing) — so all kinds
    /// are carried across.
    ///
    /// **Handle contract:** every node of `other` is remapped to a fresh handle in
    /// `self`. Returns the map `NodeId in other → NodeId in self`. (By contrast,
    /// [`Clone`] **preserves** handles in the independent copy.)
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a property of `other` cannot be written
    /// into `self` — an incoming string `tag` where `self` holds an `i32`
    /// `tag`, say. `other`'s properties are foreign data to `self`, so the two
    /// vocabularies can genuinely disagree; the merged graph would otherwise
    /// come back missing exactly the properties the caller handed over. The
    /// merge is *not* rolled back: `self` keeps what was written before the
    /// conflict, as it does for any other partial write.
    pub fn merge(&mut self, other: MolGraph) -> Result<HashMap<NodeId, NodeId>, MolRsError> {
        let mut node_map: HashMap<NodeId, NodeId> = HashMap::new();
        for old_id in other.nodes.handles() {
            let new_id = self.add_node_with(other.materialize_atom(old_id))?;
            node_map.insert(old_id, new_id);
        }

        for okid in other.kind_ids() {
            let oidx = okid.0 as usize;
            let name = &other.kind_name[oidx];
            let arity = other.kind_arity[oidx];
            let self_kind = self.register_kind(name, arity);
            let orids: Vec<RelationId> = other.relation_ids(okid).collect();
            for orid in orids {
                let rel = other.materialize_relation(okid, orid);
                let mapped: SmallVec<[NodeId; 4]> = rel.nodes.iter().map(|n| node_map[n]).collect();
                if let Ok(rid) = self.add_relation(self_kind, &mapped) {
                    self.set_relation_props(self_kind, rid, &rel.props)?;
                }
            }
        }
        Ok(node_map)
    }

    /// Place `transforms.len()` rigid copies of `template` into `self`, copy
    /// `c` moved by `transforms[c]` and stamped `frag_id = frag_ids[c]`.
    ///
    /// Returns the new node handles copy-major: template node `t` of copy `c`
    /// (template nodes in [`node_ids`](Self::node_ids) order) is at index
    /// `c * n + t`, `n` being the template's node count.
    ///
    /// - **Column-wise.** Nodes and each relation kind's properties are
    ///   appended with `EntityTable::extend_repeated`, one pass per column;
    ///   no per-node [`add_node_with`](Self::add_node_with) runs.
    /// - **Coordinates.** `x`/`y`/`z` of a template row holding the full
    ///   triple are rewritten per copy by [`crate::op::transform_points`]. A
    ///   row without the full triple is copied untransformed.
    /// - **Relations.** Every relation kind of `template` is registered on
    ///   `self` by name, its endpoints offset into each copy and its props
    ///   copied. Ports are a relation kind, so they are kept.
    /// - **`frag_id`.** The `i32` node column `frag_id` holds `frag_ids[c]` on
    ///   every node of copy `c`, overwriting any template value.
    ///
    /// Unlike [`merge`](Self::merge), this is **atomic**: every check runs
    /// before the first write, so on an error `self` is unchanged — no node,
    /// relation, column or kind is added.
    ///
    /// No transforms place no copy: `Ok` of no handles with `self` untouched —
    /// no kind registered, no column added — and no check beyond the length
    /// one runs, since nothing would be written.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when
    /// - `transforms.len() != frag_ids.len()`;
    /// - a node column of `template` has an element type other than the one
    ///   `self` holds for that key;
    /// - `frag_id` is held, by `self` or `template`, as a non-`i32` column;
    /// - a relation kind of `template` is registered on `self` at another
    ///   arity;
    /// - a relation property column of `template` contradicts the element
    ///   type `self` holds for that key in the same kind.
    pub fn replicate(
        &mut self,
        template: &MolGraph,
        transforms: &[crate::op::Rigid],
        frag_ids: &[I],
    ) -> Result<Vec<NodeId>, MolRsError> {
        // ---- checks: nothing is written until all pass ----
        if transforms.len() != frag_ids.len() {
            return Err(MolRsError::validation(format!(
                "replicate needs one frag_id per transform; got {} transforms and {} frag_ids",
                transforms.len(),
                frag_ids.len()
            )));
        }
        if transforms.is_empty() {
            return Ok(Vec::new());
        }
        self.nodes.check_extend(&template.nodes)?;
        for (owner, table) in [("self", &self.nodes), ("template", &template.nodes)] {
            if let Some(col) = table.column(FRAG_ID)
                && !matches!(col, crate::core::EntityColumn::I32(..))
            {
                return Err(MolRsError::validation(format!(
                    "'{FRAG_ID}' of {owner} is typed {}; replicate stamps an i32 {FRAG_ID}",
                    col.type_name()
                )));
            }
        }
        for tkid in template.kind_ids() {
            let tidx = tkid.0 as usize;
            let name = &template.kind_name[tidx];
            let arity = template.kind_arity[tidx];
            if let Some(skid) = self.kind_id(name) {
                let found = self.arity(skid);
                if found != arity {
                    return Err(MolRsError::validation(format!(
                        "kind '{name}' is registered with arity {found}, but the template's is {arity}"
                    )));
                }
                self.kinds[skid.0 as usize]
                    .props
                    .check_extend(&template.kinds[tidx].props)?;
            }
        }

        // ---- nodes: one column-wise append ----
        let copies = transforms.len();
        let n = template.n_nodes();
        let row0 = self.nodes.len();
        let handles = self.nodes.extend_repeated(&template.nodes, copies)?;
        for &id in &handles {
            self.adjacency.insert(id, Vec::new());
        }

        // ---- coordinates: rows with the full triple, per copy ----
        let (placed_rows, points) = template.full_triple_rows();
        if !placed_rows.is_empty() {
            let images: Vec<Vec<[F; 3]>> = transforms
                .iter()
                .map(|rigid| crate::op::transform_points(rigid, &points))
                .collect();
            for (axis, key) in [keys::X, keys::Y, keys::Z].into_iter().enumerate() {
                let (col, _) = self
                    .nodes
                    .column_f64_mut(key)
                    .expect("a template row holds the full triple, so x/y/z were appended as f64");
                for (c, image) in images.iter().enumerate() {
                    let base = row0 + c * n;
                    for (&t, p) in placed_rows.iter().zip(image) {
                        col[base + t] = p[axis];
                    }
                }
            }
        }

        // ---- frag_id: one fill per copy ----
        for (c, &fid) in frag_ids.iter().enumerate() {
            let start = row0 + c * n;
            self.nodes
                .fill_i32(FRAG_ID, start..start + n, fid)
                .expect("checked before the first write");
        }

        // ---- relations: kind by name, endpoints offset per copy ----
        for tkid in template.kind_ids() {
            let tidx = tkid.0 as usize;
            let skid = self.register_kind(&template.kind_name[tidx], template.kind_arity[tidx]);
            let tkind = &template.kinds[tidx];
            let arity = template.kind_arity[tidx];
            let rel_rows: Vec<SmallVec<[usize; 4]>> = tkind
                .props
                .handles()
                .map(|rid| {
                    tkind.endpoints[rid]
                        .iter()
                        .map(|&node| {
                            template
                                .nodes
                                .row(node)
                                .expect("a template relation names a live template node")
                        })
                        .collect()
                })
                .collect();
            let rids = self.kinds[skid.0 as usize]
                .props
                .extend_repeated(&tkind.props, copies)
                .expect("checked before the first write");
            let m = rel_rows.len();
            for (i, rid) in rids.into_iter().enumerate() {
                let base = (i / m) * n;
                let eps: SmallVec<[NodeId; 4]> =
                    rel_rows[i % m].iter().map(|&t| handles[base + t]).collect();
                if arity == 2 {
                    for &ep in &eps[..2] {
                        self.adjacency.entry(ep).or_default().push((skid, rid));
                    }
                }
                self.kinds[skid.0 as usize].endpoints.insert(rid, eps);
            }
        }
        Ok(handles)
    }

    /// Rows holding all of `x`, `y` and `z`, with their coordinates, in row
    /// order.
    fn full_triple_rows(&self) -> (Vec<usize>, Vec<[F; 3]>) {
        let (Ok((xs, xv)), Ok((ys, yv)), Ok((zs, zv))) = (
            self.nodes.column_f64(keys::X),
            self.nodes.column_f64(keys::Y),
            self.nodes.column_f64(keys::Z),
        ) else {
            return (Vec::new(), Vec::new());
        };
        (0..xs.len())
            .filter(|&r| xv.get(r) && yv.get(r) && zv.get(r))
            .map(|r| (r, [xs[r], ys[r], zs[r]]))
            .unzip()
    }

    // =====================================================================
    // Frame conversion (shared mechanism)
    //
    // The PUBLIC `to_frame` / `from_frame` API is provided by the **leaf**
    // types ([`Atomistic`](crate::core::Atomistic) /
    // [`CoarseGrain`](crate::core::CoarseGrain)), since converting to/from
    // the central [`Frame`] is a domain operation with leaf-specific block/kind
    // requirements. These `pub(crate)` methods are only the shared,
    // registry-driven implementation the leaves call — not data-struct API.
    // =====================================================================

    /// Shared implementation of leaf `to_frame`. Each node-component becomes a
    /// column in the `"atoms"` block; every non-empty relation kind becomes a
    /// block (named by the kind) with `atomi`/`atomj`/… columns referencing node
    /// row order plus one column per relation property — registry-driven.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a column is refused by the [`Block`] it
    /// is written into; the message names the column. A value whose element
    /// type the Frame schema does not declare for its key is refused at the
    /// write ([`set_node`](Self::set_node) refuses a string under `"x"`), so the property API cannot build such a column; the raw column
    /// table ([`node_table_mut`](Self::node_table_mut)) still can, because it
    /// writes an element type without consulting the key's declared dtype.
    /// That is a caller reaching past the door, not a broken invariant, so it
    /// is returned rather than asserted.
    pub(crate) fn to_frame(&self) -> Result<Frame, MolRsError> {
        let mut frame = Frame::new();

        let node_ids: Vec<NodeId> = self.nodes.handles().collect();
        let n = node_ids.len();
        let id_to_row: HashMap<NodeId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        // ---- atoms (node) block: one column per component (zero-copy reads;
        // columns are already dense and aligned to node row order), in the
        // order the components were first written ----
        let mut atoms_block = Block::new();
        for key in self.nodes.columns() {
            emit_column(&mut atoms_block, &self.nodes, key)?;
        }
        if n > 0 {
            frame.insert("atoms", atoms_block);
        }

        // ---- one block per non-empty relation kind ----
        for kid in self.kind_ids() {
            let kidx = kid.0 as usize;
            if self.kinds[kidx].props.is_empty() {
                continue;
            }
            let block = self.relation_block(kid, &id_to_row)?;
            frame.insert(&self.kind_name[kidx], block);
        }

        Ok(frame)
    }

    /// The [`Frame`] block of one relation kind: the `atomi`/`atomj`/… endpoint
    /// columns in position order, addressing nodes by the row order
    /// `id_to_row` fixes, plus one column per relation property.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] if a column is refused by the block: a
    /// column of the wrong row count, or a canonical key written at a dtype
    /// the Frame schema forbids. [`to_frame`](Self::to_frame) propagates it.
    fn relation_block(
        &self,
        kind: KindId,
        id_to_row: &HashMap<NodeId, usize>,
    ) -> Result<Block, MolRsError> {
        use ndarray::Array1;

        let kidx = kind.0 as usize;
        let k = &self.kinds[kidx];
        // Relation row order: shared by the endpoint columns and the (already
        // aligned, dense) property columns.
        let rids: Vec<RelationId> = k.props.handles().collect();
        let mut block = Block::new();

        for pos in 0..self.kind_arity[kidx] {
            let col: Vec<Idx> = rids
                .iter()
                .map(|rid| id_to_row[&k.endpoints[*rid][pos]] as Idx)
                .collect();
            block
                .insert(rel_col_name(pos), Array1::from_vec(col).into_dyn())
                .map_err(|e| MolRsError::validation(e.to_string()))?;
        }

        // Property columns read straight from the column table, in the order
        // they were first written.
        for key in k.props.columns() {
            emit_column(&mut block, &k.props, key)?;
        }
        Ok(block)
    }

    /// Read a [`Frame`] into `self`: the `"atoms"` block becomes nodes; each
    /// **already-registered** kind's block (matched by name) becomes relations
    /// via its `atomi`/`atomj`/… columns, with any extra columns read back as
    /// props.
    ///
    /// # A graph reads only its own relations
    ///
    /// A [`Frame`] is open and dynamic; a graph type is narrower and reads
    /// only the kinds it has registered (operator, 2026-09-28). A relation
    /// block whose name matches no registered kind is ignored, as are blocks
    /// that are not relation blocks (metadata, a box, a grid). A registered
    /// kind with no block in the frame is fine too. A block that *does* name
    /// a registered kind but carries fewer endpoint columns than that kind's
    /// arity — an `angles` block with `atomi` and `atomj` and no `atomk` —
    /// cannot be read and is refused.
    ///
    /// # Null cells
    ///
    /// A column's [validity mask](Block::validity) is honoured: a row the mask
    /// marks as null sets no property, so the unlabelled atoms of a partially
    /// labelled frame come back unlabelled rather than carrying the default
    /// the column had to store for them.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Parse`] when the frame has no `"atoms"` block, and
    /// [`MolRsError::Validation`] when a column of `atoms` or of a registered
    /// kind's block is not 1-D (a graph property is one value per row; the
    /// refusal names the block and the column and comes before any node or
    /// relation is added), when a column value does not fit its canonical
    /// type, when a column's dtype contradicts a component `self` already
    /// holds under that key, or when a registered kind's block is unreadable.
    pub(crate) fn extend_from_frame(&mut self, frame: &Frame) -> Result<(), MolRsError> {
        let atoms_block = frame
            .get("atoms")
            .ok_or_else(|| MolRsError::parse("Frame missing 'atoms' block"))?;

        // Ports are a capability of every graph (see `crate::core::port`),
        // so a `ports` block is read whatever the receiving type.
        if frame.get(crate::core::keys::PORTS).is_some() {
            self.try_register_kind(crate::core::keys::PORTS, 2)?;
        }
        let kind_specs: Vec<(KindId, String, usize)> = self
            .kind_ids()
            .map(|kid| {
                let i = kid.0 as usize;
                (kid, self.kind_name[i].clone(), self.kind_arity[i])
            })
            .collect();

        Self::require_1d_columns("atoms", atoms_block)?;
        for (_, block_name, _) in &kind_specs {
            if let Some(block) = frame.get(block_name) {
                Self::require_1d_columns(block_name, block)?;
            }
        }

        let node_ids = self.add_node_rows(atoms_block)?;

        for (kid, block_name, arity) in kind_specs {
            let Some(block) = frame.get(&block_name) else {
                continue;
            };
            self.add_relation_block(kid, &block_name, arity, block, &node_ids)?;
        }

        Ok(())
    }

    /// Refuse `block` if any of its columns is not 1-D.
    ///
    /// [`MaskedColumns`] reads one cell per row, which only a 1-D column has;
    /// an `(N, 3)` column would panic there. Checking every consumed block up
    /// front keeps the refusal ahead of the first write.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] naming the block, the first offending column
    /// and its rank.
    fn require_1d_columns(block_name: &str, block: &Block) -> Result<(), MolRsError> {
        for key in block.keys() {
            let ndim = block.get(key).map_or(1, |col| col.shape().len());
            if ndim != 1 {
                return Err(MolRsError::validation(format!(
                    "frame block '{block_name}' column '{key}' is {ndim}-D; a graph property \
                     column must be 1-D"
                )));
            }
        }
        Ok(())
    }

    /// Add one node per row of the `"atoms"` block, in row order, and return
    /// the handles at that order — the addressing every relation block's
    /// endpoint columns use.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a column value does not fit its
    /// canonical type, or when a property contradicts a component `self`
    /// already holds under that key.
    fn add_node_rows(&mut self, atoms: &Block) -> Result<Vec<NodeId>, MolRsError> {
        let nrows = atoms.n_rows().unwrap_or(0);
        let columns = MaskedColumns::of(atoms, &[]);
        let mut node_ids: Vec<NodeId> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let mut node = Atom::new();
            for (key, value) in columns.cells(row)? {
                node.set(key, value);
            }
            node_ids.push(self.add_node_with(node)?);
        }
        Ok(node_ids)
    }

    /// Read one relation block into the kind it names: each row becomes a
    /// relation over the nodes its endpoint columns address, carrying every
    /// other column of the block as a property.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when the block carries fewer endpoint columns
    /// than the kind's arity — the rows cannot be read at all, and reading on
    /// would hand back a graph missing relations the frame plainly stated —
    /// and when a property column's dtype contradicts the component the kind
    /// already holds under that key.
    fn add_relation_block(
        &mut self,
        kind: KindId,
        block_name: &str,
        arity: usize,
        block: &Block,
        node_ids: &[NodeId],
    ) -> Result<(), MolRsError> {
        let endpoint_names: Vec<String> = (0..arity).map(rel_col_name).collect();
        let mut endpoint_cols: Vec<&ndarray::ArrayD<Idx>> = Vec::with_capacity(arity);
        for name in &endpoint_names {
            let col = block.get(name).and_then(|c| c.as_uint()).ok_or_else(|| {
                MolRsError::validation(format!(
                    "Frame block '{block_name}' names the relation kind '{block_name}' of arity \
                     {arity} but carries no '{name}' endpoint column, so its rows cannot be read"
                ))
            })?;
            // A relation names all `arity` nodes; the value a null cell stores
            // is no atom, and reading it would tie the row to atom 0.
            if let Some(row) = block
                .validity(name)
                .and_then(|mask| mask.iter().position(|valid| !valid))
            {
                return Err(MolRsError::validation(format!(
                    "block '{block_name}' row {row} has a null '{name}' endpoint; a relation of \
                     the graph kind '{block_name}' names all {arity} of its atoms"
                )));
            }
            endpoint_cols.push(col);
        }

        // Non-endpoint columns are properties, read back at every dtype the
        // frame can carry (a string bond label, a UInt `type_id`, an `is_14`
        // flag), each honouring its validity mask.
        let props = MaskedColumns::of(block, &endpoint_names);

        for row in 0..block.n_rows().unwrap_or(0) {
            let Some(nodes) = endpoints_at(&endpoint_cols, row, node_ids) else {
                let stated: Vec<Idx> = endpoint_cols.iter().map(|col| col[[row]]).collect();
                return Err(MolRsError::validation(format!(
                    "block '{block_name}' row {row} names an atom index outside the atoms \
                     block: endpoints {stated:?}, {} atoms",
                    node_ids.len()
                )));
            };
            // Every rejection `add_relation` knows is unreachable here (the kind is
            // registered, the arity matches by construction, the handles were just
            // minted), so a failure is surfaced rather than dropped with the row.
            let rid = self.add_relation(kind, &nodes)?;
            for (key, value) in props.cells(row)? {
                self.set_relation_prop(kind, rid, key, value)?;
            }
        }
        Ok(())
    }
}

/// A graph type built from a finished [`MolGraph`]: the output factory of a
/// builder that assembles a bare graph and hands it back as whatever type the
/// caller asks for (operator, 2026-09-28: graph types are peers and no graph
/// type converts into another).
pub trait FromMolGraph: Sized {
    /// Wrap `graph` as `Self`, checking the type's invariant.
    ///
    /// # Errors
    ///
    /// The type's own refusal (an `Atomistic` needs an `element` on every
    /// node, for instance).
    fn from_molgraph(graph: MolGraph) -> Result<Self, MolRsError>;
}

impl FromMolGraph for MolGraph {
    fn from_molgraph(graph: MolGraph) -> Result<Self, MolRsError> {
        Ok(graph)
    }
}

/// Endpoint column name for the `pos`-th node of a relation block.
pub(crate) fn rel_col_name(pos: usize) -> String {
    match crate::core::keys::ENDPOINTS.get(pos) {
        Some(name) => (*name).to_owned(),
        None => format!("atom{pos}"),
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- PropValue & Atom dict-like API -----

    #[test]
    fn formal_charge_reads_int_and_integral_float_and_never_the_partial_charge() {
        let mut a = Atom::new();
        assert_eq!(a.formal_charge(), 0);
        a.set(keys::CHARGE, -0.8);
        assert_eq!(a.formal_charge(), 0);
        a.set(keys::FORMAL_CHARGE, -1.0);
        assert_eq!(a.formal_charge(), -1);
        a.set(keys::FORMAL_CHARGE, 2 as I);
        assert_eq!(a.formal_charge(), 2);
    }

    #[test]
    fn test_propvalue_from() {
        let v: PropValue = std::f64::consts::PI.into();
        assert_eq!(v, PropValue::F64(std::f64::consts::PI));
        let v: PropValue = (42 as I).into();
        assert_eq!(v, PropValue::Int(42 as I));
        let v: PropValue = "H".into();
        assert_eq!(v, PropValue::Str("H".to_owned()));
    }

    /// All three of x/y/z make a position; without z there is none.
    #[test]
    fn atom_position_reads_xyz_and_needs_all_three() {
        let mut a = Atom::xyz("C", 1.0, -2.0, 3.5);
        assert_eq!(a.position(), Some([1.0, -2.0, 3.5]));
        a.remove(keys::Z);
        assert_eq!(a.position(), None);
    }

    #[test]
    fn test_atom_dict_api() {
        let mut a = Atom::new();
        a.set("x", 1.5);
        a.set("element", "C");
        a.set("type_id", PropValue::Int(3 as I));
        assert_eq!(a.get_f64("x"), Some(1.5));
        assert_eq!(a.get_str("element"), Some("C"));
        assert_eq!(a.get_int("type_id"), Some(3 as I));
        assert_eq!(a.get_f64("missing"), None);
        assert!(a.contains_key("x"));
        assert_eq!(a.len(), 3);
        a.remove("type_id");
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn test_atom_index() {
        let mut a = Atom::xyz("O", 1.0, 2.0, 3.0);
        assert_eq!(a["x"], PropValue::F64(1.0));
        a["x"] = PropValue::F64(99.0);
        assert_eq!(a.get_f64("x"), Some(99.0));
    }

    // ----- Kind registry -----

    #[test]
    fn test_register_kind_dense_idempotent() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bond", 2);
        let angle = g.register_kind("angle", 3);
        assert_eq!(bond, KindId(0));
        assert_eq!(angle, KindId(1));
        // idempotent
        assert_eq!(g.register_kind("bond", 2), bond);
        assert_eq!(g.kind_id("angle"), Some(angle));
        assert_eq!(g.arity(bond), 2);
        assert_eq!(g.kind_name(angle), "angle");
    }

    #[test]
    #[should_panic]
    fn test_register_kind_conflicting_arity_panics() {
        let mut g = MolGraph::new();
        g.register_kind("bond", 2);
        g.register_kind("bond", 3);
    }

    // ----- Node CRUD -----

    #[test]
    fn test_add_node_field_less() {
        let mut g = MolGraph::new();
        let n = g.add_node();
        assert!(g.get_node(n).unwrap().is_empty());
        g.translate([1.0, 2.0, 3.0]);
        assert!(g.get_node(n).unwrap().get_f64("x").is_none());
        g.set_node(n, "element", "C").unwrap();
        assert_eq!(g.get_node(n).unwrap().get_str("element"), Some("C"));
    }

    #[test]
    fn test_add_remove_node() {
        let mut g = MolGraph::new();
        assert_eq!(g.n_nodes(), 0);
        let id = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        assert_eq!(g.n_nodes(), 1);
        assert_eq!(g.get_node(id).unwrap().get_str("element"), Some("C"));
        g.remove_node(id).unwrap();
        assert_eq!(g.n_nodes(), 0);
        assert!(g.get_node(id).is_err());
    }

    /// `add_node_with` builds a node out of a property bag the caller hands
    /// it, and a bag whose value contradicts an existing node column is a
    /// data condition the caller can act on — so it is returned, not
    /// asserted by a panic.
    ///
    /// The fixture key is deliberately one the Frame schema declares nothing
    /// about: under a canonical key the setter refuses the seed's element
    /// type before a column ever exists, which is a different rule. What is
    /// tested here is the *column* the graph already holds.
    #[test]
    fn add_node_with_returns_the_conflict_when_the_payload_contradicts_a_node_column() {
        let mut g = MolGraph::new();
        let seeded = g.add_node();
        g.set_node(seeded, "tag", -0.5_f64)
            .expect("a fresh 'tag' column takes an f64");

        let mut payload = Atom::new();
        payload.set("tag", "negative");

        let err = g
            .add_node_with(payload)
            .expect_err("a str 'tag' cannot enter an f64 'tag' column");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert!(
            err.to_string().contains("'tag'"),
            "the error names the conflicting key, got {err:?}"
        );
    }

    // ----- Relation CRUD -----

    #[test]
    fn test_generic_relation_crud() {
        let mut g = MolGraph::new();
        let angle = g.register_kind("angle", 3);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let rid = g.add_relation(angle, &[a, b, c]).unwrap();
        assert_eq!(g.n_relations(angle), 1);
        assert_eq!(
            g.get_relation(angle, rid).unwrap().nodes.as_slice(),
            &[a, b, c][..]
        );
        assert!(g.add_relation(angle, &[a, b]).is_err()); // wrong arity
        g.remove_node(c).unwrap();
        assert!(g.add_relation(angle, &[a, b, c]).is_err()); // missing node
        assert_eq!(g.n_relations(angle), 0); // cascaded
    }

    #[test]
    fn test_same_arity_distinct_kind() {
        let mut g = MolGraph::new();
        let dih = g.register_kind("dihedral", 4);
        let imp = g.register_kind("improper", 4);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let d = g.add_node();
        g.add_relation(dih, &[a, b, c, d]).unwrap();
        g.add_relation(imp, &[a, b, c, d]).unwrap();
        assert_eq!(g.n_relations(dih), 1);
        assert_eq!(g.n_relations(imp), 1);
    }

    #[test]
    fn test_cascade_across_all_kinds() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bond", 2);
        let angle = g.register_kind("angle", 3);
        let dih = g.register_kind("dihedral", 4);
        let imp = g.register_kind("improper", 4);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let d = g.add_node();
        g.add_relation(bond, &[a, b]).unwrap();
        g.add_relation(angle, &[b, a, c]).unwrap();
        g.add_relation(dih, &[b, a, c, d]).unwrap();
        g.add_relation(imp, &[b, a, c, d]).unwrap();
        g.remove_node(a).unwrap();
        assert_eq!(g.n_relations(bond), 0);
        assert_eq!(g.n_relations(angle), 0);
        assert_eq!(g.n_relations(dih), 0);
        assert_eq!(g.n_relations(imp), 0);
    }

    // ----- Neighbors -----

    #[test]
    fn test_neighbors() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bond", 2);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        g.add_relation(bond, &[a, b]).unwrap();
        g.add_relation(bond, &[a, c]).unwrap();
        let mut n: Vec<NodeId> = g.neighbors(a).collect();
        n.sort_by_key(|id| id.0);
        assert_eq!(n.len(), 2);
        assert!(n.contains(&b) && n.contains(&c));
        assert_eq!(g.neighbors(b).collect::<Vec<_>>(), vec![a]);
        // removing the relation clears adjacency
        let bid = g.relations(bond).next().unwrap().0;
        g.remove_relation(bond, bid).unwrap();
        assert_eq!(g.neighbors(a).count(), 1);
    }

    // ----- Spatial -----

    #[test]
    fn test_translate_and_rotate() {
        let mut g = MolGraph::new();
        let id = g
            .add_node_with(Atom::xyz("C", 1.0, 0.0, 0.0))
            .expect("fixture node");
        g.translate([10.0, 20.0, 30.0]);
        let a = g.get_node(id).unwrap();
        assert!((a.get_f64("x").unwrap() - 11.0).abs() < 1e-12);
        let id2 = g
            .add_node_with(Atom::xyz("C", 1.0, 0.0, 0.0))
            .expect("fixture node");
        g.rotate([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2, None)
            .expect("the z axis is a direction");
        let b = g.get_node(id2).unwrap();
        assert!((b.get_f64("x").unwrap()).abs() < 1e-12);
        assert!((b.get_f64("y").unwrap() - 1.0).abs() < 1e-12);
    }

    // ----- Frame round-trip (generic) -----

    #[test]
    fn test_to_frame_extend_from_frame_roundtrip() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let o = g
            .add_node_with(Atom::xyz("O", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let h1 = g
            .add_node_with(Atom::xyz("H", 0.96, 0.0, 0.0))
            .expect("fixture node");
        let h2 = g
            .add_node_with(Atom::xyz("H", -0.24, 0.93, 0.0))
            .expect("fixture node");
        g.add_relation(bond, &[o, h1]).unwrap();
        g.add_relation(bond, &[o, h2]).unwrap();
        let frame = g.to_frame().expect("a schema-conforming graph converts");
        assert!(frame.contains_key("atoms"));
        assert!(frame.contains_key("bonds"));
        assert_eq!(frame["atoms"].n_rows(), Some(3));
        assert_eq!(frame["bonds"].n_rows(), Some(2));

        // read back into a graph with the same kind registered
        let mut g2 = MolGraph::new();
        let bond2 = g2.register_kind("bonds", 2);
        g2.extend_from_frame(&frame).unwrap();
        assert_eq!(g2.n_nodes(), 3);
        assert_eq!(g2.n_relations(bond2), 2);
    }

    #[test]
    fn test_extend_from_frame_restores_int_and_str_relation_props() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("C", 1.5, 0.0, 0.0))
            .expect("fixture node");
        let rid = g.add_relation(bond, &[a, b]).unwrap();
        // A genuinely float-valued relation prop. `bond_number` is not one —
        // it is a schema uint — and a *computed* fractional bond order is
        // exactly the kind of quantity that gets its own key.
        g.set_relation_prop(bond, rid, "wiberg_index", 1.47_f64)
            .unwrap();
        g.set_relation_prop(bond, rid, "ring_size", PropValue::Int(6))
            .unwrap();
        g.set_relation_prop(bond, rid, "label", PropValue::Str("aromatic".to_owned()))
            .unwrap();

        let frame = g.to_frame().expect("a schema-conforming graph converts");

        let mut g2 = MolGraph::new();
        let bond2 = g2.register_kind("bonds", 2);
        g2.extend_from_frame(&frame).unwrap();

        let (_, r) = g2.relations(bond2).next().expect("relation round-trips");
        assert_eq!(
            r.props.get("ring_size"),
            Some(&PropValue::Int(6)),
            "int relation prop lost on round-trip"
        );
        assert_eq!(
            r.props.get("label"),
            Some(&PropValue::Str("aromatic".to_owned())),
            "string relation prop lost on round-trip"
        );
        match r.props.get("wiberg_index") {
            Some(PropValue::F64(v)) => assert!((v - 1.47).abs() < 1e-12),
            other => panic!("float relation prop lost: {other:?}"),
        }
    }

    #[test]
    fn test_to_frame_keeps_schema_unsigned_node_columns() {
        // `id` / `mol_id` are UInt in the Frame schema but live in the node
        // table as I. Emitting them signed made `Block::insert` reject the
        // column against the schema — and, with the error swallowed, the
        // identifiers vanished from the frame entirely.
        let mut g = MolGraph::new();
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("H", 1.1, 0.0, 0.0))
            .expect("fixture node");
        g.set_node(a, "id", PropValue::Int(1)).unwrap();
        g.set_node(b, "id", PropValue::Int(2)).unwrap();
        g.set_node(a, "mol_id", PropValue::Int(7)).unwrap();
        g.set_node(b, "mol_id", PropValue::Int(7)).unwrap();

        let atoms = &g.to_frame().expect("a schema-conforming graph converts")["atoms"];
        assert_eq!(
            atoms
                .get("id")
                .and_then(|c| c.as_uint())
                .expect("'id' reaches the frame")
                .iter()
                .copied()
                .collect::<Vec<Idx>>(),
            vec![1, 2]
        );
        assert_eq!(
            atoms
                .get("mol_id")
                .and_then(|c| c.as_uint())
                .expect("'mol_id' reaches the frame")
                .iter()
                .copied()
                .collect::<Vec<Idx>>(),
            vec![7, 7]
        );
    }

    #[test]
    fn test_to_frame_keeps_bool_node_columns() {
        let mut g = MolGraph::new();
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("H", 1.1, 0.0, 0.0))
            .expect("fixture node");
        g.set_node(a, "frozen", true).unwrap();
        g.set_node(b, "frozen", false).unwrap();

        assert_eq!(
            g.to_frame().expect("a schema-conforming graph converts")["atoms"]
                .get("frozen")
                .and_then(|c| c.as_bool())
                .expect("bool column reaches the frame")
                .iter()
                .copied()
                .collect::<Vec<bool>>(),
            vec![true, false]
        );
    }

    #[test]
    fn test_extend_from_frame_restores_unsigned_and_bool_node_props() {
        let mut g = MolGraph::new();
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        g.set_node(a, "id", PropValue::Int(4)).unwrap();
        g.set_node(a, "mol_id", PropValue::Int(9)).unwrap();
        g.set_node(a, "frozen", true).unwrap();

        let mut g2 = MolGraph::new();
        g2.extend_from_frame(&g.to_frame().expect("a schema-conforming graph converts"))
            .unwrap();

        let (_, atom) = g2.nodes().next().expect("node round-trips");
        assert_eq!(atom.get("id"), Some(&PropValue::Int(4)));
        assert_eq!(atom.get("mol_id"), Some(&PropValue::Int(9)));
        assert_eq!(atom.get("frozen"), Some(&PropValue::Bool(true)));
    }

    /// The Frame schema declares `x` a float, so a string is not a narrower
    /// or wider `x` — it is a different kind of thing. The door that owns the
    /// key's dtype is the setter, which has a `Result`; letting the value in
    /// and refusing it at `to_frame` leaves the graph holding a column no
    /// frame can ever carry.
    #[test]
    fn set_node_refuses_a_str_under_the_schema_float_key_x() {
        use crate::core::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "x", "left")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'x'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Float.name()),
            "the error names the declared dtype, got {msg}"
        );
        assert!(
            msg.contains(DType::String.name()),
            "the error names the offered dtype, got {msg}"
        );
    }

    /// Same refusal under a different float key: the rule is the schema's, not
    /// a special case carved out for coordinates.
    #[test]
    fn set_node_refuses_a_str_under_the_schema_float_key_charge() {
        use crate::core::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "charge", "negative")
            .expect_err("a str cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'charge'"),
            "the error names the key, got {msg}"
        );
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::String.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// A bool is numeric-adjacent and still not a float: `true` is not `1.0`
    /// under a key the schema declares float.
    #[test]
    fn set_node_refuses_a_bool_under_a_schema_float_key() {
        use crate::core::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "y", true)
            .expect_err("a bool cannot be stored at a schema-float key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'y'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Float.name()) && msg.contains(DType::Bool.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// The refusal runs in both directions: `element` is declared a string, so
    /// an atomic number written there is refused rather than creating an int
    /// column under a string key.
    #[test]
    fn set_node_refuses_an_int_under_a_schema_string_key() {
        use crate::core::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "element", PropValue::Int(6 as I))
            .expect_err("an int cannot be stored at a schema-string key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'element'"),
            "the error names the key, got {msg}"
        );
        assert!(
            msg.contains(DType::String.name()) && msg.contains(DType::Int.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// Width is not semantics: an int under a float key is still a number, so
    /// it is widened rather than refused, and reads back as the float the
    /// schema declares.
    #[test]
    fn set_node_widens_an_int_under_a_schema_float_key() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        g.set_node(n, "x", PropValue::Int(1 as I))
            .expect("an int is a number and widens into a float key");
        assert_eq!(
            g.get_node(n).expect("node exists").get("x"),
            Some(&PropValue::F64(1.0))
        );
    }

    /// An unsigned key is not a numeric key: `1.0` is a float, and storing it
    /// under `id` leaves the graph holding a float column the Frame schema
    /// declares unsigned — a column no frame can carry. The setter owns the
    /// key's dtype, so it refuses there.
    #[test]
    fn set_node_refuses_an_f64_under_a_schema_uint_key() {
        use crate::core::DType;
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "id", 1.0_f64)
            .expect_err("an f64 cannot be stored at a schema-uint key");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("'id'"), "the error names the key, got {msg}");
        assert!(
            msg.contains(DType::Uint.name()) && msg.contains(DType::Float.name()),
            "the error names both dtypes, got {msg}"
        );
    }

    /// The refusal above is about the element type, not about `id` being
    /// closed: a non-negative int is an identifier and still goes in, and
    /// reads back as the int the store holds it as.
    #[test]
    fn set_node_accepts_an_int_under_a_schema_uint_key() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        g.set_node(n, "id", 1 as I)
            .expect("a non-negative int is an identifier");
        assert_eq!(
            g.get_node(n).expect("node exists").get("id"),
            Some(&PropValue::Int(1))
        );
    }

    /// Tightening the element-type door must not restate the sign rule: a
    /// negative under an unsigned key keeps refusing with the message that
    /// names the sign, not a dtype mismatch.
    #[test]
    fn set_node_keeps_the_unsigned_message_for_a_negative_under_a_uint_key() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        let err = g
            .set_node(n, "id", PropValue::Int(-1 as I))
            .expect_err("a negative is not an identifier");
        assert!(
            err.to_string()
                .contains("declared unsigned by the Frame schema"),
            "the sign refusal keeps its own message, got {err}"
        );
    }

    /// The vocabulary is closed but the key space is open: a key the schema
    /// declares nothing about carries whatever the caller stores, because no
    /// frame column type contradicts it.
    #[test]
    fn set_node_accepts_a_str_under_a_key_the_schema_does_not_declare() {
        let mut g = MolGraph::new();
        let n = g.add_node();

        g.set_node(n, "tag", "left")
            .expect("'tag' is not in the canonical vocabulary");
        assert_eq!(
            g.get_node(n).expect("node exists").get_str("tag"),
            Some("left")
        );
    }

    #[test]
    fn test_set_node_rejects_negative_under_unsigned_key() {
        // A sign is semantics, not width: the Frame schema declares `id`
        // unsigned, so this must fail where a Result exists rather than blow up
        // later in `to_frame`, which has none.
        let mut g = MolGraph::new();
        let a = g.add_node();
        assert!(g.set_node(a, "id", PropValue::Int(-1)).is_err());
        assert!(g.set_node(a, "id", PropValue::Int(3)).is_ok());
    }

    // ----- Merge (registry-driven; covers all kinds) -----

    #[test]
    fn test_merge_transfers_all_kinds() {
        let mut src = MolGraph::new();
        let bond = src.register_kind("bond", 2);
        let imp = src.register_kind("improper", 4);
        let a = src.add_node();
        let b = src.add_node();
        let c = src.add_node();
        let d = src.add_node();
        src.add_relation(bond, &[a, b]).unwrap();
        src.add_relation(imp, &[a, b, c, d]).unwrap();

        let mut dst = MolGraph::new();
        let dbond = dst.register_kind("bond", 2);
        let dimp = dst.register_kind("improper", 4);
        let e = dst.add_node();
        let f = dst.add_node();
        dst.add_relation(dbond, &[e, f]).unwrap();

        let map = dst.merge(src).expect("merge succeeds on compatible graphs");
        assert_eq!(dst.n_nodes(), 6);
        assert_eq!(dst.n_relations(dbond), 2);
        assert_eq!(dst.n_relations(dimp), 1, "merge must carry impropers");
        assert_eq!(map.len(), 4, "merge returns one entry per node of other");
    }

    #[test]
    fn test_clone_preserves_handles() {
        let mut g = MolGraph::new();
        let bond = g.register_kind("bonds", 2);
        let a = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("H", 1.0, 0.0, 0.0))
            .expect("fixture node");
        let rid = g.add_relation(bond, &[a, b]).unwrap();
        let g2 = g.clone();
        assert!(g2.get_node(a).is_ok());
        assert!(g2.get_node(b).is_ok());
        assert!(g2.get_relation(bond, rid).is_ok());
        assert_eq!(g2.get_node(a).unwrap().get_str("element"), Some("C"));
    }

    // ----- Clone independence -----

    #[test]
    fn test_clone_independence() {
        let mut g = MolGraph::new();
        let id = g
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .expect("fixture node");
        g.add_node_with(Atom::xyz("H", 1.0, 0.0, 0.0))
            .expect("fixture node");
        let g2 = g.clone();
        g.set_node(id, "x", 99.0).unwrap();
        assert_eq!(g2.get_node(id).unwrap().get_f64("x"), Some(0.0));
        assert_eq!(g2.n_nodes(), 2);
    }

    // ----- Contract B: extend_from_frame and the node/relation writers keep data -----

    /// A frame read into a graph comes back out with its columns in the
    /// order it went in, atoms and relation props alike — never sorted.
    #[test]
    fn to_frame_keeps_the_column_order_extend_from_frame_saw() {
        use ndarray::Array1;

        let mut graph = MolGraph::new();
        graph.register_kind("bonds", 2);
        let mut atoms = Block::new();
        atoms
            .insert("z", Array1::from_vec(vec![0.0 as F, 1.0]).into_dyn())
            .unwrap();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_owned(), "O".to_owned()]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert("x", Array1::from_vec(vec![2.0 as F, 3.0]).into_dyn())
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("type", Array1::from_vec(vec!["b".to_owned()]).into_dyn())
            .unwrap();
        bonds
            .insert("order", Array1::from_vec(vec![2.0 as F]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        graph.extend_from_frame(&frame).unwrap();
        let out = graph.to_frame().unwrap();
        assert_eq!(
            out["atoms"].keys().collect::<Vec<_>>(),
            ["z", "element", "x"]
        );
        assert_eq!(
            out["bonds"].keys().collect::<Vec<_>>(),
            ["atomi", "atomj", "type", "order"]
        );
    }

    /// A relation property column whose dtype contradicts the component the
    /// kind already holds cannot be stored, and dropping the value hands back
    /// a graph whose bonds silently lost the label the frame carried.
    #[test]
    fn extend_from_frame_rejects_a_relation_prop_whose_dtype_conflicts_with_the_kind() {
        use ndarray::Array1;

        let mut graph = MolGraph::new();
        let bonds = graph.register_kind("bonds", 2);
        let a = graph.add_node_with(Atom::new()).expect("fixture node");
        let b = graph.add_node_with(Atom::new()).expect("fixture node");
        let rid = graph.add_relation(bonds, &[a, b]).unwrap();
        graph
            .set_relation_prop(bonds, rid, "tag", PropValue::Int(1 as I))
            .unwrap();

        let mut atoms = Block::new();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_owned(), "C".to_owned()]).into_dyn(),
            )
            .unwrap();
        let mut bonds_block = Block::new();
        bonds_block
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds_block
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds_block
            .insert(
                "tag",
                Array1::from_vec(vec!["single".to_owned()]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds_block);

        let err = graph
            .extend_from_frame(&frame)
            .expect_err("a str 'tag' cannot enter an int 'tag' component");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    /// A graph property is one value per row, so a 2-D column has no reading.
    /// It is refused by name before any node is added, rather than panicking
    /// in the per-row walk.
    #[test]
    fn extend_from_frame_refuses_a_2d_atoms_column_before_adding_a_node() {
        use ndarray::{Array1, Array2};

        let mut atoms = Block::new();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["C".to_owned(), "O".to_owned()]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "xyz",
                Array2::from_shape_vec((2, 3), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
                    .unwrap()
                    .into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);

        let mut graph = MolGraph::new();
        let err = graph
            .extend_from_frame(&frame)
            .expect_err("a (2, 3) column is not a per-row property");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("'atoms'"),
            "the error names the block, got {msg}"
        );
        assert!(
            msg.contains("'xyz'"),
            "the error names the column, got {msg}"
        );
        assert_eq!(graph.n_nodes(), 0, "a refusal adds no node");
    }

    /// `merge` moves node property bags into `self`'s columns. A bag whose
    /// dtype contradicts an existing column cannot be written, and dropping it
    /// returns a merged graph missing a property the caller handed over.
    #[test]
    fn merge_rejects_a_node_prop_whose_dtype_conflicts_with_an_existing_column() {
        let mut dst = MolGraph::new();
        let kept = dst.add_node_with(Atom::new()).expect("fixture node");
        dst.set_node(kept, "tag", PropValue::Int(1 as I)).unwrap();

        let mut src = MolGraph::new();
        let mut incoming = Atom::new();
        incoming.set("tag", "single");
        src.add_node_with(incoming).expect("fixture node");

        let err = dst
            .merge(src)
            .expect_err("a str 'tag' cannot enter an int 'tag' component");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    /// The same rule on the relation side: `merge` writes the incoming
    /// relation's property bag into the kind it matched by name.
    #[test]
    fn merge_rejects_a_relation_prop_whose_dtype_conflicts_with_an_existing_component() {
        let mut dst = MolGraph::new();
        let bonds = dst.register_kind("bonds", 2);
        let a = dst.add_node_with(Atom::new()).expect("fixture node");
        let b = dst.add_node_with(Atom::new()).expect("fixture node");
        let rid = dst.add_relation(bonds, &[a, b]).unwrap();
        dst.set_relation_prop(bonds, rid, "tag", PropValue::Int(1 as I))
            .unwrap();

        let mut src = MolGraph::new();
        let src_bonds = src.register_kind("bonds", 2);
        let c = src.add_node_with(Atom::new()).expect("fixture node");
        let d = src.add_node_with(Atom::new()).expect("fixture node");
        let src_rid = src.add_relation(src_bonds, &[c, d]).unwrap();
        src.set_relation_prop(src_bonds, src_rid, "tag", "single")
            .unwrap();

        let err = dst
            .merge(src)
            .expect_err("a str 'tag' cannot enter an int 'tag' component");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    // ----- Composition: replicate -----

    use crate::core::Atomistic;
    use crate::core::BondNumber;
    use crate::core::PortKind;
    use crate::op::Rigid;

    /// C (0,0,0), O (1,0,0), H handle (-1,0,0); bonds C-O and C-H; one port
    /// (anchor C, handle H). Template node order is C, O, H.
    fn replicate_template() -> Atomistic {
        let mut frag = Atomistic::new();
        let c = frag.add_atom_xyz("C", 0.0, 0.0, 0.0);
        let o = frag.add_atom_xyz("O", 1.0, 0.0, 0.0);
        let h = frag.add_atom_xyz("H", -1.0, 0.0, 0.0);
        frag.add_bond(c, o).expect("fixture bond C-O");
        frag.add_bond(c, h).expect("fixture bond C-H");
        frag.add_port(c, h, PortKind::Symmetric, "a", BondNumber::Single)
            .expect("a bonded H handle is a legal port");
        frag
    }

    /// Copy 0: identity rotation, t = (0,0,5). Copy 1: hand-derived Rz(90 deg)
    /// (x -> y, y -> -x), t = (10,0,0).
    fn replicate_transforms() -> [Rigid; 2] {
        [
            Rigid {
                rotation: Rigid::IDENTITY.rotation,
                translation: [0.0, 0.0, 5.0],
            },
            Rigid {
                rotation: [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                translation: [10.0, 0.0, 0.0],
            },
        ]
    }

    fn xyz_of(g: &MolGraph, id: NodeId) -> [f64; 3] {
        let a = g.get_node(id).expect("live replicated node");
        [
            a.get_f64("x").expect("x"),
            a.get_f64("y").expect("y"),
            a.get_f64("z").expect("z"),
        ]
    }

    fn assert_xyz(got: [f64; 3], want: [f64; 3]) {
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() < 1e-12,
                "component {k}: got {got:?}, want {want:?}"
            );
        }
    }

    /// `self` holding two bonded nodes, one of which carries `extra_key` as a
    /// string: the pre-state every refusal test compares against.
    fn replicate_target(extra_key: &str) -> MolGraph {
        let mut g = MolGraph::new();
        let bonds = g.register_kind("bonds", 2);
        let mut seed = Atom::xyz("N", 0.0, 0.0, 0.0);
        seed.set(extra_key, "bar");
        let a = g.add_node_with(seed).expect("fixture node");
        let b = g
            .add_node_with(Atom::xyz("N", 1.1, 0.0, 0.0))
            .expect("fixture node");
        g.add_relation(bonds, &[a, b]).expect("fixture bond");
        g
    }

    fn assert_target_unchanged(g: &MolGraph) {
        assert_eq!(g.n_nodes(), 2, "no node was written");
        let bonds = g.kind_id("bonds").expect("bonds kind of the fixture");
        assert_eq!(g.n_relations(bonds), 1, "no bond was written");
        assert!(g.kind_id("ports").is_none(), "no kind was registered");
    }

    /// Regression: two rigid copies of a three-atom template. Hard-coded
    /// goldens derived by hand: O (1,0,0) under (I, (0,0,5)) is (1,0,5); under
    /// (Rz(90 deg), (10,0,0)) it is (0,1,0) + (10,0,0) = (10,1,0).
    #[test]
    fn replicate_places_two_rigid_copies_and_stamps_frag_id() {
        let template = replicate_template();
        let mut out = MolGraph::new();
        let handles = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .expect("two copies of a well-formed template replicate");

        assert_xyz(xyz_of(&out, handles[1]), [1.0, 0.0, 5.0]);
        assert_xyz(xyz_of(&out, handles[3 + 1]), [10.0, 1.0, 0.0]);

        let frag_ids: Vec<I> = handles
            .iter()
            .map(|&h| {
                out.get_node(h)
                    .unwrap()
                    .get_int(FRAG_ID)
                    .expect("every replicated node carries frag_id")
            })
            .collect();
        assert_eq!(frag_ids, vec![7, 7, 7, 8, 8, 8]);
    }

    #[test]
    fn replicate_returns_copy_major_handles() {
        let template = replicate_template();
        let mut out = MolGraph::new();
        let handles = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .unwrap();

        assert_eq!(handles.len(), 6);
        assert_eq!(out.n_nodes(), 6);
        let elements: Vec<String> = handles
            .iter()
            .map(|&h| {
                out.get_node(h)
                    .unwrap()
                    .get_str("element")
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(elements, ["C", "O", "H", "C", "O", "H"]);
        // Template node t of copy c sits at c*n + t: each copy's C is its origin
        // image, i.e. that copy's translation.
        assert_xyz(xyz_of(&out, handles[0]), [0.0, 0.0, 5.0]);
        assert_xyz(xyz_of(&out, handles[3]), [10.0, 0.0, 0.0]);
        // Copy 1's H (-1,0,0) under Rz(90 deg) is (0,-1,0), then + (10,0,0).
        assert_xyz(xyz_of(&out, handles[5]), [10.0, -1.0, 0.0]);
    }

    #[test]
    fn replicate_carries_bonds_and_ports_offset_per_copy() {
        let template = replicate_template();
        let mut out = MolGraph::new();
        let handles = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .unwrap();

        let bonds = out.kind_id("bonds").expect("'bonds' registered by name");
        let ports = out.kind_id("ports").expect("'ports' registered by name");
        assert_eq!(out.n_relations(bonds), 4);
        assert_eq!(out.n_relations(ports), 2);

        let copy_of = |id: NodeId| handles.iter().position(|&h| h == id).unwrap() / 3;
        for (_, rel) in out.relations(bonds) {
            assert_eq!(
                copy_of(rel.nodes[0]),
                copy_of(rel.nodes[1]),
                "bond stays in its copy"
            );
        }
        let mut port_ends: Vec<(NodeId, NodeId)> = out
            .relations(ports)
            .map(|(_, rel)| {
                assert_eq!(
                    rel.props.get("port_kind"),
                    Some(&PropValue::Str("$".to_owned())),
                    "port props are copied"
                );
                (rel.nodes[0], rel.nodes[1])
            })
            .collect();
        port_ends.sort_by_key(|&(a, _)| copy_of(a));
        assert_eq!(
            port_ends,
            vec![(handles[0], handles[2]), (handles[3], handles[5])],
            "each port is (anchor C, handle H) of its own copy"
        );
    }

    #[test]
    fn replicate_copies_a_row_without_a_full_triple_untransformed() {
        let mut template = MolGraph::new();
        let mut planar = Atom::new();
        planar.set("element", "C");
        planar.set("x", 1.0);
        planar.set("y", 2.0);
        template.add_node_with(planar).expect("x,y-only node");
        let mut bare = Atom::new();
        bare.set("element", "N");
        template.add_node_with(bare).expect("coordinate-less node");
        template
            .add_node_with(Atom::xyz("O", 1.0, 0.0, 0.0))
            .expect("full-triple node");

        let mut out = MolGraph::new();
        let handles = out
            .replicate(&template, &replicate_transforms()[1..], &[3])
            .unwrap();

        let planar = out.get_node(handles[0]).unwrap();
        assert_eq!(planar.get_f64("x"), Some(1.0));
        assert_eq!(planar.get_f64("y"), Some(2.0));
        assert_eq!(planar.get_f64("z"), None);
        let bare = out.get_node(handles[1]).unwrap();
        assert!(bare.get_f64("x").is_none() && bare.get_f64("y").is_none());
        assert!(bare.get_f64("z").is_none());
        // The full-triple row in the same call is still moved.
        assert_xyz(xyz_of(&out, handles[2]), [10.0, 1.0, 0.0]);
    }

    #[test]
    fn replicate_overwrites_a_template_frag_id() {
        let mut template = replicate_template();
        let c = template.node_ids().next().unwrap();
        template.set_frag_id(c, 99).unwrap();

        let mut out = MolGraph::new();
        let handles = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .unwrap();
        assert_eq!(out.get_node(handles[0]).unwrap().get_int(FRAG_ID), Some(7));
        assert_eq!(out.get_node(handles[3]).unwrap().get_int(FRAG_ID), Some(8));
    }

    #[test]
    fn replicate_refuses_a_length_mismatch_and_writes_nothing() {
        let template = replicate_template();
        let mut out = replicate_target("tag");
        let err = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7])
            .expect_err("two transforms need two frag_ids");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_target_unchanged(&out);
    }

    #[test]
    fn replicate_refuses_a_column_type_conflict_and_writes_nothing() {
        let mut template = replicate_template();
        let c = template.node_ids().next().unwrap();
        template.set_node(c, "foo", 1.5).unwrap();
        let mut out = replicate_target("foo");
        let err = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .expect_err("a float 'foo' cannot enter a str 'foo' column");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_target_unchanged(&out);
    }

    #[test]
    fn replicate_refuses_a_non_int_frag_id_column_and_writes_nothing() {
        let template = replicate_template();
        let mut out = replicate_target(FRAG_ID);
        let err = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .expect_err("frag_id is an Int column; self holds a str one");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_target_unchanged(&out);
    }

    fn sorted_node_columns(g: &MolGraph) -> Vec<String> {
        let mut cols: Vec<String> = g.node_table().columns().map(str::to_owned).collect();
        cols.sort();
        cols
    }

    #[test]
    fn replicate_into_a_non_empty_target_appends_only_new_copies() {
        let template = replicate_template();
        let mut out = replicate_target("tag");
        let before: Vec<(NodeId, Atom)> = out.nodes().collect();

        let handles = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .expect("a compatible non-empty target accepts copies");

        assert_eq!(handles.len(), 6, "only the new copies are returned");
        assert_eq!(out.n_nodes(), 2 + 6);
        for (id, atom) in &before {
            assert!(!handles.contains(id), "an old handle is not returned");
            assert_eq!(&out.get_node(*id).unwrap(), atom, "old node untouched");
        }
        for (i, &h) in handles.iter().enumerate() {
            assert_eq!(
                out.node_table().row(h),
                Some(2 + i),
                "copies follow old rows"
            );
            assert!(out.get_node(h).unwrap().get("tag").is_none());
        }
        let bonds = out.kind_id("bonds").unwrap();
        assert_eq!(out.n_relations(bonds), 1 + 2 * 2, "old + copies * template");
    }

    #[test]
    fn replicate_refuses_a_kind_arity_conflict_and_writes_nothing() {
        let template = replicate_template();
        let mut out = replicate_target("tag");
        let ports = out.register_kind("ports", 3);
        let n_kinds = out.kind_ids().count();
        let cols = sorted_node_columns(&out);

        let err = out
            .replicate(template.as_molgraph(), &replicate_transforms(), &[7, 8])
            .expect_err("template 'ports' is arity 2; self holds arity 3");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(out.n_nodes(), 2);
        assert_eq!(out.kind_ids().count(), n_kinds, "no kind was registered");
        assert_eq!(out.arity(ports), 3);
        assert_eq!(out.n_relations(ports), 0);
        assert_eq!(out.n_relations(out.kind_id("bonds").unwrap()), 1);
        assert_eq!(sorted_node_columns(&out), cols, "no column was added");
    }

    #[test]
    fn replicate_refuses_a_relation_prop_type_conflict_and_writes_nothing() {
        let mut template = MolGraph::new();
        let tlinks = template.register_kind("links", 2);
        let p = template
            .add_node_with(Atom::xyz("C", 0.0, 0.0, 0.0))
            .unwrap();
        let q = template
            .add_node_with(Atom::xyz("C", 1.5, 0.0, 0.0))
            .unwrap();
        let trid = template.add_relation(tlinks, &[p, q]).unwrap();
        template.set_relation_prop(tlinks, trid, "w", 1.5).unwrap();

        let mut out = replicate_target("tag");
        let links = out.register_kind("links", 2);
        let ends: Vec<NodeId> = out.node_ids().collect();
        let rid = out.add_relation(links, &ends).unwrap();
        out.set_relation_prop(links, rid, "w", "heavy").unwrap();
        let n_kinds = out.kind_ids().count();
        let cols = sorted_node_columns(&out);

        let err = out
            .replicate(&template, &replicate_transforms(), &[7, 8])
            .expect_err("a float 'w' cannot enter a str 'w' relation column");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(out.n_nodes(), 2);
        assert_eq!(out.kind_ids().count(), n_kinds);
        assert_eq!(out.n_relations(links), 1);
        assert_eq!(out.n_relations(out.kind_id("bonds").unwrap()), 1);
        assert_eq!(
            out.get_relation(links, rid).unwrap().props.get("w"),
            Some(&PropValue::Str("heavy".to_owned()))
        );
        assert_eq!(sorted_node_columns(&out), cols);
    }

    #[test]
    fn replicate_with_no_transforms_places_nothing_and_leaves_self_untouched() {
        let template = replicate_template();
        let mut out = replicate_target("tag");
        let n_kinds = out.kind_ids().count();
        let cols = sorted_node_columns(&out);

        let handles = out
            .replicate(template.as_molgraph(), &[], &[])
            .expect("no transforms is a valid empty placement");

        assert!(handles.is_empty());
        assert_target_unchanged(&out);
        assert_eq!(out.kind_ids().count(), n_kinds, "no kind was registered");
        assert_eq!(sorted_node_columns(&out), cols, "no column was added");
        assert!(out.node_table().column(FRAG_ID).is_none());
    }

    /// Relation handles are per-kind keys: the first relation of `widgets`
    /// and the first of `bonds` share a handle value. Removing the widget must
    /// leave the bond in both endpoints' adjacency.
    #[test]
    fn relation_row_is_the_position_in_relation_ids() {
        let mut g = MolGraph::new();
        let bonds = g.register_kind("bonds", 2);
        let a = g.add_node();
        let b = g.add_node();
        let c = g.add_node();
        let ab = g.add_relation(bonds, &[a, b]).expect("ab");
        let bc = g.add_relation(bonds, &[b, c]).expect("bc");
        let ca = g.add_relation(bonds, &[c, a]).expect("ca");
        g.remove_relation(bonds, ab).expect("ab removed");

        let ids: Vec<RelationId> = g.relation_ids(bonds).collect();
        for (row, id) in ids.iter().enumerate() {
            assert_eq!(g.relation_row(bonds, *id), Some(row));
        }
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&bc) && ids.contains(&ca));
        assert_eq!(g.relation_row(bonds, ab), None);
    }

    #[test]
    fn removing_a_relation_keeps_an_equal_handle_of_another_kind() {
        let mut g = MolGraph::new();
        let bonds = g.register_kind("bonds", 2);
        let widgets = g.register_kind("widgets", 2);
        let a = g.add_node();
        let b = g.add_node();
        let bond = g.add_relation(bonds, &[a, b]).expect("bond");
        let widget = g.add_relation(widgets, &[a, b]).expect("widget");
        assert_eq!(
            bond.data().as_ffi(),
            widget.data().as_ffi(),
            "fixture shares a handle"
        );

        g.remove_relation(widgets, widget).expect("widget removed");

        let kinds_at_a: Vec<KindId> = g.neighbor_relations(a).map(|(k, _, _)| k).collect();
        let kinds_at_b: Vec<KindId> = g.neighbor_relations(b).map(|(k, _, _)| k).collect();
        assert_eq!(kinds_at_a, vec![bonds]);
        assert_eq!(kinds_at_b, vec![bonds]);
    }
}
