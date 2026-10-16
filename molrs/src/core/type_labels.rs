//! Type names and the Frame's type-id contract: [`TypeName`] and [`TypeLabels`].

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::core::FrameAccess;
use crate::core::keys;
use crate::op::Idx;

/// Starts the qualifier of a [`TypeName`]; reserved inside endpoint labels.
const QUALIFIER: char = '@';
/// Separates the fields of a qualifier; reserved inside a qualifier field.
const FIELD: char = '_';
/// Endpoint separator when no endpoint label contains `-`.
const DASH: &str = "-";
/// Endpoint separator when some endpoint label contains `-`.
const WIDE: &str = "::";

/// True if `token` is a signed integer: an optional leading `+` / `-`, then at
/// least one ASCII digit and nothing else.
pub(crate) fn is_int_token(token: &str) -> bool {
    let digits = token.strip_prefix(['+', '-']).unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// The name of a type, **built** from endpoint labels.
///
/// A type's name is an opaque identifier: a force field stores it verbatim and
/// never reads endpoints back out of it (they are given when the type is
/// defined). `TypeName` is only the conventional way to *build* such a name
/// from endpoint labels, plus [`TypeName::infer_endpoints`] for the one kind of
/// source that carries labels and nothing else.
///
/// # Construction
///
/// `endpoints[@qualifier]`
///
/// - **Endpoints** are joined with `-` (`c3-c3-h1`), or with `::` when any
///   endpoint label itself contains `-` (`tip3p-O::tip3p-H`). An empty
///   endpoint is kept as a position: `["", "CT", "CT", ""]` joins to
///   `-CT-CT-`.
/// - **Qualifier.** [`TypeName::with_qualifier`] appends `@` and `_`-separated
///   fields holding the values a type depends on that are not a function of
///   its endpoint labels (e.g. bond orders, `C_3-C_R@1.5`).
///
/// # Reserved characters
///
/// `@` may not appear in an endpoint label ([`TypeName::join`] and
/// [`TypeName::pair`] reject it), and neither `_` nor `@` may appear in a
/// qualifier field ([`TypeName::with_qualifier`] rejects them).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeName(String);

impl TypeName {
    /// Join endpoint labels (any number: none for an atom, up to five for a
    /// cmap) into a name: `-` between them, or `::` when any label contains
    /// `-`.
    ///
    /// `Err` naming the part when a part contains `@`, which would read as the
    /// start of a qualifier.
    pub fn join(parts: &[&str]) -> Result<TypeName, String> {
        if let Some(bad) = parts.iter().find(|p| p.contains(QUALIFIER)) {
            return Err(format!(
                "type-name part {bad:?} contains '{QUALIFIER}', which starts a qualifier"
            ));
        }
        let sep = if parts.iter().any(|p| p.contains(DASH)) {
            WIDE
        } else {
            DASH
        };
        Ok(TypeName(parts.join(sep)))
    }

    /// The name of the pair of atom types `a` and `b`: `a` itself for a
    /// self-pair, else [`TypeName::join`] of the two. Same `@` rule as `join`.
    pub fn pair(a: &str, b: &str) -> Result<TypeName, String> {
        if a == b {
            Self::join(&[a]).map(|_| TypeName(a.to_owned()))
        } else {
            Self::join(&[a, b])
        }
    }

    /// The whole name, qualifier included.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether a reversal-symmetric endpoint tuple reads in reverse: its
    /// reversed spelling compares smaller than `parts`, slot by slot. A
    /// palindrome (`c3-c3`, `hc-c3-c3-hc`) does not.
    ///
    /// The one orientation rule for a bond, an angle or a proper dihedral —
    /// `i-j-k-l` and `l-k-j-i` are one term, so one spelling is stored — shared
    /// by every typifier and reader so that the same physical term gets the
    /// same name wherever it is named. An improper is not reversal-symmetric
    /// (its centre has a slot) and never goes through this.
    pub fn reads_reversed(parts: &[&str]) -> bool {
        parts.iter().rev().lt(parts.iter())
    }

    /// `parts` in the orientation a reversal-symmetric term is stored in: the
    /// smaller of the tuple and its reverse ([`reads_reversed`](Self::reads_reversed)).
    pub fn orient<'a>(parts: &[&'a str]) -> Vec<&'a str> {
        if Self::reads_reversed(parts) {
            parts.iter().rev().copied().collect()
        } else {
            parts.to_vec()
        }
    }

    /// This name with the qualifier `@f1_f2_…` appended.
    ///
    /// `Err` when the name already has a qualifier, when `fields` is empty,
    /// or when a field is empty or contains `_` or `@`.
    pub fn with_qualifier(&self, fields: &[&str]) -> Result<TypeName, String> {
        if self.0.contains(QUALIFIER) {
            return Err(format!("type name {:?} already has a qualifier", self.0));
        }
        if fields.is_empty() {
            return Err(format!(
                "qualifier for type name {:?} has no fields",
                self.0
            ));
        }
        if let Some(bad) = fields
            .iter()
            .find(|f| f.is_empty() || f.contains(FIELD) || f.contains(QUALIFIER))
        {
            return Err(format!(
                "qualifier field {bad:?} for type name {:?} is empty or contains \
                 '{FIELD}' or '{QUALIFIER}'",
                self.0
            ));
        }
        let joined = fields.join(&FIELD.to_string());
        Ok(TypeName(format!("{}{QUALIFIER}{joined}", self.0)))
    }

    /// The `arity` endpoint labels a **label-only** source means by `label`.
    ///
    /// For readers whose input carries a type label and nothing else — a
    /// LAMMPS `*.ff` include (`bond_coeff CT-CT …`) or a data file's
    /// `* Coeffs` section — and so have no other place to take endpoints from.
    /// Everything else is given its endpoints when a type is defined; a
    /// force field never calls this.
    ///
    /// The inverse of [`TypeName::join`]: the part before the first `@` (the
    /// qualifier is not an endpoint) is split on `::` when it contains `::`,
    /// else on `-`. Empty positions are kept (`-CT-CT-` is four endpoints, the
    /// outer two empty). `Err` naming the label when the split does not give
    /// exactly `arity` endpoints.
    pub fn infer_endpoints(label: &str, arity: usize) -> Result<Vec<&str>, String> {
        let head = label.split_once(QUALIFIER).map_or(label, |(head, _)| head);
        let parts: Vec<&str> = if head.contains(WIDE) {
            head.split(WIDE).collect()
        } else {
            head.split(DASH).collect()
        };
        if parts.len() != arity {
            return Err(format!(
                "type label {label:?} names {} endpoint(s), expected {arity}",
                parts.len()
            ));
        }
        Ok(parts)
    }
}

impl fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Resolved type space of one block: per-row ids, ordered labels, type count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTypeLabels {
    type_ids: Vec<Idx>,
    labels: Option<Vec<String>>,
    n_types: usize,
}

impl BlockTypeLabels {
    /// The 1-based type id of every row, in row order. Empty for a block that
    /// exists only through its inventory meta key.
    pub fn type_ids(&self) -> &[Idx] {
        &self.type_ids
    }

    /// The ordered labels (label `i` has id `i + 1`), or `None` when the ids
    /// carry no labels (pure-integer labels, or a `type_id`-only block with no
    /// inventory).
    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    /// The number of types: the largest id, the number of labels, or at least
    /// 1 for `atoms`.
    pub fn n_types(&self) -> usize {
        self.n_types
    }

    /// Distinct names, sorted byte-wise, or numerically when every name is an
    /// integer (`2` before `10`).
    fn sorted(names: impl IntoIterator<Item = String>) -> Vec<String> {
        let mut items: Vec<String> = names.into_iter().collect();
        items.sort();
        items.dedup();
        if !items.is_empty() && items.iter().all(|s| is_int_token(s)) {
            items.sort_by_key(|s| s.parse::<i64>().unwrap_or(0));
        }
        items
    }

    /// Labels of the inventory meta value `"1:C,2:H"`, ordered by id.
    ///
    /// `Err` as [`parse_inventory_ids`](Self::parse_inventory_ids).
    fn parse_inventory(key: &str, raw: &str) -> Result<Vec<String>, String> {
        Ok(Self::parse_inventory_ids(key, raw)?
            .into_iter()
            .map(|(_, label)| label)
            .collect())
    }

    /// The `(id, label)` entries of the inventory meta value `"1:C,2:H"`,
    /// ordered by id, the ids as written.
    ///
    /// `Err` naming `key` and the token for a pair without `:`, a non-integer
    /// id, an empty label or a repeated id.
    fn parse_inventory_ids(key: &str, raw: &str) -> Result<Vec<(u64, String)>, String> {
        if raw.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut pairs: Vec<(u64, String)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        for token in raw.split(',') {
            let Some((id, label)) = token.split_once(':') else {
                return Err(format!(
                    "meta {key:?}: entry {token:?} is not an `id:label` pair"
                ));
            };
            let id: u64 = id.trim().parse().map_err(|_| {
                format!("meta {key:?}: entry {token:?} has a non-integer id {id:?}")
            })?;
            let label = label.trim();
            if label.is_empty() {
                return Err(format!("meta {key:?}: entry {token:?} has an empty label"));
            }
            if !seen.insert(id) {
                return Err(format!("meta {key:?}: entry {token:?} repeats id {id}"));
            }
            pairs.push((id, label.to_owned()));
        }
        pairs.sort_by_key(|(id, _)| *id);
        Ok(pairs)
    }

    /// Resolve `block`, merging the inventory under `meta_key`.
    fn resolve(
        frame: &impl FrameAccess,
        block: &str,
        meta_key: &str,
    ) -> Result<Option<BlockTypeLabels>, String> {
        let rows = RowTypes::of(frame, block)?;
        Self::build(block, rows, Self::inventory(frame, meta_key)?)
    }

    /// The labels the inventory meta value under `meta_key` declares, in id
    /// order; empty when there is none.
    fn inventory(frame: &impl FrameAccess, meta_key: &str) -> Result<Vec<String>, String> {
        match frame.meta_ref().get(meta_key) {
            None => Ok(Vec::new()),
            Some(value) => {
                let raw = value
                    .as_str()
                    .ok_or_else(|| format!("meta {meta_key:?} must be a string"))?;
                Self::parse_inventory(meta_key, raw)
            }
        }
    }

    /// The types of `block` from its rows' types and its inventory (see the
    /// [`TypeLabels`] id rules).
    fn build(
        block: &str,
        rows: Option<RowTypes>,
        inventory: Vec<String>,
    ) -> Result<Option<BlockTypeLabels>, String> {
        let has_inventory = !inventory.is_empty();
        let min_types = if block == "atoms" { 1 } else { 0 };

        let types = match rows {
            None if inventory.is_empty() => return Ok(None),
            None => {
                let n_types = inventory.len();
                return Ok(Some(BlockTypeLabels {
                    type_ids: Vec::new(),
                    labels: Some(inventory),
                    n_types,
                }));
            }
            Some(RowTypes::Ids(type_ids)) => {
                let max_id = type_ids.iter().copied().max().unwrap_or(0) as usize;
                let n_types = max_id.max(inventory.len()).max(min_types);
                let labels = if inventory.is_empty() {
                    None
                } else {
                    Some(inventory)
                };
                return Ok(Some(BlockTypeLabels {
                    type_ids,
                    labels,
                    n_types,
                }));
            }
            Some(RowTypes::Labels(types)) => types,
        };

        let unique = Self::sorted(types.iter().cloned());
        let pure_int = unique.iter().all(|t| is_int_token(t));

        if pure_int && !has_inventory {
            let mut type_ids = Vec::with_capacity(types.len());
            let mut max_id: Idx = 0;
            for t in &types {
                let id: Idx = t
                    .parse()
                    .map_err(|e| format!("{block}: type {t:?} is not a type id: {e}"))?;
                if id == 0 {
                    return Err(format!("{block}: type id 0 is invalid (ids are 1-based)"));
                }
                max_id = max_id.max(id);
                type_ids.push(id);
            }
            return Ok(Some(BlockTypeLabels {
                type_ids,
                labels: None,
                n_types: (max_id as usize).max(1),
            }));
        }

        let mut all: HashSet<String> = inventory.iter().cloned().collect();
        all.extend(unique);
        let ordered = Self::sorted(all);
        let ids: HashMap<&str, Idx> = ordered
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), (i + 1) as Idx))
            .collect();
        let type_ids = types
            .iter()
            .map(|t| {
                ids.get(t.as_str())
                    .copied()
                    .ok_or_else(|| format!("{block}: type {t:?} missing from its own ordering"))
            })
            .collect::<Result<Vec<Idx>, String>>()?;
        let n_types = ordered.len().max(min_types);
        Ok(Some(BlockTypeLabels {
            type_ids,
            labels: Some(ordered),
            n_types,
        }))
    }
}

/// The per-row types of one block, as its columns state them.
enum RowTypes {
    /// A string `type` column (it wins over `type_id`).
    Labels(Vec<String>),
    /// Numeric ids: `type_id` (unsigned or signed), else a numeric `type`.
    Ids(Vec<Idx>),
}

impl RowTypes {
    /// The types of `block`'s rows; `None` when it has none.
    ///
    /// `Err` for an empty string label, and for rows with neither `type` nor
    /// `type_id`.
    fn of(frame: &impl FrameAccess, block: &str) -> Result<Option<RowTypes>, String> {
        let n = frame
            .visit_block(block, |b| b.n_rows().unwrap_or(0))
            .unwrap_or(0);
        if n == 0 {
            return Ok(None);
        }
        // String labels take precedence over `type_id`.
        if let Some(col) = frame.column(block, keys::TYPE).and_then(|c| c.as_string()) {
            let types: Vec<String> = (0..n).map(|i| col[[i]].clone()).collect();
            if types.iter().any(|t| t.trim().is_empty()) {
                return Err(format!(
                    "{block}: empty type label; every row needs a non-empty type"
                ));
            }
            return Ok(Some(RowTypes::Labels(types)));
        }
        let type_ids: Option<Vec<Idx>> =
            if let Some(col) = frame.column(block, keys::TYPE_ID).and_then(|c| c.as_uint()) {
                Some((0..n).map(|i| col[[i]]).collect())
            } else if let Some(col) = frame.column(block, keys::TYPE_ID).and_then(|c| c.as_int()) {
                Some((0..n).map(|i| col[[i]] as Idx).collect())
            } else if let Some(col) = frame.column(block, keys::TYPE).and_then(|c| c.as_uint()) {
                Some((0..n).map(|i| col[[i]]).collect())
            } else {
                frame
                    .column(block, keys::TYPE)
                    .and_then(|c| c.as_int())
                    .map(|col| (0..n).map(|i| col[[i]] as Idx).collect())
            };
        type_ids.map(RowTypes::Ids).map(Some).ok_or_else(|| {
            format!(
                "frame[{block:?}] has {n} rows but neither 'type' nor 'type_id'; \
                 assign a 'type' or 'type_id' column first"
            )
        })
    }
}

/// The Frame's type-id contract: per block, the type id of every row, the
/// ordered labels, and the number of types.
///
/// The one rule for turning a Frame's per-row type labels into dense 1-based
/// ids, as [`TypeName`] is the one way to build a type's name. Both live in
/// `core` so that format writers and force-field code share them without
/// naming each other.
///
/// Covered blocks, each with its inventory meta key: `atoms`
/// ([`keys::ATOM_TYPE_LABELS`]), `bonds` ([`keys::BOND_TYPE_LABELS`]), `angles`
/// ([`keys::ANGLE_TYPE_LABELS`]), `dihedrals` ([`keys::DIHEDRAL_TYPE_LABELS`]),
/// `impropers` ([`keys::IMPROPER_TYPE_LABELS`]) and `cmaps`
/// ([`keys::CMAP_TYPE_LABELS`]). No block is required.
///
/// # Id rules
///
/// - Ids are 1-based and dense; label `i` of [`BlockTypeLabels::labels`] has id
///   `i + 1`.
/// - A string `type` column wins over `type_id` when both are present. Every
///   label must be non-empty.
/// - When every label is an integer (`"3"`, `"10"`) and the block has no
///   inventory, each label is its own id and no labels are kept; id `0` is an
///   error.
/// - Otherwise the labels get ids in sorted order: byte-wise, or numerically
///   when every label is an integer (`2` before `10`).
/// - The inventory meta key (`"1:C,2:H"`) declares types no row uses; they
///   are merged into the ordering, and a block with no rows is described by
///   its inventory alone. A malformed inventory (an entry without `:`, a
///   non-integer id, an empty label, a repeated id) is an error naming the key.
/// - A block with only a numeric `type_id` (or numeric `type`) column is taken
///   as is; its inventory, if any, supplies the labels.
/// - A block with rows but neither `type` nor `type_id` is an error.
/// - A label is a type name, matched exactly: `h1-c3` and `c3-h1` are two
///   labels with two ids. Whoever writes the labels (a typifier, a reader)
///   writes the name of the type it defined.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeLabels {
    blocks: Vec<(&'static str, BlockTypeLabels)>,
}

impl TypeLabels {
    /// Blocks covered, with their inventory key.
    const BLOCKS: [(&'static str, &'static str); 6] = [
        ("atoms", keys::ATOM_TYPE_LABELS),
        ("bonds", keys::BOND_TYPE_LABELS),
        ("angles", keys::ANGLE_TYPE_LABELS),
        ("dihedrals", keys::DIHEDRAL_TYPE_LABELS),
        ("impropers", keys::IMPROPER_TYPE_LABELS),
        ("cmaps", keys::CMAP_TYPE_LABELS),
    ];

    /// Resolve every covered block of `frame` (see the type-level docs).
    pub fn from_frame(frame: &impl FrameAccess) -> Result<TypeLabels, String> {
        let mut blocks = Vec::new();
        for (block, meta_key) in Self::BLOCKS {
            if let Some(types) = BlockTypeLabels::resolve(frame, block, meta_key)? {
                blocks.push((block, types));
            }
        }
        Ok(TypeLabels { blocks })
    }

    /// [`from_frame`](Self::from_frame), with the rows of the block `extra`
    /// resolved as further rows of the covered `block`, after its own: one id
    /// space over both, so `block`'s type `i` and `extra`'s are one type.
    /// [`BlockTypeLabels::type_ids`] of `block` then lists `block`'s rows,
    /// then `extra`'s. A format that writes two blocks as one section uses
    /// it: a LAMMPS data file writes a `drudes` row (a Drude core–shell
    /// spring) as a `Bonds` row.
    ///
    /// The two are joined by label, so `extra`'s rows need a string `type`,
    /// and so do `block`'s when it has rows too.
    ///
    /// # Errors
    ///
    /// The errors of [`from_frame`](Self::from_frame); a `block` this
    /// contract does not cover; and rows typed by numeric `type_id` in either
    /// block while `extra` has rows, naming both blocks.
    pub fn from_frame_folded(
        frame: &impl FrameAccess,
        block: &str,
        extra: &str,
    ) -> Result<TypeLabels, String> {
        let Some(meta_key) = Self::inventory_key(block) else {
            return Err(format!("block {block:?} has no type-label inventory"));
        };
        let mut labels = Self::from_frame(frame)?;
        let Some(more) = RowTypes::of(frame, extra)? else {
            return Ok(labels);
        };
        let joined = match (RowTypes::of(frame, block)?, more) {
            (None, RowTypes::Labels(more)) => more,
            (Some(RowTypes::Labels(mut own)), RowTypes::Labels(more)) => {
                own.extend(more);
                own
            }
            _ => {
                return Err(format!(
                    "frame[{extra:?}] rows are typed as frame[{block:?}] rows by their \
                     string 'type' labels; give both blocks a 'type' column (not a \
                     numeric 'type_id')"
                ));
            }
        };
        let inventory = BlockTypeLabels::inventory(frame, meta_key)?;
        let types = BlockTypeLabels::build(block, Some(RowTypes::Labels(joined)), inventory)?;
        let name = Self::BLOCKS
            .iter()
            .map(|(name, _)| *name)
            .find(|name| *name == block)
            .expect("inventory_key found it");
        labels.blocks.retain(|(n, _)| *n != name);
        labels.blocks.extend(types.map(|types| (name, types)));
        Ok(labels)
    }

    /// The resolved types of `block`, or `None` when the block has neither
    /// rows nor an inventory.
    pub fn block(&self, block: &str) -> Option<&BlockTypeLabels> {
        self.blocks
            .iter()
            .find(|(name, _)| *name == block)
            .map(|(_, types)| types)
    }

    /// The `(id, label)` entries `block`'s inventory meta key declares, ordered
    /// by id, **the ids as written** — not re-sorted as
    /// [`from_frame`](Self::from_frame) orders them. Empty when the frame has
    /// no inventory for `block`. This is what a format that numbers its types
    /// itself (a LAMMPS data file's `* Type Labels`) declared, for reading
    /// rows that refer to those numbers.
    ///
    /// # Errors
    ///
    /// A `block` this contract does not cover, a non-string meta value, and a
    /// malformed inventory (see the type-level docs), each naming the key.
    pub fn declared_ids(
        frame: &impl FrameAccess,
        block: &str,
    ) -> Result<Vec<(u64, String)>, String> {
        let key = Self::inventory_key(block)
            .ok_or_else(|| format!("block {block:?} has no type-label inventory"))?;
        match frame.meta_ref().get(key) {
            None => Ok(Vec::new()),
            Some(value) => {
                let raw = value
                    .as_str()
                    .ok_or_else(|| format!("meta {key:?} must be a string"))?;
                BlockTypeLabels::parse_inventory_ids(key, raw)
            }
        }
    }

    /// The inventory meta key of `block` (`"atoms"` → `"atom_type_labels"`),
    /// or `None` for a block this contract does not cover.
    pub fn inventory_key(block: &str) -> Option<&'static str> {
        Self::BLOCKS
            .iter()
            .find(|(name, _)| *name == block)
            .map(|(_, key)| *key)
    }

    /// Declare `labels` as types of `block` whether or not a row uses them,
    /// by merging them into the block's inventory meta key.
    ///
    /// The inventory becomes the union of what it declared and `labels`, in
    /// the sorted order [`TypeLabels`] gives ids in, so for a block typed by
    /// string labels the ids of the declared and the used types agree with
    /// what [`from_frame`](Self::from_frame) resolves. (A block typed only by
    /// numeric `type_id` takes its labels from the inventory *by id*, which
    /// re-sorting would scramble: declare labels on string-typed blocks.)
    ///
    /// # Errors
    ///
    /// `Err` naming the label for an empty label or one containing `,` (the
    /// inventory separator), for a `block` not covered (see the type docs),
    /// and for an existing inventory that does not parse. `frame` is
    /// unchanged on error.
    pub fn declare<S: AsRef<str>>(
        frame: &mut crate::core::Frame,
        block: &str,
        labels: impl IntoIterator<Item = S>,
    ) -> Result<(), String> {
        let key = Self::inventory_key(block)
            .ok_or_else(|| format!("block {block:?} has no type-label inventory"))?;
        let mut all: Vec<String> = match frame.meta.get(key) {
            None => Vec::new(),
            Some(value) => {
                let raw = value
                    .as_str()
                    .ok_or_else(|| format!("meta {key:?} must be a string"))?;
                BlockTypeLabels::parse_inventory(key, raw)?
            }
        };
        for label in labels {
            let label = label.as_ref().trim();
            if label.is_empty() {
                return Err(format!("{block}: an empty type label cannot be declared"));
            }
            if label.contains(',') {
                return Err(format!(
                    "{block}: type label {label:?} contains ',', the inventory separator"
                ));
            }
            all.push(label.to_owned());
        }
        let ordered = BlockTypeLabels::sorted(all);
        if ordered.is_empty() {
            return Ok(());
        }
        let packed = ordered
            .iter()
            .enumerate()
            .map(|(i, label)| format!("{}:{label}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        frame.meta.insert(key, packed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::core::Frame;
    use crate::core::keys;
    use crate::op::Idx;
    use ndarray::{ArrayD, IxDyn};

    /// Folding `drudes` into `bonds` gives one id space: the bonds' rows,
    /// then the drudes', ids by the sorted union of their labels.
    #[test]
    fn a_folded_block_shares_the_id_space_of_its_host() {
        let mut frame = Frame::new();
        frame.insert("bonds", label_block(&["c3-hc", "c3-c3", "c3-hc"]));
        frame.insert("drudes", label_block(&["c3-Dc3"]));
        let labels = TypeLabels::from_frame_folded(&frame, "bonds", "drudes").unwrap();
        let bonds = labels.block("bonds").unwrap();
        assert_eq!(bonds.labels().unwrap(), ["c3-Dc3", "c3-c3", "c3-hc"]);
        assert_eq!(bonds.type_ids(), [3, 2, 3, 1]);
        assert_eq!(bonds.n_types(), 3);

        // Drude rows alone fold into an absent host.
        let mut alone = Frame::new();
        alone.insert("drudes", label_block(&["c3-Dc3"]));
        let labels = TypeLabels::from_frame_folded(&alone, "bonds", "drudes").unwrap();
        assert_eq!(labels.block("bonds").unwrap().type_ids(), [1]);

        // No `extra` rows: exactly `from_frame`.
        frame.remove("drudes");
        assert_eq!(
            TypeLabels::from_frame_folded(&frame, "bonds", "drudes").unwrap(),
            TypeLabels::from_frame(&frame).unwrap()
        );
    }

    #[test]
    fn a_fold_over_numeric_type_ids_is_refused_naming_both_blocks() {
        let mut frame = Frame::new();
        frame.insert("bonds", type_id_block(&[1, 2]));
        frame.insert("drudes", label_block(&["c3-Dc3"]));
        let err = TypeLabels::from_frame_folded(&frame, "bonds", "drudes").unwrap_err();
        assert!(err.contains("drudes") && err.contains("bonds"), "{err}");
        assert!(TypeLabels::from_frame_folded(&frame, "widgets", "drudes").is_err());
    }

    #[test]
    fn declare_merges_into_the_sorted_inventory() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["c3", "c3"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, "1:oh");
        TypeLabels::declare(&mut frame, "atoms", ["hc", "c3"]).unwrap();
        assert_eq!(
            frame
                .meta
                .get(keys::ATOM_TYPE_LABELS)
                .and_then(|v| v.as_str()),
            Some("1:c3,2:hc,3:oh")
        );
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").unwrap();
        assert_eq!(atoms.labels().unwrap(), ["c3", "hc", "oh"]);
        assert_eq!(atoms.type_ids(), [1, 1]);
    }

    #[test]
    fn declare_refuses_empty_and_separator_labels_and_unknown_blocks() {
        let mut frame = Frame::new();
        assert!(TypeLabels::declare(&mut frame, "atoms", [" "]).is_err());
        assert!(TypeLabels::declare(&mut frame, "atoms", ["a,b"]).is_err());
        assert!(TypeLabels::declare(&mut frame, "pairs", ["a"]).is_err());
        assert!(frame.meta.get(keys::ATOM_TYPE_LABELS).is_none());
    }

    // ------------------------------------------------------------------
    // Fixture builders
    // ------------------------------------------------------------------

    /// A block whose only column is the string `type` label per row.
    fn label_block(types: &[&str]) -> Block {
        let mut block = Block::new();
        block
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    IxDyn(&[types.len()]),
                    types.iter().map(|t| t.to_string()).collect::<Vec<String>>(),
                )
                .unwrap(),
            )
            .unwrap();
        block
    }

    /// A block whose only column is the numeric `type_id` per row.
    fn type_id_block(ids: &[Idx]) -> Block {
        let mut block = Block::new();
        block
            .insert(
                keys::TYPE_ID,
                ArrayD::from_shape_vec(IxDyn(&[ids.len()]), ids.to_vec()).unwrap(),
            )
            .unwrap();
        block
    }

    // ------------------------------------------------------------------
    // TypeName: building a name from endpoint labels
    // ------------------------------------------------------------------

    #[test]
    fn join_uses_hyphen_between_plain_parts() {
        let name = TypeName::join(&["c3", "c3", "h1"]).unwrap();
        assert_eq!(name.as_str(), "c3-c3-h1");
    }

    #[test]
    fn join_takes_the_five_labels_of_a_cmap() {
        let name = TypeName::join(&["C", "NH1", "CT1", "C", "NH1"]).unwrap();
        assert_eq!(name.as_str(), "C-NH1-CT1-C-NH1");
        assert_eq!(
            TypeName::infer_endpoints(name.as_str(), 5).unwrap(),
            ["C", "NH1", "CT1", "C", "NH1"]
        );
    }

    #[test]
    fn join_uses_double_colon_when_a_part_contains_hyphen() {
        let name = TypeName::join(&["tip3p-O", "tip3p-H"]).unwrap();
        assert_eq!(name.as_str(), "tip3p-O::tip3p-H");
    }

    #[test]
    fn join_keeps_empty_wildcard_positions() {
        let name = TypeName::join(&["", "CT", "CT", ""]).unwrap();
        assert_eq!(name.as_str(), "-CT-CT-");
    }

    #[test]
    fn join_rejects_part_containing_at_and_names_it() {
        let Err(err) = TypeName::join(&["a@b", "c"]) else {
            panic!("a part containing '@' must be rejected");
        };
        assert!(err.contains("a@b"), "{err}");
    }

    #[test]
    fn pair_of_equal_types_is_the_type_itself() {
        assert_eq!(TypeName::pair("A", "A").unwrap().as_str(), "A");
    }

    #[test]
    fn pair_of_distinct_types_is_their_join() {
        assert_eq!(TypeName::pair("A", "B").unwrap().as_str(), "A-B");
        assert_eq!(
            TypeName::pair("tip3p-O", "tip3p-H").unwrap().as_str(),
            "tip3p-O::tip3p-H"
        );
    }

    #[test]
    fn pair_rejects_part_containing_at() {
        assert!(TypeName::pair("a@b", "c").is_err());
        assert!(TypeName::pair("a@b", "a@b").is_err());
    }

    #[test]
    fn reads_reversed_when_the_reversed_spelling_is_smaller() {
        assert!(TypeName::reads_reversed(&["oh", "c3"]));
        assert!(!TypeName::reads_reversed(&["c3", "oh"]));
        assert!(TypeName::reads_reversed(&["os", "c3", "ca", "hc"]));
    }

    #[test]
    fn a_palindrome_does_not_read_reversed() {
        assert!(!TypeName::reads_reversed(&["c3", "c3"]));
        assert!(!TypeName::reads_reversed(&["hc", "c3", "c3", "hc"]));
    }

    #[test]
    fn orient_gives_one_spelling_for_both_orders() {
        assert_eq!(TypeName::orient(&["oh", "c3"]), vec!["c3", "oh"]);
        assert_eq!(TypeName::orient(&["c3", "oh"]), vec!["c3", "oh"]);
        assert_eq!(
            TypeName::orient(&["oh", "c3", "hc"]),
            TypeName::orient(&["hc", "c3", "oh"])
        );
        assert_eq!(
            TypeName::orient(&["hc", "c3", "c3", "hc"]),
            vec!["hc", "c3", "c3", "hc"]
        );
    }

    #[test]
    fn display_prints_the_whole_name() {
        let name = TypeName::join(&["C_3", "C_R"])
            .unwrap()
            .with_qualifier(&["1.5"])
            .unwrap();
        assert_eq!(format!("{name}"), "C_3-C_R@1.5");
    }

    // ------------------------------------------------------------------
    // TypeName: `@` qualifier
    // ------------------------------------------------------------------

    /// `C_3-C_R`, the bond every qualifier test starts from.
    fn c3_cr() -> TypeName {
        TypeName::join(&["C_3", "C_R"]).unwrap()
    }

    #[test]
    fn with_qualifier_appends_single_field() {
        let name = c3_cr().with_qualifier(&["1.5"]).unwrap();
        assert_eq!(name.as_str(), "C_3-C_R@1.5");
    }

    #[test]
    fn with_qualifier_joins_fields_with_underscore() {
        let name = TypeName::join(&["C_3", "C_R", "O_2"])
            .unwrap()
            .with_qualifier(&["1", "1.5", "2"])
            .unwrap();
        assert_eq!(name.as_str(), "C_3-C_R-O_2@1_1.5_2");
    }

    #[test]
    fn with_qualifier_rejects_already_qualified_name() {
        let name = c3_cr().with_qualifier(&["1.5"]).unwrap();
        assert!(name.with_qualifier(&["2"]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_no_fields() {
        assert!(c3_cr().with_qualifier(&[]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_empty_field() {
        assert!(c3_cr().with_qualifier(&[""]).is_err());
        assert!(c3_cr().with_qualifier(&["1", ""]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_field_containing_underscore() {
        assert!(c3_cr().with_qualifier(&["1_5"]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_field_containing_at() {
        assert!(c3_cr().with_qualifier(&["1@5"]).is_err());
    }

    // ------------------------------------------------------------------
    // TypeName::infer_endpoints: label-only sources
    // ------------------------------------------------------------------

    #[test]
    fn infer_endpoints_splits_on_hyphen() {
        assert_eq!(
            TypeName::infer_endpoints("c3-c3-h1", 3).unwrap(),
            vec!["c3", "c3", "h1"]
        );
    }

    #[test]
    fn infer_endpoints_splits_on_double_colon_first() {
        assert_eq!(
            TypeName::infer_endpoints("tip3p-O::tip3p-H", 2).unwrap(),
            vec!["tip3p-O", "tip3p-H"]
        );
    }

    #[test]
    fn infer_endpoints_leaves_the_qualifier_out() {
        assert_eq!(
            TypeName::infer_endpoints("C_3-C_R-O_2@1_1.5_2", 3).unwrap(),
            vec!["C_3", "C_R", "O_2"]
        );
    }

    #[test]
    fn infer_endpoints_keeps_empty_wildcard_positions() {
        assert_eq!(
            TypeName::infer_endpoints("-CT-CT-", 4).unwrap(),
            vec!["", "CT", "CT", ""]
        );
    }

    #[test]
    fn infer_endpoints_off_the_arity_is_an_error_naming_the_label() {
        let Err(err) = TypeName::infer_endpoints("c3-h1", 3) else {
            panic!("two endpoints for an arity of three must be rejected");
        };
        assert!(err.contains("c3-h1"), "{err}");
        assert!(err.contains('3'), "{err}");
    }

    // ------------------------------------------------------------------
    // TypeLabels: the Frame's type-id contract
    // ------------------------------------------------------------------

    #[test]
    fn pure_integer_labels_map_to_themselves() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["3", "1", "10"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(atoms.type_ids(), &[3, 1, 10][..]);
        assert!(atoms.labels().is_none());
        assert_eq!(atoms.n_types(), 10);
    }

    #[test]
    fn integer_id_zero_is_rejected() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["0", "1"]));
        assert!(TypeLabels::from_frame(&frame).is_err());
    }

    #[test]
    fn other_labels_get_dense_ids_in_sorted_order() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["O", "C", "H", "C"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(
            atoms.labels().map(|l| l.to_vec()),
            Some(vec!["C".to_string(), "H".to_string(), "O".to_string()])
        );
        assert_eq!(atoms.type_ids(), &[3, 1, 2, 1][..]);
        assert_eq!(atoms.n_types(), 3);
    }

    #[test]
    fn integer_labels_with_inventory_sort_numerically() {
        // Lexical order would put "10" before "2".
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["10", "2"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, "1:2,2:10");
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(
            atoms.labels().map(|l| l.to_vec()),
            Some(vec!["2".to_string(), "10".to_string()])
        );
        assert_eq!(atoms.type_ids(), &[2, 1][..]);
        assert_eq!(atoms.n_types(), 2);
    }

    #[test]
    fn inventory_meta_adds_unused_types() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["H", "C"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, "1:C,2:H,3:O");
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(
            atoms.labels().map(|l| l.to_vec()),
            Some(vec!["C".to_string(), "H".to_string(), "O".to_string()])
        );
        assert_eq!(atoms.type_ids(), &[2, 1][..]);
        assert_eq!(atoms.n_types(), 3);
    }

    #[test]
    fn inventory_only_block_has_labels_and_no_rows() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C"]));
        frame.meta.insert(keys::BOND_TYPE_LABELS, "1:c3-h1,2:c3-c3");
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let bonds = labels.block("bonds").expect("bonds from inventory");
        assert!(bonds.type_ids().is_empty());
        assert_eq!(
            bonds.labels().map(|l| l.to_vec()),
            Some(vec!["c3-h1".to_string(), "c3-c3".to_string()])
        );
        assert_eq!(bonds.n_types(), 2);
    }

    #[test]
    fn absent_block_without_inventory_is_none() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        assert!(labels.block("bonds").is_none());
        assert!(labels.block("impropers").is_none());
    }

    /// A label is a type name, matched exactly: a bond labelled both ways
    /// round is two labels with two ids, in sorted order.
    #[test]
    fn reversed_bond_labels_are_two_labels() {
        let mut frame = Frame::new();
        frame.insert("bonds", label_block(&["h1-c3", "c3-h1", "h1-c3"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let bonds = labels.block("bonds").expect("bonds resolved");
        assert_eq!(
            bonds.labels().map(|l| l.to_vec()),
            Some(vec!["c3-h1".to_string(), "h1-c3".to_string()])
        );
        assert_eq!(bonds.type_ids(), &[2, 1, 2][..]);
        assert_eq!(bonds.n_types(), 2);
    }

    /// Qualified angle labels in both orientations stay two labels too.
    #[test]
    fn reversed_qualified_angle_labels_are_two_labels() {
        let mut frame = Frame::new();
        frame.insert(
            "angles",
            label_block(&["C_3-C_R-O_2@1_1.5_2", "O_2-C_R-C_3@1.5_1_2"]),
        );
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let angles = labels.block("angles").expect("angles resolved");
        assert_eq!(
            angles.labels().map(|l| l.to_vec()),
            Some(vec![
                "C_3-C_R-O_2@1_1.5_2".to_string(),
                "O_2-C_R-C_3@1.5_1_2".to_string(),
            ])
        );
        assert_eq!(angles.type_ids(), &[1, 2][..]);
    }

    #[test]
    fn reversed_improper_labels_keep_separate_ids() {
        let mut frame = Frame::new();
        frame.insert(
            "impropers",
            label_block(&["C_R-P_3+3-C_R-C_R", "C_R-C_R-P_3+3-C_R"]),
        );
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let impropers = labels.block("impropers").expect("impropers resolved");
        assert_eq!(
            impropers.labels().map(|l| l.to_vec()),
            Some(vec![
                "C_R-C_R-P_3+3-C_R".to_string(),
                "C_R-P_3+3-C_R-C_R".to_string(),
            ])
        );
        assert_eq!(impropers.type_ids(), &[2, 1][..]);
        assert_eq!(impropers.n_types(), 2);
    }

    #[test]
    fn type_id_only_block_is_taken_as_is() {
        let mut frame = Frame::new();
        frame.insert("atoms", type_id_block(&[2, 1, 2]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(atoms.type_ids(), &[2, 1, 2][..]);
        assert!(atoms.labels().is_none());
        assert_eq!(atoms.n_types(), 2);
    }

    #[test]
    fn empty_label_is_rejected() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C", ""]));
        assert!(TypeLabels::from_frame(&frame).is_err());
    }

    #[test]
    fn rows_without_type_or_type_id_are_rejected() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::X,
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f64, 1.0]).unwrap(),
            )
            .unwrap();
        frame.insert("atoms", atoms);
        assert!(TypeLabels::from_frame(&frame).is_err());
    }

    // ------------------------------------------------------------------
    // Inventory meta parser: malformed input is an error naming the key
    // ------------------------------------------------------------------

    fn inventory_error(raw: &str) -> String {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, raw);
        let Err(err) = TypeLabels::from_frame(&frame) else {
            panic!("malformed inventory {raw:?} must be an error");
        };
        err
    }

    #[test]
    fn inventory_pair_without_colon_is_rejected() {
        let err = inventory_error("1C");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
        assert!(err.contains("1C"), "{err}");
    }

    #[test]
    fn inventory_non_integer_id_is_rejected() {
        let err = inventory_error("x:C");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
        assert!(err.contains('x'), "{err}");
    }

    #[test]
    fn inventory_empty_label_is_rejected() {
        let err = inventory_error("1:");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
    }

    #[test]
    fn inventory_repeated_id_is_rejected() {
        let err = inventory_error("1:C,1:H");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
    }
}
