//! Frame: a dictionary mapping string keys to heterogeneous [`Block`]s.

use indexmap::IndexMap;
use std::ops::{Index, IndexMut};

use crate::core::Block;
use crate::core::MetaMap;
use crate::core::MolRsError;
use crate::core::SimBox;
use crate::core::schema::block_names::ATOMS;

/// A dictionary from string keys to [`Block`]s.
///
/// Frame provides a simple container for organizing multiple blocks of data,
/// typically representing different aspects of a molecular system (e.g., atoms,
/// bonds, velocities). Each block can have different numbers of rows and different
/// column types.
///
/// Blocks iterate in insertion order, with the same rules as a
/// [`Block`]'s columns: re-inserting a key keeps its position,
/// [`remove`](Self::remove) keeps the others in order, and
/// [`rename_block`](Self::rename_block) keeps the renamed block in place.
///
/// Each `Block` manages its own `nrows` invariant; `Frame` itself only manages
/// the mapping from names to blocks and does **not** enforce cross-block axis-0
/// consistency.
///
/// # Examples
///
/// ```
/// use molrs::core::Frame;
/// use molrs::core::Block;
/// use molrs::op::{F, Idx};
/// use ndarray::Array1;
///
/// let mut frame = Frame::new();
///
/// // Create an atoms block
/// let mut atoms = Block::new();
/// atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(), None).unwrap();
/// atoms.insert("y", Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(), None).unwrap();
/// atoms.insert("id", Array1::from_vec(vec![1 as Idx, 2 as Idx, 3 as Idx]).into_dyn(), None).unwrap();
///
/// frame.insert("atoms", atoms);
///
/// // Access via Index trait
/// let atoms_ref = &frame["atoms"];
/// assert_eq!(atoms_ref.n_rows(), Some(3));
///
/// // Add metadata
/// frame.meta.insert("title", "My Molecule");
/// ```
#[derive(Default, Clone)]
pub struct Frame {
    map: IndexMap<String, Block>,
    /// Exact-dtype metadata associated with the frame.
    pub meta: MetaMap,
    /// Simulation box defining periodic boundary conditions.
    pub simbox: Option<SimBox>,
}

/// Type alias for the result of into_inner().
type IntoInnerResult = (IndexMap<String, Block>, MetaMap, Option<SimBox>);

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Frame");

        // Format blocks as a map of name -> (nrows, ncols)
        let mut blocks_map = std::collections::BTreeMap::new();
        for (k, b) in &self.map {
            blocks_map.insert(k.as_str(), (b.n_rows(), b.len()));
        }
        debug_struct.field("blocks", &blocks_map);

        // Show metadata if non-empty
        if !self.meta.is_empty() {
            debug_struct.field("meta", &self.meta);
        }

        debug_struct.finish()
    }
}

impl Frame {
    /// Creates an empty Frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    ///
    /// let frame = Frame::new();
    /// assert!(frame.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            map: IndexMap::new(),
            meta: MetaMap::new(),
            simbox: None,
        }
    }

    /// A copy whose blocks share no column buffer with this frame.
    ///
    /// Every block goes through [`Block::deep_copy`]; `meta` and the box are
    /// cloned (they hold no shared buffers).
    pub fn deep_copy(&self) -> Frame {
        Frame {
            map: self
                .map
                .iter()
                .map(|(key, block)| (key.clone(), block.deep_copy()))
                .collect(),
            meta: self.meta.clone(),
            simbox: self.simbox.clone(),
        }
    }

    /// Consumes the Frame and returns the inner map of blocks, metadata, and simbox.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title", "Test");
    ///
    /// let (blocks, meta, simbox) = frame.into_inner();
    /// assert_eq!(blocks.len(), 1);
    /// assert_eq!(meta.get("title").unwrap().as_str(), Some("Test"));
    /// assert!(simbox.is_none());
    /// ```
    pub fn into_inner(self) -> IntoInnerResult {
        (self.map, self.meta, self.simbox)
    }

    /// Number of blocks (keys) in the frame.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    ///
    /// let mut frame = Frame::new();
    /// assert_eq!(frame.len(), 0);
    ///
    /// frame.insert("atoms", Block::new());
    /// assert_eq!(frame.len(), 1);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns true if the frame contains no blocks.
    ///
    /// Note: This only checks blocks, not metadata.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns true if the frame contains the specified key.
    #[inline]
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Gets an immutable reference to the block for `key` if present.
    ///
    /// For a panicking version, use the `Index` trait: `&frame["key"]`.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Block> {
        self.map.get(key)
    }

    /// Gets a mutable reference to the block for `key` if present.
    ///
    /// For a panicking version, use the `IndexMut` trait: `&mut frame["key"]`.
    #[inline]
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Block> {
        self.map.get_mut(key)
    }

    /// Inserts a block under `key`. Returns the previous block if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    ///
    /// let mut frame = Frame::new();
    /// let old = frame.insert("atoms", Block::new());
    /// assert!(old.is_none());
    ///
    /// let old = frame.insert("atoms", Block::new());
    /// assert!(old.is_some());
    /// ```
    pub fn insert(&mut self, key: impl Into<String>, block: Block) -> Option<Block> {
        self.map.insert(key.into(), block)
    }

    /// Removes and returns the block for `key`, if present.
    pub fn remove(&mut self, key: &str) -> Option<Block> {
        self.map.shift_remove(key)
    }

    /// Removes every block — like [`len`](Self::len),
    /// [`is_empty`](Self::is_empty), [`insert`](Self::insert) and
    /// [`remove`](Self::remove), it acts on the block map only. `meta` and
    /// `simbox` are public fields and are cleared directly
    /// (`frame.meta.clear()`, `frame.simbox = None`).
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    ///
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", Block::new());
    /// frame.meta.insert("title", "Test");
    ///
    /// frame.clear();
    /// assert!(frame.is_empty());
    /// assert!(!frame.meta.is_empty()); // metadata preserved
    /// ```
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Renames a block in the frame.
    ///
    /// Returns `true` if the block was successfully renamed, `false` if the old block
    /// doesn't exist or the new block name already exists.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// assert!(frame.rename_block("atoms", "molecules"));
    /// assert!(!frame.contains_key("atoms"));
    /// assert!(frame.contains_key("molecules"));
    /// ```
    pub fn rename_block(&mut self, old_key: &str, new_key: &str) -> bool {
        // Check if old_key exists and new_key doesn't exist
        if !self.map.contains_key(old_key) || self.map.contains_key(new_key) {
            return false;
        }

        // Remove the old key and re-insert the block at the same position.
        if let Some((index, _, block)) = self.map.shift_remove_full(old_key) {
            self.map.shift_insert(index, new_key.to_string(), block);
            true
        } else {
            false
        }
    }

    /// Returns an iterator over (&str, &Block).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Block)> {
        self.map.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns a mutable iterator over (&str, &mut Block).
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// for (_name, block) in frame.iter_mut() {
    ///     // Can mutate blocks
    ///     if let Some(x) = block.get_mut("x").and_then(|c| c.get_mut::<f64>()) {
    ///         x[[0]] = 99.0 as F;
    ///     }
    /// }
    /// ```
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&str, &mut Block)> {
        self.map.iter_mut().map(|(k, v)| (k.as_str(), v))
    }

    /// Returns an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.map.keys().map(|k| k.as_str())
    }

    /// Returns an iterator over block references.
    pub fn values(&self) -> impl Iterator<Item = &Block> {
        self.map.values()
    }

    /// Returns a mutable iterator over block references.
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Block> {
        self.map.values_mut()
    }

    /// Judge this frame against the canonical Frame schema.
    ///
    /// Delegates entirely to [`crate::core::schema::Validator::canonical`] —
    /// this method does **not** re-implement dtype, shape, required-column, or
    /// endpoint-range checks. Callers that need the full report (every
    /// violation at once) should use the `Validator` directly; this is the
    /// short form for the common "is this frame well-formed?" gate.
    ///
    /// # Returns
    /// - `Ok(())` if the frame conforms
    /// - `Err(MolRsError::Validation)` with the full report text otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    /// use molrs::op::F;
    /// use ndarray::Array1;
    ///
    /// let mut frame = Frame::new();
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(), None).unwrap();
    /// atoms.insert("y", Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(), None).unwrap();
    /// frame.insert("atoms", atoms);
    ///
    /// assert!(frame.validate().is_ok());
    /// ```
    pub fn validate(&self) -> Result<(), MolRsError> {
        crate::core::schema::Validator::canonical()
            .validate(self)
            .map_err(|report| MolRsError::validation(report.to_string()))
    }

    /// A new frame holding the rows `rows` of `block`, with every relation
    /// block that indexes `block` cut down to the selection and renumbered.
    ///
    /// A *relation block* is one whose rows name rows of another block by
    /// index: each `bonds` row names two `atoms` rows in its 0-based endpoint
    /// columns `atomi` / `atomj`, an `angles` row three. Selecting atoms must
    /// therefore also drop the bonds that leave the selection and rewrite the
    /// surviving endpoints to the new row numbers.
    ///
    /// - `block` is gathered in the order `rows` gives: old row `rows[k]`
    ///   becomes new row `k`. Every column and its validity mask (the per-row
    ///   flag that marks a null cell, see
    ///   [`Block::validity`](crate::core::Block::validity)) travel.
    /// - A relation block whose endpoints index `block` (canonical `bonds`,
    ///   `angles`, …, and any unspecified block carrying `atomi`..`atoml`)
    ///   keeps only the rows whose endpoints all lie in `rows`, in their
    ///   original order, with each endpoint rewritten to its new row.
    ///   Endpoints are assumed valid (run [`validate`](Self::validate)
    ///   first); a relation row whose endpoint is out of range is dropped.
    /// - References into filtered relation tables are filtered and renumbered
    ///   transitively, including reference cycles. Selected rows themselves
    ///   may not reference rows removed by that propagation.
    /// - Every unaffected block is copied unchanged, into new buffers, as are the
    ///   box and `meta`: the subset shares no buffer with this frame.
    ///
    /// An empty `rows` is legal and gives zero-row blocks.
    ///
    /// # Errors
    ///
    /// The frame itself is never modified.
    ///
    /// - [`MolRsError::NotFound`] if there is no block `block`.
    /// - [`MolRsError::Validation`] if a row is past the end of `block`, if a
    ///   row is repeated, or if a relation block indexing `block` lacks one of
    ///   its declared endpoint columns (or carries it as anything but a 1-D
    ///   `UInt` column) — copying it would leave stale indices.
    /// - [`MolRsError::Validation`] if a row reference of `block` into itself
    ///   points outside the selection.
    ///
    /// Row references follow
    /// [`relation_endpoints`](crate::core::schema::relation_endpoints):
    /// every reference into `block` of the same frame — a relation endpoint,
    /// `members.ibead`, or any column a block's `targets` declares — is
    /// renumbered, a null row references nothing, and a reference into
    /// another section (`/frame/atoms`) or an undeclared handle
    /// (`members.atom`) is copied unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    /// use molrs::op::{F, Idx};
    /// use ndarray::Array1;
    ///
    /// // 4 atoms at x = 0..3, bonded as a chain (0,1), (1,2), (2,3).
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![0.0 as F, 1.0, 2.0, 3.0]).into_dyn(), None).unwrap();
    /// let mut bonds = Block::new();
    /// bonds.insert("atomi", Array1::from_vec(vec![0 as Idx, 1, 2]).into_dyn(), None).unwrap();
    /// bonds.insert("atomj", Array1::from_vec(vec![1 as Idx, 2, 3]).into_dyn(), None).unwrap();
    /// bonds.insert("type_id", Array1::from_vec(vec![10 as Idx, 11, 12]).into_dyn(), None).unwrap();
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", atoms);
    /// frame.insert("bonds", bonds);
    ///
    /// // Old atom 2 becomes row 0 and old atom 1 row 1; only bond (1,2) lies
    /// // inside the selection, and it becomes (1,0).
    /// let one = frame.subset("atoms", &[2, 1]).unwrap();
    ///
    /// let x: Vec<F> = one["atoms"].get("x").and_then(|c| c.get::<f64>()).unwrap().iter().copied().collect();
    /// assert_eq!(x, vec![2.0, 1.0]);
    /// assert_eq!(one["bonds"].n_rows(), Some(1));
    /// assert_eq!(one["bonds"].get("atomi").and_then(|c| c.get::<u64>()).unwrap()[[0]], 1);
    /// assert_eq!(one["bonds"].get("atomj").and_then(|c| c.get::<u64>()).unwrap()[[0]], 0);
    /// assert_eq!(one["bonds"].get("type_id").and_then(|c| c.get::<u64>()).unwrap()[[0]], 11);
    /// ```
    pub fn subset(&self, block: &str, rows: &[usize]) -> Result<Frame, MolRsError> {
        let target = self.get(block).ok_or_else(|| {
            MolRsError::not_found("block", format!("frame has no '{block}' block"))
        })?;
        let nrows = target.n_rows().unwrap_or(0);
        // new_row[old] = Some(new) for a selected row.
        let mut new_row: Vec<Option<usize>> = vec![None; nrows];
        for (k, &r) in rows.iter().enumerate() {
            if r >= nrows {
                return Err(MolRsError::validation(format!(
                    "subset row {r} is past the end of '{block}' ({nrows} rows)"
                )));
            }
            if new_row[r].is_some() {
                return Err(MolRsError::validation(format!(
                    "subset row {r} of '{block}' is selected twice"
                )));
            }
            new_row[r] = Some(k);
        }

        // Retain row sets for every block. Removing a relation row can invalidate
        // references into that relation table, so propagate to a fixed point.
        let mut kept: IndexMap<String, Vec<usize>> = self
            .iter()
            .map(|(name, b)| {
                (
                    name.to_owned(),
                    if name == block {
                        rows.to_vec()
                    } else {
                        (0..b.n_rows().unwrap_or(0)).collect()
                    },
                )
            })
            .collect();
        let references: IndexMap<String, Vec<(String, String)>> = self
            .iter()
            .map(|(name, b)| {
                (
                    name.to_owned(),
                    if b.n_rows().unwrap_or(0) == 0 {
                        Vec::new()
                    } else {
                        local_reference_targets(name, b)
                            .into_iter()
                            .filter(|(col, target)| {
                                self.contains_key(target)
                                    || (b.target(col).is_some()
                                        && b.validity(col)
                                            .is_none_or(|mask| mask.iter().any(|&valid| valid)))
                            })
                            .collect()
                    },
                )
            })
            .collect();
        for (name, refs) in &references {
            let b = &self[name.as_str()];
            for (col, target) in refs {
                if !self.contains_key(target) {
                    return Err(MolRsError::validation(format!(
                        "cannot subset: '{name}.{col}' references absent block '{target}'"
                    )));
                }
                if b.get(col)
                    .and_then(|c| c.get::<u64>())
                    .is_none_or(|v| v.ndim() != 1)
                {
                    return Err(MolRsError::validation(format!(
                        "cannot subset: block '{name}' has no 1-D UInt reference column '{col}'"
                    )));
                }
            }
        }
        loop {
            let maps: IndexMap<String, Vec<Option<usize>>> = self
                .iter()
                .map(|(name, b)| {
                    let mut map = vec![None; b.n_rows().unwrap_or(0)];
                    for (new, &old) in kept[name].iter().enumerate() {
                        map[old] = Some(new);
                    }
                    (name.to_owned(), map)
                })
                .collect();
            let mut changed = false;
            for (name, b) in self.iter() {
                let inside = |row: usize| {
                    references[name].iter().all(|(col, target)| {
                        if b.validity(col).is_some_and(|m| !m[row]) {
                            return true;
                        }
                        let value = b.get(col).unwrap().get::<u64>().unwrap()[[row]];
                        usize::try_from(value)
                            .ok()
                            .and_then(|i| maps[target].get(i))
                            .is_some_and(Option::is_some)
                    })
                };
                if name == block && kept[name].iter().any(|&row| !inside(row)) {
                    return Err(MolRsError::validation(format!(
                        "cannot subset '{block}': a selected row references a row outside the selection"
                    )));
                }
                let selection = kept.get_mut(name).unwrap();
                let before = selection.len();
                selection.retain(|&row| inside(row));
                changed |= before != selection.len();
            }
            if !changed {
                break;
            }
        }
        let maps: IndexMap<String, Vec<Option<usize>>> = self
            .iter()
            .map(|(name, b)| {
                let mut map = vec![None; b.n_rows().unwrap_or(0)];
                for (new, &old) in kept[name].iter().enumerate() {
                    map[old] = Some(new);
                }
                (name.to_owned(), map)
            })
            .collect();
        let mut out = Frame::new();
        out.meta = self.meta.clone();
        out.simbox = self.simbox.clone();
        for (name, b) in self.iter() {
            let selection = &kept[name];
            let unchanged = selection.iter().copied().eq(0..b.n_rows().unwrap_or(0));
            let mut cut = if unchanged {
                b.deep_copy()
            } else {
                b.select_rows(selection)?
            };
            for (col, target) in &references[name] {
                let mask = cut.validity(col).map(<[bool]>::to_vec);
                let values = cut.get_mut(col).unwrap().get_mut::<u64>().unwrap();
                for (row, value) in values.iter_mut().enumerate() {
                    if mask.as_ref().is_some_and(|m| !m[row]) {
                        continue;
                    }
                    *value = maps[target][*value as usize]
                        .expect("retained references have retained targets")
                        as u64;
                }
            }
            out.insert(name, cut);
        }
        Ok(out)
    }

    /// A new frame holding `count` copies of this one, concatenated block by
    /// block — the inverse of [`subset`](Self::subset) for a frame that is
    /// `count` identical molecules.
    ///
    /// - Every block becomes its rows repeated `count` times, copy after copy
    ///   (copy `c` of row `r` lands at `c * nrows + r`). Every column and its
    ///   validity mask travel; a structural N-D shape does not (the result is
    ///   a row table, as with [`Block::merge`]).
    /// - A relation block (canonical `bonds`, `angles`, …, or any block
    ///   carrying `atomi`..`atoml`, per
    ///   [`relation_endpoints`](crate::core::schema::relation_endpoints))
    ///   has each endpoint of copy `c` offset by `c` times the row count of
    ///   the block it indexes, so copy `c`'s bonds join copy `c`'s atoms.
    /// - Every other column is copied verbatim — **including identifier
    ///   columns** (`id`, `mol_id`, `type_id`): the copies carry the same
    ///   labels. Regenerate them if the copies need distinct ones.
    /// - `meta` and the box are cloned unchanged. The result shares no buffer
    ///   with this frame.
    ///
    /// `count == 0` gives zero-row blocks.
    ///
    /// # Errors
    ///
    /// The frame itself is never modified.
    ///
    /// - [`MolRsError::NotFound`] if a relation block's target block (`atoms`)
    ///   is absent, so there is no row count to offset by.
    /// - [`MolRsError::Validation`] if a non-empty relation block lacks one of
    ///   its declared endpoint columns, or carries it as anything but a 1-D
    ///   `UInt` column — copying it would leave every copy pointing at the
    ///   first.
    ///
    /// Every row reference into a block of the same frame (relation
    /// endpoints, `members.ibead`, declared `targets`) is offset; a reference
    /// into another section and an undeclared handle are copied unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    /// use molrs::op::{F, Idx};
    /// use ndarray::Array1;
    ///
    /// // A diatomic: atoms 0-1 bonded.
    /// let mut atoms = Block::new();
    /// atoms.insert("x", Array1::from_vec(vec![0.0 as F, 1.0]).into_dyn(), None).unwrap();
    /// let mut bonds = Block::new();
    /// bonds.insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None).unwrap();
    /// bonds.insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None).unwrap();
    /// let mut frame = Frame::new();
    /// frame.insert("atoms", atoms);
    /// frame.insert("bonds", bonds);
    ///
    /// let three = frame.replicate(3).unwrap();
    /// assert_eq!(three["atoms"].n_rows(), Some(6));
    /// let i: Vec<Idx> = three["bonds"].get("atomi").and_then(|c| c.get::<u64>()).unwrap().iter().copied().collect();
    /// let j: Vec<Idx> = three["bonds"].get("atomj").and_then(|c| c.get::<u64>()).unwrap().iter().copied().collect();
    /// assert_eq!((i, j), (vec![0, 2, 4], vec![1, 3, 5]));
    /// ```
    pub fn replicate(&self, count: usize) -> Result<Frame, MolRsError> {
        let mut out = Frame::new();
        out.meta = self.meta.clone();
        out.simbox = self.simbox.clone();
        for (name, b) in self.iter() {
            let rows = b.n_rows().unwrap_or(0);
            let tile: Vec<usize> = (0..count).flat_map(|_| 0..rows).collect();
            let mut tiled = b.select_rows(&tile)?;
            if b.is_empty() {
                // No column to gather: carry the declared row count alone.
                tiled.resize(rows * count)?;
            }
            let references = if rows > 0 {
                local_reference_targets(name, b)
            } else {
                Vec::new()
            };
            for (col, target) in &references {
                let span = self
                    .get(target)
                    .ok_or_else(|| {
                        MolRsError::not_found(
                            "block",
                            format!(
                                "cannot replicate '{name}': it indexes '{target}', which \
                                 the frame lacks"
                            ),
                        )
                    })?
                    .n_rows()
                    .unwrap_or(0);
                let values = tiled
                    .get_mut(col)
                    .and_then(|c| c.get_mut::<u64>())
                    .filter(|v| v.ndim() == 1)
                    .ok_or_else(|| {
                        MolRsError::validation(format!(
                            "cannot replicate: block '{name}' has no 1-D UInt reference \
                             column '{col}'"
                        ))
                    })?;
                for (i, v) in values.iter_mut().enumerate() {
                    *v += ((i / rows) * span) as crate::op::Idx;
                }
            }
            out.insert(name, tiled);
        }
        Ok(out)
    }

    /// The frames joined end to end, block by block — [`replicate`](Self::replicate)
    /// for parts that differ.
    ///
    /// - Every block name any part has becomes one block holding the parts'
    ///   rows in order, stacked with [`Block::stack`]: columns in first-seen
    ///   order, and a column one part lacks is filled for its rows and marked
    ///   null.
    /// - Every row reference into a block of the same frame (relation
    ///   endpoints `atomi`..`atoml`, `members.ibead`, declared `targets`) in
    ///   part `p` is offset by the rows the referenced block has in parts
    ///   `0..p`, so part `p`'s bonds join part `p`'s atoms. A reference into
    ///   another section and an undeclared handle are copied unchanged.
    /// - Every other column is copied verbatim — **including identifier
    ///   columns** (`id`, `mol_id`, `type_id`); regenerate them if the parts
    ///   need distinct labels.
    /// - `meta` and the box are the first part's. The result shares no buffer
    ///   with any part.
    ///
    /// No parts give an empty frame.
    ///
    /// # Errors
    ///
    /// No part is ever modified.
    ///
    /// - [`MolRsError::NotFound`] if a part's relation block indexes a block
    ///   that part lacks.
    /// - [`MolRsError::Validation`] if a non-empty relation block carries an
    ///   endpoint as anything but a 1-D `UInt` column.
    /// - [`MolRsError::Block`] when two parts carry one column under
    ///   different dtypes or per-row shapes (see [`Block::stack`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    /// use molrs::op::{F, Idx};
    /// use ndarray::Array1;
    ///
    /// fn chain(n: usize) -> Frame {
    ///     let mut atoms = Block::new();
    ///     atoms.insert("x", Array1::from_vec(vec![0.0 as F; n]).into_dyn(), None).unwrap();
    ///     let mut bonds = Block::new();
    ///     bonds.insert("atomi", Array1::from_iter(0..n as Idx - 1).into_dyn(), None).unwrap();
    ///     bonds.insert("atomj", Array1::from_iter(1..n as Idx).into_dyn(), None).unwrap();
    ///     let mut frame = Frame::new();
    ///     frame.insert("atoms", atoms);
    ///     frame.insert("bonds", bonds);
    ///     frame
    /// }
    ///
    /// // A diatomic then a triatomic: the second part's bonds start at atom 2.
    /// let joined = Frame::concat([&chain(2), &chain(3)]).unwrap();
    /// assert_eq!(joined["atoms"].n_rows(), Some(5));
    /// let i: Vec<Idx> = joined["bonds"].get("atomi").and_then(|c| c.get::<u64>()).unwrap().iter().copied().collect();
    /// assert_eq!(i, vec![0, 2, 3]);
    /// ```
    pub fn concat<'a>(frames: impl IntoIterator<Item = &'a Frame>) -> Result<Frame, MolRsError> {
        let frames: Vec<&Frame> = frames.into_iter().collect();
        let Some(first) = frames.first() else {
            return Ok(Frame::new());
        };
        // Rows each block name holds in the parts before the current one.
        let mut seen_rows: IndexMap<String, usize> = IndexMap::new();
        let mut parts: IndexMap<String, Vec<Block>> = IndexMap::new();
        for frame in &frames {
            for (name, b) in frame.iter() {
                let mut part = b.clone();
                for (col, target) in local_reference_targets(name, b) {
                    if part.contains_key(&col) {
                        part.set_target(&col, &target)?;
                    }
                }
                let rows = b.n_rows().unwrap_or(0);
                let references = if rows > 0 {
                    local_reference_targets(name, b)
                } else {
                    Vec::new()
                };
                for (col, target) in &references {
                    part.set_target(col, target)?;
                    if frame.get(target).is_none() {
                        return Err(MolRsError::not_found(
                            "block",
                            format!(
                                "cannot concat '{name}': it indexes '{target}', which its \
                                 frame lacks"
                            ),
                        ));
                    }
                    let base = seen_rows.get(target).copied().unwrap_or(0);
                    let values = part
                        .get_mut(col)
                        .and_then(|c| c.get_mut::<u64>())
                        .filter(|v| v.ndim() == 1)
                        .ok_or_else(|| {
                            MolRsError::validation(format!(
                                "cannot concat: block '{name}' has no 1-D UInt reference \
                                 column '{col}'"
                            ))
                        })?;
                    if base > 0 {
                        values.mapv_inplace(|v| v + base as crate::op::Idx);
                    }
                }
                parts.entry(name.to_owned()).or_default().push(part);
            }
            for (name, b) in frame.iter() {
                *seen_rows.entry(name.to_owned()).or_default() += b.n_rows().unwrap_or(0);
            }
        }
        let mut out = Frame::new();
        out.meta = first.meta.clone();
        out.simbox = first.simbox.clone();
        for (name, blocks) in &parts {
            out.insert(name, Block::stack(blocks)?);
        }
        Ok(out)
    }

    /// The `atoms` block's positions as one `N × 3` array — see
    /// [`Block::coords`].
    ///
    /// # Errors
    ///
    /// [`MolRsError::NotFound`] without an `atoms` block, and
    /// [`MolRsError::Block`] ([`BlockError::MissingColumn`](crate::core::BlockError::MissingColumn))
    /// when it lacks `x`, `y` or `z`.
    pub fn coords(&self) -> Result<crate::op::Fnx3, MolRsError> {
        let atoms = self.get(ATOMS).ok_or_else(|| {
            MolRsError::not_found("block", format!("frame has no '{ATOMS}' block"))
        })?;
        Ok(atoms.coords()?)
    }

    /// Write an `N × 3` array into the `atoms` block's `x` / `y` / `z` — see
    /// [`Block::set_coords`]. A frame without an `atoms` block gets one
    /// holding just the coordinates.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Block`] when `coords` is not `N × 3` or `N` differs from
    /// the `atoms` row count. The frame is unchanged on error.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::op::F;
    /// use ndarray::array;
    ///
    /// let mut frame = Frame::new();
    /// frame.set_coords(array![[1.0 as F, 2.0, 3.0]].view()).unwrap();
    /// assert_eq!(frame.coords().unwrap(), array![[1.0, 2.0, 3.0]]);
    /// ```
    pub fn set_coords(&mut self, coords: crate::op::Fnx3View<'_>) -> Result<(), MolRsError> {
        match self.get_mut(ATOMS) {
            Some(atoms) => atoms.set_coords(coords)?,
            None => {
                let mut atoms = Block::new();
                atoms.set_coords(coords)?;
                self.insert(ATOMS, atoms);
            }
        }
        Ok(())
    }

    /// Checks if the frame is consistent without returning an error.
    ///
    /// This is a non-panicking version of `validate()` that returns a boolean.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::Frame;
    /// use molrs::core::Block;
    ///
    /// let frame = Frame::new();
    /// assert!(frame.is_consistent());
    /// ```
    pub fn is_consistent(&self) -> bool {
        self.validate().is_ok()
    }
}

/// The row references of block `name` into blocks of the same frame, as
/// `(column, target)` — declared `targets` honoured, absolute targets and
/// undeclared handles left out.
fn local_reference_targets(name: &str, block: &Block) -> Vec<(String, String)> {
    let declared: Vec<(&str, &str)> = block.targets().collect();
    crate::core::schema::relation_endpoints(name, |k| block.contains_key(k), &declared)
        .into_iter()
        .filter(|r| r.is_local())
        .map(|r| (r.column, r.target))
        .collect()
}

/// Builds a Frame from `(name, block)` pairs, in their iteration order, with
/// no metadata and no box.
///
/// # Examples
///
/// ```
/// use molrs::core::{Block, Frame};
///
/// let frame: Frame = [("atoms", Block::new()), ("bonds", Block::new())]
///     .into_iter()
///     .collect();
/// assert_eq!(frame.keys().collect::<Vec<_>>(), ["atoms", "bonds"]);
/// ```
impl<K: Into<String>> FromIterator<(K, Block)> for Frame {
    fn from_iter<I: IntoIterator<Item = (K, Block)>>(iter: I) -> Self {
        Self {
            map: iter.into_iter().map(|(k, b)| (k.into(), b)).collect(),
            meta: MetaMap::new(),
            simbox: None,
        }
    }
}

// Index trait for convenient access: frame["atoms"]
impl Index<&str> for Frame {
    type Output = Block;

    fn index(&self, key: &str) -> &Self::Output {
        self.get(key)
            .unwrap_or_else(|| panic!("Frame does not contain block '{}'", key))
    }
}

impl IndexMut<&str> for Frame {
    fn index_mut(&mut self, key: &str) -> &mut Self::Output {
        self.get_mut(key)
            .unwrap_or_else(|| panic!("Frame does not contain block '{}'", key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use crate::op::{F, I};
    use ndarray::Array1;

    #[test]
    fn keys_follow_insertion_order_and_survive_remove_and_rename() {
        let mut frame = Frame::new();
        for key in ["c", "a", "b", "d"] {
            frame.insert(key, Block::new());
        }
        assert_eq!(frame.keys().collect::<Vec<_>>(), ["c", "a", "b", "d"]);
        frame.remove("a");
        assert_eq!(frame.keys().collect::<Vec<_>>(), ["c", "b", "d"]);
        assert!(frame.rename_block("b", "y"));
        assert_eq!(frame.keys().collect::<Vec<_>>(), ["c", "y", "d"]);
    }

    /// A subset keeps the frame's block order: selecting from `atoms` does
    /// not move it in front of a block inserted before it.
    #[test]
    fn subset_keeps_the_block_order() {
        let mut frame = Frame::new();
        frame.insert("cell", Block::new());
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 1.0]).into_dyn(), None)
            .unwrap();
        frame.insert("atoms", atoms);
        frame.insert("tail", Block::new());

        let sub = frame.subset("atoms", &[1]).unwrap();
        assert_eq!(sub.keys().collect::<Vec<_>>(), ["cell", "atoms", "tail"]);
    }

    /// A pass-through block (one that does not index the selected block) is a
    /// new buffer too: a subset shares nothing with its source.
    #[test]
    fn subset_shares_no_buffer_with_the_source() {
        let mut cell = Block::new();
        cell.insert("lx", Array1::from_vec(vec![10.0 as F]).into_dyn(), None)
            .unwrap();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 1.0]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("cell", cell);
        frame.insert("atoms", atoms);

        let sub = frame.subset("atoms", &[1]).unwrap();

        let src = frame["cell"]
            .get("lx")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        let dst = sub["cell"].get("lx").and_then(|c| c.get::<f64>()).unwrap();
        assert_ne!(src.as_ptr(), dst.as_ptr());
        assert_eq!(dst[[0]], 10.0);
    }

    #[test]
    fn deep_copy_gives_every_block_new_buffers() {
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.meta.insert("step", crate::core::MetaValue::I64(3));

        let copy = frame.deep_copy();

        let src = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        let dst = copy
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert_ne!(src.as_ptr(), dst.as_ptr());
        assert_eq!(dst.as_slice_memory_order(), Some(&[1.0 as F, 2.0][..]));
        assert_eq!(copy.meta.get("step"), frame.meta.get("step"));
    }

    #[test]
    fn test_frame_new() {
        let frame = Frame::new();
        assert!(frame.is_empty());
        assert_eq!(frame.len(), 0);
    }

    #[test]
    fn test_frame_insert_get() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();

        frame.insert("atoms", block);
        assert_eq!(frame.len(), 1);
        assert!(frame.contains_key("atoms"));

        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(2));
    }

    #[test]
    fn test_frame_index_access() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();
        frame.insert("atoms", block);

        // Immutable index
        let atoms = &frame["atoms"];
        assert_eq!(atoms.n_rows(), Some(1));

        // Mutable index
        let atoms_mut = &mut frame["atoms"];
        if let Some(x) = atoms_mut.get_mut("x").and_then(|c| c.get_mut::<f64>()) {
            x[[0]] = 99.0;
        }
        assert_eq!(
            frame["atoms"]
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()[[0]],
            99.0
        );
    }

    #[test]
    #[should_panic(expected = "Frame does not contain block 'missing'")]
    fn test_frame_index_panic() {
        let frame = Frame::new();
        let _ = &frame["missing"];
    }

    #[test]
    fn test_frame_iter() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.insert("bonds", Block::new());

        let keys: Vec<&str> = frame.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"atoms"));
        assert!(keys.contains(&"bonds"));

        let mut count = 0;
        for (_name, _block) in frame.iter() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_frame_iter_mut() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();
        frame.insert("atoms", block);

        for (_name, block) in frame.iter_mut() {
            if let Some(x) = block.get_mut("x").and_then(|c| c.get_mut::<f64>()) {
                x[[0]] = 42.0;
            }
        }

        assert_eq!(
            frame["atoms"]
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()[[0]],
            42.0
        );
    }

    #[test]
    fn test_frame_values_mut() {
        let mut frame = Frame::new();
        let mut block = Block::new();
        block
            .insert("x", Array1::from_vec(vec![1.0 as F]).into_dyn(), None)
            .unwrap();
        frame.insert("atoms", block);

        for block in frame.values_mut() {
            if let Some(x) = block.get_mut("x").and_then(|c| c.get_mut::<f64>()) {
                x[[0]] = 77.0;
            }
        }

        assert_eq!(
            frame["atoms"]
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()[[0]],
            77.0
        );
    }

    #[test]
    fn frame_collects_from_an_index_map() {
        let mut map = IndexMap::new();
        map.insert("atoms".to_string(), Block::new());
        map.insert("bonds".to_string(), Block::new());

        let frame: Frame = map.into_iter().collect();
        assert_eq!(frame.len(), 2);
        assert!(frame.contains_key("atoms"));
        assert!(frame.contains_key("bonds"));
    }

    #[test]
    fn frame_collects_from_any_iterator_of_named_blocks() {
        let map = std::collections::HashMap::from([("atoms".to_string(), Block::new())]);
        assert_eq!(map.into_iter().collect::<Frame>().len(), 1);
        let pairs = vec![("b", Block::new()), ("a", Block::new())];
        assert_eq!(
            pairs
                .into_iter()
                .collect::<Frame>()
                .keys()
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
    }

    #[test]
    fn test_frame_into_inner() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title", "Test");

        let (blocks, meta, simbox) = frame.into_inner();
        assert_eq!(blocks.len(), 1);
        assert!(blocks.contains_key("atoms"));
        assert_eq!(meta.get("title").unwrap().as_str(), Some("Test"));
        assert!(simbox.is_none());
    }

    #[test]
    fn test_frame_clear_preserves_meta() {
        let mut frame = Frame::new();
        frame.insert("atoms", Block::new());
        frame.meta.insert("title", "Test");

        frame.clear();
        assert!(frame.is_empty());
        assert!(!frame.meta.is_empty());
        assert_eq!(frame.meta.get("title").unwrap().as_str(), Some("Test"));
    }

    #[test]
    fn test_frame_debug() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F, 3.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);
        frame.meta.insert("title", "Test");

        let debug_str = format!("{:?}", frame);
        assert!(debug_str.contains("Frame"));
        assert!(debug_str.contains("atoms"));
        assert!(debug_str.contains("title"));
    }

    #[test]
    fn test_rename_block() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![1.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![3.0 as F, 4.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        // Successful rename
        assert!(frame.rename_block("atoms", "molecules"));
        assert!(!frame.contains_key("atoms"));
        assert!(frame.contains_key("molecules"));
        assert_eq!(
            frame["molecules"]
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_slice_memory_order()
                .unwrap(),
            &[1.0, 2.0]
        );

        // Try to rename non-existent block
        assert!(!frame.rename_block("nonexistent", "new_block"));

        // Try to rename to existing block name
        let mut bonds = Block::new();
        bonds
            .insert("count", Array1::from_vec(vec![1 as I]).into_dyn(), None)
            .unwrap();
        frame.insert("bonds", bonds);
        assert!(!frame.rename_block("molecules", "bonds"));
    }

    // ---- subset ----

    use crate::core::SimBox;
    use crate::op::Idx;

    fn float_col(values: &[F]) -> ndarray::ArrayD<F> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn uint_col(values: &[Idx]) -> ndarray::ArrayD<Idx> {
        Array1::from_vec(values.to_vec()).into_dyn()
    }

    fn uint_block(cols: &[(&str, &[Idx])]) -> Block {
        let mut b = Block::new();
        for (key, values) in cols {
            b.insert(*key, uint_col(values), None).unwrap();
        }
        b
    }

    /// 4 atoms at x = 0..3, and the chain bonds (0,1), (1,2), (2,3) with
    /// `type_id` 10, 11, 12.
    fn chain_of_four() -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert("x", float_col(&[0.0, 1.0, 2.0, 3.0]), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert(
            "bonds",
            uint_block(&[
                ("atomi", &[0, 1, 2]),
                ("atomj", &[1, 2, 3]),
                ("type_id", &[10, 11, 12]),
            ]),
        );
        frame
    }

    fn uint_values(frame: &Frame, block: &str, col: &str) -> Vec<Idx> {
        frame[block]
            .get(col)
            .and_then(|c| c.get::<u64>())
            .unwrap_or_else(|| panic!("{block}.{col} is a UInt column"))
            .iter()
            .copied()
            .collect()
    }

    fn float_values(frame: &Frame, block: &str, col: &str) -> Vec<F> {
        frame[block]
            .get(col)
            .and_then(|c| c.get::<f64>())
            .unwrap_or_else(|| panic!("{block}.{col} is a Float column"))
            .iter()
            .copied()
            .collect()
    }

    #[test]
    fn subset_gathers_atoms_in_order_and_reindexes_the_bonds_inside() {
        // rows [2, 1]: old 2 -> new 0, old 1 -> new 1. Only bond (1,2) lies
        // inside; it becomes (1,0) and keeps type_id 11.
        let out = chain_of_four().subset("atoms", &[2, 1]).unwrap();

        assert_eq!(float_values(&out, "atoms", "x"), vec![2.0, 1.0]);
        assert_eq!(out["bonds"].n_rows(), Some(1));
        assert_eq!(uint_values(&out, "bonds", "atomi"), vec![1]);
        assert_eq!(uint_values(&out, "bonds", "atomj"), vec![0]);
        assert_eq!(uint_values(&out, "bonds", "type_id"), vec![11]);
    }

    #[test]
    fn subset_keeps_only_the_angles_inside_the_selection() {
        // rows [1, 2, 3]: angle (0,1,2) touches atom 0 and is dropped;
        // angle (1,2,3) becomes (0,1,2).
        let mut frame = chain_of_four();
        frame.insert(
            "angles",
            uint_block(&[("atomi", &[0, 1]), ("atomj", &[1, 2]), ("atomk", &[2, 3])]),
        );

        let out = frame.subset("atoms", &[1, 2, 3]).unwrap();

        assert_eq!(out["angles"].n_rows(), Some(1));
        assert_eq!(uint_values(&out, "angles", "atomi"), vec![0]);
        assert_eq!(uint_values(&out, "angles", "atomj"), vec![1]);
        assert_eq!(uint_values(&out, "angles", "atomk"), vec![2]);
    }

    #[test]
    fn subset_reindexes_a_relation_block_without_a_spec() {
        // `ports` has no BlockSpec; its atomi/atomj still index atoms.
        // rows [2, 3]: port (3,2) becomes (1,0); port (0,3) leaves the
        // selection and is dropped.
        let mut frame = chain_of_four();
        frame.insert(
            "ports",
            uint_block(&[("atomi", &[3, 0]), ("atomj", &[2, 3])]),
        );

        let out = frame.subset("atoms", &[2, 3]).unwrap();

        assert_eq!(out["ports"].n_rows(), Some(1));
        assert_eq!(uint_values(&out, "ports", "atomi"), vec![1]);
        assert_eq!(uint_values(&out, "ports", "atomj"), vec![0]);
    }

    #[test]
    fn subset_copies_the_box_meta_and_a_non_relation_block_unchanged() {
        let mut frame = chain_of_four();
        let mut labels = Block::new();
        labels
            .insert("weight", float_col(&[0.5, 1.5, 2.5]), None)
            .unwrap();
        frame.insert("labels", labels);
        frame.simbox = Some(
            SimBox::new(
                Cell::Lengths([10.0 as F, 20.0, 30.0]),
                [1.0 as F, 2.0, 3.0],
                [true, false, true],
            )
            .unwrap(),
        );
        frame.meta.insert("title", "melt");

        let out = frame.subset("atoms", &[0, 1]).unwrap();

        assert_eq!(out.meta, frame.meta);
        let (before, after) = (frame.simbox.as_ref().unwrap(), out.simbox.as_ref().unwrap());
        assert_eq!(after.matrix(), before.matrix());
        assert_eq!(after.origin(), before.origin());
        assert_eq!(after.pbc(), before.pbc());
        assert_eq!(out["labels"].n_rows(), Some(3));
        assert_eq!(float_values(&out, "labels", "weight"), vec![0.5, 1.5, 2.5]);
    }

    #[test]
    fn subset_renumbers_members_ibead_and_leaves_the_atom_handle() {
        // `members.ibead` references `atoms`; `atom` (undeclared, or into
        // another section) is copied unchanged.
        let mut frame = chain_of_four();
        let mut members = uint_block(&[("ibead", &[0, 3, 2]), ("atom", &[10, 11, 12])]);
        members.set_target("atom", "/frame/atoms").unwrap();
        frame.insert("members", members);

        let out = frame.subset("atoms", &[3, 2]).unwrap();
        assert_eq!(uint_values(&out, "members", "ibead"), vec![0, 1]);
        assert_eq!(uint_values(&out, "members", "atom"), vec![11, 12]);
        assert_eq!(out["members"].target("atom"), Some("/frame/atoms"));
    }

    #[test]
    fn subset_follows_a_declared_target_and_skips_null_references() {
        let mut frame = chain_of_four();
        let mut refs = uint_block(&[("site", &[1, 3, 0])]);
        refs.set_target("site", "atoms").unwrap();
        refs.set_validity("site", vec![true, true, false]).unwrap();
        frame.insert("refs", refs);
        let out = frame.subset("atoms", &[3, 1]).unwrap();
        // Row 0 (site 1 -> 1), row 1 (site 3 -> 0), row 2 null: kept as is.
        assert_eq!(uint_values(&out, "refs", "site"), vec![1, 0, 0]);
        assert_eq!(out["refs"].validity("site"), Some(&[true, true, false][..]));
    }

    #[test]
    fn subset_of_bonds_selects_bond_rows_and_leaves_the_atoms_unchanged() {
        // No block's endpoints index `bonds`, so nothing is renumbered.
        let out = chain_of_four().subset("bonds", &[1]).unwrap();

        assert_eq!(uint_values(&out, "bonds", "atomi"), vec![1]);
        assert_eq!(uint_values(&out, "bonds", "atomj"), vec![2]);
        assert_eq!(uint_values(&out, "bonds", "type_id"), vec![11]);
        assert_eq!(float_values(&out, "atoms", "x"), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn subset_with_no_rows_gives_zero_row_atoms_and_bonds() {
        let out = chain_of_four().subset("atoms", &[]).unwrap();

        assert_eq!(out["atoms"].n_rows(), Some(0));
        assert_eq!(out["bonds"].n_rows(), Some(0));
    }

    #[test]
    fn subset_refuses_a_missing_block() {
        let err = chain_of_four()
            .subset("residues", &[0])
            .expect_err("there is no residues block");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn subset_refuses_a_row_past_the_block() {
        let err = chain_of_four()
            .subset("atoms", &[4])
            .expect_err("4 atoms are rows 0..=3");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn subset_refuses_a_repeated_row() {
        let err = chain_of_four()
            .subset("atoms", &[1, 1])
            .expect_err("row 1 is selected twice");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn subset_refuses_a_bond_block_missing_an_endpoint_column() {
        let mut frame = chain_of_four();
        frame.insert("bonds", uint_block(&[("atomi", &[0, 1])]));

        let err = frame
            .subset("atoms", &[0, 1])
            .expect_err("bonds without atomj cannot be reindexed");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert!(err.to_string().contains("atomj"), "{err}");
    }

    // ---- replicate ----

    #[test]
    fn replicate_offsets_endpoints_by_the_target_row_count_per_copy() {
        let two = chain_of_four().replicate(2).unwrap();
        assert_eq!(two["atoms"].n_rows(), Some(8));
        assert_eq!(uint_values(&two, "bonds", "atomi"), [0, 1, 2, 4, 5, 6]);
        assert_eq!(uint_values(&two, "bonds", "atomj"), [1, 2, 3, 5, 6, 7]);
        // Non-endpoint columns (identifiers included) are copied verbatim.
        assert_eq!(
            uint_values(&two, "bonds", "type_id"),
            [10, 11, 12, 10, 11, 12]
        );
    }

    /// Six atoms at x = 0..5 and the two cmaps (0,1,2,3,4) and (1,2,3,4,5).
    fn two_cmaps() -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert("x", float_col(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert(
            "cmaps",
            uint_block(&[
                ("atomi", &[0, 1]),
                ("atomj", &[1, 2]),
                ("atomk", &[2, 3]),
                ("atoml", &[3, 4]),
                ("atomm", &[4, 5]),
                ("type_id", &[7, 8]),
            ]),
        );
        frame
    }

    #[test]
    fn subset_renumbers_a_cmap_through_atomm() {
        // rows [1..=5]: cmap 0 touches atom 0 and is dropped; cmap 1 becomes
        // (0,1,2,3,4) and keeps type_id 8.
        let out = two_cmaps().subset("atoms", &[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(out["cmaps"].n_rows(), Some(1));
        for (key, want) in [
            ("atomi", 0),
            ("atomj", 1),
            ("atomk", 2),
            ("atoml", 3),
            ("atomm", 4),
        ] {
            assert_eq!(uint_values(&out, "cmaps", key), [want], "{key}");
        }
        assert_eq!(uint_values(&out, "cmaps", "type_id"), [8]);

        // atomm alone leaving the selection drops the row.
        let out = two_cmaps().subset("atoms", &[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(uint_values(&out, "cmaps", "atomm"), [4]);
        assert_eq!(uint_values(&out, "cmaps", "type_id"), [7]);
    }

    #[test]
    fn replicate_offsets_every_cmap_endpoint() {
        let two = two_cmaps().replicate(2).unwrap();
        assert_eq!(uint_values(&two, "cmaps", "atomi"), [0, 1, 6, 7]);
        assert_eq!(uint_values(&two, "cmaps", "atomm"), [4, 5, 10, 11]);
        assert_eq!(uint_values(&two, "cmaps", "type_id"), [7, 8, 7, 8]);
    }

    #[test]
    fn subset_refuses_a_cmaps_block_missing_atomm() {
        let mut frame = two_cmaps();
        frame.get_mut("cmaps").unwrap().remove("atomm");
        assert!(frame.subset("atoms", &[0, 1]).is_err());
    }

    #[test]
    fn replicate_offsets_an_unspecified_relation_block_too() {
        let mut frame = chain_of_four();
        frame.insert("ports", uint_block(&[("atomi", &[3])]));
        let three = frame.replicate(3).unwrap();
        assert_eq!(uint_values(&three, "ports", "atomi"), [3, 7, 11]);
    }

    #[test]
    fn replicate_tiles_validity_masks_with_their_rows() {
        let mut frame = chain_of_four();
        frame
            .get_mut("atoms")
            .unwrap()
            .set_validity("x", vec![true, false, true, true])
            .unwrap();
        let two = frame.replicate(2).unwrap();
        assert_eq!(
            two["atoms"].validity("x"),
            Some(&[true, false, true, true, true, false, true, true][..])
        );
    }

    #[test]
    fn replicate_zero_times_gives_zero_row_blocks() {
        let none = chain_of_four().replicate(0).unwrap();
        assert_eq!(none["atoms"].n_rows(), Some(0));
        assert_eq!(none["bonds"].n_rows(), Some(0));
    }

    #[test]
    fn replicate_carries_the_row_count_of_a_columnless_block() {
        let mut frame = chain_of_four();
        let mut marker = Block::new();
        marker.resize(2).unwrap();
        frame.insert("marker", marker);
        assert_eq!(frame.replicate(3).unwrap()["marker"].n_rows(), Some(6));
    }

    #[test]
    fn replicate_refuses_a_relation_without_its_target_block() {
        let mut frame = chain_of_four();
        frame.remove("atoms");
        let err = frame.replicate(2).expect_err("bonds index a missing block");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
    }

    #[test]
    fn replicate_refuses_a_relation_missing_an_endpoint_column() {
        let mut frame = chain_of_four();
        frame.insert("bonds", uint_block(&[("atomi", &[0])]));
        let err = frame.replicate(2).expect_err("bonds without atomj");
        assert!(err.to_string().contains("atomj"), "{err}");
    }

    #[test]
    fn replicate_offsets_members_ibead_and_leaves_an_absolute_target() {
        let mut frame = chain_of_four();
        let mut members = uint_block(&[("ibead", &[0]), ("atom", &[7])]);
        members.set_target("atom", "/frame/atoms").unwrap();
        frame.insert("members", members);
        let two = frame.replicate(2).unwrap();
        assert_eq!(uint_values(&two, "members", "ibead"), vec![0, 4]);
        assert_eq!(uint_values(&two, "members", "atom"), vec![7, 7]);
    }

    #[test]
    fn replicate_shares_no_buffer_with_the_source() {
        let frame = chain_of_four();
        let one = frame.replicate(1).unwrap();
        assert_ne!(
            frame["atoms"]
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_ptr(),
            one["atoms"]
                .get("x")
                .and_then(|c| c.get::<f64>())
                .unwrap()
                .as_ptr()
        );
    }

    // ---- concat ----

    #[test]
    fn concat_of_copies_equals_replicate() {
        let frame = chain_of_four();
        let joined = Frame::concat([&frame, &frame, &frame]).unwrap();
        let tiled = frame.replicate(3).unwrap();
        for (block, col) in [("bonds", "atomi"), ("bonds", "atomj"), ("bonds", "type_id")] {
            assert_eq!(
                uint_values(&joined, block, col),
                uint_values(&tiled, block, col),
                "{block}.{col}"
            );
        }
        assert_eq!(
            float_values(&joined, "atoms", "x"),
            float_values(&tiled, "atoms", "x")
        );
    }

    #[test]
    fn concat_offsets_past_a_part_without_the_relation_block() {
        // atoms only (2 rows), then a bonded chain: its bonds start at row 2.
        let mut lone = Frame::new();
        let mut atoms = Block::new();
        atoms.insert("x", float_col(&[9.0, 9.0]), None).unwrap();
        atoms
            .insert("charge", float_col(&[0.5, -0.5]), None)
            .unwrap();
        lone.insert("atoms", atoms);
        let joined = Frame::concat([&lone, &chain_of_four()]).unwrap();
        assert_eq!(joined["atoms"].n_rows(), Some(6));
        assert_eq!(uint_values(&joined, "bonds", "atomi"), [2, 3, 4]);
        // `charge` only the first part had: null on the chain's rows.
        assert_eq!(
            joined["atoms"].validity("charge"),
            Some(&[true, true, false, false, false, false][..])
        );
    }

    #[test]
    fn concat_keeps_the_first_parts_meta_and_box() {
        let mut a = chain_of_four();
        a.meta.insert("title", "first".to_string());
        let mut b = chain_of_four();
        b.meta.insert("title", "second".to_string());
        let joined = Frame::concat([&a, &b]).unwrap();
        assert_eq!(
            joined.meta.get("title").and_then(|v| v.as_str()),
            Some("first")
        );
    }

    #[test]
    fn concat_of_nothing_is_an_empty_frame() {
        assert!(Frame::concat(std::iter::empty()).unwrap().is_empty());
    }

    #[test]
    fn concat_refuses_a_relation_without_its_target_block() {
        let mut orphan = chain_of_four();
        orphan.remove("atoms");
        let err = Frame::concat([&chain_of_four(), &orphan]).expect_err("orphan bonds");
        assert!(matches!(err, MolRsError::NotFound { .. }), "{err:?}");
    }

    // ---- coords ----

    #[test]
    fn coords_round_trip_through_the_atoms_block() {
        let mut frame = chain_of_four();
        let xyz = ndarray::array![
            [0.0 as F, 1.0, 2.0],
            [3.0, 4.0, 5.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0]
        ];
        frame.set_coords(xyz.view()).unwrap();
        assert_eq!(frame.coords().unwrap(), xyz);
        // `x` kept its place in front of the appended `y` / `z`.
        assert_eq!(frame["atoms"].keys().collect::<Vec<_>>(), ["x", "y", "z"]);
    }

    #[test]
    fn set_coords_creates_an_atoms_block_when_there_is_none() {
        let mut frame = Frame::new();
        frame
            .set_coords(ndarray::array![[1.0 as F, 2.0, 3.0]].view())
            .unwrap();
        assert_eq!(frame["atoms"].n_rows(), Some(1));
    }

    #[test]
    fn set_coords_with_the_wrong_row_count_leaves_the_frame_unchanged() {
        let mut frame = chain_of_four();
        let err = frame
            .set_coords(ndarray::array![[1.0 as F, 2.0, 3.0]].view())
            .expect_err("4 atoms, 1 row");
        assert!(
            matches!(
                err,
                MolRsError::Block(crate::core::BlockError::RaggedAxis0 { .. })
            ),
            "{err:?}"
        );
        assert_eq!(frame["atoms"].keys().collect::<Vec<_>>(), ["x"]);
    }

    #[test]
    fn coords_without_an_atoms_block_is_not_found() {
        assert!(matches!(
            Frame::new().coords(),
            Err(MolRsError::NotFound { .. })
        ));
    }
}
