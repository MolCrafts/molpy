//! Aligned columnar component store for the ECS world: [`EntityTable`].

use indexmap::IndexMap;

use slotmap::{Key, SlotMap};

use crate::core::MolRsError;
use crate::op::{F, I};

/// Per-column validity mask — one flag per row (`true` ⇒ the row holds a value).
///
/// Backed by `Vec<bool>` (one byte per row) for a directly mappable mask; a
/// packed-bit representation is a future optimization that does not change the
/// API.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Validity(Vec<bool>);

impl Validity {
    fn with_len(n: usize) -> Self {
        Validity(vec![false; n])
    }
    /// Whether `row` holds a value.
    pub fn get(&self, row: usize) -> bool {
        self.0.get(row).copied().unwrap_or(false)
    }
    fn set(&mut self, row: usize, v: bool) {
        self.0[row] = v;
    }
    fn push(&mut self, v: bool) {
        self.0.push(v);
    }
    fn swap_remove(&mut self, row: usize) {
        self.0.swap_remove(row);
    }
    fn extend_null(&mut self, count: usize) {
        self.0.resize(self.0.len() + count, false);
    }
    fn set_range(&mut self, rows: std::ops::Range<usize>, v: bool) {
        self.0[rows].fill(v);
    }
    /// The mask as a contiguous slice (zero-copy mappable).
    pub fn as_slice(&self) -> &[bool] {
        &self.0
    }
    /// Number of rows.
    pub fn len(&self) -> usize {
        self.0.len()
    }
    /// Whether the mask is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A typed component column: a dense `Vec<T>` aligned to the table's rows plus a
/// [`Validity`] mask. `data[i]` is meaningful iff `validity().get(i)`.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityColumn {
    /// 64-bit float column.
    F64(Vec<F>, Validity),
    /// 32-bit integer column.
    I32(Vec<I>, Validity),
    /// UTF-8 string column.
    Str(Vec<String>, Validity),
    /// Boolean column.
    Bool(Vec<bool>, Validity),
}

impl EntityColumn {
    /// Element type name (for error messages).
    pub fn type_name(&self) -> &'static str {
        match self {
            EntityColumn::F64(..) => "f64",
            EntityColumn::I32(..) => "i32",
            EntityColumn::Str(..) => "str",
            EntityColumn::Bool(..) => "bool",
        }
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        match self {
            EntityColumn::F64(d, _) => d.len(),
            EntityColumn::I32(d, _) => d.len(),
            EntityColumn::Str(d, _) => d.len(),
            EntityColumn::Bool(d, _) => d.len(),
        }
    }

    /// Whether the column has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The column's validity mask.
    pub fn validity(&self) -> &Validity {
        match self {
            EntityColumn::F64(_, v) => v,
            EntityColumn::I32(_, v) => v,
            EntityColumn::Str(_, v) => v,
            EntityColumn::Bool(_, v) => v,
        }
    }

    /// Append one null slot (default value, validity unset) — keeps the column
    /// aligned when a new row is spawned.
    fn push_null(&mut self) {
        match self {
            EntityColumn::F64(d, v) => {
                d.push(0.0);
                v.push(false);
            }
            EntityColumn::I32(d, v) => {
                d.push(0);
                v.push(false);
            }
            EntityColumn::Str(d, v) => {
                d.push(String::new());
                v.push(false);
            }
            EntityColumn::Bool(d, v) => {
                d.push(false);
                v.push(false);
            }
        }
    }

    /// Swap-remove `row` (move the last row into `row`, drop the last).
    fn swap_remove(&mut self, row: usize) {
        match self {
            EntityColumn::F64(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
            EntityColumn::I32(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
            EntityColumn::Str(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
            EntityColumn::Bool(d, v) => {
                d.swap_remove(row);
                v.swap_remove(row);
            }
        }
    }

    /// Append `count` null slots (default values, validity unset).
    fn extend_null(&mut self, count: usize) {
        match self {
            EntityColumn::F64(d, v) => {
                d.resize(d.len() + count, 0.0);
                v.extend_null(count);
            }
            EntityColumn::I32(d, v) => {
                d.resize(d.len() + count, 0);
                v.extend_null(count);
            }
            EntityColumn::Str(d, v) => {
                d.resize(d.len() + count, String::new());
                v.extend_null(count);
            }
            EntityColumn::Bool(d, v) => {
                d.resize(d.len() + count, false);
                v.extend_null(count);
            }
        }
    }

    /// An all-null column of `len` rows with the element type of `like`.
    fn null_like(like: &EntityColumn, len: usize) -> EntityColumn {
        let mut col = match like {
            EntityColumn::F64(..) => EntityColumn::F64(Vec::new(), Validity::default()),
            EntityColumn::I32(..) => EntityColumn::I32(Vec::new(), Validity::default()),
            EntityColumn::Str(..) => EntityColumn::Str(Vec::new(), Validity::default()),
            EntityColumn::Bool(..) => EntityColumn::Bool(Vec::new(), Validity::default()),
        };
        col.extend_null(len);
        col
    }

    /// Whether `self` and `other` hold the same element type.
    fn same_type(&self, other: &EntityColumn) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }

    /// Append `src` (data and validity) `times` times over. The element types
    /// must agree; a mismatch is a type conflict and appends nothing.
    fn extend_repeated(
        &mut self,
        key: &str,
        src: &EntityColumn,
        times: usize,
    ) -> Result<(), MolRsError> {
        fn repeat<T: Clone>(dst: &mut Vec<T>, src: &[T], times: usize) {
            dst.reserve(src.len() * times);
            for _ in 0..times {
                dst.extend_from_slice(src);
            }
        }
        match (self, src) {
            (EntityColumn::F64(d, v), EntityColumn::F64(sd, sv)) => {
                repeat(d, sd, times);
                repeat(&mut v.0, &sv.0, times);
            }
            (EntityColumn::I32(d, v), EntityColumn::I32(sd, sv)) => {
                repeat(d, sd, times);
                repeat(&mut v.0, &sv.0, times);
            }
            (EntityColumn::Str(d, v), EntityColumn::Str(sd, sv)) => {
                repeat(d, sd, times);
                repeat(&mut v.0, &sv.0, times);
            }
            (EntityColumn::Bool(d, v), EntityColumn::Bool(sd, sv)) => {
                repeat(d, sd, times);
                repeat(&mut v.0, &sv.0, times);
            }
            (dst, src) => return Err(type_conflict(key, src.type_name(), dst.type_name())),
        }
        Ok(())
    }

    fn set_valid(&mut self, row: usize, v: bool) {
        match self {
            EntityColumn::F64(_, mask) => mask.set(row, v),
            EntityColumn::I32(_, mask) => mask.set(row, v),
            EntityColumn::Str(_, mask) => mask.set(row, v),
            EntityColumn::Bool(_, mask) => mask.set(row, v),
        }
    }
}

/// A borrowed view of one component value (the dynamic counterpart to the typed
/// `get_*` accessors). Lets a caller read a cell without statically knowing its
/// element type — used to materialize a node's full property set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntityCell<'a> {
    /// 64-bit float value.
    F64(F),
    /// 32-bit integer value.
    I32(I),
    /// String value.
    Str(&'a str),
    /// Boolean value.
    Bool(bool),
}

fn missing(key: &str) -> MolRsError {
    MolRsError::NotFound {
        entity: "component",
        message: format!("component '{key}' is absent for this entity"),
    }
}

fn type_conflict(key: &str, want: &str, got: &str) -> MolRsError {
    MolRsError::Validation {
        message: format!("component '{key}' is typed {got}, not {want}"),
    }
}

/// Borrow the value at `row` of `col` as a [`EntityCell`]. Caller guarantees the row
/// is valid (present).
fn cell_at(col: &EntityColumn, row: usize) -> EntityCell<'_> {
    match col {
        EntityColumn::F64(d, _) => EntityCell::F64(d[row]),
        EntityColumn::I32(d, _) => EntityCell::I32(d[row]),
        EntityColumn::Str(d, _) => EntityCell::Str(&d[row]),
        EntityColumn::Bool(d, _) => EntityCell::Bool(d[row]),
    }
}

/// A stable-handle, aligned-column entity table — the ECS storage substrate.
///
/// `EntityTable` stores entities — identified by stable generational
/// [`slotmap`] handles — as **rows**, and their components as **shared-row
/// aligned dense columns + per-column validity masks** (the Arrow / dataframe
/// model):
///
/// - One `handle → row` map is shared by every column, so column `i` and
///   column `i` always describe the *same* entity. That makes a column a
///   contiguous `&[T]` of length `n_rows` — directly mappable to a zero-copy
///   numpy view *and* already aligned for tabular projection (`to_frame`), with
///   no gather.
/// - Sparsity is expressed by a per-column [`Validity`] mask: an entity may
///   have `charge` but not `port`; the row's validity bit is simply unset.
/// - Columns are created lazily on first write and their element type is fixed
///   at that point; a later write of a different type is a hard error (no
///   silent coercion).
/// - Deletion is swap-remove: the moved row's *handle* is unchanged (handles
///   are the stable identity), only its internal row index compacts.
///
/// This is the storage substrate for [`crate::core::MolGraph`]; it is generic
/// over the slotmap key type `K`, so the same machinery backs both the node
/// table and each relation-kind table.
#[derive(Debug, Clone)]
pub struct EntityTable<K: Key> {
    /// `handle → row index`.
    keys: SlotMap<K, u32>,
    /// `row index → handle` (iteration / alignment order).
    rows: Vec<K>,
    /// Component columns, keyed by name, in first-write order; every column
    /// has length `rows.len()`.
    cols: IndexMap<String, EntityColumn>,
}

impl<K: Key> Default for EntityTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key> EntityTable<K> {
    /// An empty table.
    pub fn new() -> Self {
        Self {
            keys: SlotMap::with_key(),
            rows: Vec::new(),
            cols: IndexMap::new(),
        }
    }

    /// Number of live entities (rows).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no entities.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Whether `k` is a live handle in this table.
    pub fn contains(&self, k: K) -> bool {
        self.keys.contains_key(k)
    }

    /// Live handles in row (alignment) order.
    pub fn handles(&self) -> impl Iterator<Item = K> + '_ {
        self.rows.iter().copied()
    }

    /// The internal row index of `k`, if live. O(1).
    pub fn row(&self, k: K) -> Option<usize> {
        self.keys.get(k).map(|&r| r as usize)
    }

    /// Registered component names.
    pub fn columns(&self) -> impl Iterator<Item = &str> {
        self.cols.keys().map(String::as_str)
    }

    /// The validity mask of column `key`, regardless of element type
    /// (`None` if the column is absent). Aligned to row order.
    pub fn col_validity(&self, key: &str) -> Option<&Validity> {
        self.cols.get(key).map(EntityColumn::validity)
    }

    /// The column `key` whatever its element type, or `None` when no entity
    /// has ever carried the component — the dynamic counterpart of the typed
    /// [`column`](Self::column), as [`value`](Self::value) is of
    /// [`get`](Self::get).
    pub fn entity_column(&self, key: &str) -> Option<&EntityColumn> {
        self.cols.get(key)
    }

    /// Spawn a new entity: appends a null row across all existing columns and
    /// returns its stable handle. O(n_columns).
    pub fn spawn(&mut self) -> K {
        let row = self.rows.len() as u32;
        let k = self.keys.insert(row);
        self.rows.push(k);
        for col in self.cols.values_mut() {
            col.push_null();
        }
        k
    }

    /// Despawn `k` via swap-remove. Returns `false` if `k` is not live.
    ///
    /// Every *other* handle stays valid — only the row previously at the end is
    /// moved into `k`'s slot, and that entity's handle is unchanged (just its
    /// internal row index). Handles are the stable identity; rows are internal.
    pub(crate) fn despawn(&mut self, k: K) -> bool {
        let row = match self.keys.get(k) {
            Some(&r) => r as usize,
            None => return false,
        };
        self.keys.remove(k);
        self.rows.swap_remove(row);
        if row < self.rows.len() {
            let moved = self.rows[row];
            self.keys[moved] = row as u32;
        }
        for col in self.cols.values_mut() {
            col.swap_remove(row);
        }
        true
    }

    fn require_row(&self, k: K) -> Result<usize, MolRsError> {
        self.row(k).ok_or(MolRsError::NotFound {
            entity: "entity",
            message: "stale or unknown entity handle".to_owned(),
        })
    }

    /// Whether entity `k` currently holds component `key`.
    pub fn has(&self, k: K, key: &str) -> bool {
        match self.row(k) {
            Some(row) => self.cols.get(key).is_some_and(|c| c.validity().get(row)),
            None => false,
        }
    }

    /// Read one component value of entity `k` without statically knowing its
    /// element type. `None` if the handle is stale, the column is absent, or the
    /// value is null for this entity.
    pub fn value(&self, k: K, key: &str) -> Option<EntityCell<'_>> {
        let row = self.row(k)?;
        let col = self.cols.get(key)?;
        if !col.validity().get(row) {
            return None;
        }
        Some(cell_at(col, row))
    }

    /// Iterate over every present `(component_name, value)` of entity `k` (skips
    /// null/absent components). Empty for a stale handle.
    pub fn row_cells(&self, k: K) -> impl Iterator<Item = (&str, EntityCell<'_>)> {
        let row = self.row(k);
        self.cols.iter().filter_map(move |(name, col)| {
            let row = row?;
            if !col.validity().get(row) {
                return None;
            }
            Some((name.as_str(), cell_at(col, row)))
        })
    }

    /// Check that every column of `source` can be appended to `self`: a column
    /// both tables hold must have one element type in both. Writes nothing.
    ///
    /// This is the check [`extend_repeated`](Self::extend_repeated) runs before
    /// its first write, exposed so a caller appending to several tables can run
    /// every check before touching any of them.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] naming the first key whose element type in
    /// `source` contradicts the one `self` holds.
    pub(crate) fn check_extend(&self, source: &EntityTable<K>) -> Result<(), MolRsError> {
        for (key, src) in &source.cols {
            if let Some(dst) = self.cols.get(key)
                && !dst.same_type(src)
            {
                return Err(type_conflict(key, src.type_name(), dst.type_name()));
            }
        }
        Ok(())
    }

    /// Append `times` copies of every row of `source`, returning the new
    /// handles copy-major: source row `r` of copy `c` is at index
    /// `c * source.len() + r`, and the new rows follow the existing ones in
    /// that order.
    ///
    /// EntityColumn-wise: each column is extended in one pass by the source column
    /// (data and validity) repeated `times` times; a column only `self` holds
    /// is extended by nulls, and a column only `source` holds is created, null
    /// for the pre-existing rows. No per-row setter runs.
    ///
    /// Atomic: [`check_extend`](Self::check_extend) runs first, so on an error
    /// `self` is unchanged.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when a column of `source` contradicts the
    /// element type `self` holds for that key.
    pub(crate) fn extend_repeated(
        &mut self,
        source: &EntityTable<K>,
        times: usize,
    ) -> Result<Vec<K>, MolRsError> {
        self.check_extend(source)?;
        let old = self.rows.len();
        let added = source.len() * times;
        let mut handles = Vec::with_capacity(added);
        self.rows.reserve(added);
        for i in 0..added {
            let k = self.keys.insert((old + i) as u32);
            self.rows.push(k);
            handles.push(k);
        }
        for (key, dst) in self.cols.iter_mut() {
            match source.cols.get(key) {
                Some(src) => dst
                    .extend_repeated(key, src, times)
                    .expect("check_extend ran before the first write"),
                None => dst.extend_null(added),
            }
        }
        for (key, src) in &source.cols {
            if !self.cols.contains_key(key) {
                let mut col = EntityColumn::null_like(src, old);
                col.extend_repeated(key, src, times)
                    .expect("a null_like column shares its source's element type");
                self.cols.insert(key.clone(), col);
            }
        }
        Ok(handles)
    }

    /// Set the `i32` component `key` to `val` over the contiguous row range
    /// `rows`, creating the column on first use. One pass over the range.
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when `key` is held at another element type;
    /// nothing is written then.
    pub(crate) fn fill_i32(
        &mut self,
        key: &str,
        rows: std::ops::Range<usize>,
        val: I,
    ) -> Result<(), MolRsError> {
        let n = self.rows.len();
        let col = self
            .cols
            .entry(key.to_owned())
            .or_insert_with(|| EntityColumn::I32(vec![0; n], Validity::with_len(n)));
        match col {
            EntityColumn::I32(data, valid) => {
                data[rows.clone()].fill(val);
                valid.set_range(rows, true);
                Ok(())
            }
            other => Err(type_conflict(key, "i32", other.type_name())),
        }
    }

    /// Clear component `key` for entity `k` (set null). No-op if the column or
    /// value is absent. Errors only on a stale handle.
    pub fn clear(&mut self, k: K, key: &str) -> Result<(), MolRsError> {
        let row = self.require_row(k)?;
        if let Some(col) = self.cols.get_mut(key) {
            col.set_valid(row, false);
        }
        Ok(())
    }
}

/// Element types an [`EntityTable`] column can hold: [`F`], [`I`], `bool`
/// and `String`, one per [`EntityColumn`] variant.
///
/// It makes the typed table surface generic — [`EntityTable::set`],
/// [`EntityTable::get`], [`EntityTable::column`] and
/// [`EntityTable::column_mut`] take the element type as their parameter. The
/// trait is sealed: the set of element types is closed.
pub trait EntityDtype: Clone + Default + 'static + sealed::Slot {
    /// What [`EntityTable::get`] hands back: the value itself for the `Copy`
    /// scalars, a `&str` borrow for `String`.
    type Ref<'a>;

    /// Read one stored value as [`Ref`](Self::Ref).
    #[doc(hidden)]
    fn read(value: &Self) -> Self::Ref<'_>;
}

mod sealed {
    use super::{EntityColumn, Validity};

    /// Which [`EntityColumn`] variant holds this element type.
    pub trait Slot: Sized {
        /// The element type's name in error messages.
        const NAME: &'static str;
        /// Wrap a data vector and its mask in this type's variant.
        fn wrap(data: Vec<Self>, valid: Validity) -> EntityColumn;
        /// The data and mask, when `col` is this type's variant.
        fn slot(col: &EntityColumn) -> Option<(&Vec<Self>, &Validity)>;
        /// The data and mask, mutably, when `col` is this type's variant.
        fn slot_mut(col: &mut EntityColumn) -> Option<(&mut Vec<Self>, &mut Validity)>;
    }
}

macro_rules! impl_entity_dtype {
    ($ty:ty, $variant:ident, $name:literal, $ref:ty, |$v:ident| $read:expr) => {
        impl EntityDtype for $ty {
            type Ref<'a> = $ref;
            fn read($v: &Self) -> Self::Ref<'_> {
                $read
            }
        }
        impl sealed::Slot for $ty {
            const NAME: &'static str = $name;
            fn wrap(data: Vec<Self>, valid: Validity) -> EntityColumn {
                EntityColumn::$variant(data, valid)
            }
            fn slot(col: &EntityColumn) -> Option<(&Vec<Self>, &Validity)> {
                match col {
                    EntityColumn::$variant(data, valid) => Some((data, valid)),
                    _ => None,
                }
            }
            fn slot_mut(col: &mut EntityColumn) -> Option<(&mut Vec<Self>, &mut Validity)> {
                match col {
                    EntityColumn::$variant(data, valid) => Some((data, valid)),
                    _ => None,
                }
            }
        }
    };
}

impl_entity_dtype!(F, F64, "f64", F, |v| *v);
impl_entity_dtype!(I, I32, "i32", I, |v| *v);
impl_entity_dtype!(bool, Bool, "bool", bool, |v| *v);
impl_entity_dtype!(String, Str, "str", &'a str, |v| v.as_str());

impl<K: Key> EntityTable<K> {
    /// Set the `T` component `key` on entity `k`, creating the column on first
    /// use.
    ///
    /// ```
    /// use molrs::core::{EntityTable, NodeId};
    ///
    /// let mut t = EntityTable::<NodeId>::new();
    /// let a = t.spawn();
    /// t.set::<f64>(a, "charge", -0.5).unwrap();
    /// t.set::<String>(a, "name", "OW").unwrap();
    /// assert_eq!(t.get::<f64>(a, "charge").unwrap(), -0.5);
    /// assert_eq!(t.get::<String>(a, "name").unwrap(), "OW");
    /// assert!(t.get::<i32>(a, "charge").is_err()); // typed f64, not i32
    /// ```
    ///
    /// # Errors
    ///
    /// On a stale handle, or if `key` already exists with a different element
    /// type (no silent coercion). Nothing is written on error.
    pub fn set<T: EntityDtype>(
        &mut self,
        k: K,
        key: &str,
        val: impl Into<T>,
    ) -> Result<(), MolRsError> {
        let row = self.require_row(k)?;
        let n = self.rows.len();
        match self.cols.get_mut(key) {
            None => {
                let mut data = vec![T::default(); n];
                let mut valid = Validity::with_len(n);
                data[row] = val.into();
                valid.set(row, true);
                self.cols.insert(key.to_owned(), T::wrap(data, valid));
            }
            Some(col) => match T::slot_mut(col) {
                Some((data, valid)) => {
                    data[row] = val.into();
                    valid.set(row, true);
                }
                None => return Err(type_conflict(key, T::NAME, col.type_name())),
            },
        }
        Ok(())
    }

    /// Get the `T` component `key` of entity `k`.
    ///
    /// # Errors
    ///
    /// If the handle is stale, the value is absent (column missing or null for
    /// this entity), or the column has a different element type. Strict by
    /// design — no fallback default.
    pub fn get<T: EntityDtype>(&self, k: K, key: &str) -> Result<T::Ref<'_>, MolRsError> {
        let row = self.require_row(k)?;
        let col = self.cols.get(key).ok_or_else(|| missing(key))?;
        match T::slot(col) {
            Some((data, valid)) if valid.get(row) => Ok(T::read(&data[row])),
            Some(_) => Err(missing(key)),
            None => Err(type_conflict(key, T::NAME, col.type_name())),
        }
    }

    /// Borrow the whole `T` column `key` (zero-copy slice) plus its validity
    /// mask.
    ///
    /// The slice has length `len()` and is aligned to row order.
    ///
    /// # Errors
    ///
    /// If the column is absent or has a different element type.
    pub fn column<T: EntityDtype>(&self, key: &str) -> Result<(&[T], &Validity), MolRsError> {
        let col = self.cols.get(key).ok_or_else(|| missing(key))?;
        match T::slot(col) {
            Some((data, valid)) => Ok((data.as_slice(), valid)),
            None => Err(type_conflict(key, T::NAME, col.type_name())),
        }
    }

    /// Mutably borrow the whole `T` column `key` (zero-copy slice) plus its
    /// validity mask, for in-place vectorized updates over the dense,
    /// row-aligned column (rows are compacted, so the slice spans exactly the
    /// live entities).
    ///
    /// # Errors
    ///
    /// If the column is absent or has a different element type.
    pub fn column_mut<T: EntityDtype>(
        &mut self,
        key: &str,
    ) -> Result<(&mut [T], &Validity), MolRsError> {
        let col = self.cols.get_mut(key).ok_or_else(|| missing(key))?;
        let name = col.type_name();
        match T::slot_mut(col) {
            Some((data, valid)) => Ok((data.as_mut_slice(), &*valid)),
            None => Err(type_conflict(key, T::NAME, name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::new_key_type;

    new_key_type! {
        struct TestId;
    }

    type T = EntityTable<TestId>;

    #[test]
    fn columns_follow_first_write_order() {
        let mut t = T::new();
        let a = t.spawn();
        t.set::<f64>(a, "c", 1.0).unwrap();
        t.set::<String>(a, "a", "x").unwrap();
        t.set::<i32>(a, "b", 2).unwrap();
        t.set::<f64>(a, "c", 3.0).unwrap();
        assert_eq!(t.columns().collect::<Vec<_>>(), ["c", "a", "b"]);
    }

    #[test]
    fn spawn_assigns_rows_in_order() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        let c = t.spawn();
        assert_eq!(t.len(), 3);
        assert_eq!(t.row(a), Some(0));
        assert_eq!(t.row(b), Some(1));
        assert_eq!(t.row(c), Some(2));
        assert_eq!(t.handles().collect::<Vec<_>>(), vec![a, b, c]);
        assert!(t.contains(a) && t.contains(b) && t.contains(c));
    }

    #[test]
    fn lazy_column_creation_and_typed_get() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        // No column yet → strict get errors.
        assert!(t.get::<f64>(a, "charge").is_err());
        t.set::<f64>(a, "charge", 0.5).unwrap();
        assert_eq!(t.get::<f64>(a, "charge").unwrap(), 0.5);
        // `b` shares the column but has no value → null → error, not a default.
        assert!(t.get::<f64>(b, "charge").is_err());
        assert!(!t.has(b, "charge"));
        assert!(t.has(a, "charge"));
    }

    #[test]
    fn sparse_via_null_mask() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        t.set::<f64>(a, "charge", 1.0).unwrap();
        t.set::<String>(b, "port", "head").unwrap();
        // a has charge not port; b has port not charge.
        assert!(t.has(a, "charge") && !t.has(a, "port"));
        assert!(t.has(b, "port") && !t.has(b, "charge"));
        // Columns are aligned to length n_rows regardless of sparsity.
        let (data, valid) = t.column::<f64>("charge").unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(valid.as_slice(), &[true, false]);
    }

    #[test]
    fn columns_share_row_order_alignment() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        t.set::<f64>(a, "x", 1.0).unwrap();
        t.set::<f64>(b, "x", 2.0).unwrap();
        t.set::<String>(a, "el", "C").unwrap();
        t.set::<String>(b, "el", "O").unwrap();
        let (xs, _) = t.column::<f64>("x").unwrap();
        let (els, _) = t.column::<String>("el").unwrap();
        // Row i of every column is the same entity.
        assert_eq!(xs[t.row(a).unwrap()], 1.0);
        assert_eq!(els[t.row(a).unwrap()], "C");
        assert_eq!(xs[t.row(b).unwrap()], 2.0);
        assert_eq!(els[t.row(b).unwrap()], "O");
    }

    #[test]
    fn type_conflict_is_an_error() {
        let mut t = T::new();
        let a = t.spawn();
        t.set::<f64>(a, "k", 1.0).unwrap();
        // Re-typing the same column is rejected, not silently coerced.
        assert!(t.set::<String>(a, "k", "x").is_err());
        assert!(t.get::<String>(a, "k").is_err());
        assert!(t.set::<i32>(a, "k", 3).is_err());
        // Original value intact.
        assert_eq!(t.get::<f64>(a, "k").unwrap(), 1.0);
    }

    #[test]
    fn despawn_swap_remove_keeps_other_handles_stable() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        let c = t.spawn();
        t.set::<f64>(a, "x", 10.0).unwrap();
        t.set::<f64>(b, "x", 20.0).unwrap();
        t.set::<f64>(c, "x", 30.0).unwrap();
        // Remove the middle entity.
        assert!(t.despawn(b));
        assert_eq!(t.len(), 2);
        assert!(!t.contains(b));
        // a and c handles still resolve to their own data (c moved rows
        // internally, but its handle and value are unchanged).
        assert_eq!(t.get::<f64>(a, "x").unwrap(), 10.0);
        assert_eq!(t.get::<f64>(c, "x").unwrap(), 30.0);
        // b's row was filled by c; the column stays compact and aligned.
        let (xs, _) = t.column::<f64>("x").unwrap();
        assert_eq!(xs.len(), 2);
        let mut got: Vec<f64> = t.handles().map(|h| t.get::<f64>(h, "x").unwrap()).collect();
        got.sort_by(|p, q| p.partial_cmp(q).unwrap());
        assert_eq!(got, vec![10.0, 30.0]);
    }

    #[test]
    fn despawn_last_and_stale_handle() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        assert!(t.despawn(b)); // remove the last row
        assert_eq!(t.len(), 1);
        assert!(!t.despawn(b)); // already gone → false, no panic
        assert!(t.get::<f64>(b, "x").is_err()); // stale handle errors
        assert!(t.contains(a));
    }

    #[test]
    fn value_and_row_cells_read_dynamically() {
        let mut t = T::new();
        let a = t.spawn();
        t.set::<f64>(a, "x", 1.5).unwrap();
        t.set::<i32>(a, "n", 7).unwrap();
        t.set::<String>(a, "el", "C").unwrap();
        assert_eq!(t.value(a, "x"), Some(EntityCell::F64(1.5)));
        assert_eq!(t.value(a, "n"), Some(EntityCell::I32(7)));
        assert_eq!(t.value(a, "el"), Some(EntityCell::Str("C")));
        assert_eq!(t.value(a, "absent"), None);
        // row_cells yields exactly the present components.
        let mut cells: Vec<(String, EntityCell<'_>)> =
            t.row_cells(a).map(|(k, v)| (k.to_owned(), v)).collect();
        cells.sort_by(|p, q| p.0.cmp(&q.0));
        assert_eq!(
            cells,
            vec![
                ("el".to_owned(), EntityCell::Str("C")),
                ("n".to_owned(), EntityCell::I32(7)),
                ("x".to_owned(), EntityCell::F64(1.5)),
            ]
        );
        // A second, sparser entity only surfaces its own present cells.
        let b = t.spawn();
        t.set::<f64>(b, "x", 9.0).unwrap();
        assert_eq!(t.value(b, "el"), None);
        assert_eq!(t.row_cells(b).count(), 1);
    }

    #[test]
    fn clear_unsets_without_removing_column() {
        let mut t = T::new();
        let a = t.spawn();
        t.set::<f64>(a, "charge", 2.0).unwrap();
        assert!(t.has(a, "charge"));
        t.clear(a, "charge").unwrap();
        assert!(!t.has(a, "charge"));
        assert!(t.get::<f64>(a, "charge").is_err());
        // EntityColumn still exists (other entities may use it).
        assert!(t.column::<f64>("charge").is_ok());
    }

    // ----- bulk append: check_extend / extend_repeated -----

    fn sorted_columns(t: &T) -> Vec<String> {
        let mut cols: Vec<String> = t.columns().map(str::to_owned).collect();
        cols.sort();
        cols
    }

    #[test]
    fn extend_repeated_creates_a_source_only_column_null_on_old_rows() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        let mut src = T::new();
        let s0 = src.spawn();
        let s1 = src.spawn();
        src.set::<String>(s0, "el", "C").unwrap();
        src.set::<String>(s1, "el", "O").unwrap();

        let handles = t.extend_repeated(&src, 2).unwrap();

        assert_eq!(handles.len(), 4);
        assert_eq!(t.len(), 6);
        for (i, &h) in handles.iter().enumerate() {
            assert_eq!(t.row(h), Some(2 + i), "new rows follow the old ones");
        }
        assert!(!t.has(a, "el") && !t.has(b, "el"), "old rows read null");
        let (data, valid) = t.column::<String>("el").unwrap();
        assert_eq!(valid.as_slice(), &[false, false, true, true, true, true]);
        assert_eq!(&data[2..], &["C", "O", "C", "O"], "copy-major repetition");
    }

    #[test]
    fn extend_repeated_nulls_a_self_only_column_on_new_rows() {
        let mut t = T::new();
        let a = t.spawn();
        let b = t.spawn();
        t.set::<f64>(a, "x", 1.0).unwrap();
        t.set::<f64>(b, "x", 2.0).unwrap();
        let mut src = T::new();
        let s = src.spawn();
        src.set::<String>(s, "el", "N").unwrap();

        let handles = t.extend_repeated(&src, 3).unwrap();

        assert_eq!(handles.len(), 3);
        let (data, valid) = t.column::<f64>("x").unwrap();
        assert_eq!(data.len(), 5);
        assert_eq!(valid.as_slice(), &[true, true, false, false, false]);
        assert_eq!(&data[..2], &[1.0, 2.0], "old rows unchanged");
        for &h in &handles {
            assert!(!t.has(h, "x"));
        }
    }

    #[test]
    fn check_extend_refuses_a_column_type_conflict_and_writes_nothing() {
        let mut t = T::new();
        let a = t.spawn();
        t.set::<f64>(a, "k", 1.0).unwrap();
        let mut src = T::new();
        let s = src.spawn();
        src.set::<String>(s, "k", "x").unwrap();
        src.set::<i32>(s, "extra", 4).unwrap();

        let err = t.check_extend(&src).expect_err("f64 'k' vs str 'k'");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(t.len(), 1);
        assert_eq!(sorted_columns(&t), vec!["k".to_owned()]);
        assert_eq!(t.get::<f64>(a, "k").unwrap(), 1.0);
    }

    #[test]
    fn extend_repeated_refuses_a_column_type_conflict_and_writes_nothing() {
        let mut t = T::new();
        let a = t.spawn();
        t.set::<f64>(a, "k", 1.0).unwrap();
        let mut src = T::new();
        let s = src.spawn();
        src.set::<String>(s, "k", "x").unwrap();
        src.set::<i32>(s, "extra", 4).unwrap();

        let err = t.extend_repeated(&src, 2).expect_err("f64 'k' vs str 'k'");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        assert_eq!(t.len(), 1);
        assert_eq!(sorted_columns(&t), vec!["k".to_owned()]);
        assert_eq!(t.column::<f64>("k").unwrap().0.len(), 1);
    }

    // ----- fill_i32 -----

    #[test]
    fn fill_i32_writes_only_the_given_row_range() {
        let mut t = T::new();
        let a = t.spawn();
        for _ in 0..3 {
            t.spawn();
        }
        t.set::<i32>(a, "fid", 5).unwrap();

        t.fill_i32("fid", 1..3, 9).unwrap();

        let (data, valid) = t.column::<i32>("fid").unwrap();
        assert_eq!(valid.as_slice(), &[true, true, true, false]);
        assert_eq!(&data[..3], &[5, 9, 9]);
    }

    #[test]
    fn fill_i32_creates_the_column_null_outside_the_range() {
        let mut t = T::new();
        for _ in 0..4 {
            t.spawn();
        }

        t.fill_i32("fid", 2..4, 7).unwrap();

        let (data, valid) = t.column::<i32>("fid").unwrap();
        assert_eq!(valid.as_slice(), &[false, false, true, true]);
        assert_eq!(&data[2..], &[7, 7]);
    }

    #[test]
    fn fill_i32_refuses_a_non_i32_column_and_writes_nothing() {
        let mut t = T::new();
        let a = t.spawn();
        t.spawn();
        t.set::<f64>(a, "fid", 0.5).unwrap();

        let err = t.fill_i32("fid", 0..2, 3).expect_err("'fid' is f64");

        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
        let (data, valid) = t.column::<f64>("fid").unwrap();
        assert_eq!(valid.as_slice(), &[true, false]);
        assert_eq!(data[0], 0.5);
    }
}
