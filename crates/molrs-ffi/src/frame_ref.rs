//! [`FrameRef`] and [`BlockRef`]: a handle paired with the
//! [`FrameArenaCell`] that owns its frame.
//!
//! The Python (`molpy`), WebAssembly (`molwasm`) and C++ (`molcxx`)
//! bindings hold one of these for a frame and forward every access through
//! the helpers defined here, so frame schema lookup, dtype dispatch and slice
//! borrowing live in one place. The C API (`molc`) keeps its own handle
//! registry over these frames for boxes, force fields and regions, and its
//! own dtype codes.
//!
//! ## Access semantics
//!
//! Column accessors return [`Result<Option<T>, FfiError>`], splitting two
//! semantically different conditions:
//!
//! * `Ok(None)` — the column does **not** exist. This is a normal, expected
//!   outcome because molrs schemas are flexible (e.g. LAMMPS data has no
//!   `element`). Callers handle it with a simple `if let Some(..)`.
//!   [`BlockRef::has`] is the same split as a bool: `Ok(false)` is absence,
//!   and a dropped block is [`FfiError::InvalidBlockHandle`], not `Ok(false)`.
//! * `Err(DTypeMismatch)` — the column **exists but has the wrong dtype**
//!   for the accessor called. Almost always a schema-level caller bug.
//! * `Err(InvalidBlockHandle)` / `Err(NonContiguous)` — structural errors
//!   (stale handle, non-memory-contiguous view). Propagate as-is.
//!
//! ## Threading
//!
//! [`FrameArenaCell`] is `Rc<RefCell<_>>`. All current FFI consumers (wasm
//! is single-threaded; python holds the GIL; cxx is used from
//! single-threaded C++ today) satisfy this. Native multi-threaded Rust
//! consumers hold a [`FrameArena`] directly or wrap it themselves in
//! `Arc<Mutex<_>>` (molc does).

use std::cell::RefCell;
use std::rc::Rc;

use molrs::core::Frame;
use molrs::core::MetaMap;
use molrs::core::SimBox;
use molrs::core::{Block, BlockDtype, DType};

use crate::error::FfiError;
use crate::frame_arena::FrameArena;
use crate::handle::{BlockHandle, FrameId};

/// Single-threaded shared ownership of a [`FrameArena`]; a new empty one is
/// `FrameArenaCell::default()`.
pub type FrameArenaCell = Rc<RefCell<FrameArena>>;

/// An owned column as `(values, shape)`, or `None` when absent.
pub type OwnedColumn<T> = Option<(Vec<T>, Vec<usize>)>;

// =====================================================================
// FrameRef
// =====================================================================

/// A paired [`FrameId`] + [`FrameArenaCell`]. Cheap to clone (two `Rc` bumps).
///
/// Bindings wrap this as their `Frame` type and add language-specific
/// attributes (e.g. `#[wasm_bindgen]`, `#[pyclass]`).
#[derive(Clone)]
pub struct FrameRef {
    pub id: FrameId,
    pub arena: FrameArenaCell,
}

impl FrameRef {
    /// Wrap an existing arena + id.
    pub fn new(arena: FrameArenaCell, id: FrameId) -> Self {
        Self { id, arena }
    }

    /// Run a closure with immutable access to the underlying [`Frame`].
    pub fn with<R>(&self, f: impl FnOnce(&Frame) -> R) -> Result<R, FfiError> {
        self.arena.borrow().with_frame(self.id, f)
    }

    /// Run a closure with mutable access to the underlying [`Frame`].
    ///
    /// Conservatively invalidates every block handle on this frame when the
    /// closure returns, since `&mut Frame` permits arbitrary block
    /// modifications.
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut Frame) -> R) -> Result<R, FfiError> {
        self.arena.borrow_mut().with_frame_mut(self.id, f)
    }

    /// Run a closure with mutable access to the frame's metadata only.
    ///
    /// Unlike [`with_mut`](Self::with_mut), block handles on this frame stay
    /// valid: the closure cannot reach a block.
    pub fn with_meta_mut<R>(&self, f: impl FnOnce(&mut MetaMap) -> R) -> Result<R, FfiError> {
        self.arena.borrow_mut().with_frame_meta_mut(self.id, f)
    }

    /// Resolve a child block key into a [`BlockRef`]. Returns
    /// `Err(KeyNotFound)` if the key is absent from this frame.
    pub fn block(&self, key: &str) -> Result<BlockRef, FfiError> {
        let handle = self.arena.borrow().get_block(self.id, key)?;
        Ok(BlockRef::new(Rc::clone(&self.arena), handle))
    }

    /// True if the frame has a block at `key`.
    ///
    /// `Ok(false)` is a live frame that lacks `key`. A handle whose frame was
    /// dropped is [`FfiError::InvalidFrameId`], never `Ok(false)`.
    pub fn has_block(&self, key: &str) -> Result<bool, FfiError> {
        self.with(|f| f.contains_key(key))
    }

    /// Clone the simbox out of the frame (if any).
    pub fn box_clone(&self) -> Result<Option<SimBox>, FfiError> {
        self.with(|f| f.simbox.clone())
    }

    /// Replace / clear the simbox.
    pub fn set_box(&self, simbox: Option<SimBox>) -> Result<(), FfiError> {
        self.arena.borrow_mut().set_frame_box(self.id, simbox)
    }

    /// Deep-clone the frame's data out of the arena.
    pub fn clone_frame(&self) -> Result<Frame, FfiError> {
        self.arena.borrow().clone_frame(self.id)
    }

    /// Drop the frame from the arena, invalidating every handle to it.
    pub fn drop_frame(self) -> Result<(), FfiError> {
        self.arena.borrow_mut().drop_frame(self.id)
    }
}

impl Default for FrameRef {
    /// A new empty frame inside a fresh [`FrameArenaCell`].
    fn default() -> Self {
        let arena = FrameArenaCell::default();
        let id = arena.borrow_mut().frame_new();
        Self { id, arena }
    }
}

// =====================================================================
// BlockRef
// =====================================================================

/// A paired [`BlockHandle`] + [`FrameArenaCell`]. Cheap to clone.
///
/// Bindings wrap this as their `Block` type. All column-access helpers
/// live on this type so the bindings don't reinvent dtype dispatch.
#[derive(Clone)]
pub struct BlockRef {
    pub handle: BlockHandle,
    pub arena: FrameArenaCell,
}

impl BlockRef {
    pub fn new(arena: FrameArenaCell, handle: BlockHandle) -> Self {
        Self { handle, arena }
    }

    /// Run a closure with immutable access to the underlying [`Block`].
    pub fn with<R>(&self, f: impl FnOnce(&Block) -> R) -> Result<R, FfiError> {
        self.arena.borrow().with_block(&self.handle, f)
    }

    /// Run a closure with mutable access; bumps the handle version.
    pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut Block) -> R) -> Result<R, FfiError> {
        self.arena.borrow_mut().with_block_mut(&mut self.handle, f)
    }

    // ---- Metadata ----

    /// Number of rows. Returns 0 for an empty block (never Err except on
    /// handle invalidation).
    pub fn n_rows(&self) -> Result<usize, FfiError> {
        self.with(|b| b.n_rows().unwrap_or(0))
    }

    /// Column names currently present on this block.
    pub fn keys(&self) -> Result<Vec<String>, FfiError> {
        self.with(|b| b.keys().map(|s| s.to_string()).collect())
    }

    /// `Ok(true)` if the column exists. `Ok(false)` is a live block that
    /// lacks `key`. A handle whose block was dropped is
    /// [`FfiError::InvalidBlockHandle`], never `Ok(false)`.
    pub fn has(&self, key: &str) -> Result<bool, FfiError> {
        self.with(|b| b.contains_key(key))
    }

    /// Column dtype. `Ok(None)` if the column doesn't exist.
    pub fn dtype(&self, key: &str) -> Result<Option<DType>, FfiError> {
        self.with(|b| b.dtype(key))
    }

    /// Column shape. `Ok(None)` if the column doesn't exist. For molrs 1-D
    /// columns this is `[nrows]`; multi-dim columns (if any) carry their
    /// full shape.
    pub fn shape(&self, key: &str) -> Result<Option<Vec<usize>>, FfiError> {
        self.with(|b| b.get(key).map(|c| c.shape().to_vec()))
    }

    /// Deep-clone the block data out of the arena.
    pub fn clone_block(&self) -> Result<Block, FfiError> {
        self.arena.borrow().clone_block(&self.handle)
    }

    // ---- Typed column access ----

    /// Borrow a column of dtype `T` as a contiguous slice (zero-copy).
    ///
    /// * `Ok(None)` — column is absent (schema flexibility).
    /// * `Ok(Some(R))` — column is `T`, closure ran, result returned.
    /// * `Err(DTypeMismatch)` — column exists but isn't `T`.
    /// * `Err(NonContiguous)` — column is `T` but not contiguous.
    /// * `Err(InvalidBlockHandle)` — handle stale.
    pub fn borrow<T: BlockDtype, R>(
        &self,
        key: &str,
        f: impl FnOnce(&[T], &[usize]) -> R,
    ) -> Result<Option<R>, FfiError> {
        self.with(|b| -> Result<Option<R>, FfiError> {
            let Some(actual) = b.dtype(key) else {
                return Ok(None);
            };
            if actual != T::dtype() {
                return Err(FfiError::DTypeMismatch {
                    key: key.to_string(),
                    expected: T::dtype(),
                    actual,
                });
            }
            let arr = b
                .get(key)
                .and_then(|c| c.get::<T>())
                .ok_or(FfiError::InvalidBlockHandle)?;
            let slice = arr
                .as_slice_memory_order()
                .ok_or_else(|| FfiError::NonContiguous {
                    key: key.to_string(),
                })?;
            Ok(Some(f(slice, arr.shape())))
        })?
    }

    /// Owned copy of a column of dtype `T` as `(values, shape)`. Same
    /// Ok/Err semantics as [`borrow`](Self::borrow).
    pub fn copy<T: BlockDtype>(&self, key: &str) -> Result<OwnedColumn<T>, FfiError> {
        self.borrow::<T, _>(key, |slice, shape| (slice.to_vec(), shape.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::op::{F, Idx};
    use ndarray::Array1;

    fn frame_with_atoms() -> FrameRef {
        let frame = FrameRef::default();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![1.0 as F, 2.0]).into_dyn(), None)
            .unwrap();
        atoms
            .insert("id", Array1::from_vec(vec![7 as Idx, 9]).into_dyn(), None)
            .unwrap();
        atoms
            .insert(
                "name",
                Array1::from_vec(vec!["a".to_string(), "b".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame
            .arena
            .borrow_mut()
            .set_block(frame.id, "atoms", atoms)
            .unwrap();
        frame
    }

    #[test]
    fn borrow_and_copy_dispatch_on_the_requested_dtype() {
        let atoms = frame_with_atoms().block("atoms").unwrap();
        let sum = atoms
            .borrow::<F, _>("x", |x, _| x.iter().sum::<F>())
            .unwrap();
        assert_eq!(sum, Some(3.0));
        assert_eq!(
            atoms.copy::<Idx>("id").unwrap(),
            Some((vec![7, 9], vec![2]))
        );
        assert_eq!(
            atoms.copy::<String>("name").unwrap().map(|(v, _)| v),
            Some(vec!["a".to_string(), "b".to_string()])
        );
        assert_eq!(atoms.copy::<F>("absent").unwrap(), None);
        assert!(matches!(
            atoms.copy::<F>("id"),
            Err(FfiError::DTypeMismatch {
                expected: DType::Float,
                actual: DType::Uint,
                ..
            })
        ));
    }

    #[test]
    fn box_clone_reads_through_the_frame_and_drop_frame_invalidates() {
        let frame = frame_with_atoms();
        assert!(frame.box_clone().unwrap().is_none());
        let stale = frame.clone();
        frame.drop_frame().unwrap();
        assert!(matches!(stale.box_clone(), Err(FfiError::InvalidFrameId)));
    }

    #[test]
    fn has_block_on_a_dropped_frame_is_not_absence() {
        let frame = frame_with_atoms();
        assert!(frame.has_block("atoms").unwrap());
        assert!(!frame.has_block("bonds").unwrap());
        let stale = frame.clone();
        frame.drop_frame().unwrap();
        assert!(matches!(
            stale.has_block("atoms"),
            Err(FfiError::InvalidFrameId)
        ));
    }
}
