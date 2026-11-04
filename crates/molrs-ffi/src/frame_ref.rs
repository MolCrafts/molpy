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
//! [`FrameArenaCell`] is `Arc<RwLock<_>>`, so [`FrameRef`] and [`BlockRef`]
//! are `Send + Sync`: a binding may hand a frame to another thread and drop it
//! there (Python thread pools, `asyncio.to_thread`).
//!
//! Every access *tries* the lock and never waits. Readers share it; a writer
//! needs it alone. An access that finds the lock held in a conflicting mode is
//! [`FfiError::FrameBusy`] — on the same thread that is a callback reaching
//! back into the frame an operation is running on (an MD calculator reading
//! the frame it steps), which waiting would deadlock. Two threads only
//! contend when a native consumer touches the frame without holding the
//! binding's interpreter lock (the Python GIL), and they get the same refusal
//! rather than a wait that could deadlock against that lock. A panic inside
//! an access does not poison the frame: the store is still taken, as the
//! former `RefCell` did.
//!
//! The lock guards the arena — the frame map, its blocks' column maps, the
//! handle versions. Column buffers are `Arc`-shared (`molrs::core::Column`),
//! so a zero-copy view a binding hands out holds its own reference and stays
//! valid after the lock is released; a later column write replaces or
//! copies-on-write the column instead of reallocating under the view.

use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

use molrs::core::Frame;
use molrs::core::MetaMap;
use molrs::core::SimBox;
use molrs::core::{Block, BlockDtype, DType};

use crate::error::FfiError;
use crate::frame_arena::FrameArena;
use crate::handle::{BlockHandle, FrameId};

/// Shared, thread-safe ownership of a [`FrameArena`]; a new empty one is
/// `FrameArenaCell::default()`. Cloning is one `Arc` bump.
#[derive(Clone, Default)]
pub struct FrameArenaCell(Arc<RwLock<FrameArena>>);

impl FrameArenaCell {
    /// The store, for reading; [`FfiError::FrameBusy`] while a writer holds it.
    pub fn read(&self) -> Result<RwLockReadGuard<'_, FrameArena>, FfiError> {
        match self.0.try_read() {
            Ok(guard) => Ok(guard),
            Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => Err(FfiError::FrameBusy),
        }
    }

    /// The store, for writing; [`FfiError::FrameBusy`] while anything holds it.
    pub fn write(&self) -> Result<RwLockWriteGuard<'_, FrameArena>, FfiError> {
        match self.0.try_write() {
            Ok(guard) => Ok(guard),
            Err(TryLockError::Poisoned(poisoned)) => Ok(poisoned.into_inner()),
            Err(TryLockError::WouldBlock) => Err(FfiError::FrameBusy),
        }
    }

    /// True if both cells share one arena.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// An owned column as `(values, shape)`, or `None` when absent.
pub type OwnedColumn<T> = Option<(Vec<T>, Vec<usize>)>;

// =====================================================================
// FrameRef
// =====================================================================

/// A paired [`FrameId`] + [`FrameArenaCell`]. Cheap to clone (one `Arc` bump).
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
        self.arena.read()?.with_frame(self.id, f)
    }

    /// Run a closure with mutable access to the underlying [`Frame`].
    ///
    /// Conservatively invalidates every block handle on this frame when the
    /// closure returns, since `&mut Frame` permits arbitrary block
    /// modifications.
    pub fn with_mut<R>(&self, f: impl FnOnce(&mut Frame) -> R) -> Result<R, FfiError> {
        self.arena.write()?.with_frame_mut(self.id, f)
    }

    /// Run a closure with mutable access to the frame's metadata only.
    ///
    /// Unlike [`with_mut`](Self::with_mut), block handles on this frame stay
    /// valid: the closure cannot reach a block.
    pub fn with_meta_mut<R>(&self, f: impl FnOnce(&mut MetaMap) -> R) -> Result<R, FfiError> {
        self.arena.write()?.with_frame_meta_mut(self.id, f)
    }

    /// Resolve a child block key into a [`BlockRef`]. Returns
    /// `Err(KeyNotFound)` if the key is absent from this frame.
    pub fn block(&self, key: &str) -> Result<BlockRef, FfiError> {
        let handle = self.arena.read()?.get_block(self.id, key)?;
        Ok(BlockRef::new(self.arena.clone(), handle))
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
        self.arena.write()?.set_frame_box(self.id, simbox)
    }

    /// Deep-clone the frame's data out of the arena.
    pub fn clone_frame(&self) -> Result<Frame, FfiError> {
        self.arena.read()?.clone_frame(self.id)
    }

    /// Drop the frame from the arena, invalidating every handle to it.
    pub fn drop_frame(self) -> Result<(), FfiError> {
        self.arena.write()?.drop_frame(self.id)
    }
}

impl Default for FrameRef {
    /// A new empty frame inside a fresh [`FrameArenaCell`].
    fn default() -> Self {
        let arena = FrameArenaCell::default();
        let id = arena
            .write()
            .expect("a fresh arena has no other holder")
            .frame_new();
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
        self.arena.read()?.with_block(&self.handle, f)
    }

    /// Run a closure with mutable access; bumps the handle version.
    pub fn with_mut<R>(&mut self, f: impl FnOnce(&mut Block) -> R) -> Result<R, FfiError> {
        self.arena.write()?.with_block_mut(&mut self.handle, f)
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
        self.arena.read()?.clone_block(&self.handle)
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
            .write()
            .unwrap()
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

    #[test]
    fn handles_cross_threads_and_share_one_store() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FrameRef>();
        assert_send_sync::<BlockRef>();

        let frame = frame_with_atoms();
        let remote = frame.clone();
        std::thread::spawn(move || {
            let mut atoms = remote.block("atoms").unwrap();
            atoms
                .with_mut(|b| b.get_mut("x").unwrap().get_mut::<F>().unwrap()[0] = 5.0)
                .unwrap();
        })
        .join()
        .unwrap();
        let atoms = frame.block("atoms").unwrap();
        assert_eq!(
            atoms.copy::<F>("x").unwrap(),
            Some((vec![5.0, 2.0], vec![2]))
        );

        let readers: Vec<_> = (0..2)
            .map(|_| {
                let atoms = atoms.clone();
                std::thread::spawn(move || {
                    (0..1000)
                        .map(|_| atoms.borrow::<F, _>("x", |x, _| x.iter().sum::<F>()))
                        .all(|sum| matches!(sum, Ok(Some(s)) if s == 7.0))
                })
            })
            .collect();
        assert!(readers.into_iter().all(|r| r.join().unwrap()));
    }

    /// Two readers share the store, so a read nested inside a read is fine.
    #[test]
    fn a_read_inside_a_read_of_the_same_frame_is_allowed() {
        let frame = frame_with_atoms();
        let n = frame.with(|_| frame.block("atoms").and_then(|b| b.n_rows()));
        assert_eq!(n.unwrap().unwrap(), 2);
    }

    /// A panic inside a write does not lock the frame away for good.
    #[test]
    fn a_panic_inside_a_write_leaves_the_frame_usable() {
        let frame = frame_with_atoms();
        let remote = frame.clone();
        let panicked = std::thread::spawn(move || {
            remote.with_mut(|_| panic!("calculator failed")).unwrap();
        })
        .join();
        assert!(panicked.is_err());
        assert!(frame.has_block("atoms").unwrap());
    }

    /// A callback that reaches back into the frame a writer is running on —
    /// an MD calculator reading the frame it steps — is refused by name, not
    /// a borrow panic.
    #[test]
    fn a_read_inside_a_write_of_the_same_frame_is_refused() {
        let frame = frame_with_atoms();
        let inner = frame.with_mut(|_| frame.with(|_| ())).unwrap();
        assert!(matches!(inner, Err(FfiError::FrameBusy)), "{inner:?}");
        let block = frame
            .with_mut(|_| frame.block("atoms").map(|_| ()))
            .unwrap();
        assert!(matches!(block, Err(FfiError::FrameBusy)), "{block:?}");
    }
}
