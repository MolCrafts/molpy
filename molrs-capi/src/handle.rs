//! `repr(C)` handle types and internal conversion utilities.
//!
//! Each public handle type is a plain-old-data struct that fits in two
//! machine words and can be freely copied, stored, and passed by value
//! in C code.  Internally, each handle packs a SlotMap key (index +
//! generation) so that stale handles are detected at runtime.
//!
//! The conversion functions in this module are `pub(crate)` and not
//! part of the C ABI.

use molrs_ffi::{BlockHandle, FrameId};
use slotmap::{Key, new_key_type};

// --- SlotMap keys for Box and ForceField ---

new_key_type! {
    /// Key for Box entries in the HandleRegistry.
    pub struct BoxKey;
}

new_key_type! {
    /// Key for ForceField entries in the HandleRegistry.
    pub struct ForceFieldKey;
}

new_key_type! {
    /// Key for Region entries in the HandleRegistry.
    pub struct RegionKey;
}

// --- repr(C) handle structs ---

/// Opaque handle to a Frame in the handle registry.
///
/// Obtained from [`molrs_frame_new`](crate::molrs_frame_new) or
/// [`molrs_read_smiles`](crate::molrs_read_smiles).
/// Freed with [`molrs_frame_drop`](crate::molrs_frame_drop).
///
/// # Layout (C)
///
/// ```c
/// typedef struct {
///     uint32_t idx;      /* slot index       */
///     uint32_t version;  /* generation count  */
/// } MolrsFrameHandle;
/// ```
///
/// Both fields are opaque -- C callers should never interpret them
/// directly.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MolrsFrameHandle {
    /// Slot index (opaque).
    pub idx: u32,
    /// Generation counter for stale-handle detection (opaque).
    pub version: u32,
}

/// Opaque handle to a Block (column group) inside a Frame.
///
/// A block handle is obtained from
/// [`molrs_frame_get_block`](crate::molrs_frame_get_block) and
/// carries an embedded [`MolrsFrameHandle`], an interned key identifier,
/// and a version counter for invalidation tracking.
///
/// When a block is mutated through `_mut` accessors, the
/// `block_version` is bumped so that stale handles from earlier reads
/// are detected.
///
/// # Layout (C)
///
/// ```c
/// typedef struct {
///     MolrsFrameHandle frame;     /* owning frame       */
///     uint32_t         key_id;    /* interned block key  */
///     uint64_t         block_version; /* version counter */
/// } MolrsBlockHandle;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MolrsBlockHandle {
    /// Handle of the frame that owns this block.
    pub frame: MolrsFrameHandle,
    /// Interned key identifier (see [`molrs_intern_key`](crate::molrs_intern_key)).
    pub key_id: u32,
    /// Version counter; bumped on mutation for stale-handle detection.
    pub block_version: u64,
}

/// Opaque handle to a Box (simulation cell) in the handle registry.
///
/// Obtained from [`molrs_box_new`](crate::molrs_box_new) or
/// [`molrs_frame_get_box`](crate::molrs_frame_get_box).
/// Freed with [`molrs_box_drop`](crate::molrs_box_drop).
///
/// # Layout (C)
///
/// ```c
/// typedef struct {
///     uint32_t idx;
///     uint32_t version;
/// } MolrsBoxHandle;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MolrsBoxHandle {
    /// Slot index (opaque).
    pub idx: u32,
    /// Generation counter (opaque).
    pub version: u32,
}

/// Opaque handle to a Region in the handle registry.
///
/// Obtained from any `molrs_region_*` constructor, or from a boolean
/// composition of two regions. Freed with
/// [`molrs_region_drop`](crate::molrs_region_drop).
///
/// # Layout (C)
///
/// ```c
/// typedef struct {
///     uint32_t idx;
///     uint32_t version;
/// } MolrsRegionHandle;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MolrsRegionHandle {
    /// Slot index (opaque).
    pub idx: u32,
    /// Generation counter (opaque).
    pub version: u32,
}

/// Opaque handle to a ForceField in the handle registry.
///
/// Obtained from [`molrs_forcefield_new`](crate::molrs_forcefield_new) or
/// [`molrs_forcefield_from_json`](crate::molrs_forcefield_from_json).
/// Freed with [`molrs_forcefield_drop`](crate::molrs_forcefield_drop).
///
/// # Layout (C)
///
/// ```c
/// typedef struct {
///     uint32_t idx;
///     uint32_t version;
/// } MolrsForceFieldHandle;
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MolrsForceFieldHandle {
    /// Slot index (opaque).
    pub idx: u32,
    /// Generation counter (opaque).
    pub version: u32,
}

// --- Conversion: FrameId ↔ MolrsFrameHandle ---

pub(crate) fn frame_id_to_handle(id: FrameId) -> MolrsFrameHandle {
    let ffi = id.data().as_ffi();
    MolrsFrameHandle {
        idx: ffi as u32,
        version: (ffi >> 32) as u32,
    }
}

pub(crate) fn handle_to_frame_id(h: MolrsFrameHandle) -> FrameId {
    let ffi = (h.version as u64) << 32 | h.idx as u64;
    FrameId::from(slotmap::KeyData::from_ffi(ffi))
}

// --- Conversion: BoxKey ↔ MolrsBoxHandle ---

pub(crate) fn box_key_to_handle(key: BoxKey) -> MolrsBoxHandle {
    let ffi = key.data().as_ffi();
    MolrsBoxHandle {
        idx: ffi as u32,
        version: (ffi >> 32) as u32,
    }
}

pub(crate) fn handle_to_box_key(h: MolrsBoxHandle) -> BoxKey {
    let ffi = (h.version as u64) << 32 | h.idx as u64;
    BoxKey::from(slotmap::KeyData::from_ffi(ffi))
}

// --- Conversion: ForceFieldKey ↔ MolrsForceFieldHandle ---

pub(crate) fn forcefield_key_to_handle(key: ForceFieldKey) -> MolrsForceFieldHandle {
    let ffi = key.data().as_ffi();
    MolrsForceFieldHandle {
        idx: ffi as u32,
        version: (ffi >> 32) as u32,
    }
}

pub(crate) fn handle_to_forcefield_key(h: MolrsForceFieldHandle) -> ForceFieldKey {
    let ffi = (h.version as u64) << 32 | h.idx as u64;
    ForceFieldKey::from(slotmap::KeyData::from_ffi(ffi))
}

// --- Conversion: BlockHandle ↔ MolrsBlockHandle ---

/// Convert a Rust `BlockHandle` to a C `MolrsBlockHandle`.
///
/// Requires the HandleRegistry to look up the interned key_id.
pub(crate) fn block_handle_to_c(
    bh: &BlockHandle,
    key_to_id: &std::collections::HashMap<String, u32>,
) -> Option<MolrsBlockHandle> {
    let key_id = key_to_id.get(bh.key())?;
    Some(MolrsBlockHandle {
        frame: frame_id_to_handle(bh.frame_id()),
        key_id: *key_id,
        block_version: bh.version(),
    })
}

/// Convert a C `MolrsBlockHandle` to a Rust `BlockHandle`.
///
/// Requires the HandleRegistry to look up the key string from key_id.
pub(crate) fn c_to_block_handle(
    ch: &MolrsBlockHandle,
    interned_keys: &[std::ffi::CString],
) -> Option<BlockHandle> {
    let key_cstr = interned_keys.get(ch.key_id as usize)?;
    let key_str = key_cstr.to_str().ok()?;
    let frame_id = handle_to_frame_id(ch.frame);
    Some(BlockHandle::new(
        frame_id,
        key_str.to_owned(),
        ch.block_version,
    ))
}

pub(crate) fn region_key_to_handle(key: RegionKey) -> MolrsRegionHandle {
    let ffi = key.data().as_ffi();
    MolrsRegionHandle {
        idx: ffi as u32,
        version: (ffi >> 32) as u32,
    }
}

pub(crate) fn handle_to_region_key(h: MolrsRegionHandle) -> RegionKey {
    let ffi = (h.version as u64) << 32 | h.idx as u64;
    RegionKey::from(slotmap::KeyData::from_ffi(ffi))
}
