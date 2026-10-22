//! Handle types for stable cross-language references.

use slotmap::new_key_type;

// Define a new key type for frames using slotmap
new_key_type! {
    /// Stable identifier for a frame in a [`crate::FrameArena`].
    ///
    /// Structure: (index: u32, generation: u32)
    /// - index: slot in the arena
    /// - generation: invalidation counter (detects use-after-free)
    ///
    /// Properties:
    /// - Copy, comparable, FFI-safe
    /// - Valid until `FrameArena::drop_frame` is called
    /// - JS can represent without precision loss (two u32 instead of u64)
    pub struct FrameId;
}

/// View handle that refers to a block inside a frame.
///
/// Structure: (frame_id, key)
///
/// Becomes invalid when:
/// - Frame is dropped
/// - Block is removed via `FrameArena::del_block`
/// - Block is replaced via `FrameArena::set_block`
/// - Frame is cleared
///
/// Not resurrected if same key is reinserted later.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BlockHandle {
    pub(crate) frame_id: FrameId,
    pub(crate) key: String,
    pub(crate) version: u64,
}

impl BlockHandle {
    pub fn new(frame_id: FrameId, key: String, version: u64) -> Self {
        Self {
            frame_id,
            key,
            version,
        }
    }

    /// Returns the frame this handle belongs to.
    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    /// Returns the block key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the block version.
    pub fn version(&self) -> u64 {
        self.version
    }
}
