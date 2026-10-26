//! Chunk/shard planning (ports molrec chunking.py::plan).
//!
//! Ported verbatim from molrec `src/molrec/chunking.py`, constants included:
//! `TARGET_CHUNK_BYTES = 512 * 1024` (:30) and `SHARD_ABOVE = 4` (:33). Both
//! implementations must size the same array the same way, so the arithmetic
//! below is molrec's, expression for expression, not an independent derivation.

/// Bytes one chunk aims for — roughly a quarter to one megabyte. Smaller and
/// per-chunk overhead dominates; larger and every read drags a whole chunk
/// through the codec.
pub(in crate::io::mrec::zarr_storage) const TARGET_CHUNK_BYTES: u64 = 512 * 1024;

/// Beyond this many chunks, collapse them into one shard file.
const SHARD_ABOVE: u64 = 4;

/// How an array should be cut up: an inner chunk extent and, above
/// [`SHARD_ABOVE`] chunks, a shard extent that packs them into one file.
///
/// Either field may be `None`, and they are two `Option<Vec<u64>>` of the same
/// type — naming them is what keeps a call site from swapping them silently.
/// `chunks` is `None` when no byte target is computable (see [`plan`]), and
/// `shards` is `None` when the array is few enough chunks to leave as files.
pub(in crate::io::mrec::zarr_storage) struct ChunkPlan {
    /// Inner chunk extent, chunked on the leading axis only.
    pub(in crate::io::mrec::zarr_storage) chunks: Option<Vec<u64>>,
    /// Shard extent spanning the whole array, or `None` for no sharding.
    pub(in crate::io::mrec::zarr_storage) shards: Option<Vec<u64>>,
}

/// Size the chunks (and possibly one shard) of a fixed-size array of `shape`
/// whose elements are `itemsize` bytes wide.
///
/// An `itemsize` of `None` means a variable-width dtype ([`DType::String`]),
/// where a byte target is not computable — the choice is left to the backend,
/// as it is for an empty leading axis or a scalar shape.
///
/// Only the leading axis is chunked: trailing axes are per-entity structure
/// (the three components of a coordinate), and splitting them makes a single
/// entity span several chunks.
///
/// [`DType::String`]: molrs::core::DType::String
pub(in crate::io::mrec::zarr_storage) fn plan(shape: &[u64], itemsize: Option<usize>) -> ChunkPlan {
    let no_plan = ChunkPlan {
        chunks: None,
        shards: None,
    };
    let (Some(&rows_total), Some(itemsize)) = (shape.first(), itemsize) else {
        return no_plan;
    };
    if rows_total == 0 {
        return no_plan;
    }
    let trailing_shape = &shape[1..];

    // Saturating throughout: molrec computes in Python's arbitrary-precision
    // integers, and saturating at `u64::MAX` reproduces its answer for any
    // shape that saturates (a row wider than the target yields one row per
    // chunk either way).
    let trailing = trailing_shape
        .iter()
        .fold(1u64, |acc, &n| acc.saturating_mul(n));
    let row_bytes = trailing.saturating_mul(itemsize as u64).max(1);

    let rows = rows_total.min((TARGET_CHUNK_BYTES / row_bytes).max(1));
    let with_trailing = |leading: u64| {
        let mut extent = Vec::with_capacity(shape.len());
        extent.push(leading);
        extent.extend_from_slice(trailing_shape);
        extent
    };
    let chunks = Some(with_trailing(rows));

    let chunk_count = rows_total.div_ceil(rows);
    if chunk_count <= SHARD_ABOVE {
        return ChunkPlan {
            chunks,
            shards: None,
        };
    }
    // A shard must be a whole multiple of its inner chunk.
    ChunkPlan {
        chunks,
        shards: Some(with_trailing(rows.saturating_mul(chunk_count))),
    }
}

#[cfg(test)]
mod tests {
    use super::{ChunkPlan, plan};

    /// Every expectation below was produced offline by molrec itself, so the
    /// numbers are its output rather than a re-derivation of its arithmetic:
    ///
    /// ```text
    /// $ uv --directory ../molrec run --no-sync python -c "
    /// from molrec.chunking import plan
    /// print(plan(SHAPE, ITEMSIZE))"
    /// ```
    ///
    /// Run 2026-08-29 against molrec 0.1.0 (git 15cf50f, CPython 3.14.5), whose
    /// `src/molrec/chunking.py` fixes `TARGET_CHUNK_BYTES = 512 * 1024` (:30)
    /// and `SHARD_ABOVE = 4` (:33). molrec's `plan` returns a bare
    /// `(chunks, shards)` tuple; the port returns the named [`ChunkPlan`], so
    /// the first tuple slot is `chunks` and the second is `shards`.
    ///
    /// | shape | itemsize | chunks | shards |
    /// |---|---|---|---|
    /// | `(100, 3)` | 8 | `(100, 3)` | `None` |
    /// | `(10000, 32, 3)` | 8 | `(682, 32, 3)` | `(10230, 32, 3)` |
    /// | `(262144,)` | 8 | `(65536,)` | `None` |
    /// | `(262145,)` | 8 | `(65536,)` | `(327680,)` |
    /// | `(100, 3)` | `None` | `None` | `None` |
    /// | `(0, 3)` | 8 | `None` | `None` |
    /// | `()` | 8 | `None` | `None` |
    /// | `(10, 1000000)` | 8 | `(1, 1000000)` | `(10, 1000000)` |
    fn assert_plan(plan: ChunkPlan, chunks: Option<&[u64]>, shards: Option<&[u64]>) {
        assert_eq!(plan.chunks.as_deref(), chunks, "chunks");
        assert_eq!(plan.shards.as_deref(), shards, "shards");
    }

    /// An array smaller than one chunk is one whole chunk, and one chunk is
    /// never worth a shard file.
    #[test]
    fn an_array_under_the_byte_target_is_a_single_unsharded_chunk() {
        // molrec: plan((100, 3), 8) -> ((100, 3), None)
        assert_plan(plan(&[100, 3], Some(8)), Some(&[100, 3]), None);
    }

    /// Past the threshold the chunks collapse into one shard that spans the
    /// whole array, rounded up to a whole multiple of the inner chunk.
    #[test]
    fn an_array_over_the_threshold_gets_one_shard_spanning_all_of_it() {
        // 96 trailing elements x 8 B = 768 B per row, so 524288 // 768 = 682
        // rows per chunk and ceil(10000 / 682) = 15 chunks.
        // molrec: plan((10000, 32, 3), 8) -> ((682, 32, 3), (10230, 32, 3))
        assert_plan(
            plan(&[10_000, 32, 3], Some(8)),
            Some(&[682, 32, 3]),
            Some(&[10_230, 32, 3]),
        );
    }

    /// `SHARD_ABOVE` is inclusive on the low side: exactly four chunks stay as
    /// four files.
    #[test]
    fn exactly_shard_above_chunks_stay_unsharded() {
        // 8 B rows => 65536 rows per chunk; 262144 = 4 * 65536 chunks exactly.
        // molrec: plan((262144,), 8) -> ((65536,), None)
        assert_plan(plan(&[262_144], Some(8)), Some(&[65_536]), None);
    }

    /// One row past the previous case is one chunk past the threshold, and the
    /// shard appears.
    #[test]
    fn one_row_past_shard_above_collapses_into_a_shard() {
        // ceil(262145 / 65536) = 5 > SHARD_ABOVE, so shards = 5 * 65536.
        // molrec: plan((262145,), 8) -> ((65536,), (327680,))
        assert_plan(plan(&[262_145], Some(8)), Some(&[65_536]), Some(&[327_680]));
    }

    /// A variable-width dtype (`DType::String`, whose `itemsize()` is `None`)
    /// has no computable byte target, so the choice is left to the backend.
    #[test]
    fn a_variable_width_dtype_yields_no_plan() {
        // molrec: plan((100, 3), None) -> (None, None)
        assert_plan(plan(&[100, 3], None), None, None);
    }

    /// Every sequence array is created at `[0, ...]`, so the empty leading axis
    /// is the shape the writer plans first; molrec declines to size it.
    #[test]
    fn an_empty_leading_axis_yields_no_plan() {
        // molrec: plan((0, 3), 8) -> (None, None)
        assert_plan(plan(&[0, 3], Some(8)), None, None);
    }

    /// A scalar has no leading axis to chunk.
    #[test]
    fn a_scalar_shape_yields_no_plan() {
        // molrec: plan((), 8) -> (None, None)
        assert_plan(plan(&[], Some(8)), None, None);
    }

    /// A single row wider than the whole byte target still gets a chunk: the
    /// rows-per-chunk floor is one, never zero.
    #[test]
    fn a_row_wider_than_the_byte_target_still_gets_one_row_per_chunk() {
        // Row is 8 MB, so 524288 // 8000000 = 0 and the max(1, ...) floor bites;
        // 10 chunks then clears SHARD_ABOVE.
        // molrec: plan((10, 1000000), 8) -> ((1, 1000000), (10, 1000000))
        assert_plan(
            plan(&[10, 1_000_000], Some(8)),
            Some(&[1, 1_000_000]),
            Some(&[10, 1_000_000]),
        );
    }
}
