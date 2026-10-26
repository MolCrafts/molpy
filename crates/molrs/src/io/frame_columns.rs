//! The 1-D columns a reader builds a Frame block from: one insert, for every
//! dtype, mapping a [`BlockError`](crate::core::BlockError) to the readers'
//! one `InvalidData` error ([`invalid_data`]).

use ndarray::{Array1, ArrayD, IxDyn};

use crate::core::Block;
use crate::core::BlockDtype;
use crate::io::invalid_data;

/// Insert `values` as the 1-D column `key` of `block`; its length is the
/// column's row count.
pub(crate) fn insert_column<T: BlockDtype>(
    block: &mut Block,
    key: &str,
    values: Vec<T>,
) -> std::io::Result<()> {
    block
        .insert(key, Array1::from_vec(values).into_dyn(), None)
        .map_err(invalid_data)
}

/// Insert `values` as the 1-D column `key` of `block`, which the file says
/// holds `n_rows` rows: a column of any other length is an error, even when it
/// is the block's first.
pub(crate) fn insert_column_of_length<T: BlockDtype>(
    block: &mut Block,
    key: &str,
    values: Vec<T>,
    n_rows: usize,
) -> std::io::Result<()> {
    let column = ArrayD::from_shape_vec(IxDyn(&[n_rows]), values).map_err(invalid_data)?;
    block.insert(key, column, None).map_err(invalid_data)
}
