//! Matrix decomposition of feature matrices: principal component analysis.
//!
//! | Method | Input | Output |
//! |--------|-------|--------|
//! | [`Pca`] | any [`FeatureMatrix`](crate::compute::FeatureMatrix) | [`PcaResult`] — `n_components` scores per row, itself a `FeatureMatrix` |
//!
//! It consumes the descriptor results ([`SteinhardtResult`](crate::compute::SteinhardtResult),
//! [`LocalDescriptorsResult`](crate::compute::LocalDescriptorsResult)) or a
//! plain `Array2`, and feeds [`Kmeans`](crate::compute::Kmeans).

mod pca;

pub use pca::{Pca, PcaInput, PcaResult};
