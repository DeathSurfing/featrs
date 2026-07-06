//! Feature selection transformers.
//!
//! Analogous to `sklearn.feature_selection`. Reduce the number of features
//! by removing low-variance columns or selecting the top-k features according
//! to a statistical test.

pub mod select_kbest;
pub mod variance_threshold;

pub use select_kbest::SelectKBest;
pub use variance_threshold::VarianceThreshold;
