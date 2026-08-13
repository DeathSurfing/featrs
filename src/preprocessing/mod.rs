//! Data preprocessing transformations.
//!
//! Analogous to `sklearn.preprocessing`. Each sub-module provides a transformer
//! that implements [`Fit`](crate::traits::Fit) and [`Transform`](crate::traits::Transform)
//! and operates on [`DataFrame`](polars::prelude::DataFrame).

pub mod auto_type;
pub mod binarizer;
pub mod constant_column_remover;
pub mod duplicate_column_remover;
pub mod encoder;
pub mod feature_hasher;
pub mod imputer;
pub mod interaction_features;
pub mod log_transformer;
pub mod missing_indicator;
pub mod normalizer;
pub mod polynomial_features;
/// Quantile-based transformation to uniform or normal distribution.
pub mod quantile_transformer;
pub mod ratio_features;
pub mod scaler;
pub mod string_cleaner;
