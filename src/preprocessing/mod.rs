//! Data preprocessing transformations.
//!
//! Analogous to `sklearn.preprocessing`. Each sub-module provides a transformer
//! that implements [`Fit`](crate::traits::Fit) and [`Transform`](crate::traits::Transform)
//! and operates on [`DataFrame`](polars::prelude::DataFrame).

pub mod auto_type;
pub mod binarizer;
pub mod encoder;
pub mod feature_hasher;
pub mod imputer;
pub mod missing_indicator;
pub mod normalizer;
pub mod polynomial_features;
pub mod scaler;
