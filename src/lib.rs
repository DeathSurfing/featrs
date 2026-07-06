//! `featrs` — feature engineering for Rust, inspired by scikit-learn.
//!
//! Built on [Polars](https://pola.rs), all transformations operate on
//! `DataFrame` and preserve column names throughout.
//!
//! # Modules
//!
//! | Module | Description |
//! |---|---|
//! | [`preprocessing`] | Scaling, encoding, normalization, imputation, binarization, polynomial features |
//! | [`pipeline`] | `Pipeline` (sequential) and `ColumnTransformer` (per-column transforms) |
//! | [`feature_selection`] | `VarianceThreshold`, `SelectKBest` with ANOVA F-value scoring |
//! | [`traits`] | Core `Fit`, `Transform`, `FitTransform` traits and error types |
//!
//! # Quick start
//!
//! ```rust,ignore
//! use featrs::preprocessing::scaler::StandardScaler;
//! use featrs::traits::{Fit, Transform};
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(df.clone(), target)?;
//! let scaled = scaler.transform(df)?;
//! ```

pub mod feature_selection;
pub mod pipeline;
pub mod preprocessing;
pub mod traits;
