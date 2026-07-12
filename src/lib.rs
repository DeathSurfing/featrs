//! `featrs` — feature engineering for Rust, inspired by scikit-learn.
//!
//! Built on [Polars](https://pola.rs), all transformations operate on
//! `DataFrame` and preserve column names throughout. Every transformer also
//! supports Polars [`LazyFrame`](polars::prelude::LazyFrame) through the
//! [`FitLazy`] and
//! [`TransformLazy`] traits, enabling
//! query-plan optimization, predicate pushdown, and streaming for large datasets.
//!
//! # Quick start — eager
//!
//! ```rust
//! use featrs::prelude::*;
//! use polars::prelude::{Column, DataFrame, NamedFrom, Series};
//!
//! let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
//! let df = DataFrame::new(3, vec![col])?;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(df.clone())?;
//! let scaled = scaler.transform(df)?;
//! assert_eq!(scaled.height(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Quick start — lazy
//!
//! ```rust
//! use featrs::prelude::*;
//! use polars::prelude::{Column, DataFrame, IntoLazy, NamedFrom, Series};
//!
//! let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
//! let df = DataFrame::new(3, vec![col])?;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit_lazy(df.clone().lazy())?;
//! let result = scaler.transform_lazy(df.lazy())?.collect()?;
//! assert_eq!(result.height(), 3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Modules
//!
//! | Module | Description |
//! |---|---|
//! | [`prelude`] | Convenient glob-import of the most common types |
//! | [`preprocessing`] | Scaling, encoding, normalization, imputation, binarization, polynomial features, feature hashing, auto-type detection |
//! | [`pipeline`] | `Pipeline` (sequential) and `ColumnTransformer` (per-column transforms) — both support lazy execution |
//! | [`feature_selection`] | `VarianceThreshold`, `SelectKBest` with ANOVA F-value scoring |
//! | [`traits`] | Core `Fit`, `Transform`, `FitTransform`, `FitLazy`, `TransformLazy` traits and error types |
//! | [`time_series`] | Lag features, rolling windows, difference, cyclical encoding |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
// Production code must not `unwrap()`/`expect()` Polars results — route every
// failure through `Error` instead. Tests are exempt.
#![deny(clippy::unwrap_used, clippy::expect_used)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]

pub mod feature_selection;
pub mod pipeline;
pub mod preprocessing;
pub mod time_series;
pub mod traits;
pub mod util;

/// Convenient glob import of the most common types.
///
/// ```rust
/// use featrs::prelude::*;
///
/// let _scaler = StandardScaler::new();
/// ```
pub mod prelude {
    pub use crate::feature_selection::SelectKBest;
    pub use crate::feature_selection::VarianceThreshold;
    pub use crate::feature_selection::select_kbest::FClassif;
    pub use crate::pipeline::ColumnTransformer;
    pub use crate::pipeline::DataFrameTransformer;
    pub use crate::pipeline::Pipeline;
    pub use crate::pipeline::column_transformer::Remainder;
    pub use crate::preprocessing::auto_type::{AutoTypeDetector, ColumnType};
    pub use crate::preprocessing::binarizer::Binarizer;
    pub use crate::preprocessing::encoder::LabelEncoder;
    pub use crate::preprocessing::encoder::OneHotEncoder;
    pub use crate::preprocessing::encoder::OrdinalEncoder;
    pub use crate::preprocessing::feature_hasher::FeatureHasher;
    pub use crate::preprocessing::imputer::SimpleImputer;
    pub use crate::preprocessing::imputer::Strategy;
    pub use crate::preprocessing::missing_indicator::MissingIndicator;
    pub use crate::preprocessing::normalizer::Norm;
    pub use crate::preprocessing::normalizer::Normalizer;
    pub use crate::preprocessing::polynomial_features::PolynomialFeatures;
    pub use crate::preprocessing::polynomial_features::PolynomialFeaturesBuilder;
    pub use crate::preprocessing::scaler::MinMaxScaler;
    pub use crate::preprocessing::scaler::RobustScaler;
    pub use crate::preprocessing::scaler::StandardScaler;
    pub use crate::time_series::cyclical::CyclicalEncoder;
    pub use crate::time_series::diff::Difference;
    pub use crate::time_series::lag::Lagger;
    pub use crate::time_series::rolling::RollingAggregator;
    pub use crate::traits::{
        Error, Fit, FitLazy, FitSupervised, FitTransform, Result, Transform, TransformLazy,
    };
}

// --- Shallow re-exports at crate root ---
// The canonical list lives in `prelude`; the root re-exports it so that
// `featrs::StandardScaler` and `featrs::prelude::StandardScaler` both work.
// Add new public types to `prelude` only.
pub use crate::prelude::*;
