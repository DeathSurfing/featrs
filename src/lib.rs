//! `featrs` — feature engineering for Rust, inspired by scikit-learn.
//!
//! Built on [Polars](https://pola.rs), all transformations operate on
//! `DataFrame` and preserve column names throughout.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use featrs::prelude::*;
//!
//! let mut scaler = StandardScaler::new();
//! scaler.fit(df.clone(), target)?;
//! let scaled = scaler.transform(df)?;
//! ```
//!
//! # Modules
//!
//! | Module | Description |
//! |---|---|
//! | [`prelude`] | Convenient glob-import of the most common types |
//! | [`preprocessing`] | Scaling, encoding, normalization, imputation, binarization, polynomial features, feature hashing, auto-type detection |
//! | [`pipeline`] | `Pipeline` (sequential) and `ColumnTransformer` (per-column transforms) |
//! | [`feature_selection`] | `VarianceThreshold`, `SelectKBest` with ANOVA F-value scoring |
//! | [`traits`] | Core `Fit`, `Transform`, `FitTransform` traits and error types |
//! | [`time_series`] | Lag features, rolling windows, difference, cyclical encoding |

pub mod feature_selection;
pub mod pipeline;
pub mod preprocessing;
pub mod time_series;
pub mod traits;

/// Convenient glob import of the most common types.
///
/// ```rust,ignore
/// use featrs::prelude::*;
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
    pub use crate::traits::missing_indicator::MissingIndicator;
    pub use crate::traits::{Error, Fit, FitTransform, Result, Transform};
}

// --- Shallow re-exports at crate root ---

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
pub use crate::traits::missing_indicator::MissingIndicator;
pub use crate::traits::{Error, Fit, FitTransform, Result, Transform};
