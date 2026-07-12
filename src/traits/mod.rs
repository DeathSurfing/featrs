//! Core traits and error types for the featrs library.
//!
//! The library is built around six traits that mirror the scikit-learn API:
//!
//! - [`Fit`] — learn parameters from unsupervised data (`fit(X)`)
//! - [`FitSupervised`] — learn parameters from data plus a target (`fit(X, y)`)
//! - [`Transform`] — apply a learned transformation (`transform(X)`)
//! - [`FitTransform`] — convenience trait providing a default `fit_transform(X)` for any type
//!   implementing both [`Fit`] and [`Transform`]
//! - [`FitLazy`] — learn parameters from a Polars [`LazyFrame`] (defaults to collecting
//!   and calling [`Fit`]; override for zero-copy lazy fit)
//! - [`TransformLazy`] — apply a learned transformation to a [`LazyFrame`] (defaults to
//!   collecting and calling [`Transform`]; override to add lazy expressions to the plan
//!   without materializing data)
//!
//! Unsupervised transformers (scalers, encoders, imputers, …) implement
//! [`Fit`]; the few supervised ones (e.g. `SelectKBest`) implement
//! [`FitSupervised`] instead. This mirrors scikit-learn's split between
//! `fit(X)` and `fit(X, y)`, so callers no longer pass a dummy target to
//! unsupervised transformers.
//!
//! # Lazy execution
//!
//! Using [`FitLazy`] and [`TransformLazy`] allows pipelines to be expressed
//! as Polars query plans. Polars can then fuse multiple column expressions,
//! apply predicate pushdown, and execute everything in a single optimized
//! pass when `.collect()` is called. Transformers that do not provide a custom
//! override still work correctly — they simply collect the `LazyFrame` eagerly,
//! apply their regular `transform`, and re-lazify the result.
//!
//! # Errors
//!
//! All fallible operations return [`Result<T>`], which wraps [`enum@Error`].
//! [`enum@Error`] has three variants:
//! - [`Error::InvalidInput`] — wrong dimensions, types, or empty data
//! - [`Error::NotFitted`] — `transform` called before `fit`
//! - [`Error::Computation`] — numerical issues (zero variance, singular matrices, etc.)

use thiserror::Error;

/// Errors that can occur during feature engineering operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Input data has invalid shape, type, or is empty.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// `transform` was called before `fit`.
    #[error("not fitted: {0}")]
    NotFitted(String),

    /// Numerical computation failed (e.g. zero variance, singular matrix).
    #[error("computation error: {0}")]
    Computation(String),
}

/// Convenience alias for `std::result::Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Learn parameters from unsupervised data.
///
/// `X` is the feature data (e.g. `DataFrame`). Implement this trait for
/// transformers that do not need a target (scalers, encoders, imputers, …).
/// Supervised transformers implement [`FitSupervised`] instead.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Fit<X> {
    /// The type returned by `fit`. Usually `()`.
    type Output;

    /// Fit the transformer to the data.
    ///
    /// After calling `fit`, the transformer stores the learned parameters
    /// internally. Calling `transform` before `fit` returns
    /// [`Error::NotFitted`].
    fn fit(&mut self, x: X) -> Result<Self::Output>;
}

/// Learn parameters from data plus a supervised target.
///
/// `X` is the feature data and `Y` is the target data (both typically
/// `DataFrame`). Implement this trait for transformers that need a target,
/// such as [`SelectKBest`](crate::feature_selection::SelectKBest). Unsupervised
/// transformers implement [`Fit`] instead.
pub trait FitSupervised<X, Y> {
    /// The type returned by `fit`. Usually `()`.
    type Output;

    /// Fit the transformer to the data and target.
    ///
    /// After calling `fit`, the transformer stores the learned parameters
    /// internally. Calling `transform` before `fit` returns
    /// [`Error::NotFitted`].
    fn fit(&mut self, x: X, y: Y) -> Result<Self::Output>;
}

/// Apply a learned transformation to data.
///
/// `X` is the input data (e.g. `DataFrame`). The output type [`Output`](Transform::Output)
/// is typically also `DataFrame`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit(df.clone())?;
/// let scaled = scaler.transform(df)?;
/// assert_eq!(scaled.width(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait Transform<X> {
    /// The type of the transformed output.
    type Output;

    /// Transform the data using the fitted parameters.
    ///
    /// Returns [`Error::NotFitted`] if the transformer has not been fitted yet.
    fn transform(&self, x: X) -> Result<Self::Output>;
}

use polars::prelude::{DataFrame, IntoLazy, LazyFrame};

/// Convenience trait for types that implement both [`Fit`] and [`Transform`].
///
/// Automatically implemented for any type satisfying both bounds. Provides a
/// default [`fit_transform`](FitTransform::fit_transform) equivalent to
/// `fit(X)` followed by `transform(X)`; types may override it with an optimized
/// single-pass implementation. Also enables type erasure via
/// [`Box<dyn DataFrameTransformer>`](crate::pipeline::DataFrameTransformer).
pub trait FitTransform<X>: Fit<X> + Transform<X> {
    /// Fit to `x`, then transform `x`, returning the transformed output.
    ///
    /// Default implementation clones `x` so it can be both fit on and
    /// transformed; override for a single-pass implementation when `X` need
    /// not be cloned.
    fn fit_transform(&mut self, x: X) -> Result<<Self as Transform<X>>::Output>
    where
        X: Clone,
    {
        self.fit(x.clone())?;
        self.transform(x)
    }
}

impl<T, X> FitTransform<X> for T where T: Fit<X> + Transform<X> {}

/// Learn parameters from unsupervised data in a lazy (query-planned) manner.
///
/// This is the lazy counterpart to [`Fit`]. The default implementation collects
/// the [`LazyFrame`] into a [`DataFrame`] and delegates to [`Fit::fit`], so any
/// transformer that implements [`Fit`] automatically gets a working `fit_lazy`
/// without any extra code.
///
/// Override this method to keep the fit logic fully within the lazy query plan
/// (e.g. running aggregations via lazy expressions and reading back only the
/// statistics you need), which avoids materializing large intermediate frames.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{FitLazy, TransformLazy};
/// use polars::prelude::{Column, DataFrame, IntoLazy, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut scaler = StandardScaler::new();
/// scaler.fit_lazy(df.clone().lazy())?;
/// let result = scaler.transform_lazy(df.lazy())?.collect()?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub trait FitLazy: Fit<DataFrame, Output = ()> {
    /// Fit the transformer to the lazy data.
    ///
    /// By default, this collects the [`LazyFrame`] and delegates to [`Fit::fit`].
    /// Transformers can override this to run fit logic (or partial collection) on the query plan.
    fn fit_lazy(&mut self, x: LazyFrame) -> Result<()> {
        let df = x.collect().map_err(|e| Error::Computation(e.to_string()))?;
        self.fit(df)
    }
}

/// Apply a learned transformation to a lazy (query-planned) frame.
///
/// This is the lazy counterpart to [`Transform`]. The default implementation
/// collects the [`LazyFrame`] into a [`DataFrame`], applies the eager
/// [`Transform::transform`], and wraps the result back in a `LazyFrame`.
///
/// Override this method to build lazy Polars expressions instead of
/// materializing data. An override should append `with_columns(exprs)` or
/// `select(exprs)` to the incoming `LazyFrame` and return the resulting plan.
/// Polars will then fuse those expressions with the rest of the pipeline and
/// execute everything in a single optimized pass.
///
/// # Lazy-optimized implementations
///
/// The following transformers override `transform_lazy` with expression-based
/// implementations that do not collect the `LazyFrame`:
///
/// | Transformer | Lazy expression |
/// |---|---|
/// | `StandardScaler` | `(col - mean) / std` |
/// | `MinMaxScaler` | `(col - min) * scale + range_min` |
/// | `RobustScaler` | `(col - center) / scale` |
/// | `Binarizer` | `when(col > threshold).then(1.0).otherwise(0.0)` |
/// | `VarianceThreshold` | `select(selected_cols)` |
/// | `SelectKBest` | `select(selected_cols)` |
///
/// All other transformers fall back to the default (collect → eager transform
/// → re-lazify). This means they still work correctly in a lazy pipeline but
/// do not benefit from query-plan fusion.
pub trait TransformLazy: Transform<DataFrame, Output = DataFrame> {
    /// Transform the lazy data using the fitted parameters.
    ///
    /// By default, this collects the [`LazyFrame`], performs the eager transformation
    /// via [`Transform::transform`], and returns the result lazily.
    /// Transformers should override this to append lazy expressions to the `LazyFrame`.
    fn transform_lazy(&self, x: LazyFrame) -> Result<LazyFrame> {
        let df = x.collect().map_err(|e| Error::Computation(e.to_string()))?;
        let transformed = self.transform(df)?;
        Ok(transformed.lazy())
    }
}
