//! Core traits and error types for the featrs library.
//!
//! The library is built around three traits that mirror the scikit-learn API:
//!
//! - [`Fit`] — learn parameters from data (`fit`)
//! - [`Transform`] — apply a learned transformation (`transform`)
//! - [`FitTransform`] — convenience blanket trait for types that implement both
//!
//! # Errors
//!
//! All fallible operations return [`Result<T>`], which wraps [`Error`].
//! [`Error`] has three variants:
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

/// Learn parameters from data.
///
/// `X` is the feature data (e.g. `DataFrame`).
/// `Y` is the target data (defaults to `X` for unsupervised transformers).
///
/// # Example
///
/// ```rust
/// use featrs::traits::Fit;
/// # use featrs::traits::Result;
/// # use polars::prelude::*;
///
/// // Every transformer implements Fit. The fitted parameters are stored
/// // on the transformer itself.
/// ```
pub trait Fit<X, Y = X> {
    /// The type returned by `fit`. Usually `()`.
    type Output;

    /// Fit the transformer to the data.
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
/// use featrs::traits::Transform;
/// # use polars::prelude::*;
///
/// // After fitting, call transform to apply the transformation.
/// ```
pub trait Transform<X> {
    /// The type of the transformed output.
    type Output;

    /// Transform the data using the fitted parameters.
    ///
    /// Returns [`Error::NotFitted`] if the transformer has not been fitted yet.
    fn transform(&self, x: X) -> Result<Self::Output>;
}

/// Convenience trait for types that implement both [`Fit`] and [`Transform`].
///
/// This trait is automatically implemented for any type that satisfies
/// both bounds. It is used to enable type erasure with
/// [`Box<dyn DataFrameTransformer>`](crate::pipeline::DataFrameTransformer).
pub trait FitTransform<X, Y = X>: Fit<X, Y> + Transform<X> {}

impl<T, X, Y> FitTransform<X, Y> for T where T: Fit<X, Y> + Transform<X> {}
