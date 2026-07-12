//! Pipeline composition utilities.
//!
//! Analogous to `sklearn.pipeline` and `sklearn.compose`.
//! - [`Pipeline`] chains multiple transformers sequentially.
//! - [`ColumnTransformer`] applies different transformers to different column subsets.
//! - [`DataFrameTransformer`] is a trait alias for type erasure.
//!
//! Both [`Pipeline`] and [`ColumnTransformer`] implement [`FitLazy`] and
//! [`TransformLazy`], so an entire pipeline can be expressed as a Polars
//! query plan and executed in a single optimized pass. Steps that do not
//! provide a custom lazy implementation degrade gracefully to eager execution.

pub mod column_transformer;

pub use column_transformer::ColumnTransformer;

use crate::traits::{Error, Fit, FitLazy, Result, Transform, TransformLazy};
use polars::prelude::*;

/// Trait alias for [`Box<dyn ...>`](Box) type erasure in [`Pipeline`] and [`ColumnTransformer`].
///
/// Automatically implemented for any type that satisfies both
/// [`Fit<DataFrame, Output = ()>`](crate::traits::Fit) and
/// [`Transform<DataFrame, Output = DataFrame>`](crate::traits::Transform),
/// plus [`FitLazy`] and [`TransformLazy`].
pub trait DataFrameTransformer:
    Fit<DataFrame, Output = ()> + Transform<DataFrame, Output = DataFrame> + FitLazy + TransformLazy
{
}
impl<T> DataFrameTransformer for T where
    T: Fit<DataFrame, Output = ()>
        + Transform<DataFrame, Output = DataFrame>
        + FitLazy
        + TransformLazy
{
}

/// Sequential pipeline of data transformations.
///
/// Each step is `(name, transformer)`. Calling `fit(X)` fits all steps
/// sequentially (passing each step's output into the next). Calling
/// `transform(X)` passes data through every step.
///
/// # Eager example
///
/// ```rust
/// use featrs::pipeline::Pipeline;
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut pipeline = Pipeline::new(vec![
///     ("scale".into(), Box::new(StandardScaler::new())),
/// ])?;
/// pipeline.fit(df.clone())?;
/// let result = pipeline.transform(df)?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Lazy example
///
/// ```rust
/// use featrs::pipeline::Pipeline;
/// use featrs::preprocessing::scaler::StandardScaler;
/// use featrs::traits::{FitLazy, TransformLazy};
/// use polars::prelude::{Column, DataFrame, IntoLazy, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[1.0_f64, 2.0, 3.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut pipeline = Pipeline::new(vec![
///     ("scale".into(), Box::new(StandardScaler::new())),
/// ])?;
/// pipeline.fit_lazy(df.clone().lazy())?;
/// let result = pipeline.transform_lazy(df.lazy())?.collect()?;
/// assert_eq!(result.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Pipeline {
    steps: Vec<(String, Box<dyn DataFrameTransformer>)>,
}

impl Pipeline {
    /// Create a new pipeline with the given steps.
    ///
    /// Each step is a `(name, transformer)` pair. The name is used for
    /// inspection and debugging.
    ///
    /// Returns [`Error::InvalidInput`] if `steps` is empty.
    pub fn new(steps: Vec<(String, Box<dyn DataFrameTransformer>)>) -> Result<Self> {
        if steps.is_empty() {
            return Err(Error::InvalidInput(
                "Pipeline::new: at least one step is required. \
                 Provide a non-empty Vec of (name, transformer) pairs."
                    .into(),
            ));
        }
        Ok(Self { steps })
    }

    /// Returns a reference to the pipeline steps.
    pub fn steps(&self) -> &[(String, Box<dyn DataFrameTransformer>)] {
        &self.steps
    }
}

impl Fit<DataFrame> for Pipeline {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "Pipeline.fit received a DataFrame with 0 rows.".into(),
            ));
        }
        let mut x_curr = x;
        let n = self.steps.len();
        for (i, (name, transformer)) in self.steps.iter_mut().enumerate() {
            let is_last = i == n - 1;
            transformer.fit(x_curr.clone()).map_err(|e| {
                Error::Computation(format!(
                    "Pipeline: step {} ('{}') failed during fit: {}",
                    i, name, e
                ))
            })?;
            if !is_last {
                x_curr = transformer.transform(x_curr).map_err(|e| {
                    Error::Computation(format!(
                        "Pipeline: step {} ('{}') failed during intermediate transform: {}",
                        i, name, e
                    ))
                })?;
            }
        }
        Ok(())
    }
}

impl Transform<DataFrame> for Pipeline {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        let mut x_curr = x;
        for (i, (name, transformer)) in self.steps.iter().enumerate() {
            x_curr = transformer.transform(x_curr).map_err(|e| {
                Error::Computation(format!(
                    "Pipeline: step {} ('{}') failed during transform: {}",
                    i, name, e
                ))
            })?;
        }
        Ok(x_curr)
    }
}

impl FitLazy for Pipeline {
    fn fit_lazy(&mut self, x: LazyFrame) -> Result<()> {
        if self.steps.is_empty() {
            return Err(Error::InvalidInput(
                "Pipeline.fit received an empty steps list.".into(),
            ));
        }
        let mut x_curr = x;
        let n = self.steps.len();
        for (i, (name, transformer)) in self.steps.iter_mut().enumerate() {
            let is_last = i == n - 1;
            transformer.fit_lazy(x_curr.clone()).map_err(|e| {
                Error::Computation(format!(
                    "Pipeline: step {} ('{}') failed during lazy fit: {}",
                    i, name, e
                ))
            })?;
            if !is_last {
                x_curr = transformer.transform_lazy(x_curr).map_err(|e| {
                    Error::Computation(format!(
                        "Pipeline: step {} ('{}') failed during intermediate lazy transform: {}",
                        i, name, e
                    ))
                })?;
            }
        }
        Ok(())
    }
}

impl TransformLazy for Pipeline {
    fn transform_lazy(&self, x: LazyFrame) -> Result<LazyFrame> {
        let mut x_curr = x;
        for (i, (name, transformer)) in self.steps.iter().enumerate() {
            x_curr = transformer.transform_lazy(x_curr).map_err(|e| {
                Error::Computation(format!(
                    "Pipeline: step {} ('{}') failed during lazy transform: {}",
                    i, name, e
                ))
            })?;
        }
        Ok(x_curr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::binarizer::Binarizer;
    use crate::preprocessing::scaler::StandardScaler;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("a".into(), &[1.0f64, 3.0, 5.0]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 4.0, 6.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_pipeline_single_step() {
        let scaler = StandardScaler::new();
        let mut pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();

        pipeline.fit(df.clone()).unwrap();
        let result = pipeline.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_pipeline_multi_step() {
        let scaler = StandardScaler::new();
        let binarizer = Binarizer::new(0.0);
        let mut pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(scaler)),
            ("binarizer".into(), Box::new(binarizer)),
        ])
        .unwrap();
        let df = make_test_df();

        pipeline.fit(df.clone()).unwrap();
        let result = pipeline.transform(df).unwrap();

        assert_eq!(result.width(), 2);
        assert_eq!(result.height(), 3);
    }

    #[test]
    fn test_pipeline_empty_steps_error() {
        let result = Pipeline::new(vec![]);
        assert!(result.is_err(), "Pipeline::new must reject empty steps");
    }

    #[test]
    fn test_pipeline_not_fitted() {
        let scaler = StandardScaler::new();
        let pipeline = Pipeline::new(vec![("scaler".into(), Box::new(scaler))]).unwrap();
        let df = make_test_df();
        assert!(pipeline.transform(df).is_err());
    }

    #[test]
    fn test_pipeline_lazy() {
        let scaler = StandardScaler::new();
        let binarizer = Binarizer::new(0.0);
        let mut pipeline = Pipeline::new(vec![
            ("scaler".into(), Box::new(scaler)),
            ("binarizer".into(), Box::new(binarizer)),
        ])
        .unwrap();
        let df = make_test_df();

        pipeline.fit_lazy(df.clone().lazy()).unwrap();
        let eager_out = pipeline.transform(df.clone()).unwrap();
        let lazy_out = pipeline
            .transform_lazy(df.clone().lazy())
            .unwrap()
            .collect()
            .unwrap();

        assert_eq!(eager_out, lazy_out);
    }
}
