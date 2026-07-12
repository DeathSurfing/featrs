//! Feature binarization.
//!
//! [`Binarizer`] thresholds numeric features: values above the threshold
//! become `1.0`, others become `0.0`. `null` values are preserved as-is;
//! `NaN` values are also propagated unchanged.
//!
//! Implements [`TransformLazy`] with a
//! `when/then/otherwise` Polars expression, so it integrates into lazy
//! pipelines without materializing intermediate `DataFrame`s.

use crate::traits::{Error, Fit, FitLazy, Result, Transform, TransformLazy};
use crate::util::replace_f64_column;
use polars::prelude::*;

/// Binarize data according to a threshold.
///
/// Values `> threshold` become `1.0`; values `<= threshold` become `0.0`.
/// `null` values are preserved as `null`; `NaN` values are propagated as `NaN`.
///
/// Implements [`TransformLazy`] with a
/// `when/then/otherwise` Polars expression that handles `null` and `NaN`
/// propagation without materializing the `LazyFrame`.
///
/// # Example
///
/// ```rust
/// use featrs::preprocessing::binarizer::Binarizer;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let col = Column::from(Series::new("x".into(), &[-1.0_f64, 0.5, 2.0]));
/// let df = DataFrame::new(3, vec![col])?;
///
/// let mut b = Binarizer::new(0.5);
/// b.fit(df.clone())?;
/// let binarized = b.transform(df)?;
/// assert_eq!(binarized.height(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Binarizer {
    fitted: bool,
    threshold: f64,
}

impl Binarizer {
    /// Create a new `Binarizer` with the given threshold.
    ///
    /// Values strictly greater than `threshold` are set to `1.0`.
    pub fn new(threshold: f64) -> Self {
        Self {
            fitted: false,
            threshold,
        }
    }
}

impl Default for Binarizer {
    /// Create a `Binarizer` with a default threshold of `0.0`.
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Fit<DataFrame> for Binarizer {
    type Output = ();

    /// Fit the binarizer by validating the input DataFrame has
    /// at least one row and one column.
    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.height() == 0 || x.width() == 0 {
            return Err(Error::InvalidInput(
                "Binarizer.fit received an empty DataFrame (0 rows or 0 columns). \
                 Provide data with at least 1 row and 1 column."
                    .into(),
            ));
        }
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for Binarizer {
    type Output = DataFrame;

    /// Binarize the data: values above the threshold become `1.0`,
    /// others become `0.0`.
    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Binarizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let col_names: Vec<String> = x
            .get_column_names()
            .iter()
            .filter_map(|name| {
                let col = x.column(name).ok()?;
                if col.dtype() == &DataType::Float64 {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .collect();

        let mut out = x.clone();
        let threshold = self.threshold;

        for name in &col_names {
            replace_f64_column(&mut out, name.as_str(), "Binarizer", |v| {
                if v.is_nan() {
                    v
                } else if v > threshold {
                    1.0
                } else {
                    0.0
                }
            })?;
        }

        Ok(out)
    }
}

impl FitLazy for Binarizer {}

impl TransformLazy for Binarizer {
    fn transform_lazy(&self, mut x: LazyFrame) -> Result<LazyFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "Binarizer has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }

        let schema = x
            .collect_schema()
            .map_err(|e| Error::Computation(e.to_string()))?;
        let mut exprs = Vec::new();
        let threshold = self.threshold;

        for (name, dtype) in schema.iter() {
            if dtype == &DataType::Float64 {
                let name_str = name.as_str();
                exprs.push(
                    when(col(name_str).is_null())
                        .then(lit(Null {}))
                        .otherwise(
                            when(col(name_str).is_nan()).then(lit(f64::NAN)).otherwise(
                                when(col(name_str).gt(lit(threshold)))
                                    .then(lit(1.0))
                                    .otherwise(lit(0.0)),
                            ),
                        )
                        .alias(name_str),
                );
            }
        }

        Ok(x.with_columns(exprs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_binarizer_default() {
        let mut b = Binarizer::default();
        let a = Column::from(Series::new("x".into(), &[-1.0f64, 0.0, 2.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_custom_threshold() {
        let mut b = Binarizer::new(5.0);
        let a = Column::from(Series::new("x".into(), &[1.0f64, 5.0, 10.0]));
        let df = DataFrame::new(3, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let result = b.transform(df).unwrap();

        let vals: Vec<f64> = result
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .flatten()
            .collect();
        assert_relative_eq!(vals[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vals[2], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_binarizer_empty_rows_rejected() {
        let a = Column::from(Series::new("x".into(), Vec::<f64>::new()));
        let df = DataFrame::new(0, vec![a]).unwrap();

        let mut b = Binarizer::new(0.5);
        let result = b.fit(df);
        assert!(
            result.is_err(),
            "a 0-row DataFrame should be rejected at fit time"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("empty DataFrame"),
            "error message should mention the empty DataFrame"
        );
    }

    #[test]
    fn test_lazy_binarizer() {
        let mut b = Binarizer::new(0.5);
        let a = Column::from(Series::new("x".into(), &[-1.0f64, 0.5, 2.0, f64::NAN]));
        let df = DataFrame::new(4, vec![a]).unwrap();

        b.fit(df.clone()).unwrap();
        let eager_out = b.transform(df.clone()).unwrap();
        let lazy_out = b
            .transform_lazy(df.clone().lazy())
            .unwrap()
            .collect()
            .unwrap();

        let eager_vals: Vec<Option<f64>> = eager_out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect();
        let lazy_vals: Vec<Option<f64>> = lazy_out
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .collect();

        assert_eq!(eager_vals.len(), lazy_vals.len());
        for i in 0..eager_vals.len() {
            match (eager_vals[i], lazy_vals[i]) {
                (Some(ev), Some(lv)) => {
                    if ev.is_nan() {
                        assert!(lv.is_nan());
                    } else {
                        assert_relative_eq!(ev, lv, epsilon = 1e-6);
                    }
                }
                (None, None) => {}
                _ => panic!(
                    "Mismatch at index {}: eager={:?}, lazy={:?}",
                    i, eager_vals[i], lazy_vals[i]
                ),
            }
        }
    }
}
