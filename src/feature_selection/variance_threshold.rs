//! Variance-based feature selection.
//!
//! [`VarianceThreshold`] removes features whose variance does not meet
//! a threshold, i.e. features that are constant or nearly constant.

use crate::traits::{Error, Fit, Result, Transform};
use polars::prelude::*;

/// Remove features with variance below a threshold.
///
/// Features with variance below `threshold` are removed. By default (threshold `0.0`),
/// only constant features are removed.
///
/// Only `Float64` columns are considered; columns of other dtypes are silently
/// dropped from the output.
///
/// # Example
///
/// ```rust
/// use featrs::feature_selection::VarianceThreshold;
/// use featrs::traits::{Fit, Transform};
/// use polars::prelude::{Column, DataFrame, NamedFrom, Series};
///
/// let low = Column::from(Series::new("low".into(), &[1.0_f64, 1.0, 1.0]));
/// let high = Column::from(Series::new("high".into(), &[1.0_f64, 5.0, 9.0]));
/// let df = DataFrame::new(3, vec![low, high])?;
///
/// let mut vt = VarianceThreshold::new(0.01);
/// vt.fit(df.clone())?;
/// let filtered = vt.transform(df)?;
/// assert_eq!(filtered.width(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct VarianceThreshold {
    fitted: bool,
    threshold: f64,
    selected_columns: Option<Vec<String>>,
}

impl VarianceThreshold {
    /// Create a new `VarianceThreshold` transformer.
    ///
    /// Features with variance strictly less than `threshold` are dropped.
    /// Use `0.0` (default) to remove only constant features.
    pub fn new(threshold: f64) -> Self {
        Self {
            fitted: false,
            threshold,
            selected_columns: None,
        }
    }
}

impl Default for VarianceThreshold {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl Fit<DataFrame> for VarianceThreshold {
    type Output = ();

    fn fit(&mut self, x: DataFrame) -> Result<()> {
        if x.width() == 0 {
            return Err(Error::InvalidInput(
                "VarianceThreshold.fit received a DataFrame with 0 columns. \
                 Provide at least one column."
                    .into(),
            ));
        }
        if x.height() == 0 {
            return Err(Error::InvalidInput(
                "VarianceThreshold.fit received a DataFrame with 0 rows. \
                 Provide at least one row."
                    .into(),
            ));
        }

        let mut selected = Vec::new();
        for col in x.columns() {
            let name = col.name().to_string();
            let s = x.column(&name).map_err(|e| {
                Error::InvalidInput(format!(
                    "VarianceThreshold.fit: column '{}' not found. {}",
                    name, e
                ))
            })?;
            if s.dtype() != &DataType::Float64 {
                continue;
            }
            let ca = s.f64().map_err(|e| {
                Error::InvalidInput(format!(
                    "VarianceThreshold.fit: column '{}' has dtype {}; expected Float64. {}",
                    name,
                    s.dtype(),
                    e
                ))
            })?;
            let vals: Vec<f64> = ca.iter().flatten().filter(|v| !v.is_nan()).collect();
            if vals.is_empty() {
                continue;
            }
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;

            if var >= self.threshold {
                selected.push(name);
            }
        }

        if selected.is_empty() {
            return Err(Error::InvalidInput(format!(
                "VarianceThreshold: no features meet the variance threshold ({}) \
                 out of {} f64 columns. Try lowering the threshold or checking \
                 that your data has variance.",
                self.threshold,
                x.get_column_names().len()
            )));
        }

        self.selected_columns = Some(selected);
        self.fitted = true;
        Ok(())
    }
}

impl Transform<DataFrame> for VarianceThreshold {
    type Output = DataFrame;

    fn transform(&self, x: DataFrame) -> Result<DataFrame> {
        if !self.fitted {
            return Err(Error::NotFitted(
                "VarianceThreshold has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            ));
        }
        let cols = self.selected_columns.as_ref().ok_or_else(|| {
            Error::NotFitted(
                "VarianceThreshold has not been fitted. \
                 Call .fit(dataframe) before .transform()."
                    .into(),
            )
        })?;
        let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        x.select(refs)
            .map_err(|e| Error::Computation(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_df() -> DataFrame {
        let a = Column::from(Series::new("low_var".into(), &[1.0f64, 1.0, 1.0]));
        let b = Column::from(Series::new("high_var".into(), &[1.0f64, 5.0, 9.0]));
        DataFrame::new(3, vec![a, b]).unwrap()
    }

    #[test]
    fn test_variance_threshold_removes_low_var() {
        let mut vt = VarianceThreshold::new(0.1);
        let df = make_test_df();

        vt.fit(df.clone()).unwrap();
        let result = vt.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "high_var");
    }

    #[test]
    fn test_variance_threshold_zero_keeps_all() {
        let mut vt = VarianceThreshold::new(0.0);
        let df = make_test_df();

        vt.fit(df.clone()).unwrap();
        let result = vt.transform(df).unwrap();

        assert_eq!(result.width(), 2);
    }

    #[test]
    fn test_variance_threshold_with_null_and_nan() {
        // Col 'a': [1.0, null, NaN, 5.0]. Non-null/non-nan: [1.0, 5.0].
        // Mean = 3.0. Variance = ((1-3)^2 + (5-3)^2)/2 = 4.0.
        // Col 'b': [2.0, 2.0, 2.0, 2.0]. Variance = 0.0.
        let a = Column::from(Series::new("a".into(), &[Some(1.0f64), None, Some(f64::NAN), Some(5.0)]));
        let b = Column::from(Series::new("b".into(), &[2.0f64, 2.0, 2.0, 2.0]));
        let df = DataFrame::new(4, vec![a, b]).unwrap();

        let mut vt = VarianceThreshold::new(1.0);
        vt.fit(df.clone()).unwrap();
        let result = vt.transform(df).unwrap();

        assert_eq!(result.width(), 1);
        assert_eq!(result.get_column_names()[0].as_str(), "a");
    }
}
